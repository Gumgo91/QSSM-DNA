"""Evaluation parts (registered into parts.PARTS).

  eval-encoder  --condition C --seed S [--datasets ...]   frozen encoder + linear probes
  kmer-lr       [--datasets ...]                          canonical k-mer logistic regression
  cnn           --seed S [--datasets ...]                 supervised CNN
The noise and runtime parts are in jobs_noise.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import datasets, encoders, evaluate
from .parts import register

# directories searched for run directories (QSSM_DNA_RUNS, os.pathsep-separated)
ROOTS = (tuple(os.environ["QSSM_DNA_RUNS"].split(os.pathsep))
         if os.environ.get("QSSM_DNA_RUNS") else ("runs",))
TASKS = ("gb_promoter", "gb_promoter_s6k", "gb_cohn", "gb_ocr")
DEFAULT_SETS = "gb_promoter,holdout_promoter,gb_cohn,gb_ocr"
PAULI_CONDITIONS = ("qssm_endpoint", "qssm_dense8", "qssm_random_init")


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _base_cfg() -> dict:
    from ..config import load_config
    return load_config("configs/endpoint.yaml")


def _sets(args) -> list[str]:
    return (getattr(args, "datasets", None) or DEFAULT_SETS).split(",")


def _holdout_extra(views_fn, feats_h, y_h: np.ndarray, iso: np.ndarray,
                   readouts) -> dict:
    v = views_fn(feats_h["test"])
    v = {k: v[k] for k in readouts}
    pos_iso = (y_h == 0) | iso
    return {"holdout_promoter": (v, y_h),
            "holdout_isolated": ({k: a[pos_iso] for k, a in v.items()},
                                 y_h[pos_iso])}


def _load(args, device):
    refs = encoders.scan_runs(ROOTS)
    if args.condition.startswith("qssm_random_init"):
        q = encoders.qubits_of_condition(args.condition)
        if q == 4:
            base = _base_cfg()
        else:
            from ..config import load_config
            base = load_config(f"configs/allpos_q{q}.yaml")
        enc, info = encoders.matched_random_init(base, args.seed, device,
                                                 refs)
        return enc, "qssm", {"condition": args.condition, "seed": args.seed,
                             "encoder_run": None, **info}
    ref = encoders.pick_run(refs, args.condition, args.seed)
    enc = encoders.load_encoder(ref, device)
    return enc, ref.family, {
        "condition": args.condition, "seed": args.seed,
        "encoder_run": ref.run_dir.name,
        "checkpoint_sha256": encoders.sha256_file(
            ref.run_dir / "checkpoint.pt"),
        "pretrain_metrics": ref.metrics}


@register("eval-encoder")
def part_eval_encoder(args, run_dir: Path) -> dict:
    device = _device()
    enc, family, meta = _load(args, device)
    key = f"{args.condition}_s{args.seed}"
    sets = _sets(args)
    rows, preds, mans = [], [], {}
    for task in [t for t in TASKS if t in sets]:
        splits, man = datasets.load(task)
        mans[task] = man
        pauli = task == "gb_promoter" and args.condition in PAULI_CONDITIONS
        feats = evaluate.cached_features(f"{key}/{task}", family, enc,
                                         splits, device, pauli=pauli)
        if family == "qssm":
            views_fn = lambda c: evaluate.qssm_views(c, enc)  # noqa: E731
        else:
            views_fn = evaluate.gru_views
        per = {s: views_fn(feats[s]) for s in feats}
        readouts = list(per["train"])
        views = {r: {s: per[s][r] for s in per} for r in readouts}
        labels = {s: splits[s]["y"] for s in splits}
        extra = None
        if task.startswith("gb_promoter") and "holdout_promoter" in sets:
            hs, hman = datasets.load("holdout_promoter")
            mans["holdout_promoter"] = hman
            fh = evaluate.cached_features(f"{key}/holdout_promoter", family,
                                          enc, hs, device, pauli=pauli)
            iso = np.load(datasets.DATA_DIR /
                          "holdout_promoter_chr9_chr10.npz")["isolated10kb"]
            extra = _holdout_extra(views_fn, fh, hs["test"]["y"], iso,
                                   readouts)
        r_, p_, _ = evaluate.evaluate_views(views, labels, extra)
        for r in r_:
            r.update(task=task, family=family, **{k: meta.get(k) for k in
                                                  ("condition", "seed",
                                                   "encoder_run")})
        rows += r_
        for p in p_:
            p["task"] = task
        preds += p_
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "metrics.csv", index=False)
    evaluate.save_predictions(run_dir / "predictions.npz",
                              {"condition": args.condition,
                               "seed": args.seed},
                              [{**p, "eval_set": f"{p['task']}:"
                                f"{p['eval_set']}"} for p in preds])
    (run_dir / "provenance.json").write_text(json.dumps(
        {"encoder": meta, "datasets": mans,
         "probe_protocol": evaluate.__doc__}, indent=2, default=str))
    return {"rows": len(df), "device": device}


# ------------------------------------------------------------ k-mer LR

def _canonical_kmer_index(k: int) -> np.ndarray:
    comp = np.array([3, 2, 1, 0])
    idx = np.arange(4 ** k)
    digits = np.stack([(idx // 4 ** (k - 1 - j)) % 4 for j in range(k)], 1)
    rcd = comp[digits[:, ::-1]]
    rc_idx = (rcd * (4 ** np.arange(k - 1, -1, -1))).sum(1)
    canon = np.minimum(idx, rc_idx)
    uniq, inv = np.unique(canon, return_inverse=True)
    return inv


def kmer_matrix(seqs: list[str], k: int) -> np.ndarray:
    """Canonical (strand-merged) k-mer frequencies; N breaks a k-mer."""
    from scipy.sparse import csr_matrix
    lut = np.full(256, -1, np.int64)
    for i, c in enumerate("ACGT"):
        lut[ord(c)] = i
    inv = _canonical_kmer_index(k)
    rows, cols = [], []
    for r, s in enumerate(seqs):
        a = lut[np.frombuffer(s.upper().encode(), np.uint8)]
        n = len(a) - k + 1
        if n <= 0:
            continue
        bad = np.convolve((a < 0).astype(np.int64), np.ones(k, np.int64),
                          "valid") > 0
        a = np.where(a < 0, 0, a)
        code = np.zeros(n, np.int64)
        for j in range(k):
            code = code * 4 + a[j:j + n]
        code = inv[code[~bad]]
        rows.append(np.full(len(code), r))
        cols.append(code)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    m = csr_matrix((np.ones(len(rows)), (rows, cols)),
                   shape=(len(seqs), int(inv.max()) + 1)).toarray()
    return m / np.maximum(m.sum(1, keepdims=True), 1)


@register("kmer-lr")
def part_kmer_lr(args, run_dir: Path) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from ..eval.metrics import binary_metrics, pick_threshold
    sets = _sets(args)
    rows, grid_rows, preds = [], [], []
    for task in [t for t in TASKS if t in sets]:
        splits, _ = datasets.load(task)
        y = {s: splits[s]["y"] for s in splits}
        best = None
        cache = {}
        for k in (3, 4, 5, 6):
            X = {s: kmer_matrix(splits[s]["seqs"], k) for s in splits}
            sc = StandardScaler().fit(X["train"])
            X = {s: sc.transform(v) for s, v in X.items()}
            cache[k] = X
            for C in (0.01, 0.1, 1.0, 10.0):
                clf = LogisticRegression(C=C, max_iter=5000)
                clf.fit(X["train"], y["train"])
                pv = clf.predict_proba(X["val"])[:, 1]
                thr = pick_threshold(y["val"], pv)
                vm = binary_metrics(y["val"], pv, thr)["mcc"]
                grid_rows.append({"task": task, "k": k, "C": C,
                                  "val_mcc": vm, "threshold": thr})
                if best is None or vm > best[0]:
                    best = (vm, k, C, clf, sc, thr)
        vm, k, C, clf, sc, thr = best
        targets = {"test": (cache[k]["test"], y["test"])}
        if task == "gb_promoter" and "holdout_promoter" in sets:
            hs, _ = datasets.load("holdout_promoter")
            Xh = sc.transform(kmer_matrix(hs["test"]["seqs"], k))
            yh = hs["test"]["y"]
            iso = np.load(datasets.DATA_DIR /
                          "holdout_promoter_chr9_chr10.npz")["isolated10kb"]
            keep = (yh == 0) | iso
            targets["holdout_promoter"] = (Xh, yh)
            targets["holdout_isolated"] = (Xh[keep], yh[keep])
        for name, (X, yy) in targets.items():
            p = clf.predict_proba(X)[:, 1]
            met = binary_metrics(yy, p, thr)
            rows.append({"task": task, "condition": "kmer_lr", "seed": None,
                         "readout": f"canonical_{k}mer", "k": k, "C": C,
                         "val_mcc": vm, "selected_threshold": thr,
                         "eval_set": name,
                         **{f"eval_{a}": b for a, b in met.items()}})
            preds.append({"readout": f"k{k}", "eval_set": f"{task}:{name}",
                          "prob": p.astype(np.float32),
                          "label": np.asarray(yy, np.int8),
                          "threshold": thr})
    pd.DataFrame(rows).to_csv(run_dir / "metrics.csv", index=False)
    pd.DataFrame(grid_rows).to_csv(run_dir / "validation_grid.csv",
                                   index=False)
    evaluate.save_predictions(run_dir / "predictions.npz",
                              {"condition": "kmer_lr"}, preds)
    return {"rows": len(rows)}


# ------------------------------------------------------------ CNN

class SmallCNN(torch.nn.Module):
    def __init__(self, ch: int = 32, kernel: int = 8):
        super().__init__()
        def block(cin):
            return [torch.nn.Conv1d(cin, ch, kernel), torch.nn.ReLU(),
                    torch.nn.BatchNorm1d(ch)]
        self.net = torch.nn.Sequential(
            *block(5), torch.nn.MaxPool1d(2), *block(ch),
            torch.nn.MaxPool1d(2), *block(ch))
        self.head = torch.nn.Linear(ch, 1)

    def forward(self, x, mask):
        """x [B,5,L] one-hot (A,C,G,T,N), mask [B,L]; masked mean pooling
        over valid output positions."""
        h = self.net(x)
        lv = mask.float().sum(1)
        # valid output length after three convs and two pools
        lo = torch.floor((torch.floor((lv - 7) / 2) - 7) / 2) - 7
        lo = lo.clamp_min(1)
        pos = torch.arange(h.shape[-1], device=x.device)
        m = (pos.unsqueeze(0) < lo.unsqueeze(1)).float()
        z = (h * m.unsqueeze(1)).sum(-1) / m.sum(1, keepdim=True)
        return self.head(z)[:, 0]


def _onehot(seqs: list[str], lmax: int):
    from ..data.tokenizer import encode
    x = np.zeros((len(seqs), 5, lmax), np.float32)
    m = np.zeros((len(seqs), lmax), bool)
    for i, s in enumerate(seqs):
        t = encode(s)[:lmax]
        x[i, np.minimum(t, 4), np.arange(len(t))] = 1.0
        m[i, :len(t)] = True
    return x, m


def _rc_onehot(x: np.ndarray, m: np.ndarray):
    """Reverse complement of left-aligned one-hot batches."""
    out = np.zeros_like(x)
    comp = [3, 2, 1, 0, 4]
    lens = m.sum(1)
    for i, l in enumerate(lens):
        out[i, :, :l] = x[i, comp, :l][:, ::-1]
    return out


def _cnn_predict(model, x, m, xr, device, bs=1024):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(x), bs):
            mm = torch.from_numpy(m[s:s + bs]).to(device)
            a = model(torch.from_numpy(x[s:s + bs]).to(device), mm)
            b = model(torch.from_numpy(xr[s:s + bs]).to(device), mm)
            out.append(torch.sigmoid(0.5 * (a + b)).cpu().numpy())
    return np.concatenate(out)


@register("cnn")
def part_cnn(args, run_dir: Path) -> dict:
    import torch.nn.functional as F
    from ..eval.metrics import binary_metrics, pick_threshold
    device = _device()
    sets = _sets(args)
    rows, preds, hist = [], [], []
    for task in [t for t in TASKS if t in sets]:
        splits, _ = datasets.load(task)
        lmax = max(len(s) for sp in splits.values() for s in sp["seqs"])
        data = {}
        for s, d in splits.items():
            x, m = _onehot(d["seqs"], lmax)
            data[s] = (x, m, _rc_onehot(x, m), d["y"].astype(np.float32))
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        model = SmallCNN().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                weight_decay=1e-4)
        xt, mt, xrt, yt = data["train"]
        best, best_state, bad = np.inf, None, 0
        for ep in range(20):
            model.train()
            order = rng.permutation(len(yt))
            for s in range(0, len(order), 128):
                b = order[s:s + 128]
                # random strand per example (RC augmentation)
                flip = rng.random(len(b)) < 0.5
                xb = np.where(flip[:, None, None], xrt[b], xt[b])
                logit = model(torch.from_numpy(xb).to(device),
                              torch.from_numpy(mt[b]).to(device))
                loss = F.binary_cross_entropy_with_logits(
                    logit, torch.from_numpy(yt[b]).to(device))
                opt.zero_grad()
                loss.backward()
                opt.step()
            xv, mv, xrv, yv = data["val"]
            pv = _cnn_predict(model, xv, mv, xrv, device)
            vl = float(F.binary_cross_entropy(
                torch.tensor(np.clip(pv, 1e-7, 1 - 1e-7)),
                torch.tensor(yv)))
            hist.append({"task": task, "epoch": ep + 1, "val_loss": vl})
            if vl < best - 1e-4:
                best, bad = vl, 0
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= 3:
                    break
        model.load_state_dict(best_state)
        xv, mv, xrv, yv = data["val"]
        pv = _cnn_predict(model, xv, mv, xrv, device)
        thr = pick_threshold(yv, pv)
        vm = binary_metrics(yv, pv, thr)["mcc"]
        targets = {"test": data["test"]}
        if task == "gb_promoter" and "holdout_promoter" in sets:
            hs, _ = datasets.load("holdout_promoter")
            xh, mh = _onehot(hs["test"]["seqs"], lmax)
            yh = hs["test"]["y"].astype(np.float32)
            iso = np.load(datasets.DATA_DIR /
                          "holdout_promoter_chr9_chr10.npz")["isolated10kb"]
            keep = (yh == 0) | iso
            xrh = _rc_onehot(xh, mh)
            targets["holdout_promoter"] = (xh, mh, xrh, yh)
            targets["holdout_isolated"] = (xh[keep], mh[keep], xrh[keep],
                                           yh[keep])
        for name, (x, m, xr, yy) in targets.items():
            p = _cnn_predict(model, x, m, xr, device)
            met = binary_metrics(yy.astype(int), p, thr)
            rows.append({"task": task, "condition": "cnn",
                         "seed": args.seed, "readout": "supervised",
                         "n_params": sum(q.numel() for q in
                                         model.parameters()),
                         "val_mcc": vm, "selected_threshold": thr,
                         "eval_set": name,
                         **{f"eval_{a}": b for a, b in met.items()}})
            preds.append({"readout": "cnn", "eval_set": f"{task}:{name}",
                          "prob": p.astype(np.float32),
                          "label": yy.astype(np.int8), "threshold": thr})
    pd.DataFrame(rows).to_csv(run_dir / "metrics.csv", index=False)
    pd.DataFrame(hist).to_csv(run_dir / "train_history.csv", index=False)
    evaluate.save_predictions(run_dir / "predictions.npz",
                              {"condition": "cnn", "seed": args.seed}, preds)
    return {"rows": len(rows), "device": device}
