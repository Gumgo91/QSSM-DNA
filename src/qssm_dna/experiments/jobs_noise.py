"""Evaluation-only noise sensitivity and per-prediction runtime parts.

  noise    --seed S [--condition C]   frozen encoder under NoiseModel settings,
                                      exact and 1024-shot features, probes
                                      fitted without noise (policy A) and
                                      refitted on noisy features (policy B)
  runtime                             wall-clock per prediction of the QSSM,
                                      GRU, CNN and k-mer models
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import datasets, encoders, evaluate
from .jobs import ROOTS, SmallCNN, _device, _onehot, _rc_onehot, kmer_matrix
from .parts import register

NOISE_SETTINGS = {
    "none": {},
    **{f"p1_{v:g}": {"p1": v} for v in (1e-5, 1e-4, 1e-3)},
    **{f"p2_{v:g}": {"p2": v} for v in (1e-4, 1e-3, 1e-2)},
    **{f"pphi_{v:g}": {"p_phi": v} for v in (1e-4, 1e-3, 1e-2, 5e-2)},
    **{f"ro_{v:g}": {"readout": v} for v in (1e-3, 1e-2)},
    "moderate": {"p1": 1e-4, "p2": 1e-3, "p_phi": 1e-3, "readout": 1e-2},
    "current": {"p1": 1e-3, "p2": 1e-2, "p_phi": 1e-2, "readout": 1e-2},
}
REFIT_SETTINGS = ("p1_0.001", "p2_0.01", "pphi_0.05", "ro_0.01", "moderate",
                  "current")
NOISE_SHOTS = 1024
SHOT_SEEDS = range(10)


@torch.no_grad()
def noisy_dirs(enc, seqs, model, device, bs: int = 1024) -> dict:
    """Exact per-direction outcome distributions under a noise model
    (readout flips applied): {"zf","xf","zr","xr"} -> [N, 2^q]."""
    from ..models.encoder import rc_tokens
    from ..quantum import density, noise
    from ..quantum import readout as rd
    q = enc.q
    n = len(seqs)
    out = {k: np.zeros((n, 1 << q)) for k in ("zf", "xf", "zr", "xr")}
    for idx, t, m in evaluate._batches(seqs, bs):
        t = torch.from_numpy(t).to(device)
        m = torch.from_numpy(m).to(device)
        for tag, tok in (("f", t), ("r", rc_tokens(t, m))):
            rho = noise.noisy_forward_state(enc, tok, m, model)
            pz = rd.z_probabilities(rho).double()
            rx = density.to_matrix(rd.hadamard_all(
                density.to_axes(rho, q), q), q)
            px = rd.z_probabilities(rx).double()
            pz = noise.readout_flip_probs(pz / pz.sum(-1, keepdim=True),
                                          model.readout, q)
            px = noise.readout_flip_probs(px / px.sum(-1, keepdim=True),
                                          model.readout, q)
            out["z" + tag][idx] = pz.cpu().numpy()
            out["x" + tag][idx] = px.cpu().numpy()
    return out


def parities(pz: np.ndarray, px: np.ndarray, q: int = 4) -> np.ndarray:
    """subset30 vector (Z parities, then X parities) from distributions."""
    from ..quantum import readout as rd
    s = rd._subset_sign_table(q).numpy()            # [15,16]
    return np.concatenate([pz @ s.T, px @ s.T], 1).astype(np.float32)


def feat_exact(d: dict) -> np.ndarray:
    return 0.5 * (parities(d["zf"], d["xf"]) + parities(d["zr"], d["xr"]))


def feat_shots(d: dict, shots: int, seed: int) -> np.ndarray:
    """Joint-bitstring sampling: one multinomial count vector per basis and
    direction feeds every parity, so raw11 and subset30 share the draws."""
    g = torch.Generator().manual_seed(seed)
    feats = []
    for z, x in (("zf", "xf"), ("zr", "xr")):
        pz = torch.from_numpy(d[z]).clamp_min(0)
        px = torch.from_numpy(d[x]).clamp_min(0)
        iz = torch.multinomial(pz, shots, replacement=True, generator=g)
        ix = torch.multinomial(px, shots, replacement=True, generator=g)
        cz = torch.zeros_like(pz).scatter_add_(1, iz, torch.ones_like(
            iz, dtype=pz.dtype))
        cx = torch.zeros_like(px).scatter_add_(1, ix, torch.ones_like(
            ix, dtype=px.dtype))
        feats.append(parities(cz.numpy() / shots, cx.numpy() / shots))
    return 0.5 * (feats[0] + feats[1])


@register("noise")
def part_noise(args, run_dir: Path) -> dict:
    from ..data.tokenizer import encode, pad_tokens
    from ..eval.metrics import binary_metrics
    from ..quantum.noise import NoiseModel, noisy_forward_state
    device = _device()
    refs = encoders.scan_runs(ROOTS)
    cond = getattr(args, "condition", None) or "qssm_dense8"
    ref = encoders.pick_run(refs, cond, args.seed)
    enc = encoders.load_encoder(ref, device)
    gb, _ = datasets.load("gb_promoter")
    ho, _ = datasets.load("holdout_promoter")
    iso = np.load(datasets.DATA_DIR /
                  "holdout_promoter_chr9_chr10.npz")["isolated10kb"]
    ri = evaluate.raw11_index()
    readouts = (("subset30", slice(None)), ("raw11", ri))
    ysets = {"test": gb["test"]["y"], "holdout": ho["test"]["y"]}
    names = {"test": "gb_promoter:test", "holdout": "holdout_promoter"}
    rows, dist_rows = [], []
    clean, probes = {}, {}
    for name, cfg in NOISE_SETTINGS.items():
        model = NoiseModel(**cfg)
        need_fit = name == "none" or name in REFIT_SETTINGS
        splits = {"test": gb["test"]["seqs"], "holdout": ho["test"]["seqs"]}
        if need_fit:
            splits.update(train=gb["train"]["seqs"], val=gb["val"]["seqs"])
        d = {k: noisy_dirs(enc, v, model, device) for k, v in splits.items()}
        fx = {k: feat_exact(v) for k, v in d.items()}
        if name == "none":
            clean = fx
            for ro, cols in readouts:
                probes[ro] = evaluate.fit_probe(
                    fx["train"][:, cols], gb["train"]["y"],
                    fx["val"][:, cols], gb["val"]["y"])
        dist_rows.append({"seed": args.seed, "condition": cond,
                          "setting": name,
                          "mean_abs_dsubset30_test": float(np.abs(
                              fx["test"] - clean["test"]).mean())})
        policies = {"A_noiseless_probe": probes}
        if name in REFIT_SETTINGS:
            policies["B_refit_probe"] = {
                ro: evaluate.fit_probe(fx["train"][:, cols],
                                       gb["train"]["y"], fx["val"][:, cols],
                                       gb["val"]["y"])
                for ro, cols in readouts}
        variants = [("exact", None, fx)]
        for ss in SHOT_SEEDS:
            variants.append((NOISE_SHOTS, ss, {
                k: feat_shots(d[k], NOISE_SHOTS,
                              10_000 * args.seed + 100 * ss + i)
                for i, k in enumerate(("test", "holdout"))}))
        for shots, ss, F in variants:
            for pol, pr in policies.items():
                for ro, cols in readouts:
                    head, thr = pr[ro]["head"], pr[ro]["threshold"]
                    for key in ("test", "holdout"):
                        y = ysets[key]
                        p = evaluate.predict(head, F[key][:, cols])
                        noise_cols = {f"noise_{k}": v for k, v in
                                      model.as_dict().items()}
                        base = {"seed": args.seed, "condition": cond,
                                "setting": name,
                                **noise_cols, "policy": pol,
                                "readout": ro, "shots": shots,
                                "shot_seed": ss, "threshold": thr}
                        met = binary_metrics(y, p, thr)
                        rows.append({**base, "eval_set": names[key],
                                     "eval_mcc": met["mcc"],
                                     "eval_auroc": met["auroc"]})
                        if key == "holdout":
                            keep = (y == 0) | iso
                            met = binary_metrics(y[keep], p[keep], thr)
                            rows.append({**base,
                                         "eval_set": "holdout_isolated",
                                         "eval_mcc": met["mcc"],
                                         "eval_auroc": met["auroc"]})
        print(f"noise seed {args.seed} {name} done", flush=True)
    pd.DataFrame(rows).to_csv(run_dir / "noise_metrics.csv", index=False)
    pd.DataFrame(dist_rows).to_csv(run_dir / "feature_distortion.csv",
                                   index=False)
    # error accumulation versus sequence length: trace distance between
    # noisy and noiseless terminal states of sequence prefixes
    seqs = gb["test"]["seqs"][:512]
    acc = []
    for name in ("p2_0.01", "pphi_0.05", "current"):
        model = NoiseModel(**NOISE_SETTINGS[name])
        for L in (8, 16, 32, 64, 128, 251):
            t, m = pad_tokens([encode(s[:L]) for s in seqs])
            t = torch.from_numpy(t).to(device)
            m = torch.from_numpy(m).to(device)
            r0 = noisy_forward_state(enc, t, m, NoiseModel())
            r1 = noisy_forward_state(enc, t, m, model)
            ev = torch.linalg.eigvalsh((r1 - r0).to(torch.complex128))
            td = 0.5 * ev.abs().sum(-1)
            acc.append({"seed": args.seed, "condition": cond,
                        "setting": name, "L": L,
                        "trace_distance_mean": float(td.mean()),
                        "trace_distance_sd": float(td.std())})
    pd.DataFrame(acc).to_csv(run_dir / "noise_vs_length.csv", index=False)
    (run_dir / "provenance.json").write_text(json.dumps({
        "encoder_run": ref.run_dir.name,
        "checkpoint_sha256": encoders.sha256_file(ref.run_dir /
                                                  "checkpoint.pt"),
        "settings": NOISE_SETTINGS, "refit": REFIT_SETTINGS,
        "shots": NOISE_SHOTS, "shot_seeds": list(SHOT_SEEDS)}, indent=2))
    return {"rows": len(rows), "device": device}


def _time(fn, cuda: bool, warm: int = 10, rep: int = 30) -> dict:
    for _ in range(warm):
        fn()
    if cuda:
        torch.cuda.synchronize()
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return {"median_s": float(np.median(ts)),
            "p10_s": float(np.percentile(ts, 10)),
            "p90_s": float(np.percentile(ts, 90))}


@register("runtime")
def part_runtime(args, run_dir: Path) -> dict:
    from ..data.tokenizer import encode, pad_tokens
    from ..models.encoder import rc_tokens
    from ..quantum import readout as rd
    from ..train.probe import _GraphedForward
    torch.set_num_threads(8)
    refs = encoders.scan_runs(ROOTS)
    gb, _ = datasets.load("gb_promoter")
    seqs = gb["test"]["seqs"]
    rows = []
    devs = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for dev in devs:
        cuda = dev == "cuda"
        enc = encoders.load_encoder(
            encoders.pick_run(refs, "qssm_dense8", 22), dev)
        g64 = encoders.load_encoder(
            encoders.pick_run(refs, "gru64_dense8", 22), dev)
        g8 = encoders.load_encoder(
            encoders.pick_run(refs, "gru8_dense8", 22), dev)
        cnn = SmallCNN().to(dev).eval()
        wd = torch.randn(30, device=dev)
        for bs in (1, 256):
            t, m = pad_tokens([encode(s) for s in seqs[:bs]])
            t = torch.from_numpy(t).to(dev)
            m = torch.from_numpy(m).to(dev)
            tr = rc_tokens(t, m)
            gf = _GraphedForward() if cuda else None

            def states():
                if gf:
                    return (gf(enc.forward_state, t, m).clone(),
                            gf(enc.forward_state, tr, m).clone())
                return enc.forward_state(t, m), enc.forward_state(tr, m)

            def qssm_exact():
                with torch.no_grad():
                    f, r = states()
                    z = 0.5 * (rd.subset_readout(f, 4)
                               + rd.subset_readout(r, 4))
                    return torch.sigmoid(z.float() @ wd)

            def qssm_shots():
                with torch.no_grad():
                    f, r = states()
                    a = rd.joint_shot_features(f, 4, NOISE_SHOTS)
                    b = rd.joint_shot_features(r, 4, NOISE_SHOTS)
                    z = 0.5 * (a["subset30"] + b["subset30"])
                    return torch.sigmoid(z.float() @ wd)

            def gru(e):
                def f():
                    with torch.no_grad():
                        return e.encode(t, m, rc_average=True)
                return f

            x, mk = _onehot(seqs[:bs], 251)
            xt = torch.from_numpy(x).to(dev)
            xrt = torch.from_numpy(_rc_onehot(x, mk)).to(dev)
            mt = torch.from_numpy(mk).to(dev)

            def cnn_f():
                with torch.no_grad():
                    return torch.sigmoid(0.5 * (cnn(xt, mt) + cnn(xrt, mt)))

            for name, fn in (("qssm_exact_rc", qssm_exact),
                             ("qssm_shots1024_rc", qssm_shots),
                             ("gru64_rc", gru(g64)), ("gru8_rc", gru(g8)),
                             ("cnn_rc", cnn_f)):
                tm = _time(fn, cuda)
                rows.append({"model": name, "device": dev, "batch": bs,
                             "seq_len": 251, **tm,
                             "ms_per_sequence": 1e3 * tm["median_s"] / bs,
                             "hardware": torch.cuda.get_device_name(0)
                             if cuda else
                             f"cpu, {torch.get_num_threads()} threads"})
                print(rows[-1], flush=True)
    wk = np.random.default_rng(0).standard_normal(2080)
    for bs in (1, 256):
        sub = seqs[:bs]
        tm = _time(lambda: kmer_matrix(sub, 6) @ wk, False)
        rows.append({"model": "kmer6_lr", "device": "cpu", "batch": bs,
                     "seq_len": 251, **tm,
                     "ms_per_sequence": 1e3 * tm["median_s"] / bs,
                     "hardware": "cpu, numpy"})
    pd.DataFrame(rows).to_csv(run_dir / "runtime.csv", index=False)
    return {"rows": len(rows)}
