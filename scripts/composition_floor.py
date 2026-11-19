"""Composition floor of the frozen-probe protocol (CPU, two threads).

  CUDA_VISIBLE_DEVICES=-1 python scripts/composition_floor.py

What a linear probe reaches from sequence composition alone on the four
evaluation sets of the paper: the promoter holdout (probes fitted on the
promoter benchmark), the benchmark test split, enhancers and open chromatin.
Splits come from qssm_dna.experiments.datasets (data/, with the overlap
clusters of data/holdout_promoter_chr9_chr10.npz), so training, validation
and test sequences are those of every encoder evaluation.

Feature sets. Each is computed on a sequence and on its reverse complement and
the two are averaged, as the encoder features are:
  gc_cpg            GC fraction and CpG observed/expected ratio
                    (#CG x bases / (#C x #G)), 2 features
  mono_di           mononucleotide and dinucleotide frequencies, 4 + 16
  leaky_integrator  exponential moving averages of the one-hot base sequence,
                    h_t = (1 - g) h_{t-1} + g onehot(x_t) with h_0 = 0, read at
                    the terminal position, for the four initial damping rates
                    g of the encoder (configs/endpoint.yaml: log-spaced from
                    initial_gamma_min to initial_gamma_max over the qubits),
                    4 x 4 features
  leaky_integrator_normalized
                    the same averages divided by 1 - (1 - g)^L, so that the
                    weights of each rate sum to one over a sequence of L bases
gc_cpg and mono_di are computed on the last 64, the last 128 and all bases of
each direction (the bases nearest to the terminal position of that direction;
the whole sequence when it is shorter than the window). N is not counted,
breaks a dinucleotide and adds nothing to a moving average.

Fits. probe_*: the probe of the paper, qssm_dna.experiments.evaluate unchanged
(AdamW, 200 full-batch epochs, no standardization, probe seed 11, threshold =
argmax validation MCC). lr_*: L2 logistic regression on standardized features
solved to convergence (Newton-Cholesky, tolerance 1e-10), C chosen by
validation MCC, threshold by the same rule.
The features and both fits are deterministic (n = 1). Intervals are 95 %
bootstrap intervals of the test MCC with the fitted model and threshold
frozen, over sequences (holdout: overlap clusters), 2,000 resamples.

Reference rows are copied from results/cells_mean_sd.csv: frozen
subset30 probes of the untrained register and of the all-position encoders
(mean and SD over ten seeds, in the probe_* columns) and the k-mer logistic
regression (in the lr_* columns).

Output: results/composition_floor.csv
"""

from __future__ import annotations

import os

THREADS = 2
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = str(THREADS)

import ctypes  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, "src")
from qssm_dna.config import load_config  # noqa: E402
from qssm_dna.eval.metrics import binary_metrics, pick_threshold  # noqa: E402
from qssm_dna.experiments import datasets, evaluate, jobs  # noqa: E402

RES = Path("results")
TASKS = ("gb_promoter", "gb_cohn", "gb_ocr")
EVALS = ["holdout_promoter", "gb_promoter:test", "gb_cohn:test",
         "gb_ocr:test"]
WINDOWS = {"last64": 64, "last128": 128, "all": None}
C_GRID = (0.01, 0.1, 1.0, 10.0)
N_BOOT = 2000
BOOT_SEED = 20261112
SIZES = {"holdout_promoter": (764, 641), "gb_promoter:test": 9034}
REFERENCES = [("qssm_random_init", "subset30"), ("qssm_allpos", "subset30"),
              ("qssm_allpos_b2500000", "subset30"), ("kmer_lr", None)]
COMP = np.array([3, 2, 1, 0, 4], np.uint8)
PAD = 5


def damping_ladder() -> np.ndarray:
    """Initial damping rates of the qubits (TokenController bias init)."""
    m = load_config("configs/endpoint.yaml")["model"]
    f = m["forgetting"]
    return np.exp(np.linspace(np.log(f["initial_gamma_min"]),
                              np.log(f["initial_gamma_max"]),
                              int(m["n_qubits"])))


def right_aligned(seqs: list[str], rc: bool) -> np.ndarray:
    """[n, Lmax] base codes (A C G T = 0..3, N = 4); every sequence ends in
    the last column and PAD fills the columns before its first base."""
    lut = np.full(256, 4, np.uint8)
    for i, c in enumerate("ACGT"):
        lut[ord(c)] = i
    lmax = max(len(s) for s in seqs)
    m = np.full((len(seqs), lmax), PAD, np.uint8)
    for r, s in enumerate(seqs):
        a = lut[np.frombuffer(s.upper().encode(), np.uint8)]
        m[r, lmax - len(a):] = COMP[a[::-1]] if rc else a
    return m


def base_counts(m: np.ndarray) -> np.ndarray:
    return np.stack([(m == b).sum(1) for b in range(4)], 1).astype(np.float64)


def gc_cpg(m: np.ndarray) -> np.ndarray:
    cnt = base_counts(m)
    n = cnt.sum(1)
    cg = ((m[:, :-1] == 1) & (m[:, 1:] == 2)).sum(1)
    den = cnt[:, 1] * cnt[:, 2]
    return np.stack([(cnt[:, 1] + cnt[:, 2]) / np.maximum(n, 1),
                     np.where(den > 0, cg * n / np.maximum(den, 1), 0.0)], 1)


def mono_di(m: np.ndarray) -> np.ndarray:
    mono = base_counts(m)
    a, b = m[:, :-1], m[:, 1:]
    pair = np.where((a < 4) & (b < 4), a * 4 + b, 16)
    di = np.stack([(pair == c).sum(1) for c in range(16)],
                  1).astype(np.float64)
    return np.concatenate(
        [mono / np.maximum(mono.sum(1, keepdims=True), 1),
         di / np.maximum(di.sum(1, keepdims=True), 1)], 1)


def leaky_integrator(m: np.ndarray, rates: np.ndarray,
                     normalized: bool) -> np.ndarray:
    """Terminal value of h_t = (1 - g) h_{t-1} + g onehot(x_t), h_0 = 0:
    sum over positions of g (1 - g)^(distance to the last base)."""
    dist = np.arange(m.shape[1] - 1, -1, -1)
    w = rates[None, :] * (1.0 - rates[None, :]) ** dist[:, None]
    h = np.stack([(m == b).astype(np.float64) @ w for b in range(4)], 2)
    if normalized:
        length = (m != PAD).sum(1)
        h = h / (1.0 - (1.0 - rates[None, :]) ** length[:, None])[:, :, None]
    return h.reshape(len(m), -1)


def features(seqs: list[str], rates: np.ndarray) -> dict[str, np.ndarray]:
    """{'<feature set>:<window>': [n, d]}, averaged over both directions."""
    out: dict[str, np.ndarray] = {}
    for rc in (False, True):
        m = right_aligned(seqs, rc)
        f = {}
        for wname, w in WINDOWS.items():
            mw = m if w is None else m[:, -w:]
            f[f"gc_cpg:{wname}"] = gc_cpg(mw)
            f[f"mono_di:{wname}"] = mono_di(mw)
        f["leaky_integrator:all"] = leaky_integrator(m, rates, False)
        f["leaky_integrator_normalized:all"] = leaky_integrator(m, rates,
                                                               True)
        for k, v in f.items():
            out[k] = out.get(k, 0.0) + 0.5 * v
    return out


def fit_lr(x: dict, y: dict, targets: dict) -> tuple[list[dict], dict]:
    """Standardized L2 logistic regression; C by validation MCC (first
    maximum kept), threshold by the rule of the paper."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(x["train"])
    xt, xv = sc.transform(x["train"]), sc.transform(x["val"])
    best = None
    for C in C_GRID:
        clf = LogisticRegression(C=C, solver="newton-cholesky", tol=1e-10,
                                 max_iter=1000)
        clf.fit(xt, y["train"])
        if int(clf.n_iter_[0]) >= clf.max_iter:
            raise RuntimeError(f"logistic regression not converged (C={C})")
        pv = clf.predict_proba(xv)[:, 1]
        thr = pick_threshold(y["val"], pv)
        vm = binary_metrics(y["val"], pv, thr)["mcc"]
        if best is None or vm > best[0]:
            best = (vm, C, clf, thr)
    vm, C, clf, thr = best
    rows, preds = [], {}
    for name, (z, yy) in targets.items():
        p = clf.predict_proba(sc.transform(z))[:, 1]
        met = binary_metrics(yy, p, thr)
        rows.append({"eval_set": name, "lr_mcc": met["mcc"],
                     "lr_auroc": met["auroc"], "lr_val_mcc": vm,
                     "lr_threshold": thr, "lr_C": C,
                     "lr_n_iter": int(clf.n_iter_[0])})
        preds[name] = (p >= thr).astype(int)
    return rows, preds


def boot_ci(y: np.ndarray, pred: np.ndarray, clusters: np.ndarray | None,
            rng: np.random.Generator) -> tuple[float, float]:
    """Percentile interval of the MCC under resampling of sequences (the
    four confusion counts are then multinomial) or of whole clusters."""
    cat = 2 * np.asarray(y, int) + pred          # tn, fp, fn, tp
    if clusters is None:
        c = rng.multinomial(len(cat), np.bincount(cat, minlength=4)
                            / len(cat), size=N_BOOT).astype(np.float64)
    else:
        units, inv = np.unique(clusters, return_inverse=True)
        tab = np.zeros((len(units), 4))
        np.add.at(tab, (inv, cat), 1.0)
        c = rng.multinomial(len(units), np.full(len(units), 1.0 / len(units)),
                            size=N_BOOT) @ tab
    tn, fp, fn, tp = c.T
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.where(den > 0, (tp * tn - fp * fn) / np.maximum(den, 1e-300),
                   0.0)
    lo, hi = np.percentile(mcc, [2.5, 97.5])
    return float(lo), float(hi)


def check_sizes(gb: dict, hs: dict) -> None:
    y = hs["test"]["y"]
    got = {"holdout_promoter": (int((y == 1).sum()), int((y == 0).sum())),
           "gb_promoter:test": len(gb["test"]["seqs"])}
    print(f"task sizes: holdout {got['holdout_promoter'][0]} positives, "
          f"{got['holdout_promoter'][1]} negatives; benchmark test "
          f"{got['gb_promoter:test']} sequences", flush=True)
    if got != SIZES:
        raise SystemExit(f"task sizes {got} differ from those of the paper "
                         f"{SIZES}")


def task_rows(task: str, rates: np.ndarray,
              rng: np.random.Generator) -> list[dict]:
    splits, _ = datasets.load(task)
    labels = {s: splits[s]["y"] for s in splits}
    feats = {s: features(splits[s]["seqs"], rates) for s in splits}
    names = list(feats["train"])
    views = {k: {s: feats[s][k] for s in feats} for k in names}
    extra, clusters = None, {}
    if task == "gb_promoter":
        hs, _ = datasets.load("holdout_promoter")
        check_sizes(splits, hs)
        fh = features(hs["test"]["seqs"], rates)
        extra = {"holdout_promoter": (fh, hs["test"]["y"])}
        clusters["holdout_promoter"] = np.load(
            datasets.DATA_DIR / "holdout_promoter_chr9_chr10.npz")["cluster"]
    r_, p_, _ = evaluate.evaluate_views(views, labels, extra)
    probe = {(r["readout"], r["eval_set"]): r for r in r_}
    probe_pred = {(p["readout"], p["eval_set"]):
                  (p["prob"] >= p["threshold"]).astype(int) for p in p_}
    rows = []
    for k in names:
        targets = {"test": (views[k]["test"], labels["test"])}
        for name, (v, yy) in (extra or {}).items():
            targets[name] = (v[k], yy)
        lr_rows, lr_pred = fit_lr(views[k], labels, targets)
        fset, window = k.split(":")
        for lr in lr_rows:
            es = lr.pop("eval_set")
            pr = probe[(k, es)]
            yy = targets[es][1]
            p_lo, p_hi = boot_ci(yy, probe_pred[(k, es)], clusters.get(es),
                                 rng)
            l_lo, l_hi = boot_ci(yy, lr_pred[es], clusters.get(es), rng)
            rows.append({
                "eval_name": f"{task}:test" if es == "test" else es,
                "kind": "composition", "features": fset, "window": window,
                "n_features": views[k]["train"].shape[1], "n": 1,
                "n_eval": len(yy),
                "probe_mcc": pr["eval_mcc"], "probe_mcc_ci95_low": p_lo,
                "probe_mcc_ci95_high": p_hi, "probe_auroc": pr["eval_auroc"],
                "probe_val_mcc": pr["val_mcc"],
                "probe_threshold": pr["selected_threshold"],
                "lr_mcc": lr["lr_mcc"], "lr_mcc_ci95_low": l_lo,
                "lr_mcc_ci95_high": l_hi,
                **{c: lr[c] for c in ("lr_auroc", "lr_val_mcc",
                                      "lr_threshold", "lr_C", "lr_n_iter")}})
            print(rows[-1]["eval_name"], k, f"probe {pr['eval_mcc']:.3f}",
                  f"lr {lr['lr_mcc']:.3f}", flush=True)
    return rows


def reference_rows() -> list[dict]:
    cells = pd.read_csv(RES / "cells_mean_sd.csv")
    rows = []
    for ev in EVALS:
        for cond, readout in REFERENCES:
            c = cells[(cells.eval_name == ev) & (cells.condition == cond)]
            if readout is not None:
                c = c[c.readout == readout]
            if len(c) != 1:
                raise SystemExit(f"{len(c)} rows for {cond} on {ev} in "
                                 "cells_mean_sd.csv")
            c = c.iloc[0]
            row = {"eval_name": ev, "kind": "reference",
                   "features": f"{cond}/{c.readout}", "n": int(c.n)}
            if cond == "kmer_lr":
                k = int(c.readout.split("_")[1][0])
                row.update(n_features=int(jobs._canonical_kmer_index(k).max())
                           + 1, lr_mcc=c.mcc_mean, lr_auroc=c.auroc_mean)
            else:
                row.update(n_features=int(c.readout.removeprefix("subset")),
                           probe_mcc=c.mcc_mean, probe_mcc_sd=c.mcc_sd,
                           probe_auroc=c.auroc_mean,
                           probe_auroc_sd=c.auroc_sd)
            rows.append(row)
    return rows


def main() -> None:
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetPriorityClass(
            ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    torch.set_num_threads(THREADS)
    rates = damping_ladder()
    print("damping ladder", rates, flush=True)
    rng = np.random.default_rng(BOOT_SEED)
    rows = []
    for task in TASKS:
        rows += task_rows(task, rates, rng)
    df = pd.DataFrame(rows + reference_rows())
    cols = ["eval_name", "kind", "features", "window", "n_features", "n",
            "n_eval", "probe_mcc", "probe_mcc_sd", "probe_mcc_ci95_low",
            "probe_mcc_ci95_high", "probe_auroc", "probe_auroc_sd",
            "probe_val_mcc", "probe_threshold", "lr_mcc", "lr_mcc_ci95_low",
            "lr_mcc_ci95_high", "lr_auroc", "lr_val_mcc", "lr_threshold",
            "lr_C", "lr_n_iter"]
    df["eval_name"] = pd.Categorical(df["eval_name"], EVALS, ordered=True)
    df = df.sort_values("eval_name", kind="stable")[cols].astype(
        {"n_eval": "Int64", "lr_n_iter": "Int64"})
    RES.mkdir(parents=True, exist_ok=True)
    df.to_csv(RES / "composition_floor.csv", index=False)
    show = ["eval_name", "features", "window", "n", "probe_mcc",
            "probe_mcc_sd", "probe_mcc_ci95_low", "probe_mcc_ci95_high",
            "probe_auroc", "lr_mcc", "lr_mcc_ci95_low", "lr_mcc_ci95_high",
            "lr_auroc", "lr_C"]
    print(df[show].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
