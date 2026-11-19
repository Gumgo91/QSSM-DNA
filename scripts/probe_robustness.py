"""Probe robustness of the frozen-feature contrasts (CPU, two threads).

  CUDA_VISIBLE_DEVICES=-1 python scripts/probe_robustness.py --verify
  CUDA_VISIBLE_DEVICES=-1 python scripts/probe_robustness.py

Every probe of the promoter and enhancer comparisons is refitted on the
cached terminal features <feature cache>/<condition>_s<seed>.npz of
scripts/pauli_features.py (feature cache: evaluate.CACHE_ROOT, by default
runs/features; the raw11 and subset30 columns of pauli255) with two
protocols, for the matched initialization, the endpoint,
dense-tail-8 and all-position encoders at both token budgets and the two
reference registers init_zerogamma and init_norzz, ten replication seeds:

  probe_*  the probe of the paper, qssm_dna.experiments.evaluate unchanged
           (AdamW, 200 full-batch epochs, no standardization, probe seed 11,
           threshold = argmax validation MCC)
  lr_*     L2-regularised logistic regression (scikit-learn, lbfgs) on
           features standardized with the mean and SD of the probe's
           training split, solved to convergence: tolerance 1e-8, and the run
           stops if a fit reports no convergence or leaves a gradient
           component of the mean penalized loss above 1e-6. C is chosen on
           the validation split from 1e-4 ... 1e2 by the validation MCC at
           the validation threshold (first maximum kept), the threshold by
           the rule of the paper (qssm_dna.eval.metrics.pick_threshold).

A probe is fitted on the training split of the promoter benchmark and scored
on its test split and on the chromosome 9/10 promoter holdout, or fitted on
the enhancer training split and scored on the enhancer test split
(gb_cohn:test). The test splits and the holdout enter no choice.

The run starts by refitting the paper's probe on the cached raw11 features of
qssm_dense8 and qssm_endpoint and stops unless the paired holdout contrast
equals the C1_supervision row of results/primary_contrasts.csv to three
decimals; --verify ends after that step. Encoders whose rows are in the
output are skipped on a restart; --refit ignores the stored rows. The
contrast tables of make_results.py in results/ are read for the paper_*
columns.

Output: results/probe_robustness.csv, one table with three levels:
  seed      one row per encoder, seed, readout and evaluation set: MCC and
            AUROC of both protocols next to the stored value of the paper
            (paper_*, results/all_metrics_long.csv; the reference
            registers have none)
  cell      mean, SD and 95 % t interval over seeds of the seed rows for
            each protocol, and lr minus probe paired over seeds (shift_*)
  contrast  first minus second, paired over seeds
            (qssm_dna.experiments.stats.paired_summary), MCC and AUROC, for
            both protocols next to the row of the paper's contrast tables
            (paper_*; paper_source names the file and the contrast)
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time
import warnings
from pathlib import Path

THREADS = 2
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = str(THREADS)
# the handle is passed as a pointer: with the default int conversion the call
# fails on 64-bit Python (error 6) and the priority stays unchanged
if sys.platform == "win32":
    _K32 = ctypes.WinDLL("kernel32")
    _K32.GetCurrentProcess.restype = ctypes.c_void_p
    _K32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    if not _K32.SetPriorityClass(_K32.GetCurrentProcess(), 0x4000):
        raise OSError("could not lower the process priority")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.stats import t as student_t  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

torch.set_num_threads(THREADS)
sys.path.insert(0, "src")
from qssm_dna.eval.metrics import binary_metrics, pick_threshold  # noqa: E402
from qssm_dna.experiments import evaluate, stats  # noqa: E402

CACHE = evaluate.CACHE_ROOT
RES = Path("results")
OUT = RES / "probe_robustness.csv"
SEEDS = stats.SEEDS
B10 = "_b2500000"
INIT = "qssm_random_init"
TRAINED = ["qssm_endpoint", "qssm_dense8", "qssm_allpos",
           "qssm_endpoint" + B10, "qssm_dense8" + B10, "qssm_allpos" + B10]
CONDS = [INIT, *TRAINED, "init_zerogamma", "init_norzz"]
READOUTS = ("raw11", "subset30")
PARTS = ("train", "val", "test")
TASKS = ("gb_promoter", "gb_cohn")
EVALS = ["holdout_promoter", "gb_promoter:test", "gb_cohn:test"]
METRICS = {"eval_mcc": "mcc", "eval_auroc": "auroc"}
C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
TOL, MAX_ITER, GRAD_TOL = 1e-8, 100000, 1e-6
CHECK_TOL = 5e-4


def short(cond: str) -> str:
    name = cond.removeprefix("qssm_")
    return name.replace(B10, "_B10") if B10 in name else name + "_B1"


def read_csv(path: Path) -> pd.DataFrame:
    """Exact floats, so that copied values and a restart change no digit."""
    return pd.read_csv(path, float_precision="round_trip")


CONTRASTS = {
    "dense8_B1_vs_endpoint_B1_raw11": (("qssm_dense8", "raw11"),
                                       ("qssm_endpoint", "raw11")),
    **{f"{short(c)}_vs_init_subset30": ((c, "subset30"), (INIT, "subset30"))
       for c in TRAINED},
    "dense8_B10_vs_endpoint_B10_raw11": (("qssm_dense8" + B10, "raw11"),
                                         ("qssm_endpoint" + B10, "raw11")),
    "allpos_B1_vs_dense8_B10_subset30": (("qssm_allpos", "subset30"),
                                         ("qssm_dense8" + B10, "subset30")),
    "allpos_B1_vs_endpoint_B10_subset30": (
        ("qssm_allpos", "subset30"), ("qssm_endpoint" + B10, "subset30")),
}
PAPER_TABLES = ("primary_contrasts.csv", "dose_placement_contrasts.csv",
                "allpos_contrasts.csv", "budget_contrasts.csv",
                "cross_budget_contrasts.csv", "headline_auroc.csv")
PAIRED = ("mean_diff", "sd_diff", "ci95_low", "ci95_high", "t", "p_t",
          "p_perm", "p_wilcoxon", "n_positive")
PAPER = ("n", "mean_first", "mean_second") + PAIRED
ORDER = {"condition": CONDS, "readout": READOUTS, "eval_set": EVALS}
KEYS = ["eval_set", "condition", "seed", "readout"]
FIT_COLS = KEYS + [
    "n_features", "n_train", "n_val", "n_eval", "lr_mcc", "lr_auroc",
    "lr_val_mcc", "lr_threshold", "lr_C", "lr_n_iter", "lr_grad_max",
    "probe_mcc", "probe_auroc", "probe_val_mcc", "probe_threshold"]
COLUMNS = (
    ["level", "eval_set", "condition", "seed", "readout", "contrast",
     "first", "second", "metric", "n", "seeds"] + FIT_COLS[4:]
    + ["paper_mcc", "paper_auroc"]
    + [f"{p}_{k}" for p in ("lr", "probe", "paper")
       for k in ("mean", "sd", "mean_first", "mean_second", "mean_diff",
                 "sd_diff", "ci95_low", "ci95_high", "t", "p_t", "p_perm",
                 "p_wilcoxon", "n_positive")]
    + ["paper_n", "paper_source", "shift_mean_diff", "shift_ci95_low",
       "shift_ci95_high", "shift_p_perm", "shift_n_positive"])
INT_COLS = ["seed", "n", "n_features", "n_train", "n_val", "n_eval",
            "lr_n_iter", "lr_n_positive", "probe_n_positive",
            "paper_n_positive", "paper_n", "shift_n_positive"]


def load_features(cond: str, seed: int
                  ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    z = np.load(CACHE / f"{cond}_s{seed}.npz")
    p = z["pauli255"]
    return ({r: p[:, z[f"{r}_cols"]] for r in READOUTS},
            z["y"].astype(np.int64), z["split"])


def task_data(feats: dict, y: np.ndarray, split: np.ndarray, task: str
              ) -> tuple[dict, dict, dict | None]:
    """views[readout][part], labels[part] and the evaluation-only sets of
    one probe-training task, as evaluate.evaluate_views takes them."""
    rows = {k: split == f"{task}:{k}" for k in PARTS}
    views = {r: {k: f[m] for k, m in rows.items()} for r, f in feats.items()}
    labels = {k: y[m] for k, m in rows.items()}
    extra = None
    if task == "gb_promoter":
        hold = split == "holdout_promoter:test"
        extra = {"holdout_promoter": ({r: f[hold] for r, f in feats.items()},
                                      y[hold])}
    return views, labels, extra


def grad_max(clf, x: np.ndarray, y: np.ndarray, C: float) -> float:
    """Largest gradient component of the mean penalized loss at the fit."""
    r = clf.predict_proba(x)[:, 1] - y
    g = (x.T @ r + clf.coef_[0] / C) / len(y)
    return float(max(np.abs(g).max(), abs(r.mean())))


def fit_lr(v: dict, labels: dict, targets: dict) -> dict[str, dict]:
    """Standardized L2 logistic regression; C by validation MCC (first
    maximum kept), threshold by the rule of the paper."""
    sc = StandardScaler().fit(v["train"].astype(np.float64))
    if not (sc.var_ > 0).all():
        raise RuntimeError("a feature is constant on the training split")
    xt = sc.transform(v["train"].astype(np.float64))
    xv = sc.transform(v["val"].astype(np.float64))
    best = None
    for C in C_GRID:
        clf = LogisticRegression(C=C, solver="lbfgs", tol=TOL,
                                 max_iter=MAX_ITER).fit(xt, labels["train"])
        grad = grad_max(clf, xt, labels["train"], C)
        if int(clf.n_iter_[0]) >= MAX_ITER or grad > GRAD_TOL:
            raise RuntimeError(f"logistic regression not converged (C={C}, "
                               f"gradient {grad:.1e})")
        pv = clf.predict_proba(xv)[:, 1]
        thr = pick_threshold(labels["val"], pv)
        vm = binary_metrics(labels["val"], pv, thr)["mcc"]
        if best is None or vm > best[0]:
            best = (vm, C, clf, thr, grad)
    vm, C, clf, thr, grad = best
    out = {}
    for name, (z, yy) in targets.items():
        met = binary_metrics(
            yy, clf.predict_proba(sc.transform(z.astype(np.float64)))[:, 1],
            thr)
        out[name] = {"lr_mcc": met["mcc"], "lr_auroc": met["auroc"],
                     "lr_val_mcc": vm, "lr_threshold": thr, "lr_C": C,
                     "lr_n_iter": int(clf.n_iter_[0]), "lr_grad_max": grad}
    return out


def encoder_rows(cond: str, seed: int) -> list[dict]:
    feats, y, split = load_features(cond, seed)
    rows = []
    for task in TASKS:
        views, labels, extra = task_data(feats, y, split, task)
        fit, _, _ = evaluate.evaluate_views(views, labels, extra)
        probe = {(r["readout"], r["eval_set"]): r for r in fit}
        for readout, v in views.items():
            targets = {"test": (v["test"], labels["test"])}
            for name, (vv, yy) in (extra or {}).items():
                targets[name] = (vv[readout], yy)
            for name, lr in fit_lr(v, labels, targets).items():
                pr = probe[(readout, name)]
                rows.append({
                    "eval_set": f"{task}:test" if name == "test" else name,
                    "condition": cond, "seed": seed, "readout": readout,
                    "n_features": v["train"].shape[1],
                    "n_train": len(labels["train"]),
                    "n_val": len(labels["val"]),
                    "n_eval": len(targets[name][1]), **lr,
                    "probe_mcc": pr["eval_mcc"],
                    "probe_auroc": pr["eval_auroc"],
                    "probe_val_mcc": pr["val_mcc"],
                    "probe_threshold": pr["selected_threshold"]})
    return rows


def verify() -> None:
    want = read_csv(RES / "primary_contrasts.csv")
    want = want[(want.contrast == "C1_supervision")
                & (want.eval_set == "holdout_promoter")
                & (want.metric == "eval_mcc")].iloc[0]
    mcc = {}
    for cond in ("qssm_dense8", "qssm_endpoint"):
        mcc[cond] = []
        for seed in SEEDS:
            feats, y, split = load_features(cond, seed)
            views, labels, extra = task_data({"raw11": feats["raw11"]}, y,
                                             split, "gb_promoter")
            fit, _, _ = evaluate.evaluate_views(views, labels, extra)
            mcc[cond].append(next(r["eval_mcc"] for r in fit
                                  if r["eval_set"] == "holdout_promoter"))
    got = stats.paired_summary(mcc["qssm_dense8"], mcc["qssm_endpoint"])
    keys = ("mean_diff", "ci95_low", "ci95_high")
    line = "{:+.4f} [{:.4f}, {:.4f}]"
    print("paper's probe on the cached features, qssm_dense8 - qssm_endpoint"
          " (raw11, holdout_promoter, MCC):",
          line.format(*(got[k] for k in keys)), "| primary_contrasts.csv:",
          line.format(*(want[k] for k in keys)), flush=True)
    worst = max(abs(got[k] - want[k]) for k in keys)
    if worst > CHECK_TOL:
        raise SystemExit(f"the cached features do not reproduce "
                         f"C1_supervision "
                         f"(largest difference {worst:.4f})")


def stored_rows() -> list[dict]:
    if not OUT.exists():
        return []
    old = read_csv(OUT)
    old = old[old.level == "seed"][FIT_COLS]
    full = old.groupby(["condition", "seed"]).eval_set.transform("size") \
        == len(READOUTS) * len(EVALS)
    return old[full].astype({c: int for c in INT_COLS if c in FIT_COLS}
                            ).to_dict("records")


def write(rows: pd.DataFrame) -> None:
    table = rows.reindex(columns=COLUMNS)
    table[INT_COLS] = table[INT_COLS].astype("Int64")
    tmp = OUT.with_name(OUT.stem + ".partial.csv")
    table.to_csv(tmp, index=False)
    tmp.replace(OUT)


def seed_table(rows: list[dict]) -> pd.DataFrame:
    """Seed rows in table order with the stored values of the paper."""
    df = pd.DataFrame(rows)
    paper = read_csv(RES / "all_metrics_long.csv")
    paper = paper[paper.seed.isin(SEEDS) & paper.condition.isin(CONDS)
                  & paper.readout.isin(READOUTS)].rename(
        columns={"eval_set": "eval_part", "eval_name": "eval_set",
                 "eval_mcc": "paper_mcc", "eval_auroc": "paper_auroc"})
    paper = paper[KEYS + ["paper_mcc", "paper_auroc"]].astype({"seed": int})
    if paper.duplicated(KEYS).any():
        raise SystemExit("all_metrics_long.csv holds a cell twice")
    df = df.merge(paper, on=KEYS, how="left", validate="one_to_one")
    df = df.sort_values(
        ["condition", "seed", "readout", "eval_set"], kind="stable",
        key=lambda s: s.map(ORDER[s.name].index) if s.name in ORDER else s)
    df.insert(0, "level", "seed")
    return df.reset_index(drop=True)


def mean_interval(x: np.ndarray) -> dict:
    n, m, sd = len(x), float(np.mean(x)), float(np.std(x, ddof=1))
    half = float(student_t.ppf(0.975, n - 1)) * sd / np.sqrt(n)
    return {"mean": m, "sd": sd, "ci95_low": m - half, "ci95_high": m + half}


def cell_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    for (ev, cond, readout), g in df.groupby(
            ["eval_set", "condition", "readout"], sort=False):
        g = g.set_index("seed").loc[SEEDS]
        for metric, m in METRICS.items():
            row = {"level": "cell", "eval_set": ev, "condition": cond,
                   "readout": readout, "metric": metric, "n": len(g),
                   "seeds": ",".join(map(str, g.index))}
            for p in ("lr", "probe", "paper"):
                x = g[f"{p}_{m}"].to_numpy(float)
                if not np.isnan(x).any():
                    row.update({f"{p}_{k}": v
                                for k, v in mean_interval(x).items()})
            shift = stats.paired_summary(g[f"lr_{m}"], g[f"probe_{m}"])
            row.update({f"shift_{k}": shift[k] for k in (
                "mean_diff", "ci95_low", "ci95_high", "p_perm",
                "n_positive")})
            rows.append(row)
    return rows


def paper_contrast(tables: dict, ev: str, first: str, second: str,
                   metric: str) -> dict:
    for name, t in tables.items():
        hit = t[(t.eval_set == ev) & (t["first"] == first)
                & (t.second == second) & (t.metric == metric)]
        if len(hit):
            r = hit.iloc[0]
            return {**{f"paper_{k}": r[k] for k in PAPER},
                    "paper_source": f"{name}:{r.contrast}"}
    raise SystemExit(f"no paper row for {first} minus {second}, {metric}, "
                     f"{ev}")


def contrast_rows(df: pd.DataFrame) -> list[dict]:
    tables = {name: read_csv(RES / name) for name in PAPER_TABLES}
    cells = df.set_index(["eval_set", "condition", "readout",
                          "seed"]).sort_index()
    rows = []
    for ev in EVALS:
        for metric, m in METRICS.items():
            for name, ((c1, r1), (c2, r2)) in CONTRASTS.items():
                a, b = cells.loc[(ev, c1, r1)], cells.loc[(ev, c2, r2)]
                seeds = [s for s in SEEDS if s in a.index and s in b.index]
                first, second = f"{c1}/{r1}", f"{c2}/{r2}"
                row = {"level": "contrast", "eval_set": ev, "contrast": name,
                       "first": first, "second": second, "metric": metric,
                       "n": len(seeds), "seeds": ",".join(map(str, seeds))}
                for p in ("lr", "probe"):
                    x = a.loc[seeds, f"{p}_{m}"].to_numpy(float)
                    y = b.loc[seeds, f"{p}_{m}"].to_numpy(float)
                    res = stats.paired_summary(x, y)
                    row.update({f"{p}_mean_first": float(x.mean()),
                                f"{p}_mean_second": float(y.mean()),
                                **{f"{p}_{k}": res[k] for k in PAIRED}})
                row.update(paper_contrast(tables, ev, first, second, metric))
                rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--refit", action="store_true")
    args = ap.parse_args()
    warnings.simplefilter("error", ConvergenceWarning)
    verify()
    if args.verify:
        return
    rows = [] if args.refit else stored_rows()
    done = {(r["condition"], r["seed"]) for r in rows}
    print(f"{len(done)} encoders in {OUT} are kept", flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    for cond in CONDS:
        for seed in SEEDS:
            if (cond, seed) in done:
                continue
            t0 = time.time()
            new = encoder_rows(cond, seed)
            rows += new
            write(seed_table(rows))
            hold = {r["readout"]: r for r in new
                    if r["eval_set"] == "holdout_promoter"}
            print(cond, seed, " ".join(
                f"{k} holdout MCC lr {r['lr_mcc']:.3f} probe "
                f"{r['probe_mcc']:.3f} (C={r['lr_C']:g}, "
                f"{r['lr_n_iter']} it)" for k, r in hold.items()),
                f"{time.time() - t0:.1f}s", flush=True)
    df = seed_table(rows)
    cells, contrasts = cell_rows(df), contrast_rows(df)
    write(pd.concat([df, pd.DataFrame(cells), pd.DataFrame(contrasts)],
                    ignore_index=True))
    gap = df[["paper_mcc", "paper_auroc"]].to_numpy() \
        - df[["probe_mcc", "probe_auroc"]].to_numpy()
    print("paper's probe on the cached features against the stored values, "
          "max |difference| of MCC and AUROC:", np.nanmax(np.abs(gap), 0),
          flush=True)
    fits = df[df.eval_set != "holdout_promoter"]
    print(f"logistic regression: {len(fits)} selected fits, largest gradient "
          f"component {fits.lr_grad_max.max():.1e}, at most "
          f"{fits.lr_n_iter.max()} iterations; C chosen: "
          f"{fits.lr_C.value_counts().sort_index().to_dict()}", flush=True)
    show = ["eval_set", "contrast", "metric", "n", "lr_mean_diff",
            "lr_ci95_low", "lr_ci95_high", "lr_p_perm", "lr_n_positive",
            "paper_mean_diff", "paper_ci95_low", "paper_ci95_high",
            "paper_p_perm", "paper_n_positive"]
    print(pd.DataFrame(contrasts)[show].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
