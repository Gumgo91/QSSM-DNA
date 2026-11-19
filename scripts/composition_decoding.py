"""Composition read from the register, and what the register adds to
composition (CPU, two threads, below-normal priority).

  CUDA_VISIBLE_DEVICES=-1 python scripts/composition_decoding.py

Inputs. The cached terminal features <feature cache>/<condition>_s<seed>.npz
of scripts/pauli_features.py (feature cache: evaluate.CACHE_ROOT, by
default runs/features; 255 Pauli expectations of the terminal
four-qubit state, averaged over a sequence and its reverse complement;
subset30 = the columns subset30_cols, the 30 two-basis parities) for the
matched initialization, the endpoint, dense-tail-8 and all-position encoders
at both token budgets and the two reference registers init_zerogamma and
init_norzz, ten replication seeds. The sequences of the promoter benchmark
(train/val/test) and of the chromosome 9/10 promoter holdout come from
qssm_dna.experiments.datasets, which gives the row order of the caches. The run
stops unless the splits of a cache follow that order, the labels of every
split (the enhancer splits included) equal the labels of the loaded
sequences row by row, and every cache holds the same label and split
columns. Composition features are those of
scripts/composition_floor.py (right_aligned, gc_cpg, mono_di,
features): each is computed on a sequence and on its reverse complement and
the two are averaged; N is not counted and breaks a dinucleotide.

A. Decodability (results/composition_decoding.csv). For each
encoder, seed and readout (subset30, 30 features; pauli255, 255 features)
the features are standardized with the mean and SD of the promoter training
split and a ridge regression (scikit-learn Ridge: unpenalized intercept,
alpha times the sum of squared coefficients added to the residual sum of
squares) is fitted on the promoter training split for four targets:
  gc_all      GC fraction of the whole sequence
  cpg_oe_all  CpG observed/expected ratio (#CG x bases / (#C x #G)) of the
              whole sequence
  gc_last64   GC fraction of the last 64 bases of each direction (the 64
              bases nearest to the terminal position of the forward read and
              of the reverse-complement read, that is, the two ends of the
              sequence), mean of the two: the last64 window of
              composition_floor.py
  gc_last16   the same with 16 bases
alpha is chosen for each target on the promoter validation split from
1e-3 ... 1e2 by the validation R^2 (first maximum kept). R^2 = 1 - SS_res /
SS_tot about the mean of the scored split (sklearn.metrics.r2_score), on the
holdout and on the benchmark test split, which enter no choice. Levels:
  label_only  reference, n = 1: R^2 of the class means of the training split
              (what the promoter label alone explains of each target)
  seed        one row per encoder, seed, readout, target and evaluation set
  cell        mean, SD and 95 % t interval of R^2 over seeds
  contrast    R^2 of an encoder minus R^2 of qssm_random_init of the same
              seed and readout, paired over seeds
              (qssm_dna.experiments.stats.paired_summary: mean difference, 95 %
              t interval, exact sign-flip permutation p, seeds positive)

B. Increment over composition (results/composition_increment.csv).
Promoter labels and the standardized L2 logistic regression of
scripts/probe_robustness.py (fit_lr unchanged: lbfgs to
convergence, C from 1e-4 ... 1e2 by the validation MCC, threshold by the
rule of the paper), fitted on the promoter training split, selected on its
validation split and scored on the holdout and on the benchmark test split
(MCC and AUROC). Feature sets, all columns standardized:
  gc_cpg            GC fraction and CpG observed/expected ratio of the whole
                    sequence, 2 features
  subset30          the 30 parities of the register
  gc_cpg+subset30   32 features
  mono_di           mononucleotide and dinucleotide frequencies of the whole
                    sequence, 4 + 16 features
  mono_di+subset30  50 features
Levels:
  composition  the two composition-only fits (deterministic, n = 1)
  seed         one row per encoder, seed, feature set and evaluation set
  cell         mean, SD and 95 % t interval over seeds, MCC and AUROC
  contrast     paired over seeds, MCC and AUROC: gc_cpg+subset30 minus gc_cpg
               and mono_di+subset30 minus mono_di for every encoder (the
               second arm is the constant of the composition row, so the
               contrast is the per-seed combined value minus that constant
               and n_positive counts the seeds above it); subset30 of an
               encoder minus subset30 of qssm_random_init of the same seed
In both tables the columns ci95_low and ci95_high belong to the mean of a
cell row and to the mean difference of a contrast row. The intervals are
over replication seeds; the composition-only arm and the evaluation
sequences are fixed, so they carry no sampling error of the evaluation set.
The logistic regressions of part B use the C grid of composition_floor.py
(0.01 ... 10) by default, so that the composition-only arm equals the value
in composition_floor.csv; --c-grid probe_robustness refits all five feature
sets with 1e-4 ... 1e2 as a sensitivity run (write it elsewhere with
--out-dir). The column c_grid names the grid of a row.

Checks, printed (CHECK lines) and not written to the tables:
  1  row alignment of the labels (above)
  2  gc_cpg and mono_di of the whole sequence against the lr_* values of
     results/composition_floor.csv, with the logistic regression of
     composition_floor.py (its solver and its C grid 0.01 ... 10) and with
     the one of part B (C grid 1e-4 ... 1e2)
  3  the subset30 fits of part B against the lr_* seed rows of
     results/probe_robustness.csv

Also reads results/composition_floor.csv and results/probe_robustness.csv
(checks 2 and 3), so composition_floor.py and probe_robustness.py run first.
"""

from __future__ import annotations

import os

THREADS = 2
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = str(THREADS)

import argparse  # noqa: E402
import ctypes  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402


def lower_priority() -> int | None:
    """Below-normal priority class on Windows, typed as in
    scripts/pauli_features.py (with the default int conversion of the
    handle the call fails on 64-bit Python). Returns the class read back."""
    if sys.platform != "win32":
        return None
    k = ctypes.WinDLL("kernel32")
    k.GetCurrentProcess.restype = ctypes.c_void_p
    k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    k.GetPriorityClass.argtypes = [ctypes.c_void_p]
    k.GetPriorityClass.restype = ctypes.c_uint32
    if not k.SetPriorityClass(k.GetCurrentProcess(), 0x4000):
        raise OSError("could not lower the process priority")
    return int(k.GetPriorityClass(k.GetCurrentProcess()))


PRIORITY = lower_priority()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import composition_floor as cf  # noqa: E402
import probe_robustness as pr  # noqa: E402
from qssm_dna.experiments import datasets, stats  # noqa: E402

RES = Path("results")
INIT = pr.INIT
CONDS = pr.CONDS
SEEDS = pr.SEEDS
SETS = ("gb_promoter", "holdout_promoter", "gb_cohn")
PART = {"gb_promoter:train": "train", "gb_promoter:val": "val",
        "gb_promoter:test": "test",
        "holdout_promoter:test": "holdout_promoter"}
EVALS = {"holdout_promoter": "holdout_promoter", "test": "gb_promoter:test"}
READOUTS = ("subset30", "pauli255")
TARGETS = ("gc_all", "cpg_oe_all", "gc_last64", "gc_last16")
ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
COMPOSITION = ("gc_cpg", "mono_di")
FEATURE_SETS = ("subset30", "gc_cpg+subset30", "mono_di+subset30")
METRICS = ("mcc", "auroc")
PAIRED = ("mean_diff", "sd_diff", "ci95_low", "ci95_high", "t", "p_t",
          "p_perm", "p_wilcoxon", "n_positive")
CHECK_TOL = 5e-4
A_COLS = ["level", "eval_set", "condition", "seed", "readout", "target",
          "contrast", "first", "second", "n", "seeds", "n_features",
          "n_train", "n_val", "n_eval", "r2", "val_r2", "train_r2", "alpha",
          "mean", "sd", "mean_first", "mean_second", *PAIRED]
B_COLS = ["level", "eval_set", "condition", "seed", "features", "contrast",
          "first", "second", "metric", "n", "seeds", "n_features", "n_train",
          "n_val", "n_eval", "mcc", "auroc", "val_mcc", "threshold", "C",
          "n_iter", "grad_max", "mean", "sd", "mean_first", "mean_second",
          *PAIRED, "c_grid"]
C_GRIDS = {"probe_robustness": pr.C_GRID, "composition_floor": cf.C_GRID}
INT_COLS = ["seed", "n", "n_features", "n_train", "n_val", "n_eval",
            "n_iter", "n_positive"]


def load_sequences() -> dict[str, tuple[list[str], np.ndarray]]:
    """{"<dataset>:<part>": (sequences, labels)} in the order in which
    pauli_features.load_sets wrote the rows of the caches."""
    out = {}
    for name in SETS:
        parts, _ = datasets.load(name)
        for part, d in parts.items():
            out[f"{name}:{part}"] = (d["seqs"],
                                     np.asarray(d["y"]).astype(np.int64))
    return out


def check_alignment(y: np.ndarray, split: np.ndarray, seqs: dict
                    ) -> list[dict]:
    """Splits of a cache in dataset order, each one block of rows, with the
    labels of the loaded sequences row by row."""
    keys, first = np.unique(split, return_index=True)
    order = [str(k) for k in keys[np.argsort(first)]]
    if order != list(seqs):
        raise SystemExit(f"split order of the cache {order} differs from "
                         f"the datasets {list(seqs)}")
    rows = []
    for key, (_, lab) in seqs.items():
        idx = np.flatnonzero(split == key)
        block = bool(idx[-1] - idx[0] + 1 == len(idx))
        same = len(idx) == len(lab) and np.array_equal(y[idx], lab)
        rows.append({"split": key, "n_cache": len(idx), "n_loaded": len(lab),
                     "one_block": block, "labels_equal": bool(same),
                     "positives": int(lab.sum())})
        if not (block and same):
            raise SystemExit(f"rows of {key} do not align: {rows[-1]}")
    return rows


def gc_window(seqs: list[str], width: int) -> np.ndarray:
    """GC fraction of the last `width` bases of each direction, mean of the
    two (the window convention of composition_floor.features)."""
    return sum(0.5 * cf.gc_cpg(cf.right_aligned(seqs, rc)[:, -width:])[:, 0]
               for rc in (False, True))


def composition(seqs: list[str], rates: np.ndarray) -> dict[str, np.ndarray]:
    f = cf.features(seqs, rates)
    if not np.array_equal(gc_window(seqs, 64), f["gc_cpg:last64"][:, 0]):
        raise SystemExit("the window convention differs from "
                         "composition_floor.features")
    return {"gc_cpg": f["gc_cpg:all"], "mono_di": f["mono_di:all"],
            "targets": np.stack([f["gc_cpg:all"][:, 0],
                                 f["gc_cpg:all"][:, 1],
                                 f["gc_cpg:last64"][:, 0],
                                 gc_window(seqs, 16)], 1)}


def label_only_rows(targets: dict, labels: dict) -> list[dict]:
    """R^2 of the class means of the training split."""
    means = np.stack([targets["train"][labels["train"] == c].mean(0)
                      for c in (0, 1)])
    rows = []
    for part, ev in EVALS.items():
        r2 = r2_score(targets[part], means[labels[part]],
                      multioutput="raw_values")
        rows += [{"level": "label_only", "eval_set": ev, "target": t,
                  "n": 1, "n_features": 1, "n_train": len(labels["train"]),
                  "n_eval": len(labels[part]), "r2": float(r2[j])}
                 for j, t in enumerate(TARGETS)]
    return rows


def ridge_rows(x: dict, targets: dict) -> list[dict]:
    """Standardized ridge regression of every target; alpha by validation
    R^2 (first maximum kept)."""
    sc = StandardScaler().fit(x["train"])
    if not (sc.var_ > 0).all():
        raise RuntimeError("a feature is constant on the training split")
    xs = {k: sc.transform(v) for k, v in x.items()}
    best: list = [None] * len(TARGETS)
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha).fit(xs["train"], targets["train"])
        r2 = {k: r2_score(targets[k], model.predict(v),
                          multioutput="raw_values") for k, v in xs.items()}
        for j in range(len(TARGETS)):
            if best[j] is None or r2["val"][j] > best[j]["val_r2"]:
                best[j] = {"val_r2": float(r2["val"][j]), "alpha": alpha,
                           "train_r2": float(r2["train"][j]),
                           **{k: float(r2[k][j]) for k in EVALS}}
    rows = []
    for j, t in enumerate(TARGETS):
        for part, ev in EVALS.items():
            rows.append({"eval_set": ev, "target": t,
                         "n_features": x["train"].shape[1],
                         "n_train": len(x["train"]), "n_val": len(x["val"]),
                         "n_eval": len(x[part]), "r2": best[j][part],
                         **{k: best[j][k] for k in ("val_r2", "train_r2",
                                                    "alpha")}})
    return rows


def lr_rows(x: dict, labels: dict) -> list[dict]:
    """probe_robustness.fit_lr on one feature set of the promoter task."""
    fit = pr.fit_lr({k: x[k] for k in pr.PARTS}, labels,
                    {k: (x[k], labels[k]) for k in EVALS})
    return [{"eval_set": EVALS[k], "n_features": x["train"].shape[1],
             "n_train": len(labels["train"]), "n_val": len(labels["val"]),
             "n_eval": len(labels[k]),
             **{c.removeprefix("lr_"): v for c, v in r.items()}}
            for k, r in fit.items()]


def cell(x: np.ndarray, seeds: list[int]) -> dict:
    return {"n": len(x), "seeds": ",".join(map(str, seeds)),
            **pr.mean_interval(np.asarray(x, float))}


def paired(a: np.ndarray, b: np.ndarray, seeds: list[int]) -> dict:
    res = stats.paired_summary(a, b)
    return {"n": len(seeds), "seeds": ",".join(map(str, seeds)),
            "mean_first": float(np.mean(a)), "mean_second": float(np.mean(b)),
            **{k: res[k] for k in PAIRED}}


def summarize_a(seed_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    df = pd.DataFrame(seed_rows)
    key = ["eval_set", "readout", "target"]
    wide = df.pivot_table(index="seed", columns=key + ["condition"],
                          values="r2", sort=False)
    cells, contrasts = [], []
    for (ev, readout, target, cond), col in wide.items():
        col = col.dropna()
        seeds = [int(s) for s in col.index]
        cells.append({"level": "cell", "eval_set": ev, "condition": cond,
                      "readout": readout, "target": target,
                      **cell(col.to_numpy(), seeds)})
        if cond == INIT or (ev, readout, target, INIT) not in wide:
            continue
        ref = wide[(ev, readout, target, INIT)].dropna()
        seeds = [s for s in seeds if s in ref.index]
        contrasts.append({
            "level": "contrast", "eval_set": ev, "condition": cond,
            "readout": readout, "target": target,
            "contrast": f"{cond}_minus_{INIT}",
            "first": f"{cond}/{readout}", "second": f"{INIT}/{readout}",
            **paired(col[seeds].to_numpy(), ref[seeds].to_numpy(), seeds)})
    return cells, contrasts


def summarize_b(seed_rows: list[dict], comp_rows: list[dict]
                ) -> tuple[list[dict], list[dict]]:
    df = pd.DataFrame(seed_rows)
    const = {(r["eval_set"], r["features"]): r for r in comp_rows}
    cells, contrasts = [], []
    for metric in METRICS:
        wide = df.pivot_table(index="seed",
                              columns=["eval_set", "features", "condition"],
                              values=metric, sort=False)
        for (ev, fset, cond), col in wide.items():
            col = col.dropna()
            seeds = [int(s) for s in col.index]
            base = {"eval_set": ev, "condition": cond, "metric": metric}
            cells.append({"level": "cell", **base, "features": fset,
                          **cell(col.to_numpy(), seeds)})
            if "+" in fset:
                alone = fset.split("+")[0]
                c = const[(ev, alone)][metric]
                contrasts.append({
                    "level": "contrast", **base,
                    "contrast": f"{fset}_minus_{alone}",
                    "first": f"{cond}/{fset}", "second": alone,
                    **paired(col.to_numpy(), np.full(len(col), c), seeds)})
            elif cond != INIT and (ev, fset, INIT) in wide:
                ref = wide[(ev, fset, INIT)].dropna()
                seeds = [s for s in seeds if s in ref.index]
                contrasts.append({
                    "level": "contrast", **base,
                    "contrast": f"{fset}_minus_{INIT}_{fset}",
                    "first": f"{cond}/{fset}", "second": f"{INIT}/{fset}",
                    **paired(col[seeds].to_numpy(), ref[seeds].to_numpy(),
                             seeds)})
    return cells, contrasts


def write(rows: list[dict], cols: list[str], path: Path) -> pd.DataFrame:
    table = pd.DataFrame(rows).reindex(columns=cols)
    ints = [c for c in INT_COLS if c in cols]
    table[ints] = table[ints].astype("Int64")
    tmp = path.with_name(path.stem + ".partial.csv")
    table.to_csv(tmp, index=False)
    tmp.replace(path)
    return table


def check_composition(x: dict, labels: dict, comp_rows: list[dict]) -> None:
    """Check 2: composition_floor.csv, window all."""
    want = pr.read_csv(RES / "composition_floor.csv")
    want = want[(want.kind == "composition") & (want.window == "all")
                ].set_index(["eval_name", "features"])
    part_b = {(r["eval_set"], r["features"]): r for r in comp_rows}
    for fset in COMPOSITION:
        own, _ = cf.fit_lr(x[fset], labels,
                           {k: (x[fset][k], labels[k]) for k in EVALS})
        for r in own:
            ev = EVALS[r["eval_set"]]
            w, b = want.loc[(ev, fset)], part_b[(ev, fset)]
            d_own = max(abs(r["lr_mcc"] - w.lr_mcc),
                        abs(r["lr_auroc"] - w.lr_auroc))
            d_b = max(abs(b["mcc"] - w.lr_mcc), abs(b["auroc"] - w.lr_auroc))
            print(f"CHECK 2 {fset}/all {ev}: composition_floor.csv MCC "
                  f"{w.lr_mcc:.4f} AUROC {w.lr_auroc:.4f} (C={w.lr_C:g}) | "
                  f"composition_floor.fit_lr on these rows MCC "
                  f"{r['lr_mcc']:.4f} AUROC {r['lr_auroc']:.4f} "
                  f"(C={r['lr_C']:g}), max |diff| {d_own:.1e} "
                  f"{'PASS' if d_own < CHECK_TOL else 'FAIL'} | part B "
                  f"protocol MCC {b['mcc']:.4f} AUROC {b['auroc']:.4f} "
                  f"(C={b['C']:g}), max |diff| {d_b:.1e} "
                  f"{'PASS' if d_b < CHECK_TOL else 'DIFFERS'}", flush=True)


def check_subset30(seed_rows: list[dict]) -> None:
    """Check 3: probe_robustness.csv, lr_* of subset30."""
    want = pr.read_csv(RES / "probe_robustness.csv")
    want = want[(want.level == "seed") & (want.readout == "subset30")
                ].astype({"seed": int})[["eval_set", "condition", "seed",
                                         "lr_mcc", "lr_auroc"]]
    got = pd.DataFrame(seed_rows)
    got = got[got.features == "subset30"].merge(
        want, on=["eval_set", "condition", "seed"], how="left",
        validate="one_to_one")
    if got.lr_mcc.isna().any():
        print("CHECK 3: probe_robustness.csv lacks",
              int(got.lr_mcc.isna().sum()), "of the fitted cells", flush=True)
    got = got.dropna(subset=["lr_mcc"])
    d = np.maximum((got.mcc - got.lr_mcc).abs(),
                   (got.auroc - got.lr_auroc).abs())
    print(f"CHECK 3 subset30, {len(got)} seed cells of "
          f"{got.condition.nunique()} encoders against "
          f"probe_robustness.csv: max |diff| of MCC and AUROC "
          f"{d.max():.1e} {'PASS' if d.max() < CHECK_TOL else 'FAIL'}",
          flush=True)
    hold = got[got.eval_set == "holdout_promoter"].groupby(
        "condition", sort=False)
    for cond, g in hold:
        if cond in (INIT, "qssm_allpos_b2500000"):
            print(f"CHECK 3 {cond} holdout MCC, mean of {len(g)} seeds: "
                  f"{g.mcc.mean():.4f} here, {g.lr_mcc.mean():.4f} in "
                  f"probe_robustness.csv; per seed here "
                  f"{np.round(g.mcc.to_numpy(), 4).tolist()}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--conditions", default=",".join(CONDS))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--out-dir", default=str(RES),
                    help="directory of the two output tables")
    ap.add_argument("--c-grid", choices=list(C_GRIDS),
                    default="composition_floor",
                    help="C grid of the logistic regressions of part B")
    args = ap.parse_args()
    pr.C_GRID = C_GRIDS[args.c_grid]
    conds = args.conditions.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    warnings.simplefilter("error", ConvergenceWarning)
    pd.set_option("display.width", 250)
    print(f"threads {THREADS}, priority class {PRIORITY and hex(PRIORITY)} "
          f"(0x4000 = below normal), CUDA_VISIBLE_DEVICES="
          f"{os.environ['CUDA_VISIBLE_DEVICES']}; part B C grid "
          f"{args.c_grid} {pr.C_GRID}; tables to {out}", flush=True)

    seqs = load_sequences()
    lens = {k: sorted({len(s) for s in v[0]}) for k, v in seqs.items()}
    print("sequence lengths:", {k: (v[0], v[-1]) for k, v in lens.items()
                                if k in PART}, flush=True)
    rates = cf.damping_ladder()
    comp = {PART[k]: composition(seqs[k][0], rates) for k in PART}
    labels = {PART[k]: seqs[k][1] for k in PART}
    targets = {k: v["targets"] for k, v in comp.items()}
    x_comp = {f: {k: v[f] for k, v in comp.items()} for f in COMPOSITION}
    print("targets, mean (SD) on the holdout:", {
        t: f"{targets['holdout_promoter'][:, j].mean():.4f} "
           f"({targets['holdout_promoter'][:, j].std():.4f})"
        for j, t in enumerate(TARGETS)}, flush=True)

    label_rows = label_only_rows(targets, labels)
    comp_rows = [{"level": "composition", "features": f, "n": 1, **r}
                 for f in COMPOSITION for r in lr_rows(x_comp[f], labels)]
    for r in comp_rows:
        print(f"{r['features']} alone, {r['eval_set']}: MCC {r['mcc']:.4f} "
              f"AUROC {r['auroc']:.4f} (C={r['C']:g}, {r['n_iter']} it)",
              flush=True)
    check_composition(x_comp, labels, comp_rows)

    a_rows, b_rows, ref, aligned = [], [], None, 0
    for cond in conds:
        for seed in seeds:
            t0 = time.time()
            z = np.load(pr.CACHE / f"{cond}_s{seed}.npz")
            y, split = z["y"].astype(np.int64), z["split"]
            if ref is None:
                ref = (y, split)
                for r in check_alignment(y, split, seqs):
                    print("CHECK 1", r, flush=True)
            elif not (np.array_equal(y, ref[0])
                      and np.array_equal(split, ref[1])):
                raise SystemExit(f"{cond} seed {seed}: label or split column "
                                 "differs from the first cache")
            aligned += 1
            p = z["pauli255"].astype(np.float64)
            x255 = {part: p[split == k] for k, part in PART.items()}
            x30 = {k: v[:, z["subset30_cols"]] for k, v in x255.items()}
            if x30["train"].shape[1] != 30:
                raise SystemExit("subset30 does not have 30 columns")
            tag = {"level": "seed", "condition": cond, "seed": seed}
            new_a = []
            for readout, x in (("subset30", x30), ("pauli255", x255)):
                new_a += [{**tag, "readout": readout, **r}
                          for r in ridge_rows(x, targets)]
            a_rows += new_a
            new_b = []
            for fset in FEATURE_SETS:
                extra = fset.split("+")[0] if "+" in fset else None
                x = x30 if extra is None else {
                    k: np.concatenate([x_comp[extra][k], v], 1)
                    for k, v in x30.items()}
                new_b += [{**tag, "features": fset, **r}
                          for r in lr_rows(x, labels)]
            b_rows += new_b
            hold = {r["features"]: r for r in new_b
                    if r["eval_set"] == "holdout_promoter"}
            r2 = {(r["readout"], r["target"]): r["r2"] for r in new_a
                  if r["eval_set"] == "holdout_promoter"}
            print(cond, seed, "holdout R2 subset30",
                  " ".join(f"{t} {r2[('subset30', t)]:.3f}" for t in TARGETS),
                  "| holdout MCC", " ".join(f"{k} {r['mcc']:.3f}"
                                            for k, r in hold.items()),
                  f"| {time.time() - t0:.1f}s", flush=True)
    print(f"CHECK 1 label and split columns identical in {aligned} caches",
          flush=True)

    a_cells, a_con = summarize_a(a_rows)
    b_cells, b_con = summarize_b(b_rows, comp_rows)
    ta = write(label_rows + a_rows + a_cells + a_con, A_COLS,
               out / "composition_decoding.csv")
    tb = write([{**r, "c_grid": args.c_grid}
                for r in comp_rows + b_rows + b_cells + b_con], B_COLS,
               out / "composition_increment.csv")
    check_subset30(b_rows)

    fits = pd.DataFrame(b_rows)
    fits = fits[fits.eval_set == "holdout_promoter"]
    print(f"logistic regression: {len(fits)} selected fits, largest gradient "
          f"component {fits.grad_max.max():.1e}, at most "
          f"{fits.n_iter.max()} iterations; C chosen: "
          f"{fits.groupby('features').C.value_counts().to_dict()}",
          flush=True)
    ridge = pd.DataFrame(a_rows)
    ridge = ridge[ridge.eval_set == "holdout_promoter"]
    print("ridge alpha chosen:",
          ridge.groupby("readout").alpha.value_counts().to_dict(), flush=True)

    print("\nA. label_only R^2")
    print(ta[ta.level == "label_only"][["eval_set", "target", "r2"]]
          .round(4).to_string(index=False))
    print("\nA. R^2 over seeds (cell rows)")
    print(ta[ta.level == "cell"][["eval_set", "readout", "target",
                                  "condition", "n", "mean", "sd", "ci95_low",
                                  "ci95_high"]].round(4).to_string(
        index=False))
    print("\nA. encoder minus qssm_random_init (contrast rows)")
    print(ta[ta.level == "contrast"][[
        "eval_set", "readout", "target", "condition", "n", "mean_first",
        "mean_second", "mean_diff", "ci95_low", "ci95_high", "p_perm",
        "n_positive"]].round(4).to_string(index=False))
    print("\nB. composition alone")
    print(tb[tb.level == "composition"][["eval_set", "features", "mcc",
                                         "auroc", "val_mcc", "threshold",
                                         "C"]].round(4).to_string(
        index=False))
    print("\nB. means over seeds (cell rows)")
    print(tb[tb.level == "cell"][["eval_set", "metric", "features",
                                  "condition", "n", "mean", "sd", "ci95_low",
                                  "ci95_high"]].round(4).to_string(
        index=False))
    print("\nB. contrast rows")
    print(tb[tb.level == "contrast"][[
        "eval_set", "metric", "contrast", "condition", "n", "mean_first",
        "mean_second", "mean_diff", "ci95_low", "ci95_high", "p_perm",
        "n_positive"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
