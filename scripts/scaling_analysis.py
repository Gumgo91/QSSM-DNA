"""Register-size comparison (q = 4, 6) under one probe protocol (CPU).

Every register size is scored with the same probe code on the same data:
probes are fitted on the fixed 6,000-sequence subset of the promoter
training split (datasets.scaling_train_index, seed 1729), thresholds on the
validation split, and evaluated on the benchmark test split and the promoter
holdout, with all same-basis parities (and the 3q-1 local observables) of the
RC-averaged terminal state as readout. Features come from the caches of the
eval-encoder part: <key>/gb_promoter (full training split, subset here) or
<key>/gb_promoter_s6k (already the subset), plus <key>/holdout_promoter,
with <key> = qssm_<variant>[_q6]_s<seed>.

  CUDA_VISIBLE_DEVICES=-1 python scripts/scaling_analysis.py [<cache_root> ...]

The cache roots default to evaluate.CACHE_ROOT (runs/features unless
QSSM_DNA_FEATURES is set); several roots are searched in the order given.

Outputs: results/scaling_long.csv, results/scaling_contrasts.csv. Where an
evaluation run under the run roots (jobs.ROOTS) also scored the
gb_promoter_s6k task directly, its rows are compared with the recomputed
ones (max |difference| printed).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from qssm_dna.experiments import datasets, evaluate, jobs, stats  # noqa: E402

RES = Path("results")
SEEDS = {4: [22, 33, 44, 55, 66, 77, 88, 99, 110, 121],
         6: [22, 33, 44, 55, 66, 77, 88, 99, 110, 121]}
VARIANTS = ("allpos", "allpos_norzz", "random_init")


def key(variant: str, q: int, seed: int) -> str:
    return f"qssm_{variant}{'' if q == 4 else f'_q{q}'}_s{seed}"


def find(roots: list[Path], k: str) -> Path | None:
    for r in roots:
        for task in ("gb_promoter", "gb_promoter_s6k"):
            if (r / k / task / "train.npz").exists() and \
                    (r / k / "holdout_promoter" / "test.npz").exists():
                return r / k
    return None


def cache_rows(roots: list[Path]) -> pd.DataFrame:
    splits, _ = datasets.load("gb_promoter")
    idx = datasets.scaling_train_index(len(splits["train"]["y"]))
    labels = {"train": splits["train"]["y"][idx],
              "val": splits["val"]["y"], "test": splits["test"]["y"]}
    hs, _ = datasets.load("holdout_promoter")
    y_h = hs["test"]["y"]
    iso = np.load(datasets.DATA_DIR /
                  "holdout_promoter_chr9_chr10.npz")["isolated10kb"]
    pos_iso = (y_h == 0) | iso
    rows = []
    for q, seeds in SEEDS.items():
        for variant in VARIANTS:
            for s in seeds:
                d = find(roots, key(variant, q, s))
                if d is None:
                    print(f"missing cache {key(variant, q, s)}")
                    continue
                full = (d / "gb_promoter" / "train.npz").exists()
                task = d / ("gb_promoter" if full else "gb_promoter_s6k")
                per = {sp: evaluate.qssm_views(dict(np.load(
                    task / f"{sp}.npz"))) for sp in ("train", "val", "test")}
                if full:
                    per["train"] = {k: v[idx] for k, v in per["train"].items()}
                assert len(next(iter(per["train"].values()))) == len(idx)
                vh = evaluate.qssm_views(dict(np.load(
                    d / "holdout_promoter" / "test.npz")))
                readouts = [r for r in per["train"] if r != "pauli255"]
                views = {r: {sp: per[sp][r] for sp in per} for r in readouts}
                extra = {"holdout_promoter": ({r: vh[r] for r in readouts},
                                              y_h),
                         "holdout_isolated": ({r: vh[r][pos_iso]
                                               for r in readouts},
                                              y_h[pos_iso])}
                r_, _, _ = evaluate.evaluate_views(views, labels, extra)
                for r in r_:
                    r.update(variant=variant, q=q, seed=s,
                             condition=key(variant, q, s).rsplit("_s", 1)[0],
                             source=f"{task.parent.name}/{task.name}")
                rows += r_
    return pd.DataFrame(rows)


def check_against_evals(df: pd.DataFrame) -> None:
    ev = stats.collect(jobs.ROOTS)
    if ev.empty:
        return
    ev = ev[ev.task == "gb_promoter_s6k"]
    m = df.merge(ev, on=["condition", "seed", "readout", "eval_set"],
                 suffixes=("", "_eval"))
    if len(m):
        print(f"recomputed vs stored evaluations: {len(m)} rows, max |diff| "
              f"{np.abs(m.eval_mcc - m.eval_mcc_eval).max():.2e}")


def main() -> None:
    roots = [Path(a) for a in sys.argv[1:]] or [evaluate.CACHE_ROOT]
    df = cache_rows(roots)
    check_against_evals(df)
    keep = ["variant", "q", "seed", "condition", "readout", "eval_set",
            "eval_mcc", "eval_auroc", "val_mcc", "selected_threshold",
            "source"]
    df = df[keep].copy()
    df["readout_kind"] = np.where(df["readout"].str.startswith("subset"),
                                  "parities", "raw")
    RES.mkdir(parents=True, exist_ok=True)
    df.sort_values(["q", "variant", "seed", "readout", "eval_set"]).to_csv(
        RES / "scaling_long.csv", index=False)
    rows = []
    for (q, kind, es), g in df.groupby(["q", "readout_kind", "eval_set"]):
        w = g.pivot_table(index="seed", columns="variant", values="eval_mcc")
        for name, (x, y) in {"allpos_minus_init": ("allpos", "random_init"),
                             "allpos_minus_norzz": ("allpos",
                                                    "allpos_norzz")}.items():
            if x not in w or y not in w:
                continue
            pair = w[[x, y]].dropna()
            if len(pair) < 3:
                continue
            rows.append({"q": q, "readout_kind": kind, "eval_set": es,
                         "contrast": name,
                         "mean_a": pair[x].mean(), "sd_a": pair[x].std(),
                         "mean_b": pair[y].mean(), "sd_b": pair[y].std(),
                         **stats.paired_summary(pair[x].values,
                                                pair[y].values)})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "scaling_contrasts.csv", index=False)
    cells = (df.groupby(["q", "readout_kind", "eval_set", "variant"])
             ["eval_mcc"].agg(["mean", "std", "count"]).round(3))
    print(cells.to_string())
    if not out.empty:
        print(out[["q", "readout_kind", "eval_set", "contrast", "n",
                   "mean_diff", "ci95_low", "ci95_high", "p_perm",
                   "n_positive"]].round(3).to_string())


if __name__ == "__main__":
    main()
