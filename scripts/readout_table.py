"""Readout table of the frozen encoders from the cached Pauli features (CPU,
two threads).

  CUDA_VISIBLE_DEVICES=-1 python scripts/readout_table.py --verify
  CUDA_VISIBLE_DEVICES=-1 python scripts/readout_table.py

For each encoder cached by pauli_features.py (the initialization of the seed,
the endpoint, dense-tail-8 and all-position encoders at both token budgets,
and the reference registers init_zerogamma and init_norzz) and each
replication seed, linear probes are fitted on the raw11, subset30 and pauli255
columns of <feature cache>/<condition>_s<seed>.npz (evaluate.CACHE_ROOT, by
default runs/features) with the protocol of
qssm_dna.experiments.evaluate, unchanged (AdamW, 200 full-batch epochs, probe
seed 11, no standardization, threshold = argmax validation MCC). The promoter
probe is fitted on the promoter benchmark and scored on its test split and on
the chromosome 9/10 promoter holdout; the enhancer probe is fitted and scored
on human_enhancers_cohn.

The run starts with qssm_random_init, qssm_endpoint and qssm_dense8 and
compares their subset30 and pauli255 holdout MCC with
results/per_seed_mcc.csv; it stops if a per-seed value differs by
more than 0.01. --verify ends after that step. Encoders whose rows are in the
output are skipped on a restart.

Output:
  results/readout_full.csv
      one row per encoder, seed, readout and evaluation set: MCC, AUROC,
      selected threshold, validation MCC
  results/readout_contrasts.csv
      paired contrasts over seeds (stats.paired_summary) of MCC and AUROC on
      each evaluation set, Holm over the contrasts of a family:
        readout_width          pauli255 minus subset30, each of the seven
                               encoders
        pretraining_pauli255   trained encoder minus qssm_random_init
        pretraining_subset30   the same with subset30
        budget_vs_readout      ten-fold encoder read with subset30 minus
                               qssm_random_init read with pauli255
        supervision_pauli255   qssm_dense8 minus qssm_endpoint
        reference_subset30     init_zerogamma, init_norzz minus
                               qssm_random_init
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time
from pathlib import Path

THREADS = 2
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = str(THREADS)
if sys.platform == "win32":
    _K32 = ctypes.WinDLL("kernel32")
    _K32.GetCurrentProcess.restype = ctypes.c_void_p
    _K32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    if not _K32.SetPriorityClass(_K32.GetCurrentProcess(), 0x4000):
        raise OSError("could not lower the process priority")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(THREADS)
sys.path.insert(0, "src")
from qssm_dna.experiments import evaluate, stats  # noqa: E402

RES = Path("results")
OUT = RES / "readout_full.csv"
OUT_CONTRASTS = RES / "readout_contrasts.csv"
FEATURES = evaluate.CACHE_ROOT
SEEDS = stats.SEEDS
INIT = "qssm_random_init"
BASE = ("qssm_endpoint", "qssm_dense8", "qssm_allpos")
TENFOLD = tuple(f"{c}_b2500000" for c in BASE)
REFERENCE = ("init_zerogamma", "init_norzz")
ENCODERS = (INIT, *BASE, *TENFOLD)
CONDS = (*ENCODERS, *REFERENCE)
READOUTS = ("raw11", "subset30", "pauli255")
TASKS = ("gb_promoter", "gb_cohn")
PARTS = ("train", "val", "test")
EVALS = ["holdout_promoter", "gb_promoter:test", "gb_cohn:test"]
METRICS = ("eval_mcc", "eval_auroc")
VERIFY_CONDS = (INIT, "qssm_endpoint", "qssm_dense8")
VERIFY_READOUTS = ("subset30", "pauli255")
VERIFY_EVAL = "holdout_promoter"
CHECK_TOL = 0.01
KEYS = ["condition", "seed", "readout", "eval_name"]
COLUMNS = ["condition", "seed", "readout", "n_features", "task", "eval_set",
           "eval_name", "eval_n", "eval_mcc", "eval_auroc",
           "selected_threshold", "val_mcc", "probe_sha256", "features"]
ORDER = {"condition": {c: i for i, c in enumerate(CONDS)},
         "readout": {r: i for i, r in enumerate(READOUTS)},
         "eval_name": {e: i for i, e in enumerate(EVALS)}}
ROWS_PER_ENCODER = len(READOUTS) * len(EVALS)


def contrast_families() -> dict[str, dict]:
    def pair(a, ra, b, rb):
        return (a, ra), (b, rb)
    return {
        "readout_width": {
            f"{c}_pauli255_minus_subset30": pair(c, "pauli255", c, "subset30")
            for c in ENCODERS},
        "pretraining_pauli255": {
            f"{c}_minus_init_pauli255": pair(c, "pauli255", INIT, "pauli255")
            for c in BASE + TENFOLD},
        "pretraining_subset30": {
            f"{c}_minus_init_subset30": pair(c, "subset30", INIT, "subset30")
            for c in BASE + TENFOLD},
        "budget_vs_readout": {
            f"{c}_subset30_minus_init_pauli255":
                pair(c, "subset30", INIT, "pauli255") for c in TENFOLD},
        "supervision_pauli255": {
            "qssm_dense8_minus_qssm_endpoint_pauli255":
                pair("qssm_dense8", "pauli255", "qssm_endpoint", "pauli255")},
        "reference_subset30": {
            f"{c}_minus_init_subset30": pair(c, "subset30", INIT, "subset30")
            for c in REFERENCE},
    }


def probe_rows(cond: str, seed: int) -> list[dict]:
    path = FEATURES / f"{cond}_s{seed}.npz"
    z = np.load(path)
    y, split, pauli = z["y"].astype(np.int64), z["split"], z["pauli255"]
    feats = {"raw11": pauli[:, z["raw11_cols"]],
             "subset30": pauli[:, z["subset30_cols"]], "pauli255": pauli}
    rows = []
    for task in TASKS:
        views = {r: {k: f[split == f"{task}:{k}"] for k in PARTS}
                 for r, f in feats.items()}
        labels = {k: y[split == f"{task}:{k}"] for k in PARTS}
        extra = None
        if task == "gb_promoter":
            hold = split == "holdout_promoter:test"
            extra = {"holdout_promoter": (
                {r: f[hold] for r, f in feats.items()}, y[hold])}
        fit, _, _ = evaluate.evaluate_views(views, labels, extra)
        for r in fit:
            ev = r["eval_set"]
            rows.append({
                "condition": cond, "seed": seed, "readout": r["readout"],
                "n_features": feats[r["readout"]].shape[1], "task": task,
                "eval_set": ev,
                "eval_name": f"{task}:test" if ev == "test" else ev,
                "eval_n": r["eval_n"], "eval_mcc": r["eval_mcc"],
                "eval_auroc": r["eval_auroc"],
                "selected_threshold": r["selected_threshold"],
                "val_mcc": r["val_mcc"], "probe_sha256": r["probe_sha256"],
                "features": path.name})
    return rows


def ordered(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        ["condition", "seed", "eval_name", "readout"],
        key=lambda c: c.map(ORDER[c.name]) if c.name in ORDER else c,
        kind="stable").reset_index(drop=True)


def write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".partial.csv")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def previous() -> pd.DataFrame:
    if not OUT.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(OUT, float_precision="round_trip")[COLUMNS]
    size = df.groupby(["condition", "seed"]).size()
    done = {k for k, n in size.items() if n == ROWS_PER_ENCODER}
    return df[[k in done for k in zip(df.condition, df.seed)]]


def run(table: pd.DataFrame, conds) -> pd.DataFrame:
    done = set(zip(table.condition, table.seed))
    for cond in conds:
        for seed in SEEDS:
            if (cond, seed) in done:
                continue
            t0 = time.time()
            new = pd.DataFrame(probe_rows(cond, seed))[COLUMNS]
            table = pd.concat([table, new], ignore_index=True) \
                if len(table) else new
            write(table, OUT)
            hold = new[new.eval_name == VERIFY_EVAL].set_index("readout")
            print(f"{cond} s{seed} {time.time() - t0:.0f} s  holdout MCC "
                  + "  ".join(f"{r}={hold.eval_mcc[r]:.3f}"
                              for r in READOUTS), flush=True)
    return table


def stored_cells(table: pd.DataFrame) -> pd.DataFrame:
    """Rows of the table next to the per-seed MCC of the stored evaluations
    (results/per_seed_mcc.csv), for every cell held there."""
    ref = pd.read_csv(RES / "per_seed_mcc.csv",
                      float_precision="round_trip").melt(
        id_vars=["eval_name", "condition", "readout"], var_name="seed",
        value_name="stored_mcc").dropna()
    ref["seed"] = ref.seed.astype(float).astype(int)
    m = table.merge(ref, on=KEYS)
    m["abs_diff"] = (m.eval_mcc - m.stored_mcc).abs()
    return m


def verify(table: pd.DataFrame) -> None:
    m = stored_cells(table)
    m = m[m.condition.isin(VERIFY_CONDS) & m.readout.isin(VERIFY_READOUTS)
          & (m.eval_name == VERIFY_EVAL)]
    want = len(VERIFY_CONDS) * len(VERIFY_READOUTS) * len(SEEDS)
    if len(m) != want:
        raise SystemExit(f"{len(m)} of {want} verification cells found in "
                         "per_seed_mcc.csv and the readout table")
    cells = m.groupby(["readout", "condition"], sort=False).agg(
        n=("abs_diff", "size"), mean_mcc=("eval_mcc", "mean"),
        mean_stored=("stored_mcc", "mean"),
        max_abs_diff=("abs_diff", "max"))
    print(cells.to_string(float_format=lambda v: f"{v:.6g}"))
    for r, v in m.groupby("readout", sort=False).abs_diff.max().items():
        print(f"check: {VERIFY_EVAL} {r}, max |per-seed MCC - "
              f"per_seed_mcc.csv| = {v:.3g} (tolerance {CHECK_TOL})",
              flush=True)
    if m.abs_diff.max() > CHECK_TOL:
        raise SystemExit("per_seed_mcc.csv not reproduced from the cached "
                         "features; no contrasts written")


def contrasts(table: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for family, defs in contrast_families().items():
        for metric in METRICS:
            c = stats.contrasts(table, defs, EVALS, metric, SEEDS)
            if len(c) != len(defs) * len(EVALS) or (c.n != len(SEEDS)).any():
                raise RuntimeError(f"{family} {metric}: a contrast lacks "
                                   "seeds")
            c.insert(0, "family", family)
            frames.append(c)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="reproduce the per_seed_mcc.csv cells and stop")
    args = ap.parse_args()
    table = run(previous(), VERIFY_CONDS)
    verify(table)
    if args.verify:
        return
    table = ordered(run(table, CONDS))
    if len(table) != len(CONDS) * len(SEEDS) * ROWS_PER_ENCODER:
        raise RuntimeError("readout table is incomplete")
    write(table, OUT)
    con = contrasts(table)
    write(con, OUT_CONTRASTS)

    m = stored_cells(table)
    print("all cells of per_seed_mcc.csv, max |per-seed MCC difference|:")
    print(m.groupby(["eval_name", "readout"], sort=False).abs_diff
          .agg(["size", "max"]).to_string(float_format=lambda v: f"{v:.3g}"))
    cells = table.pivot_table(index=["condition", "readout"],
                              columns="eval_name", values=list(METRICS),
                              aggfunc=["mean", "std"], sort=False)
    print(cells.round(3).to_string())
    print(con[con.metric == "eval_mcc"][
        ["family", "eval_set", "contrast", "n", "mean_first", "mean_second",
         "mean_diff", "ci95_low", "ci95_high", "p_perm", "p_perm_holm",
         "n_positive"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
