"""Results tables of the paper from the run directories (CPU only).

  CUDA_VISIBLE_DEVICES=-1 python scripts/make_results.py

Inputs : under the run roots (qssm_dna.experiments.jobs.ROOTS: the entries of
         QSSM_DNA_RUNS, else runs/)
           evaluation runs: metrics.csv of every finished part
             (stats.collect)
           pretraining runs: metrics.json (next-base validation NLL,
             tokens, supervised targets, updates; encoders.scan_runs)
           noise runs (*_noise_*): noise_metrics.csv, noise_vs_length.csv
           the runtime run (*_runtime*): runtime.csv
         results/nll_large.csv (scripts/nll_large.py; held-out terminal
           loss used by across_conditions.csv)
Outputs: results/
           all_metrics_long.csv, cells_mean_sd.csv, per_seed_mcc.csv
           primary_contrasts.csv, dose_placement_contrasts.csv,
           ablation_contrasts.csv, allpos_contrasts.csv,
           budget_contrasts.csv, cross_budget_contrasts.csv,
           headline_auroc.csv, classical_frame.csv
           pretrain_nll.csv, nll_contrasts.csv, conditions_table.csv,
           reference_level.csv, across_conditions.csv
           noise_metrics.csv, noise_vs_length.csv (all noise runs,
             concatenated), shot_contrasts.csv
           runtime.csv

Every contrast is first minus second, paired over the replication seeds.
The enhancer set is gb_cohn:test. The column run of all_metrics_long.csv
names the evaluation run of each row. The process uses two numerical
threads and runs below normal priority.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

THREADS = 2
os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = str(THREADS)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402

sys.path.insert(0, "src")
from qssm_dna.experiments import encoders, jobs, stats  # noqa: E402

OUT = Path("results")
SEEDS = stats.SEEDS
EVALS = ["holdout_promoter", "holdout_isolated", "gb_promoter:test",
         "gb_cohn:test", "gb_ocr:test"]
B10 = "_b2500000"
METRICS = ("eval_mcc", "eval_auroc")
INIT = "qssm_random_init"
QSSM = [INIT, "qssm_endpoint", "qssm_dense2", "qssm_dense4", "qssm_dense8",
        "qssm_spread8", "qssm_allpos", "qssm_dense8_zerogamma",
        "qssm_dense8_staticgamma", "qssm_dense8_norzz", "qssm_allpos_norzz",
        "qssm_endpoint" + B10, "qssm_dense8" + B10, "qssm_allpos" + B10,
        "qssm_dense8_w256" + B10]
GRU = ["gru8_dense8", "gru8_allpos", "gru64_endpoint", "gru64_dense8",
       "gru64_allpos", "gru64_dense8" + B10]
ABLATED = ("_norzz", "_staticgamma", "_zerogamma")
CELLS = ["qssm_endpoint", "qssm_dense2", "qssm_dense4", "qssm_dense8",
         "qssm_spread8", "qssm_allpos", "qssm_endpoint" + B10,
         "qssm_dense8" + B10, "qssm_allpos" + B10, "qssm_dense8_w256" + B10]
READOUTS = ("subset30", "raw11", "projected64", "pauli255")
# noise runs started without a condition evaluated this encoder
NOISE_DEFAULT_CONDITION = "qssm_dense8"


def limit_resources() -> None:
    torch.set_num_threads(THREADS)
    if sys.platform == "win32":
        k = ctypes.WinDLL("kernel32")
        k.GetCurrentProcess.restype = ctypes.c_void_p
        k.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint)
        k.SetPriorityClass(k.GetCurrentProcess(), 0x4000)


def cell_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df[df.seed.isin(SEEDS) | df.seed.isna()].groupby(
        ["eval_name", "condition", "readout"])
    t = g.agg(n=("eval_mcc", "size"), mcc_mean=("eval_mcc", "mean"),
              mcc_sd=("eval_mcc", "std"), auroc_mean=("eval_auroc", "mean"),
              auroc_sd=("eval_auroc", "std")).reset_index()
    return t


def run_files(pattern: str) -> list[Path]:
    """Files matching `pattern` below the run roots, one per run directory
    (the first root that holds a run wins), in order of the run name."""
    found = {}
    for root in jobs.ROOTS:
        for f in sorted(Path(root).glob(pattern)):
            found.setdefault(f.parent.name, f)
    return [found[k] for k in sorted(found)]


def noise_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """noise_metrics.csv and noise_vs_length.csv of every noise run,
    concatenated, with a missing condition filled in."""
    files = run_files("**/*_noise_*/noise_metrics.csv")
    if not files:
        raise SystemExit(f"no noise runs under {jobs.ROOTS}")
    tables = []
    for name in ("noise_metrics.csv", "noise_vs_length.csv"):
        t = pd.concat([pd.read_csv(f.with_name(name)) for f in files],
                      ignore_index=True)
        if "condition" not in t.columns:
            t["condition"] = np.nan
        t["condition"] = t["condition"].fillna(NOISE_DEFAULT_CONDITION)
        tables.append(t)
    return tables[0], tables[1]


def runtime_table() -> pd.DataFrame | None:
    """runtime.csv of the runtime run (the latest one if there are
    several)."""
    files = run_files("**/*_runtime*/runtime.csv")
    if not files:
        return None
    if len(files) > 1:
        print(f"{len(files)} runtime runs; using {files[-1].parent.name}")
    return pd.read_csv(files[-1])


def nll_table() -> pd.DataFrame:
    refs = encoders.scan_runs(jobs.ROOTS)
    rows = [{"condition": r.condition, "seed": r.seed,
             "val_nll": r.metrics.get("val_nll"),
             "tokens": r.metrics.get("tokens_seen"),
             "targets": r.metrics.get("targets_seen"),
             "updates": r.metrics.get("steps"),
             "run_dir": str(r.run_dir)} for r in refs]
    df = pd.DataFrame(rows)
    # the same run directory can be present under several roots; keep the
    # earliest run per cell, as encoders.pick_run does, and check that
    # copies agree
    df["run_name"] = df.run_dir.map(lambda p: Path(p).name)
    spread = df.groupby(["condition", "seed"]).val_nll.agg(
        lambda v: v.max() - v.min())
    same = df.groupby(["condition", "seed"]).run_name.nunique() == 1
    bad = spread[same & (spread > 1e-12)]
    if len(bad):
        raise SystemExit(f"copies of one run disagree: {bad.to_dict()}")
    df = (df.sort_values(["run_name", "run_dir"])
          .drop_duplicates(["condition", "seed"], keep="first")
          .drop(columns="run_dir").rename(columns={"run_name": "run"}))
    return df.sort_values(["condition", "seed"])


def nll_contrasts(nll: pd.DataFrame, defs: dict) -> pd.DataFrame:
    rows = []
    for name, (a, b) in defs.items():
        x = nll[nll.condition == a].set_index("seed")["val_nll"]
        y = nll[nll.condition == b].set_index("seed")["val_nll"]
        common = [s for s in SEEDS if s in x.index and s in y.index]
        if len(common) < 2:
            continue
        rows.append({"contrast": name, "first": a, "second": b,
                     "mean_first": float(x[common].mean()),
                     "mean_second": float(y[common].mean()),
                     **stats.paired_summary(x[common].values,
                                            y[common].values)})
    return pd.DataFrame(rows)


def slug(name: str) -> str:
    return name.replace(":", "_").replace("eval_", "")


def both_metrics(df: pd.DataFrame, defs: dict) -> pd.DataFrame:
    return pd.concat([stats.contrasts(df, defs, EVALS, m, adjust=False)
                      for m in METRICS], ignore_index=True)


def pretrain_means(nll: pd.DataFrame) -> pd.DataFrame:
    n = nll[nll.seed.isin(SEEDS)].copy()
    n["per_update"] = n.targets / n.updates
    return n.groupby("condition").agg(
        n_pretrain_runs=("seed", "size"), tokens_mean=("tokens", "mean"),
        updates_mean=("updates", "mean"), targets_mean=("targets", "mean"),
        targets_per_update_mean=("per_update", "mean"),
        val_nll_mean=("val_nll", "mean"), val_nll_sd=("val_nll", "std"))


def cross_budget(df: pd.DataFrame, pre: pd.DataFrame) -> pd.DataFrame:
    pairs = {
        "allpos_B1_vs_dense8_B10": ("qssm_allpos", "qssm_dense8" + B10),
        "allpos_B1_vs_endpoint_B10": ("qssm_allpos", "qssm_endpoint" + B10),
        "dense8_B1_vs_endpoint_B10": ("qssm_dense8", "qssm_endpoint" + B10),
        "dense4_B1_vs_endpoint_B10": ("qssm_dense4", "qssm_endpoint" + B10),
        "allpos_B10_vs_dense8_B10": ("qssm_allpos" + B10,
                                     "qssm_dense8" + B10),
        "endpoint_B10_vs_endpoint_B1": ("qssm_endpoint" + B10,
                                        "qssm_endpoint"),
        "dense8_w256_B10_vs_dense8_B10": ("qssm_dense8_w256" + B10,
                                          "qssm_dense8" + B10),
    }
    defs = {f"{name}_{r}": ((a, r), (b, r))
            for name, (a, b) in pairs.items() for r in ("subset30", "raw11")}
    out = both_metrics(df, defs)
    for side in ("first", "second"):
        cond = out[side].str.split("/").str[0]
        out[f"tokens_{side}"] = cond.map(pre.tokens_mean)
        out[f"targets_{side}"] = cond.map(pre.targets_mean)
    return out


def shot_contrasts(nm: pd.DataFrame) -> pd.DataFrame:
    """Finite shots against exact expectations for the base-budget encoders:
    paired difference over seeds of the holdout MCC (1,024 shots minus
    exact; the per-seed shot value is the mean over the shot draws), with
    the probe fitted on noiseless features and no hardware noise."""
    nm = nm.copy()
    nm["shots"] = nm["shots"].astype(str)
    nm = nm[(nm.setting == "none") & (nm.policy == "A_noiseless_probe")
            & (nm.eval_set == "holdout_promoter") & nm.seed.isin(SEEDS)]
    rows = []
    for (cond, readout), g in nm.groupby(["condition", "readout"]):
        for metric in METRICS:
            a = g[g.shots == "1024"].groupby("seed")[metric].mean()
            b = g[g.shots == "exact"].groupby("seed")[metric].mean()
            seeds = [s for s in SEEDS if s in a.index and s in b.index]
            if len(seeds) < 2:
                continue
            rows.append({"condition": cond, "readout": readout,
                         "eval_set": "holdout_promoter", "metric": metric,
                         "first": "1024 shots", "second": "exact",
                         "seeds": ",".join(map(str, seeds)),
                         "mean_first": float(a[seeds].mean()),
                         "sd_first": float(a[seeds].std(ddof=1)),
                         "mean_second": float(b[seeds].mean()),
                         "sd_second": float(b[seeds].std(ddof=1)),
                         **stats.paired_summary(a[seeds].values,
                                                b[seeds].values)})
    return pd.DataFrame(rows)


def headline_auroc(df: pd.DataFrame, def_sets: dict) -> pd.DataFrame:
    parts = []
    for source, defs in def_sets.items():
        t = stats.contrasts(df, defs, EVALS, "eval_auroc", adjust=False)
        t.insert(0, "source", source)
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


def conditions_table(df: pd.DataFrame, pre: pd.DataFrame) -> pd.DataFrame:
    lift = both_metrics(df, {c: ((c, "subset30"), (INIT, "subset30"))
                             for c in QSSM[1:]}).set_index(
        ["contrast", "eval_set", "metric"])
    p = pre.reindex(QSSM + GRU).fillna({"n_pretrain_runs": 0})
    rows = []
    for c in QSSM + GRU:
        family = "qssm" if c in QSSM else "gru"
        r = "subset30" if family == "qssm" else "projected64"
        row = {"condition": c, "family": family, "readout": r,
               "n_seeds": len(stats.cell(df, c, r, "holdout_promoter")),
               **p.loc[c].to_dict()}
        for metric in METRICS:
            for ev in ("holdout_promoter", "gb_cohn:test"):
                s = stats.cell(df, c, r, ev, metric)
                row[f"{slug(ev)}_{slug(metric)}_mean"] = float(s.mean())
                row[f"{slug(ev)}_{slug(metric)}_sd"] = float(s.std())
        for metric in METRICS:
            for ev in EVALS:
                d = (lift.loc[(c, ev, metric)]
                     if (c, ev, metric) in lift.index else {})
                tag = f"lift_{slug(ev)}_{slug(metric)}"
                row.update({tag: d.get("mean_diff"),
                            f"{tag}_ci95_low": d.get("ci95_low"),
                            f"{tag}_ci95_high": d.get("ci95_high"),
                            f"{tag}_p_perm": d.get("p_perm"),
                            f"{tag}_n_positive": d.get("n_positive")})
        rows.append(row)
    out = pd.DataFrame(rows)
    counts = [c for c in out.columns
              if c.endswith("_n_positive") or c == "n_pretrain_runs"]
    out[counts] = out[counts].astype("Int64")
    return out


def reference_level(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in QSSM:
        for metric in METRICS:
            for ev in EVALS:
                for r in READOUTS:
                    s = stats.cell(df, c, r, ev, metric)
                    if s.empty:
                        continue
                    row = {"condition": c, "readout": r, "eval_set": ev,
                           "metric": metric, "n": len(s),
                           "mean": float(s.mean()), "sd": float(s.std()),
                           "min": float(s.min()), "max": float(s.max()),
                           **{f"seed_{k}": v
                              for k, v in s.reindex(SEEDS).items()}}
                    ref = stats.cell(df, INIT, r, ev, metric)
                    common = sorted(set(s.index) & set(ref.index))
                    if c != INIT and len(common) > 2:
                        res = pearsonr(s[common].values, ref[common].values)
                        lo, hi = res.confidence_interval(0.95)
                        row.update({
                            "n_pairs": len(common),
                            "pearson_r_with_init": float(res.statistic),
                            "pearson_ci95_low": float(lo),
                            "pearson_ci95_high": float(hi),
                            "pearson_p": float(res.pvalue),
                            "init_architecture": "ablated" if any(
                                a in c for a in ABLATED) else "same"})
                    rows.append(row)
    out = pd.DataFrame(rows)
    out["n_pairs"] = out["n_pairs"].astype("Int64")
    return out


def classical_frame(df: pd.DataFrame) -> pd.DataFrame:
    versus = {}
    for c in CELLS:
        obj = c.removeprefix("qssm_")
        for h in (8, 64):
            for rq, rg in (("subset30", "projected64"), ("raw11", "hidden")):
                versus[f"qssm_vs_gru{h}_{obj.replace(B10, '_B10')}_{rq}"] = (
                    (c, rq), (f"gru{h}_{obj}", rg))
    pairs = {"gru64_dense8_vs_endpoint": ("gru64_dense8", "gru64_endpoint"),
             "gru64_allpos_vs_endpoint": ("gru64_allpos", "gru64_endpoint"),
             "gru64_allpos_vs_dense8": ("gru64_allpos", "gru64_dense8"),
             "gru8_allpos_vs_dense8": ("gru8_allpos", "gru8_dense8"),
             "gru64_dense8_B10_vs_B1": ("gru64_dense8" + B10, "gru64_dense8")}
    dose = {f"{name}_{r}": ((a, r), (b, r))
            for name, (a, b) in pairs.items()
            for r in ("projected64", "hidden")}
    parts = []
    for group, defs in (("qssm_vs_gru", versus), ("gru_dose", dose)):
        t = both_metrics(df, defs)
        t.insert(0, "group", group)
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


def across_conditions(df: pd.DataFrame, pre: pd.DataFrame) -> pd.DataFrame:
    p = pre.loc[CELLS]
    # terminal next-base loss on the 10,240 held-out 64/128-input windows of
    # nll_large.py, the same windows for every cell; the loss logged in
    # training used 128 windows whose length follows the training windows
    path = OUT / "nll_large.csv"
    if not path.exists():
        raise RuntimeError("run scripts/nll_large.py first: "
                           f"{path} is missing")
    held = pd.read_csv(path)
    held = held[(held.window_set == "L64_128") & held.seed.isin(SEEDS)
                & held.condition.isin(CELLS)]
    if held.condition.nunique() != len(CELLS) or not (
            held.groupby("condition").seed.nunique() == len(SEEDS)).all():
        raise RuntimeError("nll_large.csv does not hold every cell and seed")
    held = held.groupby("condition").nll_terminal.mean()
    design = pd.DataFrame({
        "condition": CELLS,
        "log10_targets_total": np.log10(p.targets_mean.values),
        "log10_targets_per_update": np.log10(
            p.targets_per_update_mean.values),
        "log10_tokens": np.log10(p.tokens_mean.values),
        "updates": p.updates_mean.values,
        "heldout_nll_terminal_mean": held.loc[CELLS].values})
    blocks = []
    for metric in METRICS:
        for ev in EVALS:
            for r in ("subset30", "raw11"):
                lift = stats.contrasts(
                    df, {c: ((c, r), (INIT, r)) for c in CELLS}, [ev],
                    metric, adjust=False).set_index("contrast").loc[CELLS]
                t = design.copy()
                t.insert(1, "mean_score", lift.mean_first.values)
                t.insert(2, "lift_over_init_mean", lift.mean_diff.values)
                foot = {"pearson_r_with_mean_score": {}, "pearson_p": {}}
                for k in t.columns[2:]:
                    res = pearsonr(t.mean_score, t[k])
                    foot["pearson_r_with_mean_score"][k] = float(
                        res.statistic)
                    foot["pearson_p"][k] = float(res.pvalue)
                t = pd.concat([t, pd.DataFrame(
                    [{"condition": k, **v} for k, v in foot.items()])],
                    ignore_index=True)
                t.insert(0, "eval_set", ev)
                t.insert(1, "metric", metric)
                t.insert(2, "readout", r)
                t.insert(3, "n_cells", len(CELLS))
                blocks.append(t)
    return pd.concat(blocks, ignore_index=True)


def main() -> None:
    limit_resources()
    OUT.mkdir(parents=True, exist_ok=True)
    df = stats.collect(jobs.ROOTS)
    df.to_csv(OUT / "all_metrics_long.csv", index=False)
    cell_table(df).to_csv(OUT / "cells_mean_sd.csv", index=False)
    per_seed = df[df.seed.isin(SEEDS + [11])].pivot_table(
        index=["eval_name", "condition", "readout"], columns="seed",
        values="eval_mcc")
    per_seed.to_csv(OUT / "per_seed_mcc.csv")

    out = []
    for metric in ("eval_mcc", "eval_auroc"):
        out.append(stats.contrasts(df, stats.PRIMARY, EVALS, metric))
    pd.concat(out).to_csv(OUT / "primary_contrasts.csv", index=False)

    q = lambda c, r="raw11": (c, r)  # noqa: E731
    dose = {f"{c}_vs_endpoint": (q(c), q("qssm_endpoint"))
            for c in ("qssm_dense2", "qssm_dense4", "qssm_dense8",
                      "qssm_allpos", "qssm_spread8")}
    dose["dense8_vs_spread8"] = (q("qssm_dense8"), q("qssm_spread8"))
    dose["allpos_vs_dense8"] = (q("qssm_allpos"), q("qssm_dense8"))
    stats.contrasts(df, dose, EVALS, adjust=False).to_csv(
        OUT / "dose_placement_contrasts.csv", index=False)

    abl = {}
    for c in ("qssm_dense8_norzz", "qssm_dense8_staticgamma",
              "qssm_dense8_zerogamma"):
        for r in ("raw11", "subset30"):
            abl[f"{c}_{r}_vs_dense8"] = ((c, r), ("qssm_dense8", r))
    abl["gru8_vs_qssm_dense8_subset30"] = (("gru8_dense8", "projected64"),
                                           ("qssm_dense8", "subset30"))
    abl["gru8_hidden_vs_qssm_dense8_raw11"] = (("gru8_dense8", "hidden"),
                                               ("qssm_dense8", "raw11"))
    abl["gru64_dense8_vs_endpoint"] = (("gru64_dense8", "projected64"),
                                       ("gru64_endpoint", "projected64"))
    abl["projected64_vs_raw11_dense8"] = (("qssm_dense8", "projected64"),
                                          ("qssm_dense8", "raw11"))
    abl["pauli255_vs_subset30_dense8"] = (("qssm_dense8", "pauli255"),
                                          ("qssm_dense8", "subset30"))
    abl["pauli255_dense8_vs_init"] = (("qssm_dense8", "pauli255"),
                                      ("qssm_random_init", "pauli255"))
    stats.contrasts(df, abl, EVALS, adjust=False).to_csv(
        OUT / "ablation_contrasts.csv", index=False)

    # all-position comparisons
    allp = {
        "qssm_allpos_vs_init": (("qssm_allpos", "subset30"),
                                ("qssm_random_init", "subset30")),
        "qssm_endpoint_vs_init": (("qssm_endpoint", "subset30"),
                                  ("qssm_random_init", "subset30")),
        "qssm_allpos_vs_dense8": (("qssm_allpos", "subset30"),
                                  ("qssm_dense8", "subset30")),
        "qssm_allpos_vs_gru64_allpos": (("qssm_allpos", "subset30"),
                                        ("gru64_allpos", "projected64")),
        "qssm_allpos_vs_gru8_allpos": (("qssm_allpos", "subset30"),
                                       ("gru8_allpos", "projected64")),
        "gru64_allpos_vs_endpoint": (("gru64_allpos", "projected64"),
                                     ("gru64_endpoint", "projected64")),
        "gru8_allpos_vs_dense8": (("gru8_allpos", "projected64"),
                                  ("gru8_dense8", "projected64")),
        "qssm_allpos_norzz_vs_allpos": (("qssm_allpos_norzz", "subset30"),
                                        ("qssm_allpos", "subset30")),
    }
    stats.contrasts(df, allp, EVALS, adjust=False).to_csv(
        OUT / "allpos_contrasts.csv", index=False)

    long = {
        "dense8_vs_endpoint_B10": (("qssm_dense8" + B10, "raw11"),
                                   ("qssm_endpoint" + B10, "raw11")),
        "dense8_B10_vs_init": (("qssm_dense8" + B10, "subset30"),
                               ("qssm_random_init", "subset30")),
        "allpos_B10_vs_init": (("qssm_allpos" + B10, "subset30"),
                               ("qssm_random_init", "subset30")),
        "dense8_w256_B10_vs_init": (("qssm_dense8_w256" + B10, "subset30"),
                                    ("qssm_random_init", "subset30")),
        "endpoint_B10_vs_init": (("qssm_endpoint" + B10, "subset30"),
                                 ("qssm_random_init", "subset30")),
        "dense8_B10_vs_B1": (("qssm_dense8" + B10, "subset30"),
                             ("qssm_dense8", "subset30")),
        "gru64_B10_vs_B1": (("gru64_dense8" + B10, "projected64"),
                            ("gru64_dense8", "projected64")),
        # C4_classical_gap at the ten-fold budget
        "dense8_B10_vs_gru64_B10": (("qssm_dense8" + B10, "subset30"),
                                    ("gru64_dense8" + B10, "projected64")),
    }
    stats.contrasts(df, long, EVALS, adjust=False).to_csv(
        OUT / "budget_contrasts.csv", index=False)

    nll = nll_table()
    nll.to_csv(OUT / "pretrain_nll.csv", index=False)
    ndefs = {f"{c}_vs_endpoint": (c, "qssm_endpoint")
             for c in ("qssm_dense2", "qssm_dense4", "qssm_dense8",
                       "qssm_allpos", "qssm_spread8")}
    ndefs["dense8_vs_endpoint_B10"] = ("qssm_dense8" + B10,
                                       "qssm_endpoint" + B10)
    ndefs["gru64_dense8_vs_endpoint"] = ("gru64_dense8", "gru64_endpoint")
    ndefs["endpoint_B10_vs_endpoint"] = ("qssm_endpoint" + B10,
                                         "qssm_endpoint")
    ndefs["qssm_dense4_vs_endpoint_B10"] = ("qssm_dense4",
                                            "qssm_endpoint" + B10)
    ndefs["qssm_allpos_vs_endpoint_B10"] = ("qssm_allpos",
                                            "qssm_endpoint" + B10)
    ndefs["qssm_dense8_zerogamma_vs_dense8"] = ("qssm_dense8_zerogamma",
                                                "qssm_dense8")
    ndefs["qssm_allpos_norzz_vs_allpos"] = ("qssm_allpos_norzz",
                                            "qssm_allpos")
    nll_contrasts(nll, ndefs).to_csv(OUT / "nll_contrasts.csv", index=False)

    pre = pretrain_means(nll)
    cross_budget(df, pre).to_csv(OUT / "cross_budget_contrasts.csv",
                                 index=False)
    headline_auroc(df, {"allpos_contrasts": allp, "budget_contrasts": long}
                   ).to_csv(OUT / "headline_auroc.csv", index=False)
    conditions_table(df, pre).to_csv(OUT / "conditions_table.csv",
                                     index=False)
    reference_level(df).to_csv(OUT / "reference_level.csv", index=False)
    classical_frame(df).to_csv(OUT / "classical_frame.csv", index=False)
    across_conditions(df, pre).to_csv(OUT / "across_conditions.csv",
                                      index=False)
    noise_metrics, noise_length = noise_tables()
    noise_metrics.to_csv(OUT / "noise_metrics.csv", index=False)
    noise_length.to_csv(OUT / "noise_vs_length.csv", index=False)
    shot_contrasts(noise_metrics).to_csv(OUT / "shot_contrasts.csv",
                                         index=False)
    runtime = runtime_table()
    if runtime is None:
        print(f"no runtime run under {jobs.ROOTS}; runtime.csv not written")
    else:
        runtime.to_csv(OUT / "runtime.csv", index=False)
    print(f"tables -> {OUT} ({len(df)} metric rows from "
          f"{len(jobs.ROOTS)} roots)")


if __name__ == "__main__":
    main()
