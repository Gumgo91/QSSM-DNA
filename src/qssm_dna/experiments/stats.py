"""Aggregation of the experiment runs and paired statistics over seeds.

Builds one long table from the metrics.csv files of the experiment runs and
computes paired contrasts between (condition, readout) cells: mean
difference, 95 % t-interval, paired t-test, exact sign-flip permutation test
and Wilcoxon signed-rank test, with Holm adjustment within an evaluation set.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEEDS = [22, 33, 44, 55, 66, 77, 88, 99, 110, 121]
PRIMARY = {
    "C1_supervision": (("qssm_dense8", "raw11"), ("qssm_endpoint", "raw11")),
    "C2_readout": (("qssm_dense8", "subset30"), ("qssm_dense8", "raw11")),
    "C3_pretraining": (("qssm_dense8", "subset30"),
                       ("qssm_random_init", "subset30")),
    "C4_classical_gap": (("qssm_dense8", "subset30"),
                         ("gru64_dense8", "projected64")),
}


def collect(roots) -> pd.DataFrame:
    """Long table of the metrics.csv files of every finished experiment run
    under the roots (status.json with status DONE and a `part` entry), one
    row per condition x seed x task x eval_set x readout. The column `run`
    holds the run directory name."""
    frames = []
    for root in roots:
        for f in sorted(Path(root).glob("**/metrics.csv")):
            st = f.parent / "status.json"
            if not st.exists():
                continue
            status = json.loads(st.read_text())
            if status.get("status") != "DONE" or "part" not in status:
                continue
            df = pd.read_csv(f)
            df["run"] = f.parent.name
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # a job repeated after an interrupted queue writes an identical second
    # run directory; keep the earliest one per cell
    key = [c for c in ("condition", "seed", "task", "eval_set", "readout")
           if c in df.columns]
    df = (df.sort_values("run", kind="stable")
          .drop_duplicates(subset=key, keep="first")
          .reset_index(drop=True))
    df["eval_name"] = np.where(df["eval_set"] == "test",
                               df["task"] + ":test", df["eval_set"])
    # probes fitted on the 6,000-sequence subset form their own sets
    s6 = (df["task"] == "gb_promoter_s6k") & (df["eval_set"] != "test")
    df.loc[s6, "eval_name"] = "gb_promoter_s6k:" + df.loc[s6, "eval_set"]
    return df


def sign_flip_p(d: np.ndarray) -> float:
    """Exact two-sided sign-flip permutation p-value for the mean."""
    d = np.asarray(d, float)
    obs = abs(d.mean())
    n = len(d)
    signs = np.array(list(itertools.product((1, -1), repeat=n)))
    means = np.abs((signs * d).mean(1))
    return float((means >= obs - 1e-12).mean())


def paired_summary(a: np.ndarray, b: np.ndarray) -> dict:
    from scipy import stats
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    m, sd = float(d.mean()), float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, n - 1))
    t = m / se if se > 0 else np.inf
    p_t = float(2 * stats.t.sf(abs(t), n - 1)) if se > 0 else 0.0
    try:
        p_w = float(stats.wilcoxon(d).pvalue)
    except ValueError:
        p_w = float("nan")
    return {"n": n, "mean_diff": m, "sd_diff": sd,
            "ci95_low": m - tcrit * se, "ci95_high": m + tcrit * se,
            "t": float(t), "df": n - 1, "p_t": p_t,
            "p_perm": sign_flip_p(d), "p_wilcoxon": p_w,
            "n_positive": int((d > 0).sum())}


def holm(pvals: dict) -> dict:
    keys = sorted(pvals, key=lambda k: pvals[k])
    m = len(keys)
    out, run = {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * pvals[k]))
        out[k] = run
    return out


def cell(df: pd.DataFrame, cond: str, readout: str, eval_name: str,
         metric: str = "eval_mcc", seeds=SEEDS) -> pd.Series:
    sub = df[(df.condition == cond) & (df.readout == readout)
             & (df.eval_name == eval_name) & df.seed.isin(seeds)]
    s = sub.groupby("seed")[metric].first()
    return s


def contrasts(df: pd.DataFrame, pairs: dict, eval_names, metric="eval_mcc",
              seeds=SEEDS, adjust: bool = True) -> pd.DataFrame:
    rows = []
    for ev in eval_names:
        block = []
        for name, ((c1, r1), (c2, r2)) in pairs.items():
            a = cell(df, c1, r1, ev, metric, seeds)
            b = cell(df, c2, r2, ev, metric, seeds)
            common = sorted(set(a.index) & set(b.index))
            if len(common) < 2:
                continue
            res = paired_summary(a[common].values, b[common].values)
            block.append({"eval_set": ev, "contrast": name,
                          "first": f"{c1}/{r1}", "second": f"{c2}/{r2}",
                          "metric": metric,
                          "mean_first": float(a[common].mean()),
                          "mean_second": float(b[common].mean()),
                          "seeds": ",".join(map(str, common)), **res})
        if adjust and block:
            adj = holm({r["contrast"]: r["p_perm"] for r in block})
            for r in block:
                r["p_perm_holm"] = adj[r["contrast"]]
        rows += block
    return pd.DataFrame(rows)


def load_predictions(run: str | Path, roots=None) -> dict:
    """Arrays of <root>/**/<run>/predictions.npz, searched under the roots
    (default jobs.ROOTS); `run` is a run directory name or path."""
    if roots is None:
        from .jobs import ROOTS as roots
    name = Path(run).name
    for root in roots:
        hits = sorted(Path(root).glob(f"**/{name}/predictions.npz"))
        if hits:
            break
    else:
        raise FileNotFoundError(f"no predictions.npz for run {name!r} "
                                f"under {list(roots)}")
    z = np.load(hits[0], allow_pickle=False)
    return {k: z[k] for k in z.files if k != "meta"}
