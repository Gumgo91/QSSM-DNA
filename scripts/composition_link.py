"""Link between the composition a register carries and its promoter transfer.

  python scripts/composition_link.py

For every register and replication seed, pairs
  r2   holdout R^2 of the ridge regression from the 30 same-basis parities
       to the GC fraction of the whole sequence
       (results/composition_decoding.csv, level seed), and
  mcc  holdout MCC of the paper's linear probe on the same 30 parities
       (results/readout_full.csv),
and writes results/composition_link.csv: Pearson and Spearman
correlation over all register-seeds, over the seven registers with the full
dynamics, and over the ten-seed means of the nine registers; and the range of
the change in holdout MCC when the 30 parities are added to the regression on
GC fraction and CpG ratio (results/composition_increment.csv, contrast
rows).

Pure pandas/numpy/scipy; no model is loaded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RES = Path("results")
REFERENCE = ("init_zerogamma", "init_norzz")


def pick(df: pd.DataFrame, **kw) -> pd.DataFrame:
    for k, v in kw.items():
        if k not in df.columns:
            raise KeyError(f"column {k} missing; have {list(df.columns)}")
        df = df[df[k] == v]
    return df


def main() -> None:
    dec = pd.read_csv(RES / "composition_decoding.csv")
    a = pick(dec, level="seed", readout="subset30", target="gc_all",
             eval_set="holdout_promoter")[["condition", "seed", "r2"]].copy()
    a["seed"] = a["seed"].astype(int)
    ro = pd.read_csv(RES / "readout_full.csv")
    b = pick(ro, readout="subset30", eval_name="holdout_promoter")[
        ["condition", "seed", "eval_mcc"]]
    m = a.merge(b, on=["condition", "seed"], validate="one_to_one")
    assert len(m) == 90, len(m)

    rows = []

    def corr(name: str, x: pd.DataFrame) -> None:
        rows.append({
            "quantity": name, "n": len(x),
            "pearson_r": stats.pearsonr(x["r2"], x["eval_mcc"])[0],
            "spearman_r": stats.spearmanr(x["r2"], x["eval_mcc"])[0],
            "min": np.nan, "max": np.nan})

    corr("gc_r2_vs_mcc_all_register_seeds", m)
    corr("gc_r2_vs_mcc_full_dynamics", m[~m["condition"].isin(REFERENCE)])
    corr("gc_r2_vs_mcc_register_means",
         m.groupby("condition")[["r2", "eval_mcc"]].mean().reset_index())

    inc = pd.read_csv(RES / "composition_increment.csv")
    c = pick(inc, level="contrast", eval_set="holdout_promoter",
             contrast="gc_cpg+subset30_minus_gc_cpg")
    for metric in ("mcc", "auroc"):
        d = c[c["metric"] == metric]
        rows.append({"quantity": f"increment_over_gc_cpg_{metric}",
                     "n": len(d), "pearson_r": np.nan, "spearman_r": np.nan,
                     "min": d["mean_diff"].min(), "max": d["mean_diff"].max()})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "composition_link.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
