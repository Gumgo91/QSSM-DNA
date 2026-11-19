"""Contrasts against matched references (CPU only, no model is run).

  CUDA_VISIBLE_DEVICES=-1 python scripts/matched_reference.py

Inputs : results/all_metrics_long.csv   (make_results.py)
         results/readout_full.csv       (readout_table.py)
Output : results/matched_reference_contrasts.csv

Four families of paired contrasts over the ten replication seeds (first
minus second), each on the promoter holdout, the benchmark test split and
enhancers, for MCC and AUROC:

  ablation_lift         an encoder trained without couplings or without
                        damping against the untrained register with the
                        same component switched off (subset30)
  lift_difference       lift of the full all-position encoder over the
                        untrained register minus the lift of the encoder
                        without couplings over the untrained register
                        without couplings (subset30)
  budget                all-position encoders at 2.5 million tokens against
                        0.25 million tokens (subset30 and raw11)
  supervision_pauli255  dense-tail-8 and all-position against endpoint
                        encoders with all 255 Pauli expectations

The untrained registers without couplings or damping and the pauli255
readout exist only in readout_full.csv, whose terminal states were computed
in double precision through the transfer matrices; the trained ablation
encoders come from the single-precision evaluation runs. The two routes
give identical holdout scores for the registers they share, which the
script checks. The budget rows take both arms from readout_full.csv, so
that they agree with the readout table built from the same file.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "2"

import pandas as pd  # noqa: E402

sys.path.insert(0, "src")
from qssm_dna.experiments import stats  # noqa: E402

OUT = Path("results")
SEEDS = stats.SEEDS
EVALS = ["holdout_promoter", "gb_promoter:test", "gb_cohn:test"]
METRICS = ("eval_mcc", "eval_auroc")
B10 = "_b2500000"


def table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df.seed.isin(SEEDS)]
    key = ["condition", "readout", "eval_name", "seed"]
    if df.duplicated(key).any():
        raise RuntimeError(f"{path}: duplicated rows for {key}")
    return df.set_index(key).sort_index()


def main() -> None:
    long = table(OUT / "all_metrics_long.csv")
    full = table(OUT / "readout_full.csv")

    def series(src: pd.DataFrame, cond: str, readout: str, ev: str,
               metric: str) -> pd.Series:
        try:
            s = src.loc[(cond, readout, ev), metric]
        except KeyError as err:
            raise RuntimeError(f"no rows for {cond}/{readout}/{ev}") from err
        if sorted(s.index) != sorted(SEEDS):
            raise RuntimeError(f"{cond}/{readout}/{ev}: seeds "
                               f"{sorted(s.index)}")
        return s.loc[SEEDS].astype(float)

    # the two routes agree on the holdout for the registers they share
    for cond in ("qssm_random_init", "qssm_allpos"):
        a = series(long, cond, "subset30", "holdout_promoter", "eval_mcc")
        b = series(full, cond, "subset30", "holdout_promoter", "eval_mcc")
        gap = float((a - b).abs().max())
        print(f"route check {cond}: max |difference| = {gap:.2e}")
        if gap > 1e-12:
            raise RuntimeError(f"{cond}: routes differ by {gap}")

    rows = []

    def add(family: str, name: str, first: pd.Series, second: pd.Series,
            label_first: str, label_second: str, ev: str,
            metric: str) -> None:
        rows.append({"family": family, "eval_set": ev, "contrast": name,
                     "first": label_first, "second": label_second,
                     "metric": metric,
                     "mean_first": float(first.mean()),
                     "mean_second": float(second.mean()),
                     "seeds": ",".join(map(str, SEEDS)),
                     **stats.paired_summary(first.values, second.values)})

    for ev in EVALS:
        for metric in METRICS:
            for cond, ref in (("qssm_allpos_norzz", "init_norzz"),
                              ("qssm_dense8_norzz", "init_norzz"),
                              ("qssm_dense8_zerogamma", "init_zerogamma")):
                add("ablation_lift", f"{cond}_minus_{ref}",
                    series(long, cond, "subset30", ev, metric),
                    series(full, ref, "subset30", ev, metric),
                    f"{cond}/subset30", f"{ref}/subset30", ev, metric)
            lift_full = (series(long, "qssm_allpos", "subset30", ev, metric)
                         - series(long, "qssm_random_init", "subset30", ev,
                                  metric))
            lift_norzz = (series(long, "qssm_allpos_norzz", "subset30", ev,
                                 metric)
                          - series(full, "init_norzz", "subset30", ev,
                                   metric))
            add("lift_difference", "allpos_lift_minus_allpos_norzz_lift",
                lift_full, lift_norzz, "qssm_allpos - qssm_random_init",
                "qssm_allpos_norzz - init_norzz", ev, metric)
            for readout in ("subset30", "raw11"):
                add("budget", f"allpos_B10_vs_allpos_B1_{readout}",
                    series(full, "qssm_allpos" + B10, readout, ev, metric),
                    series(full, "qssm_allpos", readout, ev, metric),
                    f"qssm_allpos{B10}/{readout}", f"qssm_allpos/{readout}",
                    ev, metric)
            for a, b in (("qssm_dense8" + B10, "qssm_endpoint" + B10),
                         ("qssm_allpos" + B10, "qssm_endpoint" + B10),
                         ("qssm_allpos", "qssm_endpoint")):
                add("supervision_pauli255", f"{a}_minus_{b}_pauli255",
                    series(full, a, "pauli255", ev, metric),
                    series(full, b, "pauli255", ev, metric),
                    f"{a}/pauli255", f"{b}/pauli255", ev, metric)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "matched_reference_contrasts.csv", index=False)
    show = out[(out.eval_set == "holdout_promoter")
               & (out.metric == "eval_mcc")]
    for r in show.itertuples():
        print(f"{r.family:22s} {r.contrast:58s} {r.mean_first:.3f} "
              f"{r.mean_second:.3f}  {r.mean_diff:+.3f} "
              f"[{r.ci95_low:+.3f}, {r.ci95_high:+.3f}]  "
              f"{r.n_positive}/{r.n}")
    print(f"{len(out)} rows -> {OUT / 'matched_reference_contrasts.csv'}")


if __name__ == "__main__":
    main()
