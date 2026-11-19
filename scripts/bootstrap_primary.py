"""Paired bootstrap of the primary contrasts (CPU only).

  CUDA_VISIBLE_DEVICES=-1 python scripts/bootstrap_primary.py

For every primary contrast (stats.PRIMARY) and evaluation set, the
seed-mean MCC difference is resampled over sequences (holdout: over overlap
clusters), 2,000 resamples, with the fitted probes and thresholds frozen.

Inputs : results/all_metrics_long.csv (make_results.py; its column run names
         the evaluation run of each row), the predictions.npz of those runs
         (found through stats.load_predictions under the run roots) and
         data/holdout_promoter_chr9_chr10.npz (overlap clusters)
Output : results/bootstrap_primary.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from qssm_dna.experiments import datasets, stats  # noqa: E402

RES = Path("results")
N_BOOT = 2000
PRED_KEY = {"holdout_promoter": "gb_promoter:holdout_promoter",
            "holdout_isolated": "gb_promoter:holdout_isolated",
            "gb_promoter:test": "gb_promoter:test",
            "gb_cohn:test": "gb_cohn:test", "gb_ocr:test": "gb_ocr:test"}


def mcc_from_counts(tp, tn, fp, fn):
    num = tp * tn - fp * fn
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return np.where(den > 0, num / np.maximum(den, 1e-300), 0.0)


def boot_mcc(y, yhat, idx_mat):
    """MCC for each bootstrap row of indices: idx_mat [B, n]."""
    yy = y[idx_mat]
    pp = yhat[idx_mat]
    tp = ((yy == 1) & (pp == 1)).sum(1).astype(float)
    tn = ((yy == 0) & (pp == 0)).sum(1).astype(float)
    fp = ((yy == 0) & (pp == 1)).sum(1).astype(float)
    fn = ((yy == 1) & (pp == 0)).sum(1).astype(float)
    return mcc_from_counts(tp, tn, fp, fn)


def main() -> None:
    long = pd.read_csv(RES / "all_metrics_long.csv")
    z = np.load(datasets.DATA_DIR / "holdout_promoter_chr9_chr10.npz")
    clusters_full = z["cluster"]
    iso_keep = (z["y"] == 0) | z["isolated10kb"]
    rows = []
    rng = np.random.default_rng(20261103)
    for ev, pkey in PRED_KEY.items():
        for name, ((c1, r1), (c2, r2)) in stats.PRIMARY.items():
            pairs = []
            for s in stats.SEEDS:
                d1 = long[(long.condition == c1) & (long.seed == s)
                          & (long.eval_name == ev) & (long.readout == r1)]
                d2 = long[(long.condition == c2) & (long.seed == s)
                          & (long.eval_name == ev) & (long.readout == r2)]
                if d1.empty or d2.empty:
                    continue
                p1 = stats.load_predictions(d1.run.iloc[0])
                p2 = stats.load_predictions(d2.run.iloc[0])
                k1, k2 = f"{pkey}__{r1}", f"{pkey}__{r2}"
                y = p1[f"{k1}__label"].astype(int)
                a = (p1[f"{k1}__prob"] >= p1[f"{k1}__threshold"]).astype(int)
                b = (p2[f"{k2}__prob"] >= p2[f"{k2}__threshold"]).astype(int)
                pairs.append((y, a, b))
            if len(pairs) < 2:
                continue
            y = pairs[0][0]
            n = len(y)
            if ev.startswith("holdout"):
                cl = clusters_full if ev == "holdout_promoter" else \
                    clusters_full[iso_keep]
                units = np.unique(cl)
                members = [np.flatnonzero(cl == u) for u in units]
                idx = []
                for _ in range(N_BOOT):
                    pick = rng.integers(0, len(units), len(units))
                    idx.append(np.concatenate([members[i] for i in pick]))
                # ragged -> evaluate row by row
                diffs = []
                for ix in idx:
                    dm = [boot_mcc(yy, aa, ix[None])[0] -
                          boot_mcc(yy, bb, ix[None])[0]
                          for yy, aa, bb in pairs]
                    diffs.append(np.mean(dm))
                diffs = np.array(diffs)
            else:
                diffs = np.zeros(N_BOOT)
                for start in range(0, N_BOOT, 100):
                    ix = rng.integers(0, n, (min(100, N_BOOT - start), n))
                    dm = np.mean([boot_mcc(yy, aa, ix) - boot_mcc(yy, bb, ix)
                                  for yy, aa, bb in pairs], axis=0)
                    diffs[start:start + len(dm)] = dm
            obs = np.mean([boot_mcc(yy, aa, np.arange(n)[None])[0] -
                           boot_mcc(yy, bb, np.arange(n)[None])[0]
                           for yy, aa, bb in pairs])
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            rows.append({"eval_set": ev, "contrast": name,
                         "n_seeds": len(pairs), "observed_mean_diff": obs,
                         "boot_ci95_low": lo, "boot_ci95_high": hi,
                         "n_boot": N_BOOT,
                         "unit": "cluster" if ev.startswith("holdout")
                         else "sequence"})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(RES / "bootstrap_primary.csv", index=False)


if __name__ == "__main__":
    main()
