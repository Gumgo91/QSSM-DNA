"""Holdout windows that share a 31-mer with a benchmark sequence (CPU only).

  CUDA_VISIBLE_DEVICES=-1 python scripts/holdout_kmer31.py

Inputs : data/holdout_promoter_chr9_chr10.npz (the holdout as evaluated),
         data/benchmarks.npz (written by prepare-data)
Output : results/holdout_kmer31.json

The manifest of the holdout records this count for the 1,406 candidate
windows, before the window that duplicates a benchmark training sequence is
removed. This script counts it on the evaluated holdout with the k-mer
coding of qssm_dna.experiments.holdout, for both strands of every benchmark
sequence.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "2"

import numpy as np  # noqa: E402

sys.path.insert(0, "src")
from qssm_dna.experiments import datasets  # noqa: E402
from qssm_dna.experiments import holdout as H  # noqa: E402

OUT = Path("results/holdout_kmer31.json")


def main() -> None:
    gb = np.load("data/benchmarks.npz", allow_pickle=False)
    gb_seqs = set(gb["train_seq"].tolist()) | set(gb["test_seq"].tolist())
    gb_k = np.unique(np.concatenate(
        [H._kmer_codes(t) for s in gb_seqs for t in (s, H.rc(s))]))
    hold = np.load(datasets.DATA_DIR / "holdout_promoter_chr9_chr10.npz",
                   allow_pickle=False)
    y = hold["y"]
    share = np.zeros(len(y), dtype=bool)
    for i, seq in enumerate(hold["seq"].tolist()):
        codes = H._kmer_codes(seq)
        pos = np.minimum(np.searchsorted(gb_k, codes), len(gb_k) - 1)
        share[i] = bool((gb_k[pos] == codes).any())
    res = {"windows_total": int(len(y)),
           "positives": int((y == 1).sum()),
           "negatives": int((y == 0).sum()),
           "windows_sharing_a_31mer_with_benchmark": int(share.sum()),
           "positives_sharing": int(share[y == 1].sum()),
           "negatives_sharing": int(share[y == 0].sum()),
           "benchmark_sequences": len(gb_seqs),
           "benchmark_31mers_both_strands": int(len(gb_k))}
    OUT.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
