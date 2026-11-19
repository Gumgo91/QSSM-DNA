"""Genomic Benchmarks dataset access (official splits).

The dataset name is checked against the installed registry at load time;
a missing dataset raises DatasetUnavailable.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .tokenizer import canonical_hash

KNOWN_TASKS = {
    "human_nontata_promoters": "HumanNontataPromoters",
    "human_enhancers_cohn": "HumanEnhancersCohn",
    "human_ocr_ensembl": "HumanOcrEnsembl",
}


class DatasetUnavailable(RuntimeError):
    pass


def list_available_datasets() -> list[str]:
    try:
        from genomic_benchmarks.data_check import list_datasets
        return sorted(list_datasets())
    except Exception as e:
        raise DatasetUnavailable(f"genomic_benchmarks unavailable: {e}")


def load_task(task: str, version: int | None = None,
              ) -> dict[str, list[tuple[str, int]]]:
    """Returns {"train": [(seq,label)...], "test": [...]} raw sequences.

    Uses the genomic_benchmarks package; verifies the registry exposes the
    requested dataset name first. The package manages its own cache dir.
    """
    if task not in KNOWN_TASKS:
        raise DatasetUnavailable(f"unknown task {task!r}; not in registry map")
    available = list_available_datasets()
    if task not in available:
        raise DatasetUnavailable(
            f"dataset {task!r} not in installed registry (available: "
            f"{available[:8]}...)")
    try:
        import genomic_benchmarks.dataset_getters.pytorch_datasets as gbd
        cls = getattr(gbd, KNOWN_TASKS[task], None)
        if cls is None:
            raise DatasetUnavailable(
                f"no pytorch getter for {task}")
        out = {}
        for split in ("train", "test"):
            ds = cls(split=split, version=version)
            out[split] = [(str(s), int(y)) for s, y in ds]
        return out
    except DatasetUnavailable:
        raise
    except Exception as e:
        raise DatasetUnavailable(f"load of {task} failed: {e}")


def dataset_manifest(task: str, splits: dict[str, list[tuple[str, int]]],
                     source_meta: dict | None = None) -> dict:
    """Hash-level manifest: counts, lengths, label balance, canonical
    hashes for leak checks."""
    man = {"task": task, "splits": {}, "source": source_meta or {}}
    for split, rows in splits.items():
        seqs = [s for s, _ in rows]
        labels = [y for _, y in rows]
        lens = [len(s) for s in seqs]
        hashes = sorted(canonical_hash(s) for s in seqs)
        man["splits"][split] = {
            "n": len(rows),
            "label_counts": {int(k): int(v) for k, v in
                             zip(*np.unique(labels, return_counts=True))},
            "length_min": int(min(lens)), "length_max": int(max(lens)),
            "length_mean": float(np.mean(lens)),
            "canonical_hash_sha256": hashlib.sha256(
                "".join(hashes).encode()).hexdigest(),
            "n_unique_canonical": len(set(hashes)),
        }
    return man


def find_cross_split_duplicates(
        splits: dict[str, list[tuple[str, int]]]) -> dict[str, int]:
    """Exact/RC-duplicate leakage check across split pairs."""
    hashes = {name: {canonical_hash(s): i for i, (s, _) in enumerate(rows)}
              for name, rows in splits.items()}
    out = {}
    names = list(hashes)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = set(hashes[names[i]]) & set(hashes[names[j]])
            out[f"{names[i]}_vs_{names[j]}"] = len(inter)
    return out
