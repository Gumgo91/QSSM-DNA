"""Labelled evaluation sets.

  gb_promoter       Genomic Benchmarks human_nontata_promoters: official
                    train split with a canonical-hash grouped 10 % validation
                    split (seed 1729), official test split.
  gb_promoter_s6k   gb_promoter with a fixed 6,000-sequence probe-training
                    subset (register-size comparison).
  holdout_promoter  promoter set from the held-out chromosomes chr9 and chr10
                    built by experiments.holdout; evaluation only (no probe
                    fitting, threshold or setting is chosen on it).
  gb_cohn, gb_ocr   Genomic Benchmarks human_enhancers_cohn and
                    human_ocr_ensembl; probes are fitted on their own train
                    splits (10 % grouped validation for thresholds).

Every loader returns {"train"|"val"|"test": {"seqs": [...], "y": array}}
(holdout_promoter: only "test") plus a manifest with canonical hashes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..data.tokenizer import canonical_hash

DATA_DIR = Path("data")
GB_TASK_FILES = {"gb_cohn": "human_enhancers_cohn",
                 "gb_ocr": "human_ocr_ensembl"}
# probe-training cap for the large OCR train split (drawn once with seed
# 1729, independently of any encoder)
OCR_TRAIN_CAP = 30000
SPLIT_SEED = 1729
VAL_FRAC = 0.1


def _split_hash(seqs: list[str], y: np.ndarray) -> str:
    h = hashlib.sha256()
    for s, lab in zip(seqs, y.tolist()):
        h.update(canonical_hash(s).encode())
        h.update(str(int(lab)).encode())
    return h.hexdigest()


def _manifest(name: str, splits: dict) -> dict:
    out = {"dataset": name, "splits": {}}
    for k, v in splits.items():
        lens = [len(s) for s in v["seqs"]]
        out["splits"][k] = {
            "n": len(v["seqs"]),
            "label_counts": {int(a): int(b) for a, b in
                             zip(*np.unique(v["y"], return_counts=True))},
            "length_min": int(min(lens)), "length_max": int(max(lens)),
            "length_mean": float(np.mean(lens)),
            "ordered_canonical_label_sha256": _split_hash(v["seqs"], v["y"]),
        }
    return out


def _grouped_val(seqs: list[str], y: np.ndarray):
    from ..train.probe import train_val_split
    rows = list(zip(seqs, y.tolist()))
    tr, va = train_val_split(rows, VAL_FRAC, SPLIT_SEED)
    return np.asarray(tr), np.asarray(va)


def load_gb_promoter() -> tuple[dict, dict]:
    from ..train.probe import load_promoter_benchmark
    d = load_promoter_benchmark({"data": {"benchmark": {
        "validation_fraction_of_original_train": VAL_FRAC,
        "split_seed": SPLIT_SEED}}})
    splits = {k: {"seqs": d["seqs"][k], "y": d[f"{k}_labels"]}
              for k in ("train", "val", "test")}
    return splits, _manifest("gb_promoter", splits)


def load_holdout_promoter() -> tuple[dict, dict]:
    p = DATA_DIR / "holdout_promoter_chr9_chr10.npz"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing; build it with `experiment --part holdout`")
    z = np.load(p, allow_pickle=False)
    splits = {"test": {"seqs": z["seq"].tolist(),
                       "y": z["y"].astype(np.int64)}}
    return splits, _manifest("holdout_promoter", splits)


def load_gb_task(name: str) -> tuple[dict, dict]:
    p = DATA_DIR / f"{GB_TASK_FILES[name]}.npz"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing; fetch it with `experiment --part fetch-tasks`")
    z = np.load(p, allow_pickle=False)
    seqs = z["train_seq"].tolist()
    y = z["train_y"].astype(np.int64)
    tr, va = _grouped_val(seqs, y)
    if name == "gb_ocr" and len(tr) > OCR_TRAIN_CAP:
        rng = np.random.default_rng(SPLIT_SEED)
        tr = np.sort(rng.choice(tr, size=OCR_TRAIN_CAP, replace=False))
    splits = {
        "train": {"seqs": [seqs[i] for i in tr], "y": y[tr]},
        "val": {"seqs": [seqs[i] for i in va], "y": y[va]},
        "test": {"seqs": z["test_seq"].tolist(),
                 "y": z["test_y"].astype(np.int64)},
    }
    return splits, _manifest(name, splits)


SCALING_TRAIN = 6000


def scaling_train_index(n_train: int) -> np.ndarray:
    """Fixed probe-training subset of the promoter training split used to
    compare register sizes (drawn once, seed 1729)."""
    rng = np.random.default_rng(SPLIT_SEED)
    return np.sort(rng.choice(n_train, size=SCALING_TRAIN, replace=False))


def load(name: str) -> tuple[dict, dict]:
    if name == "gb_promoter":
        return load_gb_promoter()
    if name == "gb_promoter_s6k":
        splits, _ = load_gb_promoter()
        idx = scaling_train_index(len(splits["train"]["seqs"]))
        splits["train"] = {"seqs": [splits["train"]["seqs"][i] for i in idx],
                           "y": splits["train"]["y"][idx]}
        return splits, _manifest(name, splits)
    if name == "holdout_promoter":
        return load_holdout_promoter()
    if name in GB_TASK_FILES:
        return load_gb_task(name)
    raise KeyError(name)


def fetch_gb_tasks(out_dir: Path = DATA_DIR) -> dict:
    """Download the two extra Genomic Benchmarks tasks through the
    genomic_benchmarks package and store them as npz with a manifest."""
    from ..data import benchmarks
    out_dir.mkdir(parents=True, exist_ok=True)
    man = {}
    for key, task in GB_TASK_FILES.items():
        splits = benchmarks.load_task(task, version=0)
        np.savez_compressed(
            out_dir / f"{task}.npz",
            train_seq=np.array([s for s, _ in splits["train"]]),
            train_y=np.array([y for _, y in splits["train"]]),
            test_seq=np.array([s for s, _ in splits["test"]]),
            test_y=np.array([y for _, y in splits["test"]]))
        m = benchmarks.dataset_manifest(task, splits)
        m["duplicate_check"] = benchmarks.find_cross_split_duplicates(splits)
        m["npz_sha256"] = hashlib.sha256(
            (out_dir / f"{task}.npz").read_bytes()).hexdigest()
        man[key] = m
    (out_dir / "gb_tasks_manifest.json").write_text(
        json.dumps(man, indent=2, default=str))
    return man
