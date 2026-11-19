"""Grouped splits: every group (e.g. exact or reverse-complement
duplicates) is assigned to one split as a whole."""

from __future__ import annotations

import numpy as np


def grouped_split(group_id: np.ndarray, fractions: dict[str, float],
                  seed: int) -> dict[str, np.ndarray]:
    """Assign whole groups to splits. fractions e.g.
    {"train": 0.6, "val": 0.2, "test": 0.2}; returns index arrays."""
    rng = np.random.default_rng(seed)
    groups = np.unique(group_id)
    perm = rng.permutation(len(groups))
    n = len(groups)
    bounds, acc = {}, 0.0
    names = list(fractions)
    for i, name in enumerate(names):
        if i == len(names) - 1:
            end = n
        else:
            acc += fractions[name]
            end = int(round(acc * n))
        start = 0 if i == 0 else bounds[names[i - 1]][1]
        bounds[name] = (start, end)
    out: dict[str, np.ndarray] = {}
    gset = {name: set(groups[perm[s:e]].tolist())
            for name, (s, e) in bounds.items()}
    for name, gs in gset.items():
        out[name] = np.where(np.isin(group_id, list(gs)))[0]
    return out
