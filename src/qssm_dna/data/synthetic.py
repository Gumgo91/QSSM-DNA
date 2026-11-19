"""Synthetic sequence tasks used by the training and resume tests.

Tasks:
  * delayed_base_retrieval: base at position 0 must be reproduced after a
    run of unrelated distractors. 4-class label = that base.
  * motif_order: two fixed motifs A and B embedded in matched background;
    binary label = order (AB vs BA). Paired examples share a group_id so
    pairs never cross splits.
"""

from __future__ import annotations

import numpy as np

from .tokenizer import encode

MOTIF_A = encode("ACG")  # fixed motifs; identical composition in both labels
MOTIF_B = encode("TGA")


def gen_delayed_base_retrieval(n: int, length: int,
                               rng: np.random.Generator,
                               ) -> dict[str, np.ndarray]:
    """tokens [n,L] over {A,C,G,T}; label = tokens[:,0]."""
    tokens = rng.integers(0, 4, size=(n, length), dtype=np.int64)
    labels = tokens[:, 0].copy()
    return {
        "tokens": tokens,
        "labels": labels.astype(np.int64),
        "group_id": np.arange(n, dtype=np.int64),
        "task": "delayed_base_retrieval",
        "n_classes": 4,
    }


def _motif_layout(rng: np.random.Generator, length: int,
                  ) -> tuple[np.ndarray, int, int]:
    """One background layout + non-overlapping motif positions (a before b)."""
    la, lb = len(MOTIF_A), len(MOTIF_B)
    bg = rng.integers(0, 4, size=length, dtype=np.int64)
    a_start = int(rng.integers(0, length - la - lb + 1))
    b_start = int(rng.integers(a_start + la, length - lb + 1))
    return bg, a_start, b_start


def gen_motif_order(n_pairs: int, length: int,
                    rng: np.random.Generator) -> dict[str, np.ndarray]:
    """n_pairs*2 examples; a pair shares background+positions and differs
    only in motif order -> identical GC and k-mer marginals within a pair."""
    toks, labels, groups = [], [], []
    for p in range(n_pairs):
        bg, a_start, b_start = _motif_layout(rng, length)
        seq_ab = bg.copy()
        seq_ab[a_start:a_start + len(MOTIF_A)] = MOTIF_A
        seq_ab[b_start:b_start + len(MOTIF_B)] = MOTIF_B
        seq_ba = bg.copy()
        seq_ba[a_start:a_start + len(MOTIF_B)] = MOTIF_B
        seq_ba[b_start:b_start + len(MOTIF_A)] = MOTIF_A
        for seq, lab in ((seq_ab, 0), (seq_ba, 1)):
            toks.append(seq)
            labels.append(lab)
            groups.append(p)
    return {
        "tokens": np.stack(toks).astype(np.int64),
        "labels": np.array(labels, dtype=np.int64),
        "group_id": np.array(groups, dtype=np.int64),
        "task": "motif_order",
        "n_classes": 2,
    }


def generate(task: str, n: int, length: int, seed: int
             ) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    if task == "delayed_base_retrieval":
        return gen_delayed_base_retrieval(n, length, rng)
    if task == "motif_order":
        return gen_motif_order(max(n // 2, 1), length, rng)
    raise ValueError(f"unknown synthetic task {task!r}")
