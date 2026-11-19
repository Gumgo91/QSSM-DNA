"""DNA tokenization. Vocabulary order is fixed: A C G T N PAD.

N is a distinct input token (not replaced by a random base).
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch

VOCAB = ("A", "C", "G", "T", "N", "PAD")
CHAR_TO_IDX = {c: i for i, c in enumerate(VOCAB)}
PAD_IDX = CHAR_TO_IDX["PAD"]
N_IDX = CHAR_TO_IDX["N"]

# complement on indices: A<->T, C<->G, N->N, PAD->PAD
COMPLEMENT_IDX = np.array([3, 2, 1, 0, 4, 5], dtype=np.int64)


def encode(seq: str) -> np.ndarray:
    """String -> int64 token array. Unknown chars -> N. Lowercase ok."""
    arr = np.fromiter(
        (CHAR_TO_IDX.get(ch, N_IDX) for ch in seq.upper()),
        dtype=np.int64, count=len(seq))
    return arr


def decode(tokens: np.ndarray | torch.Tensor) -> str:
    t = np.asarray(tokens)
    return "".join(VOCAB[int(i)] for i in t)


def rc_array(tokens: np.ndarray) -> np.ndarray:
    """Reverse-complement a token array (no padding semantics -- see
    rc_tokens for padded batches)."""
    return COMPLEMENT_IDX[tokens[::-1].copy()]


def pad_tokens(seqs: list[np.ndarray], max_len: int | None = None
               ) -> tuple[np.ndarray, np.ndarray]:
    """Left-aligned padding. Returns tokens [B,L] and mask [B,L] (bool)."""
    b = len(seqs)
    l = max_len or max(len(s) for s in seqs)
    toks = np.full((b, l), PAD_IDX, dtype=np.int64)
    mask = np.zeros((b, l), dtype=bool)
    for i, s in enumerate(seqs):
        n = min(len(s), l)
        toks[i, :n] = s[:n]
        mask[i, :n] = True  # N inside a window stays a valid token
    return toks, mask


def canonical_sequence(seq: str) -> str:
    """min(seq, RC(seq)) lexicographically -- for leak-detection hashing."""
    fwd = seq.upper()
    rev = decode(rc_array(encode(fwd)))
    return min(fwd, rev)


def canonical_hash(seq: str) -> str:
    return hashlib.sha256(canonical_sequence(seq).encode()).hexdigest()
