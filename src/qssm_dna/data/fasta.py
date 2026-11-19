"""FASTA download and window sampling for DNA pretraining.

The SHA-256 of every downloaded file is recorded in data_manifest.json;
the per-chromosome files of UCSC hg38/chromosomes/ are used.
"""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

import numpy as np

from .tokenizer import N_IDX, encode


def download(url: str, dest: str | Path) -> dict:
    import requests

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
            tmp.rename(dest)
    h = hashlib.sha256(dest.read_bytes()).hexdigest()
    return {"url": url, "path": Path(dest).as_posix(), "sha256": h,
            "bytes": dest.stat().st_size}


def read_fasta_sequences(path: str | Path) -> dict[str, str]:
    """Return {chrom: uppercase seq}. Handles .gz or plain."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    seqs: dict[str, list[str]] = {}
    name = None
    with opener(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                name = line[1:].split()[0]
                seqs[name] = []
            elif name is not None:
                seqs[name].append(line.upper())
    return {k: "".join(v) for k, v in seqs.items()}


class WindowSampler:
    """Samples windows of length L+1 from one chromosome; the last base is
    the next-base target. Windows do not cross chromosome boundaries, and
    windows whose input has more than max_n_fraction Ns are redrawn;
    WindowSource drops windows whose target is N."""

    def __init__(self, chrom_tokens: dict[str, np.ndarray],
                 lengths: list[int], probs: list[float] | None,
                 max_n_fraction: float, rng: np.random.Generator,
                 train_chroms: list[str] | None = None):
        self.lengths = lengths
        self.probs = (np.array(probs, dtype=float) if probs else
                      np.ones(len(lengths)) / len(lengths))
        self.probs = self.probs / self.probs.sum()
        self.max_n = max_n_fraction
        self.rng = rng
        names = train_chroms or list(chrom_tokens)
        self.chroms = [(c, chrom_tokens[c]) for c in names
                       if c in chrom_tokens]
        if not self.chroms:
            raise ValueError("no chromosomes available for sampling")
        self.cum_len = np.cumsum([len(s) for _, s in self.chroms])
        self.n_skipped_nfrac = 0
        self.n_sampled = 0

    def sample(self) -> tuple[np.ndarray, int, str, int, int]:
        """-> (input_tokens[L], target_base, chrom, start, L).
        Retries until the input satisfies the N-fraction rule."""
        for _ in range(64):
            l = int(self.rng.choice(self.lengths, p=self.probs))
            r = int(self.rng.integers(0, int(self.cum_len[-1])))
            ci = int(np.searchsorted(self.cum_len, r, side="right"))
            base = 0 if ci == 0 else int(self.cum_len[ci - 1])
            chrom, seq = self.chroms[ci]
            start = r - base
            if start + l + 1 > len(seq):
                continue  # window would cross the chromosome end
            window = seq[start:start + l + 1]
            inp, tgt = window[:-1], int(window[-1])
            nfrac = float(np.mean(inp == N_IDX))
            if nfrac > self.max_n:
                self.n_skipped_nfrac += 1
                continue
            self.n_sampled += 1
            return inp.astype(np.int64), tgt, chrom, start, l
        raise RuntimeError("could not sample a valid window after 64 tries "
                           f"(skipped {self.n_skipped_nfrac} for N-fraction)")
