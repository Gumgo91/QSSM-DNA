"""Classical GRU encoder, pretrained with the same windows, objective and
token budget as the quantum encoder."""

from __future__ import annotations

import torch
import torch.nn as nn


class GRUEncoder(nn.Module):
    """Small causal GRU -> terminal hidden -> linear projection to proj_dim."""

    def __init__(self, vocab_size: int = 6, emb: int = 16, hidden: int = 64,
                 proj_dim: int = 64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb)
        self.gru = nn.GRU(emb, hidden, batch_first=True)
        self.proj = nn.Linear(hidden, proj_dim)
        self.proj_dim = proj_dim

    def encode(self, tokens: torch.Tensor, mask: torch.Tensor,
               representation: str = "projected",
               rc_average: bool = False, **_) -> torch.Tensor:
        lengths = mask.sum(1).clamp_min(1)
        x = torch.nn.utils.rnn.pack_padded_sequence(
            self.emb(tokens.clamp_max(4)), lengths.cpu(),
            batch_first=True, enforce_sorted=False)
        _, h = self.gru(x)
        z = torch.nn.functional.gelu(self.proj(h[-1]))
        if rc_average:
            from .encoder import rc_tokens
            xr = torch.nn.utils.rnn.pack_padded_sequence(
                self.emb(rc_tokens(tokens, mask).clamp_max(4)),
                lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, hr = self.gru(xr)
            zr = torch.nn.functional.gelu(self.proj(hr[-1]))
            z = 0.5 * (z + zr)
        return z

    def encode_positions(self, tokens: torch.Tensor,
                         mask: torch.Tensor) -> torch.Tensor:
        """[B,L,proj] per-position embeddings for dense next-base
        supervision."""
        lengths = mask.sum(1).clamp_min(1)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            self.emb(tokens.clamp_max(4)), lengths.cpu(),
            batch_first=True, enforce_sorted=False)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(
            self.gru(packed)[0], batch_first=True,
            total_length=tokens.shape[1])
        return torch.nn.functional.gelu(self.proj(out))
