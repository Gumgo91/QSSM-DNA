"""Gate matrices, differentiable in torch.

Conventions:
  RY(a) = exp(-i a Y/2) = [[cos(a/2), -sin(a/2)], [sin(a/2), cos(a/2)]]
  RZ(a) = exp(-i a Z/2) = diag(e^{-ia/2}, e^{ia/2})
  RZZ(a) = exp(-i a Z⊗Z/2) = diag(e^{-ia/2}, e^{ia/2}, e^{ia/2}, e^{-ia/2})
          on basis |00>,|01>,|10>,|11> (first index = higher-order wire).

All builders accept a scalar or a batched angle tensor of shape [B] and
return [2,2] / [B,2,2] (1q) or [4,4] / [B,4,4] (RZZ). Angles are real.
"""

from __future__ import annotations

import math

import torch


def _as_batch(a: torch.Tensor) -> tuple[torch.Tensor, bool]:
    a = torch.as_tensor(a)
    if a.dim() == 0:
        return a.reshape(1), False
    return a, True


def _cdtype_of(a: torch.Tensor) -> torch.dtype:
    return torch.complex128 if a.dtype == torch.float64 else torch.complex64


def ry(a: torch.Tensor | float) -> torch.Tensor:
    if not torch.is_tensor(a):
        a = torch.tensor(float(a))
    ab, _ = _as_batch(a)
    h = ab / 2
    c, s = torch.cos(h), torch.sin(h)
    cd = _cdtype_of(ab)
    zero = torch.zeros_like(c)
    top = torch.stack([c.to(cd), (-s).to(cd)], dim=-1)
    bot = torch.stack([s.to(cd), c.to(cd)], dim=-1)
    out = torch.stack([top, bot], dim=-2)  # [B,2,2]
    return out[0] if not torch.is_tensor(a) or a.dim() == 0 else out


def rz(a: torch.Tensor | float) -> torch.Tensor:
    if not torch.is_tensor(a):
        a = torch.tensor(float(a))
    ab, _ = _as_batch(a)
    h = ab / 2
    cd = _cdtype_of(ab)
    e_neg = torch.complex(torch.cos(h), -torch.sin(h)).to(cd)
    e_pos = torch.complex(torch.cos(h), torch.sin(h)).to(cd)
    zero = torch.zeros_like(e_neg)
    top = torch.stack([e_neg, zero], dim=-1)
    bot = torch.stack([zero, e_pos], dim=-1)
    out = torch.stack([top, bot], dim=-2)
    return out[0] if a.dim() == 0 else out


def ryrz(a_ry: torch.Tensor, a_rz: torch.Tensor) -> torch.Tensor:
    """Fused RZ(a_rz) @ RY(a_ry) on one wire (RZ applied AFTER RY).
    Exact product of the two matrices -- one contraction instead of two.
    Angles may have arbitrary leading batch dims."""
    x = ry(a_ry)
    z = rz(a_rz)
    return torch.einsum("...ij,...jk->...ik", z, x)


def rzz(a: torch.Tensor | float) -> torch.Tensor:
    if not torch.is_tensor(a):
        a = torch.tensor(float(a))
    ab, _ = _as_batch(a)
    h = ab / 2
    cd = _cdtype_of(ab)
    e_neg = torch.complex(torch.cos(h), -torch.sin(h)).to(cd)
    e_pos = torch.complex(torch.cos(h), torch.sin(h)).to(cd)
    zero = torch.zeros_like(e_neg)
    rows = [
        torch.stack([e_neg, zero, zero, zero], dim=-1),
        torch.stack([zero, e_pos, zero, zero], dim=-1),
        torch.stack([zero, zero, e_pos, zero], dim=-1),
        torch.stack([zero, zero, zero, e_neg], dim=-1),
    ]
    out = torch.stack(rows, dim=-2)  # [B,4,4]
    return out[0] if a.dim() == 0 else out


def hadamard(dtype: torch.dtype = torch.complex64,
             device: torch.device | str | None = None) -> torch.Tensor:
    return torch.tensor(
        [[1.0, 1.0], [1.0, -1.0]], dtype=dtype, device=device) / math.sqrt(2.0)


def amplitude_damping_kraus(gamma: torch.Tensor | float) -> torch.Tensor:
    """Kraus operators for D_gamma. Returns [..., 2, 2, 2]: last axis indexes
    the Kraus operator (K_0 then K_1). gamma may be scalar or [B]."""
    if not torch.is_tensor(gamma):
        gamma = torch.tensor(float(gamma))
    g = gamma
    cd = _cdtype_of(g)
    one = torch.ones_like(g)
    zero = torch.zeros_like(g)
    s1 = torch.sqrt((1 - g).clamp_min(0.0)).to(cd)
    sg = torch.sqrt(g.clamp_min(0.0)).to(cd)
    k0 = torch.stack([
        torch.stack([one.to(cd), zero.to(cd)], dim=-1),
        torch.stack([zero.to(cd), s1], dim=-1),
    ], dim=-2)
    k1 = torch.stack([
        torch.stack([zero.to(cd), sg], dim=-1),
        torch.stack([zero.to(cd), zero.to(cd)], dim=-1),
    ], dim=-2)
    return torch.stack([k0, k1], dim=-1)  # [...,2,2,2]
