"""The model's learned, token-conditioned amplitude-damping channel. The
evaluation-only hardware noise is in quantum/noise.py.
"""

from __future__ import annotations

import torch

from .density import apply_1q_channel
from .gates import amplitude_damping_kraus


def amplitude_damping(rho_ax: torch.Tensor, gamma: torch.Tensor, wire: int,
                      q: int) -> torch.Tensor:
    """Apply D_gamma on `wire`. gamma may be scalar or [B] (per-sample)."""
    k = amplitude_damping_kraus(gamma)
    return apply_1q_channel(rho_ax, k, wire, q)
