"""Hardware-noise channels for evaluation (not used in training).

The strengths are hypothetical sensitivity values (configs/experiments.json,
"noise"), not a calibration of any device. The model's own token-conditioned
amplitude damping (channels.py) is a learned component and is separate.

Definitions on the [B,2]*2q axes view (row wires, then column wires):

  depolarize_1q(p, w):      rho -> (1-p) rho + p (I/2)_w (x) Tr_w rho
  depolarize_2q(p, w1, w2): rho -> (1-p) rho + p (I/4)_{w1 w2} (x) Tr_{w1 w2} rho
  dephase(p, w):            rho -> (1-p) rho + p Z_w rho Z_w
  readout bit flip e:       every measured bit flips independently with
                            probability e; a parity over a wire subset S
                            then has its expectation scaled by (1-2e)^|S|.

Token-cell noise model (NoiseModel.cell): after the learned damping layer
each memory wire receives depolarize_1q(p2) for the two-qubit interaction
with the reset ancilla that realises the damping channel; after each
RY/RZ layer every wire receives depolarize_1q(p1); every RZZ is followed
by depolarize_2q(p2) on its pair; every wire then dephases with p_phi per
token cell. PAD positions execute no cell and receive no noise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from . import channels, density, gates


@dataclass(frozen=True)
class NoiseModel:
    p1: float = 0.0       # 1q depolarizing per rotation layer per wire
    p2: float = 0.0       # 2q depolarizing per RZZ; 1q per damping dilation
    p_phi: float = 0.0    # dephasing per wire per token cell
    readout: float = 0.0  # symmetric bit-flip probability per measured bit

    def as_dict(self) -> dict:
        return asdict(self)


def _row_col(w: int, q: int) -> tuple[int, int]:
    return 1 + w, 1 + q + w


def depolarize_1q(rho_ax: torch.Tensor, p: float, w: int,
                  q: int) -> torch.Tensor:
    if p == 0.0:
        return rho_ax
    r, c = _row_col(w, q)
    tr = rho_ax.diagonal(dim1=r, dim2=c).sum(-1)
    mixed = tr.unsqueeze(r).unsqueeze(c)
    shape = [1] * rho_ax.dim()
    shape[r] = shape[c] = 2
    eye = torch.eye(2, dtype=rho_ax.dtype, device=rho_ax.device) / 2
    return (1.0 - p) * rho_ax + p * (mixed * eye.view(shape))


def depolarize_2q(rho_ax: torch.Tensor, p: float, w1: int, w2: int,
                  q: int) -> torch.Tensor:
    if p == 0.0:
        return rho_ax
    lo, hi = sorted((w1, w2))
    r_lo, c_lo = _row_col(lo, q)
    r_hi, c_hi = _row_col(hi, q)
    # trace the higher wire first so the lower wire's axes keep their index
    tr = rho_ax.diagonal(dim1=r_hi, dim2=c_hi).sum(-1)
    tr = tr.diagonal(dim1=r_lo, dim2=c_lo - 1).sum(-1)
    mixed = tr.unsqueeze(r_lo).unsqueeze(r_hi).unsqueeze(c_lo).unsqueeze(c_hi)
    shape = [1] * rho_ax.dim()
    shape[r_lo] = shape[c_lo] = shape[r_hi] = shape[c_hi] = 2
    eye_lo = torch.eye(2, dtype=rho_ax.dtype, device=rho_ax.device) / 2
    s_lo = [1] * rho_ax.dim()
    s_lo[r_lo] = s_lo[c_lo] = 2
    s_hi = [1] * rho_ax.dim()
    s_hi[r_hi] = s_hi[c_hi] = 2
    return (1.0 - p) * rho_ax + p * (mixed * eye_lo.view(s_lo)
                                     * eye_lo.view(s_hi))


def dephase(rho_ax: torch.Tensor, p: float, w: int, q: int) -> torch.Tensor:
    if p == 0.0:
        return rho_ax
    r, c = _row_col(w, q)
    f = torch.tensor([[1.0, 1.0 - 2.0 * p], [1.0 - 2.0 * p, 1.0]],
                     dtype=rho_ax.real.dtype, device=rho_ax.device)
    shape = [1] * rho_ax.dim()
    shape[r] = shape[c] = 2
    return rho_ax * f.view(shape).to(rho_ax.dtype)


def readout_flip_probs(p: torch.Tensor, e: float, q: int) -> torch.Tensor:
    """Outcome distribution [..., 2^q] after independent bit flips."""
    if e == 0.0:
        return p
    k = torch.tensor([[1.0 - e, e], [e, 1.0 - e]], dtype=p.dtype,
                     device=p.device)
    lead = p.shape[:-1]
    t = p.reshape(*lead, *([2] * q))
    for w in range(q):
        t = torch.movedim(torch.tensordot(t, k, dims=([t.dim() - q + w],
                                                      [1])),
                          -1, t.dim() - q + w)
    return t.reshape(*lead, 1 << q)


@torch.no_grad()
def noisy_forward_state(encoder, tokens: torch.Tensor, mask: torch.Tensor,
                        noise: NoiseModel) -> torch.Tensor:
    """Terminal rho [B,D,D] of a QSSMEncoder with NoiseModel cells. With a
    noiseless model this reproduces encoder.forward_state exactly (same
    operation order; tested)."""
    q = encoder.q
    b, l = tokens.shape
    theta, gamma = encoder.controls(tokens)
    rzz = encoder.rzz_angles
    rho_ax = density.to_axes(encoder.initial_state(b, tokens.device), q)
    for t in range(l):
        new = rho_ax
        for i in range(q):
            new = channels.amplitude_damping(new, gamma[:, t, i], i, q)
            new = depolarize_1q(new, noise.p2, i, q)
        for layer in range(encoder.cell_depth):
            for i in range(q):
                new = density.apply_1q_unitary(
                    new, gates.ryrz(theta[:, t, layer, i, 0],
                                    theta[:, t, layer, i, 1]), i, q)
                new = depolarize_1q(new, noise.p1, i, q)
            for i in range(q - 1):
                new = density.apply_2q_unitary(
                    new, gates.rzz(rzz[layer, i]), i, i + 1, q)
                new = depolarize_2q(new, noise.p2, i, i + 1, q)
        for i in range(q):
            new = dephase(new, noise.p_phi, i, q)
        view = mask[:, t].view(-1, *([1] * (2 * q)))
        rho_ax = torch.where(view, new, rho_ax)
    return density.to_matrix(rho_ax, q)
