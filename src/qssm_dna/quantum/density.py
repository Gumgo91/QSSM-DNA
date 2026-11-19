"""Density-matrix operations of the differentiable torch backend.

State layout:
  rho : [B, D, D] complex, D = 2**q, Hermitian, PSD, trace 1.
  Wire 0 is the most significant bit of the flat basis index.
  `to_axes` reshapes to [B, r_0..r_{q-1}, c_0..c_{q-1}] (row wires then
  col wires); `to_matrix` flattens back big-endian.

All operators act through 1- and 2-qubit tensor contractions; no dense
D x D unitary or D^2 x D^2 superoperator is built.

Implementation note: contractions use reshape-only views (no permute
copies). For wire w the row index splits as [2^w, 2, 2^(q-1-w)] and the
col index likewise, so an einsum over the size-2 axis is a strided view.
"""

from __future__ import annotations

import torch


def initial_state(batch_size: int, q: int,
                  device: torch.device | str | None = None,
                  dtype: torch.dtype = torch.complex64) -> torch.Tensor:
    d = 1 << q
    rho = torch.zeros(batch_size, d, d, dtype=dtype, device=device)
    rho[:, 0, 0] = 1.0
    return rho


def to_axes(rho: torch.Tensor, q: int) -> torch.Tensor:
    """[B,D,D] -> [B, 2]*2q (row wires 0..q-1, then col wires)."""
    b = rho.shape[0]
    return rho.reshape(b, *([2] * (2 * q)))


def to_matrix(rho_ax: torch.Tensor, q: int) -> torch.Tensor:
    b = rho_ax.shape[0]
    return rho_ax.reshape(b, 1 << q, 1 << q)


def _apply_row_1q(rho: torch.Tensor, m: torch.Tensor, w: int,
                  q: int) -> torch.Tensor:
    """out[b,i,:] = sum_a m[b?,i,a] rho[b,a,:] with i the wire-w row bit."""
    b = rho.shape[0]
    hi, lo = 1 << w, 1 << (q - 1 - w)
    v = rho.reshape(b, hi, 2, lo * (1 << q))
    if m.dim() == 2:
        out = torch.einsum("ia,bxay->bxiy", m.to(rho.dtype), v)
    else:
        out = torch.einsum("bia,bxay->bxiy", m.to(rho.dtype), v)
    return out.reshape(b, 1 << q, 1 << q)


def _apply_col_1q(rho: torch.Tensor, m: torch.Tensor, w: int,
                  q: int) -> torch.Tensor:
    """out[b,:,j] = sum_a rho[b,:,a] m[b?,j,a] -- caller passes conj(m)."""
    b = rho.shape[0]
    hi, lo = 1 << w, 1 << (q - 1 - w)
    v = rho.reshape(b, 1 << q, hi, 2, lo)
    mm = m.to(rho.dtype)
    if mm.dim() == 2:
        out = torch.einsum("ja,bdxay->bdxjy", mm, v)
    else:
        out = torch.einsum("bja,bdxay->bdxjy", mm, v)
    return out.reshape(b, 1 << q, 1 << q)


def apply_1q_unitary(rho_ax: torch.Tensor, u: torch.Tensor, wire: int,
                     q: int) -> torch.Tensor:
    """rho -> U_w rho U_w^dagger, U on `wire`. u: [2,2] or [B,2,2].
    rho_ax is the [B,2]*2q axes view (or any [B,D,D]-reshapable)."""
    rho = rho_ax.reshape(rho_ax.shape[0], 1 << q, 1 << q)
    rho = _apply_row_1q(rho, u, wire, q)
    rho = _apply_col_1q(rho, u.conj(), wire, q)
    return rho.reshape(rho_ax.shape)


def _split2(lo_w: int, hi_w: int, q: int):
    """Index-split sizes for wires (lo_w, hi_w), lo_w < hi_w."""
    a = 1 << lo_w                      # bits above lo wire
    mid = 1 << (hi_w - lo_w - 1)       # bits between
    lo = 1 << (q - 1 - hi_w)           # bits below hi wire
    return a, mid, lo


def apply_2q_unitary(rho_ax: torch.Tensor, u: torch.Tensor, w1: int,
                     w2: int, q: int) -> torch.Tensor:
    """rho -> U rho U^dagger with U a 4x4 on wires (w1,w2), big-endian
    pair index |w1 w2> (w1<w2 = high bit of pair). u: [4,4] or [B,4,4]."""
    assert w1 != w2
    lo, hi = (w1, w2) if w1 < w2 else (w2, w1)
    b = rho_ax.shape[0]
    d = 1 << q
    a, mid, l = _split2(lo, hi, q)
    mm = u.to(rho_ax.dtype)
    mm4 = mm.reshape(*mm.shape[:-2], 2, 2, 2, 2)  # (o_lo,o_hi, i_lo,i_hi)
    # row: view [B, hi_bits, bit_lo, mid, bit_hi, lo_bits, col_flat]
    v = rho_ax.reshape(b, a, 2, mid, 2, l, d)
    if mm4.dim() == 4:
        rho = torch.einsum("pqrs,bxrmsez->bxpmqez", mm4, v)
    else:
        rho = torch.einsum("bpqrs,bxrmsez->bxpmqez", mm4, v)
    rho = rho.reshape(b, d, d)
    # col: view [B, row_flat, hi_bits, bit_lo, mid, bit_hi, lo_bits]
    v = rho.reshape(b, d, a, 2, mid, 2, l)
    mcc = mm4.conj()
    if mcc.dim() == 4:
        rho = torch.einsum("pqrs,bdxrmse->bdxpmqe", mcc, v)
    else:
        rho = torch.einsum("bpqrs,bdxrmse->bdxpmqe", mcc, v)
    return rho.reshape(b, d, d).reshape(rho_ax.shape)


def apply_1q_channel(rho_ax: torch.Tensor, kraus: torch.Tensor, wire: int,
                     q: int) -> torch.Tensor:
    """rho -> sum_a K_a rho K_a^dagger on `wire`.

    kraus: [2,2,K] or [B,2,2,K] (last axis = kraus index).
    """
    if kraus.dim() == 3:
        k_list = [kraus[..., a] for a in range(kraus.shape[-1])]
    else:
        k_list = [kraus[:, :, :, a] for a in range(kraus.shape[-1])]
    out = None
    for k in k_list:
        term = _apply_row_1q(rho_ax.reshape(rho_ax.shape[0], 1 << q,
                                            1 << q), k, wire, q)
        term = _apply_col_1q(term, k.conj(), wire, q)
        out = term if out is None else out + term
    return out.reshape(rho_ax.shape)


def hadamard_all(rho_ax: torch.Tensor, q: int,
                 dtype: torch.dtype | None = None) -> torch.Tensor:
    """Apply H on every wire: rho -> H^⊗q rho H^⊗q (H is Hermitian)."""
    from .gates import hadamard

    h = hadamard(dtype=rho_ax.dtype if dtype is None else dtype,
                 device=rho_ax.device)
    for w in range(q):
        rho_ax = apply_1q_unitary(rho_ax, h, w, q)
    return rho_ax


def state_checks(rho: torch.Tensor, q: int) -> dict[str, float]:
    """Diagnostics: max |trace-1|, max |rho - rho^dagger|, min eigenvalue."""
    with torch.no_grad():
        tr = torch.diagonal(rho, dim1=-2, dim2=-1).sum(-1)
        herm = (rho - rho.conj().transpose(-1, -2)).abs().amax(dim=(-2, -1))
        eig = torch.linalg.eigvalsh(rho)
        return {
            "max_trace_err": float((tr - 1.0).abs().max()),
            "max_herm_err": float(herm.max()),
            "min_eig": float(eig.min()),
        }
