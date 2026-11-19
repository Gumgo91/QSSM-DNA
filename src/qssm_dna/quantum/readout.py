"""Terminal readouts.

raw11 (3q - 1 features at q = 4), read once at the last valid position:
    z_raw = ( <Z_i>_{i=0..q-1}, <X_i>_{i=0..q-1}, <Z_i Z_{i+1}>_{i=0..q-2} )
subset30 (2(2^q - 1) features): every Z-parity and every X-parity over a
nonempty wire subset, from the same two measurement bases.
pauli255 (4^q - 1 features): every nonidentity Pauli expectation.

Measurement groups (G = 2):
  * Z group: one bitstring draw gives all Z parities jointly, so their
    covariance is kept.
  * X group: H on all wires, then one bitstring draw gives all X parities.
On hardware every shot is a full state re-preparation; eval.resources
counts these. With shots=None the expectations are exact.
"""

from __future__ import annotations

import torch

from .density import hadamard_all, to_axes, to_matrix


def _sign_table(q: int, device=None) -> torch.Tensor:
    """[q, D] table of (-1)^{bit_i(x)}, wire 0 = most significant bit."""
    d = 1 << q
    x = torch.arange(d, device=device)
    bits = (x.unsqueeze(0) >> torch.arange(q - 1, -1, -1,
                                           device=device).unsqueeze(1)) & 1
    return 1.0 - 2.0 * bits.to(torch.float64)  # +1 for bit 0, -1 for bit 1


def z_probabilities(rho: torch.Tensor) -> torch.Tensor:
    """Diagonal of rho, clamped at zero."""
    p = rho.diagonal(dim1=-2, dim2=-1).real
    return p.clamp_min(0.0)


def terminal_readout(rho: torch.Tensor, q: int) -> torch.Tensor:
    """Exact expectations -> [B, 3q-1] (float of matching precision)."""
    b, d, _ = rho.shape
    signs = _sign_table(q, rho.device).to(rho.real.dtype)  # [q,D]
    p = z_probabilities(rho)                                # [B,D]
    z_exp = p @ signs.T                                     # [B,q]
    # <X_i> = sum_a rho[a, a ^ mask_i], mask_i = 1 << (q-1-i)
    x_cols = []
    arange_d = torch.arange(d, device=rho.device)
    for i in range(q):
        mask = 1 << (q - 1 - i)
        flipped = arange_d ^ mask
        xi = rho[:, arange_d, flipped].real.sum(-1)         # [B]
        x_cols.append(xi)
    x_exp = torch.stack(x_cols, dim=-1)                     # [B,q]
    zz_cols = []
    for i in range(q - 1):
        s = signs[i] * signs[i + 1]
        zz_cols.append(p @ s)
    zz_exp = torch.stack(zz_cols, dim=-1) if q > 1 else p.new_zeros(b, 0)
    return torch.cat([z_exp, x_exp, zz_exp], dim=-1)


def terminal_readout_shots(rho: torch.Tensor, q: int, shots: int,
                           generator: torch.Generator | None = None
                           ) -> torch.Tensor:
    """Finite-shot estimate with joint bitstring sampling per basis group.

    Returns [B, 3q-1]. Group Z: one multinomial draw on diag(rho) feeds all
    Z_i and Z_iZ_{i+1}. Group X: H^⊗q rho H^⊗q diagonal feeds all X_i.
    """
    p = z_probabilities(rho)
    p = p / p.sum(-1, keepdim=True)
    idx = torch.multinomial(p, shots, replacement=True,
                            generator=generator)            # [B,S]
    signs = _sign_table(q, rho.device).to(rho.real.dtype)   # [q,D]
    bit_signs = signs[:, idx]                               # [q,B,S] (+1/-1)
    z_est = bit_signs.mean(-1).T                            # [B,q]
    zz_cols = [(bit_signs[i] * bit_signs[i + 1]).mean(-1)
               for i in range(q - 1)]
    zz_est = torch.stack(zz_cols, -1) if q > 1 else p.new_zeros(rho.shape[0], 0)

    rho_ax = to_axes(rho, q)
    rho_x = to_matrix(hadamard_all(rho_ax, q), q)
    px = z_probabilities(rho_x)
    px = px / px.sum(-1, keepdim=True)
    idx_x = torch.multinomial(px, shots, replacement=True,
                              generator=generator)
    x_signs = signs[:, idx_x]
    x_est = x_signs.mean(-1).T                              # [B,q]
    return torch.cat([z_est, x_est, zz_est], dim=-1)


def _subset_sign_table(q: int, device=None) -> torch.Tensor:
    """[2^q-1, 2^q] table of (-1)^popcount(x & idx_mask(s)) for every
    nonempty subset s. Subset index s selects wire i iff bit i of s is
    set; wire 0 = most significant bit, so s is mapped to the index-space
    mask idx_mask(s) = sum_i bit_i(s) << (q-1-i)."""
    d = 1 << q
    x = torch.arange(d, device=device)
    s = torch.arange(1, d, device=device).unsqueeze(1)       # [S,1]
    wire_bits = (s >> torch.arange(q, device=device)
                 .unsqueeze(0)) & 1                          # [S,q]
    bitpos = (q - 1 - torch.arange(q, device=device))
    idx_mask = (wire_bits << bitpos.unsqueeze(0)).sum(-1)    # [S]
    v = x.unsqueeze(0) & idx_mask.unsqueeze(1)               # [S,D]
    pc = torch.zeros(v.shape, dtype=torch.long, device=device)
    for _ in range(q):
        pc += v & 1
        v = v >> 1
    return 1.0 - 2.0 * (pc % 2).to(torch.float64)


def subset_readout(rho: torch.Tensor, q: int) -> torch.Tensor:
    """subset30 readout: all nonempty subset parities from the same two
    measurement bases (Z, then X via H^⊗q) -> [B, 2*(2^q-1)].

    The 3q-1 readout is a subset of these (the singleton Z/X subsets and
    the adjacent two-wire Z products).
    """
    s = _subset_sign_table(q, rho.device).to(rho.real.dtype)  # [S,D]
    p_z = z_probabilities(rho)                                # [B,D]
    z_sub = p_z @ s.T                                         # [B,S]
    rho_x = to_matrix(hadamard_all(to_axes(rho, q), q), q)
    p_x = z_probabilities(rho_x)
    x_sub = p_x @ s.T
    return torch.cat([z_sub, x_sub], dim=-1)


_PAULI_MATS = torch.tensor(
    [[[1, 0], [0, 1]],   # I
     [[0, 1], [1, 0]],   # X
     [[0, -1j], [1j, 0]],  # Y
     [[1, 0], [0, -1]]], dtype=torch.complex128)  # Z


def _pauli_strings(q: int, device=None) -> torch.Tensor:
    """All nonidentity Pauli words -> [4^q-1, 2^q, 2^q] complex128."""
    words = []
    for w in range(1, 4 ** q):
        mats, rem = [], w
        for _ in range(q):
            mats.append(_PAULI_MATS[rem % 4])
            rem //= 4
        p = mats[0]
        for m in mats[1:]:
            p = torch.kron(p, m)  # last wire = least significant bit
        words.append(p)
    return torch.stack(words).to(device)


def pauli_features(rho: torch.Tensor, q: int) -> torch.Tensor:
    """pauli255 readout: expectations of all 4^q-1 nonidentity Pauli
    strings -> [B, 4^q-1], i.e. the full state. On hardware it needs more
    measurement bases than the two of raw11 and subset30.
    """
    p_all = _pauli_strings(q, rho.device)             # [S,D,D]
    return torch.einsum("bij,sji->bs",
                        rho.to(torch.complex128),
                        p_all).real.to(rho.real.dtype)


def joint_shot_features(rho: torch.Tensor, q: int, shots: int,
                        generator: torch.Generator | None = None
                        ) -> dict[str, torch.Tensor]:
    """Finite-shot features: one joint bitstring draw per basis feeds both
    the 3q-1 terminal features and the 2(2^q-1) subset features, so they
    share the samples and their covariance. Returns {"raw11", "subset30"}.
    """
    s_tab = _subset_sign_table(q, rho.device).to(rho.real.dtype)  # [S,D]
    signs = _sign_table(q, rho.device).to(rho.real.dtype)         # [q,D]
    out = {}
    for basis in ("z", "x"):
        if basis == "z":
            p = z_probabilities(rho)
        else:
            rho_x = to_matrix(hadamard_all(to_axes(rho, q), q), q)
            p = z_probabilities(rho_x)
        p = p / p.sum(-1, keepdim=True)
        idx = torch.multinomial(p, shots, replacement=True,
                                generator=generator)              # [B,S]
        sub = s_tab[:, idx].mean(-1).T                            # [B,S_sub]
        out[basis] = (sub, signs[:, idx])  # signs: [q,B,S]
    # raw11 reuses the same draws: Z singles and adjacent ZZ from the
    # z draws, X singles from the x draws
    z_singles = out["z"][1][:, :, :]                             # [q,B,S]
    z_exp = z_singles.mean(-1).T                                 # [B,q]
    zz = torch.stack([(z_singles[i] * z_singles[i + 1]).mean(-1)
                      for i in range(q - 1)], dim=-1)
    x_exp = out["x"][1].mean(-1).T
    raw11 = torch.cat([z_exp, x_exp, zz], dim=-1)
    subset = torch.cat([out["z"][0], out["x"][0]], dim=-1)
    return {"raw11": raw11, "subset30": subset}


def subset_score_tables(weight: torch.Tensor, q: int) -> torch.Tensor:
    """Direct-score lookup for a linear head on subset features.

    For basis b in {Z=0, X=1} and bitstring outcome y in [0,2^q):
        f_b(y) = sum_{S nonempty} w_{b,S} * prod_{i in S} (1-2*y_i)
    -> [2, 2^q] table. Then logit = bias + sum_b E[f_b(Y_b)].
    weight: [2*(2^q-1)] head weight in subset-readout order
    (all Z subsets, then all X subsets)."""
    s_tab = _subset_sign_table(q, weight.device).to(weight.dtype)
    w_z = weight[: (1 << q) - 1]
    w_x = weight[(1 << q) - 1:]
    return torch.stack([s_tab.T @ w_z, s_tab.T @ w_x])           # [2,D]


def score_from_probs(p_z: torch.Tensor, p_x: torch.Tensor,
                     table: torch.Tensor, bias: float) -> torch.Tensor:
    """E[f_b] under each basis distribution -> scalar logit per sample."""
    return p_z @ table[0] + p_x @ table[1] + bias


def shot_logit_variance(p_z: torch.Tensor, p_x: torch.Tensor,
                        table: torch.Tensor, n_dir: int,
                        shots: int) -> torch.Tensor:
    """Var[logit_hat] = sum_{r,b} Var[f_b(Y_{r,b})] / (R^2 * S) for R
    independent directions (RC draws) and S shots per group."""
    var_z = p_z @ (table[0] ** 2) - (p_z @ table[0]) ** 2
    var_x = p_x @ (table[1] ** 2) - (p_x @ table[1]) ** 2
    return (var_z + var_x) / (n_dir ** 2 * shots)


def raw_feature_dim(q: int) -> int:
    return 3 * q - 1
