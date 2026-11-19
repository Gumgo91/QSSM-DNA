"""State-affine (Pauli transfer) form of the trained recurrence. CPU only.

Because the controller sees only the current token, a trained encoder applies
one of five fixed CPTP maps Phi_x (x in A,C,G,T,N). In the normalised Pauli
basis rho = (I + sum_P r_P P) / 2^q, each map is affine,

    r_t = A_x r_{t-1} + c_x,   A_x in R^{(4^q-1) x (4^q-1)},

with c_x != 0 because amplitude damping is non-unital, and every readout is a
fixed linear selection of r_L. This module builds (A_x, c_x), the top
Lyapunov exponent along real sequences (the per-token decay rate of the
memory of earlier inputs) and the table of effective control parameters.
"""

from __future__ import annotations

import numpy as np
import torch

from ..quantum import density


def pauli_basis(q: int) -> torch.Tensor:
    """[4^q, D, D] complex128, index 0 = identity (big-endian wire order)."""
    from ..quantum.readout import _PAULI_MATS
    mats = []
    for w in range(4 ** q):
        digs, rem = [], w
        for _ in range(q):
            digs.append(rem % 4)
            rem //= 4
        m = _PAULI_MATS[digs[-1]]
        for dgt in reversed(digs[:-1]):
            m = torch.kron(m, _PAULI_MATS[dgt])
        mats.append(m)
    return torch.stack(mats)


@torch.no_grad()
def transfer_matrices(enc, tokens=(0, 1, 2, 3, 4)) -> dict[int, np.ndarray]:
    """R_x[P, Q] = Tr(P Phi_x(Q)) / 2^q for the trained one-token map."""
    q = enc.q
    d = 1 << q
    enc = enc.double().cpu()
    enc.cdtype = torch.complex128
    basis = pauli_basis(q)                                   # [n, D, D]
    n = basis.shape[0]
    out = {}
    for x in tokens:
        theta, gamma = enc.controller.forward_token(
            torch.full((n,), x, dtype=torch.long))
        rho_ax = density.to_axes(basis.clone(), q)
        rho_ax = enc._step_ax(rho_ax, theta, gamma,
                              torch.ones(n, dtype=torch.bool))
        img = density.to_matrix(rho_ax, q)                   # Phi_x(Q)
        r = torch.einsum("pij,qji->pq", basis, img).real / d
        out[x] = r.numpy()
    return out


def affine_parts(r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(A, c) from a Pauli transfer matrix; row/col 0 = identity."""
    return r[1:, 1:], r[1:, 0]


def lyapunov(mats: dict[int, np.ndarray], seqs: list[np.ndarray],
             renorm: int = 8) -> np.ndarray:
    """Top Lyapunov exponent per sequence: (1/L) log ||A_{x_L}...A_{x_1} v||
    for a random unit start vector, with periodic renormalisation."""
    rng = np.random.default_rng(0)
    out = []
    for toks in seqs:
        v = rng.standard_normal(next(iter(mats.values())).shape[0])
        v /= np.linalg.norm(v)
        logn = 0.0
        for i, x in enumerate(toks):
            v = mats[int(x)] @ v
            if (i + 1) % renorm == 0:
                nv = np.linalg.norm(v)
                logn += np.log(nv)
                v /= nv
        logn += np.log(np.linalg.norm(v))
        out.append(logn / len(toks))
    return np.array(out)


def control_table(enc) -> np.ndarray:
    """Effective parameters: per-token (8 angles + 4 gammas) for A,C,G,T,N
    plus the 3 RZZ angles -> 63 numbers."""
    with torch.no_grad():
        th, ga = enc.controller.forward_token(torch.arange(5))
        return np.concatenate([th.reshape(5, -1).double().numpy().ravel(),
                               ga.double().numpy().ravel(),
                               enc.rzz_angles.detach().double().numpy()
                               .ravel()])
