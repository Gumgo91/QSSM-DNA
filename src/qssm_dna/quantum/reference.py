"""Independent NumPy reference backend (dense, complex128).

Reference for the torch contraction backend in the tests, built from dense
D x D operators. Basis order is big-endian: wire 0 = most significant bit.

Ancilla dilation: the enlarged state is rho ⊗ |0><0| in the basis
|memory, ancilla>, i.e. the ancilla is the last (least significant) wire of
the q+1 qubit register.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- gates

def ry_np(a: float) -> np.ndarray:
    c, s = np.cos(a / 2), np.sin(a / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def rz_np(a: float) -> np.ndarray:
    return np.array([[np.exp(-0.5j * a), 0.0],
                     [0.0, np.exp(0.5j * a)]], dtype=np.complex128)


def rzz_np(a: float) -> np.ndarray:
    return np.diag([np.exp(-0.5j * a), np.exp(0.5j * a),
                    np.exp(0.5j * a), np.exp(-0.5j * a)]).astype(np.complex128)


def amplitude_damping_kraus_np(gamma: float) -> np.ndarray:
    """[2,2,2] array; last axis indexes K_0, K_1."""
    k0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - gamma)]],
                  dtype=np.complex128)
    k1 = np.array([[0.0, np.sqrt(gamma)], [0.0, 0.0]], dtype=np.complex128)
    return np.stack([k0, k1], axis=-1)


def u_gamma_np(gamma: float) -> np.ndarray:
    """Stinespring-type dilation unitary on (memory, ancilla), ancilla
    initialised to |0>. Basis |m,a> = |00>,|01>,|10>,|11>."""
    s1, sg = np.sqrt(1.0 - gamma), np.sqrt(gamma)
    return np.array(
        [[1.0, 0.0, 0.0, 0.0],
         [0.0, s1, sg, 0.0],
         [0.0, -sg, s1, 0.0],
         [0.0, 0.0, 0.0, 1.0]], dtype=np.complex128)


# ------------------------------------------------------- dense expanders

def expand_1q(u: np.ndarray, wire: int, q: int) -> np.ndarray:
    """Dense 2^q x 2^q operator of 1-qubit u on `wire` (big-endian)."""
    out = np.array([[1.0]], dtype=np.complex128)
    for w in range(q):
        out = np.kron(out, u if w == wire else np.eye(2))
    return out


def expand_2q(u4: np.ndarray, w1: int, w2: int, q: int) -> np.ndarray:
    """Dense 2^q x 2^q operator of 2-qubit u4 on wires (w1,w2).

    u4 is indexed |w1 w2> big-endian within the pair (w1 high bit).
    Works for arbitrary (w1 != w2), not only adjacent wires.
    """
    d = 1 << q
    bits = np.array([[(x >> (q - 1 - w)) & 1 for w in range(q)]
                     for x in range(d)])  # [d, q]
    untouched = [w for w in range(q) if w not in (w1, w2)]
    match = np.ones((d, d), dtype=bool)
    if untouched:
        match = np.all(bits[:, None, untouched] == bits[None, :, untouched],
                       axis=-1)
    ai = bits[:, w1] * 2 + bits[:, w2]  # pair index, w1 = high bit
    aj = ai
    return u4[np.ix_(ai, aj)] * match


def apply_1q_unitary_np(rho: np.ndarray, u: np.ndarray, wire: int,
                        q: int) -> np.ndarray:
    uu = expand_1q(u, wire, q)
    return uu @ rho @ uu.conj().T


def apply_2q_unitary_np(rho: np.ndarray, u4: np.ndarray, w1: int, w2: int,
                        q: int) -> np.ndarray:
    lo, hi = (w1, w2) if w1 < w2 else (w2, w1)
    uu = expand_2q(u4, lo, hi, q)
    return uu @ rho @ uu.conj().T


def amplitude_damping_np(rho: np.ndarray, gamma: float, wire: int,
                         q: int) -> np.ndarray:
    ks = amplitude_damping_kraus_np(gamma)
    out = np.zeros_like(rho)
    for a in range(ks.shape[-1]):
        kk = expand_1q(ks[..., a], wire, q)
        out = out + kk @ rho @ kk.conj().T
    return out


def kraus_completeness_np(gamma: float) -> float:
    ks = amplitude_damping_kraus_np(gamma)
    acc = sum(ks[..., a].conj().T @ ks[..., a] for a in range(ks.shape[-1]))
    return float(np.abs(acc - np.eye(2)).max())


# ------------------------------------------------------- ancilla dilation

def dilated_amplitude_damping_np(rho: np.ndarray, gamma: float, wire: int,
                                 q: int) -> np.ndarray:
    """Channel on `wire` via ancilla qubit (last wire) initialised to |0>:

        Tr_a[ U_gamma (rho ⊗ |0><0|) U_gamma† ]

    Basis |memory, ancilla>; ancilla = least significant wire.
    """
    d = rho.shape[0]
    rho_e = np.kron(rho, np.array([[1.0, 0.0], [0.0, 0.0]],
                                  dtype=np.complex128))
    u = expand_2q(u_gamma_np(gamma), min(wire, q), max(wire, q), q + 1)
    out = u @ rho_e @ u.conj().T
    # partial trace over last wire (index q of q+1 wires)
    ax = out.reshape([2] * (2 * (q + 1)))
    return np.trace(ax, axis1=q, axis2=2 * q + 1).reshape(d, d)
