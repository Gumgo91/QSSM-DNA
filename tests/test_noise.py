"""Evaluation-only noise channels (quantum/noise.py) against dense
NumPy references built from Pauli twirls and explicit bit-flip sums."""

import itertools

import numpy as np
import pytest
import torch

from qssm_dna.models.encoder import QSSMEncoder
from qssm_dna.quantum import density, noise
from conftest import random_density

pytestmark = pytest.mark.core

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _on(ops: dict, q: int) -> np.ndarray:
    """Kronecker product with ops[w] on wire w (wire 0 most significant)."""
    out = np.array([[1.0 + 0j]])
    for w in range(q):
        out = np.kron(out, ops.get(w, I2))
    return out


def _apply(fn, rho: torch.Tensor, q: int) -> np.ndarray:
    out = fn(density.to_axes(rho, q))
    return density.to_matrix(out, q).numpy()


@pytest.mark.parametrize("w", [0, 1, 2])
def test_depolarize_1q_matches_pauli_twirl(w):
    q, p = 3, 0.37
    rho = random_density(2, q, seed=w)
    got = _apply(lambda r: noise.depolarize_1q(r, p, w, q), rho, q)
    for b in range(2):
        r = rho[b].numpy()
        twirl = sum(_on({w: P}, q) @ r @ _on({w: P}, q).conj().T
                    for P in (I2, X, Y, Z)) / 4
        ref = (1 - p) * r + p * twirl
        np.testing.assert_allclose(got[b], ref, atol=1e-12)


@pytest.mark.parametrize("pair", [(0, 1), (1, 2), (0, 2), (2, 0)])
def test_depolarize_2q_matches_pauli_twirl(pair):
    q, p = 3, 0.21
    w1, w2 = pair
    rho = random_density(2, q, seed=7)
    got = _apply(lambda r: noise.depolarize_2q(r, p, w1, w2, q), rho, q)
    for b in range(2):
        r = rho[b].numpy()
        twirl = sum(_on({w1: P1, w2: P2}, q) @ r
                    @ _on({w1: P1, w2: P2}, q).conj().T
                    for P1 in (I2, X, Y, Z) for P2 in (I2, X, Y, Z)) / 16
        np.testing.assert_allclose(got[b], (1 - p) * r + p * twirl,
                                   atol=1e-12)


@pytest.mark.parametrize("w", [0, 2])
def test_dephase_matches_kraus(w):
    q, p = 3, 0.15
    rho = random_density(1, q, seed=3)
    got = _apply(lambda r: noise.dephase(r, p, w, q), rho, q)[0]
    r = rho[0].numpy()
    zw = _on({w: Z}, q)
    np.testing.assert_allclose(got, (1 - p) * r + p * zw @ r @ zw,
                               atol=1e-12)


def test_channels_preserve_trace_and_positivity():
    q = 3
    rho = density.to_axes(random_density(4, q, seed=11), q)
    out = noise.depolarize_1q(rho, 0.3, 1, q)
    out = noise.depolarize_2q(out, 0.4, 0, 2, q)
    out = noise.dephase(out, 0.2, 2, q)
    m = density.to_matrix(out, q)
    chk = density.state_checks(m, q)
    assert chk["max_trace_err"] < 1e-12
    assert chk["max_herm_err"] < 1e-12
    assert chk["min_eig"] > -1e-12


def test_readout_flip_matches_bruteforce():
    q, e = 3, 0.07
    rng = np.random.default_rng(0)
    p = rng.random((2, 8))
    p /= p.sum(1, keepdims=True)
    got = noise.readout_flip_probs(torch.tensor(p), e, q).numpy()
    ref = np.zeros_like(p)
    for x, y in itertools.product(range(8), range(8)):
        flips = bin(x ^ y).count("1")
        ref[:, y] += p[:, x] * e ** flips * (1 - e) ** (q - flips)
    np.testing.assert_allclose(got, ref, atol=1e-14)


def test_noiseless_model_reproduces_forward_state():
    torch.manual_seed(2)
    enc = QSSMEncoder(n_qubits=3, cell_depth=1, hidden_dim=16)
    toks = torch.tensor(np.random.default_rng(1).integers(0, 4, (3, 25)))
    lens = torch.tensor([25, 17, 9])
    mask = torch.arange(25).unsqueeze(0) < lens.unsqueeze(1)
    with torch.no_grad():
        ref = enc.forward_state(toks, mask)
    got = noise.noisy_forward_state(enc, toks, mask, noise.NoiseModel())
    torch.testing.assert_close(got, ref, atol=0, rtol=0)


def test_noise_moves_state_towards_maximally_mixed():
    torch.manual_seed(2)
    enc = QSSMEncoder(n_qubits=3, cell_depth=1, hidden_dim=16)
    toks = torch.tensor(np.random.default_rng(1).integers(0, 4, (2, 40)))
    mask = torch.ones(2, 40, dtype=torch.bool)
    pur = []
    for p in (0.0, 1e-2, 1e-1):
        rho = noise.noisy_forward_state(enc, toks, mask,
                                        noise.NoiseModel(p1=p, p2=p))
        pur.append(float(torch.einsum("bij,bji->b", rho, rho).real.mean()))
    assert pur[0] > pur[1] > pur[2] > 1 / 8 - 1e-6
