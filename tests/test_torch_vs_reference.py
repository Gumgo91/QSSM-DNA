"""End-to-end: torch contraction backend vs independent NumPy dense
reference on the FULL recurrent update (damping + RY/RZ + RZZ), q<=3,
complex128, tolerance 1e-8."""

import numpy as np
import pytest
import torch

from qssm_dna.models.encoder import QSSMEncoder
from qssm_dna.quantum import reference

pytestmark = pytest.mark.core


def reference_forward(tokens: np.ndarray, mask: np.ndarray, enc) -> np.ndarray:
    """Replay the model's controls through dense numpy ops."""
    q, depth = enc.q, enc.cell_depth
    with torch.no_grad():
        theta, gamma = enc.controls(
            torch.tensor(tokens))          # [B,L,depth,q,2], [B,L,q]
        theta = theta.numpy().astype(np.float64)
        gamma = gamma.numpy().astype(np.float64)
        rzz = enc.rzz_angles.detach().numpy().astype(np.float64)
    b, l = tokens.shape
    d = 1 << q
    rhos = np.zeros((b, d, d), dtype=np.complex128)
    for i in range(b):
        rho = np.zeros((d, d), np.complex128)
        rho[0, 0] = 1.0
        for t in range(l):
            if not mask[i, t]:
                continue
            for w in range(q):
                rho = reference.amplitude_damping_np(
                    rho, float(gamma[i, t, w]), w, q)
            for l_ in range(depth):
                for w in range(q):
                    rho = reference.apply_1q_unitary_np(
                        rho, reference.ry_np(float(theta[i, t, l_, w, 0])),
                        w, q)
                for w in range(q):
                    rho = reference.apply_1q_unitary_np(
                        rho, reference.rz_np(float(theta[i, t, l_, w, 1])),
                        w, q)
                for w in range(q - 1):
                    rho = reference.apply_2q_unitary_np(
                        rho, reference.rzz_np(float(rzz[l_, w])), w, w + 1,
                        q)
        rhos[i] = rho
    return rhos


@pytest.mark.parametrize("q", [1, 2, 3])
def test_full_update_matches_reference(q):
    torch.manual_seed(0)
    enc = QSSMEncoder(n_qubits=q, cell_depth=1,
                      dtype=torch.complex128).double()
    rng = np.random.default_rng(3)
    tokens = rng.integers(0, 5, size=(3, 5))
    mask = np.ones((3, 5), dtype=bool)
    mask[0, -2:] = False  # exercise PAD skip
    rho_t = enc.forward_state(torch.tensor(tokens),
                              torch.tensor(mask)).detach().numpy()
    rho_np = reference_forward(tokens, mask, enc)
    np.testing.assert_allclose(rho_t, rho_np, atol=1e-8)


@pytest.mark.parametrize("q", [2])
def test_cell_depth_two_matches(q):
    torch.manual_seed(1)
    enc = QSSMEncoder(n_qubits=q, cell_depth=2,
                      dtype=torch.complex128).double()
    rng = np.random.default_rng(9)
    tokens = rng.integers(0, 5, size=(2, 4))
    mask = np.ones((2, 4), dtype=bool)
    rho_t = enc.forward_state(torch.tensor(tokens),
                              torch.tensor(mask)).detach().numpy()
    rho_np = reference_forward(tokens, mask, enc)
    np.testing.assert_allclose(rho_t, rho_np, atol=1e-8)
