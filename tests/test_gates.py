"""Gate matrices: unitarity, correctness vs definitions, PennyLane
cross-check (skipped if PennyLane is missing or its API differs)."""

import numpy as np
import pytest
import torch

from qssm_dna.quantum import gates, reference

pytestmark = pytest.mark.core


@pytest.mark.parametrize("a", [0.0, 0.37, -1.2, np.pi, 2 * np.pi])
def test_unitary_gates(a):
    for m in (gates.ry(torch.tensor(a, dtype=torch.float64)),
              gates.rz(torch.tensor(a, dtype=torch.float64)),
              gates.rzz(torch.tensor(a, dtype=torch.float64))):
        eye = torch.eye(m.shape[-1], dtype=m.dtype)
        assert torch.allclose(m.conj().T @ m, eye, atol=1e-12)
        assert torch.allclose(m @ m.conj().T, eye, atol=1e-12)


def test_ry_definition():
    a = 0.7
    m = gates.ry(torch.tensor(a, dtype=torch.float64)).numpy()
    c, s = np.cos(a / 2), np.sin(a / 2)
    np.testing.assert_allclose(m, [[c, -s], [s, c]], atol=1e-15)


def test_rz_definition():
    a = 0.7
    m = gates.rz(torch.tensor(a, dtype=torch.float64)).numpy()
    np.testing.assert_allclose(
        m, np.diag([np.exp(-0.5j * a), np.exp(0.5j * a)]), atol=1e-15)


def test_rzz_definition():
    a = 0.7
    m = gates.rzz(torch.tensor(a, dtype=torch.float64)).numpy()
    d = np.diag([np.exp(-0.5j * a), np.exp(0.5j * a),
                 np.exp(0.5j * a), np.exp(-0.5j * a)])
    np.testing.assert_allclose(m, d, atol=1e-15)


def test_torch_vs_numpy_gate_matrices():
    for a in (0.0, 0.31, -0.8, np.pi):
        np.testing.assert_allclose(
            gates.ry(torch.tensor(a, dtype=torch.float64)).numpy(),
            reference.ry_np(a), atol=1e-15)
        np.testing.assert_allclose(
            gates.rz(torch.tensor(a, dtype=torch.float64)).numpy(),
            reference.rz_np(a), atol=1e-15)
        np.testing.assert_allclose(
            gates.rzz(torch.tensor(a, dtype=torch.float64)).numpy(),
            reference.rzz_np(a), atol=1e-15)


def test_pennylane_crosscheck():
    """Independent reference: PennyLane standard gate/channel matrices."""
    try:
        import pennylane as qml
    except ImportError:
        pytest.skip("pennylane not installed; cross-check skipped")
    a = 0.53
    np.testing.assert_allclose(
        np.asarray(qml.matrix(qml.RY(a, wires=0)), dtype=np.complex128),
        reference.ry_np(a), atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(qml.matrix(qml.RZ(a, wires=0)), dtype=np.complex128),
        reference.rz_np(a), atol=1e-12)
    rzz_op = getattr(qml, "RZZ", None) or getattr(qml, "IsingZZ", None)
    if rzz_op is None:
        pytest.skip("no RZZ/IsingZZ in installed pennylane; cross-check skipped")
    np.testing.assert_allclose(
        np.asarray(qml.matrix(rzz_op(a, wires=[0, 1])),
                   dtype=np.complex128),
        reference.rzz_np(a), atol=1e-12)
    g = 0.13
    try:
        kms = qml.AmplitudeDamping(g, wires=0).kraus_matrices()
    except AttributeError:
        pytest.skip("AmplitudeDamping.kraus_matrices API differs; "
                    "pennylane channel check skipped")
    ours = reference.amplitude_damping_kraus_np(g)
    for a_i in range(2):
        np.testing.assert_allclose(np.asarray(kms[a_i],
                                              dtype=np.complex128),
                                   ours[..., a_i], atol=1e-12)
