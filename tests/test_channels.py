"""Amplitude damping: Kraus completeness, boundary values, torch
contraction vs NumPy dense reference, ancilla-qubit dilation match."""

import numpy as np
import pytest
import torch

from qssm_dna.quantum import channels, density, gates, reference
from conftest import random_density, random_pure_density

pytestmark = pytest.mark.core

GAMMAS = [0.0, 1e-4, 0.13, 0.5, 1.0]


@pytest.mark.parametrize("gamma", GAMMAS)
def test_kraus_completeness(gamma):
    ks = gates.amplitude_damping_kraus(
        torch.tensor(gamma, dtype=torch.float64))
    acc = sum(ks[..., a].conj().T @ ks[..., a] for a in range(2))
    assert torch.allclose(acc, torch.eye(2, dtype=torch.complex128),
                          atol=1e-12)
    assert reference.kraus_completeness_np(gamma) < 1e-12


@pytest.mark.parametrize("gamma", GAMMAS)
def test_dilation_unitary(gamma):
    u = reference.u_gamma_np(gamma)
    np.testing.assert_allclose(u.conj().T @ u, np.eye(4), atol=1e-12)


@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("q", [1, 2, 3])
@pytest.mark.parametrize("wire", [0, 1, 2])
def test_dilation_matches_kraus(gamma, q, wire):
    if wire >= q:
        pytest.skip("wire out of range")
    rho = random_pure_density(q, seed=42 + wire)
    got = reference.dilated_amplitude_damping_np(rho, gamma, wire, q)
    want = reference.amplitude_damping_np(rho, gamma, wire, q)
    np.testing.assert_allclose(got, want, atol=1e-12)


@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("q", [1, 2, 3])
def test_torch_channel_vs_numpy_dense(gamma, q):
    rho_np = np.stack([random_pure_density(q, seed=i) for i in range(3)])
    rho_t = torch.tensor(rho_np, dtype=torch.complex128)
    for wire in range(q):
        ax = density.to_axes(rho_t, q)
        out_t = channels.amplitude_damping(
            ax, torch.full((3,), gamma, dtype=torch.float64), wire, q)
        out_t = density.to_matrix(out_t, q).numpy()
        out_np = np.stack([reference.amplitude_damping_np(r, gamma, wire, q)
                           for r in rho_np])
        np.testing.assert_allclose(out_t, out_np, atol=1e-10)


def test_gamma_zero_is_identity():
    q = 2
    rho = random_density(2, q, seed=5)
    out = density.to_matrix(
        channels.amplitude_damping(
            density.to_axes(rho, q),
            torch.zeros(2, dtype=torch.float64), 0, q), q)
    torch.testing.assert_close(out, rho, atol=1e-12, rtol=0)


def test_gamma_one_resets_wire():
    # gamma=1 on wire 0 of |11><11| (single) -> |01><01| (wire0 -> |0>)
    q = 2
    rho = torch.zeros(1, 4, 4, dtype=torch.complex128)
    rho[0, 3, 3] = 1.0  # |11>
    out = density.to_matrix(
        channels.amplitude_damping(
            density.to_axes(rho, q),
            torch.ones(1, dtype=torch.float64), 0, q), q)
    want = torch.zeros(1, 4, 4, dtype=torch.complex128)
    want[0, 1, 1] = 1.0  # |01>: wire0=0, wire1=1
    torch.testing.assert_close(out, want, atol=1e-12, rtol=0)


@pytest.mark.parametrize("q", [1, 2, 3])
def test_channel_preserves_state(q):
    rho = random_density(4, q, seed=7)
    gam = torch.rand(4, dtype=torch.float64) * 0.9 + 0.05
    ax = density.to_axes(rho, q)
    for w in range(q):
        ax = channels.amplitude_damping(ax, gam, w, q)
    out = density.to_matrix(ax, q)
    checks = density.state_checks(out, q)
    assert checks["max_trace_err"] < 1e-10
    assert checks["max_herm_err"] < 1e-10
    assert checks["min_eig"] > -1e-9


@pytest.mark.parametrize("q", [1, 2])
def test_torch_2q_unitary_vs_numpy_dense(q):
    rho_np = np.stack([random_pure_density(q, seed=i) for i in range(2)])
    rho_t = torch.tensor(rho_np)
    a = 0.77
    for w in range(q - 1):
        out_t = density.to_matrix(
            density.apply_2q_unitary(
                density.to_axes(rho_t, q),
                gates.rzz(torch.tensor(a, dtype=torch.float64)),
                w, w + 1, q), q).numpy()
        out_np = np.stack([reference.apply_2q_unitary_np(
            r, reference.rzz_np(a), w, w + 1, q) for r in rho_np])
        np.testing.assert_allclose(out_t, out_np, atol=1e-12)
