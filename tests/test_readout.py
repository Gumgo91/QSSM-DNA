"""Terminal readout: exact expectations bounded, X-group consistency via
H rotation, joint bitstring shot sampling, feature dim 3q-1."""

import numpy as np
import pytest
import torch

from qssm_dna.models.encoder import QSSMEncoder
from qssm_dna.quantum import density, gates, readout
from conftest import random_density

pytestmark = pytest.mark.core


@pytest.mark.parametrize("q", [1, 2, 3])
def test_expectations_bounded(q):
    rho = random_density(4, q, seed=11)
    z = readout.terminal_readout(rho, q)
    assert z.shape == (4, 3 * q - 1)
    assert float(z.abs().max()) <= 1.0 + 1e-9


def test_x_group_matches_hadamard_rotation():
    """<X_i> computed by direct sum must equal Z_i measured after H^⊗q."""
    q = 3
    rho = random_density(2, q, seed=13)
    z_direct = readout.terminal_readout(rho, q)[:, q:2 * q]
    rho_x = density.to_matrix(
        density.hadamard_all(density.to_axes(rho, q), q), q)
    signs = readout._sign_table(q, rho.device)
    p = readout.z_probabilities(rho_x)
    z_after_h = p @ signs.T.to(p.dtype)
    torch.testing.assert_close(z_direct, z_after_h.to(z_direct.dtype),
                               atol=1e-10, rtol=0)


def test_shot_readout_converges_and_covariance():
    """Joint bitstring draws: Z_i and ZZ share the same samples.
    Check means approach exact values and correlated observables stay
    correlated across seeds (covariance preserved)."""
    q = 2
    rng = np.random.default_rng(0)
    # state with strong Z0-Z1 correlation: mixture of |00>,|11>
    rho = torch.zeros(1, 4, 4, dtype=torch.complex128)
    rho[0, 0, 0] = 0.5
    rho[0, 3, 3] = 0.5
    z_exact = readout.terminal_readout(rho, q)
    g = torch.Generator().manual_seed(7)
    z_s = readout.terminal_readout_shots(rho, q, shots=8192, generator=g)
    # |00>/|11|: <Z0>=<Z1>=+1 exactly; shot noise only on the mixture
    assert torch.allclose(z_s[:, :q], z_exact[:, :q], atol=2e-2)
    # Z0Z1 = +1 exactly in both mixture components -> no shot variance
    assert abs(z_s[0, -1].item() - 1.0) < 1e-12


def test_shot_vs_exact_small_gap():
    torch.manual_seed(5)
    q = 2
    enc = QSSMEncoder(n_qubits=q)
    tokens = torch.randint(0, 4, (2, 8))
    mask = torch.ones(2, 8, dtype=torch.bool)
    rho = enc.forward_state(tokens, mask)
    z_exact = readout.terminal_readout(rho, q)
    g = torch.Generator().manual_seed(0)
    z_s = readout.terminal_readout_shots(rho, q, shots=4096, generator=g)
    assert (z_s - z_exact).abs().max() < 0.15  # ~3 sigma of shot noise


# ------------------------------------------- subset, Pauli and shot readouts

@pytest.mark.parametrize("q", [2, 3])
def test_subset_readout_dims_and_singletons(q):
    """subset_readout = all nonempty Z/X subset products, same two bases.
    Singleton rows must reproduce <Z_i>, <X_i> exactly."""
    rho = random_density(3, q, seed=17)
    z = readout.subset_readout(rho, q)
    s = (1 << q) - 1
    assert z.shape == (3, 2 * s)
    base = readout.terminal_readout(rho, q)
    # singleton Z_i: subset index s_idx=1<<i -> row (1<<i)-1
    for i in range(q):
        row = (1 << i) - 1
        torch.testing.assert_close(z[:, row], base[:, i],
                                   atol=1e-10, rtol=0)
        torch.testing.assert_close(z[:, s + row], base[:, q + i],
                                   atol=1e-10, rtol=0)


def test_subset_readout_contains_adjacent_zz():
    """Adjacent Z_i Z_{i+1} of the original readout are subset rows with
    bits i,i+1 set."""
    q = 3
    rho = random_density(3, q, seed=19)
    z = readout.subset_readout(rho, q)
    base = readout.terminal_readout(rho, q)
    for i in range(q - 1):
        row = ((1 << i) | (1 << (i + 1))) - 1
        torch.testing.assert_close(z[:, row], base[:, 2 * q + i],
                                   atol=1e-10, rtol=0)


def test_pauli_features_complete_hermitian_basis():
    """All 4^q-1 nonidentity Pauli expectations: reconstruct rho from
    features and compare to the original (completeness)."""
    q = 2
    rho = random_density(3, q, seed=23)
    feats = readout.pauli_features(rho, q)          # [B,15]
    assert feats.shape == (3, 4 ** q - 1)
    words = readout._pauli_strings(q)               # [15,D,D]
    d = 1 << q
    # rho = 2^-q (I + sum_a r_a P_a); r_a = feats
    eye = torch.eye(d, dtype=torch.complex128).expand(rho.shape[0], d, d)
    recon = (eye + torch.einsum(
        "bs,sij->bij", feats.to(torch.complex128), words)) / d
    torch.testing.assert_close(recon, rho.to(torch.complex128),
                               atol=1e-10, rtol=1e-10)


def test_pauli_features_match_manual_kron():
    """Spot-check two specific words against hand-built kron products."""
    q = 3
    rho = random_density(2, q, seed=29)
    feats = readout.pauli_features(rho, q)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
    Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
    I = torch.eye(2, dtype=torch.complex128)
    # word digits are little-endian over wires: w = d0 + 4*d1 + 16*d2
    # Z on wire0 -> digits (3,0,0) -> w=3 -> row 2
    p = torch.kron(Z, torch.kron(I, I))
    exp = torch.einsum("bij,ji->b", rho.to(torch.complex128), p).real
    torch.testing.assert_close(feats[:, 2], exp.to(feats.dtype),
                               atol=1e-10, rtol=0)
    # Y on wire1 -> digits (0,2,0) -> w=8 -> row 7
    p2 = torch.kron(I, torch.kron(Y, I))
    exp2 = torch.einsum("bij,ji->b", rho.to(torch.complex128), p2).real
    torch.testing.assert_close(feats[:, 7], exp2.to(feats.dtype),
                               atol=1e-10, rtol=0)


def test_joint_shot_features_shared_draws():
    """raw11 rows inside the shot estimate are identical samples of the
    subset30 rows (joint bitstrings, not independent noise)."""
    q = 3
    rho = random_density(3, q, seed=31)
    g1 = torch.Generator().manual_seed(5)
    g2 = torch.Generator().manual_seed(5)
    a = readout.joint_shot_features(rho, q, 512, g1)
    b = readout.joint_shot_features(rho, q, 512, g2)
    torch.testing.assert_close(a["raw11"], b["raw11"], atol=0, rtol=0)
    torch.testing.assert_close(a["subset30"], b["subset30"], atol=0, rtol=0)
    # raw11 Z singles == subset30 singleton rows of the same draws
    for i in range(q):
        torch.testing.assert_close(a["raw11"][:, i],
                                   a["subset30"][:, (1 << i) - 1],
                                   atol=0, rtol=0)
        torch.testing.assert_close(a["raw11"][:, q + i],
                                   a["subset30"][:, (1 << q) - 1 +
                                                 (1 << i) - 1],
                                   atol=0, rtol=0)


def test_score_lookup_identity():
    """f_b(y) lookup + bias must equal the linear head applied to exact
    subset features."""
    q = 2
    rho = random_density(4, q, seed=37)
    z = readout.subset_readout(rho, q)                    # [B,14]
    torch.manual_seed(0)
    w = torch.randn(z.shape[1], dtype=torch.float64)
    bias = 0.37
    table = readout.subset_score_tables(w, q)
    rho_x = density.to_matrix(
        density.hadamard_all(density.to_axes(rho, q), q), q)
    p_z = readout.z_probabilities(rho).to(torch.float64)
    p_x = readout.z_probabilities(rho_x).to(torch.float64)
    logit_lookup = readout.score_from_probs(p_z, p_x, table, bias)
    logit_dot = z.to(torch.float64) @ w + bias
    torch.testing.assert_close(logit_lookup, logit_dot, atol=1e-10,
                               rtol=1e-10)
    # the shot variance is finite and nonnegative
    v = readout.shot_logit_variance(p_z, p_x, table, n_dir=1, shots=256)
    assert torch.all(v >= -1e-9)
