"""Sequence semantics: RC involution, PAD invariance, batch independence,
chunked streaming equality, causality, N kept distinct, state reset."""

import numpy as np
import pytest
import torch

from qssm_dna.data.tokenizer import (N_IDX, PAD_IDX, decode, encode,
                                     rc_array)
from qssm_dna.models.encoder import QSSMEncoder, rc_tokens

pytestmark = pytest.mark.core


def test_rc_involution():
    x = encode("ACGTNACGT")
    np.testing.assert_array_equal(rc_array(rc_array(x)), x)


def test_rc_token_batch_roundtrip():
    tokens = torch.tensor([[0, 1, 2, 3], [0, 4, 1, 5]])  # row1 last=PAD
    mask = torch.tensor([[True] * 4, [True] * 3 + [False]])
    rc = rc_tokens(tokens, mask)
    np.testing.assert_array_equal(rc[0].numpy(), rc_array(tokens[0].numpy()))
    # row1: valid "ANT"? -> comp of [0,4,1] reversed = comp([1,4,0])=[2,4,3]
    np.testing.assert_array_equal(rc[1].numpy(), [2, 4, 3, PAD_IDX])
    rt = rc_tokens(rc, mask)
    torch.testing.assert_close(rt, tokens)


def test_pad_does_not_update_state():
    torch.manual_seed(0)
    enc = QSSMEncoder(n_qubits=2, dtype=torch.complex128)
    toks = torch.tensor([[0, 1, 2]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    rho_short = enc.forward_state(toks, mask)
    pad = torch.cat([toks, torch.full((1, 4), PAD_IDX)], dim=1)
    pmask = torch.cat([mask, torch.zeros(1, 4, dtype=torch.bool)], dim=1)
    rho_pad = enc.forward_state(pad, pmask)
    torch.testing.assert_close(rho_pad, rho_short, atol=1e-12, rtol=0)


def test_pad_only_never_updates():
    enc = QSSMEncoder(n_qubits=2)
    toks = torch.full((1, 4), PAD_IDX)
    mask = torch.zeros(1, 4, dtype=torch.bool)
    rho = enc.forward_state(toks, mask)
    assert abs(rho[0, 0, 0].item() - 1.0) < 1e-12
    assert rho.abs().sum().item() == pytest.approx(1.0)


def test_batch_independence():
    torch.manual_seed(0)
    enc = QSSMEncoder(n_qubits=2, dtype=torch.complex128)
    toks = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0], [1, 1, 4, 2]])
    mask = torch.ones(3, 4, dtype=torch.bool)
    z_batch = enc.encode(toks, mask, representation="raw")
    for i in range(3):
        z_one = enc.encode(toks[i:i + 1], mask[i:i + 1],
                           representation="raw")
        torch.testing.assert_close(z_one, z_batch[i:i + 1],
                                   atol=1e-12, rtol=0)


def test_chunked_streaming_matches_full():
    torch.manual_seed(0)
    enc = QSSMEncoder(n_qubits=3, dtype=torch.complex128)
    toks = torch.randint(0, 5, (2, 13))
    mask = torch.ones(2, 13, dtype=torch.bool)
    mask[1, 10:] = False
    full = enc.forward_state(toks, mask)
    for chunk in (1, 3, 5, 13):
        chunked = enc.forward_state_chunked(toks, mask, chunk)
        torch.testing.assert_close(chunked, full, atol=1e-12, rtol=0)


def test_causality_future_tokens_do_not_change_prefix_state():
    enc = QSSMEncoder(n_qubits=2)
    toks = torch.tensor([[0, 1, 2, 3]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    states = enc.forward_state(toks, mask, return_states=True)
    toks2 = toks.clone()
    toks2[0, 3] = 1  # change LAST token
    states2 = enc.forward_state(toks2, mask, return_states=True)
    for t in range(3):
        torch.testing.assert_close(states[t], states2[t],
                                   atol=1e-12, rtol=0)
    assert not torch.allclose(states[3], states2[3])


def test_n_is_distinct_input():
    enc = QSSMEncoder(n_qubits=2)
    base = torch.tensor([[0, 1, 2]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    variants = {}
    for tok in (0, 1, 2, 3, N_IDX):
        t = base.clone()
        t[0, 1] = tok
        variants[tok] = enc.encode(t, mask, representation="raw")
    for a in variants:
        for b in variants:
            if a != b:
                assert not torch.allclose(variants[a], variants[b],
                                          atol=1e-6), \
                    f"tokens {a},{b} produced identical features"


def test_state_resets_between_calls():
    enc = QSSMEncoder(n_qubits=2)
    toks = torch.tensor([[0, 1]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    z1 = enc.encode(toks, mask, representation="raw")
    z2 = enc.encode(toks, mask, representation="raw")
    torch.testing.assert_close(z1, z2)  # deterministic, no carryover


def test_rc_average_feature_symmetry():
    """RC-averaged raw features of x and RC(x) must match to ~1e-6."""
    torch.manual_seed(0)
    enc = QSSMEncoder(n_qubits=2, dtype=torch.complex128)
    toks = torch.tensor([[0, 1, 2, 3, 0]])
    mask = torch.ones(1, 5, dtype=torch.bool)
    z_fwd = enc.encode(toks, mask, representation="raw", rc_average=True)
    rc = rc_tokens(toks, mask)
    z_rev = enc.encode(rc, mask, representation="raw", rc_average=True)
    torch.testing.assert_close(z_fwd, z_rev, atol=1e-6, rtol=0)
