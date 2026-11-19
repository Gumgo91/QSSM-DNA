"""Training variants and ablations.

- supervision placement: tail (main), spread, all
- controller gamma modes: input (main), static, zero
- rzz_mode=off keeps the recurrence a product of local channels
- every variant draws the same initial parameters for a given seed
"""

import numpy as np
import pytest
import torch

from qssm_dna.models.encoder import QSSMEncoder
from qssm_dna.train.pretrain import WindowSource

pytestmark = pytest.mark.pipeline


class _FixedSampler:
    """Deterministic stand-in for fasta.WindowSampler."""

    def __init__(self, lengths):
        self.lengths = list(lengths)
        self.i = 0
        self.rng = np.random.default_rng(0)
        self.n_sampled = 0
        self.n_skipped_nfrac = 0

    def sample(self):
        l = self.lengths[self.i % len(self.lengths)]
        self.i += 1
        self.n_sampled += 1
        inp = self.rng.integers(0, 4, size=l).astype(np.int64)
        return inp, int(self.rng.integers(0, 4)), "chrT", 0, l


def _sup_rows(placement, k, lengths):
    src = WindowSource(_FixedSampler(lengths), 0, dense_tail=k,
                       placement=placement)
    t, m, y, yd, sup = src.next_batch(len(lengths))
    return [np.flatnonzero(sup[i].numpy()).tolist()
            for i in range(len(lengths))]


def test_tail_placement_is_last_k():
    rows = _sup_rows("tail", 8, [64, 128])
    assert rows[0] == list(range(56, 64))
    assert rows[1] == list(range(120, 128))


def test_spread_placement_even_and_ends_at_last():
    rows = _sup_rows("spread", 8, [64, 128])
    assert rows[0] == [7, 15, 23, 31, 39, 47, 55, 63]
    assert rows[1] == [15, 31, 47, 63, 79, 95, 111, 127]


def test_all_placement_covers_every_position():
    rows = _sup_rows("all", 0, [64, 16])
    assert rows[0] == list(range(64))
    assert rows[1] == list(range(16))


def test_n_inside_window_is_masked_with_a_valid_fill():
    """A window may contain N (up to the N-fraction limit); its dense target
    must be masked and must not carry the out-of-range class index 4."""
    class _NSampler(_FixedSampler):
        def sample(self):
            inp, tgt, c, s, l = super().sample()
            inp[10] = 4            # N inside the input window
            return inp, tgt, c, s, l
    for placement, k in (("all", 0), ("tail", 64), ("spread", 8)):
        src = WindowSource(_NSampler([64]), 0, dense_tail=k,
                           placement=placement)
        t, m, y, yd, sup = src.next_batch(1)
        assert int(yd.max()) < 4
        assert not bool(sup[0, 9])  # position 9 predicts the N at 10
        loss = torch.nn.functional.cross_entropy(
            torch.zeros(64, 4), yd[0], reduction="none")
        assert torch.isfinite(loss).all()


def test_dense_targets_are_next_base_for_every_placement():
    for placement, k in (("tail", 4), ("spread", 4), ("all", 0)):
        src = WindowSource(_FixedSampler([20]), 0, dense_tail=k,
                           placement=placement)
        t, m, y, yd, sup = src.next_batch(1)
        # yd[t] = x_{t+1} inside the window, and the endpoint target last
        assert torch.equal(yd[0, :-1], t[0, 1:])
        assert int(yd[0, -1]) == int(y[0])


def _enc(seed, **kw):
    torch.manual_seed(seed)
    return QSSMEncoder(n_qubits=3, cell_depth=1, hidden_dim=16, **kw)


def test_variants_share_initial_parameters():
    ref = _enc(7)
    for kw in ({"gamma_mode": "static"}, {"gamma_mode": "zero"},
               {"rzz_mode": "off"}):
        other = _enc(7, **kw)
        for (n1, p1), (n2, p2) in zip(ref.named_parameters(),
                                      other.named_parameters()):
            assert n1 == n2
            if n1 == "rzz_angles" and kw.get("rzz_mode") == "off":
                assert torch.count_nonzero(p2) == 0
                assert not p2.requires_grad
            else:
                assert torch.equal(p1, p2), n1


def test_default_encoder_unchanged_by_new_arguments():
    a = _enc(3)
    b = _enc(3, gamma_mode="input", rzz_mode="learned")
    toks = torch.tensor(np.random.default_rng(0).integers(0, 4, (2, 12)))
    mask = torch.ones(2, 12, dtype=torch.bool)
    with torch.no_grad():
        torch.testing.assert_close(a.forward_state(toks, mask),
                                   b.forward_state(toks, mask),
                                   atol=0, rtol=0)


def test_gamma_modes():
    toks = torch.tensor([[0, 1, 2, 3, 4]])
    _, g_in = _enc(1).controls(toks)
    _, g_st = _enc(1, gamma_mode="static").controls(toks)
    _, g_ze = _enc(1, gamma_mode="zero").controls(toks)
    assert not torch.allclose(g_in[0, 0], g_in[0, 1])
    assert torch.allclose(g_st[0], g_st[0, :1].expand_as(g_st[0]))
    assert torch.count_nonzero(g_ze) == 0


def _partial_trace_keep(rho, keep, q):
    """Reduced density matrix of one wire from a [D,D] matrix."""
    r = rho.reshape([2] * (2 * q))
    for w in reversed(range(q)):
        if w == keep:
            continue
        r = r.diagonal(dim1=w, dim2=w + r.dim() // 2).sum(-1)
    return r


def test_rzz_off_keeps_product_state():
    q = 3
    enc = _enc(5, rzz_mode="off").double()
    enc.cdtype = torch.complex128
    toks = torch.tensor(np.random.default_rng(4).integers(0, 4, (1, 30)))
    mask = torch.ones(1, 30, dtype=torch.bool)
    with torch.no_grad():
        rho = enc.forward_state(toks, mask)[0]
    marg = [_partial_trace_keep(rho, w, q) for w in range(q)]
    prod = marg[0]
    for m_ in marg[1:]:
        prod = torch.kron(prod, m_)
    torch.testing.assert_close(rho, prod, atol=1e-10, rtol=0)
    # with entangling gates the same check fails
    enc2 = _enc(5).double()
    enc2.cdtype = torch.complex128
    with torch.no_grad():
        rho2 = enc2.forward_state(toks, mask)[0]
    marg2 = [_partial_trace_keep(rho2, w, q) for w in range(q)]
    prod2 = torch.kron(torch.kron(marg2[0], marg2[1]), marg2[2])
    assert float((rho2 - prod2).abs().max()) > 1e-6
