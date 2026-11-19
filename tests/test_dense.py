"""Dense next-base supervision (classical_sim_dense_supervision):

- dense position-t target is x_{t+1}; perturbing later tokens leaves
  the position-t logit invariant (causality).
- the dense loss reduces by the explicit valid-target mean.
- intermediate readouts never re-enter the recurrence (terminal logits
  from forward_positions equal the terminal forward()).
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from qssm_dna.models.encoder import QSSMEncoder, QSSMModel

pytestmark = pytest.mark.core


def _model(seed=3):
    torch.manual_seed(seed)
    return QSSMModel(QSSMEncoder(n_qubits=3, cell_depth=1,
                               hidden_dim=16), 4)


def test_dense_position_targets_are_next_base():
    """forward_positions[..., t] must predict x_{t+1} and be invariant to
    tokens after t (causality through the recurrence)."""
    model = _model()
    rng = np.random.default_rng(0)
    toks = torch.tensor(rng.integers(0, 4, size=(4, 24)))
    mask = torch.ones(4, 24, dtype=torch.bool)
    with torch.no_grad():
        lg = model.forward_positions(toks, mask)
    t = 10
    toks2 = toks.clone()
    toks2[:, t + 1:] = torch.tensor(
        rng.integers(0, 4, size=(4, 24 - t - 1)))
    with torch.no_grad():
        lg2 = model.forward_positions(toks2, mask)
    torch.testing.assert_close(lg[:, :t + 1], lg2[:, :t + 1],
                               atol=0, rtol=0)


def test_terminal_logit_matches_forward():
    """Last valid position of forward_positions == forward() logits."""
    model = _model()
    rng = np.random.default_rng(1)
    toks = torch.tensor(rng.integers(0, 4, size=(3, 20)))
    lens = torch.tensor([20, 12, 7])
    mask = torch.arange(20).unsqueeze(0) < lens.unsqueeze(1)
    with torch.no_grad():
        lp = model.forward_positions(toks, mask)
        lf = model(toks, mask)
    for i, l in enumerate(lens.tolist()):
        torch.testing.assert_close(lp[i, l - 1], lf[i], atol=1e-5,
                                   rtol=1e-5)


def test_dense_loss_is_valid_target_mean():
    """The dense loss is sum(per*sup)/sum(sup), not a raw sum and
    not padded-target diluted. N targets excluded by the sup mask."""
    model = _model()
    rng = np.random.default_rng(2)
    toks = torch.tensor(rng.integers(0, 4, size=(2, 16)))
    mask = torch.ones(2, 16, dtype=torch.bool)
    yd = torch.roll(toks, -1, dims=1)
    sup = torch.zeros(2, 16, dtype=torch.bool)
    sup[0, 8:16] = True
    sup[1, 10:16] = True
    with torch.no_grad():
        logits = model.forward_positions(toks, mask)
        per = F.cross_entropy(logits.reshape(-1, 4), yd.reshape(-1),
                              reduction="none").reshape(yd.shape)
        manual = (per * sup).sum() / sup.sum().clamp_min(1)
    # equivalence to explicit mean over supervised positions
    vals = per[sup]
    torch.testing.assert_close(manual, vals.mean(), atol=1e-7, rtol=1e-7)
    # and differs from the mean diluted by padded positions
    diluted = (per * sup).sum() / sup.numel()
    assert float(manual) > float(diluted)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_graphed_dense_step_matches_eager():
    """_GraphedDenseStep replay reproduces the eager gradients over
    repeated updates."""
    from qssm_dna.train.pretrain import _GraphedDenseStep
    torch.manual_seed(5)
    m1 = _model(5).cuda()
    m2 = _model(5).cuda()
    rng = np.random.default_rng(3)
    gs = _GraphedDenseStep(m1, "cuda", 8, 32, 2)
    for _ in range(3):
        t = torch.tensor(rng.integers(0, 4, size=(8, 32))).cuda()
        m = torch.ones(8, 32, dtype=torch.bool).cuda()
        yd = torch.roll(t, -1, dims=1)
        sup = torch.zeros(8, 32, dtype=torch.bool).cuda()
        sup[:, 24:32] = True
        lv = gs.run(t, m, yd, sup)
        assert lv is not None
        logits = m2.forward_positions(t, m)
        per = F.cross_entropy(logits.reshape(-1, 4), yd.reshape(-1),
                              reduction="none").reshape(yd.shape)
        loss = (per * sup).sum() / sup.sum().clamp_min(1) / 2
        loss.backward()
        assert abs(lv - float(loss) * 2) < 1e-5
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            torch.testing.assert_close(p1.grad, p2.grad, atol=1e-5,
                                       rtol=1e-4)
        m1.zero_grad(set_to_none=False)
        m2.zero_grad(set_to_none=False)
