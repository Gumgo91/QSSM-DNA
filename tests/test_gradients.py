"""Finite-difference gradient check: q=2, L=3, real-valued CE loss,
non-symmetric init. Tolerance 1e-5 (abs or rel)."""

import numpy as np
import pytest
import torch

from qssm_dna.models.encoder import QSSMEncoder, QSSMModel

pytestmark = pytest.mark.core


def _loss(model, tokens, mask, y):
    return torch.nn.functional.cross_entropy(
        model.forward_pretrain(tokens, mask), y)


def test_finite_difference_gradients():
    torch.manual_seed(123)
    q = 2
    enc = QSSMEncoder(n_qubits=q, cell_depth=1, dtype=torch.complex128)
    model = QSSMModel(enc, 4).double()
    enc.cdtype = torch.complex128
    tokens = torch.tensor([[0, 1, 2], [3, 0, 4]])
    mask = torch.ones(2, 3, dtype=torch.bool)
    y = torch.tensor([1, 2])
    loss = _loss(model, tokens, mask, y)
    loss.backward()

    # choose params with nonzero grad (guard against symmetric zeros)
    checked = 0
    for name, p in model.named_parameters():
        g = p.grad
        idxs = np.argwhere(np.ones_like(g.detach().numpy()))[:4]
        for idx in idxs:
            idx = tuple(idx.tolist())
            gv = g[idx].item()
            orig = p.detach()[idx].item()
            for eps in (1e-6,):
                p.data[idx] = orig + eps
                lp = _loss(model, tokens, mask, y).item()
                p.data[idx] = orig - eps
                lm = _loss(model, tokens, mask, y).item()
                p.data[idx] = orig
                fd = (lp - lm) / (2 * eps)
                err = abs(fd - gv)
                rel = err / max(abs(fd), abs(gv), 1e-12)
                if abs(gv) < 1e-12 and abs(fd) < 1e-12:
                    continue  # both zero: not informative
                assert err < 1e-5 or rel < 1e-5, (
                    f"{name}{idx}: ad={gv:.8g} fd={fd:.8g} "
                    f"err={err:.3g} rel={rel:.3g}")
                checked += 1
    assert checked > 0, "no informative gradient was checked"
