"""Training checks: tiny-batch overfit signal, checkpoint eval-logit
equality, optimizer/RNG/sampler resume equivalence."""

import numpy as np
import pytest
import torch

from qssm_dna.data import synthetic
from qssm_dna.data.tokenizer import pad_tokens
from qssm_dna.models.encoder import QSSMEncoder, QSSMModel
from qssm_dna.train.checkpoint import load_checkpoint
from qssm_dna.train.pretrain import SyntheticSource, train_classifier

pytestmark = pytest.mark.pipeline

CFG = {
    "training": {"micro_batch_size": 2, "gradient_accumulation_steps": 1,
                 "optimizer": "AdamW", "learning_rate": 1e-3,
                 "weight_decay": 1e-4, "gradient_clip_norm": 1.0,
                 "max_optimizer_steps": 40},
    "execution": {"seed": 11, "precision": "complex64"},
}


def _model(seed=0):
    torch.manual_seed(seed)
    return QSSMModel(QSSMEncoder(n_qubits=2), 4)


def test_tiny_overfit_loss_decreases(tmp_path):
    data = synthetic.generate("delayed_base_retrieval", 32, 8, seed=3)
    src = SyntheticSource(data["tokens"], data["labels"], seed=11)
    model = _model()
    res = train_classifier(model, src, CFG, "cpu", tmp_path / "log.jsonl",
                           tmp_path / "ck.pt")
    assert res["losses"][-1] < res["losses"][0]


def test_checkpoint_reload_logits_equal(tmp_path):
    data = synthetic.generate("delayed_base_retrieval", 16, 8, seed=3)
    src = SyntheticSource(data["tokens"], data["labels"], seed=11)
    model = _model()
    ck = tmp_path / "ck.pt"
    train_classifier(model, src, CFG, "cpu", tmp_path / "log.jsonl", ck)
    t, m = pad_tokens([data["tokens"][i] for i in range(4)])
    model.eval()
    with torch.no_grad():
        logits1 = model(torch.from_numpy(t), torch.from_numpy(m))
    model2 = _model(seed=999)  # different init
    load_checkpoint(ck, model2, map_location="cpu")
    model2.eval()
    with torch.no_grad():
        logits2 = model2(torch.from_numpy(t), torch.from_numpy(m))
    torch.testing.assert_close(logits1, logits2, atol=1e-6, rtol=0)


def test_resume_matches_uninterrupted(tmp_path):
    data = synthetic.generate("delayed_base_retrieval", 32, 8, seed=3)

    cfg10 = dict(CFG); cfg10["training"] = dict(CFG["training"])
    cfg10["training"]["max_optimizer_steps"] = 10
    cfg15 = dict(CFG); cfg15["training"] = dict(CFG["training"])
    cfg15["training"]["max_optimizer_steps"] = 15

    # uninterrupted 15 steps
    m1 = _model()
    s1 = SyntheticSource(data["tokens"], data["labels"], seed=11)
    train_classifier(m1, s1, cfg15, "cpu", tmp_path / "a.jsonl", None)

    # 10 steps, checkpoint, resume to 15
    m2 = _model()
    s2 = SyntheticSource(data["tokens"], data["labels"], seed=11)
    train_classifier(m2, s2, cfg10, "cpu", tmp_path / "b.jsonl",
                     tmp_path / "ck.pt")
    train_classifier(m2, s2, cfg15, "cpu", tmp_path / "b2.jsonl",
                     tmp_path / "ck2.pt", resume_from=tmp_path / "ck.pt")

    for (k, v), (_, v2) in zip(m1.state_dict().items(),
                               m2.state_dict().items()):
        torch.testing.assert_close(v, v2, atol=1e-6, rtol=0,
                                   msg=lambda msg, k=k: f"{k}: {msg}")
