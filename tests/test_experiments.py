"""Experiment helpers: readout indexing, distribution inversion,
shot sampling, canonical k-mers, CNN masking, run-condition labels and the
holdout window rule. CPU only."""

import numpy as np
import pytest
import torch

from qssm_dna.config import load_config
from qssm_dna.models.encoder import QSSMEncoder
from qssm_dna.quantum import readout as rd
from qssm_dna.experiments import encoders, evaluate, holdout, jobs, jobs_noise

pytestmark = pytest.mark.pipeline


def _rho(seed=0, b=6, l=30):
    torch.manual_seed(seed)
    enc = QSSMEncoder(n_qubits=4, cell_depth=1, hidden_dim=32)
    toks = torch.randint(0, 4, (b, l))
    with torch.no_grad():
        return enc.forward_state(toks, torch.ones(b, l, dtype=torch.bool))


def test_raw11_is_a_subset_of_subset30():
    rho = _rho()
    sub = rd.subset_readout(rho, 4).double().numpy()
    raw = rd.terminal_readout(rho, 4).double().numpy()
    np.testing.assert_allclose(sub[:, evaluate.raw11_index()], raw,
                               atol=1e-6)


def test_parities_determine_both_distributions():
    rho = _rho(1)
    sub = rd.subset_readout(rho, 4).double().numpy()
    pz, px = evaluate.probs_from_subset(sub)
    np.testing.assert_allclose(pz, rd.z_probabilities(rho).double().numpy(),
                               atol=1e-6)
    np.testing.assert_allclose(jobs_noise.parities(pz, px), sub, atol=1e-6)


def test_shot_features_converge_to_exact():
    sub = rd.subset_readout(_rho(2), 4).double().numpy()
    pz, px = evaluate.probs_from_subset(sub)
    d = {"zf": pz, "xf": px, "zr": pz, "xr": px}
    err = [np.abs(jobs_noise.feat_shots(d, s, 0) - sub).mean()
           for s in (100, 10000)]
    assert err[1] < err[0] / 5


def test_canonical_kmers_are_strand_invariant():
    s = "ACGTTGCAAGGCTTAACGNNACGTAGCTAGG"
    comp = str.maketrans("ACGTN", "TGCAN")
    r = s.translate(comp)[::-1]
    for k in (2, 3, 5):
        a = jobs.kmer_matrix([s, r], k)
        np.testing.assert_allclose(a[0], a[1])
    assert jobs.kmer_matrix(["ACGT"], 2).shape[1] == 10


def test_cnn_padding_does_not_change_output():
    torch.manual_seed(0)
    cnn = jobs.SmallCNN().eval()
    seqs = ["".join(np.random.default_rng(i).choice(list("ACGT"), n))
            for i, n in enumerate((251, 180))]
    x1, m1 = jobs._onehot(seqs, 251)
    x2, m2 = jobs._onehot(seqs, 400)
    with torch.no_grad():
        a = cnn(torch.from_numpy(x1), torch.from_numpy(m1))
        b = cnn(torch.from_numpy(x2), torch.from_numpy(m2))
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_rc_onehot_matches_reverse_complement():
    s = "AACGTNGGT"
    comp = str.maketrans("ACGTN", "TGCAN")
    x, m = jobs._onehot([s], 12)
    xr = jobs._rc_onehot(x, m)
    y, _ = jobs._onehot([s.translate(comp)[::-1]], 12)
    np.testing.assert_array_equal(xr, y)


@pytest.mark.parametrize("cfg_path,family,budget,expected", [
    ("configs/endpoint.yaml", "qssm", None, "qssm_endpoint"),
    ("configs/dense8.yaml", "qssm", None, "qssm_dense8"),
    ("configs/dense4.yaml", "qssm", None, "qssm_dense4"),
    ("configs/allpos.yaml", "qssm", 2500000,
     "qssm_allpos_b2500000"),
    ("configs/spread8.yaml", "qssm", None, "qssm_spread8"),
    ("configs/dense8_norzz.yaml", "qssm", None,
     "qssm_dense8_norzz"),
    ("configs/dense8_zerogamma.yaml", "qssm", None,
     "qssm_dense8_zerogamma"),
    ("configs/dense8_w256.yaml", "qssm", 2500000,
     "qssm_dense8_w256_b2500000"),
    ("configs/gru8_dense8.yaml", "gru", None, "gru8_dense8"),
    ("configs/endpoint.yaml", "gru", None, "gru64_endpoint"),
    ("configs/dense2.yaml", "qssm", None, "qssm_dense2"),
    ("configs/dense8_staticgamma.yaml", "qssm", None,
     "qssm_dense8_staticgamma"),
    ("configs/allpos_norzz.yaml", "qssm", None, "qssm_allpos_norzz"),
    ("configs/allpos_q6.yaml", "qssm", None, "qssm_allpos_q6"),
    ("configs/gru8_allpos.yaml", "gru", None, "gru8_allpos"),
    ("configs/allpos.yaml", "gru", None, "gru64_allpos"),
    ("configs/dense8.yaml", "gru", 2500000, "gru64_dense8_b2500000"),
])
def test_condition_labels(cfg_path, family, budget, expected):
    cfg = load_config(cfg_path)
    if budget:
        cfg["training"]["processed_input_token_budget"] = budget
    assert encoders.condition_of(cfg, family) == expected


def test_views_follow_register_size():
    torch.manual_seed(0)
    enc = QSSMEncoder(n_qubits=3, cell_depth=1, hidden_dim=16)
    toks = torch.randint(0, 4, (4, 20))
    with torch.no_grad():
        rho = enc.forward_state(toks, torch.ones(4, 20, dtype=torch.bool))
        sub = rd.subset_readout(rho, 3).float().numpy()
        raw = rd.terminal_readout(rho, 3).float().numpy()
    v = evaluate.qssm_views({"subset30_fwd": sub, "subset30_rc": sub}, enc)
    assert set(v) == {"raw8", "subset14", "projected64"}
    np.testing.assert_allclose(v["raw8"], raw, atol=1e-6)


def test_register_size_labels():
    cfg = load_config("configs/allpos_norzz_q6.yaml")
    assert encoders.condition_of(cfg, "qssm") == "qssm_allpos_norzz_q6"
    assert encoders.qubits_of_condition("qssm_random_init_q8") == 8
    assert encoders.qubits_of_condition("qssm_dense8_b2500000") == 4


def test_scaling_subset_is_fixed():
    from qssm_dna.experiments import datasets
    a = datasets.scaling_train_index(24388)
    b = datasets.scaling_train_index(24388)
    assert len(a) == 6000 and np.array_equal(a, b)
    assert np.all(np.diff(a) > 0)


def test_state_affine_form_reproduces_simulator():
    from qssm_dna.experiments import theory
    torch.manual_seed(4)
    enc = QSSMEncoder(n_qubits=3, cell_depth=1, hidden_dim=16)
    seq = np.random.default_rng(1).integers(0, 5, 40)
    R = theory.transfer_matrices(enc)          # converts enc to float64
    basis = theory.pauli_basis(3)
    r = basis[1:, 0, 0].real.numpy().copy()    # |000><000|
    for x in seq:
        a, c = theory.affine_parts(R[int(x)])
        r = a @ r + c
    with torch.no_grad():
        rho = enc.forward_state(torch.tensor(seq)[None],
                                torch.ones(1, 40, dtype=torch.bool))[0]
    direct = torch.einsum("pij,ji->p", basis, rho).real.numpy()[1:]
    np.testing.assert_allclose(r, direct, atol=1e-12)
    # trace preservation: identity row of every transfer matrix
    for x in range(5):
        np.testing.assert_allclose(R[x][0], np.eye(len(R[x]))[0],
                                   atol=1e-12)


def test_sign_flip_permutation_and_holm():
    from qssm_dna.experiments import stats
    assert stats.sign_flip_p(np.ones(10)) == pytest.approx(2 / 1024)
    assert stats.sign_flip_p(np.array([1.0, -1.0])) == 1.0
    adj = stats.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})


def test_collect_reads_finished_experiment_runs(tmp_path):
    import json
    import pandas as pd
    from qssm_dna.experiments import evaluate, stats
    row = {"condition": "qssm_dense8", "seed": 22, "task": "gb_promoter",
           "eval_set": "test", "readout": "raw11", "eval_mcc": 0.5}
    for name, status in (("20260101_000000_eval_encoder_x_s22",
                          {"status": "DONE", "part": "eval-encoder"}),
                         ("20260101_000001_eval_encoder_x_s22",
                          {"status": "DONE", "part": "eval-encoder"}),
                         ("20260101_000002_failed", {"status": "FAILED",
                                                     "part": "cnn"}),
                         ("20260101_000003_other", {"status": "DONE"})):
        d = tmp_path / "runs" / name
        d.mkdir(parents=True)
        (d / "status.json").write_text(json.dumps(status))
        pd.DataFrame([row]).to_csv(d / "metrics.csv", index=False)
    evaluate.save_predictions(
        tmp_path / "runs" / "20260101_000000_eval_encoder_x_s22"
        / "predictions.npz", {}, [{"readout": "raw11", "eval_set": "test",
                                   "prob": np.ones(3, np.float32),
                                   "label": np.ones(3, np.int8),
                                   "threshold": 0.5}])
    df = stats.collect([tmp_path])
    assert list(df["run"]) == ["20260101_000000_eval_encoder_x_s22"]
    assert df["eval_name"].iloc[0] == "gb_promoter:test"
    p = stats.load_predictions(df["run"].iloc[0], roots=[tmp_path])
    assert set(p) == {"test__raw11__prob", "test__raw11__label",
                      "test__raw11__threshold"}


def test_holdout_window_puts_tss_at_base_201():
    # toy chromosome; TSS (1-based) at 1001 on both strands
    rng = np.random.default_rng(0)
    chrom = "".join(rng.choice(list("ACGT"), 3000))
    p = 1001
    s, e = holdout._window(p, "+")
    assert e - s == 251 and chrom[s:e][200] == chrom[p - 1]
    s, e = holdout._window(p, "-")
    seq = holdout.rc(chrom[s:e])
    assert e - s == 251
    assert seq[200] == holdout.rc(chrom[p - 1])
