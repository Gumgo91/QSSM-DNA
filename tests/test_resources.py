"""Resource ledger arithmetic: N_run = R*G*S, token cells = N_run*L,
prefix-restart reference count."""

import pytest

from qssm_dna.eval.resources import (resource_record,
                                     restarting_reference_cells)

pytestmark = pytest.mark.core

CFG = {
    "model": {"n_qubits": 4, "cell_depth": 1,
              "readout": {"measurement_groups": 2}},
    "execution": {"precision": "complex64"},
}


def test_run_count_identity():
    rec = resource_record(CFG, length=1024, batch=1, shots=1024, rc=True)
    assert rec["circuit_repetitions"] == 2 * 2 * 1024
    assert rec["token_cell_executions"] == 2 * 2 * 1024 * 1024


def test_exact_mode_single_run():
    rec = resource_record(CFG, length=64, batch=1, shots=None, rc=False)
    assert rec["circuit_repetitions"] == 1 * 2 * 1
    assert rec["token_cell_executions"] == 2 * 64


def test_restarting_reference():
    r = restarting_reference_cells(8)
    assert r["prefix_restart_token_cells"] == 36
    assert r["terminal_readout_token_cells"] == 8


def test_no_fabricated_runtime():
    rec = resource_record(CFG, 8, 1, shots=8, rc=False,
                          execution_mode="quantum_resource_estimate")
    assert rec["runtime_estimate_s"] is None
    assert rec["runtime_estimate_reason"]
