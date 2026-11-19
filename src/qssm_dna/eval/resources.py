"""Resource records: circuit repetitions, token-cell executions and gate
counts per prediction.

execution_mode is one of
  classical_measured        -- wall clock of the classical simulation
  quantum_resource_estimate -- counts under the stated circuit assumptions
  qpu_measured              -- measured on quantum hardware
Without a hardware target the runtime estimate is left null.
"""

from __future__ import annotations


def token_cell_gate_counts(q: int, cell_depth: int) -> dict:
    """Logical gate/channel counts per token cell (compile-independent)."""
    return {
        "amplitude_damping_channels": q,          # each = ancilla + U + reset
        "ry_gates": q * cell_depth,
        "rz_gates": q * cell_depth,
        "rzz_gates": (q - 1) * cell_depth,
        "ancilla_resets": q,                       # serial-reset width q+1
        "basis_rotations": 0,                      # readout-time only
    }


def resource_record(cfg: dict, length: int, batch: int, *,
                    shots: int | None, rc: bool,
                    execution_mode: str = "classical_measured",
                    measured_or_estimated: str = "measured",
                    wall_time_s: float | None = None,
                    backend: str = "torch_density_contraction",
                    hardware: str | None = None,
                    extra_assumptions: list[str] | None = None) -> dict:
    m = cfg["model"]
    q = m["n_qubits"]
    depth = m["cell_depth"]
    g = m["readout"]["measurement_groups"]
    r = 2 if rc else 1
    s = shots if shots is not None else 1
    n_run = r * g * s
    per_cell = token_cell_gate_counts(q, depth)
    gates = {
        **per_cell,
        "readout_basis_rotations": q,              # H on each wire (X group)
        "measurements_per_run": g,
    }
    return {
        "execution_mode": execution_mode,
        "measured_or_estimated": measured_or_estimated,
        "backend": backend,
        "hardware": hardware,
        "dtype": str(cfg["execution"].get("precision")),
        "batch": batch,
        "length": length,
        "q": q,
        "ancilla": "serial_reset_q_plus_1",
        "cell_depth": depth,
        "G": g, "R": r, "S": s,
        "circuit_repetitions": n_run,
        "token_cell_executions": n_run * length,
        "gate_counts_per_token_cell": gates,
        "reset_counts_per_token_cell": per_cell["ancilla_resets"],
        "compiled_depth": None,
        "compiled_depth_reason": "no hardware compiler target specified",
        "cache_policy": "finite_token_controller_lookup_allowed",
        "wall_time_s": wall_time_s,
        "input_io_policy": "controller_precomputed_per_token",
        "assumptions": extra_assumptions or [
            "wall_time_s is the classical simulation time, not a QPU estimate",
            "each shot/basis/direction is a full L-step re-preparation",
        ],
        "runtime_estimate_s": None,
        "runtime_estimate_reason": ("no target hardware durations provided"
                                    if execution_mode !=
                                    "classical_measured" else None),
    }


def restarting_reference_cells(length: int) -> dict:
    """Token cells of a terminal readout (L) and of reading every prefix by
    restarting from scratch (L(L+1)/2). The two protocols produce different
    outputs (1 vs L), so this is a cost comparison only."""
    return {
        "terminal_readout_token_cells": length,
        "prefix_restart_token_cells": length * (length + 1) // 2,
        "note": "output sets differ (1 vs L outputs); not a speedup claim",
    }
