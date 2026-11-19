"""Locate pretrained runs by condition and seed, and rebuild encoders.

A run's condition label is derived from its resolved config, not from its
directory name: qssm_endpoint, qssm_dense<k>, qssm_allpos, qssm_spread<k>,
gru<h>_dense<k>, ..., with the suffixes _norzz, _staticgamma, _zerogamma,
_w256 (256-base windows), _b<budget> for budgets other than 250,000 input
tokens and _q<n> for registers other than four qubits. When a (condition,
seed) pair has several finished runs, the earliest is used.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

BASE_BUDGET = 250000


def condition_of(cfg: dict, family: str) -> str:
    t = cfg["training"]
    dense = t.get("objective") == "classical_sim_dense_supervision"
    placement = t.get("supervision_placement", "tail")
    k = int(t.get("dense_tail", 0))
    if family == "qssm":
        if not dense:
            name = "qssm_endpoint"
        elif placement == "all":
            name = "qssm_allpos"
        elif placement == "spread":
            name = f"qssm_spread{k}"
        else:
            name = f"qssm_dense{k}"
        if cfg["model"].get("unitary", {}).get("rzz_mode", "learned") == "off":
            name += "_norzz"
        gm = cfg["model"].get("forgetting", {}).get("gamma_mode", "input")
        if gm != "input":
            name += f"_{gm}gamma"
    else:
        h = int(cfg.get("baselines", {}).get("gru_hidden", 64))
        sup = f"dense{k}" if dense else "endpoint"
        if dense and placement == "all":
            sup = "allpos"
        name = f"{family}{h}_{sup}"
    if list(t.get("sequence_lengths", [])) == [256]:
        name += "_w256"
    budget = int(t.get("processed_input_token_budget", BASE_BUDGET))
    if budget != BASE_BUDGET:
        name += f"_b{budget}"
    q = int(cfg["model"].get("n_qubits", 4))
    if family == "qssm" and q != 4:
        name += f"_q{q}"
    return name


def qubits_of_condition(condition: str) -> int:
    """Register size encoded in a condition label (suffix _q<n>; default 4)."""
    tail = condition.rsplit("_q", 1)
    return int(tail[1]) if len(tail) == 2 and tail[1].isdigit() else 4


@dataclass
class RunRef:
    condition: str
    seed: int
    family: str
    run_dir: Path | None
    cfg: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)


def scan_runs(roots=("runs",)) -> list[RunRef]:
    refs = []
    for root in roots:
        for d in sorted(Path(root).glob("*_pretrain_*")):
            fam = d.name.rsplit("_pretrain_", 1)[1].split("_")[0]
            st, cy = d / "status.json", d / "config_resolved.yaml"
            if not (st.exists() and cy.exists()
                    and (d / "checkpoint.pt").exists()):
                continue
            if (d / "EXCLUDED.json").exists():   # run marked as excluded
                continue
            if json.loads(st.read_text()).get("status") != "DONE":
                continue
            cfg = json.loads(cy.read_text())
            met = json.loads((d / "metrics.json").read_text()) \
                if (d / "metrics.json").exists() else {}
            refs.append(RunRef(condition_of(cfg, fam),
                               int(cfg["execution"]["seed"]), fam, d, cfg,
                               {k: v for k, v in met.items()
                                if k != "losses"}))
    return refs


def pick_run(refs: list[RunRef], condition: str, seed: int) -> RunRef:
    hits = [r for r in refs if r.condition == condition and r.seed == seed]
    if not hits:
        raise LookupError(f"no DONE run for {condition} seed {seed}")
    return sorted(hits, key=lambda r: r.run_dir.name)[0]


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def state_sha256(module: torch.nn.Module) -> str:
    buf = io.BytesIO()
    torch.save(module.state_dict(), buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def load_encoder(ref: RunRef, device: str):
    """Encoder with the run's architecture flags, weights from checkpoint."""
    from ..models.gru import GRUEncoder
    from ..models.encoder import QSSMEncoder, encoder_kwargs
    if ref.family == "qssm":
        enc = QSSMEncoder(**encoder_kwargs(ref.cfg))
    elif ref.family == "gru":
        enc = GRUEncoder(hidden=int(ref.cfg.get("baselines", {})
                                    .get("gru_hidden", 64)))
    else:
        raise ValueError(ref.family)
    payload = torch.load(ref.run_dir / "checkpoint.pt", map_location="cpu",
                         weights_only=False)
    enc.load_state_dict({k[len("encoder."):]: v
                         for k, v in payload["model"].items()
                         if k.startswith("encoder.")})
    return enc.to(device).eval()


def value_sha256(module: torch.nn.Module) -> str:
    """Platform-independent hash of parameter values (names, dtypes,
    shapes, raw bytes in sorted key order)."""
    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        t = v.detach().cpu().contiguous()
        h.update(k.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def _platform(run_dir: Path) -> str | None:
    f = run_dir / "environment.json"
    if not f.exists():
        return None
    return json.loads(f.read_text()).get("platform", "").split("-")[0]


def matched_random_init(base_cfg: dict, seed: int, device: str,
                        refs: list[RunRef]) -> tuple[torch.nn.Module, dict]:
    """Untrained encoder identical to the initialisation of the seed's
    QSSM runs. Rebuilds the full pretraining model as cli.cmd_pretrain does
    and checks its torch.save hash against every QSSM run of that seed with
    the main architecture made on the same OS (torch.save bytes differ
    between Windows and Linux for identical tensors, so other runs are
    listed as not comparable; value_sha256 allows cross-platform checks)."""
    import platform
    from ..models.encoder import QSSMEncoder, QSSMModel, encoder_kwargs
    torch.manual_seed(seed)
    enc = QSSMEncoder(**encoder_kwargs(base_cfg))
    model = QSSMModel(enc, 4)
    sha = state_sha256(model)
    here = platform.system()
    q = int(base_cfg["model"]["n_qubits"])
    checked = []
    for r in refs:
        if r.family != "qssm" or r.seed != seed:
            continue
        if int(r.cfg["model"].get("n_qubits", 4)) != q:
            continue
        if any(a in r.condition for a in ("_norzz", "_staticgamma",
                                          "_zerogamma")):
            continue
        f = r.run_dir / "init_state_sha256.txt"
        if not f.exists():
            continue
        if _platform(r.run_dir) != here:
            checked.append({"run_dir": str(r.run_dir),
                            "match": "not_comparable_other_os"})
            continue
        ok = f.read_text().strip() == sha
        checked.append({"run_dir": str(r.run_dir), "match": ok})
        if not ok:
            raise RuntimeError(
                f"random-init rebuild for seed {seed} does not match "
                f"{f}: {sha} != {f.read_text().strip()}")
    return enc.to(device).eval(), {"init_state_sha256": sha,
                                   "init_value_sha256": value_sha256(model),
                                   "platform": here,
                                   "verified_against": checked}
