"""Checkpoint save/load with full resume state: model, optimizer, and
every RNG the trainer touched (torch, torch.cuda, python, named numpy
Generators, sampler position)."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def capture_rng(generators: dict[str, np.random.Generator] | None = None
                ) -> dict:
    state = {
        "torch": torch.get_rng_state(),
        "torch_cuda": (torch.cuda.get_rng_state_all()
                       if torch.cuda.is_available() else None),
        "python": random.getstate(),
        "numpy_named": {},
    }
    for k, g in (generators or {}).items():
        state["numpy_named"][k] = g.bit_generator.state
    return state


def restore_rng(state: dict,
                generators: dict[str, np.random.Generator] | None = None
                ) -> None:
    torch.set_rng_state(state["torch"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    random.setstate(state["python"])
    for k, s in state.get("numpy_named", {}).items():
        if generators and k in generators:
            generators[k].bit_generator.state = s


def save_checkpoint(path: str | Path, model, optimizer=None, *,
                    step: int = 0, tokens_seen: int = 0,
                    targets_seen: int = 0, rng: dict | None = None,
                    source_state: dict | None = None,
                    extra: dict | None = None) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "tokens_seen": tokens_seen,
        "targets_seen": targets_seen,
        "rng": rng,
        "source_state": source_state,
        "extra": extra or {},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model, optimizer=None,
                    map_location="cpu") -> dict:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return {k: v for k, v in payload.items() if k != "model"}
