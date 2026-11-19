"""Config loading. Every run records the resolved config and the SHA-256 of
the YAML (or JSON) file it was loaded from."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    raw = path.read_bytes()
    cfg = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(cfg, dict):
        raise ConfigError(f"config {path} did not parse to a mapping")
    cfg["_config_path"] = str(path)
    cfg["_config_sha256"] = sha256_bytes(raw)
    return cfg


def resolve_device(cfg: dict[str, Any]) -> str:
    """execution.device; 'auto' -> cuda if available, else cpu."""
    import torch

    dev = cfg.get("execution", {}).get("device", "auto")
    if dev == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda" and not torch.cuda.is_available():
        raise ConfigError("config requests device=cuda but CUDA is not available")
    return dev


def config_summary_json(cfg: dict[str, Any]) -> str:
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return json.dumps(clean, indent=2, sort_keys=True, default=str)
