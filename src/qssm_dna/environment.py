"""Software and hardware environment recorded with every run."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

PACKAGES = ["torch", "numpy", "scipy", "pandas", "pyarrow",
            "scikit-learn", "PyYAML", "pennylane", "pytest",
            "genomic-benchmarks"]


def _pkg_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_commit(repo: str | Path = ".") -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo,
            capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def collect_environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {},
        "git_commit": git_commit(),
    }
    for p in PACKAGES:
        env["packages"][p] = _pkg_version(p)
    try:
        import torch
        env["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["torch_cuda_device"] = torch.cuda.get_device_name(0)
            env["torch_cuda_vram_bytes"] = torch.cuda.get_device_properties(0).total_memory
            env["torch_cudnn"] = torch.backends.cudnn.version()
    except Exception as e:  # pragma: no cover
        env["torch_error"] = repr(e)
    return env
