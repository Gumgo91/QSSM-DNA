"""Run directories: resolved config, config hash, environment, git commit,
metrics, resources and status of every run."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from ..config import config_summary_json
from ..environment import collect_environment


def make_run_dir(root: str | Path, name: str) -> Path:
    """<root>/<timestamp>_<name>, with a numeric suffix if it exists."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    run = Path(root) / f"{ts}_{name}"
    i = 1
    while run.exists():
        run = Path(root) / f"{ts}_{name}_{i}"
        i += 1
    run.mkdir(parents=True)
    return run


def write_base_artifacts(run_dir: Path, cfg: dict) -> None:
    (run_dir / "config_resolved.yaml").write_text(config_summary_json(cfg))
    (run_dir / "config_sha256.txt").write_text(cfg["_config_sha256"])
    (run_dir / "environment.json").write_text(
        json.dumps(collect_environment(), indent=2, default=str))
    if cfg.get("_config_path"):
        try:
            (run_dir / "git_commit.txt").write_text(
                subprocess.run(
                    ["git", "rev-parse", "HEAD"], capture_output=True,
                    text=True).stdout.strip())
        except Exception:
            pass


def write_status(run_dir: Path, status: str, detail: dict) -> None:
    (run_dir / "status.json").write_text(json.dumps(
        {"status": status, **detail}, indent=2, default=str))


def write_metrics(run_dir: Path, metrics: dict) -> None:
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str))


def write_resources(run_dir: Path, resources: list[dict]) -> None:
    (run_dir / "resources.json").write_text(
        json.dumps(resources, indent=2, default=str))
