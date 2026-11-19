"""Dispatcher of `python -m qssm_dna experiment --part <name>`.

Each part writes a run directory <timestamp>_<part>[_<condition>][_s<seed>]
with the config, environment and status files of the other commands; the
status records the part name, and an exception is stored in result.json
with status FAILED."""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

from ..reporting import manifest as mf

EXPERIMENTS_CFG = "configs/experiments.json"


def _experiments_cfg() -> dict:
    from ..config import load_config  # the YAML loader also reads JSON
    return load_config(EXPERIMENTS_CFG)


def part_holdout(args, run_dir: Path) -> dict:
    from .holdout import build
    return build()


def part_fetch_tasks(args, run_dir: Path) -> dict:
    from .datasets import fetch_gb_tasks
    return fetch_gb_tasks()


PARTS = {"holdout": part_holdout, "fetch-tasks": part_fetch_tasks}


def register(name: str):
    def deco(fn):
        PARTS[name] = fn
        return fn
    return deco


def run_part(name: str, args) -> int:
    from . import jobs, jobs_noise  # noqa: F401  (register parts)
    if name not in PARTS:
        print(f"unknown part {name!r}; choose from {sorted(PARTS)}")
        return 2
    cfg = _experiments_cfg()
    tag = name.replace("-", "_")
    if getattr(args, "condition", None):
        tag += f"_{args.condition}"
    if getattr(args, "seed", None) is not None:
        tag += f"_s{args.seed}"
    run_dir = mf.make_run_dir(getattr(args, "out", "runs"), tag)
    mf.write_base_artifacts(run_dir, cfg)
    t0 = time.time()
    try:
        out = PARTS[name](args, run_dir)
        status = "DONE"
    except Exception as e:
        out = {"error": f"{type(e).__name__}: {e}",
               "trace": traceback.format_exc()}
        status = "FAILED"
    (run_dir / "result.json").write_text(json.dumps(out, indent=2,
                                                    default=str))
    mf.write_status(run_dir, status, {"part": name,
                                      "wall_time_s": time.time() - t0})
    print(f"[{name}] {status} -> {run_dir} "
          f"({time.time() - t0:.1f}s)")
    if status == "FAILED":
        print(out["trace"][-3000:])
    return 0 if status == "DONE" else 1
