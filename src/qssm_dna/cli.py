"""Command line interface.

    python -m qssm_dna prepare-data --config configs/endpoint.yaml
    python -m qssm_dna pretrain --config configs/dense8.yaml --model qssm --seed 22
    python -m qssm_dna experiment --part eval-encoder --condition qssm_dense8 --seed 22
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
import torch

from .config import load_config, resolve_device
from .reporting import manifest


# ------------------------------------------------------------- prepare-data

def cmd_prepare_data(args) -> int:
    """Download the hg38 chromosomes of the config and the promoter
    benchmark, and record their SHA-256 hashes."""
    from .data import benchmarks, fasta

    cfg = load_config(args.config)
    run_dir = manifest.make_run_dir(cfg.get("outputs", {}).get("root", "runs"), "prepare-data")
    manifest.write_base_artifacts(run_dir, cfg)
    man = {"genome": {}, "benchmark": {}}
    g = cfg["data"]["genome"]
    chroms = sorted(set(
        g.get("train_chromosomes", [])
        + g.get("validation_chromosomes", [])
        + g.get("test_chromosomes", [])))
    base = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes"
    data_dir = Path("data/fasta")
    for ch in chroms:
        url = f"{base}/{ch}.fa.gz"
        try:
            info = fasta.download(url, data_dir / f"{ch}.fa.gz")
            man["genome"][ch] = info
            print(f"{ch}: {info['bytes']} bytes sha256={info['sha256'][:16]}")
        except Exception as e:
            man["genome"][ch] = {"url": url, "error": repr(e)}
            print(f"{ch}: download failed: {e}")
    Path("data").mkdir(exist_ok=True)
    task = cfg["data"]["benchmark"]["task"]
    try:
        splits = benchmarks.load_task(
            task, version=int(cfg["data"]["benchmark"].get(
                "requested_version", 0)))
        bman = benchmarks.dataset_manifest(task, splits)
        bman["duplicate_check"] = benchmarks.find_cross_split_duplicates(splits)
        man["benchmark"] = bman
        np.savez_compressed(
            "data/benchmarks.npz",
            train_seq=np.array([s for s, _ in splits["train"]]),
            train_y=np.array([y for _, y in splits["train"]]),
            test_seq=np.array([s for s, _ in splits["test"]]),
            test_y=np.array([y for _, y in splits["test"]]))
        print(f"benchmark {task}: {bman['splits']}")
    except benchmarks.DatasetUnavailable as e:
        man["benchmark"] = {"status": "UNAVAILABLE", "reason": str(e)}
        print(f"benchmark {task} unavailable: {e}")
    (run_dir / "data_manifest.json").write_text(
        json.dumps(man, indent=2, default=str))
    # a copy next to the data
    Path("data").mkdir(exist_ok=True)
    (Path("data") / "data_manifest.json").write_text(
        json.dumps(man, indent=2, default=str))
    manifest.write_status(run_dir, "DONE", man.get("benchmark", {})
                          if isinstance(man.get("benchmark"), dict) else {})
    print(f"prepare-data -> {run_dir}")
    return 0


# ----------------------------------------------------------------- pretrain

def cmd_pretrain(args) -> int:
    """Next-base pretraining of a QSSM or GRU encoder on genome windows,
    followed by the validation NLL on the held-out chromosomes."""
    import torch.nn.functional as F

    from .data import fasta
    from .data.tokenizer import encode
    from .eval.resources import resource_record
    from .models.encoder import QSSMEncoder, QSSMModel, encoder_kwargs
    from .models.gru import GRUEncoder
    from .train.probe import EncoderClassifier
    from .train.pretrain import WindowSource, train_classifier

    cfg = load_config(args.config)
    device = resolve_device(cfg)
    if args.seed is not None:
        cfg["execution"]["seed"] = int(args.seed)
    if args.token_budget is not None:
        cfg["training"]["processed_input_token_budget"] = int(
            args.token_budget)
    which = args.model
    run_dir = manifest.make_run_dir(cfg.get("outputs", {}).get("root", "runs"),
                                    f"pretrain_{which}")
    manifest.write_base_artifacts(run_dir, cfg)
    g = cfg["data"]["genome"]
    chroms = g["train_chromosomes"]
    chrom_tokens = {}
    for ch in chroms + g["validation_chromosomes"]:
        p = Path(f"data/fasta/{ch}.fa.gz")
        if not p.exists():
            raise FileNotFoundError(f"{p} missing; run prepare-data first")
        seqs = fasta.read_fasta_sequences(p)
        chrom_tokens[ch] = encode(seqs[ch])
    rng = np.random.default_rng(cfg["execution"]["seed"])
    sampler = fasta.WindowSampler(
        chrom_tokens, cfg["training"]["sequence_lengths"],
        cfg["training"].get("length_probabilities"),
        g["max_input_n_fraction"], rng, train_chroms=chroms)
    src = WindowSource(sampler, cfg["execution"]["seed"],
                       rc_augment=cfg["training"].get(
                           "rc_augmentation") ==
                       "whole_example_before_target_split",
                       dense_tail=int(cfg["training"].get("dense_tail", 0)),
                       placement=cfg["training"].get(
                           "supervision_placement", "tail"))
    torch.manual_seed(cfg["execution"]["seed"])
    if which == "qssm":
        enc = QSSMEncoder(**encoder_kwargs(cfg))
        model = QSSMModel(enc, 4)
    else:
        model = EncoderClassifier(GRUEncoder(
            hidden=int(cfg.get("baselines", {}).get("gru_hidden", 64))), 4)
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    init_sha = hashlib.sha256(buf.getvalue()).hexdigest()
    (run_dir / "init_state_sha256.txt").write_text(init_sha)
    t0 = time.time()
    res = train_classifier(
        model, src, cfg, device, run_dir / "train_log.jsonl",
        run_dir / "checkpoint.pt",
        max_steps=10 ** 9,
        token_budget=int(cfg["training"]["processed_input_token_budget"]))
    res["wall_time_s"] = time.time() - t0
    res["init_state_sha256"] = init_sha
    res["objective"] = cfg["training"].get("objective",
                                           "terminal_next_base")
    res["dense_tail"] = int(cfg["training"].get("dense_tail", 0))
    # validation NLL on the held-out chromosomes
    val_sampler = fasta.WindowSampler(
        chrom_tokens, cfg["training"]["sequence_lengths"],
        cfg["training"].get("length_probabilities"),
        g["max_input_n_fraction"], np.random.default_rng(1234),
        train_chroms=g["validation_chromosomes"])
    val_src = WindowSource(val_sampler, 1234)
    model.eval()
    nlls = []
    with torch.no_grad():
        for _ in range(8):
            t, m, y = val_src.next_batch(16)
            t, m, y = t.to(device), m.to(device), y.to(device)
            nlls.append(float(F.cross_entropy(model(t, m), y)))
    res["val_nll"] = float(np.mean(nlls))
    res["val_bits_per_base"] = res["val_nll"] / np.log(2)
    res["uniform_baseline_bits"] = 2.0
    manifest.write_metrics(run_dir, res)
    manifest.write_resources(run_dir, [resource_record(
        cfg, max(cfg["training"]["sequence_lengths"]),
        cfg["training"]["micro_batch_size"], shots=None, rc=False,
        wall_time_s=res["wall_time_s"],
        hardware=torch.cuda.get_device_name(0) if device == "cuda" else "cpu")])
    manifest.write_status(run_dir, "DONE",
                          {"tokens_seen": res["tokens_seen"],
                           "val_nll": res["val_nll"]})
    print(f"pretrain done -> {run_dir} tokens={res['tokens_seen']} "
          f"val_nll={res['val_nll']:.4f}")
    return 0


# --------------------------------------------------------------- experiment

def cmd_experiment(args) -> int:
    from .experiments.parts import run_part
    return run_part(args.part, args)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="qssm_dna")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare-data",
                       help="download genome chromosomes and the benchmark")
    p.add_argument("--config", required=True)
    p = sub.add_parser("pretrain", help="next-base pretraining of an encoder")
    p.add_argument("--config", required=True)
    p.add_argument("--model", default="qssm", choices=["qssm", "gru"])
    p.add_argument("--seed", type=int, default=None,
                   help="override execution.seed")
    p.add_argument("--token-budget", type=int, default=None,
                   help="override training.processed_input_token_budget")
    p = sub.add_parser("experiment",
                       help="evaluation, baselines, noise and runtime parts")
    p.add_argument("--part", required=True,
                   help="holdout, fetch-tasks, eval-encoder, kmer-lr, cnn, "
                        "noise or runtime")
    p.add_argument("--condition", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--datasets", default=None,
                   help="comma-separated evaluation sets")
    p.add_argument("--out", default="runs")
    args = ap.parse_args(argv)
    return {
        "prepare-data": cmd_prepare_data,
        "pretrain": cmd_pretrain,
        "experiment": cmd_experiment,
    }[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
