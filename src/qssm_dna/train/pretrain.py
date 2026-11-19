"""Next-base pretraining: batch sources, CUDA-graph training steps and the
training loop.

A batch source provides next_batch() -> (tokens [B,L] int64, mask [B,L]
bool, targets [B] int64[, dense targets [B,L], supervision mask [B,L]]) and
state_dict()/load_state_dict() for exact resume.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..data.tokenizer import PAD_IDX, N_IDX, encode, pad_tokens, rc_array
from .checkpoint import (capture_rng, load_checkpoint, restore_rng,
                         save_checkpoint)


class SyntheticSource:
    """Fixed array iterated in per-epoch seeded permutations; the position
    is determined by (seed, epoch, pos), so resume is exact."""

    def __init__(self, tokens: np.ndarray, labels: np.ndarray, seed: int):
        self.tokens = tokens
        self.labels = labels
        self.seed = seed
        self.epoch = 0
        self.pos = 0
        self._order = self._perm(self.epoch)

    def _perm(self, epoch: int) -> np.ndarray:
        return np.random.default_rng((self.seed, epoch)).permutation(
            len(self.tokens))

    def next_batch(self, bs: int):
        n = len(self.tokens)
        toks, labs = [], []
        for _ in range(bs):
            if self.pos >= n:
                self.epoch += 1
                self.pos = 0
                self._order = self._perm(self.epoch)
            i = self._order[self.pos]
            self.pos += 1
            toks.append(self.tokens[i])
            labs.append(self.labels[i])
        t, m = pad_tokens(toks)
        return (torch.from_numpy(t), torch.from_numpy(m),
                torch.from_numpy(np.asarray(labs, dtype=np.int64)))

    def state_dict(self):
        return {"epoch": self.epoch, "pos": self.pos, "seed": self.seed}

    def load_state_dict(self, s):
        self.epoch, self.pos, self.seed = s["epoch"], s["pos"], s["seed"]
        self._order = self._perm(self.epoch)


class WindowSource:
    """Streams (input [L], next base) windows from chromosome token arrays.
    Holds its own numpy Generator, so resume is exact through its
    bit_generator state. RC augmentation transforms the whole L+1 window
    before the input/target split."""

    def __init__(self, sampler, seed: int, rc_augment: bool = False,
                 dense_tail: int = 0, placement: str = "tail"):
        self.sampler = sampler
        self.rng = np.random.default_rng(seed)
        self.rc_augment = rc_augment
        self.dense_tail = int(dense_tail)  # >0: emit per-position targets
        # which positions carry a next-base loss:
        #   tail   -- the last dense_tail valid positions
        #   spread -- dense_tail positions evenly spaced, ending at the last
        #   all    -- every valid position (full causal LM objective)
        if placement not in ("tail", "spread", "all"):
            raise ValueError(f"unknown supervision placement {placement!r}")
        if placement == "spread" and self.dense_tail < 1:
            raise ValueError("spread placement needs dense_tail >= 1")
        self.placement = placement
        self.dense = self.dense_tail > 0 or placement == "all"
        self.n_target_n_skipped = 0

    def supervised_positions(self, l_i: int) -> list[int]:
        if self.placement == "all":
            return list(range(l_i))
        if self.placement == "spread":
            k = min(self.dense_tail, l_i)
            return sorted({-(-(j + 1) * l_i // k) - 1 for j in range(k)})
        return list(range(max(l_i - self.dense_tail, 0), l_i))

    def next_batch(self, bs: int):
        toks, labs, denses = [], [], []
        need = bs
        while need > 0:
            inp, tgt, chrom, start, l = self.sampler.sample()
            if tgt == N_IDX:          # N targets are excluded from loss
                self.n_target_n_skipped += 1
                continue
            if self.rc_augment and self.rng.random() < 0.5:
                full = rc_array(np.concatenate([inp, [tgt]]))
                inp, tgt = full[:-1], int(full[-1])
            toks.append(inp)
            labs.append(tgt)
            if self.dense:
                # y_next[t] = base following position t (within the same
                # L+1 window): inp[t+1] for t<l-1, tgt for t=l-1
                denses.append(np.concatenate([inp[1:], [tgt]]))
            need -= 1
        t, m = pad_tokens(toks)
        t = torch.from_numpy(t)
        m = torch.from_numpy(m)
        y = torch.from_numpy(np.asarray(labs, dtype=np.int64))
        if not self.dense:
            return t, m, y
        b, lmax = t.shape
        # fill value must be a valid class index (<4); supervision mask
        # carries the semantics, not the target value
        yd = np.zeros((b, lmax), dtype=np.int64)
        sup = np.zeros((b, lmax), dtype=bool)
        for i, d in enumerate(denses):
            l_i = len(d)
            # N targets are never supervised (sup stays False below); they
            # still need a valid class index because the per-position loss
            # is computed for every position before masking
            yd[i, :l_i] = np.where(d == N_IDX, 0, d)
            # supervise the selected valid positions whose next-base
            # target is a base (not N); tail = last min(k, l_i)
            for pos in self.supervised_positions(l_i):
                if d[pos] != N_IDX:
                    sup[i, pos] = True
        return (t, m, y,
                torch.from_numpy(yd), torch.from_numpy(sup))

    def state_dict(self):
        return {"rng": self.rng.bit_generator.state,
                "sampler_stats": {
                    "n_sampled": self.sampler.n_sampled,
                    "n_skipped_nfrac": self.sampler.n_skipped_nfrac},
                "n_target_n_skipped": self.n_target_n_skipped}

    def load_state_dict(self, s):
        self.rng.bit_generator.state = s["rng"]
        self.n_target_n_skipped = s.get("n_target_n_skipped", 0)

    def generators(self):
        return {"window_sampler": self.sampler.rng, "rc_aug": self.rng}


class _GraphedDenseStep:
    """Whole-microbatch forward + backward CUDA graph for dense
    supervision.

    Static input buffers; the batch is padded to a fixed size with zero
    supervision weight, and PAD positions (mask False) leave the state
    unchanged, so fixed-length padding gives the same result.
    optimizer.step() stays outside the graph. Capture requires that no
    eager CUDA backward has run in the process before.
    """

    _side_stream: torch.cuda.Stream | None = None

    def __init__(self, model, device: str, batch_size: int, seq_len: int,
                 accum: int):
        self.model = model
        self.accum = accum
        self.bs = batch_size
        self.s_t = torch.full((batch_size, seq_len), PAD_IDX,
                              dtype=torch.long, device=device)
        self.s_m = torch.zeros(batch_size, seq_len, dtype=torch.bool,
                               device=device)
        self.s_yd = torch.zeros(batch_size, seq_len, dtype=torch.long,
                                device=device)
        self.s_sup = torch.zeros(batch_size, seq_len, dtype=torch.bool,
                                 device=device)
        self.graph = None
        self.s_loss = None
        self.failed = False

    def _loss(self):
        logits = self.model.forward_positions(self.s_t, self.s_m)
        per = F.cross_entropy(logits.reshape(-1, 4),
                              self.s_yd.reshape(-1),
                              reduction="none").reshape(self.s_yd.shape)
        sup_w = self.s_sup.to(per.dtype)
        return (per * sup_w).sum() / sup_w.sum().clamp_min(1.0) / \
            self.accum

    def _capture(self):
        if _GraphedDenseStep._side_stream is None:
            _GraphedDenseStep._side_stream = torch.cuda.Stream()
        s = _GraphedDenseStep._side_stream
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self.model.zero_grad(set_to_none=False)
                loss = self._loss()
                loss.backward()
        torch.cuda.current_stream().wait_stream(s)
        self.model.zero_grad(set_to_none=False)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self.s_loss = self._loss()
            self.s_loss.backward()
        self.graph = g

    def run(self, t: torch.Tensor, m: torch.Tensor, yd: torch.Tensor,
            sup: torch.Tensor) -> float | None:
        """Pads tokens/mask/targets to the captured shape and replays.
        Returns the unscaled loss, or None if capture failed. Gradients
        accumulate across replays (the caller zeroes them)."""
        b, l = t.shape
        self.s_t.fill_(PAD_IDX)
        self.s_t[:b, :l].copy_(t)
        self.s_m.zero_()
        self.s_m[:b, :l].copy_(m)
        self.s_yd.zero_()
        self.s_yd[:b, :l].copy_(yd)
        self.s_sup.zero_()
        self.s_sup[:b, :l].copy_(sup)
        if self.graph is None:
            if self.failed:
                return None
            try:
                self._capture()
            except Exception:
                self.failed = True
                torch.cuda.synchronize()
                return None
        self.graph.replay()
        return float(self.s_loss) * self.accum


class _GraphedEndpointStep:
    """Whole-microbatch forward + backward CUDA graph for endpoint
    pretraining, with the same padding and capture rules as
    _GraphedDenseStep."""

    _side_stream: torch.cuda.Stream | None = None

    def __init__(self, model, device: str, batch_size: int, seq_len: int,
                 accum: int):
        self.model = model
        self.accum = accum
        self.bs = batch_size
        self.s_t = torch.full((batch_size, seq_len), PAD_IDX,
                              dtype=torch.long, device=device)
        self.s_m = torch.zeros(batch_size, seq_len, dtype=torch.bool,
                               device=device)
        self.s_y = torch.zeros(batch_size, dtype=torch.long,
                               device=device)
        self.s_w = torch.zeros(batch_size, dtype=torch.float32,
                               device=device)
        self.graph = None
        self.s_loss = None
        self.failed = False

    def _loss(self):
        logits = self.model(self.s_t, self.s_m)
        per = F.cross_entropy(logits, self.s_y, reduction="none")
        return (per * self.s_w).sum() / self.s_w.sum().clamp(min=1) \
            / self.accum

    def _capture(self):
        if _GraphedEndpointStep._side_stream is None:
            _GraphedEndpointStep._side_stream = torch.cuda.Stream()
        s = _GraphedEndpointStep._side_stream
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self.model.zero_grad(set_to_none=False)
                loss = self._loss()
                loss.backward()
        torch.cuda.current_stream().wait_stream(s)
        self.model.zero_grad(set_to_none=False)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self.s_loss = self._loss()
            self.s_loss.backward()
        self.graph = g

    def run(self, t: torch.Tensor, m: torch.Tensor,
            y: torch.Tensor) -> float | None:
        b, l = t.shape
        self.s_t.fill_(PAD_IDX)
        self.s_t[:b, :l].copy_(t)
        self.s_m.zero_()
        self.s_m[:b, :l].copy_(m)
        self.s_y.zero_()
        self.s_y[:b].copy_(y)
        self.s_w.zero_()
        self.s_w[:b] = 1.0
        if self.graph is None:
            if self.failed:
                return None
            try:
                self._capture()
            except Exception:
                self.failed = True
                torch.cuda.synchronize()
                return None
        self.graph.replay()
        return float(self.s_loss) * self.accum


def make_optimizer(model, cfg: dict) -> torch.optim.Optimizer:
    t = cfg["training"]
    return torch.optim.AdamW(
        model.parameters(), lr=float(t["learning_rate"]),
        weight_decay=float(t["weight_decay"]))


def train_classifier(model, source, cfg: dict, device: str,
                     log_path: str | Path, ckpt_path: str | Path | None,
                     resume_from: str | Path | None = None,
                     max_steps: int | None = None,
                     token_budget: int | None = None) -> dict:
    """Micro-batch -> (accum) -> optimizer step loop. Returns counters.

    With a token budget, training stops after the first complete optimizer
    step that reaches it; the actual token count is returned.
    """
    t = cfg["training"]
    bs = int(t["micro_batch_size"])
    accum = int(t["gradient_accumulation_steps"])
    steps = int(max_steps or t.get("max_optimizer_steps", 0))
    clip = float(t["gradient_clip_norm"])
    dense_tail = int(t.get("dense_tail", 0))
    placement = t.get("supervision_placement", "tail")
    dense = (t.get("objective") == "classical_sim_dense_supervision"
             and (dense_tail > 0 or placement == "all"))
    opt = make_optimizer(model, cfg)
    tokens_seen = targets_seen = 0
    start_step = 0
    generators = source.generators() if hasattr(source, "generators") else {}

    if resume_from is not None:
        meta = load_checkpoint(resume_from, model, opt, map_location=device)
        start_step = meta["step"]
        tokens_seen = meta.get("tokens_seen", 0)
        targets_seen = meta.get("targets_seen", 0)
        if meta.get("rng"):
            restore_rng(meta["rng"], generators)
        if meta.get("source_state"):
            source.load_state_dict(meta["source_state"])

    model.to(device).train()
    gstep = None
    if (device == "cuda" and torch.cuda.is_available()
            and getattr(model, "supports_dense_graph", False)):
        if dense:
            gstep = _GraphedDenseStep(model, device, bs,
                                      max(t["sequence_lengths"]), accum)
        else:
            gstep = _GraphedEndpointStep(model, device, bs,
                                         max(t["sequence_lengths"]),
                                         accum)
    log_f = open(log_path, "a", encoding="utf-8")
    losses = []
    t0 = time.time()
    for step in range(start_step, steps):
        # keep static .grad buffers alive for graph accumulation
        opt.zero_grad(set_to_none=(gstep is None))
        for _ in range(accum):
            batch = source.next_batch(bs)
            if dense:
                toks, mask, y, yd, sup = (b.to(device) for b in batch)
                lv = gstep.run(toks, mask, yd, sup) \
                    if gstep is not None else None
                if lv is None:
                    logits = model.forward_positions(toks, mask)
                    per = F.cross_entropy(
                        logits.reshape(-1, 4), yd.reshape(-1),
                        reduction="none").reshape(yd.shape)
                    loss = (per * sup).sum() / sup.sum().clamp_min(1) \
                        / accum
                    loss.backward()
                    lv = float(loss) * accum
                loss = None  # lv carries the unscaled loss for logging
                targets_seen += int(sup.sum())
            else:
                toks, mask, y = (b.to(device) for b in batch[:3])
                lv = gstep.run(toks, mask, y) \
                    if gstep is not None else None
                if lv is None:
                    logits = model(toks, mask)
                    loss = F.cross_entropy(logits, y) / accum
                    loss.backward()
                    lv = float(loss) * accum
                targets_seen += int(y.numel())
            tokens_seen += int(mask.sum())
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        losses.append(float(lv))
        rec = {"step": step + 1, "loss": losses[-1],
               "grad_norm": float(grad_norm),
               "tokens_seen": tokens_seen, "targets_seen": targets_seen,
               "elapsed_s": time.time() - t0}
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        if token_budget is not None and tokens_seen >= token_budget:
            break
    log_f.close()
    if ckpt_path is not None:
        save_checkpoint(
            ckpt_path, model, opt, step=step + 1,
            tokens_seen=tokens_seen, targets_seen=targets_seen,
            rng=capture_rng(generators),
            source_state=source.state_dict())
    return {"steps": step + 1, "losses": losses,
            "tokens_seen": tokens_seen, "targets_seen": targets_seen,
            "wall_time_s": time.time() - t0,
            "source_state": source.state_dict()}


# --------------------------------------------- replay of training windows

def chrom_tokens(cfg: dict, chroms: list[str]) -> dict[str, np.ndarray]:
    """Token arrays of the chromosomes from data/fasta/<chrom>.fa.gz."""
    from ..data import fasta
    out = {}
    for ch in chroms:
        seqs = fasta.read_fasta_sequences(Path(f"data/fasta/{ch}.fa.gz"))
        out[ch] = encode(seqs[ch])
    return out


def replay_train_windows(cfg: dict):
    """Replay the windows drawn by `pretrain` for this config, with the
    same random-number construction as cli.cmd_pretrain.

    Returns ([(chrom, start, L, target)] per accepted window, number of
    windows skipped for an N target, RC-augmentation flags). Windows are
    accepted until the count of non-N input tokens reaches
    training.processed_input_token_budget.
    """
    from ..data import fasta
    g = cfg["data"]["genome"]
    chroms = g["train_chromosomes"]
    ct = chrom_tokens(cfg, chroms)
    sampler = fasta.WindowSampler(
        ct, cfg["training"]["sequence_lengths"],
        cfg["training"].get("length_probabilities"),
        g["max_input_n_fraction"],
        np.random.default_rng(cfg["execution"]["seed"]),
        train_chroms=chroms)
    rc = cfg["training"].get("rc_augmentation") ==         "whole_example_before_target_split"
    rc_rng = np.random.default_rng(cfg["execution"]["seed"])
    need = cfg["training"]["processed_input_token_budget"]
    windows, n_tskipped, rc_flags = [], 0, []
    toks = 0
    while toks < need:
        inp, tgt, chrom, start, l = sampler.sample()
        if tgt == N_IDX:
            n_tskipped += 1
            continue
        if rc:
            rc_flags.append(bool(rc_rng.random() < 0.5))
        windows.append((chrom, int(start), int(l), int(tgt)))
        toks += int((inp != N_IDX).sum())
    return windows, n_tskipped, rc_flags
