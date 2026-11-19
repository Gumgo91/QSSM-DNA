"""Cached terminal Pauli features of frozen encoders (CPU, four threads).

  CUDA_VISIBLE_DEVICES=-1 python scripts/pauli_features.py
  CUDA_VISIBLE_DEVICES=-1 python scripts/pauli_features.py --check

For each encoder and replication seed: the 255 non-identity Pauli
expectations of the terminal four-qubit state of every sequence of the
promoter benchmark (train/val/test), the chromosome 9/10 promoter holdout and
human_enhancers_cohn (train/val/test), averaged over the sequence and its
reverse complement as qssm_dna.experiments.evaluate does. Encoders: the seed's
initialization, the endpoint, dense-tail-8 and all-position runs at both
token budgets, and two reference registers built from the initialization
(init_zerogamma: gamma_mode zero as in dense8_zerogamma.yaml; init_norzz:
rzz_mode off as in dense8_norzz.yaml).

The controller sees only the current token, so the terminal vector follows
from the token-switched affine recurrence r_t = A_x r_{t-1} + c_x of
qssm_dna.experiments.theory. The maps of BLOCK consecutive tokens are composed
once per encoder and applied to all sequences grouped by token word.

Inputs: the pretraining runs under the run roots (jobs.ROOTS), the datasets
of qssm_dna.experiments.datasets, configs/endpoint.yaml (matched
initialization) and the two ablation configs.
Output: <feature cache>/<condition>_s<seed>.npz (feature cache:
evaluate.CACHE_ROOT, by default runs/features; skipped if present)
  pauli255       [n, 255] float32, columns ordered as readout.pauli_features
  y, split       [n] label and "<dataset>:<part>" of each row, dataset order
  subset30_cols  pauli255 columns that equal the subset30 readout
  raw11_cols     pauli255 columns that equal the raw11 readout
  meta           JSON provenance (run directory, checkpoint hash, seconds)

--check compares the affine form with the forward pass of the model on 64
holdout sequences (complex64 as evaluate runs it, and complex128), refits the
probes of evaluate on the cached columns against every matching cell of
results/per_seed_mcc.csv, and compares the subset30 columns with the
feature caches that the eval-encoder part wrote in the same cache root
(<condition>_s<seed>/<task>/<split>.npz, where present); it writes
<feature cache>/pauli_features_check.json.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import json
import os
import sys
import time

THREADS = 4
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = str(THREADS)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, "src")
from qssm_dna.config import load_config  # noqa: E402
from qssm_dna.data.tokenizer import COMPLEMENT_IDX, encode  # noqa: E402
from qssm_dna.quantum import readout as rd  # noqa: E402
from qssm_dna.experiments import datasets, encoders  # noqa: E402
from qssm_dna.experiments import evaluate, jobs, theory  # noqa: E402

OUT = evaluate.CACHE_ROOT
SEEDS = [22, 33, 44, 55, 66, 77, 88, 99, 110, 121]
REFERENCE = {"init_zerogamma": "configs/dense8_zerogamma.yaml",
             "init_norzz": "configs/dense8_norzz.yaml"}
CONDS = ["qssm_random_init", "qssm_endpoint", "qssm_dense8", "qssm_allpos",
         "qssm_endpoint_b2500000", "qssm_dense8_b2500000",
         "qssm_allpos_b2500000", *REFERENCE]
SETS = ("gb_promoter", "holdout_promoter", "gb_cohn")
VOCAB = 5       # A, C, G, T, N; PAD never occurs inside a length group
BLOCK = 4


def lower_priority() -> None:
    """Below-normal priority class on Windows. The handle is passed as a
    pointer: with the default int conversion the call fails on 64-bit
    Python (error 6) and the priority stays unchanged."""
    if sys.platform != "win32":
        return
    k = ctypes.WinDLL("kernel32")
    k.GetCurrentProcess.restype = ctypes.c_void_p
    k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    if not k.SetPriorityClass(k.GetCurrentProcess(), 0x4000):
        raise OSError("could not lower the process priority")


def column_maps(q: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(row of the theory.affine_parts vector for each readout.pauli_features
    column, pauli255 columns of subset30, pauli255 columns of raw11).
    readout puts the Pauli of wire i in base-4 digit i of the word index,
    theory.pauli_basis in digit q-1-i; a subset uses wire i iff bit i is set,
    with Z (digit 3) for the first 2^q-1 entries and X (digit 1) after."""
    wires = np.arange(q)
    digits = (np.arange(1, 4 ** q)[:, None] // 4 ** wires) % 4
    order = (digits * 4 ** wires[::-1]).sum(1) - 1
    if not torch.equal(theory.pauli_basis(q)[1:][torch.from_numpy(order)],
                       rd._pauli_strings(q)):
        raise RuntimeError("Pauli ordering of theory and readout changed")
    bits = (np.arange(1, 1 << q)[:, None] >> wires) & 1
    word = (bits * 4 ** wires).sum(1)
    subset = np.concatenate([3 * word - 1, word - 1])
    return order, subset, subset[evaluate.raw11_index(q)]


def affine_tables(enc) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A [5, n, n], c [5, n] for tokens A, C, G, T, N and r_0 of |0...0>."""
    mats = theory.transfer_matrices(copy.deepcopy(enc))
    for r in mats.values():
        if np.abs(r[0] - np.eye(1, r.shape[1])[0]).max() > 1e-12:
            raise RuntimeError("one-token map is not trace preserving")
    parts = [theory.affine_parts(mats[x]) for x in range(VOCAB)]
    r0 = theory.pauli_basis(enc.q)[1:, 0, 0].real.clone()
    return (torch.from_numpy(np.stack([p[0] for p in parts])),
            torch.from_numpy(np.stack([p[1] for p in parts])), r0)


def word_maps(a: torch.Tensor, c: torch.Tensor, words: torch.Tensor
              ) -> tuple[torch.Tensor, torch.Tensor]:
    """Affine map of each token word [U, w] read left to right, returned as
    (transposed linear part [U, n, n], offset [U, n])."""
    m, v = a[words[:, 0]], c[words[:, 0]]
    for j in range(1, words.shape[1]):
        step = a[words[:, j]]
        m = torch.bmm(step, m)
        v = torch.bmm(step, v.unsqueeze(2)).squeeze(2) + c[words[:, j]]
    return m.transpose(1, 2).contiguous(), v


@torch.no_grad()
def terminal_vectors(a: torch.Tensor, c: torch.Tensor, r0: torch.Tensor,
                     tokens: np.ndarray, block: int = BLOCK) -> torch.Tensor:
    """tokens [N, L] -> r_L [N, n] of r_t = A_x r_{t-1} + c_x from r_0."""
    n, length = tokens.shape
    r = r0.repeat(n, 1)
    cut = length - length % block
    for lo, hi, width in ((0, cut, block), (cut, length, length - cut)):
        if hi == lo:
            continue
        place = VOCAB ** np.arange(width)
        codes = tokens[:, lo:hi].reshape(n, -1, width) @ place
        uniq, inv = np.unique(codes, return_inverse=True)
        inv = inv.reshape(codes.shape)
        m_t, v = word_maps(a, c, torch.from_numpy(uniq[:, None] // place
                                                  % VOCAB))
        for k in range(codes.shape[1]):
            order = np.argsort(inv[:, k], kind="stable")
            edge = np.searchsorted(inv[order, k], np.arange(len(uniq) + 1))
            for u in range(len(uniq)):
                idx = torch.from_numpy(order[edge[u]:edge[u + 1]])
                if len(idx):
                    r[idx] = torch.addmm(v[u], r[idx], m_t[u])
    return r


def length_groups(seqs: list[str]) -> list[tuple[np.ndarray, np.ndarray]]:
    toks = [encode(s) for s in seqs]
    lens = np.array([len(t) for t in toks])
    return [(idx, np.stack([toks[i] for i in idx]))
            for idx in (np.flatnonzero(lens == n) for n in np.unique(lens))]


@torch.no_grad()
def pauli255(enc, groups, order: np.ndarray) -> np.ndarray:
    """RC-averaged terminal Pauli expectations in readout column order."""
    a, c, r0 = affine_tables(enc)
    out = np.zeros((sum(len(i) for i, _ in groups), len(order)), np.float32)
    for idx, toks in groups:
        both = np.concatenate([toks, COMPLEMENT_IDX[toks[:, ::-1]]])
        r = terminal_vectors(a, c, r0, both)
        out[idx] = (0.5 * (r[:len(idx)] + r[len(idx):])).numpy()[:, order]
    return out


def build_encoder(name: str, seed: int, refs, base: dict):
    if name == "qssm_random_init":
        return encoders.matched_random_init(base, seed, "cpu", refs)
    if name in REFERENCE:
        init, _ = encoders.matched_random_init(base, seed, "cpu", refs)
        enc, info = encoders.matched_random_init(
            load_config(REFERENCE[name]), seed, "cpu", [])
        want = init.state_dict()
        for k, v in enc.state_dict().items():
            off = name == "init_norzz" and k == "rzz_angles"
            if v.any() if off else not torch.equal(v, want[k]):
                raise RuntimeError(f"{name} seed {seed}: {k} differs from "
                                   "the matched initialization")
        return enc, {**info, "config": REFERENCE[name],
                     "gamma_mode": enc.controller.gamma_mode,
                     "rzz_mode": enc.rzz_mode}
    ref = encoders.pick_run(refs, name, seed)
    return encoders.load_encoder(ref, "cpu"), {
        "run_dir": str(ref.run_dir),
        "checkpoint_sha256": encoders.sha256_file(
            ref.run_dir / "checkpoint.pt")}


def load_sets() -> tuple[list[str], np.ndarray, np.ndarray]:
    seqs, y, split = [], [], []
    for name in SETS:
        parts, _ = datasets.load(name)
        for part, d in parts.items():
            seqs += d["seqs"]
            y.append(d["y"])
            split += [f"{name}:{part}"] * len(d["seqs"])
    return seqs, np.concatenate(y).astype(np.int8), np.array(split)


def compute(conds: list[str], seeds: list[int]) -> None:
    order, subset, raw = column_maps(4)
    refs = encoders.scan_runs(jobs.ROOTS)
    base = load_config("configs/endpoint.yaml")
    seqs, y, split = load_sets()
    groups = length_groups(seqs)
    OUT.mkdir(parents=True, exist_ok=True)
    t_all, done = time.time(), 0
    for seed in seeds:
        for name in conds:
            p = OUT / f"{name}_s{seed}.npz"
            if p.exists():
                continue
            t0 = time.time()
            enc, info = build_encoder(name, seed, refs, base)
            feats = pauli255(enc, groups, order)
            meta = {"condition": name, "seed": seed, "encoder": info,
                    "datasets": list(SETS), "block": BLOCK,
                    "rc_average": "mean of both directions",
                    "seconds": round(time.time() - t0, 1)}
            tmp = p.with_name(p.stem + ".partial.npz")
            np.savez_compressed(tmp, pauli255=feats, y=y, split=split,
                                subset30_cols=subset, raw11_cols=raw,
                                meta=json.dumps(meta, default=str))
            tmp.replace(p)
            done += 1
            print(f"{p.name} {feats.shape} {time.time() - t0:.1f}s",
                  flush=True)
    spent = [json.loads(str(np.load(f)["meta"]))["seconds"]
             for f in (OUT / f"{c}_s{s}.npz" for s in seeds for c in conds)]
    print(f"this run: {done} files in {time.time() - t_all:.1f}s; "
          f"all {len(spent)} files: {sum(spent):.1f}s of encoder time",
          flush=True)


def check_forward(conds: list[str], seed: int) -> list[dict]:
    order, subset, raw = column_maps(4)
    refs = encoders.scan_runs(jobs.ROOTS)
    base = load_config("configs/endpoint.yaml")
    hs, _ = datasets.load("holdout_promoter")
    pick = np.linspace(0, len(hs["test"]["seqs"]) - 1, 64).astype(int)
    seqs = [hs["test"]["seqs"][i] for i in pick]
    rows = []
    for name in conds:
        enc, _ = build_encoder(name, seed, refs, base)
        f = pauli255(enc, length_groups(seqs), order)
        mine = {"pauli255": f, "subset30": f[:, subset], "raw11": f[:, raw]}
        exact = copy.deepcopy(enc).double()
        exact.cdtype = torch.complex128
        row = {"encoder": name, "seed": seed, "n": len(seqs)}
        for tag, model in (("", enc), ("_complex128", exact)):
            view = evaluate.qssm_views(
                evaluate.encode_qssm(model, seqs, "cpu", pauli=True))
            row.update({k + tag: float(np.abs(v - view[k]).max())
                        for k, v in mine.items()})
        rows.append(row)
        print(row, flush=True)
    return rows


def check_probe(conds: list[str], seeds: list[int]) -> list[dict]:
    """Refit the probes of evaluate on the cached columns for every cell
    that per_seed_mcc.csv holds (promoter and enhancer probes; raw11,
    subset30 and, where reported, pauli255) and compare the MCC."""
    ref = pd.read_csv("results/per_seed_mcc.csv").set_index(
        ["eval_name", "condition", "readout"])
    parts = ("train", "val", "test")
    rows = []
    for name in conds:
        for seed in seeds:
            z = np.load(OUT / f"{name}_s{seed}.npz")
            y, split, p = z["y"].astype(np.int64), z["split"], z["pauli255"]
            feats = {"raw11": p[:, z["raw11_cols"]],
                     "subset30": p[:, z["subset30_cols"]], "pauli255": p}
            for task in ("gb_promoter", "gb_cohn"):
                views = {r: {k: f[split == f"{task}:{k}"] for k in parts}
                         for r, f in feats.items()
                         if (f"{task}:test", name, r) in ref.index}
                if not views:
                    continue
                extra = None
                if task == "gb_promoter":
                    hold = split == "holdout_promoter:test"
                    extra = {"holdout_promoter": (
                        {r: feats[r][hold] for r in views}, y[hold])}
                fit, _, _ = evaluate.evaluate_views(
                    views, {k: y[split == f"{task}:{k}"] for k in parts},
                    extra)
                for r in fit:
                    ev = r["eval_set"]
                    ev = f"{task}:test" if ev == "test" else ev
                    want = float(ref.loc[(ev, name, r["readout"]),
                                         f"{float(seed)}"])
                    rows.append({"condition": name, "seed": seed,
                                 "readout": r["readout"], "eval_name": ev,
                                 "mcc": r["eval_mcc"], "mcc_paper": want,
                                 "diff": r["eval_mcc"] - want})
            print(name, seed, "probe MCC, max |difference|",
                  max((abs(r["diff"]) for r in rows if r["condition"] == name
                       and r["seed"] == seed), default=None), flush=True)
    return rows


def probe_summary(rows: list[dict]) -> list[dict]:
    """Mean and 95% t interval over seeds of the refitted MCC, next to the
    mean of the values in per_seed_mcc.csv."""
    from scipy import stats
    out = []
    for (name, readout, ev), g in pd.DataFrame(rows).groupby(
            ["condition", "readout", "eval_name"], sort=False):
        x, n = g.mcc.to_numpy(), len(g)
        half = float(stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n))
        out.append({"condition": name, "readout": readout, "eval_name": ev,
                    "n": n, "seeds": g.seed.tolist(), "mcc_mean": x.mean(),
                    "ci95_low": x.mean() - half, "ci95_high": x.mean() + half,
                    "mcc_mean_paper": g.mcc_paper.mean(),
                    "max_abs_diff": g["diff"].abs().max()})
    return out


def check_eval_cache(conds: list[str], seeds: list[int]) -> list[dict]:
    """subset30 columns against the feature caches of the eval-encoder
    part (<condition>_s<seed>/<task>/<split>.npz), where they exist."""
    rows = []
    for name in conds:
        for seed in seeds:
            d = OUT / f"{name}_s{seed}"
            if not d.exists():
                continue
            z = np.load(OUT / f"{name}_s{seed}.npz")
            sub, split = z["pauli255"][:, z["subset30_cols"]], z["split"]
            for f in sorted(d.glob("*/*.npz")):
                key = f"{f.parent.name}:{f.stem}"
                if not (split == key).any():
                    continue
                other = evaluate.qssm_views(dict(np.load(f)))["subset30"]
                rows.append({"condition": name, "seed": seed, "split": key,
                             "n": len(other), "max_abs_diff": float(
                                 np.abs(sub[split == key] - other).max())})
        got = [r["max_abs_diff"] for r in rows if r["condition"] == name]
        print(name, len(got), "cached evaluation splits, max |diff|",
              max(got, default=None), flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--conditions", default=",".join(CONDS))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    args = ap.parse_args()
    torch.set_num_threads(THREADS)
    lower_priority()
    conds = args.conditions.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    if not args.check:
        compute(conds, seeds)
        return
    out = {"forward_pass": check_forward(conds, seeds[0]),
           "probe": check_probe(conds, seeds),
           "eval_cache": check_eval_cache(conds, seeds)}
    out["probe_summary"] = probe_summary(out["probe"])
    (OUT / "pauli_features_check.json").write_text(
        json.dumps(out, indent=2, default=float))
    fw, pr, pc = (pd.DataFrame(out[k])
                  for k in ("forward_pass", "probe", "eval_cache"))
    print("forward pass, max |difference|:",
          fw.drop(columns=["encoder", "seed", "n"]).max().to_dict())
    print("probe MCC, max |difference| to per_seed_mcc.csv:",
          pr["diff"].abs().groupby([pr.eval_name, pr.readout]).max()
          .to_dict())
    print(pd.DataFrame(out["probe_summary"]).drop(columns="seeds")
          .round(4).to_string(index=False))
    print("eval-encoder caches, max |difference| of subset30:",
          pc.max_abs_diff.max() if len(pc) else None)


if __name__ == "__main__":
    main()
