"""Noise and finite shots for the reference register and the ten-fold
encoders (CPU).

The evaluation-only protocol of qssm_dna.experiments.jobs_noise (single- and
two-qubit depolarizing errors, dephasing per token, readout bit flips; exact
expectations or 1,024 sampled bitstrings per basis and direction, ten shot
seeds; probe fitted on noiseless training features and left unchanged) is
applied to the promoter holdout for the matched initialization and the three
ten-fold-budget encoders, ten replication seeds, subset30 and raw11 readouts:

  none      exact, 1,024 shots
  p2_0.001  exact
  moderate  exact, 1,024 shots

The controller sees only the current token, so a noisy token cell is one of
five fixed linear maps. Each map is read off noise.noisy_forward_state by
starting it from the Pauli basis (float64), and the terminal state of every
holdout sequence and its reverse complement follows from the recurrence
r_t = R_x r_{t-1} of qssm_dna.experiments.theory. Outcome distributions,
readout flips, parities and shot sampling are those of jobs_noise. For every encoder
and setting the features are compared with jobs_noise.noisy_dirs (the
per-token density-matrix pass, complex64) on 32 holdout sequences, and the
noiseless features with the cached holdout features of the same encoder.

The noiseless probe of an encoder is fitted on its cached training and
validation features: <feature cache>/<condition>_s<seed>/gb_promoter (written
by the eval-encoder part; the features the stored probes were fitted on),
else <feature cache>/<condition>_s<seed>.npz of pauli_features.py, else a
CPU pass (feature cache: evaluate.CACHE_ROOT, by default runs/features).
Encoders come from the pretraining runs under the run roots (jobs.ROOTS).

  CUDA_VISIBLE_DEVICES=-1 python scripts/noise_reference.py --verify
  CUDA_VISIBLE_DEVICES=-1 python scripts/noise_reference.py

The run starts by reproducing the stored qssm_allpos rows (seeds 22, 33) of
the noise_metrics.csv files of the noise runs (<run root>/**/*_noise_*/) and
stops if a per-seed holdout MCC differs by more than 0.01; --verify ends
after that step. Encoders whose rows are in the output are skipped on a
restart. Two worker processes with one numerical thread each do the encoder
passes (two threads in total). results/all_metrics_long.csv, where present,
is used to compare the noiseless rows with the stored evaluations.

Output: results/noise_reference.csv, one table with four levels:
  check     CPU value next to the stored value of the noise run (qssm_allpos)
  raw       one row per encoder, setting, readout and shot draw
  seed      per encoder seed (shots: mean over the ten draws)
  contrast  ten-fold encoder minus qssm_random_init under the same setting,
            paired over seeds (MCC and AUROC; Holm over the three encoders)
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import multiprocessing
import os
import sys
import time
from pathlib import Path

WORKERS, THREADS = 2, 1
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = str(THREADS)
if sys.platform == "win32":
    _K32 = ctypes.WinDLL("kernel32")
    _K32.GetCurrentProcess.restype = ctypes.c_void_p
    _K32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    if not _K32.SetPriorityClass(_K32.GetCurrentProcess(), 0x4000):
        raise OSError("could not lower the process priority")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(THREADS)
sys.path.insert(0, "src")
from qssm_dna.config import load_config  # noqa: E402
from qssm_dna.data.tokenizer import encode, pad_tokens  # noqa: E402
from qssm_dna.eval.metrics import binary_metrics  # noqa: E402
from qssm_dna.models.encoder import rc_tokens  # noqa: E402
from qssm_dna.quantum import density, noise  # noqa: E402
from qssm_dna.quantum.noise import NoiseModel  # noqa: E402
from qssm_dna.experiments import datasets, encoders, evaluate  # noqa: E402
from qssm_dna.experiments import jobs, stats, theory  # noqa: E402
from qssm_dna.experiments.jobs_noise import (NOISE_SETTINGS,  # noqa: E402
                                          NOISE_SHOTS, SHOT_SEEDS,
                                          feat_exact, feat_shots, noisy_dirs)

OUT = Path("results/noise_reference.csv")
SEEDS = stats.SEEDS
REFERENCE = "qssm_random_init"
TENFOLD = ("qssm_endpoint_b2500000", "qssm_dense8_b2500000",
           "qssm_allpos_b2500000")
SHOTS = str(NOISE_SHOTS)
VARIANTS = (("none", "exact"), ("none", SHOTS), ("p2_0.001", "exact"),
            ("moderate", "exact"), ("moderate", SHOTS))
CHECK_CONDITION, CHECK_SEEDS = "qssm_allpos", (22, 33)
CHECK_VARIANTS = tuple((s, m) for s in ("none", "p2_0.001", "moderate")
                       for m in ("exact", SHOTS))
CHECK_TOL = 0.01
FEATURE_TOL = 1e-3
LIBRARY_N = 32
VOCAB = 5       # A, C, G, T, N; PAD executes no cell
READOUTS = ("subset30", "raw11")
KEYS = ["condition", "seed", "setting", "shots", "readout"]
RAW_COLS = ["level", "condition", "seed", "setting", "noise_p1", "noise_p2",
            "noise_p_phi", "noise_readout", "shots", "shot_seed", "readout",
            "threshold", "eval_set", "eval_mcc", "eval_auroc", "encoder",
            "probe_features", "probe_sha256", "holdout_cache_maxdiff",
            "library_maxdiff"]
CHECK_COLS = ["level"] + KEYS + [
    "eval_mcc", "eval_auroc", "n_draws", "threshold", "stored_mcc",
    "stored_auroc", "stored_threshold", "mcc_diff", "auroc_diff",
    "max_draw_abs_diff", "encoder", "probe_features",
    "holdout_cache_maxdiff", "library_maxdiff"]
COLUMNS = ["level", "condition", "seed", "setting", "noise_p1", "noise_p2",
           "noise_p_phi", "noise_readout", "shots", "shot_seed", "readout",
           "eval_set", "threshold", "eval_mcc", "eval_auroc", "n_draws",
           "eval_mcc_draw_sd", "contrast", "metric", "mean_a", "sd_a",
           "mean_b", "sd_b", "n", "mean_diff", "sd_diff", "ci95_low",
           "ci95_high", "t", "df", "p_t", "p_perm", "p_perm_holm",
           "p_wilcoxon", "n_positive", "seeds", "stored_mcc", "stored_auroc",
           "stored_threshold", "mcc_diff", "auroc_diff", "max_draw_abs_diff",
           "encoder", "probe_features", "probe_sha256",
           "holdout_cache_maxdiff", "library_maxdiff"]
INT_COLS = ["seed", "shot_seed", "n_draws", "n", "df", "n_positive"]
ORDER = {"condition": {c: i for i, c in enumerate((REFERENCE,) + TENFOLD)},
         "setting": {s: i for i, s in enumerate(
             dict.fromkeys(s for s, _ in VARIANTS))},
         "shots": {"exact": 0, SHOTS: 1},
         "readout": {r: i for i, r in enumerate(READOUTS)}}


class BasisStart:
    """Encoder view that starts noise.noisy_forward_state from the Pauli
    basis instead of |0...0><0...0|. The cell is linear, so one token
    returns the image of every basis element under the noisy cell."""

    def __init__(self, enc, basis: torch.Tensor):
        self.enc, self.basis = enc, basis
        self.q, self.cell_depth = enc.q, enc.cell_depth
        self.rzz_angles = enc.rzz_angles

    def controls(self, tokens: torch.Tensor):
        return self.enc.controls(tokens)

    def initial_state(self, batch_size: int, device=None) -> torch.Tensor:
        return self.basis.clone()


@torch.no_grad()
def token_maps(enc, model: NoiseModel) -> torch.Tensor:
    """R_x[P, Q] = Tr(P Phi_x(Q)) / 2^q of the NoiseModel cell for tokens
    A, C, G, T, N -> [5, 4^q, 4^q] float64 (theory.transfer_matrices with
    the cell of noise.noisy_forward_state)."""
    q = enc.q
    exact = copy.deepcopy(enc).double()
    exact.cdtype = torch.complex128
    basis = theory.pauli_basis(q)
    n = basis.shape[0]
    start = BasisStart(exact, basis)
    valid = torch.ones(n, 1, dtype=torch.bool)
    unit = torch.eye(1, n, dtype=torch.float64)[0]
    maps = []
    for x in range(VOCAB):
        img = noise.noisy_forward_state(start, torch.full((n, 1), x), valid,
                                        model)
        r = torch.einsum("pij,qji->pq", basis, img) / (1 << q)
        if float(r.imag.abs().max()) > 1e-10 or \
                float((r.real[0] - unit).abs().max()) > 1e-10:
            raise RuntimeError("one-token map does not preserve "
                               "Hermiticity and trace")
        maps.append(r.real)
    return torch.stack(maps)


def readout_maps(q: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Z- and X-basis outcome probabilities as linear maps of the Pauli
    vector: [4^q, 2^q] each."""
    basis = theory.pauli_basis(q)
    turned = density.to_matrix(
        density.hadamard_all(density.to_axes(basis, q), q), q)
    return (basis.diagonal(dim1=-2, dim2=-1).real / (1 << q),
            turned.diagonal(dim1=-2, dim2=-1).real / (1 << q))


@torch.no_grad()
def terminal_vectors(maps: torch.Tensor, r0: torch.Tensor,
                     tokens: torch.Tensor, mask: torch.Tensor
                     ) -> torch.Tensor:
    """r_L [N, 4^q] of r_t = R_x r_{t-1} over the valid tokens of a row."""
    r = r0.repeat(len(tokens), 1)
    mt = maps.transpose(1, 2).contiguous()
    for t in range(tokens.shape[1]):
        for x in range(VOCAB):
            idx = torch.nonzero(mask[:, t] & (tokens[:, t] == x)).squeeze(1)
            if len(idx):
                r[idx] = r[idx] @ mt[x]
    return r


@torch.no_grad()
def transfer_dirs(enc, seqs: list[str], model: NoiseModel) -> dict:
    """jobs_noise.noisy_dirs in the Pauli transfer form: exact outcome
    distributions per direction with readout flips applied."""
    q = enc.q
    maps = token_maps(enc, model)
    r0 = theory.pauli_basis(q)[:, 0, 0].real.clone()
    t, m = pad_tokens([encode(s) for s in seqs])
    t, m = torch.from_numpy(t), torch.from_numpy(m)
    out = {}
    for tag, tok in (("f", t), ("r", rc_tokens(t, m))):
        r = terminal_vectors(maps, r0, tok, m)
        for basis, to_probs in zip("zx", readout_maps(q)):
            p = (r @ to_probs).clamp_min(0.0)
            out[basis + tag] = noise.readout_flip_probs(
                p / p.sum(-1, keepdim=True), model.readout, q).numpy()
    return out


def n_raw(variants) -> int:
    draws = sum(1 if m == "exact" else len(SHOT_SEEDS) for _, m in variants)
    return draws * len(READOUTS)


def load_encoder(refs, base, cond: str, seed: int):
    if cond == REFERENCE:
        enc, info = encoders.matched_random_init(base, seed, "cpu", refs)
        return enc, f"matched_random_init:{info['init_value_sha256'][:16]}"
    ref = encoders.pick_run(refs, cond, seed)
    return encoders.load_encoder(ref, "cpu"), str(ref.run_dir)


def cached(key: str, task: str, split: str):
    p = evaluate.CACHE_ROOT / key / task / f"{split}.npz"
    if p.exists():
        return evaluate.qssm_views(dict(np.load(p)))["subset30"], str(p)
    return None


def probe_features(enc, key: str, gb: dict) -> tuple[dict, str]:
    """Noiseless RC-averaged subset30 of the training and validation
    splits and where it came from."""
    want = {sp: len(gb[sp]["y"]) for sp in ("train", "val")}
    hit = {sp: cached(key, "gb_promoter", sp) for sp in want}
    if all(h is not None and len(h[0]) == want[sp] for sp, h in hit.items()):
        return ({sp: h[0] for sp, h in hit.items()},
                str(Path(hit["train"][1]).parent))
    flat = evaluate.CACHE_ROOT / f"{key}.npz"
    if flat.exists():
        z = np.load(flat)
        sub, split = z["pauli255"][:, z["subset30_cols"]], z["split"]
        fx = {sp: sub[split == f"gb_promoter:{sp}"] for sp in want}
        if all(len(fx[sp]) == want[sp] for sp in want):
            return fx, str(flat)
    clean = NoiseModel()
    return ({sp: feat_exact(transfer_dirs(enc, gb[sp]["seqs"], clean))
             for sp in want}, "cpu_pass")


def cache_name(p: str) -> str:
    """Name of a feature-cache entry relative to the cache root."""
    try:
        return Path(p).resolve().relative_to(
            evaluate.CACHE_ROOT.resolve()).as_posix()
    except ValueError:
        return p


def run_encoder(enc, cond: str, seed: int, source: str, gb: dict, ho: dict,
                variants) -> list[dict]:
    key = f"{cond}_s{seed}"
    y = ho["test"]["y"]
    seqs = ho["test"]["seqs"]
    pick = np.linspace(0, len(seqs) - 1, LIBRARY_N).astype(int)
    fx, probe_source = probe_features(enc, key, gb)
    cols = {"subset30": slice(None), "raw11": evaluate.raw11_index()}
    probes = {ro: evaluate.fit_probe(fx["train"][:, c], gb["train"]["y"],
                                     fx["val"][:, c], gb["val"]["y"])
              for ro, c in cols.items()}
    sha = {ro: evaluate.probe_sha(p["head"])[:16] for ro, p in probes.items()}
    rows, cache_diff = [], float("nan")
    for name in dict.fromkeys(s for s, _ in variants):
        model = NoiseModel(**NOISE_SETTINGS[name])
        d = transfer_dirs(enc, seqs, model)
        exact = feat_exact(d)
        lib_diff = float(np.abs(exact[pick] - feat_exact(noisy_dirs(
            enc, [seqs[i] for i in pick], model, "cpu"))).max())
        if lib_diff > FEATURE_TOL:
            raise RuntimeError(f"{key} {name}: transfer form differs from "
                               f"noisy_dirs by {lib_diff:.2e}")
        if name == "none":
            ref = cached(key, "holdout_promoter", "test")
            if ref is not None:
                cache_diff = float(np.abs(exact - ref[0]).max())
                if cache_diff > FEATURE_TOL:
                    raise RuntimeError(
                        f"{key}: noiseless holdout features differ from "
                        f"{ref[1]} by {cache_diff:.2e}")
        feats = []
        if (name, "exact") in variants:
            feats.append(("exact", None, exact))
        if (name, SHOTS) in variants:
            feats += [(SHOTS, ss, feat_shots(
                d, NOISE_SHOTS, 10_000 * seed + 100 * ss + 1))
                for ss in SHOT_SEEDS]
        for shots, ss, f in feats:
            for ro, c in cols.items():
                thr = probes[ro]["threshold"]
                met = binary_metrics(
                    y, evaluate.predict(probes[ro]["head"], f[:, c]), thr)
                rows.append({
                    "level": "raw", "condition": cond, "seed": seed,
                    "setting": name,
                    **{f"noise_{k}": v for k, v in model.as_dict().items()},
                    "shots": shots, "shot_seed": ss, "readout": ro,
                    "threshold": thr, "eval_set": "holdout_promoter",
                    "eval_mcc": met["mcc"], "eval_auroc": met["auroc"],
                    "encoder": Path(source).name,
                    "probe_features": cache_name(probe_source),
                    "probe_sha256": sha[ro], "library_maxdiff": lib_diff})
    for r in rows:
        r["holdout_cache_maxdiff"] = cache_diff
    return rows


_CTX: dict = {}


def job(task: tuple) -> tuple[list[dict], float]:
    """One encoder in a worker process; data and run index load once."""
    cond, seed, variants = task
    if not _CTX:
        _CTX.update(refs=encoders.scan_runs(jobs.ROOTS),
                    base=load_config("configs/endpoint.yaml"),
                    gb=datasets.load("gb_promoter")[0],
                    ho=datasets.load("holdout_promoter")[0])
    t0 = time.time()
    enc, src = load_encoder(_CTX["refs"], _CTX["base"], cond, seed)
    rows = run_encoder(enc, cond, seed, src, _CTX["gb"], _CTX["ho"],
                       variants)
    return rows, time.time() - t0


def ordered(raw: pd.DataFrame) -> pd.DataFrame:
    return raw.sort_values(
        ["seed", "condition", "setting", "shots", "shot_seed", "readout"],
        key=lambda c: c.map(ORDER[c.name]) if c.name in ORDER else c,
        kind="stable").reset_index(drop=True)


def per_seed(raw: pd.DataFrame) -> pd.DataFrame:
    first = ["noise_p1", "noise_p2", "noise_p_phi", "noise_readout",
             "threshold", "eval_set", "encoder", "probe_features",
             "probe_sha256", "holdout_cache_maxdiff", "library_maxdiff"]
    out = raw.groupby(KEYS, sort=False).agg(
        eval_mcc=("eval_mcc", "mean"), eval_auroc=("eval_auroc", "mean"),
        n_draws=("eval_mcc", "size"),
        eval_mcc_draw_sd=("eval_mcc", lambda v: v.std(ddof=1)
                          if len(v) > 1 else np.nan),
        **{c: (c, "first") for c in first}).reset_index()
    out.insert(0, "level", "seed")
    return out


def stored_rows() -> pd.DataFrame:
    # one file per noise run, also when a run directory is under two roots
    fs = {}
    for root in jobs.ROOTS:
        for f in sorted(Path(root).glob("**/*_noise_*/noise_metrics.csv")):
            fs.setdefault(f.parent.name, f)
    fs = [fs[k] for k in sorted(fs)]
    if not fs:
        raise SystemExit("no noise_metrics.csv of a noise run under "
                         f"{jobs.ROOTS}")
    nm = pd.concat([pd.read_csv(f) for f in fs])
    nm["shots"] = nm["shots"].astype(str)
    return nm[(nm.condition == CHECK_CONDITION)
              & (nm.policy == "A_noiseless_probe")
              & (nm.eval_set == "holdout_promoter")]


def check_rows(raw: pd.DataFrame) -> pd.DataFrame:
    draws = raw.merge(stored_rows(), on=KEYS + ["shot_seed"],
                      suffixes=("", "_stored"))
    if len(draws) != len(raw):
        raise RuntimeError("stored noise rows missing for the check")
    draws["d"] = (draws.eval_mcc - draws.eval_mcc_stored).abs()
    out = draws.groupby(KEYS, sort=False).agg(
        eval_mcc=("eval_mcc", "mean"), eval_auroc=("eval_auroc", "mean"),
        n_draws=("eval_mcc", "size"), threshold=("threshold", "first"),
        stored_mcc=("eval_mcc_stored", "mean"),
        stored_auroc=("eval_auroc_stored", "mean"),
        stored_threshold=("threshold_stored", "first"),
        max_draw_abs_diff=("d", "max"), encoder=("encoder", "first"),
        probe_features=("probe_features", "first"),
        holdout_cache_maxdiff=("holdout_cache_maxdiff", "first"),
        library_maxdiff=("library_maxdiff", "first")).reset_index()
    out["mcc_diff"] = out.eval_mcc - out.stored_mcc
    out["auroc_diff"] = out.eval_auroc - out.stored_auroc
    out.insert(0, "level", "check")
    return out[CHECK_COLS]


def contrast_rows(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for setting, shots in VARIANTS:
        for readout in READOUTS:
            for metric in ("eval_mcc", "eval_auroc"):
                sub = seed_df[(seed_df.setting == setting)
                              & (seed_df.shots == shots)
                              & (seed_df.readout == readout)]
                w = sub.pivot(index="seed", columns="condition",
                              values=metric)
                block = []
                for cond in TENFOLD:
                    if cond not in w or REFERENCE not in w:
                        continue
                    pair = w[[cond, REFERENCE]].dropna()
                    if len(pair) < 3:
                        continue
                    block.append({
                        "level": "contrast", "condition": cond,
                        "contrast": f"{cond}_minus_{REFERENCE}",
                        "setting": setting, "shots": shots,
                        "readout": readout, "eval_set": "holdout_promoter",
                        "metric": metric,
                        "mean_a": pair[cond].mean(),
                        "sd_a": pair[cond].std(),
                        "mean_b": pair[REFERENCE].mean(),
                        "sd_b": pair[REFERENCE].std(),
                        **stats.paired_summary(pair[cond].values,
                                               pair[REFERENCE].values),
                        "seeds": " ".join(map(str, pair.index))})
                adj = stats.holm({r["condition"]: r["p_perm"]
                                  for r in block})
                for r in block:
                    r["p_perm_holm"] = adj[r["condition"]]
                rows += block
    return pd.DataFrame(rows)


def compare_with_evals(seed_df: pd.DataFrame) -> None:
    p = Path("results/all_metrics_long.csv")
    if not p.exists():
        return
    ev = pd.read_csv(p, low_memory=False)
    ev = ev[(ev.task == "gb_promoter") & (ev.eval_name == "holdout_promoter")]
    m = seed_df[(seed_df.setting == "none") & (seed_df.shots == "exact")] \
        .merge(ev, on=["condition", "seed", "readout"],
               suffixes=("", "_eval"))
    if len(m):
        print(f"noiseless exact rows vs stored evaluations: {len(m)} rows, "
              f"max |MCC diff| "
              f"{np.abs(m.eval_mcc - m.eval_mcc_eval).max():.2e}, "
              f"max |AUROC diff| "
              f"{np.abs(m.eval_auroc - m.eval_auroc_eval).max():.2e}")


def write(frames: list[pd.DataFrame]) -> None:
    frames = [f for f in frames if len(f)]
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    extra = sorted(set(df.columns) - set(COLUMNS))
    if extra:
        raise RuntimeError(f"columns without a place in the output: {extra}")
    df = df.reindex(columns=COLUMNS)
    df[INT_COLS] = df[INT_COLS].astype("Int64")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_name(OUT.stem + ".partial.csv")
    df.to_csv(tmp, index=False)
    tmp.replace(OUT)


def previous() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not OUT.exists():
        return pd.DataFrame(), pd.DataFrame()
    df = pd.read_csv(OUT, dtype={"shots": str}, low_memory=False,
                     float_precision="round_trip")
    check = df[df.level == "check"].reindex(columns=CHECK_COLS)
    raw = df[df.level == "raw"].reindex(columns=RAW_COLS)
    return check.astype({"seed": int}), raw.astype({"seed": int})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="reproduce the stored qssm_allpos rows and stop")
    args = ap.parse_args()
    check, raw = previous()
    with multiprocessing.Pool(WORKERS) as pool:
        n_check = len(CHECK_SEEDS) * len(CHECK_VARIANTS) * len(READOUTS)
        if len(check) != n_check:
            rows = []
            tasks = [(CHECK_CONDITION, s, CHECK_VARIANTS)
                     for s in CHECK_SEEDS]
            for new, dt in pool.imap_unordered(job, tasks):
                rows += new
                print(f"check {new[0]['condition']} s{new[0]['seed']} "
                      f"{dt:.0f} s", flush=True)
            check = check_rows(pd.DataFrame(rows).sort_values(
                "seed", kind="stable"))
            write([check, raw])
        print(check[KEYS[1:] + ["eval_mcc", "stored_mcc", "mcc_diff",
                                "max_draw_abs_diff", "auroc_diff",
                                "threshold", "stored_threshold"]]
              .round(5).to_string(index=False))
        worst = float(check.mcc_diff.abs().max())
        print(f"check: max |per-seed holdout MCC - stored| = {worst:.5f} "
              f"(tolerance {CHECK_TOL}); features vs noisy_dirs "
              f"{check.library_maxdiff.max():.1e}, vs cached holdout "
              f"{check.holdout_cache_maxdiff.max():.1e}", flush=True)
        if worst > CHECK_TOL:
            raise SystemExit("stored noise protocol not reproduced on the "
                             "CPU; no reference numbers written")
        if args.verify:
            return

        done = set()
        if len(raw):
            size = raw.groupby(["condition", "seed"]).size()
            done = {k for k, n in size.items() if n == n_raw(VARIANTS)}
            raw = raw[[k in done for k in zip(raw.condition, raw.seed)]]
        tasks = [(cond, s, VARIANTS) for s in SEEDS
                 for cond in (REFERENCE,) + TENFOLD if (cond, s) not in done]
        for new, dt in pool.imap_unordered(job, tasks):
            new = pd.DataFrame(new)[RAW_COLS]
            raw = pd.concat([raw, new], ignore_index=True) if len(raw) \
                else new
            write([check, raw])
            ex = new[(new.shots == "exact") & (new.readout == "subset30")]
            print(f"{new.condition.iloc[0]} s{new.seed.iloc[0]} {dt:.0f} s  "
                  "subset30 exact MCC "
                  + "  ".join(f"{a}={b:.3f}" for a, b in
                              zip(ex.setting, ex.eval_mcc))
                  + f"  cache diff {new.holdout_cache_maxdiff.iloc[0]:.1e}"
                  + f"  library diff {new.library_maxdiff.max():.1e}",
                  flush=True)

    raw = ordered(raw)
    seed_df = per_seed(raw)
    con = contrast_rows(seed_df)
    write([check, raw, seed_df, con])
    compare_with_evals(seed_df)
    cells = seed_df.groupby(["readout", "setting", "shots", "condition"],
                            sort=False)[["eval_mcc", "eval_auroc"]] \
        .agg(["mean", "std", "count"]).round(3)
    print(cells.to_string())
    print(con[con.metric == "eval_mcc"][
        ["contrast", "setting", "shots", "readout", "n", "mean_a", "mean_b",
         "mean_diff", "ci95_low", "ci95_high", "p_perm", "p_perm_holm",
         "n_positive"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
