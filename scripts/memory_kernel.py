"""Damping rates and memory kernel of the trained recurrence (CPU).

  CUDA_VISIBLE_DEVICES=-1 python scripts/memory_kernel.py [run|verify|summary]

Two quantities, for each replication seed's matched initialization, the
0.25 M-token encoders, the no-coupling ablations and the 2.5 M-token
encoders (encoders from the pretraining runs under the run roots,
jobs.ROOTS; sequences of the promoter holdout from datasets.load). The
kernel follows from the affine form of the token maps in the normalised
Pauli basis, r_t = A_x r_{t-1} + c_x, with the 30 same-basis parities a
fixed selection M r_L of the terminal vector.

  results/damping_rates.csv
                        amplitude-damping probability gamma of each qubit
                        (1..4 = wires 0..3) for each token; gamma_mean_acgt is
                        the mean over the four nucleotides, gamma_mean_acgtn
                        the mean over all five tokens.
  results/memory_kernel.csv
                        one row per condition and seed. sens_d<d>: on 256
                        promoter-holdout sequences (drawn once, seed 1729) the
                        base at distance d from the 3' end (d = 0 is the last
                        base read) is replaced by each of the three other
                        nucleotides; the change of the 30 parities of the
                        terminal state of the forward read is
                        M A_{x_L}..A_{x_{t+1}} [(A_y - A_{x_t}) r_{t-1}
                        + c_y - c_{x_t}], its root mean square over the 30
                        parities is averaged over substitutions and sequences.
                        pooled_d<d>: root of the mean squared change over
                        parities, substitutions and sequences. d50, d90:
                        smallest d with sum_{d' <= d} sens^2 at least 50 % and
                        90 % of the sum over d' = 0..250 (a flat kernel gives
                        125 and 225); d50_pooled, d90_pooled likewise.
                        rel_d<d> = sens_d<d> / sens_d0.

run      computes both files (a condition and seed present in both is
         skipped) and prints the summary.
verify   for two encoders, compares the kernel and the individual parity
         changes at five distances with the model's own forward pass on the
         substituted sequences (complex128 and the native complex64).
summary  means and SDs over seeds, paired changes from initialization and
         per-qubit damping rates from the two files.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

THREADS = 2
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = str(THREADS)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, "src")
from qssm_dna.config import load_config  # noqa: E402
from qssm_dna.data.tokenizer import encode  # noqa: E402
from qssm_dna.quantum import readout as rd  # noqa: E402
from qssm_dna.experiments import datasets, encoders, jobs  # noqa: E402
from qssm_dna.experiments import stats, theory  # noqa: E402

CONDS = ["qssm_random_init", "qssm_endpoint", "qssm_dense8", "qssm_allpos",
         "qssm_dense8_norzz", "qssm_allpos_norzz", "qssm_endpoint_b2500000",
         "qssm_dense8_b2500000", "qssm_allpos_b2500000",
         "qssm_dense8_w256_b2500000"]
SEEDS = [22, 33, 44, 55, 66, 77, 88, 99, 110, 121]
RES = Path("results")
FILES = {"damping": RES / "damping_rates.csv",
         "kernel": RES / "memory_kernel.csv"}
TOKENS = "ACGTN"
N_SEQ = 256
DRAW_SEED = 1729
REL_D = (0, 8, 32, 64, 128, 250)
VERIFY = (("qssm_dense8", 22), ("qssm_allpos_b2500000", 121))
VERIFY_D = (0, 8, 32, 128, 250)


def readout_matrix(q: int) -> np.ndarray:
    """M [2(2^q-1), 4^q-1]: subset_readout(rho) = M r for a state rho."""
    basis = theory.pauli_basis(q).numpy()
    s = rd._subset_sign_table(q).numpy()
    had = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    hq = had
    for _ in range(q - 1):
        hq = np.kron(hq, had)
    obs = np.array([np.diag(v) for v in s] + [hq @ np.diag(v) @ hq for v in s])
    m = np.einsum("kij,pji->kp", obs, basis).real / (1 << q)
    sel = m[:, 1:]
    if not (np.allclose(sel.sum(1), 1.0) and np.allclose(sel * sel, sel)
            and np.allclose(m[:, 0], 0.0)):
        raise RuntimeError("subset readout is not a Pauli selection")
    return np.round(sel)


def initial_r(q: int) -> np.ndarray:
    """Pauli vector of |0..0><0..0|."""
    return theory.pauli_basis(q).numpy()[1:, 0, 0].real.copy()


def affine(enc) -> tuple[np.ndarray, np.ndarray]:
    """A [4, n, n], c [4, n] for A, C, G, T (converts enc to complex128)."""
    r = theory.transfer_matrices(enc, tokens=(0, 1, 2, 3))
    parts = [theory.affine_parts(r[x]) for x in range(4)]
    return (np.stack([p[0] for p in parts]), np.stack([p[1] for p in parts]))


def kernel(a: np.ndarray, c: np.ndarray, m: np.ndarray, r0: np.ndarray,
           toks: np.ndarray, keep=()) -> tuple[np.ndarray, np.ndarray, dict]:
    """sens [L], pooled [L] indexed by distance from the last token, and the
    parity changes [4, B, K] (row y = substituted base) at distances `keep`."""
    b, l = toks.shape
    n, k = a.shape[-1], m.shape[0]
    prev = np.empty((l, b, n))
    r = np.tile(r0, (b, 1))
    for t in range(l):
        prev[t] = r
        new = np.empty_like(r)
        for x in range(4):
            i = np.flatnonzero(toks[:, t] == x)
            new[i] = r[i] @ a[x].T + c[x]
        r = new
    back = np.tile(m, (b, 1, 1))
    rows = np.arange(b)
    sens, pooled, kept = np.zeros(l), np.zeros(l), {}
    for t in range(l - 1, -1, -1):
        d = l - 1 - t
        x_t = toks[:, t]
        img = np.stack([prev[t] @ a[y].T + c[y] for y in range(4)])
        change = np.einsum("bkn,ybn->ybk", back, img - img[x_t, rows])
        ms = (change ** 2).mean(-1)
        sens[d] = np.sqrt(ms).sum(0).mean() / 3.0
        pooled[d] = np.sqrt(ms.sum(0).mean() / 3.0)
        if d in keep:
            kept[d] = change
        for x in range(4):
            i = np.flatnonzero(x_t == x)
            back[i] = (back[i].reshape(-1, n) @ a[x]).reshape(len(i), k, n)
    return sens, pooled, kept


def mass_distance(sens: np.ndarray, frac: float) -> int:
    cum = np.cumsum(sens ** 2)
    return int(np.searchsorted(cum, frac * cum[-1]))


def damping_rows(cond: str, seed: int, enc) -> list[dict]:
    g = theory.control_table(enc)[40:60].reshape(5, 4)
    return [{"condition": cond, "seed": seed, "qubit": i + 1,
             **{f"gamma_{TOKENS[x]}": float(g[x, i]) for x in range(5)},
             "gamma_mean_acgt": float(g[:4, i].mean()),
             "gamma_mean_acgtn": float(g[:, i].mean())} for i in range(4)]


def kernel_row(cond: str, seed: int, sens: np.ndarray, pooled: np.ndarray,
               n_seq: int) -> dict:
    return {"condition": cond, "seed": seed, "n_sequences": n_seq,
            "seq_len": len(sens),
            "d50": mass_distance(sens, 0.5), "d90": mass_distance(sens, 0.9),
            **{f"rel_d{d}": float(sens[d] / sens[0]) for d in REL_D},
            "d50_pooled": mass_distance(pooled, 0.5),
            "d90_pooled": mass_distance(pooled, 0.9),
            **{f"sens_d{d}": float(v) for d, v in enumerate(sens)},
            **{f"pooled_d{d}": float(v) for d, v in enumerate(pooled)}}


def load_data() -> np.ndarray:
    """Kernel sequences [N_SEQ, 251] drawn from the promoter holdout."""
    hold, _ = datasets.load("holdout_promoter")
    seqs = hold["test"]["seqs"]
    pick = np.sort(np.random.default_rng(DRAW_SEED).choice(
        len(seqs), size=N_SEQ, replace=False))
    toks = np.stack([encode(seqs[i]) for i in pick])
    if toks.max() > 3:
        raise RuntimeError("holdout sequences contain N")
    return toks


def encoder(refs, base: dict, cond: str, seed: int):
    if cond == "qssm_random_init":
        return encoders.matched_random_init(base, seed, "cpu", refs)[0]
    return encoders.load_encoder(encoders.pick_run(refs, cond, seed), "cpu")


def _order(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(_s=df["seed"].map(SEEDS.index),
                   _c=df["condition"].map(CONDS.index))
    by = ["_s", "_c"] + (["qubit"] if "qubit" in df else [])
    return df.sort_values(by, kind="stable").drop(columns=["_s", "_c"]) \
        .reset_index(drop=True)


def _write(rows: dict) -> None:
    RES.mkdir(parents=True, exist_ok=True)
    for name, path in FILES.items():
        tmp = path.with_name(path.name + ".partial")
        _order(pd.DataFrame(rows[name])).to_csv(tmp, index=False)
        tmp.replace(path)


def _resume() -> tuple[dict, set]:
    if not all(p.exists() for p in FILES.values()):
        return {name: [] for name in FILES}, set()
    dfs = {name: pd.read_csv(p, float_precision="round_trip")
           for name, p in FILES.items()}
    done = set.intersection(*[set(zip(d.condition, d.seed))
                              for d in dfs.values()])
    rows = {name: [r for r in d.to_dict("records")
                   if (r["condition"], r["seed"]) in done]
            for name, d in dfs.items()}
    return rows, done


def run() -> None:
    refs = encoders.scan_runs(jobs.ROOTS)
    base = load_config("configs/endpoint.yaml")
    toks = load_data()
    m, r0 = readout_matrix(4), initial_r(4)
    rows, done = _resume()
    print(f"{len(done)} cells already present", flush=True)
    for s in SEEDS:
        for c in CONDS:
            if (c, s) in done:
                continue
            enc = encoder(refs, base, c, s)
            damp = damping_rows(c, s, enc)
            a, cvec = affine(enc)
            sens, pooled, _ = kernel(a, cvec, m, r0, toks)
            krow = kernel_row(c, s, sens, pooled, len(toks))
            rows["damping"] += damp
            rows["kernel"].append(krow)
            _write(rows)
            print(c, s, "d50", krow["d50"], "d90", krow["d90"],
                  "gamma", [round(r["gamma_mean_acgt"], 4) for r in damp],
                  flush=True)
    summary()


def _mean_sd(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    g = df.groupby("condition")[cols]
    out = pd.concat({"mean": g.mean(), "sd": g.std(ddof=1)}, axis=1) \
        .swaplevel(axis=1)[cols]
    out.insert(0, "n", g.size())
    return out.loc[[c for c in CONDS if c in out.index]]


def summary() -> None:
    pd.set_option("display.width", 250)
    damp = pd.read_csv(FILES["damping"])
    df = pd.read_csv(FILES["kernel"])[
        ["condition", "seed", "d50", "d90", "d50_pooled", "d90_pooled"]
        + [f"rel_d{d}" for d in REL_D]]
    print("\n== mean and SD over seeds ==")
    print(_mean_sd(df, ["d50", "d90"]).round(5).to_string())
    print("\n== pooled-mean-square kernel and relative sensitivity ==")
    print(_mean_sd(df, ["d50_pooled", "d90_pooled"]
                   + [f"rel_d{d}" for d in REL_D[1:]]).round(4).to_string())
    print("\n== paired change from qssm_random_init ==")
    wide = {k: df.pivot(index="seed", columns="condition", values=k)
            for k in ("d50", "d90")}
    out = []
    for c in CONDS[1:]:
        for k, w in wide.items():
            if c not in w or "qssm_random_init" not in w:
                continue
            both = w[[c, "qssm_random_init"]].dropna()
            if len(both) < 2:
                continue
            res = stats.paired_summary(both[c].values,
                                       both["qssm_random_init"].values)
            out.append({"condition": c, "quantity": k,
                        "mean": float(both[c].mean()),
                        "mean_init": float(both["qssm_random_init"].mean()),
                        **{j: res[j] for j in ("n", "mean_diff", "sd_diff",
                                               "ci95_low", "ci95_high",
                                               "p_perm", "n_positive")}})
    print(pd.DataFrame(out).round(5).to_string(index=False))
    print("\n== damping rate per qubit, mean over A, C, G, T ==")
    g = damp.groupby(["condition", "qubit"])["gamma_mean_acgt"]
    tab = pd.concat({"mean": g.mean().unstack(),
                     "sd": g.std(ddof=1).unstack()}, axis=1)
    print(tab.loc[[c for c in CONDS if c in tab.index]].round(5).to_string())
    print("\n== spread of the damping rate over the four nucleotides "
          "(max / min, mean over seeds) ==")
    cols = [f"gamma_{x}" for x in "ACGT"]
    ratio = (damp[cols].max(1) / damp[cols].min(1)).groupby(
        [damp.condition, damp.qubit]).mean().unstack()
    print(ratio.loc[[c for c in CONDS if c in ratio.index]]
          .round(3).to_string())


@torch.no_grad()
def _forward_changes(enc, tt: torch.Tensor, mask: torch.Tensor, ref,
                     t: int) -> np.ndarray:
    out = []
    for y in range(4):
        t2 = tt.clone()
        t2[:, t] = y
        sub = rd.subset_readout(enc.forward_state(t2, mask), enc.q)
        out.append(sub.double().numpy() - ref)
    return np.stack(out)


def verify() -> None:
    refs = encoders.scan_runs(jobs.ROOTS)
    base = load_config("configs/endpoint.yaml")
    toks = load_data()
    m, r0 = readout_matrix(4), initial_r(4)
    tt = torch.from_numpy(toks)
    mask = torch.ones_like(tt, dtype=torch.bool)
    l = toks.shape[1]
    print("== kernel from the affine form vs the model's forward pass ==")
    worst = {"complex128": 0.0, "complex64": 0.0}
    for cond, seed in VERIFY:
        enc = encoder(refs, base, cond, seed)
        a, c = affine(enc)
        sens, _, kept = kernel(a, c, m, r0, toks, keep=VERIFY_D)
        native = encoder(refs, base, cond, seed)
        for name, model in (("complex128", enc), ("complex64", native)):
            with torch.no_grad():
                ref = rd.subset_readout(model.forward_state(tt, mask),
                                        model.q).double().numpy()
            for d in VERIFY_D:
                ch = _forward_changes(model, tt, mask, ref, l - 1 - d)
                direct = np.sqrt((ch ** 2).mean(-1)).sum(0).mean() / 3.0
                dk = abs(direct - sens[d])
                dc = float(np.abs(ch - kept[d]).max())
                worst[name] = max(worst[name], dk, dc)
                print(cond, seed, name, "d", d, "affine", f"{sens[d]:.10f}",
                      "forward", f"{direct:.10f}", "|diff|", f"{dk:.2e}",
                      "max |parity change diff|", f"{dc:.2e}", flush=True)
    for name, w in worst.items():
        print(f"maximum absolute difference ({name}): {w:.3e}",
              "PASS" if w < 1e-4 else "FAIL")


def main() -> None:
    torch.set_num_threads(THREADS)
    if os.name == "nt":
        k32 = ctypes.windll.kernel32
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x4000)
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    {"run": run, "verify": verify, "summary": summary}[mode]()


if __name__ == "__main__":
    main()
