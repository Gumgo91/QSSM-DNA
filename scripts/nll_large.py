"""Next-base loss of every pretrained encoder on a large held-out set (CPU).

  CUDA_VISIBLE_DEVICES=-1 python scripts/nll_large.py

Inputs: the pretraining runs under the run roots (jobs.ROOTS; checkpoint.pt,
config and metrics.json), data/fasta/<chromosome>.fa.gz of the training and
validation chromosomes of configs/endpoint.yaml.

The validation NLL logged by cli.cmd_pretrain is the terminal-position
next-base loss on 128 windows of chr7/chr8 drawn by fasta.WindowSampler and
WindowSource, both seeded with 1234 (64- and 128-bp windows with equal
probability; 256-bp windows for the 256-bp runs). This script continues that
stream: draws 1-128 are the logged set, used only to check that the logged
value is reproduced; draws 129-10,368 (N_WINDOWS = 10,240) are the evaluation
set. Every checkpoint of the ten replication seeds gets, on the evaluation
set,
  nll_terminal  the logged definition (loss at the last position), and
  nll_allpos    the loss pooled over all positions whose next base is A/C/G/T.
All conditions are scored on the 64/128-bp set; the 256-bp condition is also
scored on the 256-bp set that its own logged value refers to.

A GRU is evaluated with its own forward_positions. A QSSM applies one of five
fixed channels per token, so its state is propagated as a Pauli vector with
the transfer matrices of theory.transfer_matrices (float64) and read through
the checkpoint's projection and head; route_maxdiff is the largest
per-position difference from model.forward_positions on the logged windows.

Count references, scored on the same targets with add-0.5 counts:
  uniform, unigram, markov1..markov6   estimated on the forward strand of the
      whole training chromosome (chr22);
  unigram_windows, markov1_windows, markov2_windows   estimated on the
      terminal targets of the training windows of configs/endpoint.yaml
      (seed 11, 250,000 input tokens), replayed by
      train.pretrain.replay_train_windows.
A reference sees what the encoder sees: its context is the last k bases
inside the window, backed off to the longest N-free suffix available.

se_terminal and se_allpos are standard errors over windows (windows as
clusters). In the contrast table the interval and the p-values are over
seeds, and se_windows is the standard error over windows of the seed-mean
difference.

Outputs: results/nll_large.csv (one row per checkpoint and window
set) and nll_large_contrasts.csv (paired contrasts over seeds, both losses,
64/128-bp set). The per-window losses of each checkpoint are kept in
runs/nll_large; a checkpoint whose file exists is not re-evaluated.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "2"
if os.name == "nt":
    ctypes.windll.kernel32.SetPriorityClass(
        ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.set_num_threads(2)
sys.path.insert(0, "src")
from qssm_dna.config import load_config  # noqa: E402
from qssm_dna.data import fasta  # noqa: E402
from qssm_dna.data.tokenizer import encode  # noqa: E402
from qssm_dna.models.encoder import QSSMModel  # noqa: E402
from qssm_dna.experiments import encoders, jobs, stats, theory  # noqa: E402
from qssm_dna.train.probe import EncoderClassifier  # noqa: E402
from qssm_dna.train.pretrain import (WindowSource,  # noqa: E402
                                     replay_train_windows)

OUT = Path("results")
CACHE = Path("runs/nll_large")
SEEDS = stats.SEEDS
VAL_SEED = 1234
N_LOGGED = 128
N_WINDOWS = 10240
MAX_ORDER = 6
B10 = "_b2500000"
QSSM = ["qssm_endpoint", "qssm_dense2", "qssm_dense4", "qssm_dense8",
        "qssm_spread8", "qssm_allpos", "qssm_dense8_zerogamma",
        "qssm_dense8_staticgamma", "qssm_dense8_norzz", "qssm_allpos_norzz",
        "qssm_endpoint" + B10, "qssm_dense8" + B10, "qssm_allpos" + B10,
        "qssm_dense8_w256" + B10]
GRU = ["gru8_dense8", "gru8_allpos", "gru64_endpoint", "gru64_dense8",
       "gru64_allpos", "gru64_dense8" + B10]
COUNT = {"uniform": None, "unigram": ("chromosome", 0),
         **{f"markov{k}": ("chromosome", k) for k in range(1, MAX_ORDER + 1)},
         "unigram_windows": ("windows", 0), "markov1_windows": ("windows", 1),
         "markov2_windows": ("windows", 2)}
PAIRS = [(c, "qssm_endpoint") for c in
         ("qssm_dense2", "qssm_dense4", "qssm_dense8", "qssm_spread8",
          "qssm_allpos")] + [
    ("qssm_endpoint" + B10, "qssm_endpoint"),
    ("qssm_dense8" + B10, "qssm_endpoint" + B10),
    ("qssm_allpos" + B10, "qssm_endpoint" + B10),
    ("qssm_dense4", "qssm_endpoint" + B10),
    ("qssm_allpos", "qssm_endpoint" + B10),
    ("qssm_dense8_zerogamma", "qssm_dense8"),
    ("qssm_allpos_norzz", "qssm_allpos"),
    ("gru8_allpos", "qssm_allpos"),
    ("gru64_allpos", "gru64_endpoint")] + [
    (c, m) for m in ("markov1", "markov1_windows") for c in QSSM] + [
    ("qssm_allpos" + B10, f"markov{k}") for k in range(2, MAX_ORDER + 1)]


def chrom_tokens(names: list[str]) -> dict[str, np.ndarray]:
    out = {}
    for ch in names:
        seq = fasta.read_fasta_sequences(Path(f"data/fasta/{ch}.fa.gz"))[ch]
        out[ch] = encode(seq).astype(np.uint8)
    return out


def window_params(cfg: dict) -> tuple:
    g, t = cfg["data"]["genome"], cfg["training"]
    return (tuple(t["sequence_lengths"]),
            tuple(t.get("length_probabilities") or ()),
            float(g["max_input_n_fraction"]),
            tuple(g["validation_chromosomes"]))


def set_name(cfg: dict) -> str:
    return "L" + "_".join(str(l) for l in cfg["training"]["sequence_lengths"])


def draw(tokens: dict[str, np.ndarray], params: tuple) -> tuple[dict, dict]:
    """(logged windows, evaluation windows) of the validation stream, each
    with tokens t, mask m, terminal target y, per-position target yd and the
    mask sup of positions whose target is a base."""
    lengths, probs, max_n, chroms = params
    sampler = fasta.WindowSampler(
        tokens, list(lengths), list(probs) or None, max_n,
        np.random.default_rng(VAL_SEED), train_chroms=list(chroms))
    src = WindowSource(sampler, VAL_SEED, placement="all")
    return tuple(dict(zip(("t", "m", "y", "yd", "sup"), src.next_batch(n)))
                 for n in (N_LOGGED, N_WINDOWS))


def chromosome_tables(seqs: list[np.ndarray], max_order: int,
                      chunk: int = 1 << 23) -> list[np.ndarray]:
    """P(next base | last k bases) for k = 0..max_order from every position
    of the sequences; the context index has the most recent base as its
    lowest base-4 digit."""
    tables = []
    for k in range(max_order + 1):
        counts = np.zeros(4 ** (k + 1), np.int64)
        for seq in seqs:
            for s in range(0, len(seq) - k, chunk):
                w = seq[s:s + chunk + k].astype(np.int64)
                n = len(w) - k
                code, ok = np.zeros(n, np.int64), np.ones(n, bool)
                for j in range(k + 1):
                    ok &= w[j:j + n] < 4
                    code = code * 4 + w[j:j + n]
                counts += np.bincount(code[ok], minlength=len(counts))
        c = counts.reshape(-1, 4) + 0.5
        tables.append(c / c.sum(1, keepdims=True))
    return tables


def window_tables(cfg: dict, tokens: dict[str, np.ndarray],
                  max_order: int) -> list[np.ndarray]:
    """The same tables from the terminal targets of the replayed training
    windows of one run (add-0.5 counts; a context with N is skipped)."""
    windows = replay_train_windows(cfg)[0]
    tables = []
    for k in range(max_order + 1):
        c = np.full((4 ** k, 4), 0.5)
        for ch, s, l, tgt in windows:
            ctx = tokens[ch][s + l - k:s + l]
            if (ctx < 4).all():
                c[sum(int(b) * 4 ** j for j, b in enumerate(ctx[::-1])),
                  tgt] += 1
        tables.append(c / c.sum(1, keepdims=True))
    return tables


def markov_nll(tables: list[np.ndarray], order: int, w: dict) -> np.ndarray:
    """[n, L] loss of the order-k reference at every position."""
    tok, yd = w["t"].numpy(), w["yd"].numpy()
    n, length = tok.shape
    nll = np.zeros((n, length))
    done = np.zeros((n, length), bool)
    for k in range(order, -1, -1):
        code = np.zeros((n, length), np.int64)
        ok = np.ones((n, length), bool)
        ok[:, :max(k - 1, 0)] = False
        for j in range(k):
            prev = tok[:, :length - j]
            ok[:, j:] &= prev < 4
            code[:, j:] += np.where(prev < 4, prev, 0) * 4 ** j
        use = ok & ~done
        nll[use] = -np.log(tables[k][code[use], yd[use]])
        done |= use
    return nll


def count_nll(tables: dict, name: str, w: dict) -> np.ndarray:
    if COUNT[name] is None:
        return np.full(tuple(w["t"].shape), np.log(4.0))
    source, order = COUNT[name]
    return markov_nll(tables[source], order, w)


def window_losses(nll: np.ndarray, w: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per window: loss at the last position, loss summed over the
    positions whose target is a base."""
    last = w["m"].sum(1).numpy() - 1
    return nll[np.arange(len(nll)), last], (nll * w["sup"].numpy()).sum(1)


def ratio_se(tot: np.ndarray, cnt: np.ndarray) -> float:
    r = tot - tot.sum() / cnt.sum() * cnt
    n = len(tot)
    return float(np.sqrt(n / (n - 1) * (r ** 2).sum()) / cnt.sum())


def summarise(term: np.ndarray, tot: np.ndarray, cnt: np.ndarray) -> dict:
    return {"nll_terminal": float(term.mean()),
            "se_terminal": float(term.std(ddof=1) / np.sqrt(len(term))),
            "nll_allpos": float(tot.sum() / cnt.sum()),
            "se_allpos": ratio_se(tot, cnt)}


def load_model(ref: encoders.RunRef) -> torch.nn.Module:
    enc = encoders.load_encoder(ref, "cpu")
    model = QSSMModel(enc, 4) if ref.family == "qssm" \
        else EncoderClassifier(enc, 4)
    payload = torch.load(ref.run_dir / "checkpoint.pt", map_location="cpu",
                         weights_only=False)
    model.load_state_dict(payload["model"])
    return model.eval()


def position_nll(logits: torch.Tensor, w: dict) -> np.ndarray:
    return -F.log_softmax(logits.double(), -1).gather(
        -1, w["yd"].unsqueeze(-1)).squeeze(-1).numpy()


def model_route(model: torch.nn.Module, w: dict,
                batch: int = 1024) -> np.ndarray:
    return position_nll(torch.cat(
        [model.forward_positions(w["t"][s:s + batch], w["m"][s:s + batch])
         for s in range(0, len(w["t"]), batch)]), w)


def readout_index(q: int) -> list[int]:
    """Positions of Z_i, X_i and Z_i Z_(i+1) in theory.pauli_basis (digits
    I, X, Y, Z; wire 0 is the most significant digit)."""
    w = [4 ** (q - 1 - i) for i in range(q)]
    return [3 * x for x in w] + w + \
        [3 * (w[i] + w[i + 1]) for i in range(q - 1)]


def pauli_route(model: QSSMModel):
    """Per-position loss of a QSSM from its five Pauli transfer matrices.
    Converts the model to float64."""
    enc = model.encoder
    mats = theory.transfer_matrices(enc)
    mats = [torch.from_numpy(mats[x].T.copy()) for x in range(5)]
    model.double()
    r0 = theory.pauli_basis(enc.q)[:, 0, 0].real
    idx = readout_index(enc.q)

    def run(w: dict, block: int = 2048) -> np.ndarray:
        out = []
        for s in range(0, len(w["t"]), block):
            tok = w["t"][s:s + block]
            r = r0.repeat(len(tok), 1)
            z = torch.empty(*tok.shape, len(idx), dtype=torch.float64)
            for t in range(tok.shape[1]):
                for x, mat in enumerate(mats):
                    rows = torch.nonzero(tok[:, t] == x).squeeze(1)
                    if len(rows):
                        r[rows] = r[rows] @ mat
                z[:, t] = r[:, idx]
            out.append(model.head(F.gelu(enc.proj(z))))
        return position_nll(torch.cat(out), w)
    return run


def logged_definition(model: torch.nn.Module, w: dict) -> float:
    """The validation loss exactly as cli.cmd_pretrain computes it."""
    return float(np.mean([
        float(F.cross_entropy(model(w["t"][i:i + 16], w["m"][i:i + 16]),
                              w["y"][i:i + 16]))
        for i in range(0, N_LOGGED, 16)]))


@torch.no_grad()
def evaluate(ref: encoders.RunRef, sets: dict, main: str) -> dict:
    """Per-window losses on the evaluation windows and the checks on the
    logged windows, for the main window set and the run's own."""
    names = list(dict.fromkeys([main, set_name(ref.cfg)]))
    model = load_model(ref)
    val128 = {k: logged_definition(model, sets[k][0]) for k in names}
    if ref.family == "qssm":
        check = {k: model_route(model, sets[k][0]) for k in names}
        route = pauli_route(model)
    else:
        check = {}
        route = lambda w: model_route(model, w)  # noqa: E731
    out = {}
    for k in names:
        logged, large = sets[k]
        a = route(logged)
        small = summarise(*window_losses(a, logged),
                          logged["sup"].sum(1).numpy())
        out[f"{k}__terminal"], out[f"{k}__allpos_sum"] = window_losses(
            route(large), large)
        out[f"{k}__val_nll_128"] = val128[k]
        out[f"{k}__terminal_128"] = small["nll_terminal"]
        out[f"{k}__allpos_128"] = small["nll_allpos"]
        out[f"{k}__route_maxdiff"] = float(np.abs(a - check[k])[
            logged["sup"].numpy()].max()) if k in check else np.nan
    return out


def contrasts(win: dict, cnt: np.ndarray, main: str) -> pd.DataFrame:
    """win[condition, seed] = (terminal loss, all-position loss sum) per
    evaluation window of the main set."""
    def level(v: np.ndarray, j: int) -> float:
        return float(v.mean() if j == 0 else v.sum() / cnt.sum())

    rows = []
    for j, loss in enumerate(("terminal", "allpos")):
        for a, b in PAIRS:
            common = [s for s in SEEDS if (a, s) in win and (b, s) in win]
            if len(common) < 2:
                continue
            x = np.array([level(win[a, s][j], j) for s in common])
            y = np.array([level(win[b, s][j], j) for s in common])
            d = np.mean([win[a, s][j] - win[b, s][j] for s in common], 0)
            rows.append({
                "loss": loss, "window_set": main,
                "contrast": f"{a}_minus_{b}", "first": a, "second": b,
                "mean_first": float(x.mean()), "mean_second": float(y.mean()),
                "seeds": ",".join(map(str, common)),
                **stats.paired_summary(x, y),
                "se_windows": float(d.std(ddof=1) / np.sqrt(len(d)))
                if j == 0 else ratio_se(d, cnt)})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    refs = encoders.scan_runs(jobs.ROOTS)
    todo = [encoders.pick_run(refs, c, s) for c in QSSM + GRU for s in SEEDS]
    base = load_config("configs/endpoint.yaml")
    genome = base["data"]["genome"]
    train = genome["train_chromosomes"]
    tokens = chrom_tokens(train + genome["validation_chromosomes"])
    tables = {"chromosome": chromosome_tables([tokens[c] for c in train],
                                              MAX_ORDER),
              "windows": window_tables(base, tokens, 2)}
    param_sets = {window_params(c): set_name(c)
                  for c in [base] + [r.cfg for r in todo]}
    if len(set(param_sets.values())) != len(param_sets):
        raise SystemExit("two window settings share one name")
    main_set = set_name(base)
    sets, cnt, win, rows = {}, {}, {}, []
    for params, name in param_sets.items():
        sets[name] = draw(tokens, params)
        large = sets[name][1]
        cnt[name] = large["sup"].sum(1).numpy()
        for c in COUNT:
            w = window_losses(count_nll(tables, c, large), large)
            rows.append({"condition": c, "seed": pd.NA, "family": "count",
                         "window_set": name, "n_windows": N_WINDOWS,
                         "n_targets": int(cnt[name].sum()),
                         **summarise(*w, cnt[name])})
            if name == main_set:
                win.update({(c, s): w for s in SEEDS})
            print(name, c, round(rows[-1]["nll_terminal"], 5),
                  round(rows[-1]["nll_allpos"], 5), flush=True)
    del tokens
    for ref in todo:
        f = CACHE / (f"{ref.condition}_s{ref.seed}_v{VAL_SEED}"
                     f"_n{N_WINDOWS}.npz")
        if not f.exists():
            part = f.with_name(f.stem + ".part.npz")
            np.savez(part, **evaluate(ref, sets, main_set))
            part.replace(f)
        z = np.load(f)
        for name in dict.fromkeys([main_set, set_name(ref.cfg)]):
            w = (z[f"{name}__terminal"], z[f"{name}__allpos_sum"])
            rows.append({
                "condition": ref.condition, "seed": ref.seed,
                "family": ref.family, "window_set": name,
                "n_windows": N_WINDOWS, "n_targets": int(cnt[name].sum()),
                **summarise(*w, cnt[name]),
                "val_nll_logged": ref.metrics.get("val_nll")
                if name == set_name(ref.cfg) else np.nan,
                **{k: float(z[f"{name}__{k}"]) for k in
                   ("val_nll_128", "terminal_128", "allpos_128",
                    "route_maxdiff")},
                "run": ref.run_dir.name})
            if name == main_set:
                win[ref.condition, ref.seed] = w
            r = rows[-1]
            print(r["condition"], r["seed"], name,
                  round(r["nll_terminal"], 5), round(r["nll_allpos"], 5),
                  "logged", r["val_nll_logged"], "recomputed",
                  r["val_nll_128"], "route", r["route_maxdiff"], flush=True)
    df = pd.DataFrame(rows)
    df["seed"] = df["seed"].astype("Int64")
    df.to_csv(OUT / "nll_large.csv", index=False)
    contrasts(win, cnt[main_set], main_set).to_csv(
        OUT / "nll_large_contrasts.csv", index=False)
    print(f"{len(df)} rows -> {OUT / 'nll_large.csv'}")


if __name__ == "__main__":
    main()
