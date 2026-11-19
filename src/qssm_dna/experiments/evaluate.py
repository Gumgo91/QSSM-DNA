"""Frozen-feature extraction and linear-probe evaluation.

Feature cache (one npz per encoder x dataset x split):
  QSSM: subset30 for each sequence direction (fwd, rc), and pauli255 of the
        RC-averaged state where requested. raw11 is a subset of subset30
        (Z_i, X_i, Z_i Z_{i+1}); projected64 is GELU(W raw11 + b) with the
        encoder's pretrained projection; reverse-complement averaging is the
        mean of the two directions (all readouts are linear in rho). The 30
        parities also determine both outcome distributions exactly, so
        finite-shot sampling can be done from the cache.
  GRU:  projected64 (RC-averaged after projection, as in GRUEncoder.encode)
        and the raw terminal hidden state for each direction.

Probe protocol: LinearHead, AdamW lr 1e-2, weight decay 1e-4, 200
full-batch epochs on CPU, probe seed 11, no feature standardisation,
threshold = argmax validation MCC over 97 values in [0.02, 0.98]
(prediction = prob >= threshold).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import torch

from ..data.tokenizer import encode, pad_tokens
from ..eval.metrics import binary_metrics
from ..quantum import readout as rd

PROBE_SEED = 11
CACHE_ROOT = Path(os.environ.get("QSSM_DNA_FEATURES", "runs/features"))


# ------------------------------------------------------------ batching

def _batches(seqs: list[str], batch_size: int, bucket: int = 32):
    """Length-sorted batches padded to a multiple of `bucket` (PAD never
    updates the state, so padding length does not change any output)."""
    order = np.argsort([len(s) for s in seqs], kind="stable")
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        toks = [encode(seqs[i]) for i in idx]
        lmax = max(len(t) for t in toks)
        lpad = int(np.ceil(lmax / bucket) * bucket)
        t, m = pad_tokens(toks, max_len=lpad)
        yield idx, t, m


# ------------------------------------------------------------ QSSM

def raw11_index(q: int = 4) -> list[int]:
    """Positions of (Z_i, X_i, Z_i Z_{i+1}) inside the subset30 vector."""
    half = (1 << q) - 1
    z = [(1 << i) - 1 for i in range(q)]
    x = [half + (1 << i) - 1 for i in range(q)]
    zz = [((1 << i) | (1 << (i + 1))) - 1 for i in range(q - 1)]
    return z + x + zz


@torch.no_grad()
def encode_qssm(enc, seqs: list[str], device: str, batch_size: int = 1024,
                pauli: bool = False) -> dict:
    from ..models.encoder import rc_tokens
    from ..train.probe import _GraphedForward
    q = enc.q
    # keep one state tensor near 2^25 complex entries for larger registers
    batch_size = max(16, min(batch_size, (1 << 25) // (4 ** q)))
    n = len(seqs)
    sub_f = np.zeros((n, 2 * ((1 << q) - 1)), np.float32)
    sub_r = np.zeros_like(sub_f)
    pau = np.zeros((n, 4 ** q - 1), np.float32) if pauli else None
    # CUDA graphs only for fixed-length sets: every new padded shape would
    # capture (and keep) another graph, which exhausts memory on the
    # variable-length OCR task; eager execution gives identical states
    fixed = len({len(s) for s in seqs}) == 1
    fwd = _GraphedForward() if device == "cuda" and fixed else None
    verified = fwd is None
    for idx, t, m in _batches(seqs, batch_size):
        t = torch.from_numpy(t).to(device)
        m = torch.from_numpy(m).to(device)
        tr = rc_tokens(t, m)
        if fwd is not None:
            rf = fwd(enc.forward_state, t, m).clone()
            rr = fwd(enc.forward_state, tr, m).clone()
            if not verified:
                md = float((rf - enc.forward_state(t, m)).abs().max())
                if md > 1e-5:
                    raise RuntimeError(f"graphed forward mismatch {md}")
                verified = True
        else:
            rf, rr = enc.forward_state(t, m), enc.forward_state(tr, m)
        sub_f[idx] = rd.subset_readout(rf, q).float().cpu().numpy()
        sub_r[idx] = rd.subset_readout(rr, q).float().cpu().numpy()
        if pauli:
            pau[idx] = rd.pauli_features(0.5 * (rf + rr), q).float() \
                .cpu().numpy()
    out = {"subset30_fwd": sub_f, "subset30_rc": sub_r}
    if pauli:
        out["pauli255"] = pau
    return out


def readout_names(q: int) -> tuple[str, str]:
    """('raw<3q-1>', 'subset<2(2^q-1)>'): raw11/subset30 at q = 4."""
    return f"raw{3 * q - 1}", f"subset{2 * ((1 << q) - 1)}"


def qssm_views(cache: dict, enc=None) -> dict[str, np.ndarray]:
    sub = 0.5 * (cache["subset30_fwd"] + cache["subset30_rc"])
    q = int(round(np.log2(sub.shape[1] // 2 + 1)))
    raw_name, sub_name = readout_names(q)
    views = {sub_name: sub, raw_name: sub[:, raw11_index(q)]}
    if enc is not None:
        w = enc.proj.weight.detach().cpu().double().numpy()
        b = enc.proj.bias.detach().cpu().double().numpy()
        z = views[raw_name].astype(np.float64) @ w.T + b
        views["projected64"] = torch.nn.functional.gelu(
            torch.from_numpy(z)).numpy().astype(np.float32)
    if "pauli255" in cache:
        views["pauli255"] = cache["pauli255"]
    return views


def probs_from_subset(sub30: np.ndarray, q: int = 4):
    """Exact Z- and X-basis outcome distributions from the 2(2^q-1)
    parities of one direction (inverse Walsh-Hadamard transform)."""
    s = rd._subset_sign_table(q).numpy()                      # [15,16]
    half = (1 << q) - 1
    pz = (1.0 + sub30[:, :half].astype(np.float64) @ s) / (1 << q)
    px = (1.0 + sub30[:, half:].astype(np.float64) @ s) / (1 << q)
    return np.clip(pz, 0, None), np.clip(px, 0, None)


# ------------------------------------------------------------ GRU

def _gru_hidden(enc, t: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    lengths = m.sum(1).clamp_min(1)
    x = torch.nn.utils.rnn.pack_padded_sequence(
        enc.emb(t.clamp_max(4)), lengths.cpu(), batch_first=True,
        enforce_sorted=False)
    _, h = enc.gru(x)
    return h[-1]


@torch.no_grad()
def encode_gru(enc, seqs: list[str], device: str,
               batch_size: int = 1024) -> dict:
    from ..models.encoder import rc_tokens
    n = len(seqs)
    hdim = enc.gru.hidden_size
    hf = np.zeros((n, hdim), np.float32)
    hr = np.zeros_like(hf)
    proj = np.zeros((n, enc.proj_dim), np.float32)
    for idx, t, m in _batches(seqs, batch_size):
        t = torch.from_numpy(t).to(device)
        m = torch.from_numpy(m).to(device)
        a = _gru_hidden(enc, t, m)
        b = _gru_hidden(enc, rc_tokens(t, m), m)
        hf[idx] = a.cpu().numpy()
        hr[idx] = b.cpu().numpy()
        za = torch.nn.functional.gelu(enc.proj(a))
        zb = torch.nn.functional.gelu(enc.proj(b))
        proj[idx] = (0.5 * (za + zb)).cpu().numpy()
    return {"hidden_fwd": hf, "hidden_rc": hr, "projected64": proj}


def gru_views(cache: dict) -> dict[str, np.ndarray]:
    return {"projected64": cache["projected64"],
            "hidden": 0.5 * (cache["hidden_fwd"] + cache["hidden_rc"])}


# ------------------------------------------------------------ cache

def cached_features(key: str, family: str, enc, splits: dict, device: str,
                    pauli: bool = False) -> dict:
    """{split: cache-dict}; computed once per (encoder key, dataset)."""
    out = {}
    for split, d in splits.items():
        p = CACHE_ROOT / key / f"{split}.npz"
        if p.exists():
            z = np.load(p)
            if not pauli or "pauli255" in z.files:
                out[split] = {k: z[k] for k in z.files}
                continue
        p.parent.mkdir(parents=True, exist_ok=True)
        if family == "qssm":
            c = encode_qssm(enc, d["seqs"], device, pauli=pauli)
        else:
            c = encode_gru(enc, d["seqs"], device)
        tmp = p.with_name(p.stem + ".partial.npz")
        np.savez_compressed(tmp, **c)
        tmp.replace(p)          # atomic rename: no truncated cache file
        out[split] = c
    return out


# ------------------------------------------------------------ probes

def probe_sha(head) -> str:
    buf = io.BytesIO()
    torch.save(head.state_dict(), buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def fit_probe(z_tr, y_tr, z_va, y_va):
    from ..train.probe import train_probe
    return train_probe(z_tr, y_tr, z_va, y_va, PROBE_SEED)


def predict(head, z: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return torch.sigmoid(head(torch.tensor(z, dtype=torch.float32))
                             [:, 0]).numpy()


def evaluate_views(views: dict[str, dict[str, np.ndarray]],
                   labels: dict[str, np.ndarray],
                   extra_eval: dict[str, tuple[dict, np.ndarray]] | None
                   = None) -> tuple[list[dict], list[dict], dict]:
    """views[readout][split]; fits on train, thresholds on val, scores
    test and every extra (eval-only) set with the same frozen probe.
    Returns (metric rows, prediction rows, fitted heads)."""
    rows, preds, heads = [], [], {}
    for readout, v in views.items():
        pr = fit_probe(v["train"], labels["train"], v["val"], labels["val"])
        thr = pr["threshold"]
        heads[readout] = (pr["head"], thr)
        base = {"readout": readout, "selected_threshold": thr,
                "threshold_rule": "argmax val MCC over linspace(0.02,0.98,"
                                  "97); first max kept; predict prob>=thr",
                "probe_sha256": probe_sha(pr["head"])[:16],
                "val_mcc": binary_metrics(labels["val"], pr["val_prob"],
                                          thr)["mcc"]}
        targets = {"test": (v["test"], labels["test"])} \
            if "test" in v else {}
        for name, (vv, yy) in (extra_eval or {}).items():
            targets[name] = (vv[readout], yy)
        for name, (z, y) in targets.items():
            p = predict(pr["head"], z)
            met = binary_metrics(y, p, thr)
            rows.append({**base, "eval_set": name,
                         **{f"eval_{k}": val for k, val in met.items()}})
            preds.append({"readout": readout, "eval_set": name,
                          "prob": p.astype(np.float32),
                          "label": np.asarray(y, np.int8),
                          "threshold": thr})
    return rows, preds, heads


def save_predictions(path: Path, meta: dict, preds: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrs = {}
    for p in preds:
        k = f"{p['eval_set']}__{p['readout']}"
        arrs[f"{k}__prob"] = p["prob"]
        arrs[f"{k}__label"] = p["label"]
        arrs[f"{k}__threshold"] = np.float64(p["threshold"])
    np.savez_compressed(path, meta=json.dumps(meta), **arrs)
