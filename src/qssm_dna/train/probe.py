"""Linear probes on frozen features and the promoter-benchmark splits.

The promoter benchmark keeps its official test split; a canonical-hash
grouped part of the official train split is used for validation, and
thresholds are chosen on validation only.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.tokenizer import canonical_hash, encode, pad_tokens
from ..eval.metrics import pick_threshold
from ..models.heads import LinearHead


class EncoderClassifier(nn.Module):
    """Any encoder with .encode(tokens,mask,...) -> [B,proj_dim] plus a
    linear head."""

    def __init__(self, encoder: nn.Module, n_classes: int):
        super().__init__()
        self.encoder = encoder
        self.head = LinearHead(encoder.proj_dim, n_classes)

    def forward(self, tokens, mask, rc_average=False, shots=None, seed=None):
        z = self.encoder.encode(tokens, mask, representation="projected",
                                rc_average=rc_average, shots=shots,
                                seed=seed)
        return self.head(z)

    def forward_positions(self, tokens, mask):
        """[B,L,C] per-position logits for dense next-base supervision
        (objective classical_sim_dense_supervision)."""
        return self.head(self.encoder.encode_positions(tokens, mask))


class _GraphedForward:
    """Forward-only CUDA graph replay for a fixed-shape callable.

    The token loop is launch-bound (about 600 small kernels per sequence);
    a captured graph replays the whole encode as one launch. Outputs are
    static buffers, overwritten by the next call with the same input
    shape. A pure forward can be captured at any point.
    """

    def __init__(self):
        self.cache: dict = {}
        self._stream = None

    def __call__(self, fn, *inputs: torch.Tensor) -> torch.Tensor:
        key = tuple(tuple(x.shape) for x in inputs)
        ent = self.cache.get(key)
        if ent is None:
            static = [x.clone() for x in inputs]
            try:
                if self._stream is None:
                    self._stream = torch.cuda.Stream()
                s = self._stream
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(2):
                        fn(*static)
                torch.cuda.current_stream().wait_stream(s)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    s_out = fn(*static)
                self.cache[key] = (g, static, s_out)
            except Exception:
                # e.g. cuDNN pack_padded_sequence host sync -- mark key
                # eager and run plainly
                self.cache[key] = None
                torch.cuda.synchronize()
            ent = self.cache[key]
        if ent is None:
            return fn(*inputs)
        g, static, s_out = ent
        for st, x in zip(static, inputs):
            st.copy_(x)
        g.replay()
        return s_out


def train_probe(z_train: np.ndarray, y_train: np.ndarray,
                z_val: np.ndarray, y_val: np.ndarray, seed: int,
                epochs: int = 200, lr: float = 1e-2) -> dict:
    """Linear head on frozen features. Threshold picked on validation.

    Runs on CPU deliberately: tiny model, and keeping every CUDA-side
    backward inside graph warmup/capture avoids the "legacy stream
    depends on capturing stream" failure mode of CUDA graph capture."""
    torch.manual_seed(seed)
    dev = "cpu"
    head = LinearHead(z_train.shape[1], 1).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    xt = torch.tensor(z_train, dtype=torch.float32, device=dev)
    yt = torch.tensor(y_train, dtype=torch.float32, device=dev)
    xv = torch.tensor(z_val, dtype=torch.float32, device=dev)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(
            head(xt)[:, 0], yt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pv = torch.sigmoid(head(xv)[:, 0]).cpu().numpy()
    thr = pick_threshold(y_val, pv)
    return {"head": head, "val_prob": pv, "threshold": thr}


def arrays_from_seqs(seqs: list[str], labels: list[int]):
    toks = [encode(s) for s in seqs]
    t, m = pad_tokens(toks)
    return t, m, np.asarray(labels, dtype=np.int64)


def train_val_split(rows: list[tuple[str, int]], val_frac: float,
                    seed: int):
    """Canonical-hash grouped split of the official train split."""
    keys = [canonical_hash(s) for s, _ in rows]
    group_of = {}
    for h in keys:
        if h not in group_of:
            group_of[h] = len(group_of)
    gid = np.array([group_of[h] for h in keys])
    from ..data.splits import grouped_split
    sp = grouped_split(gid, {"train": 1.0 - val_frac, "val": val_frac},
                       seed)
    return sp["train"], sp["val"]


def load_promoter_benchmark(cfg) -> dict:
    """Token arrays and sequences of the train/val/test splits of
    data/benchmarks.npz (validation fraction and split seed from
    cfg["data"]["benchmark"])."""
    npz = np.load("data/benchmarks.npz", allow_pickle=False)
    rows = list(zip(npz["train_seq"].tolist(), npz["train_y"].tolist()))
    bcfg = cfg["data"]["benchmark"]
    tr_idx, va_idx = train_val_split(
        rows, float(bcfg["validation_fraction_of_original_train"]),
        int(bcfg["split_seed"]))
    rows_tr = [rows[i] for i in tr_idx]
    rows_va = [rows[i] for i in va_idx]
    rows_te = list(zip(npz["test_seq"].tolist(), npz["test_y"].tolist()))
    out = {}
    for name, rws in (("train", rows_tr), ("val", rows_va),
                      ("test", rows_te)):
        t, m, y = arrays_from_seqs([s for s, _ in rws],
                                   [y for _, y in rws])
        out[f"{name}_tokens"], out[f"{name}_mask"], out[f"{name}_labels"] = \
            t, m, y
    out["seqs"] = {"train": [s for s, _ in rows_tr],
                   "val": [s for s, _ in rows_va],
                   "test": [s for s, _ in rows_te]}
    return out
