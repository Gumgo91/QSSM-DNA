"""Selective quantum state-space encoder (density-matrix recurrence).

Per-token update on the state rho [B,D,D], D = 2**q, starting from
|0...0><0...0|:

    1. controller (one-hot of the current token) -> theta_t, gamma_t
    2. per-wire amplitude damping D_{gamma_{t,i}}
    3. cell_depth x [RY and RZ on every wire, adjacent RZZ (learned,
       shared over time)]
    4. the state carries to the next token

PAD positions leave the state unchanged. The readout is taken once, at the
last valid position; the controller sees only the input tokens, never the
state or its expectations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..quantum import channels, density, gates, readout
from .controller import TokenController
from .heads import LinearHead

VOCAB = ("A", "C", "G", "T", "N", "PAD")
PAD_IDX = 5


_COMP_CACHE: dict = {}


def rc_tokens(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Reverse-complement valid tokens of each row; PADs stay at the end.

    tokens: [B,L] long over VOCAB indices; mask: [B,L] True for valid.
    Complement map A<->T, C<->G, N->N. Vectorized (no host<->device sync)
    so it is legal inside CUDA graph capture.
    """
    comp = _COMP_CACHE.get(tokens.device)
    if comp is None:
        comp = torch.tensor([3, 2, 1, 0, 4, 5], device=tokens.device)
        _COMP_CACHE[tokens.device] = comp
    ctok = comp[tokens]                       # complement of every position
    n = mask.long().sum(1, keepdim=True)      # [B,1] valid counts
    pos = torch.arange(tokens.shape[1], device=tokens.device)
    src = (n - 1 - pos.unsqueeze(0)).clamp(min=0)
    rev = torch.gather(ctok, 1, src)          # ctok[n-1-p] for p<n
    pad = torch.full_like(tokens, PAD_IDX)
    return torch.where(pos.unsqueeze(0) < n, rev, pad)


class _StepModule(nn.Module):
    """Parameter-free token step (learned tensors arrive as inputs)."""

    def __init__(self, q: int, cell_depth: int):
        super().__init__()
        self.q = q
        self.cell_depth = cell_depth

    def forward(self, rho_ax: torch.Tensor, theta_t: torch.Tensor,
                gamma_t: torch.Tensor, valid: torch.Tensor,
                rzz_angles: torch.Tensor) -> torch.Tensor:
        """rho_ax [B,2]*2q; theta_t [B,depth,q,2]; gamma_t [B,q];
        valid [B] bool; rzz_angles [depth, q-1]."""
        q = self.q
        new = rho_ax
        for i in range(q):
            new = channels.amplitude_damping(new, gamma_t[:, i], i, q)
        for layer in range(self.cell_depth):
            for i in range(q):
                new = density.apply_1q_unitary(
                    new, gates.ryrz(theta_t[:, layer, i, 0],
                                    theta_t[:, layer, i, 1]), i, q)
            for i in range(q - 1):
                new = density.apply_2q_unitary(
                    new, gates.rzz(rzz_angles[layer, i]), i, i + 1, q)
        view = valid.view(-1, *([1] * (2 * q)))
        return torch.where(view, new, rho_ax)


class QSSMEncoder(nn.Module):
    def __init__(self, n_qubits: int = 3, cell_depth: int = 1,
                 hidden_dim: int = 32, angle_bound: float = 3.141592653589793,
                 gamma_init_min: float = 1e-4, gamma_init_max: float = 1e-1,
                 vocab_size: int = 6, proj_dim: int = 64,
                 dtype: torch.dtype = torch.complex64,
                 gamma_mode: str = "input", rzz_mode: str = "learned"):
        super().__init__()
        self.q = n_qubits
        self.d = 1 << n_qubits
        self.cell_depth = cell_depth
        self.vocab_size = vocab_size
        self.proj_dim = proj_dim
        self.cdtype = dtype
        self.controller = TokenController(
            vocab_size, hidden_dim, n_qubits, cell_depth, angle_bound,
            gamma_init_min, gamma_init_max, gamma_mode=gamma_mode)
        # learned, time-shared RZZ angles on the open chain (q-1 edges)
        self.rzz_angles = nn.Parameter(
            0.1 * torch.randn(cell_depth, max(n_qubits - 1, 1)))
        # rzz_mode="off" (no-coupling ablation): the parameter is drawn as
        # usual (same RNG stream as the main model), then zeroed and frozen,
        # so the recurrence stays a product of local channels
        if rzz_mode not in ("learned", "off"):
            raise ValueError(f"rzz_mode {rzz_mode!r}")
        self.rzz_mode = rzz_mode
        if rzz_mode == "off":
            with torch.no_grad():
                self.rzz_angles.zero_()
            self.rzz_angles.requires_grad_(False)
        self.raw_dim = readout.raw_feature_dim(n_qubits)
        self.proj = nn.Linear(self.raw_dim, proj_dim)
        self._step_mod = _StepModule(n_qubits, cell_depth)

    # ------------------------------------------------------------- state

    def initial_state(self, batch_size: int,
                      device: torch.device | str | None = None
                      ) -> torch.Tensor:
        dev = device if device is not None else self.proj.weight.device
        return density.initial_state(batch_size, self.q, dev, self.cdtype)

    def controls(self, tokens: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.controller(tokens)

    # ------------------------------------------------------------- step

    def _step_ax(self, rho_ax: torch.Tensor, theta_t: torch.Tensor,
                 gamma_t: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """One token. theta_t [B,depth,q,2]; gamma_t [B,q]; valid [B] bool."""
        return self._step_mod(rho_ax, theta_t, gamma_t, valid,
                              self.rzz_angles)

    # --------------------------------------------------------- full pass

    def forward_state(self, tokens: torch.Tensor, mask: torch.Tensor,
                      return_states: bool = False) -> torch.Tensor | list:
        """tokens [B,L] int, mask [B,L] bool -> terminal rho [B,D,D].

        return_states=True returns the state after every position (used by
        dense supervision; the recurrence never reads them back).
        """
        b, l = tokens.shape
        theta, gamma = self.controls(tokens)
        rho_ax = density.to_axes(self.initial_state(b, tokens.device), self.q)
        states = []
        for t in range(l):
            rho_ax = self._step_ax(rho_ax, theta[:, t], gamma[:, t],
                                   mask[:, t])
            if return_states:
                states.append(density.to_matrix(rho_ax, self.q))
        if return_states:
            return states
        return density.to_matrix(rho_ax, self.q)

    def forward_state_chunked(self, tokens: torch.Tensor, mask: torch.Tensor,
                              chunk_size: int) -> torch.Tensor:
        """Streaming evaluation: the state persists across chunks, so the
        result equals forward_state on the full sequence."""
        b, l = tokens.shape
        rho = self.initial_state(b, tokens.device)
        for s in range(0, l, chunk_size):
            rho = self._forward_range(rho, tokens[:, s:s + chunk_size],
                                      mask[:, s:s + chunk_size])
        return rho

    def _forward_range(self, rho: torch.Tensor, tokens: torch.Tensor,
                       mask: torch.Tensor) -> torch.Tensor:
        theta, gamma = self.controls(tokens)
        rho_ax = density.to_axes(rho, self.q)
        for t in range(tokens.shape[1]):
            rho_ax = self._step_ax(rho_ax, theta[:, t], gamma[:, t],
                                   mask[:, t])
        return density.to_matrix(rho_ax, self.q)

    # ------------------------------------------------------------ encode

    def _raw_features(self, tokens: torch.Tensor, mask: torch.Tensor,
                      shots: int | None, seed: int | None,
                      generator: torch.Generator | None) -> torch.Tensor:
        rho = self.forward_state(tokens, mask)
        if shots is None:
            return readout.terminal_readout(rho, self.q)
        gen = generator
        if gen is None:
            gen = torch.Generator(device=rho.device)
            if seed is not None:
                gen.manual_seed(seed)
        return readout.terminal_readout_shots(rho, self.q, shots, gen)

    def encode(self, tokens: torch.Tensor, mask: torch.Tensor,
               representation: str = "projected",
               rc_average: bool = False, shots: int | None = None,
               seed: int | None = None,
               generator: torch.Generator | None = None) -> torch.Tensor:
        """-> [B, 3q-1] (raw) or [B, proj_dim] (projected)."""
        z = self._raw_features(tokens, mask, shots, seed, generator)
        if rc_average:
            gen2 = None
            if shots is not None:
                gen2 = torch.Generator(device=tokens.device)
                gen2.manual_seed((seed or 0) + 1)
            z_rc = self._raw_features(rc_tokens(tokens, mask), mask, shots,
                                      seed, gen2)
            z = 0.5 * (z + z_rc)
        if representation == "raw":
            return z
        if representation == "projected":
            return F.gelu(self.proj(z))
        raise ValueError(f"unknown representation {representation!r}")


def encoder_kwargs(cfg: dict) -> dict:
    """QSSMEncoder constructor arguments from a (resolved) run config.
    Ablation keys are optional; their defaults give the main model."""
    m = cfg["model"]
    return {"n_qubits": m["n_qubits"], "cell_depth": m["cell_depth"],
            "hidden_dim": m["controller"]["hidden_dim"],
            "gamma_mode": m.get("forgetting", {}).get("gamma_mode", "input"),
            "rzz_mode": m.get("unitary", {}).get("rzz_mode", "learned")}


class QSSMModel(nn.Module):
    """Encoder + linear head (n_classes=4 for next-base pretraining)."""

    supports_dense_graph = True

    def __init__(self, encoder: QSSMEncoder, n_classes: int):
        super().__init__()
        self.encoder = encoder
        self.head = LinearHead(encoder.proj_dim, n_classes)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor,
                rc_average: bool = False, shots: int | None = None,
                seed: int | None = None) -> torch.Tensor:
        z = self.encoder.encode(tokens, mask, representation="projected",
                                rc_average=rc_average, shots=shots, seed=seed)
        return self.head(z)

    def forward_pretrain(self, tokens: torch.Tensor,
                         mask: torch.Tensor) -> torch.Tensor:
        return self.forward(tokens, mask)

    def forward_positions(self, tokens: torch.Tensor,
                          mask: torch.Tensor) -> torch.Tensor:
        """Per-position logits [B,L,C] from the state after every position,
        for dense next-base supervision (objective
        classical_sim_dense_supervision). The intermediate readouts are
        computed in simulation only and never feed the controller or the
        next state; downstream features use the terminal state."""
        states = self.encoder.forward_state(tokens, mask,
                                            return_states=True)
        rho = torch.stack(states, dim=1)                    # [B,L,D,D]
        b, l = rho.shape[:2]
        z = readout.terminal_readout(
            rho.reshape(b * l, self.encoder.d, self.encoder.d),
            self.encoder.q)
        return self.head(F.gelu(self.encoder.proj(z))).view(b, l, -1)
