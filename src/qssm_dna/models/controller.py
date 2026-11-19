"""Classical input controller. Its only input is the one-hot of the current
token; it receives no state, expectation, future base or label.

    c_t = GELU(W_e onehot(x_t) + b_e)
    theta_t = angle_bound * tanh(W_theta c_t + b_theta)   -> RY/RZ angles
    gamma_{t,i} = sigmoid(w_i^T c_t + b_i)                -> damping probs
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


GAMMA_MODES = ("input", "static", "zero")


class TokenController(nn.Module):
    """gamma_mode (input is the main model; static and zero are ablations):
      input  -- gamma_{t,i} = sigmoid(w_i^T c_t + b_i)   (token-dependent)
      static -- gamma_i = sigmoid(b_i), identical for every token
      zero   -- gamma = 0 (no damping; unitary-only recurrence)
    Parameters are created identically in every mode, so a given seed
    gives the same initial angles/bias across ablations."""

    def __init__(self, vocab_size: int, hidden_dim: int, n_qubits: int,
                 cell_depth: int, angle_bound: float,
                 gamma_init_min: float, gamma_init_max: float,
                 gamma_mode: str = "input"):
        super().__init__()
        if gamma_mode not in GAMMA_MODES:
            raise ValueError(f"gamma_mode {gamma_mode!r} not in {GAMMA_MODES}")
        self.gamma_mode = gamma_mode
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.q = n_qubits
        self.cell_depth = cell_depth
        self.angle_bound = float(angle_bound)
        self.embed = nn.Linear(vocab_size, hidden_dim)
        # per cell layer, per qubit: (RY angle, RZ angle)
        self.n_angles = cell_depth * 2 * n_qubits
        self.angle_head = nn.Linear(hidden_dim, self.n_angles)
        self.gamma_head = nn.Linear(hidden_dim, n_qubits)
        # small random init: avoids the all-zero-gradient symmetric point
        nn.init.normal_(self.angle_head.weight, std=0.1)
        nn.init.zeros_(self.angle_head.bias)
        nn.init.normal_(self.gamma_head.weight, std=0.05)
        with torch.no_grad():
            if n_qubits == 1:
                gammas = torch.tensor([gamma_init_min])
            else:
                gammas = torch.exp(torch.linspace(
                    math.log(gamma_init_min), math.log(gamma_init_max),
                    n_qubits))
            self.gamma_head.bias.copy_(
                torch.log(gammas / (1.0 - gammas)))  # logit

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """tokens: [B,L] long -> theta [B,L,depth,q,2], gamma [B,L,q]."""
        onehot = F.one_hot(tokens, self.vocab_size).to(
            self.embed.weight.dtype)
        c = F.gelu(self.embed(onehot))
        theta = self.angle_bound * torch.tanh(self.angle_head(c))
        b, l, _ = theta.shape
        theta = theta.view(b, l, self.cell_depth, self.q, 2)
        if self.gamma_mode == "input":
            gamma = torch.sigmoid(self.gamma_head(c))
        elif self.gamma_mode == "static":
            gamma = torch.sigmoid(self.gamma_head.bias).expand(b, l, self.q)
        else:
            gamma = theta.new_zeros(b, l, self.q)
        return theta, gamma

    def forward_token(self, tokens_t: torch.Tensor
                      ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single position: tokens_t [B] -> theta [B,depth,q,2], gamma [B,q]."""
        theta, gamma = self.forward(tokens_t.unsqueeze(1))
        return theta[:, 0], gamma[:, 0]
