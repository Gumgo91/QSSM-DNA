"""Prediction heads on the projected embedding."""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(z)
