from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn

from .utils import FUTURE_LENGTH, REFERENCE_DIM, TARGET_LENGTH


class MotionGRU(nn.Module):
    """Minimal sequence model for 25x31 -> 11x31 reference prediction."""

    def __init__(
        self,
        input_dim: int = REFERENCE_DIM,
        hidden_dim: int = 128,
        num_layers: int = 1,
        output_steps: int = TARGET_LENGTH,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim != REFERENCE_DIM:
            raise ValueError(f"this experiment expects {REFERENCE_DIM} input dimensions")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_steps = output_steps
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_steps * input_dim),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or history.shape[-1] != self.input_dim:
            raise ValueError(f"history must be [batch,time,{self.input_dim}], got {tuple(history.shape)}")
        _, hidden = self.gru(history)
        residual = self.head(hidden[-1]).reshape(-1, self.output_steps, self.input_dim)
        # Predict a correction/trajectory relative to the most recent observation.
        return history[:, -1:, :] + residual


def motion_prediction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    current_weight: float = 1.0,
    future_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if prediction.shape != target.shape:
        raise ValueError(f"prediction and target shapes differ: {prediction.shape} vs {target.shape}")
    if prediction.shape[1] != FUTURE_LENGTH + 1:
        raise ValueError(f"expected {FUTURE_LENGTH + 1} target frames")
    current = torch.mean((prediction[:, 0] - target[:, 0]) ** 2)
    future = torch.mean((prediction[:, 1:] - target[:, 1:]) ** 2)
    total = current_weight * current + future_weight * future
    return total, {"current": current, "future": future}
