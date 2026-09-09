"""Small evaluation metrics kept independent from Isaac Gym imports."""

import torch


def rmse_from_mean_square(mean_square):
    """Convert an accumulated mean-square value to RMSE."""
    value = torch.as_tensor(mean_square, dtype=torch.float64)
    return torch.sqrt(value.clamp_min(0.0))


def tensor_rmse(error: torch.Tensor) -> torch.Tensor:
    """True elementwise root-mean-square error."""
    return rmse_from_mean_square(torch.mean(error.float().square()))
