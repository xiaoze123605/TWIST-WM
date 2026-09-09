from __future__ import annotations

import torch
import torch.nn.functional as F

from .utils import TARGET_LENGTH


def last_frame_hold(history: torch.Tensor, output_steps: int = TARGET_LENGTH) -> torch.Tensor:
    """Repeat the newest corrupted observation for every target horizon."""
    return history[:, -1:, :].expand(-1, output_steps, -1).clone()


def linear_extrapolation(
    history: torch.Tensor,
    output_steps: int = TARGET_LENGTH,
    fit_frames: int = 5,
) -> torch.Tensor:
    """Least-squares constant-velocity extrapolation from recent observations."""
    fit_frames = min(max(2, fit_frames), history.shape[1])
    recent = history[:, -fit_frames:, :]
    x = torch.arange(fit_frames, dtype=history.dtype, device=history.device)
    x = x - x.mean()
    slope = torch.sum(recent * x.view(1, -1, 1), dim=1) / torch.sum(x * x)
    offsets = torch.arange(output_steps, dtype=history.dtype, device=history.device)
    return history[:, -1:, :] + offsets.view(1, -1, 1) * slope[:, None, :]


def _interpolate_holds(history: torch.Tensor, tolerance: float = 1e-7) -> torch.Tensor:
    """Linearly fill observable held runs; trailing holds remain causal."""
    batch, frames, features = history.shape
    held = torch.max(torch.abs(history[:, 1:] - history[:, :-1]), dim=-1).values <= tolerance
    positions = torch.arange(frames, device=history.device).view(1, -1).expand(batch, -1)

    left_indices = torch.zeros((batch, frames), dtype=torch.long, device=history.device)
    last_valid = torch.zeros(batch, dtype=torch.long, device=history.device)
    for t in range(1, frames):
        last_valid = torch.where(held[:, t - 1], last_valid, torch.full_like(last_valid, t))
        left_indices[:, t] = last_valid

    right_indices = torch.full((batch, frames), frames - 1, dtype=torch.long, device=history.device)
    next_valid = torch.full((batch,), frames - 1, dtype=torch.long, device=history.device)
    for t in range(frames - 1, 0, -1):
        next_valid = torch.where(held[:, t - 1], next_valid, torch.full_like(next_valid, t))
        right_indices[:, t] = next_valid

    gather_shape = (-1, -1, features)
    left = torch.gather(history, 1, left_indices.unsqueeze(-1).expand(*gather_shape))
    right = torch.gather(history, 1, right_indices.unsqueeze(-1).expand(*gather_shape))
    denominator = (right_indices - left_indices).clamp_min(1)
    amount = (positions - left_indices).to(history.dtype) / denominator.to(history.dtype)
    interpolated = left + amount.unsqueeze(-1) * (right - left)
    internal_hold = torch.cat(
        (torch.zeros((batch, 1), dtype=torch.bool, device=history.device), held), dim=1
    ) & (right_indices > positions)
    return torch.where(internal_hold.unsqueeze(-1), interpolated, history)


def filtering_interpolation(
    history: torch.Tensor,
    output_steps: int = TARGET_LENGTH,
    filter_width: int = 5,
    fit_frames: int = 5,
) -> torch.Tensor:
    """Repair held samples, smooth causally, then extrapolate a local trend."""
    repaired = _interpolate_holds(history)
    width = min(max(1, filter_width), repaired.shape[1])
    values = repaired.transpose(1, 2)
    values = F.pad(values, (width - 1, 0), mode="replicate")
    kernel = torch.ones(
        repaired.shape[-1], 1, width, dtype=history.dtype, device=history.device
    ) / width
    filtered = F.conv1d(values, kernel, groups=repaired.shape[-1]).transpose(1, 2)
    return linear_extrapolation(filtered, output_steps=output_steps, fit_frames=fit_frames)


BASELINES = {
    "last_frame_hold": last_frame_hold,
    "linear_extrapolation": linear_extrapolation,
    "filtering_interpolation": filtering_interpolation,
}
