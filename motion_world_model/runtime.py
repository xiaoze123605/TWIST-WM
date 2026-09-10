"""Small, stateful runtime helpers for the frozen Motion World Model demo."""

from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Optional, Sequence

import numpy as np
import torch

from .model import MotionGRU
from .utils import HISTORY_LENGTH, REFERENCE_DIM


YAW_INDEX = 3
MIMIC_DIM_WITH_WRISTS = 33
WRIST_ROLL_INDICES = (27, 32)
WM_REFERENCE_INDICES = tuple(
    index for index in range(MIMIC_DIM_WITH_WRISTS) if index not in WRIST_ROLL_INDICES
)


def wrap_angle(angle: float) -> float:
    """Wrap one angle to [-pi, pi]."""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def unwrap_angle_near(angle: float, previous: Optional[float]) -> float:
    """Choose the 2*pi-equivalent angle closest to the previous value."""
    wrapped = wrap_angle(angle)
    if previous is None:
        return wrapped
    return float(previous + wrap_angle(wrapped - previous))


def wrap_reference_yaw(reference: np.ndarray) -> np.ndarray:
    result = np.asarray(reference, dtype=np.float32).copy()
    if result.shape != (REFERENCE_DIM,):
        raise ValueError(f"reference must have shape ({REFERENCE_DIM},), got {result.shape}")
    result[YAW_INDEX] = wrap_angle(float(result[YAW_INDEX]))
    return result


def remove_wrist_roll(reference_33d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split the 33-D Redis reference into the 31-D WM input and two wrists."""
    reference = np.asarray(reference_33d, dtype=np.float32)
    if reference.shape != (MIMIC_DIM_WITH_WRISTS,):
        raise ValueError(
            f"reference must have shape ({MIMIC_DIM_WITH_WRISTS},), got {reference.shape}"
        )
    return reference[list(WM_REFERENCE_INDICES)].copy(), reference[list(WRIST_ROLL_INDICES)].copy()


def reinsert_wrist_roll(reference_31d: np.ndarray, wrists: Sequence[float]) -> np.ndarray:
    """Restore the untouched wrist-roll values to a 31-D WM result."""
    reference = np.asarray(reference_31d, dtype=np.float32)
    wrist_values = np.asarray(wrists, dtype=np.float32)
    if reference.shape != (REFERENCE_DIM,):
        raise ValueError(f"reference must have shape ({REFERENCE_DIM},), got {reference.shape}")
    if wrist_values.shape != (len(WRIST_ROLL_INDICES),):
        raise ValueError(f"wrists must have shape ({len(WRIST_ROLL_INDICES)},), got {wrist_values.shape}")
    result = np.empty(MIMIC_DIM_WITH_WRISTS, dtype=np.float32)
    result[list(WM_REFERENCE_INDICES)] = reference
    result[list(WRIST_ROLL_INDICES)] = wrist_values
    return result


@dataclass(frozen=True)
class RuntimeCorruptionConfig:
    noise_std: float = 0.02
    hold_probability: float = 0.03
    delay_max_frames: int = 3
    lowpass_probability: float = 0.5
    lowpass_alpha_min: float = 0.6
    lowpass_alpha_max: float = 0.9

    @classmethod
    def from_preset(cls, preset: str) -> "RuntimeCorruptionConfig":
        if preset == "formal":
            return cls()
        if preset == "demo_stress":
            # Visualization-only stress test. The trained model is unchanged.
            return cls(
                noise_std=0.03,
                hold_probability=0.10,
                delay_max_frames=6,
                lowpass_probability=0.75,
                lowpass_alpha_min=0.70,
                lowpass_alpha_max=0.93,
            )
        raise ValueError(f"unknown corruption preset: {preset}")

    def validate(self) -> None:
        if self.noise_std < 0:
            raise ValueError("noise_std must be non-negative")
        if not 0.0 <= self.hold_probability <= 1.0:
            raise ValueError("hold_probability must be in [0, 1]")
        if self.delay_max_frames < 0:
            raise ValueError("delay_max_frames must be non-negative")
        if not 0.0 <= self.lowpass_probability <= 1.0:
            raise ValueError("lowpass_probability must be in [0, 1]")
        if not 0.0 <= self.lowpass_alpha_min <= self.lowpass_alpha_max <= 1.0:
            raise ValueError("low-pass alpha range must lie in [0, 1]")


class RuntimeReferenceCorruptor:
    """Causal corruption matching the Phase-1 training components."""

    def __init__(
        self,
        config: Optional[RuntimeCorruptionConfig] = None,
        seed: int = 42,
    ) -> None:
        self.config = config or RuntimeCorruptionConfig()
        self.config.validate()
        self.seed = int(seed)
        self.reset()

    @classmethod
    def from_preset(cls, preset: str, seed: int = 42) -> "RuntimeReferenceCorruptor":
        return cls(RuntimeCorruptionConfig.from_preset(preset), seed=seed)

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self._previous_clean_yaw: Optional[float] = None
        self._hold_state: Optional[np.ndarray] = None
        self._delay_buffer: Deque[np.ndarray] = deque(
            maxlen=self.config.delay_max_frames + 1
        )
        self._filter_state: Optional[np.ndarray] = None

    def corrupt(self, clean_reference: np.ndarray) -> np.ndarray:
        clean = np.asarray(clean_reference, dtype=np.float32)
        if clean.shape != (REFERENCE_DIM,):
            raise ValueError(f"reference must have shape ({REFERENCE_DIM},), got {clean.shape}")
        if not np.all(np.isfinite(clean)):
            raise ValueError("clean reference contains NaN or Inf")

        current = clean.copy()
        current[YAW_INDEX] = unwrap_angle_near(
            float(current[YAW_INDEX]), self._previous_clean_yaw
        )
        self._previous_clean_yaw = float(current[YAW_INDEX])

        if self.config.noise_std > 0.0:
            current += self.rng.normal(
                0.0, self.config.noise_std, size=current.shape
            ).astype(np.float32)

        if (
            self._hold_state is not None
            and self.rng.random() < self.config.hold_probability
        ):
            current = self._hold_state.copy()
        self._hold_state = current.copy()

        self._delay_buffer.append(current.copy())
        delay = (
            int(self.rng.integers(0, self.config.delay_max_frames + 1))
            if self.config.delay_max_frames > 0
            else 0
        )
        delayed_index = max(0, len(self._delay_buffer) - 1 - delay)
        current = self._delay_buffer[delayed_index].copy()

        if self._filter_state is None:
            self._filter_state = current.copy()
        elif self.rng.random() < self.config.lowpass_probability:
            alpha = float(
                self.rng.uniform(
                    self.config.lowpass_alpha_min,
                    self.config.lowpass_alpha_max,
                )
            )
            current = alpha * self._filter_state + (1.0 - alpha) * current
            self._filter_state = current.astype(np.float32, copy=True)
        else:
            self._filter_state = current.copy()

        return current.astype(np.float32, copy=False)


def resolve_checkpoint_path(
    checkpoint: Optional[str | Path], repository_root: Optional[str | Path] = None
) -> Path:
    """Prefer full_stable_v2/best.pt, then find another existing WM best.pt."""
    if checkpoint is not None:
        explicit = Path(checkpoint).expanduser().resolve()
        if not explicit.is_file():
            raise FileNotFoundError(f"Motion-WM checkpoint not found: {explicit}")
        return explicit

    root = (
        Path(repository_root).expanduser().resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[1]
    )
    preferred = root / "legged_gym/logs/motion_world_model/full_stable_v2/best.pt"
    if preferred.is_file():
        return preferred
    candidates = sorted(
        (root / "legged_gym/logs/motion_world_model").glob("**/best.pt")
    )
    if not candidates:
        raise FileNotFoundError(
            "No Motion-WM best.pt found under legged_gym/logs/motion_world_model"
        )
    return candidates[0].resolve()


class MotionReferenceRefiner:
    """Maintain a 25-frame history and restore only the clean current frame."""

    def __init__(self, checkpoint_path: str | Path, device: str = "auto") -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Motion-WM checkpoint not found: {self.checkpoint_path}")
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if device == "auto" else torch.device(device)
        try:
            checkpoint = torch.load(
                self.checkpoint_path, map_location=self.device, weights_only=False
            )
        except TypeError:  # PyTorch versions before weights_only was added.
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        required = (
            "model_state",
            "model_config",
            "normalization_mean",
            "normalization_std",
        )
        missing = [name for name in required if name not in checkpoint]
        if missing:
            raise ValueError(f"checkpoint is missing required fields: {missing}")

        mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float32)
        std = np.asarray(checkpoint["normalization_std"], dtype=np.float32)
        if mean.shape != (REFERENCE_DIM,) or std.shape != (REFERENCE_DIM,):
            raise ValueError(
                f"checkpoint normalization must be ({REFERENCE_DIM},), got {mean.shape}/{std.shape}"
            )
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
            raise ValueError("checkpoint normalization contains NaN or Inf")
        if np.any(std <= 0.0):
            raise ValueError("checkpoint normalization std must be positive")

        self.model = MotionGRU(**checkpoint["model_config"]).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.mean = torch.as_tensor(mean, device=self.device).view(1, 1, -1)
        self.std = torch.as_tensor(std, device=self.device).view(1, 1, -1)
        self.history: Deque[np.ndarray] = deque(maxlen=HISTORY_LENGTH)
        self._previous_input_yaw: Optional[float] = None
        self.fallback_count = 0

    @property
    def ready(self) -> bool:
        return len(self.history) == HISTORY_LENGTH

    def reset(self) -> None:
        self.history.clear()
        self._previous_input_yaw = None
        self.fallback_count = 0

    def refine(self, corrupted_reference: np.ndarray) -> np.ndarray:
        corrupted = np.asarray(corrupted_reference, dtype=np.float32)
        if corrupted.shape != (REFERENCE_DIM,):
            raise ValueError(
                f"reference must have shape ({REFERENCE_DIM},), got {corrupted.shape}"
            )
        if not np.all(np.isfinite(corrupted)):
            self.fallback_count += 1
            warnings.warn("Motion-WM input contains NaN/Inf; returning corrupted reference")
            return corrupted.copy()

        continuous = corrupted.copy()
        continuous[YAW_INDEX] = unwrap_angle_near(
            float(continuous[YAW_INDEX]), self._previous_input_yaw
        )
        self._previous_input_yaw = float(continuous[YAW_INDEX])
        self.history.append(continuous)
        fallback = wrap_reference_yaw(continuous)

        # Do not pad with a repeated first frame. Warm up on raw corrupted input.
        if not self.ready:
            return fallback

        history = torch.as_tensor(
            np.stack(self.history), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        try:
            with torch.no_grad():
                prediction = self.model((history - self.mean) / self.std)
                restored = prediction[0, 0] * self.std[0, 0] + self.mean[0, 0]
            result = restored.detach().cpu().numpy().astype(np.float32, copy=False)
            if result.shape != (REFERENCE_DIM,) or not np.all(np.isfinite(result)):
                raise ValueError("Motion-WM output is not finite")
            return wrap_reference_yaw(result)
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            self.fallback_count += 1
            warnings.warn(f"Motion-WM inference failed ({exc}); returning corrupted reference")
            return fallback
