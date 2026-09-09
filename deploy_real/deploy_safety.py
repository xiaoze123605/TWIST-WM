"""Pure helpers shared by the real-robot deployment entry points."""

import json
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np


MIMIC_OBS_DIM = 33
POLICY_ACTION_DIM = 23
POLICY_OBS_DIM = 1155


def parse_mimic_msg(
    raw_msg: Any,
    *,
    now: Optional[float] = None,
    expected_dim: int = MIMIC_OBS_DIM,
) -> Tuple[np.ndarray, float, int]:
    """Parse legacy list or timestamped dict Redis mimic messages.

    Returns ``(action_mimic, age_seconds, frame_id)``. Legacy list messages
    have no timestamp, so their age is reported as 0 and frame_id as -1.
    Invalid, non-finite, or incorrectly sized messages raise ``ValueError``.
    """
    if raw_msg is None:
        raise ValueError("empty mimic message")
    if isinstance(raw_msg, (bytes, bytearray)):
        raw_msg = raw_msg.decode("utf-8")
    if isinstance(raw_msg, str):
        try:
            raw_msg = json.loads(raw_msg)
        except json.JSONDecodeError as exc:
            raise ValueError("mimic message is not valid JSON") from exc

    timestamp = 0.0
    frame_id = -1
    if isinstance(raw_msg, dict):
        if "action_mimic" not in raw_msg:
            raise ValueError("dict mimic message has no action_mimic field")
        payload = raw_msg["action_mimic"]
        timestamp = float(raw_msg.get("timestamp", 0.0) or 0.0)
        frame_id = int(raw_msg.get("frame_id", -1))
    elif isinstance(raw_msg, (list, tuple, np.ndarray)):
        payload = raw_msg
    else:
        raise ValueError(f"unsupported mimic message type: {type(raw_msg).__name__}")

    mimic = np.asarray(payload, dtype=np.float32).reshape(-1)
    if mimic.size != expected_dim:
        raise ValueError(
            f"mimic_obs has {mimic.size} values, expected {expected_dim}"
        )
    if not np.all(np.isfinite(mimic)):
        raise ValueError("mimic_obs contains NaN or Inf")

    now = time.time() if now is None else float(now)
    age = max(0.0, now - timestamp) if timestamp > 0.0 else 0.0
    return mimic, age, frame_id


@dataclass
class SafetyFilterResult:
    target: np.ndarray
    replaced_nonfinite: bool
    joint_limit_clipped: bool
    rate_limited: bool
    delta_limited: bool


class TargetSafetyFilter:
    """Final target filter used by TRACKING, RECOVERY, and SAFE_STAND."""

    def __init__(
        self,
        default_pose: np.ndarray,
        lower_limits: np.ndarray,
        upper_limits: np.ndarray,
        control_dt: float,
        max_target_rate: float = 4.0,
        max_delta_per_step: float = 0.15,
    ):
        self.default_pose = np.asarray(default_pose, dtype=np.float32).copy()
        self.lower_limits = np.asarray(lower_limits, dtype=np.float32).copy()
        self.upper_limits = np.asarray(upper_limits, dtype=np.float32).copy()
        self.control_dt = float(control_dt)
        self.max_target_rate = float(max_target_rate)
        self.max_delta_per_step = float(max_delta_per_step)
        self.previous_target = self.default_pose.copy()

        shape = self.default_pose.shape
        if self.lower_limits.shape != shape or self.upper_limits.shape != shape:
            raise ValueError("joint limit shape must match default pose")

    def reset(self, target: Optional[np.ndarray] = None) -> None:
        value = self.default_pose if target is None else target
        self.previous_target = np.asarray(value, dtype=np.float32).copy()

    def apply(self, desired_target: np.ndarray, action_ramp: float) -> SafetyFilterResult:
        desired = np.asarray(desired_target, dtype=np.float32).reshape(-1)
        if desired.shape != self.default_pose.shape:
            raise ValueError(
                f"target has shape {desired.shape}, expected {self.default_pose.shape}"
            )

        finite_mask = np.isfinite(desired)
        replaced_nonfinite = not bool(np.all(finite_mask))
        if replaced_nonfinite:
            desired = np.where(finite_mask, desired, self.default_pose)

        ramp = float(np.clip(action_ramp, 0.0, 1.0))
        desired = self.default_pose + ramp * (desired - self.default_pose)

        clipped = np.clip(desired, self.lower_limits, self.upper_limits)
        joint_limit_clipped = not bool(np.allclose(clipped, desired))

        delta = clipped - self.previous_target
        rate_delta = self.max_target_rate * self.control_dt
        rate_clipped = np.clip(delta, -rate_delta, rate_delta)
        rate_limited = not bool(np.allclose(rate_clipped, delta))

        hard_clipped = np.clip(
            rate_clipped, -self.max_delta_per_step, self.max_delta_per_step
        )
        delta_limited = not bool(np.allclose(hard_clipped, rate_clipped))

        target = self.previous_target + hard_clipped
        target = np.clip(target, self.lower_limits, self.upper_limits).astype(np.float32)
        self.previous_target = target.copy()
        return SafetyFilterResult(
            target=target,
            replaced_nonfinite=replaced_nonfinite,
            joint_limit_clipped=joint_limit_clipped,
            rate_limited=rate_limited,
            delta_limited=delta_limited,
        )
