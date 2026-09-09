from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np


REFERENCE_DIM = 31
HISTORY_LENGTH = 25
FUTURE_LENGTH = 10
TARGET_LENGTH = FUTURE_LENGTH + 1

REFERENCE_SLICES = {
    "root_height": slice(0, 1),
    "orientation": slice(1, 4),
    "root_velocity": slice(4, 7),
    "yaw_rate": slice(7, 8),
    "joint_position": slice(8, 31),
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def motion_group_id(relative_path: str) -> str:
    """Group segmented files from one source motion into the same split."""
    path = Path(relative_path)
    stem = path.stem
    if "_seg" in stem and stem.rsplit("_seg", 1)[-1].isdigit():
        stem = stem.rsplit("_seg", 1)[0]
    return str(path.with_name(stem)).replace("\\", "/")


def stable_split(
    group_id: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> str:
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("split ratios must satisfy train > 0, val >= 0, train + val < 1")
    digest = hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "val"
    return "test"


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(x), np.cos(x))


def ensure_quaternion_continuity(quaternions: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternions, dtype=np.float64).copy()
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("root_rot contains a zero quaternion")
    q /= norms
    for i in range(1, len(q)):
        if np.dot(q[i - 1], q[i]) < 0:
            q[i] *= -1
    return q


def quaternion_xyzw_to_euler(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    x, y, z, w = (q[..., i] for i in range(4))
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.stack((roll, pitch, yaw), axis=-1)


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    out = np.asarray(q, dtype=np.float64).copy()
    out[..., :3] *= -1
    return out


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = np.moveaxis(np.asarray(q1), -1, 0)
    x2, y2, z2, w2 = np.moveaxis(np.asarray(q2), -1, 0)
    return np.stack(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        axis=-1,
    )


def quaternion_rotate_inverse(q: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(vectors, dtype=np.float64)
    q_xyz = q[..., :3]
    q_w = q[..., 3:4]
    # Conjugate rotation, equivalent to Isaac Gym's quat_rotate_inverse.
    return (
        v * (2.0 * q_w * q_w - 1.0)
        - 2.0 * q_w * np.cross(q_xyz, v)
        + 2.0 * q_xyz * np.sum(q_xyz * v, axis=-1, keepdims=True)
    )


def quaternion_to_exp_map(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    # Select the shortest equivalent rotation to avoid sign-induced spikes.
    q = np.where(q[..., 3:4] < 0, -q, q)
    xyz = q[..., :3]
    length = np.linalg.norm(xyz, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(length, np.clip(q[..., 3:4], -1.0, 1.0))
    axis = np.divide(xyz, length, out=np.zeros_like(xyz), where=length > 1e-8)
    axis = np.where(length > 1e-8, axis, np.array([0.0, 0.0, 1.0]))
    return angle * axis


def quaternion_slerp(q0: np.ndarray, q1: np.ndarray, amount: np.ndarray) -> np.ndarray:
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64).copy()
    amount = np.asarray(amount, dtype=np.float64)[..., None]
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot)
    dot = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    near = sin_theta < 1e-6
    w0 = np.where(near, 1.0 - amount, np.sin((1.0 - amount) * theta) / np.maximum(sin_theta, 1e-8))
    w1 = np.where(near, amount, np.sin(amount * theta) / np.maximum(sin_theta, 1e-8))
    out = w0 * q0 + w1 * q1
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def box_smooth(values: np.ndarray, width: int = 19) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if width <= 1:
        return values.copy()
    width = min(width, len(values) if len(values) % 2 == 1 else max(1, len(values) - 1))
    if width <= 1:
        return values.copy()
    pad = width // 2
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="constant")
    kernel = np.ones(width, dtype=np.float64) / width
    return np.stack([np.convolve(padded[:, i], kernel, mode="valid") for i in range(values.shape[1])], axis=-1)


def save_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def streaming_mean_std(arrays: Iterable[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, int]:
    count = 0
    total = np.zeros(REFERENCE_DIM, dtype=np.float64)
    total_sq = np.zeros(REFERENCE_DIM, dtype=np.float64)
    for array in arrays:
        x = np.asarray(array, dtype=np.float64)
        count += x.shape[0]
        total += x.sum(axis=0)
        total_sq += np.square(x).sum(axis=0)
    if count == 0:
        raise ValueError("cannot compute statistics from an empty training split")
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 1e-8)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32), count
