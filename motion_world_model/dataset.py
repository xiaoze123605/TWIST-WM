from __future__ import annotations

import bisect
import pickle
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, Sampler

from .utils import (
    FUTURE_LENGTH,
    HISTORY_LENGTH,
    REFERENCE_DIM,
    box_smooth,
    ensure_quaternion_continuity,
    load_json,
    motion_group_id,
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_rotate_inverse,
    quaternion_slerp,
    quaternion_to_exp_map,
    quaternion_xyzw_to_euler,
    save_json,
    stable_split,
    streaming_mean_std,
)


@dataclass(frozen=True)
class CorruptionConfig:
    noise_std: float = 0.02
    hold_probability: float = 0.03
    delay_max_frames: int = 3
    lowpass_probability: float = 0.5
    lowpass_alpha_min: float = 0.6
    lowpass_alpha_max: float = 0.9

    def validate(self) -> None:
        if self.noise_std < 0:
            raise ValueError("noise_std must be non-negative")
        if not 0 <= self.hold_probability <= 1:
            raise ValueError("hold_probability must be in [0, 1]")
        if self.delay_max_frames < 0:
            raise ValueError("delay_max_frames must be non-negative")
        if not 0 <= self.lowpass_probability <= 1:
            raise ValueError("lowpass_probability must be in [0, 1]")
        if not 0 <= self.lowpass_alpha_min <= self.lowpass_alpha_max <= 1:
            raise ValueError("low-pass alpha range must lie in [0, 1]")

    def only(self, component: str) -> "CorruptionConfig":
        if component == "full":
            return self
        if component == "clean":
            return CorruptionConfig(0.0, 0.0, 0, 0.0, 0.0, 0.0)
        empty = CorruptionConfig(0.0, 0.0, 0, 0.0, 0.0, 0.0)
        if component == "noise":
            return replace(empty, noise_std=self.noise_std)
        if component == "hold":
            return replace(empty, hold_probability=self.hold_probability)
        if component == "delay":
            return replace(empty, delay_max_frames=self.delay_max_frames)
        if component == "lowpass":
            return replace(
                empty,
                lowpass_probability=self.lowpass_probability,
                lowpass_alpha_min=self.lowpass_alpha_min,
                lowpass_alpha_max=self.lowpass_alpha_max,
            )
        raise ValueError(f"unknown corruption component: {component}")


def corrupt_reference(
    clean: np.ndarray,
    rng: np.random.Generator,
    config: CorruptionConfig,
) -> np.ndarray:
    """Apply the student reference corruptions causally to one history window."""
    config.validate()
    out = np.asarray(clean, dtype=np.float32).copy()
    if config.noise_std > 0:
        out += rng.normal(0.0, config.noise_std, size=out.shape).astype(np.float32)

    if config.hold_probability > 0:
        for t in range(1, len(out)):
            if rng.random() < config.hold_probability:
                out[t] = out[t - 1]

    if config.delay_max_frames > 0:
        delayed = out.copy()
        delays = rng.integers(0, config.delay_max_frames + 1, size=len(out))
        for t, delay in enumerate(delays):
            delayed[t] = out[max(0, t - int(delay))]
        out = delayed

    if config.lowpass_probability > 0 and len(out):
        filtered_state = out[0].copy()
        for t in range(1, len(out)):
            if rng.random() < config.lowpass_probability:
                alpha = rng.uniform(config.lowpass_alpha_min, config.lowpass_alpha_max)
                out[t] = alpha * filtered_state + (1.0 - alpha) * out[t]
            filtered_state = out[t].copy()
    return out


def _compatible_numpy_module(module: str) -> str:
    if module == "numpy._core" or module.startswith("numpy._core."):
        return "numpy.core" + module[len("numpy._core") :]
    return module


class _NumpyCompatibilityUnpickler(pickle.Unpickler):
    """Read NumPy 2 pickles in the NumPy 1.x environment used by TWIST."""

    def find_class(self, module: str, name: str):
        return super().find_class(_compatible_numpy_module(module), name)


def _load_motion_pickle(path: Path) -> Dict[str, np.ndarray]:
    with path.open("rb") as handle:
        data = _NumpyCompatibilityUnpickler(handle).load()
    required = ("fps", "root_pos", "root_rot", "dof_pos")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{path}: missing keys {missing}")
    root_pos = np.asarray(data["root_pos"])
    root_rot = np.asarray(data["root_rot"])
    dof_pos = np.asarray(data["dof_pos"])
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"{path}: root_pos must be [T,3], got {root_pos.shape}")
    if root_rot.shape != (len(root_pos), 4):
        raise ValueError(f"{path}: root_rot must be [T,4], got {root_rot.shape}")
    if dof_pos.shape != (len(root_pos), 23):
        raise ValueError(f"{path}: dof_pos must be [T,23], got {dof_pos.shape}")
    if len(root_pos) < 2:
        raise ValueError(f"{path}: motion must contain at least two frames")
    fps = float(data["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"{path}: invalid fps {fps}")
    for name, value in (("root_pos", root_pos), ("root_rot", root_rot), ("dof_pos", dof_pos)):
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{path}: {name} contains NaN or Inf")
    return {
        "fps": fps,
        "root_pos": root_pos.astype(np.float64),
        "root_rot": root_rot.astype(np.float64),
        "dof_pos": dof_pos.astype(np.float64),
    }


def motion_to_reference_50hz(path: Path, target_fps: float = 50.0) -> np.ndarray:
    """Reproduce MotionLib sampling and the G1 31-D student reference layout."""
    if not np.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("target_fps must be positive")
    motion = _load_motion_pickle(path)
    source_fps = motion["fps"]
    root_pos = motion["root_pos"]
    root_rot = ensure_quaternion_continuity(motion["root_rot"])
    dof_pos = motion["dof_pos"]
    frames = len(root_pos)

    root_vel = np.zeros_like(root_pos)
    root_vel[:-1] = source_fps * np.diff(root_pos, axis=0)
    root_vel[-1] = root_vel[-2]
    root_vel = box_smooth(root_vel, 19)

    delta_rot = quaternion_multiply(root_rot[1:], quaternion_conjugate(root_rot[:-1]))
    root_ang_vel = np.zeros_like(root_pos)
    root_ang_vel[:-1] = source_fps * quaternion_to_exp_map(delta_rot)
    root_ang_vel[-1] = root_ang_vel[-2]
    root_ang_vel = box_smooth(root_ang_vel, 19)

    duration = (frames - 1) / source_fps
    target_frames = int(np.floor(duration * target_fps + 1e-8)) + 1
    times = np.arange(target_frames, dtype=np.float64) / target_fps
    source_position = np.minimum(times * source_fps, frames - 1)
    idx0 = np.floor(source_position).astype(np.int64)
    idx1 = np.minimum(idx0 + 1, frames - 1)
    blend = source_position - idx0

    sample_pos = (1.0 - blend[:, None]) * root_pos[idx0] + blend[:, None] * root_pos[idx1]
    sample_rot = quaternion_slerp(root_rot[idx0], root_rot[idx1], blend)
    sample_dof = (1.0 - blend[:, None]) * dof_pos[idx0] + blend[:, None] * dof_pos[idx1]
    sample_root_vel = quaternion_rotate_inverse(sample_rot, root_vel[idx0])
    sample_root_ang_vel = quaternion_rotate_inverse(sample_rot, root_ang_vel[idx0])
    euler = quaternion_xyzw_to_euler(sample_rot)
    euler[:, 2] = np.unwrap(euler[:, 2])

    reference = np.concatenate(
        (
            sample_pos[:, 2:3],
            euler,
            sample_root_vel,
            sample_root_ang_vel[:, 2:3],
            sample_dof,
        ),
        axis=-1,
    ).astype(np.float32)
    if reference.shape[1] != REFERENCE_DIM:
        raise AssertionError(f"expected {REFERENCE_DIM} reference dims, got {reference.shape}")
    return reference


def _read_motion_entries(yaml_path: Path) -> Tuple[Path, List[Dict[str, object]]]:
    with yaml_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    root = Path(config["root_path"]).expanduser()
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    entries: List[Dict[str, object]] = []
    seen = set()
    for item in config["motions"]:
        relative = str(item["file"]).replace("\\", "/")
        if relative in seen:
            continue
        seen.add(relative)
        entries.append({"relative_path": relative, "weight": float(item.get("weight", 1.0))})
    return root, entries


def build_dataset_cache(
    yaml_path: Path,
    output_dir: Path,
    target_fps: float = 50.0,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    split_seed: int = 42,
    max_motions: Optional[int] = None,
) -> Dict[str, object]:
    """Convert each source motion to one isolated cached 31-D sequence."""
    yaml_path = Path(yaml_path).resolve()
    output_dir = Path(output_dir).resolve()
    sequence_dir = output_dir / "sequences"
    sequence_dir.mkdir(parents=True, exist_ok=True)
    root, entries = _read_motion_entries(yaml_path)
    if max_motions is not None:
        if max_motions <= 0:
            raise ValueError("max_motions must be positive")
        entries = entries[:max_motions]

    manifests: Dict[str, List[Dict[str, object]]] = {"train": [], "val": [], "test": []}
    skipped: List[Dict[str, str]] = []
    minimum_frames = HISTORY_LENGTH + FUTURE_LENGTH
    for index, item in enumerate(entries):
        relative = str(item["relative_path"])
        source_path = root / relative
        group = motion_group_id(relative)
        split = stable_split(group, split_seed, train_ratio, val_ratio)
        cache_name = f"{index:06d}_{Path(relative).stem}.npy"
        cache_path = sequence_dir / split / cache_name
        try:
            reference = motion_to_reference_50hz(source_path, target_fps=target_fps)
            if len(reference) < minimum_frames:
                raise ValueError(f"only {len(reference)} frames after resampling; need {minimum_frames}")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, reference, allow_pickle=False)
            manifests[split].append(
                {
                    "motion_id": relative,
                    "group_id": group,
                    "source_path": str(source_path),
                    "cache_path": str(cache_path.relative_to(output_dir)),
                    "frames": int(len(reference)),
                    "windows": int(len(reference) - minimum_frames + 1),
                    "weight": item["weight"],
                }
            )
        except Exception as exc:
            skipped.append({"motion_id": relative, "error": str(exc)})

    for split, items in manifests.items():
        save_json(output_dir / f"{split}_manifest.json", {"split": split, "motions": items})

    group_sets = {
        split: {str(item["group_id"]) for item in items}
        for split, items in manifests.items()
    }
    if (
        group_sets["train"] & group_sets["val"]
        or group_sets["train"] & group_sets["test"]
        or group_sets["val"] & group_sets["test"]
    ):
        raise AssertionError("motion group leakage detected between dataset splits")

    train_arrays = (
        np.load(output_dir / item["cache_path"], mmap_mode="r")
        for item in manifests["train"]
    )
    mean, std, stats_frames = streaming_mean_std(train_arrays)
    np.savez(output_dir / "normalization.npz", mean=mean, std=std)
    metadata = {
        "source_yaml": str(yaml_path),
        "source_root": str(root),
        "target_fps": float(target_fps),
        "history_length": HISTORY_LENGTH,
        "future_length": FUTURE_LENGTH,
        "reference_dim": REFERENCE_DIM,
        "split_seed": split_seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "motions": {split: len(items) for split, items in manifests.items()},
        "windows": {
            split: int(sum(int(item["windows"]) for item in items))
            for split, items in manifests.items()
        },
        "normalization_frames": stats_frames,
        "skipped": skipped,
    }
    save_json(output_dir / "metadata.json", metadata)
    return metadata


class MotionWindowDataset(Dataset):
    """Lazy windows that never cross cached motion boundaries."""

    def __init__(
        self,
        cache_dir: Path,
        split: str,
        corruption: Optional[CorruptionConfig] = None,
        corruption_seed: int = 1234,
        max_windows: Optional[int] = None,
    ) -> None:
        self.cache_dir = Path(cache_dir).resolve()
        manifest = load_json(self.cache_dir / f"{split}_manifest.json")
        self.entries = manifest["motions"]
        if not self.entries:
            raise ValueError(f"{split} split contains no motions")
        self.corruption = corruption or CorruptionConfig()
        self.corruption.validate()
        self.corruption_seed = int(corruption_seed)
        self.epoch = 0
        self.cumulative = np.cumsum([int(item["windows"]) for item in self.entries]).tolist()
        self.total_windows = self.cumulative[-1]
        if max_windows is not None and max_windows <= 0:
            raise ValueError("max_windows must be positive")
        self.max_windows = min(self.total_windows, max_windows) if max_windows else self.total_windows

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.max_windows

    @lru_cache(maxsize=8)
    def _sequence(self, entry_index: int) -> np.ndarray:
        path = self.cache_dir / self.entries[entry_index]["cache_path"]
        # These per-motion arrays are small. Owning the memory avoids dangling
        # memmap views when multi-worker DataLoaders evict LRU entries.
        return np.load(path, allow_pickle=False)

    def __getitem__(self, index: int) -> Dict[str, object]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        # Development caps remain distributed over the complete split instead of
        # silently selecting only its first motions.
        global_index = index
        if self.max_windows < self.total_windows:
            global_index = int(index * self.total_windows / self.max_windows)
        entry_index = bisect.bisect_right(self.cumulative, global_index)
        previous = self.cumulative[entry_index - 1] if entry_index else 0
        start = global_index - previous
        sequence = self._sequence(entry_index)
        clean_history = np.asarray(sequence[start : start + HISTORY_LENGTH], dtype=np.float32)
        target_start = start + HISTORY_LENGTH - 1
        clean_target = np.asarray(
            sequence[target_start : target_start + FUTURE_LENGTH + 1], dtype=np.float32
        )
        seed = self.corruption_seed + 1_000_003 * self.epoch + global_index
        corrupted = corrupt_reference(clean_history, np.random.default_rng(seed), self.corruption)
        return {
            "corrupted_history": torch.from_numpy(corrupted),
            "clean_history": torch.from_numpy(clean_history.copy()),
            "target": torch.from_numpy(clean_target.copy()),
            "motion_id": str(self.entries[entry_index]["motion_id"]),
        }


class MotionWeightedSampler(Sampler[int]):
    """Match MotionLib: sample a YAML-weighted motion, then a uniform time."""

    def __init__(
        self,
        dataset: MotionWindowDataset,
        num_samples: Optional[int],
        seed: int,
        block_size: int = 64,
    ) -> None:
        if dataset.max_windows != dataset.total_windows:
            raise ValueError("MotionWeightedSampler requires an uncapped dataset")
        self.dataset = dataset
        self.num_samples = int(num_samples or dataset.total_windows)
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        weights = np.asarray([float(item.get("weight", 1.0)) for item in dataset.entries])
        if np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("motion weights must be non-negative with a positive sum")
        self.probabilities = weights / weights.sum()
        self.starts = np.asarray([0] + dataset.cumulative[:-1], dtype=np.int64)
        self.counts = np.asarray([int(item["windows"]) for item in dataset.entries], dtype=np.int64)
        self.seed = int(seed)
        self.block_size = int(block_size)
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + 1_000_003 * self.epoch)
        remaining = self.num_samples
        while remaining:
            size = min(self.block_size, remaining)
            entry = int(rng.choice(len(self.probabilities), p=self.probabilities))
            fractions = rng.random(size)
            local = (fractions * self.counts[entry]).astype(np.int64)
            for index in self.starts[entry] + local:
                yield int(index)
            remaining -= size


def load_normalization(cache_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    values = np.load(Path(cache_dir) / "normalization.npz", allow_pickle=False)
    return values["mean"].astype(np.float32), values["std"].astype(np.float32)
