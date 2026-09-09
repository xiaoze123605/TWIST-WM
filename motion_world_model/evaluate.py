from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .baselines import BASELINES
from .dataset import CorruptionConfig, MotionWindowDataset, load_normalization
from .model import MotionGRU
from .utils import REFERENCE_SLICES, save_json


class MetricAccumulator:
    def __init__(self) -> None:
        self.squared: Optional[np.ndarray] = None
        self.samples = 0

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        error = prediction - target
        # Euler angles are periodic; use the shortest angular error.
        error[..., 1:4] = torch.atan2(torch.sin(error[..., 1:4]), torch.cos(error[..., 1:4]))
        squared = torch.sum(error.double() ** 2, dim=0).cpu().numpy()
        self.squared = squared if self.squared is None else self.squared + squared
        self.samples += prediction.shape[0]

    def result(self, std: np.ndarray) -> Dict[str, object]:
        if self.squared is None or self.samples == 0:
            raise ValueError("no samples were evaluated")
        mse = self.squared / self.samples
        normalized_mse = mse / np.square(np.maximum(std, 1e-8))[None, :]
        result: Dict[str, object] = {
            "overall_rmse": float(np.sqrt(mse.mean())),
            "current_rmse": float(np.sqrt(mse[0].mean())),
            "future_rmse": float(np.sqrt(mse[1:].mean())),
            "per_horizon_rmse": np.sqrt(mse.mean(axis=1)).tolist(),
            "normalized_overall_rmse": float(np.sqrt(normalized_mse.mean())),
            "normalized_current_rmse": float(np.sqrt(normalized_mse[0].mean())),
            "normalized_future_rmse": float(np.sqrt(normalized_mse[1:].mean())),
            "samples": self.samples,
        }
        for name, feature_slice in REFERENCE_SLICES.items():
            result[f"{name}_rmse"] = float(np.sqrt(mse[:, feature_slice].mean()))
        return result


@torch.no_grad()
def evaluate_predictors(
    loader: Iterable[Dict[str, object]],
    mean: np.ndarray,
    std: np.ndarray,
    model: Optional[MotionGRU] = None,
    device: str = "cpu",
    include_baselines: bool = True,
) -> Dict[str, object]:
    names = list(BASELINES) if include_baselines else []
    if model is not None:
        names.insert(0, "gru")
        model.eval()
    accumulators = {name: MetricAccumulator() for name in names}
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=device).view(1, 1, -1)
    std_t = torch.as_tensor(std, dtype=torch.float32, device=device).view(1, 1, -1)

    for batch in loader:
        history = batch["corrupted_history"].to(device)
        target = batch["target"].to(device)
        if model is not None:
            normalized = (history - mean_t) / std_t
            prediction = model(normalized) * std_t + mean_t
            accumulators["gru"].update(prediction, target)
        if include_baselines:
            for name, predictor in BASELINES.items():
                accumulators[name].update(predictor(history), target)

    metrics = {name: accumulator.result(std) for name, accumulator in accumulators.items()}
    if model is not None and include_baselines:
        best_current = min(metrics[name]["normalized_current_rmse"] for name in BASELINES)
        best_future = min(metrics[name]["normalized_future_rmse"] for name in BASELINES)
        metrics["comparison"] = {
            "primary_metric": "train-std-normalized RMSE",
            "beats_all_baselines_current": metrics["gru"]["normalized_current_rmse"] < best_current,
            "beats_all_baselines_future": metrics["gru"]["normalized_future_rmse"] < best_future,
            "confirmed": (
                metrics["gru"]["normalized_current_rmse"] < best_current
                and metrics["gru"]["normalized_future_rmse"] < best_future
            ),
        }
    return metrics


def load_checkpoint(path: Path, device: str):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = MotionGRU(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the offline 31-D motion model and baselines")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--corruption", choices=("full", "clean", "noise", "hold", "delay", "lowpass"), default="full")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    model, checkpoint = load_checkpoint(args.checkpoint, args.device)
    corruption = CorruptionConfig(**checkpoint["corruption_config"]).only(args.corruption)
    dataset = MotionWindowDataset(args.cache_dir, args.split, corruption, max_windows=args.max_windows)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    mean, std = load_normalization(args.cache_dir)
    metrics = evaluate_predictors(loader, mean, std, model, args.device)
    if args.output:
        save_json(args.output, metrics)
    print(metrics)


if __name__ == "__main__":
    main()
