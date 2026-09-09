from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import CorruptionConfig, MotionWeightedSampler, MotionWindowDataset, load_normalization
from .evaluate import evaluate_predictors
from .model import MotionGRU, motion_prediction_loss
from .utils import save_json, seed_everything


@dataclass
class TrainConfig:
    cache_dir: Path
    output_dir: Path
    epochs: int = 30
    batch_size: int = 256
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    current_weight: float = 1.0
    future_weight: float = 1.0
    seed: int = 42
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_train_windows: Optional[int] = None
    max_eval_windows: Optional[int] = None
    corruption_mode: str = "full"
    eval_corruption_mode: str = "full"
    sampling_mode: str = "motion_weighted"
    sampling_block_size: int = 64
    noise_std: float = 0.02
    hold_probability: float = 0.03
    delay_max_frames: int = 3
    lowpass_probability: float = 0.5
    lowpass_alpha_min: float = 0.6
    lowpass_alpha_max: float = 0.9

    def corruption(self, mode: Optional[str] = None) -> CorruptionConfig:
        return self.base_corruption().only(mode or self.corruption_mode)

    def base_corruption(self) -> CorruptionConfig:
        return CorruptionConfig(
            noise_std=self.noise_std,
            hold_probability=self.hold_probability,
            delay_max_frames=self.delay_max_frames,
            lowpass_probability=self.lowpass_probability,
            lowpass_alpha_min=self.lowpass_alpha_min,
            lowpass_alpha_max=self.lowpass_alpha_max,
        )


def _loader(
    dataset: MotionWindowDataset,
    config: TrainConfig,
    shuffle: bool,
    sampler=None,
) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=config.device.startswith("cuda"),
        generator=generator,
    )


def run_training(config: TrainConfig) -> Dict[str, object]:
    if config.epochs <= 0:
        raise ValueError("epochs must be positive")
    if config.current_weight < 0 or config.future_weight < 0:
        raise ValueError("loss weights must be non-negative")
    if config.current_weight == 0 and config.future_weight == 0:
        raise ValueError("at least one loss weight must be positive")
    seed_everything(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    training_corruption = config.corruption()
    evaluation_corruption = config.corruption(config.eval_corruption_mode)
    train_set = MotionWindowDataset(
        config.cache_dir,
        "train",
        training_corruption,
        corruption_seed=config.seed + 100,
        max_windows=None if config.sampling_mode == "motion_weighted" else config.max_train_windows,
    )
    val_set = MotionWindowDataset(
        config.cache_dir,
        "val",
        evaluation_corruption,
        corruption_seed=config.seed + 200,
        max_windows=config.max_eval_windows,
    )
    test_set = MotionWindowDataset(
        config.cache_dir,
        "test",
        evaluation_corruption,
        corruption_seed=config.seed + 300,
        max_windows=config.max_eval_windows,
    )
    train_sampler = None
    if config.sampling_mode == "motion_weighted":
        train_sampler = MotionWeightedSampler(
            train_set,
            num_samples=config.max_train_windows,
            seed=config.seed + 400,
            block_size=config.sampling_block_size,
        )
    train_loader = _loader(train_set, config, shuffle=True, sampler=train_sampler)
    val_loader = _loader(val_set, config, shuffle=False)
    test_loader = _loader(test_set, config, shuffle=False)

    mean, std = load_normalization(config.cache_dir)
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=config.device).view(1, 1, -1)
    std_t = torch.as_tensor(std, dtype=torch.float32, device=config.device).view(1, 1, -1)
    model_config = {
        "hidden_dim": config.hidden_dim,
        "num_layers": config.num_layers,
        "dropout": config.dropout,
    }
    model = MotionGRU(**model_config).to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_score = float("inf")
    history = []
    checkpoint_path = config.output_dir / "best.pt"

    for epoch in range(config.epochs):
        train_set.set_epoch(epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        totals = np.zeros(3, dtype=np.float64)
        samples = 0
        for batch in train_loader:
            corrupted = batch["corrupted_history"].to(config.device)
            target = batch["target"].to(config.device)
            normalized_history = (corrupted - mean_t) / std_t
            normalized_target = (target - mean_t) / std_t
            prediction = model(normalized_history)
            loss, parts = motion_prediction_loss(
                prediction,
                normalized_target,
                current_weight=config.current_weight,
                future_weight=config.future_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = corrupted.shape[0]
            totals += count * np.array(
                [loss.item(), parts["current"].item(), parts["future"].item()]
            )
            samples += count

        validation = evaluate_predictors(
            val_loader, mean, std, model, config.device, include_baselines=False
        )
        score = float(validation["gru"]["normalized_overall_rmse"])
        row = {
            "epoch": epoch + 1,
            "train_loss": float(totals[0] / samples),
            "train_current_mse": float(totals[1] / samples),
            "train_future_mse": float(totals[2] / samples),
            "val_current_rmse": validation["gru"]["current_rmse"],
            "val_future_rmse": validation["gru"]["future_rmse"],
            "val_normalized_current_rmse": validation["gru"]["normalized_current_rmse"],
            "val_normalized_future_rmse": validation["gru"]["normalized_future_rmse"],
        }
        history.append(row)
        print(row)
        if score < best_score:
            best_score = score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": model_config,
                    "corruption_config": asdict(config.base_corruption()),
                    "training_corruption_config": asdict(training_corruption),
                    "evaluation_corruption_config": asdict(evaluation_corruption),
                    "train_config": {**asdict(config), "cache_dir": str(config.cache_dir), "output_dir": str(config.output_dir)},
                    "epoch": epoch + 1,
                    "normalization_mean": mean,
                    "normalization_std": std,
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    metrics = evaluate_predictors(test_loader, mean, std, model, config.device)
    result = {
        "best_epoch": checkpoint["epoch"],
        "history": history,
        "test": metrics,
    }
    save_json(config.output_dir / "metrics.json", result)
    return result


def add_train_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--current-weight", type=float, default=1.0)
    parser.add_argument("--future-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-train-windows", type=int)
    parser.add_argument("--max-eval-windows", type=int)
    parser.add_argument("--corruption-mode", choices=("full", "clean", "noise", "hold", "delay", "lowpass"), default="full")
    parser.add_argument("--eval-corruption-mode", choices=("full", "clean", "noise", "hold", "delay", "lowpass"), default="full")
    parser.add_argument("--sampling-mode", choices=("motion_weighted", "windows"), default="motion_weighted")
    parser.add_argument("--sampling-block-size", type=int, default=64)
    parser.add_argument("--noise-std", type=float, default=0.02)
    parser.add_argument("--hold-probability", type=float, default=0.03)
    parser.add_argument("--delay-max-frames", type=int, default=3)
    parser.add_argument("--lowpass-probability", type=float, default=0.5)
    parser.add_argument("--lowpass-alpha-min", type=float, default=0.6)
    parser.add_argument("--lowpass-alpha-max", type=float, default=0.9)


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    names = TrainConfig.__dataclass_fields__.keys()
    return TrainConfig(**{name: getattr(args, name) for name in names})


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the offline GRU motion model")
    add_train_arguments(parser)
    args = parser.parse_args()
    result = run_training(config_from_args(args))
    print(result["test"]["comparison"])


if __name__ == "__main__":
    main()
