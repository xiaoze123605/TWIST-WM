from __future__ import annotations

import argparse
from dataclasses import replace

from .train import TrainConfig, add_train_arguments, config_from_args, run_training
from .utils import save_json


VARIANTS = {
    "full": {},
    "no_current_loss": {"current_weight": 0.0},
    "no_future_loss": {"future_weight": 0.0},
    "clean_input": {"corruption_mode": "clean"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run objective and corruption ablations")
    add_train_arguments(parser)
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS))
    args = parser.parse_args()
    base = config_from_args(args)
    summary = {}
    for name in args.variants:
        config = replace(base, output_dir=base.output_dir / name, **VARIANTS[name])
        print(f"running ablation: {name}")
        result = run_training(config)
        summary[name] = {
            "best_epoch": result["best_epoch"],
            "test": result["test"],
        }
    save_json(base.output_dir / "ablation_summary.json", summary)


if __name__ == "__main__":
    main()
