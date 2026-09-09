#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motion_world_model.dataset import build_dataset_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an isolated 50 Hz 31-D motion cache")
    parser.add_argument("--motion-file", type=Path, required=True, help="Original twist_dataset.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-fps", type=float, default=50.0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--max-motions", type=int, help="Development-only prefix limit")
    args = parser.parse_args()
    metadata = build_dataset_cache(
        args.motion_file,
        args.output_dir,
        target_fps=args.target_fps,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        max_motions=args.max_motions,
    )
    print(metadata)


if __name__ == "__main__":
    main()
