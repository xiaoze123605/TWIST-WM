# Offline 31-D motion world model

This package is deliberately isolated from the TWIST policy. Phase 1 neither
generates augmented PKLs nor performs long autoregressive rollouts.

## Contract

- Source: the original `legged_gym/motion_data_configs/twist_dataset.yaml`.
- Sampling: each motion is independently resampled to 50 Hz.
- Split: deterministic SHA-256 assignment by motion. Files named `_segNN` share
  one group and therefore cannot leak across train/validation/test.
- Sampling: training matches MotionLib semantics by choosing a motion according
  to its YAML weight and then choosing a window uniformly within that motion.
  Samples are emitted in small motion-local blocks to keep multi-worker reads
  efficient; this changes ordering, not the sampling distribution.
- Reference: `[height, roll, pitch, yaw, local root velocity xyz, local yaw
  rate, 23 joint positions]`, 31 values per frame.
- Input/target: corrupted frames `t-24..t` (25x31) to clean frames `t..t+10`
  (11x31). No window crosses a motion boundary.
- Objective: current-frame restoration MSE plus future-frame prediction MSE.

## Commands

Run these from the repository root in the `twist` environment.

```bash
conda run -n twist python tools/build_motion_world_model_dataset.py \
  --motion-file legged_gym/motion_data_configs/twist_dataset.yaml \
  --output-dir track_dataset/motion_world_model_50hz

conda run -n twist python -m motion_world_model.train \
  --cache-dir track_dataset/motion_world_model_50hz \
  --output-dir legged_gym/logs/motion_world_model/full

conda run -n twist python -m motion_world_model.evaluate \
  --cache-dir track_dataset/motion_world_model_50hz \
  --checkpoint legged_gym/logs/motion_world_model/full/best.pt \
  --output legged_gym/logs/motion_world_model/full/test_metrics.json

conda run -n twist python -m motion_world_model.ablate \
  --cache-dir track_dataset/motion_world_model_50hz \
  --output-dir legged_gym/logs/motion_world_model/ablation
```

Keep `--num-workers 0` on the repository's current Python 3.8 / PyTorch 2.4
environment; multiprocessing DataLoader workers were observed to segfault
during the full experiment.

All ablation variants are evaluated with the same full corruption by default;
`clean_input` changes training input only. The evaluator compares the GRU
against last-frame hold, recent linear
extrapolation, and causal filtering plus held-frame interpolation. Its
`comparison.confirmed` flag is true only when the GRU beats every baseline on
both current restoration and future prediction on the independent test split.
The gate uses train-standard-deviation-normalized RMSE because the 31 channels
mix metres and radians; raw RMSE and per-component physical-unit RMSE are also
reported.

For a cheap development build, pass `--max-motions N`; this must not be used
for the final result. The full dataset is about 4 GB before conversion, so disk
usage should be checked before building its cache.
