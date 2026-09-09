# Phase-1 offline result

Date: 2026-09-09

## Dataset

The cache was built from every unique entry in the original
`legged_gym/motion_data_configs/twist_dataset.yaml`, with each motion resampled
independently to 50 Hz. Eight clips shorter than the required 35 frames were
reported and skipped.

| Split | Motion files | Independent groups | Windows |
|---|---:|---:|---:|
| Train | 12,282 | 12,232 | 6,009,501 |
| Validation | 1,537 | 1,534 | 729,268 |
| Test | 1,475 | 1,466 | 724,836 |

The group intersections between all split pairs are empty. Training samples a
motion with its original YAML weight, then samples time uniformly inside that
motion. The formal run used 500,000 sampled train windows per epoch and 100,000
windows distributed across validation. Final testing evaluates all 724,836
windows in the independent test split.

## Model and baselines

The model is a one-layer, 128-hidden-unit GRU with a residual MLP head. It maps
corrupted `25x31` reference histories to clean `11x31` targets (current plus ten
future frames). The loss gives equal weight to current restoration MSE and
future prediction MSE. Baselines are last-frame hold, five-frame linear
extrapolation, and held-frame interpolation followed by causal filtering and
local extrapolation.

Because the channels mix metres and radians, the primary metric is RMSE after
division by each channel's train-split standard deviation. Raw RMSE and
physical component RMSE remain in the generated metrics file.

## Independent test result

The full model trained for ten epochs; the checkpoint was selected only from
validation normalized RMSE. Test used full corruption and every window from the
independent test-motion split.

| Predictor | Normalized current RMSE | Normalized future RMSE | Raw current RMSE | Raw future RMSE |
|---|---:|---:|---:|---:|
| Full GRU | **0.161149** | **0.290334** | **0.032463** | **0.078737** |
| Last-frame hold | 0.181237 | 0.369684 | 0.036336 | 0.097951 |
| Linear extrapolation | 0.181237 | 0.529610 | 0.036336 | 0.101493 |
| Filtering/interpolation | 0.211152 | 0.438123 | 0.053072 | 0.099212 |

The GRU reduces normalized RMSE by 11.1% for current restoration and 21.5% for
future prediction relative to the best baseline. It also beats last-frame hold
on every reported physical component: root height, orientation, local root
velocity, yaw rate, and joint position.

## Ablation

The three reduced variants used the same full motion pool, YAML-weighted
sampling, 500,000 train windows per epoch, common full-corruption validation and
100,000-window test sets, and five epochs. The table keeps the full model's
matching 100,000-window result. These runs establish directionality; the full
model received ten epochs.

| Variant | Normalized current RMSE | Normalized future RMSE |
|---|---:|---:|
| Full objective + full corruption | **0.160328** | **0.289963** |
| No current-restoration loss | 0.244533 | 0.302354 |
| No future-prediction loss | 0.170090 | 0.382076 |
| Clean-only training input | 0.180403 | 0.304371 |

Both objective terms are necessary for the best joint result. Corruption during
training materially improves restoration and also improves future prediction.

The same checkpoint was also tested with one corruption component enabled at a
time. It improves future RMSE over last-frame hold in every condition: clean
`0.2255 vs 0.2718`, noise `0.2578 vs 0.3124`, hold `0.2260 vs 0.2730`, delay
`0.2540 vs 0.3240`, and low-pass `0.2354 vs 0.2920`. On isolated clean/hold/
delay/low-pass inputs, last-frame hold remains better at the current frame; the
accepted joint gate is therefore specifically the configured full-corruption
distribution, not every individual corruption regime.

## Reproducibility and scope gate

- Cache: `track_dataset/motion_world_model_50hz` (ignored generated artifact).
- Full checkpoint and metrics:
  `legged_gym/logs/motion_world_model/full_stable_v2` (ignored generated artifact).
  `test_metrics_all_windows.json` is the authoritative final test result;
  `metrics.json` is the 100,000-window training-run result.
- Ablation checkpoints and metrics:
  `legged_gym/logs/motion_world_model/ablation_full_dataset` (ignored generated artifact).
- Seed: 42; batch size: 2048; device: RTX 4090.
- This machine's Python 3.8 / PyTorch 2.4 DataLoader workers segfaulted under
  multiprocessing, so the accepted formal runs use `--num-workers 0`.

The offline gate requested for phase 1 is satisfied. No TWIST integration, PKL
augmentation, or long-horizon rollout has been implemented.
