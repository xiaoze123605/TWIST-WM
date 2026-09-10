# TWIST-WM paired MuJoCo demo — 2026-09-10

All metrics use the clean reference at the same motion frame. No models were
trained or modified. Both motion servers and the policy use CPU. Physics uses
dt=0.001, decimation=20; video is 640x480, 50 fps, 380 frames (7.6 seconds).
Reference sampling uses `--steps 1`, the existing server default.
The first 24 WM frames pass corrupted input through without inference.
Clean/reference and processed/reference are published with atomic Redis MSET
and read with MGET. The optional frame handshake pairs exactly one reference
frame with each policy step, independent of rendering speed.

## Reproduce the three runs

Run from the repository root. Redis must already be running on localhost:6379.
These commands write to a new `replay_formal` directory, preserving the recorded
results in `formal`. Each high-level process must print `Waiting for sim_ready_g1`
before its simulator starts. On this machine the 3-second startup pause suffices.
Do not run groups concurrently: the existing Redis keys are shared.

```bash
cd '/home/hank/TWIST（anyadapter）'
PY=/home/hank/anaconda3/envs/twist/bin/python
MOTION=track_dataset/twist_motion_dataset/accad/B3___walk1.pkl
TWIST=legged_gym/logs/g1_stu_rl/0529_twist_rlbcstu/traced/0529_twist_rlbcstu-36500-jit.pt
WM=legged_gym/logs/motion_world_model/full_stable_v2/best.pt
PRESET=formal
OUT=wm_demo_outputs/replay_formal
mkdir -p "$OUT"

run_demo() {
  mode="$1"
  "$PY" deploy_real/server_high_level_motion_lib.py \
    --motion_file "$MOTION" --device cpu --steps 1 \
    --motion-speed 1.0 --motion-scale 1.0 --hip-yaw-scale 1.0 \
    --reference-mode "$mode" --wm-checkpoint "$WM" \
    --corruption-preset "$PRESET" --seed 42 \
    --wait-for-sim-ready --sim-ready-timeout 120 \
    > "$OUT/${mode}_high.log" 2>&1 &
  motion_pid=$!
  sleep 3
  xvfb-run -a "$PY" deploy_real/server_low_level_g1_sim.py \
    --policy_path "$TWIST" --device cpu --headless --sync-reference \
    --sim_duration 7.6 --record_video \
    --video_path "$OUT/demo_${mode}.mp4" \
    --metrics_out "$OUT/${mode}_summary.json" \
    > "$OUT/${mode}_low.log" 2>&1 || return 1
  wait "$motion_pid"
}

run_demo clean && run_demo corrupt && run_demo wm
"$PY" tools/summarize_wm_demo.py \
  --clean "$OUT/clean_summary.json" \
  --corrupt "$OUT/corrupt_summary.json" \
  --wm "$OUT/wm_summary.json"
```

For the fixed stress preset, set `PRESET=demo_stress` and
`OUT=wm_demo_outputs/replay_demo_stress`, create that directory and repeat the
three calls. Recorded stress results reuse the identical clean video/metrics
from formal; only corrupt and wm were rerun. Clean mode does not apply a preset.

## Recorded results

| Preset | Mode | Joint RMSE | Root error | Max tilt |
|---|---|---:|---:|---:|
| formal | clean | 0.14385194 | 0.31076225 | 0.25228111 |
| formal | corrupt | 0.15525292 | 0.34179525 | 0.24916000 |
| formal | wm | 0.14637697 | 0.28711595 | 0.25817103 |
| demo_stress | clean (reused) | 0.14385194 | 0.31076225 | 0.25228111 |
| demo_stress | corrupt | 0.17463031 | 0.23983292 | 0.24495689 |
| demo_stress | wm | 0.16639310 | 0.25246323 | 0.25820385 |

Improvement = 100 * (Corrupt - WM) / Corrupt; positive is better.

| Preset | Joint RMSE | Root error | Max tilt |
|---|---:|---:|---:|
| formal | +5.7171% | +15.9977% | -3.6166% |
| demo_stress | +4.7169% | -5.2663% | -5.4079% |

Joint RMSE and Max tilt are in radians. Root error retains the existing
sqrt(mean(height_error^2, roll_error^2, pitch_error^2, wrapped_yaw_error^2))
definition: it mixes metres and radians and has no single physical unit.
Consult the component metrics in each JSON; formal's root improvement is
primarily yaw, while height/roll/pitch do not improve together.

All summaries contain 380 policy frames, including the WM warmup. End-of-motion
return-to-standing is excluded. Videos were decoded and sampled at one-second
intervals for visual checking. The sampled poses show small gait differences,
but do not substantiate an obvious visual improvement or absence of temporal
jitter/latency. Formal improves joint tracking and the root aggregate; stress
does not improve overall stability. No parameter search or retraining followed.

Artifacts: `formal/demo_{clean,corrupt,wm}.mp4`,
`formal/{clean,corrupt,wm}_summary.json`, and the same filenames in `demo_stress/`.
Older files directly in `wm_demo_outputs/` predate clean-ground-truth correction
and are not the results reported here.

## Input SHA-256

TWIST: `f7d34dd2bc0b278cd4da33811dc8218e65da306a83d5c62f6fd12c1dc8202ef0`

WM: `187c11b128cbf1a6c095edcf1e3e8291fef76b63c03639f1d5618285f83afd99`

Motion: `ee25979f7d3b2cb38383096a4003fc91e5b61480a9e4e0530156ec8f56b0dacc`
