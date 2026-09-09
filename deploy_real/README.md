# TWIST G1 real-robot deployment

The original TWIST low-level entry point for a base policy is:

```bash
python server_low_level_g1_real.py --policy_path PATH/TO/BASE_POLICY.pt --net NETWORK_INTERFACE
```

The separate AnyAdapter V4 entry point is:

```bash
python server_low_level_g1_real_anyadapter.py \
  --policy_path PATH/TO/ANYADAPTER_V4_POLICY.pt \
  --net NETWORK_INTERFACE \
  --anyadapter-ema-alpha 0.0
```

`server_low_level_g1_real.py` is unchanged and does not forward to V2.
`server_low_level_g1_real_v2.py` is an obsolete experimental deployment path
and is not used by the AnyAdapter controller.

The AnyAdapter controller keeps the original real-controller behavior and adds
only the adapter observation wrapper:

- control period: `0.02 s` (50 Hz)
- action scale: `0.5`
- history length: `10`
- base TWIST policy input: `1155` = current frame `105` + historical frames
  `10 * 105`
- AnyAdapter V4 policy input: `2635` = base TWIST `1155` + adapter history
  `20 * 74`
- each AnyAdapter history frame contains the selected 51-D physical state
  (`base_ang_vel`, roll/pitch, joint position offsets, joint velocities) and
  the previous 23-D policy action
- policy output: 23 joints in leg-12 + waist-3 + arm-8 order
- Redis mimic input: 33 values; wrist roll is removed before the 31-D policy input

The AnyAdapter world model is not run during deployment. Only the exported
base actor, history encoder, and residual adapter run in the control loop.
Unsupported JIT contracts exit before DDS or motor communication begins.

Validate an exported V4 policy without Unitree communication:

```bash
python server_low_level_g1_real_anyadapter.py \
  --policy_path PATH/TO/ANYADAPTER_V4_POLICY.pt \
  --device cpu \
  --probe-only
```

Start Redis before running a sender or controller:

```bash
redis-server
```

## Legacy V2 notes

The remaining V2 stand-test, replay, and recovery notes below are retained for
reference only. They do not describe the current AnyAdapter deployment entry.

### 1. Stand test

Use this as the first powered test. It does not need GMR or Redis mimic data and
does not run the policy.

```bash
cd deploy_real
python server_low_level_g1_real_v2.py \
  --stand-test \
  --net NETWORK_INTERFACE \
  --stand-seconds 10
```

For a no-command bench check:

```bash
python server_low_level_g1_real_v2.py \
  --stand-test --dry-run --stand-seconds 2
```

## 2. Offline motion replay

Terminal A publishes a motion file. Viewer creation is disabled by default;
add `--vis` only on a machine with a working GUI.

```bash
cd deploy_real
python server_high_level_motion_lib.py \
  --motion_file PATH/TO/MOTION.pkl
```

Terminal B runs the real low-level controller:

```bash
cd deploy_real
python server_low_level_g1_real_v2.py \
  --policy_path PATH/TO/POLICY.pt \
  --net NETWORK_INTERFACE
```

The deprecated `server_low_level_g1_real.py` command accepts the same arguments
and forwards to v2. AnyAdapter detection is automatic; no enable flag is
required. `--anyadapter-ema-alpha 0.0` disables optional output smoothing.

The controller can also replay a recorded `[T, 33]` NumPy array without a
Redis sender:

```bash
python server_low_level_g1_real_v2.py \
  --policy_path PATH/TO/POLICY.pt \
  --replay-mimic PATH/TO/MIMIC.npy \
  --net NETWORK_INTERFACE
```

Add `--dry-run --device cpu --max-steps 500` to run the full observation,
policy, state-machine, safety-filter, and logging path without motor commands.

## 3. Real-time OptiTrack + GMR

Use the filtered v2 sender:

```bash
cd deploy_real
python server_motion_optitrack_gmr_clean_bufferfix_v2.py \
  --host OPTITRACK_SERVER_IP \
  --client_ip LOCAL_MOCAP_INTERFACE_IP
```

Then start the low-level controller in a second terminal:

```bash
python server_low_level_g1_real_v2.py \
  --policy_path PATH/TO/POLICY.pt \
  --net NETWORK_INTERFACE
```

`server_optitrack_redis.py` is an alternate, simpler GMR sender. Do not run two
senders at once because both write `action_mimic_g1`.

## Test and safety modes

- `--dry-run`: no Unitree environment and no motor commands.
- `--stand-test`: filtered default-pose hold; policy and GMR are not required.
- `--replay-mimic FILE.npy`: loop a `[T, 33]` mimic sequence.
- `--recovery-test`: force RECOVERY during frames 100-124.
- `--max-steps N`: stop a bench or CI run after N control steps.

The controller starts in `SAFE_STAND`. Fresh, finite Redis/replay input and low
risk must remain stable for 15 frames before it enters `TRACKING`. Elevated
risk enters `RECOVERY`; recovery lasts at least 0.4 seconds and returns to
tracking only after another stable window. Severe attitude, non-finite data,
abnormal policy targets, or a long input outage returns to `SAFE_STAND`.

All targets from `TRACKING`, `RECOVERY`, and `SAFE_STAND` pass through the same
finite check, action ramp, URDF joint-limit clamp, target-rate limit, and
per-step delta limit.

## Redis compatibility

Both message formats are accepted:

```json
[0.0, 0.0, "... 33 values total ..."]
```

```json
{
  "timestamp": 1750000000.0,
  "frame_id": 123,
  "action_mimic": [0.0, 0.0, "... 33 values total ..."]
}
```

A missing frame holds the last valid mimic observation. Before any valid frame
has arrived, fallback is `DEFAULT_MIMIC_OBS["g1"]`, never a zero vector.

## Logs

Logs are written under `deploy_real/logs/` by default:

- CSV: timing, Redis age, mode, risk, RPY, tracking errors, torque ratio, ramp.
- NPZ: full 23-D position/velocity/target/action/torque arrays, 31-D policy
  mimic input, RPY, angular velocity, risk, mode, Redis age, and loop timing.

Treat dry-run and stand-test success as prerequisites, not proof of safe dynamic
motion. Keep the robot supported and keep the remote emergency controls ready
during the first powered tracking and recovery tests.
