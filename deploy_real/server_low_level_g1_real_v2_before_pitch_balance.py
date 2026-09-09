#!/usr/bin/env python3
"""Safe real-robot low-level controller for the 23-action TWIST policy."""

import argparse
import csv
import json
import os
import time
import traceback
from collections import deque
from datetime import datetime

import numpy as np
import redis
import torch

from data_utils.params import DEFAULT_MIMIC_OBS
from data_utils.rot_utils import quatToEuler
from deploy_safety import (
    MIMIC_OBS_DIM,
    POLICY_ACTION_DIM,
    POLICY_OBS_DIM,
    TargetSafetyFilter,
    parse_mimic_msg,
)
from robot_control.common.remote_controller import KeyMap
from robot_control.config import Config
from robot_control.g1_wrapper import G1RealWorldEnv


RAMP_TIME = 5.0
TRACKING_TRANSITION_TIME = 1.0
MAX_TARGET_RATE = 4.0
MAX_DELTA_PER_STEP = 0.15
REDIS_STALE_THRESHOLD = 0.10
REDIS_RECOVERY_MISS_FRAMES = 5
REDIS_SAFE_STAND_MISS_FRAMES = 50
RISK_ROLL_THRESHOLD = 0.60
RISK_PITCH_THRESHOLD = 0.60
SEVERE_ROLL_PITCH = 0.85
RISK_ROLL_RATE_THRESHOLD = 6.0
RISK_PITCH_RATE_THRESHOLD = 6.0
RISK_TRACKING_ERROR = 0.50
RISK_TORQUE_RATIO = 0.85
RISK_LOOP_DT_RATIO = 1.5
RISK_HIGH = 0.60
RISK_LOW = 0.20
RISK_STABLE_FRAMES = 15
RECOVERY_MIN_DURATION = 0.40
RECOVERY_BLEND_TIME = 0.75
RECOVERY_ACTION_SCALE = 0.10
SHUTDOWN_BLEND_TIME = 1.0

# Policy order: left leg 6, right leg 6, waist 3, left arm 4, right arm 4.
JOINT_LOWER = np.array([
    -2.5307, -0.5236, -2.7576, -0.087267, -0.87267, -0.2618,
    -2.5307, -2.9671, -2.7576, -0.087267, -0.87267, -0.2618,
    -2.618, -0.52, -0.52,
    -3.0892, -1.5882, -2.618, -1.0472,
    -3.0892, -2.2515, -2.618, -1.0472,
], dtype=np.float32)
JOINT_UPPER = np.array([
    2.8798, 2.9671, 2.7576, 2.8798, 0.5236, 0.2618,
    2.8798, 0.5236, 2.7576, 2.8798, 0.5236, 0.2618,
    2.618, 0.52, 0.52,
    2.6704, 2.2515, 2.618, 2.0944,
    2.6704, 1.5882, 2.618, 2.0944,
], dtype=np.float32)
JOINT_LIMIT_MARGIN = 0.03


def extract_mimic_obs_to_body_and_wrist(mimic_obs):
    mimic_obs = np.asarray(mimic_obs, dtype=np.float32).reshape(-1)
    if mimic_obs.size != MIMIC_OBS_DIM:
        raise ValueError(f"mimic_obs must have {MIMIC_OBS_DIM} values")
    wrist_ids = [27, 32]
    other_ids = [idx for idx in range(MIMIC_OBS_DIM) if idx not in wrist_ids]
    return mimic_obs[other_ids], mimic_obs[wrist_ids]


def compute_risk_score(
    roll, pitch, ang_vel, dof_pos, target_dof_pos, tau_est,
    torque_limits, redis_age, loop_dt, control_dt,
):
    tracking_error = np.asarray(target_dof_pos) - np.asarray(dof_pos)
    rms_tracking_error = float(np.sqrt(np.mean(tracking_error ** 2)))
    max_tracking_error = float(np.max(np.abs(tracking_error)))
    torque_ratio = float(np.max(
        np.abs(tau_est) / (np.asarray(torque_limits) + 1e-6)
    ))
    values = {
        "roll": min(abs(float(roll)) / RISK_ROLL_THRESHOLD, 1.0),
        "pitch": min(abs(float(pitch)) / RISK_PITCH_THRESHOLD, 1.0),
        "roll_rate": min(abs(float(ang_vel[0])) / RISK_ROLL_RATE_THRESHOLD, 1.0),
        "pitch_rate": min(abs(float(ang_vel[1])) / RISK_PITCH_RATE_THRESHOLD, 1.0),
        "tracking": min(rms_tracking_error / RISK_TRACKING_ERROR, 1.0),
        "torque": min(torque_ratio / RISK_TORQUE_RATIO, 1.0),
        "redis": min(float(redis_age) / REDIS_STALE_THRESHOLD, 1.0),
        "loop_dt": min(float(loop_dt) / (control_dt * RISK_LOOP_DT_RATIO), 1.0),
    }
    weights = {
        "roll": 2.0, "pitch": 2.0, "roll_rate": 1.5, "pitch_rate": 1.5,
        "tracking": 1.0, "torque": 1.5, "redis": 0.5, "loop_dt": 0.5,
    }
    score = sum(weights[key] * values[key] for key in weights) / sum(weights.values())
    return float(score), max_tracking_error, rms_tracking_error, torque_ratio


class DeploymentLogger:
    CSV_FIELDS = [
        "timestamp", "frame_id", "loop_dt", "redis_age", "mode", "risk_score",
        "roll", "pitch", "yaw", "max_tracking_error", "rms_tracking_error",
        "max_torque_ratio", "ramp", "deploy_action_scale",
    ]
    ARRAY_FIELDS = [
        "dof_pos", "dof_vel", "target_dof_pos", "raw_action", "last_action",
        "tau_est", "action_mimic", "rpy", "ang_vel", "risk_score", "mode",
        "redis_age", "loop_dt",
    ]

    def __init__(self, prefix):
        self.prefix = prefix
        self.csv_path = f"{prefix}.csv" if prefix else None
        self.npz_path = f"{prefix}.npz" if prefix else None
        self.csv_file = None
        self.csv_writer = None
        self.arrays = {key: [] for key in self.ARRAY_FIELDS}
        if prefix:
            os.makedirs(os.path.dirname(prefix), exist_ok=True)
            self.csv_file = open(self.csv_path, "w", newline="", buffering=65536)
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.CSV_FIELDS)
            self.csv_writer.writeheader()

    def append(self, summary, arrays):
        if self.csv_writer is None:
            return
        self.csv_writer.writerow({key: summary[key] for key in self.CSV_FIELDS})
        for key in self.ARRAY_FIELDS:
            value = arrays[key]
            self.arrays[key].append(value if isinstance(value, str) else np.array(value, copy=True))

    def close(self):
        if self.csv_file is None:
            return
        self.csv_file.close()
        print(f"[LOG] CSV: {self.csv_path}  ({len(self.arrays.get('dof_pos', []))} frames)")
        try:
            np.savez_compressed(
                self.npz_path,
                **{key: np.asarray(value) for key, value in self.arrays.items()},
            )
            print(f"[LOG] NPZ: {self.npz_path}")
        except Exception as exc:
            print(f"[LOG] NPZ write failed ({exc}); CSV is still saved.")
        self.csv_file = None


class RealTimePolicyControllerV2:
    def __init__(
        self, policy_path, config_path, device="cuda", net="eno1",
        log_prefix=None, stand_test=False, dry_run=False, replay_mimic=None,
        recovery_test=False, max_steps=None, stand_seconds=10.0,
        deploy_action_scale=1.0, shutdown_behavior="hold",
    ):
        self.stand_test = stand_test
        self.dry_run = dry_run
        self.replay_mimic_path = replay_mimic
        self.recovery_test = recovery_test
        self.max_steps = max_steps
        self.stand_seconds = stand_seconds
        self.deploy_action_scale = float(deploy_action_scale)
        if not (0.0 <= self.deploy_action_scale <= 1.0):
            raise ValueError("--deploy-action-scale must be in [0.0, 1.0]")
        print(f"[SAFETY] deploy_action_scale={self.deploy_action_scale:.3f}")
        self.shutdown_behavior = str(shutdown_behavior)
        if self.shutdown_behavior not in ("hold", "zero_torque"):
            raise ValueError("--shutdown-behavior must be 'hold' or 'zero_torque'")
        print(f"[SAFETY] shutdown_behavior={self.shutdown_behavior}")

        self.redis_client = redis.Redis(
            host="localhost", port=6379, db=0,
            socket_connect_timeout=0.05, socket_timeout=0.05,
        )
        self.config = Config(config_path)
        self.env = None if dry_run else G1RealWorldEnv(net=net, config=self.config)
        self.robot_ready = False

        if device.startswith("cuda") and not torch.cuda.is_available():
            print("[WARN] CUDA unavailable; using CPU.")
            device = "cpu"
        self.device = device
        self.policy = None
        if not stand_test:
            if not policy_path:
                raise ValueError("--policy_path is required unless --stand-test is used")
            self.policy = torch.jit.load(policy_path, map_location=device)
            self.policy.eval()
            try:
                with torch.no_grad():
                    probe = self.policy(
                        torch.zeros(1, POLICY_OBS_DIM, device=self.device)
                    ).detach().cpu().numpy().reshape(-1)
            except Exception as exc:
                raise ValueError(
                    "policy does not accept the required 1155-D observation"
                ) from exc
            if probe.size != POLICY_ACTION_DIM:
                raise ValueError(
                    f"policy output has {probe.size} values, expected 23"
                )
            print(f"[POLICY] Loaded {policy_path} on {device}")
            print("[POLICY] Contract verified: 1155 -> 23")

        self.num_actions = POLICY_ACTION_DIM
        self.default_dof_pos = np.concatenate(
            [self.config.default_angles, self.config.arm_waist_target]
        ).astype(np.float32)
        if self.default_dof_pos.size != self.num_actions:
            raise ValueError("configured default pose is not 23-dimensional")
        if self.config.control_dt != 0.02 or self.config.action_scale != 0.5:
            raise ValueError("deployment requires control_dt=0.02 and action_scale=0.5")

        self.control_dt = self.config.control_dt
        self.action_scale = self.config.action_scale
        self.ang_vel_scale = self.config.ang_vel_scale
        self.dof_vel_scale = self.config.dof_vel_scale
        self.dof_pos_scale = self.config.dof_pos_scale
        self.ankle_idx = [4, 5, 10, 11]
        self.torque_limits = np.asarray(
            getattr(self.env, "torque_limits", [
                88, 139, 88, 139, 50, 50, 88, 139, 88, 139, 50, 50,
                88, 50, 50, 25, 25, 25, 25, 25, 25, 25, 25,
            ]), dtype=np.float32,
        )
        self.safety_filter = TargetSafetyFilter(
            self.default_dof_pos,
            JOINT_LOWER + JOINT_LIMIT_MARGIN,
            JOINT_UPPER - JOINT_LIMIT_MARGIN,
            self.control_dt,
            MAX_TARGET_RATE,
            MAX_DELTA_PER_STEP,
        )

        self.n_mimic_obs = 31
        self.n_proprio = self.n_mimic_obs + 3 + 2 + 3 * self.num_actions
        self.history_len = 10
        if self.n_proprio * (self.history_len + 1) != POLICY_OBS_DIM:
            raise AssertionError("policy observation dimension changed")
        self.proprio_history_buf = deque(maxlen=self.history_len)
        for _ in range(self.history_len):
            self.proprio_history_buf.append(
                np.zeros(self.n_proprio, dtype=np.float32)
            )

        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.last_valid_mimic_full = None
        self.last_wrist = np.zeros(2, dtype=np.float32)
        self.frame_id = 0
        self.redis_miss_count = 0
        self.mode = "INIT"
        self.mode_entry_time = time.monotonic()
        self.tracking_entry_time = self.mode_entry_time
        self.stable_frames = 0
        self.start_time = None
        self.ramp = 0.0
        self.logger = DeploymentLogger(log_prefix)

        self.replay_data = None
        self.replay_idx = 0
        if replay_mimic:
            loaded = np.load(replay_mimic)
            if isinstance(loaded, np.lib.npyio.NpzFile):
                key = "action_mimic" if "action_mimic" in loaded else loaded.files[0]
                loaded = loaded[key]
            self.replay_data = np.asarray(loaded, dtype=np.float32)
            if self.replay_data.ndim != 2 or self.replay_data.shape[1] != MIMIC_OBS_DIM:
                raise ValueError("replay mimic must have shape [T, 33]")
            print(f"[REPLAY] Loaded {self.replay_data.shape}")

    def _set_mode(self, new_mode, reason):
        if new_mode == self.mode:
            return
        print(f"[MODE] {self.mode} -> {new_mode}: {reason}")
        self.mode = new_mode
        self.mode_entry_time = time.monotonic()
        self.stable_frames = 0
        if new_mode == "TRACKING":
            self.tracking_entry_time = self.mode_entry_time

    def _reset_robot(self):
        if self.dry_run:
            return
        self.env.zero_torque_state()
        self.env.move_to_default_pos()
        self.env.default_pos_state()
        self.robot_ready = True
        self.safety_filter.reset(self.default_dof_pos)

    def _get_robot_state(self):
        if self.dry_run:
            return (
                self.default_dof_pos.copy(),
                np.zeros(self.num_actions, dtype=np.float32),
                np.array([1, 0, 0, 0], dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                np.zeros(self.num_actions, dtype=np.float32),
            )
        dof_pos, dof_vel, quat, ang_vel = self.env.get_robot_state()
        return (
            np.asarray(dof_pos, dtype=np.float32),
            np.asarray(dof_vel, dtype=np.float32),
            np.asarray(quat, dtype=np.float32),
            np.asarray(ang_vel, dtype=np.float32),
            np.asarray(self.env.tauj, dtype=np.float32).copy(),
        )

    def _read_mimic(self):
        if self.replay_data is not None:
            if self.replay_idx >= len(self.replay_data):
                self.replay_idx = 0
            mimic = self.replay_data[self.replay_idx].copy()
            self.replay_idx += 1
            return mimic, 0.0, self.replay_idx, True
        try:
            raw = self.redis_client.get("action_mimic_g1")
            mimic, age, frame_id = parse_mimic_msg(raw)
            return mimic, age, frame_id, True
        except (redis.RedisError, ValueError, TypeError) as exc:
            if self.redis_miss_count == 0:
                print(f"[WARN] mimic input unavailable: {exc}")
            return None, float("inf"), -1, False

    def _publish_proprio(self, obs_proprio):
        try:
            self.redis_client.set("state_body_g1", json.dumps(obs_proprio.tolist()))
        except redis.RedisError:
            pass

    def _send_action(self, target, wrist):
        if not self.dry_run:
            self.env.send_robot_action(
                target, kp_scale=1.0, kd_scale=1.0,
                left_wrist_roll=float(wrist[0]),
                right_wrist_roll=float(wrist[1]),
            )

    def _severe_fault(self, rpy, state_arrays, candidate, redis_age):
        if any(not np.all(np.isfinite(value)) for value in state_arrays):
            return "non-finite robot state"
        if not np.all(np.isfinite(candidate)):
            return "non-finite policy target"
        if max(abs(float(rpy[0])), abs(float(rpy[1]))) >= SEVERE_ROLL_PITCH:
            return "excessive roll/pitch"
        if self.redis_miss_count > REDIS_SAFE_STAND_MISS_FRAMES:
            return "long Redis/replay outage"
        if np.max(np.abs(candidate - self.default_dof_pos)) > 4.0:
            return "implausible joint target"
        if not np.isfinite(redis_age):
            return None
        return None

    def _update_state_machine(self, risk_score, input_healthy, severe_reason):
        now = time.monotonic()
        if severe_reason:
            self._set_mode("SAFE_STAND", severe_reason)
            return
        if self.mode == "INIT":
            self._set_mode("SAFE_STAND", "startup")

        if self.mode == "SAFE_STAND":
            if input_healthy and risk_score <= RISK_LOW:
                self.stable_frames += 1
                if self.stable_frames >= RISK_STABLE_FRAMES:
                    self._set_mode("TRACKING", "healthy input stable")
            else:
                self.stable_frames = 0
        elif self.mode == "TRACKING":
            if not input_healthy and self.redis_miss_count > REDIS_RECOVERY_MISS_FRAMES:
                self._set_mode("RECOVERY", "mimic input interrupted")
            elif risk_score >= RISK_HIGH:
                self._set_mode("RECOVERY", f"risk={risk_score:.3f}")
        elif self.mode == "RECOVERY":
            if input_healthy and risk_score <= RISK_LOW:
                self.stable_frames += 1
                if (self.stable_frames >= RISK_STABLE_FRAMES
                        and now - self.mode_entry_time >= RECOVERY_MIN_DURATION):
                    self._set_mode("TRACKING", "risk stable")
            else:
                self.stable_frames = 0

    def _desired_for_mode(self, policy_target):
        now = time.monotonic()
        if self.mode == "TRACKING":
            mode_ramp = min(
                1.0, (now - self.tracking_entry_time) / TRACKING_TRANSITION_TIME
            )
            global_ramp = min(1.0, (now - self.start_time) / RAMP_TIME)
            self.ramp = min(mode_ramp, global_ramp)
            return policy_target, self.ramp
        if self.mode == "RECOVERY":
            alpha = min(1.0, (now - self.mode_entry_time) / RECOVERY_BLEND_TIME)
            reduced = (
                self.default_dof_pos
                + RECOVERY_ACTION_SCALE * (policy_target - self.default_dof_pos)
            )
            desired = (1.0 - alpha) * reduced + alpha * self.default_dof_pos
            self.ramp = 1.0
            return desired, 1.0
        self.ramp = 1.0
        return self.default_dof_pos, 1.0

    def _run_stand_test(self):
        print(f"[STAND-TEST] Holding filtered default pose for {self.stand_seconds:.1f}s")
        steps = int(self.stand_seconds / self.control_dt)
        if self.max_steps is not None:
            steps = min(steps, self.max_steps)
        for _ in range(steps):
            cycle_start = time.monotonic()
            dof_pos, dof_vel, quat, ang_vel, tau_est = self._get_robot_state()
            rpy = np.asarray(quatToEuler(quat), dtype=np.float32)
            result = self.safety_filter.apply(self.default_dof_pos, 1.0)
            self._send_action(result.target, np.zeros(2, dtype=np.float32))
            loop_dt = time.monotonic() - cycle_start
            now_wall = time.time()
            self.logger.append(
                {
                    "timestamp": f"{now_wall:.6f}",
                    "frame_id": self.frame_id,
                    "loop_dt": f"{loop_dt:.6f}",
                    "redis_age": f"{float('inf'):.6f}",
                    "mode": "STAND_TEST",
                    "risk_score": f"{0.0:.6f}",
                    "roll": f"{rpy[0]:.6f}",
                    "pitch": f"{rpy[1]:.6f}",
                    "yaw": f"{rpy[2]:.6f}",
                    "max_tracking_error": f"{0.0:.6f}",
                    "rms_tracking_error": f"{0.0:.6f}",
                    "max_torque_ratio": f"{0.0:.6f}",
                    "ramp": f"{1.0:.6f}",
                    "deploy_action_scale": f"{self.deploy_action_scale:.6f}",
                },
                {
                    "dof_pos": dof_pos, "dof_vel": dof_vel,
                    "target_dof_pos": result.target,
                    "raw_action": np.zeros(self.num_actions, dtype=np.float32),
                    "last_action": np.zeros(self.num_actions, dtype=np.float32),
                    "tau_est": tau_est,
                    "action_mimic": np.zeros(self.n_mimic_obs, dtype=np.float32),
                    "rpy": rpy, "ang_vel": ang_vel,
                    "risk_score": 0.0, "mode": "STAND_TEST",
                    "redis_age": float("inf"), "loop_dt": loop_dt,
                },
            )
            self.frame_id += 1
            elapsed = time.monotonic() - cycle_start
            time.sleep(max(0.0, self.control_dt - elapsed))

    def run(self):
        self._reset_robot()
        self.start_time = time.monotonic()
        self._set_mode("SAFE_STAND", "startup")
        print(
            f"[START] dry_run={self.dry_run} stand_test={self.stand_test} "
            f"replay={self.replay_mimic_path is not None}"
        )
        try:
            if self.stand_test:
                self._run_stand_test()
                return
            while self.max_steps is None or self.frame_id < self.max_steps:
                cycle_start = time.monotonic()
                if (not self.dry_run
                        and self.env.remote_controller.button[KeyMap.select] == 1):
                    print("[INFO] Select pressed.")
                    break

                dof_pos, dof_vel, quat, ang_vel, tau_est = self._get_robot_state()
                rpy = np.asarray(quatToEuler(quat), dtype=np.float32)

                # ── SAFE_STAND fast path: skip Redis, skip policy ──
                # In SAFE_STAND the policy output is discarded anyway, but
                # Redis timeouts + CPU inference can drop the control rate
                # below 50 Hz, causing the motor watchdog to trip.  Keep the
                # loop tight so the robot holds default pose reliably.
                if self.mode == "SAFE_STAND":
                    # Periodic mimic probe (every ~0.5 s) to detect recovery.
                    probe_now = (self.redis_miss_count % 25 == 0)
                    ss_input_healthy = False
                    ss_redis_age = float("inf")
                    ss_source_frame_id = -1
                    if probe_now:
                        mimic_probe, ss_redis_age, ss_source_frame_id, ok = (
                            self._read_mimic()
                        )
                        if ok and ss_redis_age <= REDIS_STALE_THRESHOLD:
                            ss_input_healthy = True
                    if not ss_input_healthy:
                        self.redis_miss_count += 1

                    severe_reason = None
                    if max(abs(float(rpy[0])), abs(float(rpy[1]))) >= SEVERE_ROLL_PITCH:
                        severe_reason = "excessive roll/pitch"
                    self._update_state_machine(
                        0.0, ss_input_healthy, severe_reason,
                    )

                    # If we just transitioned to TRACKING, fall through to the
                    # full path so the next frame runs policy inference.
                    if self.mode == "SAFE_STAND":
                        desired, _action_ramp = self._desired_for_mode(
                            self.default_dof_pos,
                        )
                        filtered = self.safety_filter.apply(desired, 1.0)
                        self._send_action(
                            filtered.target, np.zeros(2, dtype=np.float32),
                        )

                        loop_dt = time.monotonic() - cycle_start
                        now_wall = time.time()
                        self.logger.append(
                            {
                                "timestamp": f"{now_wall:.6f}",
                                "frame_id": ss_source_frame_id,
                                "loop_dt": f"{loop_dt:.6f}",
                                "redis_age": f"{ss_redis_age:.6f}",
                                "mode": self.mode,
                                "risk_score": "0.000000",
                                "roll": f"{rpy[0]:.6f}",
                                "pitch": f"{rpy[1]:.6f}",
                                "yaw": f"{rpy[2]:.6f}",
                                "max_tracking_error": "0.000000",
                                "rms_tracking_error": "0.000000",
                                "max_torque_ratio": "0.000000",
                                "ramp": "1.000000",
                                "deploy_action_scale": f"{self.deploy_action_scale:.6f}",
                            },
                            {
                                "dof_pos": dof_pos, "dof_vel": dof_vel,
                                "target_dof_pos": filtered.target,
                                "raw_action": np.zeros(self.num_actions, dtype=np.float32),
                                "last_action": np.zeros(self.num_actions, dtype=np.float32),
                                "tau_est": tau_est,
                                "action_mimic": np.zeros(self.n_mimic_obs, dtype=np.float32),
                                "rpy": rpy, "ang_vel": ang_vel,
                                "risk_score": 0.0, "mode": self.mode,
                                "redis_age": ss_redis_age, "loop_dt": loop_dt,
                            },
                        )
                        self.frame_id += 1
                        time.sleep(max(0.0, self.control_dt - loop_dt))
                        continue
                    # Mode changed to TRACKING: fall through to full path below.

                # ── Full path: TRACKING / RECOVERY ──
                mimic_raw, redis_age, source_frame_id, valid_input = self._read_mimic()
                if valid_input and redis_age <= REDIS_STALE_THRESHOLD:
                    self.redis_miss_count = 0
                    self.last_valid_mimic_full = mimic_raw.copy()
                    input_healthy = True
                else:
                    self.redis_miss_count += 1
                    input_healthy = False
                    mimic_raw = (
                        self.last_valid_mimic_full.copy()
                        if self.last_valid_mimic_full is not None
                        else DEFAULT_MIMIC_OBS["g1"].astype(np.float32).copy()
                    )

                action_mimic, wrist = extract_mimic_obs_to_body_and_wrist(mimic_raw)
                wrist_delta = np.clip(wrist - self.last_wrist, -0.08, 0.08)
                wrist = np.clip(self.last_wrist + wrist_delta, -1.9, 1.9)
                self.last_wrist = wrist.copy()

                obs_dof_vel = dof_vel.copy()
                obs_dof_vel[self.ankle_idx] = 0.0
                obs_proprio = np.concatenate([
                    ang_vel * self.ang_vel_scale,
                    rpy[:2],
                    (dof_pos - self.default_dof_pos) * self.dof_pos_scale,
                    obs_dof_vel * self.dof_vel_scale,
                    self.last_action,
                ]).astype(np.float32)
                self._publish_proprio(obs_proprio)
                obs_full = np.concatenate([action_mimic, obs_proprio]).astype(np.float32)
                # Deployed JIT contract:
                # current full frame (105)
                # + ten historical full frames (10 * 105) = 1155.
                obs_buf = np.concatenate([
                    obs_full,
                    np.asarray(self.proprio_history_buf).reshape(-1),
                ]).astype(np.float32)
                if obs_buf.size != POLICY_OBS_DIM:
                    raise RuntimeError(f"policy input is {obs_buf.size}, expected {POLICY_OBS_DIM}")
                self.proprio_history_buf.append(obs_full)

                with torch.no_grad():
                    output = self.policy(
                        torch.from_numpy(obs_buf).unsqueeze(0).to(self.device)
                    ).detach().cpu().numpy().reshape(-1)
                action_finite = bool(np.all(np.isfinite(output)))
                raw_action = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
                raw_action = np.clip(raw_action, -10.0, 10.0).astype(np.float32)
                if raw_action.size != self.num_actions:
                    raise RuntimeError(
                        f"policy output is {raw_action.size}, expected 23"
                    )
                previous_action = self.last_action.copy()
                self.last_action = raw_action.copy()
                policy_target = (
                    self.default_dof_pos
                    + raw_action * self.action_scale * self.deploy_action_scale
                )

                inference_dt = time.monotonic() - cycle_start
                risk_score, max_error, rms_error, torque_ratio = compute_risk_score(
                    rpy[0], rpy[1], ang_vel, dof_pos,
                    self.safety_filter.previous_target, tau_est,
                    self.torque_limits, redis_age, inference_dt, self.control_dt,
                )
                if self.recovery_test and 100 <= self.frame_id < 125:
                    risk_score = 1.0
                severe_reason = self._severe_fault(
                    rpy, [dof_pos, dof_vel, rpy, ang_vel, tau_est],
                    policy_target, redis_age,
                )
                if not action_finite:
                    severe_reason = "non-finite policy action"
                self._update_state_machine(
                    risk_score, input_healthy, severe_reason
                )

                desired, action_ramp = self._desired_for_mode(policy_target)
                filtered = self.safety_filter.apply(desired, action_ramp)
                if filtered.replaced_nonfinite:
                    self._set_mode("SAFE_STAND", "safety filter replaced NaN/Inf")
                self._send_action(filtered.target, wrist)

                loop_dt = time.monotonic() - cycle_start
                now_wall = time.time()
                self.logger.append(
                    {
                        "timestamp": f"{now_wall:.6f}",
                        "frame_id": source_frame_id,
                        "loop_dt": f"{loop_dt:.6f}",
                        "redis_age": f"{redis_age:.6f}",
                        "mode": self.mode,
                        "risk_score": f"{risk_score:.6f}",
                        "roll": f"{rpy[0]:.6f}",
                        "pitch": f"{rpy[1]:.6f}",
                        "yaw": f"{rpy[2]:.6f}",
                        "max_tracking_error": f"{max_error:.6f}",
                        "rms_tracking_error": f"{rms_error:.6f}",
                        "max_torque_ratio": f"{torque_ratio:.6f}",
                        "ramp": f"{self.ramp:.6f}",
                        "deploy_action_scale": f"{self.deploy_action_scale:.6f}",
                    },
                    {
                        "dof_pos": dof_pos, "dof_vel": dof_vel,
                        "target_dof_pos": filtered.target,
                        "raw_action": raw_action, "last_action": previous_action,
                        "tau_est": tau_est, "action_mimic": action_mimic,
                        "rpy": rpy, "ang_vel": ang_vel,
                        "risk_score": risk_score, "mode": self.mode,
                        "redis_age": redis_age, "loop_dt": loop_dt,
                    },
                )
                self.frame_id += 1
                time.sleep(max(0.0, self.control_dt - loop_dt))
        except KeyboardInterrupt:
            print("\n[INFO] Keyboard interrupt.")
        except Exception as exc:
            print(f"[ERROR] Main loop: {exc}")
            traceback.print_exc()
        finally:
            # Flush logs to disk before cleanup (cleanup may block or crash).
            self.logger.close()
            self._cleanup()

    def _blend_to_default_pose(self):
        print("[SHUTDOWN] Blending to default pose.")
        steps = max(1, int(SHUTDOWN_BLEND_TIME / self.control_dt))
        for _ in range(steps):
            target = self.safety_filter.apply(self.default_dof_pos, 1.0).target
            self._send_action(target, np.zeros(2, dtype=np.float32))
            time.sleep(self.control_dt)

    def _hold_default_pose_until_exit(self):
        print("[HOLD] Holding default pose. Press Select or Ctrl+C again to exit.")
        while True:
            if (not self.dry_run
                    and self.env is not None
                    and self.env.remote_controller.button[KeyMap.select] == 1):
                print("[HOLD] Select pressed. Exiting hold.")
                break
            target = self.safety_filter.apply(self.default_dof_pos, 1.0).target
            self._send_action(target, np.zeros(2, dtype=np.float32))
            time.sleep(self.control_dt)

    def _cleanup(self):
        try:
            if self.robot_ready and not self.dry_run:
                self._blend_to_default_pose()

                # Save logs before entering long hold mode.
                self.logger.close()

                if self.shutdown_behavior == "hold":
                    self._hold_default_pose_until_exit()
                elif self.shutdown_behavior == "zero_torque":
                    print("[SHUTDOWN] Entering zero torque state.")
                    self.env.zero_torque_state()
            else:
                self.logger.close()

        except KeyboardInterrupt:
            print("\n[INFO] Hold interrupted by user.")
        finally:
            # close() is idempotent (guarded by self.csv_file is None).
            self.logger.close()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Safe TWIST G1 real-robot controller (recommended)"
    )
    parser.add_argument(
        "--policy_path",
        help="1155-input, 23-output TorchScript policy (required except stand-test).",
    )
    parser.add_argument(
        "--config_path",
        default=os.path.join(here, "robot_control/configs/g1.yaml"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--net", default="eno1")
    parser.add_argument("--log_dir", default=os.path.join(here, "logs"))
    parser.add_argument("--stand-test", action="store_true")
    parser.add_argument("--replay-mimic")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--recovery-test", action="store_true")
    parser.add_argument(
        "--max-steps", type=int,
        help="Finite loop length for automated dry-run/replay validation.",
    )
    parser.add_argument("--stand-seconds", type=float, default=10.0)
    parser.add_argument(
        "--deploy-action-scale",
        type=float,
        default=1.0,
        help="Extra real-deployment action multiplier. Use 0.2 for first real replay.",
    )
    parser.add_argument(
        "--shutdown-behavior",
        choices=["hold", "zero_torque"],
        default="hold",
        help=(
            "Behavior after normal exit: hold default pose with PD commands "
            "or enter zero torque. Use hold for real replay."
        ),
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_prefix = (
        os.path.join(args.log_dir, f"real_deploy_{timestamp}")
        if args.log_dir else None
    )
    controller = RealTimePolicyControllerV2(
        policy_path=args.policy_path,
        config_path=args.config_path,
        device=args.device,
        net=args.net,
        log_prefix=log_prefix,
        stand_test=args.stand_test,
        dry_run=args.dry_run,
        replay_mimic=args.replay_mimic,
        recovery_test=args.recovery_test,
        max_steps=args.max_steps,
        stand_seconds=args.stand_seconds,
        deploy_action_scale=args.deploy_action_scale,
        shutdown_behavior=args.shutdown_behavior,
    )
    controller.run()


if __name__ == "__main__":
    main()