#!/usr/bin/env python3
"""Original TWIST real-robot controller with AnyAdapter history support."""

import argparse
import json
import os
import time
from collections import deque

import numpy as np
import redis
import torch

from data_utils.rot_utils import quatToEuler
from robot_control.common.remote_controller import KeyMap
from robot_control.config import Config
from robot_control.g1_wrapper import G1RealWorldEnv
from server_low_level_g1_real import extract_mimic_obs_to_body_and_wrist
from twist_anyadapter_runtime import AnyAdapterRuntime, AnyAdapterRuntimeConfig


BASE_OBS_DIM = 1155
NUM_ACTIONS = 23
ANYADAPTER_HISTORY_LEN = 20
ANYADAPTER_STATE_INDICES = (
    list(range(31, 36))
    + list(range(36, 59))
    + list(range(59, 82))
)
ANYADAPTER_FRAME_DIM = len(ANYADAPTER_STATE_INDICES) + NUM_ACTIONS
ANYADAPTER_OBS_DIM = (
    BASE_OBS_DIM + ANYADAPTER_HISTORY_LEN * ANYADAPTER_FRAME_DIM
)


def make_anyadapter_runtime(policy_path, device, ema_alpha):
    return AnyAdapterRuntime(
        AnyAdapterRuntimeConfig(
            base_obs_dim=BASE_OBS_DIM,
            num_actions=NUM_ACTIONS,
            history_len=ANYADAPTER_HISTORY_LEN,
            state_indices=ANYADAPTER_STATE_INDICES,
            policy_path=policy_path,
            device=device,
            action_clip=10.0,
            action_ema_alpha=ema_alpha,
        )
    )


class RealTimePolicyControllerRealAnyAdapter:
    """The original real controller with a 2635-D AnyAdapter policy input."""

    def __init__(
        self,
        policy_path,
        config_path,
        device="cuda",
        net="eno1",
        anyadapter_ema_alpha=0.0,
    ):
        self.redis_client = redis.Redis(host="localhost", port=6379, db=0)
        self.config = Config(config_path)

        print(
            f"[DDS] Initializing the original TWIST real controller on '{net}'. "
            "Waiting for LowState..."
        )
        self.env = G1RealWorldEnv(net=net, config=self.config)

        self.device = device
        self.anyadapter_runtime = make_anyadapter_runtime(
            policy_path,
            device,
            anyadapter_ema_alpha,
        )
        print(f"[AnyAdapter] Policy loaded from {policy_path}")
        print(
            f"[AnyAdapter] Contract verified: {BASE_OBS_DIM} + "
            f"{ANYADAPTER_HISTORY_LEN} * {ANYADAPTER_FRAME_DIM} = "
            f"{ANYADAPTER_OBS_DIM} -> {NUM_ACTIONS}"
        )

        self.num_actions = NUM_ACTIONS
        self.default_dof_pos = np.concatenate(
            [self.config.default_angles, self.config.arm_waist_target], axis=0
        ).astype(np.float32)
        if self.default_dof_pos.size != self.num_actions:
            raise ValueError("configured default pose must have 23 values")

        self.ang_vel_scale = 0.25
        self.dof_vel_scale = 0.05
        self.dof_pos_scale = 1.0
        self.ankle_idx = [4, 5, 10, 11]

        self.n_mimic_obs = 8 + self.num_actions
        self.n_proprio = self.n_mimic_obs + 3 + 2 + 3 * self.num_actions
        self.history_len = 10
        self.proprio_history_buf = deque(maxlen=self.history_len)
        for _ in range(self.history_len):
            self.proprio_history_buf.append(
                np.zeros(self.n_proprio, dtype=np.float32)
            )

        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.control_dt = self.config.control_dt
        self.action_scale = self.config.action_scale

    def reset_robot(self):
        print("Entering zero torque state, waiting for user to enter `start` ...")
        self.env.zero_torque_state()

        print("Press START on remote to move to default position ...")
        self.env.move_to_default_pos()

        print("Now in default position, press A to continue ...")
        self.env.default_pos_state()

        print("Robot will hold default pos. If needed, do other checks here.")

    def _read_mimic(self):
        action_mimic_json = self.redis_client.get("action_mimic_g1")
        if action_mimic_json is None:
            raise RuntimeError("cannot get action_mimic from redis")
        payload = json.loads(action_mimic_json)
        if isinstance(payload, dict):
            payload = payload.get("action_mimic")
        if payload is None:
            raise RuntimeError("action_mimic redis message has no action_mimic field")
        mimic = np.asarray(payload, dtype=np.float32).reshape(-1)
        if mimic.size != 33:
            raise RuntimeError(
                f"action_mimic must contain 33 values, got {mimic.size}"
            )
        return extract_mimic_obs_to_body_and_wrist(mimic)

    def run(self):
        self.reset_robot()
        self.anyadapter_runtime.reset()
        print("Begin AnyAdapter policy loop. Press [Select] on remote to exit.")

        try:
            while True:
                t_start = time.time()

                if self.env.remote_controller.button[KeyMap.select] == 1:
                    print("Select pressed, exiting main loop.")
                    break

                dof_pos, dof_vel, quat, ang_vel = self.env.get_robot_state()
                rpy = quatToEuler(quat)

                obs_dof_vel = dof_vel.copy()
                obs_dof_vel[self.ankle_idx] = 0.0
                obs_proprio = np.concatenate([
                    ang_vel * self.ang_vel_scale,
                    rpy[:2],
                    (dof_pos - self.default_dof_pos) * self.dof_pos_scale,
                    obs_dof_vel * self.dof_vel_scale,
                    self.last_action,
                ]).astype(np.float32)

                self.redis_client.set(
                    "state_body_g1", json.dumps(obs_proprio.tolist())
                )
                action_mimic, wrist_dof_pos = self._read_mimic()

                obs_full = np.concatenate([
                    action_mimic,
                    obs_proprio,
                ]).astype(np.float32)
                obs_hist = np.asarray(
                    self.proprio_history_buf, dtype=np.float32
                ).reshape(-1)
                base_obs = np.concatenate([obs_full, obs_hist]).astype(np.float32)
                if base_obs.size != BASE_OBS_DIM:
                    raise RuntimeError(
                        f"base policy observation has {base_obs.size} values, "
                        f"expected {BASE_OBS_DIM}"
                    )
                self.proprio_history_buf.append(obs_full)

                raw_action = self.anyadapter_runtime.act(base_obs)
                if raw_action.size != self.num_actions:
                    raise RuntimeError(
                        f"policy output has {raw_action.size} values, "
                        f"expected {self.num_actions}"
                    )
                if not np.all(np.isfinite(raw_action)):
                    raise RuntimeError("AnyAdapter policy returned NaN or Inf")
                self.last_action = raw_action.copy()

                raw_action = np.clip(raw_action, -10.0, 10.0)
                target_dof_pos = (
                    self.default_dof_pos + raw_action * self.action_scale
                )
                self.env.send_robot_action(
                    target_dof_pos,
                    kp_scale=1.0,
                    kd_scale=1.0,
                    left_wrist_roll=wrist_dof_pos[0],
                    right_wrist_roll=wrist_dof_pos[1],
                )

                elapsed = time.time() - t_start
                if elapsed < self.control_dt:
                    time.sleep(self.control_dt - elapsed)
        except Exception as exc:
            print(f"Error in main loop: {exc}")
        finally:
            self.env.zero_torque_state()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Original TWIST G1 real controller with AnyAdapter V4"
    )
    parser.add_argument("--policy_path", required=True)
    parser.add_argument(
        "--config_path",
        default=os.path.join(here, "robot_control/configs/g1.yaml"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--net", default="eno1")
    parser.add_argument(
        "--anyadapter-ema-alpha",
        "--anyadapter_ema_alpha",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Verify the 2635 -> 23 JIT contract without Unitree communication.",
    )
    args = parser.parse_args()

    if args.probe_only:
        runtime = make_anyadapter_runtime(
            args.policy_path,
            args.device,
            args.anyadapter_ema_alpha,
        )
        action = runtime.act(np.zeros(BASE_OBS_DIM, dtype=np.float32))
        print(
            f"[AnyAdapter] Probe passed: {runtime.policy_obs_dim} -> "
            f"{action.size}, finite={bool(np.all(np.isfinite(action)))}"
        )
        return

    controller = RealTimePolicyControllerRealAnyAdapter(
        policy_path=args.policy_path,
        config_path=args.config_path,
        device=args.device,
        net=args.net,
        anyadapter_ema_alpha=args.anyadapter_ema_alpha,
    )
    controller.run()


if __name__ == "__main__":
    main()
