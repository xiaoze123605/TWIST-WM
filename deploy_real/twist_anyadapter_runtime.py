"""
Runtime wrapper for TWIST + AnyAdapter JIT policy.

Drop into:
    TWIST/deploy_real/twist_anyadapter_runtime.py

Use this from server_low_level_g1_real.py or server_low_level_g1_sim.py after you
export the adapter actor to JIT.  It maintains the history buffer required by
the adapter policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import torch


@dataclass
class AnyAdapterRuntimeConfig:
    base_obs_dim: int
    num_actions: int
    history_len: int
    state_indices: Sequence[int]
    policy_path: str
    device: str = "cpu"
    action_clip: float = 1.0
    adapter_input_is_augmented: bool = True
    adapter_context_dim: int = 0
    fill_history_on_first_observation: bool = False
    tracking_error_history_len: int = 0
    tracking_ref_dof_vel_filter_alpha: float = 0.5
    tracking_ref_dof_vel_clip: float = 20.0
    control_dt: float = 0.02
    fill_tracking_history_on_first_observation: bool = True
    # EMA smoothing factor for action output (0 = no smoothing, must be < 1).
    # 0.3–0.5 recommended to suppress high-frequency jitter from adapter.
    action_ema_alpha: float = 0.0


class AnyAdapterRuntime:
    def __init__(self, cfg: AnyAdapterRuntimeConfig):
        self.cfg = cfg
        if cfg.base_obs_dim <= 0 or cfg.num_actions <= 0 or cfg.history_len <= 0:
            raise ValueError("base_obs_dim, num_actions, and history_len must be positive")
        if not 0.0 <= float(cfg.action_ema_alpha) < 1.0:
            raise ValueError("action_ema_alpha must be in [0, 1)")
        if cfg.tracking_error_history_len < 0:
            raise ValueError("tracking_error_history_len must be non-negative")
        if not 0.0 <= float(cfg.tracking_ref_dof_vel_filter_alpha) <= 1.0:
            raise ValueError("tracking_ref_dof_vel_filter_alpha must be in [0, 1]")
        if cfg.tracking_ref_dof_vel_clip <= 0.0:
            raise ValueError("tracking_ref_dof_vel_clip must be positive")
        if cfg.control_dt <= 0.0:
            raise ValueError("control_dt must be positive")
        self.device = torch.device(cfg.device)
        self.policy = torch.jit.load(cfg.policy_path, map_location=self.device)
        self.policy.eval()
        self.state_indices = np.asarray(cfg.state_indices, dtype=np.int64)
        if self.state_indices.size == 0:
            raise ValueError("state_indices must not be empty")
        if self.state_indices.min() < 0 or self.state_indices.max() >= cfg.base_obs_dim:
            raise ValueError("state_indices must index only the base observation")
        self.hist_state_dim = len(self.state_indices)
        self.frame_dim = self.hist_state_dim + cfg.num_actions
        self.tracking_error_frame_dim = 2 * cfg.num_actions + 7
        self.policy_obs_dim = (
            cfg.base_obs_dim
            + cfg.history_len * self.frame_dim
            + cfg.tracking_error_history_len * self.tracking_error_frame_dim
            + cfg.adapter_context_dim
        )
        self.history = np.zeros((cfg.history_len, self.frame_dim), dtype=np.float32)
        self.tracking_error_history = np.zeros(
            (cfg.tracking_error_history_len, self.tracking_error_frame_dim),
            dtype=np.float32,
        )
        self._tracking_prev_ref_dof_pos = np.zeros(
            (cfg.num_actions,), dtype=np.float32
        )
        self._tracking_ref_dof_vel_est = np.zeros(
            (cfg.num_actions,), dtype=np.float32
        )
        self._tracking_reference_initialized = False
        self._tracking_history_initialized = False
        self.prev_action = np.zeros((cfg.num_actions,), dtype=np.float32)
        # EMA smoothing state
        self._ema_action: Optional[np.ndarray] = None
        self._ema_alpha = float(cfg.action_ema_alpha)
        self._history_initialized = False
        with torch.no_grad():
            probe = self.policy(
                torch.zeros(1, self.policy_obs_dim, device=self.device)
            )
        if probe.numel() != cfg.num_actions:
            raise ValueError(
                f"Policy output has {probe.numel()} values, expected {cfg.num_actions}."
            )

    def reset(self):
        self.history[:] = 0.0
        self.tracking_error_history[:] = 0.0
        self._tracking_prev_ref_dof_pos[:] = 0.0
        self._tracking_ref_dof_vel_est[:] = 0.0
        self._tracking_reference_initialized = False
        self._tracking_history_initialized = False
        self.prev_action[:] = 0.0
        self._ema_action = None
        self._history_initialized = False

    @staticmethod
    def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
        return np.arctan2(np.sin(angle), np.cos(angle))

    def _build_tracking_error_frame(
        self,
        tracking_reference: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        root_linear_velocity: np.ndarray,
        root_yaw_velocity: float,
        roll_pitch: np.ndarray,
    ) -> np.ndarray:
        """Match the DTERA training-time 53-D tracking-error definition."""
        n = self.cfg.num_actions
        reference = np.asarray(tracking_reference, dtype=np.float32).reshape(-1)
        dof_pos = np.asarray(dof_pos, dtype=np.float32).reshape(-1)
        dof_vel = np.asarray(dof_vel, dtype=np.float32).reshape(-1)
        root_linear_velocity = np.asarray(
            root_linear_velocity, dtype=np.float32
        ).reshape(-1)
        roll_pitch = np.asarray(roll_pitch, dtype=np.float32).reshape(-1)
        if reference.size != 8 + n:
            raise ValueError(
                f"Expected tracking reference dim {8 + n}, got {reference.size}."
            )
        if dof_pos.size != n or dof_vel.size != n:
            raise ValueError("dof_pos and dof_vel must match num_actions")
        if root_linear_velocity.size != 3 or roll_pitch.size != 2:
            raise ValueError(
                "root_linear_velocity and roll_pitch must have 3 and 2 values"
            )

        ref_dof_pos = reference[8 : 8 + n]
        if self._tracking_reference_initialized:
            raw_ref_dof_vel = (
                ref_dof_pos - self._tracking_prev_ref_dof_pos
            ) / float(self.cfg.control_dt)
            raw_ref_dof_vel = np.clip(
                raw_ref_dof_vel,
                -float(self.cfg.tracking_ref_dof_vel_clip),
                float(self.cfg.tracking_ref_dof_vel_clip),
            )
            alpha = float(self.cfg.tracking_ref_dof_vel_filter_alpha)
            ref_dof_vel = (
                alpha * self._tracking_ref_dof_vel_est
                + (1.0 - alpha) * raw_ref_dof_vel
            )
        else:
            # Training initializes reference velocity to the measured velocity
            # on a reset, preventing a spurious first-frame derivative spike.
            ref_dof_vel = dof_vel.copy()

        self._tracking_prev_ref_dof_pos[:] = ref_dof_pos
        self._tracking_ref_dof_vel_est[:] = ref_dof_vel
        self._tracking_reference_initialized = True

        q_error = ref_dof_pos - dof_pos
        dq_error = ref_dof_vel - dof_vel
        root_linear_error = reference[4:7] - root_linear_velocity
        root_yaw_error = np.asarray(
            [reference[7] - float(root_yaw_velocity)], dtype=np.float32
        )
        roll_pitch_error = self._wrap_to_pi(reference[1:3] - roll_pitch)
        phase_lag = np.asarray([
            np.clip(
                float(np.dot(q_error, ref_dof_vel))
                / (float(np.dot(ref_dof_vel, ref_dof_vel)) + 1e-6),
                -0.2,
                0.2,
            )
        ], dtype=np.float32)
        frame = np.concatenate([
            q_error,
            dq_error,
            root_linear_error,
            root_yaw_error,
            roll_pitch_error,
            phase_lag,
        ]).astype(np.float32)
        if frame.size != self.tracking_error_frame_dim:
            raise RuntimeError(
                f"tracking error frame has {frame.size} values, expected "
                f"{self.tracking_error_frame_dim}"
            )
        return frame

    def _build_augmented_obs(
        self,
        base_obs: np.ndarray,
        adapter_context: Optional[np.ndarray] = None,
        tracking_reference: Optional[np.ndarray] = None,
        dof_pos: Optional[np.ndarray] = None,
        dof_vel: Optional[np.ndarray] = None,
        root_linear_velocity: Optional[np.ndarray] = None,
        root_yaw_velocity: Optional[float] = None,
        roll_pitch: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        base_obs = np.asarray(base_obs, dtype=np.float32).reshape(-1)
        if base_obs.shape[0] != self.cfg.base_obs_dim:
            raise ValueError(
                f"Expected base observation dim {self.cfg.base_obs_dim}, "
                f"got {base_obs.shape[0]}."
            )
        dyn_state = base_obs[self.state_indices].astype(np.float32)
        new_frame = np.concatenate([dyn_state, self.prev_action], axis=0)
        if self.cfg.fill_history_on_first_observation and not self._history_initialized:
            self.history[:] = new_frame
        else:
            self.history[:-1] = self.history[1:]
            self.history[-1] = new_frame
        self._history_initialized = True
        if self.cfg.tracking_error_history_len > 0:
            tracking_inputs = (
                tracking_reference,
                dof_pos,
                dof_vel,
                root_linear_velocity,
                root_yaw_velocity,
                roll_pitch,
            )
            if any(value is None for value in tracking_inputs):
                raise ValueError(
                    "DTERA policy requires tracking_reference, dof_pos, dof_vel, "
                    "root_linear_velocity, root_yaw_velocity, and roll_pitch."
                )
            tracking_frame = self._build_tracking_error_frame(
                tracking_reference,
                dof_pos,
                dof_vel,
                root_linear_velocity,
                root_yaw_velocity,
                roll_pitch,
            )
            if (
                self.cfg.fill_tracking_history_on_first_observation
                and self.tracking_error_history.shape[0] > 0
                and not getattr(self, "_tracking_history_initialized", False)
            ):
                self.tracking_error_history[:] = tracking_frame
            else:
                self.tracking_error_history[:-1] = self.tracking_error_history[1:]
                self.tracking_error_history[-1] = tracking_frame
            self._tracking_history_initialized = True
        if self.cfg.adapter_context_dim > 0:
            if adapter_context is None:
                raise ValueError(
                    f"This policy requires {self.cfg.adapter_context_dim} adapter context values."
                )
            adapter_context = np.asarray(adapter_context, dtype=np.float32).reshape(-1)
            if adapter_context.shape[0] != self.cfg.adapter_context_dim:
                raise ValueError(
                    f"Expected adapter context dim {self.cfg.adapter_context_dim}, "
                    f"got {adapter_context.shape[0]}."
                )
        else:
            adapter_context = np.zeros(0, dtype=np.float32)
        return np.concatenate([
            base_obs.astype(np.float32),
            self.history.reshape(-1),
            self.tracking_error_history.reshape(-1),
            adapter_context,
        ], axis=0)

    @torch.no_grad()
    def act(
        self,
        base_obs: np.ndarray,
        adapter_context: Optional[np.ndarray] = None,
        **tracking_inputs,
    ) -> np.ndarray:
        obs = self._build_augmented_obs(
            base_obs, adapter_context, **tracking_inputs
        )
        obs_t = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
        action = self.policy(obs_t).squeeze(0).detach().cpu().numpy().astype(np.float32)
        action = action.reshape(-1)
        if action.shape[0] != self.cfg.num_actions:
            raise RuntimeError(
                f"Policy output has {action.shape[0]} values, expected {self.cfg.num_actions}."
            )
        action = np.clip(action, -self.cfg.action_clip, self.cfg.action_clip)

        # EMA smoothing to suppress high-frequency adapter jitter.
        if self._ema_alpha > 0.0:
            if self._ema_action is None:
                self._ema_action = action.copy()
            else:
                self._ema_action = (
                    self._ema_alpha * self._ema_action + (1.0 - self._ema_alpha) * action
                )
            action = self._ema_action.copy()

        self.prev_action = action.copy()
        return action
