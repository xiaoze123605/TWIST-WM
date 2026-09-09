"""
TWIST + AnyAdapter-style policy module.

Drop this file into:
    TWIST/rsl_rl/rsl_rl/modules/actor_critic_twist_anyadapter.py

Design:
    action = frozen_base_actor(obs_base) + adapter(obs_base, history_embedding)

The history encoder and world model are trained with an auxiliary forward-dynamics
loss.  The frozen base actor is loaded from a TorchScript/JIT TWIST student actor,
so this module can be added without rewriting the original TWIST actor.

Important assumptions:
    1. The environment provides an augmented observation:
           [base_obs, history_flat]
       where history_flat is history_len * history_frame_dim.
    2. Each history frame is:
           [hist_state, previous_action]
       where hist_state is a selected subset of base_obs used for dynamics ID.
    3. base_actor_jit_path points to an exported TWIST student JIT policy.

You must set base_obs_dim, history_len, history_frame_dim, hist_state_dim and
wm_target_indices in the TWIST config.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal


Normal.set_default_validate_args = False


def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "lrelu":
        return nn.LeakyReLU()
    raise ValueError(f"Unsupported activation: {name}")


def mlp(in_dim: int, hidden_dims: Sequence[int], out_dim: int, activation: str = "elu") -> nn.Sequential:
    act = get_activation(activation)
    layers = []
    last = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(last, h), act.__class__()]
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


def zero_init_last_linear(module: nn.Module) -> None:
    """Make a network initially output almost zero, preserving the base policy."""
    linears = [m for m in module.modules() if isinstance(m, nn.Linear)]
    if not linears:
        return
    nn.init.zeros_(linears[-1].weight)
    nn.init.zeros_(linears[-1].bias)


class HistoryEncoder(nn.Module):
    """Encode recent robot response history into a dynamics embedding z_t."""

    def __init__(
        self,
        history_frame_dim: int,
        history_len: int,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        activation: str = "elu",
        use_conv: bool = True,
    ) -> None:
        super().__init__()
        self.history_frame_dim = history_frame_dim
        self.history_len = history_len
        self.latent_dim = latent_dim
        self.use_conv = use_conv
        act = get_activation(activation)
        if use_conv:
            self.net = nn.Sequential(
                nn.Conv1d(history_frame_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
                act.__class__(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
                act.__class__(),
                nn.Flatten(),
                nn.Linear(hidden_dim * ((history_len + 3) // 4), latent_dim),
            )
        else:
            self.net = mlp(history_frame_dim * history_len, [hidden_dim, hidden_dim], latent_dim, activation)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        # history: [B, H, D]
        if history.ndim != 3:
            raise ValueError(f"history must be [B,H,D], got {tuple(history.shape)}")
        if self.use_conv:
            return self.net(history.transpose(1, 2))
        return self.net(history.reshape(history.shape[0], -1))


class WorldModel(nn.Module):
    """
    One-step forward dynamics proxy model.

    It predicts current/next selected robot state from previous selected state,
    previous action and dynamics embedding.  This is not a planner; it only makes
    the embedding dynamics-aware.
    """

    def __init__(
        self,
        hist_state_dim: int,
        num_actions: int,
        latent_dim: int,
        target_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        activation: str = "elu",
        predict_delta: bool = True,
    ) -> None:
        super().__init__()
        self.hist_state_dim = hist_state_dim
        self.num_actions = num_actions
        self.latent_dim = latent_dim
        self.target_dim = target_dim
        self.predict_delta = predict_delta
        self.net = mlp(hist_state_dim + num_actions + latent_dim, hidden_dims, target_dim, activation)

    def forward(self, prev_state: torch.Tensor, prev_action: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        pred = self.net(torch.cat([prev_state, prev_action, z], dim=-1))
        if self.predict_delta:
            # If target_dim <= hist_state_dim, interpret prediction as state delta.
            base = prev_state[..., : self.target_dim]
            pred = base + pred
        return pred


class ResidualAdapter(nn.Module):
    """Small zero-initialized residual correction branch."""

    def __init__(
        self,
        base_obs_dim: int,
        latent_dim: int,
        num_actions: int,
        extra_input_dim: int = 0,
        hidden_dims: Sequence[int] = (128, 128),
        activation: str = "elu",
        delta_scale: float = 0.25,
    ) -> None:
        super().__init__()
        self.delta_scale = float(delta_scale)
        self.extra_input_dim = int(extra_input_dim)
        self.net = mlp(base_obs_dim + latent_dim + self.extra_input_dim, hidden_dims, num_actions, activation)
        zero_init_last_linear(self.net)

    def forward(self, base_obs: torch.Tensor, z: torch.Tensor, extra: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.extra_input_dim > 0:
            if extra is None:
                extra = base_obs.new_zeros(base_obs.shape[0], self.extra_input_dim)
            x = torch.cat([base_obs, z, extra], dim=-1)
        else:
            x = torch.cat([base_obs, z], dim=-1)
        return self.delta_scale * torch.tanh(self.net(x))


class DynamicsResidualBranch(nn.Module):
    """Dynamics-only residual branch conditioned on state, base action and z_dyn."""

    def __init__(
        self,
        dynamics_input_dim: int,
        latent_dim: int,
        num_actions: int,
        hidden_dims: Sequence[int] = (128, 128),
        activation: str = "elu",
        delta_scale: float = 0.05,
    ) -> None:
        super().__init__()
        self.delta_scale = float(delta_scale)
        self.net = mlp(dynamics_input_dim + latent_dim, hidden_dims, num_actions, activation)
        zero_init_last_linear(self.net)

    def forward(self, dynamics_features: torch.Tensor, z_dyn: torch.Tensor) -> torch.Tensor:
        x = torch.cat([dynamics_features, z_dyn], dim=-1)
        return self.delta_scale * torch.tanh(self.net(x))


class TrackingResidualBranch(nn.Module):
    """Tracking-only residual branch with no access to the dynamics latent."""

    def __init__(
        self,
        tracking_feature_dim: int,
        num_actions: int,
        context_dim: int = 0,
        hidden_dims: Sequence[int] = (128, 128),
        activation: str = "elu",
        delta_scale: float = 0.05,
    ) -> None:
        super().__init__()
        self.delta_scale = float(delta_scale)
        self.context_dim = int(context_dim)
        input_dim = tracking_feature_dim + num_actions + self.context_dim
        self.net = mlp(input_dim, hidden_dims, num_actions, activation)
        zero_init_last_linear(self.net)

    def forward(
        self,
        tracking_features: torch.Tensor,
        base_action: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        parts = [tracking_features, base_action]
        if self.context_dim > 0:
            if context is None:
                context = base_action.new_zeros(base_action.shape[0], self.context_dim)
            parts.append(context)
        return self.delta_scale * torch.tanh(self.net(torch.cat(parts, dim=-1)))


class DualResidualAdapter(nn.Module):
    """Container that keeps both residual branches visible to PPO as one module."""

    def __init__(
        self,
        dynamics_input_dim: int,
        tracking_feature_dim: int,
        latent_dim: int,
        num_actions: int,
        context_dim: int = 0,
        hidden_dims: Sequence[int] = (128, 128),
        activation: str = "elu",
        dynamics_delta_scale: float = 0.05,
        tracking_delta_scale: float = 0.05,
    ) -> None:
        super().__init__()
        self.dynamics_branch = DynamicsResidualBranch(
            dynamics_input_dim=dynamics_input_dim,
            latent_dim=latent_dim,
            num_actions=num_actions,
            hidden_dims=hidden_dims,
            activation=activation,
            delta_scale=dynamics_delta_scale,
        )
        self.tracking_branch = TrackingResidualBranch(
            tracking_feature_dim=tracking_feature_dim,
            num_actions=num_actions,
            context_dim=context_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            delta_scale=tracking_delta_scale,
        )

    def forward(
        self,
        dynamics_features: torch.Tensor,
        z_dyn: torch.Tensor,
        tracking_features: torch.Tensor,
        base_action: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        delta_dyn = self.dynamics_branch(dynamics_features, z_dyn)
        delta_err = self.tracking_branch(tracking_features, base_action, context)
        return delta_dyn, delta_err


class TwistAnyAdapterActorCritic(nn.Module):
    """
    RSL-RL compatible actor-critic wrapper.

    Actor:
        frozen TWIST JIT base actor + trainable residual adapter.
    Critic:
        trainable MLP using critic observations.  You may replace this with the
        original TWIST critic if you want a stricter continuation of training.
    """

    is_recurrent = False

    def __init__(
        self,
        num_prop: Optional[int] = None,
        num_critic_obs: Optional[int] = None,
        num_priv_latent: int = 0,
        num_hist: int = 0,
        num_actions: Optional[int] = None,
        base_actor_jit_path: Optional[str] = None,
        base_obs_dim: int = -1,
        history_len: int = 20,
        history_frame_dim: Optional[int] = None,
        hist_state_dim: Optional[int] = None,
        wm_target_indices: Optional[Sequence[int]] = None,
        latent_dim: int = 32,
        adapter_hidden_dims: Sequence[int] = (128, 128),
        critic_hidden_dims: Sequence[int] = (512, 256, 128),
        world_model_hidden_dims: Sequence[int] = (256, 256),
        activation: str = "elu",
        init_noise_std: float = 0.2,
        fix_action_std: bool = False,
        action_delta_scale: float = 0.25,
        adapter_gain: float = 1.0,
        use_dual_branch_adapter: bool = False,
        dynamics_branch_gain: float = 1.0,
        tracking_branch_gain: float = 1.0,
        adapter_branch_mode: str = "full",
        dynamics_action_delta_scale: Optional[float] = None,
        tracking_action_delta_scale: Optional[float] = None,
        default_ref_dof_pos: Optional[Sequence[float]] = None,
        use_tracking_error_adapter_input: bool = False,
        compact_adapter_input: bool = False,
        history_policy_grad_scale: float = 0.0,
        adapter_context_dim: int = 0,
        use_conv_history: bool = True,
        freeze_base: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        if num_critic_obs is None:
            num_critic_obs = kwargs.pop("num_critic_observations", None)
        if num_critic_obs is None:
            num_critic_obs = kwargs.pop("num_observations", None)
        if base_actor_jit_path is None:
            raise ValueError("base_actor_jit_path must point to an exported TWIST student JIT actor.")
        if base_obs_dim <= 0:
            raise ValueError("base_obs_dim must be the original TWIST student obs dim before AnyAdapter history.")
        if num_actions is None:
            raise ValueError("num_actions must be provided by the runner.")
        if num_critic_obs is None:
            raise ValueError("num_critic_obs or num_critic_observations must be provided by the runner.")
        self.base_actor_jit_path = str(base_actor_jit_path)
        self.num_actions = num_actions
        self.base_obs_dim = int(base_obs_dim)
        self.history_len = int(history_len)
        self.history_frame_dim = int(history_frame_dim or (self.base_obs_dim + num_actions))
        self.hist_state_dim = int(hist_state_dim or (self.history_frame_dim - num_actions))
        if self.history_len <= 0:
            raise ValueError("history_len must be positive for AnyAdapter.")
        if self.history_frame_dim <= 0:
            raise ValueError("history_frame_dim must be len(anyadapter_state_indices) + num_actions.")
        if self.hist_state_dim <= 0:
            raise ValueError("hist_state_dim must be len(anyadapter_state_indices).")
        self.latent_dim = int(latent_dim)
        self.fix_action_std = bool(fix_action_std)
        self.wm_target_indices = None if wm_target_indices is None else torch.as_tensor(wm_target_indices, dtype=torch.long)
        self.adapter_gain = float(adapter_gain)
        self.use_dual_branch_adapter = bool(use_dual_branch_adapter)
        self.dynamics_branch_gain = float(dynamics_branch_gain)
        self.tracking_branch_gain = float(tracking_branch_gain)
        # Inference-only branch mask used by the dual-branch ablation study.
        # Only the action forward (actor_mean / act_inference / get_adapter_delta)
        # is affected; PPO losses, gradients, the world model and
        # adapter_regularization_loss (which reads the raw branch outputs) are
        # all left untouched.  "full" reproduces the default dual forward.
        self.adapter_branch_mode = str(adapter_branch_mode)
        if self.adapter_branch_mode not in ("base_only", "full", "dyn_only", "err_only"):
            raise ValueError(
                "adapter_branch_mode must be one of 'base_only', 'full', 'dyn_only', "
                f"'err_only', got {self.adapter_branch_mode!r}."
            )
        self.use_tracking_error_adapter_input = bool(use_tracking_error_adapter_input)
        self.compact_adapter_input = bool(compact_adapter_input)
        self.history_policy_grad_scale = float(history_policy_grad_scale)
        self.adapter_context_dim = int(adapter_context_dim)
        if self.adapter_context_dim < 0:
            raise ValueError("adapter_context_dim must be non-negative.")
        if not 0.0 <= self.history_policy_grad_scale <= 1.0:
            raise ValueError("history_policy_grad_scale must be in [0, 1].")
        if self.use_dual_branch_adapter and not self.use_tracking_error_adapter_input:
            raise ValueError(
                "use_tracking_error_adapter_input must be True when use_dual_branch_adapter is enabled."
            )
        self.tracking_error_feature_dim = int(num_actions + 6) if self.use_tracking_error_adapter_input else 0
        if default_ref_dof_pos is None:
            default_ref_dof_pos = [0.0] * num_actions
        if len(default_ref_dof_pos) != num_actions:
            raise ValueError(f"default_ref_dof_pos must have {num_actions} values, got {len(default_ref_dof_pos)}.")
        self.register_buffer("default_ref_dof_pos", torch.as_tensor(default_ref_dof_pos, dtype=torch.float32))

        self.base_actor = torch.jit.load(base_actor_jit_path, map_location="cpu")
        self.base_actor.eval()
        if freeze_base:
            for p in self.base_actor.parameters():
                p.requires_grad_(False)

        self.history_encoder = HistoryEncoder(
            history_frame_dim=self.history_frame_dim,
            history_len=self.history_len,
            latent_dim=latent_dim,
            hidden_dim=128,
            activation=activation,
            use_conv=use_conv_history,
        )
        adapter_policy_input_dim = (
            self.num_actions + self.hist_state_dim
            if self.compact_adapter_input
            else self.base_obs_dim
        )
        if self.use_dual_branch_adapter:
            # Compact features already contain [base_action, current_state].
            # In non-compact mode base_action is appended explicitly below.
            dynamics_input_dim = (
                adapter_policy_input_dim
                if self.compact_adapter_input
                else adapter_policy_input_dim + self.num_actions
            )
            self.adapter = DualResidualAdapter(
                dynamics_input_dim=dynamics_input_dim,
                tracking_feature_dim=self.tracking_error_feature_dim,
                latent_dim=latent_dim,
                num_actions=num_actions,
                context_dim=self.adapter_context_dim,
                hidden_dims=adapter_hidden_dims,
                activation=activation,
                dynamics_delta_scale=(
                    action_delta_scale
                    if dynamics_action_delta_scale is None
                    else dynamics_action_delta_scale
                ),
                tracking_delta_scale=(
                    action_delta_scale
                    if tracking_action_delta_scale is None
                    else tracking_action_delta_scale
                ),
            )
        else:
            self.adapter = ResidualAdapter(
                base_obs_dim=adapter_policy_input_dim,
                latent_dim=latent_dim,
                num_actions=num_actions,
                extra_input_dim=self.tracking_error_feature_dim + self.adapter_context_dim,
                hidden_dims=adapter_hidden_dims,
                activation=activation,
                delta_scale=action_delta_scale,
            )
        target_dim = len(wm_target_indices) if wm_target_indices is not None else self.hist_state_dim
        self.world_model = WorldModel(
            hist_state_dim=self.hist_state_dim,
            num_actions=num_actions,
            latent_dim=latent_dim,
            target_dim=target_dim,
            hidden_dims=world_model_hidden_dims,
            activation=activation,
            predict_delta=True,
        )
        # Per-component world model loss splits.
        # The target state selected by wm_target_indices is laid out as:
        #   [base_ang_vel(3), roll/pitch(2), dof_pos(23), dof_vel(23)] = 51 dims
        # Components  →  (start, end) in the 51-dim target vector.
        self.wm_component_splits = {
            "ang_vel":     (0, 3),   # base angular velocity
            "orientation": (3, 5),   # roll / pitch
            "dof_pos":     (5, 28),  # joint positions
            "dof_vel":     (28, 51), # joint velocities
        }
        self.wm_component_weights = {
            "ang_vel":     1.0,
            "orientation": 2.0,
            "dof_pos":     1.0,
            "dof_vel":     1.0,
        }
        self.critic = mlp(num_critic_obs, critic_hidden_dims, 1, activation)

        if self.fix_action_std:
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions), requires_grad=False)
        else:
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution: Optional[Normal] = None

    def split_obs(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        base_obs = obs[:, : self.base_obs_dim]
        hist_flat = obs[:, self.base_obs_dim : self.base_obs_dim + self.history_len * self.history_frame_dim]
        if hist_flat.numel() == 0:
            raise RuntimeError(
                "Observation does not contain history. Add AnyAdapterHistoryMixin to the TWIST env "
                "or check base_obs_dim/history_len/history_frame_dim."
            )
        history = hist_flat.reshape(obs.shape[0], self.history_len, self.history_frame_dim)
        return base_obs, history

    def _obs_slice_or_zeros(self, base_obs: torch.Tensor, start: int, width: int) -> torch.Tensor:
        if base_obs.shape[1] >= start + width:
            return base_obs[:, start : start + width]
        out = base_obs.new_zeros(base_obs.shape[0], width)
        if base_obs.shape[1] > start:
            available = base_obs.shape[1] - start
            out[:, :available] = base_obs[:, start:]
        return out

    def adapter_context(self, observations: torch.Tensor) -> Optional[torch.Tensor]:
        if self.adapter_context_dim == 0:
            return None
        context_start = self.base_obs_dim + self.history_len * self.history_frame_dim
        context_end = context_start + self.adapter_context_dim
        if observations.shape[1] < context_end:
            raise RuntimeError(
                f"Observation has {observations.shape[1]} dims, but adapter context "
                f"requires at least {context_end}."
            )
        return observations[:, context_start:context_end]

    def tracking_error_features(self, base_obs: torch.Tensor) -> torch.Tensor:
        """
        Reference-aware adapter-only features.

        These features are deliberately kept out of the history encoder and
        world model so z remains a dynamics embedding rather than a motion
        command embedding.  TWIST student obs layout used here:
            1:3   reference roll/pitch
            4:8   reference root velocity xyz + yaw velocity
            8:31  reference dof position
            34:36 actual roll/pitch
            36:59 actual dof position offset from default pose
        """
        ref_vel = self._obs_slice_or_zeros(base_obs, 4, 4)
        ref_roll_pitch = self._obs_slice_or_zeros(base_obs, 1, 2)
        actual_roll_pitch = self._obs_slice_or_zeros(base_obs, 34, 2)
        ref_dof_pos = self._obs_slice_or_zeros(base_obs, 8, self.num_actions)
        dof_pos_delta = self._obs_slice_or_zeros(base_obs, 36, self.num_actions)
        default_ref = self.default_ref_dof_pos.view(1, self.num_actions)
        actual_dof_pos = default_ref + dof_pos_delta
        return torch.cat([ref_vel, ref_roll_pitch - actual_roll_pitch, ref_dof_pos - actual_dof_pos], dim=-1)

    def encode_history_for_policy(self, history: torch.Tensor) -> torch.Tensor:
        z = self.history_encoder(history)
        if self.history_policy_grad_scale <= 0.0:
            return z.detach()
        # Forward values stay unchanged while PPO gradients are scaled down.
        z_detached = z.detach()
        return z_detached + self.history_policy_grad_scale * (z - z_detached)

    def encode_history_for_world_model(self, history: torch.Tensor) -> torch.Tensor:
        return self.history_encoder(history)

    def base_action(self, observations: torch.Tensor) -> torch.Tensor:
        base_obs, _ = self.split_obs(observations)
        with torch.no_grad():
            return self.base_actor(base_obs)

    def compact_adapter_features(
        self,
        base_obs: torch.Tensor,
        base_action: torch.Tensor,
    ) -> torch.Tensor:
        if self.wm_target_indices is None:
            current_state = base_obs[:, 31 : 31 + self.hist_state_dim]
        else:
            idx = self.wm_target_indices.to(base_obs.device)
            current_state = base_obs.index_select(dim=1, index=idx)
        return torch.cat([base_action, current_state], dim=-1)

    def action_delta_components(
        self,
        observations: torch.Tensor,
        detach_history: bool = True,
        base_action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the two raw branch outputs (legacy residual, zero in old mode)."""
        base_obs, history = self.split_obs(observations)
        if detach_history:
            z = self.encode_history_for_policy(history)
        else:
            z = self.encode_history_for_world_model(history)

        if base_action is None and (self.use_dual_branch_adapter or self.compact_adapter_input):
            base_action = self.base_action(observations)

        if self.use_dual_branch_adapter:
            if self.compact_adapter_input:
                dynamics_features = self.compact_adapter_features(base_obs, base_action)
            else:
                dynamics_features = torch.cat([base_obs, base_action], dim=-1)
            tracking_features = self.tracking_error_features(base_obs)
            context = self.adapter_context(observations)
            return self.adapter(
                dynamics_features,
                z,
                tracking_features,
                base_action,
                context,
            )

        adapter_obs = base_obs
        if self.compact_adapter_input:
            adapter_obs = self.compact_adapter_features(base_obs, base_action)
        extra_parts = []
        if self.use_tracking_error_adapter_input:
            extra_parts.append(self.tracking_error_features(base_obs))
        context = self.adapter_context(observations)
        if context is not None:
            extra_parts.append(context)
        extra = torch.cat(extra_parts, dim=-1) if extra_parts else None
        legacy_delta = self.adapter(adapter_obs, z, extra)
        return legacy_delta, torch.zeros_like(legacy_delta)

    def action_delta(
        self,
        observations: torch.Tensor,
        detach_history: bool = True,
        base_action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.adapter_branch_mode == "base_only":
            return observations.new_zeros(observations.shape[0], self.num_actions)
        delta_dyn, delta_err = self.action_delta_components(
            observations,
            detach_history=detach_history,
            base_action=base_action,
        )
        if not self.use_dual_branch_adapter:
            # Legacy single-residual modes keep their historical forward.
            return delta_dyn
        if self.adapter_branch_mode == "dyn_only":
            return self.dynamics_branch_gain * delta_dyn
        if self.adapter_branch_mode == "err_only":
            return self.tracking_branch_gain * delta_err
        return (
            self.dynamics_branch_gain * delta_dyn
            + self.tracking_branch_gain * delta_err
        )

    def actor_mean(self, observations: torch.Tensor) -> torch.Tensor:
        base_action = self.base_action(observations)
        if self.adapter_branch_mode == "base_only":
            return base_action
        delta = self.action_delta(
            observations,
            detach_history=True,
            base_action=base_action,
        )
        return base_action + self.adapter_gain * delta

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor_mean(observations)

    def get_adapter_delta(self, observations: torch.Tensor) -> torch.Tensor:
        return self.action_delta(observations, detach_history=True)

    def get_adapter_delta_components(
        self,
        observations: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.action_delta_components(observations, detach_history=True)

    def predict_world_model(self, observations: torch.Tensor, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        _, history = self.split_obs(observations)
        z = self.encode_history_for_world_model(history)
        prev_frame = history[:, -1]
        prev_state = prev_frame[:, : self.hist_state_dim]
        if actions is None:
            prev_action = prev_frame[:, self.hist_state_dim : self.hist_state_dim + self.num_actions]
        else:
            prev_action = actions
        return self.world_model(prev_state, prev_action, z)

    def update_distribution(self, observations: torch.Tensor) -> None:
        mean = self.actor_mean(observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations: torch.Tensor, eval: bool = False, **kwargs) -> torch.Tensor:
        return self.actor_mean(observations)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Call act() or update_distribution() before get_actions_log_prob().")
        return self.distribution.log_prob(actions).sum(dim=-1)

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.critic(critic_observations)

    def reset(self, dones=None) -> None:
        pass

    def if_fix_std(self) -> bool:
        return self.fix_action_std

    def update_std(self, std: float) -> None:
        self.std.data[:] = std

    def world_model_loss(self, observations: torch.Tensor, loss_type: str = "smooth_l1") -> Tuple[torch.Tensor, dict]:
        base_obs, history = self.split_obs(observations)
        z = self.encode_history_for_world_model(history)
        prev_frame = history[:, -1]
        prev_state = prev_frame[:, : self.hist_state_dim]
        prev_action = prev_frame[:, self.hist_state_dim : self.hist_state_dim + self.num_actions]

        if self.wm_target_indices is None:
            target = base_obs[:, : self.hist_state_dim]
        else:
            idx = self.wm_target_indices.to(base_obs.device)
            target = base_obs.index_select(dim=1, index=idx)

        pred = self.predict_world_model(observations)
        if loss_type == "mse":
            loss = torch.mean((pred - target) ** 2)
        else:
            loss = torch.nn.functional.smooth_l1_loss(pred, target)
        return loss, {"wm_loss": float(loss.detach().cpu())}

    def adapter_regularization_loss(self, observations: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        delta_dyn, delta_err = self.get_adapter_delta_components(observations)
        if self.use_dual_branch_adapter:
            delta = (
                self.dynamics_branch_gain * delta_dyn
                + self.tracking_branch_gain * delta_err
            )
        else:
            delta = delta_dyn
        loss = (delta ** 2).mean()
        delta_l2 = torch.norm(delta, p=2, dim=-1).mean()
        info = {
            "adapter_delta_l2": float(delta_l2.detach().cpu()),
            "adapter_reg_loss": float(loss.detach().cpu()),
            "max_abs_delta_action": float(delta.detach().abs().max().cpu()),
            "mean_abs_delta_action": float(delta.detach().abs().mean().cpu()),
            # Dual-only diagnostics remain defined for legacy adapters so the
            # runner can use one stable logging schema.
            "dynamics_delta_l2": 0.0,
            "tracking_delta_l2": 0.0,
            "dynamics_mean_abs_delta": 0.0,
            "tracking_mean_abs_delta": 0.0,
            "dynamics_max_abs_delta": 0.0,
            "tracking_max_abs_delta": 0.0,
            "branch_balance_ratio": 0.0,
            "branch_cosine_similarity": 0.0,
        }
        if self.use_dual_branch_adapter:
            dynamics_delta_l2 = torch.norm(delta_dyn, p=2, dim=-1).mean()
            tracking_delta_l2 = torch.norm(delta_err, p=2, dim=-1).mean()
            info.update({
                "dynamics_delta_l2": float(dynamics_delta_l2.detach().cpu()),
                "tracking_delta_l2": float(tracking_delta_l2.detach().cpu()),
                "dynamics_mean_abs_delta": float(delta_dyn.detach().abs().mean().cpu()),
                "tracking_mean_abs_delta": float(delta_err.detach().abs().mean().cpu()),
                "dynamics_max_abs_delta": float(delta_dyn.detach().abs().max().cpu()),
                "tracking_max_abs_delta": float(delta_err.detach().abs().max().cpu()),
                "branch_balance_ratio": float(
                    (
                        torch.minimum(dynamics_delta_l2, tracking_delta_l2)
                        / (torch.maximum(dynamics_delta_l2, tracking_delta_l2) + 1e-8)
                    ).detach().cpu()
                ),
                "branch_cosine_similarity": float(
                    torch.nn.functional.cosine_similarity(
                        delta_dyn,
                        delta_err,
                        dim=-1,
                        eps=1e-8,
                    ).mean().detach().cpu()
                ),
            })
        return loss, info

    def adapter_bias_regularization_loss(self, observations: torch.Tensor) -> torch.Tensor:
        """Penalize persistent per-joint offsets while preserving phase corrections."""
        delta = self.get_adapter_delta(observations)
        return torch.mean(torch.mean(delta, dim=0) ** 2)
