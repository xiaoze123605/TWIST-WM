"""Any2Track-style dynamics adapter for the frozen TWIST student actor.

Unlike the earlier action-residual AnyAdapter experiments, this module injects
zero-initialized adapter branches alongside every frozen actor backbone layer.
The history encoder is trained by a dynamics world model and is detached from
the PPO policy loss by default.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .actor_critic_twist_anyadapter import get_activation, mlp


Normal.set_default_validate_args = False


def _copy_linear(state, prefix: str) -> nn.Linear:
    weight = state[prefix + ".weight"].detach().cpu()
    bias = state[prefix + ".bias"].detach().cpu()
    layer = nn.Linear(weight.shape[1], weight.shape[0])
    with torch.no_grad():
        layer.weight.copy_(weight)
        layer.bias.copy_(bias)
    return layer


class Any2TrackHistoryEncoder(nn.Module):
    """Temporal encoder matching the released AnyAdapter convolution layout."""

    def __init__(
        self,
        history_frame_dim: int,
        history_len: int,
        latent_dim: int = 128,
        conv_channels: Sequence[int] = (64, 64),
        kernel_sizes: Sequence[int] = (9, 6),
        strides: Sequence[int] = (5, 3),
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if not (len(conv_channels) == len(kernel_sizes) == len(strides)):
            raise ValueError("conv_channels, kernel_sizes and strides must have equal lengths")
        self.history_frame_dim = int(history_frame_dim)
        self.history_len = int(history_len)
        self.latent_dim = int(latent_dim)

        layers = []
        in_channels = self.history_frame_dim
        out_len = self.history_len
        for channels, kernel, stride in zip(conv_channels, kernel_sizes, strides):
            out_len = (out_len - int(kernel)) // int(stride) + 1
            if out_len <= 0:
                raise ValueError(
                    f"history_len={history_len} is too short for kernels={tuple(kernel_sizes)} "
                    f"and strides={tuple(strides)}"
                )
            layers.extend([
                nn.Conv1d(in_channels, int(channels), int(kernel), stride=int(stride)),
                get_activation(activation),
            ])
            in_channels = int(channels)
        self.conv = nn.Sequential(*layers)
        self.output = nn.Linear(in_channels * out_len, self.latent_dim)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3:
            raise ValueError(f"history must be [B,H,D], got {tuple(history.shape)}")
        hidden = self.conv(history.transpose(1, 2))
        return self.output(hidden.flatten(start_dim=1))


class TwistLayerwiseAdapterActor(nn.Module):
    """Frozen 0529 TWIST actor with zero-init adapters at every backbone layer."""

    def __init__(
        self,
        base_actor_jit_path: str,
        latent_dim: int,
        base_obs_dim: int = 1155,
        num_motion_observations: int = 31,
        adapter_gain: float = 1.0,
        verify_base: bool = True,
    ) -> None:
        super().__init__()
        scripted = torch.jit.load(base_actor_jit_path, map_location="cpu").eval()
        state = scripted.state_dict()

        self.base_obs_dim = int(base_obs_dim)
        self.num_motion_observations = int(num_motion_observations)
        self.adapter_gain = float(adapter_gain)
        self.register_buffer("obs_mean", state["normalizer._mean"].detach().cpu().clone())
        self.register_buffer("obs_std", state["normalizer._std"].detach().cpu().clone())

        self.motion_input = _copy_linear(state, "actor.motion_encoder.encoder.0")
        self.motion_output = _copy_linear(state, "actor.motion_encoder.linear_output")
        self.base_layers = nn.ModuleList([
            _copy_linear(state, "actor.actor_backbone.0"),
            _copy_linear(state, "actor.actor_backbone.2"),
            _copy_linear(state, "actor.actor_backbone.4"),
            _copy_linear(state, "actor.actor_backbone.6"),
            _copy_linear(state, "actor.actor_backbone.9"),
        ])
        layer_norm_weight = state["actor.actor_backbone.7.weight"].detach().cpu()
        self.base_layer_norm = nn.LayerNorm(layer_norm_weight.numel())
        with torch.no_grad():
            self.base_layer_norm.weight.copy_(layer_norm_weight)
            self.base_layer_norm.bias.copy_(state["actor.actor_backbone.7.bias"].detach().cpu())

        adapter_inputs = [int(latent_dim)] + [layer.out_features for layer in self.base_layers[:-1]]
        adapter_outputs = [layer.out_features for layer in self.base_layers]
        self.adapter = nn.ModuleList([
            nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip(adapter_inputs, adapter_outputs)
        ])
        for layer in self.adapter:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

        for module in [self.motion_input, self.motion_output, self.base_layers, self.base_layer_norm]:
            for parameter in module.parameters():
                parameter.requires_grad_(False)

        if self.obs_mean.numel() != self.base_obs_dim:
            raise ValueError(
                f"Base JIT normalizer has {self.obs_mean.numel()} dims, expected {self.base_obs_dim}."
            )
        self.num_actions = self.base_layers[-1].out_features
        if verify_base:
            self._verify_against_scripted(scripted)

    def _normalize(self, obs: torch.Tensor) -> torch.Tensor:
        return ((obs - self.obs_mean) / (self.obs_std + 1e-4)).float()

    def _backbone_input(self, base_obs: torch.Tensor) -> torch.Tensor:
        obs = self._normalize(base_obs)
        motion_obs = obs[:, : self.num_motion_observations]
        motion_latent = self.motion_output(F.silu(self.motion_input(motion_obs)))
        return torch.cat([
            obs[:, self.num_motion_observations :],
            obs[:, : self.num_motion_observations],
            motion_latent,
        ], dim=-1)

    def _forward_impl(self, base_obs: torch.Tensor, z: torch.Tensor, use_adapter: bool) -> torch.Tensor:
        hidden = self._backbone_input(base_obs)
        for index, base_layer in enumerate(self.base_layers):
            base_hidden = base_layer(hidden)
            if use_adapter:
                adapter_input = z if index == 0 else hidden
                base_hidden = base_hidden + self.adapter_gain * self.adapter[index](adapter_input)
            if index == len(self.base_layers) - 1:
                return base_hidden
            if index == len(self.base_layers) - 2:
                base_hidden = self.base_layer_norm(base_hidden)
            hidden = F.silu(base_hidden)
        raise RuntimeError("TWIST backbone has no output layer")

    def forward(self, base_obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(base_obs, z, use_adapter=True)

    def base_forward(self, base_obs: torch.Tensor) -> torch.Tensor:
        z = base_obs.new_zeros(base_obs.shape[0], self.adapter[0].in_features)
        return self._forward_impl(base_obs, z, use_adapter=False)

    def _verify_against_scripted(self, scripted) -> None:
        with torch.no_grad():
            generator = torch.Generator(device="cpu").manual_seed(17)
            sample = torch.randn(4, self.base_obs_dim, generator=generator)
            expected = scripted(sample)
            actual = self.base_forward(sample)
            max_error = float((expected - actual).abs().max())
        if max_error > 2e-5:
            raise RuntimeError(
                "Reconstructed TWIST backbone does not match the base JIT: "
                f"max_abs_error={max_error:.3e}"
            )
        print(f"[Any2Track] frozen base reconstruction verified, max_abs_error={max_error:.3e}")


class Any2TrackWorldModel(nn.Module):
    """Structured forward model for the 51-D TWIST proprioceptive state."""

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        latent_dim: int,
        hidden_dims: Sequence[int],
        activation: str = "silu",
        dof_vel_start: int = 28,
        dof_pos_start: int = 5,
        dof_count: int = 23,
        dof_vel_scale: float = 0.05,
        control_dt: float = 0.02,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.dof_vel_start = int(dof_vel_start)
        self.dof_pos_start = int(dof_pos_start)
        self.dof_count = int(dof_count)
        self.dof_vel_scale = float(dof_vel_scale)
        self.control_dt = float(control_dt)
        # gyro delta + roll/pitch delta + joint-velocity delta
        self.delta_dim = 3 + 2 + self.dof_count
        self.net = mlp(
            self.state_dim + int(num_actions) + int(latent_dim),
            hidden_dims,
            self.delta_dim,
            activation,
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        delta = self.net(torch.cat([state, action, z], dim=-1))
        new_gyro = state[:, :3] + delta[:, :3]
        new_orientation = state[:, 3:5] + delta[:, 3:5]
        old_dof_pos = state[:, self.dof_pos_start : self.dof_pos_start + self.dof_count]
        old_scaled_dof_vel = state[:, self.dof_vel_start : self.dof_vel_start + self.dof_count]
        new_scaled_dof_vel = old_scaled_dof_vel + delta[:, 5:]
        new_dof_pos = old_dof_pos + new_scaled_dof_vel / self.dof_vel_scale * self.control_dt
        return torch.cat([new_gyro, new_orientation, new_dof_pos, new_scaled_dof_vel], dim=-1)


class TwistAny2TrackActorCritic(nn.Module):
    """RSL-RL actor-critic using the released AnyAdapter training structure."""

    is_recurrent = False

    def __init__(
        self,
        num_actions: int,
        base_actor_jit_path: str,
        base_obs_dim: int = 1155,
        history_len: int = 79,
        history_frame_dim: int = 74,
        hist_state_dim: int = 51,
        wm_target_indices: Optional[Sequence[int]] = None,
        latent_dim: int = 128,
        world_model_hidden_dims: Sequence[int] = (512, 512, 256, 256, 256, 128),
        critic_hidden_dims: Sequence[int] = (512, 256, 128),
        activation: str = "silu",
        init_noise_std: float = 0.05,
        fix_action_std: bool = False,
        adapter_gain: float = 1.0,
        freeze_base: bool = True,
        num_critic_observations: Optional[int] = None,
        num_critic_obs: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        del freeze_base  # The layerwise base is always frozen by construction.
        critic_dim = num_critic_observations if num_critic_observations is not None else num_critic_obs
        if critic_dim is None:
            raise ValueError("num_critic_observations is required")
        self.num_actions = int(num_actions)
        self.base_obs_dim = int(base_obs_dim)
        self.history_len = int(history_len)
        self.history_frame_dim = int(history_frame_dim)
        self.hist_state_dim = int(hist_state_dim)
        self.latent_dim = int(latent_dim)
        self.fix_action_std = bool(fix_action_std)
        self.adapter_gain = float(adapter_gain)
        if self.history_frame_dim != self.hist_state_dim + self.num_actions:
            raise ValueError("history_frame_dim must equal hist_state_dim + num_actions")

        if wm_target_indices is None:
            wm_target_indices = list(range(31, 36)) + list(range(36, 59)) + list(range(59, 82))
        self.register_buffer("wm_target_indices", torch.as_tensor(wm_target_indices, dtype=torch.long))
        if self.wm_target_indices.numel() != self.hist_state_dim:
            raise ValueError("wm_target_indices length must equal hist_state_dim")

        self.history_encoder = Any2TrackHistoryEncoder(
            self.history_frame_dim,
            self.history_len,
            latent_dim=self.latent_dim,
            activation=activation,
        )
        self.layerwise_actor = TwistLayerwiseAdapterActor(
            base_actor_jit_path,
            latent_dim=self.latent_dim,
            base_obs_dim=self.base_obs_dim,
            adapter_gain=self.adapter_gain,
        )
        if self.layerwise_actor.num_actions != self.num_actions:
            raise ValueError(
                f"Base actor outputs {self.layerwise_actor.num_actions} actions, expected {self.num_actions}."
            )
        self.world_model = Any2TrackWorldModel(
            self.hist_state_dim,
            self.num_actions,
            self.latent_dim,
            world_model_hidden_dims,
            activation=activation,
        )
        self.critic = mlp(int(critic_dim), critic_hidden_dims, 1, activation)
        self.std = nn.Parameter(
            float(init_noise_std) * torch.ones(self.num_actions),
            requires_grad=not self.fix_action_std,
        )
        self.distribution: Optional[Normal] = None
        self.wm_component_splits = {
            "ang_vel": (0, 3),
            "orientation": (3, 5),
            "dof_pos": (5, 28),
            "dof_vel": (28, 51),
        }

    @property
    def adapter(self):
        """Expose layer adapters without registering a duplicate module path."""
        return self.layerwise_actor.adapter

    def split_obs(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        base_obs = obs[:, : self.base_obs_dim]
        history_end = self.base_obs_dim + self.history_len * self.history_frame_dim
        if obs.shape[1] < history_end:
            raise RuntimeError(f"Expected at least {history_end} observation dims, got {obs.shape[1]}")
        history = obs[:, self.base_obs_dim : history_end].reshape(
            obs.shape[0], self.history_len, self.history_frame_dim
        )
        return base_obs, history

    def selected_state(self, obs: torch.Tensor) -> torch.Tensor:
        base_obs = obs[:, : self.base_obs_dim]
        return base_obs.index_select(1, self.wm_target_indices)

    def encode_history_for_policy(self, history: torch.Tensor) -> torch.Tensor:
        return self.history_encoder(history).detach()

    def encode_history_for_world_model(self, history: torch.Tensor) -> torch.Tensor:
        return self.history_encoder(history)

    def actor_mean(self, observations: torch.Tensor) -> torch.Tensor:
        base_obs, history = self.split_obs(observations)
        z = self.encode_history_for_policy(history)
        return self.layerwise_actor(base_obs, z)

    def base_action(self, observations: torch.Tensor) -> torch.Tensor:
        base_obs = observations[:, : self.base_obs_dim]
        with torch.no_grad():
            return self.layerwise_actor.base_forward(base_obs)

    def action_delta(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.actor_mean(observations) - self.base_action(observations)

    def get_adapter_delta(self, observations: torch.Tensor) -> torch.Tensor:
        return self.action_delta(observations)

    def predict_world_model(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if history is None:
            _, history = self.split_obs(observations)
        if state is None:
            state = self.selected_state(observations)
        z = self.encode_history_for_world_model(history)
        return self.world_model(state, actions, z)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor_mean(observations)

    def update_distribution(self, observations: torch.Tensor) -> None:
        mean = self.actor_mean(observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.actor_mean(observations)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
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

    def adapter_regularization_loss(self, observations: torch.Tensor):
        delta = self.action_delta(observations)
        loss = delta.square().mean()
        return loss, {
            "adapter_delta_l2": float(delta.norm(dim=-1).mean().detach().cpu()),
            "adapter_reg_loss": float(loss.detach().cpu()),
            "max_abs_delta_action": float(delta.abs().max().detach().cpu()),
            "mean_abs_delta_action": float(delta.abs().mean().detach().cpu()),
        }

    def reset(self, dones=None) -> None:
        pass

    def if_fix_std(self) -> bool:
        return self.fix_action_std

    def update_std(self, std: float) -> None:
        self.std.data.fill_(std)
