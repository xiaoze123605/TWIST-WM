"""DTERA: dual-feature residual adaptation with uncertainty/risk gating."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .actor_critic_twist_anyadapter import (
    DynamicsResidualBranch,
    HistoryEncoder,
    TwistAnyAdapterActorCritic,
    WorldModel,
    mlp,
    zero_init_last_linear,
)


class TrackingErrorHistoryEncoder(HistoryEncoder):
    """Independent encoder for the tracking-error history H_e."""

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        # The tracking encoder is optimized by both PPO and the error-trend
        # auxiliary loss.  Its unconstrained latent scale can otherwise grow
        # without changing either downstream objective, eventually driving
        # the residual branch's tanh fully into saturation.  Parameter-free
        # LayerNorm fixes that scale ambiguity without adding trainable state
        # or changing the latent dimension/checkpoint layout.
        latent = super().forward(history)
        return F.layer_norm(latent, (latent.shape[-1],))


class TrackingResidualBranchWithHistory(nn.Module):
    def __init__(
        self,
        error_frame_dim: int,
        tracking_latent_dim: int,
        num_actions: int,
        hidden_dims: Sequence[int],
        activation: str,
        delta_scale: float,
    ) -> None:
        super().__init__()
        self.delta_scale = float(delta_scale)
        self.net = mlp(
            error_frame_dim + tracking_latent_dim + num_actions,
            hidden_dims,
            num_actions,
            activation,
        )
        zero_init_last_linear(self.net)

    def forward(self, error_frame, z_error, base_action):
        x = torch.cat((error_frame, z_error, base_action), dim=-1)
        return self.delta_scale * torch.tanh(self.net(x))


class DTERAResidualAdapter(nn.Module):
    def __init__(
        self,
        dynamics_input_dim: int,
        dynamics_latent_dim: int,
        error_frame_dim: int,
        tracking_latent_dim: int,
        num_actions: int,
        hidden_dims: Sequence[int],
        activation: str,
        dynamics_delta_scale: float,
        tracking_delta_scale: float,
    ) -> None:
        super().__init__()
        self.dynamics_branch = DynamicsResidualBranch(
            dynamics_input_dim,
            dynamics_latent_dim,
            num_actions,
            hidden_dims,
            activation,
            dynamics_delta_scale,
        )
        self.tracking_branch = TrackingResidualBranchWithHistory(
            error_frame_dim,
            tracking_latent_dim,
            num_actions,
            hidden_dims,
            activation,
            tracking_delta_scale,
        )

    def forward(self, dynamics_features, z_dynamics, error_frame, z_error, base_action):
        return (
            self.dynamics_branch(dynamics_features, z_dynamics),
            self.tracking_branch(error_frame, z_error, base_action),
        )


class ErrorTrendPredictor(nn.Module):
    def __init__(self, error_frame_dim, num_actions, latent_dim, hidden_dims, activation):
        super().__init__()
        self.net = mlp(
            error_frame_dim + num_actions + latent_dim,
            hidden_dims,
            error_frame_dim,
            activation,
        )

    def forward(self, error_frame, action, z_error):
        return self.net(torch.cat((error_frame, action, z_error), dim=-1))


class WorldModelEnsemble(nn.Module):
    def __init__(
        self,
        ensemble_size: int,
        hist_state_dim: int,
        num_actions: int,
        latent_dim: int,
        target_dim: int,
        hidden_dims: Sequence[int],
        activation: str,
    ) -> None:
        super().__init__()
        if ensemble_size < 1:
            raise ValueError("world_model_ensemble_size must be positive")
        self.members = nn.ModuleList(
            [
                WorldModel(
                    hist_state_dim,
                    num_actions,
                    latent_dim,
                    target_dim,
                    hidden_dims,
                    activation,
                    predict_delta=True,
                )
                for _ in range(ensemble_size)
            ]
        )

    def forward_members(self, state, action, z_dynamics):
        return torch.stack(
            [member(state, action, z_dynamics) for member in self.members], dim=0
        )

    def forward(self, state, action, z_dynamics):
        return self.forward_members(state, action, z_dynamics).mean(dim=0)


class ResidualRiskPredictor(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_dims, activation):
        super().__init__()
        self.net = mlp(state_dim + num_actions, hidden_dims, 1, activation)

    def forward(self, state, action):
        return self.net(torch.cat((state, action), dim=-1)).squeeze(-1)


class TwistDTERAActorCritic(TwistAnyAdapterActorCritic):
    """Frozen TWIST actor plus DTERA dual residual and selective branch gates."""

    is_dtera = True

    def __init__(
        self,
        *args,
        tracking_history_len: int = 20,
        tracking_error_frame_dim: int = 53,
        tracking_latent_dim: int = 32,
        tracking_history_policy_grad_scale: float = 0.25,
        world_model_ensemble_size: int = 3,
        use_tracking_error_history: bool = True,
        use_error_trend_predictor: bool = True,
        use_world_model_ensemble: bool = True,
        use_adaptive_residual_gate: bool = True,
        error_predictor_hidden_dims: Sequence[int] = (128, 128),
        risk_predictor_hidden_dims: Sequence[int] = (128, 128),
        tracking_error_scales: Sequence[float] = (0.35, 2.0, 1.0, 0.35, 0.08),
        gate_demand_k: float = 1.0,
        gate_confidence_k: float = 1.0,
        gate_risk_k: float = 5.0,
        gate_mode: str = "demand_only",
        confidence_gate_strength: float = 0.0,
        use_independent_branch_gates: bool = False,
        tracking_demand_mode: str = "legacy_exp",
        tracking_demand_low: float = 0.30,
        tracking_demand_high: float = 0.80,
        dynamics_demand_low: float = 0.10,
        dynamics_demand_high: float = 0.50,
        dynamics_gate_scale: float = 1.0,
        tracking_gate_scale: float = 1.0,
        dynamics_confidence_gate_strength: float = 0.0,
        residual_warmup_iterations: int = 1000,
        freeze_residual_output_bias: bool = True,
        wm_variance_ema_decay: float = 0.99,
        base_single_obs_dim: int = 105,
        base_history_len: int = 10,
        adapter_hidden_dims: Sequence[int] = (128, 128),
        world_model_hidden_dims: Sequence[int] = (256, 256),
        activation: str = "elu",
        dynamics_action_delta_scale: float = 0.03,
        tracking_action_delta_scale: float = 0.03,
        use_dual_branch_adapter: bool = True,
        use_tracking_error_adapter_input: bool = True,
        **kwargs,
    ) -> None:
        if not use_dual_branch_adapter or not use_tracking_error_history:
            raise ValueError("DTERA requires both residual branches and tracking history")
        super().__init__(
            *args,
            adapter_hidden_dims=adapter_hidden_dims,
            world_model_hidden_dims=world_model_hidden_dims,
            activation=activation,
            dynamics_action_delta_scale=dynamics_action_delta_scale,
            tracking_action_delta_scale=tracking_action_delta_scale,
            use_dual_branch_adapter=True,
            use_tracking_error_adapter_input=use_tracking_error_adapter_input,
            **kwargs,
        )
        self.tracking_history_len = int(tracking_history_len)
        self.tracking_error_frame_dim = int(tracking_error_frame_dim)
        self.tracking_latent_dim = int(tracking_latent_dim)
        self.tracking_history_policy_grad_scale = float(
            tracking_history_policy_grad_scale
        )
        self.use_adaptive_residual_gate = bool(use_adaptive_residual_gate)
        self.use_error_trend_predictor = bool(use_error_trend_predictor)
        self.use_world_model_ensemble = bool(use_world_model_ensemble)
        self.gate_demand_k = float(gate_demand_k)
        self.gate_confidence_k = float(gate_confidence_k)
        self.gate_risk_k = float(gate_risk_k)
        self.gate_mode = str(gate_mode)
        self.confidence_gate_strength = float(confidence_gate_strength)
        self.use_independent_branch_gates = bool(use_independent_branch_gates)
        self.tracking_demand_mode = str(tracking_demand_mode)
        self.tracking_demand_low = float(tracking_demand_low)
        self.tracking_demand_high = float(tracking_demand_high)
        self.dynamics_demand_low = float(dynamics_demand_low)
        self.dynamics_demand_high = float(dynamics_demand_high)
        self.dynamics_gate_scale = float(dynamics_gate_scale)
        self.tracking_gate_scale = float(tracking_gate_scale)
        self.dynamics_confidence_gate_strength = float(
            dynamics_confidence_gate_strength
        )
        self.residual_warmup_iterations = int(residual_warmup_iterations)
        self.freeze_residual_output_bias = bool(freeze_residual_output_bias)
        self.wm_variance_ema_decay = float(wm_variance_ema_decay)
        self.base_single_obs_dim = int(base_single_obs_dim)
        self.base_history_len = int(base_history_len)
        if self.gate_mode not in ("off", "demand_only", "demand_confidence", "full"):
            raise ValueError(f"unsupported gate_mode: {self.gate_mode}")
        if not 0.0 <= self.confidence_gate_strength <= 1.0:
            raise ValueError("confidence_gate_strength must be in [0, 1]")
        if self.tracking_demand_mode not in ("legacy_exp", "smoothstep"):
            raise ValueError(
                f"unsupported tracking_demand_mode: {self.tracking_demand_mode}"
            )
        if not 0.0 <= self.tracking_demand_low < self.tracking_demand_high:
            raise ValueError(
                "tracking_demand thresholds require 0 <= low < high"
            )
        if not 0.0 <= self.dynamics_demand_low < self.dynamics_demand_high:
            raise ValueError(
                "dynamics_demand thresholds require 0 <= low < high"
            )
        for name, value in (
            ("dynamics_gate_scale", self.dynamics_gate_scale),
            ("tracking_gate_scale", self.tracking_gate_scale),
            (
                "dynamics_confidence_gate_strength",
                self.dynamics_confidence_gate_strength,
            ),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.residual_warmup_iterations < 0:
            raise ValueError("residual_warmup_iterations must be non-negative")
        if self.adapter_branch_mode not in ("base_only", "dyn_only", "err_only", "full"):
            raise ValueError(f"unsupported adapter_branch_mode: {self.adapter_branch_mode}")
        if len(tracking_error_scales) != 5 or any(float(x) <= 0 for x in tracking_error_scales):
            raise ValueError("tracking_error_scales must contain five positive group scales")

        self.tracking_error_history_encoder = TrackingErrorHistoryEncoder(
            self.tracking_error_frame_dim,
            self.tracking_history_len,
            self.tracking_latent_dim,
            hidden_dim=128,
            activation=activation,
            use_conv=True,
        )
        self.adapter = DTERAResidualAdapter(
            dynamics_input_dim=self.hist_state_dim + self.num_actions,
            dynamics_latent_dim=self.latent_dim,
            error_frame_dim=self.tracking_error_frame_dim,
            tracking_latent_dim=self.tracking_latent_dim,
            num_actions=self.num_actions,
            hidden_dims=adapter_hidden_dims,
            activation=activation,
            dynamics_delta_scale=dynamics_action_delta_scale,
            tracking_delta_scale=tracking_action_delta_scale,
        )
        if self.freeze_residual_output_bias:
            for branch in (
                self.adapter.dynamics_branch,
                self.adapter.tracking_branch,
            ):
                output_layer = [
                    module
                    for module in branch.net.modules()
                    if isinstance(module, nn.Linear)
                ][-1]
                output_layer.bias.requires_grad_(False)
        target_dim = (
            len(self.wm_target_indices)
            if self.wm_target_indices is not None
            else self.hist_state_dim
        )
        self.world_model = WorldModelEnsemble(
            world_model_ensemble_size if self.use_world_model_ensemble else 1,
            self.hist_state_dim,
            self.num_actions,
            self.latent_dim,
            target_dim,
            world_model_hidden_dims,
            activation,
        )
        self.error_trend_predictor = ErrorTrendPredictor(
            self.tracking_error_frame_dim,
            self.num_actions,
            self.tracking_latent_dim,
            error_predictor_hidden_dims,
            activation,
        )
        self.risk_predictor = ResidualRiskPredictor(
            self.hist_state_dim,
            self.num_actions,
            risk_predictor_hidden_dims,
            activation,
        )
        q, dq = self.num_actions, self.num_actions
        expanded_scales = (
            [tracking_error_scales[0]] * q
            + [tracking_error_scales[1]] * dq
            + [tracking_error_scales[2]] * 4
            + [tracking_error_scales[3]] * 2
            + [tracking_error_scales[4]]
        )
        self.register_buffer(
            "tracking_error_scale",
            torch.as_tensor(expanded_scales, dtype=torch.float32),
        )
        self.register_buffer("wm_variance_ema", torch.ones(target_dim))
        self.register_buffer("wm_variance_ema_updates", torch.zeros((), dtype=torch.long))
        self.register_buffer(
            "residual_training_iteration", torch.zeros((), dtype=torch.long)
        )

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys,
        unexpected_keys, error_msgs,
    ):
        # Checkpoints created before residual warm-up did not contain this
        # buffer.  Seed it at zero; the runner load hook then restores the
        # checkpoint iteration from checkpoint metadata.
        key = prefix + "residual_training_iteration"
        self._loaded_without_residual_iteration = key not in state_dict
        if self._loaded_without_residual_iteration:
            # Direct inference loaders may not have runner metadata.  In that
            # case an old checkpoint must preserve its historical full-scale
            # behavior; OnPolicyRunnerMimic later replaces this with the
            # checkpoint iteration when metadata is available.
            state_dict[key] = self.residual_training_iteration.new_tensor(
                self.residual_warmup_iterations
            )
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs,
        )

    def set_training_iteration(self, iteration):
        self.residual_training_iteration.fill_(max(int(iteration), 0))

    def residual_warmup_factor(self, iteration=None):
        if self.residual_warmup_iterations <= 0:
            return 1.0
        if iteration is None:
            iteration = int(self.residual_training_iteration.item())
        return min(max(float(iteration) / self.residual_warmup_iterations, 0.0), 1.0)

    @property
    def tracking_history_offset(self):
        return self.base_obs_dim + self.history_len * self.history_frame_dim

    def split_dtera_obs(self, observations):
        base_obs, dynamics_history = super().split_obs(observations)
        start = self.tracking_history_offset
        end = start + self.tracking_history_len * self.tracking_error_frame_dim
        if observations.shape[-1] < end:
            raise RuntimeError(f"DTERA observation needs at least {end} dims")
        tracking_history = observations[:, start:end].reshape(
            observations.shape[0],
            self.tracking_history_len,
            self.tracking_error_frame_dim,
        )
        return base_obs, dynamics_history, tracking_history

    def adapter_context(self, observations):
        if self.adapter_context_dim == 0:
            return None
        start = (
            self.tracking_history_offset
            + self.tracking_history_len * self.tracking_error_frame_dim
        )
        return observations[:, start : start + self.adapter_context_dim]

    def encode_tracking_history_for_policy(self, history):
        z = self.tracking_error_history_encoder(
            self.normalize_tracking_error(history)
        )
        scale = self.tracking_history_policy_grad_scale
        if scale <= 0.0:
            return z.detach()
        return z.detach() + scale * (z - z.detach())

    def encode_tracking_history_for_auxiliary(self, history):
        return self.tracking_error_history_encoder(
            self.normalize_tracking_error(history)
        )

    def normalize_tracking_error(self, error):
        # ``tracking_error_scale`` is a registered buffer, so loading/moving
        # the actor already keeps it on the same device as the model.  Passing
        # ``device=error.device`` here is unsafe for a traced export: tracing
        # on CPU records a literal ``Device("cpu")`` in the graph, and the
        # resulting JIT then moves this scale back to CPU when inference runs
        # on CUDA.  Preserve the buffer device and only follow the input dtype.
        scale = self.tracking_error_scale.to(dtype=error.dtype)
        return error / scale

    def action_delta_components(self, observations, detach_history=True, base_action=None):
        _, dynamics_history, tracking_history = self.split_dtera_obs(observations)
        z_dynamics = (
            self.encode_history_for_policy(dynamics_history)
            if detach_history
            else self.encode_history_for_world_model(dynamics_history)
        )
        z_error = (
            self.encode_tracking_history_for_policy(tracking_history)
            if detach_history
            else self.encode_tracking_history_for_auxiliary(tracking_history)
        )
        if base_action is None:
            base_action = self.base_action(observations)
        dynamics_state = dynamics_history[:, -1, : self.hist_state_dim]
        dynamics_features = torch.cat((base_action, dynamics_state), dim=-1)
        error_frame = self.normalize_tracking_error(tracking_history[:, -1])
        return self.adapter(
            dynamics_features, z_dynamics, error_frame, z_error, base_action
        )

    def _selected_candidate(self, delta_dynamics, delta_error):
        if self.adapter_branch_mode == "base_only":
            return torch.zeros_like(delta_dynamics)
        if self.adapter_branch_mode == "dyn_only":
            return self.dynamics_branch_gain * delta_dynamics
        if self.adapter_branch_mode == "err_only":
            return self.tracking_branch_gain * delta_error
        return (
            self.dynamics_branch_gain * delta_dynamics
            + self.tracking_branch_gain * delta_error
        )

    def _selected_branch_candidates(self, delta_dynamics, delta_error):
        """Return independently maskable branch contributions."""
        zero = torch.zeros_like(delta_dynamics)
        selected_dynamics = (
            self.dynamics_branch_gain * delta_dynamics
            if self.adapter_branch_mode in ("dyn_only", "full")
            else zero
        )
        selected_tracking = (
            self.tracking_branch_gain * delta_error
            if self.adapter_branch_mode in ("err_only", "full")
            else zero
        )
        return selected_dynamics, selected_tracking

    def candidate_delta_scale(self):
        dyn_scale = self.adapter.dynamics_branch.delta_scale
        err_scale = self.adapter.tracking_branch.delta_scale
        if self.adapter_branch_mode == "base_only":
            return 0.0
        if self.adapter_branch_mode == "dyn_only":
            return abs(self.dynamics_branch_gain) * dyn_scale
        if self.adapter_branch_mode == "err_only":
            return abs(self.tracking_branch_gain) * err_scale
        return (
            abs(self.dynamics_branch_gain) * dyn_scale
            + abs(self.tracking_branch_gain) * err_scale
        )

    @staticmethod
    def _smoothstep_demand(magnitude, low, high):
        width = high - low
        unit = ((magnitude - low) / width).clamp(0.0, 1.0)
        demand = unit.square() * (3.0 - 2.0 * unit)
        threshold_eps = 1e-6
        demand = torch.where(
            magnitude <= low + threshold_eps,
            torch.zeros_like(demand),
            demand,
        )
        return torch.where(
            magnitude >= high - threshold_eps,
            torch.ones_like(demand),
            demand,
        )

    def _tracking_error_magnitude(self, error_frame, normalized=False):
        normalized_error = (
            error_frame if normalized else self.normalize_tracking_error(error_frame)
        )
        return torch.sqrt(
            torch.mean(normalized_error.square(), dim=-1) + 1e-12
        )

    def tracking_demand(self, error_frame, normalized=False):
        magnitude = self._tracking_error_magnitude(error_frame, normalized)
        if self.tracking_demand_mode == "legacy_exp":
            demand = 1.0 - torch.exp(-self.gate_demand_k * magnitude)
        else:
            # Fixed normalized thresholds are deterministic in simulation and
            # on hardware. Smoothstep supplies exact closed/open regions
            # without deployment-time running-statistic drift.
            demand = self._smoothstep_demand(
                magnitude,
                self.tracking_demand_low,
                self.tracking_demand_high,
            )
        return torch.where(
            torch.all(error_frame == 0, dim=-1), torch.zeros_like(demand), demand
        ).clamp(0.0, 1.0)

    def dynamics_demand(self, error_frame, normalized=False):
        """Early-opening dynamics demand that is exactly closed at stand."""
        magnitude = self._tracking_error_magnitude(error_frame, normalized)
        demand = self._smoothstep_demand(
            magnitude,
            self.dynamics_demand_low,
            self.dynamics_demand_high,
        )
        return torch.where(
            torch.all(error_frame == 0, dim=-1), torch.zeros_like(demand), demand
        ).clamp(0.0, 1.0)

    def update_uncertainty_ema(self, member_predictions):
        # Normalize disagreement by the natural per-component prediction
        # variation across the training batch, not by raw physical units.
        ensemble_mean = member_predictions.detach().mean(dim=0)
        variance = ensemble_mean.var(dim=0, unbiased=False)
        decay = self.wm_variance_ema_decay
        self.wm_variance_ema.mul_(decay).add_(variance, alpha=1.0 - decay)
        self.wm_variance_ema_updates.add_(1)

    def confidence_from_members(self, member_predictions):
        if member_predictions.shape[0] == 1:
            batch = member_predictions.shape[1]
            return (
                member_predictions.new_ones(batch),
                member_predictions.new_zeros(batch),
            )
        variance = member_predictions.var(dim=0, unbiased=False)
        normalized = variance / self.wm_variance_ema.clamp_min(1e-6)
        uncertainty = normalized.mean(dim=-1).clamp_min(0.0)
        confidence = torch.exp(-self.gate_confidence_k * uncertainty).clamp(0.0, 1.0)
        return confidence, uncertainty

    def safety_from_delta_risk(self, delta_risk):
        return torch.exp(-self.gate_risk_k * F.relu(delta_risk)).clamp(0.0, 1.0)

    def effective_confidence(self, confidence):
        strength = self.confidence_gate_strength
        return 1.0 - strength * (1.0 - confidence)

    def dynamics_effective_confidence(self, confidence):
        strength = self.dynamics_confidence_gate_strength
        return 1.0 - strength * (1.0 - confidence)

    def risk_logits(self, observations, actions):
        _, dynamics_history, _ = self.split_dtera_obs(observations)
        state = dynamics_history[:, -1, : self.hist_state_dim]
        return self.risk_predictor(state, actions)

    def predict_world_model_members(self, observations, actions=None):
        _, dynamics_history, _ = self.split_dtera_obs(observations)
        z_dynamics = self.encode_history_for_world_model(dynamics_history)
        frame = dynamics_history[:, -1]
        state = frame[:, : self.hist_state_dim]
        if actions is None:
            actions = frame[:, self.hist_state_dim : self.hist_state_dim + self.num_actions]
        return self.world_model.forward_members(state, actions, z_dynamics)

    def predict_world_model(self, observations, actions=None):
        return self.predict_world_model_members(observations, actions).mean(dim=0)

    def _compose_gated_delta(
        self,
        selected_dynamics,
        selected_tracking,
        tracking_demand,
        dynamics_demand,
        confidence,
        safety,
    ):
        """Apply the configured shared or per-branch gates.

        This helper contains the authoritative action mathematics.  Both the
        fast actor path and the full diagnostic path call it so skipping
        unused WM/risk inference cannot change the policy output.
        """
        effective_confidence = self.effective_confidence(confidence)
        dynamics_confidence = self.dynamics_effective_confidence(confidence)
        demand_confidence_gate = tracking_demand * effective_confidence
        full_gate = demand_confidence_gate * safety

        if not self.use_adaptive_residual_gate or self.gate_mode == "off":
            legacy_gate = torch.ones_like(tracking_demand)
        elif self.gate_mode == "demand_only":
            legacy_gate = tracking_demand
        elif self.gate_mode == "demand_confidence":
            legacy_gate = demand_confidence_gate
        else:
            legacy_gate = full_gate
        if self.adapter_branch_mode == "base_only":
            legacy_gate = torch.zeros_like(legacy_gate)
        legacy_gate = legacy_gate.clamp(0.0, 1.0)

        if self.use_independent_branch_gates:
            if not self.use_adaptive_residual_gate or self.gate_mode == "off":
                dynamics_gate = torch.ones_like(dynamics_demand)
            elif self.gate_mode == "demand_only":
                dynamics_gate = dynamics_demand
            elif self.gate_mode == "demand_confidence":
                dynamics_gate = dynamics_demand * dynamics_confidence
            else:
                dynamics_gate = dynamics_demand * dynamics_confidence * safety

            if not self.use_adaptive_residual_gate or self.gate_mode == "off":
                tracking_gate = torch.ones_like(tracking_demand)
            elif self.gate_mode == "demand_only":
                tracking_gate = tracking_demand
            elif self.gate_mode == "demand_confidence":
                tracking_gate = demand_confidence_gate
            else:
                tracking_gate = full_gate

            dynamics_gate = (
                self.dynamics_gate_scale * dynamics_gate
            ).clamp(0.0, 1.0)
            tracking_gate = (
                self.tracking_gate_scale * tracking_gate
            ).clamp(0.0, 1.0)
            if self.adapter_branch_mode not in ("dyn_only", "full"):
                dynamics_gate = torch.zeros_like(dynamics_gate)
            if self.adapter_branch_mode not in ("err_only", "full"):
                tracking_gate = torch.zeros_like(tracking_gate)
            gated_dynamics = dynamics_gate.unsqueeze(-1) * selected_dynamics
            gated_tracking = tracking_gate.unsqueeze(-1) * selected_tracking
            # Preserve the historical field as the tracking-gate alias.
            gate = tracking_gate
        else:
            dynamics_gate = legacy_gate
            tracking_gate = legacy_gate
            gated_dynamics = legacy_gate.unsqueeze(-1) * selected_dynamics
            gated_tracking = legacy_gate.unsqueeze(-1) * selected_tracking
            gate = legacy_gate

        return {
            "effective_confidence": effective_confidence,
            "demand_confidence_gate": demand_confidence_gate,
            "full_gate": full_gate,
            "gate": gate,
            "dynamics_gate": dynamics_gate,
            "tracking_gate": tracking_gate,
            "gated_delta_dyn": gated_dynamics,
            "gated_delta_err": gated_tracking,
            "gated_delta": gated_dynamics + gated_tracking,
        }

    def _gate_model_requirements(self):
        """Return whether the action actually depends on WM/risk outputs."""
        if not self.use_adaptive_residual_gate or self.gate_mode in (
            "off", "demand_only"
        ):
            return False, False
        needs_confidence = self.confidence_gate_strength > 0.0
        if self.use_independent_branch_gates:
            needs_confidence = (
                needs_confidence
                or self.dynamics_confidence_gate_strength > 0.0
            )
        needs_risk = self.gate_mode == "full"
        return needs_confidence, needs_risk

    def _action_delta_fast(self, observations, base_action, detach_history=True):
        """Compute the applied residual without diagnostic-only model calls."""
        delta_dynamics, delta_error = self.action_delta_components(
            observations,
            detach_history=detach_history,
            base_action=base_action,
        )
        selected_dynamics, selected_tracking = self._selected_branch_candidates(
            delta_dynamics, delta_error
        )
        candidate = selected_dynamics + selected_tracking
        _, _, tracking_history = self.split_dtera_obs(observations)
        error_frame = tracking_history[:, -1]
        tracking_demand = self.tracking_demand(error_frame)
        dynamics_demand = self.dynamics_demand(error_frame)

        confidence = tracking_demand.new_ones(tracking_demand.shape)
        safety = tracking_demand.new_ones(tracking_demand.shape)
        needs_confidence, needs_risk = self._gate_model_requirements()
        if needs_confidence or needs_risk:
            # Gate models are auxiliary estimators and must not receive PPO
            # gradients through the actor action path.
            with torch.no_grad():
                if needs_confidence:
                    members = self.predict_world_model_members(
                        observations, base_action + candidate.detach()
                    )
                    confidence, _ = self.confidence_from_members(members)
                if needs_risk:
                    p_base = torch.sigmoid(
                        self.risk_logits(observations, base_action)
                    )
                    p_candidate = torch.sigmoid(
                        self.risk_logits(
                            observations, base_action + candidate.detach()
                        )
                    )
                    safety = self.safety_from_delta_risk(
                        p_candidate - p_base
                    )

        gates = self._compose_gated_delta(
            selected_dynamics,
            selected_tracking,
            tracking_demand,
            dynamics_demand,
            confidence,
            safety,
        )
        return self.residual_warmup_factor() * gates["gated_delta"]

    def action_diagnostics(self, observations, base_action=None):
        if base_action is None:
            base_action = self.base_action(observations)
        delta_dynamics, delta_error = self.action_delta_components(
            observations, detach_history=True, base_action=base_action
        )
        selected_dynamics, selected_tracking = self._selected_branch_candidates(
            delta_dynamics, delta_error
        )
        candidate = selected_dynamics + selected_tracking
        _, _, tracking_history = self.split_dtera_obs(observations)
        error_frame = tracking_history[:, -1]
        demand = self.tracking_demand(error_frame)
        dynamics_demand = self.dynamics_demand(error_frame)

        # WM/risk are diagnostic gate models during the PPO phase.  Their
        # parameters and the dynamics encoder must remain frozen until the
        # deferred auxiliary phase, so do not build an autograd graph here.
        with torch.no_grad():
            members = self.predict_world_model_members(
                observations, base_action + candidate.detach()
            )
            confidence, uncertainty = self.confidence_from_members(members)
            p_base = torch.sigmoid(
                self.risk_logits(observations, base_action)
            )
            p_candidate = torch.sigmoid(
                self.risk_logits(observations, base_action + candidate.detach())
            )
        delta_risk = p_candidate - p_base
        safety = self.safety_from_delta_risk(delta_risk).detach()
        gates = self._compose_gated_delta(
            selected_dynamics,
            selected_tracking,
            demand,
            dynamics_demand,
            confidence,
            safety,
        )
        alpha_res = self.residual_warmup_factor()
        applied = alpha_res * gates["gated_delta"]
        return {
            "base_action": base_action,
            "delta_dyn": delta_dynamics,
            "delta_err": delta_error,
            "candidate_delta": candidate,
            "selected_delta_dyn": selected_dynamics,
            "selected_delta_err": selected_tracking,
            "demand": demand,
            "dynamics_demand": dynamics_demand,
            "confidence": confidence,
            "effective_confidence": gates["effective_confidence"],
            "uncertainty": uncertainty,
            "p_base": p_base,
            "p_candidate": p_candidate,
            "delta_risk": delta_risk,
            "safety": safety,
            "demand_confidence_gate": gates["demand_confidence_gate"],
            "full_gate": gates["full_gate"],
            "gate": gates["gate"],
            "dynamics_gate": gates["dynamics_gate"],
            "tracking_gate": gates["tracking_gate"],
            "residual_warmup_factor": alpha_res,
            "gated_delta_dyn": gates["gated_delta_dyn"],
            "gated_delta_err": gates["gated_delta_err"],
            "gated_delta": gates["gated_delta"],
            "applied_delta": applied,
        }

    def action_delta(self, observations, detach_history=True, base_action=None):
        if self.adapter_branch_mode == "base_only":
            return observations.new_zeros(observations.shape[0], self.num_actions)
        if base_action is None:
            base_action = self.base_action(observations)
        return self._action_delta_fast(
            observations,
            base_action,
            detach_history=detach_history,
        )

    def actor_mean(self, observations):
        base_action = self.base_action(observations)
        if self.adapter_branch_mode == "base_only":
            return base_action
        applied = self._action_delta_fast(observations, base_action)
        return base_action + self.adapter_gain * applied

    def error_prediction_loss(self, observations, actions, next_observations, valid_mask):
        if not self.use_error_trend_predictor:
            zero = observations.new_zeros(())
            return zero, zero
        _, _, raw_history = self.split_dtera_obs(observations)
        _, _, raw_next_history = self.split_dtera_obs(next_observations)
        history = self.normalize_tracking_error(raw_history)
        next_history = self.normalize_tracking_error(raw_next_history)
        z_error = self.tracking_error_history_encoder(history)
        current_error = history[:, -1]
        target_delta = next_history[:, -1] - current_error
        prediction = self.error_trend_predictor(current_error, actions, z_error)
        per_sample = F.smooth_l1_loss(prediction, target_delta, reduction="none").mean(-1)
        mask = valid_mask.reshape(-1).float()
        loss = (per_sample * mask).sum() / mask.sum().clamp_min(1.0)
        return loss, z_error.detach().norm(dim=-1).mean()

    def build_synthetic_stand_observation(self, observations, root_height=0.793):
        stand = observations.clone()
        single = stand.new_zeros(stand.shape[0], self.base_single_obs_dim)
        single[:, 0] = root_height
        single[:, 8 : 8 + self.num_actions] = self.default_ref_dof_pos
        expected_base_dim = self.base_single_obs_dim * (self.base_history_len + 1)
        if expected_base_dim != self.base_obs_dim:
            raise RuntimeError(
                f"base stand layout is {expected_base_dim}, expected {self.base_obs_dim}"
            )
        stand[:, : self.base_obs_dim] = single.repeat(1, self.base_history_len + 1)
        dyn_start = self.base_obs_dim
        dyn_end = dyn_start + self.history_len * self.history_frame_dim
        stand[:, dyn_start:dyn_end] = 0.0
        err_end = dyn_end + self.tracking_history_len * self.tracking_error_frame_dim
        stand[:, dyn_end:err_end] = 0.0
        if stand.shape[-1] > err_end:
            stand[:, err_end:] = 0.0
        return stand

    @staticmethod
    def _tensor_stats(tensor):
        flat = tensor.detach().reshape(-1)
        return float(flat.mean().cpu()), float(flat.abs().max().cpu())

    @staticmethod
    def _output_bias_norm(branch):
        output_layer = [
            module
            for module in branch.net.modules()
            if isinstance(module, nn.Linear)
        ][-1]
        return float(output_layer.bias.detach().norm().cpu())

    def adapter_regularization_loss(self, observations) -> Tuple[torch.Tensor, dict]:
        diag = self.action_diagnostics(observations)
        candidate = diag["candidate_delta"]
        gated = diag["gated_delta"]
        applied = diag["applied_delta"]
        delta_dynamics = diag["delta_dyn"]
        delta_error = diag["delta_err"]
        loss = candidate.square().mean()
        dyn_l2 = torch.norm(delta_dynamics, dim=-1).mean()
        err_l2 = torch.norm(delta_error, dim=-1).mean()
        candidate_l2 = torch.norm(candidate, dim=-1).mean()
        gated_l2 = torch.norm(gated, dim=-1).mean()
        applied_l2 = torch.norm(applied, dim=-1).mean()
        gate = diag["gate"]
        dynamics_gate = diag["dynamics_gate"]
        tracking_gate = diag["tracking_gate"]
        delta_risk = diag["delta_risk"]
        dyn_scale = self.adapter.dynamics_branch.delta_scale
        err_scale = self.adapter.tracking_branch.delta_scale
        candidate_scale = self.candidate_delta_scale()
        dyn_saturation = (delta_dynamics.abs() >= 0.95 * dyn_scale).float().mean()
        err_saturation = (delta_error.abs() >= 0.95 * err_scale).float().mean()
        candidate_saturation = (
            candidate.new_zeros(())
            if candidate_scale <= 0.0
            else (
                candidate.abs() >= 0.95 * candidate_scale
            ).float().mean()
        )
        info = {
            "adapter_delta_l2": float(candidate_l2.detach().cpu()),
            "dynamics_delta_l2": float(dyn_l2.detach().cpu()),
            "tracking_delta_l2": float(err_l2.detach().cpu()),
            "dynamics_mean_abs_delta": float(delta_dynamics.detach().abs().mean().cpu()),
            "tracking_mean_abs_delta": float(delta_error.detach().abs().mean().cpu()),
            "dynamics_max_abs_delta": float(delta_dynamics.detach().abs().max().cpu()),
            "tracking_max_abs_delta": float(delta_error.detach().abs().max().cpu()),
            "branch_balance_ratio": float((torch.minimum(dyn_l2, err_l2) / (torch.maximum(dyn_l2, err_l2) + 1e-8)).detach().cpu()),
            "branch_cosine_similarity": float(F.cosine_similarity(delta_dynamics, delta_error, dim=-1, eps=1e-8).mean().detach().cpu()),
            "max_abs_delta_action": float(applied.detach().abs().max().cpu()),
            "mean_abs_delta_action": float(applied.detach().abs().mean().cpu()),
            "candidate_delta_l2": float(candidate_l2.detach().cpu()),
            "gated_delta_l2": float(gated_l2.detach().cpu()),
            "applied_delta_l2": float(applied_l2.detach().cpu()),
            "candidate_mean_abs_delta": float(candidate.detach().abs().mean().cpu()),
            "candidate_max_abs_delta": float(candidate.detach().abs().max().cpu()),
            "gated_mean_abs_delta": float(gated.detach().abs().mean().cpu()),
            "gated_max_abs_delta": float(gated.detach().abs().max().cpu()),
            "applied_mean_abs_delta": float(applied.detach().abs().mean().cpu()),
            "applied_max_abs_delta": float(applied.detach().abs().max().cpu()),
            "wm_uncertainty_mean": float(diag["uncertainty"].mean().cpu()),
            "wm_uncertainty_p90": float(torch.quantile(diag["uncertainty"], 0.90).cpu()),
            "wm_uncertainty_p95": float(torch.quantile(diag["uncertainty"], 0.95).cpu()),
            "tracking_demand_mean": float(diag["demand"].mean().detach().cpu()),
            "tracking_demand_p90": float(torch.quantile(diag["demand"].detach(), 0.90).cpu()),
            "dynamics_demand_mean": float(
                diag["dynamics_demand"].mean().detach().cpu()
            ),
            "dynamics_demand_p90": float(
                torch.quantile(diag["dynamics_demand"].detach(), 0.90).cpu()
            ),
            "gate_confidence_mean": float(diag["confidence"].mean().cpu()),
            "safety_factor_mean": float(diag["safety"].mean().cpu()),
            "demand_confidence_gate_mean": float(
                diag["demand_confidence_gate"].mean().detach().cpu()
            ),
            "full_diagnostic_gate_mean": float(
                diag["full_gate"].mean().detach().cpu()
            ),
            "residual_warmup_factor": float(diag["residual_warmup_factor"]),
            "dyn_saturation_fraction": float(dyn_saturation.detach().cpu()),
            "err_saturation_fraction": float(err_saturation.detach().cpu()),
            "candidate_saturation_fraction": float(
                candidate_saturation.detach().cpu()
            ),
            "dynamics_output_bias_norm": self._output_bias_norm(
                self.adapter.dynamics_branch
            ),
            "tracking_output_bias_norm": self._output_bias_norm(
                self.adapter.tracking_branch
            ),
            "gate_mean": float(gate.mean().detach().cpu()),
            "gate_p10": float(torch.quantile(gate.detach(), 0.10).cpu()),
            "gate_p90": float(torch.quantile(gate.detach(), 0.90).cpu()),
            "gate_fraction_lt_0_1": float((gate < 0.1).float().mean().detach().cpu()),
            "gate_fraction_gt_0_9": float((gate > 0.9).float().mean().detach().cpu()),
            "dynamics_gate_mean": float(dynamics_gate.mean().detach().cpu()),
            "dynamics_gate_p10": float(torch.quantile(dynamics_gate.detach(), 0.10).cpu()),
            "dynamics_gate_p90": float(torch.quantile(dynamics_gate.detach(), 0.90).cpu()),
            "tracking_gate_mean": float(tracking_gate.mean().detach().cpu()),
            "tracking_gate_p10": float(torch.quantile(tracking_gate.detach(), 0.10).cpu()),
            "tracking_gate_p90": float(torch.quantile(tracking_gate.detach(), 0.90).cpu()),
            "p_base_mean": float(diag["p_base"].mean().cpu()),
            "p_candidate_mean": float(diag["p_candidate"].mean().cpu()),
            "delta_risk_mean": float(delta_risk.mean().cpu()),
            "delta_risk_p95": float(torch.quantile(delta_risk, 0.95).cpu()),
        }
        return loss, info

    def residual_saturation_penalty(self, observations):
        delta_dynamics, delta_error = self.action_delta_components(observations)
        dyn_scale = self.adapter.dynamics_branch.delta_scale
        err_scale = self.adapter.tracking_branch.delta_scale
        dyn_ratio = delta_dynamics.abs() / max(dyn_scale, 1e-8)
        err_ratio = delta_error.abs() / max(err_scale, 1e-8)
        return (
            F.relu(dyn_ratio - 0.8).square().mean()
            + F.relu(err_ratio - 0.8).square().mean()
        )

    def adapter_bias_regularization_loss(self, observations):
        delta_dynamics, delta_error = self.action_delta_components(observations)
        candidate = self._selected_candidate(delta_dynamics, delta_error)
        return torch.mean(torch.mean(candidate, dim=0).square())
