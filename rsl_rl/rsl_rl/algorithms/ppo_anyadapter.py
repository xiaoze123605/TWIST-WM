"""
PPO with AnyAdapter auxiliary losses.

Drop into:
    TWIST/rsl_rl/rsl_rl/algorithms/ppo_anyadapter.py

This class is intentionally close to TWIST/rsl_rl/rsl_rl/algorithms/ppo.py, but
adds:
    - world_model_loss_coef * actor_critic.world_model_loss(obs_batch)
    - adapter_reg_coef * actor_critic.adapter_regularization_loss(obs_batch)

It assumes the environment has already appended history to actor observations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rsl_rl.algorithms.ppo import PPO


class PPOAnyAdapter(PPO):
    def __init__(
        self,
        *args,
        world_model_loss_coef: float = 0.1,
        adapter_reg_coef: float = 1e-3,
        adapter_reg_initial_coef: float = None,
        adapter_reg_anneal_iterations: int = 0,
        residual_saturation_reg_coef: float = 0.0,
        adapter_bias_reg_coef: float = 0.0,
        stand_anchor_coef: float = 1.0,
        synthetic_stand_anchor_coef: float = 0.0,
        synthetic_stand_root_height: float = 0.793,
        stand_vel_threshold: float = 0.05,
        stand_dof_threshold: float = 0.15,
        world_model_loss_type: str = "smooth_l1",
        weight_decay: float = 0.0,
        joint_encoder_optimization: bool = False,
        separate_wm_updates_history_encoder: bool = False,
        error_prediction_loss_coef: float = 0.0,
        defer_world_model_update: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.world_model_loss_coef = float(world_model_loss_coef)
        self.adapter_reg_coef = float(adapter_reg_coef)
        self.adapter_reg_initial_coef = (
            self.adapter_reg_coef
            if adapter_reg_initial_coef is None
            else float(adapter_reg_initial_coef)
        )
        self.adapter_reg_anneal_iterations = int(adapter_reg_anneal_iterations)
        self.residual_saturation_reg_coef = float(
            residual_saturation_reg_coef
        )
        self.adapter_bias_reg_coef = float(adapter_bias_reg_coef)
        self.stand_anchor_coef = float(stand_anchor_coef)
        self.synthetic_stand_anchor_coef = float(synthetic_stand_anchor_coef)
        self.synthetic_stand_root_height = float(synthetic_stand_root_height)
        self.stand_vel_threshold = float(stand_vel_threshold)
        self.stand_dof_threshold = float(stand_dof_threshold)
        self.world_model_loss_type = world_model_loss_type
        self.weight_decay = float(weight_decay)
        self.joint_encoder_optimization = bool(joint_encoder_optimization)
        self.separate_wm_updates_history_encoder = bool(
            separate_wm_updates_history_encoder
        )
        self.error_prediction_loss_coef = float(error_prediction_loss_coef)
        self.defer_world_model_update = bool(defer_world_model_update)
        self.skip_dagger_update = True
        self.requires_next_observations = True

        self.ppo_params = []
        self.ppo_params += [p for p in self.actor_critic.adapter.parameters() if p.requires_grad]
        self.ppo_params += [p for p in self.actor_critic.critic.parameters() if p.requires_grad]
        if getattr(self.actor_critic, "std", None) is not None and self.actor_critic.std.requires_grad:
            self.ppo_params.append(self.actor_critic.std)

        self.history_encoder_params = [
            p for p in self.actor_critic.history_encoder.parameters() if p.requires_grad
        ]
        self.world_model_params = [
            p for p in self.actor_critic.world_model.parameters() if p.requires_grad
        ]
        # The WM optimizer owns the encoder only in joint mode. In the legacy
        # non-joint mode the encoder stays inside ppo_optimizer (the WM loss
        # gradient is accumulated before the policy step, see update()), so it
        # remains trainable without being owned by wm_optimizer. Any2Track
        # opts out: its encoder is trained exclusively by the autoregressive
        # world-model update via separate_wm_updates_history_encoder.
        if not self.joint_encoder_optimization and not self.separate_wm_updates_history_encoder:
            self.ppo_params = self.ppo_params + self.history_encoder_params
        self.wm_params = list(self.world_model_params)
        if self.joint_encoder_optimization or self.separate_wm_updates_history_encoder:
            self.wm_params = list(self.history_encoder_params) + self.wm_params

        self.ppo_optimizer = torch.optim.Adam(
            self.ppo_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        # The L2-style weight decay of torch.optim.Adam dominates the tiny
        # WM/encoder gradients (the encoder policy path is scaled by
        # history_policy_grad_scale and the WM gradient shrinks as it
        # converges), which walks both networks to zero: in the formal run
        # the encoder collapsed by iter ~500 and the world model by iter
        # ~2000. wm_optimizer is therefore created without weight decay.
        wm_weight_decay = 0.0
        self.wm_optimizer = torch.optim.Adam(
            self.wm_params,
            lr=self.learning_rate,
            weight_decay=wm_weight_decay,
        )
        self.optimizer = self.ppo_optimizer
        ppo_optimizer_param_ids = {
            id(p) for group in self.ppo_optimizer.param_groups for p in group["params"]
        }
        wm_optimizer_param_ids = {
            id(p) for group in self.wm_optimizer.param_groups for p in group["params"]
        }
        self.history_encoder_in_ppo_optimizer = all(
            id(p) in ppo_optimizer_param_ids for p in self.history_encoder_params
        )
        self.history_encoder_in_wm_optimizer = all(
            id(p) in wm_optimizer_param_ids for p in self.history_encoder_params
        )
        print(
            "[AnyAdapter] history_encoder_in_ppo_optimizer="
            f"{self.history_encoder_in_ppo_optimizer}"
        )
        print(
            "[AnyAdapter] history_encoder_in_wm_optimizer="
            f"{self.history_encoder_in_wm_optimizer}"
        )
        self.anyadapter_metrics = {}
        self._wm_target_warning_printed = False

    def effective_adapter_reg_coef(self, iteration=None):
        if iteration is None:
            if hasattr(self.actor_critic, "residual_training_iteration"):
                iteration = int(
                    self.actor_critic.residual_training_iteration.item()
                )
            else:
                iteration = self.counter
        if self.adapter_reg_anneal_iterations <= 0:
            return self.adapter_reg_coef
        progress = min(
            max(float(iteration) / self.adapter_reg_anneal_iterations, 0.0),
            1.0,
        )
        return (
            self.adapter_reg_initial_coef
            + progress * (self.adapter_reg_coef - self.adapter_reg_initial_coef)
        )

    def _has_world_model_target(self) -> bool:
        has_next_obs = hasattr(self.storage, "next_observations") and self.storage.next_observations is not None
        has_available_mask = (
            hasattr(self.storage, "next_observations_available")
            and bool(torch.any(self.storage.next_observations_available).item())
        )
        return has_next_obs and has_available_mask

    def _warn_missing_world_model_target_once(self) -> None:
        if self._wm_target_warning_printed:
            return
        has_next_obs = hasattr(self.storage, "next_observations") and self.storage.next_observations is not None
        has_target_indices = getattr(self.actor_critic, "wm_target_indices", None) is not None
        has_predict_world_model = hasattr(self.actor_critic, "predict_world_model")
        print(
            "[AnyAdapter] world model target missing, wm loss skipped "
            f"(rollout storage has next_obs: {has_next_obs}; "
            f"wm_target_indices set: {has_target_indices}; "
            "PPOAnyAdapter active: True; "
            f"actor_critic.predict_world_model available: {has_predict_world_model}; "
            "predict_world_model called: False)"
        )
        self._wm_target_warning_printed = True

    def process_env_step(self, rewards, dones, infos):
        next_obs = infos.get("next_observations", None)
        if next_obs is not None:
            self.transition.next_observations = next_obs.detach()
        return super().process_env_step(rewards, dones, infos)

    def _world_model_loss_from_batch(self, obs_batch, actions_batch, next_obs_batch, dones_batch, available_batch):
        wm_target_indices = getattr(self.actor_critic, "wm_target_indices", None)
        if wm_target_indices is None:
            target_next_state = next_obs_batch[:, : self.actor_critic.hist_state_dim]
        else:
            idx = wm_target_indices.to(next_obs_batch.device)
            target_next_state = next_obs_batch.index_select(dim=1, index=idx)

        pred_next_state = self.actor_critic.predict_world_model(obs_batch, actions_batch)

        valid_mask = (1.0 - dones_batch.float()).reshape(-1, 1)
        if available_batch is not None:
            valid_mask = valid_mask * available_batch.float().reshape(-1, 1)
        valid_count = valid_mask.sum().clamp_min(1.0)

        splits = getattr(self.actor_critic, "wm_component_splits", None)
        weights = getattr(self.actor_critic, "wm_component_weights", None)
        if splits is not None and weights is not None:
            total_loss = pred_next_state.new_zeros(())
            component_info = {}
            for name, (start, end) in splits.items():
                w = float(weights.get(name, 1.0))
                pred_slice = pred_next_state[:, start:end]
                target_slice = target_next_state[:, start:end]
                if self.world_model_loss_type == "mse":
                    comp_loss_per_dim = (pred_slice - target_slice) ** 2
                else:
                    comp_loss_per_dim = F.smooth_l1_loss(pred_slice, target_slice, reduction="none")
                comp_loss_per_sample = comp_loss_per_dim.mean(dim=-1, keepdim=True)
                # Log the raw masked component loss. The training objective
                # keeps its existing weighted sum exactly unchanged.
                component_loss = (comp_loss_per_sample * valid_mask).sum() / valid_count
                component_info[f"wm_{name}_loss"] = float(component_loss.detach().cpu())
                total_loss = total_loss + w * component_loss
            return total_loss, component_info
        else:
            if self.world_model_loss_type == "mse":
                loss_per_dim = (pred_next_state - target_next_state) ** 2
            else:
                loss_per_dim = F.smooth_l1_loss(pred_next_state, target_next_state, reduction="none")
            loss_per_sample = loss_per_dim.mean(dim=-1, keepdim=True)
            return (loss_per_sample * valid_mask).sum() / valid_count, {}

    def _stand_anchor_loss_from_batch(self, obs_batch):
        base_obs, _ = self.actor_critic.split_obs(obs_batch)
        if base_obs.shape[1] < 31:
            zero = obs_batch.new_tensor(0.0)
            return zero, zero

        root_vel = base_obs[:, 4:7]
        yaw_ang_vel = base_obs[:, 7]
        ref_dof_pos = base_obs[:, 8:31]
        default_ref = self.actor_critic.default_ref_dof_pos.to(base_obs.device).view(1, -1)

        stand_mask = (
            (torch.norm(root_vel, dim=-1) < self.stand_vel_threshold)
            & (torch.abs(yaw_ang_vel) < self.stand_vel_threshold)
            & (torch.mean(torch.abs(ref_dof_pos - default_ref), dim=-1) < self.stand_dof_threshold)
        )
        stand_ratio = stand_mask.float().mean()
        if not torch.any(stand_mask):
            return obs_batch.new_tensor(0.0), stand_ratio

        delta = self.actor_critic.action_delta(obs_batch[stand_mask], detach_history=True)
        return (delta ** 2).mean(), stand_ratio

    def _synthetic_stand_anchor_loss_from_batch(self, obs_batch):
        if hasattr(self.actor_critic, "build_synthetic_stand_observation"):
            stand_obs = self.actor_critic.build_synthetic_stand_observation(
                obs_batch, self.synthetic_stand_root_height
            )
            diagnostics = self.actor_critic.action_diagnostics(stand_obs)
            self._last_synthetic_stand_metrics = {
                "synthetic_stand_candidate_delta": float(
                    diagnostics["candidate_delta"].detach().norm(dim=-1).mean().cpu()
                ),
                "synthetic_stand_applied_delta": float(
                    diagnostics["applied_delta"].detach().norm(dim=-1).mean().cpu()
                ),
            }
            # Demand is exactly zero for a synthetic stand. Penalize the raw
            # candidate, otherwise the gate would hide a learned fixed bias.
            return diagnostics["candidate_delta"].square().mean()
        stand_obs = obs_batch.clone()
        base_obs, _ = self.actor_critic.split_obs(stand_obs)
        default_ref = self.actor_critic.default_ref_dof_pos.to(base_obs.device)
        base_obs[:, 0] = self.synthetic_stand_root_height
        base_obs[:, 1:4] = 0.0
        base_obs[:, 4:7] = 0.0
        base_obs[:, 7] = 0.0
        base_obs[:, 8:31] = default_ref.view(1, -1)
        delta = self.actor_critic.action_delta(stand_obs, detach_history=True)
        return (delta ** 2).mean()

    @staticmethod
    def _grad_norm(parameters):
        sq_sum = None
        for p in parameters:
            if p.grad is None:
                continue
            norm_sq = torch.sum(p.grad.detach() ** 2)
            sq_sum = norm_sq if sq_sum is None else sq_sum + norm_sq
        if sq_sum is None:
            return 0.0
        return float(torch.sqrt(sq_sum).detach().cpu())

    @staticmethod
    def _gradient_values_norm(gradients):
        """L2 norm for gradients returned by torch.autograd.grad."""
        sq_sum = None
        for grad in gradients:
            if grad is None:
                continue
            norm_sq = torch.sum(grad.detach() ** 2)
            sq_sum = norm_sq if sq_sum is None else sq_sum + norm_sq
        if sq_sum is None:
            return 0.0
        return float(torch.sqrt(sq_sum).detach().cpu())

    def _diagnostic_grad_norm(self, loss, parameters):
        """Measure one loss source without changing accumulated training grads."""
        parameters = tuple(parameters)
        if not parameters or not loss.requires_grad:
            return 0.0
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        return self._gradient_values_norm(gradients)

    def update(self):
        effective_adapter_reg_coef = self.effective_adapter_reg_coef()
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_wm_loss = 0.0
        mean_adapter_reg_loss = 0.0
        mean_saturation_penalty = 0.0
        mean_adapter_bias_reg_loss = 0.0
        mean_adapter_delta_l2 = 0.0
        mean_dynamics_delta_l2 = 0.0
        mean_tracking_delta_l2 = 0.0
        mean_dynamics_mean_abs_delta = 0.0
        mean_tracking_mean_abs_delta = 0.0
        mean_dynamics_max_abs_delta = 0.0
        mean_tracking_max_abs_delta = 0.0
        mean_branch_balance_ratio = 0.0
        mean_branch_cosine_similarity = 0.0
        mean_stand_anchor_loss = 0.0
        mean_synthetic_stand_anchor_loss = 0.0
        mean_stand_sample_ratio = 0.0
        mean_history_encoder_ppo_grad_norm = 0.0
        mean_history_encoder_wm_grad_norm = 0.0
        mean_history_encoder_total_grad_norm = 0.0
        mean_adapter_grad_norm = 0.0
        mean_dynamics_branch_grad_norm = 0.0
        mean_tracking_branch_grad_norm = 0.0
        mean_max_abs_delta_action = 0.0
        mean_mean_abs_delta_action = 0.0
        mean_wm_component_losses = {
            "wm_ang_vel_loss": 0.0,
            "wm_orientation_loss": 0.0,
            "wm_dof_pos_loss": 0.0,
            "wm_dof_vel_loss": 0.0,
        }
        history_ppo_grad_measurements = 0
        history_wm_grad_measurements = 0
        wm_loss_skipped = False
        mean_error_prediction_loss = 0.0
        mean_z_e_norm = 0.0
        mean_tracking_encoder_ppo_grad_norm = 0.0
        mean_tracking_encoder_aux_grad_norm = 0.0
        mean_error_predictor_grad_norm = 0.0
        tracking_grad_measurements = 0
        dtera_info_keys = (
            "candidate_delta_l2", "gated_delta_l2", "applied_delta_l2",
            "candidate_mean_abs_delta", "candidate_max_abs_delta",
            "gated_mean_abs_delta", "gated_max_abs_delta",
            "applied_mean_abs_delta", "applied_max_abs_delta",
            "wm_uncertainty_mean", "wm_uncertainty_p90", "wm_uncertainty_p95",
            "tracking_demand_mean", "tracking_demand_p90", "gate_confidence_mean",
            "dynamics_demand_mean", "dynamics_demand_p90",
            "safety_factor_mean", "gate_mean", "gate_p10", "gate_p90",
            "demand_confidence_gate_mean", "full_diagnostic_gate_mean",
            "residual_warmup_factor", "dyn_saturation_fraction",
            "err_saturation_fraction", "candidate_saturation_fraction",
            "dynamics_output_bias_norm", "tracking_output_bias_norm",
            "gate_fraction_lt_0_1", "gate_fraction_gt_0_9",
            "dynamics_gate_mean", "dynamics_gate_p10", "dynamics_gate_p90",
            "tracking_gate_mean", "tracking_gate_p10", "tracking_gate_p90",
            "p_base_mean", "p_candidate_mean", "delta_risk_mean", "delta_risk_p95",
            "synthetic_stand_candidate_delta", "synthetic_stand_applied_delta",
        )
        mean_dtera_info = {key: 0.0 for key in dtera_info_keys}

        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator_anyadapter(self.num_mini_batches, self.num_learning_epochs)

        num_updates = self.num_learning_epochs * self.num_mini_batches

        for sample in generator:
            (
                obs_batch,
                critic_obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                hid_states_batch,
                masks_batch,
                next_obs_batch,
                dones_batch,
                next_obs_available_batch,
            ) = sample

            self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            wm_loss = obs_batch.new_tensor(0.0)
            adapter_reg_loss = obs_batch.new_tensor(0.0)
            saturation_penalty = obs_batch.new_tensor(0.0)
            adapter_bias_reg_loss = obs_batch.new_tensor(0.0)
            stand_anchor_loss = obs_batch.new_tensor(0.0)
            synthetic_stand_anchor_loss = obs_batch.new_tensor(0.0)
            stand_sample_ratio = obs_batch.new_tensor(0.0)
            wm_component_info = {}
            history_encoder_ppo_grad_norm = None
            history_encoder_wm_grad_norm = None
            if (
                hasattr(self.actor_critic, "adapter_regularization_loss")
                and effective_adapter_reg_coef > 0.0
            ):
                adapter_reg_loss, adapter_info = self.actor_critic.adapter_regularization_loss(obs_batch)
                loss = loss + effective_adapter_reg_coef * adapter_reg_loss
                mean_adapter_delta_l2 += adapter_info.get("adapter_delta_l2", adapter_reg_loss.item())
                mean_max_abs_delta_action += adapter_info.get("max_abs_delta_action", 0.0)
                mean_mean_abs_delta_action += adapter_info.get("mean_abs_delta_action", 0.0)
                mean_dynamics_delta_l2 += adapter_info.get("dynamics_delta_l2", 0.0)
                mean_tracking_delta_l2 += adapter_info.get("tracking_delta_l2", 0.0)
                mean_dynamics_mean_abs_delta += adapter_info.get("dynamics_mean_abs_delta", 0.0)
                mean_tracking_mean_abs_delta += adapter_info.get("tracking_mean_abs_delta", 0.0)
                mean_dynamics_max_abs_delta += adapter_info.get("dynamics_max_abs_delta", 0.0)
                mean_tracking_max_abs_delta += adapter_info.get("tracking_max_abs_delta", 0.0)
                mean_branch_balance_ratio += adapter_info.get("branch_balance_ratio", 0.0)
                mean_branch_cosine_similarity += adapter_info.get("branch_cosine_similarity", 0.0)
                for key in dtera_info_keys:
                    mean_dtera_info[key] += adapter_info.get(key, 0.0)
            if (
                self.residual_saturation_reg_coef > 0.0
                and hasattr(self.actor_critic, "residual_saturation_penalty")
            ):
                saturation_penalty = (
                    self.actor_critic.residual_saturation_penalty(obs_batch)
                )
                loss = (
                    loss
                    + self.residual_saturation_reg_coef * saturation_penalty
                )
            if (
                hasattr(self.actor_critic, "adapter_bias_regularization_loss")
                and self.adapter_bias_reg_coef > 0.0
            ):
                adapter_bias_reg_loss = self.actor_critic.adapter_bias_regularization_loss(obs_batch)
                loss = loss + self.adapter_bias_reg_coef * adapter_bias_reg_loss
            if self.stand_anchor_coef > 0.0:
                stand_anchor_loss, stand_sample_ratio = self._stand_anchor_loss_from_batch(obs_batch)
                loss = loss + self.stand_anchor_coef * stand_anchor_loss
            if self.synthetic_stand_anchor_coef > 0.0:
                self._last_synthetic_stand_metrics = {}
                synthetic_stand_anchor_loss = self._synthetic_stand_anchor_loss_from_batch(obs_batch)
                loss = loss + self.synthetic_stand_anchor_coef * synthetic_stand_anchor_loss
                for key, value in self._last_synthetic_stand_metrics.items():
                    mean_dtera_info[key] += value

            control_loss = loss
            error_prediction_loss = obs_batch.new_zeros(())
            z_e_norm = obs_batch.new_zeros(())
            valid_next_mask = (
                (1.0 - dones_batch.float()) * next_obs_available_batch.float()
            )
            if (
                self.error_prediction_loss_coef > 0.0
                and hasattr(self.actor_critic, "error_prediction_loss")
                and bool(torch.any(valid_next_mask).item())
            ):
                error_prediction_loss, z_e_norm = self.actor_critic.error_prediction_loss(
                    obs_batch,
                    actions_batch,
                    next_obs_batch,
                    valid_next_mask,
                )
                loss = loss + self.error_prediction_loss_coef * error_prediction_loss

            # Everything accumulated so far is the policy-side objective. In
            # joint mode the WM term is backwarded separately first so its
            # encoder gradient can be measured before policy accumulation.
            ppo_loss = loss
            wm_target_available = self._has_world_model_target()
            weighted_wm_loss = None
            if (
                self.joint_encoder_optimization
                and not self.defer_world_model_update
                and self.world_model_loss_coef > 0.0
            ):
                if hasattr(self.actor_critic, "predict_world_model") and wm_target_available:
                    wm_loss, wm_component_info = self._world_model_loss_from_batch(
                        obs_batch,
                        actions_batch,
                        next_obs_batch,
                        dones_batch,
                        next_obs_available_batch,
                    )
                    weighted_wm_loss = self.world_model_loss_coef * wm_loss
                else:
                    wm_loss_skipped = True
                    self._warn_missing_world_model_target_once()

            run_non_joint_wm_update = (
                not self.joint_encoder_optimization
                and not self.defer_world_model_update
                and hasattr(self.actor_critic, "predict_world_model")
                and self.world_model_loss_coef > 0.0
            )

            # Policy source diagnostic, read-only and measured once per update
            # before any backward frees the graph. WM gradients below are read
            # from the real WM backward right after it runs.
            if history_ppo_grad_measurements == 0:
                history_params = tuple(self.actor_critic.history_encoder.parameters())
                history_encoder_ppo_grad_norm = self._diagnostic_grad_norm(
                    ppo_loss,
                    history_params,
                )
                history_ppo_grad_measurements += 1

            tracking_encoder_ppo_grad_norm = 0.0
            tracking_encoder_aux_grad_norm = 0.0
            if hasattr(self.actor_critic, "tracking_error_history_encoder"):
                tracking_params = tuple(
                    self.actor_critic.tracking_error_history_encoder.parameters()
                )
                tracking_encoder_ppo_grad_norm = self._diagnostic_grad_norm(
                    control_loss, tracking_params
                )
                if error_prediction_loss.requires_grad:
                    tracking_encoder_aux_grad_norm = self._diagnostic_grad_norm(
                        self.error_prediction_loss_coef * error_prediction_loss,
                        tracking_params,
                    )
                tracking_grad_measurements += 1

            self.ppo_optimizer.zero_grad()
            self.wm_optimizer.zero_grad()
            if self.joint_encoder_optimization and weighted_wm_loss is not None:
                weighted_wm_loss.backward()
                history_encoder_wm_grad_norm = self._grad_norm(
                    self.actor_critic.history_encoder.parameters()
                )
                history_wm_grad_measurements += 1
            if run_non_joint_wm_update:
                if wm_target_available:
                    # Backward the WM loss before the policy step: in non-joint
                    # mode the encoder is owned by ppo_optimizer, so its WM
                    # gradient is applied by the ppo step below while
                    # wm_optimizer only consumes the world-model gradients.
                    wm_loss, wm_component_info = self._world_model_loss_from_batch(
                        obs_batch,
                        actions_batch,
                        next_obs_batch,
                        dones_batch,
                        next_obs_available_batch,
                    )
                    (self.world_model_loss_coef * wm_loss).backward()
                    history_encoder_wm_grad_norm = self._grad_norm(self.actor_critic.history_encoder.parameters())
                    history_wm_grad_measurements += 1
                else:
                    history_encoder_wm_grad_norm = 0.0
                    wm_loss_skipped = True
                    self._warn_missing_world_model_target_once()
            # In joint mode this accumulates policy gradients on top of the WM
            # gradients already stored on HistoryEncoder. The resulting .grad
            # is mathematically the gradient of ppo_loss + weighted_wm_loss;
            # in non-joint mode it is the policy gradient on top of the WM
            # gradient that the ppo step is about to consume.
            ppo_loss.backward()
            history_encoder_total_grad_norm = self._grad_norm(
                self.actor_critic.history_encoder.parameters()
            )
            adapter_grad_norm = self._grad_norm(self.actor_critic.adapter.parameters())
            error_predictor_grad_norm = 0.0
            if hasattr(self.actor_critic, "error_trend_predictor"):
                error_predictor_grad_norm = self._grad_norm(
                    self.actor_critic.error_trend_predictor.parameters()
                )
            if getattr(self.actor_critic, "use_dual_branch_adapter", False):
                dynamics_branch_grad_norm = self._grad_norm(
                    self.actor_critic.adapter.dynamics_branch.parameters()
                )
                tracking_branch_grad_norm = self._grad_norm(
                    self.actor_critic.adapter.tracking_branch.parameters()
                )
            else:
                dynamics_branch_grad_norm = 0.0
                tracking_branch_grad_norm = 0.0
            if self.joint_encoder_optimization:
                nn.utils.clip_grad_norm_(
                    self.ppo_params + self.wm_params,
                    self.max_grad_norm,
                )
            else:
                nn.utils.clip_grad_norm_(self.ppo_params, self.max_grad_norm)
            self.ppo_optimizer.step()
            if run_non_joint_wm_update and wm_target_available:
                nn.utils.clip_grad_norm_(self.wm_params, self.max_grad_norm)
                self.wm_optimizer.step()
            elif self.joint_encoder_optimization and not self.defer_world_model_update:
                self.wm_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_wm_loss += wm_loss.item()
            mean_adapter_reg_loss += adapter_reg_loss.item()
            mean_saturation_penalty += saturation_penalty.item()
            mean_adapter_bias_reg_loss += adapter_bias_reg_loss.item()
            mean_stand_anchor_loss += stand_anchor_loss.item()
            mean_synthetic_stand_anchor_loss += synthetic_stand_anchor_loss.item()
            mean_stand_sample_ratio += stand_sample_ratio.item()
            if history_encoder_ppo_grad_norm is not None:
                mean_history_encoder_ppo_grad_norm += history_encoder_ppo_grad_norm
            if history_encoder_wm_grad_norm is not None:
                mean_history_encoder_wm_grad_norm += history_encoder_wm_grad_norm
            mean_history_encoder_total_grad_norm += history_encoder_total_grad_norm
            mean_adapter_grad_norm += adapter_grad_norm
            mean_dynamics_branch_grad_norm += dynamics_branch_grad_norm
            mean_tracking_branch_grad_norm += tracking_branch_grad_norm
            mean_error_prediction_loss += error_prediction_loss.item()
            mean_z_e_norm += float(z_e_norm.detach().cpu())
            mean_tracking_encoder_ppo_grad_norm += tracking_encoder_ppo_grad_norm
            mean_tracking_encoder_aux_grad_norm += tracking_encoder_aux_grad_norm
            mean_error_predictor_grad_norm += error_predictor_grad_norm
            for key in mean_wm_component_losses:
                mean_wm_component_losses[key] += wm_component_info.get(key, 0.0)

        if self.fix_std:
            std_stage = min(max((self.counter - self.std_schedule[2]), 0) / self.std_schedule[3], 1)
            std_coef = std_stage * (self.std_schedule[1] - self.std_schedule[0]) + self.std_schedule[0]
            if hasattr(self.actor_critic, "update_std"):
                self.actor_critic.update_std(std_coef)

        self.anyadapter_metrics = {
            "ppo_learning_rate": float(self.ppo_optimizer.param_groups[0]["lr"]),
            "world_model_loss": mean_wm_loss / num_updates,
            "world_model_loss_skipped": float(wm_loss_skipped),
            "adapter_delta_l2": mean_adapter_delta_l2 / num_updates,
            "dynamics_delta_l2": mean_dynamics_delta_l2 / num_updates,
            "tracking_delta_l2": mean_tracking_delta_l2 / num_updates,
            "dynamics_mean_abs_delta": mean_dynamics_mean_abs_delta / num_updates,
            "tracking_mean_abs_delta": mean_tracking_mean_abs_delta / num_updates,
            "dynamics_max_abs_delta": mean_dynamics_max_abs_delta / num_updates,
            "tracking_max_abs_delta": mean_tracking_max_abs_delta / num_updates,
            "branch_balance_ratio": mean_branch_balance_ratio / num_updates,
            "branch_cosine_similarity": mean_branch_cosine_similarity / num_updates,
            "adapter_reg_loss": mean_adapter_reg_loss / num_updates,
            "effective_adapter_reg_coef": effective_adapter_reg_coef,
            "residual_saturation_penalty": (
                mean_saturation_penalty / num_updates
            ),
            "adapter_bias_reg_loss": mean_adapter_bias_reg_loss / num_updates,
            "stand_anchor_loss": mean_stand_anchor_loss / num_updates,
            "synthetic_stand_anchor_loss": mean_synthetic_stand_anchor_loss / num_updates,
            "stand_sample_ratio": mean_stand_sample_ratio / num_updates,
            "history_encoder_ppo_grad_norm": (
                mean_history_encoder_ppo_grad_norm / max(history_ppo_grad_measurements, 1)
            ),
            "history_encoder_wm_grad_norm": (
                mean_history_encoder_wm_grad_norm / max(history_wm_grad_measurements, 1)
            ),
            "history_encoder_total_grad_norm": mean_history_encoder_total_grad_norm / num_updates,
            "adapter_grad_norm": mean_adapter_grad_norm / num_updates,
            "dynamics_branch_grad_norm": mean_dynamics_branch_grad_norm / num_updates,
            "tracking_branch_grad_norm": mean_tracking_branch_grad_norm / num_updates,
            "max_abs_delta_action": mean_max_abs_delta_action / num_updates,
            "mean_abs_delta_action": mean_mean_abs_delta_action / num_updates,
            "surrogate_loss": mean_surrogate_loss / num_updates,
            "value_loss": mean_value_loss / num_updates,
            "error_prediction_loss": mean_error_prediction_loss / num_updates,
            "z_e_norm": mean_z_e_norm / num_updates,
            "tracking_encoder_ppo_grad_norm": (
                mean_tracking_encoder_ppo_grad_norm / max(tracking_grad_measurements, 1)
            ),
            "tracking_encoder_aux_grad_norm": (
                mean_tracking_encoder_aux_grad_norm / max(tracking_grad_measurements, 1)
            ),
            "error_predictor_grad_norm": mean_error_predictor_grad_norm / num_updates,
        }
        for key, total in mean_dtera_info.items():
            self.anyadapter_metrics[key] = total / num_updates
        for key, total in mean_wm_component_losses.items():
            value = total / num_updates
            self.anyadapter_metrics[key] = value
        # Preserve the component key schema previously consumed by the runner.
        self.anyadapter_metrics.update({
            "world_model_loss_ang_vel": self.anyadapter_metrics["wm_ang_vel_loss"],
            "world_model_loss_orientation": self.anyadapter_metrics["wm_orientation_loss"],
            "world_model_loss_dof_pos": self.anyadapter_metrics["wm_dof_pos_loss"],
            "world_model_loss_dof_vel": self.anyadapter_metrics["wm_dof_vel_loss"],
        })
        if hasattr(self, "_deferred_auxiliary_update"):
            deferred_metrics = self._deferred_auxiliary_update()
            self.anyadapter_metrics.update(deferred_metrics)
        self.storage.clear()
        self.update_counter()
        return (
            mean_value_loss / num_updates,
            mean_surrogate_loss / num_updates,
            0.0,
            self.anyadapter_metrics.get(
                "world_model_loss", mean_wm_loss / num_updates
            ),
            mean_adapter_reg_loss / num_updates,
            0.0,
        )

    def update_dagger(self):
        """AnyAdapter does not use TWIST/RMA latent DAgger updates.

        OnPolicyRunnerMimic calls update_dagger() for every algorithm whose
        class name is not exactly "PPO".  The original implementation assumes
        actor_critic.actor exposes infer_priv_latent/infer_hist_latent, which
        is specific to the RMA actor.  AnyAdapter trains its history encoder
        through PPO plus the world-model auxiliary loss instead.
        """
        return 0.0
