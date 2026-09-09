"""PPO parameter ownership and auxiliary updates for DTERA."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Optional

from .ppo_anyadapter import PPOAnyAdapter


class PPODTERA(PPOAnyAdapter):
    def __init__(
        self,
        *args,
        risk_horizon: int = 10,
        risk_pos_weight: float = 5.0,
        risk_adaptive_pos_weight: bool = True,
        risk_pos_weight_max: float = 20.0,
        risk_learning_rate: Optional[float] = None,
        fixed_action_std: Optional[float] = None,
        world_model_bootstrap: bool = True,
        world_model_bootstrap_probability: float = 0.8,
        **kwargs,
    ):
        kwargs["joint_encoder_optimization"] = True
        kwargs["defer_world_model_update"] = True
        super().__init__(*args, **kwargs)
        if not getattr(self.actor_critic, "is_dtera", False):
            raise TypeError("PPODTERA requires TwistDTERAActorCritic")
        self.risk_horizon = int(risk_horizon)
        self.risk_pos_weight = float(risk_pos_weight)
        self.risk_adaptive_pos_weight = bool(risk_adaptive_pos_weight)
        self.risk_pos_weight_max = float(risk_pos_weight_max)
        if self.risk_pos_weight_max < 1.0:
            raise ValueError("risk_pos_weight_max must be at least 1")
        self.fixed_action_std = (
            None if fixed_action_std is None else float(fixed_action_std)
        )
        if self.fixed_action_std is not None and self.fixed_action_std <= 0.0:
            raise ValueError("fixed_action_std must be positive")
        if self.fixed_action_std is not None:
            # The shared CLI helper historically overwrites policy config with
            # --fix_action_std=False when the flag is omitted.  DTERA's
            # fixed_action_std is authoritative and must also disable PPO
            # likelihood gradients on std within an update.
            self.actor_critic.std.requires_grad_(False)
        self.world_model_bootstrap = bool(world_model_bootstrap)
        self.world_model_bootstrap_probability = float(
            world_model_bootstrap_probability
        )

        adapter_critic_params = [
            p
            for module in (self.actor_critic.adapter, self.actor_critic.critic)
            for p in module.parameters()
            if p.requires_grad
        ]
        # Keep the std parameter group present even when std is frozen.  Apart
        # from making the ownership explicit, this preserves the parameter
        # group layout of older DTERA checkpoints whose std was trainable, so
        # their PPO optimizer state can still be restored for fine-tuning.
        std_params = [self.actor_critic.std]
        tracking_aux_params = [
            p
            for module in (
                self.actor_critic.tracking_error_history_encoder,
                self.actor_critic.error_trend_predictor,
            )
            for p in module.parameters()
            if p.requires_grad
        ]
        self.ppo_params = adapter_critic_params + std_params + tracking_aux_params
        self.history_encoder_params = [
            p for p in self.actor_critic.history_encoder.parameters() if p.requires_grad
        ]
        self.world_model_params = [
            p for p in self.actor_critic.world_model.parameters() if p.requires_grad
        ]
        self.wm_params = self.history_encoder_params + self.world_model_params
        self.risk_params = [
            p for p in self.actor_critic.risk_predictor.parameters() if p.requires_grad
        ]

        self.ppo_optimizer = torch.optim.Adam(
            [
                {"params": adapter_critic_params, "weight_decay": self.weight_decay},
                {"params": std_params, "weight_decay": 0.0},
                {"params": tracking_aux_params, "weight_decay": 0.0},
            ],
            lr=self.learning_rate,
        )
        self.wm_optimizer = torch.optim.Adam(
            self.wm_params, lr=self.learning_rate, weight_decay=0.0
        )
        self.risk_optimizer = torch.optim.Adam(
            self.risk_params,
            lr=self.learning_rate if risk_learning_rate is None else risk_learning_rate,
            weight_decay=0.0,
        )
        self.optimizer = self.ppo_optimizer
        self._assert_optimizer_ownership()
        self._enforce_fixed_action_std()

    @staticmethod
    def _optimizer_ids(optimizer):
        return {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }

    def _assert_optimizer_ownership(self):
        ppo_ids = self._optimizer_ids(self.ppo_optimizer)
        wm_ids = self._optimizer_ids(self.wm_optimizer)
        risk_ids = self._optimizer_ids(self.risk_optimizer)
        if ppo_ids & wm_ids or ppo_ids & risk_ids or wm_ids & risk_ids:
            raise AssertionError("a DTERA parameter is owned by more than one optimizer")
        trainable_ids = {
            id(parameter)
            for parameter in self.actor_critic.parameters()
            if parameter.requires_grad
        }
        owned_ids = ppo_ids | wm_ids | risk_ids
        allowed_non_trainable_ids = {id(self.actor_critic.std)}
        missing_ids = trainable_ids - owned_ids
        unexpected_ids = owned_ids - trainable_ids - allowed_non_trainable_ids
        if missing_ids or unexpected_ids:
            missing = len(missing_ids)
            extra = len(unexpected_ids)
            raise AssertionError(f"DTERA optimizer ownership mismatch: missing={missing}, extra={extra}")

        dyn_ids = {id(p) for p in self.actor_critic.history_encoder.parameters()}
        tracking_ids = {
            id(p) for p in self.actor_critic.tracking_error_history_encoder.parameters()
        }
        risk_predictor_ids = {id(p) for p in self.actor_critic.risk_predictor.parameters()}
        diagnostics = {
            "dynamics_encoder_in_ppo": bool(dyn_ids & ppo_ids),
            "dynamics_encoder_in_wm": dyn_ids <= wm_ids,
            "tracking_encoder_in_ppo": tracking_ids <= ppo_ids,
            "tracking_encoder_in_wm": bool(tracking_ids & wm_ids),
            "risk_predictor_only_in_risk": (
                risk_predictor_ids <= risk_ids
                and not bool(risk_predictor_ids & (ppo_ids | wm_ids))
            ),
        }
        self.optimizer_ownership = diagnostics
        for key, value in diagnostics.items():
            print(f"[DTERA] {key}={value}")
        expected = {
            "dynamics_encoder_in_ppo": False,
            "dynamics_encoder_in_wm": True,
            "tracking_encoder_in_ppo": True,
            "tracking_encoder_in_wm": False,
            "risk_predictor_only_in_risk": True,
        }
        if diagnostics != expected:
            raise AssertionError(f"unexpected DTERA optimizer ownership: {diagnostics}")

    def _enforce_fixed_action_std(self):
        if self.fixed_action_std is None:
            return
        self.actor_critic.update_std(self.fixed_action_std)

    def on_load_checkpoint(self, loaded_dict=None):
        """Restore the configured exploration scale after loading old weights."""
        self._enforce_fixed_action_std()
        if (
            loaded_dict is not None
            and getattr(
                self.actor_critic,
                "_loaded_without_residual_iteration",
                False,
            )
            and hasattr(self.actor_critic, "set_training_iteration")
        ):
            self.actor_critic.set_training_iteration(loaded_dict.get("iter", 0))
        if self.fixed_action_std is not None:
            print(f"[DTERA] fixed_action_std={self.fixed_action_std:.6f}")

    def process_env_step(self, rewards, dones, infos):
        timeouts = infos.get("time_outs", None)
        self.transition.timeouts = (
            torch.zeros_like(dones) if timeouts is None else timeouts.detach()
        )
        return super().process_env_step(rewards, dones, infos)

    def _world_model_loss_from_batch(
        self,
        obs_batch,
        actions_batch,
        next_obs_batch,
        dones_batch,
        available_batch,
    ):
        indices = self.actor_critic.wm_target_indices.to(next_obs_batch.device)
        target = next_obs_batch.index_select(1, indices)
        members = self.actor_critic.predict_world_model_members(obs_batch, actions_batch)
        self.actor_critic.update_uncertainty_ema(members)
        ensemble_mean = members.mean(dim=0)
        _, uncertainty = self.actor_critic.confidence_from_members(members)
        error_scale = self.actor_critic.wm_variance_ema.sqrt().clamp_min(1e-3)
        actual_error = torch.sqrt(
            torch.mean(
                ((ensemble_mean - target) / error_scale).square(), dim=-1
            ).clamp_min(0.0)
        )
        calibration_valid = (
            (1.0 - dones_batch.float()).reshape(-1)
            * available_batch.float().reshape(-1)
        ).bool()
        valid = (1.0 - dones_batch.float()).reshape(1, -1, 1)
        valid = valid * available_batch.float().reshape(1, -1, 1)
        if self.world_model_bootstrap:
            bootstrap = (
                torch.rand(
                    members.shape[0], members.shape[1], 1,
                    device=members.device,
                )
                < self.world_model_bootstrap_probability
            ).float()
            # Every member must retain at least one valid sample.
            empty = (bootstrap * valid).sum(dim=1, keepdim=True) == 0
            bootstrap = torch.where(empty, torch.ones_like(bootstrap), bootstrap)
            valid = valid * bootstrap

        total = members.new_zeros(())
        component_info = {}
        for name, (start, end) in self.actor_critic.wm_component_splits.items():
            pred = members[:, :, start:end]
            truth = target[None, :, start:end]
            if self.world_model_loss_type == "mse":
                per_sample = (pred - truth).square().mean(dim=-1, keepdim=True)
            else:
                per_sample = F.smooth_l1_loss(
                    pred, truth.expand_as(pred), reduction="none"
                ).mean(dim=-1, keepdim=True)
            member_counts = valid.sum(dim=1).clamp_min(1.0)
            member_losses = (per_sample * valid).sum(dim=1) / member_counts
            component_loss = member_losses.mean()
            component_info[f"wm_{name}_loss"] = float(component_loss.detach().cpu())
            total = total + float(
                self.actor_critic.wm_component_weights.get(name, 1.0)
            ) * component_loss
        component_info.update(
            self.uncertainty_error_calibration(
                uncertainty.detach(), actual_error.detach(), calibration_valid
            )
        )
        return total, component_info

    @staticmethod
    def uncertainty_error_calibration(uncertainty, actual_error, valid_mask=None):
        uncertainty = uncertainty.reshape(-1).float()
        actual_error = actual_error.reshape(-1).float()
        if valid_mask is not None:
            valid_mask = valid_mask.reshape(-1).bool()
            uncertainty = uncertainty[valid_mask]
            actual_error = actual_error[valid_mask]
        if uncertainty.numel() < 2:
            return {
                "wm_uncertainty_error_corr": 0.0,
                "actual_wm_error_low_uncertainty": 0.0,
                "actual_wm_error_high_uncertainty": 0.0,
            }
        centered_u = uncertainty - uncertainty.mean()
        centered_e = actual_error - actual_error.mean()
        denom = torch.sqrt(
            centered_u.square().sum() * centered_e.square().sum()
        ).clamp_min(1e-12)
        correlation = (centered_u * centered_e).sum() / denom
        count = max(int(uncertainty.numel() * 0.2), 1)
        order = torch.argsort(uncertainty)
        return {
            "wm_uncertainty_error_corr": float(correlation.cpu()),
            "actual_wm_error_low_uncertainty": float(
                actual_error[order[:count]].mean().cpu()
            ),
            "actual_wm_error_high_uncertainty": float(
                actual_error[order[-count:]].mean().cpu()
            ),
        }

    def _future_risk_targets(self):
        dones = self.storage.dones.bool()
        failures = dones & ~self.storage.timeouts.bool()
        targets = torch.zeros_like(dones, dtype=torch.float32)
        valid = torch.zeros_like(dones, dtype=torch.bool)
        unresolved = torch.ones_like(dones, dtype=torch.bool)
        horizon = min(self.risk_horizon, self.storage.num_transitions_per_env)
        for offset in range(horizon):
            available = torch.zeros_like(dones)
            shifted_done = torch.zeros_like(dones)
            shifted_failure = torch.zeros_like(failures)
            if offset == 0:
                available[:] = True
                shifted_done[:] = dones
                shifted_failure[:] = failures
            else:
                available[:-offset] = True
                shifted_done[:-offset] = dones[offset:]
                shifted_failure[:-offset] = failures[offset:]
            found_failure = unresolved & available & shifted_failure
            found_termination = unresolved & available & shifted_done
            targets[found_failure] = 1.0
            valid |= found_termination
            unresolved &= ~found_termination
            if offset == horizon - 1:
                valid |= unresolved & available
        return targets, valid

    def _train_risk_predictor(self, risk_targets=None, risk_valid=None):
        if risk_targets is None or risk_valid is None:
            risk_targets, risk_valid = self._future_risk_targets()
        flat_valid = risk_valid.flatten(0, 1).reshape(-1).bool()
        targets = risk_targets.flatten(0, 1).reshape(-1)[flat_valid]
        observations = self.storage.observations.flatten(0, 1)[flat_valid]
        actions = self.storage.actions.flatten(0, 1)[flat_valid]
        positive_count = int(targets.sum().item())
        negative_count = int(targets.numel() - positive_count)
        common = {
            "risk_valid_ratio": float(risk_valid.float().mean().cpu()),
            "risk_positive_ratio": (
                float(targets.mean().cpu()) if targets.numel() else 0.0
            ),
            "risk_num_positive": float(positive_count),
            "risk_num_negative": float(negative_count),
        }
        if positive_count == 0:
            return {
                **common,
                "risk_loss": 0.0,
                "risk_update_skipped": 1.0,
                "risk_effective_pos_weight": 0.0,
                "risk_prob_positive": 0.0,
                "risk_prob_negative": self._mean_risk_probability(
                    observations, actions
                ),
                "risk_probability_gap": 0.0,
            }
        batch_size = 4096
        order = torch.randperm(targets.numel(), device=targets.device)
        total_loss = 0.0
        batches = 0
        if self.risk_adaptive_pos_weight:
            effective_pos_weight = min(
                max(negative_count / max(positive_count, 1), 1.0),
                self.risk_pos_weight_max,
            )
        else:
            effective_pos_weight = self.risk_pos_weight
        pos_weight = targets.new_tensor(effective_pos_weight)
        for start in range(0, targets.numel(), batch_size):
            ids = order[start : start + batch_size]
            logits = self.actor_critic.risk_logits(observations[ids], actions[ids])
            loss = F.binary_cross_entropy_with_logits(
                logits, targets[ids], pos_weight=pos_weight
            )
            self.risk_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.risk_params, self.max_grad_norm)
            self.risk_optimizer.step()
            total_loss += loss.item()
            batches += 1
        positive_mask = targets.bool()
        prob_positive = self._mean_risk_probability(
            observations[positive_mask], actions[positive_mask]
        )
        prob_negative = self._mean_risk_probability(
            observations[~positive_mask], actions[~positive_mask]
        )
        return {
            **common,
            "risk_loss": total_loss / max(batches, 1),
            "risk_update_skipped": 0.0,
            "risk_effective_pos_weight": float(effective_pos_weight),
            "risk_prob_positive": prob_positive,
            "risk_prob_negative": prob_negative,
            "risk_probability_gap": prob_positive - prob_negative,
        }

    def _mean_risk_probability(self, observations, actions):
        if observations.shape[0] == 0:
            return 0.0
        total = 0.0
        count = 0
        with torch.no_grad():
            for start in range(0, observations.shape[0], 4096):
                probs = torch.sigmoid(
                    self.actor_critic.risk_logits(
                        observations[start : start + 4096],
                        actions[start : start + 4096],
                    )
                )
                total += float(probs.sum().cpu())
                count += probs.numel()
        return total / max(count, 1)

    def _deferred_auxiliary_update(self):
        """Run WM then Risk only after all PPO minibatches are complete."""
        metrics = {}
        wm_keys = (
            "wm_ang_vel_loss", "wm_orientation_loss", "wm_dof_pos_loss",
            "wm_dof_vel_loss", "wm_uncertainty_error_corr",
            "actual_wm_error_low_uncertainty",
            "actual_wm_error_high_uncertainty",
        )
        totals = {key: 0.0 for key in wm_keys}
        wm_loss_total = 0.0
        wm_grad_total = 0.0
        wm_batches = 0
        if self.world_model_loss_coef > 0.0 and self._has_world_model_target():
            generator = self.storage.mini_batch_generator_anyadapter(
                self.num_mini_batches, self.num_learning_epochs
            )
            for sample in generator:
                obs, _, actions, *rest = sample
                next_obs, dones, available = rest[-3:]
                self.wm_optimizer.zero_grad()
                wm_loss, info = self._world_model_loss_from_batch(
                    obs, actions, next_obs, dones, available
                )
                (self.world_model_loss_coef * wm_loss).backward()
                wm_grad_total += self._grad_norm(self.history_encoder_params)
                torch.nn.utils.clip_grad_norm_(self.wm_params, self.max_grad_norm)
                self.wm_optimizer.step()
                wm_loss_total += float(wm_loss.detach().cpu())
                for key in totals:
                    totals[key] += info.get(key, 0.0)
                wm_batches += 1
        else:
            metrics["world_model_loss_skipped"] = 1.0
        metrics["world_model_loss"] = wm_loss_total / max(wm_batches, 1)
        metrics["history_encoder_wm_grad_norm"] = (
            wm_grad_total / max(wm_batches, 1)
        )
        # Dynamics encoder receives no PPO gradient in DTERA, so after the
        # deferred WM phase its total update gradient equals its WM gradient.
        metrics["history_encoder_total_grad_norm"] = metrics[
            "history_encoder_wm_grad_norm"
        ]
        for key, total in totals.items():
            metrics[key] = total / max(wm_batches, 1)
        metrics.update({
            "world_model_loss_ang_vel": metrics["wm_ang_vel_loss"],
            "world_model_loss_orientation": metrics["wm_orientation_loss"],
            "world_model_loss_dof_pos": metrics["wm_dof_pos_loss"],
            "world_model_loss_dof_vel": metrics["wm_dof_vel_loss"],
        })
        risk_targets, risk_valid = self._future_risk_targets()
        metrics.update(self._train_risk_predictor(risk_targets, risk_valid))
        return metrics

    def update(self):
        # This also protects a resumed run before its first optimizer step if
        # the checkpoint was produced while action std was still trainable.
        self._enforce_fixed_action_std()
        self.wm_optimizer.zero_grad()
        self.risk_optimizer.zero_grad()
        result = super().update()
        self._enforce_fixed_action_std()
        if hasattr(self.actor_critic, "set_training_iteration"):
            self.actor_critic.set_training_iteration(
                int(self.actor_critic.residual_training_iteration.item()) + 1
            )
        return result
