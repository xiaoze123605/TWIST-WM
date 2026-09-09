"""PPO training loop for the layerwise TWIST Any2Track adapter."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ppo_anyadapter import PPOAnyAdapter


_DEFAULT_LEG_DOF_POS = (
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
)


class PPOAny2Track(PPOAnyAdapter):
    """Train the world model first, then PPO with a detached dynamics embedding."""

    def __init__(
        self,
        *args,
        world_model_learning_rate: float = 1e-4,
        policy_learning_rate: float = 1e-4,
        world_model_sequence_length: int = 20,
        world_model_num_epochs: int = 1,
        world_model_component_weights=(5.0, 5.0, 1.0, 0.5),
        **kwargs,
    ):
        kwargs["joint_encoder_optimization"] = False
        # Any2Track intentionally trains its history encoder only through the
        # separate autoregressive world-model update.
        kwargs["separate_wm_updates_history_encoder"] = True
        super().__init__(*args, **kwargs)
        self.world_model_sequence_length = int(world_model_sequence_length)
        self.world_model_num_epochs = int(world_model_num_epochs)
        if len(world_model_component_weights) != 4:
            raise ValueError("world_model_component_weights must contain four values")
        self.world_model_component_weights = {
            "ang_vel": float(world_model_component_weights[0]),
            "orientation": float(world_model_component_weights[1]),
            "dof_pos": float(world_model_component_weights[2]),
            "dof_vel": float(world_model_component_weights[3]),
        }
        self.learning_rate = float(policy_learning_rate)
        self.ppo_optimizer = torch.optim.Adam(
            self.ppo_params,
            lr=float(policy_learning_rate),
            weight_decay=self.weight_decay,
        )
        self.wm_optimizer = torch.optim.Adam(
            self.wm_params,
            lr=float(world_model_learning_rate),
        )
        self.optimizer = self.ppo_optimizer

    def _autoregressive_world_model_loss(
        self,
        obs_sequence,
        action_sequence,
        next_obs_sequence,
        done_sequence,
        available_sequence,
    ):
        actor_critic = self.actor_critic
        _, history = actor_critic.split_obs(obs_sequence[:, 0])
        state = actor_critic.selected_state(obs_sequence[:, 0])
        component_sums = {
            name: state.new_zeros(()) for name in actor_critic.wm_component_splits
        }
        valid_total = state.new_zeros(())

        for step in range(obs_sequence.shape[1]):
            prediction = actor_critic.predict_world_model(
                obs_sequence[:, step],
                action_sequence[:, step],
                state=state,
                history=history,
            )
            target = actor_critic.selected_state(next_obs_sequence[:, step])
            done = done_sequence[:, step].bool().reshape(-1, 1)
            available = available_sequence[:, step].float().reshape(-1, 1)
            valid = available * (~done).float()
            valid_total = valid_total + valid.sum()

            for name, (start, end) in actor_critic.wm_component_splits.items():
                if self.world_model_loss_type == "mse":
                    per_dim = (prediction[:, start:end] - target[:, start:end]).square()
                else:
                    per_dim = F.smooth_l1_loss(
                        prediction[:, start:end], target[:, start:end], reduction="none"
                    )
                component_sums[name] = component_sums[name] + (
                    per_dim.mean(dim=-1, keepdim=True) * valid
                ).sum()

            predicted_frame = torch.cat([prediction, action_sequence[:, step]], dim=-1)
            predicted_history = torch.cat(
                [history[:, 1:], predicted_frame.unsqueeze(1)], dim=1
            )
            _, reset_history = actor_critic.split_obs(next_obs_sequence[:, step])
            history = torch.where(done.unsqueeze(-1), reset_history, predicted_history)
            state = torch.where(done, target, prediction)

        denominator = valid_total.clamp_min(1.0)
        component_losses = {
            name: value / denominator for name, value in component_sums.items()
        }
        total = sum(
            self.world_model_component_weights[name] * component_losses[name]
            for name in component_losses
        )
        return total, component_losses, int(valid_total.detach().item())

    def _update_world_model(self):
        if self.world_model_loss_coef <= 0.0:
            return 0.0, {}, 0.0, True
        if not self._has_world_model_target():
            self._warn_missing_world_model_target_once()
            return 0.0, {}, 0.0, True

        total_loss = 0.0
        total_grad_norm = 0.0
        component_totals = {
            name: 0.0 for name in self.actor_critic.wm_component_splits
        }
        update_count = 0
        generator = self.storage.world_model_sequence_generator(
            self.world_model_sequence_length,
            self.num_mini_batches,
            self.world_model_num_epochs,
        )
        for sequence_batch in generator:
            loss, components, valid_count = self._autoregressive_world_model_loss(*sequence_batch)
            if valid_count == 0:
                continue
            self.wm_optimizer.zero_grad()
            (self.world_model_loss_coef * loss).backward()
            grad_norm = self._grad_norm(self.wm_params)
            nn.utils.clip_grad_norm_(self.wm_params, self.max_grad_norm)
            self.wm_optimizer.step()
            total_loss += float(loss.detach().cpu())
            total_grad_norm += grad_norm
            for name, value in components.items():
                component_totals[name] += float(value.detach().cpu())
            update_count += 1

        if update_count == 0:
            return 0.0, component_totals, 0.0, True
        return (
            total_loss / update_count,
            {name: value / update_count for name, value in component_totals.items()},
            total_grad_norm / update_count,
            False,
        )

    def _in_place_leg_mask(self, observations):
        base_obs, _ = self.actor_critic.split_obs(observations)
        reference_velocity = base_obs[:, 4:7]
        reference_yaw_velocity = base_obs[:, 7]
        reference_leg_pose = base_obs[:, 8:20]
        default_leg_pose = reference_leg_pose.new_tensor(
            _DEFAULT_LEG_DOF_POS
        ).view(1, -1)
        return (
            (
                torch.norm(reference_velocity[:, :2], dim=-1)
                < self.stand_vel_threshold
            )
            & (torch.abs(reference_yaw_velocity) < self.stand_vel_threshold)
            & (
                torch.mean(
                    torch.abs(reference_leg_pose - default_leg_pose), dim=-1
                )
                < self.stand_dof_threshold
            )
        )

    def _in_place_leg_anchor_loss(self, observations, delta):
        stand_mask = self._in_place_leg_mask(observations)
        stand_ratio = stand_mask.float().mean()
        if not torch.any(stand_mask):
            return observations.new_zeros(()), stand_ratio
        return delta[stand_mask, :12].square().mean(), stand_ratio

    def _synthetic_in_place_leg_anchor_loss(self, observations):
        stand_observations = observations.clone()
        base_obs, _ = self.actor_critic.split_obs(stand_observations)
        if self.actor_critic.base_obs_dim % 105 != 0:
            raise RuntimeError("TWIST base observation must contain 105-D frames")
        default_leg_pose = base_obs.new_tensor(_DEFAULT_LEG_DOF_POS)
        for frame_start in range(0, self.actor_critic.base_obs_dim, 105):
            base_obs[:, frame_start + 4 : frame_start + 8] = 0.0
            base_obs[:, frame_start + 8 : frame_start + 20] = default_leg_pose
        delta = self.actor_critic.action_delta(stand_observations)
        return delta[:, :12].square().mean()

    def update(self):
        wm_loss, wm_components, wm_grad_norm, wm_skipped = self._update_world_model()

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_delta_l2 = 0.0
        mean_delta_abs = 0.0
        mean_delta_max = 0.0
        mean_adapter_reg = 0.0
        mean_stand_anchor = 0.0
        mean_synthetic_stand_anchor = 0.0
        mean_stand_ratio = 0.0
        mean_adapter_grad_norm = 0.0

        # The world model has already consumed next observations above.  PPO
        # only needs the current policy/critic batch, so avoid gathering a
        # second 7001-D observation tensor for every mini-batch.  At 4096 envs
        # that unused copy alone can cost hundreds of MiB.
        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )
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
            ) = sample

            self.actor_critic.act(
                obs_batch,
                masks=masks_batch,
                hidden_states=hid_states_batch[0],
            )
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch,
                masks=masks_batch,
                hidden_states=hid_states_batch[1],
            )
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1e-5)
                        + (
                            old_sigma_batch.square()
                            + (old_mu_batch - mu_batch).square()
                        )
                        / (2.0 * sigma_batch.square())
                        - 0.5,
                        dim=-1,
                    ).mean()
                    if kl > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif 0.0 < kl < self.desired_kl / 2.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for group in self.ppo_optimizer.param_groups:
                        group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch.squeeze(-1))
            surrogate = -advantages_batch.squeeze(-1) * ratio
            surrogate_clipped = -advantages_batch.squeeze(-1) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.maximum(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_loss = torch.maximum(
                    (value_batch - returns_batch).square(),
                    (value_clipped - returns_batch).square(),
                ).mean()
            else:
                value_loss = (returns_batch - value_batch).square().mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )
            needs_delta_grad = (
                self.adapter_reg_coef > 0.0 or self.stand_anchor_coef > 0.0
            )
            if needs_delta_grad:
                delta = self.actor_critic.action_delta(obs_batch)
            else:
                with torch.no_grad():
                    delta = self.actor_critic.action_delta(obs_batch)
            adapter_reg = delta.square().mean()
            adapter_info = {
                "adapter_delta_l2": float(
                    delta.norm(dim=-1).mean().detach().cpu()
                ),
                "mean_abs_delta_action": float(delta.abs().mean().detach().cpu()),
                "max_abs_delta_action": float(delta.abs().max().detach().cpu()),
            }
            if self.adapter_reg_coef > 0.0:
                loss = loss + self.adapter_reg_coef * adapter_reg

            stand_anchor = obs_batch.new_zeros(())
            stand_ratio = obs_batch.new_zeros(())
            if self.stand_anchor_coef > 0.0:
                stand_anchor, stand_ratio = self._in_place_leg_anchor_loss(
                    obs_batch, delta
                )
                loss = loss + self.stand_anchor_coef * stand_anchor

            synthetic_stand_anchor = obs_batch.new_zeros(())
            if self.synthetic_stand_anchor_coef > 0.0:
                synthetic_stand_anchor = (
                    self._synthetic_in_place_leg_anchor_loss(obs_batch)
                )
                loss = (
                    loss
                    + self.synthetic_stand_anchor_coef * synthetic_stand_anchor
                )

            self.ppo_optimizer.zero_grad()
            loss.backward()
            adapter_grad_norm = self._grad_norm(self.actor_critic.adapter.parameters())
            nn.utils.clip_grad_norm_(self.ppo_params, self.max_grad_norm)
            self.ppo_optimizer.step()

            mean_value_loss += float(value_loss.detach().cpu())
            mean_surrogate_loss += float(surrogate_loss.detach().cpu())
            mean_entropy += float(entropy_batch.mean().detach().cpu())
            mean_adapter_reg += float(adapter_reg.detach().cpu())
            mean_stand_anchor += float(stand_anchor.detach().cpu())
            mean_synthetic_stand_anchor += float(
                synthetic_stand_anchor.detach().cpu()
            )
            mean_stand_ratio += float(stand_ratio.detach().cpu())
            mean_delta_l2 += adapter_info.get("adapter_delta_l2", 0.0)
            mean_delta_abs += adapter_info.get("mean_abs_delta_action", 0.0)
            mean_delta_max += adapter_info.get("max_abs_delta_action", 0.0)
            mean_adapter_grad_norm += adapter_grad_norm

        self.storage.clear()
        self.anyadapter_metrics = {
            "world_model_loss": wm_loss,
            "world_model_loss_skipped": float(wm_skipped),
            "world_model_loss_ang_vel": wm_components.get("ang_vel", 0.0),
            "world_model_loss_orientation": wm_components.get("orientation", 0.0),
            "world_model_loss_dof_pos": wm_components.get("dof_pos", 0.0),
            "world_model_loss_dof_vel": wm_components.get("dof_vel", 0.0),
            "history_encoder_wm_grad_norm": wm_grad_norm,
            "history_encoder_ppo_grad_norm": 0.0,
            "adapter_delta_l2": mean_delta_l2 / num_updates,
            "adapter_reg_loss": mean_adapter_reg / num_updates,
            "stand_anchor_loss": mean_stand_anchor / num_updates,
            "synthetic_stand_anchor_loss": (
                mean_synthetic_stand_anchor / num_updates
            ),
            "stand_sample_ratio": mean_stand_ratio / num_updates,
            "adapter_grad_norm": mean_adapter_grad_norm / num_updates,
            "mean_abs_delta_action": mean_delta_abs / num_updates,
            "max_abs_delta_action": mean_delta_max / num_updates,
            "surrogate_loss": mean_surrogate_loss / num_updates,
            "value_loss": mean_value_loss / num_updates,
            "entropy": mean_entropy / num_updates,
        }
        self.update_counter()
        return (
            mean_value_loss / num_updates,
            mean_surrogate_loss / num_updates,
            0.0,
            wm_loss,
            mean_adapter_reg / num_updates,
            0.0,
        )
