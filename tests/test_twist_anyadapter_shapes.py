from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "rsl_rl"))

import torch

from rsl_rl.modules import TwistAnyAdapterActorCritic


class DummyBaseActor(torch.nn.Module):
    def __init__(self, base_obs_dim: int, num_actions: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(base_obs_dim, num_actions)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.linear(obs)


def make_dummy_base_actor_jit(base_obs_dim: int, num_actions: int, path: str) -> None:
    actor = DummyBaseActor(base_obs_dim, num_actions).eval()
    example = torch.zeros(1, base_obs_dim)
    traced = torch.jit.trace(actor, example)
    traced.save(path)


def main() -> None:
    torch.manual_seed(0)

    base_obs_dim = 100
    history_len = 20
    hist_state_dim = 40
    num_actions = 29
    history_frame_dim = hist_state_dim + num_actions
    total_obs_dim = base_obs_dim + history_len * history_frame_dim

    obs = torch.randn(4, total_obs_dim)

    with tempfile.TemporaryDirectory() as tmpdir:
        base_actor_jit_path = os.path.join(tmpdir, "dummy_base_actor.pt")
        make_dummy_base_actor_jit(base_obs_dim, num_actions, base_actor_jit_path)

        actor_critic = TwistAnyAdapterActorCritic(
            num_prop=base_obs_dim,
            num_critic_obs=total_obs_dim,
            num_priv_latent=0,
            num_hist=history_len,
            num_actions=num_actions,
            base_actor_jit_path=base_actor_jit_path,
            base_obs_dim=base_obs_dim,
            history_len=history_len,
            history_frame_dim=history_frame_dim,
            hist_state_dim=hist_state_dim,
            latent_dim=16,
            adapter_hidden_dims=[32, 32],
            critic_hidden_dims=[64, 32],
            world_model_hidden_dims=[64, 32],
            use_conv_history=True,
            freeze_base=True,
        )
        actor_critic.eval()

        with torch.no_grad():
            action = actor_critic.act_inference(obs)
            value = actor_critic.evaluate(obs)
            adapter_delta = actor_critic.get_adapter_delta(obs)
            world_pred = actor_critic.predict_world_model(obs)

        assert list(action.shape) == [4, num_actions], action.shape
        assert list(value.shape) == [4, 1], value.shape
        assert list(adapter_delta.shape) == [4, num_actions], adapter_delta.shape
        assert world_pred.shape[0] == 4, world_pred.shape
        assert abs(adapter_delta.mean().item()) < 1e-6, adapter_delta.mean().item()

        tracking_actor_critic = TwistAnyAdapterActorCritic(
            num_prop=base_obs_dim,
            num_critic_obs=total_obs_dim,
            num_priv_latent=0,
            num_hist=history_len,
            num_actions=num_actions,
            base_actor_jit_path=base_actor_jit_path,
            base_obs_dim=base_obs_dim,
            history_len=history_len,
            history_frame_dim=history_frame_dim,
            hist_state_dim=hist_state_dim,
            latent_dim=16,
            adapter_hidden_dims=[32, 32],
            critic_hidden_dims=[64, 32],
            world_model_hidden_dims=[64, 32],
            action_delta_scale=0.05,
            use_tracking_error_adapter_input=True,
            use_conv_history=True,
            freeze_base=True,
        )
        tracking_actor_critic.eval()
        with torch.no_grad():
            tracking_action = tracking_actor_critic.act_inference(obs)
            tracking_delta = tracking_actor_critic.get_adapter_delta(obs)
        assert list(tracking_action.shape) == [4, num_actions], tracking_action.shape
        assert list(tracking_delta.shape) == [4, num_actions], tracking_delta.shape
        assert abs(tracking_delta.mean().item()) < 1e-6, tracking_delta.mean().item()

        compact_actor_critic = TwistAnyAdapterActorCritic(
            num_prop=base_obs_dim,
            num_critic_obs=total_obs_dim,
            num_priv_latent=0,
            num_hist=history_len,
            num_actions=num_actions,
            base_actor_jit_path=base_actor_jit_path,
            base_obs_dim=base_obs_dim,
            history_len=history_len,
            history_frame_dim=history_frame_dim,
            hist_state_dim=hist_state_dim,
            latent_dim=16,
            adapter_hidden_dims=[32, 32],
            critic_hidden_dims=[64, 32],
            world_model_hidden_dims=[64, 32],
            action_delta_scale=0.10,
            use_tracking_error_adapter_input=True,
            compact_adapter_input=True,
            history_policy_grad_scale=0.10,
            use_conv_history=True,
            freeze_base=True,
        )
        compact_actor_critic.eval()
        with torch.no_grad():
            compact_action = compact_actor_critic.act_inference(obs)
            compact_delta = compact_actor_critic.get_adapter_delta(obs)
        assert list(compact_action.shape) == [4, num_actions], compact_action.shape
        assert list(compact_delta.shape) == [4, num_actions], compact_delta.shape
        expected_adapter_input = num_actions + hist_state_dim + 16 + num_actions + 6
        assert compact_actor_critic.adapter.net[0].in_features == expected_adapter_input
        assert abs(compact_delta.mean().item()) < 1e-6, compact_delta.mean().item()

        heading_obs = torch.randn(4, total_obs_dim + 2)
        heading_actor_critic = TwistAnyAdapterActorCritic(
            num_prop=base_obs_dim,
            num_critic_obs=total_obs_dim + 2,
            num_priv_latent=0,
            num_hist=history_len,
            num_actions=num_actions,
            base_actor_jit_path=base_actor_jit_path,
            base_obs_dim=base_obs_dim,
            history_len=history_len,
            history_frame_dim=history_frame_dim,
            hist_state_dim=hist_state_dim,
            latent_dim=16,
            adapter_hidden_dims=[32, 32],
            critic_hidden_dims=[64, 32],
            world_model_hidden_dims=[64, 32],
            action_delta_scale=0.05,
            use_tracking_error_adapter_input=True,
            compact_adapter_input=True,
            history_policy_grad_scale=0.10,
            adapter_context_dim=2,
            use_conv_history=True,
            freeze_base=True,
        )
        heading_actor_critic.eval()
        with torch.no_grad():
            heading_action = heading_actor_critic.act_inference(heading_obs)
            heading_delta = heading_actor_critic.get_adapter_delta(heading_obs)
            heading_bias_loss = heading_actor_critic.adapter_bias_regularization_loss(
                heading_obs
            )
        assert list(heading_action.shape) == [4, num_actions], heading_action.shape
        assert list(heading_delta.shape) == [4, num_actions], heading_delta.shape
        assert heading_bias_loss.ndim == 0
        assert heading_actor_critic.adapter.net[0].in_features == expected_adapter_input + 2
        assert abs(heading_delta.mean().item()) < 1e-6, heading_delta.mean().item()

        # Dual branch mode uses the production action count and keeps dynamics
        # and tracking inputs structurally independent.
        dual_num_actions = 23
        dual_history_frame_dim = hist_state_dim + dual_num_actions
        dual_total_obs_dim = base_obs_dim + history_len * dual_history_frame_dim
        dual_obs = torch.randn(4, dual_total_obs_dim)
        dual_base_actor_jit_path = os.path.join(tmpdir, "dummy_dual_base_actor.pt")
        make_dummy_base_actor_jit(
            base_obs_dim,
            dual_num_actions,
            dual_base_actor_jit_path,
        )

        def make_dual_actor(history_policy_grad_scale: float):
            return TwistAnyAdapterActorCritic(
                num_prop=base_obs_dim,
                num_critic_obs=dual_total_obs_dim,
                num_priv_latent=0,
                num_hist=history_len,
                num_actions=dual_num_actions,
                base_actor_jit_path=dual_base_actor_jit_path,
                base_obs_dim=base_obs_dim,
                history_len=history_len,
                history_frame_dim=dual_history_frame_dim,
                hist_state_dim=hist_state_dim,
                latent_dim=16,
                adapter_hidden_dims=[32, 32],
                critic_hidden_dims=[64, 32],
                world_model_hidden_dims=[64, 32],
                action_delta_scale=0.25,
                use_dual_branch_adapter=True,
                use_tracking_error_adapter_input=True,
                compact_adapter_input=True,
                history_policy_grad_scale=history_policy_grad_scale,
                dynamics_action_delta_scale=0.05,
                tracking_action_delta_scale=0.05,
                use_conv_history=True,
                freeze_base=True,
            )

        dual_actor = make_dual_actor(history_policy_grad_scale=0.25)
        dual_actor.eval()
        with torch.no_grad():
            base_action = dual_actor.base_action(dual_obs)
            delta_dyn, delta_err = dual_actor.get_adapter_delta_components(dual_obs)
            delta_total = dual_actor.get_adapter_delta(dual_obs)
            dual_action = dual_actor.actor_mean(dual_obs)

        assert delta_dyn.shape == (4, dual_num_actions)
        assert delta_err.shape == (4, dual_num_actions)
        assert delta_total.shape == (4, dual_num_actions)
        assert dual_action.shape == (4, dual_num_actions)
        assert torch.equal(delta_total, delta_dyn + delta_err)
        assert delta_dyn.abs().max().item() < 1e-6
        assert delta_err.abs().max().item() < 1e-6
        assert (dual_action - base_action).abs().max().item() < 1e-6
        assert (
            dual_actor.adapter.dynamics_branch.net[0].in_features
            == dual_num_actions + hist_state_dim + 16
        )
        assert (
            dual_actor.adapter.tracking_branch.net[0].in_features
            == (dual_num_actions + 6) + dual_num_actions
        )

        dynamics_last = dual_actor.adapter.dynamics_branch.net[-1]
        tracking_last = dual_actor.adapter.tracking_branch.net[-1]
        initial_dyn = delta_dyn.clone()
        initial_err = delta_err.clone()
        with torch.no_grad():
            tracking_last.bias.fill_(0.2)
            dyn_after_tracking, err_after_tracking = dual_actor.get_adapter_delta_components(dual_obs)
        assert torch.equal(dyn_after_tracking, initial_dyn)
        assert not torch.equal(err_after_tracking, initial_err)

        with torch.no_grad():
            dynamics_last.bias.fill_(0.3)
            dyn_after_dynamics, err_after_dynamics = dual_actor.get_adapter_delta_components(dual_obs)
        assert not torch.equal(dyn_after_dynamics, dyn_after_tracking)
        assert torch.equal(err_after_dynamics, err_after_tracking)

        # A nonzero final weight is needed before z can receive a gradient from
        # a zero-initialized residual network.
        with torch.no_grad():
            dynamics_last.weight.fill_(0.01)
        dual_actor.zero_grad(set_to_none=True)
        loss = dual_actor.actor_mean(dual_obs).square().mean()
        loss.backward()
        assert any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in dual_actor.adapter.dynamics_branch.parameters()
        )
        assert any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in dual_actor.adapter.tracking_branch.parameters()
        )
        assert any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in dual_actor.history_encoder.parameters()
        )

        _, dual_reg_info = dual_actor.adapter_regularization_loss(dual_obs)
        for key in (
            "adapter_delta_l2",
            "adapter_reg_loss",
            "max_abs_delta_action",
            "mean_abs_delta_action",
            "dynamics_delta_l2",
            "tracking_delta_l2",
            "dynamics_mean_abs_delta",
            "tracking_mean_abs_delta",
        ):
            assert key in dual_reg_info

        detached_history_actor = make_dual_actor(history_policy_grad_scale=0.0)
        detached_dynamics_last = detached_history_actor.adapter.dynamics_branch.net[-1]
        detached_tracking_last = detached_history_actor.adapter.tracking_branch.net[-1]
        with torch.no_grad():
            detached_dynamics_last.weight.fill_(0.01)
            detached_dynamics_last.bias.fill_(0.3)
            detached_tracking_last.bias.fill_(0.2)
        detached_history_actor.zero_grad(set_to_none=True)
        detached_loss = detached_history_actor.actor_mean(dual_obs).square().mean()
        detached_loss.backward()
        assert all(p.grad is None for p in detached_history_actor.history_encoder.parameters())

    print("twist anyadapter shape test ok")


def test_twist_anyadapter_shapes() -> None:
    main()


if __name__ == "__main__":
    main()
