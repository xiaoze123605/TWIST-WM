"""Inference-only branch ablation tests for the Dual AnyAdapter.

Covers the four guarantees required by the ablation study:
  1. forward shapes stay [B, 23] in full / dyn_only / err_only,
  2. branch isolation: dyn_only excludes the tracking branch and err_only
     excludes the dynamics branch (bit-exact, verified with a poisoned branch),
  3. the default "full" mode reproduces the current dual forward exactly,
  4. legacy single-residual modes (AnyAdapter / Safe / V3 / V4 / V5 / V6)
     are unaffected by the switch, and the world model / regularization loss
     keep reading the raw branch outputs regardless of the mask.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "rsl_rl"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rsl_rl.modules import TwistAnyAdapterActorCritic
from test_twist_anyadapter_dual_diagnostics import (
    HIST_STATE_DIM,
    NUM_ACTIONS,
    TOTAL_OBS_DIM,
    WM_TARGET_INDICES,
    DummyBaseActor,
    grad_norm,
)

BASE_OBS_DIM = 100
HISTORY_LEN = 20
HIST_STATE_DIM = 51
HISTORY_FRAME_DIM = HIST_STATE_DIM + NUM_ACTIONS

BRANCH_MODES = ("full", "dyn_only", "err_only")


def make_dual_actor(base_actor_path: str, branch_mode: str = "full"):
    return TwistAnyAdapterActorCritic(
        num_prop=BASE_OBS_DIM,
        num_critic_obs=TOTAL_OBS_DIM,
        num_actions=NUM_ACTIONS,
        base_actor_jit_path=base_actor_path,
        base_obs_dim=BASE_OBS_DIM,
        history_len=HISTORY_LEN,
        history_frame_dim=HISTORY_FRAME_DIM,
        hist_state_dim=HIST_STATE_DIM,
        wm_target_indices=WM_TARGET_INDICES,
        latent_dim=16,
        adapter_hidden_dims=[32, 32],
        critic_hidden_dims=[64, 32],
        world_model_hidden_dims=[64, 32],
        use_dual_branch_adapter=True,
        use_tracking_error_adapter_input=True,
        compact_adapter_input=True,
        history_policy_grad_scale=0.10,
        dynamics_action_delta_scale=0.05,
        tracking_action_delta_scale=0.05,
        adapter_branch_mode=branch_mode,
        use_conv_history=True,
        freeze_base=True,
    )


def make_legacy_actor(base_actor_path: str):
    return TwistAnyAdapterActorCritic(
        num_prop=BASE_OBS_DIM,
        num_critic_obs=TOTAL_OBS_DIM,
        num_actions=NUM_ACTIONS,
        base_actor_jit_path=base_actor_path,
        base_obs_dim=BASE_OBS_DIM,
        history_len=HISTORY_LEN,
        history_frame_dim=HISTORY_FRAME_DIM,
        hist_state_dim=HIST_STATE_DIM,
        wm_target_indices=WM_TARGET_INDICES,
        latent_dim=16,
        adapter_hidden_dims=[32, 32],
        critic_hidden_dims=[64, 32],
        world_model_hidden_dims=[64, 32],
        use_dual_branch_adapter=False,
        use_tracking_error_adapter_input=False,
        compact_adapter_input=False,
        history_policy_grad_scale=0.0,
        action_delta_scale=0.25,
        use_conv_history=True,
        freeze_base=True,
    )


def poison_branch_last_layer(actor, branch_name: str, factor: float = 100.0):
    """Scale the final layer of one branch so any leak becomes obvious."""
    layer = getattr(actor.adapter, branch_name).net[-1]
    with torch.no_grad():
        layer.weight.mul_(factor)
        layer.bias.mul_(factor)


def enable_branch_outputs(actor, dyn_bias: float = 0.30, err_bias: float = 0.20):
    """Move the zero-initialized branch heads to distinct non-zero outputs."""
    dynamics_last = actor.adapter.dynamics_branch.net[-1]
    tracking_last = actor.adapter.tracking_branch.net[-1]
    with torch.no_grad():
        dynamics_last.weight.fill_(0.01)
        dynamics_last.bias.fill_(dyn_bias)
        tracking_last.weight.fill_(0.01)
        tracking_last.bias.fill_(err_bias)


class DualBranchAblationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.base_actor_path = str(Path(cls.temp_dir.name) / "dummy_base_actor.pt")
        actor = DummyBaseActor().eval()
        traced = torch.jit.trace(actor, torch.zeros(1, BASE_OBS_DIM))
        traced.save(cls.base_actor_path)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_forward_shapes_are_batch_by_23_in_all_modes(self):
        for batch in (1, 4, 17):
            obs = torch.randn(batch, TOTAL_OBS_DIM)
            for mode in BRANCH_MODES:
                with self.subTest(batch=batch, mode=mode):
                    actor = make_dual_actor(self.base_actor_path, branch_mode=mode)
                    self.assertEqual(actor.actor_mean(obs).shape, (batch, NUM_ACTIONS))
                    self.assertEqual(actor.action_delta(obs).shape, (batch, NUM_ACTIONS))
                    self.assertEqual(actor.act_inference(obs).shape, (batch, NUM_ACTIONS))

    def test_dyn_only_returns_only_dynamics_branch(self):
        torch.manual_seed(1)
        actor = make_dual_actor(self.base_actor_path, branch_mode="dyn_only")
        enable_branch_outputs(actor)
        obs = torch.randn(4, TOTAL_OBS_DIM)

        delta = actor.action_delta(obs)
        delta_dyn, delta_err = actor.get_adapter_delta_components(obs)

        self.assertTrue(torch.equal(delta, actor.dynamics_branch_gain * delta_dyn))
        self.assertFalse(torch.allclose(
            delta, actor.tracking_branch_gain * delta_err, atol=1e-6
        ))

        # Poison the excluded tracking branch: output must not change at all.
        before = actor.actor_mean(obs).clone()
        poison_branch_last_layer(actor, "tracking_branch")
        after = actor.actor_mean(obs)
        self.assertTrue(torch.equal(before, after))

    def test_err_only_returns_only_tracking_branch(self):
        torch.manual_seed(2)
        actor = make_dual_actor(self.base_actor_path, branch_mode="err_only")
        enable_branch_outputs(actor)
        obs = torch.randn(4, TOTAL_OBS_DIM)

        delta = actor.action_delta(obs)
        delta_dyn, delta_err = actor.get_adapter_delta_components(obs)

        self.assertTrue(torch.equal(delta, actor.tracking_branch_gain * delta_err))
        self.assertFalse(torch.allclose(
            delta, actor.dynamics_branch_gain * delta_dyn, atol=1e-6
        ))

        before = actor.actor_mean(obs).clone()
        poison_branch_last_layer(actor, "dynamics_branch")
        after = actor.actor_mean(obs)
        self.assertTrue(torch.equal(before, after))

    def test_full_mode_matches_default_dual_forward_exactly(self):
        torch.manual_seed(3)
        obs = torch.randn(6, TOTAL_OBS_DIM)
        actor_default = make_dual_actor(self.base_actor_path)
        enable_branch_outputs(actor_default)
        actor_full = make_dual_actor(self.base_actor_path, branch_mode="full")
        # Copy weights so the only difference between the two actors is the
        # explicit vs default branch-mode argument.
        actor_full.load_state_dict(actor_default.state_dict())

        self.assertEqual(actor_default.adapter_branch_mode, "full")
        self.assertTrue(torch.equal(
            actor_default.actor_mean(obs), actor_full.actor_mean(obs)
        ))
        self.assertTrue(torch.equal(
            actor_default.action_delta(obs), actor_full.action_delta(obs)
        ))

        # "full" is exactly the weighted sum of the two raw branch outputs.
        delta = actor_full.action_delta(obs)
        delta_dyn, delta_err = actor_full.get_adapter_delta_components(obs)
        self.assertTrue(torch.equal(
            delta,
            actor_full.dynamics_branch_gain * delta_dyn
            + actor_full.tracking_branch_gain * delta_err,
        ))

    def test_branch_mode_is_swappable_after_construction(self):
        """The evaluate script sets adapter_branch_mode post-load; this must work."""
        torch.manual_seed(4)
        obs = torch.randn(4, TOTAL_OBS_DIM)
        actor = make_dual_actor(self.base_actor_path, branch_mode="full")
        enable_branch_outputs(actor)

        delta_full = actor.action_delta(obs).clone()
        actor.adapter_branch_mode = "dyn_only"
        delta_dyn_only = actor.action_delta(obs)
        actor.adapter_branch_mode = "err_only"
        delta_err_only = actor.action_delta(obs)
        actor.adapter_branch_mode = "full"
        delta_full_again = actor.action_delta(obs)

        delta_dyn, delta_err = actor.get_adapter_delta_components(obs)
        self.assertTrue(torch.equal(
            delta_dyn_only, actor.dynamics_branch_gain * delta_dyn
        ))
        self.assertTrue(torch.equal(
            delta_err_only, actor.tracking_branch_gain * delta_err
        ))
        self.assertTrue(torch.equal(delta_full, delta_full_again))

    def test_base_only_is_exact_frozen_actor(self):
        actor = make_dual_actor(self.base_actor_path, branch_mode="base_only")
        enable_branch_outputs(actor)
        obs = torch.randn(7, TOTAL_OBS_DIM)
        self.assertTrue(torch.equal(actor.actor_mean(obs), actor.base_action(obs)))
        self.assertTrue(torch.equal(
            actor.action_delta(obs), torch.zeros(7, NUM_ACTIONS)
        ))

    def test_world_model_and_reg_loss_ignore_branch_mask(self):
        torch.manual_seed(5)
        actor = make_dual_actor(self.base_actor_path, branch_mode="dyn_only")
        enable_branch_outputs(actor)
        obs = torch.randn(4, TOTAL_OBS_DIM)

        # World model gradient chain is untouched by the mask.
        actor.zero_grad(set_to_none=True)
        target = torch.randn(4, HIST_STATE_DIM)
        torch.nn.functional.smooth_l1_loss(
            actor.predict_world_model(obs), target
        ).backward()
        self.assertGreater(grad_norm(actor.history_encoder.parameters()), 0.0)
        self.assertGreater(grad_norm(actor.world_model.parameters()), 0.0)

        # The training regularization loss keeps reading both raw branches
        # even when the inference mask excludes one of them.
        reg_loss, info = actor.adapter_regularization_loss(obs)
        self.assertEqual(reg_loss.ndim, 0)
        self.assertGreater(info["dynamics_delta_l2"], 0.0)
        self.assertGreater(info["tracking_delta_l2"], 0.0)

    def test_legacy_single_residual_modes_ignore_branch_mask(self):
        torch.manual_seed(6)
        actor = make_legacy_actor(self.base_actor_path)
        obs = torch.randn(4, TOTAL_OBS_DIM)

        delta_ref = actor.action_delta(obs).clone()
        for mode in BRANCH_MODES:
            with self.subTest(mode=mode):
                actor.adapter_branch_mode = mode
                delta = actor.action_delta(obs)
                self.assertEqual(delta.shape, (4, NUM_ACTIONS))
                self.assertTrue(torch.equal(delta, delta_ref))
                actor.zero_grad(set_to_none=True)
                actor.actor_mean(obs).sum().backward()

    def test_invalid_branch_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            make_dual_actor(self.base_actor_path, branch_mode="dyn_and_err")


if __name__ == "__main__":
    unittest.main()
