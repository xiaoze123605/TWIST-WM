from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "rsl_rl"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rsl_rl.algorithms import PPOAnyAdapter
from test_twist_anyadapter_dual_diagnostics import (
    HIST_STATE_DIM,
    TOTAL_OBS_DIM,
    DummyBaseActor,
    grad_norm,
    make_actor,
)


class WorldModelEncoderGradientTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.base_actor_path = str(Path(cls.temp_dir.name) / "dummy_base_actor.pt")
        actor = DummyBaseActor().eval()
        traced = torch.jit.trace(actor, torch.zeros(1, 100))
        traced.save(cls.base_actor_path)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_wm_loss_backward_reaches_history_encoder(self):
        torch.manual_seed(101)
        actor = make_actor(self.base_actor_path, history_policy_grad_scale=0.10)
        algorithm = PPOAnyAdapter(object(), actor, joint_encoder_optimization=True)
        observations = torch.randn(8, TOTAL_OBS_DIM)
        target = torch.randn(8, HIST_STATE_DIM)
        encoder_before = [
            parameter.detach().clone()
            for parameter in actor.history_encoder.parameters()
        ]

        algorithm.wm_optimizer.zero_grad()
        wm_loss = torch.nn.functional.smooth_l1_loss(
            actor.predict_world_model(observations),
            target,
        )
        wm_loss.backward()

        self.assertGreater(grad_norm(actor.history_encoder.parameters()), 1e-8)
        self.assertGreater(grad_norm(actor.world_model.parameters()), 1e-8)
        algorithm.wm_optimizer.step()
        self.assertTrue(any(
            not torch.equal(before, after)
            for before, after in zip(encoder_before, actor.history_encoder.parameters())
        ))

    def test_joint_wm_optimizer_contains_history_encoder_by_identity(self):
        actor = make_actor(self.base_actor_path, history_policy_grad_scale=0.10)
        algorithm = PPOAnyAdapter(object(), actor, joint_encoder_optimization=True)
        optimizer_param_ids = {
            id(parameter)
            for group in algorithm.wm_optimizer.param_groups
            for parameter in group["params"]
        }
        ppo_optimizer_param_ids = {
            id(parameter)
            for group in algorithm.ppo_optimizer.param_groups
            for parameter in group["params"]
        }

        self.assertTrue(algorithm.history_encoder_in_wm_optimizer)
        self.assertFalse(algorithm.history_encoder_in_ppo_optimizer)
        self.assertTrue(all(
            id(parameter) in optimizer_param_ids
            for parameter in actor.history_encoder.parameters()
        ))
        self.assertTrue(all(
            id(parameter) in optimizer_param_ids
            for parameter in actor.world_model.parameters()
        ))
        self.assertFalse(any(
            id(parameter) in ppo_optimizer_param_ids
            for parameter in actor.history_encoder.parameters()
        ))
        self.assertFalse(any(
            id(parameter) in ppo_optimizer_param_ids
            for parameter in actor.world_model.parameters()
        ))
        self.assertTrue(all(
            group["weight_decay"] == 0.0
            for group in algorithm.wm_optimizer.param_groups
        ))

    def test_non_joint_wm_optimizer_excludes_history_encoder(self):
        actor = make_actor(self.base_actor_path, history_policy_grad_scale=0.0)
        algorithm = PPOAnyAdapter(object(), actor, joint_encoder_optimization=False)
        optimizer_param_ids = {
            id(parameter)
            for group in algorithm.wm_optimizer.param_groups
            for parameter in group["params"]
        }
        world_model_param_ids = {id(parameter) for parameter in actor.world_model.parameters()}

        self.assertFalse(algorithm.history_encoder_in_wm_optimizer)
        self.assertFalse(any(
            id(parameter) in optimizer_param_ids
            for parameter in actor.history_encoder.parameters()
        ))
        self.assertEqual(optimizer_param_ids, world_model_param_ids)

    def test_legacy_actor_still_forwards_with_non_joint_optimizer(self):
        actor = make_actor(
            self.base_actor_path,
            history_policy_grad_scale=0.0,
            dual=False,
        )
        algorithm = PPOAnyAdapter(object(), actor, joint_encoder_optimization=False)
        observations = torch.randn(4, TOTAL_OBS_DIM)

        actions = actor.actor_mean(observations)
        reg_loss, _ = actor.adapter_regularization_loss(observations)
        self.assertEqual(actions.shape, (4, 23))
        self.assertEqual(reg_loss.ndim, 0)
        self.assertFalse(algorithm.history_encoder_in_wm_optimizer)
        # Legacy non-joint mode keeps the encoder inside ppo_optimizer so the
        # WM-loss gradient (backwarded before the policy step) is still applied.
        self.assertTrue(algorithm.history_encoder_in_ppo_optimizer)


if __name__ == "__main__":
    unittest.main()
