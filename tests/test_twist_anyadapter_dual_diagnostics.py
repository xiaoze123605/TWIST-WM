from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "rsl_rl"))

from rsl_rl.algorithms import PPOAnyAdapter
from rsl_rl.modules import TwistAnyAdapterActorCritic


BASE_OBS_DIM = 100
HISTORY_LEN = 20
HIST_STATE_DIM = 51
NUM_ACTIONS = 23
HISTORY_FRAME_DIM = HIST_STATE_DIM + NUM_ACTIONS
TOTAL_OBS_DIM = BASE_OBS_DIM + HISTORY_LEN * HISTORY_FRAME_DIM
WM_TARGET_INDICES = list(range(31, 82))


class DummyBaseActor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(BASE_OBS_DIM, NUM_ACTIONS)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.linear(obs)


def make_actor(base_actor_path: str, history_policy_grad_scale: float, dual: bool = True):
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
        use_dual_branch_adapter=dual,
        use_tracking_error_adapter_input=dual,
        compact_adapter_input=dual,
        history_policy_grad_scale=history_policy_grad_scale,
        dynamics_action_delta_scale=0.05,
        tracking_action_delta_scale=0.05,
        use_conv_history=True,
        freeze_base=True,
    )


def grad_norm(parameters) -> float:
    norm_sq = sum(
        float(torch.sum(parameter.grad.detach() ** 2))
        for parameter in parameters
        if parameter.grad is not None
    )
    return norm_sq ** 0.5


def enable_policy_gradient_paths(actor: TwistAnyAdapterActorCritic) -> None:
    """Move past exact zero init so a policy loss can reach both branches and z."""
    with torch.no_grad():
        dynamics_last = actor.adapter.dynamics_branch.net[-1]
        tracking_last = actor.adapter.tracking_branch.net[-1]
        dynamics_last.weight.fill_(0.01)
        dynamics_last.bias.fill_(0.30)
        tracking_last.weight.fill_(0.01)
        tracking_last.bias.fill_(0.20)


def load_config_with_minimal_base_classes():
    """Execute the real config module without importing the Isaac Gym env package."""
    module_names = (
        "legged_gym",
        "legged_gym.envs",
        "legged_gym.envs.g1",
        "legged_gym.envs.g1.g1_mimic_distill_config",
    )
    previous_modules = {name: sys.modules.get(name) for name in module_names}

    class BaseEnvCfg:
        class env:
            pass

        class motion:
            pass

        class domain_rand:
            pass

        class rewards:
            class scales:
                pass

    class BaseTrainCfg:
        class runner:
            pass

        class policy:
            pass

        class algorithm:
            pass

    try:
        for name in module_names[:-1]:
            package = types.ModuleType(name)
            package.__path__ = []
            sys.modules[name] = package
        base_config = types.ModuleType(module_names[-1])
        base_config.G1MimicStuRLCfg = BaseEnvCfg
        base_config.G1MimicPrivCfgPPO = BaseTrainCfg
        sys.modules[module_names[-1]] = base_config

        config_path = (
            REPO_ROOT
            / "legged_gym/legged_gym/envs/g1/g1_mimic_distill_anyadapter_config.py"
        )
        spec = importlib.util.spec_from_file_location(
            "_dual_anyadapter_config_under_test",
            config_path,
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


class DualAnyAdapterDiagnosticsTest(unittest.TestCase):
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

    def test_dual_config_resolves_v4_gradient_settings(self):
        config_module = load_config_with_minimal_base_classes()
        config = config_module.G1MimicStuAnyAdapterDualCfgPPO()

        self.assertIs(config.policy.use_dual_branch_adapter, True)
        self.assertIs(config.policy.use_tracking_error_adapter_input, True)
        self.assertIs(config.policy.compact_adapter_input, True)
        self.assertAlmostEqual(config.policy.history_policy_grad_scale, 0.10)
        self.assertIs(config.algorithm.joint_encoder_optimization, True)

    def test_dtera_uses_fixed_ppo_learning_rate_schedule(self):
        config_module = load_config_with_minimal_base_classes()
        config = config_module.G1MimicStuAnyAdapterDTERACfgPPO()

        self.assertEqual(config.algorithm.schedule, "fixed")
        self.assertAlmostEqual(config.algorithm.learning_rate, 2e-4)
        self.assertIs(config.policy.freeze_residual_output_bias, True)

    def test_selective_dtera_config_preserves_scales_and_separates_gates(self):
        config_module = load_config_with_minimal_base_classes()
        legacy = config_module.G1MimicStuAnyAdapterDTERACfgPPO()
        selective = config_module.G1MimicStuAnyAdapterDTERASelectiveCfgPPO()

        self.assertFalse(hasattr(legacy.policy, "use_independent_branch_gates"))
        self.assertTrue(selective.policy.use_independent_branch_gates)
        self.assertEqual(selective.policy.tracking_demand_mode, "smoothstep")
        self.assertEqual(selective.policy.dynamics_demand_low, 0.10)
        self.assertEqual(selective.policy.dynamics_demand_high, 0.50)
        self.assertEqual(selective.policy.dynamics_gate_scale, 0.5)
        self.assertEqual(selective.policy.tracking_gate_scale, 1.0)
        self.assertEqual(
            selective.policy.dynamics_action_delta_scale,
            legacy.policy.dynamics_action_delta_scale,
        )
        self.assertEqual(
            selective.policy.tracking_action_delta_scale,
            legacy.policy.tracking_action_delta_scale,
        )
        self.assertEqual(
            selective.algorithm.learning_rate, legacy.algorithm.learning_rate
        )

    def test_history_encoder_is_owned_by_joint_wm_optimizer(self):
        actor = make_actor(self.base_actor_path, history_policy_grad_scale=0.10)
        algorithm = PPOAnyAdapter(object(), actor, joint_encoder_optimization=True)
        ppo_optimizer_param_ids = {
            id(parameter)
            for group in algorithm.ppo_optimizer.param_groups
            for parameter in group["params"]
        }
        wm_optimizer_param_ids = {
            id(parameter)
            for group in algorithm.wm_optimizer.param_groups
            for parameter in group["params"]
        }

        self.assertFalse(any(
            id(parameter) in ppo_optimizer_param_ids
            for parameter in actor.history_encoder.parameters()
        ))
        self.assertTrue(all(
            id(parameter) in wm_optimizer_param_ids
            for parameter in actor.history_encoder.parameters()
        ))

    def test_policy_history_gradient_scale(self):
        for scale, expect_gradient in ((0.10, True), (0.0, False)):
            with self.subTest(scale=scale):
                torch.manual_seed(10)
                actor = make_actor(self.base_actor_path, history_policy_grad_scale=scale)
                enable_policy_gradient_paths(actor)
                obs = torch.randn(4, TOTAL_OBS_DIM)

                actor.zero_grad(set_to_none=True)
                actor.actor_mean(obs).square().mean().backward()
                history_grad = grad_norm(actor.history_encoder.parameters())

                if expect_gradient:
                    self.assertGreater(history_grad, 0.0)
                else:
                    self.assertLess(history_grad, 1e-12)

    def test_world_model_and_dual_branch_gradients(self):
        torch.manual_seed(20)
        actor = make_actor(self.base_actor_path, history_policy_grad_scale=0.10)
        obs = torch.randn(4, TOTAL_OBS_DIM)

        actor.zero_grad(set_to_none=True)
        target = torch.randn(4, HIST_STATE_DIM)
        torch.nn.functional.smooth_l1_loss(actor.predict_world_model(obs), target).backward()
        self.assertGreater(grad_norm(actor.history_encoder.parameters()), 0.0)
        self.assertGreater(grad_norm(actor.world_model.parameters()), 0.0)

        enable_policy_gradient_paths(actor)
        actor.zero_grad(set_to_none=True)
        actor.actor_mean(obs).square().mean().backward()
        self.assertGreater(grad_norm(actor.adapter.dynamics_branch.parameters()), 0.0)
        self.assertGreater(grad_norm(actor.adapter.tracking_branch.parameters()), 0.0)

    def test_world_model_component_diagnostics_keep_weighted_total(self):
        torch.manual_seed(30)
        actor = make_actor(self.base_actor_path, history_policy_grad_scale=0.10)
        algorithm = PPOAnyAdapter(object(), actor, joint_encoder_optimization=True)
        obs = torch.randn(4, TOTAL_OBS_DIM)
        next_obs = torch.randn(4, TOTAL_OBS_DIM)
        actions = torch.randn(4, NUM_ACTIONS)

        total_loss, component_info = algorithm._world_model_loss_from_batch(
            obs,
            actions,
            next_obs,
            torch.zeros(4),
            torch.ones(4),
        )
        expected_keys = {
            "wm_ang_vel_loss",
            "wm_orientation_loss",
            "wm_dof_pos_loss",
            "wm_dof_vel_loss",
        }
        self.assertEqual(set(component_info), expected_keys)
        self.assertTrue(all(component_info[key] >= 0.0 for key in expected_keys))
        self.assertTrue(any(component_info[key] > 0.0 for key in expected_keys))
        expected_total = (
            component_info["wm_ang_vel_loss"]
            + 2.0 * component_info["wm_orientation_loss"]
            + component_info["wm_dof_pos_loss"]
            + component_info["wm_dof_vel_loss"]
        )
        self.assertAlmostEqual(total_loss.item(), expected_total, places=6)

    def test_joint_update_reports_separate_gradient_sources(self):
        torch.manual_seed(40)
        actor = make_actor(self.base_actor_path, history_policy_grad_scale=0.10)
        enable_policy_gradient_paths(actor)
        algorithm = PPOAnyAdapter(
            object(),
            actor,
            joint_encoder_optimization=True,
            num_learning_epochs=1,
            num_mini_batches=1,
            desired_kl=None,
            stand_anchor_coef=0.0,
            synthetic_stand_anchor_coef=0.0,
        )
        algorithm.init_storage(
            num_envs=4,
            num_transitions_per_env=1,
            actor_obs_shape=[TOTAL_OBS_DIM],
            critic_obs_shape=[TOTAL_OBS_DIM],
            action_shape=[NUM_ACTIONS],
        )
        storage = algorithm.storage
        obs = torch.randn(4, TOTAL_OBS_DIM)
        next_obs = torch.randn(4, TOTAL_OBS_DIM)
        with torch.no_grad():
            actor.update_distribution(obs)
            actions = actor.action_mean + 0.10
            old_log_prob = actor.get_actions_log_prob(actions)
            old_mean = actor.action_mean.clone()
            old_std = actor.action_std.clone()

        storage.observations[0].copy_(obs)
        storage.privileged_observations[0].copy_(obs)
        storage.next_observations[0].copy_(next_obs)
        storage.next_observations_available[0].fill_(1)
        storage.actions[0].copy_(actions)
        storage.actions_log_prob[0, :, 0].copy_(old_log_prob)
        storage.mu[0].copy_(old_mean)
        storage.sigma[0].copy_(old_std)
        storage.advantages[0, :, 0].copy_(torch.tensor([1.0, -0.5, 0.75, -1.25]))
        storage.returns[0, :, 0].copy_(torch.tensor([0.2, -0.1, 0.3, -0.2]))

        algorithm.update()
        metrics = algorithm.anyadapter_metrics
        self.assertGreater(metrics["history_encoder_ppo_grad_norm"], 0.0)
        self.assertGreater(metrics["history_encoder_wm_grad_norm"], 0.0)
        self.assertGreater(metrics["history_encoder_total_grad_norm"], 0.0)
        self.assertGreater(metrics["dynamics_branch_grad_norm"], 0.0)
        self.assertGreater(metrics["tracking_branch_grad_norm"], 0.0)
        self.assertGreater(metrics["dynamics_max_abs_delta"], 0.0)
        self.assertGreater(metrics["tracking_max_abs_delta"], 0.0)
        self.assertGreaterEqual(metrics["branch_balance_ratio"], 0.0)
        self.assertLessEqual(metrics["branch_balance_ratio"], 1.0)
        self.assertGreaterEqual(metrics["branch_cosine_similarity"], -1.0)
        self.assertLessEqual(metrics["branch_cosine_similarity"], 1.0)
        self.assertTrue(any(
            metrics[key] > 0.0
            for key in (
                "wm_ang_vel_loss",
                "wm_orientation_loss",
                "wm_dof_pos_loss",
                "wm_dof_vel_loss",
            )
        ))

    def test_non_joint_update_applies_wm_gradient_to_encoder(self):
        """Legacy non-joint mode: WM loss must still reach and update the encoder."""
        torch.manual_seed(45)
        actor = make_actor(
            self.base_actor_path,
            history_policy_grad_scale=0.0,
            dual=False,
        )
        algorithm = PPOAnyAdapter(
            object(),
            actor,
            joint_encoder_optimization=False,
            num_learning_epochs=1,
            num_mini_batches=1,
            desired_kl=None,
            stand_anchor_coef=0.0,
            synthetic_stand_anchor_coef=0.0,
        )
        algorithm.init_storage(
            num_envs=4,
            num_transitions_per_env=1,
            actor_obs_shape=[TOTAL_OBS_DIM],
            critic_obs_shape=[TOTAL_OBS_DIM],
            action_shape=[NUM_ACTIONS],
        )
        storage = algorithm.storage
        obs = torch.randn(4, TOTAL_OBS_DIM)
        next_obs = torch.randn(4, TOTAL_OBS_DIM)
        with torch.no_grad():
            actor.update_distribution(obs)
            actions = actor.action_mean + 0.10
            old_log_prob = actor.get_actions_log_prob(actions)
            old_mean = actor.action_mean.clone()
            old_std = actor.action_std.clone()

        storage.observations[0].copy_(obs)
        storage.privileged_observations[0].copy_(obs)
        storage.next_observations[0].copy_(next_obs)
        storage.next_observations_available[0].fill_(1)
        storage.actions[0].copy_(actions)
        storage.actions_log_prob[0, :, 0].copy_(old_log_prob)
        storage.mu[0].copy_(old_mean)
        storage.sigma[0].copy_(old_std)
        storage.advantages[0, :, 0].copy_(torch.tensor([1.0, -0.5, 0.75, -1.25]))
        storage.returns[0, :, 0].copy_(torch.tensor([0.2, -0.1, 0.3, -0.2]))

        encoder_before = [
            parameter.detach().clone()
            for parameter in actor.history_encoder.parameters()
        ]
        world_model_before = [
            parameter.detach().clone()
            for parameter in actor.world_model.parameters()
        ]

        algorithm.update()
        metrics = algorithm.anyadapter_metrics

        self.assertGreater(metrics["history_encoder_wm_grad_norm"], 0.0)
        # history_policy_grad_scale == 0.0, so the policy contributes no
        # encoder gradient of its own in this legacy configuration.
        self.assertEqual(metrics["history_encoder_ppo_grad_norm"], 0.0)
        self.assertTrue(algorithm.history_encoder_in_ppo_optimizer)
        self.assertFalse(algorithm.history_encoder_in_wm_optimizer)
        self.assertTrue(any(
            not torch.equal(before, after)
            for before, after in zip(encoder_before, actor.history_encoder.parameters())
        ))
        self.assertTrue(any(
            not torch.equal(before, after)
            for before, after in zip(world_model_before, actor.world_model.parameters())
        ))

    def test_legacy_mode_forward_backward_and_zero_dual_diagnostics(self):
        actor = make_actor(
            self.base_actor_path,
            history_policy_grad_scale=0.0,
            dual=False,
        )
        obs = torch.randn(4, TOTAL_OBS_DIM)

        actor.zero_grad(set_to_none=True)
        actor.actor_mean(obs).sum().backward()
        reg_loss, info = actor.adapter_regularization_loss(obs)

        self.assertEqual(reg_loss.ndim, 0)
        self.assertGreaterEqual(info["adapter_delta_l2"], 0.0)
        for key in (
            "dynamics_delta_l2",
            "tracking_delta_l2",
            "dynamics_mean_abs_delta",
            "tracking_mean_abs_delta",
            "dynamics_max_abs_delta",
            "tracking_max_abs_delta",
            "branch_balance_ratio",
            "branch_cosine_similarity",
        ):
            self.assertEqual(info[key], 0.0)


if __name__ == "__main__":
    unittest.main()
