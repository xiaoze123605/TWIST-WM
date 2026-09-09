from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT / "rsl_rl"))

from rsl_rl.algorithms import PPODTERA
from rsl_rl.modules import TwistDTERAActorCritic


NUM_ACTIONS = 4
BASE_SINGLE_DIM = 30
BASE_HISTORY_LEN = 1
BASE_OBS_DIM = BASE_SINGLE_DIM * (BASE_HISTORY_LEN + 1)
HISTORY_LEN = 4
HIST_STATE_DIM = 51
DYN_FRAME_DIM = HIST_STATE_DIM + NUM_ACTIONS
ERROR_FRAME_DIM = 2 * NUM_ACTIONS + 7
TOTAL_OBS_DIM = (
    BASE_OBS_DIM
    + HISTORY_LEN * DYN_FRAME_DIM
    + HISTORY_LEN * ERROR_FRAME_DIM
)


class DummyBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(BASE_OBS_DIM, NUM_ACTIONS)

    def forward(self, observations):
        return self.linear(observations)


def make_actor(
    path,
    branch_mode="full",
    gate_mode=None,
    fix_action_std=False,
    **overrides,
):
    kwargs = dict(
        num_critic_obs=TOTAL_OBS_DIM,
        num_actions=NUM_ACTIONS,
        base_actor_jit_path=path,
        base_obs_dim=BASE_OBS_DIM,
        base_single_obs_dim=BASE_SINGLE_DIM,
        base_history_len=BASE_HISTORY_LEN,
        history_len=HISTORY_LEN,
        history_frame_dim=DYN_FRAME_DIM,
        hist_state_dim=HIST_STATE_DIM,
        wm_target_indices=list(range(HIST_STATE_DIM)),
        latent_dim=8,
        tracking_history_len=HISTORY_LEN,
        tracking_error_frame_dim=ERROR_FRAME_DIM,
        tracking_latent_dim=8,
        adapter_hidden_dims=[16, 16],
        critic_hidden_dims=[32, 16],
        world_model_hidden_dims=[16, 16],
        error_predictor_hidden_dims=[16, 16],
        risk_predictor_hidden_dims=[16, 16],
        default_ref_dof_pos=[0.0] * NUM_ACTIONS,
        adapter_branch_mode=branch_mode,
        tracking_error_scales=[1.0] * 5,
        residual_warmup_iterations=0,
        fix_action_std=fix_action_std,
        freeze_base=True,
    )
    if gate_mode is not None:
        kwargs["gate_mode"] = gate_mode
    kwargs.update(overrides)
    return TwistDTERAActorCritic(**kwargs)


def grad_norm(parameters):
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += parameter.grad.square().sum().item()
    return total ** 0.5


class DTERATest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.base_path = str(Path(cls.tmp.name) / "base.pt")
        traced = torch.jit.trace(DummyBase().eval(), torch.zeros(1, BASE_OBS_DIM))
        traced.save(cls.base_path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_history_and_latent_shapes(self):
        actor = make_actor(self.base_path)
        obs = torch.randn(3, TOTAL_OBS_DIM)
        base, dynamics, tracking = actor.split_dtera_obs(obs)
        self.assertEqual(base.shape, (3, BASE_OBS_DIM))
        self.assertEqual(dynamics.shape, (3, HISTORY_LEN, DYN_FRAME_DIM))
        self.assertEqual(tracking.shape, (3, HISTORY_LEN, ERROR_FRAME_DIM))
        self.assertEqual(actor.history_encoder(dynamics).shape, (3, 8))
        self.assertEqual(actor.tracking_error_history_encoder(tracking).shape, (3, 8))
        self.assertEqual(
            actor.predict_world_model_members(obs).shape,
            (3, 3, HIST_STATE_DIM),
        )

    def test_tracking_latent_is_scale_bounded_and_keeps_gradient(self):
        actor = make_actor(self.base_path)
        history = (
            1.0e4 * torch.randn(6, HISTORY_LEN, ERROR_FRAME_DIM)
        ).requires_grad_()
        latent = actor.tracking_error_history_encoder(history)
        expected_norm = latent.new_tensor(actor.tracking_latent_dim).sqrt()
        self.assertTrue(torch.allclose(
            latent.norm(dim=-1),
            expected_norm.expand(latent.shape[0]),
            rtol=2e-3,
            atol=2e-3,
        ))
        weights = torch.arange(
            1, actor.tracking_latent_dim + 1, dtype=latent.dtype
        )
        (latent * weights).mean().backward()
        self.assertGreater(history.grad.norm().item(), 0.0)

    def test_traced_tracking_normalization_has_no_fixed_cpu_device(self):
        actor = make_actor(self.base_path).eval()

        class TrackingNormalizer(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, error):
                return self.model.normalize_tracking_error(error)

        example = torch.randn(2, HISTORY_LEN, ERROR_FRAME_DIM)
        traced = torch.jit.trace(TrackingNormalizer(actor), example)
        self.assertTrue(torch.equal(traced(example), actor.normalize_tracking_error(example)))
        self.assertNotIn(
            'Device = prim::Constant[value="cpu"]()',
            str(traced.inlined_graph),
        )

    def test_residual_output_biases_are_zero_and_frozen(self):
        actor = make_actor(self.base_path)
        for branch in (
            actor.adapter.dynamics_branch,
            actor.adapter.tracking_branch,
        ):
            output_layer = [
                module
                for module in branch.net.modules()
                if isinstance(module, torch.nn.Linear)
            ][-1]
            self.assertFalse(output_layer.bias.requires_grad)
            self.assertEqual(output_layer.bias.abs().max().item(), 0.0)

        obs = torch.randn(8, TOTAL_OBS_DIM)
        loss = actor.action_delta_components(obs)[0].square().mean()
        loss.backward()
        for branch in (
            actor.adapter.dynamics_branch,
            actor.adapter.tracking_branch,
        ):
            output_layer = [
                module
                for module in branch.net.modules()
                if isinstance(module, torch.nn.Linear)
            ][-1]
            self.assertIsNone(output_layer.bias.grad)

    def test_zero_init_and_base_only(self):
        obs = torch.randn(5, TOTAL_OBS_DIM)
        actor = make_actor(self.base_path)
        diag = actor.action_diagnostics(obs)
        self.assertLess(diag["delta_dyn"].abs().max().item(), 1e-8)
        self.assertLess(diag["delta_err"].abs().max().item(), 1e-8)
        self.assertTrue(torch.equal(actor.actor_mean(obs), actor.base_action(obs)))

        actor.adapter_branch_mode = "base_only"
        self.assertTrue(torch.equal(actor.actor_mean(obs), actor.base_action(obs)))
        self.assertEqual(actor.action_delta(obs).abs().max().item(), 0.0)

    def test_branch_feature_isolation(self):
        torch.manual_seed(7)
        actor = make_actor(self.base_path, gate_mode="off")
        with torch.no_grad():
            actor.adapter.dynamics_branch.net[-1].weight.fill_(0.01)
            actor.adapter.tracking_branch.net[-1].weight.fill_(0.01)
        obs = torch.randn(3, TOTAL_OBS_DIM)
        dyn0, err0 = actor.get_adapter_delta_components(obs)

        tracking_changed = obs.clone()
        tracking_changed[:, actor.tracking_history_offset:] += 3.0
        dyn1, err1 = actor.get_adapter_delta_components(tracking_changed)
        self.assertTrue(torch.equal(dyn0, dyn1))
        self.assertFalse(torch.equal(err0, err1))

        dynamics_changed = obs.clone()
        dynamics_changed[:, BASE_OBS_DIM:actor.tracking_history_offset] += 3.0
        dyn2, err2 = actor.get_adapter_delta_components(dynamics_changed)
        self.assertFalse(torch.equal(dyn0, dyn2))
        self.assertTrue(torch.equal(err0, err2))

    def test_gate_mathematics(self):
        actor = make_actor(self.base_path)
        self.assertEqual(actor.gate_mode, "demand_only")
        zeros = torch.zeros(4, ERROR_FRAME_DIM)
        self.assertTrue(torch.equal(actor.tracking_demand(zeros), torch.zeros(4)))

        consistent = torch.zeros(3, 4, HIST_STATE_DIM)
        disagreement = consistent.clone()
        disagreement[0] = 4.0
        disagreement[1] = -4.0
        confidence0, _ = actor.confidence_from_members(consistent)
        confidence1, _ = actor.confidence_from_members(disagreement)
        self.assertTrue(torch.all(confidence1 < confidence0))

        risks = torch.tensor([-0.2, 0.0, 0.1, 0.4])
        safety = actor.safety_from_delta_risk(risks)
        self.assertTrue(torch.equal(safety[:2], torch.ones(2)))
        self.assertGreater(safety[2].item(), safety[3].item())

        diag = actor.action_diagnostics(torch.randn(9, TOTAL_OBS_DIM))
        self.assertTrue(torch.all((diag["gate"] >= 0) & (diag["gate"] <= 1)))

    def test_smoothstep_tracking_demand_has_closed_and_open_regions(self):
        actor = make_actor(
            self.base_path,
            tracking_demand_mode="smoothstep",
            tracking_demand_low=0.30,
            tracking_demand_high=0.80,
        )
        normalized_error = torch.tensor([0.0, 0.20, 0.30, 0.55, 0.80, 1.20])
        normalized_error = normalized_error[:, None].repeat(1, ERROR_FRAME_DIM)
        demand = actor.tracking_demand(normalized_error, normalized=True)
        self.assertTrue(torch.equal(demand[:3], torch.zeros(3)))
        self.assertAlmostEqual(demand[3].item(), 0.5, places=6)
        self.assertTrue(torch.equal(demand[4:], torch.ones(2)))
        self.assertTrue(torch.all(demand[1:] >= demand[:-1]))

    def test_independent_branch_gates_apply_per_branch(self):
        actor = make_actor(
            self.base_path,
            gate_mode="demand_only",
            use_independent_branch_gates=True,
            tracking_demand_mode="smoothstep",
            tracking_demand_low=0.30,
            tracking_demand_high=0.80,
            dynamics_gate_scale=0.50,
            tracking_gate_scale=1.0,
            dynamics_confidence_gate_strength=0.0,
        )
        with torch.no_grad():
            actor.adapter.dynamics_branch.net[-1].bias.fill_(0.4)
            actor.adapter.tracking_branch.net[-1].bias.fill_(0.2)

        obs = torch.zeros(3, TOTAL_OBS_DIM)
        tracking = obs[:, actor.tracking_history_offset:].reshape(
            3, HISTORY_LEN, ERROR_FRAME_DIM
        )
        tracking[:, -1].fill_(1.0)
        diag = actor.action_diagnostics(obs)
        self.assertTrue(torch.allclose(
            diag["dynamics_gate"], torch.full((3,), 0.5)
        ))
        self.assertTrue(torch.equal(diag["tracking_gate"], torch.ones(3)))
        expected = (
            0.5 * diag["selected_delta_dyn"]
            + diag["selected_delta_err"]
        )
        self.assertTrue(torch.allclose(diag["gated_delta"], expected))

        tracking[:, -1].fill_(0.1)
        low_error_diag = actor.action_diagnostics(obs)
        self.assertTrue(torch.equal(
            low_error_diag["tracking_gate"], torch.zeros(3)
        ))
        self.assertTrue(torch.equal(
            low_error_diag["dynamics_gate"], torch.zeros(3)
        ))
        self.assertTrue(torch.equal(
            low_error_diag["gated_delta"],
            torch.zeros_like(low_error_diag["gated_delta"]),
        ))

    def test_demand_only_actor_fast_path_skips_wm_and_risk(self):
        actor = make_actor(
            self.base_path,
            gate_mode="demand_only",
            use_independent_branch_gates=True,
            tracking_demand_mode="smoothstep",
        )
        obs = torch.randn(5, TOTAL_OBS_DIM)
        with mock.patch.object(
            actor, "predict_world_model_members",
            side_effect=AssertionError("WM must not run on demand-only action path"),
        ), mock.patch.object(
            actor, "risk_logits",
            side_effect=AssertionError("risk must not run on demand-only action path"),
        ):
            actor.actor_mean(obs)

    def test_fast_action_matches_full_diagnostics(self):
        actor = make_actor(
            self.base_path,
            gate_mode="demand_only",
            use_independent_branch_gates=True,
            tracking_demand_mode="smoothstep",
        )
        with torch.no_grad():
            actor.adapter.dynamics_branch.net[-1].bias.fill_(0.4)
            actor.adapter.tracking_branch.net[-1].bias.fill_(0.2)
        obs = torch.randn(6, TOTAL_OBS_DIM)
        base_action = actor.base_action(obs)
        expected = actor.action_diagnostics(
            obs, base_action=base_action
        )["applied_delta"]
        actual = actor.action_delta(obs, base_action=base_action)
        self.assertTrue(torch.allclose(actual, expected))

    def test_gate_off_matches_dual_formula(self):
        actor = make_actor(self.base_path, gate_mode="off")
        with torch.no_grad():
            actor.adapter.dynamics_branch.net[-1].bias.fill_(0.3)
            actor.adapter.tracking_branch.net[-1].bias.fill_(0.2)
        obs = torch.randn(4, TOTAL_OBS_DIM)
        diag = actor.action_diagnostics(obs)
        expected = (
            actor.dynamics_branch_gain * diag["delta_dyn"]
            + actor.tracking_branch_gain * diag["delta_err"]
        )
        self.assertTrue(torch.equal(diag["candidate_delta"], expected))
        self.assertTrue(torch.equal(diag["applied_delta"], expected))

    def test_residual_warmup_factor_and_application(self):
        actor = make_actor(self.base_path, residual_warmup_iterations=1000)
        with torch.no_grad():
            actor.adapter.dynamics_branch.net[-1].bias.fill_(0.4)
            actor.adapter.tracking_branch.net[-1].bias.fill_(0.2)
        obs = torch.randn(3, TOTAL_OBS_DIM)
        actor.set_training_iteration(0)
        self.assertEqual(actor.residual_warmup_factor(), 0.0)
        self.assertEqual(actor.action_delta(obs).abs().max().item(), 0.0)
        actor.set_training_iteration(250)
        diag = actor.action_diagnostics(obs)
        self.assertEqual(actor.residual_warmup_factor(), 0.25)
        self.assertTrue(torch.allclose(
            diag["applied_delta"], 0.25 * diag["gated_delta"]
        ))
        actor.set_training_iteration(1000)
        self.assertEqual(actor.residual_warmup_factor(), 1.0)
        actor.set_training_iteration(1200)
        self.assertEqual(actor.residual_warmup_factor(), 1.0)

    def test_saturation_fractions_and_regularization(self):
        actor = make_actor(self.base_path, gate_mode="off")
        obs = torch.randn(8, TOTAL_OBS_DIM)
        self.assertEqual(actor.residual_saturation_penalty(obs).item(), 0.0)
        with torch.no_grad():
            actor.adapter.dynamics_branch.net[-1].bias.fill_(5.0)
            actor.adapter.tracking_branch.net[-1].bias.fill_(5.0)
        penalty = actor.residual_saturation_penalty(obs)
        _, info = actor.adapter_regularization_loss(obs)
        self.assertGreater(penalty.item(), 0.0)
        self.assertEqual(info["dyn_saturation_fraction"], 1.0)
        self.assertEqual(info["err_saturation_fraction"], 1.0)
        self.assertEqual(info["candidate_saturation_fraction"], 1.0)

    def test_confidence_strength(self):
        actor = make_actor(self.base_path)
        confidence = torch.tensor([0.2, 0.8])
        actor.confidence_gate_strength = 0.0
        demand = torch.tensor([0.3, 0.7])
        effective = actor.effective_confidence(confidence)
        self.assertTrue(torch.equal(effective, torch.ones_like(confidence)))
        self.assertTrue(torch.equal(demand * effective, demand))
        actor.confidence_gate_strength = 1.0
        effective = actor.effective_confidence(confidence)
        self.assertTrue(torch.allclose(effective, confidence))
        self.assertTrue(torch.allclose(demand * effective, demand * confidence))

    def test_tracking_history_is_normalized_before_encoder(self):
        actor = make_actor(
            self.base_path,
            tracking_error_scales=[0.5, 2.0, 1.5, 0.25, 0.1],
        )
        obs = torch.zeros(2, TOTAL_OBS_DIM)
        raw = 2.0 * actor.tracking_error_scale
        start = actor.tracking_history_offset
        obs[:, start : start + HISTORY_LEN * ERROR_FRAME_DIM] = raw.repeat(
            HISTORY_LEN
        )
        captured = {}

        def capture(_, inputs):
            captured["history"] = inputs[0].detach().clone()

        handle = actor.tracking_error_history_encoder.register_forward_pre_hook(
            capture
        )
        branch_handle = actor.adapter.tracking_branch.register_forward_pre_hook(
            lambda _, inputs: captured.update(
                {"error_frame": inputs[0].detach().clone()}
            )
        )
        actor.action_delta_components(obs)
        handle.remove()
        branch_handle.remove()
        self.assertTrue(torch.allclose(
            captured["history"], torch.full_like(captured["history"], 2.0)
        ))
        self.assertTrue(torch.allclose(
            captured["error_frame"],
            torch.full_like(captured["error_frame"], 2.0),
        ))

    def test_wm_and_error_auxiliary_gradients(self):
        actor = make_actor(self.base_path)
        obs = torch.randn(6, TOTAL_OBS_DIM)
        next_obs = torch.randn_like(obs)
        actions = torch.randn(6, NUM_ACTIONS)

        actor.zero_grad(set_to_none=True)
        actor.predict_world_model_members(obs, actions).square().mean().backward()
        self.assertGreater(grad_norm(actor.history_encoder.parameters()), 1e-8)
        self.assertGreater(grad_norm(actor.world_model.parameters()), 1e-8)

        actor.zero_grad(set_to_none=True)
        loss, _ = actor.error_prediction_loss(
            obs, actions, next_obs, torch.ones(6, 1)
        )
        loss.backward()
        self.assertGreater(
            grad_norm(actor.tracking_error_history_encoder.parameters()), 1e-8
        )
        self.assertGreater(grad_norm(actor.error_trend_predictor.parameters()), 1e-8)

    def test_optimizer_ownership_and_weight_decay(self):
        actor = make_actor(self.base_path)
        algorithm = PPODTERA(
            object(), actor,
            world_model_loss_coef=0.1,
            error_prediction_loss_coef=0.05,
            weight_decay=1e-4,
        )
        expected = {
            "dynamics_encoder_in_ppo": False,
            "dynamics_encoder_in_wm": True,
            "tracking_encoder_in_ppo": True,
            "tracking_encoder_in_wm": False,
            "risk_predictor_only_in_risk": True,
        }
        self.assertEqual(algorithm.optimizer_ownership, expected)
        self.assertTrue(all(g["weight_decay"] == 0 for g in algorithm.wm_optimizer.param_groups))
        self.assertTrue(all(g["weight_decay"] == 0 for g in algorithm.risk_optimizer.param_groups))
        tracking_ids = {id(p) for p in actor.tracking_error_history_encoder.parameters()}
        tracking_groups = [
            group for group in algorithm.ppo_optimizer.param_groups
            if any(id(p) in tracking_ids for p in group["params"])
        ]
        self.assertTrue(tracking_groups)
        self.assertTrue(all(group["weight_decay"] == 0 for group in tracking_groups))
        self.assertTrue(algorithm.defer_world_model_update)

    def test_adapter_regularization_schedule(self):
        actor = make_actor(self.base_path)
        algorithm = PPODTERA(
            object(), actor,
            adapter_reg_coef=0.02,
            adapter_reg_initial_coef=0.10,
            adapter_reg_anneal_iterations=1000,
        )
        self.assertAlmostEqual(algorithm.effective_adapter_reg_coef(0), 0.10)
        self.assertAlmostEqual(algorithm.effective_adapter_reg_coef(500), 0.06)
        self.assertAlmostEqual(algorithm.effective_adapter_reg_coef(1000), 0.02)
        self.assertAlmostEqual(algorithm.effective_adapter_reg_coef(2000), 0.02)

    def test_uncertainty_error_calibration(self):
        uncertainty = torch.arange(10, dtype=torch.float32)
        actual_error = 2.0 * uncertainty + 1.0
        metrics = PPODTERA.uncertainty_error_calibration(
            uncertainty, actual_error
        )
        self.assertGreater(metrics["wm_uncertainty_error_corr"], 0.99)
        self.assertGreater(
            metrics["actual_wm_error_high_uncertainty"],
            metrics["actual_wm_error_low_uncertainty"],
        )

    def test_fixed_action_std_and_legacy_optimizer_compatibility(self):
        legacy_actor = make_actor(self.base_path, fix_action_std=False)
        legacy = PPODTERA(object(), legacy_actor)

        fixed_actor = make_actor(self.base_path, fix_action_std=True)
        fixed = PPODTERA(object(), fixed_actor, fixed_action_std=0.05)
        self.assertFalse(fixed_actor.std.requires_grad)
        self.assertTrue(torch.allclose(
            fixed_actor.std, torch.full_like(fixed_actor.std, 0.05)
        ))

        cli_overridden_actor = make_actor(
            self.base_path, fix_action_std=False
        )
        PPODTERA(
            object(), cli_overridden_actor, fixed_action_std=0.05
        )
        self.assertFalse(cli_overridden_actor.std.requires_grad)

        # An old checkpoint has one trainable std entry in this parameter
        # group.  The fixed-std optimizer deliberately retains that layout.
        fixed.ppo_optimizer.load_state_dict(legacy.ppo_optimizer.state_dict())
        fixed_actor.std.fill_(0.4)
        fixed.on_load_checkpoint()
        self.assertTrue(torch.allclose(
            fixed_actor.std, torch.full_like(fixed_actor.std, 0.05)
        ))

    def test_end_to_end_auxiliary_update(self):
        torch.manual_seed(11)
        actor = make_actor(self.base_path)
        algorithm = PPODTERA(
            object(), actor,
            num_learning_epochs=1,
            num_mini_batches=1,
            world_model_loss_coef=0.1,
            error_prediction_loss_coef=0.05,
            adapter_reg_coef=0.02,
            adapter_bias_reg_coef=0.1,
            stand_anchor_coef=0.0,
            synthetic_stand_anchor_coef=2.0,
        )
        algorithm.init_storage(
            2, 4, [TOTAL_OBS_DIM], [TOTAL_OBS_DIM], [NUM_ACTIONS]
        )
        storage = algorithm.storage
        storage.observations.normal_()
        storage.next_observations.normal_()
        storage.next_observations_available.fill_(1)
        storage.privileged_observations.copy_(storage.observations)
        storage.actions.normal_()
        storage.dones.zero_()
        storage.timeouts.zero_()
        storage.values.normal_()
        storage.returns.normal_()
        storage.advantages.normal_()
        storage.mu.normal_()
        storage.sigma.fill_(0.5)
        storage.actions_log_prob.zero_()

        result = algorithm.update()
        self.assertEqual(len(result), 6)
        for key in (
            "world_model_loss", "error_prediction_loss", "risk_loss",
            "risk_positive_ratio", "gate_mean", "candidate_delta_l2",
            "applied_delta_l2", "tracking_encoder_aux_grad_norm",
        ):
            self.assertIn(key, algorithm.anyadapter_metrics)
            self.assertTrue(torch.isfinite(torch.tensor(
                algorithm.anyadapter_metrics[key]
            )))
        self.assertEqual(
            algorithm.anyadapter_metrics["history_encoder_total_grad_norm"],
            algorithm.anyadapter_metrics["history_encoder_wm_grad_norm"],
        )

    def test_ppo_phase_freezes_dynamics_wm_and_risk(self):
        torch.manual_seed(13)
        actor = make_actor(self.base_path)
        algorithm = PPODTERA(
            object(), actor,
            num_learning_epochs=1,
            num_mini_batches=1,
            world_model_loss_coef=0.1,
            error_prediction_loss_coef=0.05,
            stand_anchor_coef=0.0,
            synthetic_stand_anchor_coef=0.0,
        )
        algorithm.init_storage(
            2, 4, [TOTAL_OBS_DIM], [TOTAL_OBS_DIM], [NUM_ACTIONS]
        )
        storage = algorithm.storage
        for tensor in (
            storage.observations, storage.next_observations, storage.actions,
            storage.values, storage.returns, storage.advantages, storage.mu,
        ):
            tensor.normal_()
        storage.next_observations_available.fill_(1)
        storage.privileged_observations.copy_(storage.observations)
        storage.dones.zero_()
        storage.timeouts.zero_()
        storage.sigma.fill_(0.5)
        storage.actions_log_prob.zero_()
        frozen = [
            parameter.detach().clone()
            for module in (
                actor.history_encoder, actor.world_model, actor.risk_predictor
            )
            for parameter in module.parameters()
        ]
        original_deferred = algorithm._deferred_auxiliary_update

        def checked_deferred():
            current = [
                parameter.detach()
                for module in (
                    actor.history_encoder, actor.world_model,
                    actor.risk_predictor,
                )
                for parameter in module.parameters()
            ]
            self.assertTrue(all(
                torch.equal(before, after)
                for before, after in zip(frozen, current)
            ))
            return original_deferred()

        algorithm._deferred_auxiliary_update = checked_deferred
        algorithm.update()

    def test_future_risk_targets_ignore_timeout_and_stop_at_reset(self):
        actor = make_actor(self.base_path)
        algorithm = PPODTERA(object(), actor, risk_horizon=5)
        algorithm.init_storage(
            1, 5, [TOTAL_OBS_DIM], [TOTAL_OBS_DIM], [NUM_ACTIONS]
        )
        algorithm.storage.dones.zero_()
        algorithm.storage.timeouts.zero_()
        algorithm.storage.dones[2, 0, 0] = 1
        algorithm.storage.timeouts[2, 0, 0] = 1
        algorithm.storage.dones[4, 0, 0] = 1
        targets, valid = algorithm._future_risk_targets()
        targets = targets[:, 0, 0]
        valid = valid[:, 0, 0]
        self.assertTrue(torch.equal(
            targets, torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0])
        ))
        self.assertTrue(torch.all(valid))

    def test_risk_tail_valid_mask(self):
        actor = make_actor(self.base_path)
        algorithm = PPODTERA(object(), actor, risk_horizon=3)
        algorithm.init_storage(
            1, 5, [TOTAL_OBS_DIM], [TOTAL_OBS_DIM], [NUM_ACTIONS]
        )
        algorithm.storage.dones.zero_()
        algorithm.storage.timeouts.zero_()
        targets, valid = algorithm._future_risk_targets()
        self.assertTrue(torch.equal(targets, torch.zeros_like(targets)))
        self.assertTrue(torch.equal(
            valid[:, 0, 0], torch.tensor([True, True, True, False, False])
        ))

    def test_risk_zero_positive_skip(self):
        actor = make_actor(self.base_path)
        algorithm = PPODTERA(object(), actor)
        algorithm.init_storage(
            1, 6, [TOTAL_OBS_DIM], [TOTAL_OBS_DIM], [NUM_ACTIONS]
        )
        targets = torch.zeros(6, 1, 1)
        valid = torch.ones(6, 1, 1, dtype=torch.bool)
        metrics = algorithm._train_risk_predictor(targets, valid)
        self.assertEqual(metrics["risk_update_skipped"], 1.0)
        self.assertEqual(metrics["risk_num_positive"], 0.0)
        self.assertEqual(metrics["risk_num_negative"], 6.0)

    def test_adaptive_risk_weight_and_probability_diagnostics(self):
        actor = make_actor(self.base_path)
        algorithm = PPODTERA(
            object(), actor, risk_adaptive_pos_weight=True,
            risk_pos_weight_max=20.0,
        )
        algorithm.init_storage(
            1, 10, [TOTAL_OBS_DIM], [TOTAL_OBS_DIM], [NUM_ACTIONS]
        )
        algorithm.storage.observations.normal_()
        algorithm.storage.actions.normal_()
        targets = torch.zeros(10, 1, 1)
        targets[:2] = 1.0
        valid = torch.ones_like(targets, dtype=torch.bool)
        metrics = algorithm._train_risk_predictor(targets, valid)
        self.assertEqual(metrics["risk_effective_pos_weight"], 4.0)
        self.assertEqual(metrics["risk_update_skipped"], 0.0)
        self.assertAlmostEqual(
            metrics["risk_probability_gap"],
            metrics["risk_prob_positive"] - metrics["risk_prob_negative"],
        )

    def test_synthetic_stand_and_checkpoint_buffers(self):
        actor = make_actor(self.base_path)
        stand = actor.build_synthetic_stand_observation(
            torch.randn(2, TOTAL_OBS_DIM)
        )
        first = stand[:, :BASE_SINGLE_DIM]
        second = stand[:, BASE_SINGLE_DIM:BASE_OBS_DIM]
        self.assertTrue(torch.equal(first, second))
        _, dynamics, tracking = actor.split_dtera_obs(stand)
        self.assertTrue(torch.equal(dynamics, torch.zeros_like(dynamics)))
        self.assertTrue(torch.equal(tracking, torch.zeros_like(tracking)))
        diag = actor.action_diagnostics(stand)
        self.assertLess(diag["candidate_delta"].abs().max().item(), 1e-8)
        self.assertLess(diag["applied_delta"].abs().max().item(), 1e-8)

        actor.wm_variance_ema.fill_(2.5)
        actor.wm_variance_ema_updates.fill_(17)
        actor.set_training_iteration(321)
        restored = make_actor(self.base_path)
        restored.load_state_dict(actor.state_dict())
        self.assertTrue(torch.equal(restored.wm_variance_ema, actor.wm_variance_ema))
        self.assertTrue(torch.equal(
            restored.wm_variance_ema_updates, actor.wm_variance_ema_updates
        ))
        self.assertEqual(restored.residual_training_iteration.item(), 321)
        restored_algorithm = PPODTERA(object(), restored)
        restored_algorithm.on_load_checkpoint({"iter": 0})
        self.assertEqual(restored.residual_training_iteration.item(), 321)

        legacy_state = dict(actor.state_dict())
        legacy_state.pop("residual_training_iteration")
        legacy_restored = make_actor(
            self.base_path, residual_warmup_iterations=1000
        )
        legacy_restored.load_state_dict(legacy_state)
        self.assertEqual(
            legacy_restored.residual_training_iteration.item(), 1000
        )
        legacy_algorithm = PPODTERA(object(), legacy_restored)
        legacy_algorithm.on_load_checkpoint({"iter": 500})
        self.assertEqual(
            legacy_restored.residual_training_iteration.item(), 500
        )


class HistoryMixinTest(unittest.TestCase):
    def test_reset_repeated_fill_for_both_histories(self):
        path = REPO_ROOT / "legged_gym/legged_gym/envs/g1/anyadapter_history_mixin.py"
        spec = importlib.util.spec_from_file_location("history_mixin_for_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class Env(module.AnyAdapterHistoryMixin):
            pass

        env = Env()
        env.cfg = type("Cfg", (), {})()
        env.cfg.env = type("EnvCfg", (), {
            "use_anyadapter": True,
            "anyadapter_history_len": 4,
            "anyadapter_state_indices": [0, 1, 2],
            "anyadapter_hist_state_dim": 3,
            "anyadapter_history_frame_dim": 3 + NUM_ACTIONS,
            "anyadapter_context_dim": 0,
            "anyadapter_fill_history_on_reset": True,
            "use_tracking_error_history": True,
            "tracking_error_history_len": 4,
            "tracking_error_frame_dim": ERROR_FRAME_DIM,
        })()
        env.device = "cpu"
        env.num_envs = 2
        env.num_actions = NUM_ACTIONS
        env.num_obs = 6
        env.obs_buf = torch.zeros(2, 6)
        env.episode_length_buf = torch.zeros(2)
        env.actions = torch.ones(2, NUM_ACTIONS)
        env.yaw = torch.zeros(2)
        env.roll = torch.zeros(2)
        env.pitch = torch.zeros(2)
        env.dof_pos = torch.zeros(2, NUM_ACTIONS)
        env.dof_vel = torch.zeros(2, NUM_ACTIONS)
        env.base_lin_vel = torch.zeros(2, 3)
        env.base_ang_vel = torch.zeros(2, 3)
        env.global_obs = False
        env.dt = 0.02
        env._ref_dof_pos = torch.ones(2, NUM_ACTIONS)
        env._ref_dof_vel = torch.ones(2, NUM_ACTIONS)
        env.root_states = torch.zeros(2, 13)
        env._ref_root_vel = torch.ones(2, 3)
        env._ref_root_ang_vel = torch.zeros(2, 3)
        env._ref_root_rot = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1)
        env._init_anyadapter_history()
        base = torch.arange(12, dtype=torch.float32).reshape(2, 6) + 1.0
        tracking_reference = torch.zeros(2, 8 + NUM_ACTIONS)
        tracking_reference[:, 8:] = 1.0
        result = env._append_anyadapter_history(
            base, env.actions, tracking_reference=tracking_reference
        )

        self.assertEqual(result.shape, (2, 6 + 4 * 7 + 4 * ERROR_FRAME_DIM))
        for history in (env.anyadapter_history, env.tracking_error_history):
            self.assertTrue(torch.equal(
                history, history[:, :1].expand_as(history)
            ))
        self.assertGreater(env.anyadapter_history[:, :, :3].abs().sum().item(), 0)
        self.assertEqual(
            env.anyadapter_history[:, :, 3:].abs().max().item(), 0.0
        )
        self.assertGreater(env.tracking_error_history.abs().sum().item(), 0)

    def test_degraded_reference_independence_and_local_velocity_frame(self):
        path = REPO_ROOT / "legged_gym/legged_gym/envs/g1/anyadapter_history_mixin.py"
        spec = importlib.util.spec_from_file_location("history_mixin_independence", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class Env(module.AnyAdapterHistoryMixin):
            pass

        env = Env()
        env.cfg = type("Cfg", (), {})()
        env.cfg.env = type("EnvCfg", (), {
            "use_anyadapter": True,
            "anyadapter_history_len": 2,
            "anyadapter_state_indices": [0],
            "use_tracking_error_history": True,
            "tracking_error_history_len": 2,
            "tracking_error_frame_dim": ERROR_FRAME_DIM,
        })()
        env.device = "cpu"
        env.num_envs = 1
        env.num_actions = NUM_ACTIONS
        env.num_obs = 2
        env.obs_buf = torch.zeros(1, 2)
        env.episode_length_buf = torch.zeros(1)
        env.dof_pos = torch.zeros(1, NUM_ACTIONS)
        env.dof_vel = torch.zeros(1, NUM_ACTIONS)
        env.base_lin_vel = torch.tensor([[1.0, 2.0, 3.0]])
        env.base_ang_vel = torch.tensor([[0.0, 0.0, 0.4]])
        env.root_states = torch.zeros(1, 13)
        env.root_states[:, 7:10] = 99.0
        env.roll = torch.zeros(1)
        env.pitch = torch.zeros(1)
        env.global_obs = False
        env.dt = 0.02
        env._init_anyadapter_history()
        reference = torch.zeros(1, 8 + NUM_ACTIONS)
        reference[:, 4:7] = env.base_lin_vel
        reference[:, 7] = env.base_ang_vel[:, 2]
        reference[:, 8:] = 0.5
        frame0 = env._build_tracking_error_frame(reference).clone()
        env._ref_dof_pos = torch.full((1, NUM_ACTIONS), 100.0)
        env._ref_dof_vel = torch.full((1, NUM_ACTIONS), 100.0)
        env._ref_root_vel = torch.full((1, 3), 100.0)
        env.tracking_ref_initialized.fill_(False)
        frame1 = env._build_tracking_error_frame(reference).clone()
        self.assertTrue(torch.equal(frame0, frame1))
        root_error_start = 2 * NUM_ACTIONS
        self.assertEqual(
            frame0[:, root_error_start : root_error_start + 4]
            .abs().max().item(),
            0.0,
        )
        env.episode_length_buf.fill_(2)
        next_reference = reference.clone()
        next_reference[:, 8:] += 0.2
        next_frame = env._build_tracking_error_frame(next_reference)
        dq_start = NUM_ACTIONS
        # raw dq_ref=(0.7-0.5)/0.02=10, alpha=0.5 -> filtered 5.
        self.assertTrue(torch.allclose(
            next_frame[:, dq_start : dq_start + NUM_ACTIONS],
            torch.full((1, NUM_ACTIONS), 5.0),
        ))


class EvaluationMetricTest(unittest.TestCase):
    def test_rmse_correctness(self):
        path = REPO_ROOT / "legged_gym/legged_gym/scripts/evaluation_metrics.py"
        spec = importlib.util.spec_from_file_location("evaluation_metrics", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        error = torch.tensor([3.0, 4.0])
        self.assertAlmostEqual(
            module.tensor_rmse(error).item(), (12.5) ** 0.5, places=6
        )


if __name__ == "__main__":
    unittest.main()
