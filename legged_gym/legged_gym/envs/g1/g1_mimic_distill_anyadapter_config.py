"""
G1 TWIST student + AnyAdapter configuration.

This config keeps the original TWIST student observation as the base actor
input and appends a separate AnyAdapter history after it:

    base_obs_dim = 1155
    anyadapter history = 20 * (51 selected state dims + 23 actions) = 1480
    final actor obs dim = 2635

The selected state deliberately excludes reference motion, TWIST's built-in
history block, and the action_history_buf slice already present in the base obs.
"""

from legged_gym.envs.g1.g1_mimic_distill_config import G1MimicStuRLCfg, G1MimicPrivCfgPPO


ANYADAPTER_STATE_INDICES = (
    list(range(31, 36)) +  # base_ang_vel + roll/pitch
    list(range(36, 59)) +  # dof_pos - default
    list(range(59, 82))    # dof_vel
)

DEFAULT_REF_DOF_POS = [
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.4, 0.0, 1.2,
    0.0, -0.4, 0.0, 1.2,
]


class G1MimicStuAnyAdapterCfg(G1MimicStuRLCfg):
    class env(G1MimicStuRLCfg.env):
        use_anyadapter = True
        normalize_obs = False

        # Original TWIST student policy input before appending AnyAdapter history.
        base_obs_dim = 1155

        anyadapter_history_len = 20
        anyadapter_state_indices = ANYADAPTER_STATE_INDICES
        anyadapter_hist_state_dim = 51
        anyadapter_history_frame_dim = 74

        # For reference/documentation only.  Do not assign this to
        # num_observations here; the mixin updates env.num_obs after BaseTask
        # has allocated the original TWIST observation buffer.
        anyadapter_added_obs_dim = 1480
        anyadapter_final_policy_obs_dim = 2635


class G1MimicStuAnyAdapterCfgPPO(G1MimicPrivCfgPPO):
    class runner(G1MimicPrivCfgPPO.runner):
        policy_class_name = "TwistAnyAdapterActorCritic"
        algorithm_class_name = "PPOAnyAdapter"
        runner_class_name = "OnPolicyRunnerMimic"
        experiment_name = "g1_twist_anyadapter"
        run_name = ""

    class policy(G1MimicPrivCfgPPO.policy):
        # Must point to the frozen exported TWIST student JIT actor.
        base_actor_jit_path = "/home/hank/TWIST（anyadapter）/legged_gym/logs/g1_stu_rl/0529_twist_rlbcstu/traced/0529_twist_rlbcstu-36500-jit.pt"

        base_obs_dim = 1155
        history_len = 20
        hist_state_dim = 51
        history_frame_dim = 74
        wm_target_indices = ANYADAPTER_STATE_INDICES

        latent_dim = 32
        adapter_hidden_dims = [128, 128]
        world_model_hidden_dims = [256, 256]
        critic_hidden_dims = [512, 256, 128]
        action_delta_scale = 0.25
        init_noise_std = 0.2
        freeze_base = True
        activation = "elu"
        use_conv_history = True

    class algorithm(G1MimicPrivCfgPPO.algorithm):
        world_model_loss_coef = 0.1
        adapter_reg_coef = 1e-3
        world_model_loss_type = "smooth_l1"


# ======================== V2 Conservative ========================

class G1MimicStuAnyAdapterV2Cfg(G1MimicStuRLCfg):
    """V2 conservative: smaller adapter delta, stronger regularization, lower wm coef.

    CRITICAL: normalize_obs is set to False because the JIT base actor contains its
    own internal Normalizer fitted during the original TWIST student training.  If
    the runner also normalizes observations the base actor receives double-normalized
    inputs, producing severely wrong actions.
    """

    class env(G1MimicStuRLCfg.env):
        use_anyadapter = True
        normalize_obs = False

        # Original TWIST student policy input before appending AnyAdapter history.
        base_obs_dim = 1155

        anyadapter_history_len = 20
        anyadapter_state_indices = ANYADAPTER_STATE_INDICES
        anyadapter_hist_state_dim = 51
        anyadapter_history_frame_dim = 74

        # For reference/documentation only.
        anyadapter_added_obs_dim = 1480
        anyadapter_final_policy_obs_dim = 2635

    class motion(G1MimicStuRLCfg.motion):
        # Slow curriculum: adapter sees easy motions for much longer before
        # progressing to max difficulty.  This prevents overfitting to extreme
        # dynamics and keeps corrections small for standing / simple gaits.
        motion_curriculum_gamma = 0.002


class G1MimicStuAnyAdapterV2CfgPPO(G1MimicPrivCfgPPO):
    class runner(G1MimicPrivCfgPPO.runner):
        policy_class_name = "TwistAnyAdapterActorCritic"
        algorithm_class_name = "PPOAnyAdapter"
        runner_class_name = "OnPolicyRunnerMimic"
        experiment_name = "g1_twist_anyadapter_v2"
        run_name = ""

    class policy(G1MimicPrivCfgPPO.policy):
        # Must point to the frozen exported TWIST student JIT actor.
        base_actor_jit_path = "/home/hank/TWIST（anyadapter）/legged_gym/logs/g1_stu_rl/0529_twist_rlbcstu/traced/0529_twist_rlbcstu-36500-jit.pt"

        base_obs_dim = 1155
        history_len = 20
        hist_state_dim = 51
        history_frame_dim = 74
        wm_target_indices = ANYADAPTER_STATE_INDICES

        latent_dim = 32
        adapter_hidden_dims = [128, 128]
        world_model_hidden_dims = [256, 256]
        critic_hidden_dims = [512, 256, 128]
        action_delta_scale = 0.02
        init_noise_std = 0.05
        freeze_base = True
        activation = "elu"
        use_conv_history = True

    class algorithm(G1MimicPrivCfgPPO.algorithm):
        world_model_loss_coef = 0.1
        adapter_reg_coef = 2e-1
        world_model_loss_type = "smooth_l1"
        # Weight decay on adapter + history_encoder params to prevent large
        # weight norms that cause tanh saturation / action jitter.
        weight_decay = 1e-4


# ======================== Safe Stand-Preserving ========================

class G1MimicStuAnyAdapterSafeCfg(G1MimicStuAnyAdapterV2Cfg):
    class env(G1MimicStuAnyAdapterV2Cfg.env):
        use_anyadapter = True
        normalize_obs = False


class G1MimicStuAnyAdapterSafeCfgPPO(G1MimicPrivCfgPPO):
    class runner(G1MimicPrivCfgPPO.runner):
        policy_class_name = "TwistAnyAdapterActorCritic"
        algorithm_class_name = "PPOAnyAdapter"
        runner_class_name = "OnPolicyRunnerMimic"
        experiment_name = "g1_twist_anyadapter_safe"
        run_name = ""

    class policy(G1MimicPrivCfgPPO.policy):
        # Must point to the frozen exported TWIST student JIT actor.
        base_actor_jit_path = "/home/hank/TWIST（anyadapter）/legged_gym/logs/g1_stu_rl/0529_twist_rlbcstu/traced/0529_twist_rlbcstu-36500-jit.pt"

        base_obs_dim = 1155
        history_len = 20
        hist_state_dim = 51
        history_frame_dim = 74
        wm_target_indices = ANYADAPTER_STATE_INDICES
        default_ref_dof_pos = DEFAULT_REF_DOF_POS

        latent_dim = 32
        adapter_hidden_dims = [128, 128]
        world_model_hidden_dims = [256, 256]
        critic_hidden_dims = [512, 256, 128]
        action_delta_scale = 0.02
        adapter_gain = 1.0
        init_noise_std = 0.05
        freeze_base = True
        activation = "elu"
        use_conv_history = True

    class algorithm(G1MimicPrivCfgPPO.algorithm):
        world_model_loss_coef = 0.05
        adapter_reg_coef = 0.05
        stand_anchor_coef = 2.0
        synthetic_stand_anchor_coef = 2.0
        synthetic_stand_root_height = 0.793
        stand_vel_threshold = 0.05
        stand_dof_threshold = 0.15
        world_model_loss_type = "smooth_l1"
        weight_decay = 1e-4


# ======================== V3 Tracking-Error-Aware Adapter ========================

class G1MimicStuAnyAdapterV3Cfg(G1MimicStuAnyAdapterSafeCfg):
    """V3: let only the residual adapter see tracking error features.

    The history encoder and world model remain dynamics-only:
        history frame = selected robot state + previous action

    The adapter additionally receives:
        reference root velocity/yaw velocity,
        reference roll/pitch - actual roll/pitch,
        reference dof position - actual dof position.
    """

    class env(G1MimicStuAnyAdapterSafeCfg.env):
        use_anyadapter = True
        normalize_obs = False


class G1MimicStuAnyAdapterV3CfgPPO(G1MimicStuAnyAdapterSafeCfgPPO):
    class runner(G1MimicStuAnyAdapterSafeCfgPPO.runner):
        policy_class_name = "TwistAnyAdapterActorCritic"
        algorithm_class_name = "PPOAnyAdapter"
        runner_class_name = "OnPolicyRunnerMimic"
        experiment_name = "g1_twist_anyadapter_v3"
        run_name = ""

    class policy(G1MimicStuAnyAdapterSafeCfgPPO.policy):
        use_tracking_error_adapter_input = True
        action_delta_scale = 0.05
        adapter_gain = 1.0

    class algorithm(G1MimicStuAnyAdapterSafeCfgPPO.algorithm):
        world_model_loss_coef = 0.05
        adapter_reg_coef = 0.05
        stand_anchor_coef = 2.0
        synthetic_stand_anchor_coef = 2.0
        world_model_loss_type = "smooth_l1"
        weight_decay = 1e-4


# ======================== V4 Joint Dynamics-Control Adapter ========================

class G1MimicStuAnyAdapterV4Cfg(G1MimicStuAnyAdapterV3Cfg):
    """V4 uses learnable-but-meaningful dynamics variation for residual training."""

    class env(G1MimicStuAnyAdapterV3Cfg.env):
        use_anyadapter = True
        normalize_obs = False

    class domain_rand(G1MimicStuAnyAdapterV3Cfg.domain_rand):
        gravity_range = (-0.10, 0.10)
        friction_range = [0.20, 2.0]
        added_mass_range = [-3.0, 3.0]
        added_com_range = [-0.05, 0.05]
        max_push_vel_xy = 0.8
        max_push_force_end_effector = 15.0
        motor_strength_range = [0.8, 1.2]
        mimic_obs_noise_std = 0.01
        mimic_obs_dropout_prob = 0.02
        mimic_obs_delay_max = 2


class G1MimicStuAnyAdapterV4CfgPPO(G1MimicStuAnyAdapterV3CfgPPO):
    class runner(G1MimicStuAnyAdapterV3CfgPPO.runner):
        policy_class_name = "TwistAnyAdapterActorCritic"
        algorithm_class_name = "PPOAnyAdapter"
        runner_class_name = "OnPolicyRunnerMimic"
        experiment_name = "g1_twist_anyadapter_v4_0529"
        run_name = ""
        save_interval = 500

    class policy(G1MimicStuAnyAdapterV3CfgPPO.policy):
        # Explicit here so V4 cannot accidentally inherit a different base.
        base_actor_jit_path = "/home/hank/TWIST（anyadapter）/legged_gym/logs/g1_stu_rl/0529_twist_rlbcstu/traced/0529_twist_rlbcstu-36500-jit.pt"
        use_tracking_error_adapter_input = True
        compact_adapter_input = True
        history_policy_grad_scale = 0.10
        action_delta_scale = 0.10
        adapter_gain = 1.0
        init_noise_std = 0.05
        freeze_base = True

    class algorithm(G1MimicStuAnyAdapterV3CfgPPO.algorithm):
        joint_encoder_optimization = True
        world_model_loss_coef = 0.10
        adapter_reg_coef = 0.01
        stand_anchor_coef = 2.0
        synthetic_stand_anchor_coef = 2.0
        world_model_loss_type = "smooth_l1"
        weight_decay = 1e-4


# ======================== Dual Dynamics / Tracking-Error Branches ========================

class G1MimicStuAnyAdapterDualCfg(G1MimicStuAnyAdapterV4Cfg):
    """V4 environment with structurally independent residual branches."""

    class env(G1MimicStuAnyAdapterV4Cfg.env):
        use_anyadapter = True
        normalize_obs = False


class G1MimicStuAnyAdapterDualCfgPPO(G1MimicStuAnyAdapterV4CfgPPO):
    class runner(G1MimicStuAnyAdapterV4CfgPPO.runner):
        policy_class_name = "TwistAnyAdapterActorCritic"
        algorithm_class_name = "PPOAnyAdapter"
        runner_class_name = "OnPolicyRunnerMimic"
        experiment_name = "g1_twist_anyadapter_dual"
        run_name = ""

    class policy(G1MimicStuAnyAdapterV4CfgPPO.policy):
        use_dual_branch_adapter = True
        use_tracking_error_adapter_input = True
        compact_adapter_input = True
        dynamics_branch_gain = 1.0
        tracking_branch_gain = 1.0
        dynamics_action_delta_scale = 0.05
        tracking_action_delta_scale = 0.05
        adapter_gain = 1.0


# ======================== DTERA Uncertainty/Risk-Gated Dual Adapter ========================

class G1MimicStuAnyAdapterDTERACfg(G1MimicStuAnyAdapterV4Cfg):
    """Independent DTERA task; existing Dual/V4-V6 configurations are unchanged."""

    class env(G1MimicStuAnyAdapterV4Cfg.env):
        use_anyadapter = True
        normalize_obs = False
        base_obs_dim = 1155

        anyadapter_history_len = 20
        anyadapter_state_indices = ANYADAPTER_STATE_INDICES
        anyadapter_hist_state_dim = 51
        anyadapter_history_frame_dim = 74
        anyadapter_fill_history_on_reset = True

        use_tracking_error_history = True
        tracking_error_history_len = 20
        tracking_error_frame_dim = 53
        tracking_ref_dof_vel_filter_alpha = 0.5
        tracking_ref_dof_vel_clip = 20.0

        anyadapter_context_dim = 0
        anyadapter_added_obs_dim = 20 * 74 + 20 * 53
        anyadapter_final_policy_obs_dim = 1155 + anyadapter_added_obs_dim


class G1MimicStuAnyAdapterDTERACfgPPO(G1MimicPrivCfgPPO):
    class runner(G1MimicPrivCfgPPO.runner):
        policy_class_name = "TwistDTERAActorCritic"
        algorithm_class_name = "PPODTERA"
        runner_class_name = "OnPolicyRunnerMimic"
        experiment_name = "g1_twist_dtera_gate_v1"
        run_name = ""
        resume = False
        save_interval = 50

    class policy(G1MimicPrivCfgPPO.policy):
        base_actor_jit_path = "/home/hank/TWIST（anyadapter）/legged_gym/logs/g1_stu_rl/0529_twist_rlbcstu/traced/0529_twist_rlbcstu-36500-jit.pt"
        base_obs_dim = 1155
        base_single_obs_dim = 105
        base_history_len = 10

        use_dual_branch_adapter = True
        use_tracking_error_adapter_input = True
        use_tracking_error_history = True
        use_error_trend_predictor = True
        use_world_model_ensemble = True
        use_adaptive_residual_gate = True

        history_len = 20
        hist_state_dim = 51
        history_frame_dim = 74
        wm_target_indices = ANYADAPTER_STATE_INDICES
        latent_dim = 32
        history_policy_grad_scale = 0.0

        tracking_history_len = 20
        tracking_error_frame_dim = 53
        tracking_latent_dim = 32
        tracking_history_policy_grad_scale = 0.25

        adapter_hidden_dims = [128, 128]
        world_model_hidden_dims = [256, 256]
        error_predictor_hidden_dims = [128, 128]
        risk_predictor_hidden_dims = [128, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"
        use_conv_history = True
        default_ref_dof_pos = DEFAULT_REF_DOF_POS

        dynamics_branch_gain = 1.0
        tracking_branch_gain = 1.0
        dynamics_action_delta_scale = 0.03
        tracking_action_delta_scale = 0.03
        adapter_gain = 1.0
        adapter_branch_mode = "full"
        init_noise_std = 0.05
        # The frozen base actor is already competent and each residual branch
        # is capped at 0.03.  Letting entropy optimization grow exploration to
        # 0.38 overwhelms the residual learning signal and hurts deterministic
        # evaluation, so DTERA keeps the intended small exploration scale.
        fix_action_std = True
        freeze_base = True

        world_model_ensemble_size = 3
        tracking_error_scales = [0.35, 2.0, 1.0, 0.35, 0.08]
        gate_demand_k = 1.0
        gate_confidence_k = 1.0
        gate_risk_k = 5.0
        gate_mode = "demand_only"
        confidence_gate_strength = 0.0
        residual_warmup_iterations = 1000
        freeze_residual_output_bias = True
        wm_variance_ema_decay = 0.99

    class algorithm(G1MimicPrivCfgPPO.algorithm):
        # With fixed exploration std and residual warm-up, policy KL is very
        # small early on.  The inherited adaptive schedule therefore ramps
        # the shared PPO/tracking-aux optimizer from 2e-4 toward 1e-2 and
        # collapses the tracking branch into tanh saturation.
        schedule = "fixed"
        learning_rate = 2e-4
        entropy_coef = 0.0
        std_schedule = [0.05, 0.05, 0, 1]
        fixed_action_std = 0.05
        joint_encoder_optimization = True
        world_model_loss_coef = 0.10
        error_prediction_loss_coef = 0.05
        adapter_reg_coef = 0.02
        adapter_reg_initial_coef = 0.10
        adapter_reg_anneal_iterations = 1000
        residual_saturation_reg_coef = 0.05
        adapter_bias_reg_coef = 0.10
        stand_anchor_coef = 0.0
        synthetic_stand_anchor_coef = 2.0
        synthetic_stand_root_height = 0.793
        world_model_loss_type = "smooth_l1"
        weight_decay = 1e-4

        risk_horizon = 10
        risk_adaptive_pos_weight = True
        risk_pos_weight_max = 20.0
        defer_world_model_update = True
        world_model_bootstrap = True
        world_model_bootstrap_probability = 0.8


# ======================== DTERA Selective Independent Gates ========================

class G1MimicStuAnyAdapterDTERASelectiveCfg(G1MimicStuAnyAdapterDTERACfg):
    """DTERA environment kept identical for a controlled policy comparison."""


class G1MimicStuAnyAdapterDTERASelectiveCfgPPO(
    G1MimicStuAnyAdapterDTERACfgPPO
):
    """Conservative independent gates for the next sim-to-real training run.

    Dynamics compensation opens earlier than tracking compensation, but both
    branches close exactly at stand. WM confidence and learned risk stay
    diagnostic until multi-seed evaluation demonstrates reliable ranking.
    """

    class runner(G1MimicStuAnyAdapterDTERACfgPPO.runner):
        experiment_name = "g1_twist_dtera_selective_gate_v1"
        run_name = ""

    class policy(G1MimicStuAnyAdapterDTERACfgPPO.policy):
        use_independent_branch_gates = True
        tracking_demand_mode = "smoothstep"
        tracking_demand_low = 0.30
        tracking_demand_high = 0.80
        dynamics_demand_low = 0.10
        dynamics_demand_high = 0.50
        dynamics_gate_scale = 0.50
        tracking_gate_scale = 1.0
        dynamics_confidence_gate_strength = 0.0
        gate_mode = "demand_only"
        confidence_gate_strength = 0.0


# ======================== V5 Heading-Aware, Bias-Controlled ========================

class G1MimicStuAnyAdapterV5Cfg(G1MimicStuAnyAdapterV4Cfg):
    """V5 adds adapter-only heading feedback without changing the base actor."""

    class env(G1MimicStuAnyAdapterV4Cfg.env):
        anyadapter_context_dim = 2
        anyadapter_added_obs_dim = 1482
        anyadapter_final_policy_obs_dim = 2637


class G1MimicStuAnyAdapterV5CfgPPO(G1MimicStuAnyAdapterV4CfgPPO):
    class runner(G1MimicStuAnyAdapterV4CfgPPO.runner):
        policy_class_name = "TwistAnyAdapterActorCritic"
        algorithm_class_name = "PPOAnyAdapter"
        runner_class_name = "OnPolicyRunnerMimic"
        experiment_name = "g1_twist_anyadapter_v5_heading_0529"
        run_name = ""
        save_interval = 500

    class policy(G1MimicStuAnyAdapterV4CfgPPO.policy):
        base_actor_jit_path = "/home/hank/TWIST（anyadapter）/legged_gym/logs/g1_stu_rl/0529_twist_rlbcstu/traced/0529_twist_rlbcstu-36500-jit.pt"
        adapter_context_dim = 2
        action_delta_scale = 0.05
        adapter_gain = 1.0
        freeze_base = True

    class algorithm(G1MimicStuAnyAdapterV4CfgPPO.algorithm):
        adapter_reg_coef = 0.05
        adapter_bias_reg_coef = 0.20
        world_model_loss_coef = 0.10
        joint_encoder_optimization = True


# ======================== V6 Any2Track-Style Layer Adapters ========================

class G1MimicStuAnyAdapterV6Cfg(G1MimicStuAnyAdapterV4Cfg):
    """Long-history adaptation with in-place stability protection."""

    class env(G1MimicStuAnyAdapterV4Cfg.env):
        use_anyadapter = True
        normalize_obs = False
        anyadapter_history_len = 79
        anyadapter_hist_state_dim = 51
        anyadapter_history_frame_dim = 74
        anyadapter_added_obs_dim = 79 * 74
        anyadapter_final_policy_obs_dim = 1155 + 79 * 74
        anyadapter_context_dim = 0
        anyadapter_fill_history_on_reset = True

    class rewards(G1MimicStuAnyAdapterV4Cfg.rewards):
        # Mocap root estimates contain small translational noise. Do not reward
        # stepping until the reference has a clear locomotion command.
        locomotion_ref_vel_threshold = 0.12
        in_place_ref_vel_threshold = 0.12
        in_place_ref_yaw_vel_threshold = 0.12

        class scales(G1MimicStuAnyAdapterV4Cfg.rewards.scales):
            tracking_root_vel = 1.5
            feet_slip = -0.2
            action_rate = -0.02
            in_place_root_motion = -1.0
            in_place_feet_motion = -0.25


class G1MimicStuAnyAdapterV6CfgPPO(G1MimicPrivCfgPPO):
    class runner(G1MimicPrivCfgPPO.runner):
        policy_class_name = "TwistAny2TrackActorCritic"
        algorithm_class_name = "PPOAny2Track"
        runner_class_name = "OnPolicyRunnerMimic"
        experiment_name = "g1_twist_any2track_v6_0529"
        run_name = ""
        save_interval = 500
        # A 24-step rollout contains a contiguous 20-step world-model window.
        num_steps_per_env = 24

    class policy(G1MimicPrivCfgPPO.policy):
        base_actor_jit_path = "/home/hank/TWIST（anyadapter）/legged_gym/logs/g1_stu_rl/0529_twist_rlbcstu/traced/0529_twist_rlbcstu-36500-jit.pt"
        base_obs_dim = 1155
        history_len = 79
        hist_state_dim = 51
        history_frame_dim = 74
        wm_target_indices = ANYADAPTER_STATE_INDICES
        latent_dim = 128
        world_model_hidden_dims = [512, 512, 256, 256, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "silu"
        init_noise_std = 0.05
        fix_action_std = False
        adapter_gain = 1.0
        freeze_base = True

    class algorithm(G1MimicPrivCfgPPO.algorithm):
        # 7001-D V6 observations make the inherited 4-way mini-batches too
        # large for a 24 GiB GPU at 4096 envs.  This keeps the same rollout and
        # number of PPO epochs while reducing the per-update memory peak by 4x.
        num_mini_batches = 16
        policy_learning_rate = 5e-5
        world_model_learning_rate = 1e-4
        world_model_loss_coef = 1.0
        world_model_loss_type = "smooth_l1"
        world_model_sequence_length = 20
        world_model_num_epochs = 1
        world_model_component_weights = [5.0, 5.0, 1.0, 0.5]
        # Layer adapters can otherwise grow far beyond the frozen base action.
        adapter_reg_coef = 0.05
        adapter_bias_reg_coef = 0.0
        # For stationary references with neutral legs, constrain only the
        # 12 leg-action deltas. Arm and waist tracking remain unrestricted.
        stand_anchor_coef = 0.5
        synthetic_stand_anchor_coef = 0.0
        stand_vel_threshold = 0.12
        stand_dof_threshold = 0.18
        weight_decay = 1e-5
