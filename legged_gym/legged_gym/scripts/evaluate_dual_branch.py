"""Inference-only branch ablation evaluation for the Dual AnyAdapter.

Loads ONE Dual AnyAdapter checkpoint (a = a_base + adapter_gain * delta) and
rolls it out with branch masks (and, for DTERA, gate ablations) by changing
inference-only attributes:
inference time:

    base_only -> a_base
    full      -> a_base + adapter_gain * (dyn_gain * delta_dyn + err_gain * delta_err)
    dyn_only  -> a_base + adapter_gain * (dyn_gain * delta_dyn)
    err_only  -> a_base + adapter_gain * (err_gain * delta_err)

The mask is applied only through TwistAnyAdapterActorCritic.adapter_branch_mode;
network weights, the world model, and the optimizer are never touched, so a
single checkpoint serves all modes.  DTERA additionally supports
full_gate_off/full_demand_only/full_demand_confidence/full_gate. Fairness:
identical task config,
checkpoint, seed, motion file, and domain randomization across modes; only the
branch mask changes (run each mode in a separate process with the same flags).

Results are written to <out_dir>/<branch_mode>.json (full report) and
<out_dir>/<branch_mode>.csv (flat summary row).

Every run also loads or creates a scenario manifest.  Each environment then
replays one fixed motion/start/DR scenario and contributes exactly one
fixed-horizon result, so action-dependent reset timing cannot change the
motion sequence used by another ablation mode.
"""

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

# NOTE: isaacgym must be imported before torch; legged_gym.envs does that.
from legged_gym.envs import *  # noqa: F401,F403  (registers tasks, imports isaacgym)
from legged_gym.gym_utils import task_registry
from legged_gym.envs.base.legged_robot import euler_from_quaternion
from legged_gym.envs.base.humanoid_char import convert_to_local_root_body_pos
from legged_gym.envs.g1.g1_mimic_distill import G1MimicDistill
from legged_gym.scripts.evaluation_metrics import rmse_from_mean_square

import torch
from isaacgym import gymapi

BRANCH_MODES = ("base_only", "full", "dyn_only", "err_only")
DTERA_MODES = (
    "full_gate_off",
    "full_demand_only",
    "full_demand_confidence",
    "full_gate",
)
EVAL_MODES = BRANCH_MODES + DTERA_MODES

# Termination reason labels (priority order matches the env's reset semantics).
REASON_NONE = "none"
REASON_TIMEOUT = "timeout"
REASON_MOTION_END = "motion_end"
REASON_POSE_FAIL = "pose_fail"
REASON_CONTACT = "contact"
REASON_ROLL_PITCH = "roll_pitch"
REASON_HEIGHT = "height"
REASON_VELOCITY = "velocity"
REASON_OTHER = "other"
REASON_HORIZON = "horizon"
SCENARIO_MANIFEST_VERSION = 1


def instrument_termination_reasons() -> None:
    """Monkey-patch check_termination to record a per-env reason buffer.

    Pure read-only instrumentation: the original function runs first and its
    behavior is unchanged; the wrapper only fills env._term_reason_buf with
    the first condition that triggered reset_buf for each env.
    """
    original = G1MimicDistill.check_termination

    def instrumented_check_termination(self):
        original(self)

        num_envs = self.num_envs
        reason = [REASON_NONE] * num_envs

        time_out = self.time_out_buf
        motion_end = (
            self.episode_length_buf * self.dt
            >= self._motion_lib.get_motion_length(self._motion_ids)
        )
        contact = torch.any(
            torch.norm(
                self.contact_forces[:, self.termination_contact_indices, :],
                dim=-1,
            )
            > 1.0,
            dim=1,
        )
        roll_cut = torch.abs(self.roll) > self.cfg.rewards.termination_roll
        pitch_cut = torch.abs(self.pitch) > self.cfg.rewards.termination_pitch
        height = (
            torch.abs(self.root_states[:, 2] - self._ref_root_pos[:, 2])
            > self.cfg.rewards.root_height_diff_threshold
        )
        velocity = torch.norm(self.root_states[:, 7:10], dim=-1) > 5.0

        pose_fail = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        if self._pose_termination:
            body_pos = (
                self.rigid_body_states[:, self._key_body_ids, 0:3]
                - self.rigid_body_states[:, 0:1, 0:3]
            )
            tar_body_pos = (
                self._ref_body_pos[:, self._key_body_ids]
                - self._ref_root_pos[:, None, :]
            )
            if not self.global_obs:
                body_pos = convert_to_local_root_body_pos(
                    self.root_states[:, 3:7], body_pos
                )
                tar_body_pos = convert_to_local_root_body_pos(
                    self._ref_root_rot, tar_body_pos
                )
            body_pos_dist = (
                (tar_body_pos - body_pos).square().sum(dim=-1).max(dim=-1)[0]
            )
            pose_fail = body_pos_dist > self._pose_termination_dist ** 2
            if self._track_root:
                root_dist = (
                    self._ref_root_pos - self.root_states[:, 0:3]
                ).square().sum(dim=-1)
                pose_fail |= (
                    root_dist.squeeze(-1)
                    > self._root_tracking_termination_dist ** 2
                )

        reset = self.reset_buf.cpu().numpy()
        for i in range(num_envs):
            if not reset[i]:
                continue
            if time_out[i]:
                reason[i] = REASON_TIMEOUT
            elif motion_end[i]:
                reason[i] = REASON_MOTION_END
            elif pose_fail[i]:
                reason[i] = REASON_POSE_FAIL
            elif contact[i]:
                reason[i] = REASON_CONTACT
            elif roll_cut[i] or pitch_cut[i]:
                reason[i] = REASON_ROLL_PITCH
            elif height[i]:
                reason[i] = REASON_HEIGHT
            elif velocity[i]:
                reason[i] = REASON_VELOCITY
            else:
                reason[i] = REASON_OTHER

        self._term_reason_buf = reason

    G1MimicDistill.check_termination = instrumented_check_termination


def build_eval_args(args: argparse.Namespace, headless: bool) -> argparse.Namespace:
    """Namespace with every field task_registry/parse_sim_params expects."""
    return argparse.Namespace(
        task=args.task,
        seed=args.seed,
        num_envs=args.num_envs,
        physics_engine=gymapi.SIM_PHYSX,
        sim_device_type="cuda" if "cuda" in args.device else "cpu",
        sim_device_id=0,
        compute_device_id=0,
        graphics_device_id=0,
        sim_device="cuda" if "cuda" in args.device else "cpu",
        rl_device=args.device,
        device=args.device,
        headless=headless,
        use_gpu=True,
        use_gpu_pipeline=True,
        subscenes=0,
        num_threads=0,
        teleop_mode=False,
        rows=None,
        cols=None,
        record_video=False,
        no_rand=args.no_dr,
        resume=False,
        max_iterations=None,
        experiment_name=None,
        run_name=None,
        load_run=None,
        checkpoint=None,
        fix_action_std=False,
        teacher_exptid="mimic",
        teacher_checkpoint=-1,
        eval_student=False,
        proj_name="g1",
        exptid=None,
        resumeid=None,
    )


def checkpoint_md5(path: str) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_tensor_cpu(env, name):
    value = getattr(env, name, None)
    return None if value is None else value.detach().cpu()


def _scenario_rows_from_env(env, max_steps):
    motion_ids = env._motion_ids.detach().cpu().tolist()
    motion_times = env._motion_time_offsets.detach().cpu().tolist()
    mass = _optional_tensor_cpu(env, "mass_params_tensor")
    friction = _optional_tensor_cpu(env, "friction_coeffs_tensor")
    motor = _optional_tensor_cpu(env, "motor_strength")
    rows = []
    for env_id in range(env.num_envs):
        rows.append({
            "scenario_id": env_id,
            "motion_id": int(motion_ids[env_id]),
            "start_time_s": float(motion_times[env_id]),
            "max_steps": int(max_steps),
            "domain_randomization": {
                "mass_com": (
                    [] if mass is None else [float(x) for x in mass[env_id]]
                ),
                "friction": (
                    None if friction is None else float(friction[env_id])
                ),
                "motor_p": (
                    [] if motor is None else [float(x) for x in motor[0, env_id]]
                ),
                "motor_d": (
                    [] if motor is None else [float(x) for x in motor[1, env_id]]
                ),
            },
        })
    return rows


def _dynamic_dr_config(env):
    cfg = env.cfg.domain_rand
    names = (
        "domain_rand_general", "randomize_gravity", "gravity_rand_interval_s",
        "gravity_range", "push_robots", "push_interval_s", "max_push_vel_xy",
        "push_end_effector", "push_end_effector_interval_s",
        "max_push_force_end_effector", "randomize_mimic_obs",
        "mimic_obs_noise_std", "mimic_obs_dropout_prob",
        "mimic_obs_delay_max", "mimic_obs_lpf_prob",
        "mimic_obs_lpf_alpha_range", "action_delay", "action_buf_len",
    )
    result = {}
    for name in names:
        if hasattr(cfg, name):
            value = getattr(cfg, name)
            if isinstance(value, tuple):
                value = list(value)
            result[name] = value
    return result


def _manifest_digest(manifest):
    payload = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_and_apply_scenario_manifest(env, args, manifest_path):
    expected = {
        "version": SCENARIO_MANIFEST_VERSION,
        "task": args.task,
        "seed": args.seed,
        "num_envs": env.num_envs,
        "max_steps": args.max_steps,
        "device": args.device,
        "scenario_start_mode": args.scenario_start_mode,
        "dt": float(env.dt),
        "motion_file": str(env.cfg.motion.motion_file),
        "domain_rand_enabled": bool(env.cfg.domain_rand.domain_rand_general),
        "dynamic_domain_randomization": _dynamic_dr_config(env),
    }
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(
                    f"Scenario manifest mismatch for {key}: "
                    f"expected {value!r}, found {manifest.get(key)!r}"
                )
        if len(manifest.get("scenarios", [])) != env.num_envs:
            raise ValueError("Scenario manifest does not contain one row per env")
        created = False
    else:
        manifest = dict(expected)
        manifest["created_at"] = datetime.now().isoformat(timespec="seconds")
        manifest["scenarios"] = _scenario_rows_from_env(env, args.max_steps)
        if args.scenario_start_mode == "uniform":
            start_generator = torch.Generator(device="cpu")
            start_generator.manual_seed(int(args.seed) + 130363)
            phases = torch.rand(env.num_envs, generator=start_generator)
            motion_ids = torch.tensor(
                [row["motion_id"] for row in manifest["scenarios"]],
                dtype=torch.long,
                device=env.device,
            )
            motion_lengths = env._motion_lib.get_motion_length(
                motion_ids
            ).detach().cpu()
            start_times = phases * (motion_lengths - float(env.dt)).clamp_min(0.0)
            for env_id, row in enumerate(manifest["scenarios"]):
                row["start_time_s"] = float(start_times[env_id])
        # HumanoidChar normally draws a fresh 0.8..1.2 joint-position scale on
        # every reset. Generate it from an isolated CPU RNG so creating the
        # manifest does not shift the rollout's global random stream.
        dof_generator = torch.Generator(device="cpu")
        dof_generator.manual_seed(int(args.seed) + 104729)
        dof_scale = 0.8 + 0.4 * torch.rand(
            env.num_envs, env.num_dof, generator=dof_generator
        )
        for env_id, row in enumerate(manifest["scenarios"]):
            row["initial_dof_pos_scale"] = [
                float(x) for x in dof_scale[env_id]
            ]
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        created = True

    rows = sorted(manifest["scenarios"], key=lambda row: row["scenario_id"])
    if [row["scenario_id"] for row in rows] != list(range(env.num_envs)):
        raise ValueError("Scenario IDs must be exactly 0..num_envs-1")
    if any(row.get("max_steps") != args.max_steps for row in rows):
        raise ValueError("Every scenario max_steps must match the manifest horizon")
    manifest["scenarios"] = rows

    # Static physical randomization is applied during environment creation.
    # Requiring an exact replay prevents a same-seed run from silently using
    # different mass/friction when RNG consumption or config changes.
    observed_rows = _scenario_rows_from_env(env, args.max_steps)
    for expected_row, observed_row in zip(rows, observed_rows):
        if expected_row["domain_randomization"] != observed_row["domain_randomization"]:
            raise ValueError(
                "Static domain randomization does not match scenario manifest "
                f"for env {expected_row['scenario_id']}"
            )

    device = env.device
    env._eval_scenario_motion_ids = torch.tensor(
        [row["motion_id"] for row in rows], dtype=torch.long, device=device
    )
    env._eval_scenario_motion_times = torch.tensor(
        [row["start_time_s"] for row in rows], dtype=torch.float, device=device
    )
    if any(
        len(row.get("initial_dof_pos_scale", [])) != env.num_dof
        for row in rows
    ):
        raise ValueError(
            "Each scenario must contain one initial_dof_pos_scale per DoF"
        )
    env._eval_scenario_dof_pos_scale = torch.tensor(
        [row["initial_dof_pos_scale"] for row in rows],
        dtype=torch.float,
        device=device,
    )
    motion_lengths = env._motion_lib.get_motion_length(
        env._eval_scenario_motion_ids
    )
    if torch.any(env._eval_scenario_motion_times < 0) or torch.any(
        env._eval_scenario_motion_times >= motion_lengths
    ):
        raise ValueError("Scenario start_time_s must lie inside its motion")

    all_env_ids = torch.arange(env.num_envs, device=device, dtype=torch.long)
    env.reset_idx(all_env_ids)
    env.compute_observations()
    digest = _manifest_digest(manifest)
    state = "created" if created else "loaded"
    print(
        f"[evaluate_dual_branch] scenario manifest {state}: {manifest_path} "
        f"sha256={digest[:12]}"
    )
    return manifest, digest


def combine_branches(actor_critic, delta_dyn, delta_err, mode):
    """Mirror of TwistAnyAdapterActorCritic.action_delta for the dual adapter."""
    if mode == "base_only":
        return torch.zeros_like(delta_dyn)
    if mode == "dyn_only":
        return actor_critic.dynamics_branch_gain * delta_dyn
    if mode == "err_only":
        return actor_critic.tracking_branch_gain * delta_err
    return (
        actor_critic.dynamics_branch_gain * delta_dyn
        + actor_critic.tracking_branch_gain * delta_err
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=str, default="g1_stu_anyadapter_dual")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the trained Dual AnyAdapter checkpoint (*.pt).")
    parser.add_argument("--branch_mode", type=str, choices=EVAL_MODES,
                        default="full")
    parser.add_argument("--num_envs", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=2000,
                        help="Number of control steps per rollout.")
    parser.add_argument("--episode_length_s", type=float, default=None,
                        help="Override env episode length (seconds). Default: task config.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--show_viewer", action="store_true",
                        help="Attach a viewer instead of running headless.")
    parser.add_argument("--no_dr", action="store_true",
                        help="Disable domain randomization (controlled runs).")
    parser.add_argument("--adapter_gain", type=float, default=None,
                        help="Inference-only global residual gain override.")
    parser.add_argument("--dynamics_gain", type=float, default=None,
                        help="Inference-only dynamics branch gain override.")
    parser.add_argument("--tracking_gain", type=float, default=None,
                        help="Inference-only tracking branch gain override.")
    gate_group = parser.add_mutually_exclusive_group()
    gate_group.add_argument(
        "--independent_branch_gates", dest="independent_branch_gates",
        action="store_const", const=True, default=None,
        help="Use separate dynamics/tracking gates for DTERA inference.",
    )
    gate_group.add_argument(
        "--shared_gate", dest="independent_branch_gates",
        action="store_const", const=False,
        help="Force the legacy shared DTERA gate.",
    )
    parser.add_argument("--dynamics_gate_scale", type=float, default=None)
    parser.add_argument("--tracking_gate_scale", type=float, default=None)
    parser.add_argument(
        "--tracking_demand_mode", choices=("legacy_exp", "smoothstep"),
        default=None,
    )
    parser.add_argument("--tracking_demand_low", type=float, default=None)
    parser.add_argument("--tracking_demand_high", type=float, default=None)
    parser.add_argument("--dynamics_demand_low", type=float, default=None)
    parser.add_argument("--dynamics_demand_high", type=float, default=None)
    parser.add_argument(
        "--scenario_manifest", type=str, default=None,
        help=(
            "Fixed scenario manifest. If omitted, create/reuse one inside "
            "out_dir, keyed by seed/DR/env-count/horizon."
        ),
    )
    parser.add_argument(
        "--scenario_start_mode", choices=("uniform", "zero"),
        default="uniform",
        help="Start-frame policy used only when creating a new manifest.",
    )
    parser.add_argument(
        "--result_name", type=str, default=None,
        help="Output filename stem; useful for gain sweeps without overwrites.",
    )
    parser.add_argument("--out_dir", type=str,
                        default=str(REPO_ROOT / "results" / "dual_ablation"))
    args = parser.parse_args()

    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.scenario_manifest is None:
        dr_tag = "no_dr" if args.no_dr else "dr"
        scenario_manifest_path = out_dir / (
            f"scenario_seed{args.seed}_{dr_tag}_{args.num_envs}env_"
            f"{args.max_steps}steps_{args.scenario_start_mode}.json"
        )
    else:
        scenario_manifest_path = Path(args.scenario_manifest)

    instrument_termination_reasons()
    eval_args = build_eval_args(args, headless=not args.show_viewer)

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # Identical env config across modes; only the branch mask changes.
    env_cfg.env.record_video = False
    env_cfg.env.rand_reset = False
    env_cfg.env.randomize_start_pos = False
    env_cfg.env.randomize_start_yaw = False
    if args.episode_length_s is not None:
        env_cfg.env.episode_length_s = args.episode_length_s
    if hasattr(env_cfg, "motion"):
        env_cfg.motion.motion_curriculum = False
    if hasattr(env_cfg, "noise"):
        env_cfg.noise.add_noise = False
    if args.no_dr:
        # update_cfg_from_args only flips domain_rand_general, while this
        # repository's creation/reset code reads the individual flags. Disable
        # them explicitly so --no_dr is a real no-randomization condition.
        for name in (
            "randomize_gravity", "randomize_friction",
            "randomize_base_mass", "randomize_base_com", "push_robots",
            "push_end_effector", "randomize_motor", "action_delay",
            "randomize_mimic_obs",
        ):
            if hasattr(env_cfg.domain_rand, name):
                setattr(env_cfg.domain_rand, name, False)

    env, _ = task_registry.make_env(name=args.task, args=eval_args, env_cfg=env_cfg)
    scenario_manifest, scenario_manifest_sha256 = (
        _validate_and_apply_scenario_manifest(
            env, args, scenario_manifest_path
        )
    )
    print(f"[evaluate_dual_branch] env num_envs={env.num_envs} "
          f"dt={env.dt} num_actions={env.num_actions}")

    # ``no_rand`` has already been consumed while constructing the env.  The
    # shared config helper also interprets it as a request to replace the
    # configured policy/algorithm with the generic ActorCritic/PPO pair, which
    # is invalid for AnyAdapter/DTERA checkpoint evaluation.  Clear only the
    # runner-side flag; the already-created environment remains deterministic.
    eval_args.no_rand = False

    train_cfg.runner.resume = False
    ppo_runner, _ = task_registry.make_alg_runner(
        log_root=None, env=env, name=args.task, args=eval_args,
        train_cfg=train_cfg, init_wandb=False,
    )
    ppo_runner.load(args.checkpoint, load_optimizer=False)

    actor_critic = ppo_runner.get_actor_critic(device=env.device)
    policy = ppo_runner.get_inference_policy(device=env.device)
    dtera_mode_map = {
        "full_gate_off": ("full", "off", 0.0),
        "full_demand_only": ("full", "demand_only", 0.0),
        "full_demand_confidence": ("full", "demand_confidence", 1.0),
        "full_gate": ("full", "full", 1.0),
    }
    if args.branch_mode in dtera_mode_map:
        if not getattr(actor_critic, "is_dtera", False):
            raise ValueError(f"{args.branch_mode} requires the DTERA task/checkpoint")
        (
            actor_critic.adapter_branch_mode,
            actor_critic.gate_mode,
            actor_critic.confidence_gate_strength,
        ) = dtera_mode_map[args.branch_mode]
    else:
        actor_critic.adapter_branch_mode = args.branch_mode
    for argument, attribute in (
        (args.adapter_gain, "adapter_gain"),
        (args.dynamics_gain, "dynamics_branch_gain"),
        (args.tracking_gain, "tracking_branch_gain"),
    ):
        if argument is not None:
            setattr(actor_critic, attribute, float(argument))
    dtera_overrides = (
        (args.independent_branch_gates, "use_independent_branch_gates"),
        (args.dynamics_gate_scale, "dynamics_gate_scale"),
        (args.tracking_gate_scale, "tracking_gate_scale"),
        (args.tracking_demand_mode, "tracking_demand_mode"),
        (args.tracking_demand_low, "tracking_demand_low"),
        (args.tracking_demand_high, "tracking_demand_high"),
        (args.dynamics_demand_low, "dynamics_demand_low"),
        (args.dynamics_demand_high, "dynamics_demand_high"),
    )
    if any(value is not None for value, _ in dtera_overrides):
        if not getattr(actor_critic, "is_dtera", False):
            raise ValueError("DTERA gate overrides require a DTERA task/checkpoint")
        for value, attribute in dtera_overrides:
            if value is not None:
                setattr(actor_critic, attribute, value)
        if actor_critic.tracking_demand_low >= actor_critic.tracking_demand_high:
            raise ValueError("tracking_demand_low must be below tracking_demand_high")
        if actor_critic.dynamics_demand_low >= actor_critic.dynamics_demand_high:
            raise ValueError("dynamics_demand_low must be below dynamics_demand_high")
    print(f"[evaluate_dual_branch] branch mode: {actor_critic.adapter_branch_mode} "
          f"gate mode: {getattr(actor_critic, 'gate_mode', 'n/a')} "
          f"independent: {getattr(actor_critic, 'use_independent_branch_gates', False)} "
          f"gains: adapter={actor_critic.adapter_gain:g}, "
          f"dyn={actor_critic.dynamics_branch_gain:g}, "
          f"err={actor_critic.tracking_branch_gain:g}")

    obs = env.get_observations()
    if env.cfg.env.normalize_obs:
        normalizer = ppo_runner.get_normalizer(device=env.device)
    else:
        normalizer = None

    # --- accumulators -----------------------------------------------------
    num_envs = env.num_envs
    scenario_return = torch.zeros(num_envs, device=env.device)
    scenario_steps = torch.zeros(num_envs, dtype=torch.long, device=env.device)
    scenario_active = torch.ones(num_envs, dtype=torch.bool, device=env.device)
    scenario_terminated = torch.zeros(
        num_envs, dtype=torch.bool, device=env.device
    )
    scenario_reasons = [REASON_HORIZON] * num_envs
    fall_count = 0
    timeout_count = 0
    motion_end_count = 0
    reason_counts = {label: 0 for label in (
        REASON_TIMEOUT, REASON_MOTION_END, REASON_POSE_FAIL, REASON_CONTACT,
        REASON_ROLL_PITCH, REASON_HEIGHT, REASON_VELOCITY, REASON_OTHER,
    )}
    mean_rew = 0.0
    mean_root_pos_err = 0.0
    mean_roll_err = 0.0
    mean_pitch_err = 0.0
    mean_joint_err = 0.0
    mean_joint_vel_err = 0.0
    mean_root_vel_err = 0.0
    mean_keybody_err = 0.0
    root_pos_mse = 0.0
    roll_mse = 0.0
    pitch_mse = 0.0
    joint_pos_mse = 0.0
    joint_vel_mse = 0.0
    root_vel_mse = 0.0
    keybody_mse = 0.0
    mean_dyn_res = 0.0
    mean_err_res = 0.0
    mean_applied_res = 0.0
    max_applied_res = 0.0
    mean_gate = 0.0
    mean_dynamics_gate = 0.0
    mean_tracking_gate = 0.0
    mean_feet_slip = 0.0
    mean_action_rate = 0.0
    previous_actions = None
    metric_samples = 0
    action_rate_samples = 0
    executed_steps = 0
    startup = []
    # Env's own episode logger (extras), weighted by number of resets per step.
    extras_metric_sums = {}
    extras_metric_weight = 0.0

    for step in range(args.max_steps):
        active_mask = scenario_active.clone()
        if not active_mask.any():
            break
        executed_steps = step + 1
        if normalizer is not None:
            normalized_obs = normalizer.normalize(obs.detach())
        else:
            normalized_obs = obs.detach()
        actions = policy(normalized_obs)
        with torch.inference_mode():
            base_action = actor_critic.base_action(normalized_obs)
            if hasattr(actor_critic, "action_diagnostics"):
                diagnostics = actor_critic.action_diagnostics(
                    normalized_obs, base_action=base_action
                )
                delta_dyn = diagnostics["delta_dyn"]
                delta_err = diagnostics["delta_err"]
                candidate = diagnostics["candidate_delta"]
                gated = diagnostics["gated_delta"]
                applied = actor_critic.adapter_gain * diagnostics["applied_delta"]
                gate = diagnostics["gate"]
                dynamics_gate = diagnostics.get("dynamics_gate", gate)
                tracking_gate = diagnostics.get("tracking_gate", gate)
            else:
                delta_dyn, delta_err = actor_critic.get_adapter_delta_components(normalized_obs)
                candidate = combine_branches(actor_critic, delta_dyn, delta_err,
                                             args.branch_mode)
                gated = candidate
                applied = actor_critic.adapter_gain * candidate
                gate = torch.zeros(candidate.shape[0], device=candidate.device) \
                    if args.branch_mode == "base_only" else torch.ones(candidate.shape[0], device=candidate.device)
                dynamics_gate = gate
                tracking_gate = gate
            dyn_res = torch.norm(delta_dyn, dim=-1)
            err_res = torch.norm(delta_err, dim=-1)
            applied_res = torch.norm(applied, dim=-1)

            if step < 50:
                dyn_scale = actor_critic.adapter.dynamics_branch.delta_scale
                err_scale = actor_critic.adapter.tracking_branch.delta_scale
                candidate_scale = (
                    actor_critic.candidate_delta_scale()
                    if hasattr(actor_critic, "candidate_delta_scale")
                    else (
                        abs(actor_critic.dynamics_branch_gain) * dyn_scale
                        + abs(actor_critic.tracking_branch_gain) * err_scale
                    )
                )
                current_roll, current_pitch, _ = euler_from_quaternion(
                    env.root_states[:, 3:7]
                )
                startup.append({
                    "step": step,
                    "roll_mean": float(current_roll.mean().cpu()),
                    "roll_abs_max": float(current_roll.abs().max().cpu()),
                    "pitch_mean": float(current_pitch.mean().cpu()),
                    "pitch_abs_max": float(current_pitch.abs().max().cpu()),
                    "root_height_mean": float(env.root_states[:, 2].mean().cpu()),
                    "base_action_mean_abs": float(base_action.abs().mean().cpu()),
                    "base_action_max_abs": float(base_action.abs().max().cpu()),
                    "delta_dyn_mean_abs": float(delta_dyn.abs().mean().cpu()),
                    "delta_dyn_max_abs": float(delta_dyn.abs().max().cpu()),
                    "delta_err_mean_abs": float(delta_err.abs().mean().cpu()),
                    "delta_err_max_abs": float(delta_err.abs().max().cpu()),
                    "candidate_delta_mean_abs": float(candidate.abs().mean().cpu()),
                    "candidate_delta_max_abs": float(candidate.abs().max().cpu()),
                    "gated_delta_mean_abs": float(gated.abs().mean().cpu()),
                    "gated_delta_max_abs": float(gated.abs().max().cpu()),
                    "applied_delta_mean_abs": float(applied.abs().mean().cpu()),
                    "applied_delta_max_abs": float(applied.abs().max().cpu()),
                    "gate_mean": float(gate.mean().cpu()),
                    "dynamics_gate_mean": float(dynamics_gate.mean().cpu()),
                    "tracking_gate_mean": float(tracking_gate.mean().cpu()),
                    "alpha_res": float(
                        diagnostics.get("residual_warmup_factor", 1.0)
                    ) if hasattr(actor_critic, "is_dtera") else 1.0,
                    "dyn_saturation_fraction": float(
                        (delta_dyn.abs() >= 0.95 * dyn_scale).float().mean().cpu()
                    ),
                    "err_saturation_fraction": float(
                        (delta_err.abs() >= 0.95 * err_scale).float().mean().cpu()
                    ),
                    "candidate_saturation_fraction": float(
                        0.0 if candidate_scale <= 0.0 else
                        (candidate.abs() >= 0.95 * candidate_scale)
                        .float().mean().cpu()
                    ),
                    "demand_mean": float(
                        diagnostics.get("demand", gate).mean().cpu()
                    ),
                    "confidence_mean": float(
                        diagnostics.get("confidence", torch.ones_like(gate))
                        .mean().cpu()
                    ),
                    "safety_mean": float(
                        diagnostics.get("safety", torch.ones_like(gate))
                        .mean().cpu()
                    ),
                })

        obs, _, rews, dones, infos = env.step(actions.detach())

        scenario_return[active_mask] += rews[active_mask]
        scenario_steps[active_mask] += 1

        # Tracking errors from the env's current state vs reference motion.
        root_pos_error_vector = env.root_states[:, :3] - env._ref_root_pos
        root_pos_err = torch.norm(root_pos_error_vector, dim=-1)
        current_roll, current_pitch, _ = euler_from_quaternion(
            env.root_states[:, 3:7]
        )
        roll_ref, pitch_ref, _ = euler_from_quaternion(env._ref_root_rot)
        roll_err = torch.abs(torch.atan2(
            torch.sin(current_roll - roll_ref), torch.cos(current_roll - roll_ref)
        ))
        pitch_err = torch.abs(torch.atan2(
            torch.sin(current_pitch - pitch_ref), torch.cos(current_pitch - pitch_ref)
        ))
        joint_error_vector = env.dof_pos - env._ref_dof_pos
        joint_vel_error_vector = env.dof_vel - env._ref_dof_vel
        root_vel_error_vector = env.root_states[:, 7:10] - env._ref_root_vel
        keybody_error_vector = (
            env.rigid_body_states[:, env._key_body_ids, 0:3]
            - env._ref_body_pos[:, env._key_body_ids]
        )
        joint_err = joint_error_vector.abs().mean(dim=-1)
        joint_vel_err = joint_vel_error_vector.abs().mean(dim=-1)
        root_vel_err = torch.norm(root_vel_error_vector, dim=-1)
        keybody_err = torch.norm(keybody_error_vector, dim=-1).mean(dim=-1)

        active_count = int(active_mask.sum().item())
        metric_samples += active_count
        mean_rew += float(rews[active_mask].sum().detach().cpu())
        mean_root_pos_err += float(root_pos_err[active_mask].sum().detach().cpu())
        mean_roll_err += float(roll_err[active_mask].sum().detach().cpu())
        mean_pitch_err += float(pitch_err[active_mask].sum().detach().cpu())
        mean_joint_err += float(joint_err[active_mask].sum().detach().cpu())
        mean_joint_vel_err += float(
            joint_vel_err[active_mask].sum().detach().cpu()
        )
        mean_root_vel_err += float(
            root_vel_err[active_mask].sum().detach().cpu()
        )
        mean_keybody_err += float(
            keybody_err[active_mask].sum().detach().cpu()
        )
        root_pos_mse += float(
            root_pos_error_vector.square().mean(dim=-1)[active_mask].sum().cpu()
        )
        roll_mse += float(roll_err.square()[active_mask].sum().cpu())
        pitch_mse += float(pitch_err.square()[active_mask].sum().cpu())
        joint_pos_mse += float(
            joint_error_vector.square().mean(dim=-1)[active_mask].sum().cpu()
        )
        joint_vel_mse += float(
            joint_vel_error_vector.square().mean(dim=-1)[active_mask].sum().cpu()
        )
        root_vel_mse += float(
            root_vel_error_vector.square().mean(dim=-1)[active_mask].sum().cpu()
        )
        keybody_mse += float(
            keybody_error_vector.square().mean(dim=(1, 2))[active_mask].sum().cpu()
        )
        mean_dyn_res += float(dyn_res[active_mask].sum().detach().cpu())
        mean_err_res += float(err_res[active_mask].sum().detach().cpu())
        mean_applied_res += float(
            applied_res[active_mask].sum().detach().cpu()
        )
        max_applied_res = max(
            max_applied_res,
            float(applied_res[active_mask].max().detach().cpu()),
        )
        mean_gate += float(gate[active_mask].sum().detach().cpu())
        mean_dynamics_gate += float(
            dynamics_gate[active_mask].sum().detach().cpu()
        )
        mean_tracking_gate += float(
            tracking_gate[active_mask].sum().detach().cpu()
        )
        feet_vel = env.rigid_body_states[:, env.feet_indices, 7:10]
        contacts = (env.contact_forces[:, env.feet_indices, 2] > 5.0).float()
        feet_slip = (
            feet_vel[:, :, :2].norm(dim=-1) * contacts
        ).mean(dim=-1)
        mean_feet_slip += float(feet_slip[active_mask].sum().detach().cpu())
        if previous_actions is not None:
            action_rate_samples += active_count
            mean_action_rate += float(
                (actions - previous_actions).norm(dim=-1)[active_mask]
                .sum().detach().cpu()
            )
        previous_actions = actions.detach().clone()

        all_done_mask = dones.bool()
        done_mask = all_done_mask & active_mask
        if done_mask.any():
            scenario_terminated[done_mask] = True
            timeout_count += int((done_mask & env.time_out_buf).sum())
            fall_count += int((done_mask & ~env.time_out_buf).sum())
            done_ids = done_mask.nonzero(as_tuple=False).flatten().cpu().tolist()
            for env_id in done_ids:
                scenario_reasons[env_id] = env._term_reason_buf[env_id]
            for label in reason_counts:
                reason_counts[label] += sum(
                    1 for i in done_ids if env._term_reason_buf[i] == label
                )
            scenario_active[done_mask] = False

            # Env's own per-episode reward-component logger (extras["episode"]).
            # Only consumed on reset steps: reset_idx refills it exclusively for
            # the envs that just reset, and it stays stale between resets.
            # Once an inactive env finishes a later replay, the aggregate is
            # no longer attributable to first-episode scenarios, so skip it.
            if (
                torch.equal(all_done_mask, done_mask)
                and "episode" in infos
                and infos["episode"]
            ):
                weight = float(done_mask.sum())
                for key, value in infos["episode"].items():
                    if value is not None:
                        extras_metric_sums[key] = (
                            extras_metric_sums.get(key, 0.0)
                            + float(value.mean().detach().cpu()) * weight
                        )
                extras_metric_weight += weight

    completed_returns = scenario_return.detach().cpu().tolist()
    completed_lengths = scenario_steps.detach().cpu().tolist()
    steps = executed_steps
    num_episodes = num_envs
    motion_end_count = reason_counts[REASON_MOTION_END]
    sample_denominator = max(metric_samples, 1)
    action_rate_denominator = max(action_rate_samples, 1)
    scenario_outcomes = [
        {
            "scenario_id": env_id,
            "motion_id": scenario_manifest["scenarios"][env_id]["motion_id"],
            "start_time_s": scenario_manifest["scenarios"][env_id]["start_time_s"],
            "return": completed_returns[env_id],
            "length_frames": completed_lengths[env_id],
            "terminated": bool(scenario_terminated[env_id].item()),
            "termination_reason": scenario_reasons[env_id],
        }
        for env_id in range(num_envs)
    ]
    results = {
        "meta": {
            "task": args.task,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_md5": checkpoint_md5(args.checkpoint),
            "branch_mode": args.branch_mode,
            "branch_mask": {
                "base_only": "0",
                "full": "delta_dyn + delta_err",
                "dyn_only": "delta_dyn",
                "err_only": "delta_err",
                "full_gate_off": "(delta_dyn + delta_err), gate=1",
                "full_demand_only": "(delta_dyn + delta_err), gate=D",
                "full_demand_confidence": "(delta_dyn + delta_err), gate=D*C",
                "full_gate": "(delta_dyn + delta_err), gate=D*C*S",
            }[args.branch_mode],
            "adapter_gain": float(actor_critic.adapter_gain),
            "dynamics_branch_gain": float(actor_critic.dynamics_branch_gain),
            "tracking_branch_gain": float(actor_critic.tracking_branch_gain),
            "use_independent_branch_gates": bool(getattr(
                actor_critic, "use_independent_branch_gates", False
            )),
            "dynamics_gate_scale": float(getattr(
                actor_critic, "dynamics_gate_scale", 1.0
            )),
            "tracking_gate_scale": float(getattr(
                actor_critic, "tracking_gate_scale", 1.0
            )),
            "tracking_demand_mode": getattr(
                actor_critic, "tracking_demand_mode", None
            ),
            "tracking_demand_low": float(getattr(
                actor_critic, "tracking_demand_low", 0.0
            )),
            "tracking_demand_high": float(getattr(
                actor_critic, "tracking_demand_high", 1.0
            )),
            "dynamics_demand_low": float(getattr(
                actor_critic, "dynamics_demand_low", 0.0
            )),
            "dynamics_demand_high": float(getattr(
                actor_critic, "dynamics_demand_high", 1.0
            )),
            "seed": args.seed,
            "num_envs": num_envs,
            "max_steps": args.max_steps,
            "executed_steps": steps,
            "dt": float(env.dt),
            "episode_length_s": float(env.cfg.env.episode_length_s),
            "domain_rand_enabled": bool(env.cfg.domain_rand.domain_rand_general),
            "motion_file": str(env.cfg.motion.motion_file),
            "scenario_manifest": str(scenario_manifest_path.resolve()),
            "scenario_manifest_sha256": scenario_manifest_sha256,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
        "performance": {
            "mean_reward_per_step": mean_rew / sample_denominator,
            "num_episodes": num_episodes,
            "mean_episode_return": (sum(completed_returns) / num_episodes) if num_episodes else None,
            "std_episode_return": float(torch.tensor(completed_returns).float().std()) if num_episodes > 1 else None,
            "mean_episode_length_s": (sum(completed_lengths) / num_episodes * env.dt) if num_episodes else None,
            "episode_return": completed_returns,
            "episode_length_frames": completed_lengths,
            "fixed_scenario_outcomes": scenario_outcomes,
        },
        "tracking": {
            "mean_root_pos_error_m": mean_root_pos_err / sample_denominator,
            "mean_roll_error_rad": mean_roll_err / sample_denominator,
            "mean_pitch_error_rad": mean_pitch_err / sample_denominator,
            "mean_joint_pos_error_rad": mean_joint_err / sample_denominator,
            "mean_joint_vel_error_rad_s": mean_joint_vel_err / sample_denominator,
            "mean_root_vel_error_m_s": mean_root_vel_err / sample_denominator,
            "mean_keybody_pos_error_m": mean_keybody_err / sample_denominator,
            "root_position_rmse_m": float(
                rmse_from_mean_square(root_pos_mse / sample_denominator)
            ),
            "root_velocity_rmse_m_s": float(
                rmse_from_mean_square(root_vel_mse / sample_denominator)
            ),
            "roll_rmse_rad": float(
                rmse_from_mean_square(roll_mse / sample_denominator)
            ),
            "pitch_rmse_rad": float(
                rmse_from_mean_square(pitch_mse / sample_denominator)
            ),
            "joint_position_rmse_rad": float(
                rmse_from_mean_square(joint_pos_mse / sample_denominator)
            ),
            "joint_velocity_rmse_rad_s": float(
                rmse_from_mean_square(joint_vel_mse / sample_denominator)
            ),
            "keybody_rmse_m": float(
                rmse_from_mean_square(keybody_mse / sample_denominator)
            ),
            # Reward-component metrics from the env's own episode logger.
            "env_episode_reward_components": {
                key: value / extras_metric_weight if extras_metric_weight > 0 else None
                for key, value in extras_metric_sums.items()
            },
        },
        "stability": {
            "termination_reason_counts": reason_counts,
            "fall_count": fall_count,
            "timeout_count": timeout_count,
            "motion_end_count": motion_end_count,
            "fall_rate_per_episode": (fall_count / num_episodes) if num_episodes else None,
        },
        "residual": {
            "mean_delta_dyn_l2": mean_dyn_res / sample_denominator,
            "mean_delta_err_l2": mean_err_res / sample_denominator,
            "mean_applied_residual_l2": mean_applied_res / sample_denominator,
            "max_applied_residual_l2": max_applied_res,
            "mean_gate": mean_gate / sample_denominator,
            "mean_dynamics_gate": mean_dynamics_gate / sample_denominator,
            "mean_tracking_gate": mean_tracking_gate / sample_denominator,
        },
        "control": {
            "mean_feet_slip_m_s": mean_feet_slip / sample_denominator,
            "mean_action_rate_l2": mean_action_rate / action_rate_denominator,
        },
        "startup_first_50_steps": startup,
    }

    result_name = args.result_name or args.branch_mode
    if Path(result_name).name != result_name or result_name in ("", ".", ".."):
        raise ValueError("result_name must be a plain filename stem")
    json_path = out_dir / f"{result_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    flat = {
        "branch_mode": args.branch_mode,
        "checkpoint": results["meta"]["checkpoint"],
        "seed": args.seed,
        "num_envs": num_envs,
        "max_steps": args.max_steps,
        "mean_reward_per_step": results["performance"]["mean_reward_per_step"],
        "mean_episode_return": results["performance"]["mean_episode_return"],
        "mean_episode_length_s": results["performance"]["mean_episode_length_s"],
        "mean_root_pos_error_m": results["tracking"]["mean_root_pos_error_m"],
        "mean_roll_error_rad": results["tracking"]["mean_roll_error_rad"],
        "mean_pitch_error_rad": results["tracking"]["mean_pitch_error_rad"],
        "mean_joint_pos_error_rad": results["tracking"]["mean_joint_pos_error_rad"],
        "mean_joint_vel_error_rad_s": results["tracking"]["mean_joint_vel_error_rad_s"],
        "mean_root_vel_error_m_s": results["tracking"]["mean_root_vel_error_m_s"],
        "mean_keybody_pos_error_m": results["tracking"]["mean_keybody_pos_error_m"],
        "root_position_rmse_m": results["tracking"]["root_position_rmse_m"],
        "root_velocity_rmse_m_s": results["tracking"]["root_velocity_rmse_m_s"],
        "roll_rmse_rad": results["tracking"]["roll_rmse_rad"],
        "pitch_rmse_rad": results["tracking"]["pitch_rmse_rad"],
        "joint_position_rmse_rad": results["tracking"]["joint_position_rmse_rad"],
        "joint_velocity_rmse_rad_s": results["tracking"]["joint_velocity_rmse_rad_s"],
        "keybody_rmse_m": results["tracking"]["keybody_rmse_m"],
        "fall_count": fall_count,
        "timeout_count": timeout_count,
        "motion_end_count": motion_end_count,
        "mean_delta_dyn_l2": results["residual"]["mean_delta_dyn_l2"],
        "mean_delta_err_l2": results["residual"]["mean_delta_err_l2"],
        "mean_applied_residual_l2": results["residual"]["mean_applied_residual_l2"],
        "max_applied_residual_l2": results["residual"]["max_applied_residual_l2"],
        "mean_gate": results["residual"]["mean_gate"],
        "mean_dynamics_gate": results["residual"]["mean_dynamics_gate"],
        "mean_tracking_gate": results["residual"]["mean_tracking_gate"],
        "mean_feet_slip_m_s": results["control"]["mean_feet_slip_m_s"],
        "mean_action_rate_l2": results["control"]["mean_action_rate_l2"],
    }
    for label in reason_counts:
        flat[f"termination_{label}"] = reason_counts[label]

    csv_path = out_dir / f"{result_name}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)

    print(f"[evaluate_dual_branch] mode={args.branch_mode} "
          f"episodes={num_episodes} falls={fall_count} timeouts={timeout_count} "
          f"mean_episode_return={results['performance']['mean_episode_return']}")
    print(f"[evaluate_dual_branch] results -> {json_path}")
    print(f"[evaluate_dual_branch] results -> {csv_path}")


if __name__ == "__main__":
    main()
