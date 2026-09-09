"""
Generate physically-consistent motion dataset using a trained teacher policy.

Usage:
    python legged_gym/legged_gym/scripts/generate_cleaned_dataset.py \
        --teacher_exptid 0523_twist_teacher \
        --teacher_checkpoint 22500 \
        --motion_config legged_gym/motion_data_configs/twist_dataset.yaml \
        --output_dir /home/hank/TWIST/track_dataset/twist_dataset_cleaned \
        --device cuda:0
"""

import os
import sys
import copy
import pickle
import argparse
from tqdm import tqdm

# Ensure TWIST root and rsl_rl sub-package are importable
_TWIST_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
for _p in (_TWIST_ROOT, os.path.join(_TWIST_ROOT, "rsl_rl")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from isaacgym import gymapi
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.gym_utils import task_registry
from pose.utils.motion_lib_pkl import smooth
from pose.utils.torch_utils import quat_diff, quat_to_exp_map


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_exptid", type=str, required=True)
    parser.add_argument("--teacher_checkpoint", type=int, required=True)
    parser.add_argument("--motion_config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--motion_ids", type=str, default=None,
                        help="Comma-separated motion IDs to process (default: all)")
    parser.add_argument("--max_motions", type=int, default=None,
                        help="Max number of motions to process (for testing)")
    return parser.parse_args()


def make_recording_env(motion_config_path, device):
    """Create a 1-env teacher environment with domain randomization disabled."""
    env_cfg, train_cfg = task_registry.get_cfgs(name="g1_priv_mimic")
    env_cfg = copy.deepcopy(env_cfg)
    env_cfg.env.num_envs = 1
    env_cfg.env.obs_type = "priv"
    env_cfg.env.rand_reset = False
    env_cfg.env.pose_termination = False
    env_cfg.env.episode_length_s = 120
    env_cfg.motion.motion_file = motion_config_path
    env_cfg.motion.motion_curriculum = False
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.domain_rand_general = True  # must match training obs format
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = True
    env_cfg.domain_rand.randomize_base_com = True
    env_cfg.domain_rand.randomize_motor = True
    env_cfg.domain_rand.action_delay = False

    args = argparse.Namespace()
    args.task = "g1_priv_mimic"
    args.proj_name = "g1_priv_mimic"
    args.exptid = "recording"
    args.device = device
    args.rl_device = device
    args.resume = False
    args.headless = True
    args.physics_engine = gymapi.SIM_PHYSX
    args.sim_device_type = device
    args.sim_device = device
    args.compute_device_id = 0
    args.sim_device_id = 0
    args.use_gpu = True
    args.use_gpu_pipeline = True
    args.subscenes = 0
    args.num_threads = 0
    args.num_envs = None
    args.seed = None
    args.rows = None
    args.cols = None
    args.record_video = False
    args.no_rand = False
    args.teleop_mode = False
    args.horovod = False
    args.no_wandb = False
    args.checkpoint = -1
    args.debug = False
    args.web = False
    args.run_name = ""
    args.experiment_name = ""
    args.load_run = ""
    args.max_iterations = None
    args.resumeid = None
    args.fix_action_std = False
    env, _ = task_registry.make_env(name="g1_priv_mimic", args=args, env_cfg=env_cfg)
    return env, train_cfg


# Original 38-body names from the motion dataset (must match this exact order)
_ORIG_38_BODY_NAMES = [
    "pelvis", "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
    "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link", "left_toe_link",
    "pelvis_contour_link", "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
    "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link", "right_toe_link",
    "waist_yaw_link", "waist_roll_link", "torso_link", "head_link", "head_mocap", "imu_in_torso",
    "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link",
    "left_elbow_link", "left_wrist_roll_link", "left_wrist_pitch_link", "left_wrist_yaw_link",
    "left_rubber_hand", "right_shoulder_pitch_link", "right_shoulder_roll_link",
    "right_shoulder_yaw_link", "right_elbow_link", "right_wrist_roll_link",
    "right_wrist_pitch_link", "right_wrist_yaw_link", "right_rubber_hand",
]


def build_sim_to_orig_38_map(sim_body_names):
    """Build a mapping: for each of the 38 original bodies, find its index in sim_body_names.
    Returns (indices_tensor, valid_mask) where indices_tensor[i] = sim index for orig body i,
    and valid_mask[i] = True if the body exists in sim.
    """
    sim_name_to_idx = {name: i for i, name in enumerate(sim_body_names)}
    indices = []
    valid = []
    for orig_name in _ORIG_38_BODY_NAMES:
        if orig_name in sim_name_to_idx:
            indices.append(sim_name_to_idx[orig_name])
            valid.append(True)
        else:
            indices.append(-1)
            valid.append(False)
    return indices, valid


def load_teacher(env, train_cfg, teacher_exptid, teacher_checkpoint, device):
    """Load trained teacher policy from checkpoint."""
    from rsl_rl.modules.actor_critic_mimic import ActorCriticMimic

    ckpt_path = os.path.join(
        LEGGED_GYM_ROOT_DIR, "logs", "g1_priv_mimic",
        teacher_exptid, f"model_{teacher_checkpoint}.pt",
    )
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Teacher checkpoint not found: {ckpt_path}")

    print(f"Loading teacher from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    num_motion_steps = len(env.cfg.env.tar_obs_steps)
    teacher = ActorCriticMimic(
        num_observations=env.num_obs,
        num_critic_observations=env.num_privileged_obs,
        num_motion_observations=env.cfg.env.n_priv_mimic_obs,
        num_motion_steps=num_motion_steps,
        num_actions=env.num_actions,
        actor_hidden_dims=train_cfg.policy.actor_hidden_dims,
        critic_hidden_dims=train_cfg.policy.critic_hidden_dims,
        activation=train_cfg.policy.activation,
        init_noise_std=train_cfg.policy.init_noise_std,
        motion_latent_dim=train_cfg.policy.motion_latent_dim,
        layer_norm=train_cfg.policy.layer_norm,
        tanh_encoder_output=False,
    ).to(device)

    teacher.load_state_dict(checkpoint["model_state_dict"])
    teacher.eval()

    normalizer = checkpoint.get("normalizer", None)
    if normalizer is None:
        raise ValueError("Teacher checkpoint has no normalizer — was it trained with normalize_obs=True?")
    return teacher, normalizer


def generate_cleaned_dataset(env, teacher, normalizer, output_dir, motion_ids=None, device="cuda:0"):
    """Run teacher policy rollouts for each motion and save physically-consistent trajectories."""
    os.makedirs(output_dir, exist_ok=True)

    motion_lib = env._motion_lib
    num_motions = motion_lib.num_motions()
    if motion_ids is None:
        motion_ids = list(range(num_motions))
    else:
        motion_ids = [int(x) for x in motion_ids.split(",")]

    sim_body_names = env.body_names[1:]  # skip "world" at index 0
    sim_to_orig_idx, _ = build_sim_to_orig_38_map(sim_body_names)
    fps = 50
    dt = env.cfg.sim.dt * env.cfg.control.decimation  # action dt = 0.02s

    from legged_gym.envs.base.humanoid_char import compute_local_body_pos

    from isaacgym.torch_utils import quat_rotate_inverse
    from legged_gym.envs.base.legged_robot import euler_from_quaternion

    for mid in tqdm(motion_ids, desc="Generating cleaned motions"):
        motion_len = motion_lib.get_motion_length(torch.tensor([mid], device=device)).item()
        num_steps = int(motion_len / dt)

        # Use standard training reset with specific motion_id
        env.reset_idx(torch.tensor([0], device=device), motion_ids=torch.tensor([mid], device=device))

        # reset_idx doesn't call compute_observations; manually refresh & compute
        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.gym.refresh_dof_state_tensor(env.sim)
        env.gym.refresh_net_contact_force_tensor(env.sim)
        env.gym.refresh_rigid_body_state_tensor(env.sim)
        env.gym.refresh_force_sensor_tensor(env.sim)
        env.base_quat[:] = env.root_states[:, 3:7]
        env.base_lin_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 7:10])
        env.base_ang_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 10:13])
        env.projected_gravity[:] = quat_rotate_inverse(env.base_quat, env.gravity_vec)
        env.roll, env.pitch, env.yaw = euler_from_quaternion(env.base_quat)
        env.compute_observations()

        # Record frames manually
        rec_frames = []

        def _record_frame():
            body_pos_global = env.rigid_body_states[:, 1:, :3]
            local_body_pos = compute_local_body_pos(
                env.root_states[:, :3], env.root_states[:, 3:7], body_pos_global)
            rec_frames.append({
                'root_pos': env.root_states[:, :3].clone(),
                'root_rot': env.root_states[:, 3:7].clone(),
                'dof_pos': env.dof_pos.clone(),
                'local_body_pos': local_body_pos.clone(),
            })

        _record_frame()  # record initial state after reset
        obs = env.get_observations()

        for t in range(num_steps):
            norm_obs = normalizer.normalize(obs.detach())
            with torch.no_grad():
                actions = teacher.act_inference(norm_obs)
            obs, _, _, _, _ = env.step(actions.detach())
            _record_frame()

            if env.reset_buf[0].item():
                rec_frames.pop()  # remove fallen frame
                break

        frames = rec_frames
        if len(frames) < 2:
            print(f"  Skipping motion {mid}: too few frames ({len(frames)})")
            continue

        # Stack recorded frames
        root_pos = torch.cat([f["root_pos"] for f in frames], dim=0)  # (T, 3)
        root_rot = torch.cat([f["root_rot"] for f in frames], dim=0)  # (T, 4)
        dof_pos = torch.cat([f["dof_pos"] for f in frames], dim=0)    # (T, 23)

        # Remap 51 sim bodies → 38 original bodies
        sim_body_pos = torch.cat([f["local_body_pos"] for f in frames], dim=0)  # (T, 51, 3)
        local_body_pos_38 = torch.zeros(len(frames), 38, 3)
        for orig_i in range(38):
            sim_i = sim_to_orig_idx[orig_i]
            if sim_i >= 0:
                local_body_pos_38[:, orig_i, :] = sim_body_pos[:, sim_i, :]
            # else: "pelvis" stays at (0,0,0) which is correct — root IS pelvis
        local_body_pos = local_body_pos_38

        # Compute velocities from positions with smoothing
        box_pts = min(19, root_pos.shape[0])
        root_pos_smooth = smooth(root_pos, box_pts, device)
        root_vel = torch.zeros_like(root_pos)
        root_vel[1:] = fps * (root_pos_smooth[1:] - root_pos_smooth[:-1])
        root_vel = smooth(root_vel, box_pts, device)

        root_ang_vel = torch.zeros_like(root_pos)
        for t_idx in range(1, root_rot.shape[0]):
            dq = quat_diff(root_rot[t_idx - 1:t_idx], root_rot[t_idx:t_idx + 1])
            root_ang_vel[t_idx] = quat_to_exp_map(dq) * fps
        root_ang_vel = smooth(root_ang_vel, box_pts, device)

        dof_pos_smooth = smooth(dof_pos, box_pts, device)
        dof_vel = torch.zeros_like(dof_pos)
        dof_vel[1:] = fps * (dof_pos_smooth[1:] - dof_pos_smooth[:-1])
        dof_vel = smooth(dof_vel, box_pts, device)

        pkl_data = {
            "fps": float(fps),
            "root_pos": root_pos.cpu().numpy(),
            "root_rot": root_rot.cpu().numpy(),
            "dof_pos": dof_pos.cpu().numpy(),
            "local_body_pos": local_body_pos.cpu().numpy(),
            "link_body_list": list(_ORIG_38_BODY_NAMES),
        }

        output_path = os.path.join(output_dir, f"cleaned_{mid}.pkl")
        with open(output_path, "wb") as f:
            pickle.dump(pkl_data, f)
        print(f"  Saved motion {mid}: {len(frames)} frames to {output_path}")


def generate_cleaned_yaml(output_dir, motion_ids):
    """Generate a yaml config pointing to the cleaned dataset."""
    yaml_path = os.path.join(
        LEGGED_GYM_ROOT_DIR, "motion_data_configs", "twist_dataset_cleaned.yaml",
    )
    lines = [f"root_path: {output_dir}", "motions:"]
    for mid in motion_ids:
        pkl_path = os.path.join(output_dir, f"cleaned_{mid}.pkl")
        if os.path.exists(pkl_path):
            lines.append(f"  - file: cleaned_{mid}.pkl")
            lines.append(f"    weight: 150.0")
            lines.append(f"    description: cleaned general movement (teacher rollout)")
    with open(yaml_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Generated yaml config: {yaml_path}")


def main():
    args = parse_args()
    device = args.device

    print("=" * 60)
    print("Creating recording environment...")
    env, train_cfg = make_recording_env(args.motion_config, device)

    print("Loading teacher policy...")
    teacher, normalizer = load_teacher(env, train_cfg, args.teacher_exptid, args.teacher_checkpoint, device)

    # Clean old output
    if os.path.exists(args.output_dir):
        import shutil
        shutil.rmtree(args.output_dir)
        print(f"Cleaned old output: {args.output_dir}")

    print("Generating cleaned dataset...")
    # If max_motions, only process first N
    if args.max_motions is not None and args.motion_ids is None:
        motion_ids = ",".join(str(i) for i in range(args.max_motions))
    else:
        motion_ids = args.motion_ids
    generate_cleaned_dataset(
        env, teacher, normalizer,
        output_dir=args.output_dir,
        motion_ids=motion_ids,
        device=device,
    )

    # Count actual motions in motion lib
    motion_lib = env._motion_lib
    all_motion_ids = list(range(motion_lib.num_motions()))
    if args.motion_ids is not None:
        all_motion_ids = [int(x) for x in args.motion_ids.split(",")]
    generate_cleaned_yaml(args.output_dir, all_motion_ids)

    print("Done.")


if __name__ == "__main__":
    main()
