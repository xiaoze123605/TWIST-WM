"""
Parallel batched version of generate_cleaned_dataset.
Processes motions in batches across multiple parallel envs for ~100x speedup.

Usage:
    python legged_gym/legged_gym/scripts/generate_cleaned_dataset_parallel.py \
        --teacher_exptid 0529_twist_teacher --teacher_checkpoint 35000 \
        --motion_config legged_gym/motion_data_configs/twist_dataset.yaml \
        --output_dir /home/hank/TWIST/track_dataset/twist_dataset_cleaned \
        --num_envs 1024 --device cuda:0
"""
import os, sys, copy, pickle, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../rsl_rl")))

from isaacgym import gymapi
import torch
from tqdm import tqdm
import numpy as np

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.gym_utils import task_registry
from pose.utils.motion_lib_pkl import smooth
from pose.utils.torch_utils import quat_diff, quat_to_exp_map
from isaacgym.torch_utils import quat_rotate_inverse
from legged_gym.envs.base.legged_robot import euler_from_quaternion
from legged_gym.envs.base.humanoid_char import compute_local_body_pos


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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_exptid", type=str, required=True)
    p.add_argument("--teacher_checkpoint", type=int, required=True)
    p.add_argument("--motion_config", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--num_envs", type=int, default=1024)
    p.add_argument("--start_batch", type=int, default=0)
    p.add_argument("--num_batches", type=int, default=None)
    return p.parse_args()


def make_parallel_env(motion_config_path, num_envs, device):
    env_cfg, train_cfg = task_registry.get_cfgs(name="g1_priv_mimic")
    env_cfg = copy.deepcopy(env_cfg)
    env_cfg.env.num_envs = num_envs
    env_cfg.env.obs_type = "priv"
    env_cfg.env.rand_reset = False
    env_cfg.env.pose_termination = False
    env_cfg.env.enable_early_termination = False
    env_cfg.env.episode_length_s = 120
    env_cfg.motion.motion_file = motion_config_path
    env_cfg.motion.motion_curriculum = False
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.domain_rand_general = True
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = True
    env_cfg.domain_rand.randomize_base_com = True
    env_cfg.domain_rand.randomize_motor = True
    env_cfg.domain_rand.action_delay = False

    args = argparse.Namespace()
    args.task = "g1_priv_mimic"; args.proj_name = "g1_priv_mimic"; args.exptid = "rec"
    args.device = device; args.rl_device = device
    args.resume = False; args.headless = True
    args.physics_engine = gymapi.SIM_PHYSX
    args.sim_device_type = device; args.sim_device = device
    args.compute_device_id = 0; args.sim_device_id = 0
    args.use_gpu = True; args.use_gpu_pipeline = True
    args.subscenes = 0; args.num_threads = 0
    args.num_envs = None; args.seed = 1; args.motion_file = None
    args.rows = None; args.cols = None; args.record_video = False
    args.no_rand = False; args.teleop_mode = False; args.horovod = False
    args.no_wandb = False; args.checkpoint = -1; args.debug = False; args.web = False
    args.run_name = ""; args.experiment_name = ""; args.load_run = ""
    args.max_iterations = None; args.resumeid = None; args.fix_action_std = False
    env, _ = task_registry.make_env(name="g1_priv_mimic", args=args, env_cfg=env_cfg)
    return env, train_cfg


def load_teacher(env, train_cfg, teacher_exptid, teacher_checkpoint, device):
    from rsl_rl.modules.actor_critic_mimic import ActorCriticMimic
    ckpt_path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "g1_priv_mimic",
                             teacher_exptid, f"model_{teacher_checkpoint}.pt")
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
    normalizer = checkpoint["normalizer"]
    if normalizer is None:
        raise ValueError("No normalizer in checkpoint")
    return teacher, normalizer


def post_reset_refresh(env):
    """Refresh tensors and compute derived quantities after reset_idx."""
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


def save_pkl(output_path, frames_list, sim_body_names, sim_to_orig_idx, fps=50, device="cuda:0"):
    """Convert recorded frames to pkl and save."""
    T = len(frames_list)
    if T < 2:
        return

    root_pos = torch.cat([f[0].cpu() for f in frames_list], dim=0)
    root_rot = torch.cat([f[1].cpu() for f in frames_list], dim=0)
    dof_pos = torch.cat([f[2].cpu() for f in frames_list], dim=0)
    sim_body_pos = torch.cat([f[3].cpu() for f in frames_list], dim=0)

    # Remap 51 → 38 bodies
    local_body_pos_38 = torch.zeros(T, 38, 3)
    for orig_i in range(38):
        sim_i = sim_to_orig_idx[orig_i]
        if sim_i >= 0:
            local_body_pos_38[:, orig_i, :] = sim_body_pos[:, sim_i, :]

    # Compute velocities
    box_pts = min(19, T)
    root_pos_s = smooth(root_pos, box_pts, torch.device('cpu'))
    root_vel = torch.zeros_like(root_pos)
    root_vel[1:] = fps * (root_pos_s[1:] - root_pos_s[:-1])
    root_vel = smooth(root_vel, box_pts, torch.device('cpu'))

    root_ang_vel = torch.zeros_like(root_pos)
    for t_idx in range(1, T):
        dq = quat_diff(root_rot[t_idx - 1:t_idx], root_rot[t_idx:t_idx + 1])
        root_ang_vel[t_idx] = quat_to_exp_map(dq) * fps
    root_ang_vel = smooth(root_ang_vel, box_pts, torch.device('cpu'))

    dof_pos_s = smooth(dof_pos, box_pts, torch.device('cpu'))
    dof_vel = torch.zeros_like(dof_pos)
    dof_vel[1:] = fps * (dof_pos_s[1:] - dof_pos_s[:-1])
    dof_vel = smooth(dof_vel, box_pts, torch.device('cpu'))

    pkl_data = {
        "fps": float(fps),
        "root_pos": root_pos.numpy(),
        "root_rot": root_rot.numpy(),
        "dof_pos": dof_pos.numpy(),
        "local_body_pos": local_body_pos_38.numpy(),
        "link_body_list": list(_ORIG_38_BODY_NAMES),
    }
    with open(output_path, "wb") as f:
        pickle.dump(pkl_data, f)


def read_original_weights(motion_config_path):
    """Read original motion weights from yaml in order."""
    import yaml
    with open(motion_config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    weights = []
    for entry in config["motions"]:
        weights.append(entry["weight"])
    return weights


def generate_yaml(output_dir, num_motions, original_weights):
    yaml_path = os.path.join(LEGGED_GYM_ROOT_DIR, "motion_data_configs", "twist_dataset_cleaned.yaml")
    lines = [f"root_path: {output_dir}", "motions:"]
    for mid in range(num_motions):
        p = os.path.join(output_dir, f"cleaned_{mid}.pkl")
        if os.path.exists(p):
            w = original_weights[mid] if mid < len(original_weights) else 150.0
            lines.append(f"  - file: cleaned_{mid}.pkl")
            lines.append(f"    weight: {w}")
            lines.append(f"    description: cleaned (teacher rollout)")
    with open(yaml_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Yaml: {yaml_path}")


def main():
    args = parse_args()
    device = args.device
    N = args.num_envs

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Creating env with {N} parallel environments...")
    env, train_cfg = make_parallel_env(args.motion_config, N, device)
    teacher, normalizer = load_teacher(env, train_cfg, args.teacher_exptid,
                                       args.teacher_checkpoint, device)

    sim_body_names = env.body_names[1:]
    sim_to_orig_idx = [sim_body_names.index(name) if name in sim_body_names else -1
                       for name in _ORIG_38_BODY_NAMES]

    motion_lib = env._motion_lib
    total_motions = motion_lib.num_motions()
    dt = env.cfg.sim.dt * env.cfg.control.decimation
    fps = 50

    num_batches = (total_motions + N - 1) // N
    start_b = args.start_batch
    end_b = min(start_b + args.num_batches, num_batches) if args.num_batches else num_batches

    print(f"Total motions: {total_motions}, batch size: {N}, batches: {num_batches}")
    print(f"Processing batches [{start_b}, {end_b}), ETA: ~{(end_b-start_b)*30/3600:.1f}h")

    for batch in range(start_b, end_b):
        start_mid = batch * N
        end_mid = min(start_mid + N, total_motions)
        batch_motions = list(range(start_mid, end_mid))
        batch_N = len(batch_motions)

        # Determine max steps for this batch
        max_steps = 0
        for mid in batch_motions:
            mlen = motion_lib.get_motion_length(torch.tensor([mid], device=device)).item()
            nsteps = int(mlen / dt)
            if nsteps > max_steps:
                max_steps = nsteps

        # Reset each env to its assigned motion
        env_ids = torch.arange(batch_N, device=device)
        motion_ids = torch.tensor(batch_motions, device=device)
        env.reset_idx(env_ids, motion_ids=motion_ids)
        post_reset_refresh(env)

        # Initialize per-env recording
        # Each element: list of (root_pos, root_rot, dof_pos, local_body_pos) tensors on GPU
        recordings = [[] for _ in range(batch_N)]
        active = torch.ones(batch_N, dtype=torch.bool, device=device)

        # Record initial state
        for i in range(batch_N):
            bp = env.rigid_body_states[i:i+1, 1:, :3].clone()
            recordings[i].append((
                env.root_states[i:i+1, :3].clone(),
                env.root_states[i:i+1, 3:7].clone(),
                env.dof_pos[i:i+1].clone(),
                bp,
            ))

        obs = env.get_observations()

        for t in range(max_steps):
            norm_obs = normalizer.normalize(obs.detach())
            with torch.no_grad():
                actions = teacher.act_inference(norm_obs)
            obs, _, _, _, _ = env.step(actions.detach())

            # Record state for active envs
            for i in range(batch_N):
                if active[i]:
                    # Check if this env's motion has ended (time >= motion_len)
                    mlen = motion_lib.get_motion_length(torch.tensor([batch_motions[i]], device=device))
                    cur_time = (t + 2) * dt  # +2 because initial frame was step -1
                    if cur_time > mlen.item():
                        active[i] = False
                    else:
                        bp = env.rigid_body_states[i:i+1, 1:, :3].clone()
                        recordings[i].append((
                            env.root_states[i:i+1, :3].clone(),
                            env.root_states[i:i+1, 3:7].clone(),
                            env.dof_pos[i:i+1].clone(),
                            bp,
                        ))

        # Save pkl for each env in batch
        for i, mid in enumerate(batch_motions):
            frames = recordings[i]
            if len(frames) < 2:
                print(f"  Skipping motion {mid}: {len(frames)} frames")
                continue
            output_path = os.path.join(args.output_dir, f"cleaned_{mid}.pkl")
            save_pkl(output_path, frames, sim_body_names, sim_to_orig_idx, fps)
            if batch % 10 == 0 or i == 0:
                print(f"  Batch {batch} motion {mid}: {len(frames)} frames saved")

        print(f"Batch {batch}/{num_batches} done ({batch_N} motions, {max_steps} steps)")

    original_weights = read_original_weights(args.motion_config)
    generate_yaml(args.output_dir, total_motions, original_weights)
    print("Done.")


if __name__ == "__main__":
    main()
