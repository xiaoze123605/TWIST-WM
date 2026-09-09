"""Minimal script to debug teacher recording. Runs 1 motion and logs key metrics."""
import os, sys, copy, pickle, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../rsl_rl")))

from isaacgym import gymapi
import torch
from tqdm import tqdm

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.gym_utils import task_registry
from rsl_rl.modules.actor_critic_mimic import ActorCriticMimic


def main():
    device = "cuda:0"
    teacher_exptid = "0529_twist_teacher"
    teacher_checkpoint = 35000
    motion_config = os.path.join(LEGGED_GYM_ROOT_DIR, "motion_data_configs/twist_dataset.yaml")
    mid = 0

    # 1. Create env (standard training config)
    env_cfg, train_cfg = task_registry.get_cfgs(name="g1_priv_mimic")
    env_cfg = copy.deepcopy(env_cfg)
    env_cfg.env.num_envs = 1
    env_cfg.env.obs_type = "priv"
    env_cfg.env.rand_reset = False
    env_cfg.env.pose_termination = False
    env_cfg.env.enable_early_termination = False
    env_cfg.env.episode_length_s = 120
    env_cfg.motion.motion_file = motion_config
    env_cfg.motion.motion_curriculum = False
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.domain_rand_general = True  # keep True to match training obs format
    env_cfg.domain_rand.randomize_friction = True   # needed by compute_observations
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = True  # needed by compute_observations
    env_cfg.domain_rand.randomize_base_com = True   # needed by compute_observations
    env_cfg.domain_rand.randomize_motor = True      # needed by compute_observations
    env_cfg.domain_rand.action_delay = False

    args = argparse.Namespace()
    args.task = "g1_priv_mimic"; args.proj_name = "g1_priv_mimic"; args.exptid = "test"
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
    print(f"Env created. num_envs={env.num_envs}, num_obs={env.num_obs}, num_actions={env.num_actions}")

    # 2. Load teacher
    ckpt_path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "g1_priv_mimic", teacher_exptid, f"model_{teacher_checkpoint}.pt")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    num_motion_steps = len(env_cfg.env.tar_obs_steps)
    teacher = ActorCriticMimic(
        num_observations=env.num_obs,
        num_critic_observations=env.num_privileged_obs,
        num_motion_observations=env_cfg.env.n_priv_mimic_obs,
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
    print(f"Teacher loaded. normalizer type: {type(normalizer).__name__}")

    # 3. Reset to specific motion
    motion_lib = env._motion_lib
    motion_len = motion_lib.get_motion_length(torch.tensor([mid], device=device)).item()
    dt = env_cfg.sim.dt * env_cfg.control.decimation
    num_steps = min(int(motion_len / dt), 500)
    print(f"Motion {mid}: len={motion_len:.1f}s, steps={num_steps}, dt={dt}")

    env._motion_ids[:] = mid
    env._motion_time_offsets[:] = 0.0
    env.reset_idx(torch.tensor([0], device=device), motion_ids=torch.tensor([mid], device=device))

    # CRITICAL: reset_idx doesn't call compute_observations or compute roll/pitch.
    # We must refresh tensors and compute derived quantities before getting observations.
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_net_contact_force_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    env.gym.refresh_force_sensor_tensor(env.sim)
    from isaacgym.torch_utils import quat_rotate_inverse
    env.base_quat[:] = env.root_states[:, 3:7]
    env.base_lin_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 7:10])
    env.base_ang_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 10:13])
    env.projected_gravity[:] = quat_rotate_inverse(env.base_quat, env.gravity_vec)
    from legged_gym.envs.base.legged_robot import euler_from_quaternion
    env.roll, env.pitch, env.yaw = euler_from_quaternion(env.base_quat)
    env.compute_observations()

    print("After reset_idx:")
    print(f"  reset_buf={env.reset_buf[0].item()}")
    print(f"  root_z={env.root_states[0,2].item():.4f}")
    print(f"  ref_root_z={env._ref_root_pos[0,2].item():.4f}")
    print(f"  roll={env.roll[0].item():.4f} pitch={env.pitch[0].item():.4f}")
    print(f"  contact_max={env.contact_forces[0,:,2].max().item():.2f}")

    # 4. Run teacher
    obs = env.get_observations()
    print(f"\nInitial obs: min={obs.min().item():.3f} max={obs.max().item():.3f} mean={obs.mean().item():.3f}")
    norm_obs = normalizer.normalize(obs.detach())
    print(f"Initial obs norm: min={norm_obs.min().item():.3f} max={norm_obs.max().item():.3f}")

    normalizer_obj = normalizer
    survived = 0
    for t in range(num_steps):
        norm_obs = normalizer_obj.normalize(obs.detach())
        with torch.no_grad():
            action = teacher.act_inference(norm_obs)
        obs, _, rew, done, _ = env.step(action.detach())

        if t == 0:
            print(f"\nAfter step 0:")
            print(f"  action norm={action.norm().item():.3f}")
            print(f"  reset_buf={env.reset_buf[0].item()}")
            print(f"  root_z={env.root_states[0,2].item():.4f}")
            print(f"  roll={env.roll[0].item():.4f} pitch={env.pitch[0].item():.4f}")

        if env.reset_buf[0].item():
            print(f"\nTerminated at step {t}!")
            print(f"  root_z={env.root_states[0,2].item():.4f}")
            print(f"  ref_root_z={env._ref_root_pos[0,2].item():.4f}")
            print(f"  roll={env.roll[0].item():.4f} pitch={env.pitch[0].item():.4f}")
            print(f"  contact_max={env.contact_forces[0,:,2].max().item():.2f}")
            break
        survived += 1

    print(f"\nSurvived {survived}/{num_steps} steps ({survived*dt:.1f}s)")

if __name__ == "__main__":
    main()
