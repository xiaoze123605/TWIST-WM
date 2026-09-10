#!/usr/bin/env python
import argparse
import sys
import time
import redis
import json
import numpy as np
import isaacgym
import torch
from rich import print
import os
import mujoco
from mujoco.viewer import launch_passive

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
# ---------------------------------------------------------------------
# Example imports: adapt to your actual file structure
# ---------------------------------------------------------------------
from pose.utils.motion_lib_pkl import MotionLib
from data_utils.rot_utils import euler_from_quaternion, quat_rotate_inverse, quat_rotate_inverse_torch

from data_utils.params import DEFAULT_MIMIC_OBS, DEFAULT_ACTION_HAND
from motion_world_model.runtime import (
    MotionReferenceRefiner,
    RuntimeReferenceCorruptor,
    reinsert_wrist_roll,
    remove_wrist_roll,
    resolve_checkpoint_path,
    wrap_reference_yaw,
)


def _wrap_to_pi(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _quat_xyzw_from_euler(roll, pitch, yaw):
    half_roll = 0.5 * roll
    half_pitch = 0.5 * pitch
    half_yaw = 0.5 * yaw
    cr, sr = torch.cos(half_roll), torch.sin(half_roll)
    cp, sp = torch.cos(half_pitch), torch.sin(half_pitch)
    cy, sy = torch.cos(half_yaw), torch.sin(half_yaw)
    return torch.stack((
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ), dim=-1)


# ---------------------------------------------------------------------
# A small helper to replicate "mimic obs" logic from your code
# ---------------------------------------------------------------------
def build_mimic_obs(
    motion_lib: MotionLib,
    t_step: int,
    control_dt: float,
    tar_obs_steps,
    robot_type: str = "g1",
    motion_speed: float = 1.0,
    heading_offset: float = 0.0,
    hip_yaw_scale: float = 1.0,
):
    """
    Build the mimic_obs at time-step t_step, referencing the code in MimicRunner.
    """
    device = torch.device(motion_lib._device)
    # Build times
    motion_times = torch.tensor([t_step * control_dt * motion_speed], device=device).unsqueeze(-1)
    obs_motion_times = tar_obs_steps * control_dt * motion_speed + motion_times
    obs_motion_times = obs_motion_times.flatten()
    
    # Suppose we only have a single motion in the .pkl
    motion_ids = torch.zeros(len(tar_obs_steps), dtype=torch.int, device=device)
    
    # Retrieve motion frames
    root_pos, root_rot, root_vel, root_ang_vel, dof_pos, _, body_pos = motion_lib.calc_motion_frame(motion_ids, obs_motion_times)
    source_root_rot = root_rot

    # Convert to euler (roll, pitch, yaw)
    roll, pitch, yaw = euler_from_quaternion(root_rot)
    yaw = _wrap_to_pi(yaw - float(heading_offset))
    display_root_rot = _quat_xyzw_from_euler(roll, pitch, yaw)
    roll = roll.reshape(1, -1, 1)
    pitch = pitch.reshape(1, -1, 1)
    yaw = yaw.reshape(1, -1, 1)

    # Transform velocities to root frame
    root_vel = quat_rotate_inverse_torch(source_root_rot, root_vel).reshape(1, -1, 3)
    root_ang_vel = quat_rotate_inverse_torch(source_root_rot, root_ang_vel).reshape(1, -1, 3)
    root_vel = root_vel * motion_speed
    root_ang_vel = root_ang_vel * motion_speed

    root_pos = root_pos.reshape(1, -1, 3)
    dof_pos = dof_pos.reshape(1, -1, dof_pos.shape[-1])
    if robot_type == "g1" and hip_yaw_scale != 1.0:
        dof_pos = dof_pos.clone()
        dof_pos[..., [2, 8]] *= float(hip_yaw_scale)
    
    if robot_type == "g1":
        dof_pos_with_wrist = torch.zeros(25, device=device).reshape(1, 1, 25)
        wrist_ids = [19, 24]
        other_ids = [f for f in range(25) if f not in wrist_ids]
        dof_pos_with_wrist[..., other_ids] = dof_pos
        dof_pos = dof_pos_with_wrist
        
    mimic_obs_buf = torch.cat((
                root_pos[..., 2:3],
                roll, pitch, yaw,
                root_vel,
                root_ang_vel[..., 2:3],
                dof_pos
            ), dim=-1)[:, 0:1]  # shape (1, 1, ?)
    mimic_obs_buf = mimic_obs_buf.reshape(1, -1)
    
    return mimic_obs_buf.detach().cpu().numpy().squeeze(), root_pos.detach().cpu().numpy().squeeze(), \
        display_root_rot.detach().cpu().numpy().squeeze(), dof_pos.detach().cpu().numpy().squeeze(), \
            root_vel.detach().cpu().numpy().squeeze(), root_ang_vel.detach().cpu().numpy().squeeze()


def process_mimic_reference(mimic_obs, reference_mode, corruptor=None, refiner=None):
    """Apply the demo pipeline while leaving the two wrist-roll values untouched."""
    reference_31d, wrists = remove_wrist_roll(mimic_obs)
    if reference_mode == "clean":
        processed = reference_31d
    else:
        if corruptor is None:
            raise ValueError(f"{reference_mode} mode requires a corruptor")
        corrupted = corruptor.corrupt(reference_31d)
        if reference_mode == "corrupt":
            processed = wrap_reference_yaw(corrupted)
        elif reference_mode == "wm":
            if refiner is None:
                raise ValueError("wm mode requires a MotionReferenceRefiner")
            processed = refiner.refine(corrupted)
        else:
            raise ValueError(f"unknown reference mode: {reference_mode}")
    return reinsert_wrist_roll(processed, wrists)


def main(args, xml_file, robot_base):
    if args.motion_speed <= 0.0:
        raise ValueError("--motion-speed must be > 0")
    if not (0.0 <= args.motion_scale <= 1.0):
        raise ValueError("--motion-scale must be in [0, 1]")
    if not (0.0 <= args.hip_yaw_scale <= 1.0):
        raise ValueError("--hip-yaw-scale must be in [0, 1]")
    if args.sim_ready_timeout <= 0.0:
        raise ValueError("--sim-ready-timeout must be > 0")

    if args.vis:
        sim_model = mujoco.MjModel.from_xml_path(xml_file)
        sim_data = mujoco.MjData(sim_model)
        viewer = launch_passive(model=sim_model, data=sim_data, show_left_ui=False, show_right_ui=False)
        
        # Print DoF names in order
        print("Degrees of Freedom (DoF) names and their order:")
        for i in range(sim_model.nv):  # 'nv' is the number of DoFs
            dof_name = mujoco.mj_id2name(sim_model, mujoco.mjtObj.mjOBJ_JOINT, sim_model.dof_jntid[i])
            print(f"DoF {i}: {dof_name}")

        # print("Body names and their IDs:")
        # for i in range(self.model.nbody):  # 'nbody' is the number of bodies
        #     body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
        #     print(f"Body ID {i}: {body_name}")
        
        print("Motor (Actuator) names and their IDs:")
        for i in range(sim_model.nu):  # 'nu' is the number of actuators (motors)
            motor_name = mujoco.mj_id2name(sim_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            print(f"Motor ID {i}: {motor_name}")
            
    # 1. Connect to Redis
    redis_client = redis.Redis(host="localhost", port=6379, db=0)
    redis_client.ping()
    sim_ready_key = f"sim_ready_{args.robot}"
    if args.wait_for_sim_ready:
        redis_client.delete(sim_ready_key)
        redis_client.delete(f"action_mimic_frame_{args.robot}")

    # 2. Load motion library
    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else args.device
    print(f"[Motion Server] Torch device: {device}")
    motion_lib = MotionLib(args.motion_file, device=device)

    corruptor = None
    refiner = None
    if args.reference_mode != "clean":
        corruptor = RuntimeReferenceCorruptor.from_preset(
            args.corruption_preset, seed=args.seed
        )
    if args.reference_mode == "wm":
        checkpoint_path = resolve_checkpoint_path(args.wm_checkpoint)
        refiner = MotionReferenceRefiner(checkpoint_path, device=device)
        print(f"[Motion Server] Motion-WM checkpoint: {checkpoint_path}")
    print(
        f"[Motion Server] Reference mode: {args.reference_mode}; "
        f"corruption={args.corruption_preset}; seed={args.seed}"
    )

    initial_motion_id = torch.zeros(1, dtype=torch.long, device=device)
    initial_motion_time = torch.zeros(1, dtype=torch.float, device=device)
    initial_root_pos, initial_root_rot, *_ = motion_lib.calc_motion_frame(
        initial_motion_id,
        initial_motion_time,
    )
    _, _, initial_yaw = euler_from_quaternion(initial_root_rot)
    heading_offset = 0.0 if args.keep_absolute_heading else float(initial_yaw.item())
    initial_root_xy = initial_root_pos[0, :2].detach().cpu().numpy()
    heading_cos = np.cos(-heading_offset)
    heading_sin = np.sin(-heading_offset)
    heading_rotation = np.array([
        [heading_cos, -heading_sin],
        [heading_sin, heading_cos],
    ])
    heading_mode = "absolute" if args.keep_absolute_heading else "relative-to-first-frame"
    print(
        f"[Motion Server] Heading mode: {heading_mode}; "
        f"source initial yaw={float(initial_yaw.item()):.4f} rad"
    )
    motion_length = motion_lib.get_motion_length(initial_motion_id)
    final_time = torch.clamp(motion_length - 1e-4, min=0.0)
    final_root_pos, final_root_rot, *_ = motion_lib.calc_motion_frame(
        initial_motion_id,
        final_time,
    )
    _, _, final_yaw = euler_from_quaternion(final_root_rot)
    relative_yaw = _wrap_to_pi(final_yaw - initial_yaw)
    canonical_displacement = heading_rotation @ (
        final_root_pos[0, :2].detach().cpu().numpy() - initial_root_xy
    )
    print(
        "[Motion Server] Motion summary: "
        f"relative_yaw={float(relative_yaw.item()):.4f} rad, "
        f"canonical_xy=({canonical_displacement[0]:.3f}, "
        f"{canonical_displacement[1]:.3f}) m"
    )
    
    # 3. Prepare the steps array
    tar_obs_steps = [int(x.strip()) for x in args.steps.split(",")]
    tar_obs_steps_tensor = torch.tensor(tar_obs_steps, device=device, dtype=torch.int)

    # 4. Loop over time steps and publish mimic obs
    control_dt = 0.02
    # compute num_steps based on motion length
    motion_id = torch.tensor([0], device=device, dtype=torch.long)
    motion_length = motion_lib.get_motion_length(motion_id)
    num_steps = int(motion_length / (control_dt * args.motion_speed))

    if args.wait_for_sim_ready:
        print(f"[Motion Server] Waiting for {sim_ready_key}...")
        deadline = time.monotonic() + args.sim_ready_timeout
        while redis_client.get(sim_ready_key) is None:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"MuJoCo did not publish {sim_ready_key} within "
                    f"{args.sim_ready_timeout:.1f}s"
                )
            time.sleep(0.01)
        print("[Motion Server] MuJoCo ready; starting reference frame 0")
    
    print(
        f"[Motion Server] Streaming for {num_steps} steps at dt={control_dt:.3f} seconds, "
        f"motion_speed={args.motion_speed:.3f}, motion_scale={args.motion_scale:.3f}..."
    )

    last_mimic_obs = DEFAULT_MIMIC_OBS[args.robot]
    last_clean_mimic_obs = DEFAULT_MIMIC_OBS[args.robot].copy()
    vis_root_vel = False
    vis_root_ang_vel = False
    if vis_root_vel:
        root_vel_list = []
    if vis_root_ang_vel:
        root_ang_vel_list = []
        
    try:
        loop_idx = 0
        global_frame_id = 0
        while True:
            if args.loop:
                print(f"[Motion Server] Loop {loop_idx}...")
            for t_step in range(num_steps):
                t0 = time.time()
                if args.wait_for_sim_ready:
                    deadline = time.monotonic() + args.sim_ready_timeout
                    while int(redis_client.get(sim_ready_key) or -1) < global_frame_id:
                        if time.monotonic() > deadline:
                            raise TimeoutError('MuJoCo stopped requesting reference frames')
                        time.sleep(0.001)

                # Build a mimic obs from the motion library
                mimic_obs, root_pos, root_rot, dof_pos, root_vel, root_ang_vel = build_mimic_obs(
                    motion_lib=motion_lib,
                    t_step=t_step,
                    control_dt=control_dt,
                    tar_obs_steps=tar_obs_steps_tensor,
                    robot_type=args.robot,
                    motion_speed=args.motion_speed,
                    heading_offset=heading_offset,
                    hip_yaw_scale=args.hip_yaw_scale,
                )
                mimic_obs = DEFAULT_MIMIC_OBS[args.robot] + args.motion_scale * (
                    mimic_obs - DEFAULT_MIMIC_OBS[args.robot]
                )
                clean_mimic_obs = mimic_obs.copy()
                mimic_obs = process_mimic_reference(
                    mimic_obs,
                    args.reference_mode,
                    corruptor=corruptor,
                    refiner=refiner,
                )
                if vis_root_vel:
                    root_vel_list.append(root_vel)
                if vis_root_ang_vel:
                    root_ang_vel_list.append(root_ang_vel)

                # Convert to JSON (list) to put into Redis
                mimic_obs_list = mimic_obs.tolist() if mimic_obs.ndim == 1 else mimic_obs.flatten().tolist()
                redis_client.mset({
                    f"action_mimic_{args.robot}": json.dumps(mimic_obs_list),
                    f"action_mimic_clean_{args.robot}": json.dumps(clean_mimic_obs.tolist()),
                    f"action_mimic_frame_{args.robot}": global_frame_id,
                })
                redis_client.set(f"action_hand_{args.robot}", json.dumps(DEFAULT_ACTION_HAND[args.robot].tolist()))
                last_mimic_obs = mimic_obs
                last_clean_mimic_obs = clean_mimic_obs
                # Print or log it
                print(
                    f"Loop {loop_idx:3d} step {t_step:4d}/{num_steps} "
                    f"=> mimic_obs shape = {mimic_obs.shape} published...",
                    end="\r",
                )
                global_frame_id += 1

                if args.vis:
                    if not args.keep_absolute_heading:
                        root_pos = root_pos.copy()
                        root_pos[:2] = heading_rotation @ (root_pos[:2] - initial_root_xy)
                    sim_data.qpos[:3] = root_pos
                    # filp rot
                    # root_rot = root_rot[[1,2,3,0]]
                    root_rot = root_rot[[3,0,1,2]]
                    sim_data.qpos[3:7] = root_rot
                    sim_data.qpos[7:] = dof_pos
                    mujoco.mj_forward(sim_model, sim_data)
                    robot_base_pos = sim_data.xpos[sim_model.body(robot_base).id]
                    viewer.cam.lookat = robot_base_pos
                    # set distance to pelvis
                    viewer.cam.distance = 2.0
                    viewer.sync()
                    
                # Sleep to maintain real-time pace
                elapsed = time.time() - t0
                if elapsed < control_dt:
                    time.sleep(control_dt - elapsed)

            if not args.loop:
                break
            loop_idx += 1
            if args.loop_pause > 0.0:
                time.sleep(args.loop_pause)
        
    except KeyboardInterrupt:
        print("[Motion Server] Keyboard interrupt. Interpolating to default mimic_obs...")
        # do linear interpolation to the last mimic_obs
        time_back_to_default = 2.0
        for i in range(int(time_back_to_default / control_dt)):
            interp_mimic_obs = last_mimic_obs + (DEFAULT_MIMIC_OBS[args.robot] - last_mimic_obs) * (i / (time_back_to_default / control_dt))
            clean_interp = last_clean_mimic_obs + (DEFAULT_MIMIC_OBS[args.robot] - last_clean_mimic_obs) * (i / (time_back_to_default / control_dt))
            redis_client.mset({
                f"action_mimic_{args.robot}": json.dumps(interp_mimic_obs.tolist()),
                f"action_mimic_clean_{args.robot}": json.dumps(clean_interp.tolist()),
            })
            redis_client.set(f"action_hand_{args.robot}", json.dumps(DEFAULT_ACTION_HAND[args.robot].tolist()))
            time.sleep(control_dt)
        redis_client.mset({
            f"action_mimic_{args.robot}": json.dumps(DEFAULT_MIMIC_OBS[args.robot].tolist()),
            f"action_mimic_clean_{args.robot}": json.dumps(DEFAULT_MIMIC_OBS[args.robot].tolist()),
        })
        redis_client.set(f"action_hand_{args.robot}", json.dumps(DEFAULT_ACTION_HAND[args.robot].tolist()))
        last_mimic_obs = DEFAULT_MIMIC_OBS[args.robot]
        exit()
    finally:
        print("[Motion Server] Exiting...Interpolating to default mimic_obs...")
        # do linear interpolation to the last mimic_obs
        time_back_to_default = 2.0
        for i in range(int(time_back_to_default / control_dt)):
            interp_mimic_obs = last_mimic_obs + (DEFAULT_MIMIC_OBS[args.robot] - last_mimic_obs) * (i / (time_back_to_default / control_dt))
            clean_interp = last_clean_mimic_obs + (DEFAULT_MIMIC_OBS[args.robot] - last_clean_mimic_obs) * (i / (time_back_to_default / control_dt))
            redis_client.mset({
                f"action_mimic_{args.robot}": json.dumps(interp_mimic_obs.tolist()),
                f"action_mimic_clean_{args.robot}": json.dumps(clean_interp.tolist()),
            })
            redis_client.set(f"action_hand_{args.robot}", json.dumps(DEFAULT_ACTION_HAND[args.robot].tolist()))
            time.sleep(control_dt)
        redis_client.mset({
            f"action_mimic_{args.robot}": json.dumps(DEFAULT_MIMIC_OBS[args.robot].tolist()),
            f"action_mimic_clean_{args.robot}": json.dumps(DEFAULT_MIMIC_OBS[args.robot].tolist()),
        })
        redis_client.set(f"action_hand_{args.robot}", json.dumps(DEFAULT_ACTION_HAND[args.robot].tolist()))
        last_mimic_obs = DEFAULT_MIMIC_OBS[args.robot]
        exit()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_file", help="Path to your *.pkl motion file for MotionLib", 
                        default=os.path.join(REPO_ROOT, "track_dataset/twist_motion_dataset/accad/B3___walk1.pkl"))
    parser.add_argument("--robot", type=str, default="g1", choices=["g1"])
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for MotionLib: auto, cpu, cuda, or cuda:0",
    )
    parser.add_argument("--steps", type=str,
                        default="1",
                        help="Comma-separated steps for future frames (tar_obs_steps)")
    parser.add_argument("--loop", action="store_true",
                        help="Loop the motion continuously until Ctrl+C.")
    parser.add_argument("--loop-pause", type=float, default=0.0,
                        help="Seconds to wait between loops when --loop is set.")
    parser.add_argument("--motion-speed", type=float, default=1.0,
                        help="Motion playback speed. Use 0.5 for half-speed real deployment.")
    parser.add_argument("--motion-scale", type=float, default=1.0,
                        help="Scale mimic obs around default pose. Use 0.6 to reduce motion amplitude.")
    parser.add_argument(
        "--hip-yaw-scale",
        type=float,
        default=1.0,
        help="Scale left/right hip-yaw references; try 0.5 to reduce out-toeing.",
    )
    parser.add_argument(
        "--keep-absolute-heading",
        action="store_true",
        help="Publish the dataset's absolute yaw instead of making the first frame yaw zero.",
    )
    parser.add_argument(
        "--reference-mode",
        choices=("clean", "corrupt", "wm"),
        default="clean",
        help="Reference path used by the frozen TWIST policy.",
    )
    parser.add_argument(
        "--wm-checkpoint",
        default=None,
        help="Motion-WM best.pt. Defaults to full_stable_v2, then another existing best.pt.",
    )
    parser.add_argument(
        "--corruption-preset",
        choices=("formal", "demo_stress"),
        default="formal",
        help="formal matches training; demo_stress is visualization-only.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic corruption seed.")
    parser.add_argument(
        "--wait-for-sim-ready",
        action="store_true",
        help="Wait for the low-level MuJoCo process before publishing frame 0.",
    )
    parser.add_argument(
        "--sim-ready-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait when --wait-for-sim-ready is used.",
    )
    parser.add_argument("--vis", action="store_true", help="Visualize the motion")
    args = parser.parse_args()

    print("Robot type: ", args.robot)
    print("Motion file: ", args.motion_file)
    print("Steps: ", args.steps)
    print("Motion speed: ", args.motion_speed)
    print("Motion scale: ", args.motion_scale)
    print("Reference mode: ", args.reference_mode)
    
    HERE = os.path.dirname(os.path.abspath(__file__))
    
    if args.robot == "g1":
        xml_file = f"{HERE}/../assets/g1/g1_mocap_with_wrist_roll.xml"
        robot_base = "pelvis"
    else:
        raise ValueError(f"robot type {args.robot} not supported")
    
    
    main(args, xml_file, robot_base)
