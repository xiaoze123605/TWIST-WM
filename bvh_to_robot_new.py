import argparse
import pathlib
import time
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.lafan1 import load_bvh_file
from general_motion_retargeting.utils import auto_calibrate_human_scale, clean_qpos
from rich import print
from tqdm import tqdm
import os
import numpy as np

if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bvh_file",
        help="BVH motion file to load.",
        required=True,
        type=str,
    )
    
    parser.add_argument(
        "--format",
        choices=["lafan1", "nokov"],
        default="lafan1",
    )
    
    parser.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Loop the motion.",
    )
    
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "booster_t1", "stanford_toddy", "fourier_n1", "engineai_pm01", "pal_talos"],
        default="unitree_g1",
    )
    
    
    parser.add_argument(
        "--record_video",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--video_path",
        type=str,
        default="videos/example.mp4",
    )

    parser.add_argument(
        "--rate_limit",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion.",
    )
    
    parser.add_argument(
        "--motion_fps",
        default=None,
        type=float,
        help=(
            "Override the motion fps for the viewer. If not set, uses the BVH "
            "header's native frame rate."
        ),
    )

    parser.add_argument(
        "--target_fps",
        default=50.0,
        type=float,
        help=(
            "Target fps for the saved pkl file. The retargeted motion will be "
            "downsampled from the BVH native fps to this value. Default 50 "
            "(matches TWIST official mocap dataset)."
        ),
    )

    parser.add_argument(
        "--time_scale",
        default=1.0,
        type=float,
        help=(
            "Speed up (>1) or slow down (<1) the motion. A value of 1.6 "
            "means the output plays 1.6x faster than the original BVH, "
            "which increases step frequency proportionally. Useful when "
            "the source BVH has a slower gait than the reference dataset."
        ),
    )

    parser.add_argument(
        "--auto_calibrate_scale",
        action="store_true",
        default=False,
        help=(
            "Recompute human_scale_table from the BVH's first frame so each "
            "limb segment matches the target robot's link lengths. Strongly "
            "recommended when the BVH skeleton differs from the one the IK "
            "config was originally tuned for (e.g. custom Motive captures)."
        ),
    )

    parser.add_argument(
        "--ground_align",
        action="store_true",
        default=False,
        help=(
            "Per frame, lift/lower the human pose so the lowest foot sits "
            "just above the ground. Useful when the BVH root height and the "
            "robot's home pelvis height do not match."
        ),
    )

    parser.add_argument(
        "--clean_output",
        action="store_true",
        default=False,
        help=(
            "Apply post-processing to the saved trajectory: slerp resample, "
            "Butterworth low-pass, dof velocity clip and joint-limit clip. "
            "Strongly recommended for sim2sim and RL tracking."
        ),
    )
    parser.add_argument(
        "--lowpass_hz", default=5.0, type=float,
        help="Butterworth low-pass cutoff (Hz). 0 disables.",
    )
    parser.add_argument(
        "--max_dof_vel", default=30.0, type=float,
        help="Cap |Δq/Δt| per dof at this many rad/s. 0 disables.",
    )
    parser.add_argument(
        "--limit_margin_deg", default=5.0, type=float,
        help="Clip dof_pos to URDF limits with this margin (deg).",
    )
    parser.add_argument(
        "--root_z_mean", default=None, type=float,
        help=(
            "If set, translate root_pos[:,2] so its mean equals this value. "
            "Useful for matching a reference dataset's pelvis height "
            "(e.g. 0.762 for TWIST mocap). Ignored if --foot_ground_clearance "
            "is also set."
        ),
    )
    parser.add_argument(
        "--foot_ground_clearance", default=0.035, type=float,
        help=(
            "If set (meters), run FK to find the lowest foot z across the "
            "trajectory and lift root z so that foot sits at this clearance. "
            "Default 0.035 m (G1 ankle-to-sole distance). Set to 0 to disable. "
            "Takes precedence over --root_z_mean."
        ),
    )

    args = parser.parse_args()
    
    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:  # Only create directory if it's not empty
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []

    
    # Load BVH trajectory (now also returns native fps from BVH header)
    lafan1_data_frames, actual_human_height, bvh_fps = load_bvh_file(
        args.bvh_file, format=args.format
    )

    # Use BVH native fps unless user explicitly overrides
    motion_fps = args.motion_fps if args.motion_fps is not None else bvh_fps
    print(f"[bold]BVH native fps: {bvh_fps:.1f}[/bold]")
    print(f"Viewer motion_fps: {motion_fps:.1f}")
    if args.save_path:
        # Effective source fps accounts for time_scale: if time_scale=1.6,
        # we treat the source as if it were captured at bvh_fps*1.6, so that
        # resampling to target_fps keeps fewer frames → motion plays faster.
        effective_src_fps = bvh_fps * args.time_scale
        print(f"Save target_fps: {args.target_fps:.1f}")
        if args.time_scale != 1.0:
            print(f"Time scale: {args.time_scale:.2f}x  "
                  f"(effective src fps for resampling: {effective_src_fps:.1f})")
    
    # Initialize the retargeting system
    retargeter = GMR(
        src_human=f"bvh_{args.format}",
        tgt_robot=args.robot,
        actual_human_height=actual_human_height,
    )

    if args.auto_calibrate_scale:
        auto_calibrate_human_scale(retargeter, lafan1_data_frames[0])

    robot_motion_viewer = RobotMotionViewer(robot_type=args.robot,
                                            motion_fps=motion_fps,
                                            transparent_robot=0,
                                            record_video=args.record_video,
                                            video_path=args.video_path,
                                            # video_width=2080,
                                            # video_height=1170
                                            )
    
    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds
    
    # Create tqdm progress bar for the total number of frames
    pbar = tqdm(total=len(lafan1_data_frames), desc="Retargeting")
    
    # Start the viewer
    i = 0

    while True:
        
        # FPS measurement
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time
            
        # Update progress bar
        pbar.update(1)

        # Update task targets.
        smplx_data = lafan1_data_frames[i]

        # retarget
        qpos = retargeter.retarget(smplx_data, offset_to_ground=args.ground_align)
        
        # visualize
        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=retargeter.scaled_human_data,
            rate_limit=args.rate_limit,
            follow_camera=True,
        )

        if args.save_path is not None:
            qpos_list.append(qpos)

        if args.loop:
            i = (i + 1) % len(lafan1_data_frames)
        else:
            i += 1
            if i >= len(lafan1_data_frames):
                break
    
    if args.save_path is not None:
        import pickle

        qpos_array = np.array(qpos_list)  # (N, nq)

        # effective_src_fps incorporates time_scale: bvh_fps * time_scale.
        # When time_scale > 1, we pretend the source was faster, so resampling
        # to target_fps keeps fewer frames → motion plays faster.
        eff_src_fps = bvh_fps * args.time_scale

        if args.clean_output:
            foot_gc = args.foot_ground_clearance if args.foot_ground_clearance > 0 else None
            qpos_resampled, _info = clean_qpos(
                qpos_array,
                src_fps=eff_src_fps,
                tgt_fps=args.target_fps,
                model=retargeter.model,
                lowpass_hz=args.lowpass_hz,
                max_dof_vel=args.max_dof_vel,
                limit_margin_deg=args.limit_margin_deg,
                root_z_mean=args.root_z_mean,
                foot_ground_clearance=foot_gc,
                verbose=True,
            )
            tgt_fps = args.target_fps
        else:
            # Legacy path: naive linear interpolation, no filtering.
            from scipy.interpolate import interp1d

            n_frames_src = qpos_array.shape[0]
            src_fps = eff_src_fps
            tgt_fps = args.target_fps

            if abs(src_fps - tgt_fps) > 0.5:
                duration = n_frames_src / src_fps
                n_frames_tgt = int(round(duration * tgt_fps))
                t_src = np.linspace(0, duration, n_frames_src, endpoint=False)
                t_tgt = np.linspace(0, duration, n_frames_tgt, endpoint=False)

                interp_fn = interp1d(t_src, qpos_array, axis=0, kind='linear')
                qpos_resampled = interp_fn(t_tgt)

                # Renormalize quaternion after linear interp.
                root_rot_raw = qpos_resampled[:, 3:7]
                norms = np.linalg.norm(root_rot_raw, axis=1, keepdims=True)
                norms[norms < 1e-8] = 1.0
                qpos_resampled[:, 3:7] = root_rot_raw / norms

                print(
                    f"Downsampled: {n_frames_src} frames @ {src_fps:.1f} fps -> "
                    f"{n_frames_tgt} frames @ {tgt_fps:.1f} fps "
                    f"(duration {duration:.2f}s)"
                )
            else:
                qpos_resampled = qpos_array
                tgt_fps = src_fps
                print(f"No resampling needed (src={src_fps:.1f}, tgt={tgt_fps:.1f})")

        root_pos = qpos_resampled[:, :3]
        # save from wxyz to xyzw
        root_rot = qpos_resampled[:, 3:7][:, [1, 2, 3, 0]]
        dof_pos = qpos_resampled[:, 7:]
        
        motion_data = {
            "fps": float(tgt_fps),
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": None,
            "link_body_list": None,
        }
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path} ({qpos_resampled.shape[0]} frames @ {tgt_fps:.0f} fps)")

    # Close progress bar
    pbar.close()
    
    robot_motion_viewer.close()
