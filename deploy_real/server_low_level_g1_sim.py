import argparse
import json
import time
import sys
import numpy as np
import redis
import mujoco
import torch
from rich import print
from collections import deque
import mujoco.viewer as mjv
from tqdm import tqdm
from data_utils.params import DEFAULT_MIMIC_OBS
import os
from data_utils.rot_utils import quatToEuler, quat_rotate_inverse
from deploy_safety import MIMIC_OBS_DIM, parse_mimic_msg

BASE_OBS_DIM = 1155
ANYADAPTER_OBS_DIM = 2635
ANYADAPTER_HEADING_OBS_DIM = 2637
DTERA_OBS_DIM = 3695
ANY2TRACK_OBS_DIM = 7001
NUM_ACTIONS = 23
ANYADAPTER_HISTORY_LEN = 20
ANYADAPTER_STATE_INDICES = list(range(31, 36)) + list(range(36, 59)) + list(range(59, 82))
REDIS_STALE_THRESHOLD = 0.5
_POLICY_PROBE_ERRORS = {}
_POLICY_OBS_DIM_CACHE = {}
SUPPORTED_POLICY_OBS_DIMS = (
    BASE_OBS_DIM,
    ANYADAPTER_OBS_DIM,
    ANYADAPTER_HEADING_OBS_DIM,
    DTERA_OBS_DIM,
    ANY2TRACK_OBS_DIM,
)

# AnyAdapter runtime support (optional)
try:
    from twist_anyadapter_runtime import AnyAdapterRuntime, AnyAdapterRuntimeConfig
    _ANYADAPTER_AVAILABLE = True
except ImportError:
    _ANYADAPTER_AVAILABLE = False

def draw_root_velocity(mujoco_model, mujoco_data, mujoco_viewer, tgt_root_vel, init_geom_id, root_name, rgba_velocity=[1, 1, 0, 1]):
    """
    Draws an arrow representing velocity, for debug/visualization.
    """
    mujoco_viewer.user_scn.ngeom = init_geom_id
    root_body_id = mujoco_model.body(root_name).id
    root_pos = mujoco_data.xpos[root_body_id]
    root_vel = tgt_root_vel
    vel_scale = 1.0

    mujoco.mjv_initGeom(
        mujoco_viewer.user_scn.geoms[mujoco_viewer.user_scn.ngeom],
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        size=np.zeros(3),
        pos=np.zeros(3),
        mat=np.zeros(9),
        rgba=rgba_velocity,
    )
    mujoco.mjv_connector(
        mujoco_viewer.user_scn.geoms[mujoco_viewer.user_scn.ngeom],
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        width=0.01,
        from_=root_pos,
        to=root_pos + vel_scale * np.array(root_vel),
    )
    mujoco_viewer.user_scn.ngeom += 1
    return mujoco_viewer.user_scn.ngeom


# -------------------------------------------------------------------
# Main low-level policy controller that:
#   - reads mimic obs from Redis
#   - feeds into policy
#   - runs the sim
# -------------------------------------------------------------------
def extract_mimic_obs_to_body_and_wrist(mimic_obs):
    total_degrees = 33
    wrist_ids = [27, 32]
    other_ids = [f for f in range(total_degrees) if f not in wrist_ids]
    policy_target = mimic_obs[other_ids]
    wrist_dof_pos = mimic_obs[wrist_ids]
    
    return policy_target, wrist_dof_pos

def aggregate_wrist_dof_pos(body_dof_pos, wrist_dof_pos):
    total_degrees = 25
    wrist_ids = [19, 24]
    other_ids = [f for f in range(total_degrees) if f not in wrist_ids]
    whole_body_pd_target = np.zeros(total_degrees)
    whole_body_pd_target[other_ids] = body_dof_pos
    whole_body_pd_target[wrist_ids] = wrist_dof_pos
    
    return whole_body_pd_target


def _detect_policy_obs_dim(policy_path, device):
    """Load TorchScript once and probe all supported observation contracts."""
    cache_key = (os.path.abspath(policy_path), str(device))
    if cache_key in _POLICY_OBS_DIM_CACHE:
        return _POLICY_OBS_DIM_CACHE[cache_key]
    try:
        policy = torch.jit.load(policy_path, map_location=device)
        policy = policy.to(device)
        policy.eval()
    except Exception as exc:
        detail = str(exc).splitlines()[-1]
        for obs_dim in SUPPORTED_POLICY_OBS_DIMS:
            _POLICY_PROBE_ERRORS[(str(device), obs_dim)] = detail
        _POLICY_OBS_DIM_CACHE[cache_key] = None
        return None
    with torch.no_grad():
        for obs_dim in SUPPORTED_POLICY_OBS_DIMS:
            try:
                out = policy(torch.zeros(1, obs_dim, device=device))
                if out.shape[-1] == NUM_ACTIONS:
                    _POLICY_PROBE_ERRORS.pop((str(device), obs_dim), None)
                    _POLICY_OBS_DIM_CACHE[cache_key] = obs_dim
                    return obs_dim
                detail = f"policy output dim is {out.shape[-1]}, expected {NUM_ACTIONS}"
            except Exception as exc:
                detail = str(exc).splitlines()[-1]
            _POLICY_PROBE_ERRORS[(str(device), obs_dim)] = detail
    _POLICY_OBS_DIM_CACHE[cache_key] = None
    return None


def _policy_accepts_obs_dim(policy_path, device, obs_dim):
    return _detect_policy_obs_dim(policy_path, device) == obs_dim


def _should_use_anyadapter(
    policy_path, device, requested_anyadapter, detected_obs_dim=None
):
    if detected_obs_dim is None:
        detected_obs_dim = _detect_policy_obs_dim(policy_path, device)
    if detected_obs_dim == BASE_OBS_DIM and not requested_anyadapter:
        return False
    if detected_obs_dim == ANYADAPTER_OBS_DIM:
        print(
            "[AnyAdapter] Detected 2635-D AnyAdapter policy; "
            "enabling runtime history wrapper automatically."
        )
        return True
    if detected_obs_dim == ANYADAPTER_HEADING_OBS_DIM:
        print(
            "[AnyAdapter] Detected 2637-D heading-aware AnyAdapter policy; "
            "enabling runtime history and heading context automatically."
        )
        return True
    if detected_obs_dim == DTERA_OBS_DIM:
        print(
            "[DTERA] Detected 3695-D dual-history policy; enabling dynamics "
            "and tracking-error runtime histories automatically."
        )
        return True
    if detected_obs_dim == ANY2TRACK_OBS_DIM:
        print(
            "[Any2Track] Detected 7001-D layer-adapter policy; "
            "enabling 79-frame runtime history automatically."
        )
        return True
    if detected_obs_dim == BASE_OBS_DIM and requested_anyadapter:
        raise RuntimeError(
            "--use_anyadapter was requested, but the policy accepts only the "
            f"{BASE_OBS_DIM}-D base TWIST observation."
        )
    dtera_probe_error = _POLICY_PROBE_ERRORS.get(
        (str(device), DTERA_OBS_DIM), "unknown error"
    )
    if "same device" in dtera_probe_error.lower():
        raise RuntimeError(
            f"The policy reached the {DTERA_OBS_DIM}-D DTERA graph, but its "
            f"TorchScript tensors are not device-portable on {device}:\n"
            f"{dtera_probe_error}\n"
            "This is not an observation-dimension change. Re-export the JIT "
            "with the device-portable DTERA exporter, or use --device cpu as "
            "a temporary workaround."
        )
    raise RuntimeError(
        f"Policy does not accept {BASE_OBS_DIM}-D TWIST obs, "
        f"{ANYADAPTER_OBS_DIM}-D AnyAdapter obs, or "
        f"{ANYADAPTER_HEADING_OBS_DIM}-D heading-aware obs, or "
        f"{DTERA_OBS_DIM}-D DTERA obs, or "
        f"{ANY2TRACK_OBS_DIM}-D Any2Track obs: {policy_path}\n"
        f"Probe on {device} failed. Last DTERA error: "
        f"{dtera_probe_error}\n"
        "The device only selects where TorchScript inference runs; it does not "
        "change the policy observation contract."
    )
    
class RealTimePolicyController:
    def __init__(self,
                 xml_file,
                 policy_path,
                 device='cuda',
                 record_video=False,
                 use_anyadapter=False,
                 anyadapter_ema_alpha=0.0,
                 debug_policy_stats=False,
                 debug_policy_stats_steps=20):

        self.redis_client = None
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        except Exception as e:
            print(f"Error connecting to Redis: {e}")

        self.device = device
        self.policy_obs_dim = _detect_policy_obs_dim(policy_path, device)
        self.use_anyadapter = _should_use_anyadapter(
            policy_path, device, use_anyadapter, self.policy_obs_dim
        )
        self.use_dtera = self.policy_obs_dim == DTERA_OBS_DIM
        self.use_any2track = self.policy_obs_dim == ANY2TRACK_OBS_DIM
        self.anyadapter_context_dim = (
            2 if self.policy_obs_dim == ANYADAPTER_HEADING_OBS_DIM else 0
        )

        if self.use_anyadapter:
            if not _ANYADAPTER_AVAILABLE:
                raise ImportError(
                    "AnyAdapter runtime not found. Ensure deploy_real/ is on PYTHONPATH."
                )
            # AnyAdapterRuntime wraps the combined JIT (base + adapter) and
            # maintains the history buffer externally.
            self.anyadapter_cfg = AnyAdapterRuntimeConfig(
                base_obs_dim=BASE_OBS_DIM,
                num_actions=NUM_ACTIONS,
                history_len=79 if self.use_any2track else ANYADAPTER_HISTORY_LEN,
                state_indices=ANYADAPTER_STATE_INDICES,
                policy_path=policy_path,
                device=device,
                action_clip=10.0,
                action_ema_alpha=anyadapter_ema_alpha,
                adapter_context_dim=self.anyadapter_context_dim,
                fill_history_on_first_observation=(
                    self.use_any2track or self.use_dtera
                ),
                tracking_error_history_len=(
                    ANYADAPTER_HISTORY_LEN if self.use_dtera else 0
                ),
                tracking_ref_dof_vel_filter_alpha=0.5,
                tracking_ref_dof_vel_clip=20.0,
                control_dt=0.02,
                fill_tracking_history_on_first_observation=True,
            )
            self.anyadapter_runtime = AnyAdapterRuntime(self.anyadapter_cfg)
            self.policy = None  # not used directly
            print(f"[AnyAdapter] Runtime loaded, policy: {policy_path}")
        else:
            # Load original TWIST JIT policy directly
            self.policy = torch.jit.load(policy_path, map_location=device)
            print(f"Policy loaded from {policy_path}")

        # Create MuJoCo sim
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.model.opt.timestep = 0.001
        self.data = mujoco.MjData(self.model)
        
        # Print DoF names in order
        print("Degrees of Freedom (DoF) names and their order:")
        for i in range(self.model.nv):  # 'nv' is the number of DoFs
            dof_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.model.dof_jntid[i])
            print(f"DoF {i}: {dof_name}")

        # print("Body names and their IDs:")
        # for i in range(self.model.nbody):  # 'nbody' is the number of bodies
        #     body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
        #     print(f"Body ID {i}: {body_name}")
        
        print("Motor (Actuator) names and their IDs:")
        for i in range(self.model.nu):  # 'nu' is the number of actuators (motors)
            motor_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            print(f"Motor ID {i}: {motor_name}")
            

        self.viewer = mjv.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = 0
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 0
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = 0
        self.viewer.cam.distance = 2.0

        # Example defaults & placeholders
        self.num_actions = 23
        self.sim_duration = 100000.0
        self.sim_dt = 0.001
        self.sim_decimation = 20

        self.last_action = np.zeros(self.num_actions, dtype=np.float32)

        # PD Gains, etc. (adapt as needed)
        self.default_dof_pos = np.array([
                -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # left leg (6)
                -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # right leg (6)
                0.0, 0.0, 0.0, # torso (1)
                0.0, 0.4, 0.0, 1.2,
                0.0, -0.4, 0.0, 1.2,
            ])
        """
        mimic_obs = np.concatenate([
        root_pos[2:3],      # just the z for height
        rpy,                # roll, pitch, yaw
        root_vel_relative,  # local root vel
        dof_pos])
        """
        self.default_mimic_obs = DEFAULT_MIMIC_OBS["g1"]
        self.mujoco_default_dof_pos = np.concatenate([
            np.array([0, 0, 0.793]),
            np.array([0, 0, 0, 1]),
             np.array([-0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # left leg (6)
                -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # right leg (6)
                0.0, 0.0, 0.0, # torso (1)
                0.0, 0.2, 0.0, 1.2, 0.0, # left arm (4)
                0.0, -0.2, 0.0, 1.2, 0.0, # right arm (4)
                ])
        ])
        self.stiffness = np.array([
                100, 100, 100, 150, 40, 40,
                100, 100, 100, 150, 40, 40,
                150, 150, 150,
                40, 40, 40, 40, 20,
                40, 40, 40, 40, 20,
            ])
        self.damping = np.array([
                2, 2, 2, 4, 2, 2,
                2, 2, 2, 4, 2, 2,
                4, 4, 4,
                5, 5, 5, 5, 1,
                5, 5, 5, 5, 1,
            ])
        self.torque_limits = np.array([
                88, 139, 88, 139, 50, 50,
                88, 139, 88, 139, 50, 50,
                88, 50, 50,
                25, 25, 25, 25, 25,
                25, 25, 25, 25, 25,
            ])
        
        self.action_scale = 0.5

        
        self.ankle_idx = [4, 5, 10, 11]
        
        # For multi-step history
        self.n_mimic_obs = 31
        self.n_proprio = self.n_mimic_obs + 3 + 2 + 3*self.num_actions
        self.proprio_history_buf = deque(maxlen=10)
        for _ in range(10):
            self.proprio_history_buf.append(np.zeros(self.n_proprio))

        self.record_video = record_video
        self.debug_policy_stats = debug_policy_stats
        self.debug_policy_stats_steps = int(debug_policy_stats_steps)
        self._debug_policy_stats_count = 0
        self._last_valid_mimic_full = None
        self._last_mimic_warning_time = 0.0

    def _read_mimic_reference(self):
        """Read either supported Redis format without stopping the simulator."""
        try:
            raw = self.redis_client.get("action_mimic_g1")
            mimic, age, _ = parse_mimic_msg(raw, expected_dim=MIMIC_OBS_DIM)
            if age > REDIS_STALE_THRESHOLD:
                raise ValueError(
                    f"stale action_mimic_g1 message (age={age:.2f}s)"
                )
            self._last_valid_mimic_full = mimic.copy()
            return mimic
        except (redis.RedisError, ValueError, TypeError) as exc:
            now = time.monotonic()
            if now - self._last_mimic_warning_time >= 2.0:
                source = (
                    "last valid reference"
                    if self._last_valid_mimic_full is not None
                    else "default standing reference"
                )
                print(f"[Redis][WARN] {exc}; using {source}")
                self._last_mimic_warning_time = now
            if self._last_valid_mimic_full is not None:
                return self._last_valid_mimic_full.copy()
            return self.default_mimic_obs.astype(np.float32).copy()

    def _print_policy_debug_stats(self, step, action_mimic, obs_proprio, raw_action):
        if not self.debug_policy_stats or self._debug_policy_stats_count >= self.debug_policy_stats_steps:
            return
        ref_root = action_mimic[:8]
        ref_dof = action_mimic[8:31]
        actual_state = obs_proprio[:51]
        print(
            "[PolicyDebug] "
            f"step={step} "
            f"use_anyadapter={self.use_anyadapter} "
            f"ref_root_absmax={np.max(np.abs(ref_root)):.4f} "
            f"ref_dof_absmax={np.max(np.abs(ref_dof)):.4f} "
            f"actual_state_absmax={np.max(np.abs(actual_state)):.4f} "
            f"raw_action_absmax={np.max(np.abs(raw_action)):.4f} "
            f"raw_action_mean={np.mean(raw_action):.4f} "
            f"raw_action_std={np.std(raw_action):.4f}"
        )
        self._debug_policy_stats_count += 1

    def extract_data(self):
        qpos = self.data.qpos.astype(np.float32)
        qvel = self.data.qvel.astype(np.float32)
        
        body_ids = [0,1,2,3,4,5,
                    6,7,8,9,10,11,
                    12,13,14,
                    15,16,17,18,# 19
                    20,21,22,23, # 24
                    ]
        wrist_ids = [19, 24]
        
        whole_body_dof = qpos[7:]
        whole_body_dof_vel = qvel[6:]
        body_dof_pos = qpos[[f+7 for f in body_ids]]
        body_dof_vel = qvel[[f+6 for f in body_ids]]
        wrist_dof_pos = qpos[[f+7 for f in wrist_ids]]
        wrist_dof_vel = qvel[[f+6 for f in wrist_ids]]

        quat = self.data.sensor('orientation').data.astype(np.float32)
        ang_vel = self.data.sensor('angular-velocity').data.astype(np.float32)
        return whole_body_dof, whole_body_dof_vel, body_dof_pos, body_dof_vel, wrist_dof_pos, wrist_dof_vel, quat, ang_vel

    def reset_sim(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def reset(self, mujoco_dof_pos=None):
        # body & hand
        self.data.qpos[:] = mujoco_dof_pos
        mujoco.mj_forward(self.model, self.data)
       
    def run(self):
        # Optionally record video
        if self.record_video:
            import imageio
            video_name = "debug_sim.mp4"
            print(f"Saving video to {video_name}")
            mp4_writer = imageio.get_writer(video_name, fps=50)
        else:
            mp4_writer = None

        self.reset_sim()
        self.reset(self.mujoco_default_dof_pos)
        if self.use_anyadapter:
            self.anyadapter_runtime.reset()

        steps = int(self.sim_duration / self.sim_dt)
        pbar = tqdm(range(steps), desc="Simulating...")

        # send initial proprio to redis
        proprio_json = json.dumps(self.proprio_history_buf[0].tolist())
        self.redis_client.set("state_body_g1", proprio_json)
        self.redis_client.set("state_hand_g1", json.dumps(np.zeros(14).tolist()))
        try:
            for i in pbar:
                
                t_start = time.time()
                whole_body_dof, whole_body_dof_vel, body_dof_pos, body_dof_vel, wrist_dof_pos, wrist_dof_vel, quat, ang_vel = self.extract_data()
                
                if i % self.sim_decimation == 0:
                    
                    # Build a "proprio" vector for your policy, e.g.:
                    rpy = quatToEuler(quat)
                    obs_body_dof_vel = body_dof_vel.copy()
                    obs_body_dof_vel[self.ankle_idx] = 0.
                    obs_proprio = np.concatenate([
                        ang_vel * 0.25,
                        rpy[:2],
                        (body_dof_pos - self.default_dof_pos),
                        obs_body_dof_vel * 0.05,
                        self.last_action
                    ])
                    # send proprio to redis
                    self.redis_client.set("state_body_g1", json.dumps(obs_proprio.tolist()))
                    self.redis_client.set("state_hand_g1", json.dumps(np.zeros(14).tolist()))

                    # Use a safe standing reference until a fresh Redis frame arrives.
                    action_mimic_full = self._read_mimic_reference()
                    action_mimic, wrist_dof_pos = extract_mimic_obs_to_body_and_wrist(
                        action_mimic_full
                    )

                    obs_full = np.concatenate([action_mimic, obs_proprio])
                    obs_hist = np.array(self.proprio_history_buf).flatten()
                    obs_buf = np.concatenate([obs_full, obs_hist])
                    self.proprio_history_buf.append(obs_full)

                    obs_tensor = torch.from_numpy(obs_buf).float().unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        if self.use_anyadapter:
                            adapter_context = None
                            if self.anyadapter_context_dim == 2:
                                heading_error = np.arctan2(
                                    np.sin(action_mimic[3] - rpy[2]),
                                    np.cos(action_mimic[3] - rpy[2]),
                                )
                                adapter_context = np.array([
                                    np.sin(heading_error),
                                    1.0 - np.cos(heading_error),
                                ], dtype=np.float32)
                            raw_action = self.anyadapter_runtime.act(
                                obs_buf,
                                adapter_context=adapter_context,
                                **(
                                    {
                                        "tracking_reference": action_mimic,
                                        "dof_pos": body_dof_pos,
                                        "dof_vel": body_dof_vel,
                                        "root_linear_velocity": quat_rotate_inverse(
                                            np.asarray([
                                                quat[1], quat[2], quat[3], quat[0]
                                            ], dtype=np.float32).reshape(1, 4),
                                            np.asarray(
                                                self.data.qvel[:3], dtype=np.float32
                                            ).reshape(1, 3),
                                        ).reshape(3),
                                        "root_yaw_velocity": float(ang_vel[2]),
                                        "roll_pitch": rpy[:2],
                                    }
                                    if self.use_dtera else {}
                                ),
                            )
                        else:
                            raw_action = self.policy(obs_tensor).cpu().numpy().squeeze()

                    self._print_policy_debug_stats(i, action_mimic, obs_proprio, raw_action)
                    
                    self.last_action = raw_action
                    raw_action = np.clip(raw_action, -10., 10.)
                    scaled_actions = raw_action * self.action_scale
                    pd_target = scaled_actions + self.default_dof_pos
                    pd_target = aggregate_wrist_dof_pos(pd_target, wrist_dof_pos)
                    # debug draw velocity arrow if you want
                    self.viewer.user_scn.ngeom = 0
                    draw_root_velocity(self.model, self.data, self.viewer, [0,0,0], 0, "pelvis", [1,0,0,1])
                    
                    # make camera follow the pelvis
                    pelvis_pos = self.data.xpos[self.model.body("pelvis").id]
                    self.viewer.cam.lookat = pelvis_pos
                    self.viewer.sync()
                    if mp4_writer is not None:
                        img = self.viewer.read_pixels()
                        mp4_writer.append_data(img)

                # PD control
                torque = (pd_target - whole_body_dof) * self.stiffness - whole_body_dof_vel * self.damping
                torque = np.clip(torque, -self.torque_limits, self.torque_limits)
                
                self.data.ctrl[:] = torque
                
                mujoco.mj_step(self.model, self.data)
                # sleep to maintain real-time pace
                elapsed = time.time() - t_start
                if elapsed < self.sim_dt:
                    time.sleep(self.sim_dt - elapsed)
        except Exception as e:
            print(f"Error in run: {e}")
            pass
        finally:
            if mp4_writer is not None:
                mp4_writer.close()
                print("Video saved")

            self.viewer.close()


def main_low_level_sim(args):
    controller = RealTimePolicyController(
        xml_file=args.xml_file,
        policy_path=args.policy_path,
        device=args.device,
        record_video=args.record_video,
        use_anyadapter=args.use_anyadapter,
        anyadapter_ema_alpha=args.anyadapter_ema_alpha,
        debug_policy_stats=args.debug_policy_stats,
        debug_policy_stats_steps=args.debug_policy_stats_steps,
    )
    controller.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    HERE = os.path.dirname(os.path.abspath(__file__))
    
    parser.add_argument("--xml_file", default=os.path.join(HERE, "../assets/g1/g1_sim2sim_with_wrist_roll.xml"), help="Mujoco XML file")
    
    parser.add_argument("--policy_path",  help="Path to the policy",
                        default="../assets/twist_general_motion_tracker.pt"
                        )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch policy device (for example cuda, cuda:0, or cpu)",
    )
                        
    parser.add_argument("--record_video", action="store_true", help="Record a video")
    parser.add_argument("--use_anyadapter", action="store_true", help="Use AnyAdapter runtime wrapper")
    parser.add_argument("--anyadapter_ema_alpha", type=float, default=0.0, help="EMA smoothing for AnyAdapter action output; use 0 for fair A/B/C comparison")
    parser.add_argument("--debug_policy_stats", action="store_true", help="Print reference/proprio/action statistics for the first few policy steps")
    parser.add_argument("--debug_policy_stats_steps", type=int, default=20, help="Number of policy steps to print when --debug_policy_stats is enabled")
    args = parser.parse_args()

    args.record_proprio = True
    
    main_low_level_sim(args)
