"""
Stable OptiTrack -> GMR -> TWIST Redis bridge.

The upstream demo enables both G1 IK tables and warm-starts each solve from the
previous frame.  A bad hand orientation or an infeasible full-body target can
therefore push the solver into a folded local minimum.  This bridge keeps the
upstream mocap filters, but adds the constraints required by the TWIST runtime:
velocity-limited IK, table-2-only matching, wrist-task softening, branch-jump
guards, and output-distribution matching.

Usage:
  conda activate gmr
  python server_optitrack_redis.py
  python server_optitrack_redis.py --vis  # use the current desktop DISPLAY
"""

import argparse, json, os, socket, threading, time
from collections import deque
from queue import Empty
import numpy as np
import redis
from rich import print

import mujoco
import mujoco.viewer

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.optitrack_vendor.NatNetClient import setup_optitrack
from general_motion_retargeting.utils.realtime_filter import (
    RealtimeMotionFilter, BufferSmoother,
)
from data_utils.params import DEFAULT_ACTION_HAND, DEFAULT_MIMIC_OBS
from data_utils.rot_utils import quatToEuler, quat_rotate_inverse
# ── Helpers (same as v2 server) ─────────────────────────────────────────────

def _wrap_to_pi(x):
    return float(np.arctan2(np.sin(x), np.cos(x)))

def _quat_wxyz_to_yaw(q):
    w, x, y, z = map(float, q[:4])
    return float(np.arctan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z)))

def _quat_diff_yaw_rate(q0, q1, dt):
    if dt <= 1e-6: return 0.0
    return float(_wrap_to_pi(_quat_wxyz_to_yaw(q1) - _quat_wxyz_to_yaw(q0)) / dt)

def _lowpass_1st(prev, x, dt, fc_hz):
    if fc_hz is None or fc_hz <= 0: return float(x)
    rc = 1.0/(2.0*np.pi*fc_hz)
    return float((1.0 - dt/(rc+dt))*prev + (dt/(rc+dt))*x)


def _route_source_ip(server_ip):
    """Return the local IPv4 address selected by the kernel route table."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((server_ip, 1510))
        return sock.getsockname()[0]
    finally:
        sock.close()


def _same_ipv4_subnet_24(ip_a, ip_b):
    return ip_a.rsplit('.', 1)[0] == ip_b.rsplit('.', 1)[0]


# ── DoF mapping ─────────────────────────────────────────────────────────────

POLICY_DOF_INDEX = np.array([
    0,1,2,3,4,5,       # left  leg
    6,7,8,9,10,11,     # right leg
    12,13,14,           # waist
    15,16,17,18,19,     # left  arm (skip wrist_pitch=20, wrist_yaw=21)
    22,23,24,25,26,     # right arm (skip wrist_pitch=27, wrist_yaw=28)
], dtype=int)

# GMR's 29 actuated DoFs after the floating base. TWIST only consumes wrist
# roll, and its offline G1 reference inserts zero for both wrist-roll entries.
RAW_WRIST_DOF_INDEX = np.array([19, 20, 21, 26, 27, 28], dtype=int)
TWIST_WRIST_DOF_INDEX = np.array([19, 24], dtype=int)


def _quat_normalize_wxyz(q):
    q = np.asarray(q, dtype=float).copy()
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / norm


def _soften_wrist_orientation_tasks(gmr, orientation_cost=0.0):
    """Keep hand position targets without forcing incompatible hand frames."""
    changed = 0
    for table_name in ("ik_match_table1", "ik_match_table2"):
        table = getattr(gmr, table_name, None)
        if not isinstance(table, dict):
            continue
        for frame_name, entry in list(table.items()):
            frame_lower = str(frame_name).lower()
            if "wrist" not in frame_lower and "hand" not in frame_lower:
                continue
            if not isinstance(entry, (list, tuple)) or len(entry) < 5:
                continue
            body_name, pos_cost, _, pos_offset, rot_offset = entry[:5]
            table[frame_name] = [
                body_name, pos_cost, float(orientation_cost), pos_offset, rot_offset
            ]
            changed += 1

    if changed and hasattr(gmr, "setup_retarget_configuration"):
        gmr.setup_retarget_configuration()
    print(f"[GMR] softened {changed} wrist/hand orientation tasks "
          f"to cost={orientation_cost}")


def _lock_raw_wrist_joints(qpos, enabled=True):
    qpos = np.asarray(qpos, dtype=float).copy()
    if enabled and len(qpos) > 7 + int(RAW_WRIST_DOF_INDEX.max()):
        qpos[7 + RAW_WRIST_DOF_INDEX] = 0.0
    return qpos


def _limit_qpos_jump(qpos, previous, max_root_jump=0.25, max_joint_jump=0.45):
    """Clamp one-frame IK branch changes before they reach the viewer/policy."""
    qpos = np.asarray(qpos, dtype=float).copy()
    if not np.all(np.isfinite(qpos)):
        raise FloatingPointError("GMR qpos contains NaN/Inf")
    qpos[3:7] = _quat_normalize_wxyz(qpos[3:7])
    if previous is None:
        return qpos, False

    previous = np.asarray(previous, dtype=float)
    jumped = False
    if np.dot(qpos[3:7], previous[3:7]) < 0.0:
        qpos[3:7] *= -1.0

    root_delta = qpos[:3] - previous[:3]
    root_norm = np.linalg.norm(root_delta)
    if max_root_jump > 0 and root_norm > max_root_jump:
        qpos[:3] = previous[:3] + root_delta * (
            max_root_jump / (root_norm + 1e-8)
        )
        jumped = True

    if max_joint_jump > 0 and len(qpos) == len(previous) and len(qpos) > 7:
        joint_delta = qpos[7:] - previous[7:]
        clipped_delta = np.clip(joint_delta, -max_joint_jump, max_joint_jump)
        if np.any(clipped_delta != joint_delta):
            qpos[7:] = previous[7:] + clipped_delta
            jumped = True
    return qpos, jumped


def _pose_safety_reason(
    qpos,
    min_root_height=0.65,
    max_root_height=1.05,
    max_root_tilt=0.5,
    max_knee=2.4,
    max_knee_asymmetry=1.2,
):
    """Return a reason when an IK solution is outside the live G1 envelope."""
    qpos = np.asarray(qpos, dtype=float)
    if not np.all(np.isfinite(qpos)):
        return "NaN/Inf"
    if qpos.shape[0] < 19:
        return f"short qpos ({qpos.shape[0]})"

    root_z = float(qpos[2])
    if root_z < min_root_height or root_z > max_root_height:
        return f"root_z={root_z:.3f} outside [{min_root_height:.3f}, {max_root_height:.3f}]"

    roll, pitch, _ = quatToEuler(_quat_normalize_wxyz(qpos[3:7]))
    if max(abs(float(roll)), abs(float(pitch))) > max_root_tilt:
        return f"root tilt roll/pitch=({roll:+.3f}, {pitch:+.3f})"

    dof = qpos[7:]
    left_knee, right_knee = float(dof[3]), float(dof[9])
    if max(left_knee, right_knee) > max_knee:
        return f"knee limit=({left_knee:+.3f}, {right_knee:+.3f})"
    if abs(left_knee - right_knee) > max_knee_asymmetry:
        return f"knee asymmetry=({left_knee:+.3f}, {right_knee:+.3f})"
    return None


# ── mimic_obs builder ────────────────────────────────────────────────────────

def build_mimic_obs(qpos, qdot, robot_type, *,
                    prev_root_rot_wxyz=None, dt=None,
                    init_yaw=0.0, root_z_filtered=None,
                    max_root_vel=3.0, max_yaw_rate=4.0):
    root_pos, root_rot = qpos[:3], _quat_normalize_wxyz(qpos[3:7])
    dof_pos = qpos[7:][POLICY_DOF_INDEX].copy()
    if robot_type == "g1":
        dof_pos[TWIST_WRIST_DOF_INDEX] = 0.0
    roll, pitch, yaw = quatToEuler(root_rot)
    yaw = _wrap_to_pi(yaw - init_yaw)
    root_h = np.array([float(root_z_filtered)]) if root_z_filtered is not None else root_pos[2:3]
    root_vel = np.asarray(qdot[:3], dtype=float).copy()
    root_vel_norm = np.linalg.norm(root_vel)
    if max_root_vel > 0 and root_vel_norm > max_root_vel:
        root_vel *= max_root_vel / (root_vel_norm + 1e-8)
    root_rot_xyzw = root_rot.reshape(1,4)[:,[1,2,3,0]]
    root_vel_rel = quat_rotate_inverse(root_rot_xyzw, root_vel.reshape(1,3)).ravel()
    yr = _quat_diff_yaw_rate(prev_root_rot_wxyz, root_rot, dt) if prev_root_rot_wxyz is not None and dt and dt>1e-6 else 0.0
    if max_yaw_rate > 0:
        yr = float(np.clip(yr, -max_yaw_rate, max_yaw_rate))
    obs = np.concatenate([root_h, [roll,pitch,yaw], root_vel_rel, [yr], dof_pos])
    if not np.all(np.isfinite(obs)):
        raise FloatingPointError("mimic_obs contains NaN/Inf")
    return obs


# ── Server ───────────────────────────────────────────────────────────────────

class OptiTrackRedisServer:
    def __init__(self, args):
        self.args = args
        self.robot_type = args.robot

        # ── OptiTrack (exact copy from optitrack_to_robot.py) ──
        route_ip = _route_source_ip(args.host)
        if args.client_ip == "auto":
            self.client_ip = route_ip
        else:
            self.client_ip = args.client_ip
            if self.client_ip != route_ip:
                print(
                    f"[NatNet][WARN] requested client_ip={self.client_ip}, "
                    f"but route to {args.host} uses {route_ip}"
                )

        transport = args.transport
        if args.use_multicast:
            transport = "multicast"
        if transport == "auto":
            transport = (
                "multicast"
                if _same_ipv4_subnet_24(self.client_ip, args.host)
                else "unicast"
            )
        self.use_multicast = transport == "multicast"
        print(
            f"Connecting to {args.host} from {self.client_ip} "
            f"using {transport} NatNet ..."
        )
        self.client = setup_optitrack(
            args.host, self.client_ip, self.use_multicast
        )
        t = threading.Thread(target=self.client.run, daemon=True); t.start()

        # Wait for connection (exact copy from optitrack_to_robot.py lines 87-95)
        timeout = 10
        t0 = time.time()
        while not self.client.connected():
            if time.time() - t0 > timeout:
                print(f"Failed to connect within {timeout}s")
                self.client.shutdown()
                raise RuntimeError("OptiTrack connection failed")
            time.sleep(0.1)
        print(f"Connected: {self.client.connected()}")

        # A velocity-limited solve prevents one bad mocap frame from moving the
        # warm-started IK configuration directly into a folded branch.
        tgt = "unitree_g1" if args.robot == "g1" else "unitree_t1"
        self.gmr = GMR(
            src_human="fbx",
            tgt_robot=tgt,
            actual_human_height=args.actual_human_height,
            damping=args.gmr_damping,
            use_velocity_limit=not args.no_velocity_limit,
        )
        if not args.enable_table1:
            self.gmr.use_ik_match_table1 = False
            print("[GMR] table1 disabled; using position-aware table2")

        # The stock FBX config gives each toe task position/orientation costs
        # 100/50, but gives the free-floating pelvis only 10/5. With live data,
        # the solver can reduce total foot/limb error by tipping the whole robot
        # onto its side. A balanced pelvis cost keeps the base upright while
        # leaving enough freedom for leg tracking.
        pelvis_task = self.gmr.ik_match_table2.get("pelvis")
        if pelvis_task is not None and len(pelvis_task) >= 3:
            pelvis_task[1] = float(args.pelvis_task_cost)
            pelvis_task[2] = float(args.pelvis_task_cost)
            print(f"[GMR] table2 pelvis position/orientation cost="
                  f"{args.pelvis_task_cost}")
        if not args.keep_wrist_orientation:
            _soften_wrist_orientation_tasks(
                self.gmr, orientation_cost=args.wrist_orient_cost
            )
        elif hasattr(self.gmr, "setup_retarget_configuration"):
            self.gmr.setup_retarget_configuration()

        self._required_mocap_bodies = set(self.gmr.human_body_to_task2.keys())
        self._dropped_frame_count = 0
        self._invalid_body_count = 0
        self._missing_body_count = 0
        self._last_mocap_bodies = {}
        self._last_mocap_body_time = {}

        # ── Filters (exact copy from optitrack_to_robot.py) ──
        self.motion_filter = RealtimeMotionFilter(
            pos_smoothing=0.5,
            rot_smoothing=0.4,
            max_pos_jump=0.15,
            max_rot_jump_deg=60.0,
            max_hand_pos_jump=0.05,
            max_hand_rot_jump_deg=30.0,
            wrist_roll_limit_deg=60.0,
            hand_orient_smooth=0.85,
            quat_avg_window=5,
            flip_detect=True,
        )
        self.median_filter = BufferSmoother(window_size=3, mode='median')

        # ── Viewer (exact copy from optitrack_to_robot.py) ──
        self.viewer = None
        if args.vis:
            self.viewer = RobotMotionViewer(robot_type=tgt)

        # ── Redis ──
        self.redis = redis.Redis(host="localhost", port=6379, db=0)

        # ── Post-IK filter state (streaming Butterworth for root + legs) ──
        self._bw_sos_pos = None; self._bw_zi_pos = None
        self._bw_sos_quat = None; self._bw_zi_quat = None; self._bw_prev_quat = None
        self._bw_sos_legs = None; self._bw_zi_legs = None
        self._bw_sos_arms = None; self._bw_zi_arms = None
        self._prev_qpos_filt = None  # for velocity clipping
        self._qpos_history = deque(maxlen=11)  # for regression-based velocity
        self._qdot_smoothed = None  # EMA-smoothed velocity

        # FK workspace for output-side ground alignment. Keep this separate from
        # GMR's warm-started configuration so visualization cannot perturb IK.
        self._ground_data = mujoco.MjData(self.gmr.model)
        self._ground_foot_ids = []
        for body_name in (
            "left_ankle_roll_link", "right_ankle_roll_link",
            "left_toe_link", "right_toe_link",
        ):
            body_id = mujoco.mj_name2id(
                self.gmr.model, mujoco.mjtObj.mjOBJ_BODY, body_name
            )
            if body_id >= 0:
                self._ground_foot_ids.append(body_id)
        self._ground_shift = None
        self._last_output_qpos = None

        # ── Teleop state ──
        self.last_mimic_obs = DEFAULT_MIMIC_OBS[args.robot].copy()
        self.init_yaw = 0.0
        self.root_z_baseline = None
        # Match DEFAULT_MIMIC_OBS and the TWIST training distribution.
        self.root_z_target_mean = float(DEFAULT_MIMIC_OBS[args.robot][0])
        self.root_z_filt = None
        self._last_safe_qpos = None
        self._jump_count = 0
        self._rejected_pose_count = 0

    def _decode_mocap_data(self, mocap_data):
        """Convert one NatNet packet using dynamic full IDs before static IDs."""
        self.client.latest_frame_number = mocap_data.prefix_data.frame_number
        frame = {}
        id_map = self.client.rigid_body_id_map

        def add_rigid_body(rb, overwrite):
            if not rb.tracking_valid:
                self._invalid_body_count += 1
                if self.args.reject_invalid_tracking:
                    return
            full_id = int(rb.id_num)
            local_id = full_id & 0xFFFF
            full_name = id_map.get(full_id)
            local_name = id_map.get(local_id)
            # Prefer dynamic names only when they match GMR's expected FBX body
            # names. Some Motive versions report prefixed model-description names.
            if full_name in self._required_mocap_bodies:
                body_name = full_name
            elif local_name in self._required_mocap_bodies:
                body_name = local_name
            else:
                body_name = full_name if full_name is not None else local_name
            if body_name is None:
                return
            value = [rb.pos, np.roll(rb.rot, 1)]
            if overwrite or body_name not in frame:
                frame[body_name] = value

        skeleton_data = mocap_data.skeleton_data
        if skeleton_data is not None and skeleton_data.skeleton_list:
            # TWIST/GMR currently supports one streamed human skeleton.
            for rb in skeleton_data.skeleton_list[0].rigid_body_list:
                add_rigid_body(rb, overwrite=True)

        rigid_body_data = mocap_data.rigid_body_data
        if rigid_body_data is not None:
            for rb in rigid_body_data.rigid_body_list:
                add_rigid_body(rb, overwrite=False)

        now = time.monotonic()
        for body_name, value in frame.items():
            self._last_mocap_bodies[body_name] = value
            self._last_mocap_body_time[body_name] = now

        missing = self._required_mocap_bodies.difference(frame)
        for body_name in tuple(missing):
            age = now - self._last_mocap_body_time.get(body_name, -np.inf)
            if age <= self.args.missing_body_hold:
                frame[body_name] = self._last_mocap_bodies[body_name]
                missing.remove(body_name)
        if missing:
            self._missing_body_count += 1
            if self._missing_body_count % 60 == 1:
                print(f"[NatNet][DROP] missing required bodies: {sorted(missing)}")
            return None
        return frame

    def _get_latest_frame(self, timeout):
        """Drain NatNet's bounded queue and process only the newest packet."""
        queue = self.client.data_queue
        try:
            mocap_data = queue.get(block=True, timeout=timeout)
        except Empty:
            return None

        dropped = 0
        while True:
            try:
                mocap_data = queue.get_nowait()
                dropped += 1
            except Empty:
                break
        self._dropped_frame_count += dropped
        return self._decode_mocap_data(mocap_data)

    def _pose_safety_reason(self, qpos):
        return _pose_safety_reason(
            qpos,
            min_root_height=self.args.min_root_height,
            max_root_height=self.args.max_root_height,
            max_root_tilt=self.args.max_root_tilt,
            max_knee=self.args.max_knee,
            max_knee_asymmetry=self.args.max_knee_asymmetry,
        )

    def _sync_gmr_configuration(self, qpos):
        """Keep GMR's warm start aligned with the protected output qpos."""
        configuration = getattr(self.gmr, "configuration", None)
        if configuration is not None and hasattr(configuration, "update"):
            configuration.update(np.asarray(qpos, dtype=float))

    def _publish(self, obs, frame_id=0):
        obs = np.asarray(obs, dtype=float)
        if obs.shape != DEFAULT_MIMIC_OBS[self.robot_type].shape:
            raise ValueError(
                f"mimic_obs shape mismatch: got {obs.shape}, "
                f"expected {DEFAULT_MIMIC_OBS[self.robot_type].shape}"
            )
        self.last_mimic_obs = obs.copy()
        if self.args.dry_run:
            return

        payload = obs.tolist()
        if self.args.timestamped_redis:
            payload = {
                "timestamp": time.time(),
                "frame_id": int(frame_id),
                "action_mimic": payload,
            }
        self.redis.set(f"action_mimic_{self.robot_type}", json.dumps(payload))
        if self.robot_type == "g1":
            self.redis.set("action_hand_g1", json.dumps(DEFAULT_ACTION_HAND["g1"].tolist()))

    def _post_filter(self, qpos):
        """Real-time equivalent of offline clean_qpos (bvh_to_robot_new.py).

        Offline: 4th-order zero-phase Butterworth at 5 Hz on ALL dofs.
        Real-time: 4th-order causal Butterworth at 5 Hz on root+legs+waist,
        plus a lighter configurable arm filter.

        Also: velocity clip at 30 rad/s + joint-limit clip at 5 deg.
        """
        from scipy.signal import butter, sosfilt, sosfilt_zi
        out = qpos.copy()
        CUTOFF = 5.0   # Hz — match offline clean_qpos
        ORDER = 4       # match offline clean_qpos
        FS = float(self.args.freq)

        # ── Root position (x, y, z) ──
        pos = qpos[0:3].reshape(1, 3)
        if self._bw_sos_pos is None:
            self._bw_sos_pos = butter(ORDER, CUTOFF, 'low', fs=FS, output='sos')
            self._bw_zi_pos = sosfilt_zi(self._bw_sos_pos)[:,:,np.newaxis] * pos[0]
        pos_f, self._bw_zi_pos = sosfilt(self._bw_sos_pos, pos, axis=0, zi=self._bw_zi_pos)
        out[0:3] = pos_f.ravel()

        # ── Root rotation (w, x, y, z) — unflip, filter, renormalize ──
        quat = qpos[3:7].copy()
        if self._bw_prev_quat is not None and np.dot(quat, self._bw_prev_quat) < 0:
            quat = -quat
        self._bw_prev_quat = quat.copy()
        quat_u = quat.reshape(1, 4)
        if self._bw_zi_quat is None:
            self._bw_sos_quat = butter(ORDER, CUTOFF, 'low', fs=FS, output='sos')
            self._bw_zi_quat = sosfilt_zi(self._bw_sos_quat)[:,:,np.newaxis] * quat_u[0]
        quat_f, self._bw_zi_quat = sosfilt(self._bw_sos_quat, quat_u, axis=0, zi=self._bw_zi_quat)
        quat_f = quat_f.ravel()
        out[3:7] = quat_f / max(np.linalg.norm(quat_f), 1e-10)

        # ── Leg + waist dofs (GMR indices 0-14 = qpos[7:22]) ──
        # Arm dofs (GMR indices 15-28 = qpos[22:36]) → UNFILTERED
        leg_end = min(15, len(qpos) - 7)
        if leg_end > 0:
            legs = qpos[7:7+leg_end].reshape(1, -1)
            if self._bw_zi_legs is None:
                self._bw_sos_legs = butter(ORDER, CUTOFF, 'low', fs=FS, output='sos')
                self._bw_zi_legs = sosfilt_zi(self._bw_sos_legs)[:,:,np.newaxis] * legs[0]
            legs_f, self._bw_zi_legs = sosfilt(self._bw_sos_legs, legs, axis=0, zi=self._bw_zi_legs)
            out[7:7+leg_end] = legs_f.ravel()

        arm_start = 7 + leg_end
        if arm_start < len(qpos) and self.args.arm_filter_cutoff > 0.0:
            arms = qpos[arm_start:].reshape(1, -1)
            if self._bw_zi_arms is None:
                self._bw_sos_arms = butter(
                    ORDER,
                    float(self.args.arm_filter_cutoff),
                    'low',
                    fs=FS,
                    output='sos',
                )
                self._bw_zi_arms = (
                    sosfilt_zi(self._bw_sos_arms)[:, :, np.newaxis] * arms[0]
                )
            arms_f, self._bw_zi_arms = sosfilt(
                self._bw_sos_arms,
                arms,
                axis=0,
                zi=self._bw_zi_arms,
            )
            out[arm_start:] = arms_f.ravel()

        # ── Velocity clip: 30 rad/s on ALL dofs (same as clean_qpos) ──
        if self._prev_qpos_filt is not None and len(out) > 7:
            dq_max = 30.0 / FS
            delta = out[7:] - self._prev_qpos_filt[7:]
            out[7:] = self._prev_qpos_filt[7:] + np.clip(delta, -dq_max, dq_max)
        self._prev_qpos_filt = out.copy()

        return out

    def _align_feet_to_ground(self, qpos, dt):
        """Align the lowest G1 foot body to the viewer ground using FK."""
        out = np.asarray(qpos, dtype=float).copy()
        if self.args.no_ground_align or not self._ground_foot_ids:
            return out, None, np.nan

        self._ground_data.qpos[:] = 0.0
        nq = min(out.shape[0], self.gmr.model.nq)
        self._ground_data.qpos[:nq] = out[:nq]
        mujoco.mj_forward(self.gmr.model, self._ground_data)
        min_foot_z = min(
            float(self._ground_data.xpos[body_id, 2])
            for body_id in self._ground_foot_ids
        )
        desired_shift = float(self.args.foot_ground_clearance - min_foot_z)
        if abs(desired_shift) > self.args.max_ground_shift:
            return (
                out,
                f"ground shift {desired_shift:+.3f}m exceeds "
                f"{self.args.max_ground_shift:.3f}m",
                min_foot_z,
            )

        if self._ground_shift is None:
            shift = desired_shift
        elif desired_shift >= self._ground_shift:
            # Never smooth a correction that prevents visible penetration.
            shift = desired_shift
        else:
            cutoff = max(float(self.args.ground_align_cutoff), 1e-3)
            rc = 1.0 / (2.0 * np.pi * cutoff)
            alpha = float(np.clip(dt / (rc + dt), 0.0, 1.0))
            shift = self._ground_shift + alpha * (
                desired_shift - self._ground_shift
            )
        self._ground_shift = float(shift)
        out[2] += self._ground_shift
        return out, None, min_foot_z + self._ground_shift

    def safe_stand(self, freq=50, seconds=2.0):
        target = DEFAULT_MIMIC_OBS[self.robot_type].copy()
        steps = max(int(seconds*freq), 1)
        start = self.last_mimic_obs.copy()
        for i in range(steps):
            a = i/steps
            self._publish(start + (target - start) * a)
            time.sleep(1.0/freq)
        self._publish(target)
        print("[Safe Stand] Done")

    def run(self, freq=50, debug_arms=False):
        max_cutoff = max(5.0, float(self.args.arm_filter_cutoff))
        if freq <= 2.0 * max_cutoff:
            raise ValueError(
                f"--freq must be greater than {2.0 * max_cutoff:g} Hz "
                "for the configured filters"
            )
        print("="*60)
        mode = "DRY RUN (Redis disabled)" if self.args.dry_run else "Redis publishing"
        print(f"[Server] Starting stable OptiTrack pipeline: {mode} ...")
        print("="*60)

        self.safe_stand(freq=freq)

        # Wait for first frame to lock init_yaw
        print("[Teleop] Waiting for first frame ...")
        last_wait_warning = time.monotonic()
        while True:
            frame = self._get_latest_frame(timeout=2.0)
            if frame and 'Hips' in frame:
                break
            now = time.monotonic()
            if now - last_wait_warning >= 5.0:
                mode = "multicast" if self.use_multicast else "unicast"
                print(
                    f"[NatNet][WAIT] command channel connected but no complete "
                    f"skeleton frame; server={self.args.host}, "
                    f"client={self.client_ip}, transport={mode}"
                )
                last_wait_warning = now
            time.sleep(0.02)
        frame = self.motion_filter(frame)
        frame = self.median_filter(frame)
        first_qpos = np.asarray(self.gmr.retarget(frame), dtype=float)
        first_qpos = _lock_raw_wrist_joints(
            first_qpos,
            enabled=(self.robot_type == "g1" and not self.args.unlock_raw_wrist),
        )
        first_qpos, _ = _limit_qpos_jump(
            first_qpos,
            None,
            max_root_jump=self.args.max_root_jump,
            max_joint_jump=self.args.max_joint_jump,
        )
        first_reject_reason = self._pose_safety_reason(first_qpos)
        if first_reject_reason is not None:
            raise RuntimeError(
                f"Unsafe first GMR pose: {first_reject_reason}. "
                "Check Motive Global/FBX/Z-axis streaming and stand neutrally."
            )
        self._sync_gmr_configuration(first_qpos)
        self._last_safe_qpos = first_qpos.copy()
        first_qpos_filt = self._post_filter(first_qpos)
        first_qpos_filt, ground_reason, first_foot_z = self._align_feet_to_ground(
            first_qpos_filt, 1.0 / freq
        )
        if ground_reason is not None:
            raise RuntimeError(f"Unsafe first GMR ground alignment: {ground_reason}")
        aligned_reason = self._pose_safety_reason(first_qpos_filt)
        if aligned_reason is not None:
            raise RuntimeError(f"Unsafe first aligned GMR pose: {aligned_reason}")
        self._last_output_qpos = first_qpos_filt.copy()

        self.init_yaw = _quat_wxyz_to_yaw(first_qpos[3:7])
        self.root_z_baseline = float(first_qpos_filt[2])
        self.root_z_filt = self.root_z_target_mean
        print(f"[Teleop] init_yaw={np.degrees(self.init_yaw):+.1f}deg  "
              f"root_z_baseline={self.root_z_baseline:.3f}m")
        if self.args.debug:
            first_dof = first_qpos_filt[7:]
            print(
                f"[first qpos] root_z={first_qpos_filt[2]:.3f} "
                f"support_z={first_foot_z:.3f} "
                f"knees=({first_dof[3]:+.3f}, {first_dof[9]:+.3f}) "
                f"hips_pitch=({first_dof[0]:+.3f}, {first_dof[6]:+.3f})"
            )

        # Warmup
        ref0_obs = build_mimic_obs(
            first_qpos_filt, np.zeros_like(first_qpos_filt), self.robot_type,
            prev_root_rot_wxyz=first_qpos_filt[3:7], dt=1.0/freq,
            init_yaw=self.init_yaw, root_z_filtered=self.root_z_target_mean)
        warmup_n = max(int(2.0*freq), 1)
        warmup_start = self.last_mimic_obs.copy()
        print(f"[Teleop] Warmup 2.0s ...")
        for i in range(warmup_n):
            a = (i+1)/warmup_n
            self._publish(warmup_start + (ref0_obs - warmup_start) * a)
            time.sleep(1.0/freq)

        # ── Main loop (EXACT copy of optitrack_to_robot.py lines 152-202) ──
        first_sample_t = time.monotonic()
        self._qpos_history.append((first_sample_t, first_qpos_filt.copy()))
        prev_root_rot = first_qpos_filt[3:7].copy()
        prev_t = first_sample_t
        period = 1.0 / freq
        next_tick = first_sample_t
        frame_count = 0
        print("[Teleop] Running ...")

        while True:
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0.0:
                time.sleep(delay)
            elif delay < -period:
                next_tick = time.monotonic()

            # Get frame (line 153)
            frame = self._get_latest_frame(timeout=5.0)
            if frame is None or len(frame) == 0:
                continue
            if 'Hips' not in frame:
                continue

            # Filter (lines 163-166)
            frame = self.motion_filter(frame)
            frame = self.median_filter(frame)

            # Retarget (line 196)
            qpos = np.asarray(self.gmr.retarget(frame), dtype=float)
            qpos = _lock_raw_wrist_joints(
                qpos,
                enabled=(self.robot_type == "g1" and not self.args.unlock_raw_wrist),
            )
            reject_reason = self._pose_safety_reason(qpos)
            if reject_reason is not None and self._last_safe_qpos is not None:
                self._rejected_pose_count += 1
                qpos = self._last_safe_qpos.copy()
                self._sync_gmr_configuration(qpos)
                jumped = False
                if self.args.debug and self._rejected_pose_count % 30 == 1:
                    print(f"[GMR][REJECT] {reject_reason}; holding last safe pose")
            else:
                qpos, jumped = _limit_qpos_jump(
                    qpos,
                    self._last_safe_qpos,
                    max_root_jump=self.args.max_root_jump,
                    max_joint_jump=self.args.max_joint_jump,
                )
                self._sync_gmr_configuration(qpos)
                self._last_safe_qpos = qpos.copy()
                if jumped:
                    self._jump_count += 1
            frame_count += 1

            # Debug: print raw GMR arm angles
            if debug_arms and frame_count % 100 == 0:
                la = qpos[22:29]
                ra = qpos[29:36]
                print(f"[F{frame_count}] L(elbow={la[3]:+.4f} shld_r={la[1]:+.4f} shld_y={la[2]:+.4f}) "
                      f"R(elbow={ra[3]:+.4f} shld_r={ra[1]:+.4f} shld_y={ra[2]:+.4f})")

            # ── TWIST-specific: publish to redis ──
            # Apply post-IK smoothing (matches offline clean_qpos).
            qpos_filt = self._post_filter(qpos)
            qpos_filt, ground_reason, support_z = self._align_feet_to_ground(
                qpos_filt, dt=1.0 / freq
            )
            aligned_reason = (
                ground_reason
                if ground_reason is not None
                else self._pose_safety_reason(qpos_filt)
            )
            if aligned_reason is not None and self._last_output_qpos is not None:
                self._rejected_pose_count += 1
                qpos_filt = self._last_output_qpos.copy()
                if self.args.debug and self._rejected_pose_count % 30 == 1:
                    print(
                        f"[GMR][OUTPUT REJECT] {aligned_reason}; "
                        "holding last aligned pose"
                    )
            else:
                self._last_output_qpos = qpos_filt.copy()

            # Display exactly the protected reference used to build Redis obs,
            # rather than an unsafe raw IK solution.
            if self.viewer is not None:
                try:
                    self.viewer.step(
                        root_pos=qpos_filt[:3],
                        root_rot=qpos_filt[3:7],
                        dof_pos=qpos_filt[7:],
                        rate_limit=False,
                    )
                except Exception as exc:
                    print(f"[Viewer][WARN] {exc!r}")

            # Velocity via timestamped regression over recent output frames.
            sample_t = time.monotonic()
            self._qpos_history.append((sample_t, qpos_filt.copy()))
            if len(self._qpos_history) >= 7:
                t_reg = np.array(
                    [sample[0] for sample in self._qpos_history], dtype=float
                )
                t_reg -= t_reg[0]
                t_mean = np.mean(t_reg)
                q_arr = np.array(
                    [sample[1] for sample in self._qpos_history], dtype=float
                )
                denom = np.sum((t_reg - t_mean) ** 2)
                qdot_raw = (np.sum((t_reg - t_mean)[:, None] * (q_arr - np.mean(q_arr, axis=0)), axis=0)
                            / max(denom, 1e-10))  # per-second units
            elif len(self._qpos_history) >= 2:
                t0, q0 = self._qpos_history[-2]
                t1, q1 = self._qpos_history[-1]
                qdot_raw = (q1 - q0) / max(t1 - t0, 1e-4)
            else:
                qdot_raw = np.zeros_like(qpos_filt)

            # EMA-smooth the velocity (further reduces noise)
            alpha_qdot = 0.3  # ~3-frame effective window
            if self._qdot_smoothed is None:
                self._qdot_smoothed = qdot_raw.copy()
            else:
                self._qdot_smoothed = (1-alpha_qdot) * self._qdot_smoothed + alpha_qdot * qdot_raw

            dt_real = float(np.clip(
                sample_t - prev_t,
                1.0 / (4.0 * freq),
                4.0 / freq,
            ))
            prev_t = sample_t

            z_shifted = float(qpos_filt[2]) + (self.root_z_target_mean - self.root_z_baseline)
            self.root_z_filt = _lowpass_1st(self.root_z_filt, z_shifted, dt_real, 1.5)

            cur_rot = qpos_filt[3:7].copy()
            obs = build_mimic_obs(
                qpos_filt, self._qdot_smoothed, self.robot_type,
                prev_root_rot_wxyz=prev_root_rot, dt=dt_real,
                init_yaw=self.init_yaw, root_z_filtered=self.root_z_filt)
            prev_root_rot = cur_rot

            self._publish(obs, frame_id=frame_count)

            if self.args.debug and frame_count % max(int(2.0 * freq), 1) == 0:
                dof = qpos_filt[7:]
                print(
                    f"[frame {frame_count}] root_z={qpos_filt[2]:.3f} "
                    f"support_z={support_z:.3f} "
                    f"roll/pitch={np.round(obs[1:3], 3)} "
                    f"knees=({dof[3]:+.3f}, {dof[9]:+.3f}) "
                    f"jump_guards={self._jump_count} "
                    f"rejected={self._rejected_pose_count} "
                    f"queue_dropped={self._dropped_frame_count} "
                    f"invalid_bodies={self._invalid_body_count} "
                    f"missing_frames={self._missing_body_count}"
                )


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stable OptiTrack -> G1 -> TWIST Redis bridge")
    parser.add_argument("--host", default="192.168.3.103")
    parser.add_argument("--client_ip", default="auto",
                        help="local NatNet interface address; default: route auto-detect")
    parser.add_argument("--transport", choices=["auto", "multicast", "unicast"],
                        default="auto",
                        help="NatNet transport; auto uses unicast across subnets")
    parser.add_argument("--use_multicast", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--robot", default="g1", choices=["g1","t1"])
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--freq", type=float, default=50.0,
                        help="GMR processing/Redis output rate in Hz")
    parser.add_argument("--arm_filter_cutoff", type=float, default=8.0,
                        help="causal arm DoF low-pass cutoff in Hz; 0 disables")
    parser.add_argument("--actual_human_height", type=float, default=1.6)
    parser.add_argument("--gmr_damping", type=float, default=1.0)
    parser.add_argument("--pelvis_task_cost", type=float, default=100.0,
                        help="table2 pelvis position/orientation cost; 100 prevents base collapse")
    parser.add_argument("--no_velocity_limit", action="store_true",
                        help="disable GMR IK velocity limits (unsafe for live mocap)")
    parser.add_argument("--enable_table1", action="store_true",
                        help="enable the conflicting rotation-only IK table")
    parser.add_argument("--keep_wrist_orientation", action="store_true",
                        help="keep strong OptiTrack hand-orientation IK costs")
    parser.add_argument("--wrist_orient_cost", type=float, default=0.0)
    parser.add_argument("--unlock_raw_wrist", action="store_true",
                        help="allow raw GMR wrist roll/pitch/yaw in qpos")
    parser.add_argument("--max_root_jump", type=float, default=0.25)
    parser.add_argument("--max_joint_jump", type=float, default=0.20)
    parser.add_argument("--min_root_height", type=float, default=0.65)
    parser.add_argument("--max_root_height", type=float, default=1.05)
    parser.add_argument("--max_root_tilt", type=float, default=0.5)
    parser.add_argument("--max_knee", type=float, default=2.4)
    parser.add_argument("--max_knee_asymmetry", type=float, default=1.2)
    parser.add_argument("--no_ground_align", action="store_true",
                        help="disable output-side G1 foot FK ground alignment")
    parser.add_argument("--foot_ground_clearance", type=float, default=0.035,
                        help="target z of the lowest G1 foot body in meters")
    parser.add_argument("--ground_align_cutoff", type=float, default=3.0,
                        help="low-pass cutoff used only while lowering floating feet")
    parser.add_argument("--max_ground_shift", type=float, default=0.35,
                        help="reject frames requiring a larger root-z correction")
    parser.add_argument("--reject_invalid_tracking", action="store_true",
                        help="strictly reject NatNet bodies marked tracking-invalid")
    parser.add_argument("--missing_body_hold", type=float, default=0.20,
                        help="seconds to reuse the last valid value for a missing body")
    parser.add_argument("--timestamped_redis", action="store_true",
                        help="publish dict messages for v2 consumers; default is TWIST-compatible list")
    parser.add_argument("--dry_run", action="store_true",
                        help="run OptiTrack/GMR diagnostics without writing Redis")
    parser.add_argument("--debug_arms", action="store_true",
                        help="Print raw GMR arm angles every 100 frames")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    server = OptiTrackRedisServer(args)
    try:
        server.run(freq=args.freq, debug_arms=args.debug_arms)
    except KeyboardInterrupt:
        print("\n[Server] Interrupted.")
    finally:
        try:
            server.safe_stand(freq=args.freq, seconds=1.0)
        except Exception as exc:
            print(f"[Safe Stand][WARN] {exc!r}")
        try:
            server.client.shutdown()
        except Exception:
            pass

if __name__ == "__main__":
    main()
