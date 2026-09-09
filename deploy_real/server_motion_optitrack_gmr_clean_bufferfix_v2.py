"""
v2 修订（相对 server_motion_optitrack_gmr_clean_bufferfix.py 的改动）：
  1. **真实 yaw_rate 进 obs**：原版把 obs[7] (root_ang_vel_yaw_relative) 写死为 0，
     policy 因此一直以为 root 不在转身，导致下肢平衡控制失误。改成对相邻帧
     的 root_rot 做 quat-diff 求 yaw rate。
  2. **yaw lock to init**：参考 server_high_level_motion_lib_v2.py，把发布给
     policy 的 yaw 改为相对于 teleop 启动时的第 0 帧 yaw，避免 OptiTrack
     第 0 帧朝向 OOD 把 policy 拖偏。
  3. **root_z 实时低通**：对 root_z（obs 第 0 维）做一阶 1.5 Hz 低通，吸收
     OptiTrack 单帧抖动；离线 mocap 我们用了同样的处理。
  4. **真正的 warmup**：原 safe_stand_mode 从 last_mimic_obs 插值到
     DEFAULT_MIMIC_OBS（站立姿）。修复后 teleop 进入时再做一次 2s warmup，
     从 DEFAULT 平滑过渡到当前 OptiTrack 第 0 帧 mimic_obs，避免 PD 力矩
     瞬间饱和。
"""

import argparse
import json
import os
import threading
import time
from collections import deque

import mujoco
import mujoco.viewer
import numpy as np
import redis
from rich import print

# GMR imports
from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting as GMR
from general_motion_retargeting.optitrack_vendor.NatNetClient import setup_optitrack
from general_motion_retargeting.utils.realtime_filter import (
    RealtimeMotionFilter, BufferSmoother,
)

# TWIST utils
from data_utils.params import DEFAULT_MIMIC_OBS, DEX31_QPOS_OPEN, DEX31_QPOS_CLOSE
from data_utils.rot_utils import quatToEuler, quat_rotate_inverse
from loop_rate_limiters import RateLimiter


# -----------------------------
# Quaternion / yaw utilities
# -----------------------------
def _wrap_to_pi(x):
    return float(np.arctan2(np.sin(x), np.cos(x)))


def _quat_wxyz_to_yaw(q_wxyz):
    """root_rot 在 GMR 输出里是 wxyz；返回 yaw (rad)。"""
    w, x, y, z = float(q_wxyz[0]), float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3])
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny, cosy))


def _quat_diff_yaw_rate(q0_wxyz, q1_wxyz, dt):
    """两帧 root_rot (wxyz) 的差分 -> body frame 下的 yaw rate (rad/s)。
    用 yaw 直接差分再 unwrap 一次，对小步长 dt 已经足够稳。"""
    if dt <= 1e-6:
        return 0.0
    yaw0 = _quat_wxyz_to_yaw(q0_wxyz)
    yaw1 = _quat_wxyz_to_yaw(q1_wxyz)
    dyaw = _wrap_to_pi(yaw1 - yaw0)
    return float(dyaw / dt)


# 一阶 IIR 低通：alpha = dt / (rc + dt)，rc = 1 / (2π fc)
def _lowpass_1st(prev, x, dt, fc_hz):
    if fc_hz is None or fc_hz <= 0:
        return float(x)
    rc = 1.0 / (2.0 * np.pi * fc_hz)
    alpha = dt / (rc + dt)
    return float((1.0 - alpha) * prev + alpha * x)


# -----------------------------
# Real-time filtering utilities
# -----------------------------
def _slerp_update(prev_q_wxyz, new_q_wxyz, alpha):
    """Slerp between prev_q and new_q (both wxyz). alpha in [0,1] = weight for new.

    Uses the standard quaternion slerp formula.  When the two quaternions are
    very close the function falls back to lerp + renormalise to avoid numerical
    issues around arccos(1).
    """
    prev_q = np.asarray(prev_q_wxyz, dtype=float)
    new_q = np.asarray(new_q_wxyz, dtype=float)

    # Ensure same hemisphere
    dot = float(np.dot(prev_q, new_q))
    if dot < 0.0:
        new_q = -new_q
        dot = -dot

    # If very close, use lerp + renormalise
    if dot > 0.9995:
        result = prev_q + alpha * (new_q - prev_q)
        norm = float(np.linalg.norm(result))
        if norm < 1e-10:
            return prev_q.copy()
        return result / norm

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * alpha
    sin_theta = np.sin(theta)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return s0 * prev_q + s1 * new_q


def _ema_update(prev, new, alpha):
    """Exponential moving average: result = (1-alpha)*prev + alpha*new."""
    return (1.0 - alpha) * np.asarray(prev, dtype=float) + alpha * np.asarray(new, dtype=float)


def _filter_mocap_frame(raw_frame, prev_state, alpha_pos=0.35, alpha_rot=0.30):
    """Apply EMA to body positions and slerp to body quaternions.

    This is a **causal** filter (only past data), suitable for real-time
    streaming.  It reduces OptiTrack camera jitter *before* the data enters
    GMR, which is the single most effective place to suppress IK noise.

    Args:
        raw_frame:  dict  body_name → [pos(3,),  quat(4, wxyz)]
        prev_state: dict  body_name → [pos(3,),  quat(4, wxyz)]  or None
        alpha_pos:  EMA alpha for positions  (0-1; higher = more responsive)
        alpha_rot:  EMA alpha for rotations

    Returns:
        filtered_frame: same format as raw_frame
        new_state:      updated prev_state for the next call
    """
    filtered = {}
    new_state = {}

    for body_name in raw_frame.keys():
        raw_pos, raw_quat = raw_frame[body_name]
        raw_pos = np.asarray(raw_pos, dtype=float)
        raw_quat = np.asarray(raw_quat, dtype=float)

        if prev_state is not None and body_name in prev_state:
            prev_pos, prev_quat = prev_state[body_name]
            filt_pos = _ema_update(prev_pos, raw_pos, alpha_pos)
            filt_quat = _slerp_update(prev_quat, raw_quat, alpha_rot)
        else:
            filt_pos = raw_pos.copy()
            filt_quat = raw_quat.copy()

        filtered[body_name] = [filt_pos, filt_quat]
        new_state[body_name] = [filt_pos.copy(), filt_quat.copy()]

    return filtered, new_state


def _compute_velocity_from_history(qpos_history, dt_per_frame):
    """Compute velocity via linear regression over a window of qpos values.

    Much more stable than single-frame finite differences when the input is
    noisy.  Equivalent to a Savitzky-Golay first-derivative filter with a
    linear polynomial.

    Args:
        qpos_history:  list / deque of qpos arrays  (oldest first)
        dt_per_frame:  nominal seconds between consecutive entries

    Returns:
        qdot:  velocity array, same shape as each qpos entry
    """
    n = len(qpos_history)
    if n < 2:
        return np.zeros_like(qpos_history[-1])
    if n == 2:
        return (np.asarray(qpos_history[-1]) - np.asarray(qpos_history[-2])) / max(dt_per_frame, 1e-6)

    t = np.arange(n, dtype=float) * dt_per_frame
    t_mean = float(np.mean(t))
    qpos_arr = np.asarray(qpos_history, dtype=float)          # (n, d)
    q_mean = np.mean(qpos_arr, axis=0)                         # (d,)

    t_diff = t - t_mean                                        # (n,)
    denom = float(np.sum(t_diff ** 2))
    if denom < 1e-10:
        return np.zeros_like(q_mean)

    slope = np.sum(t_diff[:, None] * (qpos_arr - q_mean), axis=0) / denom  # (d,)
    return slope


def _filter_qpos(qpos_raw, prev_state, *,
                 alpha_root_xy=0.20,
                 alpha_root_z=None,
                 alpha_root_rot=0.30,
                 alpha_dof=0.25):
    """Post-GMR causal low-pass on a single qpos vector.

    - root_pos x,y  → mild EMA  (horizontal drift is real signal)
    - root_pos z    → caller handles this separately via 1.5 Hz lowpass
    - root_rot      → slerp
    - dof_pos       → per-element EMA

    Args:
        qpos_raw:   raw qpos  [root_pos(3), root_rot(4,wxyz), dof_pos(D)]
        prev_state: previous filtered qpos (same shape) or None
        alpha_*:    EMA / slerp coefficients

    Returns:
        qpos_filt:  filtered qpos
        new_state:  updated prev_state
    """
    qpos_raw = np.asarray(qpos_raw, dtype=float)
    if prev_state is None:
        return qpos_raw.copy(), qpos_raw.copy()

    prev = np.asarray(prev_state, dtype=float)
    out = qpos_raw.copy()

    # root_pos x,y — mild EMA
    out[0] = _ema_update(prev[0], qpos_raw[0], alpha_root_xy)
    out[1] = _ema_update(prev[1], qpos_raw[1], alpha_root_xy)

    # root_pos z — caller handles externally; keep raw here
    if alpha_root_z is not None:
        out[2] = _ema_update(prev[2], qpos_raw[2], alpha_root_z)

    # root_rot — slerp
    out[3:7] = _slerp_update(prev[3:7], qpos_raw[3:7], alpha_root_rot)

    # dof_pos — per-element EMA
    out[7:] = _ema_update(prev[7:], qpos_raw[7:], alpha_dof)

    return out, out.copy()


# -----------------------------
# Streaming real-time qpos cleaner
# -----------------------------
class StreamingQposCleaner:
    """Real-time causal version of the offline ``clean_qpos`` pipeline.

    Applies, frame by frame:
      1. Butterworth low-pass (causal, 4th-order, configurable cutoff)
      2. Frame-to-frame dof velocity clipping
      3. Joint-limit clipping with configurable margin

    This makes the real-time retargeting quality approach the offline
    post-processed quality, reducing foot sliding and unstable standing.
    """

    def __init__(
        self,
        model,
        fps: float,
        lowpass_hz: float = 5.0,
        max_dof_vel: float = 30.0,
        limit_margin_deg: float = 5.0,
    ):
        self.model = model
        self.fps = float(fps)
        self.lowpass_hz = float(lowpass_hz) if lowpass_hz else 0.0
        self.max_dof_vel = float(max_dof_vel) if max_dof_vel else 0.0
        self.limit_margin_deg = float(limit_margin_deg)

        self._prev_qpos = None
        self._prev_quat_wxyz = None  # for streaming quat unflip

        # Butterworth filter state — created lazily when first frame arrives
        self._sos_pos = None
        self._zi_pos = None
        self._sos_quat = None
        self._zi_quat = None
        self._sos_dof = None
        self._zi_dof = None

        # Joint limits from model
        self._lower_dof = None
        self._upper_dof = None
        if model is not None:
            self._lower_dof, self._upper_dof = _read_dof_limits_from_model(model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def __call__(self, qpos: np.ndarray) -> np.ndarray:
        """Process a single qpos frame.  Returns the cleaned qpos."""
        out = np.asarray(qpos, dtype=np.float64).copy()

        # 1. Butterworth low-pass (position, quaternion, and dof)
        if self.lowpass_hz > 0:
            out = self._butter_step(out)

        # 2. Velocity clip
        if self.max_dof_vel > 0 and self._prev_qpos is not None:
            out = self._vel_clip_step(out)

        # 3. Limit clip
        if self._lower_dof is not None:
            out = self._limit_clip_step(out)

        self._prev_qpos = out.copy()
        return out

    def reset(self):
        """Reset internal filter state (call when teleop restarts)."""
        self._prev_qpos = None
        self._prev_quat_wxyz = None
        self._zi_pos = None
        self._zi_quat = None
        self._zi_dof = None

    # ------------------------------------------------------------------
    # Butterworth low-pass (causal, SOS-based)
    # ------------------------------------------------------------------
    def _butter_step(self, qpos: np.ndarray) -> np.ndarray:
        from scipy.signal import butter, sosfilt, sosfilt_zi

        out = qpos.copy()

        # --- Position (x, y, z) ---
        pos = qpos[0:3].reshape(1, 3)  # (1 time sample, 3 channels)
        if self._sos_pos is None:
            self._sos_pos = butter(4, self.lowpass_hz, 'low', fs=self.fps, output='sos')
            # zi: (n_sections, 2) → (n_sections, 2, 1) → broadcast with pos[0] (3,) → (n_sections, 2, 3)
            self._zi_pos = sosfilt_zi(self._sos_pos)[:, :, np.newaxis] * pos[0]
        pos_filt, self._zi_pos = sosfilt(self._sos_pos, pos, axis=0, zi=self._zi_pos)
        out[0:3] = pos_filt.ravel()

        # --- Quaternion (w, x, y, z) — unflip first ---
        quat = qpos[3:7].copy()
        if self._prev_quat_wxyz is not None:
            if np.dot(quat, self._prev_quat_wxyz) < 0:
                quat = -quat
        self._prev_quat_wxyz = quat.copy()

        quat = quat.reshape(1, 4)  # (1 time sample, 4 channels)
        if self._sos_quat is None:
            self._sos_quat = butter(4, self.lowpass_hz, 'low', fs=self.fps, output='sos')
            self._zi_quat = sosfilt_zi(self._sos_quat)[:, :, np.newaxis] * quat[0]
        quat_filt, self._zi_quat = sosfilt(self._sos_quat, quat, axis=0, zi=self._zi_quat)
        quat_filt = quat_filt.ravel()
        norm = float(np.linalg.norm(quat_filt))
        out[3:7] = quat_filt / max(norm, 1e-10)

        # --- Dof positions ---
        if len(qpos) > 7:
            dof = qpos[7:].reshape(1, -1)  # (1 time sample, N channels)
            if self._sos_dof is None:
                self._sos_dof = butter(4, self.lowpass_hz, 'low', fs=self.fps, output='sos')
                self._zi_dof = sosfilt_zi(self._sos_dof)[:, :, np.newaxis] * dof[0]
            dof_filt, self._zi_dof = sosfilt(self._sos_dof, dof, axis=0, zi=self._zi_dof)
            out[7:] = dof_filt.ravel()

        return out

    # ------------------------------------------------------------------
    # Velocity clipping
    # ------------------------------------------------------------------
    def _vel_clip_step(self, qpos: np.ndarray) -> np.ndarray:
        out = qpos.copy()
        if len(qpos) <= 7 or self._prev_qpos is None:
            return out
        dq_max = self.max_dof_vel / self.fps
        delta = out[7:] - self._prev_qpos[7:]
        clipped = np.clip(delta, -dq_max, dq_max)
        out[7:] = self._prev_qpos[7:] + clipped
        return out

    # ------------------------------------------------------------------
    # Joint-limit clipping
    # ------------------------------------------------------------------
    def _limit_clip_step(self, qpos: np.ndarray) -> np.ndarray:
        out = qpos.copy()
        if len(qpos) <= 7 or self._lower_dof is None:
            return out
        n_dof = len(qpos) - 7
        margin_rad = float(np.deg2rad(self.limit_margin_deg))
        lo = self._lower_dof[:n_dof] + margin_rad
        hi = self._upper_dof[:n_dof] - margin_rad
        bad = lo > hi
        lo[bad] = -np.inf
        hi[bad] = np.inf
        out[7:] = np.clip(out[7:], lo, hi)
        return out


def _read_dof_limits_from_model(model):
    """Read joint position limits from a MuJoCo model.

    Returns (lower, upper), each of shape (nv,).  Joints without explicit
    limits get ±inf.
    """
    nv = model.nv
    lower = np.full(nv, -np.inf, dtype=np.float64)
    upper = np.full(nv, np.inf, dtype=np.float64)
    for jid in range(model.njnt):
        jnt_type = model.jnt_type[jid]
        if jnt_type != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        qpos_adr = model.jnt_qposadr[jid]
        dof_adr = model.jnt_dofadr[jid]
        if dof_adr < 0 or dof_adr >= nv:
            continue
        if model.jnt_limited[jid]:
            lower[dof_adr] = model.jnt_range[jid][0]
            upper[dof_adr] = model.jnt_range[jid][1]
    return lower, upper


# -----------------------------
# Optional / missing deps stubs
# -----------------------------
try:
    from robot_control.joycon_wrapper import JoyConController
except Exception:
    class JoyConController:
        def get_state(self):
            return {
                "right": {"x": 1, "b": 0, "zr": 0},
                "left": {"zl": 0},
            }


try:
    from robot_control.speaker import Speaker
except Exception:
    class Speaker:
        def speak(self, text):
            print(f"[Speaker disabled] {text}")


# -----------------------------
# Constants
# -----------------------------
BUFFER_START_IDX = 0
LOOKAHEAD = 2
SEED_BUFFER_SAMPLES = 3

# 1(root_height) + 3(rpy) + 3(root_vel_relative) + 1(root_ang_vel_yaw)
MIMIC_OBS_FIXED_DIM = 8

# GMR 29-dof -> old policy 25-dof
# old policy dof order:
# 0-5   left leg
# 6-11  right leg
# 12-14 waist
# 15-19 left arm   = shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll
# 20-24 right arm  = shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll
POLICY_DOF_INDEX = {
    "g1": np.array([
        0, 1, 2, 3, 4, 5,
        6, 7, 8, 9, 10, 11,
        12, 13, 14,
        15, 16, 17, 18, 19,
        22, 23, 24, 25, 26,
    ], dtype=int),
    "t1": np.array([
        0, 1, 2, 3, 4, 5,
        6, 7, 8, 9, 10, 11,
        12, 13, 14,
        15, 16, 17, 18, 19,
        22, 23, 24, 25, 26,
    ], dtype=int),
}

# GMR qpos -> vis qpos remap
VIS_DOF_INDEX = {
    "g1": np.array([
        0, 1, 2, 3, 4, 5,
        6, 7, 8, 9, 10, 11,
        12, 13, 14,
        15, 16, 17, 18, 19,
        22, 23, 24, 25, 26,
    ], dtype=int),
    "t1": np.array([
        0, 1, 2, 3, 4, 5,
        6, 7, 8, 9, 10, 11,
        12, 13, 14,
        15, 16, 17, 18, 19,
        22, 23, 24, 25, 26,
    ], dtype=int),
}

DOF_SIGN = {
    "g1": np.array([
        1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1,
        1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
    ], dtype=float),
    "t1": np.array([
        1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1,
        1, 1, 1,
        1, 1, 1, 1, 1,
        1, -1, -1, 1, 1,
    ], dtype=float),
}

DOF_OFFSET = {
    "g1": np.zeros(25, dtype=float),
    "t1": np.zeros(25, dtype=float),
}

# =========================
# TEST MODE CONFIG
# =========================
TEST_MODE = False

TEST_JOINT_IDX = 21
TEST_DELTA = 0.5

# -----------------------------
# Small helpers
# -----------------------------
def get_id_data(times, qpos, t0=None):
    times = np.asarray(times, dtype=float)
    qpos = np.asarray(qpos, dtype=float)

    if len(times) == 0:
        raise ValueError("times is empty")
    if len(times) != len(qpos):
        raise ValueError("times and qpos length mismatch")

    if t0 is None:
        t0 = times[-1]

    if len(times) == 1:
        return qpos[0].copy(), np.zeros_like(qpos[0])

    idx = np.searchsorted(times, t0)
    idx = int(np.clip(idx, 1, len(times) - 1))
    t1, t2 = times[idx - 1], times[idx]
    q1, q2 = qpos[idx - 1], qpos[idx]

    if abs(t2 - t1) < 1e-8:
        alpha = 0.0
        q0 = q2.copy()
        qdot = np.zeros_like(q2)
    else:
        alpha = np.clip((t0 - t1) / (t2 - t1), 0.0, 1.0)
        q0 = (1 - alpha) * q1 + alpha * q2
        qdot = (q2 - q1) / (t2 - t1)

    return q0, qdot


def get_target_mimic_obs_dim(robot_type):
    return len(DEFAULT_MIMIC_OBS[robot_type])


def get_target_dof_dim(robot_type):
    target_obs_dim = get_target_mimic_obs_dim(robot_type)
    dof_dim = target_obs_dim - MIMIC_OBS_FIXED_DIM
    if dof_dim <= 0:
        raise ValueError(f"Invalid target obs dim for {robot_type}: {target_obs_dim}")
    return dof_dim


def remap_and_fix_dof(qpos, robot_type):
    dof_pos_all = qpos[7:]
    dof_index = POLICY_DOF_INDEX[robot_type]

    max_idx = int(np.max(dof_index))
    if len(dof_pos_all) <= max_idx:
        raise ValueError(
            f"dof_pos dim too small for {robot_type}: "
            f"got {len(dof_pos_all)}, need index up to {max_idx}"
        )

    dof_pos = dof_pos_all[dof_index].copy()

    expected_dof_dim = get_target_dof_dim(robot_type)
    if len(dof_pos) != expected_dof_dim:
        raise ValueError(
            f"dof_pos remap dim mismatch for {robot_type}: "
            f"got {len(dof_pos)}, expected {expected_dof_dim}"
        )

    dof_pos = dof_pos * DOF_SIGN[robot_type] + DOF_OFFSET[robot_type]
    return dof_pos


def remap_qpos_for_vis(qpos, robot_type, sim_qpos_dim):
    qpos = np.asarray(qpos, dtype=float)
    if len(qpos) < 7:
        raise ValueError(f"qpos too short: {len(qpos)}")

    out = np.zeros(sim_qpos_dim, dtype=float)

    root_dim = min(7, sim_qpos_dim)
    out[:root_dim] = qpos[:root_dim]

    if sim_qpos_dim <= 7:
        return out

    dof_src = qpos[7:]
    dof_idx = VIS_DOF_INDEX[robot_type]

    max_idx = int(np.max(dof_idx))
    if len(dof_src) <= max_idx:
        raise ValueError(
            f"vis remap source dof too short for {robot_type}: "
            f"got {len(dof_src)}, need index up to {max_idx}"
        )

    dof_dst = dof_src[dof_idx]

    needed = sim_qpos_dim - 7
    if len(dof_dst) < needed:
        raise ValueError(
            f"vis remap dst dof too short: got {len(dof_dst)}, need {needed}"
        )

    out[7:sim_qpos_dim] = dof_dst[:needed]
    return out


def _get_mimic_obs(
    qpos,
    qdot,
    robot_type,
    *,
    prev_root_rot_wxyz=None,
    dt=None,
    init_yaw=0.0,
    root_z_filtered=None,
):
    """build mimic_obs from instantaneous (qpos, qdot).

    Extra v2 args:
      prev_root_rot_wxyz, dt :  用于真实 yaw_rate 计算（取代写死 0）。
      init_yaw                :  把当前 yaw 锁到相对值，避免 OptiTrack 启动时
                                  机器人朝向 != 0 引起的 OOD。
      root_z_filtered         :  若给出，用它替换 raw root_z，作为 obs 第 0 维。
                                  调用方负责喂入 1 阶低通后的 z。
    """
    root_pos = qpos[:3]
    root_rot = qpos[3:7]  # wxyz (GMR 输出)

    dof_pos = remap_and_fix_dof(qpos, robot_type)

    roll, pitch, yaw = quatToEuler(root_rot)
    yaw = _wrap_to_pi(float(yaw) - init_yaw)

    # root_z（obs 第 0 维）。
    if root_z_filtered is not None:
        root_height = np.array([float(root_z_filtered)], dtype=float)
    else:
        root_height = root_pos[2:3]

    # root linear vel (world) -> rotate into root frame.
    root_vel = qdot[:3]
    root_rot_xyzw = root_rot.reshape(1, 4)[:, [1, 2, 3, 0]]
    root_vel = root_vel.reshape(1, 3)
    root_vel_relative = quat_rotate_inverse(root_rot_xyzw, root_vel).reshape(3)

    # **真正的** yaw_rate（之前写死 0 是 deploy 端最大的 bug 之一）。
    if prev_root_rot_wxyz is not None and dt is not None and dt > 1e-6:
        yaw_rate = _quat_diff_yaw_rate(prev_root_rot_wxyz, root_rot, dt)
    else:
        yaw_rate = 0.0
    root_ang_vel_relative_yaw = np.array([yaw_rate], dtype=float)

    mimic_obs = np.concatenate([
        root_height,
        [roll, pitch, yaw],
        root_vel_relative,
        root_ang_vel_relative_yaw,
        dof_pos,
    ])

    expected_dim = len(DEFAULT_MIMIC_OBS[robot_type])
    if len(mimic_obs) != expected_dim:
        raise ValueError(
            f"mimic_obs dim mismatch for {robot_type}: "
            f"got {len(mimic_obs)}, expected {expected_dim}"
        )

    return mimic_obs


def make_test_mimic_obs(robot_type="g1", test_joint_idx=None, test_delta=0.0):
    obs = DEFAULT_MIMIC_OBS[robot_type].copy()

    if test_joint_idx is not None:
        if not (0 <= test_joint_idx < 25):
            raise ValueError(f"test_joint_idx must be in [0, 24], got {test_joint_idx}")
        obs[8 + test_joint_idx] += test_delta

    return obs


# -----------------------------
# Data buffer driven by GMR
# -----------------------------
class GMRDataBuffer:
    def __init__(self, client, robot_type, actual_human_height=1.6, frame_rate_hz=120.0):
        self.client = client
        self.robot_type = robot_type
        self.frame_rate_hz = frame_rate_hz

        if robot_type == "g1":
            tgt_robot = "unitree_g1"
        elif robot_type == "t1":
            tgt_robot = "unitree_t1"
        else:
            raise ValueError(f"Unsupported robot type for GMR: {robot_type}")

        self.gmr = GMR(
            src_human="fbx",
            tgt_robot=tgt_robot,
            actual_human_height=actual_human_height,
            # Match GMR optitrack_to_robot.py defaults exactly:
            # damping=0.5, no velocity limits, both IK tables.
        )

        # Re-enable table-1: GMR's RealtimeMotionFilter handles flips via
        # outlier clamping + flip detection, so the IK can safely use both
        # table1 (rotation) and table2 (position+rotation) for better tracking.
        print("[GMR] Using both ik_match_table1 + table2 "
              "(pre-IK outlier rejection prevents arm flips).")

        self._buffer = []

        # --- Pre-GMR mocap filtering (GMR's RealtimeMotionFilter) ---
        # This is the SAME filter used by GMR's optitrack_to_robot.py.
        # Key features: per-body EMA/SLERP, outlier clamping, flip detection,
        # wrist roll constraint, quaternion averaging for hands.
        self._filter_mocap = True
        self._motion_filter = RealtimeMotionFilter(
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
        self._median_filter = BufferSmoother(window_size=3, mode='median')
        self._prev_qpos_raw = None  # velocity clip state

    def reset_gmr_configuration(self):
        """Re-initialize the GMR IK configuration to the model default pose.

        Call this before seeding when entering teleop mode to avoid carrying
        over a flipped/diverged IK state from a previous session.
        """
        import mink
        self.gmr.configuration = mink.Configuration(self.gmr.model)

    def reset_mocap_filters(self):
        """Reset the GMR RealtimeMotionFilter and BufferSmoother state."""
        self._motion_filter.reset()
        self._median_filter.reset()
        self._prev_qpos_raw = None
        self._skip_count = 0
        # Clear Butterworth filter state
        for attr in ('_bw_zi_pos', '_bw_zi_quat', '_bw_zi_legs', '_bw_prev_quat'):
            if hasattr(self, attr):
                setattr(self, attr, None)

    def set_mocap_filter(self, enable, alpha_pos=None, alpha_rot=None):
        """Enable/disable pre-GMR mocap filtering on the fly."""
        self._filter_mocap = enable
        if alpha_pos is not None:
            self._motion_filter.pos_alpha = alpha_pos
        if alpha_rot is not None:
            self._motion_filter.rot_alpha = alpha_rot
        if not enable:
            self._motion_filter.reset()
            self._median_filter.reset()

    def set_qpos_filter(self, enable, alpha_root_xy=None, alpha_root_rot=None, alpha_dof=None):
        """Configure post-GMR qpos filter.

        Set any alpha to 1.0 to effectively disable that component.
        Set enable=False + any alpha=1.0 to bypass all filtering.
        """
        if alpha_root_xy is not None:
            self._qpos_alpha_root_xy = alpha_root_xy
        if alpha_root_rot is not None:
            self._qpos_alpha_root_rot = alpha_root_rot
        if alpha_dof is not None:
            self._qpos_alpha_dof = alpha_dof
        if not enable:
            self._qpos_filt_prev = None

    def push(self):
        try:
            mocap_frame = self.client.get_frame(timeout=2.0)
            if mocap_frame is None:
                print("[WARN] get_frame timed out — no OptiTrack data")
                return False
        except Exception as e:
            print(f"[WARN] get_frame error: {repr(e)}")
            return False

        # Skip frames that lost skeleton tracking (no root bone)
        if not mocap_frame or 'Hips' not in mocap_frame:
            if not hasattr(self, '_skip_count'):
                self._skip_count = 0
            self._skip_count += 1
            if self._skip_count <= 3 or self._skip_count % 100 == 0:
                print(f"[WARN] Frame missing 'Hips' (tracking lost?), "
                      f"skipping (#{self._skip_count})")
            return False

        # --- Pre-GMR mocap filtering (GMR RealtimeMotionFilter) ---
        if self._filter_mocap:
            mocap_frame = self._motion_filter(mocap_frame)
            mocap_frame = self._median_filter(mocap_frame)

        frame_number = self.client.get_frame_number()
        qpos = self.gmr.retarget(mocap_frame)

        # --- Post-IK processing (matches offline clean_qpos, real-time causal) ---
        # 1. Targeted lowpass: root + legs + waist only (NOT arms — keep hands responsive)
        # 2. Velocity clip: prevents >30 rad/s single-frame jumps
        # 3. Joint-limit clip: keeps joints within model limits
        qpos = self._postprocess_qpos(qpos)

        timestamp = frame_number / self.frame_rate_hz
        self._buffer.append((timestamp, np.array(qpos, dtype=float).copy(), dict(mocap_frame)))
        return True

    def _postprocess_qpos(self, qpos):
        """Post-IK processing matching offline clean_qpos, adapted for streaming.

        1. 2nd-order Butterworth lowpass at 8 Hz on root + legs + waist
           (NOT arms — keep hands expressive, matching offline fidelity)
        2. Velocity clip at 30 rad/s
        3. Joint-limit clip at 5 deg margin
        """
        out = np.asarray(qpos, dtype=float).copy()

        # --- 1. Targeted Butterworth (root + legs + waist only) ---
        out = self._butter_stability(out)

        # --- 2. Velocity clip ---
        if self._prev_qpos_raw is not None and len(out) > 7:
            dq_max = 30.0 / 50.0  # 30 rad/s at 50 fps
            delta = out[7:] - self._prev_qpos_raw[7:]
            out[7:] = self._prev_qpos_raw[7:] + np.clip(delta, -dq_max, dq_max)
        self._prev_qpos_raw = out.copy()

        # --- 3. Joint-limit clip ---
        if len(out) > 7:
            out = self._limit_clip_qpos(out)
        return out

    def _butter_stability(self, qpos):
        """2nd-order Butterworth at 8 Hz — causal, ~10 ms delay.

        Filters root position (0:3), root rotation (3:7), and leg+waist
        dofs (7:22).  Arm dofs (22:36) pass through unfiltered so hand
        tracking stays crisp.
        """
        from scipy.signal import butter, sosfilt, sosfilt_zi
        out = qpos.copy()
        fps = 50.0
        cutoff = 8.0
        n_dof = len(qpos) - 7  # 29 for GMR model

        # Lazy init filter state
        if not hasattr(self, '_bw_sos_pos'):
            self._bw_sos_pos = butter(2, cutoff, 'low', fs=fps, output='sos')
            self._bw_zi_pos = None
            self._bw_sos_quat = butter(2, cutoff, 'low', fs=fps, output='sos')
            self._bw_zi_quat = None
            self._bw_prev_quat = None
            self._bw_sos_legs = butter(2, cutoff, 'low', fs=fps, output='sos')
            self._bw_zi_legs = None

        # Root position (x, y, z) — 3 channels
        pos = qpos[0:3].reshape(1, 3)
        if self._bw_zi_pos is None:
            self._bw_zi_pos = sosfilt_zi(self._bw_sos_pos)[:, :, np.newaxis] * pos[0]
        pos_f, self._bw_zi_pos = sosfilt(self._bw_sos_pos, pos, axis=0, zi=self._bw_zi_pos)
        out[0:3] = pos_f.ravel()

        # Root rotation (w, x, y, z) — unflip then filter
        quat = qpos[3:7].copy()
        if self._bw_prev_quat is not None:
            if np.dot(quat, self._bw_prev_quat) < 0:
                quat = -quat
        self._bw_prev_quat = quat.copy()
        quat = quat.reshape(1, 4)
        if self._bw_zi_quat is None:
            self._bw_zi_quat = sosfilt_zi(self._bw_sos_quat)[:, :, np.newaxis] * quat[0]
        quat_f, self._bw_zi_quat = sosfilt(self._bw_sos_quat, quat, axis=0, zi=self._bw_zi_quat)
        quat_f = quat_f.ravel()
        out[3:7] = quat_f / max(np.linalg.norm(quat_f), 1e-10)

        # Leg + waist dofs only (GMR dof indices 0-14 = qpos[7:22])
        # Arm dofs (GMR indices 15-28 = qpos[22:36]) pass through unfiltered
        if n_dof > 0:
            leg_waist_end = min(15, n_dof)  # 15 dofs: 6 left_leg + 6 right_leg + 3 waist
            if leg_waist_end > 0:
                legs = qpos[7:7 + leg_waist_end].reshape(1, -1)
                if self._bw_zi_legs is None:
                    self._bw_zi_legs = sosfilt_zi(self._bw_sos_legs)[:, :, np.newaxis] * legs[0]
                legs_f, self._bw_zi_legs = sosfilt(self._bw_sos_legs, legs, axis=0, zi=self._bw_zi_legs)
                out[7:7 + leg_waist_end] = legs_f.ravel()
            # Arm dofs: keep as-is (no filtering)

        return out

    def _limit_clip_qpos(self, qpos):
        """Clip dof_pos to model joint limits with 5 deg margin."""
        out = np.asarray(qpos, dtype=float).copy()
        n_dof = len(out) - 7
        lower = np.full(n_dof, -np.inf)
        upper = np.full(n_dof, np.inf)
        for jid in range(self.gmr.model.njnt):
            if self.gmr.model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            dof_adr = self.gmr.model.jnt_dofadr[jid]
            if dof_adr < 0 or dof_adr >= n_dof:
                continue
            if self.gmr.model.jnt_limited[jid]:
                lower[dof_adr] = self.gmr.model.jnt_range[jid][0]
                upper[dof_adr] = self.gmr.model.jnt_range[jid][1]
        margin = np.deg2rad(5.0)
        lo = lower + margin
        hi = upper - margin
        lo[lo > hi] = -np.inf
        hi[lo > hi] = np.inf
        out[7:] = np.clip(out[7:], lo, hi)
        return out

    def pushN(self, n, max_wait_s=10.0):
        start_len = self.length
        t0 = time.time()
        while self.length - start_len < n:
            ok = self.push()
            if not ok:
                if time.time() - t0 > max_wait_s:
                    raise TimeoutError(
                        f"GMRDataBuffer: no OptiTrack frames after {max_wait_s:.0f}s. "
                        f"Check that OptiTrack server is streaming (Motive is running and streaming frames)."
                    )
                time.sleep(0.02)

    def clear(self):
        self._buffer = []

    @property
    def length(self):
        return len(self._buffer)

    @property
    def times(self):
        return np.array([t for t, _, _ in self._buffer], dtype=float)

    @property
    def qpos(self):
        return np.array([q for _, q, _ in self._buffer], dtype=float)

    @property
    def mocap_data(self):
        return np.array([m for _, _, m in self._buffer], dtype=object)


# -----------------------------
# Server
# -----------------------------
class OptiTrackMimicObsServerGMR:
    def __init__(
        self,
        server_ip,
        client_ip,
        use_multicast,
        robot_type,
        xml_file,
        vis=False,
        use_hand=False,
        actual_human_height=1.6,
    ):
        self.robot_type = robot_type
        self.use_hand = use_hand
        self.vis = vis

        if not TEST_MODE:
            self.client = setup_optitrack(
                server_address=server_ip,
                client_address=client_ip,
                use_multicast=use_multicast,
            )
            thread = threading.Thread(target=self.client.run, daemon=True)
            thread.start()
            time.sleep(2)
            print(f"[DEBUG] client connected: {self.client.connected()}")
        else:
            self.client = None
            print("[TEST_MODE] Skipping OptiTrack connection")
        print("Client initialized")

        self.redis_client = redis.Redis(host="localhost", port=6379, db=0)
        self.vicon_data_buffer = GMRDataBuffer(
            self.client,
            robot_type=robot_type,
            actual_human_height=actual_human_height,
        )

        self.curr_buffer_idx = BUFFER_START_IDX
        self.last_mimic_obs = DEFAULT_MIMIC_OBS[self.robot_type].copy()
        self.joycon_controller = JoyConController()
        self.speaker = Speaker()
        self.qdot_history = deque(maxlen=10)
        self.qpos_history = deque(maxlen=11)  # for regression-based velocity estimation

        # ----- v2 state -----
        # 在 teleop 启动时记录 init_yaw / init_root_z 作 baseline。
        self.init_yaw = 0.0
        self.root_z_baseline = None       # 启动时的 OptiTrack root_z 平均（m）
        self.root_z_target_mean = 0.762   # G1 home pelvis z（与离线 mocap 对齐）
        self.root_z_filt = None           # 一阶低通后的 root_z
        self.prev_root_rot_wxyz = None    # 上一帧 root_rot，用于算 yaw_rate
        self.prev_t = None
        # root_z 低通截止；离线 mocap 我们也用了 ~1.5 Hz
        self.root_z_lpf_hz = 1.5

        if self.robot_type == "g1":
            self.robot_base = "pelvis"
        elif self.robot_type == "t1":
            self.robot_base = "Waist"
        else:
            self.robot_base = "pelvis"

        self.sim_model = None
        self.sim_data = None
        self.viewer = None
        if self.vis:
            self.sim_model = mujoco.MjModel.from_xml_path(xml_file)
            self.sim_data = mujoco.MjData(self.sim_model)
            self.viewer = mujoco.viewer.launch_passive(
                model=self.sim_model,
                data=self.sim_data,
                show_left_ui=False,
                show_right_ui=False,
            )
            print("Degrees of Freedom (DoF) names and their order:")
            for i in range(self.sim_model.nv):
                dof_name = mujoco.mj_id2name(
                    self.sim_model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    self.sim_model.dof_jntid[i]
                )
                print(f"DoF {i}: {dof_name}")
            print("Motor (Actuator) names and their IDs:")
            for i in range(self.sim_model.nu):
                motor_name = mujoco.mj_id2name(
                    self.sim_model,
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    i
                )
                print(f"Motor ID {i}: {motor_name}")

    def vis_qpos(self, qpos):
        if not self.vis or self.viewer is None:
            return

        try:
            vis_qpos = remap_qpos_for_vis(
                qpos=qpos,
                robot_type=self.robot_type,
                sim_qpos_dim=len(self.sim_data.qpos),
            )
        except Exception as e:
            print(f"[WARN] vis remap failed: {e}")
            return

        self.sim_data.qpos[:] = vis_qpos
        mujoco.mj_forward(self.sim_model, self.sim_data)
        robot_base_pos = self.sim_data.xpos[self.sim_model.body(self.robot_base).id]
        self.viewer.cam.lookat = robot_base_pos
        self.viewer.cam.distance = 2.0
        self.viewer.sync()

    def safe_stand_mode(self, freq=50, seconds=2.0):
        print("[Into safe stand mode...]")
        self.speaker.speak("Start Safe Standing Mode")

        target = DEFAULT_MIMIC_OBS[self.robot_type].copy()

        if len(self.last_mimic_obs) != len(target):
            print(
                f"[WARN] safe_stand_mode dim mismatch: "
                f"last={len(self.last_mimic_obs)}, target={len(target)}. "
                f"Resetting last_mimic_obs to target directly."
            )
            self.last_mimic_obs = target.copy()

        total_steps = max(int(seconds * freq), 1)
        start_obs = self.last_mimic_obs.copy()

        for i in range(total_steps):
            alpha = i / total_steps
            interp = start_obs + (target - start_obs) * alpha
            self.redis_client.set(
                f"action_mimic_{self.robot_type}",
                json.dumps(interp.tolist())
            )
            time.sleep(1.0 / freq)

        self.redis_client.set(
            f"action_mimic_{self.robot_type}",
            json.dumps(target.tolist())
        )
        self.last_mimic_obs = target.copy()
        print("[Finished safe stand mode...]")
        self.speaker.speak("Finished Safe Standing Mode")

    def safe_transition_mode(self, current_mimic_obs, target_mimic_obs, freq=50, seconds=1.0):
        print("[Into safe transition mode...]")
        self.speaker.speak("Start Transition Mode")

        if len(current_mimic_obs) != len(target_mimic_obs):
            raise ValueError(
                f"safe_transition_mode dim mismatch: "
                f"current={len(current_mimic_obs)}, target={len(target_mimic_obs)}"
            )

        total_steps = max(int(seconds * freq), 1)
        for i in range(total_steps):
            alpha = i / total_steps
            interp = current_mimic_obs + (target_mimic_obs - current_mimic_obs) * alpha
            self.redis_client.set(
                f"action_mimic_{self.robot_type}",
                json.dumps(interp.tolist())
            )
            time.sleep(1.0 / freq)

        self.redis_client.set(
            f"action_mimic_{self.robot_type}",
            json.dumps(target_mimic_obs.tolist())
        )
        self.speaker.speak("Finished Transition Mode")

    def teleop_mode(self, freq=50):
        print(f"[Teleop Mode] Start running at {freq} Hz ...")
        self.speaker.speak("Start Teleoperation Mode")
        rate = RateLimiter(frequency=freq, warn=False)

        step_count = 0

        if not TEST_MODE:
            print(f"[Teleop Mode] Resetting GMR configuration & filters ...")
            self.vicon_data_buffer.reset_gmr_configuration()
            self.vicon_data_buffer.reset_mocap_filters()
            print(f"[Teleop Mode] Seeding buffer with {SEED_BUFFER_SAMPLES} samples...")
            self.vicon_data_buffer.pushN(SEED_BUFFER_SAMPLES + LOOKAHEAD)
            self.curr_buffer_idx = BUFFER_START_IDX

            # ===== v2: init_yaw + root_z_baseline + warmup =====
            # 用 buffer 里最新的 SEED_BUFFER_SAMPLES 帧做平均，作为启动时姿态。
            qpos_arr = self.vicon_data_buffer.qpos[-SEED_BUFFER_SAMPLES:]
            init_root_rot_wxyz = qpos_arr[-1, 3:7]
            self.init_yaw = _quat_wxyz_to_yaw(init_root_rot_wxyz)
            print(f"[Teleop Mode] init_yaw locked at {self.init_yaw:+.4f} rad "
                  f"({np.degrees(self.init_yaw):+.1f} deg)")

            self.root_z_baseline = float(np.mean(qpos_arr[:, 2]))
            print(f"[Teleop Mode] OptiTrack root_z baseline = {self.root_z_baseline:.3f} m, "
                  f"target = {self.root_z_target_mean:.3f} m, "
                  f"shift = {self.root_z_target_mean - self.root_z_baseline:+.3f} m")

            # 用 init 帧构造 ref0 mimic_obs（warmup 终点）
            ref0_qpos = qpos_arr[-1]
            ref0_qdot = np.zeros_like(ref0_qpos)
            self.root_z_filt = self.root_z_target_mean  # warmup 起点已对齐
            ref0_mimic_obs = _get_mimic_obs(
                ref0_qpos,
                ref0_qdot,
                self.robot_type,
                prev_root_rot_wxyz=ref0_qpos[3:7],
                dt=1.0 / freq,
                init_yaw=self.init_yaw,
                root_z_filtered=self.root_z_target_mean,
            )

            # 2 秒 warmup：从当前 last_mimic_obs（safe stand 之后已经是 DEFAULT）
            # 平滑过渡到 OptiTrack 第 0 帧 mimic_obs。
            warmup_seconds = 2.0
            warmup_steps = max(int(warmup_seconds * freq), 1)
            warmup_start = self.last_mimic_obs.copy()
            print(f"[Teleop Mode] Warmup {warmup_seconds:.1f}s "
                  f"(DEFAULT -> first OptiTrack frame, {warmup_steps} steps) ...")
            for i in range(warmup_steps):
                alpha = (i + 1) / warmup_steps
                interp = warmup_start + (ref0_mimic_obs - warmup_start) * alpha
                self.redis_client.set(
                    f"action_mimic_{self.robot_type}",
                    json.dumps(interp.tolist()),
                )
                self.last_mimic_obs = interp
                rate.sleep()

            # 初始化 v2 内部状态
            self.prev_root_rot_wxyz = ref0_qpos[3:7].copy()
            self.prev_t = time.time()
            self.root_z_filt = self.root_z_target_mean

        while self.flag_teleop_mode and self.running:
            joycon_state = self.joycon_controller.get_state()
            self.flag_teleop_mode = joycon_state["right"]["x"] == 1
            if joycon_state["right"]["b"] == 1:
                self.running = False

            t_start = time.time()
            q0 = None

            if TEST_MODE:
                mimic_obs = make_test_mimic_obs(
                    robot_type=self.robot_type,
                    test_joint_idx=TEST_JOINT_IDX,
                    test_delta=TEST_DELTA,
                )
                print("[DEBUG] TEST MODE mimic_obs shape:", mimic_obs.shape, "first6:", mimic_obs[:6])
                print("[DEBUG] TEST MODE right_arm_dof:", mimic_obs[-5:])
                print(f"[DEBUG] TEST MODE joint_idx={TEST_JOINT_IDX}, delta={TEST_DELTA}")

            else:
                ok = self.vicon_data_buffer.push()
                if not ok:
                    rate.sleep()
                    continue

                times = self.vicon_data_buffer.times
                qpos_arr = self.vicon_data_buffer.qpos

                min_frames = max(LOOKAHEAD + 3, 5)
                if len(times) < min_frames:
                    print(f"[DEBUG] waiting buffer, len(times)={len(times)}")
                    rate.sleep()
                    continue

                window_size = 100
                end_idx = len(times)
                start_idx = max(0, end_idx - window_size)
                q0, qdot0 = get_id_data(
                    times[start_idx:end_idx],
                    qpos_arr[start_idx:end_idx],
                    t0=times[-2],
                )

                # --- Improved velocity estimation ---
                # Maintain a window of recent filtered qpos and compute
                # velocity via linear regression (equivalent to Savitzky-Golay
                # first-derivative filter).  Much more stable than 2-frame
                # finite differences followed by box averaging.
                self.qpos_history.append(q0.copy())
                dt_nominal = 1.0 / freq
                if len(self.qpos_history) >= 5:
                    qdot0_sm = _compute_velocity_from_history(
                        self.qpos_history, dt_nominal
                    )
                elif len(self.qpos_history) >= 2:
                    # Not enough frames for regression — fall back to simple diff
                    qdot0_sm = (
                        (self.qpos_history[-1] - self.qpos_history[-2])
                        / dt_nominal
                    )
                else:
                    qdot0_sm = qdot0

                # v2: 真实 dt + 真实 yaw_rate + root_z 低通 + yaw lock
                now = time.time()
                if self.prev_t is None:
                    dt_real = 1.0 / freq
                else:
                    dt_real = max(now - self.prev_t, 1.0 / (4.0 * freq))
                self.prev_t = now

                # 1) root_z 校正：减去 baseline 偏差，再一阶低通
                z_raw = float(q0[2])
                z_shifted = z_raw + (self.root_z_target_mean - self.root_z_baseline)
                if self.root_z_filt is None:
                    self.root_z_filt = z_shifted
                else:
                    self.root_z_filt = _lowpass_1st(
                        self.root_z_filt, z_shifted, dt_real, self.root_z_lpf_hz
                    )

                cur_root_rot_wxyz = q0[3:7].copy()
                mimic_obs = _get_mimic_obs(
                    q0,
                    qdot0_sm,
                    self.robot_type,
                    prev_root_rot_wxyz=self.prev_root_rot_wxyz,
                    dt=dt_real,
                    init_yaw=self.init_yaw,
                    root_z_filtered=self.root_z_filt,
                )
                self.prev_root_rot_wxyz = cur_root_rot_wxyz

            expected_dim = len(DEFAULT_MIMIC_OBS[self.robot_type])
            if len(mimic_obs) != expected_dim:
                raise ValueError(
                    f"teleop mimic_obs dim mismatch: got {len(mimic_obs)}, expected {expected_dim}"
                )

            if step_count == 0:
                self.last_mimic_obs = mimic_obs.copy()

            self.redis_client.set(
                f"action_mimic_{self.robot_type}",
                json.dumps(mimic_obs.tolist())
            )
            self.last_mimic_obs = mimic_obs.copy()

            if self.use_hand and self.robot_type == "g1":
                if joycon_state["left"]["zl"]:
                    left_q_target = DEX31_QPOS_OPEN["left"]
                else:
                    left_q_target = DEX31_QPOS_CLOSE["left"]

                if joycon_state["right"]["zr"]:
                    right_q_target = DEX31_QPOS_OPEN["right"]
                else:
                    right_q_target = DEX31_QPOS_CLOSE["right"]

                dex31_qpos = np.concatenate([left_q_target, right_q_target])
                self.redis_client.set(
                    f"action_hand_{self.robot_type}",
                    json.dumps(dex31_qpos.tolist())
                )

            step_count += 1

            if self.vis and (q0 is not None):
                self.vis_qpos(q0)

            rate.sleep()
            _fps = 1 / max(time.time() - t_start, 1e-6)

        self.speaker.speak("Finished Teleoperation Mode")

    def main_loop(self, freq=50):
        self.safe_stand_mode(freq=freq, seconds=2.0)
        print("=" * 50)
        print("[DEBUG] SAFE STAND FINISHED. ABOUT TO START MAIN LOOP.")
        print("=" * 50)

        self.running = True
        try:
            while self.running:
                print("[DEBUG] Top of main while loop")
                joycon_state = self.joycon_controller.get_state()
                self.flag_teleop_mode = joycon_state["right"]["x"] == 1
                if joycon_state["right"]["b"] == 1:
                    self.running = False

                if self.flag_teleop_mode:
                    try:
                        self.teleop_mode(freq)
                    except Exception as e:
                        print(f"exit teleop mode: {e}")
                        print("Into safe stand mode...")
                        self.safe_stand_mode(freq, seconds=3.0)
                        break
                else:
                    self.redis_client.set(
                        f"action_mimic_{self.robot_type}",
                        json.dumps(self.last_mimic_obs.tolist()),
                    )
                    time.sleep(1.0 / freq)
        except Exception as e:
            print(f"main loop error: {e}")

        self.safe_stand_mode(freq, seconds=3.0)


def main():
    parser = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--host", default="192.168.3.103", help="OptiTrack Server Address")
    parser.add_argument("--client_ip", default="192.168.3.176", help="Local client IP")
    parser.add_argument("--use_multicast", type=bool, default=True)
    parser.add_argument("--robot", default="g1", choices=["g1", "t1"])
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--use_hand", action="store_true")
    parser.add_argument(
        "--no_mocap_filter", action="store_true",
        help="Disable pre-GMR mocap body filtering (GMR RealtimeMotionFilter).",
    )
    # GMR RealtimeMotionFilter parameters (match optitrack_to_robot.py defaults)
    parser.add_argument(
        "--filter_pos_smooth", type=float, default=0.5,
        help="EMA factor for body position (0-1, higher=smoother). Default 0.5.",
    )
    parser.add_argument(
        "--filter_rot_smooth", type=float, default=0.4,
        help="SLERP factor for body rotation (0-1, higher=smoother). Default 0.4.",
    )
    parser.add_argument(
        "--filter_hand_orient_smooth", type=float, default=0.85,
        help="Extra SLERP for hand orientation (0-1). Default 0.85.",
    )
    parser.add_argument(
        "--filter_wrist_roll_limit", type=float, default=60.0,
        help="Max wrist roll relative to forearm (deg, 0=disable). Default 60.",
    )
    parser.add_argument(
        "--filter_median_window", type=int, default=3,
        help="Sliding median window for positions (0=disable). Default 3.",
    )
    args = parser.parse_args()

    if args.robot == "g1":
        xml_file = f"{here}/../assets/g1/g1_mocap_with_wrist_roll.xml"
    elif args.robot == "t1":
        xml_file = f"{here}/../assets/t1/t1_mocap.xml"
    else:
        raise ValueError(f"Unsupported robot type: {args.robot}")

    server = OptiTrackMimicObsServerGMR(
        server_ip=args.host,
        client_ip=args.client_ip,
        use_multicast=args.use_multicast,
        robot_type=args.robot,
        xml_file=xml_file,
        vis=args.vis,
        use_hand=args.use_hand,
    )

    # Configure GMR RealtimeMotionFilter from CLI (same as optitrack_to_robot.py)
    buf = server.vicon_data_buffer
    buf.set_mocap_filter(enable=not args.no_mocap_filter)
    if not args.no_mocap_filter:
        buf._motion_filter.pos_alpha = args.filter_pos_smooth
        buf._motion_filter.rot_alpha = args.filter_rot_smooth
        buf._motion_filter.hand_orient_alpha = args.filter_hand_orient_smooth
        buf._motion_filter.wrist_roll_limit = (
            np.deg2rad(args.filter_wrist_roll_limit)
            if args.filter_wrist_roll_limit > 0 else None
        )
        if args.filter_median_window > 0:
            buf._median_filter.window = args.filter_median_window

    print(f"[Filter] GMR RealtimeMotionFilter={'ON' if not args.no_mocap_filter else 'OFF'}"
          f" (pos={args.filter_pos_smooth}, rot={args.filter_rot_smooth},"
          f" hand_orient={args.filter_hand_orient_smooth},"
          f" wrist_limit={args.filter_wrist_roll_limit}deg,"
          f" median_win={args.filter_median_window})")

    server.main_loop(freq=50)


if __name__ == "__main__":
    main()