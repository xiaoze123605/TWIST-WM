#!/usr/bin/env python3
import argparse
import csv
import glob
import os
from collections import Counter

import numpy as np


JOINT_NAMES = [
    "L_hip_pitch", "L_hip_roll", "L_hip_yaw", "L_knee", "L_ankle_pitch", "L_ankle_roll",
    "R_hip_pitch", "R_hip_roll", "R_hip_yaw", "R_knee", "R_ankle_pitch", "R_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "L_shoulder_pitch", "L_shoulder_roll", "L_shoulder_yaw", "L_elbow",
    "R_shoulder_pitch", "R_shoulder_roll", "R_shoulder_yaw", "R_elbow",
]

GROUPS = {
    "leg": list(range(12)),
    "left_leg": list(range(6)),
    "right_leg": list(range(6, 12)),
    "hip_pitch": [0, 6],
    "hip_roll": [1, 7],
    "hip_yaw": [2, 8],
    "knee": [3, 9],
    "ankle_pitch": [4, 10],
    "ankle_roll": [5, 11],
    "waist": [12, 13, 14],
    "arm": list(range(15, 23)),
}


def find_latest_log(log_dir: str):
    npz_files = sorted(glob.glob(os.path.join(log_dir, "real_deploy_*.npz")))
    if not npz_files:
        raise FileNotFoundError(f"No real_deploy_*.npz found in {log_dir}")
    npz_path = npz_files[-1]
    csv_path = npz_path.replace(".npz", ".csv")
    return csv_path, npz_path


def resolve_paths(path, log_dir):
    if path is None:
        return find_latest_log(log_dir)

    path = os.path.abspath(path)
    if path.endswith(".npz"):
        npz_path = path
        csv_path = path.replace(".npz", ".csv")
    elif path.endswith(".csv"):
        csv_path = path
        npz_path = path.replace(".csv", ".npz")
    else:
        raise ValueError("path must be a .npz or .csv file")

    if not os.path.exists(npz_path):
        raise FileNotFoundError(npz_path)
    return csv_path, npz_path


def load_csv(csv_path: str):
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, "r") as f:
        return list(csv.DictReader(f))


def safe_arr(data, key, default=None):
    if key in data.files:
        return data[key]
    return default


def rms(x, axis=None):
    return np.sqrt(np.mean(np.asarray(x) ** 2, axis=axis))


def max_abs(x, axis=None):
    return np.max(np.abs(np.asarray(x)), axis=axis)


def print_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_group_errors(err):
    print_header("GROUP TRACKING ERROR")
    for name, idx in GROUPS.items():
        e = err[:, idx]
        print(
            f"{name:14s} "
            f"rms={rms(e):.6f}  "
            f"max={max_abs(e):.6f}  "
            f"signed_mean={np.mean(e):+.6f}"
        )


def print_joint_table(target, dof, err, tau=None):
    print_header("PER-JOINT TARGET / DOF / ERROR")
    print(
        "idx joint                 "
        "target_mean   dof_mean      err_mean     err_rms      err_max      "
        "tau_max_abs"
    )
    for i, name in enumerate(JOINT_NAMES):
        target_mean = np.mean(target[:, i])
        dof_mean = np.mean(dof[:, i])
        err_mean = np.mean(err[:, i])
        err_rms = rms(err[:, i])
        err_max = max_abs(err[:, i])
        if tau is not None:
            tau_max = max_abs(tau[:, i])
        else:
            tau_max = float("nan")
        print(
            f"{i:02d}  {name:20s} "
            f"{target_mean:+.6f}  {dof_mean:+.6f}  "
            f"{err_mean:+.6f}  {err_rms:.6f}  {err_max:.6f}  "
            f"{tau_max:.3f}"
        )


def print_pitch_roll_summary(rpy, ang_vel=None):
    print_header("BODY RPY / FALL DIRECTION")
    roll = rpy[:, 0]
    pitch = rpy[:, 1]
    yaw = rpy[:, 2]

    roll_change = roll[-1] - roll[0]
    pitch_change = pitch[-1] - pitch[0]

    print(f"roll  first={roll[0]:+.6f}  last={roll[-1]:+.6f}  maxabs={max_abs(roll):.6f}  change={roll_change:+.6f}")
    print(f"pitch first={pitch[0]:+.6f}  last={pitch[-1]:+.6f}  maxabs={max_abs(pitch):.6f}  change={pitch_change:+.6f}")
    print(f"yaw   first={yaw[0]:+.6f}  last={yaw[-1]:+.6f}  maxabs={max_abs(yaw):.6f}")

    if abs(pitch_change) > abs(roll_change):
        print("dominant direction: pitch / front-back")
    else:
        print("dominant direction: roll / left-right")

    if ang_vel is not None:
        print("ang_vel max abs:", np.round(max_abs(ang_vel, axis=0), 6).tolist())


def print_loop_summary(loop_dt):
    print_header("LOOP TIMING")
    print(f"loop_dt mean: {np.mean(loop_dt):.6f}")
    print(f"loop_dt max : {np.max(loop_dt):.6f}")
    print(f"loop_dt > 0.02: {int(np.sum(loop_dt > 0.02))}")
    print(f"loop_dt > 0.04: {int(np.sum(loop_dt > 0.04))}")


def print_mimic_policy_summary(data):
    raw = safe_arr(data, "raw_action")
    mimic = safe_arr(data, "action_mimic")

    print_header("MIMIC / POLICY OUTPUT")
    if mimic is not None:
        print(f"action_mimic shape: {mimic.shape}")
        print(f"action_mimic max abs: {max_abs(mimic):.6f}")
        if len(mimic) >= 2:
            print(f"action_mimic max delta: {max_abs(np.diff(mimic, axis=0)):.6f}")

    if raw is not None:
        print(f"raw_action shape: {raw.shape}")
        print(f"raw_action max abs: {max_abs(raw):.6f}")
        if len(raw) >= 2:
            print(f"raw_action max delta: {max_abs(np.diff(raw, axis=0)):.6f}")


def print_diagnosis(target, dof, err, tau, rpy, loop_dt):
    print_header("DIAGNOSIS HINTS")

    leg_err_rms = rms(err[:, GROUPS["leg"]])
    leg_err_max = max_abs(err[:, GROUPS["leg"]])
    ankle_pitch_rms = rms(err[:, GROUPS["ankle_pitch"]], axis=0)
    hip_pitch_rms = rms(err[:, GROUPS["hip_pitch"]], axis=0)
    knee_rms = rms(err[:, GROUPS["knee"]], axis=0)

    roll_change = rpy[-1, 0] - rpy[0, 0]
    pitch_change = rpy[-1, 1] - rpy[0, 1]

    print(f"leg_err_rms: {leg_err_rms:.6f}")
    print(f"leg_err_max: {leg_err_max:.6f}")
    print(f"ankle_pitch_rms [L, R]: {np.round(ankle_pitch_rms, 6).tolist()}")
    print(f"hip_pitch_rms   [L, R]: {np.round(hip_pitch_rms, 6).tolist()}")
    print(f"knee_rms        [L, R]: {np.round(knee_rms, 6).tolist()}")

    if np.max(loop_dt) > 0.02:
        print("- Loop has overruns: first fix control frequency / CPU / Redis blocking.")
    else:
        print("- Loop timing looks OK.")

    if leg_err_rms > 0.08 or leg_err_max > 0.20:
        print("- Joint tracking error is large. Check PD gains, torque, motor mapping, or joint zero offsets.")
    else:
        print("- Joint tracking error is not too large. Falling may be due to lack of active balance / COM issue.")

    if np.max(ankle_pitch_rms) > 0.10:
        print("- Ankle pitch error is large. Prioritize ankle_pitch stiffness / target compensation / pitch balance.")

    if np.max(hip_pitch_rms) > 0.08:
        print("- Hip pitch error is large. Front-back support may be insufficient.")

    if abs(pitch_change) > abs(roll_change):
        print("- Main fall direction is front-back. Focus on hip_pitch, knee, ankle_pitch, pitch compensation.")
    else:
        print("- Main fall direction is left-right. Focus on hip_roll, ankle_roll, roll compensation.")

    if tau is not None:
        leg_tau = max_abs(tau[:, GROUPS["leg"]], axis=0)
        print("leg_tau_max_abs:", np.round(leg_tau, 3).tolist())
        if np.max(leg_tau) < 20:
            print("- Leg torque is not extremely high. This looks more like low stiffness / balance issue than torque saturation.")
        else:
            print("- Some leg torque is high. Be careful increasing gains; check saturation/contact.")


def main():
    parser = argparse.ArgumentParser(description="Analyze TWIST real deployment logs.")
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Optional path to real_deploy_xxx.npz or .csv. If omitted, use latest log.",
    )
    parser.add_argument(
        "--log-dir",
        default="/home/hank/TWIST/deploy_real/logs",
        help="Log directory used when path is omitted.",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=0,
        help="Analyze only the last N frames. 0 means all frames.",
    )
    args = parser.parse_args()

    csv_path, npz_path = resolve_paths(args.path, args.log_dir)

    print("CSV:", csv_path)
    print("NPZ:", npz_path)

    rows = load_csv(csv_path)
    if rows:
        print("CSV rows:", len(rows))
        print("mode counts:", Counter(r.get("mode", "") for r in rows))

    data = np.load(npz_path, allow_pickle=True)

    required = ["target_dof_pos", "dof_pos", "rpy", "loop_dt"]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise KeyError(f"Missing required arrays in npz: {missing}")

    target = data["target_dof_pos"]
    dof = data["dof_pos"]
    rpy = data["rpy"]
    loop_dt = data["loop_dt"]

    tau = safe_arr(data, "tau_est")
    ang_vel = safe_arr(data, "ang_vel")

    if args.tail and args.tail > 0:
        target = target[-args.tail:]
        dof = dof[-args.tail:]
        rpy = rpy[-args.tail:]
        loop_dt = loop_dt[-args.tail:]
        if tau is not None:
            tau = tau[-args.tail:]
        if ang_vel is not None:
            ang_vel = ang_vel[-args.tail:]

    err = target - dof

    print_header("BASIC")
    print("frames:", len(target))
    print("target shape:", target.shape)
    print("dof shape:", dof.shape)
    print("target finite:", bool(np.all(np.isfinite(target))))
    print("dof finite:", bool(np.all(np.isfinite(dof))))
    print("target max delta:", max_abs(np.diff(target, axis=0)) if len(target) >= 2 else 0.0)
    print("dof max delta:", max_abs(np.diff(dof, axis=0)) if len(dof) >= 2 else 0.0)

    print_loop_summary(loop_dt)
    print_pitch_roll_summary(rpy, ang_vel)
    print_group_errors(err)
    print_joint_table(target, dof, err, tau)
    print_mimic_policy_summary(data)
    print_diagnosis(target, dof, err, tau, rpy, loop_dt)


if __name__ == "__main__":
    main()