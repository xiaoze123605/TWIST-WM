#!/usr/bin/env python3
"""
Comprehensive motion data quality checker for TWIST training.
Checks data validity, physical feasibility, and flags bad data.
"""
import pickle
import os
import sys
import numpy as np
from collections import defaultdict

# --- Config ---
FPS_TOLERANCE = (20, 120)        # fps should be in this range
MIN_FRAMES = 30                  # minimum frames to be useful
MAX_ROOT_HEIGHT = 3.0            # max root height (meters) - above this is likely air/error
MIN_ROOT_HEIGHT = 0.05           # min root height - below ground is error
MAX_ROOT_VEL = 10.0              # max root linear velocity (m/s) - beyond is teleportation
MAX_ROOT_ANGVEL = 20.0           # max root angular velocity (rad/s)
MAX_JOINT_VEL = 30.0             # max joint velocity (rad/s)
MAX_POS_JUMP_STD = 8.0           # flag if any position change exceeds N std devs
QUAT_NORM_TOL = 0.01             # tolerance for quaternion unit norm check
LOCAL_BODY_MAX_DIST = 2.0        # max distance of any local body from root
MIN_BODY_HEIGHT = -0.1           # min body height below root (feet shouldn't be too far below)

def check_pkl(filepath):
    """Returns list of issues for a single pkl file."""
    issues = []
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        return [f"CANNOT LOAD: {e}"]

    # --- Basic structure checks ---
    for key in ['fps', 'root_pos', 'root_rot', 'dof_pos']:
        if key not in data:
            issues.append(f"MISSING key: {key}")
    if issues:
        return issues

    fps = data['fps']
    root_pos = data['root_pos']
    root_rot = data['root_rot']
    dof_pos = data['dof_pos']

    T = root_pos.shape[0]

    # --- FPS check ---
    if not (FPS_TOLERANCE[0] <= fps <= FPS_TOLERANCE[1]):
        issues.append(f"FPS={fps:.1f} out of range [{FPS_TOLERANCE[0]}, {FPS_TOLERANCE[1]}]")

    # --- Frame count ---
    if T < MIN_FRAMES:
        issues.append(f"Too few frames: {T} < {MIN_FRAMES}")

    # --- Shape consistency ---
    if root_rot.shape[0] != T:
        issues.append(f"root_rot frames mismatch: {root_rot.shape[0]} vs {T}")
    if dof_pos.shape[0] != T:
        issues.append(f"dof_pos frames mismatch: {dof_pos.shape[0]} vs {T}")

    # --- NaN/Inf check ---
    if np.any(np.isnan(root_pos)):
        issues.append(f"NaN in root_pos: {np.sum(np.isnan(root_pos))} values")
    if np.any(np.isinf(root_pos)):
        issues.append(f"Inf in root_pos: {np.sum(np.isinf(root_pos))} values")
    if np.any(np.isnan(root_rot)):
        issues.append(f"NaN in root_rot: {np.sum(np.isnan(root_rot))} values")
    if np.any(np.isinf(root_rot)):
        issues.append(f"Inf in root_rot: {np.sum(np.isinf(root_rot))} values")
    if np.any(np.isnan(dof_pos)):
        nc = np.sum(np.isnan(dof_pos))
        issues.append(f"NaN in dof_pos: {nc} values")
    if np.any(np.isinf(dof_pos)):
        issues.append(f"Inf in dof_pos: {np.sum(np.isinf(dof_pos))} values")

    # --- Quaternion validity ---
    quat_norms = np.linalg.norm(root_rot, axis=1)
    bad_quats = np.sum(np.abs(quat_norms - 1.0) > QUAT_NORM_TOL)
    if bad_quats > 0:
        issues.append(f"Non-unit quaternions: {bad_quats}/{T} frames (tol={QUAT_NORM_TOL})")

    # --- Root height ---
    root_z = root_pos[:, 2]
    min_z, max_z = np.min(root_z), np.max(root_z)
    if max_z > MAX_ROOT_HEIGHT:
        issues.append(f"Root too high: max_z={max_z:.3f}m > {MAX_ROOT_HEIGHT}m")
    if min_z < MIN_ROOT_HEIGHT:
        issues.append(f"Root below ground: min_z={min_z:.3f}m < {MIN_ROOT_HEIGHT}m")
    mean_z = np.mean(root_z)
    if mean_z < 0.3:
        issues.append(f"Root mean height very low: mean_z={mean_z:.3f}m")

    # --- Root velocity ---
    dt = 1.0 / fps
    root_vel = np.linalg.norm(np.diff(root_pos, axis=0), axis=1) / dt
    max_rv = np.max(root_vel)
    mean_rv = np.mean(root_vel)
    if max_rv > MAX_ROOT_VEL:
        bad_frames = np.sum(root_vel > MAX_ROOT_VEL)
        issues.append(f"Root velocity spike: max={max_rv:.2f} m/s > {MAX_ROOT_VEL} m/s ({bad_frames} frames)")
    # Check for position jumps using std
    pos_diffs = np.linalg.norm(np.diff(root_pos, axis=0), axis=1)
    std_diff = np.std(pos_diffs)
    mean_diff = np.mean(pos_diffs)
    jump_thresh = mean_diff + MAX_POS_JUMP_STD * std_diff
    jumps = pos_diffs > max(jump_thresh, 0.5)
    njumps = np.sum(jumps)
    if njumps > 0 and max_rv > MAX_ROOT_VEL:
        # Already reported above; add detail
        pass

    # --- Root angular velocity ---
    # Approximate: angle between consecutive quaternions
    q_diff = np.zeros(T-1)
    for i in range(T-1):
        q1 = root_rot[i]
        q2 = root_rot[i+1]
        # Normalize
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        dot = np.abs(np.dot(q1, q2))
        dot = np.clip(dot, 0.0, 1.0)
        q_diff[i] = 2 * np.arccos(dot)
    root_angvel = q_diff / dt
    max_rav = np.max(root_angvel)
    if max_rav > MAX_ROOT_ANGVEL:
        bad_frames = np.sum(root_angvel > MAX_ROOT_ANGVEL)
        issues.append(f"Root angular velocity spike: max={max_rav:.2f} rad/s > {MAX_ROOT_ANGVEL} rad/s ({bad_frames} frames)")

    # --- Joint velocity ---
    dof_vel = np.abs(np.diff(dof_pos, axis=0)) / dt
    max_jv = np.max(dof_vel)
    max_jv_joint = np.argmax(np.max(dof_vel, axis=0))
    if max_jv > MAX_JOINT_VEL:
        bad_frames = np.sum(np.any(dof_vel > MAX_JOINT_VEL, axis=1))
        issues.append(f"Joint velocity spike: max={max_jv:.2f} rad/s > {MAX_JOINT_VEL} rad/s "
                      f"(joint {max_jv_joint}, {bad_frames} frames)")

    # --- Joint range check ---
    # Typical humanoid joints should be roughly in [-π, π]
    j_min = np.min(dof_pos, axis=0)
    j_max = np.max(dof_pos, axis=0)
    j_range = j_max - j_min
    weird_joints = []
    for j in range(dof_pos.shape[1]):
        if j_min[j] < -2 * np.pi or j_max[j] > 2 * np.pi:
            weird_joints.append(f"j{j}: [{j_min[j]:.1f}, {j_max[j]:.1f}]")
        elif j_range[j] < 0.01:
            weird_joints.append(f"j{j}: near-constant [{j_min[j]:.3f}, {j_max[j]:.3f}]")
    if weird_joints:
        issues.append(f"Abnormal joint ranges: {'; '.join(weird_joints[:5])}"
                      + (f" ... +{len(weird_joints)-5} more" if len(weird_joints) > 5 else ""))

    # --- Local body position check ---
    if 'local_body_pos' in data:
        lbp = data['local_body_pos']
        if lbp.shape[0] != T:
            issues.append(f"local_body_pos frames mismatch: {lbp.shape[0]} vs {T}")
        else:
            # Check for NaN in local body pos
            if np.any(np.isnan(lbp)):
                issues.append(f"NaN in local_body_pos: {np.sum(np.isnan(lbp))} values")
            # Check max distance of any body from root
            body_dists = np.linalg.norm(lbp, axis=2)  # (T, 38)
            max_body_dist = np.max(body_dists)
            if max_body_dist > LOCAL_BODY_MAX_DIST:
                bad_bodies = np.where(np.max(body_dists, axis=0) > LOCAL_BODY_MAX_DIST)[0]
                issues.append(f"Body parts too far from root: max_dist={max_body_dist:.3f}m "
                              f"(body indices: {bad_bodies.tolist()})")
            # Check for bodies below root by too much
            body_z = lbp[:, :, 2]  # (T, 38)
            min_body_z = np.min(body_z)
            if min_body_z < -1.5:
                issues.append(f"Body parts very low: min_body_z={min_body_z:.3f}m below root")
            # Check local body velocity smoothness
            body_vel = np.linalg.norm(np.diff(lbp, axis=0), axis=2) / dt  # (T-1, 38)
            max_bv = np.max(body_vel)
            if max_bv > MAX_ROOT_VEL * 2:
                issues.append(f"Excessive body part velocity: max={max_bv:.1f} m/s")

    # --- Summary stats ---
    if not issues:
        duration = T / fps
        return [], {
            'frames': T,
            'duration': duration,
            'fps': fps,
            'mean_root_z': float(mean_z),
            'max_root_vel': float(max_rv),
            'mean_root_vel': float(mean_rv),
            'max_root_angvel': float(max_rav),
            'max_joint_vel': float(max_jv),
            'min_joint': float(np.min(j_min)),
            'max_joint': float(np.max(j_max)),
        }
    return issues, None


def main():
    base = '/home/hank/TWIST/track_dataset/twist_motion_dataset'
    dirs = ['mydata', 'mydata2', 'mydata3']

    all_results = {}
    bad_files = []
    good_files = []
    stats = []

    for d in dirs:
        dp = os.path.join(base, d)
        if not os.path.isdir(dp):
            continue
        files = sorted([f for f in os.listdir(dp) if f.endswith('.pkl')])
        for fname in files:
            fpath = os.path.join(dp, fname)
            label = f"{d}/{fname}"
            issues, s = check_pkl(fpath)
            if issues:
                bad_files.append((label, issues, fpath))
            else:
                good_files.append((label, s, fpath))
                stats.append((label, s))

    # --- Group bad files by category ---
    print("=" * 80)
    print("                     MOTION DATA QUALITY REPORT")
    print("=" * 80)
    print(f"\nTotal files checked: {len(bad_files) + len(good_files)}")
    print(f"Good files: {len(good_files)}")
    print(f"Bad files:  {len(bad_files)}")

    # --- Print bad files by severity ---
    severity_order = [
        'CANNOT LOAD',
        'MISSING key',
        'NaN in',
        'Inf in',
        'Too few frames:',
        'FPS=',
        'Root too high',
        'Root below ground',
        'Root velocity spike',
        'Root angular velocity spike',
        'Joint velocity spike',
        'Non-unit quaternions',
        'Abnormal joint ranges',
        'Excessive body part velocity',
        'Body parts too far from root',
        'Body parts very low',
        'Root mean height very low',
        'frames mismatch',
    ]

    def severity(iss):
        for i, pat in enumerate(severity_order):
            if iss.startswith(pat):
                return i
        return len(severity_order)

    bad_files_sorted = sorted(bad_files, key=lambda x: min(severity(i) for i in x[1]))

    print("\n" + "=" * 80)
    print("                         BAD / PROBLEMATIC FILES")
    print("=" * 80)

    if not bad_files:
        print("\n  ✅ No bad files found!")
    else:
        for label, issues, fpath in bad_files_sorted:
            print(f"\n  📛 {label}")
            for iss in issues:
                print(f"     • {iss}")

    # --- Summary of good files ---
    print("\n" + "=" * 80)
    print("                         GOOD FILES SUMMARY")
    print("=" * 80)

    if stats:
        frames_arr = np.array([s['frames'] for _, s in stats])
        dur_arr = np.array([s['duration'] for _, s in stats])
        mrv_arr = np.array([s['mean_root_vel'] for _, s in stats])
        max_rv_arr = np.array([s['max_root_vel'] for _, s in stats])
        max_rav_arr = np.array([s['max_root_angvel'] for _, s in stats])
        max_jv_arr = np.array([s['max_joint_vel'] for _, s in stats])

        print(f"\n  Count: {len(stats)}")
        print(f"  Total duration: {np.sum(dur_arr):.1f}s ({np.sum(dur_arr)/60:.1f} min)")
        print(f"  Frames per clip: min={np.min(frames_arr)}, max={np.max(frames_arr)}, "
              f"median={np.median(frames_arr):.0f}")
        print(f"  Duration per clip: min={np.min(dur_arr):.1f}s, max={np.max(dur_arr):.1f}s, "
              f"median={np.median(dur_arr):.1f}s")
        print(f"  Mean root velocity: {np.mean(mrv_arr):.2f} m/s (max clip avg: {np.max(mrv_arr):.2f})")
        print(f"  Max root velocity:  {np.max(max_rv_arr):.2f} m/s")
        print(f"  Max root angvel:    {np.max(max_rav_arr):.2f} rad/s")
        print(f"  Max joint vel:      {np.max(max_jv_arr):.2f} rad/s")

        # Per-directory breakdown
        print("\n  Per-directory breakdown:")
        for d in dirs:
            d_stats = [(l, s) for l, s in stats if l.startswith(d + '/')]
            if d_stats:
                d_dur = sum(s['duration'] for _, s in d_stats)
                print(f"    {d}: {len(d_stats)} good clips, {d_dur:.1f}s total")

    # --- Count by issue category ---
    print("\n" + "=" * 80)
    print("                       ISSUE FREQUENCY BREAKDOWN")
    print("=" * 80)
    issue_counts = defaultdict(int)
    for _, issues, _ in bad_files:
        for iss in issues:
            cat = iss.split(':')[0] if ':' in iss else iss
            issue_counts[cat] += 1
    for cat, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count} files")

    print("\n" + "=" * 80)
    print("Done.")
    print("=" * 80)


if __name__ == '__main__':
    main()
