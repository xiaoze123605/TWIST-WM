#!/usr/bin/env python3
import time
import numpy as np

from robot_control.config import Config
from robot_control.g1_wrapper import G1RealWorldEnv


NET = "enp4s0"
CONFIG_PATH = "/home/hank/TWIST/deploy_real/robot_control/configs/g1.yaml"


def main():
    cfg = Config(CONFIG_PATH)

    print("[READ] Creating G1RealWorldEnv...")
    print("[READ] 注意：这个脚本只读状态，不发送控制命令。")
    env = G1RealWorldEnv(net=NET, config=cfg)

    print("[READ] Connected. Start reading current official standing pose...")
    print("[READ] Keep the robot in official standing mode. Do not run low-level controller.")

    samples = []
    duration = 5.0
    dt = 0.02
    steps = int(duration / dt)

    for i in range(steps):
        dof_pos, dof_vel, quat, ang_vel = env.get_robot_state()
        samples.append(np.asarray(dof_pos, dtype=np.float64))
        time.sleep(dt)

    samples = np.asarray(samples)

    # 用后 3 秒的数据求均值，避开刚开始读取的不稳定部分
    tail = samples[int(2.0 / dt):]
    mean = tail.mean(axis=0)
    std = tail.std(axis=0)

    names = [
        "L_hip_pitch", "L_hip_roll", "L_hip_yaw", "L_knee", "L_ankle_pitch", "L_ankle_roll",
        "R_hip_pitch", "R_hip_roll", "R_hip_yaw", "R_knee", "R_ankle_pitch", "R_ankle_roll",
        "waist_yaw", "waist_roll", "waist_pitch",
        "L_shoulder_pitch", "L_shoulder_roll", "L_shoulder_yaw", "L_elbow",
        "R_shoulder_pitch", "R_shoulder_roll", "R_shoulder_yaw", "R_elbow",
    ]

    print("\n[RESULT] official standing dof_pos mean/std:")
    for i, (name, m, s) in enumerate(zip(names, mean, std)):
        print(f"{i:02d} {name:18s} mean={m:+.6f}  std={s:.6f}")

    print("\n[RESULT] full 23-dof list:")
    print([round(float(x), 6) for x in mean])

    print("\n[RESULT] default_angles, first 12:")
    print([round(float(x), 6) for x in mean[:12]])

    print("\n[RESULT] arm_waist_target, last 11:")
    print([round(float(x), 6) for x in mean[12:]])

    out_path = "/tmp/g1_official_stand_dof_pos.npy"
    np.save(out_path, mean)
    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()
