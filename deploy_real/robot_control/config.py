import numpy as np
import yaml


class Config:
    def __init__(self, file_path) -> None:
        with open(file_path, "r") as f:
            config = yaml.safe_load(f.read())
            self.control_dt = config["control_dt"]

            self.msg_type = config["msg_type"]
            self.imu_type = config["imu_type"]

            self.weak_motor = []
            if "weak_motor" in config:
                self.weak_motor = config["weak_motor"]

            self.lowcmd_topic = config["lowcmd_topic"]
            self.lowstate_topic = config["lowstate_topic"]

            self.policy_path = config["policy_path"] if "policy_path" in config else None

            self.leg_joint2motor_idx = config["leg_joint2motor_idx"]
            self.kps = config["kps"]
            self.kds = config["kds"]
            self.default_angles = np.array(config["default_angles"], dtype=np.float32)

            self.arm_waist_joint2motor_idx = config["arm_waist_joint2motor_idx"] if "arm_waist_joint2motor_idx" in config else None
            self.arm_waist_kps = config["arm_waist_kps"] if "arm_waist_kps" in config else None
            self.arm_waist_kds = config["arm_waist_kds"] if "arm_waist_kds" in config else None
            self.arm_waist_target = np.array(config["arm_waist_target"], dtype=np.float32) if "arm_waist_target" in config else None

            self.wrist_joint2motor_idx = config["wrist_joint2motor_idx"] if "wrist_joint2motor_idx" in config else None
            self.wrist_kps = config["wrist_kps"] if "wrist_kps" in config else None
            self.wrist_kds = config["wrist_kds"] if "wrist_kds" in config else None
            self.wrist_target = np.array(config["wrist_target"], dtype=np.float32) if "wrist_target" in config else None

            self.ang_vel_scale = config["ang_vel_scale"] if "ang_vel_scale" in config else None
            self.dof_pos_scale = config["dof_pos_scale"] if "dof_pos_scale" in config else None
            self.dof_vel_scale = config["dof_vel_scale"] if "dof_vel_scale" in config else None
            self.action_scale = config["action_scale"] if "action_scale" in config else None
            self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32) if "cmd_scale" in config else None
            self.max_cmd = np.array(config["max_cmd"], dtype=np.float32) if "max_cmd" in config else None

            self.num_actions = config["num_actions"] if "num_actions" in config else None
            self.num_obs = config["num_obs"] if "num_obs" in config else None
