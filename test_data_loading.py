import sys
import os
import yaml

sys.path.append(os.path.expanduser("~/TWIST"))
sys.path.append(os.path.expanduser("~/TWIST/legged_gym"))
sys.path.append(os.path.expanduser("~/TWIST/pose"))

config_path = "/home/hank/TWIST/legged_gym/motion_data_configs/twist_dataset.yaml"

try:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print("YAML config OK")
except Exception as e:
    print("Config parse error:", e)
    exit()

try:
    import torch
    from pose.utils.motion_lib_pkl import MotionLib

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    motion_lib = MotionLib(motion_file=config_path, device=device)
    print(f"Data loaded: {motion_lib.num_motions()} motions")
    for i in range(min(3, motion_lib.num_motions())):
        length = motion_lib.get_motion_length(torch.tensor([i], device=device)).item()
        print(f"  Motion {i}: {length:.2f}s ({motion_lib.get_motion_names()[i]})")
except Exception as e:
    print("Motion data loading failed:", e)
