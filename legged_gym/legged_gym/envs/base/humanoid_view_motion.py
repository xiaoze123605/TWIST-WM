from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os
import cv2

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from .humanoid_char_config import HumanoidCharCfg
from .humanoid_char import HumanoidChar

# import pose.anim.motion_lib as motion_lib
from pose.utils.motion_lib_pkl import MotionLib
from .humanoid_mimic import convert_to_global_root_body_pos


class HumanoidViewMotion(HumanoidChar):
    def __init__(self, cfg: HumanoidCharCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        self._episode_length = self._get_max_motion_len()
        self.motor_strength *= 0.0
        
    def _init_buffers(self):
        self._load_motions()
        super()._init_buffers()
        
    def _load_motions(self):
        self._motion_lib = MotionLib(motion_file=self.cfg.motion.motion_file,
                                                device=self.device)
        return
    
    def _get_max_motion_len(self):
        max_len = 0
        num_motions = self._motion_lib.num_motions()
        for i in range(num_motions):
            curr_len = self._motion_lib.get_motion_length(i)
            max_len = max(max_len, curr_len)
            
        return max_len
    
    def _sync_motion(self):
        num_motions = self._motion_lib.num_motions()
        motion_ids = torch.arange(self.num_envs, device=self.device)
        motion_ids = torch.remainder(motion_ids, num_motions)
        motion_times = self.episode_length_buf * self.dt % self._motion_lib.get_motion_length(motion_ids)
        root_pos, root_rot, root_vel, root_ang_vel, joint_dof, dof_vel, body_pos = self._motion_lib.calc_motion_frame(motion_ids, motion_times=motion_times)
        root_pos[:, 2] += self.cfg.motion.height_offset
        
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.int32)
        self._set_env_state(env_ids=env_ids, root_pos=root_pos, root_rot=root_rot, dof_pos=joint_dof, root_vel=0.0, root_ang_vel=0.0, dof_vel=0.0)
        
        self._ref_body_pos = convert_to_global_root_body_pos(root_pos=root_pos, root_rot=root_rot, body_pos=body_pos)
        
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids),
                                                     len(env_ids))
        
        if self.dof_state is not None:
            self.gym.set_dof_state_tensor_indexed(self.sim,
                                                  gymtorch.unwrap_tensor(self.dof_state),
                                                  gymtorch.unwrap_tensor(env_ids),
                                                  len(env_ids))
        
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
    def post_physics_step(self):
        super().post_physics_step()
        self._sync_motion()
        
    def check_termination(self):
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            
