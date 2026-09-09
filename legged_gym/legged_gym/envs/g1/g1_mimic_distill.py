from isaacgym import gymtorch
from isaacgym.torch_utils import *

import torch

from legged_gym.envs.base.humanoid_mimic import HumanoidMimic
from .g1_mimic_distill_config import G1MimicPrivCfg, G1MimicStuCfg
from legged_gym.envs.g1.anyadapter_history_mixin import AnyAdapterHistoryMixin
from legged_gym.gym_utils.math import *
from pose.utils import torch_utils
from legged_gym.envs.base.legged_robot import euler_from_quaternion
from legged_gym.envs.base.humanoid_char import convert_to_local_root_body_pos, convert_to_global_root_body_pos, compute_local_body_pos

def g1_body_from_38_to_52(body_pos_38: torch.Tensor) -> torch.Tensor:
    """
    将形状 (N, 38, 3) 的关节坐标转换为形状 (N, 52, 3)。
    多出来的手指等关节在输出中将填充为 (0, 0, 0)。

    参数:
    -------
        body_pos_38 : torch.Tensor
            大小为 (N, 38, 3) 的关节坐标，N 为批量大小

    返回:
    -------
        body_pos_52 : torch.Tensor
            大小为 (N, 52, 3) 的关节坐标
    """

    # 构建一个大小为 52 的整型索引张量，用于说明：
    # “52-link 的每个关节，对应到 38-link 中的哪一个下标？”
    # 若对应不到（例如手指关节），则为 -1。
    idx_map_52_list = [-1] * 52
    # 直接在列表中指定 38->52 的映射：
    # 0~29 不变, 30->37, 31->38, 32->39, 33->40, 34->41, 35->42, 36->43, 37->44
    # ---------------------------------------------------------------------
    idx_map_52_list[0]  = 0   # pelvis
    idx_map_52_list[1]  = 1
    idx_map_52_list[2]  = 2
    idx_map_52_list[3]  = 3
    idx_map_52_list[4]  = 4
    idx_map_52_list[5]  = 5
    idx_map_52_list[6]  = 6
    idx_map_52_list[7]  = 7
    idx_map_52_list[8]  = 8
    idx_map_52_list[9]  = 9
    idx_map_52_list[10] = 10
    idx_map_52_list[11] = 11
    idx_map_52_list[12] = 12
    idx_map_52_list[13] = 13
    idx_map_52_list[14] = 14
    idx_map_52_list[15] = 15
    idx_map_52_list[16] = 16
    idx_map_52_list[17] = 17
    idx_map_52_list[18] = 18
    idx_map_52_list[19] = 19
    idx_map_52_list[20] = 20
    idx_map_52_list[21] = 21
    idx_map_52_list[22] = 22
    idx_map_52_list[23] = 23
    idx_map_52_list[24] = 24
    idx_map_52_list[25] = 25
    idx_map_52_list[26] = 26
    idx_map_52_list[27] = 27
    idx_map_52_list[28] = 28
    idx_map_52_list[29] = 29
    idx_map_52_list[37] = 30
    idx_map_52_list[38] = 31
    idx_map_52_list[39] = 32
    idx_map_52_list[40] = 33
    idx_map_52_list[41] = 34
    idx_map_52_list[42] = 35
    idx_map_52_list[43] = 36
    idx_map_52_list[44] = 37
    # 其余下标(手指相关)依旧保持 -1

    # 转换成 PyTorch 张量，放到和输入相同的 device 上
    idx_map_52 = torch.tensor(idx_map_52_list, 
                              dtype=torch.long, 
                              device=body_pos_38.device)

    # 创建输出张量，大小 (N, 52, 3)，默认填零
    N = body_pos_38.shape[0]
    body_pos_52 = torch.zeros((N, 52, 3), 
                              dtype=body_pos_38.dtype, 
                              device=body_pos_38.device)

    # 构建一个布尔掩码，筛选出 idx_map_52 >= 0 的关节
    valid_mask = (idx_map_52 >= 0)

    # 对有效关节通过高级索引直接复制，无需对 N 进行循环
    body_pos_52[:, valid_mask, :] = body_pos_38[:, idx_map_52[valid_mask], :]

    return body_pos_52



class G1MimicDistill(AnyAdapterHistoryMixin, HumanoidMimic):
    def __init__(self, cfg: G1MimicPrivCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.obs_type = cfg.env.obs_type
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.last_feet_z = 0.05
        self.episode_length = torch.zeros((self.num_envs), device=self.device)
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        if self.obs_type == 'student':
            self.total_env_steps_counter = 24 * 100000
            self.global_counter = 24 * 100000
            # self.motion_difficulty = torch.ones_like(self.motion_difficulty)

    def _reset_ref_motion(self, env_ids, motion_ids=None):
        n = len(env_ids)
        fixed_motion_ids = getattr(self, "_eval_scenario_motion_ids", None)
        fixed_motion_times = getattr(self, "_eval_scenario_motion_times", None)
        if fixed_motion_ids is not None:
            # Evaluation-only replay hook. Normal training never defines these
            # tensors. It prevents action-dependent episode termination from
            # consuming a different motion RNG sequence in each ablation mode.
            motion_ids = fixed_motion_ids[env_ids]
            motion_times = fixed_motion_times[env_ids]
        elif motion_ids is None:
            motion_ids = self._motion_lib.sample_motions(n, motion_difficulty=self.motion_difficulty)
            if self._rand_reset:
                motion_times = self._motion_lib.sample_time(motion_ids)
            else:
                motion_times = torch.zeros(
                    motion_ids.shape, device=self.device, dtype=torch.float
                )
        else:
            if self._rand_reset:
                motion_times = self._motion_lib.sample_time(motion_ids)
            else:
                motion_times = torch.zeros(
                    motion_ids.shape, device=self.device, dtype=torch.float
                )
        
        self._motion_ids[env_ids] = motion_ids
        self._motion_time_offsets[env_ids] = motion_times
        
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos = self._motion_lib.calc_motion_frame(motion_ids, motion_times)
        root_pos[:, 2] += self.cfg.motion.height_offset
        
        self._ref_root_pos[env_ids] = root_pos
        self._ref_root_rot[env_ids] = root_rot
        self._ref_root_vel[env_ids] = root_vel
        self._ref_root_ang_vel[env_ids] = root_ang_vel
        self._ref_dof_pos[env_ids] = dof_pos
        self._ref_dof_vel[env_ids] = dof_vel
        if body_pos.shape[1] != self._ref_body_pos[env_ids].shape[1]:
            body_pos = g1_body_from_38_to_52(body_pos)
        self._ref_body_pos[env_ids] = convert_to_global_root_body_pos(root_pos=root_pos, root_rot=root_rot, body_pos=body_pos)
    
    
    def _update_ref_motion(self):
        motion_ids = self._motion_ids
        motion_times = self._get_motion_times()
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos = self._motion_lib.calc_motion_frame(motion_ids, motion_times)
        root_pos[:, 2] += self.cfg.motion.height_offset
        root_pos[:, :2] += self.episode_init_origin[:, :2]
        
        self._ref_root_pos[:] = root_pos
        self._ref_root_rot[:] = root_rot
        self._ref_root_vel[:] = root_vel
        self._ref_root_ang_vel[:] = root_ang_vel
        self._ref_dof_pos[:] = dof_pos
        self._ref_dof_vel[:] = dof_vel
        if body_pos.shape[1] != self._ref_body_pos.shape[1]:
            body_pos = g1_body_from_38_to_52(body_pos)
        self._ref_body_pos[:] = convert_to_global_root_body_pos(root_pos=root_pos, root_rot=root_rot, body_pos=body_pos)
        
    def _update_motion_difficulty(self, env_ids):
        if self.obs_type == 'priv':
            super()._update_motion_difficulty(env_ids)
        elif self.obs_type == 'student':
            super()._update_motion_difficulty(env_ids) # currently we use the same strategy for student
        else:
            return

    def _get_body_indices(self):
        upper_arm_names = [s for s in self.body_names if self.cfg.asset.upper_arm_name in s]
        lower_arm_names = [s for s in self.body_names if self.cfg.asset.lower_arm_name in s]
        torso_name = [s for s in self.body_names if self.cfg.asset.torso_name in s]
        self.torso_indices = torch.zeros(len(torso_name), dtype=torch.long, device=self.device,
                                                 requires_grad=False)
        for j in range(len(torso_name)):
            self.torso_indices[j] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                                  torso_name[j])
        self.upper_arm_indices = torch.zeros(len(upper_arm_names), dtype=torch.long, device=self.device,
                                                     requires_grad=False)
        for j in range(len(upper_arm_names)):
            self.upper_arm_indices[j] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                                upper_arm_names[j])
        self.lower_arm_indices = torch.zeros(len(lower_arm_names), dtype=torch.long, device=self.device,
                                                requires_grad=False)
        for j in range(len(lower_arm_names)):
            self.lower_arm_indices[j] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                                lower_arm_names[j])
        knee_names = [s for s in self.body_names if self.cfg.asset.shank_name in s]
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], knee_names[i])
    
    def _init_buffers(self):
        super()._init_buffers()
        self.obs_history_buf = torch.zeros((self.num_envs, self.cfg.env.history_len, self.cfg.env.n_obs_single), device=self.device)
        self.privileged_obs_history_buf = torch.zeros((self.num_envs, self.cfg.env.history_len, self.cfg.env.n_priv_obs_single), device=self.device)

        delay_max = int(getattr(self.cfg.domain_rand, "mimic_obs_delay_max", 0))
        delay_buf_len = max(1, delay_max + 1)

        self.mimic_obs_delay_buf = torch.zeros(
            (self.num_envs, delay_buf_len, self.cfg.env.n_mimic_obs),
            device=self.device,
        )

        self.mimic_obs_lpf_buf = torch.zeros(
            (self.num_envs, self.cfg.env.n_mimic_obs),
            device=self.device,
        )
        self._init_anyadapter_history()

    def reset_idx(self, env_ids, motion_ids=None):
        super().reset_idx(env_ids, motion_ids=motion_ids)
        self._reset_anyadapter_history(env_ids)
    
    def _get_noise_scale_vec(self, cfg):
        noise_scale_vec = torch.zeros(1, self.cfg.env.n_proprio, device=self.device)
        if not self.cfg.noise.add_noise:
            return noise_scale_vec
        ang_vel_dim = 3
        imu_dim = 2
        
        noise_scale_vec[:, 0:ang_vel_dim] = self.cfg.noise.noise_scales.ang_vel
        noise_scale_vec[:, ang_vel_dim:ang_vel_dim+imu_dim] = self.cfg.noise.noise_scales.imu
        noise_scale_vec[:, ang_vel_dim+imu_dim:ang_vel_dim+imu_dim+self.num_dof] = self.cfg.noise.noise_scales.dof_pos
        noise_scale_vec[:, ang_vel_dim+imu_dim+self.num_dof:ang_vel_dim+imu_dim+2*self.num_dof] = self.cfg.noise.noise_scales.dof_vel
        
        return noise_scale_vec
            
    def _get_mimic_obs(self):
        num_steps = self._tar_obs_steps.shape[0]
        assert num_steps > 0, "Invalid number of target observation steps"
        motion_times = self._get_motion_times().unsqueeze(-1)
        obs_motion_times = self._tar_obs_steps * self.dt + motion_times
        motion_ids_tiled = torch.broadcast_to(self._motion_ids.unsqueeze(-1), obs_motion_times.shape)
        motion_ids_tiled = motion_ids_tiled.flatten()
        obs_motion_times = obs_motion_times.flatten()
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos = self._motion_lib.calc_motion_frame(motion_ids_tiled, obs_motion_times)
        
        roll, pitch, yaw = euler_from_quaternion(root_rot)
        roll = roll.reshape(self.num_envs, num_steps, 1)
        pitch = pitch.reshape(self.num_envs, num_steps, 1)
        yaw = yaw.reshape(self.num_envs, num_steps, 1)
        if not self.global_obs:
            root_vel = quat_rotate_inverse(root_rot, root_vel)
            root_ang_vel = quat_rotate_inverse(root_rot, root_ang_vel)
      
        whole_key_body_pos = body_pos[:, self._key_body_ids_motion, :]
        if self.global_obs:
            whole_key_body_pos = convert_to_global_root_body_pos(root_pos=root_pos, root_rot=root_rot, body_pos=whole_key_body_pos)
        whole_key_body_pos = whole_key_body_pos.reshape(self.num_envs, num_steps, -1)
        
        root_pos = root_pos.reshape(self.num_envs, num_steps, root_pos.shape[-1])
        root_vel = root_vel.reshape(self.num_envs, num_steps, root_vel.shape[-1])
        root_rot = root_rot.reshape(self.num_envs, num_steps, root_rot.shape[-1])
        root_ang_vel = root_ang_vel.reshape(self.num_envs, num_steps, root_ang_vel.shape[-1])
        dof_pos = dof_pos.reshape(self.num_envs, num_steps, dof_pos.shape[-1])
     
        # teacher v0
        priv_mimic_obs_buf = torch.cat((
            root_pos[..., 2:3], # 1 dim
            roll, pitch, yaw, # 3 dims
            root_vel, # 3 dims
            root_ang_vel[..., 2:3], # 1 dim, yaw only
            dof_pos, # num_dof dims
            whole_key_body_pos, # num_bodies * 3 dims
        ), dim=-1) # shape: (num_envs, num_steps, 7 + num_dof + num_key_bodies * 3)
        
        
        # v6, align mocap
        mimic_obs_buf = torch.cat((
            root_pos[..., 2:3], # 1 dim
            roll, pitch, yaw, # 3 dims
            root_vel, # 3 dims
            root_ang_vel[..., 2:3], # 1 dim, yaw only
            dof_pos, # num_dof dims
        ), dim=-1)[:, 0:1] # shape: (num_envs, 1, 7 + num_dof)
        
        
        return priv_mimic_obs_buf.reshape(self.num_envs, -1), mimic_obs_buf.reshape(self.num_envs, -1)

    def _randomize_mimic_obs(self, mimic_obs):
        """Randomize student mimic input for sim2real robustness.

        This simulates real deployment issues:
        - mocap / retargeting noise
        - Redis frame hold or packet loss
        - high-level input delay
        - deployment-side low-pass filtering

        Only affects student observations.
        Does not change observation dimension.
        """
        if not getattr(self.cfg.domain_rand, "randomize_mimic_obs", False):
            return mimic_obs

        if getattr(self, "obs_type", None) != "student":
            return mimic_obs

        out = mimic_obs

        # 1. Add small noise to mimic input.
        noise_std = float(getattr(self.cfg.domain_rand, "mimic_obs_noise_std", 0.0))
        if noise_std > 0.0:
            out = out + noise_std * torch.randn_like(out)

        # Make sure buffers exist. This is defensive, in case parent init order changes.
        if not hasattr(self, "mimic_obs_delay_buf"):
            delay_max = int(getattr(self.cfg.domain_rand, "mimic_obs_delay_max", 0))
            delay_buf_len = max(1, delay_max + 1)
            self.mimic_obs_delay_buf = torch.zeros(
                (self.num_envs, delay_buf_len, self.cfg.env.n_mimic_obs),
                device=self.device,
            )

        if not hasattr(self, "mimic_obs_lpf_buf"):
            self.mimic_obs_lpf_buf = torch.zeros(
                (self.num_envs, self.cfg.env.n_mimic_obs),
                device=self.device,
            )

        # 2. Random frame hold / dropout.
        dropout_prob = float(getattr(self.cfg.domain_rand, "mimic_obs_dropout_prob", 0.0))
        if dropout_prob > 0.0:
            keep_old = torch.rand(self.num_envs, 1, device=self.device) < dropout_prob
            out = torch.where(keep_old, self.mimic_obs_delay_buf[:, -1], out)

        # 3. Update delay buffer and randomly use delayed mimic obs.
        delay_max = int(getattr(self.cfg.domain_rand, "mimic_obs_delay_max", 0))
        if delay_max > 0 and self.mimic_obs_delay_buf.shape[1] > 1:
            delay_max = min(delay_max, self.mimic_obs_delay_buf.shape[1] - 1)

            self.mimic_obs_delay_buf = torch.cat(
                [self.mimic_obs_delay_buf[:, 1:], out.unsqueeze(1)],
                dim=1,
            )

            delay_ids = torch.randint(
                low=0,
                high=delay_max + 1,
                size=(self.num_envs,),
                device=self.device,
            )

            gather_ids = self.mimic_obs_delay_buf.shape[1] - 1 - delay_ids

            out = self.mimic_obs_delay_buf[
                torch.arange(self.num_envs, device=self.device),
                gather_ids,
            ]
        else:
            self.mimic_obs_delay_buf[:, -1] = out

        # 4. Random low-pass filtering.
        lpf_prob = float(getattr(self.cfg.domain_rand, "mimic_obs_lpf_prob", 0.0))
        if lpf_prob > 0.0:
            alpha_range = getattr(
                self.cfg.domain_rand,
                "mimic_obs_lpf_alpha_range",
                [0.6, 0.9],
            )

            alpha = torch.empty(self.num_envs, 1, device=self.device).uniform_(
                float(alpha_range[0]),
                float(alpha_range[1]),
            )

            filtered = alpha * self.mimic_obs_lpf_buf + (1.0 - alpha) * out
            use_lpf = torch.rand(self.num_envs, 1, device=self.device) < lpf_prob
            out = torch.where(use_lpf, filtered, out)

            self.mimic_obs_lpf_buf = out.detach()

        return out

    def compute_observations(self):
        imu_obs = torch.stack((self.roll, self.pitch), dim=1)
        self.base_yaw_quat = quat_from_euler_xyz(0*self.yaw, 0*self.yaw, self.yaw)
        priv_mimic_obs, mimic_obs = self._get_mimic_obs()
        mimic_obs = self._randomize_mimic_obs(mimic_obs)

        proprio_obs_buf = torch.cat((
                            self.base_ang_vel  * self.obs_scales.ang_vel,   # 3 dims
                            imu_obs,    # 2 dims
                            self.reindex((self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos),
                            self.reindex(self.dof_vel * self.obs_scales.dof_vel),
                            self.reindex(self.action_history_buf[:, -1]),
                            ),dim=-1)
        
        if self.cfg.noise.add_noise and self.headless:
            proprio_obs_buf += (2 * torch.rand_like(proprio_obs_buf) - 1) * self.noise_scale_vec * min(self.total_env_steps_counter / (self.cfg.noise.noise_increasing_steps * 24),  1.)
        elif self.cfg.noise.add_noise and not self.headless:
            proprio_obs_buf += (2 * torch.rand_like(proprio_obs_buf) - 1) * self.noise_scale_vec
        else:
            proprio_obs_buf += 0.
        dof_vel_start_dim = 5 + self.dof_pos.shape[1]

        # disable ankle dof
        ankle_idx = [4, 5, 10, 11]
        proprio_obs_buf[:, [dof_vel_start_dim + i for i in ankle_idx]] = 0.
        
        key_body_pos = self.rigid_body_states[:, self._key_body_ids, :3]
        key_body_pos = key_body_pos - self.root_states[:, None, :3]
        if not self.global_obs:
            key_body_pos = convert_to_local_root_body_pos(self.root_states[:, 3:7], key_body_pos)
        key_body_pos = key_body_pos.reshape(self.num_envs, -1) # shape: (num_envs, num_key_bodies * 3)
        
        if self.cfg.domain_rand.domain_rand_general:
            priv_info = torch.cat((
                self.base_lin_vel, # 3 dims
                self.root_states[:, 2:3], # 1 dim
                key_body_pos, # num_bodies * 3 dims
                self.contact_forces[:, self.feet_indices, 2] > 5., # 2 dims, foot contact
                self.mass_params_tensor,
                self.friction_coeffs_tensor,
                self.motor_strength[0] - 1, 
                self.motor_strength[1] - 1,
            ), dim=-1)
        else:
            priv_info = torch.zeros((self.num_envs, self.cfg.env.n_priv_info), device=self.device)
        
        obs_buf = torch.cat((
            mimic_obs,
            proprio_obs_buf,
        ), dim=-1)
        
        priv_obs_buf = torch.cat((
            priv_mimic_obs,
            proprio_obs_buf,
            priv_info,
        ), dim=-1)
        
        self.privileged_obs_buf = priv_obs_buf
        
        if self.obs_type == 'priv':
            self.obs_buf = priv_obs_buf
        elif self.obs_type == 'student':
            self.obs_buf = torch.cat([obs_buf, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        
        if self.cfg.env.history_len > 0:
            self.privileged_obs_history_buf = torch.where(
                (self.episode_length_buf <= 1)[:, None, None], 
                torch.stack([priv_obs_buf] * self.cfg.env.history_len, dim=1),
                torch.cat([
                    self.privileged_obs_history_buf[:, 1:],
                    priv_obs_buf.unsqueeze(1)
                ], dim=1)
            )
            if self.obs_type == 'priv':
                self.obs_history_buf[:] = self.privileged_obs_history_buf[:]
            elif self.obs_type == 'student':
                self.obs_history_buf = torch.where(
                    (self.episode_length_buf <= 1)[:, None, None], 
                    torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
                    torch.cat([
                        self.obs_history_buf[:, 1:],
                        obs_buf.unsqueeze(1)
                    ], dim=1)
                )
        self.obs_buf = self._append_anyadapter_history(
            self.obs_buf,
            self.actions,
            tracking_reference=mimic_obs,
        )


############################################################################################################
##################################### Extra Reward Functions################################################
############################################################################################################

    def _reward_waist_dof_acc(self):
        waist_dof_idx = [13, 14]
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt)[:, waist_dof_idx], dim=1)
    
    def _reward_waist_dof_vel(self):
        waist_dof_idx = [13, 14]
        return torch.sum(torch.square(self.dof_vel[:, waist_dof_idx]), dim=1)
    
    def _reward_ankle_dof_acc(self):
        ankle_dof_idx = [4, 5, 10, 11]
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt)[:, ankle_dof_idx], dim=1)
    
    def _reward_ankle_dof_vel(self):
        ankle_dof_idx = [4, 5, 10, 11]
        return torch.sum(torch.square(self.dof_vel[:, ankle_dof_idx]), dim=1)
    
    def _reward_ankle_action(self):
        return torch.norm(self.action_history_buf[:, -1, [4, 5, 10, 11]], dim=1)

    def _in_place_reference_mask(self):
        linear_threshold = float(
            getattr(self.cfg.rewards, "in_place_ref_vel_threshold", 0.12)
        )
        yaw_threshold = float(
            getattr(self.cfg.rewards, "in_place_ref_yaw_vel_threshold", 0.12)
        )
        return (
            (torch.norm(self._ref_root_vel[:, :2], dim=1) < linear_threshold)
            & (torch.abs(self._ref_root_ang_vel[:, 2]) < yaw_threshold)
        )

    def _reward_in_place_root_motion(self):
        """Penalize uncommanded planar drift during in-place references."""
        root_motion = torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1)
        root_motion += 0.25 * torch.square(self.base_ang_vel[:, 2])
        return root_motion * self._in_place_reference_mask().float()

    def _reward_in_place_feet_motion(self):
        """Penalize support switching and foot shuffling for in-place motions."""
        feet_velocity = self.rigid_body_states[:, self.feet_indices, 7:10]
        feet_motion = torch.sum(torch.square(feet_velocity), dim=(1, 2))
        return feet_motion * self._in_place_reference_mask().float()


class G1MimicRecorder(G1MimicDistill):
    """Environment subclass that records physically-consistent trajectories from teacher rollouts."""
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self._rec_frames = []

    def _init_recording(self):
        self._rec_frames = []

    def _record_current_state(self):
        body_pos_global = self.rigid_body_states[:, 1:, :3]  # skip world (index 0)
        local_body_pos = compute_local_body_pos(
            self.root_states[:, :3],
            self.root_states[:, 3:7],
            body_pos_global,
        )
        self._rec_frames.append({
            'root_pos': self.root_states[:, :3].clone(),
            'root_rot': self.root_states[:, 3:7].clone(),
            'dof_pos': self.dof_pos.clone(),
            'local_body_pos': local_body_pos.clone(),
        })

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self._record_current_state()

    def check_termination(self):
        """Relaxed termination: only catastrophic failures (fall, roll/pitch), no pose tracking."""
        self.reset_buf[:] = 0
        self.reset_buf = torch.where(
            self.contact_forces[:, self.termination_contact_indices, :].max(dim=-1).values.max(dim=-1).values > 1.,
            torch.ones_like(self.reset_buf), self.reset_buf,
        )
        self.reset_buf = torch.where(
            self.root_states[:, 2] < self.cfg.rewards.root_height_diff_threshold,
            torch.ones_like(self.reset_buf), self.reset_buf,
        )
        self.reset_buf = torch.where(
            torch.abs(self.roll) > 1.2,
            torch.ones_like(self.reset_buf), self.reset_buf,
        )
        self.reset_buf = torch.where(
            torch.abs(self.pitch) > 1.2,
            torch.ones_like(self.reset_buf), self.reset_buf,
        )
        vel_too_large = torch.norm(self.root_states[:, 7:10], dim=-1) > 5.
        self.reset_buf |= vel_too_large
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf

    def reset_to_motion(self, motion_id):
        """Reset env to a specific motion at time 0 and begin recording."""
        env_ids = torch.tensor([0], device=self.device)
        motion_ids = torch.tensor([motion_id], device=self.device)
        self._reset_ref_motion(env_ids=env_ids, motion_ids=motion_ids)
        vel_factor = 0.8
        self._reset_dofs(env_ids, self._ref_dof_pos, self._ref_dof_vel * vel_factor)
        self._reset_root_states(env_ids=env_ids, root_vel=self._ref_root_vel * vel_factor,
                                root_quat=self._ref_root_rot)
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(
            torch.zeros(self.num_envs, self.num_dof, device=self.device)))
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        # Compute derived quantities needed by compute_observations
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)
        self.last_actions[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.compute_observations()  # must recompute obs after state change
        self._init_recording()
        self._record_current_state()
