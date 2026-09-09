# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import os
import json
from collections import deque
import statistics
from rich import print
# from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
import wandb
# import ml_runlog
import datetime

import builtins


import numpy as np
from rsl_rl.algorithms import PPORMA, PPO, PPOAnyAdapter, PPOAny2Track, PPODTERA
from rsl_rl.modules import *
from rsl_rl.storage.replay_buffer import ReplayBuffer
from rsl_rl.env import VecEnv
import sys
from copy import copy, deepcopy
import warnings
# from rsl_rl.utils.running_mean_std import RunningMeanStd
from rsl_rl.utils.normalizer import Normalizer


class OnPolicyRunnerMimic:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 init_wandb=True,
                 device='cpu', **kwargs):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        self.normalize_obs = env.cfg.env.normalize_obs

        policy_class = eval(self.cfg["policy_class_name"])
        if "Transformer" in self.cfg["policy_class_name"]:
            actor_critic = policy_class(num_prop=self.env.cfg.env.n_proprio,
                                        num_critic_obs=self.env.num_privileged_obs,
                                        num_priv_latent=self.env.cfg.env.n_priv_latent,
                                        num_hist=self.env.cfg.env.history_len,
                                        num_actions=self.env.num_actions,
                                        **self.policy_cfg).to(self.device)
            print("Number of parameters: ", sum(p.numel() for p in actor_critic.parameters()))
        else:
            actor_critic = policy_class(num_observations=self.env.num_obs,
                                        num_critic_observations=self.env.num_privileged_obs,
                                        num_motion_observations=self.env.cfg.env.n_priv_mimic_obs,
                                        num_motion_steps=len(self.env.cfg.env.tar_obs_steps),
                                        num_actions=self.env.num_actions,
                                        **self.policy_cfg).to(self.device)

        share_normalizer = (self.env.num_obs == self.env.num_privileged_obs) or self.env.num_privileged_obs is None

        if self.normalize_obs:
            if share_normalizer:
                self.normalizer = Normalizer(shape=self.env.num_obs, device=self.device, dtype=env.obs_buf.dtype)
                self.critic_normalizer = None
            else:
                self.normalizer = Normalizer(shape=self.env.num_obs, device=self.device, dtype=env.obs_buf.dtype)
                self.critic_normalizer = Normalizer(shape=self.env.num_privileged_obs, device=self.device, dtype=env.obs_buf.dtype)
        else:
            self.normalizer = None
            self.critic_normalizer = None

        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        self.alg = alg_class(self.env,
                                  actor_critic,
                                  device=self.device, **self.alg_cfg)
        self._print_anyadapter_init_debug(actor_critic)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.dagger_update_freq = self.alg_cfg["dagger_update_freq"]

        if "Transformer" in self.cfg["policy_class_name"]:
            self.alg.init_storage(
                self.env.num_envs,
                self.num_steps_per_env,
                [self.policy_cfg["obs_context_len"], self.env.num_obs],
                [self.policy_cfg["obs_context_len"], self.env.num_privileged_obs],
                [self.env.num_actions],
            )
        else:
            self.alg.init_storage(
                self.env.num_envs,
                self.num_steps_per_env,
                [self.env.num_obs],
                [self.env.num_privileged_obs],
                [self.env.num_actions],
            )

        self.learn = self.learn_RL

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

    def _print_anyadapter_init_debug(self, actor_critic):
        is_anyadapter = (
            "AnyAdapter" in self.cfg["policy_class_name"]
            or "AnyAdapter" in self.cfg["algorithm_class_name"]
            or "base_obs_dim" in self.policy_cfg
        )
        if not is_anyadapter:
            return

        base_obs_dim = self.policy_cfg.get("base_obs_dim", None)
        history_len = self.policy_cfg.get("history_len", None)
        hist_state_dim = self.policy_cfg.get("hist_state_dim", None)
        history_frame_dim = self.policy_cfg.get("history_frame_dim", None)
        adapter_context_dim = self.policy_cfg.get("adapter_context_dim", 0)
        tracking_history_len = self.policy_cfg.get("tracking_history_len", 0)
        tracking_error_frame_dim = self.policy_cfg.get("tracking_error_frame_dim", 0)
        action_delta_scale = self.policy_cfg.get("action_delta_scale", None)
        adapter_gain = self.policy_cfg.get("adapter_gain", None)
        use_tracking_error_adapter_input = self.policy_cfg.get("use_tracking_error_adapter_input", False)
        compact_adapter_input = self.policy_cfg.get("compact_adapter_input", False)
        history_policy_grad_scale = self.policy_cfg.get("history_policy_grad_scale", 0.0)
        base_actor_jit_path = self.policy_cfg.get("base_actor_jit_path", None)
        freeze_base = self.policy_cfg.get("freeze_base", None)
        init_noise_std = self.policy_cfg.get("init_noise_std", None)
        adapter_reg_coef = self.alg_cfg.get("adapter_reg_coef", None)
        adapter_bias_reg_coef = self.alg_cfg.get("adapter_bias_reg_coef", None)
        world_model_loss_coef = self.alg_cfg.get("world_model_loss_coef", None)
        joint_encoder_optimization = self.alg_cfg.get("joint_encoder_optimization", False)
        weight_decay = self.alg_cfg.get("weight_decay", None)

        expected_full_obs_dim = None
        anyadapter_history_dim = None
        if base_obs_dim is not None and history_len is not None and history_frame_dim is not None:
            anyadapter_history_dim = int(history_len) * int(history_frame_dim)
            expected_full_obs_dim = (
                int(base_obs_dim)
                + anyadapter_history_dim
                + int(tracking_history_len) * int(tracking_error_frame_dim)
                + int(adapter_context_dim)
            )

        builtins.print("[AnyAdapter] ========================================")
        builtins.print("[AnyAdapter] runner init")
        builtins.print(f"[AnyAdapter] policy_class_name: {self.cfg['policy_class_name']}")
        builtins.print(f"[AnyAdapter] algorithm_class_name: {self.cfg['algorithm_class_name']}")
        builtins.print(f"[AnyAdapter] actor_critic class: {actor_critic.__class__.__name__}")
        builtins.print(f"[AnyAdapter] alg class: {self.alg.__class__.__name__}")
        builtins.print(f"[AnyAdapter] cfg.policy.base_actor_jit_path: {base_actor_jit_path}")
        builtins.print(f"[AnyAdapter] cfg.policy.base_obs_dim: {base_obs_dim}")
        builtins.print(f"[AnyAdapter] cfg.policy.history_len: {history_len}")
        builtins.print(f"[AnyAdapter] cfg.policy.hist_state_dim: {hist_state_dim}")
        builtins.print(f"[AnyAdapter] cfg.policy.history_frame_dim: {history_frame_dim}")
        builtins.print(f"[AnyAdapter] cfg.policy.tracking_history_len: {tracking_history_len}")
        builtins.print(f"[AnyAdapter] cfg.policy.tracking_error_frame_dim: {tracking_error_frame_dim}")
        builtins.print(f"[AnyAdapter] cfg.policy.adapter_context_dim: {adapter_context_dim}")
        builtins.print(f"[AnyAdapter] cfg.policy.action_delta_scale: {action_delta_scale}")
        builtins.print(f"[AnyAdapter] cfg.policy.adapter_gain: {adapter_gain}")
        builtins.print(f"[AnyAdapter] cfg.policy.use_tracking_error_adapter_input: {use_tracking_error_adapter_input}")
        builtins.print(f"[AnyAdapter] cfg.policy.compact_adapter_input: {compact_adapter_input}")
        builtins.print(f"[AnyAdapter] cfg.policy.history_policy_grad_scale: {history_policy_grad_scale}")
        builtins.print(f"[AnyAdapter] cfg.policy.init_noise_std: {init_noise_std}")
        builtins.print(f"[AnyAdapter] cfg.policy.freeze_base: {freeze_base}")
        builtins.print(f"[AnyAdapter] cfg.algorithm.adapter_reg_coef: {adapter_reg_coef}")
        builtins.print(f"[AnyAdapter] cfg.algorithm.adapter_bias_reg_coef: {adapter_bias_reg_coef}")
        builtins.print(f"[AnyAdapter] cfg.algorithm.world_model_loss_coef: {world_model_loss_coef}")
        builtins.print(f"[AnyAdapter] cfg.algorithm.joint_encoder_optimization: {joint_encoder_optimization}")
        builtins.print(f"[AnyAdapter] cfg.algorithm.weight_decay: {weight_decay}")
        builtins.print(f"[AnyAdapter] full obs dim: {self.env.num_obs}")
        builtins.print(f"[AnyAdapter] base_obs dim: {base_obs_dim}")
        builtins.print(f"[AnyAdapter] anyadapter history dim: {anyadapter_history_dim}")
        builtins.print(f"[AnyAdapter] expected full obs dim: {expected_full_obs_dim}")
        builtins.print("[AnyAdapter] ========================================")


    def learn_RL(self, num_learning_iterations, init_at_random_ep_len=False):
        mean_value_loss = 0.
        mean_surrogate_loss = 0.
        mean_disc_loss = 0.
        mean_disc_acc = 0.
        mean_hist_latent_loss = 0.
        mean_priv_reg_loss = 0.
        priv_reg_coef = 0.
        entropy_coef = 0.
        grad_penalty_coef = 0.

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        if self.normalize_obs:
            obs = self.normalizer.normalize(obs)
            critic_obs = self.normalizer.normalize(critic_obs) if self.critic_normalizer is None else self.critic_normalizer.normalize(critic_obs)
        infos = {}
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        rew_explr_buffer = deque(maxlen=100)
        rew_entropy_buffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_reward_explr_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_reward_entropy_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        task_rew_buf = deque(maxlen=100)
        cur_task_rew_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        self.start_learning_iteration = copy(self.current_learning_iteration)

        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            hist_encoding = it % self.dagger_update_freq == 0
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs, infos, hist_encoding)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)  # obs has changed to next_obs !! if done obs has been reset
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)

                    if self.normalize_obs:
                        before_norm_obs = obs.clone()
                        before_norm_critic_obs = critic_obs.clone()
                        obs = self.normalizer.normalize(obs)
                        critic_obs = self.normalizer.normalize(critic_obs) if self.critic_normalizer is None else self.critic_normalizer.normalize(critic_obs)
                        if self._need_normalizer_update(it, self.alg_cfg["normalizer_update_iterations"]):
                            self.normalizer.record(before_norm_obs)
                            if self.critic_normalizer is not None:
                                self.critic_normalizer.record(before_norm_critic_obs)

                    if getattr(self.alg, "requires_next_observations", False):
                        infos["next_observations"] = obs.detach().clone()

                    total_rew = self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += total_rew
                        cur_reward_explr_sum += 0
                        cur_reward_entropy_sum += 0
                        cur_episode_length += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)

                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        rew_explr_buffer.extend(cur_reward_explr_sum[new_ids][:, 0].cpu().numpy().tolist())
                        rew_entropy_buffer.extend(cur_reward_entropy_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())

                        cur_reward_sum[new_ids] = 0
                        cur_reward_explr_sum[new_ids] = 0
                        cur_reward_entropy_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                stop = time.time()
                collection_time = stop - start
                if self.normalize_obs:
                    if self._need_normalizer_update(it, self.alg_cfg["normalizer_update_iterations"]):
                        self.normalizer.update()
                        if self.critic_normalizer is not None:
                            self.critic_normalizer.update()

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            regularization_scale = self.env.cfg.rewards.regularization_scale if hasattr(self.env.cfg.rewards, "regularization_scale") else 1
            average_episode_length = torch.mean(self.env.episode_length.float()).item() if hasattr(self.env, "episode_length") else 0
            mean_motion_difficulty = self.env.mean_motion_difficulty if hasattr(self.env, "mean_motion_difficulty") else 0
            mean_value_loss, mean_surrogate_loss, mean_priv_reg_loss, priv_reg_coef, mean_grad_penalty_loss, grad_penalty_coef = self.alg.update()
            if hist_encoding and not self.cfg["algorithm_class_name"] == "PPO" and not getattr(self.alg, "skip_dagger_update", False):
                print("Updating dagger...")
                mean_hist_latent_loss = self.alg.update_dagger()

            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            completed_iteration = it + 1
            if completed_iteration < 2500:
                if completed_iteration % self.save_interval == 0:
                    self.save(
                        os.path.join(
                            self.log_dir,
                            'model_{}.pt'.format(completed_iteration),
                        ),
                        iteration=completed_iteration,
                    )
            elif completed_iteration < 5000:
                if completed_iteration % (2*self.save_interval) == 0:
                    self.save(
                        os.path.join(
                            self.log_dir,
                            'model_{}.pt'.format(completed_iteration),
                        ),
                        iteration=completed_iteration,
                    )
            else:
                if completed_iteration % (5*self.save_interval) == 0:
                    self.save(
                        os.path.join(
                            self.log_dir,
                            'model_{}.pt'.format(completed_iteration),
                        ),
                        iteration=completed_iteration,
                    )
            ep_infos.clear()

        # Keep the final checkpoint name aligned with the completed iteration.
        # Resume loading already derives this value from model_<iteration>.pt.
        self.current_learning_iteration = tot_iter
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def _need_normalizer_update(self, iterations, update_iterations):
        return iterations < update_iterations

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        wandb_dict = {}
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # wandb_dict['Episode_rew/' + key] = value
                if "metric" in key:
                    wandb_dict['Episode_rew_metrics/' + key] = value
                else:
                    if "tracking" in key:
                        wandb_dict['Episode_rew_tracking/' + key] = value
                    elif "curriculum" in key:
                        wandb_dict['Episode_curriculum/' + key] = value
                    else:
                        wandb_dict['Episode_rew_regularization/' + key] = value
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n""" # dont print metrics
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        wandb_dict['Loss/value_func'] = locs['mean_value_loss']
        wandb_dict['Loss/surrogate'] = locs['mean_surrogate_loss']
        wandb_dict['Loss/entropy_coef'] = locs['entropy_coef']
        wandb_dict['Loss/learning_rate'] = self.alg.learning_rate

        anyadapter_metrics = getattr(self.alg, "anyadapter_metrics", None)
        anyadapter_log_string = ""
        if anyadapter_metrics:
            wandb_dict['AnyAdapter/world_model_loss'] = anyadapter_metrics.get("world_model_loss", 0.0)
            wandb_dict['AnyAdapter/world_model_loss_skipped'] = anyadapter_metrics.get("world_model_loss_skipped", 0.0)
            for component in ("ang_vel", "orientation", "dof_pos", "dof_vel"):
                key = f"world_model_loss_{component}"
                if key in anyadapter_metrics:
                    wandb_dict[f'AnyAdapter/{key}'] = anyadapter_metrics[key]
                canonical_key = f"wm_{component}_loss"
                wandb_dict[f'AnyAdapter/{canonical_key}'] = anyadapter_metrics.get(canonical_key, 0.0)
            wandb_dict['AnyAdapter/adapter_delta_l2'] = anyadapter_metrics.get("adapter_delta_l2", 0.0)
            wandb_dict['AnyAdapter/dynamics_delta_l2'] = anyadapter_metrics.get("dynamics_delta_l2", 0.0)
            wandb_dict['AnyAdapter/tracking_delta_l2'] = anyadapter_metrics.get("tracking_delta_l2", 0.0)
            wandb_dict['AnyAdapter/dynamics_mean_abs_delta'] = anyadapter_metrics.get("dynamics_mean_abs_delta", 0.0)
            wandb_dict['AnyAdapter/tracking_mean_abs_delta'] = anyadapter_metrics.get("tracking_mean_abs_delta", 0.0)
            wandb_dict['AnyAdapter/dynamics_max_abs_delta'] = anyadapter_metrics.get("dynamics_max_abs_delta", 0.0)
            wandb_dict['AnyAdapter/tracking_max_abs_delta'] = anyadapter_metrics.get("tracking_max_abs_delta", 0.0)
            wandb_dict['AnyAdapter/branch_balance_ratio'] = anyadapter_metrics.get("branch_balance_ratio", 0.0)
            wandb_dict['AnyAdapter/branch_cosine_similarity'] = anyadapter_metrics.get("branch_cosine_similarity", 0.0)
            wandb_dict['AnyAdapter/adapter_reg_loss'] = anyadapter_metrics.get("adapter_reg_loss", 0.0)
            wandb_dict['AnyAdapter/effective_adapter_reg_coef'] = anyadapter_metrics.get("effective_adapter_reg_coef", 0.0)
            wandb_dict['AnyAdapter/residual_saturation_penalty'] = anyadapter_metrics.get("residual_saturation_penalty", 0.0)
            wandb_dict['AnyAdapter/adapter_bias_reg_loss'] = anyadapter_metrics.get("adapter_bias_reg_loss", 0.0)
            wandb_dict['AnyAdapter/stand_anchor_loss'] = anyadapter_metrics.get("stand_anchor_loss", 0.0)
            wandb_dict['AnyAdapter/synthetic_stand_anchor_loss'] = anyadapter_metrics.get("synthetic_stand_anchor_loss", 0.0)
            wandb_dict['AnyAdapter/stand_sample_ratio'] = anyadapter_metrics.get("stand_sample_ratio", 0.0)
            wandb_dict['AnyAdapter/history_encoder_ppo_grad_norm'] = anyadapter_metrics.get("history_encoder_ppo_grad_norm", 0.0)
            wandb_dict['AnyAdapter/history_encoder_wm_grad_norm'] = anyadapter_metrics.get("history_encoder_wm_grad_norm", 0.0)
            wandb_dict['AnyAdapter/history_encoder_total_grad_norm'] = anyadapter_metrics.get("history_encoder_total_grad_norm", 0.0)
            wandb_dict['AnyAdapter/adapter_grad_norm'] = anyadapter_metrics.get("adapter_grad_norm", 0.0)
            wandb_dict['AnyAdapter/dynamics_branch_grad_norm'] = anyadapter_metrics.get("dynamics_branch_grad_norm", 0.0)
            wandb_dict['AnyAdapter/tracking_branch_grad_norm'] = anyadapter_metrics.get("tracking_branch_grad_norm", 0.0)
            wandb_dict['AnyAdapter/max_abs_delta_action'] = anyadapter_metrics.get("max_abs_delta_action", 0.0)
            wandb_dict['AnyAdapter/mean_abs_delta_action'] = anyadapter_metrics.get("mean_abs_delta_action", 0.0)
            wandb_dict['AnyAdapter/surrogate_loss'] = anyadapter_metrics.get("surrogate_loss", locs['mean_surrogate_loss'])
            wandb_dict['AnyAdapter/value_loss'] = anyadapter_metrics.get("value_loss", locs['mean_value_loss'])
            anyadapter_log_string = (
                f"""{'AnyAdapter wm loss:':>{pad}} {anyadapter_metrics.get('world_model_loss', 0.0):.6f}\n"""
                f"""{'AnyAdapter wm skipped:':>{pad}} {anyadapter_metrics.get('world_model_loss_skipped', 0.0):.0f}\n"""
                f"""{'AnyAdapter wm gyro:':>{pad}} {anyadapter_metrics.get('world_model_loss_ang_vel', 0.0):.6f}\n"""
                f"""{'AnyAdapter wm orient:':>{pad}} {anyadapter_metrics.get('world_model_loss_orientation', 0.0):.6f}\n"""
                f"""{'AnyAdapter wm dof pos:':>{pad}} {anyadapter_metrics.get('world_model_loss_dof_pos', 0.0):.6f}\n"""
                f"""{'AnyAdapter wm dof vel:':>{pad}} {anyadapter_metrics.get('world_model_loss_dof_vel', 0.0):.6f}\n"""
                f"""{'AnyAdapter delta L2:':>{pad}} {anyadapter_metrics.get('adapter_delta_l2', 0.0):.6f}\n"""
                f"""{'AnyAdapter dyn delta L2:':>{pad}} {anyadapter_metrics.get('dynamics_delta_l2', 0.0):.6f}\n"""
                f"""{'AnyAdapter err delta L2:':>{pad}} {anyadapter_metrics.get('tracking_delta_l2', 0.0):.6f}\n"""
                f"""{'AnyAdapter dyn mean |delta|:':>{pad}} {anyadapter_metrics.get('dynamics_mean_abs_delta', 0.0):.6f}\n"""
                f"""{'AnyAdapter err mean |delta|:':>{pad}} {anyadapter_metrics.get('tracking_mean_abs_delta', 0.0):.6f}\n"""
                f"""{'AnyAdapter dyn max |delta|:':>{pad}} {anyadapter_metrics.get('dynamics_max_abs_delta', 0.0):.6f}\n"""
                f"""{'AnyAdapter err max |delta|:':>{pad}} {anyadapter_metrics.get('tracking_max_abs_delta', 0.0):.6f}\n"""
                f"""{'AnyAdapter branch balance:':>{pad}} {anyadapter_metrics.get('branch_balance_ratio', 0.0):.6f}\n"""
                f"""{'AnyAdapter branch cosine:':>{pad}} {anyadapter_metrics.get('branch_cosine_similarity', 0.0):.6f}\n"""
                f"""{'AnyAdapter adapter reg:':>{pad}} {anyadapter_metrics.get('adapter_reg_loss', 0.0):.6f}\n"""
                f"""{'AnyAdapter effective reg:':>{pad}} {anyadapter_metrics.get('effective_adapter_reg_coef', 0.0):.6f}\n"""
                f"""{'AnyAdapter saturation reg:':>{pad}} {anyadapter_metrics.get('residual_saturation_penalty', 0.0):.6f}\n"""
                f"""{'AnyAdapter bias reg:':>{pad}} {anyadapter_metrics.get('adapter_bias_reg_loss', 0.0):.6f}\n"""
                f"""{'AnyAdapter stand anchor:':>{pad}} {anyadapter_metrics.get('stand_anchor_loss', 0.0):.6f}\n"""
                f"""{'AnyAdapter synth stand:':>{pad}} {anyadapter_metrics.get('synthetic_stand_anchor_loss', 0.0):.6f}\n"""
                f"""{'AnyAdapter stand ratio:':>{pad}} {anyadapter_metrics.get('stand_sample_ratio', 0.0):.6f}\n"""
                f"""{'AnyAdapter hist PPO grad:':>{pad}} {anyadapter_metrics.get('history_encoder_ppo_grad_norm', 0.0):.6e}\n"""
                f"""{'AnyAdapter hist WM grad:':>{pad}} {anyadapter_metrics.get('history_encoder_wm_grad_norm', 0.0):.6e}\n"""
                f"""{'AnyAdapter hist total grad:':>{pad}} {anyadapter_metrics.get('history_encoder_total_grad_norm', 0.0):.6e}\n"""
                f"""{'AnyAdapter dyn grad:':>{pad}} {anyadapter_metrics.get('dynamics_branch_grad_norm', 0.0):.6e}\n"""
                f"""{'AnyAdapter err grad:':>{pad}} {anyadapter_metrics.get('tracking_branch_grad_norm', 0.0):.6e}\n"""
                f"""{'AnyAdapter adapter grad:':>{pad}} {anyadapter_metrics.get('adapter_grad_norm', 0.0):.6e}\n"""
                f"""{'AnyAdapter max |delta|:':>{pad}} {anyadapter_metrics.get('max_abs_delta_action', 0.0):.6f}\n"""
                f"""{'AnyAdapter mean |delta|:':>{pad}} {anyadapter_metrics.get('mean_abs_delta_action', 0.0):.6f}\n"""
                f"""{'AnyAdapter surrogate:':>{pad}} {anyadapter_metrics.get('surrogate_loss', locs['mean_surrogate_loss']):.6f}\n"""
                f"""{'AnyAdapter value loss:':>{pad}} {anyadapter_metrics.get('value_loss', locs['mean_value_loss']):.6f}\n"""
            )
            dtera_metric_labels = {
                "ppo_learning_rate": "DTERA PPO learning rate",
                "error_prediction_loss": "DTERA error pred loss",
                "tracking_encoder_ppo_grad_norm": "DTERA error enc PPO grad",
                "tracking_encoder_aux_grad_norm": "DTERA error enc aux grad",
                "error_predictor_grad_norm": "DTERA error pred grad",
                "z_e_norm": "DTERA z_e norm",
                "wm_uncertainty_mean": "DTERA WM uncertainty",
                "wm_uncertainty_p90": "DTERA WM uncertainty p90",
                "wm_uncertainty_p95": "DTERA WM uncertainty p95",
                "wm_uncertainty_error_corr": "DTERA WM unc/error corr",
                "actual_wm_error_low_uncertainty": "DTERA WM error low unc",
                "actual_wm_error_high_uncertainty": "DTERA WM error high unc",
                "tracking_demand_mean": "DTERA demand mean",
                "tracking_demand_p90": "DTERA demand p90",
                "dynamics_demand_mean": "DTERA dyn demand mean",
                "dynamics_demand_p90": "DTERA dyn demand p90",
                "gate_confidence_mean": "DTERA confidence mean",
                "safety_factor_mean": "DTERA safety mean",
                "demand_confidence_gate_mean": "DTERA D*Ceff mean",
                "full_diagnostic_gate_mean": "DTERA D*Ceff*S mean",
                "residual_warmup_factor": "DTERA residual alpha",
                "dyn_saturation_fraction": "DTERA dyn saturation",
                "err_saturation_fraction": "DTERA err saturation",
                "candidate_saturation_fraction": "DTERA candidate saturation",
                "dynamics_output_bias_norm": "DTERA dyn output bias norm",
                "tracking_output_bias_norm": "DTERA err output bias norm",
                "gate_mean": "DTERA gate mean",
                "gate_p10": "DTERA gate p10",
                "gate_p90": "DTERA gate p90",
                "gate_fraction_lt_0_1": "DTERA gate frac <.1",
                "gate_fraction_gt_0_9": "DTERA gate frac >.9",
                "dynamics_gate_mean": "DTERA dyn gate mean",
                "dynamics_gate_p10": "DTERA dyn gate p10",
                "dynamics_gate_p90": "DTERA dyn gate p90",
                "tracking_gate_mean": "DTERA err gate mean",
                "tracking_gate_p10": "DTERA err gate p10",
                "tracking_gate_p90": "DTERA err gate p90",
                "risk_loss": "DTERA risk loss",
                "risk_positive_ratio": "DTERA risk positive",
                "risk_valid_ratio": "DTERA risk valid",
                "risk_num_positive": "DTERA risk positives",
                "risk_num_negative": "DTERA risk negatives",
                "risk_update_skipped": "DTERA risk skipped",
                "risk_effective_pos_weight": "DTERA risk pos weight",
                "risk_prob_positive": "DTERA risk p positive",
                "risk_prob_negative": "DTERA risk p negative",
                "risk_probability_gap": "DTERA risk p gap",
                "p_base_mean": "DTERA p_base",
                "p_candidate_mean": "DTERA p_candidate",
                "delta_risk_mean": "DTERA delta risk",
                "delta_risk_p95": "DTERA delta risk p95",
                "candidate_delta_l2": "DTERA candidate L2",
                "gated_delta_l2": "DTERA gated L2",
                "applied_delta_l2": "DTERA applied L2",
                "candidate_mean_abs_delta": "DTERA candidate mean",
                "candidate_max_abs_delta": "DTERA candidate max",
                "gated_mean_abs_delta": "DTERA gated mean",
                "gated_max_abs_delta": "DTERA gated max",
                "applied_mean_abs_delta": "DTERA applied mean",
                "applied_max_abs_delta": "DTERA applied max",
                "synthetic_stand_candidate_delta": "DTERA stand candidate",
                "synthetic_stand_applied_delta": "DTERA stand applied",
            }
            dtera_lines = []
            for key, label in dtera_metric_labels.items():
                if key in anyadapter_metrics:
                    value = anyadapter_metrics[key]
                    wandb_dict[f"AnyAdapter/{key}"] = value
                    dtera_lines.append(f"{label:>{pad}} {value:.6e}\n")
            anyadapter_log_string += "".join(dtera_lines)

        wandb_dict['Adaptation/hist_latent_loss'] = locs['mean_hist_latent_loss']
        wandb_dict['Adaptation/priv_reg_loss'] = locs['mean_priv_reg_loss']
        wandb_dict['Adaptation/priv_ref_lambda'] = locs['priv_reg_coef']

        wandb_dict['Scale/regularization_scale'] = locs["regularization_scale"]
        if locs['grad_penalty_coef'] != 0:
            wandb_dict['Loss/grad_penalty_loss'] = locs['mean_grad_penalty_loss']
            wandb_dict['Scale/grad_penalty_coef'] = locs["grad_penalty_coef"]

        if locs['mean_motion_difficulty'] != 0:
            wandb_dict['Scale/motion_difficulty'] = locs["mean_motion_difficulty"]

        wandb_dict['Policy/mean_noise_std'] = mean_std.item()
        wandb_dict['Perf/total_fps'] = fps
        wandb_dict['Perf/collection time'] = locs['collection_time']
        wandb_dict['Perf/learning_time'] = locs['learn_time']
        if len(locs['rewbuffer']) > 0:
            wandb_dict['Train/mean_reward'] = statistics.mean(locs['rewbuffer'])
            wandb_dict['Train/mean_episode_length'] = statistics.mean(locs['lenbuffer'])
            # wandb_dict['Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            # wandb_dict['Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        wandb.log(wandb_dict, step=locs['it'])
        if self.log_dir is not None:
            json_record = {
                "iteration": int(locs["it"]),
                "completed_updates": int(locs["it"]) + 1,
            }
            for key, value in wandb_dict.items():
                if isinstance(value, torch.Tensor):
                    if value.numel() != 1:
                        continue
                    value = value.detach().cpu().item()
                if isinstance(value, (int, float, bool, np.number)):
                    json_record[key] = float(value)
            metrics_path = os.path.join(self.log_dir, "train_metrics.jsonl")
            with open(metrics_path, "a", encoding="utf-8") as metrics_file:
                metrics_file.write(json.dumps(json_record, sort_keys=True) + "\n")

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        scale_str = f"""{'Regularization_scale:':>{pad}} {locs['regularization_scale']:.4f}\n"""
        average_episode_length = f"""{'Average_episode_length:':>{pad}} {locs['average_episode_length']:.4f}\n"""
        gp_scale_str = f"""{'Grad_penalty_coef:':>{pad}} {locs['grad_penalty_coef']:.4f}\n"""
        motion_difficulty_str = f"""{'Mean_motion_difficulty:':>{pad}} {locs['mean_motion_difficulty']:.4f}\n"""
        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Experiment Name:':>{pad}} {os.path.basename(self.log_dir)}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{anyadapter_log_string}"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward (total):':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{anyadapter_log_string}"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")

        log_string += f"""{'-' * width}\n"""
        log_string += ep_string
        log_string += f"""{'-' * width}\n"""
        log_string += scale_str
        log_string += average_episode_length
        log_string += gp_scale_str
        log_string += motion_difficulty_str
        curr_it = locs['it'] - self.start_learning_iteration
        eta = self.tot_time / (curr_it + 1) * (locs['num_learning_iterations'] - curr_it)
        mins = eta // 60
        secs = eta % 60
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {mins:.0f} mins {secs:.1f} s\n""")
        builtins.print(log_string)

    def save(self, path, infos=None, iteration=None):
        saved_iteration = (
            self.current_learning_iteration
            if iteration is None
            else int(iteration)
        )
        if self.normalize_obs:
            state_dict = {
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': saved_iteration,
            'normalizer': self.normalizer,
            'critic_normalizer': self.critic_normalizer,
            'infos': infos,
            }
        else:
            state_dict = {
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': saved_iteration,
            'infos': infos,
            }
        if hasattr(self.alg, "ppo_optimizer"):
            state_dict['ppo_optimizer_state_dict'] = self.alg.ppo_optimizer.state_dict()
        if hasattr(self.alg, "wm_optimizer"):
            state_dict['wm_optimizer_state_dict'] = self.alg.wm_optimizer.state_dict()
        if hasattr(self.alg, "risk_optimizer"):
            state_dict['risk_optimizer_state_dict'] = self.alg.risk_optimizer.state_dict()
        torch.save(state_dict, path)

    def load(self, path, load_optimizer=True):
        print("*" * 80)
        print("Loading model from {}...".format(path))
        loaded_dict = torch.load(path, map_location=self.device)
        filename_iteration = int(
            os.path.basename(path).split("_")[1].split(".")[0]
        )
        # Older checkpoints were written while
        # current_learning_iteration stayed at zero inside the loop.  Preserve
        # their established filename-based resume behavior, while new
        # checkpoints carry the exact completed-update count in ``iter``.
        if int(loaded_dict.get("iter", 0)) == 0 and filename_iteration > 0:
            loaded_dict["iter"] = filename_iteration
        if hasattr(self.alg.actor_critic, 'base_actor'):
            current_base_state = self.alg.actor_critic.base_actor.state_dict()
            checkpoint_state = loaded_dict['model_state_dict']
            mismatched_base_keys = []
            for key, current_value in current_base_state.items():
                checkpoint_key = 'base_actor.' + key
                checkpoint_value = checkpoint_state.get(checkpoint_key)
                if checkpoint_value is None or not torch.equal(
                    current_value.detach().cpu(), checkpoint_value.detach().cpu()
                ):
                    mismatched_base_keys.append(checkpoint_key)
            if mismatched_base_keys:
                configured_path = getattr(
                    self.alg.actor_critic, 'base_actor_jit_path', '<unknown>'
                )
                raise RuntimeError(
                    "AnyAdapter checkpoint base actor does not match the configured "
                    f"base_actor_jit_path={configured_path}. Start a fresh experiment "
                    "directory instead of resuming this checkpoint. First mismatch: "
                    f"{mismatched_base_keys[0]}"
                )
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if hasattr(self.alg, "on_load_checkpoint"):
            self.alg.on_load_checkpoint(loaded_dict)
        if self.normalize_obs:
            self.normalizer = loaded_dict['normalizer']
            self.critic_normalizer = loaded_dict['critic_normalizer']
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
            if hasattr(self.alg, "ppo_optimizer") and 'ppo_optimizer_state_dict' in loaded_dict:
                self.alg.ppo_optimizer.load_state_dict(loaded_dict['ppo_optimizer_state_dict'])
            if hasattr(self.alg, "wm_optimizer") and 'wm_optimizer_state_dict' in loaded_dict:
                self.alg.wm_optimizer.load_state_dict(loaded_dict['wm_optimizer_state_dict'])
                # load_state_dict copies param_groups verbatim, including the
                # old run's weight_decay=1e-4 that collapsed the encoder/WM.
                # Re-apply the constructor's wd=0 policy after loading.
                for group in self.alg.wm_optimizer.param_groups:
                    group['weight_decay'] = 0.0
            if hasattr(self.alg, "risk_optimizer") and 'risk_optimizer_state_dict' in loaded_dict:
                self.alg.risk_optimizer.load_state_dict(
                    loaded_dict['risk_optimizer_state_dict']
                )
                for group in self.alg.risk_optimizer.param_groups:
                    group['weight_decay'] = 0.0
        self.current_learning_iteration = int(
            loaded_dict.get("iter", filename_iteration)
        )
        self.env.global_counter = self.current_learning_iteration * 24
        self.env.total_env_steps_counter = self.current_learning_iteration * 24
        print("*" * 80)
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_actor_critic(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic

    def get_normalizer(self, device=None):
        if device is not None:
            self.normalizer.to(device)
        return self.normalizer
