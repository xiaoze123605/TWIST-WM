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

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot

from .base.humanoid import Humanoid
from .base.humanoid_view_motion import HumanoidViewMotion
from .base.humanoid_mimic import HumanoidMimic


from .g1.g1_mimic_distill import G1MimicDistill, G1MimicRecorder
from .g1.g1_mimic_distill_config import G1MimicPrivCfg, G1MimicPrivCfgPPO
from .g1.g1_mimic_distill_config import G1MimicStuRLCfg, G1MimicStuRLCfgDAgger
from .g1.g1_mimic_distill_config import G1MimicStuCleanedCfg, G1MimicStuCleanedCfgDAgger
from .g1.g1_mimic_distill_anyadapter_config import G1MimicStuAnyAdapterCfg, G1MimicStuAnyAdapterCfgPPO
from .g1.g1_mimic_distill_anyadapter_config import G1MimicStuAnyAdapterV2Cfg, G1MimicStuAnyAdapterV2CfgPPO
from .g1.g1_mimic_distill_anyadapter_config import G1MimicStuAnyAdapterSafeCfg, G1MimicStuAnyAdapterSafeCfgPPO
from .g1.g1_mimic_distill_anyadapter_config import G1MimicStuAnyAdapterV3Cfg, G1MimicStuAnyAdapterV3CfgPPO
from .g1.g1_mimic_distill_anyadapter_config import G1MimicStuAnyAdapterV4Cfg, G1MimicStuAnyAdapterV4CfgPPO
from .g1.g1_mimic_distill_anyadapter_config import G1MimicStuAnyAdapterDualCfg, G1MimicStuAnyAdapterDualCfgPPO
from .g1.g1_mimic_distill_anyadapter_config import G1MimicStuAnyAdapterDTERACfg, G1MimicStuAnyAdapterDTERACfgPPO
from .g1.g1_mimic_distill_anyadapter_config import G1MimicStuAnyAdapterDTERASelectiveCfg, G1MimicStuAnyAdapterDTERASelectiveCfgPPO
from .g1.g1_mimic_distill_anyadapter_config import G1MimicStuAnyAdapterV5Cfg, G1MimicStuAnyAdapterV5CfgPPO
from .g1.g1_mimic_distill_anyadapter_config import G1MimicStuAnyAdapterV6Cfg, G1MimicStuAnyAdapterV6CfgPPO

from legged_gym.gym_utils.task_registry import task_registry

# ======================= environment registration =======================

task_registry.register("g1_priv_mimic", G1MimicDistill, G1MimicPrivCfg(), G1MimicPrivCfgPPO())

task_registry.register("g1_stu_rl", G1MimicDistill, G1MimicStuRLCfg(), G1MimicStuRLCfgDAgger())

task_registry.register("g1_cleaned_gen", G1MimicRecorder, G1MimicPrivCfg(), G1MimicPrivCfgPPO())

task_registry.register("g1_stu_rl_cleaned", G1MimicDistill, G1MimicStuCleanedCfg(), G1MimicStuCleanedCfgDAgger())

task_registry.register("g1_stu_anyadapter", G1MimicDistill, G1MimicStuAnyAdapterCfg(), G1MimicStuAnyAdapterCfgPPO())
task_registry.register("g1_stu_anyadapter_v2", G1MimicDistill, G1MimicStuAnyAdapterV2Cfg(), G1MimicStuAnyAdapterV2CfgPPO())
task_registry.register("g1_stu_anyadapter_safe", G1MimicDistill, G1MimicStuAnyAdapterSafeCfg(), G1MimicStuAnyAdapterSafeCfgPPO())
task_registry.register("g1_stu_anyadapter_v3", G1MimicDistill, G1MimicStuAnyAdapterV3Cfg(), G1MimicStuAnyAdapterV3CfgPPO())
task_registry.register("g1_stu_anyadapter_v4", G1MimicDistill, G1MimicStuAnyAdapterV4Cfg(), G1MimicStuAnyAdapterV4CfgPPO())
task_registry.register("g1_stu_anyadapter_dual", G1MimicDistill, G1MimicStuAnyAdapterDualCfg(), G1MimicStuAnyAdapterDualCfgPPO())
task_registry.register("g1_stu_anyadapter_dtera", G1MimicDistill, G1MimicStuAnyAdapterDTERACfg(), G1MimicStuAnyAdapterDTERACfgPPO())
task_registry.register("g1_stu_anyadapter_dtera_selective", G1MimicDistill, G1MimicStuAnyAdapterDTERASelectiveCfg(), G1MimicStuAnyAdapterDTERASelectiveCfgPPO())
task_registry.register("g1_stu_anyadapter_v5", G1MimicDistill, G1MimicStuAnyAdapterV5Cfg(), G1MimicStuAnyAdapterV5CfgPPO())
task_registry.register("g1_stu_anyadapter_v6", G1MimicDistill, G1MimicStuAnyAdapterV6Cfg(), G1MimicStuAnyAdapterV6CfgPPO())
