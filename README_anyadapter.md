# TWIST + AnyAdapter 双支路 JIT 导出指南

针对双支路 AnyAdapter 策略（`use_dual_branch_adapter=True`）的 TorchScript 导出与部署说明。

策略结构：

```
a = a_base + adapter_gain * (dynamics_branch_gain * Δdyn + tracking_branch_gain * Δerr)

Δdyn = dynamics_delta_scale  * tanh(MLP([a_base, 当前状态51维] + z))   # 动力学支路
Δerr = tracking_delta_scale  * tanh(MLP([参考误差29维] + a_base))       # 跟踪支路
```

其中 `a_base` 是冻结的 TWIST 学生策略，`z` 是历史编码器隐变量。
delta_scale / branch_gain / adapter_gain 均为**运行时参数**（不在权重里），导出时可改，无需重训。

## 导出脚本

`legged_gym/scripts/export_twist_anyadapter_dual_jit.py`

特点（与旧单支路脚本 `export_twist_anyadapter_jit.py` 的区别）：

- 双支路训练配置硬编码（无需 preset）
- 严格 key 校验：缺 key / 多 key 直接报错，不会静默丢弃支路权重
- 内置 4 项验证：trace 与 eager 逐位一致、支路模式结构校验、修正量上界校验、支路存活检查
- 输入维度守卫：JIT 只接受 2635 维观测（其余维度干净报错），保证 deploy server 正确识别为 AnyAdapter 策略，不会误判成 Any2Track

## 导出命令

```bash
cd /home/hank/TWIST（anyadapter）

CKPT=legged_gym/logs/g1_twist_anyadapter_dual/dual_formal_v1/model_30000.pt

# 1) 只导 full 模式（直接给仿真/实机 server 用）
/home/hank/anaconda3/envs/twist/bin/python legged_gym/scripts/export_twist_anyadapter_dual_jit.py \
  --ckpt ${CKPT} \
  --device cpu

# 2) 三种模式都导（消融评估用：full / dyn_only / err_only）
for mode in full dyn_only err_only; do
  /home/hank/anaconda3/envs/twist/bin/python legged_gym/scripts/export_twist_anyadapter_dual_jit.py \
    --ckpt ${CKPT} \
    --device cpu \
    --branch_mode ${mode}
done
```

输出（自动命名，带迭代号，不会覆盖其他迭代的文件）：

```
legged_gym/logs/g1_twist_anyadapter_dual/dual_formal_v1/traced/
├── dual_formal_v1-30000-full-jit.pt
├── dual_formal_v1-30000-dyn_only-jit.pt
└── dual_formal_v1-30000-err_only-jit.pt
```

## 导出时调参（不需要重训）

```bash
# 例：加大跟踪支路修正量 0.05 -> 0.15
--tracking_delta_scale 0.15

# 例：整体削弱修正
--adapter_gain 0.5

# 例：只留动力学支路并放大 3 倍
--branch_mode dyn_only --dynamics_branch_gain 3.0
```

全部参数：`--branch_mode {full,dyn_only,err_only}`、`--adapter_gain`、
`--dynamics_delta_scale`、`--tracking_delta_scale`、
`--dynamics_branch_gain`、`--tracking_branch_gain`。

## 接入 deploy server

```bash
cd deploy_real

# GPU 被训练占用时加 --device cpu（MuJoCo 仿真本身在 CPU，只有策略推理走 device）
python server_low_level_g1_sim.py \
  --policy_path /home/hank/TWIST（anyadapter）/legged_gym/logs/g1_twist_anyadapter_dual/dual_formal_v1/traced/dual_formal_v1-30000-full-jit.pt \
  --device cpu
```

**验收标准**：server 启动时必须打印
`[AnyAdapter] Detected 2635-D AnyAdapter policy; enabling runtime history wrapper automatically.`
（若打印 Any2Track / heading-aware 则为误判，说明用了没有输入守卫的旧文件）。

## 导出 DTERA checkpoint（g1_stu_anyadapter_dtera）

`legged_gym/scripts/export_twist_dtera_jit.py`

针对 DTERA 架构（`TwistDTERAActorCritic`：双支路 + demand/confidence/risk 门控 +
跟踪误差历史编码器 + 误差趋势/风险预测器）的导出脚本，与双支路脚本同样的
严格 key 校验 + 4 项验证，观测维度为 **3695**（1155 base + 20×74 动力学历史 +
20×53 跟踪误差历史）：

```bash
cd /home/hank/TWIST（anyadapter）

CKPT=legged_gym/logs/g1_twist_dtera_revision4/dtera_revision4_frozen_output_bias_overnight/model_4800.pt

# 训练配置门控导出（gate_mode=demand_only）
/home/hank/anaconda3/envs/twist/bin/python legged_gym/scripts/export_twist_dtera_jit.py \
  --ckpt ${CKPT} \
  --device cpu

# 其他门控模式（与 evaluate_dual_branch.py 的消融模式一致）
# --gate_mode {off,demand_only,demand_confidence,full}
# --confidence_gate_strength 1.0（demand_confidence / full 建议）
# --independent_branch_gates --tracking_demand_mode smoothstep
# --tracking_demand_low 0.30 --tracking_demand_high 0.80（selective 配置）
```

输出：`<run_dir>/traced/<run>-<迭代>-dtera-<gate_mode>-jit.pt`

注意：DTERA 的 3695 维观测不在 deploy server 的探测列表
（1155/2635/2637/7001）里，server 会报告无法识别而不是误判；导出文件自带
维度守卫，只接受 3695 维输入，可直接在仿真/自定义加载器中使用。

## 训练与恢复（dual_formal_v2，wd 修复版）

### 启动全新训练

```bash
cd /home/hank/TWIST（anyadapter）

nohup /home/hank/anaconda3/envs/twist/bin/python legged_gym/legged_gym/scripts/train.py \
  --task g1_stu_anyadapter_dual \
  --proj_name g1_twist_anyadapter_dual \
  --exptid dual_formal_v2 \
  --num_envs 4096 \
  --max_iterations 30000 \
  --seed 42 \
  --no_wandb \
  > legged_gym/legged_gym/scripts/train_dual_formal_v2.log 2>&1 &
```

### 中断训练

```bash
pgrep -f "train.py.*dual_formal_v2"     # 找到 PID（nohup 外层 bash 和 python 本体都要杀）
kill <PID1> <PID2>
```

中断会丢失最近保存点之后的进度。保存间隔：`<2500` 轮每 500 保存、`2500~5000` 每 1000、
`>5000` 每 2500、30000 结束必保存。建议在保存点之后中断。

### 恢复训练（从 checkpoint 续训）

```bash
cd /home/hank/TWIST（anyadapter）

nohup /home/hank/anaconda3/envs/twist/bin/python legged_gym/legged_gym/scripts/train.py \
  --task g1_stu_anyadapter_dual \
  --proj_name g1_twist_anyadapter_dual \
  --exptid dual_formal_v2 \
  --resume \
  --checkpoint 4000 \
  --max_iterations 26000 \
  --seed 42 \
  --no_wandb \
  > legged_gym/legged_gym/scripts/train_dual_formal_v2_resume.log 2>&1 &
```

要点：

1. `--resume` 从 `logs/{proj_name}/{exptid}` 加载 checkpoint；`--checkpoint -1` 表示取最新
2. **`--max_iterations` 是追加轮数**（`tot_iter = 当前轮 + max_iterations`）。
   例：从 model_4000 续训到 30000 → 传 `26000`；传 `30000` 会练到 34000
3. resume 会从 checkpoint 加载 optimizer 状态，其中旧 param_groups 携带的
   weight_decay 会被 runner 里的补丁强制归零（`wm_optimizer` 恒为 wd=0），
   修复过的代码在续训时同样有效
4. 健康判断标准：`AnyAdapter wm loss` 低于 ~0.15 方差地板、
   `hist WM grad` 非零、checkpoint 中 encoder/WM 权重范数持续增长
   （v1 的 wd 崩塌症状是 wm loss 死在地板 + 权重范数归零）

## 注意事项

1. **必须使用本脚本重新导出的文件**。旧导出文件没有输入维度守卫，会被
   deploy server 误判为 Any2Track（79 帧历史包装器），动作全错。
2. `--device cpu` 导出足够快（<1 分钟），不会干扰训练 GPU。
3. **已知问题**：v1 的 checkpoint（model_25000 / model_30000 等）的 HistoryEncoder /
   World Model 权重因历史版本权重衰减配置归零（`hist WM grad = 0`，wm loss 停在方差
   地板），z ≡ 0，动力学支路退化为无历史的静态修正——**这些 checkpoint 只用于
   单/双支路结构对比，不代表完整设计**。`rsl_rl/rsl_rl/algorithms/ppo_anyadapter.py`
   已将 wm_optimizer 的 weight_decay 改为 0；dual_formal_v2（wd 修复版重训）在
   iter 3200 体检通过：encoder/WM 权重持续增长、hist WM grad 非零、
   wm loss 低于地板。后续导出优先用 v2 的 checkpoint。
