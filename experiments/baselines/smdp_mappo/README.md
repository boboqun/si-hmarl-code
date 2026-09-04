# SMDP-MAPPO: Semi-Markov Decision Process Aware Multi-Agent PPO

## 概述

本目录实现了 **SMDP-MAPPO**——一种针对宏动作层次化多智能体系统的 MAPPO 内核改进。

标准 MAPPO 的三个核心数学操作在宏动作 HMARL 系统中产生了结构性失配：

| MAPPO 核心操作 | 失配问题 | 我们的修正 |
|---|---|---|
| GAE 递归 `A_t = δ_t + γλ A_{t+1}` | 宏动作跨越上千步，γ^1200 → 0，信度归零 | **Macro-GAE**：γ^τ · λ^τ 折叠递归 |
| Actor Loss 均值池 `L = mean(L_i)` | 93% 样本是 NOOP 硬锁，零梯度稀释有效信号 | **Decisional Masking**：`L = mean(L_i · mask_i)` |

## 文件说明

| 文件 | 职责 |
|---|---|
| `smdp_postprocess.py` | 核心①：Macro-GAE（承诺修正 GAE） |
| `smdp_policy.py` | 核心②：Decisional Actor Masking |
| `smdp_train.py` | 训练入口 |
| `enjoy_smdp.py` | 推理可视化 |

## 与现有代码的关系

**`my_method/` 下的所有文件零修改。**

- 环境 `HierarchicalEnvV2.py` → 通过 import 复用
- 模型 `hierarchical_models.py` → 通过 import 复用
- 航点规划 `boustrophedon_planner.py` → 通过 import 复用

## 运行

```bash
# 配置 D（完整 SMDP-MAPPO）
python smdp_train.py

# 配置 A（退化为标准 MAPPO 基线）
USE_MACRO_GAE=False USE_DECISION_MASKING=False python smdp_train.py

# 配置 B（仅 Macro-GAE）
USE_DECISION_MASKING=False python smdp_train.py

# 配置 C（仅 Masking）
USE_MACRO_GAE=False python smdp_train.py
```

### 全自动消融实验

```bash
for macro in True False; do
  for mask in True False; do
    USE_MACRO_GAE=$macro USE_DECISION_MASKING=$mask \
      nohup python smdp_train.py &> log_${macro}_${mask}.txt &
  done
done
```

## 推理

```bash
python enjoy_smdp.py
python enjoy_smdp.py --checkpoint /path/to/checkpoint
```
