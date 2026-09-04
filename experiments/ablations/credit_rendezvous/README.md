# 隔离式消融实验：信用分配 × 会合机制 × 宏策略结构

本目录回应审稿意见 **M3**：原文与 Standard H-MARL 的对比同时改变了「奖励结构」
与「会合方式」两个变量，无法把 28.3% makespan / 78.1% deadhead 的增益干净地
归因到某一个机制；同时贡献 1（空间交叉注意力）和贡献 2 的两条信用通道也从未
被单独消融。本套实验在**单一变量**原则下重新训练并评测，把每个机制的贡献分离开。

## 实现方式

全部变体通过 `HierarchicalEnvV2` 新增的四个开关实现，**默认值精确还原原始
SA-HMARL**，因此主结果与既有 checkpoint 不受任何影响：

| 开关 | 取值 | 含义 |
|---|---|---|
| `reward_mode` | `dual_channel`(默认) / `flat_shared` | 双通道信用分配 vs 扁平稠密共享距离奖励 |
| `rendezvous_mode` | `continuous`(默认) / `node` | 连续中途会合 vs 节点受限会合 |
| `enable_proactive` | `True`(默认) / `False` | 主动错峰通道（PSR-proj 私有正奖励）开关 |
| `enable_reactive` | `True`(默认) / `False` | 被动接驳冲击惩罚通道开关 |

交叉注意力消融则通过 `mlp_commander.py`（去掉空间交叉注意力、改用等参数级 MLP，
I/O 完全一致）在训练时切换 UAV 宏策略主干实现。

## 变体矩阵（共 8 个）

**A. 奖励 × 会合 2×2 因子**（分离“信用分配”与“连续会合”的贡献）

| 变体 | reward_mode | rendezvous_mode |
|---|---|---|
| `full`（参照） | dual_channel | continuous |
| `flat_continuous` | flat_shared | continuous |
| `dual_node` | dual_channel | node |
| `flat_node` | flat_shared | node |

**B. 双通道逐通道隔离**

| 变体 | 说明 |
|---|---|
| `full`（参照） | 主动 + 被动 两通道全开 |
| `proactive_only` | 仅主动错峰通道 |
| `reactive_only` | 仅被动冲击惩罚通道 |
| `no_credit` | 两通道全关 |

**C. 宏策略结构**

| 变体 | 说明 |
|---|---|
| `full`（参照） | 空间交叉注意力 |
| `mlp_commander` | 普通 MLP（去掉空间交叉注意力） |

## 运行

```bash
cd experiments/ablations/credit_rendezvous

# 训练单个变体（默认 3000 iters，保证收敛）
python run_ablation_matrix.py --variant flat_continuous
python run_ablation_matrix.py --variant mlp_commander --iters 3000

# 顺序训练全部 8 个变体
python run_ablation_matrix.py --variant all

# 在 10 个固定种子(1001–1010)上统一评测，汇总对比表
python evaluate_ablations.py
```

- 每个变体权重写入 `experiments/results/ablations/<variant>/checkpoints/latest/`。
- 评测明细写入 `experiments/results/ablations/ablation_metrics.csv`，
  指标定义（Makespan / Deadhead / Redundant Scan）与 `scripts/evaluate_performance.py`、
  正文 Table 5.1 完全一致，额外记录 idle 悬停 tick（用于量化 node 会合的悬停浪费）。

## 预期解读

- `full` vs `flat_continuous`：在会合方式相同的前提下，单独度量**信用分配**的贡献。
- `full` vs `dual_node`：在奖励相同的前提下，单独度量**连续会合**的贡献。
- 2×2 四格联合：验证两机制是否存在交互效应，排除原文的混淆对比。
- `proactive_only` / `reactive_only` / `no_credit`：拆出两条信用通道各自的作用。
- `full` vs `mlp_commander`：单独量化**空间交叉注意力**（贡献 1）的增益。

## 依赖与说明

- 依赖与主项目一致（`ray[rllib]`、`torch`、`pettingzoo`、`gymnasium`、`networkx`、`numpy`、`pandas`）。
- 训练为多智能体 PPO，需 GPU/多核 CPU；单变体 1000 iters 的算力与主实验同量级。
- 代码已通过 `py_compile` 语法校验；具体数值需在配备上述依赖的机器上训练后由
  `evaluate_ablations.py` 产出。
