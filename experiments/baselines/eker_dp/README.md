# Eker 最优会合基线 (Eker & Öncü, IEEE T-AES 2025)

> Eker, A. H., & Öncü, A. *Optimal Rendezvous Scheduling for Charging Coordination
> between Aerial-Ground Vehicles*, IEEE T-AES 2025.

回应审稿意见 **M6**：补充与已发表 SOTA 的正面对比。`scripts/evaluate_performance.py`
已内置 `METHODS["Eker_DP"]`，自动在 10 个种子(1001–1010)上评测并纳入对比图表。

## 算法：在线滚动时域最优会合

原文把「在何处/何时会合充电」建模为最优控制/调度问题以最小化总完成时间。
本实现以**在线滚动时域（receding-horizon）最优会合节点选择**再现其精髓：
每个决策步，对最亟需充电的 UAV，在全部路网候选节点上求解

```
n* = argmin_n  max( d_air(uav, n)/v_air ,  d_gnd(ugv, n)/v_gnd )
```

即令空地两端先后到达同一会合节点的**会合完成时间**最小的节点，据此驱动 UGV
预置位并决定 UAV 返航时机。这是单次会合的时间最优解。

## 与 AG-CVG 的本质区别（确保两个 SOTA 不重复）

| | 会合点求解 | 时机 |
|---|---|---|
| AG-CVG | 离线、全局：匈牙利匹配能量簇端点 ↔ 节点 | 一次性规划 |
| **Eker（本目录）** | **在线、逐事件：每步重算时间最优节点** | 滚动时域 |

二者覆盖路径相同（BCD 分区 + serpentine），仅**会合调度范式**不同——一个离线全局匹配、
一个在线逐事件最优，从而构成两个真正不同的对比点。

## 相对此前版本的修正（重要）

此前版本用「飞到固定充电站再飞回」的**静态后向归纳 DP**，其成本模型与本系统的
**移动 UGV** 动力学不符，且输出经松散映射 + 「每 2 块兜底」后几乎不影响实际行为，
UGV 也退化为环境原生 escort。现已改为与移动 UGV 一致、且真正驱动 UGV 与返航时机的
在线最优会合。候选节点到达时间用欧氏距离作标准松弛（UGV 实际沿路网行驶），以便在
大规模稠密路网上每步实时求解；会合沿用环境默认连续对接（与所有基线一致）。

## UGV 调度（两阶段，避免抖动）

- **预置位阶段**：滞回承诺一个目标 UAV，UGV 驶向其**在线时间最优**会合节点；该节点仅在
  切换目标或每 25 步刷新一次（跟踪目标 UAV 移动），而非每步重算——既保留在线滚动时域特征，
  又避免 UGV 抖动、空驶激增。
- **末段**：一旦该 UAV 进入返航，UGV 直接逼近其实时位置完成对接。
- 能量感知返航：按「飞抵 UGV 当前位置所需电量 + 安全余量」触发。

## 依赖与运行

- 纯规则 / 在线规划，**无需训练，无需 scipy**；依赖与主项目一致。

```bash
python experiments/scripts/evaluate_performance.py
python -c "from experiments.baselines.eker_dp.eker_dp_policy import EkerDPController; print('import OK')"
```

接口：`EkerDPController(env)` + `get_actions(obs) -> {"uav_0":a0,"uav_1":a1,"ugv_0":a_ugv}`。
