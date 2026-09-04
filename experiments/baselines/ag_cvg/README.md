# AG-CVG 基线 (Karapetyan et al., ICRA 2024)

> Karapetyan, N., Asghar, A. B., Bhaskar, A., Shi, G., Manocha, D., & Tokekar, P.
> *AG-CVG: Coverage Planning with a Mobile Recharging UGV and an Energy-Constrained UAV*, ICRA 2024.

回应审稿意见 **M6**：补充与**已发表 SOTA 会合/协作方法**的正面对比。本目录提供
AG-CVG 的忠实适配实现作为强离线规划基线（`scripts/evaluate_performance.py` 已内置
`METHODS["AG_CVG"]`，会自动在 10 个种子(1001–1010)上评测并纳入对比图表）。

## 算法（与原文 Algorithm 1 对应）

1. 每个 UAV 分区生成 BCD（牛耕）覆盖路径；
2. 按单次满电航程把路径切成若干**能量簇**；
3. 用**匈牙利算法**把各能量簇端点二分图匹配到路网候选会合节点 →
   得到「每个充电周期的预定会合节点」（离线、全局、一次性求解）；
4. 执行期：UAV 沿规划路径覆盖，UGV **主动预置位**到下一充电周期所匹配的会合节点。

AG-CVG 的核心贡献在于 UGV 依据离线匹配结果**提前**驶向预定会合节点（而非被动跟随），
本实现通过实例级 `_step_ugv` monkey-patch 真正驱动 UGV 到匹配节点来再现这一行为。

## 区别于本仓库既有 Heuristic 基线

| | UGV 策略 | 会合点 |
|---|---|---|
| Heuristic MACPP | 质心跟随 + 电量优先级救援（反应式） | 连续对接 |
| **AG-CVG（本目录）** | **离线匈牙利匹配 → 预置位到匹配节点（前瞻式）** | 连续对接 |

## 公平性与设定

- 覆盖策略 `use_resume_scan=False`（断点续扫是 SA-HMARL 专属创新），与其余规则基线一致。
- 采用**能量感知主动返航**：电量仅够飞抵匹配会合节点 + 安全余量时返航。
- 会合沿用环境默认的**连续对接**（与所有基线一致），避免给基线强加节点受限对接而失之不公；
  「节点受限 vs 连续」的对比由正文 §5.4.1 的 D1 消融单独承担。
- 纯规则、**无需训练**，可直接评测。

## UGV 调度（两阶段，避免抖动）

- **预置位阶段**：滞回承诺一个目标 UAV（仅当另一机明显更紧迫才切换），UGV 驶向其
  **离线匈牙利匹配**的会合节点——不每步重算、不在两机间抖动，这是 AG-CVG 的预置位特征。
- **末段**：一旦该 UAV 进入返航，UGV 直接逼近其实时位置完成对接（任何移动 UGV 的常规末段）。
- 能量感知返航：按「飞抵 UGV 当前位置所需电量 + 安全余量」触发。

## 依赖与运行

- **需要 `scipy`**（匈牙利匹配）：`pip install scipy`；其余依赖与主项目一致。

```bash
python experiments/scripts/evaluate_performance.py     # 随主评测脚本一起出对比图
python -c "from experiments.baselines.ag_cvg.ag_cvg_policy import AGCVGController; print('import OK')"
```

接口：`AGCVGController(env)` + `get_actions(obs) -> {"uav_0":a0,"uav_1":a1,"ugv_0":a_ugv}`。
