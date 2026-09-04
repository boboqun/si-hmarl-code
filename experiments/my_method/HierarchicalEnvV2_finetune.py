"""
HierarchicalEnvV2_finetune.py
==============================
微调版环境 —— 基于 HierarchicalEnvV2.py，仅修改时间惩罚系数

修改内容：
    - 时间惩罚从 -0.3/tick → -0.8/tick（第1092行等效位置）
    
目的：
    加大每步时间代价，让模型更积极追求缩短 ep_len。
    从 iter_5400 checkpoint 恢复训练。

原始文件 HierarchicalEnvV2.py 完全不变，保持成果隔离。
"""

# ── 直接导入原始模块的所有内容，仅覆盖需要修改的部分 ──
import math
import numpy as np

# 导入原始环境的所有常量和类
from HierarchicalEnvV2 import (
    DT, MAX_EPISODE_STEPS, MAP_SIZE, MAX_NODES,
    UAV_SPEED_WORK, UAV_SPEED_RECHARGE, UAV_SCAN_WIDTH,
    UAV_FULL_BATTERY, UAV_LOW_BATTERY, UAV_RECHARGE_DIST,
    WAYPOINT_ARRIVE_DIST, UGV_SPEED,
    MACRO_ROWS, MACRO_COLS, MACRO_BLOCK_SIZE,
    ACT_NOOP, ACT_BLOCK_START, ACT_BLOCK_END, ACT_RECHARGE,
    HierarchicalEnv as _OriginalEnv,
)


# ════════════════════════════════════════════════════════════════════
#  微调版环境：仅覆盖 step() 中的时间惩罚系数
# ════════════════════════════════════════════════════════════════════

# 微调参数（唯一变化点）
TIME_PENALTY_PER_TICK = 0.8   # 原始: 0.3, 微调: 0.8


class HierarchicalEnvFinetune(_OriginalEnv):
    """
    HierarchicalEnvV2 的微调子类。
    
    唯一修改：step() 中未达100%覆盖时的时间惩罚
        原始: base_shared_reward -= 0.3
        微调: base_shared_reward -= 0.8
    
    继承关系确保所有其他逻辑（观测、动作、PSR、UGV追踪等）与原版完全一致。
    """

    def step(self, actions):
        """
        重写 step 方法，调用父类 step 后修正时间惩罚差额。
        
        策略：不复制整个 step 函数（避免维护两份代码），
        而是在父类 step 返回后，对 rewards 做差额修正。
        
        差额 = -(TIME_PENALTY_PER_TICK - 0.3) = -(0.8 - 0.3) = -0.5/tick
        仅在覆盖率 < 100% 时应用。
        """
        observations, rewards, terminations, truncations, infos = super().step(actions)
        
        # 差额修正：父类已扣 -0.3，我们需要再扣 -(0.8 - 0.3) = -0.5
        cur_cov = self._compute_coverage_ratio()
        if cur_cov < 1.0:
            penalty_delta = -(TIME_PENALTY_PER_TICK - 0.3)
            for agent in rewards:
                rewards[agent] += penalty_delta
        
        return observations, rewards, terminations, truncations, infos
