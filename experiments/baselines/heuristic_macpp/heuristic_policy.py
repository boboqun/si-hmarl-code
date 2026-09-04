"""
HeuristicController
===================
基于贪心与优先级的纯规则多智能体调度器 (Heuristic MACPP Agent).
1. UAV: 贪心最近未分配区块分配 + 阈值强制返航。
2. UGV: 多机质心跟随 (Centroid) + 电量优先级的防死锁救援 (Rescue with Priority Queue)。
"""

import math
import random
import types
import numpy as np

ACT_NOOP = 0
ACT_RECHARGE = 26
UAV_LOW_BATTERY_THRESHOLD = 0.25 # 25% 时强制返航

class HeuristicController:
    def __init__(self, env):
        self.env = env
        
        # 为了在不修改原环境代码的前提下，给该实例的 UGV action=26 注入“电量优先级救援”与“质心引力”逻辑
        # 我们进行实例级别的方法绑定 (Monkey Patching)
        self._original_step_ugv = self.env._step_ugv
        
        def upgraded_step_ugv(self_env, action: int):
            self_env.car['semantic_action'] = action
            v = self_env.car['edge_v']
            
            if action == 0:
                target_node = v
            elif 1 <= action <= 25:
                (x0, y0), (x1, y1) = self_env._block_id_to_rect(action)
                cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                target_node = min(self_env.G.nodes(), key=lambda n: math.hypot(
                    float(self_env.G.nodes[n]['x']) - cx, float(self_env.G.nodes[n]['y']) - cy))
            elif action == 26:
                # ── 突破性改进：Priority Queue based on remaining battery ──
                returning_uavs = [u for u in self_env.uavs.values() if u['is_returning']]
                if returning_uavs:
                    # 获取当前电量绝对值更低的那架 UAV
                    critical_uav = min(returning_uavs, key=lambda u: float(u['battery']))
                    target_node = min(self_env.G.nodes(), key=lambda n: math.hypot(
                        float(self_env.G.nodes[n]['x']) - critical_uav['x'],
                        float(self_env.G.nodes[n]['y']) - critical_uav['y']))
                else:
                    # 作业伴随：驶向工作机群重心 (Centroid/Weber)
                    working = [u for u in self_env.uavs.values() 
                               if u.get('is_busy') and not u['is_returning'] and not u['is_swapping']]
                    if working:
                        cx = sum(u['x'] for u in working) / len(working)
                        cy = sum(u['y'] for u in working) / len(working)
                        target_node = min(self_env.G.nodes(), key=lambda n: math.hypot(
                            float(self_env.G.nodes[n]['x']) - cx, float(self_env.G.nodes[n]['y']) - cy))
                    else:
                        target_node = v
            else:
                target_node = v
                
            self_env.car['target_node'] = target_node
            
            # 使用环境中定义的常数如果可用，不可用则硬编码对应的物理值
            try:
                from experiments.my_method.HierarchicalEnv import UGV_SPEED, DT
                budget = UGV_SPEED * DT
            except ImportError:
                budget = 15.0 * 1.0 # fallback
                
            self_env._ugv_move_toward(target_node, budget)
            
        # 绑定到当前环境实例
        self.env._step_ugv = types.MethodType(upgraded_step_ugv, self.env)

    def get_actions(self, obs_dict):
        """
        根据全局状态生成下一步动作
        """
        actions = {}
        
        full_battery = 0.0
        try:
            from experiments.my_method.HierarchicalEnv import UAV_FULL_BATTERY
            full_battery = UAV_FULL_BATTERY
        except ImportError:
            full_battery = 400.0
            
        uav_low_limit = full_battery * UAV_LOW_BATTERY_THRESHOLD
        
        grid = self.env.coverage_grid
        macro_rows = 5
        macro_cols = 5
        rows_per_block = max(1, len(grid) // macro_rows)
        cols_per_block = max(1, len(grid[0]) // macro_cols)
        
        # 统计尚未全覆盖的区块 (1-indexed)
        uncovered_blocks = []
        for block_id in range(1, 26):
            idx = block_id - 1
            row = idx // macro_cols
            col = idx % macro_cols
            r0, r1 = row * rows_per_block, (row + 1) * rows_per_block
            c0, c1 = col * cols_per_block, (col + 1) * cols_per_block
            sub = grid[r0:r1, c0:c1]
            if sub.size > 0 and not sub.all():
                uncovered_blocks.append(block_id)
                
        # 提取当前已被占用的区块
        claimed_blocks = set()
        for a in ['uav_0', 'uav_1']:
            if self.env.uavs[a]['is_busy']:
                claimed_blocks.add(self.env.uavs[a]['current_block'])
                
        # 为每架 UAV 派发任务
        for a in ['uav_0', 'uav_1']:
            u = self.env.uavs[a]
            
            # 1. 强制换电判断
            if u['battery'] < uav_low_limit and not u['is_swapping'] and not u['is_returning']:
                actions[a] = ACT_RECHARGE
                continue
                
            # 2. 正常工作判断
            if u['is_busy'] or u['is_returning'] or u['is_swapping']:
                actions[a] = ACT_NOOP # 继续干活 / 继续换电
            else:
                # 从未分配的待覆盖区块中随机选取一个 (Random Dispatch)
                candidate_blocks = [b for b in uncovered_blocks if b not in claimed_blocks]
                if candidate_blocks:
                    chosen_block = random.choice(candidate_blocks)
                    actions[a] = chosen_block
                    claimed_blocks.add(chosen_block)
                else:
                    actions[a] = ACT_NOOP

        # UGV 的行为完全交给环境强化版动作 26 (质心调度 + 优先级救援)
        actions['ugv_0'] = 26
        
        return actions
