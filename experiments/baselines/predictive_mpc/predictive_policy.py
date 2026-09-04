"""
PredictiveMPCController  (reviewer #4 — strong predictive-heuristic / MPC baseline)
===================================================================================
This is the STRONG predictive baseline that isolates whether SI-HMARL's gain comes
from the MARL policy or merely from the engineered predictive pre-positioning.

It is identical to HeuristicController (same greedy UAV block dispatch, same forced-
return threshold, same continuous docking via the env) EXCEPT the UGV:

  - Heuristic (reactive): UGV drives toward the *current position* of the lowest-
    battery *returning* UAV.
  - Predictive MPC (this): every tick, the UGV re-plans toward the *predicted low-
    battery landing point* p_fb (= the UAV's `fallback_return_point`, the SAME signal
    SI-HMARL's reward routing uses) of the most urgent UAV, INCLUDING UAVs still
    sweeping — i.e. it pre-positions BEFORE the return is triggered.

If SI-HMARL still beats this baseline at equal physical coverage, the advantage is the
learned policy, not the engineered pre-positioning. If this baseline closes the gap,
the gain was mostly the p_fb heuristic.
"""

import math
import random
import types

ACT_NOOP = 0
ACT_RECHARGE = 26
UAV_LOW_BATTERY_THRESHOLD = 0.25  # 25% -> forced return (same as Heuristic)


class PredictiveMPCController:
    def __init__(self, env):
        self.env = env
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
                # ── PREDICTIVE pre-positioning toward p_fb (rolling / re-planned each tick) ──
                # Consider all non-swapping UAVs (sweeping OR returning); urgency = lowest battery
                # (soonest to need recharge). Target its predicted landing point p_fb if known.
                candidates = [u for u in self_env.uavs.values() if not u['is_swapping']]
                if candidates:
                    critical_uav = min(candidates, key=lambda u: float(u['battery']))
                    fbp = critical_uav.get('fallback_return_point')
                    if fbp is not None:
                        tx, ty = float(fbp[0]), float(fbp[1])   # p_fb: predicted landing point
                    else:
                        tx, ty = float(critical_uav['x']), float(critical_uav['y'])
                    target_node = min(self_env.G.nodes(), key=lambda n: math.hypot(
                        float(self_env.G.nodes[n]['x']) - tx,
                        float(self_env.G.nodes[n]['y']) - ty))
                else:
                    target_node = v
            else:
                target_node = v

            self_env.car['target_node'] = target_node
            try:
                from experiments.my_method.HierarchicalEnv import UGV_SPEED, DT
                budget = UGV_SPEED * DT
            except ImportError:
                budget = 15.0 * 1.0
            self_env._ugv_move_toward(target_node, budget)

        self.env._step_ugv = types.MethodType(upgraded_step_ugv, self.env)

    def get_actions(self, obs_dict):
        """UAV dispatch + forced-return: IDENTICAL to HeuristicController (isolates the UGV change)."""
        actions = {}
        try:
            from experiments.my_method.HierarchicalEnv import UAV_FULL_BATTERY
            full_battery = UAV_FULL_BATTERY
        except ImportError:
            full_battery = 400.0
        uav_low_limit = full_battery * UAV_LOW_BATTERY_THRESHOLD

        grid = self.env.coverage_grid
        macro_cols = 5
        rows_per_block = max(1, len(grid) // 5)
        cols_per_block = max(1, len(grid[0]) // macro_cols)

        uncovered_blocks = []
        for block_id in range(1, 26):
            idx = block_id - 1
            row, col = idx // macro_cols, idx % macro_cols
            r0, r1 = row * rows_per_block, (row + 1) * rows_per_block
            c0, c1 = col * cols_per_block, (col + 1) * cols_per_block
            sub = grid[r0:r1, c0:c1]
            if sub.size > 0 and not sub.all():
                uncovered_blocks.append(block_id)

        claimed_blocks = set()
        for a in ['uav_0', 'uav_1']:
            if self.env.uavs[a]['is_busy']:
                claimed_blocks.add(self.env.uavs[a]['current_block'])

        for a in ['uav_0', 'uav_1']:
            u = self.env.uavs[a]
            if u['battery'] < uav_low_limit and not u['is_swapping'] and not u['is_returning']:
                actions[a] = ACT_RECHARGE
                continue
            if u['is_busy'] or u['is_returning'] or u['is_swapping']:
                actions[a] = ACT_NOOP
            else:
                candidate_blocks = [b for b in uncovered_blocks if b not in claimed_blocks]
                if candidate_blocks:
                    chosen_block = random.choice(candidate_blocks)
                    actions[a] = chosen_block
                    claimed_blocks.add(chosen_block)
                else:
                    actions[a] = ACT_NOOP

        actions['ugv_0'] = 26
        return actions
