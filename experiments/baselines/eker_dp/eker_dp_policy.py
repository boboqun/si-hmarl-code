"""
Eker Optimal-Rendezvous Baseline (Eker & Öncü, IEEE T-AES 2025)
===============================================================
"Optimal Rendezvous Scheduling for Charging Coordination between Aerial-Ground Vehicles".

忠实要点：原文把「在何处/何时会合充电」建模为最优控制/调度问题，最小化总完成时间。
本实现以 **在线滚动时域（receding-horizon）最优会合节点选择** 再现其精髓：
每个决策步，对最亟需充电的 UAV，在全部路网候选节点上求解

        n*  =  argmin_n  max( d_air(uav, n) / v_air ,  d_gnd(ugv, n) / v_gnd )

即令「空地两端先后到达同一会合节点」的会合完成时间最小的节点，并据此驱动 UGV
预置位、决定 UAV 返航时机。这是单次会合的时间最优解（rendezvous-time optimal）。

与 AG-CVG 的本质区别（避免两个基线退化为重复）：
  - AG-CVG：**离线、全局** 用匈牙利匹配把能量簇端点一次性指派到节点；
  - Eker  ：**在线、逐事件** 重算单次会合的时间最优节点（滚动时域）。

相对此前版本的关键修正：
  - 原版用「飞到固定充电站再飞回」的静态 DP 成本模型，与本系统的「移动 UGV」
    动力学不符，且其输出几乎不影响行为（松散映射 + 每 2 块兜底）。现改为与移动
    UGV 一致的在线最优会合，且真正驱动 UGV 与返航时机。
  - 候选节点上的到达时间用欧氏距离作标准松弛（UGV 沿路网行驶，欧氏为下界近似），
    以保证每步可在大规模稠密路网上实时求解；会合采用环境默认连续对接。

接口：EkerDPController(env) + get_actions(obs) -> {"uav_0":a0,"uav_1":a1,"ugv_0":a_ugv}
"""

import math
import types

# ── 物理常数（须与 HierarchicalEnvV2 执行环境一致）──────────────────────
UAV_SPEED_WORK   = 10.0
UAV_SPEED_RETURN = 20.0
UAV_SCAN_WIDTH   = 50.0
UAV_FULL_BATTERY = 900
UAV_LOW_BATTERY  = 200
UGV_SPEED        = 15.0
DT               = 1.0
SWAP_DURATION    = 120
RETURN_SAFETY    = 70

ACT_NOOP     = 0
ACT_RECHARGE = 26


def _lawnmower_waypoints(rect_min, rect_max, sweep_width):
    x0, y0 = rect_min
    x1, y1 = rect_max
    if x1 - x0 < 1e-3 or y1 - y0 < 1e-3:
        return [(x0, y0)]
    wps, y, d = [], y0 + sweep_width / 2.0, 1
    while y < y1:
        wps += [(x0, y), (x1, y)] if d == 1 else [(x1, y), (x0, y)]
        y += sweep_width
        d *= -1
    return wps


class EkerDPController:
    def __init__(self, env):
        self.env = env
        self._planned = False
        self._order = {'uav_0': [], 'uav_1': []}
        self._ugv_target_xy = None
        # 缓存候选节点坐标，避免每步重建
        self._node_xy = None
        self._target_uav = None       # 滞回承诺的目标 UAV（防抖动）
        self._committed_xy = None     # 已承诺的会合节点（周期刷新，非每步）
        self._t = 0

        env_orig_step_ugv = env._step_ugv

        def patched_step_ugv(self_env, action):
            tgt = self._ugv_target_xy
            if tgt is None:
                return env_orig_step_ugv(action)
            tx, ty = tgt
            target_node = min(self_env.G.nodes(), key=lambda n: math.hypot(
                float(self_env.G.nodes[n]['x']) - tx,
                float(self_env.G.nodes[n]['y']) - ty))
            self_env.car['semantic_action'] = action
            self_env.car['target_node'] = target_node
            self_env._ugv_move_toward(target_node, UGV_SPEED * DT)

        env._step_ugv = types.MethodType(patched_step_ugv, env)

    # ── 覆盖路径：与 AG-CVG 相同的分区 + serpentine（仅会合调度不同）──────
    def _generate_plan(self):
        env = self.env
        k = env.macro_k
        left, right = [], []
        for bid in range(1, env.n_blocks + 1):
            (right if (bid - 1) % k >= 3 else left).append(bid)
        self._order['uav_0'] = self._serpentine(left, k)
        self._order['uav_1'] = self._serpentine(right, k)
        self._node_xy = [(float(env.G.nodes[n]['x']), float(env.G.nodes[n]['y']))
                         for n in env.G.nodes()]
        self._planned = True

    @staticmethod
    def _serpentine(block_ids, k):
        if not block_ids:
            return []
        rows = {}
        for b in block_ids:
            rows.setdefault((b - 1) // k, []).append(b)
        out = []
        for i, rk in enumerate(sorted(rows)):
            rb = sorted(rows[rk], key=lambda b: (b - 1) % k)
            if i % 2 == 1:
                rb.reverse()
            out += rb
        return out

    def _optimal_rendezvous_xy(self, uav):
        """
        在线滚动时域：返回令会合完成时间 max(air_eta, gnd_eta) 最小的候选节点坐标。
        air_eta = 距离(UAV→节点)/返航速度；gnd_eta = 距离(UGV→节点)/车速。
        """
        if not self._node_xy:
            return None
        car = self.env.car
        ux, uy = uav['x'], uav['y']
        gx, gy = car['x'], car['y']
        best, best_cost = None, float('inf')
        for (nx, ny) in self._node_xy:
            air_eta = math.hypot(ux - nx, uy - ny) / UAV_SPEED_RETURN
            gnd_eta = math.hypot(gx - nx, gy - ny) / UGV_SPEED
            cost = max(air_eta, gnd_eta)        # 会合完成时间（两端先后到达）
            if cost < best_cost:
                best_cost, best = cost, (nx, ny)
        return best

    def get_actions(self, obs_dict):
        if not self._planned:
            self._generate_plan()
        env = self.env
        completed = env._get_completed_blocks()
        actions = {}

        for agent in ('uav_0', 'uav_1'):
            uav = env.uavs[agent]

            # 1) 能量感知主动返航：电量仅够飞抵 UGV 当前(预置)位置 + 安全余量时返航
            d_ugv = math.hypot(env.car['x'] - uav['x'], env.car['y'] - uav['y'])
            need = d_ugv / UAV_SPEED_RETURN + RETURN_SAFETY
            if (uav['battery'] <= max(UAV_LOW_BATTERY, need)
                    and not uav['is_swapping'] and not uav['is_returning']):
                actions[agent] = ACT_RECHARGE
                continue

            # 2) 忙/返航/换电中 → NOOP
            if uav['is_busy'] or uav['is_returning'] or uav['is_swapping']:
                actions[agent] = ACT_NOOP
                continue

            # 3) 空闲 → 取下一个未完成、未占用区块（serpentine 序）
            chosen = ACT_NOOP
            busy_blocks = {env.uavs[a].get('current_block', 0) for a in ('uav_0', 'uav_1')}
            for bid in self._order[agent]:
                if bid not in completed and bid not in busy_blocks:
                    chosen = bid
                    break
            actions[agent] = chosen

        # ── UGV 调度（两阶段 + 滞回 + 周期刷新，消除每步重算导致的抖动）────────
        self._t += 1
        returning = [a for a in ('uav_0', 'uav_1') if env.uavs[a]['is_returning']]
        if returning:
            # 末段：逼近返航 UAV 的实时位置（移动 UGV 的常规末段）
            tgt = min(returning, key=lambda a: env.uavs[a]['battery'])
            self._ugv_target_xy = (env.uavs[tgt]['x'], env.uavs[tgt]['y'])
            self._target_uav = None
        else:
            # 预置位：滞回承诺一个目标 UAV；仅在切换或每 25 步重算其【在线时间最优】会合节点，
            # 而非每步重算——这是 Eker 区别于 AG-CVG(离线匹配) 的在线滚动时域特征，但保持稳定。
            primary = min((a for a in ('uav_0', 'uav_1') if not env.uavs[a]['is_swapping']),
                          key=lambda a: env.uavs[a]['battery'], default=None)
            cur = self._target_uav
            switch = (cur is None or env.uavs.get(cur, {}).get('is_swapping', True)
                      or (primary is not None and primary != cur
                          and env.uavs[primary]['battery'] < env.uavs[cur]['battery'] - 60))
            if switch:
                self._target_uav = primary
                self._committed_xy = (self._optimal_rendezvous_xy(env.uavs[primary])
                                      if primary else None)
            elif self._target_uav is not None and self._t % 25 == 0:
                self._committed_xy = self._optimal_rendezvous_xy(env.uavs[self._target_uav])
            self._ugv_target_xy = self._committed_xy
        actions['ugv_0'] = ACT_RECHARGE   # 实际由 patched _step_ugv 驱动到最优节点
        return actions
