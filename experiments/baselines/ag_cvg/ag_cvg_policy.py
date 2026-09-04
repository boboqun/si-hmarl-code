"""
AG-CVG Adapted Baseline (Karapetyan et al., ICRA 2024)
======================================================
"AG-CVG: Coverage Planning with a Mobile Recharging UGV and an Energy-Constrained UAV".

忠实要点（与原文 Algorithm 1 对应）：
  1. 为每个 UAV 分区生成 BCD（牛耕）覆盖路径；
  2. 按单次满电航程把路径切分为能量簇（energy clusters）；
  3. 用匈牙利算法把各能量簇的端点 **二分图匹配** 到路网候选会合节点，
     得到「每个充电周期的预定会合节点」（离线、全局一次性求解）；
  4. 执行期：UAV 沿规划路径覆盖，UGV **主动预置位** 到下一充电周期所匹配的
     会合节点等待 —— 这正是 AG-CVG 区别于简单跟随式 UGV 的核心贡献。

本实现相对此前版本的关键修正：
  - UGV 不再退化为环境原生 escort（即论文方法自带的预定位启发式），而是真正
    驱动到「离线匈牙利匹配得到的会合节点」，从而忠实再现 AG-CVG 的预置位行为；
  - 采用 **能量感知主动返航**（按到匹配会合节点的距离决定返航时机），而非固定阈值；
  - 通过实例级 monkey-patch 让 UGV 可被驱动到任意指定节点（与 HeuristicController 同构）。

接口：AGCVGController(env) + get_actions(obs) -> {"uav_0":a0,"uav_1":a1,"ugv_0":a_ugv}
依赖：scipy（匈牙利匹配）。会合采用环境默认的连续对接（与其余基线一致，保证公平）。
"""

import math
import types
import numpy as np
from scipy.optimize import linear_sum_assignment

# ── 物理常数（须与 HierarchicalEnvV2 执行环境一致）──────────────────────
UAV_SPEED_WORK   = 10.0     # 扫描速度 m/s
UAV_SPEED_RETURN = 20.0     # 返航速度 m/s（env: UAV_SPEED_RECHARGE）
UAV_SCAN_WIDTH   = 50.0     # 扫宽 m
UAV_FULL_BATTERY = 900      # 满电 ticks
UAV_LOW_BATTERY  = 200      # 安全下限 ticks
UGV_SPEED        = 15.0     # 车速 m/s
DT               = 1.0
SWAP_DURATION    = 120
RETURN_SAFETY    = 70       # 返航安全余量 ticks

# 单次满电最大扫描航程（保守预算，预留返航与对接余量）
MAX_FLIGHT_DIST  = (UAV_FULL_BATTERY - UAV_LOW_BATTERY - SWAP_DURATION) * UAV_SPEED_WORK
ENERGY_CLUSTER_FRAC = 0.70   # 每个能量簇用满电航程的 70%

ACT_NOOP     = 0
ACT_RECHARGE = 26


# ──────────────────────────────────────────────────────────────────────
#  几何 / 规划工具
# ──────────────────────────────────────────────────────────────────────
def _lawnmower_waypoints(rect_min, rect_max, sweep_width):
    """矩形区域的牛耕（boustrophedon）航点序列。"""
    x0, y0 = rect_min
    x1, y1 = rect_max
    if x1 - x0 < 1e-3 or y1 - y0 < 1e-3:
        return [(x0, y0)]
    wps, y, direction = [], y0 + sweep_width / 2.0, 1
    while y < y1:
        wps += [(x0, y), (x1, y)] if direction == 1 else [(x1, y), (x0, y)]
        y += sweep_width
        direction *= -1
    return wps


def _cluster_by_energy(waypoints, max_dist):
    """把连续航点切分为若干能量簇，使每簇路径长度 ≤ max_dist。"""
    if not waypoints:
        return []
    clusters, cur, cur_d = [], [waypoints[0]], 0.0
    for i in range(1, len(waypoints)):
        seg = math.hypot(waypoints[i][0] - waypoints[i-1][0],
                         waypoints[i][1] - waypoints[i-1][1])
        if cur_d + seg > max_dist and len(cur) > 1:
            clusters.append(cur)
            cur, cur_d = [waypoints[i-1]], 0.0
        cur.append(waypoints[i])
        cur_d += seg
    if cur:
        clusters.append(cur)
    return clusters


def _hungarian_match(cluster_endpoints, road_nodes):
    """
    匈牙利算法：把每个能量簇端点匹配到路网节点（最小化端点→节点总距离）。
    cluster_endpoints: [(x,y), ...]    road_nodes: [(nid,x,y), ...]
    返回与 cluster_endpoints 等长的 [(nid,x,y), ...]。
    """
    if not cluster_endpoints or not road_nodes:
        return []
    nc, nn = len(cluster_endpoints), len(road_nodes)
    cost = np.zeros((nc, nn))
    for i, (ex, ey) in enumerate(cluster_endpoints):
        for j, (_nid, nx, ny) in enumerate(road_nodes):
            cost[i, j] = math.hypot(ex - nx, ey - ny)
    # 簇数可多于节点数：按需横向复制列，保证可行指派
    if nc > nn:
        reps = nc // nn + 1
        cost_ext = np.tile(cost, (1, reps))
        nodes_ext = road_nodes * reps
    else:
        cost_ext, nodes_ext = cost, road_nodes
    row, col = linear_sum_assignment(cost_ext)
    matched = [None] * nc
    for r, c in zip(row, col):
        if r < nc:
            matched[r] = nodes_ext[c]
    # 兜底
    for i in range(nc):
        if matched[i] is None:
            matched[i] = road_nodes[0]
    return matched


# ──────────────────────────────────────────────────────────────────────
#  AG-CVG 控制器
# ──────────────────────────────────────────────────────────────────────
class AGCVGController:
    def __init__(self, env):
        self.env = env
        self._planned = False
        self._order = {'uav_0': [], 'uav_1': []}          # 每机的 serpentine 区块序
        self._rv_nodes = {'uav_0': [], 'uav_1': []}        # 每机各充电周期的匹配会合节点 (nid,x,y)
        self._rv_idx = {'uav_0': 0, 'uav_1': 0}            # 当前充电周期索引
        self._was_returning = {'uav_0': False, 'uav_1': False}
        self._ugv_target_xy = None                         # 控制器指定的 UGV 目标点
        self._target_uav = None                            # 当前承诺预置位的目标 UAV（滞回，防抖动）

        # 实例级 monkey-patch：让 UGV 可被驱动到任意指定坐标最近的节点（预置位）
        env_orig_step_ugv = env._step_ugv

        def patched_step_ugv(self_env, action):
            tgt = self._ugv_target_xy
            if tgt is None:
                return env_orig_step_ugv(action)           # 回退到原生 escort
            tx, ty = tgt
            target_node = min(self_env.G.nodes(), key=lambda n: math.hypot(
                float(self_env.G.nodes[n]['x']) - tx,
                float(self_env.G.nodes[n]['y']) - ty))
            self_env.car['semantic_action'] = action
            self_env.car['target_node'] = target_node
            self_env._ugv_move_toward(target_node, UGV_SPEED * DT)

        env._step_ugv = types.MethodType(patched_step_ugv, env)

    # ── 离线规划：分区 → BCD → 能量聚类 → 匈牙利匹配 ───────────────────
    def _generate_plan(self):
        env = self.env
        k = env.macro_k
        left, right = [], []
        for bid in range(1, env.n_blocks + 1):
            (right if (bid - 1) % k >= 3 else left).append(bid)
        self._order['uav_0'] = self._serpentine(left, k)
        self._order['uav_1'] = self._serpentine(right, k)

        road_nodes = [(n, float(env.G.nodes[n]['x']), float(env.G.nodes[n]['y']))
                      for n in env.G.nodes()]
        for agent in ('uav_0', 'uav_1'):
            wps = []
            for bid in self._order[agent]:
                (x0, y0), (x1, y1) = env._block_id_to_rect(bid)
                wps += _lawnmower_waypoints((x0, y0), (x1, y1), UAV_SCAN_WIDTH)
            clusters = _cluster_by_energy(wps, MAX_FLIGHT_DIST * ENERGY_CLUSTER_FRAC)
            endpoints = [c[-1] for c in clusters] if clusters else []
            self._rv_nodes[agent] = _hungarian_match(endpoints, road_nodes)
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

    def _current_rv_xy(self, agent):
        """该机当前充电周期所匹配的会合节点坐标（用于 UGV 预置位与返航时机）。"""
        nodes = self._rv_nodes[agent]
        if not nodes:
            return None
        idx = min(self._rv_idx[agent], len(nodes) - 1)
        _nid, x, y = nodes[idx]
        return (x, y)

    def get_actions(self, obs_dict):
        if not self._planned:
            self._generate_plan()
        env = self.env
        completed = env._get_completed_blocks()
        actions = {}

        for agent in ('uav_0', 'uav_1'):
            uav = env.uavs[agent]
            other = 'uav_1' if agent == 'uav_0' else 'uav_0'

            # 充电周期推进：检测「本帧刚进入返航」的上升沿
            if uav['is_returning'] and not self._was_returning[agent]:
                self._rv_idx[agent] += 1
            self._was_returning[agent] = uav['is_returning']

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

            # 3) 空闲 → 按 serpentine 计划取下一个未完成、未占用区块
            chosen = ACT_NOOP
            busy_blocks = {env.uavs[a].get('current_block', 0) for a in ('uav_0', 'uav_1')}
            for bid in self._order[agent]:
                if bid not in completed and bid not in busy_blocks:
                    chosen = bid
                    break
            actions[agent] = chosen

        # ── UGV 调度（两阶段，消除抖动）──────────────────────────────────────
        # 末段：一旦有 UAV 返航，直接逼近其实时位置（任何移动 UGV 的常规末段行为，
        #       与离线匹配的预置位是两个阶段，不属于借用本方法的预测式 escort）。
        returning = [a for a in ('uav_0', 'uav_1') if env.uavs[a]['is_returning']]
        if returning:
            tgt = min(returning, key=lambda a: env.uavs[a]['battery'])
            self._ugv_target_xy = (env.uavs[tgt]['x'], env.uavs[tgt]['y'])
            self._target_uav = None          # 末段结束后重新承诺，避免抖动
        else:
            # 预置位阶段：滞回承诺一个目标 UAV（仅当另一机明显更紧迫才切换），
            # 驶向其【离线匈牙利匹配】的会合节点——AG-CVG 的预置位特征，且不每步重算。
            primary = min((a for a in ('uav_0', 'uav_1') if not env.uavs[a]['is_swapping']),
                          key=lambda a: env.uavs[a]['battery'], default=None)
            cur = self._target_uav
            need_switch = (
                cur is None
                or env.uavs.get(cur, {}).get('is_swapping', True)
                or (primary is not None and primary != cur
                    and env.uavs[primary]['battery'] < env.uavs[cur]['battery'] - 60))
            if need_switch:
                self._target_uav = primary
            self._ugv_target_xy = (self._current_rv_xy(self._target_uav)
                                   if self._target_uav else None)
        actions['ugv_0'] = ACT_RECHARGE   # 实际由 patched _step_ugv 驱动到目标节点
        return actions
