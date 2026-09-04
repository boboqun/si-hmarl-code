"""
Porcelli-CACPP Adapted Baseline (Porcelli et al., *Optimization* 2025)
======================================================================
"Context-aware coverage path planning for a swarm of UAVs using mobile
ground stations for battery-swapping."

这是与本文问题最贴近的已发表方法（UAV 群 + 移动地面换电站 + 完整覆盖 + 在线重规划），
作为「移动换电规划类 SOTA」对照。本实现与现有 AG-CVG 基线同构（同样的 env 接口、
同样的 _step_ugv monkey-patch、同样的 get_actions 协议），仅在调度策略上忠实再现
Porcelli 的两个区别性特征：

  忠实要点（对应原文 Plan + Adjust 两阶段）：
   1. Plan：每架 UAV 分区 → BCD（牛耕）覆盖路径 → 按单次满电航程切分为能量段，
      每段末端即一个 **Agreed Swap Point (ASP)**（沿途换电点，UAV 在此换电后原地续扫）；
   2. BSS(=UGV) 路由：把所有 ASP 视为带 **时间窗** 的待访问点，UGV 按「最早到达时间」
      顺序去服务（TSP-with-Time-Windows 的贪心 EDF 解；原文对大实例亦用启发式）；
   3. Adjust：任务进行中每 ADJUST_INTERVAL ticks，依据 UAV 的**实际位置/电量**重算
      ASP 与 BSS 路由 —— 这是 Porcelli「上下文感知重规划」的核心，AG-CVG 是一次性离线。

  与本文方法的对照点（保留给 SA-HMARL 的优势）：
   - 换电只发生在**离散 ASP**（贴到最近路网节点），而非本文的**连续中途对接**；
   - 调度是**周期性重优化**，而非本文的**端到端在线学习**。
   故预期：能完成任务，但 deadhead 高于 SA-HMARL（绕到 ASP）、makespan 中等。

  环境差异处理（须在论文 5.1.2 用 "In our adaptation, ..." 说明）：
   - 原文含风/温上下文，本仿真环境无 → 去掉天气项（其余能量/位置约束保留）；
   - ASP 贴到路网节点（UGV 仅能在路网上行驶），与其余基线一致用环境默认对接判定。

接口：PorcelliCACPPController(env) + get_actions(obs) -> {"uav_0","uav_1","ugv_0"}
依赖：无（纯几何 + 贪心 TSP-TW）。如需精确 TSP-TW 可换 OR-Tools，不改变接口。
"""

import math
import types
import numpy as np

# ── 物理常数（须与 HierarchicalEnvV2 执行环境一致；与 AG-CVG 基线相同）──────
UAV_SPEED_WORK   = 10.0     # 扫描速度 m/s
UAV_SPEED_RETURN = 20.0     # 返航速度 m/s
UAV_SCAN_WIDTH   = 50.0     # 扫宽 m
UAV_FULL_BATTERY = 900      # 满电 ticks
UAV_LOW_BATTERY  = 200      # 安全下限 ticks
UGV_SPEED        = 15.0     # 车速 m/s
DT               = 1.0
SWAP_DURATION    = 120
RETURN_SAFETY    = 70       # 返航安全余量 ticks

MAX_FLIGHT_DIST     = (UAV_FULL_BATTERY - UAV_LOW_BATTERY - SWAP_DURATION) * UAV_SPEED_WORK
ENERGY_CLUSTER_FRAC = 0.70  # 每个能量段用满电航程的 70%（与 AG-CVG 一致，保证公平）
ADJUST_INTERVAL     = 300   # Porcelli 的 Adjust 周期（ticks）：每隔此步数依实况重规划

ACT_NOOP = 0


# ──────────────────────────────────────────────────────────────────────
#  几何 / 规划工具（与 AG-CVG 同源，保证两基线公平）
# ──────────────────────────────────────────────────────────────────────
def _lawnmower_waypoints(rect_min, rect_max, sweep_width):
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


def _asps_with_arrival(waypoints, max_dist, start_time):
    """
    沿航点路径切分能量段，返回每段末端的 ASP 及其**预计到达时间**（构成时间窗下界）。
    返回 [(x, y, arrival_tick), ...]。
    """
    if not waypoints:
        return []
    asps, cur_d, t = [], 0.0, float(start_time)
    for i in range(1, len(waypoints)):
        seg = math.hypot(waypoints[i][0] - waypoints[i-1][0],
                         waypoints[i][1] - waypoints[i-1][1])
        cur_d += seg
        t += seg / UAV_SPEED_WORK
        if cur_d >= max_dist:
            asps.append((waypoints[i][0], waypoints[i][1], t))
            cur_d = 0.0
            t += SWAP_DURATION          # 换电耗时计入下一段到达时间
    return asps


# ──────────────────────────────────────────────────────────────────────
#  Porcelli-CACPP 控制器
# ──────────────────────────────────────────────────────────────────────
class PorcelliCACPPController:
    def __init__(self, env):
        self.env = env
        self.act_recharge = env.n_blocks + 1        # K-鲁棒（K=5 → 26）
        self._order = {'uav_0': [], 'uav_1': []}
        self._rv_idx = {'uav_0': 0, 'uav_1': 0}     # 当前充电周期（ASP）索引
        self._asps = {'uav_0': [], 'uav_1': []}     # 每机 ASP 列表 (x,y,due_tick)
        self._was_returning = {'uav_0': False, 'uav_1': False}
        self._ugv_target_xy = None
        self._last_plan_t = -10**9

        # 实例级 monkey-patch：UGV 可被驱动到任意指定坐标最近的路网节点（与 AG-CVG 同构）
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

    # ── Plan / Adjust：分区 → BCD → 能量段 → ASP(带到达时间) ─────────────
    def _plan(self, t_now):
        env = self.env
        k = env.macro_k
        # 静态分区（一次定，左右两片；与 AG-CVG 同一分区，保证可比）
        if not self._order['uav_0']:
            left, right = [], []
            for bid in range(1, env.n_blocks + 1):
                (right if (bid - 1) % k >= (k + 1) // 2 else left).append(bid)
            self._order['uav_0'] = self._serpentine(left, k)
            self._order['uav_1'] = self._serpentine(right, k)

        completed = set(env._get_completed_blocks())
        for agent in ('uav_0', 'uav_1'):
            # 仅对**尚未完成**的区块重排路径（Adjust 用实况）
            wps, start = [], (env.uavs[agent]['x'], env.uavs[agent]['y'])
            wps.append(start)
            for bid in self._order[agent]:
                if bid in completed:
                    continue
                (x0, y0), (x1, y1) = env._block_id_to_rect(bid)
                wps += _lawnmower_waypoints((x0, y0), (x1, y1), UAV_SCAN_WIDTH)
            self._asps[agent] = _asps_with_arrival(
                wps, MAX_FLIGHT_DIST * ENERGY_CLUSTER_FRAC, t_now)
            self._rv_idx[agent] = 0
        self._last_plan_t = t_now

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

    def _next_asp_xy(self, agent):
        asps = self._asps[agent]
        if not asps:
            return None
        idx = min(self._rv_idx[agent], len(asps) - 1)
        x, y, _due = asps[idx]
        return (x, y)

    def _next_asp_due(self, agent):
        asps = self._asps[agent]
        if not asps:
            return float('inf')
        idx = min(self._rv_idx[agent], len(asps) - 1)
        return asps[idx][2]

    def get_actions(self, obs_dict):
        env = self.env
        t_now = getattr(env, 't', 0)

        # Adjust：周期性 / 首帧重规划（依实际位置与电量）
        if t_now - self._last_plan_t >= ADJUST_INTERVAL:
            self._plan(t_now)

        completed = env._get_completed_blocks()
        actions = {}

        for agent in ('uav_0', 'uav_1'):
            uav = env.uavs[agent]

            # 充电周期推进：检测「本帧刚进入返航」的上升沿 → 该机服务完一个 ASP
            if uav['is_returning'] and not self._was_returning[agent]:
                self._rv_idx[agent] += 1
            self._was_returning[agent] = uav['is_returning']

            # 1) 能量感知主动返航（电量仅够飞抵 UGV 当前位置 + 安全余量）
            d_ugv = math.hypot(env.car['x'] - uav['x'], env.car['y'] - uav['y'])
            need = d_ugv / UAV_SPEED_RETURN + RETURN_SAFETY
            if (uav['battery'] <= max(UAV_LOW_BATTERY, need)
                    and not uav['is_swapping'] and not uav['is_returning']):
                actions[agent] = self.act_recharge
                continue

            # 2) 忙/返航/换电 → NOOP
            if uav['is_busy'] or uav['is_returning'] or uav['is_swapping']:
                actions[agent] = ACT_NOOP
                continue

            # 3) 空闲 → 按 serpentine 取下一个未完成、未占用区块
            chosen = ACT_NOOP
            busy_blocks = {env.uavs[a].get('current_block', 0) for a in ('uav_0', 'uav_1')}
            for bid in self._order[agent]:
                if bid not in completed and bid not in busy_blocks:
                    chosen = bid
                    break
            actions[agent] = chosen

        # ── UGV：TSP-TW 贪心(EDF) 预置位 + 末段实时跟踪 ─────────────────────
        returning = [a for a in ('uav_0', 'uav_1') if env.uavs[a]['is_returning']]
        if returning:
            # 末段：逼近最紧迫返航 UAV 的实时位置（常规末段行为，所有移动 UGV 共有）
            tgt = min(returning, key=lambda a: env.uavs[a]['battery'])
            self._ugv_target_xy = (env.uavs[tgt]['x'], env.uavs[tgt]['y'])
        else:
            # 预置位：服务「最早到达时间(time-window)」的那个 ASP —— TSP-TW 的贪心 EDF
            cand = [a for a in ('uav_0', 'uav_1') if not env.uavs[a]['is_swapping']]
            if cand:
                edf = min(cand, key=self._next_asp_due)
                self._ugv_target_xy = self._next_asp_xy(edf)
            else:
                self._ugv_target_xy = None
        actions['ugv_0'] = self.act_recharge   # 实际由 patched _step_ugv 驱动到目标节点
        return actions
