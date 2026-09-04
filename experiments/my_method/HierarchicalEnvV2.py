"""
HierarchicalEnvV2.py
====================
分层强化学习环境 V2 -- 错峰换电 + UGV主动预定位奖励版

在 HierarchicalEnv (V1) 基础上新增三类奖励信号，诱导更高阶协作行为：

  [新1] UAV 主动提早返航奖励 (Proactive Early-Return Bonus)
        当 UAV 主动发出 action=26 且电量在 30%~60% 之间（远高于安全线）
        时，给予正向奖励。鼓励模型学会主动让路，为后续错峰换电埋伏。

  [新2] UGV 主动预定位奖励 (Proactive Pre-positioning Bonus)
        当有 UAV 处于返航状态时，若 UGV 正在移动则持续正奖励 +0.5；
        若 UGV 此时静止则持续负惩罚 -1.0。
        打破 UGV 只在接到换电指令后才行动的被动模式。

  [新3] 快速汇合奖励 (Fast Rendezvous Bonus)
        UAV 返航期间，每步奖励与 UAV-UGV 距离成反比（越近奖越多）。
        引导 UGV 在最优截击点主动伏击等待，而非等 UAV 飞过来找它。

  [保留] 双机并联换电惩罚 + 动态安全返航线 (来自 V1).
"""

import math
import random
import functools
import numpy as np
import networkx as nx
from typing import Dict, Any, Optional, Tuple, List

from pettingzoo import ParallelEnv
from gymnasium import spaces

from boustrophedon_planner import SmartBoustrophedonPlanner

# ──────────────────────────────────────────────
# 全局物理常数
# ──────────────────────────────────────────────

DT               = 1.0       # 时间步长（秒）
MAX_EPISODE_STEPS = 30000    # 超时安全截断（约 8h 模拟时间，5+ 换电周期）
                             # 主要结束条件：100% 覆盖(termination) 或 UAV 断电(termination)
MAP_SIZE         = 2000.0   # 地图边长 (m) —— 2km 真实农田场景
MAX_NODES        = 200      # 路网节点索引上限（Padding 容量）

# UAV
UAV_SPEED_WORK   = 10.0     # 工作态（沿航点飞行）速度 m/s
UAV_SPEED_RECHARGE = 20.0   # 返回充电时速度 m/s
UAV_SCAN_WIDTH   = 50.0     # 扫描带宽 m
UAV_FULL_BATTERY = 900      # 满电续航 ticks (30分钟)
UAV_LOW_BATTERY  = 200      # 低电量阈值 ticks (~22%)
UAV_RECHARGE_DIST = 2.0     # 触发充电的接触距离 m
WAYPOINT_ARRIVE_DIST = 2.0  # 视为到达航点的距离 m

# UGV
UGV_SPEED        = 15.0     # 机耕道车速 m/s（54 km/h）

# 宏区块布局：5×5 = 25 块，每块 400×400m（2km 场景）
MACRO_ROWS       = 5
MACRO_COLS       = 5
MACRO_BLOCK_SIZE = MAP_SIZE / MACRO_COLS   # 400.0m (2000/5)；原注释 40.0m 为笔误

# 动作编号
ACT_NOOP         = 0         # UAV: 保持当前状态
ACT_BLOCK_START  = 1         # UAV: 区块指令起始编号
ACT_BLOCK_END    = 25        # UAV: 区块指令结束编号
ACT_RECHARGE     = 26        # UAV: 返回充电


# ================================================================
#  HierarchicalEnv 主类 (PettingZoo ParallelEnv)
# ================================================================

class HierarchicalEnv(ParallelEnv):
    """
    分层控制版覆盖率环境。

    观测结构（每个 agent 返回 Dict）：
        "observation" : {
            "coverage_grid" : uint8 (20×20)
            "self_state"    : float32 (UGV:5维, UAV:6维)
            "allies_state"  : float32
        }
        "action_mask"  : int8  (UGV: MAX_NODES维, UAV: 27维)
    """

    metadata = {
        'render_modes': ['human'],
        'name': 'hierarchical_coverage_v0',
    }

    def __init__(self, map_env, use_resume_scan: bool = True, macro_k: int = 5,
                 reward_mode: str = 'dual_channel', rendezvous_mode: str = 'continuous',
                 enable_proactive: bool = True, enable_reactive: bool = True,
                 reward_weights: dict = None, return_power_ratio: float = 1.0):
        """
        Args:
            map_env:          地图环境（RandomMapEnv 实例）
            use_resume_scan:  是否启用 SA-HMARL 独有的动态断点续扫（默认 True）。
                              基线算法（Heuristic 等）应传 False，令其对整块重新扫描，
                              确保对照实验的公平性。
            macro_k:          宏区块每边划分数量，默认 5（5×5=25 区块）。
                              K 消融实验可传入 4~8 重现对应尺度配置。
            reward_mode:      'dual_channel'（默认，完整 SA-HMARL 双通道信用分配）或
                              'flat_shared'（扁平稠密共享距离奖励，模拟 Standard H-MARL）。
                              用于「奖励 × 会合」2×2 因子消融。
            rendezvous_mode:  'continuous'（默认，连续中途动态会合）或
                              'node'（节点受限会合，UAV 须飞至固定路网节点等待 UGV）。
            enable_proactive: 是否启用主动错峰通道（PSR-proj 私有正奖励）。默认 True。
            enable_reactive:  是否启用被动接驳冲击惩罚通道。默认 True。
            说明：以上四个开关的默认值精确还原原始 SA-HMARL 行为，主结果不受影响。
        """
        super().__init__()
        self.map_env         = map_env
        self.grid_rows       = map_env.rows
        self.grid_cols       = map_env.cols
        self.planner         = SmartBoustrophedonPlanner()
        self.use_resume_scan = use_resume_scan
        # ── [ablation] 消融实验开关（默认值保持原始 SA-HMARL 行为，不影响主结果）──
        self.reward_mode      = reward_mode
        self.rendezvous_mode  = rendezvous_mode
        # [energy-sensitivity] 返航态相对扫描态的每步功耗比；默认 1.0 = 原恒定能耗模型，不影响主结果
        self.return_power_ratio = return_power_ratio
        self.enable_proactive = enable_proactive
        self.enable_reactive  = enable_reactive
        # ── [M9] 可配置奖励权重（默认值精确还原主结果；仅供奖励敏感性扫描覆盖）──
        #   coverage:       覆盖率增益系数 α（论文 R1）
        #   time_penalty:   每 tick 时间惩罚 η（论文 R2）
        #   psrproj:        主动错峰私有奖励系数（论文 R4 / PSR-proj）
        #   prepos:         UGV 预定位奖励系数 ρ（论文 R5 / Φ_pre，单位 /tick）
        #   shock:          接驳冲击惩罚总权重 β（dist/UGV_SPEED 的乘子，论文 P2）
        #   shock_ugv_frac: 被动返航冲击惩罚由 UGV 承担的比例（默认 0.5 → 50/50 拆分）
        #   crash:          UAV 断电坠机惩罚
        _DEFAULT_W = dict(coverage=1000.0, time_penalty=0.3, psrproj=1000.0,
                          prepos=3.0, shock=1.0, shock_ugv_frac=0.5, crash=500.0)
        self.W = dict(_DEFAULT_W, **(reward_weights or {}))
        # ── K-参数化宏区块配置（K 消融实验核心接口）──────────────────────
        self.macro_k          = macro_k
        self.n_blocks         = macro_k * macro_k      # 总区块数（e.g. K=5 → 25）
        self.macro_block_size = MAP_SIZE / macro_k     # 单区块边长（e.g. K=5 → 400m）
        self.act_block_end    = self.n_blocks           # UAV 区块指令最大值
        self.act_recharge     = self.n_blocks + 1       # UAV 返回充电动作编号
        self.n_actions        = self.n_blocks + 2       # 语义动作空间维度

        # 稠密化路网
        self.G            = self._densify_graph(map_env.G_proj, max_len=20.0)
        self.nodes_list   = list(self.G.nodes())
        self.node_to_idx  = {n: i for i, n in enumerate(self.nodes_list)}
        self.idx_to_node  = {i: n for i, n in enumerate(self.nodes_list)}
        self.num_nodes    = len(self.nodes_list)   # 实际节点数（≤ MAX_NODES）

        # 智能体
        self.possible_agents = ['ugv_0', 'uav_0', 'uav_1']
        self.agents          = self.possible_agents[:]

        # 内部状态
        self.t              = 0
        self.coverage_grid  = np.zeros((self.grid_rows, self.grid_cols), dtype=np.uint8)
        self.car            = {}
        self.uavs           = {}

    # ─────────────────────────────────────────────────────────────
    #  图论工具（继承自 SimulationEnv）
    # ─────────────────────────────────────────────────────────────

    def _densify_graph(self, G_orig: nx.Graph, max_len: float) -> nx.Graph:
        G = nx.Graph()
        for n, data in G_orig.nodes(data=True):
            G.add_node(n, **data)
        virtual_id_counter = 0
        for u, v, data in G_orig.edges(data=True):
            ux, uy = float(G_orig.nodes[u]['x']), float(G_orig.nodes[u]['y'])
            vx, vy = float(G_orig.nodes[v]['x']), float(G_orig.nodes[v]['y'])
            length = math.hypot(vx - ux, vy - uy)
            if length <= max_len:
                extra = {k: val for k, val in data.items() if k != 'length'}
                G.add_edge(u, v, length=length, **extra)
            else:
                num_segments = math.ceil(length / max_len)
                seg_len      = length / num_segments
                prev_node    = u
                for i in range(1, num_segments):
                    vnode = f"vnode_{u}_{v}_{virtual_id_counter}"
                    virtual_id_counter += 1
                    ratio = i / num_segments
                    nx_ = ux + ratio * (vx - ux)
                    ny_ = uy + ratio * (vy - uy)
                    G.add_node(vnode, x=nx_, y=ny_)
                    G.add_edge(prev_node, vnode, length=seg_len)
                    prev_node = vnode
                G.add_edge(prev_node, v, length=seg_len)
        return G

    # ─────────────────────────────────────────────────────────────
    #  空间定义
    # ─────────────────────────────────────────────────────────────

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Dict:
        coverage_sp = spaces.Box(
            low=0, high=1, shape=(self.grid_rows, self.grid_cols), dtype=np.uint8)

        if agent.startswith('ugv'):
            # self_state: [x_norm, y_norm, speed_norm, target_node_norm, is_moving]
            self_sp    = spaces.Box(0.0, 1.0, shape=(5,), dtype=np.float32)
            # allies 16维: 2 × UAV(8) = [x, y, battery, is_returning, is_swapping,
            #                            task_block, fb_x, fb_y]
            # fb_x/fb_y: 扫图中=兜底点, 返航中=UAV实时位置, 空闲=(0,0)
            allies_sp  = spaces.Box(0.0, 1.0, shape=(16,), dtype=np.float32)
            obs_sp     = spaces.Dict({
                'coverage_grid': coverage_sp,
                'self_state':    self_sp,
                'allies_state':  allies_sp,
            })
            mask_sp    = spaces.Box(0, 1, shape=(self.n_actions,), dtype=np.int8)
        else:
            # UAV self_state 10维: [x, y, battery, is_busy, task_block,
            #                      is_returning, is_swapping, fb_x, fb_y, phase_diff]
            # phase_diff = (my_bat - other_bat)/FULL ∈ [-1,1]，便于UAV直接感知当前错峰程度
            self_sp    = spaces.Box(-1.0, 1.0, shape=(11,), dtype=np.float32)
            # allies 9维: UGV(3) + other_UAV(6)
            # [ugv_x, ugv_y, ugv_speed, other_x, other_y, other_bat,
            #  other_is_ret, other_is_swap, other_block]
            allies_sp  = spaces.Box(0.0, 1.0, shape=(9,), dtype=np.float32)
            obs_sp     = spaces.Dict({
                'coverage_grid': coverage_sp,
                'self_state':    self_sp,
                'allies_state':  allies_sp,
            })
            mask_sp    = spaces.Box(0, 1, shape=(self.n_actions,), dtype=np.int8)

        return spaces.Dict({'observation': obs_sp, 'action_mask': mask_sp})

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Discrete:
        # 动作空间维度随 macro_k 动态变化：0=noop, 1~n_blocks=区块, n_blocks+1=返航
        return spaces.Discrete(self.n_actions)

    # ─────────────────────────────────────────────────────────────
    #  动作掩码
    # ─────────────────────────────────────────────────────────────

    def _get_ugv_action_mask(self) -> np.ndarray:
        """
        UGV 动作掩码 —— n_actions 维语义动作版（随 macro_k 动态适配）。
        0: 停靠待命， 1~n_blocks: 驶向宏区块中心待命， n_blocks+1: 动态跟车/接驳
        """
        mask = np.zeros(self.n_actions, dtype=np.int8)

        # FSM-2: 换电中 → 强制停靠 (动作 0)
        swapping = any(self.uavs[a]['is_swapping'] for a in ['uav_0', 'uav_1'])
        if swapping:
            mask[0] = 1
            return mask

        # FSM-1: 返航中 → 强制动态接驳
        returning = any(self.uavs[a]['is_returning'] for a in ['uav_0', 'uav_1'])
        if returning:
            mask[self.act_recharge] = 1
            return mask

        # 动作承诺锁：正在驶向区块中心时，不允许中途换边
        semantic_act = self.car.get('semantic_action')
        if semantic_act is not None and 1 <= semantic_act <= self.act_block_end:
            target = self.car.get('target_node')
            if target is not None:
                u, v = self.car['edge_u'], self.car['edge_v']
                arrived = (v == target) and (u == v or self.car['progress'] == 0.0)
                if not arrived:
                    try:
                        if nx.has_path(self.G, v, target):
                            mask[semantic_act] = 1
                            return mask
                    except nx.NodeNotFound:
                        pass

        # 常规：开放所有语义动作
        mask[:] = 1
        return mask

    def _get_uav_action_mask(self, agent: str) -> np.ndarray:
        """
        UAV 掩码 (n_actions 位)：0=NOOP, 1~n_blocks=区块, n_blocks+1=RECHARGE

        状态硬锁（优先级由高到低）：
          is_returning / is_swapping → 绝对只允许 NOOP，防止动作刷屏重置 FSM
          is_busy (扫图中)           → 允许 NOOP 或主动中止去充电 (RECHARGE)
          空闲                       → 全开，但屏蔽队友正在执行的区块（协作破缺）
        """
        mask = np.zeros(self.n_actions, dtype=np.int8)
        uav  = self.uavs[agent]

        # 1. 强制硬锁：返航或换电中，绝对只允许 NOOP，防止 countdown 被重置
        if uav['is_returning'] or uav['is_swapping']:
            mask[ACT_NOOP] = 1
            return mask

        # 2. 扫图中：允许继续（NOOP）或提早中止去充电（RECHARGE）
        #    双重限制：伙伴必须空闲 + 自身已消耗≥400电量（bat≤0 = 必须工作到8小时才可提早返航）
        if uav['is_busy']:
            mask[ACT_NOOP] = 1
            other = 'uav_1' if agent == 'uav_0' else 'uav_0'
            other_free = (not self.uavs[other]['is_returning']
                          and not self.uavs[other]['is_swapping'])
            work_done   = UAV_FULL_BATTERY - uav['battery']   # 自上次换电以来的消耗
            
            # [V2核心修改: 破局锁]
            # 错峰换电的意义在于物理空间中的 UGV 接驳距离最优（天然节省时间）。
            # 仅当双机处于“同步/未错峰”状态（电量差 < 25%）时，才允许用手动 RECHARGE 破局。
            # 一旦电量差 > 250，说明错峰已成，关闭手动换电，交由 E1 物理安全线自然返航。
            phase_diff = abs(uav['battery'] - self.uavs[other]['battery'])
            if other_free and work_done >= 400 and phase_diff <= 250:
                mask[self.act_recharge] = 1
            return mask

        # 3. 完全空闲：全区块 + 返航，交给 AI 自主决策
        mask[1:self.n_actions] = 1

        # [Bug2修复] IDLE 状态完全禁止 RECHARGE
        # 旧: work_done>=400 gate 可被绕过（NOOP×400tick被动耗电=零覆盖也满足条件）
        # 新: IDLE状态下 RECHARGE 永远封锁。
        #   - UAV 必须选择区块开始扫图 (is_busy=True)，才能在扫图中途调用 RECHARGE
        #   - 低电量时 E1 强制返航，无需在 IDLE 主动 RECHARGE
        mask[self.act_recharge] = 0  # 完全禁止从 IDLE 状态主动换电

        # 4. 协作破缺：屏蔽队友正在执行的区块，鼓励分工覆盖
        other = 'uav_1' if agent == 'uav_0' else 'uav_0'
        tm_block = self.uavs[other].get('current_block', 0)
        if 1 <= tm_block <= self.act_block_end:
            mask[tm_block] = 0

        # 5. 已完成区块黑名单：屏蔽 100% 覆盖的区块，防止重复劳动
        for b in self._get_completed_blocks():
            if 1 <= b <= self.act_block_end:
                mask[b] = 0

        # [Fix] 防止 mask 全零导致 KL=NaN 和权重崩溃
        if mask.sum() == 0:
            mask[ACT_NOOP] = 1

        return mask


    def _get_completed_blocks(self) -> set:
        """
        计算覆盖率已达 100% 的宏区块编号集合（已完成黑名单）。

        宏区块与覆盖网格的映射计算：
          - 地图分为 MACRO_ROWS × MACRO_COLS 个宏区块
          - coverage_grid 大小为 GRID_ROWS × GRID_COLS
          - 每个宏区块对应 (GRID_ROWS/MACRO_ROWS) × (GRID_COLS/MACRO_COLS) 个格子
        """
        completed = set()
        grid = self.coverage_grid
        rows_per_block = max(1, len(grid)     // self.macro_k)
        cols_per_block = max(1, len(grid[0])  // self.macro_k)

        for block_id in range(1, self.n_blocks + 1):
            idx = block_id - 1
            row = idx // self.macro_k
            col = idx  % self.macro_k
            r0, r1 = row * rows_per_block, (row + 1) * rows_per_block
            c0, c1 = col * cols_per_block, (col + 1) * cols_per_block
            sub = grid[r0:r1, c0:c1]
            if sub.size > 0 and sub.all():   # 所有元素均 == 1
                completed.add(block_id)
        return completed

    # ─────────────────────────────────────────────────────────────
    #  宏区块工具
    # ─────────────────────────────────────────────────────────────

    def _block_id_to_rect(self, block_id: int) -> Tuple[Tuple, Tuple]:
        """
        将区块编号 1~n_blocks (=macro_k^2) 转换为 (rect_min, rect_max)。
        区块从左下角开始，按行优先（row-major）编号。
        block_id=1 → 第0行第0列，block_id=n_blocks → 第(K-1)行第(K-1)列。
        """
        idx  = block_id - 1                       # 0-indexed
        row  = idx // self.macro_k
        col  = idx %  self.macro_k
        x0   = self.map_env.min_x + col * self.macro_block_size
        y0   = self.map_env.min_y + row * self.macro_block_size
        x1   = x0 + self.macro_block_size
        y1   = y0 + self.macro_block_size
        # 边界裁剪，防止浮点越界
        x1   = min(x1, self.map_env.max_x)
        y1   = min(y1, self.map_env.max_y)
        return (x0, y0), (x1, y1)

    # ─────────────────────────────────────────────────────────────
    #  观测生成
    # ─────────────────────────────────────────────────────────────

    def _get_obs(self, agent: str) -> Dict[str, Any]:
        mx = self.map_env.max_x
        my = self.map_env.max_y

        cov = self.coverage_grid.copy()

        if agent == 'ugv_0':
            car = self.car
            is_moving = float(car['target_node'] is not None and
                              car['target_node'] != car['edge_u'])
            self_state = np.array([
                car['x'] / mx,
                car['y'] / my,
                UGV_SPEED / 30.0,
                (self.node_to_idx.get(car['target_node'], 0)
                 / max(1, self.num_nodes - 1)) if car['target_node'] else 0.0,
                is_moving,
            ], dtype=np.float32)
            self_state = np.nan_to_num(self_state, nan=0.0, posinf=1.0, neginf=-1.0)
            self_state = np.clip(self_state, 0.0, 1.0)

            # allies_state 16维: 2 × UAV(8)
            # 每个 UAV token: [x, y, battery, is_returning, is_swapping, task_block, fb_x, fb_y]
            # fb_x/fb_y: 扫图中=兜底点(固定), 返航中=UAV实时位置, 其余=(0,0)
            # UGV 已直接可见两机电量，无需冗余的 signed_phase 字段
            allies = []
            for a in ['uav_0', 'uav_1']:
                u   = self.uavs[a]
                bat = max(0.0, float(u['battery'])) / UAV_FULL_BATTERY
                fbp = u.get('fallback_return_point')
                if fbp is not None:
                    fb_x_n, fb_y_n = fbp[0] / mx, fbp[1] / my
                elif u['is_returning']:
                    fb_x_n, fb_y_n = u['x'] / mx, u['y'] / my
                else:
                    fb_x_n, fb_y_n = 0.0, 0.0
                allies.extend([
                    u['x'] / mx,
                    u['y'] / my,
                    bat,
                    float(u['is_returning']),
                    float(u['is_swapping']),
                    float(u.get('current_block', 0)) / self.n_blocks,
                    fb_x_n,
                    fb_y_n,
                ])
            allies_state = np.array(allies, dtype=np.float32)
            allies_state = np.nan_to_num(allies_state, nan=0.0, posinf=1.0, neginf=-1.0)
            allies_state = np.clip(allies_state, 0.0, 1.0)  # clip防止浮点精度误差越界(如-3.47e-21)

            obs = {
                'coverage_grid': cov,
                'self_state':    self_state,
                'allies_state':  allies_state,
            }
            mask = self._get_ugv_action_mask()

        else:
            u = self.uavs[agent]
            task_block_norm = float(u.get('current_block', 0)) / self.n_blocks

            # self_state 11维: [x, y, battery, is_busy, task_block, is_returning, is_swapping, fb_x, fb_y, phase_diff, agent_id]
            # agent_id: 0.0=uav_0, 1.0=uav_1 —— 打破对称性，使共享策略能学出「分角色」行为
            # （无 agent_id 时两机观测相同 → 纳什均衡锁死 → 无法学习主动提早返航）
            fbp_self = u.get('fallback_return_point')
            if fbp_self is not None:
                self_fb_x, self_fb_y = fbp_self[0] / mx, fbp_self[1] / my
            elif u['is_returning']:
                self_fb_x, self_fb_y = u['x'] / mx, u['y'] / my
            else:
                self_fb_x, self_fb_y = 0.0, 0.0

            other_uav = 'uav_1' if agent == 'uav_0' else 'uav_0'
            ou = self.uavs[other_uav]

            self_state = np.array([
                u['x'] / mx,
                u['y'] / my,
                max(0.0, float(u['battery'])) / UAV_FULL_BATTERY,
                float(u['is_busy']),
                task_block_norm,
                float(u['is_returning']),
                float(u['is_swapping']),
                self_fb_x,
                self_fb_y,
                float(np.clip(
                    (max(0.0, float(u['battery'])) - max(0.0, float(ou['battery']))) / UAV_FULL_BATTERY,
                    -1.0, 1.0)),    # [V3] signed_phase_diff = (my_bat - other_bat) / FULL
                float(0 if agent == 'uav_0' else 1),  # agent_id: 对称性破缺关键特征
            ], dtype=np.float32)
            self_state = np.nan_to_num(self_state, nan=0.0, posinf=1.0, neginf=-1.0)
            self_state = np.clip(self_state, -1.0, 1.0)  # clip防float误差越界(space=Box(-1,1,11))

            car = self.car
            # allies_state 9维: UGV(3) + other_UAV(6)
            # [ugv_x, ugv_y, ugv_speed, other_x, other_y, other_bat, other_is_ret, other_is_swap, other_block]
            # 注：UAV 的相位感知已由 self_state 中的 phase_diff 字段承担，此处不重复
            allies_state = np.array([
                car['x'] / mx, car['y'] / my, UGV_SPEED / 30.0,
                ou['x'] / mx,  ou['y'] / my,
                max(0.0, float(ou['battery'])) / UAV_FULL_BATTERY,
                float(ou['is_returning']),
                float(ou['is_swapping']),
                float(ou.get('current_block', 0)) / self.n_blocks,
            ], dtype=np.float32)
            allies_state = np.nan_to_num(allies_state, nan=0.0, posinf=1.0, neginf=-1.0)
            allies_state = np.clip(allies_state, 0.0, 1.0)  # clip防float误差越界(space=Box(0,1,9))

            obs = {
                'coverage_grid': cov,
                'self_state':    self_state,
                'allies_state':  allies_state,
            }
            mask = self._get_uav_action_mask(agent)

        return {'observation': obs, 'action_mask': mask}

    # ─────────────────────────────────────────────────────────────
    #  Reset
    # ─────────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None,
              options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        if hasattr(self.map_env, "regenerate"):
            self.map_env.regenerate(seed=seed)
            self.grid_rows  = self.map_env.rows
            self.grid_cols  = self.map_env.cols
            self.G            = self._densify_graph(self.map_env.G_proj, max_len=20.0)
            self.nodes_list   = list(self.G.nodes())
            self.node_to_idx  = {n: i for i, n in enumerate(self.nodes_list)}
            self.idx_to_node  = {i: n for i, n in enumerate(self.nodes_list)}
            self.num_nodes    = len(self.nodes_list)

        self.agents        = self.possible_agents[:]
        self.t             = 0
        self.coverage_grid = np.zeros_like(self.map_env.coverage_grid)
        # ── 实测计时器（精确，不估算）─────────────────────────────────────────
        # 每架 UAV 每 tick 恰好处于四种 FSM 状态之一，累计统计
        self.scan_ticks     = 0   # UAV 正在沿航点扫图（is_busy & not returning/swapping）
        self.deadhead_ticks = 0   # UAV 正在飞向 UGV 换电（is_returning）
        self.charge_ticks   = 0   # UAV 正在机械换电（is_swapping）
        self.idle_ticks     = 0   # UAV 悬停待命（not busy, not returning, not swapping）

        # UGV 初始化
        start_node = random.choice(self.nodes_list)
        self.car = {
            'edge_u':      start_node,
            'edge_v':      start_node,
            'progress':    0.0,
            'x':           float(self.G.nodes[start_node]['x']),
            'y':           float(self.G.nodes[start_node]['y']),
            'target_node': None,
            'semantic_action': None,
        }

        # UAV 初始化
        cx = (self.map_env.min_x + self.map_env.max_x) / 2
        cy = (self.map_env.min_y + self.map_env.max_y) / 2
        spread = min(
            self.map_env.max_x - self.map_env.min_x,
            self.map_env.max_y - self.map_env.min_y,
        ) * 0.1

        self.uavs = {}
        for a in ['uav_0', 'uav_1']:
            ux = cx + random.uniform(-spread, spread)
            uy = cy + random.uniform(-spread, spread)
            self.uavs[a] = {
                'x':                     ux,
                'y':                     uy,
                'prev_x':                ux,
                'prev_y':                uy,
                'battery':               UAV_FULL_BATTERY,
                'is_busy':               False,
                'is_returning':          False,
                'is_swapping':           False,
                'swap_countdown':        0,
                'waypoint_queue':        [],
                'current_block':         0,
                'fallback_return_point': None,   # [V2] 兜底返航点，派遣时计算，返航/换电完成时清空
                'rendezvous_node':       None,   # [ablation] 节点受限会合模式下的固定会合节点
            }

        # PSR-delta 所需历史状态
        self._prev_phase_diff    = 0.0
        self._prev_both_swapping = False
        self._psr_proj_count = {'uav_0': 0, 'uav_1': 0}  # 每 episode PSR-proj 触发次数上限

        observations = {a: self._get_obs(a) for a in self.agents}
        infos        = {a: {} for a in self.agents}
        return observations, infos

    # ─────────────────────────────────────────────────────────────
    #  UGV 移动（继承 SimulationEnv 的路网推演逻辑）
    # ─────────────────────────────────────────────────────────────

    def _step_ugv(self, action: int):
        """n_actions (=K^2+2) 维语义动作 → 底层路网目标节点，驱动物理引擎。"""
        self.car['semantic_action'] = action
        v = self.car['edge_v']

        if action == 0:
            target_node = v  # 停靠

        elif 1 <= action <= self.act_block_end:
            # 预判行为：驶向目标宏区块实物中心最近路网节点
            (x0, y0), (x1, y1) = self._block_id_to_rect(action)
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            target_node = min(self.G.nodes(), key=lambda n: math.hypot(
                float(self.G.nodes[n]['x']) - cx, float(self.G.nodes[n]['y']) - cy))

        elif action == self.act_recharge:
            # 动态伴随/接驳模式 (Auto-Escort)
            # 全面对齐 R5 最佳追随预定位奖励逻辑
            best_target_pt = None
            best_urgency   = -1.0
            
            for a in ['uav_0', 'uav_1']:
                uav = self.uavs[a]
                if uav['is_swapping']:
                    continue
                urgency = 1.0 - uav['battery'] / UAV_FULL_BATTERY
                if urgency > best_urgency:
                    if uav['is_returning']:
                        best_target_pt = (uav['x'], uav['y'])
                        best_urgency = urgency
                    elif uav['is_busy'] and uav.get('fallback_return_point') is not None:
                        best_target_pt = uav['fallback_return_point']
                        best_urgency = urgency
                        
            if best_target_pt is not None:
                fx, fy = best_target_pt
                target_node = min(self.G.nodes(), key=lambda n: math.hypot(
                    float(self.G.nodes[n]['x']) - fx, float(self.G.nodes[n]['y']) - fy))
            else:
                target_node = v
        else:
            target_node = v

        self.car['target_node'] = target_node
        budget = UGV_SPEED * DT
        self._ugv_move_toward(target_node, budget)

    def _ugv_move_toward(self, target_node, budget: float):
        """沿路网最短路径向 target_node 移动，消耗 budget 米的移动预算。"""
        car = self.car
        u   = car['edge_u']
        v   = car['edge_v']
        prog = car['progress']

        # 已在目标节点
        if u == target_node and (u == v or prog == 0.0):
            return

        # 如果正在某条边上，先走完剩余距离到 v
        if u != v and prog > 0.0:
            edge_len  = self.G[u][v].get('length', 1.0)
            dist_rem  = edge_len - prog
            if budget >= dist_rem:
                budget -= dist_rem
                car['edge_u'] = v
                car['edge_v'] = v
                car['progress'] = 0.0
                car['x'] = float(self.G.nodes[v]['x'])
                car['y'] = float(self.G.nodes[v]['y'])
                u = v
            else:
                car['progress'] += budget
                ratio = car['progress'] / edge_len
                car['x'] = (self.G.nodes[u]['x'] +
                             ratio * (self.G.nodes[v]['x'] - self.G.nodes[u]['x']))
                car['y'] = (self.G.nodes[u]['y'] +
                             ratio * (self.G.nodes[v]['y'] - self.G.nodes[u]['y']))
                return

        # 现在 car 在节点 u，按最短路找下一跳
        if u == target_node or budget <= 0:
            return

        try:
            path = nx.shortest_path(self.G, u, target_node, weight='length')
        except nx.NetworkXNoPath:
            return   # 路网不连通，原地等待

        if len(path) < 2:
            return

        next_node = path[1]
        edge_len  = self.G[u][next_node].get('length', 1.0)

        if budget >= edge_len:
            car['edge_u']   = next_node
            car['edge_v']   = next_node
            car['progress'] = 0.0
            car['x']        = float(self.G.nodes[next_node]['x'])
            car['y']        = float(self.G.nodes[next_node]['y'])
        else:
            car['edge_v']   = next_node
            car['progress'] = budget
            ratio = budget / edge_len
            car['x'] = (self.G.nodes[u]['x'] +
                        ratio * (self.G.nodes[next_node]['x'] - self.G.nodes[u]['x']))
            car['y'] = (self.G.nodes[u]['y'] +
                        ratio * (self.G.nodes[next_node]['y'] - self.G.nodes[u]['y']))

    def _compute_fallback_point(
        self,
        start: Tuple[float, float],
        waypoints: List[Tuple[float, float]],
        budget_m: float,
    ) -> Tuple[float, float]:
        """
        [V2] 计算兜底返航点：沿牛耕式航点序列走 budget_m 米后所处的位置。

        数学保证：该点在 UAV 扫图全程中保持不变（见设计文档推导）。

        Args:
            start:      UAV 派遣时的起始坐标 (x, y)
            waypoints:  完整航点队列（派遣时已确定，不会变化）
            budget_m:   可用扫图飞行距离 = (battery - UAV_LOW_BATTERY) × UAV_SPEED_WORK

        Returns:
            (x, y): 兜底返航点坐标
        """
        if budget_m <= 0 or not waypoints:
            return start

        cx, cy = start
        remaining = budget_m

        for wx, wy in waypoints:
            seg = math.hypot(wx - cx, wy - cy)
            if seg <= 0:
                cx, cy = wx, wy
                continue
            if remaining >= seg:
                remaining -= seg
                cx, cy = wx, wy
            else:
                # 在本段中插值：走剩余预算后停止
                ratio = remaining / seg
                cx = cx + ratio * (wx - cx)
                cy = cy + ratio * (wy - cy)
                break

        return (cx, cy)

    # ─────────────────────────────────────────────────────────────
    #  UAV 宏指令处理 & 底层航点执行
    # ─────────────────────────────────────────────────────────────

    def _dispatch_uav_command(self, agent: str, action: int):
        """
        处理上层宏指令，填充航点队列。
        仅在 is_busy=False 时响应任务指令（NOOP 和 RECHARGE 任何时候均有效）。
        """
        uav = self.uavs[agent]

        if action == ACT_NOOP:
            # 保持当前状态，底层继续执行
            pass

        elif action == self.act_recharge:
            # 主动返航：清空航点、设置 FSM、清除兜底返航点缓存
            if uav['is_busy']:
                uav['waypoint_queue'] = []
                uav['current_block']  = 0

            uav['is_busy']                = True
            uav['is_returning']           = True
            uav['is_swapping']            = False
            uav['swap_countdown']         = 0
            uav['fallback_return_point']  = None   # [V2] 主动返航，兜底点失效

            # 接驳冲击惩罚：按呼叫瞬间 UAV 与 UGV 的物理距离计算（不连坐队友）
            # [ablation] reactive 通道开关：仅 dual_channel 且 enable_reactive 时计入
            if self.reward_mode == 'dual_channel' and self.enable_reactive:
                true_dist = math.hypot(self.car['x'] - uav['x'], self.car['y'] - uav['y'])
                self._ugv_shock_penalty += (true_dist / UGV_SPEED) * self.W['shock']

            # ── PSR-proj: 预期错峰质量奖励（Projected Stagger Reward）─────────
            # 在决策同帧解析计算「此次换电完成后的错峰质量」，建立
            # 「提早返航动作 → 改善错峰效果」的零时延直接因果链。
            # 关键：不用等待 120 tick，用物理定律当场计算未来局面。
            other     = 'uav_1' if agent == 'uav_0' else 'uav_0'
            self_bat  = uav['battery']
            other_bat = self.uavs[other]['battery']
            # 只当对方仍在正常工作时才能辽辑预测（对方返航/换电中则其电量冻结，预测无效）
            cond_other_working = (not self.uavs[other]['is_returning']
                                  and not self.uavs[other]['is_swapping'])
            # [ablation] proactive 通道开关：flat_shared 或关闭 proactive 时禁用 PSR-proj
            if self.reward_mode != 'dual_channel' or not self.enable_proactive:
                cond_other_working = False
            if cond_other_working:
                # ── 防滥用三重门槛 ────────────────────────────────────────────────
                # 1) work_done 门：至少消耗 400 tick 电量（bat≤500）才算真正「提早」返航
                #    与动作掩码保持一致：mask 同样要求 work_done≥400 才允许 RECHARGE
                work_done = UAV_FULL_BATTERY - self_bat
                if work_done < 400:
                    cond_other_working = False   # 消耗不足，抑制 PSR-proj

                # 2) 次数门：每 episode 每架 UAV 最多获得 1 次 PSR-proj 奖励
                #    1次足以建立一次错峰，不允许反复提早换电
                _MAX_PSR_PER_EPISODE = 1
                if self._psr_proj_count.get(agent, 0) >= _MAX_PSR_PER_EPISODE:
                    cond_other_working = False   # 已达上限，抑制 PSR-proj

                # 3) 覆盖率门：仅在任务初期（覆盖率 < 50%）才触发
                #    末期（覆盖率 > 50%）提早返航已无实际意义，避免末期才触发的伪信号
                cov_now = self._compute_coverage_ratio()
                if cov_now >= 0.50:
                    cond_other_working = False   # 任务已过半，抑制 PSR-proj

            if cond_other_working:
                # 对方在换电期间以 1/tick 继续消耗，120 tick 后电量如下：
                _SWAP_TICKS     = 120
                # 目标错峰度 0.40（360 tick 偏移）：
                #   物理下限 ≈ 0.30（swap+transit），用户直觉目标 ≈ 0.50，取折中 0.40
                #   确保 UGV 有足够时间跨区移动（0.40 × 900 = 360 tick = 5400m @ 15m/s）
                _TARGET_STAGGER = 0.40
                projected_other = max(0, other_bat - _SWAP_TICKS)
                projected_diff  = abs(UAV_FULL_BATTERY - projected_other) / UAV_FULL_BATTERY
                current_diff    = abs(self_bat - other_bat) / UAV_FULL_BATTERY
                # improvement 设置目标上限：只计算「向目标(0.40)的提升」，超过目标的部分不计入
                # → current_diff>=0.40 时 improvement<=0，奖励为零，不激励过度错峰
                improvement = min(projected_diff, _TARGET_STAGGER) - current_diff
                # advance_ratio: 越提早（电量越高于 E1 阈值）奖励越大，接近 E1 时归零
                advance_ratio = max(0.0, min(1.0, (self_bat - UAV_LOW_BATTERY)
                                             / (UAV_FULL_BATTERY - UAV_LOW_BATTERY)))
                if improvement > 0.05 and advance_ratio > 0.05:
                    bonus = improvement * advance_ratio * self.W['psrproj']  # 大奖励系数：一次探索即可形成清晰学习信号（count=1保护）
                    # ★ 私有奖励：只给调用 act_recharge 的 UAV（不共享）
                    # 修复：共享奖励导致 uav_1 也学到「让 uav_0 返航 = 我受益」，
                    # 共享策略叠加后变成「谁都想返航 → P1同步换电惩罚」的纳什陷阱。
                    # 私有奖励确保：uav_0(agent_id=0) 学「返航→大奖」，
                    #               uav_1(agent_id=1) 不受 PSR-proj 信号干扰。
                    self._private_psrproj[agent] = self._private_psrproj.get(agent, 0.0) + bonus
                    self._psr_proj_count[agent] = self._psr_proj_count.get(agent, 0) + 1


        elif 1 <= action <= self.act_block_end:
            if not uav['is_busy']:
                # 1. 获取原始区块的物理边界
                (x0, y0), (x1, y1) = self._block_id_to_rect(action)

                if self.use_resume_scan:
                    # ── SA-HMARL 专属：动态断点续扫 ──────────────────────────────
                    # 提取该区块内未覆盖区域的紧凑包围盒，仅对剩余部分生成航点，
                    # 避免重扫已覆盖区域，是 SA-HMARL 相对于 Heuristic 的核心优势。
                    res = self.map_env.grid_resolution
                    r0, c0 = self.map_env.xy_to_grid(x0, y0)
                    r1, c1 = self.map_env.xy_to_grid(x1 - 1e-3, y1 - 1e-3)

                    sub_grid  = self.coverage_grid[r0:r1+1, c0:c1+1]
                    uncovered = np.where(sub_grid == 0)

                    if len(uncovered[0]) > 0:
                        min_r, max_r = int(np.min(uncovered[0])), int(np.max(uncovered[0]))
                        min_c, max_c = int(np.min(uncovered[1])), int(np.max(uncovered[1]))
                        rect_min = (self.map_env.min_x + (c0 + min_c) * res,
                                    self.map_env.min_y + (r0 + min_r) * res)
                        rect_max = (self.map_env.min_x + (c0 + max_c + 1) * res,
                                    self.map_env.min_y + (r0 + max_r + 1) * res)
                    else:
                        return  # 该区块已全覆盖，取消任务

                    # print(f"\n[Debug 航点生成] 分配区块 {action}")
                    # print(f"  -> 原始物理边界: X({x0:.1f}~{x1:.1f}), Y({y0:.1f}~{y1:.1f})")
                    # print(f"  -> 续扫包围盒:   X({rect_min[0]:.1f}~{rect_max[0]:.1f}), Y({rect_min[1]:.1f}~{rect_max[1]:.1f})")
                else:
                    # ── 基线模式（Heuristic 等）：整块全量扫描 ───────────────────
                    # 不使用断点续扫，对整个区块从头规划牛耕式航点。
                    # 确保对照实验中 Heuristic 无法借用 SA-HMARL 的续扫优势。
                    rect_min = (x0, y0)
                    rect_max = (x1, y1)

                start_ref = (uav['x'], uav['y'])
                end_ref   = (self.car['x'], self.car['y'])

                # 2. 将区块边界传给规划器，生成牛耕式航点
                wps = self.planner.generate_optimal_waypoints(
                    rect_min=rect_min,
                    rect_max=rect_max,
                    sweep_width=UAV_SCAN_WIDTH,
                    start_reference=start_ref,
                    end_reference=end_ref,
                )
                if wps:
                    uav['waypoint_queue'] = list(wps)
                    uav['is_busy']        = True
                    uav['is_returning']   = False
                    uav['is_swapping']    = False
                    uav['current_block']  = action
                    # [V2] 派遣时一次性计算兜底返航点并缓存
                    _budget_m = max(0.0,
                        (uav['battery'] - UAV_LOW_BATTERY) * UAV_SPEED_WORK)
                    uav['fallback_return_point'] = self._compute_fallback_point(
                        start=(uav['x'], uav['y']),
                        waypoints=uav['waypoint_queue'],
                        budget_m=_budget_m,
                    )

    def _step_uav(self, agent: str) -> float:
        """
        UAV 底层单步物理执行 (FSM)，返回本步奖励修正量。

        FSM 状态（互斥）：
          is_returning=True → 直飞 UGV，正常耗电，产生 deadhead_penalty
          is_swapping=True  → 绑定在 UGV 上换电，不耗电，倒计时归零后 +50
          else              → 沿航点扫图或悬停待命
        """
        uav  = self.uavs[agent]
        car  = self.car
        bonus = 0.0

        uav['prev_x'] = uav['x']
        uav['prev_y'] = uav['y']

        # ── 实测计时：在 FSM 执行前记录本 tick 的真实状态 ─────────────────────
        if uav['is_returning']:
            self.deadhead_ticks += 1
        elif uav['is_swapping']:
            self.charge_ticks += 1
        elif uav['is_busy'] and uav.get('waypoint_queue'):
            self.scan_ticks += 1
        else:
            self.idle_ticks += 1

        if uav['is_returning']:
            # ── FSM-1: 返航追踪 ──
            if self.rendezvous_mode == 'node':
                # [ablation] 节点受限会合：UAV 只能飞至固定路网节点并在该节点等待 UGV 到达
                rn = uav.get('rendezvous_node')
                if rn is None or rn not in self.G:
                    rn = min(self.G.nodes(), key=lambda n: math.hypot(
                        float(self.G.nodes[n]['x']) - uav['x'],
                        float(self.G.nodes[n]['y']) - uav['y']))
                    uav['rendezvous_node'] = rn
                rx, ry = float(self.G.nodes[rn]['x']), float(self.G.nodes[rn]['y'])
            else:
                # 连续会合（默认）：实时将 UGV 的当前连续物理坐标作为追踪目标（支持路段任意位置接驳）
                rx, ry = car['x'], car['y']

            dx, dy = rx - uav['x'], ry - uav['y']
            dist   = math.hypot(dx, dy)
            budget = UAV_SPEED_RECHARGE * DT

            if dist > 1e-6:
                moved = min(dist, budget)
                uav['x'] += (dx / dist) * moved
                uav['y'] += (dy / dist) * moved
            else:
                uav['x'], uav['y'] = rx, ry

            actual_moved = math.hypot(uav['x'] - uav['prev_x'], uav['y'] - uav['prev_y'])
            bonus -= 0.05 * actual_moved   # 飞行能耗惩罚

            if self.rendezvous_mode == 'node':
                # 节点受限：UAV 到达会合节点且 UGV 也到达该节点附近时才换电；否则原地悬停（产生 idle 浪费）
                uav_at_node = math.hypot(uav['x'] - rx, uav['y'] - ry) <= UAV_RECHARGE_DIST
                ugv_at_node = math.hypot(car['x'] - rx, car['y'] - ry) <= UAV_RECHARGE_DIST
                if uav_at_node and ugv_at_node:
                    uav['is_returning']    = False
                    uav['is_swapping']     = True
                    uav['swap_countdown']  = 120
                    uav['rendezvous_node'] = None
            else:
                # 连续会合：接驳不依赖 Node，物理距离足够近即可在路段任意位置换电
                new_dist = math.hypot(uav['x'] - rx, uav['y'] - ry)
                if new_dist <= UAV_RECHARGE_DIST:
                    uav['is_returning']   = False
                    uav['is_swapping']    = True
                    uav['swap_countdown'] = 120

        elif uav['is_swapping']:
            # ── FSM-2: 机械换电（坐标绑定 UGV，step() 不扣电）──
            uav['x'] = car['x']   # 绑定 UGV 坐标（随车移动）
            uav['y'] = car['y']
            uav['swap_countdown'] -= 1

            if uav['swap_countdown'] <= 0:
                # 换电完成：重置 FSM，清除兜底返航点缓存
                uav['battery']               = UAV_FULL_BATTERY
                uav['is_swapping']           = False
                uav['is_busy']               = False
                uav['current_block']         = 0
                uav['fallback_return_point'] = None   # [V2] 换电完成，等待下次派遣重新计算
                bonus += 0.0

        elif uav['is_busy'] and uav['waypoint_queue']:
            # ── FSM-3: 扫图模式：沿航点飞行，速度 10m/s ──
            wx, wy = uav['waypoint_queue'][0]
            dx, dy = wx - uav['x'], wy - uav['y']
            dist   = math.hypot(dx, dy)
            budget = UAV_SPEED_WORK * DT   # 10m

            if dist <= WAYPOINT_ARRIVE_DIST:
                uav['waypoint_queue'].pop(0)
                if not uav['waypoint_queue']:
                    uav['is_busy']       = False
                    uav['current_block'] = 0
            else:
                uav['x'] += (dx / dist) * min(budget, dist)
                uav['y'] += (dy / dist) * min(budget, dist)

        else:
            # ── FSM-4: 悬停待命 ──
            if uav['is_busy'] and not uav['waypoint_queue']:
                uav['is_busy']       = False
                uav['current_block'] = 0

        return bonus

    # ─────────────────────────────────────────────────────────────
    #  覆盖率几何（继承自 SimulationEnv）
    # ─────────────────────────────────────────────────────────────

    def _update_coverage(self):
        """仅 UAV 在扫图航点模式下产生覆盖（UGV 不扫图）。"""
        for a in ['uav_0', 'uav_1']:
            uav = self.uavs[a]
            # 仅当 is_busy 且非返航/换电时才扫图
            if uav['is_busy'] and not uav['is_returning'] and not uav['is_swapping']:
                move_dist = math.hypot(
                    uav['x'] - uav['prev_x'],
                    uav['y'] - uav['prev_y'],
                )
                if move_dist > 1e-3:
                    self._fill_uav_swath(
                        uav['prev_x'], uav['prev_y'],
                        uav['x'],      uav['y'],
                        UAV_SCAN_WIDTH / 2.0,
                    )
                else:
                    self._fill_circle(uav['x'], uav['y'], UAV_SCAN_WIDTH / 2.0)

    def _fill_circle(self, cx: float, cy: float, radius: float):
        res      = self.map_env.grid_resolution
        r_cells  = int(math.ceil(radius / res))
        cr, cc   = self.map_env.xy_to_grid(cx, cy)

        for r in range(max(0, cr - r_cells),
                       min(self.map_env.rows - 1, cr + r_cells) + 1):
            for c in range(max(0, cc - r_cells),
                       min(self.map_env.cols - 1, cc + r_cells) + 1):
                px = self.map_env.min_x + (c + 0.5) * res
                py = self.map_env.min_y + (r + 0.5) * res
                if math.hypot(px - cx, py - cy) <= radius:
                    self.coverage_grid[r, c] = 1

    def _fill_uav_swath(self, x0, y0, x1, y1, half_width):
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return
        nx_ =  dy / length
        ny_ = -dx / length
        verts = [
            (x0 + nx_ * half_width, y0 + ny_ * half_width),
            (x1 + nx_ * half_width, y1 + ny_ * half_width),
            (x1 - nx_ * half_width, y1 - ny_ * half_width),
            (x0 - nx_ * half_width, y0 - ny_ * half_width),
        ]
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        res     = self.map_env.grid_resolution
        col_min = max(0, int((min(xs) - self.map_env.min_x) // res))
        col_max = min(self.map_env.cols - 1,
                      int((max(xs) - self.map_env.min_x) // res) + 1)
        row_min = max(0, int((min(ys) - self.map_env.min_y) // res))
        row_max = min(self.map_env.rows - 1,
                      int((max(ys) - self.map_env.min_y) // res) + 1)
        for r in range(row_min, row_max + 1):
            for c in range(col_min, col_max + 1):
                px = self.map_env.min_x + (c + 0.5) * res
                py = self.map_env.min_y + (r + 0.5) * res
                if self._point_in_convex_polygon(px, py, verts):
                    self.coverage_grid[r, c] = 1

    @staticmethod
    def _point_in_convex_polygon(px, py, verts) -> bool:
        n    = len(verts)
        sign = None
        for i in range(n):
            ax, ay = verts[i]
            bx, by = verts[(i + 1) % n]
            cross  = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
            if abs(cross) < 1e-9:
                continue
            s = 1 if cross > 0 else -1
            if sign is None:
                sign = s
            elif sign != s:
                return False
        return True

    def _compute_coverage_ratio(self) -> float:
        total = self.coverage_grid.size
        if total == 0:
            return 0.0
        return float(np.sum(self.coverage_grid)) / total

    # ─────────────────────────────────────────────────────────────
    #  Step
    # ─────────────────────────────────────────────────────────────

    def step(self, actions: Dict[str, Any]):
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        self.t      += 1
        prev_cov     = self._compute_coverage_ratio()

        # ── 1. 处理宏指令（含当帧互斥锁 + 已完成区块拦截）────────────────────
        self._dispatch_bonus = 0.0
        self._ugv_shock_penalty = 0.0   # UGV 专属接驳惩罚累加器
        self._private_psrproj  = {'uav_0': 0.0, 'uav_1': 0.0}   # PSR-proj 私有奖励：只发给调用 act_recharge 的那架 UAV
        recharge_this_tick = False   # 当帧互斥锁：防双机同 tick 同时自愿返航 → P1
        claimed_this_tick = set()   # 当帧已被抒定的区块
        for a in ['uav_0', 'uav_1']:
            if a in actions:
                act = int(actions[a])

                # ★ 防双机同帧同时自愿返航（P1延迟代价-960抹去PSR-proj+262）
                # 每帧只允许第一个主动调 act_recharge 的 UAV 执行，第二个降级为 NOOP。
                # E1 强制返航不受限（在 step 末段单独处理）。
                if act == self.act_recharge and recharge_this_tick:
                    act = ACT_NOOP   # 降级：本帧已有 UAV 在返航，等下一帧

                # 区块接令：拦截冲突与重复劳动
                if 1 <= act <= self.act_block_end:
                    other = 'uav_1' if a == 'uav_0' else 'uav_0'
                    tm_block  = self.uavs[other].get('current_block', 0)
                    completed = self._get_completed_blocks()
                    # 队友正在执行 或 当帧已被抒定 或 已 100% 覆盖 → 强制 NOOP
                    if act == tm_block or act in claimed_this_tick or act in completed:
                        act = ACT_NOOP
                    else:
                        claimed_this_tick.add(act)

                self._dispatch_uav_command(a, act)
                if act == self.act_recharge:
                    recharge_this_tick = True   # 锁定：本帧已有自愿返航

        if 'ugv_0' in actions:
            self._step_ugv(int(actions['ugv_0']))

        # ── 2. UAV 物理推演 ────────────────────────────────────────
        uav_reward_bonus = self._dispatch_bonus  # 含 dispatch 塑形惩罚
        for a in ['uav_0', 'uav_1']:
            uav_reward_bonus += self._step_uav(a)

        # ── 3. 电量扣减（is_swapping 换电中不耗电；is_returning 飞行中正常耗电）──
        dead_uav    = False
        fail_reason = ''
        for a in ['uav_0', 'uav_1']:
            if a in self.agents and not self.uavs[a]['is_swapping']:
                # [energy-sensitivity] 返航态按 return_power_ratio 扣电；ratio=1.0 时退化为原恒定模型
                self.uavs[a]['battery'] -= (self.return_power_ratio
                                            if self.uavs[a]['is_returning'] else 1.0)
                if self.uavs[a]['battery'] < 0:
                    dead_uav    = True
                    fail_reason = a

        # ── 4. 覆盖率更新 ──────────────────────────────────────────
        self._update_coverage()
        cur_cov = self._compute_coverage_ratio()

        # ── 5. 奖励计算（PSR 版）──────────────────────────────────────────

        # R1+R2+R3: 基础共享奖励（覆盖率收益 + 时间惩罚 + UAV 飞行能耗）
        base_shared_reward = (cur_cov - prev_cov) * self.W['coverage']
        if cur_cov < 1.0:
            base_shared_reward -= self.W['time_penalty']   # [审查修复4] 加强时间惩罚 0.1→0.3/tick，提高完成覆盖的紧迫感
        base_shared_reward += uav_reward_bonus

        # [Bug1修复] P1在下方 both_swapping 块统一计算，此处已删除重复的P1
        # 原代码两处各扣-8/tick = 实际-16/tick，120tick总计-1920（设计应为-960）

        # ── PSR + PSR-delta: 错峰结构奖励（方案C精化版 + ISCB）──────────────────
        # 修正要点：
        #   1. 仅当【两台都在换电】时屏蔽（封最坏情形），一台SWAP+一台WORK = 理想状态，不屏蔽！
        #   2. PSR-delta: 相位差主动增大时额外奖励，精确捕获错峰创建时段
        bat_0 = self.uavs['uav_0']['battery']
        bat_1 = self.uavs['uav_1']['battery']
        if 'uav_0' in self.agents and 'uav_1' in self.agents:
            swapping_0    = self.uavs['uav_0']['is_swapping']
            swapping_1    = self.uavs['uav_1']['is_swapping']
            both_swapping = swapping_0 and swapping_1

            # ══════════════════════════════════════════════════════════════════
            # 【人工奖励剥离】
            # 已移除 P1（并联换电惩罚 -8/tick）。原因：P1 过度惊吓了模型，导致其一到时间就死板换电。
            # 去掉 P1 后，我们回归“物理真实”：错峰换电的收益完全由自然降低的接驳等待时间、
            # 飞行 deadhead 损耗（由 time penalty -0.3/tick 和 P2 冲击惩罚承载）主导。
            # 这是真正的“自然涌现”。
            # ══════════════════════════════════════════════════════════════════


            # ══════════════════════════════════════════════════════════════════
            # 【奖励简化处理】用户明确要求：
            #   「当且仅当某一架提早换电造成错峰那一刻给奖励；
            #     其余时间的错峰换电都不应给予任何奖励」
            # 已保留：PSR-proj（私有、一次性）--- 在 _dispatch_uav_command 中计算并存入 _private_psrproj
            # 已移除：PSR-abs（每 tick 持续奖励 → 造成每200tick就返航的资本主义循环）
            # 已移除：SPP （同步压力惩罚  → 驱动模型在 bat=700 刻所就 RECHARGE）
            # ══════════════════════════════════════════════════════════════════

            self._prev_phase_diff    = abs(bat_0 - bat_1) / UAV_FULL_BATTERY
            self._prev_both_swapping = both_swapping

        # R5: UGV 最佳追随预定位奖励（仅 ugv_0 私有）
        # 始终追踪最亟需接驳（电量最低）的 UAV，促使 UGV 尽早奔赴目标点。
        ugv_positional_bonus = 0.0
        best_target_pt = None
        best_urgency   = -1.0
        
        for a in ['uav_0', 'uav_1']:
            uav = self.uavs[a]
            # 若正趴在车上换电，不需要追它
            if uav['is_swapping']:
                continue
                
            urgency = 1.0 - uav['battery'] / UAV_FULL_BATTERY  # 0=满电 → 1=空电
            if urgency > best_urgency:
                if uav['is_returning']:
                    # UAV 已经在返航路上：直接朝 UAV 的实时动态位置开（双向奔赴）
                    best_target_pt = (uav['x'], uav['y'])
                    best_urgency = urgency
                elif uav['is_busy'] and uav.get('fallback_return_point') is not None:
                    # UAV 正在干活：去它所在的区块最后兜底点等它
                    best_target_pt = uav['fallback_return_point']
                    best_urgency = urgency
                    
        if best_target_pt is not None:
            fx, fy = best_target_pt
            dist_to_target = math.hypot(self.car['x'] - fx, self.car['y'] - fy)
            MAX_DIST = MAP_SIZE * 1.5        # 归一化基准（约对角线 2828m）
            proximity = max(0.0, 1.0 - dist_to_target / MAX_DIST)
            # [修复]: 剥离 <0.30起步门槛 和 urgency 惩罚。只要在正确的接应点，就给全额奖励
            # 促使 UGV 哪怕在 UAV 仅仅消耗10%电量时，也全速前往最优接应位置抢占身位！
            ugv_positional_bonus = proximity * self.W['prepos']  # 最高 +prepos/tick（默认 3.0）

        # 初始化各智能体私有奖励字典
        rewards = {a: base_shared_reward for a in self.agents}
        if self.reward_mode == 'flat_shared':
            # [ablation] 扁平共享奖励：位置接近度奖励对所有 agent 共享（模拟 Standard H-MARL
            # 的稠密共享距离奖励），且不发放任何私有信用信号（PSR-proj / 冲击惩罚均关闭）。
            for _a in rewards:
                rewards[_a] += ugv_positional_bonus
        else:
            # 双通道（默认）：位置奖励仅 UGV 私有
            if 'ugv_0' in rewards:
                rewards['ugv_0'] += ugv_positional_bonus
            # PSR-proj 私有奖励（主动错峰通道）：精准发放给调用 act_recharge 的 UAV
            # 确保学习信号只影响决策者，消除共享奖励的「交叉污染」
            for _a in ['uav_0', 'uav_1']:
                if _a in rewards:
                    rewards[_a] += self._private_psrproj.get(_a, 0.0)

        # E1: 动态安全返航线（环境强制执行，所有模式下都保留以防坠机）
        #     + P2: 接驳冲击惩罚（reactive 通道，仅 dual_channel 且 enable_reactive 时计入）
        for a in ['uav_0', 'uav_1']:
            uav = self.uavs[a]
            dist_to_ugv = math.hypot(self.car['x'] - uav['x'], self.car['y'] - uav['y'])
            # [energy-sensitivity] 返航预留按 return_power_ratio 放大（E_safe∝P_return）；ratio=1.0 时与原式一致
            dynamic_low = max(UAV_LOW_BATTERY, int(dist_to_ugv / UAV_SPEED_RECHARGE * self.return_power_ratio) + 50)

            if uav['battery'] < dynamic_low and not uav['is_returning'] and not uav['is_swapping']:
                # [Diagnostic Hook] — silenced by default (floods logs with many envs);
                # set env var ESAFE_VERBOSE=1 to re-enable.
                import os as _os
                if dynamic_low > UAV_LOW_BATTERY and _os.environ.get("ESAFE_VERBOSE"):
                    print(f"[E_safe Trigger] {a} at ({uav['x']:.1f}, {uav['y']:.1f}), UGV at ({self.car['x']:.1f}, {self.car['y']:.1f}). Dist: {dist_to_ugv:.1f}m. Triggering at dynamic threshold {dynamic_low} (Static limit would be {UAV_LOW_BATTERY})")

                uav['waypoint_queue']        = []
                uav['is_busy']               = True
                uav['is_returning']          = True
                uav['is_swapping']           = False
                uav['swap_countdown']        = 0
                uav['fallback_return_point'] = None  # 被动返航，兜底点失效
                # P2: 被动触发的接驳冲击惩罚（reactive 通道，UAV+UGV 各承 50%）
                if self.reward_mode == 'dual_channel' and self.enable_reactive:
                    true_dist   = math.hypot(self.car['x'] - uav['x'], self.car['y'] - uav['y'])
                    shock_total = (true_dist / UGV_SPEED) * self.W['shock']
                    self._ugv_shock_penalty += shock_total * self.W['shock_ugv_frac']        # UGV 承受 shock_ugv_frac（默认 50%）
                    if a in rewards:
                        rewards[a] -= shock_total * (1.0 - self.W['shock_ugv_frac'])          # UAV 承受其余（默认 50%，即时）

        # 结算 UGV 接驳冲击惩罚
        if 'ugv_0' in rewards:
            rewards['ugv_0'] -= self._ugv_shock_penalty



        # ── 6. 终止检测 ────────────────────────────────────────────
        terminations = {a: False for a in self.agents}
        truncations  = {a: False for a in self.agents}
        infos        = {a: {} for a in self.agents}

        if dead_uav:
            for a in self.agents:
                rewards[a] = rewards.get(a, 0.0) - self.W['crash']
                terminations[a] = True
                infos[a]['reason'] = f"Crash: {fail_reason} battery depleted."

        if cur_cov >= 1.0:
            for a in self.agents:
                terminations[a] = True
                infos[a]['reason'] = "Success: 100% coverage."

        # 时间截断：防止 episode 永远不结束
        if self.t >= MAX_EPISODE_STEPS:
            for a in self.agents:
                truncations[a] = True
                infos[a]['reason'] = infos[a].get('reason', f"Timeout: {self.t} steps, cov={cur_cov:.2%}")


        if any(terminations.values()) or any(truncations.values()):
            self.agents = []

        # ── 6.5. 最终 NaN 保护（防止 reward 爆炸导致 PPO loss NaN） ──
        for a in rewards:
            if not math.isfinite(rewards[a]):
                rewards[a] = 0.0

        # ── 7. 观测生成 ────────────────────────────────────────────
        agents_for_obs = self.possible_agents if not self.agents else self.agents
        observations   = {a: self._get_obs(a) for a in agents_for_obs}

        return observations, rewards, terminations, truncations, infos


# ================================================================
#  单元测试 / 冒烟测试
# ================================================================

class _MockMapEnv:
    """仅用于单元测试的地图存根。"""
    def __init__(self, size=2000.0, grid_resolution=100.0):
        self.min_x = 0.0
        self.min_y = 0.0
        self.max_x = size
        self.max_y = size
        self.grid_resolution = grid_resolution
        self.rows = int(size / grid_resolution)
        self.cols = int(size / grid_resolution)
        self.coverage_grid = np.zeros((self.rows, self.cols), dtype=np.uint8)
        self.G_proj = self._build_grid(size, step=500.0)

    @staticmethod
    def _build_grid(size, step):
        G = nx.Graph()
        n = int(size / step) + 1
        for i in range(n):
            for j in range(n):
                nid = f"n_{i}_{j}"
                G.add_node(nid, x=float(j * step), y=float(i * step))
        for i in range(n):
            for j in range(n):
                nid = f"n_{i}_{j}"
                if j + 1 < n:
                    G.add_edge(nid, f"n_{i}_{j+1}", length=step)
                if i + 1 < n:
                    G.add_edge(nid, f"n_{i+1}_{j}", length=step)
        return G

    def xy_to_grid(self, x, y):
        col = int(np.clip(int((x - self.min_x) // self.grid_resolution), 0, self.cols - 1))
        row = int(np.clip(int((y - self.min_y) // self.grid_resolution), 0, self.rows - 1))
        return row, col


# if __name__ == '__main__':
#     print("=" * 65)
#     print("  HierarchicalEnv 冒烟测试")
#     print("=" * 65)

#     map_stub = _MockMapEnv()
#     env      = HierarchicalEnv(map_stub)

#     print(f"  路网节点数      : {env.num_nodes}")
#     print(f"  宏区块布局      : {MACRO_ROWS}×{MACRO_COLS} = {MACRO_ROWS*MACRO_COLS} 块")
#     print(f"  UAV 动作空间    : Discrete(27)")
#     print(f"  UGV 动作空间    : Discrete({MAX_NODES})")
#     print("-" * 65)

#     obs, info = env.reset(seed=0)

#     # 验证观测结构
#     for a in env.agents:
#         o = obs[a]
#         assert 'observation' in o and 'action_mask' in o, f"{a} 缺少 observation 或 action_mask"
#         mask = o['action_mask']
#         if a == 'ugv_0':
#             assert mask.shape == (27,), f"UGV mask shape 错误: {mask.shape}"
#         else:
#             assert mask.shape == (27,), f"UAV mask shape 错误: {mask.shape}"
#     print("  [✓] 观测结构 (observation + action_mask) 验证通过")

#     # 跑 20 步
#     total_rew = 0.0
#     for step_id in range(1, 21):
#         actions = {}
#         for a in env.agents:
#             mask = obs[a]['action_mask']
#             valid = np.where(mask == 1)[0]
#             if a == 'ugv_0':
#                 actions[a] = int(np.random.choice(valid))
#             else:
#                 # 前10步随机下达宏指令，后10步测试返航
#                 if step_id <= 10:
#                     actions[a] = int(np.random.choice(valid))
#                 else:
#                     actions[a] = ACT_RECHARGE

#         obs, rews, terms, truns, infos = env.step(actions)
#         total_rew += sum(rews.values())

#         if step_id % 5 == 0:
#             cov = env._compute_coverage_ratio() * 100
#             for a in ['uav_0', 'uav_1']:
#                 u = env.uavs[a]
#                 print(f"  [Tick {step_id:02d}] {a}: battery={u['battery']}  "
#                       f"is_busy={u['is_busy']}  "
#                       f"is_returning={u['is_returning']}  is_swapping={u['is_swapping']}  "
#                       f"swap_cd={u['swap_countdown']}  wps={len(u['waypoint_queue'])}")
#             print(f"           coverage={cov:.2f}%  cum_reward={total_rew:+.3f}")

#         if any(terms.values()):
#             print(f"  [终止] {infos}")
#             break

#     print("-" * 65)
#     print("  [✓] HierarchicalEnv 冒烟测试完成")
#     print("=" * 65)

if __name__ == '__main__':
    print("=" * 65)
    print("  断点续扫 (Resume Scanning) 专项验证测试")
    print("=" * 65)

    # 使用高分辨率 Mock 地图以便观察包围盒缩小 (10m一个格子)
    class _HighResMockMapEnv(_MockMapEnv):
        def __init__(self):
            super().__init__(size=2000.0, grid_resolution=10.0) 

    env = HierarchicalEnv(_HighResMockMapEnv())
    obs, info = env.reset(seed=42)

    uav = env.uavs['uav_0']
    
    print("\n================ [阶段 1] 首次派发区块 1 任务 ================")
    # 给 uav_0 下达动作 1 (去扫 1 号区块)
    env.step({'uav_0': 1, 'uav_1': 0, 'ugv_0': 0})
    print(f"当前生成的航点数量: {len(uav['waypoint_queue'])} 个")
    
    print("\n================ [阶段 2] 无人机干活中 (推演 350 步)... ================")
    for _ in range(350):
        env.step({'uav_0': 0, 'uav_1': 0, 'ugv_0': 0})  # 全体 NOOP，让底层飞
    
    print(f"当前剩余航点: {len(uav['waypoint_queue'])} 个")
    print(f"全局覆盖率进度: {env._compute_coverage_ratio()*100:.2f}%")

    print("\n================ [阶段 3] 强行打断召回 ================")
    env.step({'uav_0': 26, 'uav_1': 0, 'ugv_0': 0})  # 动作 26: 返回充电
    print(f"UAV 状态 -> 正在返航: {uav['is_returning']}, 航点是否被清空: {len(uav['waypoint_queue'])==0}")

    print("\n================ [阶段 4] 模拟换电完成，重返区块 1 ================")
    # 暴力重置状态（假装已经换完电了，省得等它慢慢飞回车顶）
    uav['is_returning'] = False
    uav['is_busy'] = False
    
    # 再次给 uav_0 下达动作 1
    env.step({'uav_0': 1, 'uav_1': 0, 'ugv_0': 0})
    print(f"续扫生成的航点数量: {len(uav['waypoint_queue'])} 个")
    print("=" * 65)