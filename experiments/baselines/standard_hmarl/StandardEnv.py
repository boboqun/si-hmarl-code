"""
StandardEnv.py
==================
分层强化学习环境 (Hierarchical Control Architecture)

架构说明：
  - 上层策略 (High-level Policy): 发出宏指令（去哪个区块扫、何时返回充电）
  - 底层执行 (Low-level Executor): SmartBoustrophedonPlanner 自动生成航点，
    每个 tick 沿航点直线移动 10m/s

角色设定：
  - uav_0, uav_1 : 扫图无人机，满电续航 300 ticks，扫描带宽 50m
  - ugv_0        : 移动充电站，沿路网行驶，不扫图

动作空间：
  - UAV : Discrete(27)  [0=保持, 1~25=前往区块, 26=返回充电]
  - UGV : Discrete(MAX_NODES)  [0~MAX_NODES-1=前往目标节点索引]

观测格式：{"observation": obs_dict, "action_mask": mask_array}

继承自 SimulationEnv.py 的：
  - _densify_graph, _fill_circle, _fill_uav_swath,
    _point_in_convex_polygon, _compute_coverage_ratio, _move_ugv

"""

import math
import random
import functools
import numpy as np
import networkx as nx
from typing import Dict, Any, Optional, Tuple, List

from pettingzoo import ParallelEnv
from gymnasium import spaces

import sys
import os
SYS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if SYS_ROOT not in sys.path:
    sys.path.insert(0, SYS_ROOT)

from experiments.my_method.boustrophedon_planner import SmartBoustrophedonPlanner

# ──────────────────────────────────────────────
# 全局物理常数
# ──────────────────────────────────────────────

DT               = 1.0       # 时间步长（秒）
MAX_EPISODE_STEPS = 30000    # 超时安全截断（约 8h 模拟时间，5+ 换电周期）
                             # 主要结束条件：100% 覆盖(termination) 或 UAV 断电(termination)
MAP_SIZE         = 2000.0   # 地图边长 (m) —— 2km 真实农田场景
MAX_NODES        = 800      # 路网节点索引上限（Padding 容量）

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
MACRO_BLOCK_SIZE = MAP_SIZE / MACRO_COLS   # 40.0m

# 动作编号
ACT_NOOP         = 0         # UAV: 保持当前状态
ACT_BLOCK_START  = 1         # UAV: 区块指令起始编号
ACT_BLOCK_END    = 25        # UAV: 区块指令结束编号
ACT_RECHARGE     = 26        # UAV: 返回充电


# ================================================================
#  StandardEnv 主类 (PettingZoo ParallelEnv)
# ================================================================

class StandardEnv(ParallelEnv):
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

    def __init__(self, map_env):
        super().__init__()
        self.map_env    = map_env
        self.grid_rows  = map_env.rows
        self.grid_cols  = map_env.cols
        self.planner    = SmartBoustrophedonPlanner()

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
            # allies: 2 UAV × 6 = [x, y, battery, is_returning, is_swapping, task_block]
            allies_sp  = spaces.Box(0.0, 1.0, shape=(12,), dtype=np.float32)
            obs_sp     = spaces.Dict({
                'coverage_grid': coverage_sp,
                'self_state':    self_sp,
                'allies_state':  allies_sp,
            })
            mask_sp    = spaces.Box(0, 1, shape=(27,), dtype=np.int8)
        else:
            # UAV self_state: [x, y, battery, is_busy, task_block, is_returning, is_swapping]
            self_sp    = spaces.Box(0.0, 1.0, shape=(7,), dtype=np.float32)
            # allies: UGV(3) + other_UAV(6) = [ugv_x, ugv_y, ugv_speed,
            #         uav_x, uav_y, uav_battery, uav_is_returning, uav_is_swapping, uav_task_block]
            allies_sp  = spaces.Box(0.0, 1.0, shape=(9,), dtype=np.float32)
            obs_sp     = spaces.Dict({
                'coverage_grid': coverage_sp,
                'self_state':    self_sp,
                'allies_state':  allies_sp,
            })
            mask_sp    = spaces.Box(0, 1, shape=(27,), dtype=np.int8)

        return spaces.Dict({'observation': obs_sp, 'action_mask': mask_sp})

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Discrete:
        if agent.startswith('ugv'):
            return spaces.Discrete(27)  # 与 UAV 统一为 27 维语义动作空间
        else:
            return spaces.Discrete(27)   # 0=noop, 1~25=block, 26=recharge

    # ─────────────────────────────────────────────────────────────
    #  动作掩码
    # ─────────────────────────────────────────────────────────────

    def _get_ugv_action_mask(self) -> np.ndarray:
        """
        UGV 动作掩码 —— 27 维语义动作版。
        0: 停靠待命， 1~25: 驶向宏区块中心待命， 26: 动态跟车/接驳
        """
        mask = np.zeros(27, dtype=np.int8)

        # FSM-2: 换电中 → 强制停靠 (动作 0)
        swapping = any(self.uavs[a]['is_swapping'] for a in ['uav_0', 'uav_1'])
        if swapping:
            mask[0] = 1
            return mask

        # FSM-1: 返航中 → 强制动态接驳 (动作 26)
        returning = any(self.uavs[a]['is_returning'] for a in ['uav_0', 'uav_1'])
        if returning:
            mask[26] = 1
            return mask

        # 动作承诺锁：正在驶向区块中心时，不允许中途换边
        semantic_act = self.car.get('semantic_action')
        if semantic_act is not None and 1 <= semantic_act <= 25:
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

        # 常规：开放所有 27 维语义动作
        mask[:] = 1
        return mask

    def _get_uav_action_mask(self, agent: str) -> np.ndarray:
        """
        UAV 掩码 (27位)：0=NOOP, 1~25=区块, 26=RECHARGE

        状态硬锁（优先级由高到低）：
          is_returning / is_swapping → 绝对只允许 NOOP，防止动作刷屏重置 FSM
          is_busy (扫图中)           → 允许 NOOP 或主动中止去充电 (RECHARGE)
          空闲                       → 全开，但屏蔽队友正在执行的区块（协作破缺）
        """
        mask = np.zeros(27, dtype=np.int8)
        uav  = self.uavs[agent]

        # 1. 强制硬锁：返航或换电中，绝对只允许 NOOP，防止 countdown 被重置
        if uav['is_returning'] or uav['is_swapping']:
            mask[ACT_NOOP] = 1
            return mask

        # 2. 扫图中：允许继续（NOOP）或紧急中止去充电（RECHARGE）
        if uav['is_busy']:
            mask[ACT_NOOP]     = 1
            mask[ACT_RECHARGE] = 1
            return mask

        # 3. 完全空闲：全开放区块 1~25 + 返航 26，权权交给 AI 自主决策
        #    早退的代价由 _dispatch_uav_command 的奖励塑形惩罚来调节
        mask[1:27] = 1

        # 4. 协作破缺：屏蔽队友正在执行的区块，鼓励分工覆盖
        other = 'uav_1' if agent == 'uav_0' else 'uav_0'
        tm_block = self.uavs[other].get('current_block', 0)
        if 1 <= tm_block <= 25:
            mask[tm_block] = 0

        # 5. 已完成区块黑名单：屏蔽 100% 覆盖的区块，防止重复劳动
        for b in self._get_completed_blocks():
            if 1 <= b <= 25:
                mask[b] = 0

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
        rows_per_block = max(1, len(grid)     // MACRO_ROWS)
        cols_per_block = max(1, len(grid[0])  // MACRO_COLS)

        for block_id in range(1, MACRO_ROWS * MACRO_COLS + 1):
            idx = block_id - 1
            row = idx // MACRO_COLS
            col = idx  % MACRO_COLS
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
        将区块编号 1~25 转换为 (rect_min, rect_max)。
        区块从左下角开始，按行优先（row-major）编号。
        block_id=1 → 第0行第0列，block_id=25 → 第4行第4列。
        """
        idx  = block_id - 1                       # 0-indexed
        row  = idx // MACRO_COLS
        col  = idx %  MACRO_COLS
        x0   = self.map_env.min_x + col * MACRO_BLOCK_SIZE
        y0   = self.map_env.min_y + row * MACRO_BLOCK_SIZE
        x1   = x0 + MACRO_BLOCK_SIZE
        y1   = y0 + MACRO_BLOCK_SIZE
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
            # is_moving: 是否有明确目标节点正在行驶
            is_moving = float(car['target_node'] is not None and
                              car['target_node'] != car['edge_u'])
            self_state = np.array([
                car['x'] / mx,
                car['y'] / my,
                UGV_SPEED / 30.0,   # 归一化速度
                (self.node_to_idx.get(car['target_node'], 0)
                 / max(1, self.num_nodes - 1)) if car['target_node'] else 0.0,
                is_moving,
            ], dtype=np.float32)

            # allies_state 10维: 2 × UAV(5) = [x, y, battery, is_returning, is_swapping]
            allies = []
            for a in ['uav_0', 'uav_1']:
                u = self.uavs[a]
                allies.extend([
                    u['x'] / mx,
                    u['y'] / my,
                    max(0.0, float(u['battery'])) / UAV_FULL_BATTERY,
                    float(u['is_returning']),
                    float(u['is_swapping']),
                    float(u.get('current_block', 0)) / 25.0,
                ])
            allies_state = np.array(allies, dtype=np.float32)

            obs = {
                'coverage_grid': cov,
                'self_state':    self_state,
                'allies_state':  allies_state,
            }
            mask = self._get_ugv_action_mask()

        else:
            u = self.uavs[agent]
            # 当前任务区块归一化（0 = 无任务）
            task_block_norm = float(u.get('current_block', 0)) / 25.0

            # self_state 7维: [x, y, battery, is_busy, task_block, is_returning, is_swapping]
            self_state = np.array([
                u['x'] / mx,
                u['y'] / my,
                max(0.0, float(u['battery'])) / UAV_FULL_BATTERY,
                float(u['is_busy']),
                task_block_norm,
                float(u['is_returning']),
                float(u['is_swapping']),
            ], dtype=np.float32)

            other_uav = 'uav_1' if agent == 'uav_0' else 'uav_0'
            ou  = self.uavs[other_uav]
            car = self.car
            # allies_state 9维: UGV(3) + other_UAV(6)
            # other_UAV: [x, y, battery, is_returning, is_swapping, task_block]
            allies_state = np.array([
                car['x'] / mx, car['y'] / my, UGV_SPEED / 30.0,
                ou['x'] / mx,  ou['y'] / my,
                max(0.0, float(ou['battery'])) / UAV_FULL_BATTERY,
                float(ou['is_returning']),
                float(ou['is_swapping']),
                float(ou.get('current_block', 0)) / 25.0,
            ], dtype=np.float32)

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

        # UGV 初始化
        start_node = random.choice(self.nodes_list)
        self.car = {
            'edge_u':      start_node,
            'edge_v':      start_node,
            'progress':    0.0,
            'x':           float(self.G.nodes[start_node]['x']),
            'y':           float(self.G.nodes[start_node]['y']),
            'target_node': None,    # 当前导航目标节点
            'semantic_action': None,  # 当前宏观语义动作
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
                'x':              ux,
                'y':              uy,
                'prev_x':         ux,
                'prev_y':         uy,
                'battery':        UAV_FULL_BATTERY,
                'is_busy':        False,   # 底层是否正在执行航点任务
                'is_returning':   False,   # FSM: 正在飞向汇合点（耗电）
                'is_swapping':    False,   # FSM: 已降落，正在换电（不耗电）
                'swap_countdown': 0,       # 换电倒计时（ticks）
                'waypoint_queue': [],      # 底层航点队列
                'current_block':  0,       # 当前任务区块（0=无）
            }

        observations = {a: self._get_obs(a) for a in self.agents}
        infos        = {a: {} for a in self.agents}
        return observations, infos

    # ─────────────────────────────────────────────────────────────
    #  UGV 移动（继承 SimulationEnv 的路网推演逻辑）
    # ─────────────────────────────────────────────────────────────

    def _step_ugv(self, action: int):
        """27 维语义动作 → 底层路网目标节点，驱动物理引擎。"""
        self.car['semantic_action'] = action
        v = self.car['edge_v']

        if action == 0:
            target_node = v  # 停靠

        elif 1 <= action <= 25:
            # 预判行为：驶向目标宏区块实物中心最近路网节点
            (x0, y0), (x1, y1) = self._block_id_to_rect(action)
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            target_node = min(self.G.nodes(), key=lambda n: math.hypot(
                float(self.G.nodes[n]['x']) - cx, float(self.G.nodes[n]['y']) - cy))

        elif action == 26:
            # 动态伴随/接驳模式
            returning_uav = next((u for u in self.uavs.values() if u['is_returning']), None)
            if returning_uav:
                # 救援优先：驶向返航 UAV 当前最近的路网节点，底层 _step_uav 完成物理吸附
                target_node = min(self.G.nodes(), key=lambda n: math.hypot(
                    float(self.G.nodes[n]['x']) - returning_uav['x'],
                    float(self.G.nodes[n]['y']) - returning_uav['y']))
            else:
                # 作业伴随：驶向工作机群重心最近节点
                working = [u for u in self.uavs.values()
                           if u.get('is_busy') and not u['is_returning'] and not u['is_swapping']]
                if working:
                    cx = sum(u['x'] for u in working) / len(working)
                    cy = sum(u['y'] for u in working) / len(working)
                    target_node = min(self.G.nodes(), key=lambda n: math.hypot(
                        float(self.G.nodes[n]['x']) - cx, float(self.G.nodes[n]['y']) - cy))
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

        elif action == ACT_RECHARGE:
            # 仅保留任务中途放弃的极小惩罚，防止无意义的频繁状态横跳
            if uav['is_busy']:
                self._dispatch_bonus -= 10.0
                uav['waypoint_queue'] = []
                uav['current_block']  = 0

            uav['is_busy']         = True
            uav['is_returning']    = True
            uav['is_swapping']     = False
            uav['swap_countdown']  = 0

            # 删除稀疏突发冲击惩罚 (Sparse Shock Penalty)
            pass


        elif ACT_BLOCK_START <= action <= ACT_BLOCK_END:
            if not uav['is_busy']:
                # 1. 获取原始区块的物理边界
                (x0, y0), (x1, y1) = self._block_id_to_rect(action)

                # 2. 动态断点续扫：提取该区块内未覆盖区域的紧凑包围盒
                res = self.map_env.grid_resolution
                r0, c0 = self.map_env.xy_to_grid(x0, y0)
                r1, c1 = self.map_env.xy_to_grid(x1 - 1e-3, y1 - 1e-3)  # 防浮点边界溢出

                # 裁剪出当前区块的覆盖率子视图
                sub_grid = self.coverage_grid[r0:r1+1, c0:c1+1]

                # 找到所有未覆盖 (值为 0) 的局部索引
                uncovered = np.where(sub_grid == 0)

                if len(uncovered[0]) > 0:
                    # 传统方法 (Standard H-MARL) 不具备基于断点的 "Resume Scanning" (提取局部包围盒) 能力
                    # 只要该区块存在未扫区域，UAV 就会死板地从零开始完整扫过整个大方格
                    rect_min = (x0, y0)
                    rect_max = (x1, y1)
                else:
                    # 该区块已全覆盖，取消任务
                    return

                start_ref = (uav['x'], uav['y'])
                end_ref   = (self.car['x'], self.car['y'])

                # 静默传统基线的 log，以免与我们的方法混淆
                # print(f"\n[Debug 航点生成] 分配区块 {action}")
                # print(f"  -> 原始物理边界: X({x0:.1f}~{x1:.1f}), Y({y0:.1f}~{y1:.1f})")
                # print(f"  -> 不支持断点续扫，直接分配大区块: X({x0:.1f}~{x1:.1f}), Y({y0:.1f}~{y1:.1f})")

                # 3. 将缩水的未覆盖边界丢给规划器，生成续扫航点
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

        if uav['is_returning']:
            # ── FSM-1: 动态返航追踪 UGV ──
            # 实时将 UGV 的当前连续物理坐标作为追踪目标（支持路段任意位置接驳）
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

            # 🚨 核心：接驳不依赖 Node！只要物理距离足够近，可在马路中央随时换电！
            if dist <= UAV_RECHARGE_DIST:
                uav['is_returning']   = False
                uav['is_swapping']    = True
                uav['swap_countdown'] = 120

        elif uav['is_swapping']:
            # ── FSM-2: 机械换电（坐标绑定 UGV，step() 不扣电）──
            uav['x'] = car['x']   # 绑定 UGV 坐标（随车移动）
            uav['y'] = car['y']
            uav['swap_countdown'] -= 1

            if uav['swap_countdown'] <= 0:
                # 换电完成
                uav['battery']      = UAV_FULL_BATTERY
                uav['is_swapping']  = False
                uav['is_busy']      = False
                uav['current_block'] = 0
                # 🚨 防作弊锁：取消换电完成奖励，唯一正向信号只能来自覆盖率提升
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
        claimed_this_tick = set()   # 当帧已被抒定的区块
        for a in ['uav_0', 'uav_1']:
            if a in actions:
                act = int(actions[a])

                # 区块接令：拦截冲突与重复劳动
                if ACT_BLOCK_START <= act <= ACT_BLOCK_END:
                    other = 'uav_1' if a == 'uav_0' else 'uav_0'
                    tm_block  = self.uavs[other].get('current_block', 0)
                    completed = self._get_completed_blocks()
                    # 队友正在执行 或 当帧已被抒定 或 已 100% 覆盖 → 强制 NOOP
                    if act == tm_block or act in claimed_this_tick or act in completed:
                        act = ACT_NOOP
                    else:
                        claimed_this_tick.add(act)

                self._dispatch_uav_command(a, act)

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
                self.uavs[a]['battery'] -= 1
                if self.uavs[a]['battery'] < 0:
                    dead_uav    = True
                    fail_reason = a

        # ── 4. 覆盖率更新 ──────────────────────────────────────────
        self._update_coverage()
        cur_cov = self._compute_coverage_ratio()

        # ── 5. 奖励计算（私有化版）───────────────────────────────────────
        # 基础共享奖励：覆盖率收益 + 全局时间惩罚 + UAV 微操扣分
        base_shared_reward = (cur_cov - prev_cov) * 1000.0
        if cur_cov < 1.0:
            base_shared_reward -= 0.1
        base_shared_reward += uav_reward_bonus

        # 初始化各智能体的私有奖励字典
        rewards = {a: base_shared_reward for a in self.agents}

        # QoS 牵徕 + Auto-RTH：低电量强制接管
        for a in ['uav_0', 'uav_1']:
            uav = self.uavs[a]
            
            # ── 🚨动态安全返航线 (Dynamic RTH) ──
            dist_to_ugv = math.hypot(self.car['x'] - uav['x'], self.car['y'] - uav['y'])
            dynamic_low = max(UAV_LOW_BATTERY, int(dist_to_ugv / 15.0) + 50)
            
            if uav['battery'] < dynamic_low and not uav['is_returning'] and not uav['is_swapping']:
                uav['waypoint_queue']  = []
                uav['is_busy']         = True
                uav['is_returning']    = True
                uav['is_swapping']     = False
                uav['swap_countdown']  = 0

        # 统一结算 UGV 的私有惩罚：密集空间距离约束 (Dense Spatial Penalty)
        # 传统 H-MARL 使用每步的距离惩罚牵制 UGV，使得 UGV 退化为被动短视跟随
        if 'ugv_0' in rewards:
            d1 = math.hypot(self.car['x'] - self.uavs['uav_0']['x'], self.car['y'] - self.uavs['uav_0']['y'])
            d2 = math.hypot(self.car['x'] - self.uavs['uav_1']['x'], self.car['y'] - self.uavs['uav_1']['y'])
            MAX_DIST = math.hypot(self.map_env.max_x - self.map_env.min_x, self.map_env.max_y - self.map_env.min_y)
            # 均摊密集拉扯惩罚
            dense_penalty = - (d1 + d2) / max(1.0, MAX_DIST) * 0.1
            rewards['ugv_0'] += dense_penalty


        # ── 6. 终止检测 ────────────────────────────────────────────
        terminations = {a: False for a in self.agents}
        truncations  = {a: False for a in self.agents}
        infos        = {a: {} for a in self.agents}

        if dead_uav:
            for a in self.agents:
                rewards[a] = rewards.get(a, 0.0) - 500.0
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
#     print("  StandardEnv 冒烟测试")
#     print("=" * 65)

#     map_stub = _MockMapEnv()
#     env      = StandardEnv(map_stub)

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
#     print("  [✓] StandardEnv 冒烟测试完成")
#     print("=" * 65)

if __name__ == '__main__':
    print("=" * 65)
    print("  断点续扫 (Resume Scanning) 专项验证测试")
    print("=" * 65)

    # 使用高分辨率 Mock 地图以便观察包围盒缩小 (10m一个格子)
    class _HighResMockMapEnv(_MockMapEnv):
        def __init__(self):
            super().__init__(size=2000.0, grid_resolution=10.0) 

    env = StandardEnv(_HighResMockMapEnv())
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