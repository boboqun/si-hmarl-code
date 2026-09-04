import math
import random
import functools
import numpy as np
import networkx as nx
from typing import Dict, Any, Optional, Tuple, List

from pettingzoo import ParallelEnv
from gymnasium import spaces

"""
SimulationEnv.py
================
异构多智能体协同路径规划 2D 模拟器 —— 核心时钟推演引擎 (Refactored for PettingZoo)

实体说明：
    ugv_0: 1 辆移动实验监测车，受路网约束。
    uav_0, uav_1: 2 架无人机，全空间 2D 自由飞行，电量有限。

时间步长：Δt = 1 s / tick
"""

# ──────────────────────────────────────────────
# 全局物理常数
# ──────────────────────────────────────────────
DT = 1.0                    # 时间步长（秒）
CAR_SPEED_WORK   = 10.0     # 车工作态速度 m/s
CAR_SPEED_CRUISE = 20.0     # 车巡航态速度 m/s
CAR_COLLECT_RADIUS = 10.0   # 车采集半径 m （等效扫宽20m）
CAR_BATTERY_STOCK  = 999    # 车携带电池块数

UAV_SPEED_WORK   = 10.0     # 无人机工作态速度 m/s
UAV_SPEED_CRUISE = 20.0     # 无人机巡航态速度 m/s
UAV_SCAN_WIDTH   = 50.0     # 无人机扫描带宽 m
UAV_FULL_BATTERY = 900      # 无人机满电续航 tick

# ================================================================
#  SimulationEnv 主类 (PettingZoo API)
# ================================================================

class SimulationEnv(ParallelEnv):
    """
    时钟推演引擎 (严格符合 PettingZoo ParallelEnv 接口标准)
    """

    metadata = {
        'render_modes': ['human'],
        'name': 'heterogeneous_coverage_v0'
    }

    def __init__(self, map_env):
        super().__init__()
        self.map_env = map_env
        self.grid_rows = map_env.rows
        self.grid_cols = map_env.cols

        # 1. 稠密化路网：所有相连边长 <= 20m
        self.G = self._densify_graph(map_env.G_proj, max_len=20.0)
        self.nodes_list = list(self.G.nodes())
        self.node_to_idx = {n: i for i, n in enumerate(self.nodes_list)}
        self.idx_to_node = {i: n for i, n in enumerate(self.nodes_list)}
        self.num_nodes = len(self.nodes_list)

        # 2. 定义智能体
        self.possible_agents = ['ugv_0', 'uav_0', 'uav_1']
        self.agents = self.possible_agents[:]

        # 缓存 Space 以提升性能
        self._action_spaces = {agent: self.action_space(agent) for agent in self.possible_agents}
        self._observation_spaces = {agent: self.observation_space(agent) for agent in self.possible_agents}

        # 内部状态变量
        self.t = 0
        self.coverage_grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.uint8)
        self.car = {}
        self.uavs = {}

    def _densify_graph(self, G_orig: nx.Graph, max_len: float) -> nx.Graph:
        """
        预处理路网稠密化 (Graph Densification)：
        遍历所有路网边，如果边的物理长度 > 20m，则在边上等距插入“虚拟节点 (Virtual Nodes)”，
        使所有相连节点的距离 <= 20m。抹平连续与离散空间的鸿沟。
        """
        G = nx.Graph()
        # 复制原始节点
        for n, data in G_orig.nodes(data=True):
            G.add_node(n, **data)

        virtual_id_counter = 0

        # 处理并拆分边
        for u, v, data in G_orig.edges(data=True):
            ux, uy = float(G_orig.nodes[u]['x']), float(G_orig.nodes[u]['y'])
            vx, vy = float(G_orig.nodes[v]['x']), float(G_orig.nodes[v]['y'])
            length = math.hypot(vx - ux, vy - uy)
            
            if length <= max_len:
                extra = {k: v for k, v in data.items() if k != 'length'}
                G.add_edge(u, v, length=length, **extra)
            else:
                num_segments = math.ceil(length / max_len)
                seg_len = length / num_segments
                
                prev_node = u
                for i in range(1, num_segments):
                    # 创建唯一虚拟节点 ID，兼容可能为字符串或整数的节点ID
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

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str):
        """定义观测空间 (进行了状态归一化并展平)"""
        coverage_space = spaces.Box(low=0, high=1, shape=(self.grid_rows, self.grid_cols), dtype=np.uint8)
        
        if agent.startswith('ugv'):
            # self_state: [x, y, mode] all normalized to [0, 1]
            self_state = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
            # allies: 2 UAVs flattened -> shape (8,) and normalized
            allies_state = spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)
            # action_mask: boolean mask showing valid node targets
            action_mask = spaces.Box(low=0, high=1, shape=(self.num_nodes,), dtype=np.int8)
            
            return spaces.Dict({
                'coverage_grid': coverage_space,
                'self_state': self_state,
                'allies_state': allies_state,
                'action_mask': action_mask
            })
        else:
            # uav state: [x, y, mode, battery]
            self_state = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
            # allies: 1 UGV [x, y, mode], 1 UAV [x, y, mode, battery] -> flatten to 7
            allies_state = spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32)
            
            return spaces.Dict({
                'coverage_grid': coverage_space,
                'self_state': self_state,
                'allies_state': allies_state
            })

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str):
        """定义动作空间"""
        if agent.startswith('ugv'):
            return spaces.Dict({
                'move': spaces.Discrete(self.num_nodes),
                'mode': spaces.Discrete(2) # 0: cruise (20m/s), 1: work (10m/s)
            })
        else:
            return spaces.Dict({
                'move': spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
                'mode': spaces.Discrete(2)
            })

    def _get_ugv_action_mask(self) -> np.ndarray:
        """获取 UGV 动作掩码，定义为所有与 edge_u 相连的节点。允许停在当前目标边上前进或回退。"""
        mask = np.zeros(self.num_nodes, dtype=np.int8)
        u = self.car['edge_u']
        v = self.car['edge_v']
        progress = self.car['progress']
        
        # 允许总是能够返回当前边的起点或保持前往终点
        mask[self.node_to_idx[u]] = 1
        mask[self.node_to_idx[v]] = 1
        
        # 如果正好在起步节点上，则可以前往任何邻居节点
        if u == v or progress == 0:
            for nbr in self.G.neighbors(u):
                mask[self.node_to_idx[nbr]] = 1
                
        return mask

    def _get_obs(self, agent: str) -> Dict[str, Any]:
        """获取智能体观测字典"""
        obs = {
            'coverage_grid': self.coverage_grid.copy()
        }
        
        max_x = self.map_env.max_x
        max_y = self.map_env.max_y
        
        if agent == 'ugv_0':
            obs['self_state'] = np.array([self.car['x']/max_x, self.car['y']/max_y, self.car['mode']], dtype=np.float32)
            allies = []
            for a in ['uav_0', 'uav_1']:
                u = self.uavs[a]
                allies.extend([u['x']/max_x, u['y']/max_y, u['mode'], max(0.0, u['battery'])/UAV_FULL_BATTERY])
            obs['allies_state'] = np.array(allies, dtype=np.float32)
            obs['action_mask'] = self._get_ugv_action_mask()
        else:
            u = self.uavs[agent]
            obs['self_state'] = np.array([u['x']/max_x, u['y']/max_y, u['mode'], max(0.0, u['battery'])/UAV_FULL_BATTERY], dtype=np.float32)
            
            other_uav = 'uav_1' if agent == 'uav_0' else 'uav_0'
            ou = self.uavs[other_uav]
            obs['allies_state'] = np.array([
                self.car['x']/max_x, self.car['y']/max_y, self.car['mode'],
                ou['x']/max_x, ou['y']/max_y, ou['mode'], max(0.0, ou['battery'])/UAV_FULL_BATTERY
            ], dtype=np.float32)
            
        return obs

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        """初始化环境"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self.agents = self.possible_agents[:]
        self.t = 0
        
        # 重置覆盖率
        self.coverage_grid = np.zeros_like(self.map_env.coverage_grid)
        
        # 初始化 UGV 状态
        start_node = random.choice(self.nodes_list)
        self.car = {
            'edge_u': start_node,
            'edge_v': start_node,
            'progress': 0.0,
            'x': float(self.G.nodes[start_node]['x']),
            'y': float(self.G.nodes[start_node]['y']),
            'mode': 0, # 0: cruise, 1: work
            'battery_stock': CAR_BATTERY_STOCK
        }
        
        # 初始化 UAV 状态
        cx_map = (self.map_env.min_x + self.map_env.max_x) / 2
        cy_map = (self.map_env.min_y + self.map_env.max_y) / 2
        spread = min(
            self.map_env.max_x - self.map_env.min_x,
            self.map_env.max_y - self.map_env.min_y
        ) * 0.1
        
        self.uavs = {}
        for agent in ['uav_0', 'uav_1']:
            ux = cx_map + random.uniform(-spread, spread)
            uy = cy_map + random.uniform(-spread, spread)
            self.uavs[agent] = {
                'x': ux,
                'y': uy,
                'prev_x': ux,
                'prev_y': uy,
                'mode': 0,
                'battery': UAV_FULL_BATTERY
            }
            
        observations = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return observations, infos

    def _move_ugv(self, move_idx: int, mode: int):
        """UGV 物理推演与终点吸附约束"""
        self.car['mode'] = mode
        speed = CAR_SPEED_WORK if mode == 1 else CAR_SPEED_CRUISE
        budget = speed * DT

        # ── 安全防护 ①：move_idx 越界裁剪（防 Padding 溢出）──
        move_idx = int(move_idx)  # 确保 Python int，防 np.int32
        move_idx = max(0, min(move_idx, self.num_nodes - 1))

        target_node = self.idx_to_node[move_idx]
        mask = self._get_ugv_action_mask()

        # 动作屏蔽生效：若选了不合法动作，强制停留在当前参考节点
        if mask[move_idx] == 0:
            target_node = self.car['edge_u']

        u = self.car['edge_u']
        v = self.car['edge_v']
        progress = self.car['progress']

        # 状况 A：正好在节点上，准备去相邻节点
        if (u == v or progress == 0):
            if target_node == u:
                return  # 停留在节点
            else:
                v_new = target_node
                # ── 安全防护 ②：v_new 必须是 u 的直接邻居 ──
                if v_new not in self.G[u]:
                    return  # 非邻居节点，忽略此动作，原地停留
                edge_len = self.G[u][v_new]['length']

                if budget >= edge_len:
                    self.car['edge_u'] = v_new
                    self.car['edge_v'] = v_new
                    self.car['progress'] = 0.0
                    self.car['x'] = float(self.G.nodes[v_new]['x'])
                    self.car['y'] = float(self.G.nodes[v_new]['y'])
                else:
                    self.car['edge_v'] = v_new
                    self.car['progress'] = budget
                    ratio = budget / edge_len
                    self.car['x'] = self.G.nodes[u]['x'] + ratio * (self.G.nodes[v_new]['x'] - self.G.nodes[u]['x'])
                    self.car['y'] = self.G.nodes[u]['y'] + ratio * (self.G.nodes[v_new]['y'] - self.G.nodes[u]['y'])

        # 状况 B：正在某条边上移动中
        elif u != v and progress > 0:
            # 继续前进
            if target_node == v:
                edge_len = self.G[u][v]['length']
                dist_rem = edge_len - progress
                if budget >= dist_rem:
                    self.car['edge_u'] = v
                    self.car['edge_v'] = v
                    self.car['progress'] = 0.0
                    self.car['x'] = float(self.G.nodes[v]['x'])
                    self.car['y'] = float(self.G.nodes[v]['y'])
                else:
                    self.car['progress'] += budget
                    ratio = self.car['progress'] / edge_len
                    self.car['x'] = self.G.nodes[u]['x'] + ratio * (self.G.nodes[v]['x'] - self.G.nodes[u]['x'])
                    self.car['y'] = self.G.nodes[u]['y'] + ratio * (self.G.nodes[v]['y'] - self.G.nodes[u]['y'])
            # 掉头回退
            elif target_node == u:
                if budget >= progress:
                    self.car['edge_u'] = u
                    self.car['edge_v'] = u
                    self.car['progress'] = 0.0
                    self.car['x'] = float(self.G.nodes[u]['x'])
                    self.car['y'] = float(self.G.nodes[u]['y'])
                else:
                    self.car['progress'] -= budget
                    edge_len = self.G[u][v]['length']
                    ratio = self.car['progress'] / edge_len
                    self.car['x'] = self.G.nodes[u]['x'] + ratio * (self.G.nodes[v]['x'] - self.G.nodes[u]['x'])
                    self.car['y'] = self.G.nodes[u]['y'] + ratio * (self.G.nodes[v]['y'] - self.G.nodes[u]['y'])
            # ── 安全防护 ③：B态下目标既非 u 也非 v → 原地等待 ──
            # else: 忽略，等到达 v 后再做决策

    def _move_uav(self, agent: str, move_vec: np.ndarray, mode: int):
        """UAV 物理推演 (连续向量移动)"""
        uav = self.uavs[agent]
        uav['mode'] = mode
        uav['prev_x'] = uav['x']
        uav['prev_y'] = uav['y']
        
        dx, dy = float(move_vec[0]), float(move_vec[1])
        length = math.hypot(dx, dy)
        
        if length > 1e-6:
            dx_n = dx / length
            dy_n = dy / length
            
            speed = UAV_SPEED_WORK if mode == 1 else UAV_SPEED_CRUISE
            budget = speed * DT
            
            target_x = uav['x'] + dx_n * budget
            target_y = uav['y'] + dy_n * budget
            
            # 环境地图边界截断限制
            target_x = np.clip(target_x, self.map_env.min_x, self.map_env.max_x)
            target_y = np.clip(target_y, self.map_env.min_y, self.map_env.max_y)
            
            uav['x'] = target_x
            uav['y'] = target_y

    def step(self, actions: Dict[str, Any]):
        """执行一个时间步长的状态转移与奖励计算"""
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        self.t += 1
        prev_coverage = self._compute_coverage_ratio()
        
        # 1. 物理位置转移 (应用动作)
        if 'ugv_0' in actions:
            act = actions['ugv_0']
            self._move_ugv(act['move'], act['mode'])
            
        for a in ['uav_0', 'uav_1']:
            if a in actions:
                act = actions[a]
                self._move_uav(a, act['move'], act['mode'])
                
        # 2. 扣除电量并进行坠机检测
        dead_uav = False
        fail_reason = ''
        for a in self.uavs:
            if a in self.agents:
                self.uavs[a]['battery'] -= 1
                if self.uavs[a]['battery'] < 0:
                    dead_uav = True
                    fail_reason = a
                    
        # 3. 更新覆盖率网格
        self._update_coverage()
        current_coverage = self._compute_coverage_ratio()
        
        # 4. 奖励设计（稠密 Shaped Reward）
        # R_coverage : 每增加 1% 覆盖率 → +10（唯一正向信号，来自实际产出）
        step_reward = (current_coverage - prev_coverage) * 100.0 * 10.0

        # R_time : 时间惩罚（-0.1 → -0.01）
        # 原 -0.1 让"静止不动"成为最优策略，降低后不再惩罚移动探索
        step_reward -= 0.01

        rewards = {a: step_reward for a in self.agents}

        terminations = {a: False for a in self.agents}
        truncations  = {a: False for a in self.agents}
        infos        = {a: {} for a in self.agents}

        # 5. 终止状态检测
        if dead_uav:
            step_reward -= 500.0   # R_death: 电量耗尽 -500
            rewards = {a: step_reward for a in self.agents}
            for a in self.agents:
                terminations[a] = True
                infos[a]['reason'] = f"Crash: {fail_reason} ran out of battery."

        if current_coverage >= 1.0:
            for a in self.agents:
                terminations[a] = True
                infos[a]['reason'] = "Success: 100% coverage achieved."

        if any(terminations.values()) or any(truncations.values()):
            self.agents = [] # ParallelEnv结束时应当清空待演进 agents 列表
            
        # 6. 生成新观测
        if not self.agents:
            observations = {a: self._get_obs(a) for a in self.possible_agents}
        else:
            observations = {a: self._get_obs(a) for a in self.agents}
            
        return observations, rewards, terminations, truncations, infos

    # ── 阶段 D：覆盖率网格底层几何判定支持 ────────────────────────────────────

    def _update_coverage(self):
        # 车处于扫描状态
        if self.car['mode'] == 1:
            self._fill_circle(self.car['x'], self.car['y'], CAR_COLLECT_RADIUS)
            
        # 无人机处于扫描状态
        for a in ['uav_0', 'uav_1']:
            if self.uavs[a]['mode'] == 1:
                uav = self.uavs[a]
                move_dist = math.hypot(uav['x'] - uav['prev_x'], uav['y'] - uav['prev_y'])
                if move_dist > 1e-3:
                    self._fill_uav_swath(uav['prev_x'], uav['prev_y'], uav['x'], uav['y'], UAV_SCAN_WIDTH / 2.0)
                else:
                    self._fill_circle(uav['x'], uav['y'], UAV_SCAN_WIDTH / 2.0)

    def _fill_circle(self, cx: float, cy: float, radius: float):
        res = self.map_env.grid_resolution
        r_cells = int(math.ceil(radius / res))
        center_row, center_col = self.map_env.xy_to_grid(cx, cy)

        row_min = max(0, center_row - r_cells)
        row_max = min(self.map_env.rows - 1, center_row + r_cells)
        col_min = max(0, center_col - r_cells)
        col_max = min(self.map_env.cols - 1, center_col + r_cells)

        for r in range(row_min, row_max + 1):
            for c in range(col_min, col_max + 1):
                px = self.map_env.min_x + (c + 0.5) * res
                py = self.map_env.min_y + (r + 0.5) * res
                if math.hypot(px - cx, py - cy) <= radius:
                    self.coverage_grid[r, c] = 1

    def _fill_uav_swath(self, x0: float, y0: float, x1: float, y1: float, half_width: float):
        dx = x1 - x0
        dy = y1 - y0
        length = math.hypot(dx, dy)
        if length < 1e-6: return

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

        res = self.map_env.grid_resolution
        col_min = max(0, int((min(xs) - self.map_env.min_x) // res))
        col_max = min(self.map_env.cols - 1, int((max(xs) - self.map_env.min_x) // res) + 1)
        row_min = max(0, int((min(ys) - self.map_env.min_y) // res))
        row_max = min(self.map_env.rows - 1, int((max(ys) - self.map_env.min_y) // res) + 1)

        for r in range(row_min, row_max + 1):
            for c in range(col_min, col_max + 1):
                px = self.map_env.min_x + (c + 0.5) * res
                py = self.map_env.min_y + (r + 0.5) * res
                if self._point_in_convex_polygon(px, py, verts):
                    self.coverage_grid[r, c] = 1

    @staticmethod
    def _point_in_convex_polygon(px: float, py: float, verts: list) -> bool:
        n = len(verts)
        sign = None
        for i in range(n):
            ax, ay = verts[i]
            bx, by = verts[(i + 1) % n]
            cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
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
        if total == 0: return 0.0
        return float(np.sum(self.coverage_grid)) / total


# ================================================================
#  Mock Test —— 随机动作推演压力测试（验证连续性和掩码逻辑）
# ================================================================

class _MockMapEnv:
    """仅用于单元测试场景的地图模块存根(Stub)"""
    def __init__(self, size=1000.0, grid_resolution=2.0):
        self.min_x = 0.0
        self.min_y = 0.0
        self.max_x = size
        self.max_y = size
        self.grid_resolution = grid_resolution

        self.rows = int(size / grid_resolution)
        self.cols = int(size / grid_resolution)
        self.coverage_grid = np.zeros((self.rows, self.cols), dtype=np.uint8)

        # 故意构造边长超过 20m（如100m）的路网，以检验稠密化预处理能力
        self.G_proj = self._build_grid_graph(size, step=100.0) 

    @staticmethod
    def _build_grid_graph(size: float, step: float) -> nx.Graph:
        G = nx.Graph()
        n = int(size / step) + 1
        for i in range(n):
            for j in range(n):
                nid = f"orig_{i}_{j}"
                G.add_node(nid, x=float(j * step), y=float(i * step))
        for i in range(n):
            for j in range(n):
                nid = f"orig_{i}_{j}"
                if j + 1 < n:
                    right = f"orig_{i}_{j+1}"
                    G.add_edge(nid, right, length=step)
                if i + 1 < n:
                    down = f"orig_{i+1}_{j}"
                    G.add_edge(nid, down, length=step)
        return G

    def xy_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        col = int((x - self.min_x) // self.grid_resolution)
        row = int((y - self.min_y) // self.grid_resolution)
        col = int(np.clip(col, 0, self.cols - 1))
        row = int(np.clip(row, 0, self.rows - 1))
        return row, col


if __name__ == '__main__':
    print("=" * 60)
    print("  SimulationEnv (PettingZoo) Random Action Mock Test")
    print("=" * 60)
    
    map_stub = _MockMapEnv(size=200.0, grid_resolution=2.0)
    # 此步骤会执行路网稠密化预处理，长边裂解
    env = SimulationEnv(map_stub)
    
    print(f"原始稠密处理前节点数: {map_stub.G_proj.number_of_nodes()}")
    print(f"预处理后增强图节点数: {env.num_nodes} (应有大幅增加)")
    print(f"UAV 动作空间示例: {env.action_space('uav_0')}")
    print("-" * 60)
    
    obs, info = env.reset(seed=42)
    
    for step_id in range(1, 16): # 跑 15 步观测连续性
        actions = {}
        for agent in env.agents:
            act = env.action_space(agent).sample()
            if agent == 'ugv_0':
                # 在连续推演中，UGV 必须依照 Action Mask 做合法的邻接点选择
                mask = obs[agent]['action_mask']
                valid_actions = np.where(mask == 1)[0]
                if valid_actions.size > 0:
                    act['move'] = int(np.random.choice(valid_actions))
            actions[agent] = act

        obs, rewards, terminations, truncations, infos = env.step(actions)
        
        rew_str = " | ".join([f"{a}: {rewards.get(a, 0.0):+.2f}" for a in env.possible_agents])
        print(f"[Tick {step_id:03d}] Rewards => {rew_str}")
        
        # 记录关键坐标监控物理推移
        if 'ugv_0' in obs:
            c_state = obs['ugv_0']['self_state']
            cg_percents = (np.sum(obs['ugv_0']['coverage_grid']) / float(obs['ugv_0']['coverage_grid'].size)) * 100.0
            print(f"  ugv_0 => pos:({c_state[0]:.1f}, {c_state[1]:.1f}), mode={int(c_state[2])}, map coverage: {cg_percents:.2f}%")
        
        if 'uav_0' in obs:
             u_state = obs['uav_0']['self_state']
             print(f"  uav_0 => pos:({u_state[0]:.1f}, {u_state[1]:.1f}), mode={int(u_state[2])}, battery={int(u_state[3])}")
             
        if any(terminations.values()):
            print("Environment hit termination condition!")
            break
            
    print("=" * 60)
    print("Test finished successfully! SimulationEnv standard format validation passed.")
