"""
FlatEnv.py
==========
MAPPO 扁平架构连续动作基线环境

动作空间：
  - UAV : Box(-1.0, 1.0, shape=(2,))  [vx_norm, vy_norm]
  - UGV : Box(-1.0, 1.0, shape=(2,))  [vx_norm, vy_norm]

观测：
  无语义封装。
"""

import math
import random
import functools
import numpy as np
import networkx as nx
from typing import Dict, Any, Optional, Tuple, List

from pettingzoo import ParallelEnv
from gymnasium import spaces

# ──────────────────────────────────────────────
# 全局物理常数 (必须匹配 Fairness Audit)
# ──────────────────────────────────────────────
DT               = 1.0       # 时间步长（秒）
MAX_EPISODE_STEPS = 30000    
MAP_SIZE         = 2000.0   
MAX_NODES        = 200      

# UAV
UAV_SPEED_WORK   = 10.0     
UAV_SPEED_RECHARGE = 20.0   
UAV_SCAN_WIDTH   = 50.0     
UAV_FULL_BATTERY = 900      
UAV_LOW_BATTERY  = 200      
UAV_RECHARGE_DIST = 2.0     

# UGV
UGV_SPEED        = 15.0     

MACRO_ROWS       = 5
MACRO_COLS       = 5

class FlatEnv(ParallelEnv):
    metadata = {
        'render_modes': ['human'],
        'name': 'flat_coverage_v0',
    }

    def __init__(self, map_env):
        super().__init__()
        self.map_env    = map_env
        self.grid_rows  = map_env.rows
        self.grid_cols  = map_env.cols

        # 稠密化路网
        self.G            = self._densify_graph(map_env.G_proj, max_len=20.0)
        self.nodes_list   = list(self.G.nodes())
        self.node_to_idx  = {n: i for i, n in enumerate(self.nodes_list)}
        self.num_nodes    = len(self.nodes_list)

        self.possible_agents = ['ugv_0', 'uav_0', 'uav_1']
        self.agents          = self.possible_agents[:]

        self.t              = 0
        self.coverage_grid  = np.zeros((self.grid_rows, self.grid_cols), dtype=np.uint8)
        self.car            = {}
        self.uavs           = {}

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

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Dict:
        coverage_sp = spaces.Box(low=0, high=1, shape=(self.grid_rows, self.grid_cols), dtype=np.uint8)
        if agent.startswith('ugv'):
            self_sp    = spaces.Box(0.0, 1.0, shape=(5,), dtype=np.float32)
            allies_sp  = spaces.Box(0.0, 1.0, shape=(10,), dtype=np.float32)
        else:
            self_sp    = spaces.Box(0.0, 1.0, shape=(7,), dtype=np.float32)
            allies_sp  = spaces.Box(0.0, 1.0, shape=(8,), dtype=np.float32)

        # Remove action_mask for FLAT baseline tests
        return spaces.Dict({'observation': spaces.Dict({
            'coverage_grid': coverage_sp,
            'self_state':    self_sp,
            'allies_state':  allies_sp,
        })})

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Box:
        # MAPPO baseline continuous action [-1, 1] 2D vector
        return spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def _get_obs(self, agent: str) -> Dict[str, Any]:
        mx, my = self.map_env.max_x, self.map_env.max_y
        cov = self.coverage_grid.copy()

        if agent == 'ugv_0':
            is_moving = float(self.car['target_node'] is not None and self.car['target_node'] != self.car['edge_u'])
            self_state = np.array([
                self.car['x'] / mx, self.car['y'] / my, UGV_SPEED / 30.0,
                self.node_to_idx.get(self.car['target_node'], 0) / max(1, self.num_nodes - 1) if self.car['target_node'] else 0.0,
                is_moving
            ], dtype=np.float32)
            
            allies = []
            for a in ['uav_0', 'uav_1']:
                u = self.uavs[a]
                allies.extend([u['x'] / mx, u['y'] / my, max(0.0, float(u['battery'])) / UAV_FULL_BATTERY, float(u['is_returning']), float(u['is_swapping'])])
            
            return {'observation': {'coverage_grid': cov, 'self_state': self_state, 'allies_state': np.array(allies, dtype=np.float32)}}
            
        else:
            u = self.uavs[agent]
            self_state = np.array([
                u['x'] / mx, u['y'] / my, max(0.0, float(u['battery'])) / UAV_FULL_BATTERY, 
                float(u['battery'] < UAV_LOW_BATTERY), 0.0, float(u['is_returning']), float(u['is_swapping'])
            ], dtype=np.float32)

            other = 'uav_1' if agent == 'uav_0' else 'uav_0'
            ou = self.uavs[other]
            allies_state = np.array([
                self.car['x'] / mx, self.car['y'] / my, UGV_SPEED / 30.0,
                ou['x'] / mx, ou['y'] / my, max(0.0, float(ou['battery'])) / UAV_FULL_BATTERY, 
                float(ou['is_returning']), float(ou['is_swapping'])
            ], dtype=np.float32)

            return {'observation': {'coverage_grid': cov, 'self_state': self_state, 'allies_state': allies_state}}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
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

        start_node = random.choice(self.nodes_list)
        self.car = {
            'edge_u': start_node, 'edge_v': start_node, 'progress': 0.0,
            'x': float(self.G.nodes[start_node]['x']), 'y': float(self.G.nodes[start_node]['y']),
            'target_node': start_node
        }

        cx, cy = (self.map_env.min_x + self.map_env.max_x)/2, (self.map_env.min_y + self.map_env.max_y)/2
        spread = min(self.map_env.max_x - self.map_env.min_x, self.map_env.max_y - self.map_env.min_y) * 0.1

        self.uavs = {}
        for a in ['uav_0', 'uav_1']:
            ux = cx + random.uniform(-spread, spread)
            uy = cy + random.uniform(-spread, spread)
            self.uavs[a] = {
                'x': ux, 'y': uy, 'prev_x': ux, 'prev_y': uy,
                'battery': UAV_FULL_BATTERY, 'is_returning': False, 'is_swapping': False, 'swap_countdown': 0
            }

        return {a: self._get_obs(a) for a in self.agents}, {a: {} for a in self.agents}

    def _step_ugv(self, act_vector):
        """Map [-1, 1] 2D vector to the nearest target road node incrementally"""
        vx, vy = act_vector
        tx = self.car['x'] + vx * UGV_SPEED * 10.0 # Propose a target 10s ahead
        ty = self.car['y'] + vy * UGV_SPEED * 10.0
        
        target_node = min(self.G.nodes(), key=lambda n: math.hypot(
                float(self.G.nodes[n]['x']) - tx, float(self.G.nodes[n]['y']) - ty))
        
        self.car['target_node'] = target_node
        self._ugv_move_toward(target_node, UGV_SPEED * DT)

    def _ugv_move_toward(self, target_node, budget):
        car = self.car
        u, v, prog = car['edge_u'], car['edge_v'], car['progress']

        if u == target_node and (u == v or prog == 0.0): return
        if u != v and prog > 0.0:
            edge_len  = self.G[u][v].get('length', 1.0)
            dist_rem  = edge_len - prog
            if budget >= dist_rem:
                budget -= dist_rem
                car['edge_u'], car['edge_v'], car['progress'] = v, v, 0.0
                car['x'], car['y'] = float(self.G.nodes[v]['x']), float(self.G.nodes[v]['y'])
                u = v
            else:
                car['progress'] += budget
                ratio = car['progress'] / edge_len
                car['x'] = self.G.nodes[u]['x'] + ratio * (self.G.nodes[v]['x'] - self.G.nodes[u]['x'])
                car['y'] = self.G.nodes[u]['y'] + ratio * (self.G.nodes[v]['y'] - self.G.nodes[u]['y'])
                return
        if u == target_node or budget <= 0: return

        try:
            path = nx.shortest_path(self.G, u, target_node, weight='length')
            if len(path) < 2: return
            n2 = path[1]
            el = self.G[u][n2].get('length', 1.0)
            if budget >= el:
                car['edge_u'], car['edge_v'], car['progress'] = n2, n2, 0.0
                car['x'], car['y'] = float(self.G.nodes[n2]['x']), float(self.G.nodes[n2]['y'])
            else:
                car['edge_v'], car['progress'] = n2, budget
                r = budget / el
                car['x'] = self.G.nodes[u]['x'] + r*(self.G.nodes[n2]['x'] - self.G.nodes[u]['x'])
                car['y'] = self.G.nodes[u]['y'] + r*(self.G.nodes[n2]['y'] - self.G.nodes[u]['y'])
        except nx.NetworkXNoPath:
            pass

    def _step_uav(self, agent: str, act_vector) -> float:
        uav, car = self.uavs[agent], self.car
        uav['prev_x'], uav['prev_y'] = uav['x'], uav['y']
        bonus = 0.0

        if uav['is_returning']:
            dx, dy = car['x'] - uav['x'], car['y'] - uav['y']
            dist = math.hypot(dx, dy)
            budget = UAV_SPEED_RECHARGE * DT
            if dist > 1e-6:
                m = min(dist, budget)
                uav['x'] += (dx/dist)*m
                uav['y'] += (dy/dist)*m
            else:
                uav['x'], uav['y'] = car['x'], car['y']
            
            bonus -= 0.05 * math.hypot(uav['x'] - uav['prev_x'], uav['y'] - uav['prev_y'])
            if dist <= UAV_RECHARGE_DIST:
                uav['is_returning'], uav['is_swapping'], uav['swap_countdown'] = False, True, 120

        elif uav['is_swapping']:
            uav['x'], uav['y'] = car['x'], car['y']
            uav['swap_countdown'] -= 1
            if uav['swap_countdown'] <= 0:
                uav['battery'], uav['is_swapping'] = UAV_FULL_BATTERY, False

        else:
            # Continuous map flight based on act_vector [vx_norm, vy_norm]
            vx, vy = act_vector
            mag = math.hypot(vx, vy)
            if mag > 1.0: vx, vy = vx/mag, vy/mag  # clip vector

            uav['x'] = np.clip(uav['x'] + vx * UAV_SPEED_WORK * DT, self.map_env.min_x, self.map_env.max_x)
            uav['y'] = np.clip(uav['y'] + vy * UAV_SPEED_WORK * DT, self.map_env.min_y, self.map_env.max_y)

        return bonus

    def _update_coverage(self):
        for a in ['uav_0', 'uav_1']:
            u = self.uavs[a]
            if not u['is_returning'] and not u['is_swapping']:
                if math.hypot(u['x'] - u['prev_x'], u['y'] - u['prev_y']) > 1e-3:
                    self._fill_uav_swath(u['prev_x'], u['prev_y'], u['x'], u['y'], UAV_SCAN_WIDTH/2.0)
                else:
                    self._fill_circle(u['x'], u['y'], UAV_SCAN_WIDTH/2.0)

    def _fill_circle(self, cx, cy, rad):
        res = self.map_env.grid_resolution
        rc = int(math.ceil(rad/res))
        cr, cc = self.map_env.xy_to_grid(cx, cy)
        for r in range(max(0, cr-rc), min(self.map_env.rows-1, cr+rc)+1):
            for c in range(max(0, cc-rc), min(self.map_env.cols-1, cc+rc)+1):
                px = self.map_env.min_x + (c+0.5)*res
                py = self.map_env.min_y + (r+0.5)*res
                if math.hypot(px-cx, py-cy) <= rad:
                    self.coverage_grid[r, c] = 1

    def _fill_uav_swath(self, x0, y0, x1, y1, hw):
        dx, dy = x1-x0, y1-y0
        l = math.hypot(dx, dy)
        if l < 1e-6: return
        nx, ny = dy/l, -dx/l
        verts = [(x0+nx*hw, y0+ny*hw), (x1+nx*hw, y1+ny*hw), (x1-nx*hw, y1-ny*hw), (x0-nx*hw, y0-ny*hw)]
        xs, ys = [v[0] for v in verts], [v[1] for v in verts]
        res = self.map_env.grid_resolution
        cmin = max(0, int((min(xs) - self.map_env.min_x)//res))
        cmax = min(self.map_env.cols-1, int((max(xs) - self.map_env.min_x)//res)+1)
        rmin = max(0, int((min(ys) - self.map_env.min_y)//res))
        rmax = min(self.map_env.rows-1, int((max(ys) - self.map_env.min_y)//res)+1)
        for r in range(rmin, rmax+1):
            for c in range(cmin, cmax+1):
                px = self.map_env.min_x + (c+0.5)*res
                py = self.map_env.min_y + (r+0.5)*res
                if self._point_in_convex_polygon(px, py, verts): self.coverage_grid[r, c] = 1

    @staticmethod
    def _point_in_convex_polygon(px, py, verts) -> bool:
        n = len(verts)
        sign = None
        for i in range(n):
            ax, ay = verts[i]
            bx, by = verts[(i+1)%n]
            cross = (bx-ax)*(py-ay) - (by-ay)*(px-ax)
            if abs(cross) < 1e-9: continue
            s = 1 if cross>0 else -1
            if sign is None: sign = s
            elif sign != s: return False
        return True

    def _compute_coverage_ratio(self) -> float:
        total = self.coverage_grid.size
        return float(np.sum(self.coverage_grid))/total if total > 0 else 0.0

    def step(self, actions: Dict[str, Any]):
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        self.t += 1
        prev_cov = self._compute_coverage_ratio()

        if 'ugv_0' in actions:
            self._step_ugv(actions['ugv_0'])

        uav_reward_bonus = 0.0
        for a in ['uav_0', 'uav_1']:
            if a in actions:
                uav_reward_bonus += self._step_uav(a, actions[a])

        dead_uav, fail_reason = False, ''
        for a in ['uav_0', 'uav_1']:
            if a in self.agents and not self.uavs[a]['is_swapping']:
                self.uavs[a]['battery'] -= 1
                if self.uavs[a]['battery'] < 0:
                    dead_uav, fail_reason = True, a

        self._update_coverage()
        cur_cov = self._compute_coverage_ratio()

        # REWARD ALIGNMENT: scale down massive per-step penalty, keeping absolute time significance.
        # HierarchicalEnv has implicitly large time steps where agents don't get -time_penalty every second, 
        # or they do but sparse. Here we use -0.01 per second which is equivalent to -1 per 100 seconds.
        base_shared_reward = (cur_cov - prev_cov) * 1000.0 - 0.01
        base_shared_reward += uav_reward_bonus

        rewards = {a: base_shared_reward for a in self.agents}

        for a in ['uav_0', 'uav_1']:
            u = self.uavs[a]
            
            # ── 🚨动态安全返航线 (Dynamic RTH) ──
            dist_to_ugv = math.hypot(self.car['x'] - u['x'], self.car['y'] - u['y'])
            dynamic_low = max(UAV_LOW_BATTERY, int(dist_to_ugv / 15.0) + 50)
            
            if u['battery'] < dynamic_low and not u['is_returning'] and not u['is_swapping']:
                u['is_returning'] = True
                u['is_swapping'] = False
                u['swap_countdown'] = 0

        terminations = {a: False for a in self.agents}
        truncations  = {a: False for a in self.agents}
        infos        = {a: {} for a in self.agents}

        if dead_uav:
            for a in self.agents:
                rewards[a] -= 500.0
                terminations[a], infos[a]['reason'] = True, f"Crash: {fail_reason}"
        if cur_cov >= 1.0:
            for a in self.agents:
                terminations[a], infos[a]['reason'] = True, "Success: 100%."
        if self.t >= MAX_EPISODE_STEPS:
            for a in self.agents:
                truncations[a], infos[a]['reason'] = True, f"Timeout: {self.t}"

        obs = {a: self._get_obs(a) for a in self.agents}

        if any(terminations.values()) or any(truncations.values()):
            self.agents = []

        return obs, rewards, terminations, truncations, infos