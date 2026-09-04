import math
import numpy as np
import networkx as nx
from typing import Dict, Any

from pettingzoo.utils.wrappers import BaseParallelWrapper
from gymnasium import spaces

from SimulationEnv import SimulationEnv, _MockMapEnv, UAV_FULL_BATTERY, UAV_SPEED_CRUISE, CAR_SPEED_CRUISE, DT

class DispatcherWrapper(BaseParallelWrapper):
    """
    高层调度器 Wrapper：
    监控 UAV 电量，引入排队 (Queue) 与休眠 (Sleep) 机制。
    当多架无人机同时低优时，排队前往汇聚点降落休眠；车只给 current_swapping_uav 充电 120 帧。
    """
    def __init__(self, env):
        super().__init__(env)
        self.BATTERY_WARNING = 250
        self.SWAP_DURATION = 120
        
        self.rescue_queue = []
        self.current_swapping_uav = None
        
        self.rendezvous_node = None
        self.swap_countdown = 0
        
    def reset(self, seed=None, options=None):
        self.rescue_queue = []
        self.current_swapping_uav = None
        self.rendezvous_node = None
        self.swap_countdown = 0
        return super().reset(seed=seed, options=options)

    @property
    def is_rescue_mode(self):
        return len(self.rescue_queue) > 0 or self.current_swapping_uav is not None

    def step(self, actions: Dict[str, Any]):
        env_u = self.unwrapped
        
        # ── 1. 触发与入队检测 ──
        for uav_id in ['uav_0', 'uav_1']:
            if uav_id in env_u.agents:
                uav = env_u.uavs[uav_id]
                if uav['battery'] < self.BATTERY_WARNING:
                    if uav_id not in self.rescue_queue and uav_id != self.current_swapping_uav:
                        self.rescue_queue.append(uav_id)
                        print(f"[{env_u.t:03d}] ⚡️ [调度器介入] {uav_id} 电量告警({uav['battery']})！加入救援队列。")
                        
                        # 如果车闲置并且刚进人，计算唯一相遇点
                        if self.rendezvous_node is None or (not self.current_swapping_uav and len(self.rescue_queue) == 1):
                             self.rendezvous_node = self._select_best_rendezvous_node(self.rescue_queue[0])
                             print(f"[{env_u.t:03d}] 🎯 [目标计算] 为队列首位 {self.rescue_queue[0]} 分配契约节点: {self.rendezvous_node}")

        # ── 2. 接管动作层 ──
        if self.is_rescue_mode:
            car = env_u.car
            target_data = env_u.G.nodes[self.rendezvous_node] if self.rendezvous_node else None
            if target_data:
                tx, ty = float(target_data['x']), float(target_data['y'])
            else:
                tx, ty = car['x'], car['y']
            
            dist_car = math.hypot(tx - car['x'], ty - car['y'])
            
            # --- 1. 换电触发判定 ---
            if self.current_swapping_uav is None and dist_car < 1.0 and len(self.rescue_queue) > 0:
                # 检查队列中是否有“已降落”具备换电条件的飞机
                for idx, uav_id in enumerate(self.rescue_queue):
                    if uav_id not in env_u.uavs: continue
                    u = env_u.uavs[uav_id]
                    if math.hypot(tx - u['x'], ty - u['y']) < 1.0:
                        self.current_swapping_uav = self.rescue_queue.pop(idx)
                        self.swap_countdown = self.SWAP_DURATION
                        print(f"[{env_u.t:03d}] 🔋 [换电死锁] 开始给 {self.current_swapping_uav} 充电 120s！剩余队列: {self.rescue_queue}")
                        break
                        
            # --- 2. 处理队列里的等待/降落机制 (不含刚被 pop 的飞机) ---
            for uav_id in self.rescue_queue:
                if uav_id not in actions: continue
                uav = env_u.uavs[uav_id]
                dist_uav = math.hypot(tx - uav['x'], ty - uav['y'])
                
                # 强制覆盖模式：归零巡航
                actions[uav_id]['mode'] = 0
                
                if dist_uav < 1.0 and dist_car < 1.0:
                    # 【降落休眠】
                    actions[uav_id]['move'] = np.array([0.0, 0.0], dtype=np.float32)
                    uav['battery'] += 1 # 抵消底层扣电
                else:
                    # 赶路
                    dx = tx - uav['x']
                    dy = ty - uav['y']
                    dist_to_go = math.hypot(dx, dy)
                    if dist_to_go <= UAV_SPEED_CRUISE * DT + 1e-3:
                        uav['x'], uav['y'] = tx, ty
                        actions[uav_id]['move'] = np.array([0.0, 0.0], dtype=np.float32)
                    else:
                        dx_norm = dx / dist_to_go
                        dy_norm = dy / dist_to_go
                        actions[uav_id]['move'] = np.array([dx_norm, dy_norm], dtype=np.float32)
                        
            # --- 3. 换电死锁队列管理 ---
            if self.current_swapping_uav is not None:
                # 正在死锁中
                if 'ugv_0' in actions:
                    actions['ugv_0']['mode'] = 0
                    actions['ugv_0']['move'] = env_u.node_to_idx[car['edge_u']]
                    
                if self.current_swapping_uav in actions:
                    actions[self.current_swapping_uav]['mode'] = 0
                    actions[self.current_swapping_uav]['move'] = np.array([0.0, 0.0], dtype=np.float32)
                    env_u.uavs[self.current_swapping_uav]['battery'] += 1 # 充电抵扣耗电
                    
                self.swap_countdown -= 1
                
                if self.swap_countdown <= 0:
                    print(f"[{env_u.t:03d}] ✅ [换电完成] {self.current_swapping_uav} 满电释放！")
                    env_u.uavs[self.current_swapping_uav]['battery'] = UAV_FULL_BATTERY
                    if env_u.car['battery_stock'] > 0: env_u.car['battery_stock'] -= 1
                    
                    self.current_swapping_uav = None
                    if not self.rescue_queue:
                         self.rendezvous_node = None # 如果队列空了，清空集合点
            else:
                # 如果没有人在换电，说明还在赶路中
                if 'ugv_0' in actions and self.rendezvous_node:
                    actions['ugv_0']['mode'] = 0
                    if car['edge_u'] != car['edge_v'] and car['progress'] > 0:
                        actions['ugv_0']['move'] = env_u.node_to_idx[car['edge_v']]
                    else:
                        try:
                            path = nx.shortest_path(env_u.G, car['edge_v'], self.rendezvous_node, weight='length')
                            next_node = path[1] if len(path) > 1 else path[0]
                            actions['ugv_0']['move'] = env_u.node_to_idx[next_node]
                        except nx.NetworkXNoPath:
                            actions['ugv_0']['move'] = env_u.node_to_idx[car['edge_v']]

        return super().step(actions)

    def _select_best_rendezvous_node(self, uav_id: str) -> str:
        env_u = self.unwrapped
        uav = env_u.uavs[uav_id]
        car = env_u.car
        
        uncovered_coords = np.argwhere(env_u.coverage_grid == 0)
        if len(uncovered_coords) > 0:
            c_row, c_col = np.mean(uncovered_coords[:, 0]), np.mean(uncovered_coords[:, 1])
            center_x = env_u.map_env.min_x + (c_col + 0.5) * env_u.map_env.grid_resolution
            center_y = env_u.map_env.min_y + (c_row + 0.5) * env_u.map_env.grid_resolution
        else:
            center_x, center_y = env_u.map_env.max_x / 2, env_u.map_env.max_y / 2
            
        try:
            ugv_paths = nx.single_source_dijkstra_path_length(env_u.G, car['edge_v'], weight='length')
        except Exception:
            ugv_paths = {}
            
        best_node, best_cost = None, float('inf')
        w1, w2, w3 = 1.0, 1.0, 0.5
        
        for node in env_u.G.nodes():
            nx_val, ny_val = float(env_u.G.nodes[node]['x']), float(env_u.G.nodes[node]['y'])
            
            dist_uav = math.hypot(nx_val - uav['x'], ny_val - uav['y'])
            t_uav = dist_uav / UAV_SPEED_CRUISE
            
            if t_uav > (uav['battery'] - 5): continue
            if node not in ugv_paths: continue
            
            dist_ugv = ugv_paths[node] + car['progress'] 
            t_ugv = dist_ugv / CAR_SPEED_CRUISE
            dist_uncovered = math.hypot(nx_val - center_x, ny_val - center_y)
            
            cost = w1 * t_uav + w2 * t_ugv + w3 * dist_uncovered
            if cost < best_cost:
                best_cost, best_node = cost, node
                
        return best_node if best_node else car['edge_v']


if __name__ == '__main__':
    print("=" * 60)
    print("  DispatcherWrapper Mock Test (Queue & Sleep)")
    print("=" * 60)
    
    map_stub = _MockMapEnv(size=300.0, grid_resolution=2.0)
    env = DispatcherWrapper(SimulationEnv(map_stub))
    
    obs, info = env.reset(seed=42)
    env.unwrapped.uavs['uav_0']['battery'] = 260
    env.unwrapped.uavs['uav_1']['battery'] = 250
    
    print(f"[Init] UAVs battery artificially set to 260 and 250 to check queue & sleep logic.")
    
    for step_idx in range(1, 450): # Extend length to allow sequential swaps
        actions = {}
        for agent in env.agents:
            act = env.action_space(agent).sample()
            if agent == 'ugv_0':
                mask = obs[agent]['action_mask']
                valid_actions = np.where(mask == 1)[0]
                if valid_actions.size > 0: act['move'] = int(np.random.choice(valid_actions))
            actions[agent] = act
            
        obs, rewards, terminations, truncations, infos = env.step(actions)
        
        if env.is_rescue_mode:
             c_x, c_y = env.unwrapped.car['x'], env.unwrapped.car['y']
             b0, b1 = env.unwrapped.uavs['uav_0']['battery'], env.unwrapped.uavs['uav_1']['battery']
             x0, y0 = env.unwrapped.uavs['uav_0']['x'], env.unwrapped.uavs['uav_0']['y']
             x1, y1 = env.unwrapped.uavs['uav_1']['x'], env.unwrapped.uavs['uav_1']['y']
             
             print(f"[{env.unwrapped.t:03d}] Q:{env.rescue_queue} | Swap:{env.current_swapping_uav}({env.swap_countdown}) | UGV:({c_x:.1f},{c_y:.1f})")
             print(f"      uav_0:({x0:.1f},{y0:.1f}) B={b0} | uav_1:({x1:.1f},{y1:.1f}) B={b1}")

        if any(terminations.values()):
            print(f"[{env.unwrapped.t:03d}] 🛑 Environment terminated (Crash or 100% covered).")
            break
            
    print("=" * 60)
    print(f"Final Tick: {env.unwrapped.t}")
