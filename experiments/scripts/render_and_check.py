"""
render_and_check.py
===================
可视化测试脚本 (Prompt 2.4)
用于单步/连续推演验证各基线环境的物理引擎状态。
"""

import os
import sys
import argparse
import time
import math
import numpy as np
import pygame

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SYS_ROOT = os.path.abspath(os.path.join(PROJECT_DIR, "..", ".."))
if SYS_ROOT not in sys.path:
    sys.path.insert(0, SYS_ROOT)

from experiments.my_method.env_defs import RandomMapEnv, GRID_RES

# 颜色常数
C_BG = (16, 20, 32)
C_GRID_COVERED = (40, 180, 60)
C_UAV1 = (255, 165, 45)
C_UAV2 = (70, 225, 200)
C_UGV = (80, 170, 255)
C_TEXT = (255, 255, 255)

class LightweightVisualizer:
    def __init__(self, env):
        self.env = env
        pygame.init()
        self.W, self.H = 800, 800
        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption("Environment Render Check")
        self.font = pygame.font.SysFont(None, 24)
        self.clock = pygame.time.Clock()
        
        # 建立物理标度
        self.map_x0 = env.map_env.min_x
        self.map_y0 = env.map_env.min_y
        scale_x = self.W / (env.map_env.max_x - env.map_env.min_x)
        scale_y = self.H / (env.map_env.max_y - env.map_env.min_y)
        self.scale = min(scale_x, scale_y)
        self.range_y = env.map_env.max_y - env.map_env.min_y

    def to_screen(self, px, py):
        sx = int((px - self.map_x0) * self.scale)
        sy = int((self.range_y - (py - self.map_y0)) * self.scale)
        return sx, sy

    def render(self, step, rewards):
        self.screen.fill(C_BG)
        
        # 1. 渲染覆盖率网格
        grid = getattr(self.env, "coverage_grid", None)
        if grid is not None:
            rows, cols = grid.shape
            cell_w = max(1, int(self.scale * GRID_RES))
            cell_h = max(1, int(self.scale * GRID_RES))
            
            cov_rows, cov_cols = np.where(grid == 1)
            for r, c in zip(cov_rows, cov_cols):
                px = self.map_x0 + (c + 0.5) * GRID_RES
                py = self.map_y0 + (r + 0.5) * GRID_RES
                sx, sy = self.to_screen(px, py)
                pygame.draw.rect(self.screen, C_GRID_COVERED, (sx - cell_w//2, sy - cell_h//2, cell_w, cell_h))

        # 2. 渲染路网节点
        for n in self.env.map_env.G_proj.nodes:
            nd = self.env.map_env.G_proj.nodes[n]
            sx, sy = self.to_screen(nd['x'], nd['y'])
            pygame.draw.circle(self.screen, (50, 50, 80), (sx, sy), 2)

        # 3. 渲染 UAVs
        for i, a in enumerate(['uav_0', 'uav_1']):
            if a in self.env.uavs:
                u = self.env.uavs[a]
                sx, sy = self.to_screen(u['x'], u['y'])
                color = C_UAV1 if i == 0 else C_UAV2
                pygame.draw.circle(self.screen, color, (sx, sy), 6)
                
                # 绘制电量条
                b = max(0, float(u['battery'])) / 400.0  # 假设 FULL 为 400
                pygame.draw.rect(self.screen, (255,0,0), (sx-10, sy-15, 20, 4))
                pygame.draw.rect(self.screen, (0,255,0), (sx-10, sy-15, 20*b, 4))

        # 4. 渲染 UGV
        if self.env.car:
            c = self.env.car
            sx, sy = self.to_screen(c['x'], c['y'])
            pygame.draw.circle(self.screen, C_UGV, (sx, sy), 8)
            pygame.draw.circle(self.screen, (255, 255, 255), (sx, sy), 8, 1)

        # 5. 信息叠层
        total_cov = getattr(self.env, "_compute_coverage_ratio", lambda: 0.0)() * 100.0
        texts = [
            f"Step: {step}",
            f"Coverage: {total_cov:.1f}%",
            f"Reward uav0: {rewards.get('uav_0', 0):.2f}",
            f"Reward ugv0: {rewards.get('ugv_0', 0):.2f}",
        ]
        if 'uav_0' in self.env.uavs:
            texts.append(f"UAV0 Batt: {self.env.uavs['uav_0']['battery']:.1f}")
        if 'uav_1' in self.env.uavs:
            texts.append(f"UAV1 Batt: {self.env.uavs['uav_1']['battery']:.1f}")
            
        for i, t in enumerate(texts):
            surf = self.font.render(t, True, C_TEXT)
            self.screen.blit(surf, (10, 10 + i * 25))

        pygame.display.flip()
        
        # 事件处理（防止卡死）
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        
        self.clock.tick(30) # 30 FPS
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True, choices=['flat', 'standard', 'heuristic', 'my_method'])
    args = parser.parse_args()
    
    seed = 42
    map_env = RandomMapEnv(grid_resolution=GRID_RES, seed=seed)
    
    print(f"[*] 启动可视化: {args.env.upper()}")
    
    if args.env == 'flat':
        from experiments.baselines.mappo_flat.FlatEnv import FlatEnv
        env = FlatEnv(map_env)
        controller = None
    elif args.env == 'standard':
        from experiments.baselines.standard_hmarl.StandardEnv import StandardEnv
        env = StandardEnv(map_env)
        controller = None
    elif args.env == 'heuristic':
        from experiments.my_method.HierarchicalEnv import HierarchicalEnv
        from experiments.baselines.heuristic_macpp.heuristic_policy import HeuristicController
        # 绑定优先级实例重写
        env = HierarchicalEnv(map_env)
        controller = HeuristicController(env)
    elif args.env == 'my_method':
        from experiments.my_method.HierarchicalEnv import HierarchicalEnv
        env = HierarchicalEnv(map_env)
        controller = None

    obs, info = env.reset(seed=seed)
    vis = LightweightVisualizer(env)
    
    step = 0
    rewards = {}
    
    running = True
    while running:
        # 获取动作
        if controller:
            actions = controller.get_actions(obs)
        else:
            actions = {}
            for agent in env.agents:
                act_space = env.action_space(agent)
                if hasattr(act_space, 'sample'):
                    actions[agent] = act_space.sample()
                else:
                    # Flat 维度是空间 (-1, 1) 的连续值
                    actions[agent] = np.random.uniform(-1.0, 1.0, size=(2,))
        
        # 步骤推演
        obs, rewards, terminations, truncations, infos = env.step(actions)
        step += 1
        
        # 控制台按秒打印核心信息
        if step % 30 == 0:
            u_batt = [env.uavs[a]['battery'] for a in ['uav_0', 'uav_1'] if a in env.uavs]
            print(f"[Step {step}] 覆盖率={env._compute_coverage_ratio()*100:.1f}% "
                  f"电量={u_batt} "
                  f"奖励=[uav0: {rewards.get('uav_0',0):.2f}, ugv: {rewards.get('ugv_0',0):.2f}]")
        
        # GUI 渲染
        if not vis.render(step, rewards):
            print("\n[!] 窗口已关闭，停止测试。")
            break
            
        if any(terminations.values()) or any(truncations.values()):
            print(f"\n[✓] 运行完毕 (Step: {step})")
            time.sleep(1)
            break

if __name__ == "__main__":
    main()
