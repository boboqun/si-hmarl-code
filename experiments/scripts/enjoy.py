#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
enjoy.py
=====================
可视化与查看由于最新策略调整导致的 SA-HMARL 行驶路径规划全貌
直接适配位于 experiments 目录下的模型注册和最新代码结构。
"""
import os
import sys
import numpy as np
import pygame
import ray
import argparse
from ray.rllib.algorithms.algorithm import Algorithm

# 添加跨级挂载目录
SYS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if SYS_ROOT not in sys.path:
    sys.path.insert(0, SYS_ROOT)

from experiments.scripts.evaluate_performance import PerformanceEvaluator, env_creator_my_method
from experiments.my_method.SimulationVisualizer import SimulationVisualizer
from experiments.my_method.HierarchicalEnv import HierarchicalEnv

# ================================================================
#  HierarchicalRLVisualizer
# ================================================================
class HierarchicalRLVisualizer(SimulationVisualizer):
    def __init__(self, algo: Algorithm, env):
        self.algo = algo
        
        # 逐级解包获取真实的 HierarchicalEnv
        base_env = env
        while hasattr(base_env, "unwrapped") and base_env.unwrapped != base_env:
            base_env = base_env.unwrapped
        if hasattr(base_env, "env"): base_env = getattr(base_env, "env")
        if hasattr(base_env, "par_env"): base_env = getattr(base_env, "par_env")
            
        self._hier_env = base_env
        self._random_map = self._hier_env.map_env

        super().__init__(map_env=self._random_map)

        self.sim = self._hier_env
        self._patch_sim_compat(self._hier_env)

        self._do_reset()
        self.paused = True

        print("[✓] 可视化器初始化完成")
        print("    按 [SPACE] 开始 RL 推理演示")
        print("    按 [R]     重置为新随机地图")
        print("    按 [↑/↓]   调整帧率")
        print("    按 [Q/ESC] 退出")

    @staticmethod
    def _patch_sim_compat(env: HierarchicalEnv):
        type(env).rescue_queue = property(lambda self: [])
        def _current_swapping(self):
            for a in ['uav_0', 'uav_1']:
                if self.uavs[a]['is_swapping']:
                    return a
            return None
        type(env).current_swapping_uav = property(_current_swapping)

        def _swap_countdown(self):
            for a in ['uav_0', 'uav_1']:
                if self.uavs[a]['is_swapping']:
                    return self.uavs[a]['swap_countdown']
            return 0
        type(env).swap_countdown = property(_swap_countdown)

        type(env).unwrapped = property(lambda self: self)

    def _do_reset(self):
        from experiments.my_method.SimulationVisualizer import (
            CoordTransform, MAP_PADDING, DRAW_W, DRAW_H, MAP_W, MAP_H
        )

        self.obs, _ = self._hier_env.reset()
        self.state  = self._decode_obs_to_state(self.obs)
        self.done       = False
        self.reward_acc = 0.0
        self.tick_cnt   = 0
        self.paused     = True
        self._uav_trails = [[], []]

        self.tf = CoordTransform(
            self._random_map.min_x, self._random_map.min_y,
            self._random_map.max_x, self._random_map.max_y,
            MAP_PADDING, MAP_PADDING,
            DRAW_W, DRAW_H,
        )

        self._pre_render_roads()
        self._swath_surface = pygame.Surface((MAP_W, MAP_H), pygame.SRCALPHA)
        self._swath_surface.fill((0, 0, 0, 0))

    def _decode_obs_to_state(self, obs) -> dict:
        env = self._hier_env
        grid = env.coverage_grid
        cov_pct = float(np.sum(grid)) / max(1, grid.size) * 100.0

        car = env.car
        swapping_uav = None
        for a in ['uav_0', 'uav_1']:
            if env.uavs[a]['is_swapping']:
                swapping_uav = env.uavs[a]
                break

        if swapping_uav is not None:
            car_status, car_mode, car_swap_cd = 'swap_locked', 'swap_wait', int(swapping_uav['swap_countdown'])
        elif car.get('target_node') and car['target_node'] != car['edge_u']:
            car_status, car_mode, car_swap_cd = 'moving', 'cruise', 0
        else:
            car_status, car_mode, car_swap_cd = 'idle', 'cruise', 0

        car_state = {
            'x': car['x'], 'y': car['y'], 'status': car_status, 'mode': car_mode,
            'battery_stock': 999, 'swap_countdown': car_swap_cd, 'current_node': car['edge_u'],
        }

        uav_states = {}
        for idx, agent_key in enumerate(['uav_0', 'uav_1']):
            uav = env.uavs[agent_key]
            vis_key = f'uav_{idx + 1}'
            battery = max(0, int(uav['battery']))
            swap_cd = int(uav['swap_countdown'])

            if uav['is_swapping']:        status, mode = 'swap_locked', 'swap'
            elif uav['is_returning']:     status, mode = 'returning', 'cruise'
            elif uav['is_busy']:          status, mode = 'working', 'work'
            else:                         status, mode = 'idle', 'idle'

            uav_states[vis_key] = {'x': uav['x'], 'y': uav['y'], 'status': status, 'mode': mode, 'battery': battery, 'swap_countdown': swap_cd}

        return {'t': env.t, 'coverage_pct': cov_pct, 'car': car_state, 'uav_1': uav_states['uav_1'], 'uav_2': uav_states['uav_2']}

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                elif event.type == pygame.KEYDOWN: running = self._handle_key(event.key)

            if not self.paused and not self.done and getattr(self, 'obs', None) is not None:
                actions = {}
                for agent in self._hier_env.agents:
                    obs_agent = self.obs.get(agent)
                    if obs_agent is None: continue
                    try:
                        pol_id = "ugv_policy" if "ugv" in agent else "uav_policy"
                        action = self.algo.compute_single_action(observation=obs_agent, policy_id=pol_id, explore=False)
                        actions[agent] = int(action[0]) if isinstance(action, tuple) else int(action)
                    except Exception as e:
                        actions[agent] = 0

                self.obs, rewards, terminations, truncations, infos = self._hier_env.step(actions)
                self.state = self._decode_obs_to_state(self.obs)
                self.done = any(terminations.values()) or any(truncations.values())
                self.reward_acc += sum(rewards.values()) if rewards else 0.0
                self.tick_cnt += 1
                self._update_trails()
                self._update_swath_surface()

                if self.done:
                    print(f"\n[Episode 结束] t={self.tick_cnt} 累计奖励={self.reward_acc:+.3f} 覆盖率={self.state['coverage_pct']:.2f}%")
                    print("    → 3 秒后自动加载新地图...")
                    pygame.time.delay(3000)
                    self._do_reset()
                    self.paused = False

            self._draw_frame()
            actual_w, actual_h = self.screen.get_size()
            scaled = pygame.transform.smoothscale(self._canvas, (actual_w, actual_h))
            self.screen.blit(scaled, (0, 0))

            lbl = self.font_sm_bold.render("OUR SA-HMARL", True, (255, 200, 60))
            self.screen.blit(lbl, (actual_w - lbl.get_width() - 10, 8))
            pygame.display.flip()
            self.clock.tick(self.fps)

        pygame.quit()
        sys.exit()

    def _handle_key(self, key) -> bool:
        if key in (pygame.K_q, pygame.K_ESCAPE): return False
        elif key == pygame.K_SPACE: self.paused = not self.paused
        elif key == pygame.K_r:
            self._do_reset()
            self.paused = False
        elif key in (pygame.K_UP, pygame.K_EQUALS, pygame.K_PLUS): self.fps = min(120, self.fps + 10)
        elif key in (pygame.K_DOWN, pygame.K_MINUS): self.fps = max(1, self.fps - 10)
        return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=os.path.join(SYS_ROOT, "experiments/results/my_method/checkpoints"))
    args = parser.parse_args()

    evaluator = PerformanceEvaluator()
    evaluator.init_ray_if_needed()

    print(f"正在加载最优权重: {args.checkpoint}")
    algo = Algorithm.from_checkpoint(args.checkpoint)
    
    # 构建兼容环境
    env = env_creator_my_method({})
    vis = HierarchicalRLVisualizer(algo=algo, env=env)
    vis.run()

if __name__ == "__main__":
    main()
