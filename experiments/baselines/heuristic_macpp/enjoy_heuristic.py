#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
enjoy_heuristic.py
==================
Heuristic MACPP 基线的交互式可视化演示脚本。
与 experiments/scripts/enjoy.py 结构完全对齐，但不需要 Ray / RLlib，
使用 HeuristicController 驱动 HierarchicalEnv 进行物理推演。

用法:
    python enjoy_heuristic.py [--seed SEED] [--fps FPS]

快捷键:
    SPACE  — 开始 / 暂停
    R      — 重置为新随机地图（按当前 seed+1 累进）
    ↑ / ↓  — 加速 / 减速
    Q/ESC  — 退出
"""

import os
import sys
import argparse
import numpy as np
import pygame

# ── 路径注册 ──────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SYS_ROOT    = os.path.abspath(os.path.join(PROJECT_DIR, "..", "..", ".."))
for p in (SYS_ROOT, PROJECT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments.my_method.env_defs       import RandomMapEnv, GRID_RES
from experiments.my_method.HierarchicalEnv import HierarchicalEnv
from experiments.my_method.SimulationVisualizer import (
    SimulationVisualizer, CoordTransform,
    MAP_PADDING, DRAW_W, DRAW_H, MAP_W, MAP_H,
)
from heuristic_policy import HeuristicController


# ================================================================
#  HeuristicVisualizer
# ================================================================
class HeuristicVisualizer(SimulationVisualizer):
    """
    SimulationVisualizer 子类：用 HeuristicController 代替 RL 策略网络。
    渲染管线（坐标变换、扫幅绘制、轨迹、HUD）与 enjoy.py 完全一致。
    """

    def __init__(self, seed: int = 42, fps: int = 30):
        self._seed = seed
        self._fps_init = fps

        # 先建一次地图，让父类完成 pygame 初始化
        map_env = RandomMapEnv(grid_resolution=GRID_RES, seed=self._seed)
        super().__init__(map_env=map_env)

        self._do_reset()
        self.paused = True

        print("[✓] Heuristic MACPP 可视化器初始化完成")
        print("    按 [SPACE] 开始演示")
        print("    按 [R]     重置（下一个 seed）")
        print("    按 [↑/↓]   调整帧率")
        print("    按 [Q/ESC] 退出")

    # ------------------------------------------------------------------
    #  环境 / 控制器初始化
    # ------------------------------------------------------------------
    def _build_env(self):
        """根据当前 seed 构建新的环境和控制器。"""
        map_env      = RandomMapEnv(grid_resolution=GRID_RES, seed=self._seed)
        # use_resume_scan=False：Heuristic 不应拥有 SA-HMARL 独有的断点续扫能力，
        # 对整块重新规划航点，确保对照实验公平性。
        env          = HierarchicalEnv(map_env, use_resume_scan=False)
        _patch_sim_compat(env)
        controller   = HeuristicController(env)
        return map_env, env, controller

    def _do_reset(self):
        """重置：构建新环境，重置可视化状态。"""
        self._random_map, self._env, self._controller = self._build_env()

        # 让父类的 sim 指向新环境（SimulationVisualizer 的绘图方法依赖 self.sim）
        self.sim = self._env
        self.map_env = self._random_map

        self.obs, _ = self._env.reset(seed=self._seed)
        self.state  = self._decode_obs_to_state()

        self.done       = False
        self.reward_acc = 0.0
        self.tick_cnt   = 0
        self.paused     = True
        self._uav_trails = [[], []]

        self.fps = self._fps_init

        # 坐标变换（地图物理坐标 → 屏幕像素）
        self.tf = CoordTransform(
            self._random_map.min_x, self._random_map.min_y,
            self._random_map.max_x, self._random_map.max_y,
            MAP_PADDING, MAP_PADDING,
            DRAW_W, DRAW_H,
        )
        self._pre_render_roads()

        # 透明扫幅层
        self._swath_surface = pygame.Surface((MAP_W, MAP_H), pygame.SRCALPHA)
        self._swath_surface.fill((0, 0, 0, 0))

    # ------------------------------------------------------------------
    #  状态解码（与 enjoy.py 完全对齐）
    # ------------------------------------------------------------------
    def _decode_obs_to_state(self) -> dict:
        env = self._env
        grid    = env.coverage_grid
        cov_pct = float(np.sum(grid)) / max(1, grid.size) * 100.0

        car          = env.car
        swapping_uav = None
        for a in ['uav_0', 'uav_1']:
            if env.uavs[a]['is_swapping']:
                swapping_uav = env.uavs[a]
                break

        if swapping_uav is not None:
            car_status, car_mode, swap_cd = 'swap_locked', 'swap_wait', int(swapping_uav['swap_countdown'])
        elif car.get('target_node') and car['target_node'] != car['edge_u']:
            car_status, car_mode, swap_cd = 'moving', 'cruise', 0
        else:
            car_status, car_mode, swap_cd = 'idle', 'cruise', 0

        car_state = {
            'x': car['x'], 'y': car['y'],
            'status': car_status, 'mode': car_mode,
            'battery_stock': 999, 'swap_countdown': swap_cd,
            'current_node': car['edge_u'],
        }

        uav_states = {}
        for idx, key in enumerate(['uav_0', 'uav_1']):
            uav = env.uavs[key]
            vis_key = f'uav_{idx + 1}'
            battery = max(0, int(uav['battery']))
            cd      = int(uav['swap_countdown'])

            if uav['is_swapping']:    status, mode = 'swap_locked', 'swap'
            elif uav['is_returning']: status, mode = 'returning',   'cruise'
            elif uav['is_busy']:      status, mode = 'working',     'work'
            else:                     status, mode = 'idle',        'idle'

            uav_states[vis_key] = {
                'x': uav['x'], 'y': uav['y'],
                'status': status, 'mode': mode,
                'battery': battery, 'swap_countdown': cd,
            }

        return {
            't': env.t,
            'coverage_pct': cov_pct,
            'car':   car_state,
            'uav_1': uav_states['uav_1'],
            'uav_2': uav_states['uav_2'],
        }

    # ------------------------------------------------------------------
    #  主循环
    # ------------------------------------------------------------------
    def run(self):
        running = True
        while running:
            # 事件处理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self._handle_key(event.key)

            # 推演一步
            if not self.paused and not self.done:
                actions = self._controller.get_actions(self.obs)

                self.obs, rewards, terminations, truncations, infos = \
                    self._env.step(actions)

                self.state      = self._decode_obs_to_state()
                self.done       = any(terminations.values()) or any(truncations.values())
                self.reward_acc += sum(rewards.values()) if rewards else 0.0
                self.tick_cnt   += 1

                self._update_trails()
                self._update_swath_surface()

                if self.done:
                    cov = self.state['coverage_pct']
                    print(f"\n[Episode 结束]  seed={self._seed}  "
                          f"t={self.tick_cnt}  覆盖率={cov:.2f}%  "
                          f"累计奖励={self.reward_acc:+.3f}")
                    print("    → 3 秒后自动加载下一个 seed...")
                    pygame.time.delay(3000)
                    self._seed += 1
                    self._do_reset()
                    self.paused = False

            # 绘制
            self._draw_frame()
            actual_w, actual_h = self.screen.get_size()
            scaled = pygame.transform.smoothscale(self._canvas, (actual_w, actual_h))
            self.screen.blit(scaled, (0, 0))

            # 右上角算法标签
            lbl = self.font_sm_bold.render(
                f"HEURISTIC MACPP  seed={self._seed}", True, (100, 230, 120))
            self.screen.blit(lbl, (actual_w - lbl.get_width() - 10, 8))

            pygame.display.flip()
            self.clock.tick(self.fps)

        pygame.quit()
        sys.exit()

    # ------------------------------------------------------------------
    #  键盘处理
    # ------------------------------------------------------------------
    def _handle_key(self, key) -> bool:
        if key in (pygame.K_q, pygame.K_ESCAPE):
            return False
        elif key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key == pygame.K_r:
            self._seed += 1
            self._do_reset()
            self.paused = False
        elif key in (pygame.K_UP, pygame.K_EQUALS, pygame.K_PLUS):
            self.fps = min(120, self.fps + 10)
        elif key in (pygame.K_DOWN, pygame.K_MINUS):
            self.fps = max(1, self.fps - 10)
        return True


# ================================================================
#  SimulationVisualizer 兼容性补丁（与 enjoy.py 保持一致）
# ================================================================
def _patch_sim_compat(env: HierarchicalEnv):
    """为 SimulationVisualizer 所需的额外接口打补丁。"""
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


# ================================================================
#  入口
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="Heuristic MACPP 交互式可视化")
    parser.add_argument("--seed", type=int, default=42,
                        help="初始随机地图种子 (默认: 42)")
    parser.add_argument("--fps",  type=int, default=30,
                        help="初始渲染帧率 (默认: 30)")
    args = parser.parse_args()

    vis = HeuristicVisualizer(seed=args.seed, fps=args.fps)
    vis.run()


if __name__ == "__main__":
    main()
