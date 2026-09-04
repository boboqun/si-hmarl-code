"""
enjoy_v2.py
===========
SA-HMARL V2 推理可视化脚本

与 enjoy_staggered.py 的核心差异：
    ① 环境：HierarchicalEnvV2（含 fallback_return_point / 兜底返航点）
    ② 模型注册名：RLlibUAVModel_v2 / RLlibUGVModel_v2（与 V1 隔离）
    ③ Checkpoint 双轨读取：
         latest/              ← 每 10 轮覆盖更新，默认使用，快速验证效果
         milestones/iter_NNNN ← 每 100 轮独立存档，用于公平对比（指定 --checkpoint）
    ④ HUD 额外显示：
         - 兜底返航点的地图标记（黄色虚线圆圈）
         - UAV 电量紧迫度（urgency bar）
         - 实时电池相位差 (Phase Diff / PSR)
         - 每次换电事件的时间戳和电量记录

运行：
    python enjoy_v2.py                                                        # 快速验证（读 latest/）
    python enjoy_v2.py --checkpoint ../results/sa_hmarl_v2/checkpoints/milestones/iter_0500
"""

import os
import sys
import glob
import argparse
import warnings
import datetime
import csv

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

import numpy as np
import pygame

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import ray
import torch
import torch.nn as nn
from ray import tune
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

from env_defs import RandomMapEnv, GRID_RES
from HierarchicalEnvV2 import HierarchicalEnv          # ← V2 环境
from SimulationVisualizer import SimulationVisualizer
from hierarchical_models import UAVSpatialCommander, UGVNodeDispatcher


# ================================================================
#  RLlib 包装器（名称带 _v2 后缀，防止与 V1 enjoy 脚本冲突）
# ================================================================

class RLlibUAVModel_v2(TorchModelV2, nn.Module):
    """UAVSpatialCommander 包装器（V2，self_state=9维）。"""
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        cfg = model_config.get("custom_model_config", {})
        self.core = UAVSpatialCommander(
            cnn_channels=cfg.get("cnn_channels", 32),
            token_dim   =cfg.get("token_dim",    64),
            state_dim   =cfg.get("state_dim",    64),
            nhead       =cfg.get("nhead",          4),
            dropout     =cfg.get("dropout",      0.05),
        )
        self._cur_value = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        oi  = obs["observation"]
        od  = {"coverage_grid": oi["coverage_grid"].float(),
               "self_state":    oi["self_state"].float(),
               "allies_state":  oi["allies_state"].float()}
        logits, value, _ = self.core(od, obs["action_mask"].float())
        self._cur_value = value
        return logits, state

    def value_function(self):
        return self._cur_value.squeeze(-1)


class RLlibUGVModel_v2(TorchModelV2, nn.Module):
    """UGVNodeDispatcher 包装器（V2，allies_state=16维）。"""
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        cfg = model_config.get("custom_model_config", {})
        self.core = UGVNodeDispatcher(
            cnn_channels=cfg.get("cnn_channels", 32),
            hidden_dim  =cfg.get("hidden_dim",  128),
            nhead       =cfg.get("nhead",          2),
            dropout     =cfg.get("dropout",      0.05),
        )
        self._cur_value = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        oi  = obs["observation"]
        od  = {"coverage_grid": oi["coverage_grid"].float(),
               "self_state":    oi["self_state"].float(),
               "allies_state":  oi["allies_state"].float()}
        logits, value = self.core(od, obs["action_mask"].float())
        self._cur_value = value
        return logits, state

    def value_function(self):
        return self._cur_value.squeeze(-1)


ModelCatalog.register_custom_model("RLlibUAVModel_v2", RLlibUAVModel_v2)
ModelCatalog.register_custom_model("RLlibUGVModel_v2", RLlibUGVModel_v2)


# ================================================================
#  环境注册（名称与 hierarchical_train_v2.py 完全一致）
# ================================================================

def env_creator(cfg):
    return ParallelPettingZooEnv(HierarchicalEnv(RandomMapEnv(grid_resolution=GRID_RES)))

tune.register_env("hierarchical_coverage_v2_env", env_creator)


def _get_policy_spaces():
    env = env_creator({})
    spaces = (env.observation_space["ugv_0"], env.action_space["ugv_0"],
              env.observation_space["uav_0"], env.action_space["uav_0"])
    env.close()
    return spaces


def get_latest_checkpoint(base_dir: str) -> str:
    base_dir   = os.path.abspath(base_dir)
    candidates = sorted(glob.glob(os.path.join(base_dir, "checkpoint_*")))
    if candidates:
        print(f"[*] 找到最新 Checkpoint: {candidates[-1]}")
        return candidates[-1]
    rllib_json = os.path.join(base_dir, "rllib_checkpoint.json")
    if os.path.exists(rllib_json):
        print(f"[*] 使用 Checkpoint 根目录: {base_dir}")
        return base_dir
    raise FileNotFoundError(
        f"在 {base_dir} 下未找到 RLlib Checkpoint。\n"
        "请先运行 hierarchical_train_v2.py 完成至少一次保存。"
    )


def get_policy_id(agent_id: str) -> str:
    return "ugv_policy" if agent_id == "ugv_0" else "uav_policy"


# ================================================================
#  V2 可视化器
# ================================================================

# 可视化专用颜色
_COLOR_UAV0_FB  = (255, 220,  50)   # UAV_0 兜底返航点 —— 金黄
_COLOR_UAV1_FB  = (100, 220, 255)   # UAV_1 兜底返航点 —— 天蓝
_COLOR_STAGGER  = ( 80, 255, 120)   # 破局成功标记 —— 绿
_COLOR_WARN     = (255,  80,  80)   # 高紧迫度警告 —— 红


class HierarchicalV2Visualizer(SimulationVisualizer):
    """
    HierarchicalEnvV2 推理可视化器。

    在 enjoy_staggered.py 基础上新增：
      - 地图上绘制兜底返航点（虚线圆圈）
      - 诊断面板显示电量紧迫度、破局状态、换电时间戳
    """

    def __init__(self, algo: Algorithm):
        self.algo = algo
        self._random_map = RandomMapEnv(grid_resolution=GRID_RES)
        self._hier_env   = HierarchicalEnv(map_env=self._random_map)

        super().__init__(map_env=self._random_map)

        self.sim     = self._hier_env
        self.map_env = self._random_map
        self._patch_sim_compat(self._hier_env)

        # V2 专属统计
        self._swap_log      = []       # [(tick, agent, battery_at_swap)]

        # V2 日志记录 (CSV)
        self._csv_filename = f"v2_reward_monitor_{datetime.datetime.now().strftime('%m%d_%H%M%S')}.csv"
        self._csv_file = open(self._csv_filename, "w", newline='', encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "episode", "tick", "uav_0_bat", "uav_1_bat", "phase_diff", 
            "psr_val", "uav_0_state", "uav_1_state", "coverage_pct", "swap_event"
        ])
        self._ep_count = 1

        self._do_reset()
        self.paused = True

        print("[✓] HierarchicalV2Visualizer 初始化完成")
        print(f"    [!] 本地日志正在写入: {self._csv_filename}")
        print("    [SPACE] 开始/暂停  [R] 重置地图  [D] 诊断  [↑/↓] 调速  [Q/ESC] 退出")

    # ─────────────────────────────────────────────────────────────
    #  兼容性 Shim（与 enjoy_staggered 一致）
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _patch_sim_compat(env: HierarchicalEnv):
        import types
        type(env).rescue_queue = property(lambda self: [])

        def _swapping(self):
            for a in ['uav_0', 'uav_1']:
                if self.uavs[a]['is_swapping']:
                    return a
            return None
        type(env).current_swapping_uav = property(_swapping)

        def _swap_cd(self):
            for a in ['uav_0', 'uav_1']:
                if self.uavs[a]['is_swapping']:
                    return self.uavs[a]['swap_countdown']
            return 0
        type(env).swap_countdown  = property(_swap_cd)
        type(env).unwrapped       = property(lambda self: self)

    # ─────────────────────────────────────────────────────────────
    #  重置
    # ─────────────────────────────────────────────────────────────

    def _do_reset(self):
        if hasattr(self, 'tick_cnt') and self.tick_cnt > 0:
            self._ep_count += 1
            
        from SimulationVisualizer import (
            CoordTransform, MAP_PADDING, DRAW_W, DRAW_H, MAP_W, MAP_H
        )
        self.obs, _     = self._hier_env.reset()
        self.state      = self._decode_obs_to_state(self.obs)
        self.done       = False
        self.reward_acc = 0.0
        self.tick_cnt   = 0
        self.paused     = True
        self._uav_trails    = [[], []]
        self._swap_log      = []

        self.tf = CoordTransform(
            self._random_map.min_x, self._random_map.min_y,
            self._random_map.max_x, self._random_map.max_y,
            MAP_PADDING, MAP_PADDING, DRAW_W, DRAW_H,
        )
        self._pre_render_roads()
        self._swath_surface = pygame.Surface((MAP_W, MAP_H), pygame.SRCALPHA)
        self._swath_surface.fill((0, 0, 0, 0))
        self.random_agent = None
        self.tester       = None

    # ─────────────────────────────────────────────────────────────
    #  FSM 状态解析（与 enjoy_staggered 相同，额外记录换电日志）
    # ─────────────────────────────────────────────────────────────

    def _decode_obs_to_state(self, obs) -> dict:
        env    = self._hier_env
        grid   = env.coverage_grid
        cov    = float(np.sum(grid)) / max(1, grid.size) * 100.0
        car    = env.car

        # 检测本帧是否发生换电开始事件
        for a in ['uav_0', 'uav_1']:
            u = env.uavs[a]
            if u['is_swapping'] and u['swap_countdown'] == 120:
                # 刚进入换电（countdown 从 120 开始），记录日志
                entry = (self.tick_cnt, a, u['battery'])
                if not self._swap_log or self._swap_log[-1] != entry:
                    self._swap_log.append(entry)

        swapping_uav = next(
            (env.uavs[a] for a in ['uav_0', 'uav_1'] if env.uavs[a]['is_swapping']),
            None
        )
        if swapping_uav:
            car_status, car_mode, car_swap_cd = 'swap_locked', 'swap_wait', int(swapping_uav['swap_countdown'])
        elif car.get('target_node') and car['target_node'] != car['edge_u']:
            car_status, car_mode, car_swap_cd = 'moving', 'cruise', 0
        else:
            car_status, car_mode, car_swap_cd = 'idle', 'cruise', 0

        car_state = {
            'x': car['x'], 'y': car['y'],
            'status': car_status, 'mode': car_mode,
            'battery_stock': 999, 'swap_countdown': car_swap_cd,
            'current_node': car['edge_u'],
        }

        uav_states = {}
        for idx, ak in enumerate(['uav_0', 'uav_1']):
            u = env.uavs[ak]
            if u['is_swapping']:
                st, md = 'swap_locked', 'swap'
            elif u['is_returning']:
                st, md = 'returning', 'cruise'
            elif u['is_busy']:
                st, md = 'working', 'work'
            else:
                st, md = 'idle', 'idle'
            uav_states[f'uav_{idx+1}'] = {
                'x': u['x'], 'y': u['y'],
                'status': st, 'mode': md,
                'battery': max(0, int(u['battery'])),
                'swap_countdown': int(u['swap_countdown']),
            }

        return {
            't': env.t, 'coverage_pct': cov,
            'car': car_state,
            'uav_1': uav_states['uav_1'],
            'uav_2': uav_states['uav_2'],
        }

    # ─────────────────────────────────────────────────────────────
    #  V2 专属绘制：兜底返航点标记
    # ─────────────────────────────────────────────────────────────

    def _draw_fallback_points(self):
        """在地图上绘制两架 UAV 的兜底返航点（虚线圆圈 + 文字标注）。"""
        env    = self._hier_env
        canvas = self._canvas

        colors = {'uav_0': _COLOR_UAV0_FB, 'uav_1': _COLOR_UAV1_FB}
        labels = {'uav_0': 'FB-0', 'uav_1': 'FB-1'}

        for ak, color in colors.items():
            u   = env.uavs[ak]
            fbp = u.get('fallback_return_point')

            # 扫图中且已计算兜底点 → 固定预测坐标
            # 返航中 → UAV 当前位置（已在路上）
            if fbp is not None:
                fx, fy = fbp
            elif u['is_returning']:
                fx, fy = u['x'], u['y']
            else:
                continue   # 空闲/换电中，无需标记

            # 地图坐标 → 屏幕坐标
            sx, sy = self.tf.to_screen(fx, fy)

            # 画虚线圆圈（用 8 段小弧近似）
            import math
            R = 18  # 屏幕像素半径
            n_seg = 16
            pts = [
                (int(sx + R * math.cos(2*math.pi*i/n_seg)),
                 int(sy - R * math.sin(2*math.pi*i/n_seg)))
                for i in range(n_seg)
            ]
            for i in range(0, n_seg, 2):   # 每隔一段跳过 → 虚线效果
                pygame.draw.line(canvas, color, pts[i], pts[(i+1) % n_seg], 2)

            # 中心十字
            pygame.draw.line(canvas, color, (sx-6, sy), (sx+6, sy), 2)
            pygame.draw.line(canvas, color, (sx, sy-6), (sx, sy+6), 2)

            # 文字标注
            lbl = self.font_sm.render(labels[ak], True, color)
            canvas.blit(lbl, (sx + 10, sy - 8))

            # 如果是"已在返航"，加"→"箭头指示从 UAV 到 fallback 的含义
            if fbp is None and u['is_returning']:
                note = self.font_sm.render("(RTH)", True, color)
                canvas.blit(note, (sx + 10, sy + 6))

    # ─────────────────────────────────────────────────────────────
    #  V2 专属 HUD：右侧诊断面板
    # ─────────────────────────────────────────────────────────────

    def _draw_v2_panel(self):
        """在右下角专门绘制 V2 面板，覆盖原有的操作快捷键说明，绝对不遮挡左侧地图！"""
        from SimulationVisualizer import HUD_X, HUD_W, CANVAS_H, SS_SCALE, C_HUD_BG
        env    = self._hier_env
        canvas = self._canvas
        UAV_FULL = 900

        S = SS_SCALE
        
        # 在右下角画一个很大的背景块，把默认的 "SPACE 暂停" 等控制说明彻底抹杀
        clear_h = 210 * S
        clear_y = CANVAS_H - clear_h
        pygame.draw.rect(canvas, C_HUD_BG, (HUD_X, clear_y, HUD_W, clear_h))
        
        # 加上一条分割线好看点
        from SimulationVisualizer import C_HUD_LINE
        pygame.draw.line(canvas, C_HUD_LINE, (HUD_X + S*8, clear_y), (HUD_X + HUD_W - S*8, clear_y), S)

        panel_x = HUD_X + 18 * S
        y       = clear_y + 12 * S

        def txt(s, color=(200, 200, 200), bold=False, line_spacing=18):
            nonlocal y
            f   = self.font_sm_bold if bold else self.font_sm
            lbl = f.render(s, True, color)
            canvas.blit(lbl, (panel_x, y))
            y += line_spacing * S

        txt("── V2 错峰换电监控 ──", (255, 200, 60), bold=True, line_spacing=22)

        # 1. 错峰状态一句话显示
        if 'uav_0' in env.uavs and 'uav_1' in env.uavs:
            b0 = max(0, env.uavs['uav_0']['battery'])
            b1 = max(0, env.uavs['uav_1']['battery'])
            diff = abs(b0 - b1) / UAV_FULL
            if diff > 0.15:
                psr = max(0.0, diff - 0.10) * 8.0
                txt(f"奖励激活 🔄 错峰率 {diff*100:.1f}% (+{psr:.2f} PSR)", _COLOR_STAGGER, bold=True)
            else:
                txt(f"同步警告 ⚠️ 错峰率仅 {diff*100:.1f}%", _COLOR_WARN)
        y += 4 * S

        # 2. 当前电量与兜底目标紧凑版
        for ak, lbl, col in [('uav_0', 'U0', _COLOR_UAV0_FB), ('uav_1', 'U1', _COLOR_UAV1_FB)]:
            u = env.uavs[ak]
            pct = int(max(0, u['battery']) / UAV_FULL * 100)
            fbp = u.get('fallback_return_point')
            if fbp:
                st = f"FB ({fbp[0]:.0f}, {fbp[1]:.0f})"
            else:
                st = "RTH" if u['is_returning'] else ("SWAP" if u['is_swapping'] else "IDLE")
            
            txt(f"{lbl} : {pct}% bat  |  {st}", col, line_spacing=18)
            
        y += 8 * S
        txt("── 提早换电记录 ──", (255, 200, 60), bold=True, line_spacing=20)
        
        if not self._swap_log:
            txt("  (暂无记录)", (120, 120, 120))
        else:
            # 只显示倒数 3 条最紧凑的，免得溢出
            for tick, agent, bat in self._swap_log[-3:]:
                pct = int(bat / UAV_FULL * 100)
                col = _COLOR_STAGGER if pct >= 38 else (160, 160, 160)
                tag = "(★提早)" if pct >= 38 else ""
                txt(f"  t={tick:>4} {agent} 余电:{pct}% {tag}", col, line_spacing=16)

    # ─────────────────────────────────────────────────────────────
    #  主事件循环
    # ─────────────────────────────────────────────────────────────

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self._handle_key(event.key)

            if not self.paused and not self.done and self.obs is not None:
                actions = {}
                for agent in self._hier_env.agents:
                    obs_a = self.obs.get(agent)
                    if obs_a is None:
                        continue
                    try:
                        action = self.algo.compute_single_action(
                            observation=obs_a,
                            policy_id=get_policy_id(agent),
                            explore=False,
                        )
                        actions[agent] = int(action[0] if isinstance(action, tuple) else action)
                    except Exception as e:
                        mask  = obs_a.get("action_mask")
                        valid = np.where(np.array(mask) == 1)[0] if mask is not None else []
                        actions[agent] = int(np.random.choice(valid)) if valid.size > 0 else 0

                self.obs, rewards, terminations, truncations, _ = \
                    self._hier_env.step(actions)
                self.state      = self._decode_obs_to_state(self.obs)
                self.done       = any(terminations.values()) or any(truncations.values())
                self.reward_acc += sum(rewards.values()) if rewards else 0.0
                self.tick_cnt   += 1
                self._update_trails()
                self._update_swath_surface()
                
                # 记录这帧的数据到 CSV
                self._record_csv_tick()

                if self.done:
                    print(f"\n[Episode 结束] t={self.tick_cnt}  奖励={self.reward_acc:+.3f}  "
                          f"覆盖率={self.state['coverage_pct']:.2f}%")
                    if self._swap_log:
                        print("  换电记录:")
                        for tk, ag, bat in self._swap_log:
                            pct = int(bat / 900 * 100)
                            tag = " ← 提早返航!" if pct >= 38 else ""
                            print(f"    t={tk:>4}  {ag}  bat={pct}%{tag}")
                    pygame.time.delay(3000)
                    self._do_reset()
                    self.paused = False

            # ── 绘制 ────────────────────────────────────────────────
            self._draw_frame()                # 父类基础绘制
            self._draw_fallback_points()      # V2 新增：兜底返航点标记
            self._draw_v2_panel()             # V2 新增：右侧奖励监控面板

            actual_w, actual_h = self.screen.get_size()
            self.screen.blit(
                pygame.transform.smoothscale(self._canvas, (actual_w, actual_h)), (0, 0)
            )

            # 右上角版本标识
            lbl = self.font_sm_bold.render("🚁  SA-HMARL V2", True, (100, 220, 255))
            self.screen.blit(lbl, (actual_w - lbl.get_width() - 10, 8))

            pygame.display.flip()
            self.clock.tick(self.fps)

        if hasattr(self, '_csv_file'):
            self._csv_file.close()
            
        pygame.quit()
        sys.exit()

    def _record_csv_tick(self):
        """将当前帧的数据写入 CSV 文件"""
        env = self._hier_env
        if 'uav_0' not in env.uavs or 'uav_1' not in env.uavs:
            return
            
        b0 = max(0, env.uavs['uav_0']['battery'])
        b1 = max(0, env.uavs['uav_1']['battery'])
        diff = abs(b0 - b1) / 900.0
        psr = max(0.0, diff - 0.10) * 8.0 if diff > 0.15 else 0.0
        
        st0 = "RTH" if env.uavs['uav_0']['is_returning'] else ("SWAP" if env.uavs['uav_0']['is_swapping'] else "IDLE/WORK")
        st1 = "RTH" if env.uavs['uav_1']['is_returning'] else ("SWAP" if env.uavs['uav_1']['is_swapping'] else "IDLE/WORK")
        
        # 寻找这一帧有没有刚触发换电的事件 (通过对比 tick 和 swap_log 尾部)
        swap_evt = ""
        if self._swap_log and self._swap_log[-1][0] == self.tick_cnt:
            swap_evt = f"{self._swap_log[-1][1]}_bat{int(self._swap_log[-1][2])}"
            
        cov_pct = round(self.state.get('coverage_pct', 0), 2)
        
        self._csv_writer.writerow([
            self._ep_count, self.tick_cnt, b0, b1, round(diff, 4), round(psr, 4),
            st0, st1, cov_pct, swap_evt
        ])
        
        # 每隔 10 帧刷入磁盘一次，防异常退出丢失数据
        if self.tick_cnt % 10 == 0:
            self._csv_file.flush()

    def _handle_key(self, key) -> bool:
        if key in (pygame.K_q, pygame.K_ESCAPE):
            return False
        elif key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key == pygame.K_r:
            print("\n[手动重置] 加载新随机地图...")
            self._do_reset()
            self.paused = False
        elif key in (pygame.K_UP, pygame.K_EQUALS, pygame.K_PLUS):
            self.fps = min(60, self.fps + 2)
        elif key in (pygame.K_DOWN, pygame.K_MINUS):
            self.fps = max(1, self.fps - 2)
        elif key == pygame.K_d:
            env = self._hier_env
            print(f"\n{'='*65}")
            print(f"  [D 诊断]  t={self.tick_cnt}  奖励={self.reward_acc:+.3f}  "
                  f"FPS={self.fps}")
            print(f"  覆盖率:  {self.state.get('coverage_pct', 0):.2f}%")
            if 'uav_0' in env.uavs and 'uav_1' in env.uavs:
                pd = abs(env.uavs['uav_0']['battery'] - env.uavs['uav_1']['battery']) / 900
                print(f"  当前错峰率: {pd*100:.1f}% (PSR>0 if >10%)")
            for a in ['uav_0', 'uav_1']:
                u   = env.uavs[a]
                pct = int(u['battery'] / 900 * 100)
                fbp = u.get('fallback_return_point')
                fb_str = f"({fbp[0]:.0f},{fbp[1]:.0f})" if fbp else \
                         "RTH-tracking" if u['is_returning'] else "None"
                print(f"  {a}: bat={pct}%  busy={u['is_busy']}  "
                      f"returning={u['is_returning']}  swapping={u['is_swapping']}  "
                      f"fallback={fb_str}")
            car = env.car
            print(f"  ugv_0:  ({car['x']:.1f},{car['y']:.1f})  "
                  f"edge={car['edge_u']}→{car['edge_v']}")
            print(f"  换电记录 ({len(self._swap_log)} 次):")
            for tk, ag, bat in self._swap_log:
                pct = int(bat / 900 * 100)
                tag = " ← 提早返航!" if pct >= 38 else ""
                print(f"    t={tk:>4}  {ag}  bat={pct}%{tag}")
            print(f"{'='*65}")
        return True


# ================================================================
#  主入口
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="SA-HMARL V2 推理可视化")
    default_ckpt = os.path.join(
        PROJECT_DIR, "..", "results", "sa_hmarl_v2", "checkpoints", "latest"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=default_ckpt,
        help="Checkpoint 目录\n"
             "  快速验证: .../checkpoints/latest/         （默认，覆盖式，每10轮更新）\n"
             "  公平对比: .../checkpoints/milestones/iter_0500/  （固定存档，不覆盖）",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("  SA-HMARL V2 推理可视化  ——  错峰换电 + UGV预定位版")
    print("=" * 65)

    print("[1/3] 初始化 Ray...")
    ray.init(ignore_reinit_error=True, log_to_driver=False)

    print("[2/3] 加载 V2 Checkpoint...")
    ckpt_path = get_latest_checkpoint(args.checkpoint)
    try:
        ugv_obs, ugv_act, uav_obs, uav_act = _get_policy_spaces()
        algo = Algorithm.from_checkpoint(
            ckpt_path,
            override_config={
                "multiagent": {
                    "policies": {
                        "ugv_policy": (None, ugv_obs, ugv_act, {}),
                        "uav_policy": (None, uav_obs, uav_act, {}),
                    },
                    "policy_mapping_fn": lambda aid, *_, **__:
                        "ugv_policy" if aid == "ugv_0" else "uav_policy",
                    "policies_to_train": ["ugv_policy", "uav_policy"],
                },
            },
        )
        print("      [✓] Checkpoint 加载成功！")
    except Exception as e:
        print(f"      [✗] 加载失败: {e}")
        print("\n  提示: 请先运行 hierarchical_train_v2.py 完成至少一次保存。")
        ray.shutdown()
        sys.exit(1)

    print("[3/3] 启动 Pygame 推理可视化...")
    print("=" * 65)

    vis = HierarchicalV2Visualizer(algo=algo)
    vis.run()
    ray.shutdown()


if __name__ == "__main__":
    main()
