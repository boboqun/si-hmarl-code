"""
hierarchical_enjoy.py
=====================
分层宏观指令 MARL 模型推理可视化脚本

与 advanced_enjoy.py 的核心差异：
    ① 环境使用 HierarchicalEnv（无 DispatcherWrapper）
    ② 注册 RLlibUAVModel / RLlibUGVModel（不是 STG2ANModel）
    ③ 重写 _decode_obs_to_state：适配 FSM 状态（is_returning / is_swapping）
    ④ 动作回退掩码适配嵌套 Dict obs 结构
    ⑤ 默认 Checkpoint 目录为 checkpoints_hierarchical/

运行：
    python hierarchical_enjoy.py
    python hierarchical_enjoy.py --checkpoint ./checkpoints_hierarchical
"""

import os
import sys
import glob
import argparse
import warnings

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

# ── 自定义模块 ────────────────────────────────────────────────────
from env_defs import RandomMapEnv, GRID_RES
from HierarchicalEnv import HierarchicalEnv
from SimulationVisualizer import SimulationVisualizer
from hierarchical_models import UAVSpatialCommander, UGVNodeDispatcher


# ================================================================
#  RLlib TorchModelV2 包装器（必须在 from_checkpoint 之前重新注册）
#  与 hierarchical_train.py 中定义完全一致
# ================================================================

class RLlibUAVModel(TorchModelV2, nn.Module):
    """UAVSpatialCommander 的 RLlib 包装器。"""
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
        obs       = input_dict["obs"]
        obs_inner = obs["observation"]
        cov_grid  = obs_inner["coverage_grid"].float()
        self_st   = obs_inner["self_state"].float()
        allies    = obs_inner["allies_state"].float()
        mask      = obs["action_mask"].float()
        obs_dict  = {"coverage_grid": cov_grid, "self_state": self_st, "allies_state": allies}
        logits, value, _ = self.core(obs_dict, mask)
        self._cur_value  = value
        return logits, state

    def value_function(self):
        return self._cur_value.squeeze(-1)


class RLlibUGVModel(TorchModelV2, nn.Module):
    """UGVNodeDispatcher 的 RLlib 包装器。"""
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
        obs       = input_dict["obs"]
        obs_inner = obs["observation"]
        cov_grid  = obs_inner["coverage_grid"].float()
        self_st   = obs_inner["self_state"].float()
        allies    = obs_inner["allies_state"].float()
        mask      = obs["action_mask"].float()
        obs_dict  = {"coverage_grid": cov_grid, "self_state": self_st, "allies_state": allies}
        logits, value = self.core(obs_dict, mask)
        self._cur_value = value
        return logits, state

    def value_function(self):
        return self._cur_value.squeeze(-1)


# 必须在 Algorithm.from_checkpoint() 之前注册！
ModelCatalog.register_custom_model("RLlibUAVModel", RLlibUAVModel)
ModelCatalog.register_custom_model("RLlibUGVModel", RLlibUGVModel)


# ================================================================
#  环境注册（必须与 hierarchical_train.py 完全一致）
# ================================================================

def env_creator(cfg):
    """不包裹 DispatcherWrapper，直接使用 HierarchicalEnv。"""
    random_map = RandomMapEnv(grid_resolution=GRID_RES)
    hier_env   = HierarchicalEnv(map_env=random_map)
    return ParallelPettingZooEnv(hier_env)


tune.register_env("hierarchical_coverage_env", env_creator)


def _get_policy_spaces():
    """从当前 HierarchicalEnv 实例读取最新 obs/action 空间（与 hierarchical_train.py 保持一致）。"""
    env = env_creator({})
    ugv_obs = env.observation_space["ugv_0"]
    ugv_act = env.action_space["ugv_0"]
    uav_obs = env.observation_space["uav_0"]
    uav_act = env.action_space["uav_0"]
    env.close()
    return ugv_obs, ugv_act, uav_obs, uav_act



# ────────────────────────────────────────────────
#  工具
# ────────────────────────────────────────────────

def get_latest_checkpoint(base_dir: str) -> str:
    base_dir   = os.path.abspath(base_dir)
    candidates = sorted(glob.glob(os.path.join(base_dir, "checkpoint_*")))
    if candidates:
        latest = candidates[-1]
        print(f"[*] 找到最新 Checkpoint: {latest}")
        return latest
    rllib_json = os.path.join(base_dir, "rllib_checkpoint.json")
    if os.path.exists(rllib_json):
        print(f"[*] 使用 Checkpoint 根目录: {base_dir}")
        return base_dir
    raise FileNotFoundError(
        f"在 {base_dir} 下未找到任何 RLlib Checkpoint。\n"
        f"请确认 hierarchical_train.py 已完成至少一次 Checkpoint 保存。"
    )


def get_policy_id(agent_id: str) -> str:
    return "ugv_policy" if agent_id == "ugv_0" else "uav_policy"


# ================================================================
#  HierarchicalRLVisualizer
#  继承 SimulationVisualizer，重写：
#    __init__              ← 使用 HierarchicalEnv
#    _do_reset             ← 直接调用 HierarchicalEnv.reset()
#    _decode_obs_to_state  ← 读取 FSM 状态变量
#    run                   ← 适配嵌套 Dict obs 的动作完整采样
# ================================================================

class HierarchicalRLVisualizer(SimulationVisualizer):
    """
    HierarchicalEnv 版推理可视化器。

    核心改动：
      1. __init__: 不使用 DispatcherWrapper，直接持有 HierarchicalEnv 实例
      2. _do_reset: 重置 HierarchicalEnv（内部已有 coverage_grid 等状态）
      3. _decode_obs_to_state: 从 env.uavs / env.car 读取 FSM 字段
      4. run: 动作采样适配嵌套 obs["action_mask"]
    """

    def __init__(self, algo: Algorithm):
        self.algo = algo

        # 构造 RandomMap + HierarchicalEnv（不包裹 DispatcherWrapper）
        self._random_map = RandomMapEnv(grid_resolution=GRID_RES)
        self._hier_env   = HierarchicalEnv(map_env=self._random_map)

        # 父类 __init__(map_env=...) 会进入 SIMULATION 状态，
        # 但它会用 SimulationEnv 重建 sim，我们先初始化父类再覆盖
        super().__init__(map_env=self._random_map)

        # 覆盖父类的 sim，改为我们的 HierarchicalEnv
        self.sim     = self._hier_env
        self.map_env = self._random_map

        # ── 兼容性 Shim：SimulationVisualizer._draw_hud / _draw_coverage_grid
        #    等方法直接读 self.sim.* 属性，HierarchicalEnv 没有这些旧字段。
        #    在此动态挂载只读属性，供父类渲染器使用，无需修改 HierarchicalEnv.py。
        self._patch_sim_compat(self._hier_env)

        self._do_reset()
        self.paused = True

        print("[✓] HierarchicalRLVisualizer 初始化完成")
        print("    按 [SPACE] 开始 RL 推理演示")
        print("    按 [R]     重置为新随机地图")
        print("    按 [↑/↓]   调整帧率")
        print("    按 [D]     打印诊断信息")
        print("    按 [Q/ESC] 退出")

    # ─────────────────────────────────────────────────────────────
    #  兼容性 Shim：动态挂载父类渲染器期望的旧 Wrapper 属性
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _patch_sim_compat(env: HierarchicalEnv):
        """
        SimulationVisualizer 的若干渲染方法（_draw_hud / _draw_coverage_grid /
        _decode_obs_to_state 父类版本）直接读 self.sim.* 旧字段。
        在此为 HierarchicalEnv 实例动态挂载兼容属性，使其透明通过：

          rescue_queue          → 空列表（HierarchicalEnv 无救援队列概念）
          current_swapping_uav  → 当前正在换电的 UAV id 或 None
          swap_countdown        → 当前换电剩余 ticks
          unwrapped             → self（HierarchicalEnv 自身即展开版）
          coverage_grid         → 直接读 env.coverage_grid（同名，无需 shim）
        """
        import types

        # rescue_queue: _draw_hud L1012 用于 HUD 显示救援等待队列
        # HierarchicalEnv 没有救援队列，始终返回空列表
        type(env).rescue_queue = property(lambda self: [])

        # current_swapping_uav: 返回正在 is_swapping 的 UAV id 或 None
        def _current_swapping(self):
            for a in ['uav_0', 'uav_1']:
                if self.uavs[a]['is_swapping']:
                    return a
            return None
        type(env).current_swapping_uav = property(_current_swapping)

        # swap_countdown: 当前换电倒计时（取正在换电的 UAV 的 swap_countdown）
        def _swap_countdown(self):
            for a in ['uav_0', 'uav_1']:
                if self.uavs[a]['is_swapping']:
                    return self.uavs[a]['swap_countdown']
            return 0
        type(env).swap_countdown = property(_swap_countdown)

        # unwrapped: 父类 _decode_obs_to_state 用 self.sim.unwrapped 访问内部 env
        # HierarchicalEnv 自身就是内层 env
        type(env).unwrapped = property(lambda self: self)

    # ─────────────────────────────────────────────────────────────
    #  重置逻辑：直接调用 HierarchicalEnv
    # ─────────────────────────────────────────────────────────────

    def _do_reset(self):
        """
        重置 HierarchicalEnv，并重建父类渲染所需的所有图层缓存。
        必须镜像父类 SimulationVisualizer._do_reset() 的关键步骤：
          1. 重建坐标变换器 self.tf
          2. 调用 _pre_render_roads() 生成 _road_surface
          3. 初始化扫轨 Surface
        """
        from SimulationVisualizer import (
            CoordTransform, MAP_PADDING, DRAW_W, DRAW_H, MAP_W, MAP_H
        )

        self.obs, _ = self._hier_env.reset()
        self.state  = self._decode_obs_to_state(self.obs)
        self.done       = False
        self.reward_acc = 0.0
        self.tick_cnt   = 0
        self.paused     = True
        self._uav_trails = [[], []]

        # ── 重建坐标变换器（父类 _draw_frame / _pre_render_roads 必须有 self.tf）
        self.tf = CoordTransform(
            self._random_map.min_x, self._random_map.min_y,
            self._random_map.max_x, self._random_map.max_y,
            MAP_PADDING, MAP_PADDING,
            DRAW_W, DRAW_H,
        )

        # ── 重新渲染路网到 _road_surface（父类 _draw_frame L761 必须有）
        self._pre_render_roads()

        # ── 扫轨 Surface
        self._swath_surface = pygame.Surface((MAP_W, MAP_H), pygame.SRCALPHA)
        self._swath_surface.fill((0, 0, 0, 0))

        self.random_agent = None
        self.tester       = None

    # ─────────────────────────────────────────────────────────────
    #  FSM 状态解析（完全重写，适配 HierarchicalEnv 内部状态）
    # ─────────────────────────────────────────────────────────────

    def _decode_obs_to_state(self, obs) -> dict:
        """
        从 HierarchicalEnv 内部变量读取实体状态，
        构建符合 SimulationVisualizer._draw_frame() 期望的 state dict。

        state 格式：
          {
            't':           int,
            'coverage_pct': float,       # 0~100
            'car': {
              'x', 'y', 'status', 'mode', 'battery_stock',
              'swap_countdown', 'current_node'
            },
            'uav_1': {
              'x', 'y', 'status', 'mode', 'battery', 'swap_countdown'
            },
            'uav_2': { ... }
          }
        """
        env = self._hier_env

        # ── 覆盖率 ────────────────────────────────────────────────
        import numpy as np
        grid  = env.coverage_grid
        cov_pct = float(np.sum(grid)) / max(1, grid.size) * 100.0

        # ── UGV 状态解析 ──────────────────────────────────────────
        car = env.car

        # 判断是否有 UAV 正在换电（绑定在 UGV 旁）
        swapping_uav = None
        for a in ['uav_0', 'uav_1']:
            if env.uavs[a]['is_swapping']:
                swapping_uav = env.uavs[a]
                break

        if swapping_uav is not None:
            car_status      = 'swap_locked'
            car_mode        = 'swap_wait'
            car_swap_cd     = int(swapping_uav['swap_countdown'])
        elif car.get('target_node') and car['target_node'] != car['edge_u']:
            car_status  = 'moving'
            car_mode    = 'cruise'
            car_swap_cd = 0
        else:
            car_status  = 'idle'
            car_mode    = 'cruise'
            car_swap_cd = 0

        car_state = {
            'x':             car['x'],
            'y':             car['y'],
            'status':        car_status,
            'mode':          car_mode,
            'battery_stock': 999,         # HierarchicalEnv 无限电池存量
            'swap_countdown': car_swap_cd,
            'current_node':  car['edge_u'],
        }

        # ── UAV 状态解析（uav_0 → uav_1，uav_1 → uav_2）─────────
        uav_states = {}
        for idx, agent_key in enumerate(['uav_0', 'uav_1']):
            uav       = env.uavs[agent_key]
            vis_key   = f'uav_{idx + 1}'    # uav_1 / uav_2

            battery    = max(0, int(uav['battery']))
            swap_cd    = int(uav['swap_countdown'])

            if uav['is_swapping']:
                status  = 'swap_locked'
                mode    = 'swap'
            elif uav['is_returning']:
                status  = 'returning'
                mode    = 'cruise'
            elif uav['is_busy']:
                status  = 'working'
                mode    = 'work'
            else:
                status  = 'idle'
                mode    = 'idle'

            uav_states[vis_key] = {
                'x':            uav['x'],
                'y':            uav['y'],
                'status':       status,
                'mode':         mode,
                'battery':      battery,
                'swap_countdown': swap_cd,
            }

        return {
            't':            env.t,
            'coverage_pct': cov_pct,
            'car':          car_state,
            'uav_1':        uav_states['uav_1'],
            'uav_2':        uav_states['uav_2'],
        }

    # ─────────────────────────────────────────────────────────────
    #  主事件循环
    # ─────────────────────────────────────────────────────────────

    def run(self):
        """RL 推理主循环，适配嵌套 Dict obs 和 Discrete 动作空间。"""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self._handle_key(event.key)

            if not self.paused and not self.done and getattr(self, 'obs', None) is not None:
                actions = {}

                for agent in self._hier_env.agents:
                    obs_agent = self.obs.get(agent)
                    if obs_agent is None:
                        continue

                    try:
                        # RL 推理：HierarchicalEnv 动作为 Discrete
                        action = self.algo.compute_single_action(
                            observation=obs_agent,
                            policy_id=get_policy_id(agent),
                            explore=False,
                        )
                        if isinstance(action, tuple):
                            action = action[0]
                        actions[agent] = int(action)

                    except Exception as e:
                        print(f"[!] 推理异常 ({agent}): {e}，回退掩码随机动作")
                        # 从嵌套 Dict obs 中提取 action_mask
                        mask  = obs_agent.get("action_mask", None)
                        if mask is not None:
                            valid = np.where(np.array(mask) == 1)[0]
                            if valid.size > 0:
                                actions[agent] = int(np.random.choice(valid))
                                continue
                        # 无掩码时用默认安全动作
                        actions[agent] = 0   # UAV: noop；UGV: 节点0

                # ── 环境步进 ──────────────────────────────────────
                self.obs, rewards, terminations, truncations, infos = \
                    self._hier_env.step(actions)
                self.state      = self._decode_obs_to_state(self.obs)
                self.done = any(terminations.values()) or any(truncations.values())
                self.reward_acc += sum(rewards.values()) if rewards else 0.0
                self.tick_cnt  += 1
                self._update_trails()
                self._update_swath_surface()

                if self.done:
                    print(f"\n[Episode 结束] t={self.tick_cnt}  "
                          f"累计奖励={self.reward_acc:+.3f}  "
                          f"覆盖率={self.state['coverage_pct']:.2f}%")
                    print("    → 3 秒后自动加载新地图...")
                    pygame.time.delay(3000)
                    self._do_reset()
                    self.paused = False

            # ── 绘制 ────────────────────────────────────────────
            self._draw_frame()
            actual_w, actual_h = self.screen.get_size()
            scaled = pygame.transform.smoothscale(self._canvas, (actual_w, actual_h))
            self.screen.blit(scaled, (0, 0))

            # HUD 右上角标识（区分于其他 enjoy 脚本）
            lbl = self.font_sm_bold.render("🏗  HIER EVAL", True, (255, 200, 60))
            self.screen.blit(lbl, (actual_w - lbl.get_width() - 10, 8))

            pygame.display.flip()
            self.clock.tick(self.fps)

        pygame.quit()
        sys.exit()

    def _handle_key(self, key) -> bool:
        if key in (pygame.K_q, pygame.K_ESCAPE):
            return False
        elif key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key == pygame.K_r:
            print("\n[手动重置] 正在加载新随机地图...")
            self._do_reset()
            self.paused = False
        elif key in (pygame.K_UP, pygame.K_EQUALS, pygame.K_PLUS):
            self.fps = min(60, self.fps + 2)
        elif key in (pygame.K_DOWN, pygame.K_MINUS):
            self.fps = max(1, self.fps - 2)
        elif key == pygame.K_d:
            env = self._hier_env
            print(f"\n{'='*60}")
            print(f"  [D] 诊断  t={self.tick_cnt}  累计奖励={self.reward_acc:+.3f}")
            print(f"  覆盖率: {self.state.get('coverage_pct', 0.0):.2f}%")
            for a in ['uav_0', 'uav_1']:
                u = env.uavs[a]
                print(f"  {a}: bat={u['battery']}  busy={u['is_busy']}  "
                      f"returning={u['is_returning']}  swapping={u['is_swapping']}  "
                      f"swap_cd={u['swap_countdown']}")
            car = env.car
            print(f"  ugv_0: ({car['x']:.1f},{car['y']:.1f})  "
                  f"edge={car['edge_u']}→{car['edge_v']}  prog={car['progress']:.2f}")
            print(f"  FPS目标={self.fps}")
            print(f"{'='*60}")
        return True


# ================================================================
#  主入口
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="分层宏观指令 RL 推理可视化")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(PROJECT_DIR, "checkpoints_hierarchical"),
        help="Checkpoint 目录（默认: ./checkpoints_hierarchical）",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("  分层宏观指令 MARL RL 推理评估 (hierarchical_enjoy.py)")
    print("=" * 65)

    print("[1/3] 初始化 Ray...")
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    print("      Ray 初始化完成")

    print("[2/3] 搜索并加载 Hierarchical Checkpoint...")
    ckpt_path = get_latest_checkpoint(args.checkpoint)

    print(f"      正在加载: {ckpt_path}")
    try:
        # 覆盖策略空间为当前环境实际空间，修复旧 checkpoint 的履歴遗留不匹配问题
        ugv_obs, ugv_act, uav_obs, uav_act = _get_policy_spaces()
        algo = Algorithm.from_checkpoint(
            ckpt_path,
            override_config={
                "multiagent": {
                    "policies": {
                        "ugv_policy": (None, ugv_obs, ugv_act, {}),
                        "uav_policy": (None, uav_obs, uav_act, {}),
                    },
                    "policy_mapping_fn": lambda agent_id, *_, **__:
                        "ugv_policy" if agent_id == "ugv_0" else "uav_policy",
                    "policies_to_train": ["ugv_policy", "uav_policy"],
                },
            },
        )
        print("      [✓] Checkpoint 加载成功！")
    except Exception as e:
        print(f"      [✗] 加载失败: {e}")
        print("\n  提示: 请先运行 hierarchical_train.py 完成至少一次 Checkpoint 保存。")
        ray.shutdown()
        sys.exit(1)

    print("[3/3] 启动 Pygame 推理可视化...")
    print("=" * 65)

    vis = HierarchicalRLVisualizer(algo=algo)
    vis.run()

    ray.shutdown()


if __name__ == "__main__":
    main()
