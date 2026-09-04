"""
enjoy_smdp.py
==============
SMDP-MAPPO 推理可视化脚本

加载 smdp_train.py 训练的 checkpoint，运行推理并可视化。
使用 my_method/ 的可视化器（零修改），仅替换 checkpoint 路径。

运行：
    cd experiments/smdp_mappo
    python enjoy_smdp.py
    python enjoy_smdp.py --checkpoint /path/to/custom/checkpoint

默认加载：
    ../results/smdp_mappo/macroON_maskON/checkpoints/latest/
"""

import os
import sys
import glob
import argparse
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

import numpy as np

# ── 路径注入 ──
SMDP_DIR = os.path.dirname(os.path.abspath(__file__))
MY_METHOD_DIR = os.path.join(SMDP_DIR, '..', 'my_method')
if MY_METHOD_DIR not in sys.path:
    sys.path.insert(0, MY_METHOD_DIR)
if SMDP_DIR not in sys.path:
    sys.path.insert(0, SMDP_DIR)

import ray
import torch
import torch.nn as nn
from ray import tune
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

from env_defs import RandomMapEnv, GRID_RES
from HierarchicalEnvV2 import HierarchicalEnv
from hierarchical_models import UAVSpatialCommander, UGVNodeDispatcher

# 复用 smdp_train 中的模型注册和 Policy
from smdp_train import RLlibUAVModel, RLlibUGVModel, env_creator
from smdp_policy import SMDPPPOTorchPolicy

# 复用 my_method 的可视化器
from SimulationVisualizer import SimulationVisualizer


# ================================================================
#  Checkpoint 工具
# ================================================================

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
        "请先运行 smdp_train.py 完成至少一次保存。"
    )


def _get_policy_spaces():
    env = env_creator({})
    spaces = (env.observation_space["ugv_0"], env.action_space["ugv_0"],
              env.observation_space["uav_0"], env.action_space["uav_0"])
    env.close()
    return spaces


def get_policy_id(agent_id: str) -> str:
    return "ugv_policy" if agent_id == "ugv_0" else "uav_policy"


# ================================================================
#  主入口
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SMDP-MAPPO Inference Visualization"
    )
    default_ckpt = os.path.join(
        SMDP_DIR, '..', 'results', 'smdp_mappo',
        'macroON_maskON', 'checkpoints', 'latest'
    )
    parser.add_argument(
        "--checkpoint", type=str, default=default_ckpt,
        help="Path to an RLlib checkpoint directory."
    )
    args = parser.parse_args()

    print("=" * 65)
    print("  SMDP-MAPPO 推理可视化")
    print("=" * 65)

    print("[1/3] 初始化 Ray...")
    ray.init(ignore_reinit_error=True, log_to_driver=False)

    print("[2/3] 加载 SMDP-MAPPO Checkpoint...")
    ckpt_path = get_latest_checkpoint(args.checkpoint)
    try:
        ugv_obs, ugv_act, uav_obs, uav_act = _get_policy_spaces()
        algo = Algorithm.from_checkpoint(
            ckpt_path,
            override_config={
                "multiagent": {
                    "policies": {
                        "ugv_policy": (SMDPPPOTorchPolicy, ugv_obs, ugv_act, {}),
                        "uav_policy": (SMDPPPOTorchPolicy, uav_obs, uav_act, {}),
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
        print("\n  提示: 请先运行 smdp_train.py 完成至少一次保存。")
        ray.shutdown()
        sys.exit(1)

    # 复用 enjoy_v2.py 中的可视化逻辑
    # 由于 enjoy_v2.py 的 HierarchicalV2Visualizer 较复杂且依赖 pygame，
    # 这里导入并直接使用
    try:
        sys.path.insert(0, MY_METHOD_DIR)
        from enjoy_v2 import HierarchicalV2Visualizer
        print("[3/3] 启动 Pygame 推理可视化...")
        print("=" * 65)
        vis = HierarchicalV2Visualizer(algo=algo)
        vis.run()
    except ImportError:
        # 如果 pygame 未安装，使用纯文本推理
        print("[3/3] pygame 不可用，执行文本推理...")
        print("=" * 65)
        _text_inference(algo)

    ray.shutdown()


def _text_inference(algo):
    """
    纯文本推理模式（无 pygame 依赖），
    运行 5 个 episode 并打印统计结果。
    """
    random_map = RandomMapEnv(grid_resolution=GRID_RES)
    hier_env   = HierarchicalEnv(map_env=random_map)

    for ep in range(1, 6):
        obs, _ = hier_env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            actions = {}
            for agent_id, ob in obs.items():
                policy_id = get_policy_id(agent_id)
                action = algo.compute_single_action(
                    ob, policy_id=policy_id,
                    explore=False
                )
                actions[agent_id] = action

            obs, rewards, terms, truncs, infos = hier_env.step(actions)
            total_reward += sum(rewards.values())
            steps += 1

            done = any(terms.values()) or any(truncs.values())

        cov = hier_env._compute_coverage_ratio()
        print(
            f"  Episode {ep}: steps={steps:>5}  "
            f"reward={total_reward:+8.1f}  "
            f"coverage={cov*100:.1f}%"
        )

    print("\n推理完成。")


if __name__ == "__main__":
    main()
