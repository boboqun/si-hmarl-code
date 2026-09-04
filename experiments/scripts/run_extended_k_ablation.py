#!/usr/bin/env python3
"""
run_extended_k_ablation.py
===========================
为消融实验补充等比训练预算：对指定单个 K 值训练 5000 iter（是原始 K=5 的 5 倍）。
每个 K 作为独立进程运行，K=7/K=8 可并行启动互不干扰。
每一个迭代都输出日志（flush=True 保证 nohup 后台实时写入）。

用法（分别在不同终端启动）:
  nohup .venv/bin/python experiments/scripts/run_extended_k_ablation.py --K 7 > k7_ablation.log 2>&1 &
  nohup .venv/bin/python experiments/scripts/run_extended_k_ablation.py --K 8 > k8_ablation.log 2>&1 &
"""

import os
import sys
import math
import argparse
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple
from gymnasium import spaces

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPOConfig

from experiments.my_method.env_defs import RandomMapEnv, GRID_RES, MAP_SIZE
from experiments.my_method.HierarchicalEnv import HierarchicalEnv, UGV_SPEED, DT
from experiments.my_method.hierarchical_train import RLlibUAVModel, RLlibUGVModel, ParallelPettingZooEnv
from experiments.my_method.hierarchical_models import UAVSpatialCommander, UGVNodeDispatcher, MLPBlock, UAV_SELF_DIM, UAV_ALLIES_DIM
from experiments.scripts.run_resolution_ablation import (
    DynamicHierarchicalEnv, DynamicRLlibUAVModel, DynamicRLlibUGVModel
)

NUM_ITER = 5000  # 5x 原始预算，保证公平性


def make_dynamic_uav_cls(K_val):
    class _Cls(DynamicRLlibUAVModel):
        def __init__(self, obs, act, out, conf, nm):
            super().__init__(obs, act, out, conf, nm, K=K_val)
    return _Cls


def make_dynamic_ugv_cls(K_val):
    class _Cls(DynamicRLlibUGVModel):
        def __init__(self, obs, act, out, conf, nm):
            super().__init__(obs, act, out, conf, nm, K=K_val)
    return _Cls


def run_for_k(K):
    save_dir = os.path.join(PROJECT_DIR, "experiments", "results_ablation", f"resolution_{K}x{K}_5k")
    os.makedirs(save_dir, exist_ok=True)

    env_name = f"dynamic_hier_env_k{K}_ext"
    uav_name = f"DynUAV_K{K}_ext"
    ugv_name = f"DynUGV_K{K}_ext"

    def env_creator(config):
        return ParallelPettingZooEnv(
            DynamicHierarchicalEnv(RandomMapEnv(grid_resolution=GRID_RES), resolution_K=K)
        )

    register_env(env_name, env_creator)
    ModelCatalog.register_custom_model(uav_name, make_dynamic_uav_cls(K))
    ModelCatalog.register_custom_model(ugv_name, make_dynamic_ugv_cls(K))

    sample_env = env_creator({})
    uav_obs_sp = sample_env.par_env.observation_space("uav_0")
    uav_act_sp = sample_env.par_env.action_space("uav_0")
    ugv_obs_sp = sample_env.par_env.observation_space("ugv_0")
    ugv_act_sp = sample_env.par_env.action_space("ugv_0")

    config = (
        PPOConfig()
        .environment(env_name)
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .env_runners(
            num_env_runners=1,
            num_envs_per_env_runner=2,
            rollout_fragment_length="auto"
        )
        .training(
            train_batch_size=2000,
            vf_loss_coeff=0.5,
            entropy_coeff=0.01,
        )
        .resources(num_gpus=0)
        .multi_agent(
            policies={
                "uav_policy": (None, uav_obs_sp, uav_act_sp, {"model": {"custom_model": uav_name}}),
                "ugv_policy": (None, ugv_obs_sp, ugv_act_sp, {"model": {"custom_model": ugv_name}}),
            },
            policy_mapping_fn=lambda agent_id, *a, **kw:
                "ugv_policy" if "ugv" in agent_id else "uav_policy",
        )
        .debugging(logger_config={"type": "ray.tune.logger.TBXLogger", "logdir": save_dir})
    )

    algo = config.build()
    print(f"\n{'='*60}", flush=True)
    print(f"  Extended Ablation: K={K} | Target={NUM_ITER} iters", flush=True)
    print(f"  Save dir: {save_dir}", flush=True)
    print(f"{'='*60}\n", flush=True)

    best_reward = -float('inf')
    for i in range(1, NUM_ITER + 1):
        result  = algo.train()
        rew     = result.get("episode_reward_mean", float("nan"))
        eplen   = result.get("episode_len_mean", float("nan"))
        steps   = result.get("timesteps_total", 0)

        is_best = not math.isnan(rew) and rew > best_reward
        if is_best:
            best_reward = rew
            algo.save(checkpoint_dir=f"{save_dir}/best_ckpt")

        # 每迭代都输出，★ 标注新的最优 checkpoint
        best_mark = " ★NEW BEST" if is_best else ""
        print(
            f"  [K={K}] {i:>5}/{NUM_ITER} | "
            f"rew={rew:+8.2f} | len={eplen:>7.1f} | steps={steps:>10,}{best_mark}",
            flush=True   # 关键：确保 nohup 日志实时刷新
        )

        # 每 100 iter 存一个例行 checkpoint
        if i % 100 == 0:
            algo.save(checkpoint_dir=f"{save_dir}/ckpt_{i}")
            print(f"  [K={K}] >>> Checkpoint saved at iter {i}", flush=True)

    algo.save(checkpoint_dir=f"{save_dir}/ckpt_final")
    algo.stop()
    print(f"\n  [K={K}] Training complete. Best reward: {best_reward:.2f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extended K-ablation (single K per process, parallel-friendly)"
    )
    parser.add_argument(
        "--K", type=int, required=True, choices=[4, 5, 6, 7, 8],
        help="Target resolution K to train. Launch separate processes for each K."
    )
    args = parser.parse_args()

    import ray
    ray.init(ignore_reinit_error=True, log_to_driver=False)

    run_for_k(args.K)

    ray.shutdown()
    print(f"\n[✓] K={args.K} extended ablation complete.", flush=True)
