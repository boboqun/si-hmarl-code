"""
train_flat.py
=============
MAPPO 扁平架构连续动作基线训练脚本。
彻底暴露“大尺度农田 + 逐秒微控”带来的维度灾难和非平稳性。
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("RAY_ENABLE_MAC_LARGE_OBJECT_STORE", "1")
os.environ.setdefault("RAY_memory_usage_threshold", "1.0")

import ray
import numpy as np
import torch
import torch.nn as nn

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# Ensure root dir is also accessible for common env definitions
SYS_ROOT = os.path.abspath(os.path.join(PROJECT_DIR, "..", "..", ".."))
if SYS_ROOT not in sys.path:
    sys.path.insert(0, SYS_ROOT)

MY_METHOD_DIR = os.path.abspath(os.path.join(PROJECT_DIR, "..", "..", "my_method"))
if MY_METHOD_DIR not in sys.path:
    sys.path.insert(0, MY_METHOD_DIR)

from experiments.my_method.env_defs import RandomMapEnv, MAP_SIZE, MAX_NODES, GRID_RES, GRID_ROWS, GRID_COLS
from FlatEnv import FlatEnv
from flat_models import RLlibFlatUAVModel, RLlibFlatUGVModel


NUM_ITER        = int(os.environ.get("NUM_ITER",        2000))
CHECKPOINT_FREQ = int(os.environ.get("CHECKPOINT_FREQ",  10))
CHECKPOINT_DIR  = os.environ.get("CHECKPOINT_DIR", os.path.join(PROJECT_DIR, "checkpoints_flat"))

TRAIN_BATCH_SIZE = 8000
MINIBATCH_SIZE   = 512
NUM_SGD_ITER     = 5
NUM_WORKERS      = 6   # Slightly lower since FlatEnv generates 10x more steps due to high freq
GAMMA            = 0.99
LR               = 3e-4

# checkpoint 双轨策略
LATEST_DIR    = os.path.join(CHECKPOINT_DIR, "latest")
MILESTONE_DIR = os.path.join(CHECKPOINT_DIR, "milestones")
os.makedirs(LATEST_DIR,    exist_ok=True)
os.makedirs(MILESTONE_DIR, exist_ok=True)

ModelCatalog.register_custom_model("RLlibFlatUAVModel", RLlibFlatUAVModel)
ModelCatalog.register_custom_model("RLlibFlatUGVModel", RLlibFlatUGVModel)

def env_creator(config: dict):
    random_map = RandomMapEnv(grid_resolution=GRID_RES)
    flat_env   = FlatEnv(map_env=random_map)
    return ParallelPettingZooEnv(flat_env)

tune.register_env("flat_coverage_env", env_creator)

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return "ugv_policy" if agent_id == "ugv_0" else "uav_policy"

def get_policy_spaces():
    env     = env_creator({})
    ugv_obs = env.observation_space["ugv_0"]
    ugv_act = env.action_space["ugv_0"]
    uav_obs = env.observation_space["uav_0"]
    uav_act = env.action_space["uav_0"]
    env.close()
    return ugv_obs, ugv_act, uav_obs, uav_act

UAV_MODEL_CONFIG = {
    "cnn_channels": 32, "hidden_dim":  128, "dropout": 0.05,
}
UGV_MODEL_CONFIG = {
    "cnn_channels": 32, "hidden_dim":  128, "dropout": 0.05,
}

def build_ppo_config(ugv_obs, ugv_act, uav_obs, uav_act):
    return (
        PPOConfig()
        .environment(env="flat_coverage_env", disable_env_checking=False,)
        .framework("torch")
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .resources(num_gpus=0)
        .env_runners(num_env_runners=NUM_WORKERS, num_envs_per_env_runner=4, rollout_fragment_length="auto", batch_mode="truncate_episodes")
        .training(
            gamma=GAMMA, lr=LR, train_batch_size=TRAIN_BATCH_SIZE, minibatch_size=MINIBATCH_SIZE,
            num_epochs=NUM_SGD_ITER, clip_param=0.2, vf_clip_param=10.0, entropy_coeff=0.01,
            kl_coeff=0.2, lambda_=0.95, grad_clip=0.5,
        )
        .multi_agent(
            policies={
                "ugv_policy": (None, ugv_obs, ugv_act, {"model": {"custom_model": "RLlibFlatUGVModel", "custom_model_config": UGV_MODEL_CONFIG}}),
                "uav_policy": (None, uav_obs, uav_act, {"model": {"custom_model": "RLlibFlatUAVModel", "custom_model_config": UAV_MODEL_CONFIG}}),
            },
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["ugv_policy", "uav_policy"],
        )
    )

def main():
    print("=" * 72)
    print("  MAPPO 扁平架构连续动作训练 (Dimensionality Curse Baseline)")
    print("=" * 72)
    torch.set_num_threads(2)
    
    ray.init(ignore_reinit_error=True, log_to_driver=True, num_cpus=10)
    ugv_obs, ugv_act, uav_obs, uav_act = get_policy_spaces()
    config = build_ppo_config(ugv_obs, ugv_act, uav_obs, uav_act)
    
    algo = config.build()

    import datetime
    best_reward = float("-inf")
    start_iter  = 0
    # 断点续训：优先从 latest/ 恢复
    _resume_path = os.path.join(LATEST_DIR, "rllib_checkpoint.json")
    if os.path.exists(_resume_path):
        print(f"[*] 发现 latest/ 检查点，恢复训练: {LATEST_DIR}")
        algo.restore(LATEST_DIR)
        start_iter = algo.iteration
        print(f"[*] 从 iter {start_iter + 1} 继续...")

    for i in range(start_iter + 1, NUM_ITER + 1):
        result = algo.train()

        # 鲁棒提取指标（兼容 RLlib 2.x：env_runners 子字典）
        def _extract(key):
            for sub in ["env_runners", "sampler_results"]:
                v = result.get(sub, {}).get(key)
                if v is not None:
                    return v
            v = result.get(key)
            return v if v is not None else float("nan")

        rew   = _extract("episode_reward_mean")
        eplen = _extract("episode_len_mean")
        ts    = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] iter {i:>4} | rew: {rew:+8.3f} | len: {eplen:>7.1f}")

        if rew > best_reward:
            best_reward = rew
        if i % CHECKPOINT_FREQ == 0 or i == NUM_ITER:
            algo.save(LATEST_DIR)
        if i % 500 == 0 or i == NUM_ITER:
            algo.save(os.path.join(MILESTONE_DIR, f"iter_{i:04d}"))
            
    algo.stop()
    ray.shutdown()

if __name__ == "__main__":
    main()
