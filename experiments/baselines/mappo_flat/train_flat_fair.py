"""
train_flat_fair.py
==================
MAPPO Flat 公平对照训练脚本（公平版）

与 train_flat.py 的差异：
  - TRAIN_BATCH_SIZE: 8000  → 16000（与 SA-HMARL 统一）
  - MINIBATCH_SIZE:   512   → 1024
  - NUM_SGD_ITER:     5     → 5（不变）
  - NUM_WORKERS:      6     → 14（充分利用 M4 Max 16 核）
  - num_envs_per_env_runner: 4 → 8
  - MAX_EPISODE_STEPS: 30000 → 50000（放开步数上限，让 MAPPO 有机会探索并完成任务）
  - NUM_ITER:          2000 → 6000（与 SA-HMARL 轮数一致）
  - 保存到 checkpoints_flat_fair/（不覆盖原有结果）

目的：确保超参公平，同时给 MAPPO 足够的回合时长积累成功经验。
原训练结果（checkpoints_flat/）完整保留。

运行：
    cd <REPO_ROOT>/experiments/baselines/mappo_flat
    NUM_ITER=6000 python train_flat_fair.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("RAY_ENABLE_MAC_LARGE_OBJECT_STORE", "1")
os.environ.setdefault("RAY_memory_usage_threshold", "1.0")
os.environ.setdefault("RAY_DISABLE_METRICS_REPORTER", "1")
os.environ.setdefault("RAY_object_spilling_threshold", "0.99")
os.environ.setdefault("RAY_filesystem_min_available_threshold", "0")
os.environ.setdefault("RAY_DEDUP_LOGS", "0")

# ★ 在 import FlatEnv 之前 patch MAX_EPISODE_STEPS
#   FlatEnv.py 第 28 行: MAX_EPISODE_STEPS = 30000（模块级常量）
#   step() 函数在运行时读取该变量，patch 后立即生效
import importlib, importlib.util

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_flatenv_path = os.path.join(PROJECT_DIR, "FlatEnv.py")
_flatenv_spec = importlib.util.spec_from_file_location("FlatEnv", _flatenv_path)
_flatenv_mod  = importlib.util.module_from_spec(_flatenv_spec)
_flatenv_spec.loader.exec_module(_flatenv_mod)
_flatenv_mod.MAX_EPISODE_STEPS = 50000   # ★ 放开步数上限
sys.modules["FlatEnv"] = _flatenv_mod
from FlatEnv import FlatEnv

import ray
import numpy as np
import torch
import torch.nn as nn

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
SYS_ROOT = os.path.abspath(os.path.join(PROJECT_DIR, "..", "..", ".."))
if SYS_ROOT not in sys.path:
    sys.path.insert(0, SYS_ROOT)
MY_METHOD_DIR = os.path.abspath(os.path.join(PROJECT_DIR, "..", "..", "my_method"))
if MY_METHOD_DIR not in sys.path:
    sys.path.insert(0, MY_METHOD_DIR)

from experiments.my_method.env_defs import RandomMapEnv, MAP_SIZE, MAX_NODES, GRID_RES, GRID_ROWS, GRID_COLS
from flat_models import RLlibFlatUAVModel, RLlibFlatUGVModel

# ── 公平版超参 ────────────────────────────────────────────────────────────────
NUM_ITER         = int(os.environ.get("NUM_ITER",        6000))
CHECKPOINT_FREQ  = int(os.environ.get("CHECKPOINT_FREQ",   10))
CHECKPOINT_DIR   = os.environ.get("CHECKPOINT_DIR",
                       os.path.join(PROJECT_DIR, "checkpoints_flat_fair"))

TRAIN_BATCH_SIZE = 8000    # ★ 与 SA-HMARL 严格统一
MINIBATCH_SIZE   = 512     # ★ 与 SA-HMARL 严格统一
NUM_SGD_ITER     = 5       # 不变
NUM_WORKERS      = int(os.environ.get("NUM_WORKERS", 14))   # 顺序运行时全用14核
RAY_CPUS         = int(os.environ.get("RAY_CPUS",   16))   # 单任务占满全部16核
GAMMA            = 0.99
LR               = 3e-4

MAX_EP_STEPS_NEW = 50000   # ★ 放开步数上限（原 30000）

LATEST_DIR    = os.path.join(CHECKPOINT_DIR, "latest")
MILESTONE_DIR = os.path.join(CHECKPOINT_DIR, "milestones")
os.makedirs(LATEST_DIR,    exist_ok=True)
os.makedirs(MILESTONE_DIR, exist_ok=True)

ModelCatalog.register_custom_model("RLlibFlatUAVModel", RLlibFlatUAVModel)
ModelCatalog.register_custom_model("RLlibFlatUGVModel", RLlibFlatUGVModel)


def env_creator(config: dict):
    random_map = RandomMapEnv(grid_resolution=GRID_RES)
    flat_env   = FlatEnv(map_env=random_map)   # 已经是 50K 版本
    return ParallelPettingZooEnv(flat_env)


tune.register_env("flat_coverage_fair_env", env_creator)


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


UAV_MODEL_CONFIG = {"cnn_channels": 32, "hidden_dim": 128, "dropout": 0.05}
UGV_MODEL_CONFIG = {"cnn_channels": 32, "hidden_dim": 128, "dropout": 0.05}


def build_ppo_config(ugv_obs, ugv_act, uav_obs, uav_act):
    return (
        PPOConfig()
        .environment(env="flat_coverage_fair_env", disable_env_checking=False)
        .framework("torch")
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .resources(num_gpus=0)
        .env_runners(
            num_env_runners=NUM_WORKERS,
            num_envs_per_env_runner=8,           # 14×8 = 112 并发环境（12 导致超时）
            rollout_fragment_length="auto",
            batch_mode="truncate_episodes",  # PPO 标准做法，截断 + GAE 自举
        )
        .training(
            gamma=GAMMA,
            lr=LR,
            train_batch_size=TRAIN_BATCH_SIZE,   # 8000（与 SA-HMARL 严格统一）
            minibatch_size=MINIBATCH_SIZE,
            num_epochs=NUM_SGD_ITER,
            clip_param=0.2,
            vf_clip_param=500.0,             # ★ 与 SA-HMARL 统一（原 10.0 导致 NaN）
            entropy_coeff=0.05,              # ★ 与 SA-HMARL 统一（原 0.01 探索不足）
            entropy_coeff_schedule=[          # ★ 与 SA-HMARL 统一衰减
                [0,         0.10],
                [2000000,   0.05],
                [8000000,   0.01],
                [20000000,  0.001],
            ],
            kl_coeff=0.2,
            lambda_=0.95,
            # grad_clip: SA-HMARL 不需要（reward 小），但 Flat 的 reward 可达 -16000
            # 导致梯度爆炸 → NaN。40.0 足够宽松，仅在极端情况兜底
            grad_clip=40.0,
        )
        .multi_agent(
            policies={
                "ugv_policy": (None, ugv_obs, ugv_act,
                               {"model": {"custom_model": "RLlibFlatUGVModel",
                                          "custom_model_config": UGV_MODEL_CONFIG}}),
                "uav_policy": (None, uav_obs, uav_act,
                               {"model": {"custom_model": "RLlibFlatUAVModel",
                                          "custom_model_config": UAV_MODEL_CONFIG}}),
            },
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["ugv_policy", "uav_policy"],
        )
    )


def main():
    ENVS_PER_RUNNER = 12
    print("=" * 72, flush=True)
    print("  MAPPO Flat 公平对照训练（M4 Max 128GB 全力版）", flush=True)
    print(f"  NUM_ITER={NUM_ITER}  BATCH={TRAIN_BATCH_SIZE}  WORKERS={NUM_WORKERS}×{ENVS_PER_RUNNER}={NUM_WORKERS*ENVS_PER_RUNNER}env", flush=True)
    print(f"  MAX_EPISODE_STEPS={MAX_EP_STEPS_NEW}（原 30000，放开探索上限）", flush=True)
    print(f"  CHECKPOINT → {CHECKPOINT_DIR}", flush=True)
    print("=" * 72, flush=True)

    torch.set_num_threads(4)              # SGD 阶段 14 Worker 全空闲，多线程不影响公平性
    torch.set_num_interop_threads(2)      # PyTorch 算子间并行

    SYS_ROOT_PATH = os.path.abspath(os.path.join(PROJECT_DIR, "..", "..", ".."))
    ray.init(
        ignore_reinit_error=True,
        log_to_driver=True,
        num_cpus=RAY_CPUS,
        object_store_memory=int(48 * 1024**3),   # ★ 128GB 系统，48GB 给 object store 充分利用
        runtime_env={
            "env_vars": {
                "PYTHONPATH": ":".join([
                    SYS_ROOT_PATH,
                    PROJECT_DIR,
                    MY_METHOD_DIR,
                    os.environ.get("PYTHONPATH", ""),
                ]),
                "PYTORCH_ENABLE_MPS_FALLBACK": "1",
            }
        },
    )

    ugv_obs, ugv_act, uav_obs, uav_act = get_policy_spaces()
    config = build_ppo_config(ugv_obs, ugv_act, uav_obs, uav_act)
    algo   = config.build()

    import datetime
    start_iter = 0
    _resume_path = os.path.join(LATEST_DIR, "rllib_checkpoint.json")
    if os.path.exists(_resume_path):
        print(f"[*] 发现 latest/ 检查点，恢复训练: {LATEST_DIR}")
        algo.restore(LATEST_DIR)
        start_iter = algo.iteration
        print(f"[*] 从 iter {start_iter + 1} 继续...")

    for i in range(start_iter + 1, NUM_ITER + 1):
        result = algo.train()

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
        print(f"[{ts}] iter {i:>5} | rew: {rew:+8.3f} | len: {eplen:>8.1f}", flush=True)

        if i % CHECKPOINT_FREQ == 0 or i == NUM_ITER:
            algo.save(LATEST_DIR)
        if i % 500 == 0 or i == NUM_ITER:
            algo.save(os.path.join(MILESTONE_DIR, f"iter_{i:05d}"))
            print(f"  [✓] Milestone saved: iter_{i:05d}", flush=True)

    algo.stop()
    ray.shutdown()
    print("[*] 训练完成。")


if __name__ == "__main__":
    main()
