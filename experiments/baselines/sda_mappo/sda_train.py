"""
sda_train.py — SDA-MATD3* (adapted) 学习基线训练脚本
=====================================================
Chen et al., RA-L 2026《Sparse Dual-Attention RL》的注意力架构，移植进 mappo_flat 管线。
本脚本与 baselines/mappo_flat/train_flat_fair.py **逐项对齐**（同 FlatEnv、同 obs/动作、
同 PPO 超参、同 batch/worker/步数上限），**唯一区别**是把自定义模型从 Flat 换成 SDA
（sda_models.RLlibSDAUAVModel / RLlibSDAUGVModel）。因此这是"只改编码器"的公平学习基线对照。

运行：
    cd <REPO_ROOT>/experiments/baselines/sda_mappo
    NUM_ITER=6000 python sda_train.py
评测：与 MAPPO Flat 基线同法（载入 checkpoints_sda_fair/latest 跑确定性策略，
      用与主实验相同的 10 个种子 1001-1010 统计 makespan/deadhead/redundant）。
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("RAY_ENABLE_MAC_LARGE_OBJECT_STORE", "1")
os.environ.setdefault("RAY_memory_usage_threshold", "1.0")
os.environ.setdefault("RAY_object_spilling_threshold", "0.99")
os.environ.setdefault("RAY_filesystem_min_available_threshold", "0")
os.environ.setdefault("RAY_DEDUP_LOGS", "0")

import importlib.util

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))                 # .../baselines/sda_mappo
FLAT_DIR    = os.path.abspath(os.path.join(PROJECT_DIR, "..", "mappo_flat"))

# 复用 mappo_flat 的 FlatEnv（同一环境，保证公平），并对齐 50K 步数上限
_flatenv_path = os.path.join(FLAT_DIR, "FlatEnv.py")
_spec = importlib.util.spec_from_file_location("FlatEnv", _flatenv_path)
_mod  = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
_mod.MAX_EPISODE_STEPS = 50000
sys.modules["FlatEnv"] = _mod
from FlatEnv import FlatEnv

import ray
import torch
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog

for _p in (PROJECT_DIR, FLAT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
SYS_ROOT = os.path.abspath(os.path.join(PROJECT_DIR, "..", "..", ".."))
if SYS_ROOT not in sys.path:
    sys.path.insert(0, SYS_ROOT)
MY_METHOD_DIR = os.path.abspath(os.path.join(PROJECT_DIR, "..", "..", "my_method"))
if MY_METHOD_DIR not in sys.path:
    sys.path.insert(0, MY_METHOD_DIR)

from experiments.my_method.env_defs import RandomMapEnv, GRID_RES
from sda_models import RLlibSDAUAVModel, RLlibSDAUGVModel   # ★ 唯一区别：SDA 编码器

# ── 与 train_flat_fair.py 完全一致的超参 ──────────────────────────────
NUM_ITER         = int(os.environ.get("NUM_ITER", 6000))
CHECKPOINT_FREQ  = int(os.environ.get("CHECKPOINT_FREQ", 10))
CHECKPOINT_DIR   = os.environ.get("CHECKPOINT_DIR",
                       os.path.join(PROJECT_DIR, "checkpoints_sda_fair"))
TRAIN_BATCH_SIZE = 8000
MINIBATCH_SIZE   = 512
NUM_SGD_ITER     = 5
NUM_WORKERS      = int(os.environ.get("NUM_WORKERS", 14))
RAY_CPUS         = int(os.environ.get("RAY_CPUS", 16))
GAMMA, LR        = 0.99, 3e-4   # 恢复 3e-4，grad_clip=5 足以防止 NaN

LATEST_DIR    = os.path.join(CHECKPOINT_DIR, "latest")
MILESTONE_DIR = os.path.join(CHECKPOINT_DIR, "milestones")
os.makedirs(LATEST_DIR, exist_ok=True)
os.makedirs(MILESTONE_DIR, exist_ok=True)

ModelCatalog.register_custom_model("RLlibSDAUAVModel", RLlibSDAUAVModel)
ModelCatalog.register_custom_model("RLlibSDAUGVModel", RLlibSDAUGVModel)

UAV_MODEL_CONFIG = {"cnn_channels": 32, "hidden_dim": 128, "heads": 4, "dropout": 0.05}
UGV_MODEL_CONFIG = {"cnn_channels": 32, "hidden_dim": 128, "heads": 4, "dropout": 0.05}


def env_creator(config: dict):
    return ParallelPettingZooEnv(FlatEnv(map_env=RandomMapEnv(grid_resolution=GRID_RES)))


tune.register_env("sda_coverage_fair_env", env_creator)


def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return "ugv_policy" if agent_id == "ugv_0" else "uav_policy"


def get_policy_spaces():
    env = env_creator({})
    sp = (env.observation_space["ugv_0"], env.action_space["ugv_0"],
          env.observation_space["uav_0"], env.action_space["uav_0"])
    env.close()
    return sp


def build_ppo_config(ugv_obs, ugv_act, uav_obs, uav_act):
    return (
        PPOConfig()
        .environment(env="sda_coverage_fair_env", disable_env_checking=False)
        .framework("torch")
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .resources(num_gpus=0)
        .env_runners(num_env_runners=NUM_WORKERS, num_envs_per_env_runner=8,
                     rollout_fragment_length="auto", batch_mode="truncate_episodes")
        .training(
            gamma=GAMMA, lr=LR,
            train_batch_size=TRAIN_BATCH_SIZE, minibatch_size=MINIBATCH_SIZE,
            num_epochs=NUM_SGD_ITER, clip_param=0.2, vf_clip_param=50.0,  # 从 500 降到 50
            entropy_coeff=0.05,
            entropy_coeff_schedule=[[0, 0.10], [2000000, 0.05],
                                    [8000000, 0.01], [20000000, 0.001]],
            kl_coeff=0.2, lambda_=0.95, grad_clip=5.0,   # 从 40 降到 5，抑制注意力梯度爆炸
        )
        .multi_agent(
            policies={
                "ugv_policy": (None, ugv_obs, ugv_act,
                               {"model": {"custom_model": "RLlibSDAUGVModel",
                                          "custom_model_config": UGV_MODEL_CONFIG}}),
                "uav_policy": (None, uav_obs, uav_act,
                               {"model": {"custom_model": "RLlibSDAUAVModel",
                                          "custom_model_config": UAV_MODEL_CONFIG}}),
            },
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["ugv_policy", "uav_policy"],
        )
    )


def main():
    print("=" * 72, flush=True)
    print("  SDA-MATD3* (adapted) 学习基线训练 —— 与 MAPPO-Flat 同管线，仅换 SDA 编码器", flush=True)
    print(f"  NUM_ITER={NUM_ITER}  BATCH={TRAIN_BATCH_SIZE}  WORKERS={NUM_WORKERS}", flush=True)
    print(f"  CHECKPOINT → {CHECKPOINT_DIR}", flush=True)
    print("=" * 72, flush=True)

    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)

    ray.init(ignore_reinit_error=True, log_to_driver=True, num_cpus=RAY_CPUS,
             object_store_memory=int(48 * 1024**3),
             runtime_env={"env_vars": {
                 "PYTHONPATH": ":".join([SYS_ROOT, PROJECT_DIR, FLAT_DIR, MY_METHOD_DIR,
                                         os.environ.get("PYTHONPATH", "")]),
                 "PYTORCH_ENABLE_MPS_FALLBACK": "1"}})

    ugv_obs, ugv_act, uav_obs, uav_act = get_policy_spaces()
    algo = build_ppo_config(ugv_obs, ugv_act, uav_obs, uav_act).build()

    import datetime, math
    start_iter = 0

    # best/ 目录：保存训练期最优奖励的权重快照，防止后期策略退化覆盖好的权重
    BEST_DIR = os.path.join(CHECKPOINT_DIR, "best")
    os.makedirs(BEST_DIR, exist_ok=True)

    if os.path.exists(os.path.join(LATEST_DIR, "rllib_checkpoint.json")):
        print(f"[*] 恢复训练: {LATEST_DIR}")
        algo.restore(LATEST_DIR); start_iter = algo.iteration

    best_reward = float("-inf")
    for i in range(start_iter + 1, NUM_ITER + 1):
        result = algo.train()
        def _extract(key):
            for sub in ["env_runners", "sampler_results"]:
                v = result.get(sub, {}).get(key)
                if v is not None:
                    return v
            return result.get(key, float("nan"))
        rew = _extract('episode_reward_mean')
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        is_best = ""
        if isinstance(rew, (int, float)) and not math.isnan(rew) and rew > best_reward:
            best_reward = rew
            algo.save(BEST_DIR)
            is_best = " ★ best"
        print(f"[{ts}] iter {i:>5} | rew: {rew:+8.3f} "
              f"| len: {_extract('episode_len_mean'):>8.1f}{is_best}", flush=True)
        if i % CHECKPOINT_FREQ == 0 or i == NUM_ITER:
            algo.save(LATEST_DIR)
        if i % 500 == 0 or i == NUM_ITER:
            algo.save(os.path.join(MILESTONE_DIR, f"iter_{i:05d}"))

    algo.stop(); ray.shutdown()
    print(f"[*] SDA 训练完成。best_reward={best_reward:.3f}")


if __name__ == "__main__":
    main()
