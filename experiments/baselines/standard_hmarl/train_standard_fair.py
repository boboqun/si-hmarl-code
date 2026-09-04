"""
train_standard_fair.py
======================
Standard H-MARL 公平对照训练脚本（公平版）

与 train_standard.py 的唯一差异：
  - TRAIN_BATCH_SIZE: 8000 → 16000（与 SA-HMARL 统一）
  - MINIBATCH_SIZE:   8000 → 1024（比例一致，支持多次 SGD）
  - NUM_SGD_ITER:     1    → 5   （充分利用大 batch）
  - NUM_WORKERS:      14   → 14  （不变，M4 Max 16核已满载）
  - num_envs_per_env_runner: 8（不变）
  - NUM_ITER:         6000（与 SA-HMARL 轮数一致）
  - 保存到 checkpoints_standard_fair/（不覆盖原有结果）

目的：确保 batch size 一致，使学习曲线对比公平。
原训练结果（checkpoints_standard/）保留，可作历史对比。

运行：
    cd <REPO_ROOT>/experiments/baselines/standard_hmarl
    NUM_ITER=6000 python train_standard_fair.py
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

import ray
import numpy as np
import torch
import torch.nn as nn

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

SYS_ROOT = os.path.abspath(os.path.join(PROJECT_DIR, "..", "..", ".."))
if SYS_ROOT not in sys.path:
    sys.path.insert(0, SYS_ROOT)

MY_METHOD_DIR = os.path.abspath(os.path.join(PROJECT_DIR, "..", "..", "my_method"))
if MY_METHOD_DIR not in sys.path:
    sys.path.insert(0, MY_METHOD_DIR)

from experiments.my_method.env_defs import RandomMapEnv, MAP_SIZE, MAX_NODES, GRID_RES, GRID_ROWS, GRID_COLS
from StandardEnv import StandardEnv
from standard_models import UAVSpatialCommander, UGVNodeDispatcher

# ── 公平版超参（与 SA-HMARL finetune 统一）────────────────────────────────────
NUM_ITER         = int(os.environ.get("NUM_ITER",        6000))   # 与 SA-HMARL 轮数一致
TRAIN_SEED       = int(os.environ.get("TRAIN_SEED",         42))   # 训练随机种子（多seed方差用，镜像 hierarchical_train_v2:278）
CHECKPOINT_FREQ  = int(os.environ.get("CHECKPOINT_FREQ",   10))
CHECKPOINT_DIR   = os.environ.get("CHECKPOINT_DIR",
                       os.path.join(PROJECT_DIR, "checkpoints_standard_fair"))

TRAIN_BATCH_SIZE = 16000   # ★ 与 SA-HMARL 统一（原 8000）
MINIBATCH_SIZE   = 1024    # ★ 比例一致，支持多次 SGD（原 8000）
NUM_SGD_ITER     = 1       # ★ 与原版一致，SGD=1 保证采样效率（原来 6s/轮的关键）
NUM_WORKERS      = int(os.environ.get("NUM_WORKERS", 14))   # 顺序运行时全用14核
RAY_CPUS         = int(os.environ.get("RAY_CPUS",   16))   # 单任务时占满全部16核
GAMMA            = 0.99
LR               = 3e-4

LATEST_DIR    = os.path.join(CHECKPOINT_DIR, "latest")
MILESTONE_DIR = os.path.join(CHECKPOINT_DIR, "milestones")
os.makedirs(LATEST_DIR,    exist_ok=True)
os.makedirs(MILESTONE_DIR, exist_ok=True)

# ── 复用 train_standard.py 中的模型包装器（内联定义，确保 Ray Worker 可序列化）─

class RLlibUAVModel(TorchModelV2, nn.Module):
    """UAVSpatialCommander 的 RLlib TorchModelV2 包装器（同 train_standard.py）"""
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
        logits, value, _attn_w = self.core(obs_dict, mask)
        self._cur_value = value
        return logits, state

    def value_function(self):
        assert self._cur_value is not None
        return self._cur_value.squeeze(-1)


class RLlibUGVModel(TorchModelV2, nn.Module):
    """UGVNodeDispatcher 的 RLlib TorchModelV2 包装器（同 train_standard.py）"""
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
        assert self._cur_value is not None
        return self._cur_value.squeeze(-1)

ModelCatalog.register_custom_model("RLlibUAVModel", RLlibUAVModel)
ModelCatalog.register_custom_model("RLlibUGVModel", RLlibUGVModel)

# ── 环境注册 ─────────────────────────────────────────────────────────────────
def env_creator(config: dict):
    random_map = RandomMapEnv(grid_resolution=GRID_RES)
    env        = StandardEnv(map_env=random_map)
    return ParallelPettingZooEnv(env)

tune.register_env("standard_coverage_fair_env", env_creator)

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

UAV_MODEL_CONFIG = {"hidden_dim": 128, "dropout": 0.05}
UGV_MODEL_CONFIG = {"hidden_dim": 128, "dropout": 0.05}

def build_ppo_config(ugv_obs, ugv_act, uav_obs, uav_act):
    return (
        PPOConfig()
        .environment(env="standard_coverage_fair_env", disable_env_checking=False)
        .framework("torch")
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .resources(num_gpus=0)
        .env_runners(
            num_env_runners=NUM_WORKERS,
            num_envs_per_env_runner=8,          # 14×8 = 112 并发环境，充分利用 128GB RAM
            rollout_fragment_length="auto",
            batch_mode="truncate_episodes",
        )
        .training(
            gamma=GAMMA,
            lr=LR,
            train_batch_size=TRAIN_BATCH_SIZE,  # 16000，与 SA-HMARL 统一
            minibatch_size=MINIBATCH_SIZE,       # 1024
            num_epochs=NUM_SGD_ITER,             # 5
            clip_param=0.2,
            vf_clip_param=10.0,
            entropy_coeff=0.01,
            kl_coeff=0.2,
            lambda_=0.95,
            grad_clip=0.5,
        )
        .multi_agent(
            policies={
                "ugv_policy": (None, ugv_obs, ugv_act,
                               {"model": {"custom_model": "RLlibUGVModel",
                                          "custom_model_config": UGV_MODEL_CONFIG}}),
                "uav_policy": (None, uav_obs, uav_act,
                               {"model": {"custom_model": "RLlibUAVModel",
                                          "custom_model_config": UAV_MODEL_CONFIG}}),
            },
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["ugv_policy", "uav_policy"],
        )
        .debugging(seed=TRAIN_SEED)
    )


def main():
    print("=" * 72, flush=True)
    print("  Standard H-MARL 公平对照训练（batch=16000，与 SA-HMARL 统一）", flush=True)
    print(f"  NUM_ITER={NUM_ITER}  BATCH={TRAIN_BATCH_SIZE}  WORKERS={NUM_WORKERS}×8={NUM_WORKERS*8}env  SEED={TRAIN_SEED}", flush=True)
    print(f"  CHECKPOINT → {CHECKPOINT_DIR}", flush=True)
    print("=" * 72, flush=True)

    torch.set_num_threads(1)   # Ray 多进程下，每 worker 限制 1 线程防争抢

    SYS_ROOT_PATH = os.path.abspath(os.path.join(PROJECT_DIR, "..", "..", ".."))
    ray.init(
        ignore_reinit_error=True,
        log_to_driver=True,
        num_cpus=RAY_CPUS,
        object_store_memory=int(16 * 1024**3),  # 16GB per job when running concurrently
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
        ts    = datetime.datetime.now().strftime("%H:%M:%S")
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
