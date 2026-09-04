"""
hierarchical_train_v2_finetune.py
==================================
SA-HMARL V2 微调训练脚本 —— 加大时间惩罚版

与 hierarchical_train_v2.py 的核心差异：
    ① 环境：HierarchicalEnvFinetune（时间惩罚 -0.3 → -0.8）
    ② 从 iter_5400 checkpoint 恢复训练
    ③ 结果完全隔离到 results/sa_hmarl_v2_finetune/

运行：
    python hierarchical_train_v2_finetune.py

    # 自定义训练轮数（默认 2000）
    NUM_ITER=3000 python hierarchical_train_v2_finetune.py
"""

import os
import sys
import warnings
import datetime
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── Apple Silicon / Ray 环境变量 ──────────────────────────────────
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

# ── 项目模块路径 ───────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from env_defs import RandomMapEnv, MAP_SIZE, MAX_NODES, GRID_RES, GRID_ROWS, GRID_COLS
from HierarchicalEnvV2_finetune import HierarchicalEnvFinetune   # ← 微调版环境
from hierarchical_models import UAVSpatialCommander, UGVNodeDispatcher


# ────────────────────────────────────────────────
#  全局训练超参
# ────────────────────────────────────────────────

NUM_ITER        = int(os.environ.get("NUM_ITER",        2000))   # 微调默认 2000 轮
CHECKPOINT_FREQ = int(os.environ.get("CHECKPOINT_FREQ",   10))

# ══════════════════════════════════════════════════════════════════
#  微调关键：从 V2 iter_5400 恢复
# ══════════════════════════════════════════════════════════════════
_FINETUNE_LATEST = os.path.join(
    os.path.dirname(PROJECT_DIR),
    "results", "sa_hmarl_v2_finetune", "checkpoints", "latest"
)
RESUME_FROM = os.environ.get("RESUME_FROM", _FINETUNE_LATEST)
START_ITER  = int(os.environ.get("START_ITER", 5410))

# 里程碑存档轮次
MILESTONE_ITERS = set(range(5500, 8001, 100))

# ══════════════════════════════════════════════════════════════════
#  微调版 Checkpoint 目录（与 V2 完全隔离）
# ══════════════════════════════════════════════════════════════════
_DEFAULT_CKPT = os.path.join(
    os.path.dirname(PROJECT_DIR),
    "results", "sa_hmarl_v2_finetune", "checkpoints"
)
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", _DEFAULT_CKPT)

LATEST_DIR    = os.path.join(CHECKPOINT_DIR, "latest")
MILESTONE_DIR = os.path.join(CHECKPOINT_DIR, "milestones")

# ══════════════════════════════════════════════════════════════════
#  日志文件（与 V2 完全隔离）
# ══════════════════════════════════════════════════════════════════
_LOG_DIR = os.path.join(
    os.path.dirname(PROJECT_DIR),
    "results", "sa_hmarl_v2_finetune"
)
os.makedirs(_LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(
    _LOG_DIR,
    f"train_finetune_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

TRAIN_BATCH_SIZE = 16000    # 加大 batch（原 8000）
MINIBATCH_SIZE   = 1024     # 同步放大 minibatch（原 512）
NUM_SGD_ITER     = 5
NUM_WORKERS      = 12
GAMMA            = 0.99
LR               = 3e-4

os.makedirs(LATEST_DIR,    exist_ok=True)
os.makedirs(MILESTONE_DIR, exist_ok=True)


# ================================================================
#  RLlib TorchModelV2 包装器（与 V2 完全相同）
# ================================================================

class RLlibUAVModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        cfg = model_config.get("custom_model_config", {})
        self.core = UAVSpatialCommander(
            cnn_channels = cfg.get("cnn_channels", 32),
            token_dim    = cfg.get("token_dim",    64),
            state_dim    = cfg.get("state_dim",    64),
            nhead        = cfg.get("nhead",          4),
            dropout      = cfg.get("dropout",      0.05),
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
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        cfg = model_config.get("custom_model_config", {})
        self.core = UGVNodeDispatcher(
            cnn_channels = cfg.get("cnn_channels", 32),
            hidden_dim   = cfg.get("hidden_dim",  128),
            nhead        = cfg.get("nhead",          2),
            dropout      = cfg.get("dropout",      0.05),
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


# ── 注册微调版模型名称（使用与 V2 相同的名称以兼容 checkpoint 恢复）──
ModelCatalog.register_custom_model("RLlibUAVModel_v2", RLlibUAVModel)
ModelCatalog.register_custom_model("RLlibUGVModel_v2", RLlibUGVModel)


# ================================================================
#  环境工厂：使用微调版环境
# ================================================================

def env_creator(config: dict):
    """供 RLlib Worker 调用，每次生成微调版环境实例。"""
    random_map = RandomMapEnv(grid_resolution=GRID_RES)
    hier_env   = HierarchicalEnvFinetune(map_env=random_map)    # ← 微调版
    return ParallelPettingZooEnv(hier_env)


# 注册环境（使用与 V2 相同的名称以兼容 checkpoint 恢复）
tune.register_env("hierarchical_coverage_v2_env", env_creator)


# ────────────────────────────────────────────────
#  策略映射
# ────────────────────────────────────────────────

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return "ugv_policy" if agent_id == "ugv_0" else "uav_policy"


# ────────────────────────────────────────────────
#  探测空间
# ────────────────────────────────────────────────

def get_policy_spaces():
    env     = env_creator({})
    ugv_obs = env.observation_space["ugv_0"]
    ugv_act = env.action_space["ugv_0"]
    uav_obs = env.observation_space["uav_0"]
    uav_act = env.action_space["uav_0"]
    env.close()
    return ugv_obs, ugv_act, uav_obs, uav_act


# ────────────────────────────────────────────────
#  PPO 配置（与 V2 完全相同）
# ────────────────────────────────────────────────

UAV_MODEL_CONFIG = {
    "cnn_channels": 32,
    "token_dim":    64,
    "state_dim":    64,
    "nhead":         4,
    "dropout":      0.05,
}

UGV_MODEL_CONFIG = {
    "cnn_channels": 32,
    "hidden_dim":  128,
    "nhead":         2,
    "dropout":      0.05,
}


def build_ppo_config(ugv_obs, ugv_act, uav_obs, uav_act):
    return (
        PPOConfig()
        .environment(
            env="hierarchical_coverage_v2_env",
            env_config={},
            disable_env_checking=False,
        )
        .framework("torch")
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .resources(num_gpus=0)
        .env_runners(
            num_env_runners=NUM_WORKERS,
            num_envs_per_env_runner=8,     # 加大并发（原 4）
            rollout_fragment_length="auto",
            batch_mode="complete_episodes",
        )
        .training(
            gamma=GAMMA,
            lr=LR,
            train_batch_size=TRAIN_BATCH_SIZE,
            minibatch_size=MINIBATCH_SIZE,
            num_epochs=NUM_SGD_ITER,
            clip_param=0.2,
            vf_clip_param=500.0,
            entropy_coeff=0.05,
            entropy_coeff_schedule=[
                [0,         0.10],
                [2000000,   0.05],
                [8000000,   0.01],
                [20000000,  0.001],
            ],
            kl_coeff=0.2,
            lambda_=0.95,
        )
        .multi_agent(
            policies={
                "ugv_policy": (
                    None, ugv_obs, ugv_act,
                    {
                        "model": {
                            "custom_model": "RLlibUGVModel_v2",
                            "custom_model_config": UGV_MODEL_CONFIG,
                        }
                    },
                ),
                "uav_policy": (
                    None, uav_obs, uav_act,
                    {
                        "model": {
                            "custom_model": "RLlibUAVModel_v2",
                            "custom_model_config": UAV_MODEL_CONFIG,
                        }
                    },
                ),
            },
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["ugv_policy", "uav_policy"],
        )
    )


# ================================================================
#  主训练循环
# ================================================================

def main():
    print("=" * 72)
    print("  SA-HMARL V2 微调训练  ——  加大时间惩罚版 (-0.3 → -0.8)")
    print("=" * 72)
    print(f"  时间惩罚    = -0.8/tick（原始 -0.3/tick）")
    print(f"  恢复自      = {RESUME_FROM}")
    print(f"  起始轮次    = {START_ITER}")
    print(f"  训练轮数    = {NUM_ITER}")
    print(f"  CHECKPOINT  → {CHECKPOINT_DIR}")
    print(f"  日志文件    → {LOG_FILE}")
    print()

    # 同时输出到终端和日志文件
    import io

    class TeeOutput:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()
        def flush(self):
            for s in self.streams:
                s.flush()

    log_fh = open(LOG_FILE, "w", buffering=1)
    sys.stdout = TeeOutput(sys.__stdout__, log_fh)

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    print(f"  PyTorch CPU 线程: {torch.get_num_threads()}")
    print(f"  并发环境数: {NUM_WORKERS} × 4 = {NUM_WORKERS * 4}")

    _ray_tmp = "/tmp/ray_tmp_hierarchical_v2_finetune"   # 隔离 Ray 临时目录
    os.makedirs(_ray_tmp, exist_ok=True)
    ray.init(
        ignore_reinit_error=True,
        log_to_driver=True,
        num_cpus=16,              # 用满 16 核（原 14）
        _temp_dir=_ray_tmp,
        object_store_memory=16 * 1024 ** 3,   # 16GB（原 8GB）
    )

    print("\n[*] 探测环境空间定义...")
    ugv_obs, ugv_act, uav_obs, uav_act = get_policy_spaces()

    print("\n[*] 构建 PPO 配置...")
    config = build_ppo_config(ugv_obs, ugv_act, uav_obs, uav_act)

    print("[*] 初始化 PPO 算法...")
    algo = config.build()

    # ── 从 V2 iter_5400 恢复权重 ──
    print(f"[*] 从 checkpoint 恢复: {RESUME_FROM}")
    algo.restore(RESUME_FROM)
    print(f"[✓] 权重恢复成功！从 iter {START_ITER + 1} 继续训练")
    print("[✓] 初始化完成！微调训练开始...\n")

    best_reward = float("-inf")
    best_ep_len = float("inf")    # 微调目标：追踪最短 ep_len

    for i in range(1, NUM_ITER + 1):
        global_iter = START_ITER + i
        result = algo.train()

        # ── 鲁棒提取 episode 指标 ──
        def _extract(key):
            for sub in ["env_runners", "sampler_results"]:
                v = result.get(sub, {}).get(key)
                if v is not None:
                    return v
            v = result.get(key)
            if v is not None:
                return v
            return float("nan")

        rew   = _extract("episode_reward_mean")
        eplen = _extract("episode_len_mean")
        steps = result.get("timesteps_total", 0)

        def policy_loss(key):
            return (result.get("info", {})
                          .get("learner", {})
                          .get(key, {})
                          .get("learner_stats", {})
                          .get("policy_loss", float("nan")))

        # 微调版：同时追踪 reward 和 ep_len
        rew_marker = "📈" if rew > best_reward else "  "
        len_marker = "🚀" if eplen < best_ep_len else "  "
        
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"[{ts}] "
            f"iter {global_iter:>4} | rew: {rew:+8.3f} {rew_marker} | "
            f"len: {eplen:>7.1f} {len_marker}| "
            f"loss_ugv: {policy_loss('ugv_policy'):>8.4f} | "
            f"loss_uav: {policy_loss('uav_policy'):>8.4f} | "
            f"steps: {steps:>10,}",
            flush=True
        )

        if rew > best_reward:
            best_reward = rew
        if eplen < best_ep_len:
            best_ep_len = eplen

        # ── Checkpoint 保存 ────────────────────────────────────────
        if i % CHECKPOINT_FREQ == 0 or i == NUM_ITER:
            algo.save(LATEST_DIR)
            print(f"  💾 latest/ 更新 (iter={global_iter})", flush=True)

        is_milestone = (global_iter in MILESTONE_ITERS) or (i == NUM_ITER)
        if is_milestone:
            iter_dir = os.path.join(MILESTONE_DIR, f"iter_{global_iter:04d}")
            os.makedirs(iter_dir, exist_ok=True)
            algo.save(iter_dir)
            print(f"  📌 milestones/iter_{global_iter:04d}/ 存档完成", flush=True)

    print("\n" + "=" * 72)
    print(f"  微调训练完成！")
    print(f"  最佳 reward   = {best_reward:.4f}")
    print(f"  最短 ep_len   = {best_ep_len:.1f}")
    print(f"  Checkpoint    → {CHECKPOINT_DIR}")
    print(f"  日志          → {LOG_FILE}")
    print("=" * 72)

    algo.stop()
    ray.shutdown()
    log_fh.close()


if __name__ == "__main__":
    main()
