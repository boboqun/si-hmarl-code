"""
hierarchical_train.py
=====================
分层宏观指令架构 —— MARL PPO 训练脚本

与 advanced_train.py 的核心差异：
    ① 环境：HierarchicalEnv（宏区块 + Boustrophedon 底层规划）
    ② UAV 模型：UAVSpatialCommander（空间交叉注意力，27维离散动作）
    ③ UGV 模型：UGVNodeDispatcher（UAV电量注意力，800维节点调度，2km农田场景）
    ④ 观测为嵌套 Dict：{"observation": {...}, "action_mask": array}
    ⑤ Checkpoint 保存到 checkpoints_hierarchical/

运行：
    python hierarchical_train.py
    NUM_ITER=300 python hierarchical_train.py
    CHECKPOINT_FREQ=5 python hierarchical_train.py
"""

import os
import sys
import warnings
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
from HierarchicalEnv import HierarchicalEnv
from hierarchical_models import UAVSpatialCommander, UGVNodeDispatcher


# ────────────────────────────────────────────────
#  全局训练超参
# ────────────────────────────────────────────────

NUM_ITER        = int(os.environ.get("NUM_ITER",        2000))
CHECKPOINT_FREQ = int(os.environ.get("CHECKPOINT_FREQ",  10))
CHECKPOINT_DIR  = os.environ.get("CHECKPOINT_DIR", os.path.join(PROJECT_DIR, "checkpoints_sa_hmarl_tuned"))

TRAIN_BATCH_SIZE = 8000
MINIBATCH_SIZE   = 512
NUM_SGD_ITER     = 5
NUM_WORKERS      = 12
GAMMA            = 0.99
LR               = 3e-4    # 略高于 advanced_train，离散动作空间更容易收敛

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ================================================================
#  RLlib TorchModelV2 包装器
# ================================================================

class RLlibUAVModel(TorchModelV2, nn.Module):
    """
    UAVSpatialCommander 的 RLlib TorchModelV2 包装器。

    观测结构（来自 HierarchicalEnv）：
        input_dict["obs"]["observation"]["coverage_grid"]  → [B, 20, 20] uint8
        input_dict["obs"]["observation"]["self_state"]     → [B, 7]  float32
        input_dict["obs"]["observation"]["allies_state"]   → [B, 8]  float32
        input_dict["obs"]["action_mask"]                   → [B, 27] int8
    """

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name)
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
        obs      = input_dict["obs"]
        obs_inner = obs["observation"]   # 嵌套的 Dict

        # 提取并确保 float（coverage_grid 来自 uint8，CNN 需要 float）
        cov_grid = obs_inner["coverage_grid"].float()   # [B, 20, 20]
        self_st  = obs_inner["self_state"].float()      # [B, 7]
        allies   = obs_inner["allies_state"].float()    # [B, 8]

        # action_mask: int8 → float，形状 [B, 27]
        mask = obs["action_mask"].float()

        obs_dict = {
            "coverage_grid": cov_grid,
            "self_state":    self_st,
            "allies_state":  allies,
        }

        logits, value, _attn_w = self.core(obs_dict, mask)
        self._cur_value = value   # 暂存，供 value_function() 使用
        return logits, state

    def value_function(self):
        assert self._cur_value is not None, "forward() must be called first"
        return self._cur_value.squeeze(-1)   # [B]


class RLlibUGVModel(TorchModelV2, nn.Module):
    """
    UGVNodeDispatcher 的 RLlib TorchModelV2 包装器。

    观测结构（来自 HierarchicalEnv）：
        input_dict["obs"]["observation"]["coverage_grid"]  → [B, 20, 20] uint8
        input_dict["obs"]["observation"]["self_state"]     → [B, 5]   float32
        input_dict["obs"]["observation"]["allies_state"]   → [B, 10]  float32
        input_dict["obs"]["action_mask"]                   → [B, 400] int8
    """

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name)
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

        cov_grid = obs_inner["coverage_grid"].float()   # [B, 20, 20]
        self_st  = obs_inner["self_state"].float()      # [B, 5]
        allies   = obs_inner["allies_state"].float()    # [B, 10]

        # action_mask: [B, 400]
        mask = obs["action_mask"].float()

        obs_dict = {
            "coverage_grid": cov_grid,
            "self_state":    self_st,
            "allies_state":  allies,
        }

        logits, value = self.core(obs_dict, mask)
        self._cur_value = value
        return logits, state

    def value_function(self):
        assert self._cur_value is not None, "forward() must be called first"
        return self._cur_value.squeeze(-1)   # [B]


# ── 向 RLlib ModelCatalog 注册 ────────────────────────────────────
ModelCatalog.register_custom_model("RLlibUAVModel", RLlibUAVModel)
ModelCatalog.register_custom_model("RLlibUGVModel", RLlibUGVModel)


# ================================================================
#  环境工厂：RandomMapEnv → HierarchicalEnv → PettingZooEnv
# ================================================================

def env_creator(config: dict):
    """供 RLlib Worker 调用，每次生成独立环境实例。"""
    random_map  = RandomMapEnv(grid_resolution=GRID_RES)
    hier_env    = HierarchicalEnv(map_env=random_map)
    return ParallelPettingZooEnv(hier_env)


tune.register_env("sa_hmarl_tuned_env", env_creator)


# ────────────────────────────────────────────────
#  策略映射
# ────────────────────────────────────────────────

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    """ugv_0 → ugv_policy；uav_0 / uav_1 → uav_policy（共享权重）。"""
    return "ugv_policy" if agent_id == "ugv_0" else "uav_policy"


# ────────────────────────────────────────────────
#  探测空间
# ────────────────────────────────────────────────

def get_policy_spaces():
    """临时实例化环境，读取各智能体的 obs / action space。"""
    env     = env_creator({})
    ugv_obs = env.observation_space["ugv_0"]
    ugv_act = env.action_space["ugv_0"]
    uav_obs = env.observation_space["uav_0"]
    uav_act = env.action_space["uav_0"]
    env.close()
    return ugv_obs, ugv_act, uav_obs, uav_act


# ────────────────────────────────────────────────
#  PPO 配置
# ────────────────────────────────────────────────

# 模型超参（扩大 Brain Capacity 以处理 Sparse Penalty）
UAV_MODEL_CONFIG = {
    "cnn_channels": 32,
    "token_dim":    64,
    "state_dim":    64,
    "nhead":         8,     # UAV 增加注意力头
    "dropout":      0.05,
}

UGV_MODEL_CONFIG = {
    "cnn_channels": 32,
    "hidden_dim":  256,    # UGV 脑容量翻倍，强化时空长程记忆
    "nhead":         4,     # UGV 增加注意力头
    "dropout":      0.05,
}


def build_ppo_config(ugv_obs, ugv_act, uav_obs, uav_act):
    return (
        PPOConfig()
        .environment(
            env="sa_hmarl_tuned_env",
            env_config={},
            disable_env_checking=False,
        )
        .framework("torch")
        # 关闭新 API 栈，使用旧版 TorchModelV2 管道
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .resources(num_gpus=0)
        .env_runners(
            num_env_runners=NUM_WORKERS,
            num_envs_per_env_runner=4,
            rollout_fragment_length="auto",
            batch_mode="truncate_episodes",
        )
        .training(
            gamma=GAMMA,
            lr=LR,
            # 引入突击退火 LR Schedule，初期大步长破局，中后期锁定
            lr_schedule=[
                [0, 3e-4],
                [5000000, 5e-5],
                [15000000, 1e-5]
            ],
            train_batch_size=TRAIN_BATCH_SIZE,
            minibatch_size=MINIBATCH_SIZE,
            num_epochs=NUM_SGD_ITER,
            clip_param=0.2,
            vf_clip_param=500.0,   # 🚨 为防止新加的 -600(排队换电扣分) 引发 Value 截断失效，大幅放开 vf clip
            # 引入极端探索压缩曲线：初期保持高探索以试探“提前返航代码 (action=26)”，防止早衰
            entropy_coeff=0.05, 
            entropy_coeff_schedule=[
                [0, 0.08],
                [3000000, 0.01],
                [15000000, 0.001]
            ],
            kl_coeff=0.2,
            lambda_=0.95,
        )
        .multi_agent(
            policies={
                # ── UGV 策略：节点调度网络 ─────────────────────────
                "ugv_policy": (
                    None, ugv_obs, ugv_act,
                    {
                        "model": {
                            "custom_model": "RLlibUGVModel",
                            "custom_model_config": UGV_MODEL_CONFIG,
                        }
                    },
                ),
                # ── UAV 策略：空间交叉注意力网络（双机共享权重）──────
                "uav_policy": (
                    None, uav_obs, uav_act,
                    {
                        "model": {
                            "custom_model": "RLlibUAVModel",
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
    print("  分层宏观指令 MARL PPO 训练  ——  HierarchicalEnv 版")
    print("=" * 72)
    print(f"  MAP_SIZE    = {MAP_SIZE}m × {MAP_SIZE}m")
    print(f"  MAX_NODES   = {MAX_NODES}  (UGV Discrete 空间容量)")
    print(f"  GRID        = {GRID_ROWS} × {GRID_COLS} @ {GRID_RES}m/格")
    print(f"  宏区块      = 5 × 5 = 25 块  @ 400m × 400m  (2km场景)")
    print(f"  UAV 动作    = Discrete(27)  [0=noop, 1~25=区块, 26=换电]")
    print(f"  NUM_ITER    = {NUM_ITER}   BATCH = {TRAIN_BATCH_SIZE}   WORKERS = {NUM_WORKERS}")
    print(f"  LR          = {LR}")

    # Learner 线程限制（防止与 Worker 争核）
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    print(f"  PyTorch CPU 线程: {torch.get_num_threads()} threads 用于 Learner SGD")
    print(f"  并发环境数: {NUM_WORKERS} × 4 = {NUM_WORKERS * 4}")

    # Ray 初始化
    _ray_tmp = "/tmp/ray_tmp_hierarchical"
    os.makedirs(_ray_tmp, exist_ok=True)
    ray.init(
        ignore_reinit_error=True,
        log_to_driver=True,
        num_cpus=14,
        _temp_dir=_ray_tmp,
        object_store_memory=8 * 1024 ** 3,
    )

    print("\n[*] 探测环境空间定义...")
    ugv_obs, ugv_act, uav_obs, uav_act = get_policy_spaces()
    print(f"    UGV obs: {ugv_obs}")
    print(f"    UGV act: {ugv_act}")
    print(f"    UAV obs: {uav_obs}")
    print(f"    UAV act: {uav_act}")

    print("\n[*] 构建 PPO 配置（分层宏观指令版）...")
    config = build_ppo_config(ugv_obs, ugv_act, uav_obs, uav_act)

    print("[*] 初始化 PPO 算法...")
    algo = config.build()
    print("[✓] 初始化完成！训练开始...\n")

    best_reward = float("-inf")

    for i in range(1, NUM_ITER + 1):
        result = algo.train()

        rew   = result.get("episode_reward_mean", float("nan"))
        eplen = result.get("episode_len_mean",    float("nan"))
        steps = result.get("timesteps_total",     0)

        def policy_loss(key):
            return (result.get("info", {})
                          .get("learner", {})
                          .get(key, {})
                          .get("learner_stats", {})
                          .get("policy_loss", float("nan")))

        trend = "📈" if rew > best_reward else "  "
        print(
            f"iter {i:>4} | rew: {rew:+8.3f} {trend} | "
            f"len: {eplen:>7.1f} | "
            f"loss_ugv: {policy_loss('ugv_policy'):>8.4f} | "
            f"loss_uav: {policy_loss('uav_policy'):>8.4f} | "
            f"steps: {steps:>10,}"
        )

        if rew > best_reward:
            best_reward = rew

        if i % CHECKPOINT_FREQ == 0 or i == NUM_ITER:
            ckpt = algo.save(CHECKPOINT_DIR)
            print(f"  💾 Checkpoint → {ckpt}")

    print("\n" + "=" * 72)
    print(f"  训练完成！最佳 episode_reward_mean = {best_reward:.4f}")
    print(f"  Checkpoint 目录: {CHECKPOINT_DIR}")
    print("=" * 72)

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
