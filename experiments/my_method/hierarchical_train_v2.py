"""
hierarchical_train_v2.py
=========================
SA-HMARL V2 训练脚本 —— 错峰换电 + UGV 主动预定位奖励版

与 hierarchical_train.py 的核心差异：
    ① 环境：HierarchicalEnvV2（新增三类协作奖励信号）
    ② Checkpoint 按迭代次数独立存储（不覆盖），便于公平对比
    ③ 实验结果与 V1 完全隔离，原始成果不受影响

Checkpoint 保存策略：
    - 每 CHECKPOINT_FREQ 轮保存一次（常规）
    - MILESTONE_ITERS 中的轮次必定保存（对比基准点）
    - 每次保存到独立子目录 iter_NNNN/，历史不覆盖
    目录结构示例：
        results/sa_hmarl_v2/checkpoints/
            iter_0100/    ← 常规存档
            iter_0200/
            iter_0500/    ← 与其他基线公平对比的基准点 ✅
            iter_1000/    ← 最终模型

运行：
    python hierarchical_train_v2.py
    NUM_ITER=1000 python hierarchical_train_v2.py
    CHECKPOINT_FREQ=50 python hierarchical_train_v2.py
"""

import os
import sys
import warnings
import datetime
import torch
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
from HierarchicalEnvV2 import HierarchicalEnv          # ← V2 环境
from hierarchical_models import UAVSpatialCommander, UGVNodeDispatcher


# ────────────────────────────────────────────────
#  全局训练超参
# ────────────────────────────────────────────────

NUM_ITER        = int(os.environ.get("NUM_ITER",        1000))
CHECKPOINT_FREQ = int(os.environ.get("CHECKPOINT_FREQ",   10))  # 快速验证轨：覆盖式，每 N 轮更新一次

# 断点续训：指定 checkpoint 路径，从上次结束处继续训练
# 用法: RESUME_FROM=/path/to/checkpoint python hierarchical_train_v2.py
RESUME_FROM = os.environ.get("RESUME_FROM", "")
START_ITER  = int(os.environ.get("START_ITER", 0))  # 续训起始轮次（用于正确编号）

# 里程碑存档轮次：必定保留在独立子目录（用于公平对比）
MILESTONE_ITERS = set(range(100, 6001, 100))  # 每 100 轮存一个，覆盖到 6000

# V2 专属 Checkpoint 根目录（与 V1 完全隔离）
_DEFAULT_CKPT = os.path.join(
    os.path.dirname(PROJECT_DIR),         # experiments/
    "results", "sa_hmarl_v2", "checkpoints"
)
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", _DEFAULT_CKPT)

# 外层目录分支：
LATEST_DIR    = os.path.join(CHECKPOINT_DIR, "latest")      # 快速验证：覆盖式存档（enjoy_v2.py 默认指向这里）
MILESTONE_DIR = os.path.join(CHECKPOINT_DIR, "milestones")  # 里程碑：按 iter_NNNN/ 永久保留

TRAIN_BATCH_SIZE = 8000
MINIBATCH_SIZE   = 512
NUM_SGD_ITER     = 5
NUM_WORKERS      = int(os.environ.get("NUM_WORKERS", 12))
ENVS_PER_RUNNER  = int(os.environ.get("ENVS_PER_RUNNER", 4))
RAY_CPUS         = int(os.environ.get("RAY_CPUS", 32))
RAY_OBJECT_STORE_GB = int(os.environ.get("RAY_OBJECT_STORE_GB", 16))
_ROLLOUT_FRAGMENT_LENGTH_ENV = os.environ.get("ROLLOUT_FRAGMENT_LENGTH", "auto")
ROLLOUT_FRAGMENT_LENGTH = (
    _ROLLOUT_FRAGMENT_LENGTH_ENV
    if _ROLLOUT_FRAGMENT_LENGTH_ENV == "auto"
    else int(_ROLLOUT_FRAGMENT_LENGTH_ENV)
)
BATCH_MODE      = os.environ.get("BATCH_MODE", "truncate_episodes")
SAMPLE_TIMEOUT_S = int(os.environ.get("SAMPLE_TIMEOUT_S", 300))
# [kl-test] make KL penalty + grad clipping env-controllable; defaults = current behavior.
# headline seed 42 was trained with kl_coeff=0.2 and NO grad_clip:
#   set  KL_COEFF=0.2  GRAD_CLIP=none  to reproduce 42's optimizer config.
KL_COEFF = float(os.environ.get("KL_COEFF", "0.0"))
_GC = os.environ.get("GRAD_CLIP", "10.0")
GRAD_CLIP = None if _GC.lower() in ("none", "null", "") else float(_GC)
TORCH_NUM_THREADS = int(os.environ.get("TORCH_NUM_THREADS", 2))
GAMMA            = 0.99
LR               = 3e-4

os.makedirs(LATEST_DIR,    exist_ok=True)
os.makedirs(MILESTONE_DIR, exist_ok=True)


# ================================================================
#  RLlib TorchModelV2 包装器（与 V1 完全相同，复用同一套网络结构）
# ================================================================

class RLlibUAVModel(TorchModelV2, nn.Module):
    """
    UAVSpatialCommander 的 RLlib TorchModelV2 包装器。

    观测结构（来自 HierarchicalEnvV2）：
        input_dict["obs"]["observation"]["coverage_grid"]  → [B, 20, 20] uint8
        input_dict["obs"]["observation"]["self_state"]     → [B, 7]  float32
        input_dict["obs"]["observation"]["allies_state"]   → [B, 8]  float32
        input_dict["obs"]["action_mask"]                   → [B, 27] int8
    """

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
    """
    UGVNodeDispatcher 的 RLlib TorchModelV2 包装器。

    观测结构（来自 HierarchicalEnvV2）：
        input_dict["obs"]["observation"]["coverage_grid"]  → [B, 20, 20] uint8
        input_dict["obs"]["observation"]["self_state"]     → [B, 5]   float32
        input_dict["obs"]["observation"]["allies_state"]   → [B, 10]  float32
        input_dict["obs"]["action_mask"]                   → [B, 400] int8
    """

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


# ── 向 RLlib ModelCatalog 注册（V2 专属名称，防止与 V1 冲突）──────
ModelCatalog.register_custom_model("RLlibUAVModel_v2", RLlibUAVModel)
ModelCatalog.register_custom_model("RLlibUGVModel_v2", RLlibUGVModel)


# ================================================================
#  环境工厂：RandomMapEnv → HierarchicalEnvV2 → PettingZooEnv
# ================================================================

def env_creator(config: dict):
    """供 RLlib Worker 调用，每次生成独立 V2 环境实例。"""
    random_map = RandomMapEnv(grid_resolution=GRID_RES)
    hier_env   = HierarchicalEnv(map_env=random_map)
    return ParallelPettingZooEnv(hier_env)


tune.register_env("hierarchical_coverage_v2_env", env_creator)


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
        .debugging(seed=int(os.environ.get("TRAIN_SEED", 42)))
        .resources(num_gpus=0)
        .env_runners(
            num_env_runners=NUM_WORKERS,
            num_envs_per_env_runner=ENVS_PER_RUNNER,
            rollout_fragment_length=ROLLOUT_FRAGMENT_LENGTH,
            batch_mode=BATCH_MODE,
            sample_timeout_s=SAMPLE_TIMEOUT_S,
        )
        .training(
            gamma=GAMMA,
            lr=LR,
            train_batch_size=TRAIN_BATCH_SIZE,
            minibatch_size=MINIBATCH_SIZE,
            num_epochs=NUM_SGD_ITER,
            grad_clip=GRAD_CLIP,    # 默认 10.0；GRAD_CLIP=none 关闭（复现 42）
            clip_param=0.2,
            vf_clip_param=500.0,    # 保持大裁剪，适应 R5 奖励的动态范围
            vf_loss_coeff=1.0,
            entropy_coeff=0.05,
            kl_coeff=KL_COEFF,      # 默认 0.0（禁用，防 masked-action KL 爆炸）；KL_COEFF=0.2 复现 42
            entropy_coeff_schedule=[
                [0,         0.10],   # 初期高熵，鼓励探索提早返航行为
                [2000000,   0.05],
                [8000000,   0.01],
                [20000000,  0.001],
            ],
            lambda_=0.95,
        )
        .multi_agent(
            policies={
                # ── UGV 策略：节点调度网络 ─────────────────────────
                "ugv_policy": (
                    None, ugv_obs, ugv_act,
                    {
                        "model": {
                            "custom_model": "RLlibUGVModel_v2",
                            "custom_model_config": UGV_MODEL_CONFIG,
                        }
                    },
                ),
                # ── UAV 策略：空间交叉注意力网络（双机共享权重）──────
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
    print("  SA-HMARL V2 训练  ——  错峰换电 + UGV主动预定位奖励版")
    print("=" * 72)
    print(f"  MAP_SIZE    = {MAP_SIZE}m × {MAP_SIZE}m")
    print(f"  MAX_NODES   = {MAX_NODES}  (UGV Discrete 空间容量)")
    print(f"  GRID        = {GRID_ROWS} × {GRID_COLS} @ {GRID_RES}m/格")
    print(f"  宏区块      = 5 × 5 = 25 块  @ 400m × 400m")
    print(f"  UAV 动作    = Discrete(27)  [0=noop, 1~25=区块, 26=换电]")
    print(f"  NUM_ITER    = {NUM_ITER}   BATCH = {TRAIN_BATCH_SIZE}   WORKERS = {NUM_WORKERS}")
    print(f"  LR          = {LR}")
    print(f"  CHECKPOINT  → {CHECKPOINT_DIR}")
    print(f"    latest/     ← 快速验证，每 {CHECKPOINT_FREQ} 轮覆盖更新")
    print(f"    milestones/ ← 公平对比，每 100 轮独立存档（永不覆盖）")
    print()
    print("  [V2 奖励信号]（与 HierarchicalEnvV2 实际实现一致）")
    print("  R1+R2: 覆盖率增益 α=1000 ·Δcov − 时间惩罚 η=0.3/tick")
    print("  R4(PSR-proj): 主动错峰私有正奖励 (系数1000; 门控 work_done≥400 & ≤1次/局 & cov<0.5)")
    print("  R5: UGV预定位奖励   (proximity × ρ=3.0/tick, D_norm=3000m, 目标=最低电UAV兜底点)")
    print("  P2: 接驳冲击惩罚    (dist/v_g; 被动返航 UGV/UAV 各50%, 主动返航 100%UGV)")
    print("  注: P1(并联换电惩罚) 已移除; crash=-500。详见 HierarchicalEnvV2 奖励段注释。")

    torch.set_num_threads(TORCH_NUM_THREADS)
    torch.set_num_interop_threads(2)
    print(f"\n  PyTorch CPU 线程: {torch.get_num_threads()} (Learner SGD)")
    print(f"  Ray CPU: {RAY_CPUS}")
    print(f"  Ray object store: {RAY_OBJECT_STORE_GB} GB")
    print(f"  rollout_fragment_length: {ROLLOUT_FRAGMENT_LENGTH}")
    print(f"  batch_mode: {BATCH_MODE}")
    print(f"  kl_coeff: {KL_COEFF}   grad_clip: {GRAD_CLIP}   (42原配置: kl=0.2, grad_clip=None)")
    print(f"  sample_timeout_s: {SAMPLE_TIMEOUT_S}")
    print(f"  并发环境数: {NUM_WORKERS} × {ENVS_PER_RUNNER} = {NUM_WORKERS * ENVS_PER_RUNNER}")

    _ray_tmp = "/tmp/ray_tmp_hierarchical_v2"
    os.makedirs(_ray_tmp, exist_ok=True)
    ray.init(
        ignore_reinit_error=True,
        log_to_driver=True,
        num_cpus=RAY_CPUS,
        _temp_dir=_ray_tmp,
        object_store_memory=RAY_OBJECT_STORE_GB * 1024 ** 3,
    )

    print("\n[*] 探测环境空间定义...")
    ugv_obs, ugv_act, uav_obs, uav_act = get_policy_spaces()
    print(f"    UGV obs: {ugv_obs}")
    print(f"    UGV act: {ugv_act}")
    print(f"    UAV obs: {uav_obs}")
    print(f"    UAV act: {uav_act}")

    print("\n[*] 构建 PPO 配置（V2 奖励版）...")
    config = build_ppo_config(ugv_obs, ugv_act, uav_obs, uav_act)

    print("[*] 初始化 PPO 算法...")
    algo = config.build()

    # ── 断点续训：从指定 checkpoint 恢复权重 ──
    if RESUME_FROM:
        print(f"[*] 从 checkpoint 恢复: {RESUME_FROM}")
        algo.restore(RESUME_FROM)
        # ── 强制重置 KL coefficient ──
        # checkpoint 中保存的 kl_coeff 可能因 masked actions 导致 PPO 自适应 KL 机制失控而爆炸
        # 配置中已设置 kl_coeff=0.0 禁用 KL 惩罚，但 restore 会覆盖配置值
        for pid in ["ugv_policy", "uav_policy"]:
            policy = algo.get_policy(pid)
            if hasattr(policy, 'kl_coeff') and policy.kl_coeff > 1.0:
                print(f"  [!] {pid} kl_coeff={policy.kl_coeff:.2e} → 重置为 0.0")
                policy.kl_coeff = 0.0
        print(f"[✓] 权重恢复成功！从 iter {START_ITER + 1} 继续训练")
    
    print("[✓] 初始化完成！训练开始...\n")

    best_reward = float("-inf")

    for i in range(1, NUM_ITER + 1):
        global_iter = START_ITER + i  # 全局轮次编号（续训时正确递增）
        result = algo.train()

        # ── 首轮 debug：打印 result 顶层 keys，帮助定位指标路径 ──
        if i == 1:
            print(f"  [DEBUG] result top keys: {list(result.keys())}")
            for _dbg_key in ["sampler_results", "env_runners", "episode_reward_mean"]:
                if _dbg_key in result:
                    _val = result[_dbg_key]
                    if isinstance(_val, dict):
                        print(f"  [DEBUG] result['{_dbg_key}'] keys: {list(_val.keys())}")
                    else:
                        print(f"  [DEBUG] result['{_dbg_key}'] = {_val}")

        # ── 鲁棒提取 episode 指标（兼容多个 RLlib 版本）──
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
            learner_info = result.get("info", {}).get("learner", {}).get(key, {})
            # try different paths for Ray 2.x compatibility
            if "learner_stats" in learner_info and "policy_loss" in learner_info["learner_stats"]:
                return learner_info["learner_stats"]["policy_loss"]
            if "policy_loss" in learner_info:
                return learner_info["policy_loss"]
            
            # recursive search just in case
            def find_loss(d):
                if not isinstance(d, dict): return None
                if "policy_loss" in d: return d["policy_loss"]
                for k, v in d.items():
                    res = find_loss(v)
                    if res is not None: return res
                return None
                
            res = find_loss(learner_info)
            return res if res is not None else 0.0

        trend = "📈" if rew > best_reward else "  "
        ts    = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"[{ts}] "
            f"iter {global_iter:>4} | rew: {rew:+8.3f} {trend} | "
            f"len: {eplen:>7.1f} | "
            f"loss_ugv: {policy_loss('ugv_policy'):>8.4f} | "
            f"loss_uav: {policy_loss('uav_policy'):>8.4f} | "
            f"steps: {steps:>10,}",
            flush=True
        )

        if rew > best_reward:
            best_reward = rew

        # ── 双轨 Checkpoint 保存策略 ────────────────────────────────────────
        # 轨道①（快速验证）: 每 CHECKPOINT_FREQ 轮覆盖 latest/，随时可用
        if i % CHECKPOINT_FREQ == 0 or i == NUM_ITER:
            algo.save(LATEST_DIR)
            print(f"  💾 latest/ 更新 (iter={global_iter})", flush=True)

        # 轨道②（里程碑存档）: 每 100 轮或到达配置里程碑，存入独立 iter_NNNN/ 永不覆盖
        is_milestone = (global_iter in MILESTONE_ITERS) or (i == NUM_ITER)
        if is_milestone:
            iter_dir = os.path.join(MILESTONE_DIR, f"iter_{global_iter:04d}")
            os.makedirs(iter_dir, exist_ok=True)
            algo.save(iter_dir)
            print(f"  📌 milestones/iter_{global_iter:04d}/ 存档完成", flush=True)

    print("\n" + "=" * 72)
    print(f"  训练完成！最佳 episode_reward_mean = {best_reward:.4f}")
    print(f"  Checkpoint 目录: {CHECKPOINT_DIR}")
    print("=" * 72)

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
