"""
smdp_train.py
==============
SMDP-MAPPO 训练入口

与 my_method/hierarchical_train_v2.py 的关系：
    - 环境：复用 HierarchicalEnvV2（零修改，通过 import 引用）
    - 网络：复用 UAVSpatialCommander / UGVNodeDispatcher（零修改）
    - 策略：替换为 SMDPPPOTorchPolicy（Macro-GAE + Decisional Masking）
    - 超参：与 V2 完全一致，确保消融唯一变量是算法改进

消融配置（通过环境变量控制）：
    USE_MACRO_GAE=True|False        控制 Macro-GAE 开关
    USE_DECISION_MASKING=True|False 控制 Decisional Actor Masking 开关

运行：
    # 配置 D（完整 SMDP-MAPPO）：
    python smdp_train.py

    # 配置 A（退化为标准 MAPPO 基线）：
    USE_MACRO_GAE=False USE_DECISION_MASKING=False python smdp_train.py

    # 配置 B（仅 Macro-GAE）：
    USE_DECISION_MASKING=False python smdp_train.py

    # 配置 C（仅 Masking）：
    USE_MACRO_GAE=False python smdp_train.py

    # 控制迭代次数：
    NUM_ITER=500 python smdp_train.py
"""

import os
import sys
import warnings
import datetime
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── Apple Silicon / Ray 环境变量（与 V2 一致）──
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

# ── 路径注入：引用 my_method/ 中的环境和模型 ──
SMDP_DIR = os.path.dirname(os.path.abspath(__file__))
MY_METHOD_DIR = os.path.abspath(os.path.join(SMDP_DIR, '..', 'my_method'))
if MY_METHOD_DIR not in sys.path:
    sys.path.insert(0, MY_METHOD_DIR)
if SMDP_DIR not in sys.path:
    sys.path.insert(0, SMDP_DIR)

# Ray Worker 进程不继承主进程的 sys.path，
# 需要通过环境变量注入，确保 Worker 能找到 env_defs / HierarchicalEnvV2 等模块
os.environ["PYTHONPATH"] = MY_METHOD_DIR + os.pathsep + SMDP_DIR + os.pathsep + os.environ.get("PYTHONPATH", "")

# 从 my_method/ 复用环境和模型定义（零修改）
from env_defs import RandomMapEnv, MAP_SIZE, MAX_NODES, GRID_RES, GRID_ROWS, GRID_COLS
from HierarchicalEnvV2 import HierarchicalEnv
from hierarchical_models import UAVSpatialCommander, UGVNodeDispatcher

# 从 smdp_mappo/ 引入自定义 Policy
from smdp_policy import SMDPPPOTorchPolicy


# ────────────────────────────────────────────────
#  全局训练超参（与 V2 完全一致）
# ────────────────────────────────────────────────

NUM_ITER         = int(os.environ.get("NUM_ITER",        1000))
CHECKPOINT_FREQ  = int(os.environ.get("CHECKPOINT_FREQ",   10))
MILESTONE_ITERS  = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000}

TRAIN_BATCH_SIZE = 8000
MINIBATCH_SIZE   = 512
NUM_SGD_ITER     = 5
NUM_WORKERS      = 14
GAMMA            = 0.99
LR               = 3e-4

# ── 消融开关（通过环境变量控制）──
USE_MACRO_GAE = os.environ.get("USE_MACRO_GAE", "True") == "True"
USE_DECISION_MASKING = os.environ.get("USE_DECISION_MASKING", "True") == "True"

# 配置名称（自动生成）
_cfg_name = f"macro{'ON' if USE_MACRO_GAE else 'OFF'}_mask{'ON' if USE_DECISION_MASKING else 'OFF'}"

# ── 运行标签（时间戳开头，方便 TensorBoard 排序）──
_RUN_TAG = os.environ.get("RUN_TAG", "v6transit")     # 可通过环境变量自定义
_RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M")
_RUN_NAME = f"{_RUN_TIMESTAMP}_{_cfg_name}_{_RUN_TAG}"

# SMDP-MAPPO 专属 Checkpoint 根目录（按运行隔离）
_DEFAULT_CKPT = os.path.join(
    os.path.dirname(SMDP_DIR),   # experiments/
    "results", "smdp_mappo", _RUN_NAME, "checkpoints"
)
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", _DEFAULT_CKPT)
LATEST_DIR     = os.path.join(CHECKPOINT_DIR, "latest")
MILESTONE_DIR  = os.path.join(CHECKPOINT_DIR, "milestones")

os.makedirs(LATEST_DIR,    exist_ok=True)
os.makedirs(MILESTONE_DIR, exist_ok=True)


# ================================================================
#  RLlib TorchModelV2 包装器（与 V2 完全相同，仅复制注册代码）
# ================================================================
# 注意：不修改 my_method/hierarchical_train_v2.py
# 这里直接定义包装器，因为 RLlib Worker 需要在各自进程中
# 通过 ModelCatalog 找到已注册的 custom_model 名称。

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
        logits, value, _attn_w = self.core(obs_dict, mask)
        self._cur_value = value
        return logits, state

    def value_function(self):
        assert self._cur_value is not None
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
        assert self._cur_value is not None
        return self._cur_value.squeeze(-1)


# 向 RLlib ModelCatalog 注册
ModelCatalog.register_custom_model("RLlibUAVModel_v2", RLlibUAVModel)
ModelCatalog.register_custom_model("RLlibUGVModel_v2", RLlibUGVModel)


# ================================================================
#  V6: Transit Coverage Reward Shaping
#  赶路途中物理覆盖照常（地图变绿），但不计入奖励。
#  仅修改奖励信号，不修改 MDP 的状态转移函数。
#  HierarchicalEnvV2.py 零改动。
# ================================================================

class TransitCoverageShaping(HierarchicalEnv):
    """
    SMDP 专属奖励塑形：抑制赶路途中的覆盖奖励。

    原理：
      - UAV 飞往目标区块途中，物理扫描照常发生（coverage_grid 正常更新）
      - 但 step() 返回的 reward 中，减去赶路路径上扫出的覆盖增量
      - 这样「飞远处」= 纯时间/电量成本，「飞近处」= 更快开始有效扫描
    """

    def step(self, actions):
        # ── 1. 快照：记录 step 前的覆盖格状态 ──
        grid_before = self.coverage_grid.copy()

        # ── 2. 识别赶路中的 UAV（在 step 前判断，因为 step 会改变位置）──
        transit_uavs = []
        for uid in ['uav_0', 'uav_1']:
            uav = self.uavs[uid]
            if (uav['is_busy'] and not uav['is_returning']
                    and not uav['is_swapping'] and uav.get('current_block', 0) > 0):
                (bx0, by0), (bx1, by1) = self._block_id_to_rect(uav['current_block'])
                # UAV 当前位置不在目标区块内 → 正在赶路
                if not (bx0 <= uav['x'] <= bx1 and by0 <= uav['y'] <= by1):
                    transit_uavs.append(uid)

        # ── 3. 正常执行父类 step（覆盖、奖励、状态全部正常计算）──
        obs, rewards, dones, truncs, infos = super().step(actions)

        # ── 4. 如果有赶路 UAV，扣除赶路途中扫出的覆盖奖励 ──
        if transit_uavs and rewards:
            grid_after = self.coverage_grid
            # 本步新增的覆盖格
            new_cells = (grid_before == 0) & (grid_after == 1)

            if np.any(new_cells):
                # 构建所有"有效区块"的掩码（所有 UAV 正在工作的目标区块）
                valid_block_mask = np.zeros_like(grid_after, dtype=bool)
                for uid in ['uav_0', 'uav_1']:
                    blk = self.uavs[uid].get('current_block', 0)
                    if blk > 0:
                        (bx0, by0), (bx1, by1) = self._block_id_to_rect(blk)
                        r0, c0 = self.map_env.xy_to_grid(bx0, by0)
                        r1, c1 = self.map_env.xy_to_grid(
                            bx1 - 1e-3, by1 - 1e-3)
                        valid_block_mask[r0:r1+1, c0:c1+1] = True

                # 赶路覆盖 = 新增格子中，不在任何有效区块内的部分
                transit_cells = int(np.sum(new_cells & ~valid_block_mask))

                if transit_cells > 0:
                    total_cells = grid_after.size
                    # 与环境奖励系数一致：(Δcov) × 1000
                    transit_reward = (transit_cells / total_cells) * 1000.0
                    for a in rewards:
                        rewards[a] -= transit_reward

        return obs, rewards, dones, truncs, infos


# ================================================================
#  环境工厂
# ================================================================

def env_creator(config: dict):
    """供 RLlib Worker 调用，每次生成独立环境实例。
    V6: 使用 TransitCoverageShaping 子类（赶路不计覆盖奖励）。
    """
    random_map = RandomMapEnv(grid_resolution=GRID_RES)
    hier_env   = TransitCoverageShaping(map_env=random_map)
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
#  模型配置
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


# ────────────────────────────────────────────────
#  PPO 配置（SMDP-MAPPO 版）
# ────────────────────────────────────────────────

def build_smdp_ppo_config(ugv_obs, ugv_act, uav_obs, uav_act,
                           use_macro_gae=True, use_decision_masking=True):
    """
    构建 SMDP-MAPPO 的 PPO 配置。

    消融开关通过 custom_model_config 注入到每个 Policy，
    在 smdp_postprocess 和 smdp_policy 中通过
    policy.config["model"]["custom_model_config"] 读取。
    """
    # 注入消融开关到 model config
    uav_cfg = {
        **UAV_MODEL_CONFIG,
        "use_macro_gae": use_macro_gae,
        "use_decision_masking": use_decision_masking,
    }
    ugv_cfg = {
        **UGV_MODEL_CONFIG,
        "use_macro_gae": use_macro_gae,
        "use_decision_masking": use_decision_masking,
    }

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
            num_envs_per_env_runner=6,
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
                # ── UGV 策略（使用 SMDP-MAPPO Policy）──
                "ugv_policy": (
                    SMDPPPOTorchPolicy, ugv_obs, ugv_act,
                    {
                        "model": {
                            "custom_model": "RLlibUGVModel_v2",
                            "custom_model_config": ugv_cfg,
                        }
                    },
                ),
                # ── UAV 策略（使用 SMDP-MAPPO Policy，双机共享权重）──
                "uav_policy": (
                    SMDPPPOTorchPolicy, uav_obs, uav_act,
                    {
                        "model": {
                            "custom_model": "RLlibUAVModel_v2",
                            "custom_model_config": uav_cfg,
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
    # ── 日志持久化：同时输出到终端和文件 ──
    LOG_DIR = os.path.join(CHECKPOINT_DIR, "..", "logs")
    os.makedirs(LOG_DIR, exist_ok=True)
    log_filename = os.path.join(
        LOG_DIR,
        f"train_{_cfg_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    class Tee:
        """同时写入终端和文件的流包装器。"""
        def __init__(self, filepath, stream):
            self._file = open(filepath, "a", encoding="utf-8", buffering=1)
            self._stream = stream
            self.encoding = getattr(stream, "encoding", "utf-8")
        def write(self, msg):
            self._stream.write(msg)
            try:
                self._file.write(msg)
            except Exception:
                pass
        def flush(self):
            self._stream.flush()
            self._file.flush()
        def fileno(self):
            return self._stream.fileno()
        def isatty(self):
            return self._stream.isatty()

    sys.stdout = Tee(log_filename, sys.stdout)
    sys.stderr = Tee(log_filename, sys.stderr)

    print("=" * 72)
    print("  SMDP-MAPPO 训练  ——  Macro-GAE + Decisional Actor Masking")
    print("=" * 72)
    print(f"  📝 训练日志 → {log_filename}")
    print(f"  MAP_SIZE       = {MAP_SIZE}m × {MAP_SIZE}m")
    print(f"  MAX_NODES      = {MAX_NODES}  (UGV Discrete 空间容量)")
    print(f"  GRID           = {GRID_ROWS} × {GRID_COLS} @ {GRID_RES}m/格")
    print(f"  UAV 动作       = Discrete(27)  [0=noop, 1~25=区块, 26=换电]")
    print(f"  NUM_ITER       = {NUM_ITER}   BATCH = {TRAIN_BATCH_SIZE}   WORKERS = {NUM_WORKERS}")
    print(f"  LR             = {LR}")
    print()
    print(f"  [SMDP-MAPPO 消融配置]")
    print(f"  USE_MACRO_GAE        = {USE_MACRO_GAE}")
    print(f"  USE_DECISION_MASKING = {USE_DECISION_MASKING}")
    print(f"  配置名称             = {_cfg_name}")
    print(f"  CHECKPOINT           → {CHECKPOINT_DIR}")
    print()
    print("  [算法改进]")
    print("  ① Macro-GAE:    宏动作折叠 + γ^τ·λ^τ 递归，修复时域信度断裂")
    print("  ② Actor Masking: 仅决策步参与策略梯度，消除零梯度稀释")

    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    print(f"\n  PyTorch CPU 线程: {torch.get_num_threads()} (Learner SGD)")
    print(f"  并发环境数: {NUM_WORKERS} × 6 = {NUM_WORKERS * 6}")

    _ray_tmp = "/tmp/ray_tmp_smdp_mappo"
    os.makedirs(_ray_tmp, exist_ok=True)
    ray.init(
        ignore_reinit_error=True,
        log_to_driver=True,
        num_cpus=16,
        _temp_dir=_ray_tmp,
        object_store_memory=16 * 1024 ** 3,
        runtime_env={
            "env_vars": {
                "PYTHONPATH": MY_METHOD_DIR + os.pathsep + SMDP_DIR,
            },
        },
    )

    print("\n[*] 探测环境空间定义...")
    ugv_obs, ugv_act, uav_obs, uav_act = get_policy_spaces()
    print(f"    UGV obs: {ugv_obs}")
    print(f"    UGV act: {ugv_act}")
    print(f"    UAV obs: {uav_obs}")
    print(f"    UAV act: {uav_act}")

    print("\n[*] 构建 SMDP-MAPPO 配置...")
    config = build_smdp_ppo_config(
        ugv_obs, ugv_act, uav_obs, uav_act,
        use_macro_gae=USE_MACRO_GAE,
        use_decision_masking=USE_DECISION_MASKING,
    )

    print("[*] 初始化 PPO 算法...")
    algo = config.build()

    # ── V6: 禁用 RLlib 硬编码的 batch 级 advantage 标准化 ──
    # RLlib PPO._training_step_old_api_stack() 会调用
    # standardize_fields(train_batch, ["advantages"])
    # 但我们已在 postprocess 中完成 Episode 级标准化
    # 7976 个 0 + 24 个已标准化值会被 re-normalize → advantage 放大 ~17x
    # 最小侵入方案：将 standardize_fields 替换为恒等函数
    import ray.rllib.algorithms.ppo.ppo as _ppo_module
    _ppo_module.standardize_fields = lambda samples, fields: samples
    print("[V6] ✅ 已禁用 RLlib batch 级 advantage 标准化（由 postprocess Episode 级标准化接管）")
    print("[✓] 初始化完成！训练开始...\n")

    best_reward = float("-inf")

    for i in range(1, NUM_ITER + 1):
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
        def _extract_episode_stat(key):
            """按优先级搜索 result 字典中的 episode 指标。"""
            # 路径 1: 新版 RLlib (env_runners)
            v = result.get("env_runners", {}).get(key)
            if v is not None:
                return v
            # 路径 2: 旧版 RLlib (sampler_results)
            v = result.get("sampler_results", {}).get(key)
            if v is not None:
                return v
            # 路径 3: 顶层
            v = result.get(key)
            if v is not None:
                return v
            return float("nan")

        rew   = _extract_episode_stat("episode_reward_mean")
        eplen = _extract_episode_stat("episode_len_mean")
        steps = result.get("timesteps_total", 0)

        def policy_stat(policy_key, stat_key):
            """从 learner 子字典提取 policy 级指标。"""
            # 路径 1: info.learner.{policy}.learner_stats.{stat}
            v = (result.get("info", {})
                       .get("learner", {})
                       .get(policy_key, {})
                       .get("learner_stats", {})
                       .get(stat_key))
            if v is not None:
                return v
            # 路径 2: info.learner.{policy}.{stat}（部分版本扁平化）
            v = (result.get("info", {})
                       .get("learner", {})
                       .get(policy_key, {})
                       .get(stat_key))
            if v is not None:
                return v
            return float("nan")

        # 读取 SMDP-MAPPO 专属监控指标
        dec_ratio = policy_stat("uav_policy", "decision_ratio")

        trend = "📈" if rew > best_reward else "  "
        ts    = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"[{ts}] "
            f"iter {i:>4} | rew: {rew:+8.3f} {trend} | "
            f"len: {eplen:>7.1f} | "
            f"loss_uav: {policy_stat('uav_policy', 'policy_loss'):>8.4f} | "
            f"dec_ratio: {dec_ratio:>5.3f} | "
            f"steps: {steps:>10,}",
            flush=True
        )

        if rew > best_reward:
            best_reward = rew

        # ── 双轨 Checkpoint 保存策略（与 V2 一致）──
        if i % CHECKPOINT_FREQ == 0 or i == NUM_ITER:
            algo.save(LATEST_DIR)
            print(f"  💾 latest/ 更新 (iter={i})", flush=True)

        is_milestone = (i in MILESTONE_ITERS) or (i == NUM_ITER)
        if is_milestone:
            iter_dir = os.path.join(MILESTONE_DIR, f"iter_{i:04d}")
            os.makedirs(iter_dir, exist_ok=True)
            algo.save(iter_dir)
            print(f"  📌 milestones/iter_{i:04d}/ 存档完成", flush=True)

    print("\n" + "=" * 72)
    print(f"  训练完成！最佳 episode_reward_mean = {best_reward:.4f}")
    print(f"  配置: {_cfg_name}")
    print(f"  Checkpoint 目录: {CHECKPOINT_DIR}")
    print("=" * 72)

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
