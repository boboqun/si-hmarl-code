"""
run_ablation_matrix.py — 隔离式消融训练器（对应审稿意见 M3）
============================================================
在「动作空间 / 物理引擎 / 训练超参全部固定」的前提下，仅改变单一变量来训练
SA-HMARL 的一系列消融变体，从而把每个机制的贡献干净地归因开来。这直接回应
审稿意见 M3：原文与 Standard H-MARL 的对比同时改变了「奖励」与「会合方式」
两个变量，无法分离增益来源。

变体矩阵
--------
A) 奖励 × 会合 2×2 因子（隔离信用分配 vs 连续会合）：
     full            = dual_channel + continuous   (= 完整 SA-HMARL，参照基准)
     flat_continuous = flat_shared  + continuous
     dual_node       = dual_channel + node
     flat_node       = flat_shared  + node
B) 双通道信用分配的逐通道隔离：
     full            = 主动 + 被动 两通道（参照基准）
     proactive_only  = 仅主动错峰通道   (enable_reactive=False)
     reactive_only   = 仅被动冲击惩罚通道(enable_proactive=False)
     no_credit       = 两通道均关闭
C) 宏策略结构（隔离交叉注意力本身）：
     full            = 空间交叉注意力（参照基准）
     mlp_commander   = 普通 MLP（去掉空间交叉注意力）

注：上述变体均通过 HierarchicalEnvV2 的 reward_mode / rendezvous_mode /
enable_proactive / enable_reactive 开关实现，这些开关的默认值精确还原原始
SA-HMARL，因此主结果不受影响（详见 HierarchicalEnvV2.__init__ 注释）。

用法
----
    python run_ablation_matrix.py --variant flat_continuous
    python run_ablation_matrix.py --variant mlp_commander --iters 1000
    python run_ablation_matrix.py --variant all          # 顺序训练全部变体

每个变体的权重写入：
    experiments/results/ablations/<variant>/checkpoints/latest/
随后用 evaluate_ablations.py 在 10 个种子上统一评测并汇总对比。
"""

import os
import sys
import argparse
import datetime
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("RAY_ENABLE_MAC_LARGE_OBJECT_STORE", "1")
os.environ.setdefault("RAY_memory_usage_threshold", "1.0")
os.environ.setdefault("RAY_DISABLE_METRICS_REPORTER", "1")
os.environ.setdefault("RAY_object_spilling_threshold", "0.99")
os.environ.setdefault("RAY_filesystem_min_available_threshold", "0")
os.environ.setdefault("RAY_DEDUP_LOGS", "0")

# ── 路径注入：my_method（环境/模型）与本目录（mlp_commander）──────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_MY_METHOD = os.path.abspath(os.path.join(_HERE, "..", "..", "my_method"))
for _p in (_MY_METHOD, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ray  # noqa: E402
import torch.nn as nn  # noqa: E402
from ray import tune  # noqa: E402
from ray.rllib.algorithms.ppo import PPOConfig  # noqa: E402
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv  # noqa: E402
from ray.rllib.models import ModelCatalog  # noqa: E402
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2  # noqa: E402

from env_defs import RandomMapEnv, GRID_RES  # noqa: E402
from HierarchicalEnvV2 import HierarchicalEnv  # noqa: E402
# 复用主训练脚本里已验证的 RLlib 包装器、模型配置、策略映射与超参
from hierarchical_train_v2 import (  # noqa: E402
    RLlibUAVModel, RLlibUGVModel,
    UAV_MODEL_CONFIG, UGV_MODEL_CONFIG,
    policy_mapping_fn,
    TRAIN_BATCH_SIZE, MINIBATCH_SIZE, NUM_SGD_ITER,
    NUM_WORKERS, GAMMA, LR,
)
from mlp_commander import MLPCommander  # noqa: E402


# ================================================================
#  MLP 版 UAV 模型包装器（CA 消融专用）
# ================================================================
class RLlibUAVModel_MLP(TorchModelV2, nn.Module):
    """MLPCommander 的 RLlib TorchModelV2 包装器，I/O 与 RLlibUAVModel 一致。"""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        cfg = model_config.get("custom_model_config", {})
        self.core = MLPCommander(
            cnn_channels=cfg.get("cnn_channels", 32),
            token_dim=cfg.get("token_dim", 64),
            state_dim=cfg.get("state_dim", 64),
            nhead=cfg.get("nhead", 4),
            dropout=cfg.get("dropout", 0.05),
        )
        self._cur_value = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        inner = obs["observation"]
        obs_dict = {
            "coverage_grid": inner["coverage_grid"].float(),
            "self_state": inner["self_state"].float(),
            "allies_state": inner["allies_state"].float(),
        }
        logits, value, _ = self.core(obs_dict, obs["action_mask"].float())
        self._cur_value = value
        return logits, state

    def value_function(self):
        assert self._cur_value is not None
        return self._cur_value.squeeze(-1)


ModelCatalog.register_custom_model("RLlibUAVModel_MLP", RLlibUAVModel_MLP)


# ================================================================
#  变体矩阵定义
# ================================================================
VARIANTS = {
    # A) 奖励 × 会合 2×2 因子
    "full":            dict(env_kwargs={},                                                  uav_model="RLlibUAVModel_v2"),
    "flat_continuous": dict(env_kwargs={"reward_mode": "flat_shared"},                      uav_model="RLlibUAVModel_v2"),
    "dual_node":       dict(env_kwargs={"rendezvous_mode": "node"},                         uav_model="RLlibUAVModel_v2"),
    "flat_node":       dict(env_kwargs={"reward_mode": "flat_shared", "rendezvous_mode": "node"}, uav_model="RLlibUAVModel_v2"),
    # B) 双通道逐通道隔离
    "proactive_only":  dict(env_kwargs={"enable_reactive": False},                          uav_model="RLlibUAVModel_v2"),
    "reactive_only":   dict(env_kwargs={"enable_proactive": False},                         uav_model="RLlibUAVModel_v2"),
    "no_credit":       dict(env_kwargs={"enable_proactive": False, "enable_reactive": False}, uav_model="RLlibUAVModel_v2"),
    # C) 宏策略结构
    "mlp_commander":   dict(env_kwargs={},                                                  uav_model="RLlibUAVModel_MLP"),
    # D) [M9] 奖励权重敏感性扫描（reward_weights 经 env_kwargs 透传到 HierarchicalEnvV2）
    #    回应审稿意见 M9：50/50 拆分比例、预定位系数 ρ、冲击惩罚权重 β 此前为硬编码，
    #    缺乏敏感性证据。以下变体各自只改单一权重，其余精确还原主结果。
    "split_30":        dict(env_kwargs={"reward_weights": {"shock_ugv_frac": 0.3}},          uav_model="RLlibUAVModel_v2"),
    "split_70":        dict(env_kwargs={"reward_weights": {"shock_ugv_frac": 0.7}},          uav_model="RLlibUAVModel_v2"),
    "split_100":       dict(env_kwargs={"reward_weights": {"shock_ugv_frac": 1.0}},          uav_model="RLlibUAVModel_v2"),
    "prepos_lo":       dict(env_kwargs={"reward_weights": {"prepos": 1.5}},                  uav_model="RLlibUAVModel_v2"),
    "prepos_hi":       dict(env_kwargs={"reward_weights": {"prepos": 6.0}},                  uav_model="RLlibUAVModel_v2"),
    "shock_lo":        dict(env_kwargs={"reward_weights": {"shock": 0.5}},                   uav_model="RLlibUAVModel_v2"),
    "shock_hi":        dict(env_kwargs={"reward_weights": {"shock": 2.0}},                   uav_model="RLlibUAVModel_v2"),
}


def make_env_creator(env_kwargs: dict):
    """返回一个把指定消融开关注入 HierarchicalEnvV2 的 env_creator。"""
    def _creator(config):
        random_map = RandomMapEnv(grid_resolution=GRID_RES)
        return ParallelPettingZooEnv(HierarchicalEnv(map_env=random_map, **env_kwargs))
    return _creator


def build_config(variant_name: str):
    v = VARIANTS[variant_name]
    env_name = f"ablation_env_{variant_name}"
    tune.register_env(env_name, make_env_creator(v["env_kwargs"]))

    # 探测各 agent 的 obs/action space
    probe = make_env_creator(v["env_kwargs"])({})
    ugv_obs, ugv_act = probe.observation_space["ugv_0"], probe.action_space["ugv_0"]
    uav_obs, uav_act = probe.observation_space["uav_0"], probe.action_space["uav_0"]
    probe.close()

    return (
        PPOConfig()
        .environment(env=env_name, env_config={}, disable_env_checking=False)
        .framework("torch")
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .resources(num_gpus=0)
        .env_runners(
            num_env_runners=int(os.environ.get("NUM_WORKERS", 14)),
            num_envs_per_env_runner=4,
            rollout_fragment_length="auto",
            batch_mode="truncate_episodes",
        )
        .training(
            gamma=GAMMA, lr=LR,
            train_batch_size=TRAIN_BATCH_SIZE,
            minibatch_size=MINIBATCH_SIZE,
            num_epochs=NUM_SGD_ITER,
            clip_param=0.2, vf_clip_param=500.0,
            entropy_coeff=0.05,
            entropy_coeff_schedule=[
                [0,         0.10],
                [2000000,   0.05],
                [8000000,   0.01],
                [20000000,  0.001],
            ],
            kl_coeff=0.2, lambda_=0.95,
        )
        .multi_agent(
            policies={
                "ugv_policy": (
                    None, ugv_obs, ugv_act,
                    {"model": {"custom_model": "RLlibUGVModel_v2", "custom_model_config": UGV_MODEL_CONFIG}},
                ),
                "uav_policy": (
                    None, uav_obs, uav_act,
                    {"model": {"custom_model": v["uav_model"], "custom_model_config": UAV_MODEL_CONFIG}},
                ),
            },
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["ugv_policy", "uav_policy"],
        )
    )


def update_training_log(variant_name, i, num_iter, rew, is_ckpt, log_file, out_dir=""):
    lines = []
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            lines = f.readlines()
            
    if lines and "(running...)" in lines[-1]:
        lines.pop()
        
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    if is_ckpt:
        lines.append(f"[{ts}]  [{variant_name}] iter {i:>4}/{num_iter} | rew={rew:10.2f} | 💾 {out_dir}\n")
    else:
        lines.append(f"[{ts}]  [{variant_name}] iter {i:>4}/{num_iter} | rew={rew:10.2f} (running...)\n")
        
    with open(log_file, "w") as f:
        f.writelines(lines)

def train_variant(variant_name: str, num_iter: int, ckpt_freq: int, log_file: str):
    out_dir = os.path.abspath(os.path.join(
        _HERE, "..", "..", "results", "ablations", variant_name, "checkpoints", "latest"))
    os.makedirs(out_dir, exist_ok=True)
    # [新增] best/ 目录：保存训练期最优奖励的权重快照，防止后期崩溃把好策略覆盖。
    best_dir = os.path.abspath(os.path.join(
        _HERE, "..", "..", "results", "ablations", variant_name, "checkpoints", "best"))
    os.makedirs(best_dir, exist_ok=True)

    print("=" * 64)
    print(f"  消融变体: {variant_name}")
    print(f"  env_kwargs = {VARIANTS[variant_name]['env_kwargs']}")
    print(f"  uav_model  = {VARIANTS[variant_name]['uav_model']}")
    print(f"  iters={num_iter}  -> {out_dir}")
    print("=" * 64)

    config = build_config(variant_name)
    algo = config.build()
    
    # 自动断点续训逻辑
    if os.path.exists(os.path.join(out_dir, "rllib_checkpoint.json")):
        try:
            algo.restore(out_dir)
            print(f"  [!] 检测到历史断点，已成功恢复权重和进度: {out_dir}")
        except Exception as e:
            print(f"  [!] 恢复断点失败，将从头开始训练: {e}")

    best = float("-inf")
    while True:
        result = algo.train()
        i = result.get("training_iteration", 0)
        
        rew = result.get("env_runners", {}).get("episode_reward_mean")
        if rew is None:
            rew = result.get("episode_reward_mean", float("nan"))

        if i % ckpt_freq == 0 or i == num_iter:
            algo.save(out_dir)   # latest/ ：断点续训点（每次覆盖）
            # [新增] 仅当奖励刷新历史最优时，额外存一份到 best/（评测优先用 best/）
            if isinstance(rew, (int, float)) and rew > best:
                best = rew
                algo.save(best_dir)
            update_training_log(variant_name, i, num_iter, rew, True, log_file, out_dir)
        else:
            update_training_log(variant_name, i, num_iter, rew, False, log_file)
            
        if i >= num_iter:
            break
            
    algo.stop()
    print(f"[✓] 变体 {variant_name} 训练完成，best_reward={best:.3f}")


def main():
    ap = argparse.ArgumentParser(description="SA-HMARL 隔离式消融训练器")
    ap.add_argument("--variant", choices=list(VARIANTS) + ["all"], default="full",
                    help="要训练的消融变体；'all' 顺序训练全部变体")
    ap.add_argument("--iters", type=int, default=int(os.environ.get("NUM_ITER", 3000)))
    ap.add_argument("--ckpt-freq", type=int, default=10)
    ap.add_argument("--reverse", action="store_true", help="反向顺序训练 (结合 --variant all 使用)")
    ap.add_argument("--log-file", type=str, default="experiments/results/ablations/training_log.txt")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)

    ray.init(ignore_reinit_error=True, log_to_driver=True,
             num_cpus=14,
             object_store_memory=8 * 1024 ** 3,
             runtime_env={"env_vars": {
                 "PYTHONPATH": f"{_MY_METHOD}:{_HERE}:{os.pathsep}".rstrip(os.pathsep)
             }})
    variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    if args.reverse and args.variant == "all":
        variants.reverse()
    
    log_file = args.log_file
    # 如果是续训，不要清空之前的文件
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("================================================================\n")
            f.write(f"  消融训练启动 (总变体数: {len(variants)}, 任务文件: {log_file})\n")
            f.write("================================================================\n")
        
    for v in variants:
        train_variant(v, args.iters, args.ckpt_freq, log_file)
    ray.shutdown()


if __name__ == "__main__":
    main()
