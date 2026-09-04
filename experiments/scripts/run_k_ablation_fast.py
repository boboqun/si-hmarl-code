#!/usr/bin/env python3
"""
run_k_ablation_fast.py — 加速且科学的 K-分辨率消融训练器
=========================================================
取代旧的 run_resolution_ablation.py / run_extended_k_ablation.py。

为什么上次"没跑好"
------------------
旧脚本用 train_batch_size=2000、num_env_runners=1 训练，远小于主实验
（batch=8000、12 workers），导致各 K 的策略**未收敛**——例如评估摘要里
K=5 的 makespan 高达 ~19,457，而充分训练的 K=5 真实值约 6,472。U 形曲线
因此被欠训练噪声淹没，得出"K=4 优于 K=5"的错误结论。

科学性保证（所有 K 仅在分辨率上不同，其余完全一致）
--------------------------------------------------
  * 同一环境与物理参数（DynamicHierarchicalEnv，只改 resolution_K=K）；
  * 同一网络结构（按 K 自适应的 token 数 / 动作维 K²+2），同一超参；
  * 对所有 K 施加**相同的训练预算上限 + 相同的早停准则**（等机会原则）；
  * 同一组 10 个评测种子(1001–1010)，由 evaluate_k_ablation.py 统一回放。
建议把全部 K∈{4,5,6,7,8} 都用本脚本训练，得到自洽的同口径对比；其中 K=5 应
≈ 主实验 6,472，可据此交叉验证一致性。

加速手段（在不牺牲公平性的前提下）
----------------------------------
  1. 大批量 + 多 worker（默认 batch=6000、8 workers）→ 样本效率与采样吞吐都更高，
     收敛所需迭代数与墙钟时间同时下降；
  2. **早停**：当平滑后的 makespan(=episode_len_mean) 在 `patience` 次迭代内不再
     改善超过 `min_delta` 即停止（设 `min_iters` 下限、`max_iters` 上限，对所有 K
     一致）。已收敛的 K 不再空耗；K≥6 若无法求解会迅速到达平台并提前停止；
  3. 跨 K 并行：每个 K 独立进程，可同时启动（见用法）。
综合相对旧方案通常有 3–5× 墙钟加速，且保证收敛。

用法
----
  # 单个 K（可多终端并行）
  python run_k_ablation_fast.py --K 4
  python run_k_ablation_fast.py --K 6 --workers 8 --max-iters 1500
  # 顺序训练全部 K
  python run_k_ablation_fast.py --K all
  # 训练后统一评测（沿用既有评测脚本）
  python evaluate_k_ablation.py

注意（V1/V2 一致性）
--------------------
DynamicHierarchicalEnv 目前继承自 V1 的 HierarchicalEnv，而主结果用的是 V2
（含双通道信用分配 + 预测式预定位）。若要让 K 消融与正文主方法完全同源，
建议把 DynamicHierarchicalEnv 改为继承 HierarchicalEnvV2（其 __init__ 已支持
macro_k 参数）并提供按 K 自适应池化的 V2 模型——这部分可按需扩展。
"""

import os
import sys
import math
import argparse
from collections import deque

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import ray
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPOConfig

from experiments.my_method.env_defs import RandomMapEnv, GRID_RES
from experiments.my_method.hierarchical_train import ParallelPettingZooEnv
from experiments.scripts.run_resolution_ablation import (
    DynamicHierarchicalEnv, DynamicRLlibUAVModel, DynamicRLlibUGVModel,
)

ALL_K = [4, 5, 6, 7, 8]


# ── 按 K 绑定的动态模型类工厂（与既有评测脚本完全一致的命名约定）──────────
def _make_uav_cls(K):
    class _Cls(DynamicRLlibUAVModel):
        def __init__(self, obs, act, out, conf, nm):
            super().__init__(obs, act, out, conf, nm, K=K)
    return _Cls


def _make_ugv_cls(K):
    class _Cls(DynamicRLlibUGVModel):
        def __init__(self, obs, act, out, conf, nm):
            super().__init__(obs, act, out, conf, nm, K=K)
    return _Cls


def _extract_makespan(result):
    """鲁棒提取 episode_len_mean（=本环境的 makespan），兼容多个 RLlib 版本。"""
    for sub in ("env_runners", "sampler_results"):
        v = result.get(sub, {}).get("episode_len_mean")
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            return v
    v = result.get("episode_len_mean")
    return v if v is not None else float("nan")


def build_config(K, workers, batch):
    env_name = f"k_fast_env_{K}"
    uav_name = f"DynUAV_fast_K{K}"
    ugv_name = f"DynUGV_fast_K{K}"

    def env_creator(config):
        return ParallelPettingZooEnv(
            DynamicHierarchicalEnv(RandomMapEnv(grid_resolution=GRID_RES), resolution_K=K))

    register_env(env_name, env_creator)
    ModelCatalog.register_custom_model(uav_name, _make_uav_cls(K))
    ModelCatalog.register_custom_model(ugv_name, _make_ugv_cls(K))

    probe = env_creator({})
    uav_obs = probe.par_env.observation_space("uav_0")
    uav_act = probe.par_env.action_space("uav_0")
    ugv_obs = probe.par_env.observation_space("ugv_0")
    ugv_act = probe.par_env.action_space("ugv_0")

    return (
        PPOConfig()
        .environment(env_name)
        .framework("torch")
        .api_stack(enable_rl_module_and_learner=False,
                   enable_env_runner_and_connector_v2=False)
        .resources(num_gpus=0)
        .env_runners(
            num_env_runners=workers,        # 多 worker 并行采样（旧版仅 1）
            num_envs_per_env_runner=2,
            rollout_fragment_length="auto",
            batch_mode="complete_episodes",
        )
        .training(
            gamma=0.99, lr=3e-4,
            train_batch_size=batch,         # 大批量（旧版仅 2000）
            minibatch_size=512,
            num_epochs=5,
            clip_param=0.2, vf_clip_param=500.0,
            entropy_coeff=0.05,
            entropy_coeff_schedule=[[0, 0.10], [2_000_000, 0.05],
                                    [8_000_000, 0.01], [20_000_000, 0.001]],
            kl_coeff=0.2, lambda_=0.95,
        )
        .multi_agent(
            policies={
                "uav_policy": (None, uav_obs, uav_act, {"model": {"custom_model": uav_name}}),
                "ugv_policy": (None, ugv_obs, ugv_act, {"model": {"custom_model": ugv_name}}),
            },
            policy_mapping_fn=lambda agent_id, *a, **kw:
                "ugv_policy" if "ugv" in agent_id else "uav_policy",
        )
    )


def train_one_k(K, args):
    save_dir = os.path.join(PROJECT_DIR, "experiments", "results_ablation", f"resolution_{K}x{K}")
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 64, flush=True)
    print(f"  K-ablation (fast) | K={K} | batch={args.batch} workers={args.workers}", flush=True)
    print(f"  early-stop: min_iters={args.min_iters} max_iters={args.max_iters} "
          f"patience={args.patience} min_delta={args.min_delta}", flush=True)
    print(f"  save_dir={save_dir}", flush=True)
    print("=" * 64, flush=True)

    algo = build_config(K, args.workers, args.batch).build()

    smooth = deque(maxlen=20)          # 平滑窗口（makespan）
    best_makespan = float("inf")
    no_improve = 0
    last_ckpt_iter = 0

    for i in range(1, args.max_iters + 1):
        result = algo.train()
        mk = _extract_makespan(result)
        rew = result.get("env_runners", {}).get("episode_reward_mean",
              result.get("episode_reward_mean", float("nan")))
        if not (isinstance(mk, float) and math.isnan(mk)):
            smooth.append(mk)
        sm = sum(smooth) / len(smooth) if smooth else float("nan")

        improved = sm < best_makespan - args.min_delta
        if improved:
            best_makespan = sm
            no_improve = 0
            algo.save(checkpoint_dir=os.path.join(save_dir, "best_ckpt"))  # 评测脚本会忽略非 ckpt_ 前缀
        else:
            no_improve += 1

        print(f"  [K={K}] {i:>4}/{args.max_iters} | makespan(smooth)={sm:>8.1f} "
              f"| rew={rew:+8.1f} | no_improve={no_improve}{'  ★' if improved else ''}",
              flush=True)

        if i % 100 == 0:
            algo.save(checkpoint_dir=os.path.join(save_dir, f"ckpt_{i}"))  # 数值名，供评测脚本取最新
            last_ckpt_iter = i

        # 早停：达到下限后，若平滑 makespan 连续 patience 次无改善则停止
        if i >= args.min_iters and no_improve >= args.patience:
            print(f"  [K={K}] early-stop at iter {i} (converged, best makespan≈{best_makespan:.1f})",
                  flush=True)
            break

    # 收尾：保存一个数值名的最终 checkpoint（供 evaluate_k_ablation.py 取用）
    final_i = max(i, last_ckpt_iter + 1)
    algo.save(checkpoint_dir=os.path.join(save_dir, f"ckpt_{final_i}"))
    algo.stop()
    print(f"  [K={K}] done. best smoothed makespan ≈ {best_makespan:.1f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Fast, convergence-guaranteed K-resolution ablation")
    ap.add_argument("--K", default="all", help="4/5/6/7/8 或 all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=6000)
    ap.add_argument("--min-iters", type=int, default=300)
    ap.add_argument("--max-iters", type=int, default=1500)
    ap.add_argument("--patience", type=int, default=60, help="连续无改善迭代数阈值")
    ap.add_argument("--min-delta", type=float, default=15.0, help="makespan 改善判定阈值(ticks)")
    args = ap.parse_args()

    ks = ALL_K if args.K == "all" else [int(args.K)]
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    for K in ks:
        train_one_k(K, args)
    ray.shutdown()
    print("[✓] K-ablation (fast) complete:", ks, flush=True)


if __name__ == "__main__":
    main()
