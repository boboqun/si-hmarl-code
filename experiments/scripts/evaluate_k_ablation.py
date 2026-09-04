#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_k_ablation.py
======================
真实 rollout 评估，逐 K 统计 Makespan / Deadhead / Redundant Scan。

数据来源:
  ─ K ≠ 5: experiments/results_ablation/resolution_{K}x{K}/ckpt_*
            用 DynamicHierarchicalEnv + Dynamic 模型类（消融训练配置）
  ─ K = 5: experiments/results/my_method/checkpoints/
            用主 HierarchicalEnv(macro_k=5) + RLlibUGVModel（主训练配置）

结果追加写入 experiments/results_ablation/k_ablation_eval.csv（支持断点续跑）。
"""

import os, sys, math
import numpy as np
import pandas as pd
from typing import Optional

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from experiments.my_method.env_defs import RandomMapEnv, GRID_RES
from experiments.my_method.HierarchicalEnv import UAV_SCAN_WIDTH

K_VALUES   = [4, 5, 6, 7, 8]
EVAL_SEEDS = list(range(1001, 1011))     # 10 个独立随机种子
MAX_STEPS  = 10000                       # 单 episode 步数上限（防策略未收敛死循环）

ABLATION_DIR = os.path.join(PROJECT_DIR, "experiments", "results_ablation")
MAIN_CKPT    = os.path.join(PROJECT_DIR, "experiments", "results",
                             "my_method", "checkpoints")
OUTPUT_CSV   = os.path.join(ABLATION_DIR, "k_ablation_eval.csv")


# ──────────────────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────────────────

def _latest_ablation_ckpt(k: int) -> Optional[str]:
    base = os.path.join(ABLATION_DIR, f"resolution_{k}x{k}")
    if not os.path.isdir(base):
        return None
    ckpts = sorted(
        [d for d in os.listdir(base) if d.startswith("ckpt_")],
        key=lambda x: int(x.split("_")[1])
    )
    return os.path.join(base, ckpts[-1]) if ckpts else None


def _already_done(k: int) -> bool:
    """断点续跑：CSV 中已有该 K 的完整数据则跳过。"""
    if not os.path.exists(OUTPUT_CSV):
        return False
    try:
        df = pd.read_csv(OUTPUT_CSV)
        return int((df["K"] == k).sum()) >= len(EVAL_SEEDS)
    except Exception:
        return False


def _rollout_metrics(env, ugv_policy, uav_policy, seed: int) -> dict:
    """
    在给定 env 上跑一个 episode，返回 makespan / deadhead / redundant_scan。
    """
    obs_dict, _ = env.reset(seed=seed)
    step_count   = 0
    deadhead     = 0.0
    scan_dist    = 0.0
    prev_pos     = {a: (env.uavs[a]['x'], env.uavs[a]['y'])
                    for a in ['uav_0', 'uav_1']}
    terminated   = False

    while step_count < MAX_STEPS:
        actions = {}
        for aid, obs in obs_dict.items():
            if aid == 'ugv_0':
                act, _, _ = ugv_policy.compute_single_action(obs)
            else:
                act, _, _ = uav_policy.compute_single_action(obs)
            actions[aid] = act

        obs_dict, _, terms, truncs, _ = env.step(actions)
        step_count += 1

        for a in ['uav_0', 'uav_1']:
            u = env.uavs[a]
            d = math.hypot(u['x'] - prev_pos[a][0],
                           u['y'] - prev_pos[a][1])
            if not u.get('is_busy') or u.get('is_returning') or u.get('is_swapping'):
                deadhead  += d
            else:
                scan_dist += d
            prev_pos[a] = (u['x'], u['y'])

        if any(terms.values()) or any(truncs.values()):
            terminated = True
            break

    covered_area   = float(np.sum(env.coverage_grid)) * GRID_RES * GRID_RES
    redundant_scan = max(0.0, scan_dist - covered_area / UAV_SCAN_WIDTH)
    total_wasted   = deadhead + redundant_scan
    status         = "OK" if terminated else "MAX_STEPS"

    print(f"  Seed {seed} [{status}]: Steps={step_count}, "
          f"Deadhead={deadhead:.0f}m, "
          f"RedundantScan={redundant_scan:.0f}m, "
          f"TotalWasted={total_wasted:.0f}m")

    return {
        "Global Makespan":        step_count,
        "Deadhead Distance (m)":  deadhead,
        "Redundant Scan (m)":     redundant_scan,
        "Total Wasted (m)":       total_wasted,
        "Terminated":             terminated,
    }


# ──────────────────────────────────────────────────────────────────────────────
# K=5 专用：主 SA-HMARL checkpoint + HierarchicalEnv(macro_k=5)
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_k5_main(seeds: list) -> list:
    """用主训练 checkpoint 评估 K=5，数据与消融实验同口径。"""
    from ray.rllib.policy.policy import Policy
    from ray.rllib.models import ModelCatalog
    from experiments.my_method.hierarchical_train import RLlibUGVModel, RLlibUAVModel
    from experiments.my_method.HierarchicalEnv import HierarchicalEnv

    ckpt_dir = MAIN_CKPT
    if not os.path.isdir(ckpt_dir):
        print(f"  [!] 主 checkpoint 未找到: {ckpt_dir}")
        return []

    print(f"\n[K=5] 加载主 checkpoint: {os.path.basename(ckpt_dir)}")
    ModelCatalog.register_custom_model("RLlibUGVModel", RLlibUGVModel)
    ModelCatalog.register_custom_model("RLlibUAVModel", RLlibUAVModel)

    ugv_policy = Policy.from_checkpoint(
        os.path.join(ckpt_dir, "policies", "ugv_policy"))
    uav_policy = Policy.from_checkpoint(
        os.path.join(ckpt_dir, "policies", "uav_policy"))

    records = []
    for seed in seeds:
        map_env = RandomMapEnv(grid_resolution=GRID_RES, seed=seed)
        env     = HierarchicalEnv(map_env, macro_k=5)
        metrics = _rollout_metrics(env, ugv_policy, uav_policy, seed)
        records.append({"K": 5, "Seed": seed, **metrics})

    return records


# ──────────────────────────────────────────────────────────────────────────────
# K ≠ 5：消融 checkpoint + DynamicHierarchicalEnv
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_k_ablation_run(k: int, seeds: list) -> list:
    from ray.rllib.policy.policy import Policy
    from ray.rllib.models import ModelCatalog
    from experiments.scripts.run_resolution_ablation import (
        DynamicHierarchicalEnv, DynamicRLlibUAVModel, DynamicRLlibUGVModel)

    ckpt_dir = _latest_ablation_ckpt(k)
    if ckpt_dir is None:
        print(f"  [!] K={k}: 未找到消融 checkpoint，跳过。")
        return []

    print(f"\n[K={k}] 加载消融 checkpoint: {os.path.basename(ckpt_dir)}")

    def make_uav_cls(K_val):
        class _Cls(DynamicRLlibUAVModel):
            def __init__(self, obs, act, out, conf, nm):
                super().__init__(obs, act, out, conf, nm, K=K_val)
        return _Cls

    def make_ugv_cls(K_val):
        class _Cls(DynamicRLlibUGVModel):
            def __init__(self, obs, act, out, conf, nm):
                super().__init__(obs, act, out, conf, nm, K=K_val)
        return _Cls

    ModelCatalog.register_custom_model("DynamicRLlibUAVModel", make_uav_cls(k))
    ModelCatalog.register_custom_model("DynamicRLlibUGVModel", make_ugv_cls(k))

    ugv_policy = Policy.from_checkpoint(
        os.path.join(ckpt_dir, "policies", "ugv_policy"))
    uav_policy = Policy.from_checkpoint(
        os.path.join(ckpt_dir, "policies", "uav_policy"))

    records = []
    for seed in seeds:
        map_env = RandomMapEnv(grid_resolution=GRID_RES, seed=seed)
        env     = DynamicHierarchicalEnv(map_env, resolution_K=k)
        metrics = _rollout_metrics(env, ugv_policy, uav_policy, seed)
        records.append({"K": k, "Seed": seed, **metrics})

    return records


# ──────────────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  K-Resolution 消融实验 —— 真实 checkpoint rollout 评估")
    print(f"  MAX_STEPS/episode = {MAX_STEPS}  |  Seeds = {len(EVAL_SEEDS)}")
    print("=" * 62)

    os.makedirs(ABLATION_DIR, exist_ok=True)

    for k in K_VALUES:
        if _already_done(k):
            print(f"\n[K={k}] 已有完整数据，跳过（断点续跑）。")
            continue

        # 选择数据来源
        if k == 5:
            recs = evaluate_k5_main(EVAL_SEEDS)
        else:
            recs = evaluate_k_ablation_run(k, EVAL_SEEDS)

        if not recs:
            continue

        # 逐 K 追加写入，中途崩溃不丢数据
        df_k = pd.DataFrame(recs)
        if os.path.exists(OUTPUT_CSV):
            df_old = pd.read_csv(OUTPUT_CSV)
            df_old = df_old[df_old["K"] != k]
            df_all = pd.concat([df_old, df_k], ignore_index=True)
        else:
            df_all = df_k
        df_all.sort_values(["K", "Seed"]).to_csv(OUTPUT_CSV, index=False)
        print(f"  [✓] K={k} 数据已写入 CSV")

    if not os.path.exists(OUTPUT_CSV):
        print("\n[!] 无有效结果，退出。")
        return

    # 最终摘要
    df = pd.read_csv(OUTPUT_CSV)
    summary = df.groupby("K").agg(
        Steps_mean=("Global Makespan", "mean"),
        Steps_std=("Global Makespan", "std"),
        Deadhead_mean=("Deadhead Distance (m)", "mean"),
        Deadhead_std=("Deadhead Distance (m)", "std"),
        Wasted_mean=("Total Wasted (m)", "mean"),
    ).round(1)
    print("\n=== K 值消融评估摘要 ===")
    print(summary.to_string())
    print("=" * 62)


if __name__ == "__main__":
    main()
