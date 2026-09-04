#!/usr/bin/env python3
"""
reeval_heuristic_only.py
========================
仅重新评测 Heuristic MACPP（随机派发版），并将结果写回 performance_metrics.csv，
保持其余算法（SA-HMARL / Standard H-MARL / MAPPO Flat）的历史数据不变。
"""

import os
import sys
import math
import numpy as np
import pandas as pd

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from experiments.my_method.env_defs import RandomMapEnv, GRID_RES
from experiments.my_method.HierarchicalEnv import HierarchicalEnv
from experiments.baselines.heuristic_macpp.heuristic_policy import HeuristicController

RESULTS_CSV = os.path.join(PROJECT_DIR, "experiments", "results", "performance_metrics.csv")
TEST_SEEDS  = list(range(1001, 1011))
UAV_SCAN_WIDTH = 50.0   # 与 HierarchicalEnv.py 保持一致

HEURISTIC_LABEL = "Heuristic\nMACPP"   # 与 CSV 原始标签保持一致


def eval_heuristic():
    records = []
    print(f"\n[*] 正在评测 Heuristic MACPP（随机派发版），共 {len(TEST_SEEDS)} 个种子...\n")

    for seed in TEST_SEEDS:
        env = HierarchicalEnv(
            RandomMapEnv(grid_resolution=GRID_RES, seed=seed),
            use_resume_scan=False   # Heuristic 无断点续扫，公平对照
        )
        controller = HeuristicController(env)
        obs, _ = env.reset(seed=seed)

        step_count = 0
        deadhead_dist = 0.0
        total_scan_dist = 0.0
        prev_c = {a: (env.uavs[a]['x'], env.uavs[a]['y']) for a in ['uav_0', 'uav_1']}

        while True:
            actions = controller.get_actions(obs)
            obs, _, terms, truncs, _ = env.step(actions)
            step_count += 1

            for a in ['uav_0', 'uav_1']:
                u = env.uavs[a]
                d = math.hypot(u['x'] - prev_c[a][0], u['y'] - prev_c[a][1])
                if not u.get('is_busy') or u.get('is_returning') or u.get('is_swapping'):
                    deadhead_dist += d
                else:
                    total_scan_dist += d
                prev_c[a] = (u['x'], u['y'])

            if any(terms.values()) or any(truncs.values()):
                break

        _cov_area = float(np.sum(env.coverage_grid)) * GRID_RES * GRID_RES
        redundant_scan_dist = max(0.0, total_scan_dist - _cov_area / UAV_SCAN_WIDTH)
        total_wasted = deadhead_dist + redundant_scan_dist

        print(f"  Seed {seed}: Makespan={step_count:>6}  "
              f"Deadhead={deadhead_dist:>10.1f}m  "
              f"RedundantScan={redundant_scan_dist:>10.1f}m  "
              f"TotalWasted={total_wasted:>11.1f}m")

        records.append({
            "Algorithm":             HEURISTIC_LABEL,
            "Seed":                  seed,
            "Global Makespan":       step_count,
            "Deadhead Distance (m)": deadhead_dist,
            "Redundant Scan (m)":    redundant_scan_dist,
            "Total Wasted (m)":      total_wasted,
        })

    return pd.DataFrame(records)


def merge_and_save(new_df):
    if os.path.exists(RESULTS_CSV):
        old_df = pd.read_csv(RESULTS_CSV)
        # 删除旧 Heuristic 行（兼容多行标签格式）
        mask = old_df["Algorithm"].str.contains("Heuristic", na=False)
        old_df = old_df[~mask]
        merged = pd.concat([old_df, new_df], ignore_index=True)
    else:
        merged = new_df

    merged.to_csv(RESULTS_CSV, index=False)
    print(f"\n[✓] 已将新 Heuristic 数据合并写入: {RESULTS_CSV}")

    # 打印摘要统计
    print("\n── 新 Heuristic MACPP 摘要 ──")
    print(f"  Global Makespan     : {new_df['Global Makespan'].mean():.0f} ± {new_df['Global Makespan'].std():.0f} steps")
    print(f"  Deadhead Distance   : {new_df['Deadhead Distance (m)'].mean():.1f} ± {new_df['Deadhead Distance (m)'].std():.1f} m")
    print(f"  Redundant Scan      : {new_df['Redundant Scan (m)'].mean():.1f} ± {new_df['Redundant Scan (m)'].std():.1f} m")
    print(f"  Total Wasted        : {new_df['Total Wasted (m)'].mean():.1f} ± {new_df['Total Wasted (m)'].std():.1f} m")

    # 与全部算法对比
    print("\n── 全算法当前均值对比 ──")
    summary = merged.groupby("Algorithm")["Global Makespan"].agg(["mean", "std"]).reset_index()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    new_heuristic_df = eval_heuristic()
    merge_and_save(new_heuristic_df)
    print("\n[✓] 完成。可重新运行 evaluate_performance.py 的 plot_academic_charts 刷新图表。")
