"""
evaluate_heuristic.py
=====================
测试 MACPP 启发式基线 (Heuristic MACPP) ，在 10 个种子下建立性能下界。
"""

import os
import sys
import numpy as np
import pandas as pd
import math

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

SYS_ROOT = os.path.abspath(os.path.join(PROJECT_DIR, "..", "..", ".."))
if SYS_ROOT not in sys.path:
    sys.path.insert(0, SYS_ROOT)

from experiments.my_method.env_defs import RandomMapEnv, GRID_RES
from experiments.my_method.HierarchicalEnv import HierarchicalEnv
from heuristic_policy import HeuristicController

def main():
    print("=" * 65)
    print("  Heuristic MACPP 全局启发式基线跑分 (Performance Lower Bound)")
    print("=" * 65)
    
    seeds = list(range(101, 111))
    results = []
    
    os.makedirs(os.path.join(PROJECT_DIR, "results"), exist_ok=True)
    csv_path = os.path.join(PROJECT_DIR, "results", "heuristic_macpp_results.csv")
    
    for seed in seeds:
        print(f"\n[Seed {seed}] 正在评估启发式调度器...")
        map_env = RandomMapEnv(grid_resolution=GRID_RES, seed=seed)
        # use_resume_scan=False：Heuristic 不应拥有 SA-HMARL 独有的断点续扫能力
        env = HierarchicalEnv(map_env, use_resume_scan=False)
        controller = HeuristicController(env)
        
        obs, info = env.reset(seed=seed)
        
        step_count = 0
        deadhead_dist = 0.0
        
        while True:
            # 记录历史坐标用于计算 deadhead
            prev_coords = {}
            for a in ['uav_0', 'uav_1']:
                prev_coords[a] = (env.uavs[a]['x'], env.uavs[a]['y'])
                
            # 控制器决策
            actions = controller.get_actions(obs)
            
            # 环境推演
            obs, rewards, terminations, truncations, infos = env.step(actions)
            step_count += 1
            
            # 统计冗余飞行距离 (Deadhead distance: 非扫图状态下的移动距离)
            for a in ['uav_0', 'uav_1']:
                uav = env.uavs[a]
                if not uav.get('is_busy') or uav.get('is_returning') or uav.get('is_swapping'):
                    dist = math.hypot(uav['x'] - prev_coords[a][0], uav['y'] - prev_coords[a][1])
                    deadhead_dist += dist

            if any(terminations.values()) or any(truncations.values()):
                cov = env._compute_coverage_ratio() * 100.0
                is_crash = any("Crash" in str(infos[a].get('reason', '')) for a in env.agents if a in infos)
                
                print(f"  -> 结束！总步数 (Makespan): {step_count}")
                print(f"  -> 最终覆盖率: {cov:.2f}%")
                print(f"  -> UAV 死区冗余距离 (Deadhead): {deadhead_dist:.1f} m")
                if is_crash:
                    print(f"  -> ⚠️ 警告: 发生了电量耗尽坠毁！")
                    
                results.append({
                    "Algorithm": "Heuristic_MACPP",
                    "Seed": seed,
                    "Makespan": step_count,
                    "Deadhead_Distance": deadhead_dist,
                    "Crash": is_crash,
                    "Final_Coverage": cov
                })
                break
                
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print("\n" + "=" * 65)
    print(f"性能下界基准评估完成：")
    print(f"平均完工时间 (Avg Makespan): {df['Makespan'].mean():.1f} ± {df['Makespan'].std():.1f}")
    print(f"平均沉没距离 (Avg Deadhead): {df['Deadhead_Distance'].mean():.1f} ± {df['Deadhead_Distance'].std():.1f}")
    print(f"坠机率 (Crash Rate): {df['Crash'].mean() * 100:.1f}%")
    print(f"详细数据已保存至: {csv_path}")
    print("=" * 65)

if __name__ == "__main__":
    main()
