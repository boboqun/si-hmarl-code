"""
Full 10-seed evaluation for AG-CVG and Eker DP baselines only.
Appends results to existing performance_metrics.csv.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import math
import time
import numpy as np
import pandas as pd

from experiments.my_method.env_defs import RandomMapEnv, GRID_RES
from experiments.my_method.HierarchicalEnvV2 import HierarchicalEnv
from experiments.baselines.ag_cvg.ag_cvg_policy import AGCVGController
from experiments.baselines.eker_dp.eker_dp_policy import EkerDPController
from experiments.baselines.porcelli_cacpp.porcelli_policy import PorcelliCACPPController

TEST_SEEDS = list(range(1001, 1011))
UAV_SCAN_WIDTH = 50.0
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'experiments', 'results')

METHODS = {
    "AG_CVG": ("AG-CVG\n(Karapetyan 2024)", AGCVGController),
    "Eker_DP": ("Eker DP\n(Eker 2025)", EkerDPController),
    "Porcelli_CACPP": ("Porcelli-CACPP*\n(Porcelli 2025)", PorcelliCACPPController),
}


def evaluate_method(method_id, label, controller_class):
    records = []
    print(f"\n{'='*60}")
    print(f"Evaluating: {label.replace(chr(10), ' ')}")
    print(f"{'='*60}")
    
    for seed in TEST_SEEDS:
        t0 = time.time()
        env = HierarchicalEnv(RandomMapEnv(grid_resolution=GRID_RES, seed=seed),
                              use_resume_scan=False)
        controller = controller_class(env)
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
        elapsed = time.time() - t0
        
        print(f"  Seed {seed}: Makespan={step_count}, Deadhead={deadhead_dist:.0f}, "
              f"Redundant={redundant_scan_dist:.0f}, Time={elapsed:.1f}s")
        
        records.append({
            "Algorithm": label,
            "Seed": seed,
            "Global Makespan": step_count,
            "Deadhead Distance (m)": deadhead_dist,
            "Redundant Scan (m)": redundant_scan_dist,
            "Total Wasted (m)": deadhead_dist + redundant_scan_dist,
        })
    
    return records


if __name__ == "__main__":
    all_records = []
    
    for method_id, (label, cls) in METHODS.items():
        records = evaluate_method(method_id, label, cls)
        all_records.extend(records)
    
    df_new = pd.DataFrame(all_records)
    
    # Load existing results and append
    csv_path = os.path.join(RESULTS_DIR, "performance_metrics.csv")
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        # Remove any existing AG-CVG / Eker DP rows
        df_old = df_old[~df_old['Algorithm'].str.contains('AG-CVG|Eker DP', na=False)]
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_combined = df_new
    
    df_combined.to_csv(csv_path, index=False)
    print(f"\n✅ Results saved to: {csv_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY (mean ± std)")
    print("="*80)
    for label in df_new['Algorithm'].unique():
        sub = df_new[df_new['Algorithm'] == label]
        print(f"\n{label.replace(chr(10), ' ')}:")
        for metric in ['Global Makespan', 'Deadhead Distance (m)', 'Redundant Scan (m)']:
            m = sub[metric].mean()
            s = sub[metric].std()
            print(f"  {metric}: {m:.0f} ± {s:.0f}")
