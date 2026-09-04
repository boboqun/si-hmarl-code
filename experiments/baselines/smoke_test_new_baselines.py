"""
Quick smoke test for AG-CVG and Eker DP baselines.
Runs 1 seed each to verify correctness before full evaluation.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import math
import numpy as np

from experiments.my_method.env_defs import RandomMapEnv, GRID_RES
from experiments.my_method.HierarchicalEnvV2 import HierarchicalEnv
from experiments.baselines.ag_cvg.ag_cvg_policy import AGCVGController
from experiments.baselines.eker_dp.eker_dp_policy import EkerDPController
from experiments.baselines.porcelli_cacpp.porcelli_policy import PorcelliCACPPController

SEED = 1001

def run_test(method_name, controller_class):
    env = HierarchicalEnv(RandomMapEnv(grid_resolution=GRID_RES, seed=SEED),
                          use_resume_scan=False)
    controller = controller_class(env)
    obs, _ = env.reset(seed=SEED)
    
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
        
        if step_count % 5000 == 0:
            cov = float(np.sum(env.coverage_grid)) / env.coverage_grid.size * 100
            print(f"  [{method_name}] t={step_count}, coverage={cov:.1f}%")
        
        if any(terms.values()) or any(truncs.values()):
            break
    
    cov = float(np.sum(env.coverage_grid)) / env.coverage_grid.size * 100
    _cov_area = float(np.sum(env.coverage_grid)) * GRID_RES * GRID_RES
    redundant = max(0.0, total_scan_dist - _cov_area / 50.0)
    
    print(f"\n{'='*60}")
    print(f"[{method_name}] RESULT (seed={SEED}):")
    print(f"  Makespan:       {step_count}")
    print(f"  Deadhead:       {deadhead_dist:.0f} m")
    print(f"  Redundant Scan: {redundant:.0f} m")
    print(f"  Coverage:       {cov:.1f}%")
    print(f"{'='*60}\n")
    
    return step_count


if __name__ == "__main__":
    print("="*60)
    print("SMOKE TEST: AG-CVG Baseline")
    print("="*60)
    t1 = run_test("AG-CVG", AGCVGController)
    
    print("="*60)
    print("SMOKE TEST: Eker DP Baseline")
    print("="*60)
    t2 = run_test("Eker DP", EkerDPController)
    
    print("="*60)
    print("SMOKE TEST: Porcelli CACPP Baseline")
    print("="*60)
    t3 = run_test("Porcelli CACPP", PorcelliCACPPController)
    
    print(f"\n✅ All baselines completed. AG-CVG={t1} ticks, Eker DP={t2} ticks, Porcelli={t3} ticks")
