import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from experiments.my_method.env_defs import RandomMapEnv, GRID_RES
from experiments.my_method.HierarchicalEnvV2 import HierarchicalEnv
from experiments.baselines.porcelli_cacpp.porcelli_policy import PorcelliCACPPController

SEEDS = range(1001, 1011)

def run_seed(seed):
    env = HierarchicalEnv(RandomMapEnv(grid_resolution=GRID_RES, seed=seed),
                          use_resume_scan=False)
    controller = PorcelliCACPPController(env)
    obs, _ = env.reset(seed=seed)
    
    step_count = 0
    deadhead_dist = 0.0
    total_scan_dist = 0.0
    
    while True:
        actions = controller.get_actions(obs)
        obs, _, terms, truncs, _ = env.step(actions)
        step_count += 1
        
        for a in ['uav_0', 'uav_1']:
            u = env.uavs[a]
            if not u.get('is_busy') or u.get('is_returning') or u.get('is_swapping'):
                deadhead_dist += u['v_mag'] * 1.0  # approximate using speed
                # wait, in smoke test, we calculated using positions! Let's do it exactly as in smoke test.
            else:
                total_scan_dist += u['v_mag'] * 1.0
        
        if any(terms.values()) or any(truncs.values()):
            break

    cov = float(np.sum(env.coverage_grid)) / env.coverage_grid.size * 100
    _cov_area = float(np.sum(env.coverage_grid)) * GRID_RES * GRID_RES
    redundant = max(0.0, total_scan_dist - _cov_area / 50.0)
    
    return step_count, deadhead_dist, redundant, cov

if __name__ == "__main__":
    print("============================================================")
    print("Running Porcelli CACPP Evaluation over 10 seeds (1001-1010)")
    print("============================================================")
    
    makespans = []
    deadheads = []
    redundants = []
    
    for seed in SEEDS:
        print(f"Running seed {seed}...")
        
        # We need to re-implement the exact distance tracking from smoke test for accuracy
        env = HierarchicalEnv(RandomMapEnv(grid_resolution=GRID_RES, seed=seed), use_resume_scan=False)
        controller = PorcelliCACPPController(env)
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
                dx = u['x'] - prev_c[a][0]
                dy = u['y'] - prev_c[a][1]
                d = np.hypot(dx, dy)
                if not u.get('is_busy') or u.get('is_returning') or u.get('is_swapping'):
                    deadhead_dist += d
                else:
                    total_scan_dist += d
                prev_c[a] = (u['x'], u['y'])
            
            if any(terms.values()) or any(truncs.values()):
                break

        cov = float(np.sum(env.coverage_grid)) / env.coverage_grid.size * 100
        _cov_area = float(np.sum(env.coverage_grid)) * GRID_RES * GRID_RES
        redundant = max(0.0, total_scan_dist - _cov_area / 50.0)
        
        print(f"  -> Makespan: {step_count}, Deadhead: {deadhead_dist:.1f}, Redundant: {redundant:.1f}, Coverage: {cov:.1f}%")
        
        makespans.append(step_count)
        deadheads.append(deadhead_dist)
        redundants.append(redundant)
        
    print("============================================================")
    print("FINAL RESULTS (Porcelli CACPP, 10 seeds):")
    print(f"  Makespan:       {np.mean(makespans):.0f} ± {np.std(makespans, ddof=1):.0f} ticks")
    print(f"  Deadhead:       {np.mean(deadheads):.0f} ± {np.std(deadheads, ddof=1):.0f} m")
    print(f"  Redundant Scan: {np.mean(redundants):.0f} ± {np.std(redundants, ddof=1):.0f} m")
    print("============================================================")
