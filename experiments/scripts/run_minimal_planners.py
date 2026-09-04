#!/usr/bin/env python3
import os
import sys
import math
import time
import argparse
import csv
import json
import numpy as np

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, "experiments", "results")

def append_csv(csv_path, method, scale_name, scale_mult, n_success, makespans, n_total):
    sr = n_success / max(1, n_total) * 100
    mean_ms = np.mean(makespans) if makespans else float("nan")
    std_ms = np.std(makespans) if len(makespans) > 1 else 0.0
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([method, scale_name, scale_mult, f"{sr:.1f}",
                          f"{mean_ms:.1f}" if not np.isnan(mean_ms) else "nan",
                          f"{std_ms:.1f}", n_total])

def eval_heuristic(env_factory, controller_class, seeds, max_steps):
    results = []
    for seed in seeds:
        env = env_factory(seed)
        controller = controller_class(env)
        obs, _ = env.reset(seed=seed)
        step_count = 0
        success = False

        while step_count < max_steps:
            actions = controller.get_actions(obs)
            obs, _, terms, truncs, _ = env.step(actions)
            step_count += 1
            if any(terms.values()):
                if env._compute_coverage_ratio() >= 0.98:
                    success = True
                break
            if any(truncs.values()):
                break

        cov = env._compute_coverage_ratio()
        results.append({"success": success, "makespan": step_count, "cov": cov})
        status = "OK" if success else f"FAIL(cov={cov:.2f})"
        print(f"      seed {seed}: {status}, steps={step_count}")
    return results

def main():
    seeds = list(range(1001, 1011))
    scales = [[0.25, "0.25x (500m)"], [0.5, "0.5x (1000m)"]]
    csv_path = os.path.join(RESULTS_DIR, "scalability_results_20260611_084847.csv")
    
    heuristic_controllers = {}
    heuristic_list = [
        ("AG-CVG", "experiments.baselines.ag_cvg.ag_cvg_policy", "AGCVGController"),
        ("Eker DP", "experiments.baselines.eker_dp.eker_dp_policy", "EkerDPController"),
        ("Porcelli CACPP", "experiments.baselines.porcelli_cacpp.porcelli_policy", "PorcelliCACPPController"),
    ]
    for name, module_path, class_name in heuristic_list:
        mod = __import__(module_path, fromlist=[class_name])
        heuristic_controllers[name] = getattr(mod, class_name)

    import re
    import importlib
    TARGET_FILES = [
        "experiments/my_method/env_defs.py",
        "experiments/my_method/HierarchicalEnv.py",
        "experiments/my_method/HierarchicalEnvV2.py",
    ]

    def patch_and_reload(scale_mult):
        target_map = 2000.0 * scale_mult
        target_res = 100.0 * scale_mult
        area_mult = scale_mult ** 2
        target_max_steps = int(max(30000, 30000 * area_mult))
        min_nodes = max(5, int(20 * area_mult))
        max_nodes = max(10, int(50 * area_mult))

        for relpath in TARGET_FILES:
            fp = os.path.join(PROJECT_DIR, relpath)
            if not os.path.exists(fp): continue
            with open(fp, 'r') as f: content = f.read()
            content = re.sub(r'MAP_SIZE\s*=\s*[\d\.]+', f'MAP_SIZE         = {target_map}', content)
            content = re.sub(r'GRID_RES\s*=\s*[\d\.]+', f'GRID_RES  = {target_res}', content)
            content = re.sub(r'MAX_EPISODE_STEPS\s*=\s*\d+', f'MAX_EPISODE_STEPS = {target_max_steps}', content)
            content = re.sub(r'rng\.randint\(\d+,\s*\d+\)', f'rng.randint({min_nodes}, {max_nodes})', content)
            with open(fp, 'w') as f: f.write(content)

        for mod_name in ["experiments.my_method.env_defs", "experiments.my_method.HierarchicalEnvV2", "experiments.my_method.HierarchicalEnv"]:
            if mod_name in sys.modules:
                try: importlib.reload(sys.modules[mod_name])
                except Exception: pass

    try:
        for scale_mult, scale_name in scales:
            patch_and_reload(scale_mult)
            env_defs = importlib.import_module("experiments.my_method.env_defs")
            grid_res = env_defs.GRID_RES
            area_mult = scale_mult ** 2
            max_steps = int(max(30000, 30000 * area_mult))

            for name, ctrl_cls in heuristic_controllers.items():
                print(f"\n  [{name}] scale={scale_name} ...")
                t0 = time.time()
                def h_factory(seed, gr=grid_res):
                    HierEnv = importlib.import_module("experiments.my_method.HierarchicalEnvV2").HierarchicalEnv
                    RMapEnv = importlib.import_module("experiments.my_method.env_defs").RandomMapEnv
                    return HierEnv(RMapEnv(grid_resolution=gr, seed=seed), use_resume_scan=False)

                results = eval_heuristic(h_factory, ctrl_cls, seeds, max_steps)
                elapsed = time.time() - t0
                n_success = sum(r["success"] for r in results)
                makespans = [r["makespan"] for r in results if r["success"]]
                append_csv(csv_path, name, scale_name, scale_mult, n_success, makespans, len(seeds))
    finally:
        patch_and_reload(1.0)

if __name__ == "__main__":
    main()
