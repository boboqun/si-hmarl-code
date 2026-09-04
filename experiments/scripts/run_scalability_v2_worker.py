#!/usr/bin/env python3
"""
run_scalability_v2_worker.py
============================
Two-phase worker:
  Phase 1: Load ALL RL checkpoints under original 2km constants (before any patching).
  Phase 2: For EACH scale, patch constants, create fresh envs, run inference.

This avoids the bug where Algorithm.from_checkpoint tries to create rollout workers
with mismatched dimensions.
"""
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

# Also inject mappo_flat dir for pickle compatibility
_flat_dir = os.path.join(PROJECT_DIR, "experiments", "baselines", "mappo_flat")
if _flat_dir not in sys.path:
    sys.path.insert(0, _flat_dir)
_pp = os.environ.get("PYTHONPATH", "")
if _flat_dir not in _pp:
    os.environ["PYTHONPATH"] = _flat_dir + os.pathsep + PROJECT_DIR + os.pathsep + _pp

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


def eval_rl(algo, raw_env_factory, seeds, max_steps):
    """Evaluate an RL method by manually creating envs and calling compute_single_action."""
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    results = []
    for seed in seeds:
        raw_env = raw_env_factory(seed)
        env = ParallelPettingZooEnv(raw_env)
        obs, _ = env.reset(seed=seed)
        step_count = 0
        success = False

        while step_count < max_steps:
            actions = {}
            for agent_id, agent_obs in obs.items():
                policy_id = "ugv_policy" if "ugv" in agent_id else "uav_policy"
                actions[agent_id] = algo.compute_single_action(
                    agent_obs, policy_id=policy_id, explore=False)
            obs, _, terms, truncs, _ = env.step(actions)
            step_count += 1
            if any(terms.values()):
                if raw_env._compute_coverage_ratio() >= 0.98:
                    success = True
                break
            if any(truncs.values()):
                break

        cov = raw_env._compute_coverage_ratio()
        results.append({"success": success, "makespan": step_count, "cov": cov})
        status = "OK" if success else f"FAIL(cov={cov:.2f})"
        print(f"      seed {seed}: {status}, steps={step_count}")
    return results


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--n_seeds", type=int, default=30)
    parser.add_argument("--scales", type=str, required=True,
                        help="JSON list of [mult, name] pairs")
    args = parser.parse_args()

    seeds = list(range(1001, 1001 + args.n_seeds))
    scales = json.loads(args.scales)  # e.g. [[2.0, "2x (4000m)"], [4.0, "4x (8000m)"]]

    # ════════════════════════════════════════════════════════
    #  PHASE 1: Load ALL RL checkpoints under ORIGINAL 2km constants
    # ════════════════════════════════════════════════════════
    print("[Phase 1] Loading RL checkpoints under original 2km constants...")

    import ray
    ray.init(ignore_reinit_error=True, log_to_driver=False)

    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.models import ModelCatalog
    from ray.tune.registry import register_env
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

    # Import model classes
    from experiments.my_method.env_defs import RandomMapEnv, GRID_RES
    from experiments.my_method.hierarchical_train_v2 import (
        RLlibUGVModel as V2UGVModel, RLlibUAVModel as V2UAVModel)
    from experiments.baselines.standard_hmarl.train_standard_fair import (
        RLlibUGVModel as StdUGV, RLlibUAVModel as StdUAV)
    from experiments.baselines.mappo_flat.flat_models import (
        RLlibFlatUAVModel, RLlibFlatUGVModel)

    # Register non-conflicting model names
    ModelCatalog.register_custom_model("StandardRLlibUGVModel", StdUGV)
    ModelCatalog.register_custom_model("StandardRLlibUAVModel", StdUAV)
    ModelCatalog.register_custom_model("RLlibFlatUAVModel", RLlibFlatUAVModel)
    ModelCatalog.register_custom_model("RLlibFlatUGVModel", RLlibFlatUGVModel)

    # Register env creators (at 2km scale, for checkpoint loading only)
    def _ec_sa(config):
        from experiments.my_method.HierarchicalEnvV2 import HierarchicalEnv
        return ParallelPettingZooEnv(
            HierarchicalEnv(RandomMapEnv(grid_resolution=GRID_RES, seed=config.get("seed", 42))))
    def _ec_std(config):
        from experiments.baselines.standard_hmarl.StandardEnv import StandardEnv
        return ParallelPettingZooEnv(
            StandardEnv(RandomMapEnv(grid_resolution=GRID_RES, seed=config.get("seed", 42))))
    def _ec_flat(config):
        from experiments.baselines.mappo_flat.FlatEnv import FlatEnv
        return ParallelPettingZooEnv(
            FlatEnv(RandomMapEnv(grid_resolution=GRID_RES, seed=config.get("seed", 42))))

    register_env("hierarchical_coverage_v2_env", _ec_sa)
    register_env("standard_coverage_fair_env", _ec_std)
    register_env("flat_coverage_fair_env", _ec_flat)

    # Load checkpoints (with proper model name switching)
    algos = {}
    _model_map = {"sa": (V2UGVModel, V2UAVModel), "std": (StdUGV, StdUAV)}
    ckpt_specs = [
        ("SA-HMARL", "sa",
         os.path.join(RESULTS_DIR, "sa_hmarl_v2", "checkpoints", "latest")),
        ("Standard H-MARL", "std",
         os.path.join(PROJECT_DIR, "experiments", "baselines",
                      "standard_hmarl", "checkpoints_standard_fair", "latest")),
        ("MAPPO Flat", None,
         os.path.join(PROJECT_DIR, "experiments", "baselines",
                      "mappo_flat", "checkpoints_flat_fair", "milestones", "iter_01000")),
    ]
    for name, model_key, ckpt_path in ckpt_specs:
        try:
            if model_key:
                _ugv, _uav = _model_map[model_key]
                ModelCatalog.register_custom_model("RLlibUGVModel", _ugv)
                ModelCatalog.register_custom_model("RLlibUAVModel", _uav)
            algos[name] = Algorithm.from_checkpoint(ckpt_path)
            print(f"  [OK] {name}")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")

    # Load heuristic controllers
    heuristic_controllers = {}
    heuristic_list = [
        ("Heuristic MACPP", "experiments.baselines.heuristic_macpp.heuristic_policy", "HeuristicController"),
        ("AG-CVG", "experiments.baselines.ag_cvg.ag_cvg_policy", "AGCVGController"),
        ("Eker DP", "experiments.baselines.eker_dp.eker_dp_policy", "EkerDPController"),
        ("Porcelli CACPP", "experiments.baselines.porcelli_cacpp.porcelli_policy", "PorcelliCACPPController"),
    ]
    for name, module_path, class_name in heuristic_list:
        try:
            mod = __import__(module_path, fromlist=[class_name])
            heuristic_controllers[name] = getattr(mod, class_name)
            print(f"  [OK] {name}")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")

    print(f"\n[Phase 1 complete] Loaded {len(algos)} RL algos + {len(heuristic_controllers)} heuristic controllers")

    # ════════════════════════════════════════════════════════
    #  PHASE 2: For EACH scale, patch constants, create fresh envs, run inference
    # ════════════════════════════════════════════════════════
    import re
    import importlib

    TARGET_FILES = [
        "experiments/my_method/env_defs.py",
        "experiments/my_method/HierarchicalEnv.py",
        "experiments/my_method/HierarchicalEnvV2.py",
        "experiments/baselines/standard_hmarl/StandardEnv.py",
        "experiments/baselines/mappo_flat/FlatEnv.py",
    ]

    def patch_and_reload(scale_mult):
        """Patch source files AND reload modules so new constants take effect."""
        target_map = 2000.0 * scale_mult
        target_res = 100.0 * scale_mult
        area_mult = scale_mult ** 2
        target_max_steps = int(max(30000, 30000 * area_mult))
        min_nodes = max(5, int(20 * area_mult))
        max_nodes = max(10, int(50 * area_mult))

        for relpath in TARGET_FILES:
            fp = os.path.join(PROJECT_DIR, relpath)
            if not os.path.exists(fp):
                continue
            with open(fp, 'r') as f:
                content = f.read()
            content = re.sub(r'MAP_SIZE\s*=\s*[\d\.]+', f'MAP_SIZE         = {target_map}', content)
            content = re.sub(r'GRID_RES\s*=\s*[\d\.]+', f'GRID_RES  = {target_res}', content)
            content = re.sub(r'MAX_EPISODE_STEPS\s*=\s*\d+', f'MAX_EPISODE_STEPS = {target_max_steps}', content)
            content = re.sub(r'rng\.randint\(\d+,\s*\d+\)', f'rng.randint({min_nodes}, {max_nodes})', content)
            with open(fp, 'w') as f:
                f.write(content)

        # Reload modules so new constants take effect in THIS process
        for mod_name in [
            "experiments.my_method.env_defs",
            "experiments.my_method.HierarchicalEnvV2",
            "experiments.my_method.HierarchicalEnv",
            "experiments.baselines.standard_hmarl.StandardEnv",
            "experiments.baselines.mappo_flat.FlatEnv",
        ]:
            if mod_name in sys.modules:
                try:
                    importlib.reload(sys.modules[mod_name])
                except Exception:
                    pass

    try:
        for scale_mult, scale_name in scales:
            print(f"\n{'='*60}")
            print(f"[Phase 2] Scale: {scale_name} (mult={scale_mult})")
            print(f"{'='*60}")

            patch_and_reload(scale_mult)

            # Re-import to get updated constants
            env_defs = importlib.import_module("experiments.my_method.env_defs")
            grid_res = env_defs.GRID_RES
            area_mult = scale_mult ** 2
            max_steps = int(max(30000, 30000 * area_mult))

            print(f"  MAP_SIZE={env_defs.MAP_SIZE}, GRID_RES={grid_res}, MAX_STEPS={max_steps}")

            # RL methods
            for name, algo in algos.items():
                print(f"\n  [{name}] scale={scale_name} ...")
                t0 = time.time()

                if name == "SA-HMARL":
                    def factory(seed, gr=grid_res):
                        HierEnv = importlib.import_module("experiments.my_method.HierarchicalEnvV2").HierarchicalEnv
                        RMapEnv = importlib.import_module("experiments.my_method.env_defs").RandomMapEnv
                        return HierEnv(RMapEnv(grid_resolution=gr, seed=seed))
                elif name == "Standard H-MARL":
                    def factory(seed, gr=grid_res):
                        StdEnv = importlib.import_module("experiments.baselines.standard_hmarl.StandardEnv").StandardEnv
                        RMapEnv = importlib.import_module("experiments.my_method.env_defs").RandomMapEnv
                        return StdEnv(RMapEnv(grid_resolution=gr, seed=seed))
                elif name == "MAPPO Flat":
                    def factory(seed, gr=grid_res):
                        FlatEnv = importlib.import_module("experiments.baselines.mappo_flat.FlatEnv").FlatEnv
                        RMapEnv = importlib.import_module("experiments.my_method.env_defs").RandomMapEnv
                        return FlatEnv(RMapEnv(grid_resolution=gr, seed=seed))

                results = eval_rl(algo, factory, seeds, max_steps)
                elapsed = time.time() - t0
                n_success = sum(r["success"] for r in results)
                makespans = [r["makespan"] for r in results if r["success"]]
                print(f"    -> SR={n_success/len(seeds)*100:.0f}%, elapsed={elapsed:.0f}s")
                append_csv(args.csv_path, name, scale_name, scale_mult,
                           n_success, makespans, len(seeds))

            # Heuristic methods
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
                print(f"    -> SR={n_success/len(seeds)*100:.0f}%, elapsed={elapsed:.0f}s")
                append_csv(args.csv_path, name, scale_name, scale_mult,
                           n_success, makespans, len(seeds))

    finally:
        # Always restore to 1x
        print("\n[cleanup] Restoring to 1x baseline...")
        patch_and_reload(1.0)

    ray.shutdown()
    print("[done] All scales complete.")


if __name__ == "__main__":
    main()
