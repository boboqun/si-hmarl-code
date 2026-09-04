"""
eval_sda_mappo.py — 评测 SDA-MAPPO baseline（与其他 baseline 同标准）
使用 FlatEnv + SDA 编码器权重，10 seeds 确定性推理。
"""
import os, sys, math, argparse
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SDA_DIR = os.path.join(PROJECT_ROOT, "experiments", "baselines", "sda_mappo")
FLAT_DIR = os.path.join(PROJECT_ROOT, "experiments", "baselines", "mappo_flat")
for p in (SDA_DIR, FLAT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

from experiments.my_method.env_defs import RandomMapEnv, GRID_RES
from experiments.baselines.mappo_flat.FlatEnv import FlatEnv
from experiments.baselines.sda_mappo.sda_models import RLlibSDAUAVModel, RLlibSDAUGVModel

SUCCESS_COV = 0.98
MAX_STEPS = 30000


def env_creator(config):
    return ParallelPettingZooEnv(
        FlatEnv(map_env=RandomMapEnv(grid_resolution=GRID_RES, seed=config.get("seed", 42)))
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1001, 1011)))
    ap.add_argument("--ckpt", type=str, default=os.path.join(SDA_DIR, "checkpoints_sda_fair", "best"))
    ap.add_argument("--max_steps", type=int, default=MAX_STEPS)
    ap.add_argument("--out", type=str, default=os.path.join(PROJECT_ROOT, "experiments", "results", "sda_mappo_eval_results.csv"))
    args = ap.parse_args()

    ModelCatalog.register_custom_model("RLlibSDAUAVModel", RLlibSDAUAVModel)
    ModelCatalog.register_custom_model("RLlibSDAUGVModel", RLlibSDAUGVModel)
    register_env("sda_coverage_fair_env", env_creator)

    ray.init(ignore_reinit_error=True, log_to_driver=False)
    print(f"Loading checkpoint: {args.ckpt}")
    algo = Algorithm.from_checkpoint(args.ckpt)

    results = []
    print(f"{'seed':>6s} {'status':>10s} {'makespan':>9s} {'coverage':>9s} {'deadhead':>10s}")
    for seed in args.seeds:
        raw_env = FlatEnv(map_env=RandomMapEnv(grid_resolution=GRID_RES, seed=seed))
        env = ParallelPettingZooEnv(raw_env)
        obs, _ = env.reset(seed=seed)

        prev = {a: (raw_env.uavs[a]['x'], raw_env.uavs[a]['y']) for a in ['uav_0', 'uav_1']}
        deadhead = 0.0
        step = 0
        outcome = "timeout"

        while step < args.max_steps:
            actions = {}
            for aid, o in obs.items():
                pid = "ugv_policy" if "ugv" in aid else "uav_policy"
                actions[aid] = algo.compute_single_action(o, policy_id=pid, explore=False)
            obs, _, terms, truncs, _ = env.step(actions)
            step += 1

            # deadhead tracking
            for a in ['uav_0', 'uav_1']:
                u = raw_env.uavs[a]
                d = math.hypot(u['x'] - prev[a][0], u['y'] - prev[a][1])
                if u.get('is_returning') or u.get('is_swapping') or (not u.get('is_busy')):
                    deadhead += d
                prev[a] = (u['x'], u['y'])

            if any(terms.values()):
                cov = raw_env._compute_coverage_ratio()
                outcome = "complete" if cov >= SUCCESS_COV else "crash"
                break
            if any(truncs.values()):
                outcome = "truncated"
                break

        cov = raw_env._compute_coverage_ratio()
        success = cov >= SUCCESS_COV
        print(f"{seed:>6d} {outcome:>10s} {step:>9d} {cov:>9.4f} {deadhead:>10.1f}")
        results.append(dict(seed=seed, outcome=outcome, makespan=step,
                            coverage=cov, deadhead=deadhead, success=success))

    # Summary
    n = len(results)
    sr = sum(r['success'] for r in results) / n * 100
    succ = [r for r in results if r['success']]
    mk = np.mean([r['makespan'] for r in succ]) if succ else float('nan')
    dh = np.mean([r['deadhead'] for r in succ]) if succ else float('nan')
    print(f"\n=== SDA-MAPPO Summary ===")
    print(f"Success Rate: {sr:.1f}%  Makespan: {mk:.1f}  Deadhead: {dh:.1f}")

    # Save CSV
    import pandas as pd
    out = args.out
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"Saved: {out}")

    ray.shutdown()


if __name__ == "__main__":
    main()
