import os
import sys
import math
import csv

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.abspath(os.path.join(_HERE, "../.."))
sys.path.insert(0, os.path.join(_PROJECT_DIR, "experiments/my_method"))
sys.path.insert(0, _PROJECT_DIR)

import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

from experiments.my_method.env_defs import RandomMapEnv, GRID_RES
from experiments.my_method.HierarchicalEnvV2 import HierarchicalEnv
from experiments.ablations.credit_rendezvous.run_ablation_matrix import VARIANTS, make_env_creator
from ray import tune
from ray.rllib.models import ModelCatalog
from experiments.my_method.hierarchical_train_v2 import RLlibUGVModel, RLlibUAVModel

ModelCatalog.register_custom_model("RLlibUGVModel", RLlibUGVModel)
ModelCatalog.register_custom_model("RLlibUAVModel", RLlibUAVModel)

TEST_SEEDS = list(range(1001, 1011))
RESULTS_DIR = os.path.join(_PROJECT_DIR, "experiments", "results", "ablations")
MAIN_CKPT = os.path.join(_PROJECT_DIR, "experiments", "results", "sa_hmarl_v2", "checkpoints", "latest")

def _find_checkpoint(path):
    if not os.path.exists(path): return None
    if os.path.exists(os.path.join(path, "rllib_checkpoint.json")): return path
    cks = [os.path.join(path, d) for d in os.listdir(path) if d.startswith("checkpoint_")]
    if not cks: return None
    cks.sort(key=os.path.getmtime)
    return cks[-1]

def _unwrap(env):
    base = env
    for _ in range(5):
        if hasattr(base, "uavs"): return base
        if hasattr(base, "par_env"): base = base.par_env
        elif hasattr(base, "env"): base = base.env
        elif hasattr(base, "unwrapped") and base.unwrapped is not base: base = base.unwrapped
        else: break
    return base

def main():
    ray.init(ignore_reinit_error=True, log_to_driver=False, runtime_env={"env_vars": {"PYTHONPATH": f"{os.path.join(_PROJECT_DIR, 'experiments/my_method')}:{_PROJECT_DIR}"}})
    
    variants_to_test = ["full", "dual_node", "flat_node"]
    records = []
    
    loaded_algos = {}
    
    for name in variants_to_test:
        print(f"\nEvaluating variant: {name}")
        env_kwargs = VARIANTS[name]["env_kwargs"] if name in VARIANTS else {}
        env_name = f"ablation_env_{name}"
        if name != "full":
            tune.register_env(env_name, make_env_creator(env_kwargs))
        
        if name in ["full", "dual_node"]:
            ckpt = _find_checkpoint(MAIN_CKPT)
        else:
            ckpt = _find_checkpoint(os.path.join(RESULTS_DIR, name, "checkpoints", "best"))
            if not ckpt:
                ckpt = _find_checkpoint(os.path.join(RESULTS_DIR, name, "checkpoints", "latest"))
        
        if ckpt not in loaded_algos:
            print(f"Loading checkpoint for {name}: {ckpt}")
            loaded_algos[ckpt] = Algorithm.from_checkpoint(ckpt)
        algo = loaded_algos[ckpt]
        
        for seed in TEST_SEEDS:
            raw_env = HierarchicalEnv(RandomMapEnv(grid_resolution=GRID_RES, seed=seed), **env_kwargs)
            env = ParallelPettingZooEnv(raw_env)
            obs, _ = env.reset(seed=seed)
            base = _unwrap(env)
            
            step_count = 0
            status = "timeout"
            while step_count < 30000:
                actions = {}
                for agent_id, agent_obs in obs.items():
                    policy_id = "ugv_policy" if "ugv" in agent_id else "uav_policy"
                    actions[agent_id] = algo.compute_single_action(agent_obs, policy_id=policy_id, explore=False)
                
                obs, _, terms, truncs, _ = env.step(actions)
                step_count += 1
                
                if any(terms.values()):
                    cov = raw_env._compute_coverage_ratio()
                    if cov >= 0.98:
                        status = "completed"
                    else:
                        status = "crash"
                    break
                if any(truncs.values()):
                    status = "timeout"
                    break
            
            print(f"  [{name}] Seed {seed}: Makespan={step_count}, Status={status}")
            records.append([name, seed, step_count, status])
    
    ray.shutdown()
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_p = os.path.join(RESULTS_DIR, "node_ablation_status_paired.csv")
    with open(csv_p, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Variant", "Seed", "Makespan", "Status"])
        writer.writerows(records)
    print(f"\n[✓] Stateful pairing saved to: {csv_p}")

if __name__ == "__main__":
    main()
