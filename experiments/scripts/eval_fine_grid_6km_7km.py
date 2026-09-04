import sys, os, time, subprocess, shutil, re, tempfile

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

TARGET_FILES = [
    "experiments/my_method/env_defs.py",
    "experiments/my_method/HierarchicalEnv.py",
    "experiments/my_method/HierarchicalEnvV2.py",
]

eval_script = """
import sys, os, time, csv, numpy as np
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog

PROJECT_DIR = sys.argv[1]
SCALE_MULT = float(sys.argv[2])
SCALE_NAME = sys.argv[3]
CSV_PATH = os.path.join(PROJECT_DIR, "experiments/results/fine_grid_audit_multiscale.csv")

ray.init(ignore_reinit_error=True, log_to_driver=False)
from experiments.my_method.hierarchical_train_v2 import RLlibUGVModel, RLlibUAVModel
ModelCatalog.register_custom_model("RLlibUGVModel", RLlibUGVModel)
ModelCatalog.register_custom_model("RLlibUAVModel", RLlibUAVModel)
ckpt = os.path.join(PROJECT_DIR, "experiments/results/sa_hmarl_v2/checkpoints/latest")
algo = Algorithm.from_checkpoint(ckpt)

from experiments.my_method.HierarchicalEnvV2 import HierarchicalEnv
from experiments.my_method.env_defs import RandomMapEnv
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from experiments.scripts.fine_grid_audit import attach_auditor

n_seeds = 10

print(f"\\nStarting strictly sandboxed evaluation for {SCALE_NAME} on {n_seeds} seeds...")

for seed in range(1001, 1001 + n_seeds):
    raw_env = HierarchicalEnv(RandomMapEnv(seed=seed))
    env = ParallelPettingZooEnv(raw_env)
    obs, _ = env.reset(seed=seed)
    
    auditor = attach_auditor(raw_env, 50.0)
    
    step_count = 0
    max_steps = int(max(30000, 30000 * (SCALE_MULT**2)))
    success = False

    while step_count < max_steps:
        actions = {}
        for agent_id, agent_obs in obs.items():
            pol = "ugv_policy" if "ugv" in agent_id else "uav_policy"
            out = algo.compute_single_action(agent_obs, policy_id=pol, explore=False)
            actions[agent_id] = out[0] if isinstance(out, tuple) else out
        
        obs, _, terms, truncs, _ = env.step(actions)
        step_count += 1
        
        if any(terms.values()):
            if raw_env._compute_coverage_ratio() >= 0.98:
                success = True
            break
        if any(truncs.values()):
            break
            
    coarse_cov = raw_env._compute_coverage_ratio()
    fine_cov = auditor.coverage_ratio() if auditor else float('nan')
    print(f"  Seed {seed}: Success={success}, Coarse={coarse_cov:.4f}, Fine={fine_cov:.4f}")
    
    with open(CSV_PATH, 'a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(["Our SA-HMARL", SCALE_NAME, seed, coarse_cov, fine_cov, success])

ray.shutdown()
"""

def run_scale(mult, name):
    temp_dir = tempfile.mkdtemp(prefix=f"route_sandbox_{name.replace(' ', '_')}_")
    print(f"\\n[Sandbox] Creating isolated environment for {name} in {temp_dir}")
    
    for relpath in TARGET_FILES:
        src_path = os.path.join(PROJECT_DIR, relpath)
        dst_path = os.path.join(temp_dir, relpath)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy2(src_path, dst_path)
        
        target_map = 2000.0 * mult
        target_res = 100.0 * mult
        area_mult = mult ** 2
        target_max_steps = int(max(30000, 30000 * area_mult))
        
        with open(dst_path, 'r') as f:
            content = f.read()
        content = re.sub(r'MAP_SIZE\s*=\s*[\d\.]+', f'MAP_SIZE         = {target_map}', content)
        content = re.sub(r'GRID_RES\s*=\s*[\d\.]+', f'GRID_RES  = {target_res}', content)
        content = re.sub(r'MAX_EPISODE_STEPS\s*=\s*\d+', f'MAX_EPISODE_STEPS = {target_max_steps}', content)
        with open(dst_path, 'w') as f:
            f.write(content)
            
    script_path = os.path.join(temp_dir, "eval_worker.py")
    with open(script_path, 'w') as f:
        f.write(eval_script)
        
    cmd = [sys.executable, script_path, PROJECT_DIR, str(mult), name]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = f"{temp_dir}:{PROJECT_DIR}:{env.get('PYTHONPATH', '')}"
    
    try:
        subprocess.run(cmd, env=env, check=True)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    run_scale(3.0, "3x (6000m)")
    run_scale(3.5, "3.5x (7000m)")
