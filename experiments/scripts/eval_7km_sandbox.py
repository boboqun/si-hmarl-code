import sys, os, time, subprocess, shutil, re, tempfile

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

TARGET_FILES = [
    "experiments/my_method/env_defs.py",
    "experiments/my_method/HierarchicalEnv.py",
    "experiments/my_method/HierarchicalEnvV2.py",
]

eval_script = """
import sys, os, time, numpy as np
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog

PROJECT_DIR = sys.argv[1]

ray.init(ignore_reinit_error=True, log_to_driver=False)
from experiments.my_method.hierarchical_train_v2 import RLlibUGVModel, RLlibUAVModel
ModelCatalog.register_custom_model("RLlibUGVModel", RLlibUGVModel)
ModelCatalog.register_custom_model("RLlibUAVModel", RLlibUAVModel)
ckpt = os.path.join(PROJECT_DIR, "experiments/results/sa_hmarl_v2/checkpoints/latest")
algo = Algorithm.from_checkpoint(ckpt)

from experiments.my_method.HierarchicalEnvV2 import HierarchicalEnv
from experiments.my_method.env_defs import RandomMapEnv

scale_name = "7km x 7km"
scale_mult = 3.5
n_seeds = 10

print(f"\\nStarting strictly sandboxed evaluation for {scale_name} on {n_seeds} seeds...")

successes = 0
makespans = []

for seed in range(n_seeds):
    raw_env = RandomMapEnv(seed=seed)
    env = HierarchicalEnv(raw_env)
    obs, _ = env.reset(seed=seed)
    step_count = 0
    max_steps = int(max(30000, 30000 * (scale_mult**2)))

    while step_count < max_steps:
        actions = {}
        for agent_id, agent_obs in obs.items():
            pol = "ugv_policy" if "ugv" in agent_id else "uav_policy"
            out = algo.compute_single_action(agent_obs, policy_id=pol)
            actions[agent_id] = out[0] if isinstance(out, tuple) else out
        
        obs, _, terminateds, truncateds, _ = env.step(actions)
        step_count += 1
        
        is_term = terminateds.get("__all__", False) or all(v for k,v in terminateds.items() if k != "__all__")
        is_trunc = truncateds.get("__all__", False) or all(v for k,v in truncateds.items() if k != "__all__")
        
        if is_term or is_trunc:
            cov = env._compute_coverage_ratio()
            if cov >= 0.95:
                successes += 1
                makespans.append(step_count)
                print(f"  Seed {seed}: Success in {step_count} steps (cov: {cov:.4f})")
            else:
                print(f"  Seed {seed}: Failed (cov: {cov:.4f})")
            break
            
sr = successes / n_seeds * 100
mean_m = np.mean(makespans) if makespans else 0
std_m = np.std(makespans) if makespans else 0
print(f"FINAL RESULT for {scale_name}: SR = {sr:.1f}%, Makespan = {mean_m:.1f} +- {std_m:.1f}")
ray.shutdown()
"""

mult = 3.5
name = "7km x 7km"

temp_dir = tempfile.mkdtemp(prefix="route_sandbox_7km_")

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
    
cmd = [sys.executable, script_path, PROJECT_DIR]
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
env["PYTHONPATH"] = f"{temp_dir}:{PROJECT_DIR}:{env.get('PYTHONPATH', '')}"

try:
    subprocess.run(cmd, env=env, check=False)
finally:
    shutil.rmtree(temp_dir, ignore_errors=True)
