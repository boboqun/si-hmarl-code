import sys, os, time, subprocess, shutil, re, tempfile

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

TARGET_FILES = [
    "experiments/my_method/env_defs.py",
    "experiments/my_method/HierarchicalEnv.py",
    "experiments/my_method/HierarchicalEnvV2.py",
]

eval_script = """
import sys, os, time
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog

PROJECT_DIR = sys.argv[1]
SCALE_MULT = float(sys.argv[2])

ray.init(ignore_reinit_error=True, log_to_driver=False)
from experiments.my_method.hierarchical_train_v2 import RLlibUGVModel, RLlibUAVModel
ModelCatalog.register_custom_model("RLlibUGVModel", RLlibUGVModel)
ModelCatalog.register_custom_model("RLlibUAVModel", RLlibUAVModel)
ckpt = os.path.join(PROJECT_DIR, "experiments/results/sa_hmarl_v2/checkpoints/latest")
algo = Algorithm.from_checkpoint(ckpt)

from experiments.my_method.HierarchicalEnvV2 import HierarchicalEnv
from experiments.my_method.env_defs import RandomMapEnv
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

raw_env = HierarchicalEnv(RandomMapEnv(seed=1001))
env = ParallelPettingZooEnv(raw_env)
obs, _ = env.reset(seed=1001)

print(f"\\n[Diagnostic] Starting evaluation on 8km map to trigger E_safe...")

for step in range(300000):
    actions = {}
    for agent_id, agent_obs in obs.items():
        pol = "ugv_policy" if "ugv" in agent_id else "uav_policy"
        out = algo.compute_single_action(agent_obs, policy_id=pol, explore=False)
        actions[agent_id] = out[0] if isinstance(out, tuple) else out
    
    obs, _, terms, truncs, _ = env.step(actions)
    
    # Check if uav is returning due to dynamic trigger
    if any(terms.values()) or any(truncs.values()):
        break

print("[Diagnostic] Finished.")
ray.shutdown()
"""

def run_diagnostic():
    temp_dir = tempfile.mkdtemp(prefix="route_sandbox_diagnostic_")
    
    for relpath in TARGET_FILES:
        src_path = os.path.join(PROJECT_DIR, relpath)
        dst_path = os.path.join(temp_dir, relpath)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy2(src_path, dst_path)
        
        target_map = 8000.0
        target_res = 400.0
        target_max_steps = 300000
        
        with open(dst_path, 'r') as f:
            content = f.read()
        content = re.sub(r'MAP_SIZE\s*=\s*[\d\.]+', f'MAP_SIZE         = {target_map}', content)
        content = re.sub(r'GRID_RES\s*=\s*[\d\.]+', f'GRID_RES  = {target_res}', content)
        content = re.sub(r'MAX_EPISODE_STEPS\s*=\s*\d+', f'MAX_EPISODE_STEPS = {target_max_steps}', content)
        
        # Inject our diagnostic print into HierarchicalEnvV2
        if "HierarchicalEnvV2.py" in relpath:
            old_code = "if uav['battery'] < dynamic_low and not uav['is_returning'] and not uav['is_swapping']:"
            new_code = "if uav['battery'] < dynamic_low and not uav['is_returning'] and not uav['is_swapping']:\n                import os, sys\n                if dynamic_low > UAV_LOW_BATTERY:\n                    with open(os.path.join(os.environ['PROJECT_DIR_ENV'], 'experiments/results/E_safe_trigger.log'), 'a') as logf:\n                        logf.write(f'[E_safe Trigger] {a} at ({uav[\"x\"]:.1f}, {uav[\"y\"]:.1f}), UGV at ({self.car[\"x\"]:.1f}, {self.car[\"y\"]:.1f}). Dist: {dist_to_ugv:.1f}m. Triggering at dynamic threshold {dynamic_low} (Static limit would be {UAV_LOW_BATTERY})\\n')\n                    print(f'[E_safe Trigger] {a} triggered at {dynamic_low}')\n                    sys.exit(0) # Exit early once we log it!"
            content = content.replace(old_code, new_code)
            
        with open(dst_path, 'w') as f:
            f.write(content)
            
    script_path = os.path.join(temp_dir, "eval_worker.py")
    with open(script_path, 'w') as f:
        f.write(eval_script)
        
    cmd = [sys.executable, script_path, PROJECT_DIR, "4.0"]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = f"{temp_dir}:{PROJECT_DIR}:{env.get('PYTHONPATH', '')}"
    env["PROJECT_DIR_ENV"] = PROJECT_DIR
    
    # Clear old log
    log_path = os.path.join(PROJECT_DIR, "experiments/results/E_safe_trigger.log")
    if os.path.exists(log_path):
        os.remove(log_path)
    
    try:
        subprocess.run(cmd, env=env, check=False)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    run_diagnostic()
