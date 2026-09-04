import sys, os, time
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_DIR)

print("Init ray...")
ray.init(ignore_reinit_error=True, log_to_driver=False)

print("Loading SA-HMARL...")
from experiments.my_method.hierarchical_train_v2 import RLlibUGVModel, RLlibUAVModel
ModelCatalog.register_custom_model("RLlibUGVModel", RLlibUGVModel)
ModelCatalog.register_custom_model("RLlibUAVModel", RLlibUAVModel)
ckpt = os.path.join(PROJECT_DIR, "experiments/results/sa_hmarl_v2/checkpoints/latest")
algo = Algorithm.from_checkpoint(ckpt)

print("Patching constants for 5000m...")
import experiments.my_method.env_defs as env_defs
env_defs.MAP_SIZE = 5000
env_defs.BLOCK_SIZE = env_defs.MAP_SIZE / env_defs.GRID_RES
env_defs.NUM_BLOCKS = env_defs.GRID_RES * env_defs.GRID_RES
import experiments.my_method.HierarchicalEnvV2 as H
H.env_defs = env_defs

from experiments.my_method.HierarchicalEnvV2 import HierarchicalEnv
from experiments.my_method.env_defs import RandomMapEnv

print("Evaluating 1 seed for 5000m...")
seed = 0
raw_env = RandomMapEnv(seed=seed)
env = HierarchicalEnv(raw_env)

obs, _ = env.reset(seed=seed)
step_count = 0
max_steps = int(max(30000, 30000 * (2.5**2)))

success = False
print("Stepping...")
t0 = time.time()
last_print = t0
while step_count < max_steps:
    actions = {}
    for agent_id, agent_obs in obs.items():
        pol = "ugv_policy" if "ugv" in agent_id else "uav_policy"
        out = algo.compute_single_action(agent_obs, policy_id=pol)
        actions[agent_id] = out[0] if isinstance(out, tuple) else out
    
    obs, rewards, terminateds, truncateds, infos = env.step(actions)
    step_count += 1
    
    if time.time() - last_print > 3.0:
        cov = env._compute_coverage_ratio()
        print(f"  Step {step_count}/{max_steps}, cov: {cov:.4f}")
        sys.stdout.flush()
        last_print = time.time()

    is_term = terminateds.get("__all__", False) or all(v for k,v in terminateds.items() if k != "__all__")
    is_trunc = truncateds.get("__all__", False) or all(v for k,v in truncateds.items() if k != "__all__")
    
    if is_term or is_trunc:
        success = env.is_success()
        break

cov = env._compute_coverage_ratio()
print(f"Finished seed {seed}: success={success}, steps={step_count}, cov={cov:.4f}, time={time.time()-t0:.1f}s")
ray.shutdown()
