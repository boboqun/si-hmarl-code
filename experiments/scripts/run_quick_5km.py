import sys, os, time, importlib, json
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_DIR)

print("Init ray...")
ray.init(ignore_reinit_error=True, log_to_driver=False)

print("Loading SA-HMARL...")
from experiments.my_method.hierarchical_train_v2 import RLlibUGVModel as V2UGVModel, RLlibUAVModel as V2UAVModel
ModelCatalog.register_custom_model("RLlibUGVModel", V2UGVModel)
ModelCatalog.register_custom_model("RLlibUAVModel", V2UAVModel)
ckpt = os.path.join(PROJECT_DIR, "experiments/results/sa_hmarl_v2/checkpoints/latest")
algo = Algorithm.from_checkpoint(ckpt)

print("Patching constants for 2.5x (5000m)...")
worker = importlib.import_module("experiments.scripts.run_scalability_v2_worker")
worker.patch_and_reload(2.5)

env_defs = importlib.import_module("experiments.my_method.env_defs")
from experiments.my_method.HierarchicalEnvV2 import HierarchicalEnv
from experiments.my_method.env_defs import RandomMapEnv

print("Evaluating 1 seed for 5000m...")
seed = 0
raw_env = RandomMapEnv(seed=seed)
env = HierarchicalEnv(raw_env, grid_res=env_defs.GRID_RES)

obs, _ = env.reset(seed=seed)
step_count = 0
max_steps = int(max(30000, 30000 * (2.5**2)))

success = False
state = algo.get_policy("ugv_policy").get_initial_state()
uav_state = algo.get_policy("uav_policy").get_initial_state()

print("Stepping...")
t0 = time.time()
last_print = t0
while step_count < max_steps:
    actions = {}
    for agent_id, agent_obs in obs.items():
        if "ugv" in agent_id:
            a, state, _ = algo.compute_single_action(agent_obs, state=state, policy_id="ugv_policy")
        else:
            a, uav_state, _ = algo.compute_single_action(agent_obs, state=uav_state, policy_id="uav_policy")
        actions[agent_id] = a
    
    obs, rewards, terminateds, truncateds, infos = env.step(actions)
    step_count += 1
    
    if time.time() - last_print > 5.0:
        cov = raw_env._compute_coverage_ratio()
        print(f"  Step {step_count}/{max_steps}, cov: {cov:.4f}")
        sys.stdout.flush()
        last_print = time.time()

    if terminateds["__all__"] or truncateds["__all__"]:
        success = env.is_success()
        break

cov = raw_env._compute_coverage_ratio()
print(f"Finished seed {seed}: success={success}, steps={step_count}, cov={cov:.4f}, time={time.time()-t0:.1f}s")

worker.patch_and_reload(1.0)
ray.shutdown()
