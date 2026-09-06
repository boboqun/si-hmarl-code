"""[2026-09-04 seed-selectable variant of eval_envelope_constnodes.py: adds --seed_lo/--seed_hi; used for the
scale-coupled comparison on the held-out seeds 1001-1010. Otherwise identical.]
Envelope-table re-run under the CONSTANT-NODE protocol (m13 fix, 2026-07-03).

The paper's envelope table (tab:scalability) mixes protocols: its 4 km / 8 km rows
came from generate_scalability_data.py (node count x area), while the 500 m-7 km
siblings came from the sandbox family (node count constant 20-50, matching the
manuscript text L953). This script re-produces the 4 km / 8 km rows under the
constant-node protocol so the whole table is homogeneous.

Mechanism = verbatim eval_sandbox.py (the proven producer): copy the three env
modules into a temp dir, regex-patch MAP_SIZE / GRID_RES / MAX_EPISODE_STEPS on
the COPIES (node-count literal untouched -> constant 20-50), run a worker with
PYTHONPATH prioritising the temp dir. Source tree is never modified -> safe to
run beside live trainings. Adds vs eval_sandbox: seeds 0-9 (matching
scalability_5km_6km_results.csv), per-seed CSV rows in the same schema, resume
(skips seeds already in the CSV), and capped ray.init so the Mac trainings are
not starved.

Usage:
  .venv/bin/python experiments/scripts/eval_envelope_constnodes.py \
      --scale_mult 2.0 --scale_name "4km x 4km" \
      --out experiments/results/envelope_constnodes_4km.csv
"""
import sys, os, subprocess, shutil, re, tempfile, argparse

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

TARGET_FILES = [
    "experiments/my_method/env_defs.py",
    "experiments/my_method/HierarchicalEnv.py",
    "experiments/my_method/HierarchicalEnvV2.py",
]

WORKER = r'''
import sys, os, csv, numpy as np
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog

PROJECT_DIR, scale_name, scale_mult, out_csv = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
SEED_LO, SEED_HI = int(sys.argv[5]), int(sys.argv[6])

# NOTE: do NOT cap num_cpus here — Algorithm.from_checkpoint restores the
# training config's full rollout-worker pool and deadlocks if Ray has fewer
# CPUs than it needs (learned 2026-07-03 the hard way).
ray.init(ignore_reinit_error=True, log_to_driver=False,
         object_store_memory=2 * 1024**3)
from experiments.my_method.hierarchical_train_v2 import RLlibUGVModel, RLlibUAVModel
ModelCatalog.register_custom_model("RLlibUGVModel", RLlibUGVModel)
ModelCatalog.register_custom_model("RLlibUAVModel", RLlibUAVModel)
ckpt = os.path.join(PROJECT_DIR, "experiments/results/sa_hmarl_v2/checkpoints/latest")
algo = Algorithm.from_checkpoint(ckpt)

from experiments.my_method.HierarchicalEnvV2 import HierarchicalEnv
from experiments.my_method.env_defs import RandomMapEnv, MAP_SIZE, GRID_RES, GRID_ROWS

print(f"[sandbox check] MAP_SIZE={MAP_SIZE} GRID_RES={GRID_RES} GRID_ROWS={GRID_ROWS}", flush=True)
assert abs(MAP_SIZE - 2000.0 * scale_mult) < 1e-6, "patched MAP_SIZE not picked up!"

done = set()
if os.path.exists(out_csv):
    with open(out_csv) as f:
        for r in csv.DictReader(f):
            done.add(int(r["Seed"]))
new_file = not os.path.exists(out_csv)
fout = open(out_csv, "a", newline="")
writer = csv.writer(fout)
if new_file:
    writer.writerow(["Seed", "Scale", "Makespan", "Coverage", "Status"]); fout.flush()

max_steps = int(max(30000, 30000 * (scale_mult ** 2)))
for seed in range(SEED_LO, SEED_HI + 1):
    if seed in done:
        print(f"  Seed {seed}: (done, skip)", flush=True); continue
    raw_env = RandomMapEnv(seed=seed)
    env = HierarchicalEnv(raw_env)
    obs, _ = env.reset(seed=seed)
    step_count = 0
    while step_count < max_steps:
        actions = {}
        for agent_id, agent_obs in obs.items():
            pol = "ugv_policy" if "ugv" in agent_id else "uav_policy"
            out = algo.compute_single_action(agent_obs, policy_id=pol)
            actions[agent_id] = out[0] if isinstance(out, tuple) else out
        obs, _, terminateds, truncateds, _ = env.step(actions)
        step_count += 1
        is_term = terminateds.get("__all__", False) or all(v for k, v in terminateds.items() if k != "__all__")
        is_trunc = truncateds.get("__all__", False) or all(v for k, v in truncateds.items() if k != "__all__")
        if is_term or is_trunc:
            break
    cov = env._compute_coverage_ratio()
    ok = cov >= 1.0
    writer.writerow([seed, scale_name, step_count if ok else 0, f"{cov:.4f}",
                     "Success" if ok else "Failed"]); fout.flush()
    print(f"  Seed {seed}: {'Success in %d steps' % step_count if ok else 'Failed'} (cov: {cov:.4f})", flush=True)

fout.close()
rows = [r for r in csv.DictReader(open(out_csv)) if r["Scale"] == scale_name]
ms = [int(r["Makespan"]) for r in rows if r["Status"] == "Success"]
print(f"FINAL {scale_name} (const-node): SR = {len(ms)}/{len(rows)}, "
      f"Makespan = {np.mean(ms):.1f} +- {np.std(ms):.1f}" if ms else
      f"FINAL {scale_name} (const-node): SR = 0/{len(rows)}", flush=True)
ray.shutdown()
'''

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale_mult", type=float, required=True)
    ap.add_argument("--scale_name", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--seed_lo", type=int, default=0)
    ap.add_argument("--seed_hi", type=int, default=9)
    args = ap.parse_args()

    temp_dir = tempfile.mkdtemp(prefix="route_sandbox_constnodes_")
    for relpath in TARGET_FILES:
        src_path = os.path.join(PROJECT_DIR, relpath)
        dst_path = os.path.join(temp_dir, relpath)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy2(src_path, dst_path)
        target_map = 2000.0 * args.scale_mult
        target_res = 100.0 * args.scale_mult
        target_max_steps = int(max(30000, 30000 * args.scale_mult ** 2))
        with open(dst_path) as f:
            content = f.read()
        content = re.sub(r'MAP_SIZE\s*=\s*[\d\.]+', f'MAP_SIZE         = {target_map}', content)
        content = re.sub(r'GRID_RES\s*=\s*[\d\.]+', f'GRID_RES  = {target_res}', content)
        content = re.sub(r'MAX_EPISODE_STEPS\s*=\s*\d+', f'MAX_EPISODE_STEPS = {target_max_steps}', content)
        # node-count literal (NUM_NODES_MIN/MAX) deliberately NOT touched -> constant 20-50
        with open(dst_path, "w") as f:
            f.write(content)
    print(f"Prepared const-node sandbox for {args.scale_name} in {temp_dir}", flush=True)

    script_path = os.path.join(temp_dir, "eval_worker.py")
    with open(script_path, "w") as f:
        f.write(WORKER)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = f"{temp_dir}:{PROJECT_DIR}:{env.get('PYTHONPATH', '')}"
    env["OMP_NUM_THREADS"] = "1"; env["MKL_NUM_THREADS"] = "1"
    cmd = [sys.executable, script_path, PROJECT_DIR, args.scale_name, str(args.scale_mult), args.out, str(args.seed_lo), str(args.seed_hi)]
    try:
        subprocess.run(cmd, env=env, check=False)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
