"""
eval_native_makespan.py — Greedy held-out makespan eval for a NATIVE main-method
================================================================================
Evaluates a checkpoint trained by experiments/my_method/hierarchical_train_v2.py
(native SI-HMARL, 2 km / K=5 / 27 actions) on a set of held-out map seeds and
reports per-seed makespan (first step at which coverage reaches 1.0), success,
and coverage. Writes/append a CSV with a `label` column so multiple checkpoints
(e.g. the headline run and a variance seed) land in one comparable table.

Why this script: the scale-coupled eval registers DIFFERENT model classes and
cannot restore a native checkpoint. This one imports hierarchical_train_v2 (which
registers RLlibUAVModel_v2 / RLlibUGVModel_v2 at import) and builds the native env,
using the same coverage-based makespan loop validated for the scale-coupled eval
(PettingZoo term/trunc dicts have no "__all__" key, so we check coverage directly).

Usage (run once per checkpoint into the SAME --out so they're directly comparable):
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/scripts/eval_native_makespan.py \
        --ckpt experiments/results/sa_hmarl_v2/checkpoints/latest \
        --label headline_seed42 --seeds 1001-1010 --out native_makespan_eval.csv

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/scripts/eval_native_makespan.py \
        --ckpt experiments/results/retrain_seeds_audit/st_hmarl_seed2/latest \
        --label seed2_truncate --seeds 1001-1010 --out native_makespan_eval.csv
"""
import os, sys, csv, argparse
import numpy as np

# --- make BOTH import styles work: package (experiments.my_method.X) and the
#     bare imports used inside hierarchical_train_v2 (hierarchical_models, env_defs, ...)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MY   = os.path.join(ROOT, "experiments", "my_method")
sys.path.insert(0, ROOT)
sys.path.insert(0, MY)

# The restored checkpoint config keeps num_env_runners>0, so RLlib spawns
# RolloutWorker subprocesses to validate spaces. Those fresh processes do NOT
# inherit the driver's sys.path, so they fail to `import hierarchical_train_v2`
# (which registers the custom env + models) -> ModuleNotFoundError in every
# worker -> get_spaces() returns [] -> "IndexError: list index out of range".
# Ray workers DO inherit the driver's PYTHONPATH, so exporting it here (before
# `import ray`) makes the module importable in the workers too.
os.environ["PYTHONPATH"] = os.pathsep.join(
    p for p in (MY, ROOT, os.environ.get("PYTHONPATH", "")) if p
)


def parse_seeds(s):
    if "-" in s:
        a, b = s.split("-"); return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to checkpoint dir (the one containing rllib_checkpoint.json)")
    ap.add_argument("--label", default="", help="tag written into the CSV (e.g. headline_seed42, seed2_truncate)")
    ap.add_argument("--seeds", default="1001-1010", help="held-out eval seeds, e.g. 1001-1010 or 1,2,3")
    ap.add_argument("--cap", type=int, default=30000, help="hard step cap (safety; episode normally completes ~6.5k)")
    ap.add_argument("--out", default="native_makespan_eval.csv")
    ap.add_argument("--explore", action="store_true", help="stochastic actions (default: greedy/deployment)")
    args = ap.parse_args()
    # RLlib's from_checkpoint hands the path to pyarrow.fs.from_uri, which rejects
    # relative paths ("URI has empty scheme"). Make it absolute.
    args.ckpt = os.path.abspath(args.ckpt)

    import ray
    from ray.rllib.algorithms.algorithm import Algorithm
    # importing the training module registers the native custom models (RLlib*Model_v2)
    import hierarchical_train_v2  # noqa: F401
    from env_defs import RandomMapEnv
    from HierarchicalEnvV2 import HierarchicalEnv

    print(f"[native-eval] ckpt={args.ckpt}  label={args.label}  seeds={args.seeds}  greedy={not args.explore}")
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    algo = Algorithm.from_checkpoint(args.ckpt)

    rows = []
    for seed in parse_seeds(args.seeds):
        env = HierarchicalEnv(RandomMapEnv(seed=seed))
        obs, _ = env.reset(seed=seed)
        step = 0
        makespan = None
        while step < args.cap:
            actions = {}
            for aid, ob in obs.items():
                pol = "ugv_policy" if "ugv" in aid else "uav_policy"
                out = algo.compute_single_action(ob, policy_id=pol, explore=args.explore)
                actions[aid] = out[0] if isinstance(out, tuple) else out
            obs, _, term, trunc, _ = env.step(actions)
            step += 1
            # makespan = first step at which coverage completes (paper definition).
            if env._compute_coverage_ratio() >= 1.0:
                makespan = step
                break
            if (term and all(term.values())) or (trunc and all(trunc.values())):
                break
        cov = env._compute_coverage_ratio()
        success = cov >= 1.0
        deadhead = getattr(env, "deadhead_ticks", float("nan"))
        rows.append(dict(label=args.label, seed=seed, success=int(success),
                         makespan=(makespan if success else ""),
                         coverage=round(cov, 4), deadhead_ticks=deadhead))
        print(f"  [{args.label}] seed {seed}: success={success} makespan={makespan} cov={cov:.4f}")

    ray.shutdown()
    mks = [r["makespan"] for r in rows if r["makespan"] != ""]
    sr = 100.0 * np.mean([r["success"] for r in rows])
    if mks:
        print(f"[native-eval] {args.label}: success={sr:.0f}%  makespan={np.mean(mks):.0f}±{np.std(mks):.0f}  (n={len(mks)})")
    else:
        print(f"[native-eval] {args.label}: success={sr:.0f}%  (no successful makespan)")

    write_header = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"appended -> {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
