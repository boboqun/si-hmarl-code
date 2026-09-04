"""
eval_standard_equalcov.py — Standard H-MARL makespan at a fixed-grid coverage
threshold (M3: the missing Standard@97% row of the equal-coverage table).

Rolls out a Standard H-MARL fair checkpoint in ITS OWN StandardEnv (node-
constrained rendezvous, native config, no resume-scan) with the 50 m
FineGridAuditor attached, and records the first step at which fine coverage
crosses each threshold. One run per training-seed checkpoint into the same
--out CSV (schema mirrors equal_coverage_2km.csv rows).

Usage (Mac .venv312 — checkpoints are cloud py3.12):
    .venv312/bin/python experiments/scripts/eval_standard_equalcov.py \
        --ckpt ~/cloud_ckpt_backup/cq120_standard/seed42/milestones/iter_06000 \
        --label standard_seed42 --seeds 1001-1010 \
        --out experiments/results/equal_coverage_2km_standard.csv
"""
import os, sys, csv, argparse
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.scripts.fine_grid_audit import attach_auditor
from experiments.my_method.env_defs import RandomMapEnv, GRID_RES
from experiments.baselines.standard_hmarl.StandardEnv import StandardEnv

MAX_STEPS = 30000


def parse_seeds(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--seeds", default="1001-1010")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.97, 0.98])
    ap.add_argument("--fine_res", type=float, default=50.0)
    ap.add_argument("--out", default=os.path.join(ROOT, "experiments", "results",
                                                  "equal_coverage_2km_standard.csv"))
    args = ap.parse_args()

    import ray
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.models import ModelCatalog
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    from ray.tune.registry import register_env
    from experiments.baselines.standard_hmarl.train_standard_fair import (
        RLlibUGVModel, RLlibUAVModel,
    )

    ModelCatalog.register_custom_model("RLlibUGVModel", RLlibUGVModel)
    ModelCatalog.register_custom_model("RLlibUAVModel", RLlibUAVModel)
    ModelCatalog.register_custom_model("StandardRLlibUGVModel", RLlibUGVModel)
    ModelCatalog.register_custom_model("StandardRLlibUAVModel", RLlibUAVModel)

    def env_creator(config):
        return ParallelPettingZooEnv(
            StandardEnv(RandomMapEnv(grid_resolution=GRID_RES, seed=config.get("seed", 42)))
        )
    register_env("standard_coverage_fair_env", env_creator)

    ray.init(ignore_reinit_error=True, log_to_driver=False)
    print(f"[std-eval] loading {args.ckpt}")
    algo = Algorithm.from_checkpoint(os.path.abspath(os.path.expanduser(args.ckpt)))

    rows = []
    for seed in parse_seeds(args.seeds):
        env = StandardEnv(RandomMapEnv(grid_resolution=GRID_RES, seed=seed))
        auditor = attach_auditor(env, fine_res=args.fine_res)
        obs, _ = env.reset(seed=seed)
        reached = {th: None for th in args.thresholds}
        step = 0
        while step < MAX_STEPS:
            actions = {}
            for aid, ob in obs.items():
                pol = "ugv_policy" if "ugv" in aid else "uav_policy"
                out = algo.compute_single_action(ob, policy_id=pol, explore=False)
                actions[aid] = out[0] if isinstance(out, tuple) else out
            obs, _, term, trunc, _ = env.step(actions)
            step += 1
            fc = auditor.coverage_ratio()
            for th in args.thresholds:
                if reached[th] is None and fc >= th:
                    reached[th] = step
            if (term and all(term.values())) or (trunc and all(trunc.values())):
                break
        fc = auditor.coverage_ratio()
        for th in args.thresholds:
            rows.append(dict(method=args.label, seed=seed, threshold=th,
                             makespan_at=(reached[th] if reached[th] is not None else ""),
                             reached=int(reached[th] is not None),
                             final_fine_cov=round(fc, 4), term_makespan=step))
        print(f"  [{args.label}] seed {seed}: term={step} fine_cov={fc:.4f} "
              + " ".join(f"@{th}={reached[th]}" for th in args.thresholds))

    ray.shutdown()
    import statistics as st
    for th in args.thresholds:
        mk = [r["makespan_at"] for r in rows if r["threshold"] == th and r["makespan_at"] != ""]
        n = len([r for r in rows if r["threshold"] == th])
        if mk:
            print(f"[std-eval] {args.label} @{th}: reached {len(mk)}/{n}, "
                  f"makespan {st.mean(mk):,.1f} ± {(st.pstdev(mk) if len(mk)>1 else 0):,.1f}")
        else:
            print(f"[std-eval] {args.label} @{th}: reached 0/{n}")

    write_header = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"appended -> {args.out}")


if __name__ == "__main__":
    main()
