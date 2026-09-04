"""
eval_scale_coupled.py — Evaluate a trained SCALE-COUPLED checkpoint
===================================================================
Runs the held-out evaluation seeds for a scale-coupled checkpoint at ITS training
scale and writes success-rate / makespan / deadhead to CSV. Compare these against
SI-HMARL's existing zero-shot transfer numbers (paper Table `tab:scalability`) and
against the per-scale training-convergence curves (from train logs) to make the C1
point:  the size-invariant interface trains stably and transfers with no retraining,
whereas the scale-coupled interface must be retrained per scale and (per the paper's
motivation) degrades as the action/observation dimension grows.

Usage:
    python eval_scale_coupled.py --scale_km 4.0 \
        --ckpt runs/scale_coupled_4000m_seed1 --seeds 1001-1010 \
        --out scale_coupled_eval.csv
"""
import os, sys, csv, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "my_method")))
sys.path.insert(0, HERE)
from train_scale_coupled import configure_scale     # reuse the exact scale override


def parse_seeds(s):
    if "-" in s:
        a, b = s.split("-"); return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale_km", type=float, required=True)
    ap.add_argument("--cell_phys_m", type=float, default=400.0)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--seeds", type=str, default="1001-1010")
    # Default output under experiments/results/scale_coupled/ (project convention)
    _DEFAULT_EVAL_OUT = os.path.join(os.path.dirname(HERE), "..", "results",
                                     "scale_coupled", "scale_coupled_eval.csv")
    ap.add_argument("--out", type=str, default=os.path.abspath(_DEFAULT_EVAL_OUT))
    args = ap.parse_args()

    L, grid_res, grid_rows, macro_k = configure_scale(args.scale_km, args.cell_phys_m)
    n_actions = macro_k * macro_k + 2
    print(f"[eval] L={L:.0f} cell={grid_res:.0f} grid={grid_rows}x{grid_rows} "
          f"macro_k={macro_k} action_dim={n_actions} ckpt={args.ckpt}")

    import ray
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    from env_defs import RandomMapEnv, GRID_RES
    from HierarchicalEnvV2 import HierarchicalEnv
    import scalable_models  # noqa: F401  (registers custom models for checkpoint restore)

    MY_METHOD = os.path.abspath(os.path.join(HERE, "..", "..", "my_method"))
    ray.init(
        ignore_reinit_error=True,
        runtime_env={"env_vars": {
            "PYTHONPATH": f"{MY_METHOD}:{HERE}:" + os.environ.get("PYTHONPATH", ""),
        }},
    )
    from ray import tune
    def env_creator(cfg):
        configure_scale(args.scale_km, args.cell_phys_m)
        import env_defs as _ed
        random_map = _ed.RandomMapEnv(grid_resolution=_ed.GRID_RES)
        hier = HierarchicalEnv(map_env=random_map, macro_k=macro_k)
        return ParallelPettingZooEnv(hier)
    tune.register_env("scale_coupled_env", env_creator)
    algo = Algorithm.from_checkpoint(args.ckpt)

    rows = []
    for seed in parse_seeds(args.seeds):
        env = HierarchicalEnv(map_env=RandomMapEnv(grid_resolution=GRID_RES, seed=seed),
                              macro_k=macro_k)
        obs, _ = env.reset(seed=seed)
        cap = int(30000 * (args.scale_km ** 2))
        step = 0
        makespan = None                                        # first step at which coverage completes
        while step < cap:
            actions = {}
            for aid, ob in obs.items():
                pol = "ugv_policy" if "ugv" in aid else "uav_policy"
                out = algo.compute_single_action(ob, policy_id=pol)
                actions[aid] = out[0] if isinstance(out, tuple) else out
            obs, _, term, trunc, _ = env.step(actions)
            step += 1
            # Makespan M = min{t : CoverageRatio(t)=1} (paper definition). This env is a
            # PettingZoo ParallelEnv whose term/trunc are PER-AGENT dicts with NO "__all__"
            # key, so the old `term.get("__all__")` check never fired and the loop ran to the
            # cap on every seed. Check coverage directly; fall back to all-agents-done.
            if env._compute_coverage_ratio() >= 1.0:
                makespan = step
                break
            if (term and all(term.values())) or (trunc and all(trunc.values())):
                break
        cov = env._compute_coverage_ratio()
        success = cov >= 1.0                                   # operational coarse-grid criterion
        deadhead = getattr(env, "deadhead_ticks", float("nan"))   # ticks (HierarchicalEnvV2 attr)
        rows.append(dict(scale_km=args.scale_km, macro_k=macro_k, action_dim=n_actions,
                         seed=seed, success=int(success),
                         makespan=(makespan if success else ""),
                         coarse_cov=round(cov, 4), deadhead_ticks=deadhead))
        print(f"  seed {seed}: success={success} makespan={makespan} cov={cov:.4f}")

    ray.shutdown()
    sr = 100.0 * np.mean([r["success"] for r in rows])
    mk = [r["makespan"] for r in rows if r["makespan"] != ""]
    print(f"[eval] scale={args.scale_km}km K={macro_k}: success={sr:.0f}%  "
          f"makespan={np.mean(mk):.0f}±{np.std(mk):.0f}" if mk else f"[eval] success={sr:.0f}%")

    write_header = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"appended -> {args.out}")


if __name__ == "__main__":
    main()
