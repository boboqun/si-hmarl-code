"""
E1 — Equal-physical-coverage comparison  (reviewer #1, top priority)
===================================================================
The main table compares methods at the operational 20x20 coarse-grid 100% termination,
where realized 50 m physical coverage differs (SI-HMARL ~98% vs planners ~99.5-99.7%).
Reviewer #1 rightly asks: does SI-HMARL's makespan advantage hold at *equal physical
coverage*? This script answers that directly.

For each method it runs the rollout while logging the fixed 50 m fine-grid coverage at
every step (via FineGridAuditor, the same auditor that produced the paper's 50 m numbers),
then reads off the makespan (and deadhead) at which each method first reaches each physical
coverage threshold (0.97 / 0.98 / 0.99 / 0.995). Methods that terminate before a threshold
(e.g. SI-HMARL caps at ~98% at coarse completion) are marked N/A there -> that gap is what
the gap-filling experiment (E3) addresses.

Reuses fine_grid_audit.py (FineGridAuditor, controllers, SI-HMARL loader). Covers the five
coverage-completing methods that share HierarchicalEnv:
    SA-HMARL, Porcelli, AG-CVG, Eker, Heuristic
(Standard H-MARL / MAPPO Flat use different envs; added in a v2 if needed. MAPPO Flat never
completes coverage so it is irrelevant to an equal-coverage comparison.)

Usage (run on the Mac .venv, where the headline checkpoint + baselines live):
    .venv/bin/python experiments/scripts/eval_equal_coverage.py \
        --methods SA-HMARL Porcelli AG-CVG Eker Heuristic --seeds 1001-1010 \
        --out experiments/results/equal_coverage_2km.csv
"""
import os, sys, csv, argparse
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.scripts.fine_grid_audit import (
    attach_auditor, _controllers, load_rllib_algo,
)
from experiments.my_method.env_defs import RandomMapEnv, GRID_RES
from experiments.my_method.HierarchicalEnvV2 import HierarchicalEnv


def parse_seeds(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def rollout_with_audit(method, seed, algo, ctrls, max_steps, fine_res, resume_scan=None):
    """Run one episode -> per-step series [(step, fine_cov, coarse_cov, deadhead_ticks)].
    resume_scan: None=auto (True only for SA-HMARL); True/False forces it (resume-scan ablation)."""
    rs = (method == "SA-HMARL") if resume_scan is None else resume_scan
    import experiments.my_method.env_defs as _ed  # read CURRENT (possibly scale-patched) GRID_RES
    raw_env = HierarchicalEnv(RandomMapEnv(grid_resolution=_ed.GRID_RES, seed=seed),
                              use_resume_scan=rs)
    # attach_auditor monkey-patches env._fill_uav_swath so the 50 m grid mirrors the env's
    # EXACT swath marking each step (same method that produced the paper's 50 m numbers).
    auditor = attach_auditor(raw_env, fine_res=fine_res)
    series = []
    step = 0

    def _log():
        series.append((step,
                       auditor.coverage_ratio(),
                       raw_env._compute_coverage_ratio(),
                       float(getattr(raw_env, "deadhead_ticks", float("nan")))))

    if method == "SA-HMARL":
        from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
        env = ParallelPettingZooEnv(raw_env)
        obs, _ = env.reset(seed=seed)
        while True:
            actions = {aid: algo.compute_single_action(
                           o, policy_id="ugv_policy" if "ugv" in aid else "uav_policy",
                           explore=False)
                       for aid, o in obs.items()}
            obs, _, terms, truncs, _ = env.step(actions); step += 1
            _log()  # auditor mirrors env marking via the monkey-patch; just read it
            if any(terms.values()) or any(truncs.values()) or step >= max_steps:
                break
    else:
        obs, _ = raw_env.reset(seed=seed)
        ctrl = ctrls[method](raw_env)
        while True:
            obs, _, terms, truncs, _ = raw_env.step(ctrl.get_actions(obs)); step += 1
            _log()  # auditor mirrors env marking via the monkey-patch; just read it
            if any(terms.values()) or any(truncs.values()) or step >= max_steps:
                break
    return series


def makespan_at(series, thr):
    """First (step, deadhead_ticks) at which fine coverage >= thr; (None, None) if never."""
    for (s, fcov, ccov, dh) in series:
        if fcov >= thr:
            return s, dh
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+",
                    default=["SA-HMARL", "Porcelli", "AG-CVG", "Eker", "Heuristic"])
    ap.add_argument("--seeds", default="1001-1010")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.97, 0.98, 0.99, 0.995])
    ap.add_argument("--fine_res", type=float, default=50.0)
    ap.add_argument("--map_km", type=float, default=2.0,
                    help="map side length in km; patches env_defs.MAP_SIZE exactly like "
                         "eval_5km_6km.py (the paper's scale-transfer protocol; GRID_RES stays 100 m)")
    ap.add_argument("--max_steps", type=int, default=None,
                    help="default: max(30000, 30000*(map_km/2)^2), matching eval_5km_6km.py")
    ap.add_argument("--resume_scan", choices=["auto", "on", "off"], default="auto",
                    help="auto=per-method default; on/off forces it (resume-scan ablation)")
    ap.add_argument("--out", default=os.path.join(ROOT, "experiments", "results", "equal_coverage_2km.csv"))
    args = ap.parse_args()

    # Scale patch — runtime equivalent of generate_scalability_data.modify_constants()
    # (the file-rewrite protocol that produced the paper's scale-transfer data):
    # MAP_SIZE x mult, GRID_RES x mult (coarse grid STAYS 20x20, cells grow),
    # road-network node count x mult^2 (constant road density),
    # MAX_EPISODE_STEPS x mult^2.  NOTE: patching MAP_SIZE alone collapses the
    # operational layer onto the SW 2 km quadrant (xy_to_grid clips) -> artifact.
    mult = args.map_km / 2.0
    import experiments.my_method.env_defs as env_defs
    import experiments.my_method.HierarchicalEnvV2 as H
    env_defs.MAP_SIZE = 2000.0 * mult
    env_defs.GRID_RES = 100.0 * mult
    env_defs.GRID_ROWS = int(env_defs.MAP_SIZE / env_defs.GRID_RES)
    env_defs.GRID_COLS = int(env_defs.MAP_SIZE / env_defs.GRID_RES)
    env_defs.NUM_NODES_MIN = max(5, int(20 * mult ** 2))
    env_defs.NUM_NODES_MAX = max(10, int(50 * mult ** 2))
    H.MAX_EPISODE_STEPS = int(max(3000, 30000 * mult ** 2))
    if args.max_steps is None:
        args.max_steps = H.MAX_EPISODE_STEPS
    print(f"[scale] map {args.map_km:g} km: MAP_SIZE={env_defs.MAP_SIZE:g} GRID_RES={env_defs.GRID_RES:g} "
          f"grid {env_defs.GRID_ROWS}x{env_defs.GRID_COLS} nodes[{env_defs.NUM_NODES_MIN},{env_defs.NUM_NODES_MAX}] "
          f"max_steps={args.max_steps}")

    seeds = parse_seeds(args.seeds)
    ctrls = _controllers()
    algo = load_rllib_algo() if "SA-HMARL" in args.methods else None

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fieldnames = ["method", "seed", "threshold", "makespan_at", "deadhead_ticks_at",
                  "reached", "final_fine_cov", "term_makespan"]
    # resumable: skip (method, seed) already written (Mac may pause mid-run -> just rerun)
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for r in csv.DictReader(f):
                done.add((r["method"], int(r["seed"])))
    new_file = not os.path.exists(args.out)
    fout = open(args.out, "a", newline="")
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    if new_file:
        writer.writeheader(); fout.flush()

    hdr = " ".join(f"ms@{t*100:g}".rjust(9) for t in args.thresholds)
    print(f"{'method':10s} {'seed':>5s} {'term_ms':>8s} {'final_fcov':>10s} " + hdr)
    for method in args.methods:
        for seed in seeds:
            if (method, seed) in done:
                print(f"{method:10s} {seed:>5d}  (done, skip)"); continue
            try:
                series = rollout_with_audit(method, seed, algo, ctrls, args.max_steps, args.fine_res,
                                            {"auto": None, "on": True, "off": False}[args.resume_scan])
            except Exception as e:
                print(f"{method:10s} {seed:>5d}  [ERROR] {type(e).__name__}: {e}  (will retry on rerun)")
                continue
            term_ms = series[-1][0]; final_fcov = series[-1][1]
            cells = []
            for thr in args.thresholds:
                ms, dh = makespan_at(series, thr)
                writer.writerow(dict(method=method, seed=seed, threshold=thr,
                                     makespan_at=(ms if ms is not None else ""),
                                     deadhead_ticks_at=("" if dh is None or np.isnan(dh) else dh),
                                     reached=int(ms is not None),
                                     final_fine_cov=round(final_fcov, 4),
                                     term_makespan=term_ms))
                cells.append(str(ms) if ms is not None else "N/A")
            fout.flush()  # incremental write -> safe to pause/resume
            print(f"{method:10s} {seed:>5d} {term_ms:>8d} {final_fcov:>10.4f} "
                  + " ".join(c.rjust(9) for c in cells))
    fout.close()
    print(f"\n-> {args.out}")

    # summary from the full CSV (also reports mean final fine-cov per method -> reconciles vs paper table)
    allrows = list(csv.DictReader(open(args.out)))
    print("\n=== summary: equal-physical-coverage makespan + final fine-cov ===")
    for method in args.methods:
        fcovs = [float(r["final_fine_cov"]) for r in allrows if r["method"] == method]
        fc = f"fcov {np.mean(fcovs)*100:.1f}±{np.std(fcovs)*100:.1f}%" if fcovs else "fcov N/A"
        line = [f"{method:10s} {fc}"]
        for thr in args.thresholds:
            vals = [float(r["makespan_at"]) for r in allrows
                    if r["method"] == method and abs(float(r["threshold"]) - thr) < 1e-9 and r["makespan_at"] != ""]
            line.append(f"@{thr:g}:{np.mean(vals):.0f}±{np.std(vals):.0f}(n{len(vals)})" if vals else f"@{thr:g}:N/A")
        print("  " + " ".join(line))


if __name__ == "__main__":
    main()
