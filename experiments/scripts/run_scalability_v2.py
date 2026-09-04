#!/usr/bin/env python3
"""
run_scalability_v2.py
=====================
Driver for comprehensive scalability evaluation.

Delegates to run_scalability_v2_worker.py which:
  Phase 1: Loads all RL checkpoints under original 2km constants
  Phase 2: Patches constants per scale and evaluates all methods

Usage:
  python run_scalability_v2.py --dry-run          # 1 seed, 4km+8km only
  python run_scalability_v2.py --seeds 30         # full 30 seeds, all scales
  python run_scalability_v2.py --seeds 10 --scales 2.0 4.0   # custom
"""
import os
import sys
import json
import subprocess
import datetime
import argparse
import csv

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RESULTS_DIR = os.path.join(PROJECT_DIR, "experiments", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SCALES_ALL = [
    [0.25, "0.25x (500m)"],
    [0.5,  "0.5x (1000m)"],
    [1.0,  "1x (2000m)"],
    [2.0,  "2x (4000m)"],
    [4.0,  "4x (8000m)"],
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="1 seed, only 2x+4x scales")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--scales", nargs="+", type=float, default=None)
    args = parser.parse_args()

    n_seeds = 1 if args.dry_run else args.seeds

    if args.scales:
        scales = [s for s in SCALES_ALL if s[0] in args.scales]
    elif args.dry_run:
        scales = [s for s in SCALES_ALL if s[0] in (2.0, 4.0)]
    else:
        scales = SCALES_ALL

    tag = "dryrun" if args.dry_run else f"n{n_seeds}"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(RESULTS_DIR, f"scalability_v2_{tag}_{timestamp}.csv")

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Algorithm", "Scale", "Scale_Mult", "Success_Rate",
                          "Mean_Makespan", "Std_Makespan", "N_Seeds"])

    print(f"{'='*70}")
    print(f"  Scalability v2: seeds={n_seeds}, scales={[s[1] for s in scales]}")
    print(f"  Output: {csv_path}")
    print(f"{'='*70}")

    worker = os.path.join(os.path.dirname(__file__), "run_scalability_v2_worker.py")
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_DIR + os.pathsep + env.get("PYTHONPATH", "")
    # Also inject mappo_flat for pickle compat
    _flat = os.path.join(PROJECT_DIR, "experiments", "baselines", "mappo_flat")
    env["PYTHONPATH"] = _flat + os.pathsep + env["PYTHONPATH"]

    cmd = [
        sys.executable, worker,
        "--csv_path", csv_path,
        "--n_seeds", str(n_seeds),
        "--scales", json.dumps(scales),
    ]
    subprocess.run(cmd, check=True, env=env)

    print(f"\n{'='*70}")
    print(f"  Results saved to: {csv_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
