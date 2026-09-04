#!/usr/bin/env python3
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
    [2.5,  "2.5x (5000m)"],
    [3.0,  "3x (6000m)"],
]

def main():
    n_seeds = 2
    scales = SCALES_ALL

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(RESULTS_DIR, f"scalability_custom_quick_{timestamp}.csv")

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Algorithm", "Scale", "Scale_Mult", "Success_Rate",
                          "Mean_Makespan", "Std_Makespan", "N_Seeds"])

    print(f"Running Custom Scalability: seeds={n_seeds}, scales={[s[1] for s in scales]}")
    
    worker = os.path.join(os.path.dirname(__file__), "run_scalability_v2_worker.py")
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_DIR + os.pathsep + env.get("PYTHONPATH", "")
    _flat = os.path.join(PROJECT_DIR, "experiments", "baselines", "mappo_flat")
    env["PYTHONPATH"] = _flat + os.pathsep + env["PYTHONPATH"]
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        sys.executable, worker,
        "--csv_path", csv_path,
        "--n_seeds", str(n_seeds),
        "--scales", json.dumps(scales),
    ]
    subprocess.run(cmd, check=True, env=env)

if __name__ == "__main__":
    main()
