#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import re
import csv
import subprocess
import datetime

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RESULTS_DIR = os.path.join(PROJECT_DIR, "experiments", "results")

# 我们要操作的核心定义文件
TARGET_FILES = [
    "experiments/my_method/env_defs.py",
    "experiments/my_method/HierarchicalEnv.py",
    "experiments/my_method/HierarchicalEnvV2.py",
    "experiments/baselines/standard_hmarl/StandardEnv.py",
    "experiments/baselines/mappo_flat/FlatEnv.py"
]

SCALES = [
    (0.25, "0.25x (500m)"),
    (0.5,  "0.5x (1000m)"),
    (1.0,  "1x (2000m)"),
    (2.0,  "2x (4000m)"),
    (4.0,  "4x (8000m)")
]

def modify_constants(scale_mult):
    """
    使用正则直接修改源文件中的 MAP_SIZE 和 GRID_RES 常量。
    通过动态修改物理源文件，可以完美欺骗 RLlib 的静态图和环境初始化，
    做到真正的 Zero-Shot 测试，而不需要重构整个庞大的系统。
    """
    target_map_size = 2000.0 * scale_mult
    target_grid_res = 100.0 * scale_mult
    
    print(f"\n[🔧] 正在修补底层物理引擎文件 -> MAP_SIZE={target_map_size}, GRID_RES={target_grid_res}")
    
    for relative_path in TARGET_FILES:
        filepath = os.path.join(PROJECT_DIR, relative_path)
        if not os.path.exists(filepath):
            continue
            
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 匹配 MAP_SIZE = xxxx.x 或 MAP_SIZE  = xxxx.x
        content = re.sub(r'MAP_SIZE\s*=\s*[\d\.]+', f'MAP_SIZE         = {target_map_size}', content)
        # 匹配 GRID_RES = xxxx.x
        content = re.sub(r'GRID_RES\s*=\s*[\d\.]+', f'GRID_RES  = {target_grid_res}', content)
        # 匹配 num_nodes 确保路网密度在物理空间扩展时保持一致
        # 由于地图面积扩大了 scale_mult^2 倍，如果想保持密度，我们应该将生成点数乘以面积倍数
        area_mult = scale_mult ** 2
        min_nodes = max(5, int(20 * area_mult))
        max_nodes = max(10, int(50 * area_mult))
        content = re.sub(r'rng\.randint\(\d+,\s*\d+\)', f'rng.randint({min_nodes}, {max_nodes})', content)
        
        # 同步成比例放大物理环境自带的安全超时销毁阈值 (MAX_EPISODE_STEPS)
        target_max_steps = int(max(3000, 30000 * area_mult))
        content = re.sub(r'MAX_EPISODE_STEPS\s*=\s*\d+', f'MAX_EPISODE_STEPS = {target_max_steps}', content)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=30, help="Number of seeds (default 30)")
    parser.add_argument("--scales", nargs="+", type=float, default=None, help="Override: only run these scale mults (e.g. --scales 2.0 4.0)")
    parser.add_argument("--resume_csv", type=str, default=None, help="Path to existing scalability_results.csv to resume from")
    args = parser.parse_args()

    scales_to_run = [(s, n) for s, n in SCALES if args.scales is None or s in args.scales]

    print(f"[*] 正在制备多尺度零样本跨域评估数据 (seeds={args.n_seeds}, scales={[n for _, n in scales_to_run]}) ...")
    
    if args.resume_csv and os.path.exists(args.resume_csv):
        csv_path = args.resume_csv
        print(f"[*] 侦测到恢复模式，将继续追加至: {csv_path}")
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"scalability_results_{timestamp}.csv"
        csv_path = os.path.join(RESULTS_DIR, csv_filename)
        print(f"[*] 全局导出数据将安全存储至: {csv_path}")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Algorithm", "Scale", "Success Rate (%)", "Makespan Multiplier"])
    
    # 确保最终一定要把文件改回来
    try:
        for scale_mult, scale_name in scales_to_run:
            modify_constants(scale_mult)
            
            evaluator_path = os.path.join(PROJECT_DIR, "experiments", "scripts", "run_scalability_evaluator_subprocess.py")
            cmd = [
                sys.executable, evaluator_path,
                "--scale_mult", str(scale_mult),
                "--scale_name", scale_name,
                "--csv_path", csv_path,
                "--n_seeds", str(args.n_seeds),
            ]
            
            print(f"[*] 启动子进程推演 {scale_name} ...")
            subprocess.run(cmd, check=True)
            
    finally:
        print("\n[*] 正在清理并恢复所有底层物理引擎常量至 1x Baseline (MAP_SIZE=2000.0) ...")
        modify_constants(1)

if __name__ == "__main__":
    main()
