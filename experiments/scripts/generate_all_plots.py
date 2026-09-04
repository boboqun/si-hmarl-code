#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
全量绘图调度脚本 (All-in-One Plot Generator)
用于一次性调用所有的出图脚本，并统一将生成的 PDF 论文配图导流到指派的路径下。

使用方法:
    python experiments/scripts/generate_all_plots.py
"""

import os
import sys
import subprocess

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYS_ROOT = os.path.abspath(os.path.join(PROJECT_DIR, ".."))

# 统一存放论文配图的目标终极路径
UNIFIED_OUTPUT_DIR = os.path.join(SYS_ROOT, "paper_figures")

# 列出我们要跑的所有绘图脚本的相对路径
# 注意：evaluate_k_ablation.py 必须在 plot_k_resolution_analysis.py 之前运行，
#       以确保真实 deadhead 数据已写入 CSV 后再绘图。
PLOTTING_SCRIPTS = [
    "experiments/ablations/plot_trajectory_heatmaps.py",
    "experiments/scripts/plot_recharge_response.py",
    "experiments/ablations/docking_comparison/plot_docking_ablation.py",
    "experiments/scripts/plot_learning_curves.py",
    "experiments/scripts/plot_energy_trace.py",
    "experiments/scripts/plot_scalability_generalization.py",
    "experiments/scripts/evaluate_performance.py",
    "experiments/scripts/evaluate_k_ablation.py",          # ← 先跑真实 rollout
    "experiments/scripts/plot_k_resolution_analysis.py",   # ← 再读 CSV 出图
]

def main():
    os.makedirs(UNIFIED_OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("  开始全局出图流水线 ...")
    print(f"  出图目标路径被统一接管至: {UNIFIED_OUTPUT_DIR}")
    print("=" * 60)
    
    # 继承当前系统环境，注入统一输出路径
    env = os.environ.copy()
    env["PLOT_OUTPUT_DIR"] = UNIFIED_OUTPUT_DIR
    
    python_exec = sys.executable
    if os.path.exists(os.path.join(SYS_ROOT, ".venv", "bin", "python")):
        python_exec = os.path.join(SYS_ROOT, ".venv", "bin", "python")
        
    failed = []

    for rel_path in PLOTTING_SCRIPTS:
        script_path = os.path.join(SYS_ROOT, rel_path)
        if not os.path.exists(script_path):
            print(f"\n[!] 警告: 未找到绘图脚本 {rel_path} 实体")
            failed.append(rel_path)
            continue
            
        print(f"\n[*] 正在运行绘图脚本 -> {os.path.basename(rel_path)} ...")
        
        process = subprocess.Popen(
            [python_exec, script_path],
            env=env,
            cwd=SYS_ROOT
        )
        process.wait()
        
        if process.returncode != 0:
            print(f"  [!] 出错: 脚本 {os.path.basename(rel_path)} 返回错误码 {process.returncode}")
            failed.append(rel_path)
        else:
            print(f"  [✓] 完成: {os.path.basename(rel_path)}")
            
    print("\n" + "=" * 60)
    if not failed:
        print(f"🎉 全部出图完成！所有的 PDF 均已静静躺在:\n   -> {UNIFIED_OUTPUT_DIR}")
    else:
        print(f"⚠️ 部分脚本挂了: {failed}")
    print("=" * 60)

if __name__ == "__main__":
    main()
