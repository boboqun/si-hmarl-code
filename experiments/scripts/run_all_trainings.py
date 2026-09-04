"""
run_all_trainings.py
====================
多基线强化学习自动化发射架 (Training Launcher)
对应论文 5.2 节所用模型权重的全自动化集中生产线。
"""

import os
import sys
import argparse
import subprocess
import time

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SYS_ROOT = os.path.abspath(os.path.join(PROJECT_DIR, "..", ".."))

# ─────────────────────────────────────────────────────────────
#  全局默认配置（可被 TARGET_OVERRIDES 覆盖）
# ─────────────────────────────────────────────────────────────
GLOBAL_CONFIG = {
    "NUM_ITER":       "500",
    "CHECKPOINT_FREQ": "10",
    "NUM_WORKERS":    "12",
}

# ─────────────────────────────────────────────────────────────
#  注册支持的训练目标
# ─────────────────────────────────────────────────────────────
TARGETS = {
    "my_method": {
        "script_path": os.path.join(SYS_ROOT, "experiments", "my_method", "hierarchical_train.py"),
        "result_dir":  os.path.join(SYS_ROOT, "experiments", "results", "my_method", "checkpoints"),
    },
    "standard_hmarl": {
        "script_path": os.path.join(SYS_ROOT, "experiments", "baselines", "standard_hmarl", "train_standard.py"),
        "result_dir":  os.path.join(SYS_ROOT, "experiments", "results", "standard_hmarl", "checkpoints"),
    },
    "mappo_flat": {
        "script_path": os.path.join(SYS_ROOT, "experiments", "baselines", "mappo_flat", "train_flat.py"),
        "result_dir":  os.path.join(SYS_ROOT, "experiments", "results", "mappo_flat", "checkpoints"),
    },
    "sa_hmarl_tuned": {
        "script_path": os.path.join(SYS_ROOT, "experiments", "my_method", "hierarchical_train_tuned.py"),
        "result_dir":  os.path.join(SYS_ROOT, "experiments", "results", "sa_hmarl_tuned", "checkpoints"),
    },
    "sa_hmarl_staggered": {
        "script_path": os.path.join(SYS_ROOT, "experiments", "my_method", "hierarchical_train.py"),
        "result_dir":  os.path.join(SYS_ROOT, "experiments", "results", "sa_hmarl_staggered", "checkpoints"),
    },
}

# ─────────────────────────────────────────────────────────────
#  各 target 的独立训练配置（覆盖 GLOBAL_CONFIG）
#
#  Standard H-MARL：与 SA-HMARL v2 完全等步（6000 iter ≈ 48M steps），
#    确保对比方案在迭代次数层面对审稿人无可置疑。
# ─────────────────────────────────────────────────────────────
TARGET_OVERRIDES = {
    "standard_hmarl": {
        "NUM_ITER":       "6000",
        "NUM_WORKERS":    "14",    # 16核 - 2(系统) = 14 workers，Learner 另占 12 PyTorch 线程
        "CHECKPOINT_FREQ": "10",
    },
}

# ─────────────────────────────────────────────────────────────
#  不需要（重）训练的 target
#
#  mappo_flat       — 架构根本性不能收敛（动作维度灾难），
#                     Makespan 全部触及上限 30000，多训无意义；
#                     保留现有 checkpoint 作"无法收敛"证据。
#  heuristic_macpp  — 规则算法，无训练过程。
#  sa_hmarl_tuned   — 内部参数敏感性探索，不写入论文主对比。
#  sa_hmarl_staggered — Conv 崩溃中止，已放弃该方向。
# ─────────────────────────────────────────────────────────────
SKIP_TARGETS = {
    "mappo_flat":        "架构不能收敛（维度灾难），保留现有 checkpoint 作'未收敛'证据",
    "heuristic_macpp":   "规则算法，无需训练",
    "sa_hmarl_tuned":    "内部参数敏感性探索，不写入主对比",
    "sa_hmarl_staggered":"ConvolutionBackward 崩溃中止，已放弃该方向",
}

# baselines_only 快捷方式：只重训需要重训的基线
BASELINES_ONLY = ["standard_hmarl"]


def resolve_config(target_name: str) -> dict:
    """合并全局配置与 target 专属覆盖。"""
    cfg = dict(GLOBAL_CONFIG)
    cfg.update(TARGET_OVERRIDES.get(target_name, {}))
    return cfg


def run_training_pipeline(target_name: str, config: dict, resume: bool = False) -> bool:
    # ── 跳过不需要重训的 target ──────────────────────────────
    if target_name in SKIP_TARGETS:
        print(f"\n⏭️  跳过 [{target_name.upper()}]")
        print(f"   原因: {SKIP_TARGETS[target_name]}\n")
        return True

    info = TARGETS.get(target_name)
    if not info:
        print(f"[!] Target '{target_name}' 找不到注册信息。")
        return False

    script_path   = info["script_path"]
    checkpoint_dir = info["result_dir"]

    if not os.path.exists(script_path):
        print(f"[!] 找不到启动脚本: {script_path}")
        return False

    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── 日志文件（tee 到 results/<target>/logs/）────────────
    log_dir  = os.path.join(os.path.dirname(checkpoint_dir), "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path  = os.path.join(log_dir, f"train_{target_name}_{timestamp}.log")

    # ── 环境变量注入 ─────────────────────────────────────────
    env = os.environ.copy()
    env["NUM_ITER"]        = config["NUM_ITER"]
    env["NUM_WORKERS"]     = config["NUM_WORKERS"]
    env["CHECKPOINT_FREQ"] = config["CHECKPOINT_FREQ"]
    env["CHECKPOINT_DIR"]  = checkpoint_dir
    if resume:
        # 训练脚本读取此变量决定是否从已有 checkpoint 续训
        env["RESUME_CHECKPOINT"] = checkpoint_dir

    print("=" * 70)
    print(f"🚀 发射独立训练进程: {target_name.upper()}")
    print(f"  └─ 脚本     : {script_path}")
    print(f"  └─ Checkpoint: {checkpoint_dir}")
    print(f"  └─ 日志文件  : {log_path}")
    print(f"  └─ ITER={config['NUM_ITER']}  WORKERS={config['NUM_WORKERS']}  RESUME={resume}")
    print("=" * 70)

    python_exec = sys.executable
    if os.path.exists(os.path.join(SYS_ROOT, ".venv", "bin", "python")):
        python_exec = os.path.join(SYS_ROOT, ".venv", "bin", "python")

    try:
        with open(log_path, "w", buffering=1) as log_f:
            log_f.write(f"# Training log: {target_name}\n")
            log_f.write(f"# Start  : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_f.write(f"# Config : {config}\n")
            log_f.write(f"# Script : {script_path}\n\n")

            process = subprocess.Popen(
                [python_exec, script_path],
                env=env,
                cwd=SYS_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # 实时透传输出，同时写文件
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_f.write(line)

            process.wait()
            log_f.write(f"\n# End: {time.strftime('%Y-%m-%d %H:%M:%S')}  returncode={process.returncode}\n")

        if process.returncode == 0:
            print(f"\n[✓] {target_name.upper()} 训练完成！日志: {log_path}\n")
            return True
        else:
            print(f"\n[!] {target_name.upper()} 进程异常退出 (code={process.returncode})。日志: {log_path}\n")
            return False

    except KeyboardInterrupt:
        print(f"\n[!] 手动中断，{target_name.upper()} 强制停止。日志已保存至 {log_path}\n")
        process.terminate()
        return False


def main():
    parser = argparse.ArgumentParser(description="Multi-Baseline Training Launcher")
    parser.add_argument(
        "--target", type=str, required=True,
        choices=["all", "baselines_only"] + list(TARGETS.keys()),
        help=(
            "训练目标:\n"
            "  all            — 全部 target（受 SKIP_TARGETS 过滤）\n"
            "  baselines_only — 只重训需要重训的基线 (standard_hmarl)\n"
            "  <name>         — 单个 target"
        ),
    )
    parser.add_argument("--resume", action="store_true",
                        help="注入 RESUME_CHECKPOINT 环境变量，由各训练脚本决定是否续训")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印计划，不实际执行")

    args = parser.parse_args()

    if args.target == "all":
        targets_to_run = list(TARGETS.keys())
    elif args.target == "baselines_only":
        targets_to_run = list(BASELINES_ONLY)
    else:
        targets_to_run = [args.target]

    print("\n" + "=" * 70)
    print("  自动化多基线训练调度器")
    print("=" * 70)
    for t in targets_to_run:
        if t in SKIP_TARGETS:
            label = f"⏭️  跳过  — {SKIP_TARGETS[t]}"
        else:
            cfg = resolve_config(t)
            label = f"✅  训练  ITER={cfg['NUM_ITER']}  WORKERS={cfg['NUM_WORKERS']}"
        print(f"  {t:25s}: {label}")
    print("=" * 70 + "\n")

    if args.dry_run:
        print("[*] Dry-run 模式，验证完成，跳过实际执行。")
        sys.exit(0)

    start_time = time.time()

    for t_name in targets_to_run:
        cfg = resolve_config(t_name)
        success = run_training_pipeline(t_name, cfg, resume=args.resume)
        if not success:
            print(f"\n[!] 调度器停止。因为 {t_name} 未能成功完成。")
            sys.exit(1)

    total_time = (time.time() - start_time) / 3600.0
    print("=" * 70)
    print(f"🎉 全部训练计划完成！总计用时: {total_time:.2f} 小时")
    print("=" * 70)


if __name__ == "__main__":
    main()
