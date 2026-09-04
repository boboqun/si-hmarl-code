"""
evaluate_ablations.py — 统一评测消融变体并汇总对比表
====================================================
载入 run_ablation_matrix.py 训练出的各变体 checkpoint，在 10 个固定测试种子
(1001–1010) 上做确定性回放，计算与正文 Table 5.1 完全一致的三项指标
（Makespan / Deadhead / Redundant Scan）以及 idle 悬停 tick，输出对比 CSV。

评测循环严格复用 scripts/evaluate_performance.py 的指标定义，确保与主实验可比。

用法
----
    python evaluate_ablations.py                 # 评测全部已训练变体
    python evaluate_ablations.py --variant flat_continuous dual_node
输出:
    experiments/results/ablations/ablation_metrics.csv
"""

import os
import sys
import math
import argparse
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_MY_METHOD = os.path.abspath(os.path.join(_HERE, "..", "..", "my_method"))
_SYS_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
for _p in (_MY_METHOD, _HERE, _SYS_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ray  # noqa: E402
from ray.rllib.algorithms.algorithm import Algorithm  # noqa: E402
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv  # noqa: E402

from env_defs import RandomMapEnv, GRID_RES  # noqa: E402
from HierarchicalEnvV2 import HierarchicalEnv  # noqa: E402
# 导入 run_ablation_matrix 会一并注册 RLlibUAVModel_v2 / RLlibUGVModel_v2 /
# RLlibUAVModel_MLP（checkpoint 反序列化时按名查找这些 custom_model）。
from run_ablation_matrix import VARIANTS, make_env_creator  # noqa: E402
from ray import tune

UAV_SCAN_WIDTH = 50.0
TEST_SEEDS = list(range(1001, 1031))
RESULTS_DIR = os.path.join(_SYS_ROOT, "experiments", "results", "ablations")
# full = 主 SA-HMARL 配置（dual_channel + continuous + 交叉注意力），直接复用主 checkpoint
MAIN_CKPT = os.path.join(_SYS_ROOT, "experiments", "results", "sa_hmarl_v2", "checkpoints", "latest")


def _find_checkpoint(path):
    """返回合法 checkpoint 目录：本身是 checkpoint，或取目录下最新 checkpoint_*。"""
    if not os.path.exists(path):
        return None
    if os.path.exists(os.path.join(path, "rllib_checkpoint.json")):
        return path
    cks = [os.path.join(path, d) for d in os.listdir(path) if d.startswith("checkpoint_")]
    if not cks:
        return None
    cks.sort(key=os.path.getmtime)
    return cks[-1]


def _unwrap(env):
    """逐层剥离 PettingZoo 封装，拿到带有 .uavs / .coverage_grid 的底层环境。"""
    base = env
    for _ in range(5):
        if hasattr(base, "uavs"):
            return base
        if hasattr(base, "par_env"):
            base = base.par_env
        elif hasattr(base, "env"):
            base = base.env
        elif hasattr(base, "unwrapped") and base.unwrapped is not base:
            base = base.unwrapped
        else:
            break
    return base


def eval_variant(name: str):
    # full = 主 SA-HMARL 配置：直接复用已验证的主 checkpoint（≈6,472 makespan），
    # 避免消融训练后期崩溃污染整个对比的参照基准。
    # 其余变体优先加载 best/（最优奖励快照），回退到 latest/（断点续训点）。
    env_name = f"ablation_env_{name}"
    if name != "full":
        tune.register_env(env_name, make_env_creator(VARIANTS[name]["env_kwargs"]))

    if name == "full":
        ckpt = _find_checkpoint(MAIN_CKPT)
        src = "主 SA-HMARL checkpoint"
    else:
        ckpt = _find_checkpoint(os.path.join(RESULTS_DIR, name, "checkpoints", "best"))
        src = "best/"
        if ckpt is None:
            ckpt = _find_checkpoint(os.path.join(RESULTS_DIR, name, "checkpoints", "latest"))
            src = "latest/"
    if ckpt is None:
        print(f"[skip] 变体 {name} 未找到 checkpoint，请先训练。")
        return []

    print(f"\n[{name}] 加载权重（{src}）: {ckpt}")
    algo = Algorithm.from_checkpoint(ckpt)
    env_kwargs = VARIANTS[name]["env_kwargs"]
    records = []

    for seed in TEST_SEEDS:
        env = ParallelPettingZooEnv(
            HierarchicalEnv(RandomMapEnv(grid_resolution=GRID_RES, seed=seed), **env_kwargs))
        obs, _ = env.reset(seed=seed)
        base = _unwrap(env)

        prev_c = {a: (base.uavs[a]["x"], base.uavs[a]["y"]) for a in ["uav_0", "uav_1"]}
        step_count, deadhead_dist, total_scan_dist = 0, 0.0, 0.0
        while True:
            actions = {}
            for agent_id, agent_obs in obs.items():
                policy_id = "ugv_policy" if "ugv" in agent_id else "uav_policy"
                actions[agent_id] = algo.compute_single_action(
                    agent_obs, policy_id=policy_id, explore=False)
            obs, _, terms, truncs, _ = env.step(actions)
            step_count += 1
            for a in ["uav_0", "uav_1"]:
                u = base.uavs.get(a)
                if not u or a not in prev_c:
                    continue
                d = math.hypot(u["x"] - prev_c[a][0], u["y"] - prev_c[a][1])
                if (not u.get("is_busy")) or u.get("is_returning") or u.get("is_swapping"):
                    deadhead_dist += d
                else:
                    total_scan_dist += d
                prev_c[a] = (u["x"], u["y"])
            if any(terms.values()) or any(truncs.values()):
                break

        cov_area = float(np.sum(base.coverage_grid)) * GRID_RES * GRID_RES
        redundant = max(0.0, total_scan_dist - cov_area / UAV_SCAN_WIDTH)
        idle = getattr(base, "idle_ticks", float("nan"))
        print(f"  seed {seed}: makespan={step_count}  deadhead={deadhead_dist:.1f}  "
              f"redundant={redundant:.1f}  idle={idle}")
        records.append({
            "Variant": name,
            "env_kwargs": str(env_kwargs),
            "Seed": seed,
            "Global Makespan": step_count,
            "Deadhead Distance (m)": deadhead_dist,
            "Redundant Scan (m)": redundant,
            "Idle Hover (ticks)": idle,
        })
    algo.stop()
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", nargs="*", default=list(VARIANTS),
                    help="要评测的变体名（默认全部）")
    args = ap.parse_args()

    ray.init(ignore_reinit_error=True, log_to_driver=False, runtime_env={"env_vars": {"PYTHONPATH": f"{_MY_METHOD}:{_HERE}:{os.pathsep}".rstrip(os.pathsep)}})
    all_records = []
    for v in args.variant:
        all_records += eval_variant(v)
    ray.shutdown()

    if not all_records:
        print("\n[!] 没有任何变体被成功评测（可能尚未训练）。")
        return

    df = pd.DataFrame(all_records)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_p = os.path.join(RESULTS_DIR, "ablation_metrics.csv")
    df.to_csv(csv_p, index=False)

    print("\n================ 消融对比（mean ± std，10 seeds）================")
    summary = df.groupby("Variant").agg(
        Makespan_mean=("Global Makespan", "mean"),
        Makespan_std=("Global Makespan", "std"),
        Deadhead_mean=("Deadhead Distance (m)", "mean"),
        Redundant_mean=("Redundant Scan (m)", "mean"),
    )
    print(summary.round(1).to_string())
    print(f"\n[✓] 明细已保存至: {csv_p}")


if __name__ == "__main__":
    main()
