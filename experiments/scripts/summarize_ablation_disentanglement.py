"""
summarize_ablation_disentanglement.py — M1 归因汇总
=====================================================
消费 evaluate_ablations.py 产出的 ablation_metrics.csv（每变体 × 10 种子的
makespan / deadhead / redundant），生成回应审稿意见 M1 的「贡献分解表」：

  1) 奖励 × 会合 2×2 因子：full / flat_continuous / dual_node / flat_node
     —— 在所有基线都获得**相同连续会合**的前提下，单独剥离「信用分配」净贡献，
        以及在固定奖励下「连续 vs 节点会合」的净贡献。
  2) 双通道逐通道隔离：full / proactive_only / reactive_only / no_credit
  3) 宏策略结构：full / mlp_commander

对每个变体相对 full（完整 SA-HMARL）报告 mean±std、相对劣化%、配对 signed-rank p、
Cliff's δ。统计实现与 compute_significance_v2.py 完全一致（精确 signed-rank、numpy）。

用法:
    python summarize_ablation_disentanglement.py \
        --csv experiments/results/ablations/ablation_metrics.csv
"""
import os
import argparse
import numpy as np
import pandas as pd
from itertools import combinations

T95 = {9: 2.262}
GROUPS = {
    "A) reward × rendezvous (2x2)": ["full", "flat_continuous", "dual_node", "flat_node"],
    "B) per-channel credit":        ["full", "proactive_only", "reactive_only", "no_credit"],
    "C) macro-policy structure":    ["full", "mlp_commander"],
    "D) reward-weight sensitivity (M9)": ["full", "split_30", "split_70", "split_100",
                                          "prepos_lo", "prepos_hi", "shock_lo", "shock_hi"],
}
METRICS = ["Global Makespan", "Deadhead Distance (m)", "Redundant Scan (m)"]


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))


def ci95(x):
    x = np.asarray(x, float); n = len(x)
    m = x.mean(); se = x.std(ddof=1) / np.sqrt(n)
    return m, T95.get(n - 1, 2.262) * se


def signed_rank_p(a, b):
    d = np.asarray(a, float) - np.asarray(b, float); d = d[d != 0]; n = len(d)
    if n == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.abs(d))) + 1.0
    Wobs = min(ranks[d > 0].sum(), ranks.sum() - ranks[d > 0].sum())
    allr = np.arange(1, n + 1); dist = []
    for k in range(n + 1):
        for pos in combinations(range(n), k):
            wp = allr[list(pos)].sum(); dist.append(min(wp, allr.sum() - wp))
    return float(np.mean(np.array(dist) <= Wobs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="experiments/results/ablations/ablation_metrics.csv")
    args = ap.parse_args()
    if not os.path.exists(args.csv):
        print(f"[!] 未找到 {args.csv}。请先运行 evaluate_ablations.py 生成消融评测结果。")
        return
    df = pd.read_csv(args.csv)
    col = "Variant" if "Variant" in df.columns else df.columns[0]
    present = set(df[col].unique())
    for gname, variants in GROUPS.items():
        avail = [v for v in variants if v in present]
        if len(avail) < 2:
            print(f"\n### {gname}: 数据不足（已有 {avail}），跳过。"); continue
        print(f"\n### {gname}")
        full = df[df[col] == "full"]
        for met in METRICS:
            print(f"  [{met}]")
            fm, _ = ci95(full[met].values) if "full" in avail else (np.nan, 0)
            for v in avail:
                vv = df[df[col] == v][met].values
                m, h = ci95(vv)
                if v == "full":
                    print(f"    {v:<16} {m:>10.1f} ± {vv.std(ddof=1):>7.1f}   (参照)")
                else:
                    p = signed_rank_p(full[met].values, vv)
                    delta = cliffs_delta(full[met].values, vv)
                    degr = (m - fm) / fm * 100
                    print(f"    {v:<16} {m:>10.1f} ± {vv.std(ddof=1):>7.1f}   "
                          f"劣化 {degr:>+6.1f}%   signed-rank p={p:.4f}   Cliff δ={delta:+.2f}")


if __name__ == "__main__":
    main()
