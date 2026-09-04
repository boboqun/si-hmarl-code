#!/usr/bin/env python3
"""
compute_significance.py
========================
基于 10 个随机种子的评测数据，对 SA-HMARL 与各基线算法进行
Wilcoxon Rank-Sum 检验（双侧，α=0.05），生成学术论文用统计显著性表格。

输出：
  1. 控制台打印 LaTeX 表格代码（可直接粘贴进论文）
  2. significance_table.csv（完整数值供核查）
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RESULTS_CSV = os.path.join(PROJECT_DIR, "experiments", "results", "performance_metrics.csv")
OUTPUT_CSV  = os.path.join(PROJECT_DIR, "paper_figures", "significance_table.csv")

METRICS = {
    "Global Makespan":       "Makespan (steps) ↓",
    "Deadhead Distance (m)": "Deadhead Dist. (m) ↓",
    "Redundant Scan (m)":    "Redundant Scan (m) ↓",
    "Total Wasted (m)":      "Total Wasted (m) ↓",
}

# SA-HMARL 标签必须与 CSV 里的完全匹配
OUR_METHOD_LABEL = "Our\nSA-HMARL"
ALPHA = 0.05


def load_per_algo(df, label):
    """提取某算法所有种子的数据行"""
    return df[df["Algorithm"] == label]


def wilcoxon_symbol(our_vals, their_vals, lower_is_better=True):
    """
    返回 (symbol, p_value)
    lower_is_better=True: 我们的值越小越好
    symbol: '+' 我们显著胜出, '≈' 无显著差异, '-' 我们显著较差
    """
    # mannwhitneyu 双侧检验
    stat, p = stats.mannwhitneyu(our_vals, their_vals, alternative='two-sided')
    if p >= ALPHA:
        return "≈", p

    if lower_is_better:
        # 我们均值更小 → 我们更好
        return ("+", p) if np.mean(our_vals) < np.mean(their_vals) else ("-", p)
    else:
        return ("+", p) if np.mean(our_vals) > np.mean(their_vals) else ("-", p)


def main():
    df = pd.read_csv(RESULTS_CSV)

    # 标准化标签（CSV 里有换行符）
    df["Algorithm"] = df["Algorithm"].str.strip()

    algos = df["Algorithm"].unique().tolist()
    our_label = next((a for a in algos if "SA-HMARL" in a or "SA_HMARL" in a), None)
    if our_label is None:
        print(f"[!] 找不到 SA-HMARL 标签，当前算法列表: {algos}")
        return

    baselines = [a for a in algos if a != our_label]
    our_df    = load_per_algo(df, our_label)

    print(f"[*] SA-HMARL 标签 = '{our_label}'")
    print(f"[*] 基线算法: {baselines}")
    print(f"[*] 每算法种子数: {len(our_df)}\n")

    # ── 构建结果表 ─────────────────────────────────────────────────────────────
    rows = []
    for metric_col, metric_name in METRICS.items():
        our_vals = our_df[metric_col].values
        our_mean = np.mean(our_vals)
        our_std  = np.std(our_vals, ddof=1)

        for baseline in baselines:
            bdf        = load_per_algo(df, baseline)
            their_vals = bdf[metric_col].values
            their_mean = np.mean(their_vals)
            their_std  = np.std(their_vals, ddof=1)

            sym, pval  = wilcoxon_symbol(our_vals, their_vals, lower_is_better=True)

            rows.append({
                "Metric":           metric_name,
                "Baseline":         baseline.replace("\n", " "),
                "SA-HMARL Mean":    f"{our_mean:.1f}",
                "SA-HMARL Std":     f"{our_std:.1f}",
                "Baseline Mean":    f"{their_mean:.1f}",
                "Baseline Std":     f"{their_std:.1f}",
                "p-value":          f"{pval:.4f}",
                "Symbol":           sym,
                "Significant?":     "Yes" if pval < ALPHA else "No",
            })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"[✓] 详细结果已保存至: {OUTPUT_CSV}\n")

    # ── 控制台汇总打印 ──────────────────────────────────────────────────────────
    print("=" * 75)
    print("  Wilcoxon Rank-Sum Test: SA-HMARL vs. Baselines (α=0.05, two-sided)")
    print("=" * 75)
    print(f"{'Metric':<28} {'vs.':<22} {'SA-HMARL':>12} {'Baseline':>12} {'p':>8} {'Sig'}  ")
    print("-" * 75)

    for _, row in result_df.iterrows():
        sig_mark = f"({row['Symbol']})" if row["Significant?"] == "Yes" else "(≈)"
        print(f"{row['Metric']:<28} {row['Baseline']:<22} "
              f"{row['SA-HMARL Mean']:>7}±{row['SA-HMARL Std']:<5} "
              f"{row['Baseline Mean']:>7}±{row['Baseline Std']:<5} "
              f"{row['p-value']:>8}  {sig_mark}")

    # ── 生成 LaTeX 表格代码 ──────────────────────────────────────────────────────
    print("\n\n" + "=" * 75)
    print("  LaTeX Table (paste directly into paper)")
    print("=" * 75)

    baselines_clean = [b.replace("\n", " ") for b in baselines]
    col_spec = "l" + "c" * len(baselines)

    print(f"\\begin{{table}}[t]")
    print(f"\\centering")
    print(f"\\caption{{Quantitative comparison of SA-HMARL against baselines (mean $\\pm$ std over 10 random seeds). Symbols: (+) SA-HMARL significantly better (Wilcoxon rank-sum, $\\alpha$=0.05); ($\\approx$) no significant difference; ($-$) significantly worse.}}")
    print(f"\\label{{tab:performance_comparison}}")
    print(f"\\begin{{tabular}}{{{col_spec}}}")
    print(f"\\toprule")

    header = "Metric & " + " & ".join(baselines_clean) + " \\\\"
    print(header)
    print("\\midrule")

    for metric_name in METRICS.values():
        sub = result_df[result_df["Metric"] == metric_name]
        # SA-HMARL 均值只打印一次（每一行相同）
        our_str = f"\\textbf{{{sub.iloc[0]['SA-HMARL Mean']}$\\pm${sub.iloc[0]['SA-HMARL Std']}}}"
        cells = []
        for _, row in sub.iterrows():
            sym = row["Symbol"] if row["Significant?"] == "Yes" else "\\approx"
            pval_str = row["p-value"]
            cells.append(f"{row['Baseline Mean']}$\\pm${row['Baseline Std']} $({sym})$")
        print(f"{metric_name} & " + " & ".join(cells) + " \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


if __name__ == "__main__":
    main()
