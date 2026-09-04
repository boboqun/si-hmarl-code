"""
compute_significance_v2.py — 配对统计复核（回应审稿意见 M3）
=================================================================
对 results/performance_metrics.csv 中 10 个配对随机种子（1001–1010）的结果，
对每个基线 vs SA-HMARL、每个指标计算：

  • 配对 Wilcoxon **signed-rank** 检验（精确零分布，n=10 → 枚举 2^10 个符号组合）
    —— 注意：旧版 compute_significance.py 误用了 rank-sum（独立样本）检验；
       由于所有方法共用同一组种子（配对设计），signed-rank 才是正确选择。
  • Cliff's δ 效应量（非参数随机优势，[-1,1]；lower-is-better 指标下
    δ=-1 表示「我们的每个 run 都优于该基线的每个 run」= 完全占优）
    —— 旧版代码根本没有计算 Cliff's δ，论文中的 δ 值此前无可复现来源。
  • 均值 ± 标准差 与 95% t 置信区间（df=n-1）。

不依赖 scipy（环境无网络），用 numpy 自实现精确 signed-rank。

用法:
    python compute_significance_v2.py
输出:
    paper_figures/significance_table_v2.csv
"""
import os
import numpy as np
import pandas as pd
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CSV_IN = os.path.join(ROOT, "experiments", "results", "performance_metrics.csv")
CSV_OUT = os.path.join(ROOT, "paper_figures", "significance_table_v2.csv")

OURS = "Our SA-HMARL"
METRICS = ["Global Makespan", "Deadhead Distance (m)", "Redundant Scan (m)"]
# t_{0.975, df}; 仅需 df=9 (n=10)，保留几档便于不同 n
T95 = {4: 2.776, 9: 2.262, 14: 2.145, 19: 2.093, 29: 2.045}


def cliffs_delta(a, b):
    """a=ours, b=baseline. 返回 P(a>b)-P(a<b)。lower-is-better 指标下越负越好。"""
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))


def ci95(x):
    x = np.asarray(x, float); n = len(x)
    m = x.mean(); se = x.std(ddof=1) / np.sqrt(n)
    h = T95.get(n - 1, 2.262) * se
    return m, m - h, m + h


def wilcoxon_signed_rank_exact(a, b):
    """配对 signed-rank 两侧精确 p（n 较小，枚举全部符号组合的零分布）。"""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[d != 0]; n = len(d)
    if n == 0:
        return float("nan"), float("nan")
    ranks = np.argsort(np.argsort(np.abs(d))) + 1.0
    Wplus = ranks[d > 0].sum()
    Wobs = min(Wplus, ranks.sum() - Wplus)
    allr = np.arange(1, n + 1)
    dist = []
    for k in range(n + 1):
        for pos in combinations(range(n), k):
            wp = allr[list(pos)].sum()
            dist.append(min(wp, allr.sum() - wp))
    dist = np.array(dist)
    p = float(np.mean(dist <= Wobs))
    return float(Wplus), p


def main():
    df = pd.read_csv(CSV_IN)
    df["Algorithm"] = df["Algorithm"].str.replace("\n", " ", regex=False).str.strip()
    ours = df[df.Algorithm == OURS].sort_values("Seed")
    rows = []
    for met in METRICS:
        o = ours[met].values
        om, ol, oh = ci95(o)
        rows.append(dict(Metric=met, Method="SA-HMARL (ours)",
                         Mean=round(om, 1), Std=round(o.std(ddof=1), 1),
                         CI_low=round(ol, 1), CI_high=round(oh, 1),
                         Improvement_pct="", signed_rank_p="", cliffs_delta="", dominance=""))
        for alg in df.Algorithm.unique():
            if alg == OURS:
                continue
            b = df[df.Algorithm == alg].sort_values("Seed")[met].values
            if np.all(b == 0):
                rows.append(dict(Metric=met, Method=alg, Mean="N/A", Std="", CI_low="",
                                 CI_high="", Improvement_pct="", signed_rank_p="",
                                 cliffs_delta="", dominance="no coverage"))
                continue
            bm, bl, bh = ci95(b)
            delta = cliffs_delta(o, b)
            _, p = wilcoxon_signed_rank_exact(o, b)
            imp = (bm - om) / bm * 100.0
            rows.append(dict(Metric=met, Method=alg, Mean=round(bm, 1),
                             Std=round(b.std(ddof=1), 1), CI_low=round(bl, 1),
                             CI_high=round(bh, 1), Improvement_pct=round(imp, 1),
                             signed_rank_p=round(p, 5), cliffs_delta=round(delta, 3),
                             dominance=("complete" if abs(delta) == 1.0 else "partial")))
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    out.to_csv(CSV_OUT, index=False)
    pd.set_option("display.width", 160, "display.max_columns", 20)
    print(out.to_string(index=False))
    print(f"\n[saved] {CSV_OUT}")


if __name__ == "__main__":
    main()
