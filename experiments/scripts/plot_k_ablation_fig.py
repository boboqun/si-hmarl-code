"""
plot_k_ablation_fig.py — Fig. 9 (K-resolution ablation), restyled to match the
other bar charts (serif fonts, hatched black-edged bars, unified width, value
labels, signature blue for the chosen K=5). Panels follow the caption:
(a) global makespan, (b) deadhead distance, (c) task success rate.

Data are the paper's reported values: makespan/success from the D2 text (which
reuse the main SI-HMARL checkpoint at K=5), and deadhead means from
experiments/results_ablation/k_ablation_eval_v2.csv (11,934 m at K=4); the K=5
deadhead is the main-table value (3,792 m) for consistency with the rest of the
paper. Hardcoded so it regenerates anywhere.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import apply_plot_style, format_ax

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'paper_figures', 'Ch5_Ablation')
os.makedirs(OUT_DIR, exist_ok=True)

ELEM_W = 0.8
K_LABELS  = ["4×4", "5×5", "6×6", "7×7", "8×8"]
HERO_IDX  = 1  # K = 5

MAKESPAN     = [6726, 6472, 25609, 30000, 30000]
MAKESPAN_STD = [ 148,  214,  9258,     0,     0]
DEADHEAD     = [11934,  3792, 36430, 41497, 43183]   # deadhead (m); K5 from main table (3,792), others v2-eval means
SUCCESS      = [100, 100, 20, 0, 0]

HERO = "#2878B5"
OTHER = "#659266"
TIMEOUT = {2: "", 3: "timeout", 4: "timeout"}  # K6 partial, K7/K8 full timeout


def _colors():
    return [HERO if i == HERO_IDX else OTHER for i in range(len(K_LABELS))]


def _hatches():
    return ['xx' if i == HERO_IDX else '//' for i in range(len(K_LABELS))]


def _bars(ax, vals, std, title, ylabel, fmt="{:,}", annotate_timeout=False):
    x = np.arange(len(K_LABELS))
    bars = ax.bar(x, vals, width=ELEM_W, color=_colors(), edgecolor='black',
                  linewidth=1.0, alpha=0.85, zorder=3)
    for b, h in zip(bars, _hatches()):
        b.set_hatch(h)
    if std is not None and any(std):
        ax.errorbar(x, vals, yerr=std, fmt='none', ecolor='#2b2b2b',
                    capsize=4, capthick=1.2, elinewidth=1.2, zorder=4)
    ymax = max(v + (s if std else 0) for v, s in zip(vals, std or [0] * len(vals)))
    ax.set_ylim(0, ymax * 1.20)
    for xi, v in enumerate(vals):
        s = std[xi] if std else 0
        ax.text(xi, v + s + ymax * 0.02, fmt.format(int(v)), ha='center', va='bottom',
                fontweight='bold', fontsize=7.5, color='#111111')
        if annotate_timeout and TIMEOUT.get(xi):
            ax.text(xi, v * 0.5, TIMEOUT[xi], ha='center', va='center', rotation=90,
                    fontsize=6.5, style='italic', color='white', zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(K_LABELS, fontweight='bold')
    ax.set_xlabel("Resolution $K$", fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: format(int(v), ',')))
    format_ax(ax)


def main():
    apply_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.9))
    _bars(axes[0], MAKESPAN, MAKESPAN_STD, "(a) Global makespan", "Makespan (ticks)",
          annotate_timeout=True)
    _bars(axes[1], DEADHEAD, None, "(b) Deadhead distance", "Distance (m)")
    _bars(axes[2], SUCCESS, None, "(c) Task success rate", "Success (%)", fmt="{}%")
    axes[2].set_ylim(0, 118)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "ablation_K_comprehensive_analysis.pdf"), facecolor='white')
    fig.savefig(os.path.join(OUT_DIR, "ablation_K_comprehensive_analysis.png"), facecolor='white')
    plt.close(fig)
    print("Saved ablation_K_comprehensive_analysis.{pdf,png}")


if __name__ == "__main__":
    main()
