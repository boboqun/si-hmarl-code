"""
plot_component_ablation.py — Fig. 8 (Component ablation), restyled to match the
performance comparison charts (plot_only.py): serif fonts, hatched bars with
black edges, unified bar width (0.6), bold value labels, error bars, the shared
per-variant palette, and the signature blue (#2878B5) for the full SI-HMARL model.

Values are the paper's reported means: the learning-component ablation table
(10 seeds) for panels (a)/(b), and the paired node-constrained status audit table
for panel (c). They are hardcoded so the figure regenerates anywhere, without
TensorBoard / ray_results.
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

ELEM_W = 0.8  # wider bars (matplotlib default); panels are packed, so 0.6 read thin
TICK_FS = 7.5

# (a)/(b) learning-component variants — means/std from the component-ablation table
VARIANTS     = ["Full", "Proactive", "Reactive", "No credit", "Flat reward", "MLP commander"]
MAKESPAN     = [6472, 6505,  6909, 6886,  7076,  7182]
MAKESPAN_STD = [ 214,  212,   320,  229,   204,   204]
DEADHEAD     = [3792, 3425, 10761, 9473, 11628,  9947]
DEADHEAD_STD = [1162, 1364,  1968, 1234,  1420,  2040]
# Shared palette (comparison-chart colors); Full keeps the signature SI-HMARL blue
V_COLORS = ["#2878B5", "#E68A5C", "#9467BD", "#F8CB7F", "#8C564B", "#659266"]
V_HATCH  = ['//', '\\\\', None, '--', 'xx', '..']   # 竖线纹理改手绘(内置 hatch 相位锚定画布,柱内分布不可控)

# (c) paired node-constrained status audit
STATUS_LABELS = ["Continuous", "Node", "Flat + node"]
STATUS_PCT    = [100, 90, 20]
STATUS_COLORS = ["#659266", "#E68A5C", "#C25E5E"]   # green / orange / red (traffic-light)
STATUS_HATCH  = ['//', None, 'xx']
STATUS_NOTE   = ["", "1 crash", "8 crashes"]


def _manual_vhatch(ax, bar, n=5, lw=0.9, alpha=0.9):
    """竖线纹理手绘版:以柱中心对称、两侧等边距,替代相位不可控的 '||'。"""
    bx, bw, bh = bar.get_x(), bar.get_width(), bar.get_height()
    for i in range(n):
        xi = bx + bw * (i + 0.5) / n
        ax.plot([xi, xi], [0, bh], color='black', lw=lw, alpha=alpha,
                zorder=3.4, solid_capstyle='butt')


def _xticklabels(ax, labels, rotation=30, fs=TICK_FS):
    ax.set_xticks(np.arange(len(labels)))
    ha = 'right' if rotation else 'center'
    ax.set_xticklabels(labels, fontweight='bold', rotation=rotation, ha=ha,
                       rotation_mode='anchor', fontsize=fs)


def _value_bars(ax, vals, std, title, ylabel):
    x = np.arange(len(VARIANTS))
    bars = ax.bar(x, vals, width=ELEM_W, color=V_COLORS, edgecolor='black',
                  linewidth=1.0, alpha=0.85, zorder=3)
    for b, h in zip(bars, V_HATCH):
        if h is None:
            _manual_vhatch(ax, b)
        else:
            b.set_hatch(h)
    ax.errorbar(x, vals, yerr=std, fmt='none', ecolor='#2b2b2b',
                capsize=4, capthick=1.2, elinewidth=1.2, zorder=4)
    ymax = max(v + s for v, s in zip(vals, std))
    ax.set_ylim(0, ymax * 1.20)
    for xi, (v, s) in enumerate(zip(vals, std)):
        ax.text(xi, v + s + ymax * 0.02, f"{int(v):,}", ha='center', va='bottom',
                fontweight='bold', fontsize=7, color='#111111')
    _xticklabels(ax, VARIANTS)
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: format(int(v), ',')))
    format_ax(ax)


def main():
    apply_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.5),
                             gridspec_kw={'width_ratios': [6, 6, 3]})

    _value_bars(axes[0], MAKESPAN, MAKESPAN_STD, "(a) Global makespan", "Makespan (ticks)")
    _value_bars(axes[1], DEADHEAD, DEADHEAD_STD, "(b) Deadhead distance", "Deadhead (m)")

    # (c) status audit
    ax = axes[2]
    x = np.arange(len(STATUS_LABELS))
    bars = ax.bar(x, STATUS_PCT, width=ELEM_W, color=STATUS_COLORS, edgecolor='black',
                  linewidth=1.0, alpha=0.85, zorder=3)
    for b, h in zip(bars, STATUS_HATCH):
        if h is None:
            _manual_vhatch(ax, b)
        else:
            b.set_hatch(h)
    ax.set_ylim(0, 122)
    for xi, (p, note) in enumerate(zip(STATUS_PCT, STATUS_NOTE)):
        ax.text(xi, p + 3, f"{p}%", ha='center', va='bottom',
                fontweight='bold', fontsize=8, color='#111111')
        if note:
            ny = p * 0.5 if p >= 50 else p + 16
            ax.text(xi, ny, note, ha='center', va='center', fontsize=7,
                    style='italic', color='#333333', zorder=5,
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                              alpha=0.78, edgecolor='none'))
    _xticklabels(ax, STATUS_LABELS, rotation=35, fs=6.5)
    ax.set_ylabel("Completed seeds (%)", fontweight='bold')
    ax.set_title("(c) Node-constrained status audit", fontweight='bold')
    format_ax(ax)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "Fig_5_4_Component_Ablation.pdf"), facecolor='white')
    fig.savefig(os.path.join(OUT_DIR, "Fig_5_4_Component_Ablation.png"), facecolor='white')
    plt.close(fig)
    print("Saved Fig_5_4_Component_Ablation.{pdf,png}")


if __name__ == "__main__":
    main()
