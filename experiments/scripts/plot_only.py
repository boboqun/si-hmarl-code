import sys
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import apply_plot_style, format_ax

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'experiments', 'results')
PLOTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'paper_figures', 'Ch5_Performance')
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Unified geometry ─────────────────────────────────────────────────────────
# Identical figure size and element width across all three comparison figures so
# that bars and boxes read at the same visual width (previously 0.55 bar vs 0.45
# box, and 6 vs 7 categories, which made the widths look inconsistent).
FIGSIZE = (7.2, 4.4)
ELEM_W  = 0.6

PALETTE = {
    "Our\nSI-HMARL": "#2878B5",
    "Standard\nH-MARL": "#E68A5C",
    "AG-CVG\n(Karapetyan 2024)": "#9467BD",
    "Porcelli CACPP\n(Porcelli 2025)": "#F8CB7F",
    "Eker DP\n(Eker 2025)": "#8C564B",
    "MAPPO\nFlat": "#C25E5E",
    "SDA-MAPPO\n(Chen 2026)": "#7F7F7F",
    "Heuristic\nMACPP": "#659266",
}
HATCHES = ['//', '\\\\', '||', '--', 'xx', '..', 'oo']

# Shared box-plot styling (width identical to the bar width above)
_BOX_KW = dict(
    showmeans=True, fliersize=0, width=ELEM_W, zorder=2, legend=False,
    meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 5},
    boxprops={'edgecolor': 'black', 'linewidth': 1.0, 'alpha': 0.85},
    whiskerprops={'color': 'black', 'linewidth': 1.0},
    capprops={'color': 'black', 'linewidth': 1.0},
    medianprops={'color': '#2b2b2b', 'linewidth': 1.3, 'linestyle': '-'},
)


def _save(fig, name):
    fig.savefig(os.path.join(PLOTS_DIR, name + ".pdf"), facecolor='white')
    fig.savefig(os.path.join(PLOTS_DIR, name + ".png"), facecolor='white')
    plt.close(fig)


def _na_marker(ax, idx, label="N/A\n(no coverage)"):
    """Keep the MAPPO slot present (for consistent category widths) but mark it N/A."""
    y0, y1 = ax.get_ylim()
    ax.text(idx, y0 + (y1 - y0) * 0.04, label, ha='center', va='bottom',
            fontsize=7, style='italic', color='#999999', zorder=5)


def _style_xaxis(ax, algos):
    ax.set_xticks(range(len(algos)))
    ax.set_xticklabels(algos, fontweight='bold', rotation=18, ha='right', rotation_mode='anchor')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: format(int(v), ',')))


def plot_academic_charts(df):
    apply_plot_style()
    algos = list(PALETTE.keys())
    colors = list(PALETTE.values())
    mappo_idx = algos.index("MAPPO\nFlat")
    sda_idx = algos.index("SDA-MAPPO\n(Chen 2026)")

    # 1. Global Makespan (bar; MAPPO shown at its 30,000-step timeout) ──────────
    fig, ax = plt.subplots(figsize=FIGSIZE)
    g = df.groupby("Algorithm")["Global Makespan"].agg(['mean', 'std']).reindex(algos)
    x = np.arange(len(algos))
    bars = ax.bar(x, g['mean'], color=colors, edgecolor='black', linewidth=1.0,
                  width=ELEM_W, zorder=3, alpha=0.85)
    for i, b in enumerate(bars):
        b.set_hatch(HATCHES[i % len(HATCHES)])
    ax.errorbar(x, g['mean'], yerr=g['std'], fmt='none', ecolor='#2b2b2b',
                capsize=4, capthick=1.2, elinewidth=1.2, zorder=4)
    ymax = (g['mean'] + g['std']).max()
    ax.set_ylim(0, ymax * 1.2)
    for b, std in zip(bars, g['std']):
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + std + ymax * 0.03,
                f"{round(h):,}", ha='center', va='bottom', fontweight='bold',
                fontsize=8, color='#111111')
    ax.set_ylabel("Execution Steps (Makespan)", fontweight='bold')
    _style_xaxis(ax, algos)
    format_ax(ax)
    fig.tight_layout()
    _save(fig, "Fig_5_2_Makespan_Comparison")

    # 2. Deadhead (box; MAPPO ~391k drift is not a meaningful return distance) ──
    fig, ax = plt.subplots(figsize=FIGSIZE)
    df_c = df[~df["Algorithm"].str.contains("MAPPO")]
    sns.boxplot(data=df_c, x="Algorithm", y="Deadhead Distance (m)", order=algos,
                hue="Algorithm", palette=PALETTE, ax=ax, **_BOX_KW)
    sns.stripplot(data=df_c, x="Algorithm", y="Deadhead Distance (m)", order=algos,
                  hue="Algorithm", palette=PALETTE, alpha=0.4, jitter=0.15, size=4,
                  ax=ax, zorder=1, legend=False)
    ax.set_ylabel("Deadhead Distance (meters)", fontweight='bold')
    ax.set_xlabel("")
    _style_xaxis(ax, algos)
    format_ax(ax)
    _na_marker(ax, mappo_idx)
    _na_marker(ax, sda_idx)
    fig.tight_layout()
    _save(fig, "Fig_5_3_Deadhead_Comparison")

    # 3. Redundant Scan (box; MAPPO never completes coverage -> N/A) ────────────
    fig, ax = plt.subplots(figsize=FIGSIZE)
    df_c = df[~df["Algorithm"].str.contains("MAPPO")]
    sns.boxplot(data=df_c, x="Algorithm", y="Redundant Scan (m)", order=algos,
                hue="Algorithm", palette=PALETTE, ax=ax, **_BOX_KW)
    sns.stripplot(data=df_c, x="Algorithm", y="Redundant Scan (m)", order=algos,
                  hue="Algorithm", palette=PALETTE, alpha=0.4, jitter=0.15, size=4,
                  ax=ax, zorder=1, legend=False)
    ax.set_ylabel("Redundant Scan Distance (meters)", fontweight='bold')
    ax.set_xlabel("")
    _style_xaxis(ax, algos)
    format_ax(ax)
    _na_marker(ax, mappo_idx)
    _na_marker(ax, sda_idx)
    fig.tight_layout()
    _save(fig, "Fig_5_4_RedundantScan_Comparison")

    # 4. Total Wasted Stacked (bar) ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=FIGSIZE)
    dh = df.groupby("Algorithm")["Deadhead Distance (m)"].mean().reindex(algos)
    rs = df.groupby("Algorithm")["Redundant Scan (m)"].mean().reindex(algos)
    x = np.arange(len(algos))
    ax.bar(x, dh, width=ELEM_W, label="Deadhead (Return-to-Recharge)", color=colors,
           edgecolor='black', linewidth=1.0, alpha=0.85, hatch='//')
    ax.bar(x, rs, width=ELEM_W, bottom=dh, label="Redundant Scan (Re-coverage)", color=colors,
           edgecolor='black', linewidth=1.0, alpha=0.45, hatch='xx')
    ymax = (dh + rs).max()
    ax.set_ylim(0, ymax * 1.25)
    for i, (a, b) in enumerate(zip(dh, rs)):
        t = a + b
        if not np.isnan(t):
            ax.text(x[i], t + ymax * 0.02, f"{round(t):,}", ha='center', va='bottom',
                    fontweight='bold', fontsize=8, color='#111111')
    ax.set_ylabel("Total Wasted Flight Distance (meters)", fontweight='bold')
    _style_xaxis(ax, algos)
    ax.legend(loc='upper left', framealpha=0.9)
    format_ax(ax)
    fig.tight_layout()
    _save(fig, "Fig_5_5_TotalWasted_Stacked")


if __name__ == "__main__":
    csv_path = os.path.join(RESULTS_DIR, "performance_metrics.csv")
    df = pd.read_csv(csv_path)
    df["Algorithm"] = df["Algorithm"].str.replace("SA-HMARL", "SI-HMARL", regex=False)
    sda = pd.read_csv(os.path.join(RESULTS_DIR, "sda_mappo_eval_results.csv"))
    sda_rows = pd.DataFrame({
        "Algorithm": "SDA-MAPPO\n(Chen 2026)",
        "Seed": sda["seed"],
        "Global Makespan": sda["makespan"],
        "Deadhead Distance (m)": sda["deadhead"],
        "Redundant Scan (m)": np.nan,
        "Total Wasted (m)": np.nan,
    })
    df = pd.concat([df, sda_rows], ignore_index=True)
    plot_academic_charts(df)
    print("Plots generated successfully!")
