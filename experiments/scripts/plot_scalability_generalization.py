"""
plot_scalability_generalization.py
==================================
零样本泛化能力 (Zero-Shot Scalability) 评估脚本（重设计版）

左图：分组柱状图 - 各算法在不同尺度下的绝对完工时间（Absolute Makespan）
      MAPPO Flat 以超时上限标注，直观展示失败
右图：对数坐标增长趋势图 - 附理论基准线揭示近理论最优的扩展性
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import apply_plot_style, format_ax, FONT_SIZE, LABEL_SIZE, TICK_SIZE, ANNOTATION_SIZE

apply_plot_style()

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import glob

OUTPUT_DIR = os.path.abspath(os.path.join(PROJECT_DIR, "paper_figures", "Ch5_Scalability"))
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    """Load and merge scalability CSVs.

    Strategy:
      1. Load the 10-seed run (June 11, has 1x/2x/4x + baselines + 0.25x/0.5x planners).
    """
    RESULTS_DIR = os.path.join(PROJECT_DIR, "experiments", "results")

    # Primary: 10-seed results
    primary = os.path.join(RESULTS_DIR, "scalability_results_20260611_084847.csv")

    if not os.path.exists(primary):
        print("[!] 未找到 scalability_results_20260611_084847.csv！")
        return None

    df = pd.read_csv(primary)
    print(f"[*] 加载: {primary}  ({len(df)} rows)")

    # Fill small-scale rows missing from the primary run (SI-HMARL, Standard
    # H-MARL, Heuristic MACPP, MAPPO Flat at 0.25x/0.5x) from the complete
    # April 23 comparison run; primary rows win on conflict.
    fallback = os.path.join(RESULTS_DIR, "scalability_results_20260423_221754.csv")
    if os.path.exists(fallback):
        fb = pd.read_csv(fallback)
        fb["Algorithm"] = fb["Algorithm"].str.replace("Porcelli CACPP", "Porcelli", regex=False)
        fb.loc[fb["Algorithm"] == "Porcelli", "Algorithm"] = "Porcelli CACPP"
        df = pd.concat([df, fb], ignore_index=True)
        print(f"[*] 合并小尺度补充: {fallback}")

    # Keep first occurrence
    df = df.drop_duplicates(subset=["Algorithm", "Scale"], keep="first")
    # Remove 4x (8000m) — exceeds safe-return limit, not plotted
    df = df[df["Scale"] != "4x (8000m)"]
    df["Algorithm"] = df["Algorithm"].str.replace("SA-HMARL", "SI-HMARL", regex=False)
    df["Algorithm"] = df["Algorithm"].str.replace("Porcelli CACPP", "Porcelli", regex=False)
    return df


def plot_scalability():
    df = load_data()
    if df is None:
        return

    # ── 配置 ────────────────────────────────────────────────────────────────
    ALGOS = ["Our SI-HMARL", "Standard H-MARL", "Heuristic MACPP", "MAPPO Flat", "Porcelli", "AG-CVG", "Eker DP"]
    SCALES_ORDERED = ["0.25x (500m)", "0.5x (1000m)", "1x (2000m)", "2x (4000m)"]
    SCALE_LABELS   = ["500×500 m", "1000×1000 m", "2000×2000 m", "4000×4000 m"]
    # 各尺度超时上限
    TIMEOUT_LIMITS = {"0.25x (500m)": max(3000, int(30_000 * 0.0625)), 
                      "0.5x (1000m)": max(3000, int(30_000 * 0.25)), 
                      "1x (2000m)": 30_000, 
                      "2x (4000m)": int(30_000 * 4)}

    COLORS = {                          # shared palette (matches the other restyled figures)
        "Our SI-HMARL":    "#2878B5",   # blue (was red)
        "Standard H-MARL": "#E68A5C",   # orange (was blue -> avoids clashing with SI-HMARL)
        "Heuristic MACPP": "#659266",   # green
        "MAPPO Flat":      "#C25E5E",   # red
        "Porcelli":        "#F8CB7F",   # tan
        "AG-CVG":          "#9467BD",   # purple
        "Eker DP":         "#8C564B",   # brown
    }
    HATCHES = {
        "Our SI-HMARL":    "//",
        "Standard H-MARL": "\\\\",
        "Heuristic MACPP": "..",
        "MAPPO Flat":      "xx",
        "Porcelli":        "**",
        "AG-CVG":          "++",
        "Eker DP":         "oo",
    }

    # ── 构建数据矩阵 ─────────────────────────────────────────────────────────
    # absolute makespan：成功则用 Makespan Multiplier 列（实际步数），失败用超时上限
    abs_makespan = {algo: [] for algo in ALGOS}
    is_failed    = {algo: [] for algo in ALGOS}

    for scale in SCALES_ORDERED:
        timeout = TIMEOUT_LIMITS[scale]
        for algo in ALGOS:
            row = df[(df["Algorithm"] == algo) & (df["Scale"] == scale)]
            if row.empty:
                abs_makespan[algo].append(timeout)
                is_failed[algo].append(True)
                continue
            sr  = float(row["Success Rate (%)"].values[0])
            msp = row["Makespan Multiplier"].values[0]
            if sr > 0 and not pd.isna(msp):
                abs_makespan[algo].append(float(msp))
                is_failed[algo].append(False)
            else:
                abs_makespan[algo].append(timeout)
                is_failed[algo].append(True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ══════════════════════════════════════════════════════════════════════════
    # 左图：分组柱状图 - 绝对完工时间 (Absolute Makespan)
    # ══════════════════════════════════════════════════════════════════════════
    ax0 = axes[0]
    n_scales = len(SCALES_ORDERED)
    n_algos  = len(ALGOS)
    group_w  = 0.86                   # total width occupied by one scale group
    bar_w    = group_w / n_algos      # per-method width -> no overlap for 7 methods
    x_base   = np.arange(n_scales)
    Y_FLOOR  = 300                    # log-axis floor (below the smallest makespan)

    for i, algo in enumerate(ALGOS):
        offsets = (i - (n_algos - 1) / 2) * bar_w   # symmetric, centered on each group
        vals    = abs_makespan[algo]
        failed  = is_failed[algo]

        bars = ax0.bar(
            x_base + offsets, vals,
            width=bar_w * 0.9, label=algo,
            color=COLORS[algo], edgecolor='black',
            linewidth=0.7, alpha=0.85,
            hatch=HATCHES[algo], zorder=3,
        )

        for bar, f, v in zip(bars, failed, vals):
            cx = bar.get_x() + bar.get_width() / 2
            if f:
                # failure: small cross marker
                ax0.text(cx, bar.get_height() * 1.04, "✕", ha='center', va='bottom',
                         fontsize=7, color='#b00000', fontweight='bold', zorder=5)
            else:
                # success: vertical value label (fits the narrow bars)
                label = f"{v/1000:.1f}k" if v >= 1000 else f"{int(v)}"
                ax0.text(cx, bar.get_height() * 1.05, label, ha='center', va='bottom',
                         fontsize=6, color='#111111', fontweight='bold', rotation=90, zorder=5)

    ax0.set_yscale('log')
    ax0.set_ylim(Y_FLOOR, 2.2e5)
    ax0.set_xticks(x_base)
    ax0.set_xticklabels(SCALE_LABELS, fontsize=11)
    ax0.set_xlabel("Map Scale (Area)", fontsize=12, fontweight='bold')
    ax0.set_ylabel("Global Makespan (steps, log)", fontsize=12, fontweight='bold')
    ax0.set_title("(a) Absolute Makespan at Each Scale  ( ✕ = timeout / not completed )",
                  fontsize=11, fontweight='bold', pad=8)
    ax0.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: (f"{x/1000:.0f}k" if x >= 1000 else f"{int(x)}")))
    ax0.legend(fontsize=8, framealpha=0.9, loc='upper left', ncol=2,
               columnspacing=0.8, handlelength=1.4, handletextpad=0.4)
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.grid(axis='y', which='both', linestyle='--', alpha=0.30, zorder=0)
    ax0.set_axisbelow(True)

    # ══════════════════════════════════════════════════════════════════════════
    # 右图：标注热力图 — 各算法相对 SI-HMARL 的完工时间倍数
    #   行  = 4 个算法   列 = 3 个尺度
    #   颜色 = 倍数（白/绿 = 好, 深红 = 差/超时）
    #   单元格文字 = 倍数 + 成功率
    # ══════════════════════════════════════════════════════════════════════════
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches

    ax1 = axes[1]
    ax1.set_aspect('equal')

    sa_vals = abs_makespan["Our SI-HMARL"]   # list [1x, 2x, 4x]

    HMAP_ALGOS  = ["Our SI-HMARL", "Heuristic MACPP", "Standard H-MARL", "MAPPO Flat", "Porcelli", "AG-CVG", "Eker DP"]
    HMAP_LABELS = ["Our SI-HMARL\n(Proposed)", "Heuristic MACPP\n(No learning)",
                   "Standard H-MARL\n(No energy credit)", "MAPPO Flat\n(No hierarchy)",
                   "Porcelli CACPP\n(Porcelli 2025)", "AG-CVG\n(Karapetyan 2024)", "Eker DP\n(Eker 2025)"]
    COL_LABELS  = ["500×500 m\n(0.25 km²)", "1000×1000 m\n(1.0 km²)", "2000×2000 m\n(4.0 km²)", "4000×4000 m\n(16.0 km²)"]
    n_rows, n_cols = len(HMAP_ALGOS), len(SCALES_ORDERED)

    # ── Build ratio matrix and annotation strings ─────────────────────────────
    ratio_matrix = np.zeros((n_rows, n_cols))
    cell_text    = [[""]*n_cols for _ in range(n_rows)]
    is_timeout   = [[False]*n_cols for _ in range(n_rows)]
    is_partial   = [[False]*n_cols for _ in range(n_rows)]

    for i, algo in enumerate(HMAP_ALGOS):
        row_df = df[df["Algorithm"] == algo]
        for j, scale in enumerate(SCALES_ORDERED):
            row = row_df[row_df["Scale"] == scale]
            sr  = float(row["Success Rate (%)"].values[0]) if not row.empty else 0.0
            timeout_val = TIMEOUT_LIMITS[scale]

            if is_failed[algo][j]:           # 0 % success → use timeout
                ratio = timeout_val / sa_vals[j]
                is_timeout[i][j] = True
                cell_text[i][j] = f"Timeout\n(0% Succ)"
            elif sr < 100.0:                 # partial success
                ratio = abs_makespan[algo][j] / sa_vals[j]
                is_partial[i][j] = True
                cell_text[i][j] = f"{ratio:.2f}×\n({sr:.0f}% Succ)"
            else:                            # full success
                ratio = abs_makespan[algo][j] / sa_vals[j]
                cell_text[i][j] = f"{ratio:.2f}×\n(100% Succ)"
            ratio_matrix[i][j] = ratio

    # ── Custom colormap: white at 1.0 → yellow → orange → deep red ────────────
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "hmap_cmap",
        [(0.00, "#2ca02c"),   # deep green  (< 1.0, shouldn't occur but safety)
         (0.00, "#ffffff"),   # white  at ratio = 1.0
         (0.07, "#fff5cc"),   # very light yellow
         (0.25, "#ffd966"),   # yellow
         (0.45, "#f4a261"),   # orange
         (0.70, "#e63946"),   # red
         (1.00, "#7b0000")],  # dark crimson (max)
    )
    # Map ratio 1.0 → 0.0 in the colormap, ratio_max → 1.0
    ratio_max = max(ratio_matrix.max(), 7.0)
    norm = mcolors.Normalize(vmin=1.0, vmax=ratio_max)

    # ── Draw cells ────────────────────────────────────────────────────────────
    cell_h, cell_w = 1.0, 1.6   # height and width of each cell in data units
    for i in range(n_rows):
        for j in range(n_cols):
            r = ratio_matrix[i][j]
            color = cmap(norm(r))
            rect = mpatches.FancyBboxPatch(
                (j * cell_w, (n_rows - 1 - i) * cell_h),
                cell_w, cell_h,
                boxstyle="round,pad=0.04",
                linewidth=0.8,
                edgecolor="#dddddd",
                facecolor=color,
                zorder=2,
            )
            ax1.add_patch(rect)

            # Hatch pattern for timeout cells
            if is_timeout[i][j]:
                hatch_rect = mpatches.FancyBboxPatch(
                    (j * cell_w, (n_rows - 1 - i) * cell_h),
                    cell_w, cell_h,
                    boxstyle="round,pad=0.04",
                    linewidth=0,
                    edgecolor="#ffffff",
                    facecolor="none",
                    hatch="////",
                    zorder=3,
                    alpha=0.25,
                )
                ax1.add_patch(hatch_rect)

            # Text color: dark on light cells, white on dark
            luminance = 0.299*color[0] + 0.587*color[1] + 0.114*color[2]
            txt_color = "#111111" if luminance > 0.45 else "#ffffff"

            lines = cell_text[i][j].split("\n")
            cx = j * cell_w + cell_w / 2
            cy = (n_rows - 1 - i) * cell_h + cell_h / 2

            # First line: ratio (larger, bold)
            ax1.text(cx, cy + 0.14, lines[0],
                     ha="center", va="center", fontsize=13,
                     fontweight="bold", color=txt_color, zorder=4)
            # Second line: success rate (smaller)
            ax1.text(cx, cy - 0.20, lines[1] if len(lines) > 1 else "",
                     ha="center", va="center", fontsize=9.5,
                     color=txt_color, style="italic", zorder=4)

    # ── Row labels (algorithm names on the left) ───────────────────────────────
    for i, lbl in enumerate(HMAP_LABELS):
        cy = (n_rows - 1 - i) * cell_h + cell_h / 2
        ax1.text(-0.12, cy, lbl,
                 ha="right", va="center", fontsize=10,
                 transform=ax1.get_yaxis_transform(),
                 fontweight="bold" if i == 0 else "normal")

    # ── Column labels (scale names on top) ────────────────────────────────────
    for j, lbl in enumerate(COL_LABELS):
        cx = j * cell_w + cell_w / 2
        ax1.text(cx, n_rows * cell_h + 0.08, lbl,
                 ha="center", va="bottom", fontsize=11, fontweight="bold")

    # ── SI-HMARL row border highlight ─────────────────────────────────────────
    highlight = mpatches.FancyBboxPatch(
        (0, (n_rows - 1) * cell_h),
        n_cols * cell_w, cell_h,
        boxstyle="round,pad=0.05",
        linewidth=2.5,
        edgecolor="#2878B5",
        facecolor="none",
        zorder=5,
    )
    ax1.add_patch(highlight)

    # ── Colorbar ──────────────────────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax1, fraction=0.046, pad=0.04, aspect=20)
    cbar.set_label("Relative Makespan vs. SI-HMARL  (×)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_ticks([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    cbar.set_ticklabels(["1.0×\n(SI-HMARL)", "2×", "3×", "4×", "5×", "6×", "7×+"])

    # ── Axes cosmetics ────────────────────────────────────────────────────────
    ax1.set_xlim(-0.05, n_cols * cell_w + 0.05)
    ax1.set_ylim(-0.05, n_rows * cell_h + 0.35)
    ax1.axis("off")
    ax1.set_title(
        "(b) Fixed-Interface Scale Transfer: Performance Heatmap\n"
        "Color = Relative Makespan  |  Green: Faster  |  White: 1.0x  |  Red: Slower / Fail",
        fontsize=12, fontweight='bold', pad=14,
    )

    plt.tight_layout(pad=2.0)
    save_path = os.path.join(OUTPUT_DIR, "Fig_5_5_Scalability.pdf")
    plt.savefig(save_path, dpi=600, format='pdf', bbox_inches='tight')
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [✓] 重设计版跨尺度泛化图表已保存: {save_path}")


if __name__ == "__main__":
    plot_scalability()
