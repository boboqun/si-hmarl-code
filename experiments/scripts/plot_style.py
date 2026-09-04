"""
plot_style.py — 期刊统一绘图风格
==========================================
所有绘图脚本通过 `from plot_style import apply_plot_style, format_ax` 导入，
确保字体、字号、线宽、颜色等在所有论文配图中完全一致。

排版要求:
  - 所有图表字体和尺寸一致
  - 缩放到印刷尺寸后最小文字不低于 8pt
  - 首选 serif 字体 (Times New Roman)
  - pdf.fonttype=42 嵌入 TrueType (避免审查报错)

单栏宽 3.5" (21 pica), 双栏全宽 7.16"
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── 双栏期刊标准列宽（英寸）──────────────────────────────────────────────────────
COL_WIDTH  = 3.5    # single-column
FULL_WIDTH = 7.16   # double-column (textwidth)

# ── 统一字号体系 ──────────────────────────────────────────────────────────────
# 所有图的 figsize 与下列字号搭配，确保最终印刷 >= 8pt
FONT_SIZE       = 9     # 基础字号
LABEL_SIZE      = 10    # 轴标签
TITLE_SIZE      = 10    # 标题
TICK_SIZE        = 8     # 刻度标签
LEGEND_SIZE     = 8     # 图例
ANNOTATION_SIZE = 7.5   # 图内注释

# ── 统一 rcParams ─────────────────────────────────────────────────────────────
_JOURNAL_RC = {
    # 字体
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'Times', 'Liberation Serif', 'Nimbus Roman', 'DejaVu Serif', 'serif'],
    'mathtext.fontset':  'stix',

    # 字号
    'font.size':         FONT_SIZE,
    'axes.labelsize':    LABEL_SIZE,
    'axes.titlesize':    TITLE_SIZE,
    'xtick.labelsize':   TICK_SIZE,
    'ytick.labelsize':   TICK_SIZE,
    'legend.fontsize':   LEGEND_SIZE,

    # 线宽
    'axes.linewidth':    0.8,
    'grid.linewidth':    0.5,
    'lines.linewidth':   1.5,

    # PDF 嵌入
    'pdf.fonttype':      42,
    'ps.fonttype':       42,

    # DPI
    'figure.dpi':        150,
    'savefig.dpi':       600,
    'savefig.bbox':      'tight',
    'savefig.pad_inches': 0.02,

    # 图例
    'legend.framealpha':  0.9,
    'legend.edgecolor':   '#cccccc',
    'legend.fancybox':    True,
}


def apply_plot_style():
    """在绘图脚本最顶部调用，设置全局 rcParams。"""
    plt.rcParams.update(_JOURNAL_RC)


def format_ax(ax, grid_axis='y'):
    """统一的轴外观：隐藏 top/right spine，添加灰色虚线网格。"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if grid_axis:
        ax.grid(True, axis=grid_axis, linestyle='--', alpha=0.35,
                color='#888888', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def thousands_formatter():
    """返回千分位数字格式器，如 '12,043'。"""
    return ticker.FuncFormatter(lambda x, _: f"{int(x):,}")


def k_formatter():
    """返回 'Nk' 格式器，如 '12k'。"""
    return ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k")
