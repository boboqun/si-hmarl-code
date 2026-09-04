"""
plot_learning_curves.py
=======================
绘制真实训练日志的收敛曲线（Episode Makespan = episode_len_mean）。

数据来源（全部为真实 RLlib progress.csv，无任何 mock）：
  SI-HMARL (2段拼接，代表标准系统):
    主段:   PPO_hierarchical_coverage_v2_env_2026-04-20_12-02-08  steps 0→13.8M
    续训A:  PPO_hierarchical_coverage_v2_env_2026-05-05_20-35-13  steps 13.8M→79.4M（无缝接续）
    ⚠ 续训B (May-10, 0→101M) 使用了修改后的奖励函数（去除错峰奖励以追求极致 ep_len），
      属于不同系统配置，仅用于论文扩展讨论，不纳入此主对比图。

  Standard H-MARL:
    续训:   PPO_standard_coverage_env_2026-05-11_00-40-58  steps 0→48M（独立完整运行）

  MAPPO Flat:
    主段:   PPO_flat_coverage_env_2026-04-14_21-25-42      steps 0→4M（ep_len始终~30000，未收敛）

展示窗口：0 → 6000 训练轮次。
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import apply_plot_style, format_ax, FONT_SIZE, ANNOTATION_SIZE

apply_plot_style()

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR  = os.path.abspath(os.path.join(PROJECT_DIR, "paper_figures", "Ch5_Learning_Curves"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

EP_LEN_COL = "env_runners/episode_len_mean"

# ── 真实 CSV 路径 ─────────────────────────────────────────────────────────────
SA_SEGS = [
    "<RAY_RESULTS_DIR>/PPO_hierarchical_coverage_v2_env_2026-04-20_12-02-0874y8y9zw/progress.csv",
    "<RAY_RESULTS_DIR>/PPO_hierarchical_coverage_v2_env_2026-05-05_20-35-13rvwck0xy/progress.csv",
    # 续训B (May-10) 使用修改后奖励函数（去除错峰奖励），不纳入主对比图
    # "<RAY_RESULTS_DIR>/PPO_hierarchical_coverage_v2_env_2026-05-10_15-53-13w8fts8g6/progress.csv",
]
# Standard H-MARL: 公平版（超参对齐，May-12, 6000 轮独立运行）
STD_CSV  = "<RAY_RESULTS_DIR>/PPO_standard_coverage_fair_env_2026-05-12_15-20-54qthf7qbx/progress.csv"
# MAPPO Flat: 公平版（超参对齐，May-15, 1676 轮后 NaN 崩溃，1476 行有效数据）
FLAT_CSV = "<RAY_RESULTS_DIR>/PPO_flat_coverage_fair_env_2026-05-15_08-11-15hwrsg0ki/progress.csv"

STYLE_CONFIG = {
    "Our SI-HMARL":       {"color": "#2878B5", "linestyle": "-",  "zorder": 4},
    "Standard H-MARL":    {"color": "#E68A5C", "linestyle": "--", "zorder": 3},
    "MAPPO Flat": {"color": "#7f7f7f", "linestyle": ":",  "zorder": 2},
}


def load_single(csv_path: str) -> pd.DataFrame:
    """读取单段 CSV，返回 [timesteps_total, episode_len_mean] 非空行。"""
    assert os.path.exists(csv_path), f"[ERROR] 文件不存在: {csv_path}"
    df = pd.read_csv(csv_path, usecols=["timesteps_total", EP_LEN_COL])
    df = df.rename(columns={EP_LEN_COL: "episode_len_mean"})
    df = df.dropna(subset=["episode_len_mean"]).reset_index(drop=True)
    return df


def load_sa_hmarl() -> pd.DataFrame:
    """
    拼接 SI-HMARL 两段真实训练日志（标准系统，相同奖励函数）。
      主段:   Apr-20, timesteps 0 → 13.8M
      续训A:  May-05, timesteps 13.8M → 79.4M（无缝接续，相同奖励配置）
    续训B (May-10) 已修改奖励函数，不纳入此图。
    """
    seg_main = load_single(SA_SEGS[0])
    seg_a    = load_single(SA_SEGS[1])

    # 确认续训A无缝接续主段
    main_max = seg_main["timesteps_total"].max()
    a_min    = seg_a["timesteps_total"].min()
    print(f"  SI-HMARL 主段末尾: {main_max:,.0f}  续训A起点: {a_min:,.0f}")
    assert abs(main_max - a_min) < 50_000, \
        f"主段与续训A之间有间隔 {main_max - a_min:,.0f} steps，请检查"

    df = pd.concat([seg_main, seg_a], ignore_index=True)
    df = df.sort_values("timesteps_total").reset_index(drop=True)
    print(f"  SI-HMARL 合并后: {len(df)} 行, steps {df['timesteps_total'].min():,.0f} → "
          f"{df['timesteps_total'].max():,.0f}")
    return df


def load_standard_hmarl() -> pd.DataFrame:
    """
    公平版（May-12, 超参对齐, 0→48M steps）。
    独立新运行（timesteps从0开始），ep_len从~10522降至~9358。
    """
    df = load_single(STD_CSV)
    print(f"  Standard H-MARL: {len(df)} 行, steps {df['timesteps_total'].min():,.0f} → "
          f"{df['timesteps_total'].max():,.0f}, "
          f"ep_len {df['episode_len_mean'].min():.0f}→{df['episode_len_mean'].iloc[-1]:.0f}")
    return df


def load_mappo() -> pd.DataFrame:
    df = load_single(FLAT_CSV)
    print(f"  MAPPO Flat: {len(df)} 行, steps {df['timesteps_total'].min():,.0f} → "
          f"{df['timesteps_total'].max():,.0f}, "
          f"ep_len {df['episode_len_mean'].mean():.0f} (几乎不收敛)")
    return df


def smooth(y_arr, window=80):
    s = pd.Series(y_arr)
    sm  = s.rolling(window=window, min_periods=1, center=True).mean()
    std = s.rolling(window=window, min_periods=1, center=True).std().fillna(0)
    return sm.values, std.values


def draw_break(ax_upper, ax_lower, d=0.012):
    kw_u = dict(transform=ax_upper.transAxes, color='k', clip_on=False, linewidth=1.1)
    ax_upper.plot((-d, +d), (-d * 2.5, +d * 2.5), **kw_u)
    ax_upper.plot((1 - d, 1 + d), (-d * 2.5, +d * 2.5), **kw_u)
    kw_l = dict(transform=ax_lower.transAxes, color='k', clip_on=False, linewidth=1.1)
    ax_lower.plot((-d, +d), (1 - d, 1 + d), **kw_l)
    ax_lower.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw_l)


def plot_learning_curves():
    print("[*] 读取真实训练日志 ...")

    df_sa  = load_sa_hmarl()
    df_std = load_standard_hmarl()
    df_mp  = load_mappo()

    # ── X 轴：使用训练轮数（iteration index）使曲线对齐到同一尺度 ────────
    # SI-HMARL ~6000轮, Standard H-MARL ~5896轮有效, MAPPO ~1402轮有效（1676轮后NaN崩溃）
    def add_iteration(df):
        df = df.copy().reset_index(drop=True)
        df["iteration"] = np.arange(1, len(df) + 1)
        return df

    df_sa  = add_iteration(df_sa)
    df_std = add_iteration(df_std)

    # MAPPO 仅 ~1402 轮有效数据（1676轮后NaN崩溃），延伸最后值到 max_iter 以便视觉对比
    df_mp = df_mp.copy().reset_index(drop=True)
    df_mp["iteration"] = np.arange(1, len(df_mp) + 1)
    last_mp_ep  = df_mp["episode_len_mean"].iloc[-1]
    max_iter    = max(len(df_sa), len(df_std))   # 6000
    df_mp_extra = pd.DataFrame({"iteration":        [df_mp["iteration"].max() + 1, max_iter],
                                 "episode_len_mean": [last_mp_ep, last_mp_ep]})
    df_mp_full  = pd.concat([df_mp, df_mp_extra], ignore_index=True)

    sa_iters  = len(df_sa)
    std_iters = len(df_std)
    mp_valid_iters = len(df_mp)
    print(f"\n  轮数对比: SI-HMARL={sa_iters}, Standard H-MARL={std_iters}, MAPPO={mp_valid_iters}")
    print(f"  注：续训B (May-10, 修改奖励) 未纳入，仅用于扩展讨论")

    series = {
        "Our SI-HMARL":       df_sa,
        "Standard H-MARL":    df_std,
        "MAPPO Flat": df_mp_full,
    }

    # ── 单轴 + 对数 Y 刻度 ──────────────────────────────────────────────────────
    # episode makespan 跨越 ~6k(SI-HMARL) -> ~30k(MAPPO Flat)。对数纵轴让三条曲线
    # 同屏可比，并清晰呈现“共同起点 -> 分叉”的收敛动态（替代旧的三段断轴，去除割裂与留白）。
    fig, ax = plt.subplots(figsize=(7.16, 3.8))

    for algo, df in series.items():
        x_iter = df["iteration"].values
        y      = df["episode_len_mean"].values
        win    = 100 if len(y) > 200 else max(5, len(y) // 10)
        sm, sd = smooth(y, win)
        cfg = STYLE_CONFIG[algo]
        ax.plot(x_iter, sm, label=algo, color=cfg["color"],
                linewidth=2.0, linestyle=cfg["linestyle"], zorder=cfg["zorder"])
        ax.fill_between(x_iter, sm - 0.5 * sd, sm + 0.5 * sd,
                        color=cfg["color"], alpha=0.13, edgecolor="none",
                        zorder=cfg["zorder"] - 1)

    # 对数纵轴：自然容纳 6k -> 30k 的量级差，同时保留收敛动态
    ax.set_yscale("log")
    ax.set_xlim(0, max_iter)
    ax.set_ylim(5600, 33000)
    ax.yaxis.set_major_locator(ticker.FixedLocator([6000, 8000, 10000, 15000, 20000, 30000]))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v/1000:g}k"))

    ax.annotate("MAPPO Flat saturates near 30k\n(never completes within budget)",
                xy=(0.32, 0.84), xycoords="axes fraction",
                fontsize=ANNOTATION_SIZE, color="#555555", style="italic")

    ax.set_title("Training convergence: episode makespan",
                 fontweight="bold", pad=8)
    ax.set_xlabel("Training iterations (rounds)", fontweight="bold")
    ax.set_ylabel(r"Episode makespan (steps), log scale  $\downarrow$", fontweight="bold")

    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    ax.legend(seen.values(), seen.keys(), loc="center right",
              framealpha=0.9, edgecolor="#cccccc")

    format_ax(ax)
    fig.tight_layout()

    # ── 保存 ──────────────────────────────────────────────────────────────────
    save_path = os.path.join(OUTPUT_DIR, "Fig_3_3_Learning_Curves.pdf")
    plt.savefig(save_path, dpi=300, format="pdf", bbox_inches="tight")
    png_path = save_path.replace(".pdf", ".png")
    plt.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\n  [✓] 图表已保存: {save_path}")
    print(f"  [✓] PNG 预览:    {png_path}")


if __name__ == "__main__":
    plot_learning_curves()
