"""
plot_energy_trace.py
====================
单回合详细微观推演：记录两架无人机电量状态以及补能窗口 (is_swapping)。
学术级双层图表，通过底层浅灰纹理阴影无遮挡标出连续中途接驳区间。

数据来源：从真实 SI-HMARL checkpoint 推演单回合，不使用任何 Mock 数据。
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYS_ROOT    = os.path.abspath(os.path.join(PROJECT_DIR, ".."))
for p in (SYS_ROOT, PROJECT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import apply_plot_style, format_ax

OUTPUT_DIR = os.path.abspath(os.path.join(SYS_ROOT, "paper_figures", "Ch5_Energy_Trace"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

CKPT_DIR = os.path.join(SYS_ROOT, "experiments", "results", "sa_hmarl_v2", "checkpoints", "latest")


def collect_real_energy_trace(seed: int = 42, max_steps: int = 4000):
    """
    从真实 SI-HMARL checkpoint 推演单回合，收集两架 UAV 的逐步电量与换电状态。
    返回 (trace_0, trace_1)，每条是 [(t, battery, is_swapping), ...] 列表。
    """
    from experiments.my_method.env_defs import RandomMapEnv, GRID_RES
    from experiments.my_method.HierarchicalEnvV2 import HierarchicalEnv as HierarchicalEnvV2
    from ray.rllib.policy.policy import Policy
    from ray.rllib.models import ModelCatalog
    from experiments.my_method.hierarchical_train_v2 import RLlibUGVModel, RLlibUAVModel

    ModelCatalog.register_custom_model("RLlibUGVModel", RLlibUGVModel)
    ModelCatalog.register_custom_model("RLlibUAVModel", RLlibUAVModel)

    ugv_policy = Policy.from_checkpoint(os.path.join(CKPT_DIR, "policies", "ugv_policy"))
    uav_policy = Policy.from_checkpoint(os.path.join(CKPT_DIR, "policies", "uav_policy"))

    map_env = RandomMapEnv(grid_resolution=GRID_RES, seed=seed)
    env     = HierarchicalEnvV2(map_env)
    obs_dict, _ = env.reset(seed=seed)

    trace_0, trace_1 = [], []

    for t in range(max_steps):
        # 记录本步电量（在 step 之前读取，与 t 时刻对齐）
        trace_0.append((t, float(env.uavs['uav_0']['battery']),
                        bool(env.uavs['uav_0']['is_swapping'])))
        trace_1.append((t, float(env.uavs['uav_1']['battery']),
                        bool(env.uavs['uav_1']['is_swapping'])))

        actions = {}
        for agent_id, obs in obs_dict.items():
            if agent_id == 'ugv_0':
                act, _, _ = ugv_policy.compute_single_action(obs)
            else:
                act, _, _ = uav_policy.compute_single_action(obs)
            actions[agent_id] = act

        obs_dict, _, terms, truncs, _ = env.step(actions)
        if any(terms.values()) or any(truncs.values()):
            break

    return trace_0, trace_1




def plot_energy_trace():
    print("[*] 从真实 SI-HMARL checkpoint 推演单回合电量轨迹 ...")
    apply_plot_style()

    if not os.path.exists(CKPT_DIR):
        print(f"[!] 错误：未找到 checkpoint 目录 {CKPT_DIR}")
        print("    请先完成 SI-HMARL 训练后再运行此脚本。")
        return

    try:
        trace_0, trace_1 = collect_real_energy_trace(seed=42)
        print(f"  [✓] 真实推演完成，共 {len(trace_0)} 步")
    except Exception as e:
        print(f"  [!] 真实推演失败：{e}")
        print("    请确认 SI-HMARL checkpoint 存在且路径正确。")
        return

    from experiments.my_method.HierarchicalEnvV2 import UAV_FULL_BATTERY

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True,
                             gridspec_kw={'hspace': 0.15})

    traces = [("(a) UAV 0 Dynamic Energy Trace", trace_0, axes[0]),
              "(b) UAV 1 Dynamic Energy Trace", trace_1, axes[1]]
    colors = ["#2878B5", "#E68A5C"]

    for (title, tr, ax), color in zip(
        [("(a) UAV 0 Dynamic Energy Trace", trace_0, axes[0]),
         ("(b) UAV 1 Dynamic Energy Trace", trace_1, axes[1])],
        colors
    ):
        ts   = [x[0] for x in tr]
        bats = [max(0.0, x[1] / UAV_FULL_BATTERY * 100) for x in tr]
        swaps = [x[2] for x in tr]

        ax.plot(ts, bats, color=color, linewidth=2.0, zorder=3)
        ax.set_ylabel("Battery Level (%)", fontweight='bold')
        ax.set_title(title, fontweight='bold', loc='left', pad=10)
        ax.set_ylim(-5, 115)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(25))

        # 阴影标注接驳区间
        is_swapping, start_ts = False, 0
        for i, swp in enumerate(swaps):
            if swp and not is_swapping:
                is_swapping, start_ts = True, ts[i]
            elif not swp and is_swapping:
                is_swapping = False
                ax.axvspan(start_ts, ts[i], facecolor='#E0E0E0',
                           edgecolor='none', alpha=0.7, zorder=1)
                ax.axvspan(start_ts, ts[i], facecolor='none',
                           edgecolor='#808080', hatch='///', alpha=0.8,
                           linewidth=0, zorder=2)
        if is_swapping:
            ax.axvspan(start_ts, ts[-1], facecolor='#E0E0E0',
                       edgecolor='none', alpha=0.7, zorder=1)
            ax.axvspan(start_ts, ts[-1], facecolor='none',
                       edgecolor='#808080', hatch='///', alpha=0.8,
                       linewidth=0, zorder=2)

        format_ax(ax)

    axes[1].set_xlabel("Physical Time Steps", fontweight='bold')

    line_legend = [plt.Line2D([0], [0], color=c, lw=2) for c in colors]
    span_legend  = mpatches.Patch(facecolor='#E0E0E0', edgecolor='#808080',
                                  hatch='///', alpha=0.8, linewidth=0)
    fig.legend(
        line_legend + [span_legend],
        ['UAV 0 Level', 'UAV 1 Level', 'Mid-segment Docking & Recharging'],
        loc='upper center', bbox_to_anchor=(0.5, 1.05),
        ncol=3, frameon=False, fontsize=12
    )

    save_path = os.path.join(OUTPUT_DIR, "Fig_4_Energy_Trace.pdf")
    plt.savefig(save_path, transparent=True)
    plt.close()
    print(f"  [✓] 图表已保存: {save_path}")


if __name__ == "__main__":
    plot_energy_trace()
