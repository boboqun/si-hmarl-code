"""
plot_recharge_response.py
=========================
方案1 可视化：换电响应距离分布图

核心思路：
  在每次 UAV 发出换电指令（is_returning 状态从 False → True）的帧，
  记录此刻 UGV 与该 UAV 的欧式直线距离。

预期结论：
  - SI-HMARL V2  : UGV 已提前预定位，响应距离峰值 50–300 m（短）
  - Standard H-MARL : UGV 被动尾随，响应距离峰值 500–1500 m（长）

对应出图路径: paper_figures/Fig_5_9_Recharge_Response_Dist.pdf
"""

import os
import sys
import math
from typing import Optional, List, Dict
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
try:
    from ray.rllib.policy import Policy
except ImportError:        # only needed for live rollout; re-plotting from cache does not require ray
    Policy = None

# ── sys.path 必须在一切 experiments.* import 之前设置 ────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))              # experiments/scripts
SYS_ROOT    = os.path.abspath(os.path.join(PROJECT_DIR, "..", "..")) # repo root
for _p in [
    SYS_ROOT,
    os.path.join(SYS_ROOT, "experiments"),
    os.path.join(SYS_ROOT, "experiments", "my_method"),
    os.path.join(SYS_ROOT, "experiments", "baselines"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from my_method.env_defs import RandomMapEnv, GRID_RES as _GRID_RES_DEFAULT  # noqa: E402
except Exception:        # rollout deps (networkx/gymnasium) absent -> cache-only replot still works
    RandomMapEnv = None
    _GRID_RES_DEFAULT = 100.0

# ── 路径常量 ─────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(SYS_ROOT, "experiments", "results")
OUTPUT_DIR  = os.path.join(SYS_ROOT, "paper_figures", "Ch5_Performance")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 实验参数 ─────────────────────────────────────────────────────────────────
TEST_SEEDS   = list(range(1001, 1011))
GRID_RES     = _GRID_RES_DEFAULT
MAX_STEPS    = 30_000
N_EVENTS_CAP = 500

# ── 样式 ──────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import apply_plot_style, format_ax, FONT_SIZE, LABEL_SIZE, ANNOTATION_SIZE

apply_plot_style()

ALGO_CONFIG = {
    "SI-HMARL V2": {
        "color": "#2878B5",
        "ckpt_dir": os.path.join(RESULTS_DIR, "sa_hmarl_v2", "checkpoints", "latest"),
        "env_type": "v2",
        "display_name": "SI-HMARL",
    },
    "Standard H-MARL": {
        "color": "#E68A5C",
        # 双轨策略：优先 latest/（本次 6000 iter 重训结果）
        "ckpt_dir": os.path.join(RESULTS_DIR, "standard_hmarl", "checkpoints", "latest"),
        "env_type": "standard",
        "display_name": "Standard H-MARL",
    },
}


# ── 环境 & 模型初始化工具 ─────────────────────────────────────────────────────

def _find_latest_checkpoint(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    if os.path.exists(os.path.join(path, "rllib_checkpoint.json")):
        return path
    ckpts = [os.path.join(path, d) for d in os.listdir(path)
             if d.startswith("checkpoint_")]
    if not ckpts:
        return None
    ckpts.sort(key=os.path.getmtime)
    return ckpts[-1]


def _make_env(env_type: str, seed: int):
    """构造对应算法的底层 PettingZoo 环境（不加 RLlib 包装）。"""
    if env_type == "v2":
        from my_method.HierarchicalEnvV2 import HierarchicalEnv
        return HierarchicalEnv(RandomMapEnv(grid_resolution=GRID_RES, seed=seed))
    else:
        from standard_hmarl.StandardEnv import StandardEnv
        return StandardEnv(RandomMapEnv(grid_resolution=GRID_RES, seed=seed))


def _load_policies(algo_name: str, ckpt_dir: str):
    """加载 UGV / UAV Policy，并在加载前按算法切换模型注册。"""
    from ray.rllib.models import ModelCatalog

    if algo_name == "SI-HMARL V2":
        from my_method.hierarchical_train_v2 import RLlibUGVModel, RLlibUAVModel
        ModelCatalog.register_custom_model("RLlibUGVModel", RLlibUGVModel)
        ModelCatalog.register_custom_model("RLlibUAVModel", RLlibUAVModel)
    else:
        from standard_hmarl.train_standard import (
            RLlibUGVModel as StdUGV, RLlibUAVModel as StdUAV,
        )
        ModelCatalog.register_custom_model("RLlibUGVModel", StdUGV)
        ModelCatalog.register_custom_model("RLlibUAVModel", StdUAV)

    ugv_path = os.path.join(ckpt_dir, "policies", "ugv_policy")
    uav_path = os.path.join(ckpt_dir, "policies", "uav_policy")
    ugv_policy = Policy.from_checkpoint(ugv_path)
    uav_policy = Policy.from_checkpoint(uav_path)
    return ugv_policy, uav_policy


# ── 核心采集函数 ──────────────────────────────────────────────────────────────

def collect_recharge_distances(algo_name: str, cfg: dict) -> List[float]:
    """
    对所有 seeds 跑真实推演，在每次 UAV 发出换电请求的帧记录 UGV-UAV 欧式距离。
    返回距离列表（单位：m）。
    """
    ckpt = _find_latest_checkpoint(cfg["ckpt_dir"])
    if ckpt is None:
        print(f"  [!] {algo_name}: 未找到 checkpoint，跳过")
        return []

    print(f"  [*] {algo_name} 加载 checkpoint: {ckpt}")
    ugv_policy, uav_policy = _load_policies(algo_name, ckpt)

    all_dists: List[float] = []

    for seed in TEST_SEEDS:
        env = _make_env(cfg["env_type"], seed)
        obs, _ = env.reset(seed=seed)

        # 记录上一帧 is_returning 状态，用于检测 False→True 的转变边沿
        prev_returning: Dict[str, bool] = {"uav_0": False, "uav_1": False}

        for _ in range(MAX_STEPS):
            actions = {}
            for agent_id, agent_obs in obs.items():
                policy = ugv_policy if "ugv" in agent_id else uav_policy
                act, _, _ = policy.compute_single_action(agent_obs)
                actions[agent_id] = act

            obs, _, terms, truncs, _ = env.step(actions)

            # 检测换电触发事件（is_returning: False → True）
            for uav_id in ["uav_0", "uav_1"]:
                if uav_id not in env.uavs:
                    continue
                cur_returning = env.uavs[uav_id]["is_returning"]
                if cur_returning and not prev_returning[uav_id]:
                    # 此帧刚触发换电请求，记录 UGV-UAV 欧式距离
                    ux, uy = env.uavs[uav_id]["x"], env.uavs[uav_id]["y"]
                    cx, cy = env.car["x"],           env.car["y"]
                    dist = math.hypot(ux - cx, uy - cy)
                    all_dists.append(dist)
                prev_returning[uav_id] = cur_returning

            if any(terms.values()) or any(truncs.values()):
                break

        print(f"    seed {seed}: 累计换电事件 {len(all_dists)} 次")

        if len(all_dists) >= N_EVENTS_CAP:
            break   # 已收集足够样本

    print(f"  [✓] {algo_name}: 共收集 {len(all_dists)} 个换电响应距离样本")
    return all_dists


# ── 出图 ──────────────────────────────────────────────────────────────────────

def plot_recharge_response(use_cache: bool = True):
    CACHE_DIR = os.path.join(OUTPUT_DIR, ".cache")
    os.makedirs(CACHE_DIR, exist_ok=True)

    data: Dict[str, List[float]] = {}
    need_rollout = False
    for algo_name in ALGO_CONFIG:
        cache_file = os.path.join(CACHE_DIR, f"recharge_dist_{algo_name.replace(' ', '_')}.npy")
        if use_cache and os.path.exists(cache_file):
            data[algo_name] = np.load(cache_file).tolist()
            print(f"[cache] {algo_name}: 读取缓存 {len(data[algo_name])} 条")
        else:
            need_rollout = True

    if need_rollout:
        print("[*] 正在采集换电响应距离数据（真实 rollout）...")
        import ray
        ray.init(ignore_reinit_error=True, log_to_driver=False)
        for algo_name, cfg in ALGO_CONFIG.items():
            if algo_name in data:
                continue
            print(f"\n── {algo_name} ──")
            dists = collect_recharge_distances(algo_name, cfg)
            data[algo_name] = dists
            cache_file = os.path.join(CACHE_DIR, f"recharge_dist_{algo_name.replace(' ', '_')}.npy")
            np.save(cache_file, np.array(dists))
            print(f"  [cache] 已保存至 {cache_file}")

    print("\n[*] 绘制换电响应距离分布图...")

    fig, ax = plt.subplots(figsize=(8, 5))

    for algo_name, dists in data.items():
        if not dists:
            continue
        color = ALGO_CONFIG[algo_name]["color"]
        arr   = np.array(dists)

        # ── 直方图（直接统计，无平滑，自然无负值）──────────────────────────
        # stat='density': 纵轴为概率密度（各桶面积之和=1），便于两组对比
        display = ALGO_CONFIG[algo_name].get("display_name", algo_name)
        n_bins = max(10, int(1 + 3.322 * np.log10(len(arr))))  # Sturges' rule
        ax.hist(arr, bins=n_bins, density=True,
                color=color, alpha=0.16, edgecolor='none',
                linewidth=0.0, zorder=1)

        # ── 叠加细线 KDE，仅用于视觉上平滑曲线轮廓，不作为主展示 ──────────
        sns.kdeplot(arr, ax=ax, color=color, linewidth=2.0,
                    fill=True, alpha=0.22, clip=(0, None),
                    label=f"{display}  (n={len(arr)})", zorder=2)

        # ── 中位数竖线 ──────────────────────────────────────────────────────
        med = np.median(arr)
        ax.axvline(med, color=color, linestyle="--", linewidth=1.5, alpha=0.9)

    ax.set_xlabel("UGV–UAV Distance at Recharge Request (m)", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    # (in-figure title removed; the LaTeX \caption describes the figure)
    ax.set_xlim(left=0)   # 距离不可能为负，x 轴从 0 开始
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(fontsize=12, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)
    sns.despine(ax=ax)

    # ── 右下角统计标注 ────────────────────────────────────────────────────────
    y_pos = 0.62
    for algo_name, dists in data.items():
        if not dists:
            continue
        arr  = np.array(dists)
        color = ALGO_CONFIG[algo_name]["color"]
        display = ALGO_CONFIG[algo_name].get("display_name", algo_name)
        label = (f"{display}\n"
                 f"  mean={arr.mean():.0f} m\n"
                 f"  median={np.median(arr):.0f} m\n"
                 f"  n={len(arr)}")
        ax.text(0.97, y_pos, label, transform=ax.transAxes,
                ha="right", va="top", fontsize=9.5, color=color,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor=color,
                          linewidth=0.8, pad=4))
        y_pos -= 0.30

    # quantitative takeaway: median ratio between the two methods
    _meds = [float(np.median(np.array(d))) for d in data.values() if d]
    if len(_meds) == 2 and min(_meds) > 0:
        ax.text(0.97, 0.015,
                f"$\\approx\\,${max(_meds)/min(_meds):.1f}$\\times$ closer median under SI-HMARL",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
                fontweight="bold", color="#222222",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#f7f7f7",
                          edgecolor="#bbbbbb", linewidth=0.8))

    plt.tight_layout()
    save_pdf = os.path.join(OUTPUT_DIR, "Fig_5_9_Recharge_Response_Dist.pdf")
    save_png = save_pdf.replace(".pdf", ".png")
    plt.savefig(save_pdf, dpi=600, format="pdf", bbox_inches="tight")
    plt.savefig(save_png, dpi=300, format="png", bbox_inches="tight",
                facecolor='white')
    plt.close()
    print(f"\n  [✓] 图表已保存: {save_pdf}")
    print(f"  [✓] PNG 预览:    {save_png}")


if __name__ == "__main__":
    plot_recharge_response()
