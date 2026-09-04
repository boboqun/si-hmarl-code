"""
evaluate_performance.py
=======================
核心评价指标抽取与顶级学术级图表渲染模块 (符合期刊出版规范)。
支持根据预存 Checkpoint 载入各算法最优权重，并生成具有动态防重叠标注、严谨双侧误差棒与灰度兼容纹理的 PDF 矢量图表。
"""

import os
import sys
import math
import numpy as np
import pandas as pd
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.algorithm import Algorithm

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SYS_ROOT = os.path.abspath(os.path.join(PROJECT_DIR, "..", ".."))
if SYS_ROOT not in sys.path:
    sys.path.insert(0, SYS_ROOT)

from experiments.my_method.env_defs import RandomMapEnv, GRID_RES

# ==============================================================================
#  学术图表全局高阶配置 (期刊出版标准)
# ==============================================================================
def set_academic_style():
    """配置顶级期刊所需的干净、冷峻且支持矢量打印的绘图参数"""
    sns.set_theme(style="ticks")
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'mathtext.fontset': 'stix',     # 数学字体对齐 Times 风格
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'axes.linewidth': 1.2,          # 边框加粗
        'pdf.fonttype': 42,             # 保证 PDF 字体嵌入且可编辑 (顶刊强制要求)
        'ps.fonttype': 42,
        'figure.dpi': 300,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',        # 自动裁剪多余白边
    })

TEST_SEEDS = list(range(1001, 1011))

# UAV 扫幅宽度（与 HierarchicalEnvV2.py 保持一致），用于计算理论最小覆盖里程
# 理论最小扫描距离 = 已覆盖面积(m²) / 扫幅宽度(m)
UAV_SCAN_WIDTH = 50.0   # meters

RESULTS_DIR = os.path.join(SYS_ROOT, "experiments", "results")
PLOTS_DIR = os.path.join(SYS_ROOT, "paper_figures", "Ch5_Performance")
os.makedirs(PLOTS_DIR, exist_ok=True)

# 统一的方法命名池 (使用 \n 换行代替原本的 rotation=15 倾斜，视觉更端庄扎实)
METHODS = {
    "Our_SA_HMARL": "Our\nST-HMARL",
    "Standard_H_MARL": "Standard\nH-MARL",
    "AG_CVG": "AG-CVG\n(Karapetyan 2024)",
    "Eker_DP": "Eker DP\n(Eker 2025)",
    "MAPPO_Flat": "MAPPO\nFlat",
    "Heuristic_MACPP": "Heuristic\nMACPP"
}

# 学术高级色板 (莫兰迪色系，降低饱和度，对色弱友好，显高级)
PALETTE = {
    "Our\nST-HMARL": "#2878B5",                # 稳重学术蓝 (主推方法)
    "Standard\nH-MARL": "#E68A5C",             # 哑光莫兰迪橙
    "AG-CVG\n(Karapetyan 2024)": "#9467BD",     # 学术紫 (外部已发表基线)
    "Eker DP\n(Eker 2025)": "#8C564B",          # 沉稳棕 (外部已发表基线)
    "MAPPO\nFlat": "#C25E5E",                   # 铁锈红
    "Heuristic\nMACPP": "#659266"               # 森系灰绿
}

# 黑白打印兼容的独立纹理映射 (Hatches)
HATCHES = ['//', '\\\\', '||', '--', 'xx', '..']

# --------------------------------------------------------------------------
# 环境生成器代理
# --------------------------------------------------------------------------
def env_creator_my_method(config):
    from experiments.my_method.HierarchicalEnvV2 import HierarchicalEnv as HierarchicalEnvV2
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    return ParallelPettingZooEnv(HierarchicalEnvV2(RandomMapEnv(grid_resolution=GRID_RES, seed=config.get("seed", 42))))

def env_creator_standard(config):
    from experiments.baselines.standard_hmarl.StandardEnv import StandardEnv
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    return ParallelPettingZooEnv(StandardEnv(RandomMapEnv(grid_resolution=GRID_RES, seed=config.get("seed", 42))))

def env_creator_flat(config):
    from experiments.baselines.mappo_flat.FlatEnv import FlatEnv
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    return ParallelPettingZooEnv(FlatEnv(RandomMapEnv(grid_resolution=GRID_RES, seed=config.get("seed", 42))))

def env_creator_standard_fair(config):
    """Standard H-MARL 公平版环境（与 standard_coverage_fair_env checkpoint 匹配）"""
    from experiments.baselines.standard_hmarl.StandardEnv import StandardEnv
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    return ParallelPettingZooEnv(StandardEnv(RandomMapEnv(grid_resolution=GRID_RES, seed=config.get("seed", 42))))

def env_creator_flat_fair(config):
    """MAPPO Flat 公平版环境（与 flat_coverage_fair_env checkpoint 匹配）"""
    from experiments.baselines.mappo_flat.FlatEnv import FlatEnv
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    return ParallelPettingZooEnv(FlatEnv(RandomMapEnv(grid_resolution=GRID_RES, seed=config.get("seed", 42))))


class PerformanceEvaluator:
    def __init__(self):
        self.ray_initialized = False

    def init_ray_if_needed(self):
        if not ray.is_initialized():
            # ---------------------------------------------------------------
            # 关键修复 (第二层): Ray 的 RolloutWorker 是独立子进程，不继承
            # 主进程运行时修改的 sys.path。必须在 ray.init() 之前通过
            # PYTHONPATH 环境变量注入，才能让所有 worker 进程也找到
            # 裸模块名 "flat_models"（checkpoint pickle 时固化的模块路径）。
            # ---------------------------------------------------------------
            _flat_dir = os.path.abspath(
                os.path.join(SYS_ROOT, "experiments", "baselines", "mappo_flat")
            )
            # 1. 注入 PYTHONPATH — ray.init() 后衍生的所有 worker 子进程将继承此变量
            _existing_pp = os.environ.get("PYTHONPATH", "")
            if _flat_dir not in _existing_pp:
                os.environ["PYTHONPATH"] = _flat_dir + os.pathsep + _existing_pp
            # 2. 同步更新当前主进程的 sys.path
            if _flat_dir not in sys.path:
                sys.path.insert(0, _flat_dir)

            ray.init(ignore_reinit_error=True, log_to_driver=False)
            from ray.tune.registry import register_env
            from ray.rllib.models import ModelCatalog

            from experiments.my_method.hierarchical_train_v2 import RLlibUGVModel as V2UGVModel, RLlibUAVModel as V2UAVModel
            from experiments.baselines.standard_hmarl.train_standard_fair import RLlibUGVModel as StandardRLlibUGVModel, RLlibUAVModel as StandardRLlibUAVModel
            from experiments.baselines.mappo_flat.flat_models import RLlibFlatUAVModel, RLlibFlatUGVModel

            # 注册可唯一发现的模型名称。注意："RLlibUGVModel" / "RLlibUAVModel" 不在此注册，
            # 将在加载各算法 checkpoint 前按需分别覆盖注册，避免 V2 编码覆盖基线编码。
            ModelCatalog.register_custom_model("StandardRLlibUGVModel", StandardRLlibUGVModel)
            ModelCatalog.register_custom_model("StandardRLlibUAVModel", StandardRLlibUAVModel)
            ModelCatalog.register_custom_model("RLlibFlatUAVModel", RLlibFlatUAVModel)
            ModelCatalog.register_custom_model("RLlibFlatUGVModel", RLlibFlatUGVModel)
            # 保存中间引用以便后续按算法切换注册
            self._model_classes = {
                "v2":      (V2UGVModel,           V2UAVModel),
                "standard":(StandardRLlibUGVModel, StandardRLlibUAVModel),
                "flat":    (RLlibFlatUGVModel,     RLlibFlatUAVModel),
            }
            
            register_env("hierarchical_coverage_env", env_creator_my_method)
            register_env("standard_coverage_env", env_creator_standard)
            register_env("flat_coverage_env", env_creator_flat)
            # 公平版环境名称（与 fair 版 checkpoint 中记录的 env 名称匹配）
            register_env("standard_coverage_fair_env", env_creator_standard_fair)
            register_env("flat_coverage_fair_env", env_creator_flat_fair)
            self.ray_initialized = True

    def _find_latest_checkpoint(self, path):
        if not os.path.exists(path):
            return None
        # 如果该目录本身就是一个合法的 checkpoint 目录
        if os.path.exists(os.path.join(path, "rllib_checkpoint.json")):
            return path
            
        ckpts = [os.path.join(path, d) for d in os.listdir(path) if d.startswith("checkpoint_")]
        if not ckpts:
            return None
        ckpts.sort(key=os.path.getmtime)
        return ckpts[-1]

    def run_eval_loop(self):
        records = []
        for method_id, method_label in METHODS.items():
            print(f"\n[{method_label.replace(chr(10), ' ')}] 准备评估核心循环...")
            algo = None
            
            if method_id in ("Heuristic_MACPP", "AG_CVG", "Eker_DP"):
                try:
                    from experiments.my_method.HierarchicalEnvV2 import HierarchicalEnv
                    if method_id == "Heuristic_MACPP":
                        from experiments.baselines.heuristic_macpp.heuristic_policy import HeuristicController
                    elif method_id == "AG_CVG":
                        from experiments.baselines.ag_cvg.ag_cvg_policy import AGCVGController
                    elif method_id == "Eker_DP":
                        from experiments.baselines.eker_dp.eker_dp_policy import EkerDPController
                except ImportError as e:
                    raise RuntimeError(
                        f"[FATAL] {method_id} 模块导入失败，已中止评估。\n"
                        f"  错误: {e}\n"
                    ) from e
            else:
                self.init_ray_if_needed()
                # SA-HMARL V2 使用独立的 checkpoint 目录；基线模型路径不变
                if method_id == "Our_SA_HMARL":
                    ckpt_dir = os.path.join(RESULTS_DIR, "sa_hmarl_v2", "checkpoints", "latest")
                elif method_id == "Standard_H_MARL":
                    # 公平版：超参对齐 SA-HMARL 的 6000 轮训练
                    ckpt_dir = os.path.join(
                        SYS_ROOT, "experiments", "baselines", "standard_hmarl",
                        "checkpoints_standard_fair", "latest"
                    )
                elif method_id == "MAPPO_Flat":
                    # 公平版 iter_01000 里程碑（iter_01500+ 已出现 NaN 带毒权重）
                    ckpt_dir = os.path.join(
                        SYS_ROOT, "experiments", "baselines", "mappo_flat",
                        "checkpoints_flat_fair", "milestones", "iter_01000"
                    )
                else:
                    ckpt_dir = os.path.join(RESULTS_DIR, method_id.lower(), "checkpoints")
                latest_ckpt = self._find_latest_checkpoint(ckpt_dir)
                
                if latest_ckpt:
                    print(f"  └─ 加载最优权重: {latest_ckpt}")
                    from ray.rllib.models import ModelCatalog as _MC
                    if method_id == "Our_SA_HMARL":
                        _ugv, _uav = self._model_classes["v2"]
                        _MC.register_custom_model("RLlibUGVModel", _ugv)
                        _MC.register_custom_model("RLlibUAVModel", _uav)
                    elif method_id == "Standard_H_MARL":
                        _ugv, _uav = self._model_classes["standard"]
                        _MC.register_custom_model("RLlibUGVModel", _ugv)
                        _MC.register_custom_model("RLlibUAVModel", _uav)
                    # MAPPO_Flat 使用 RLlibFlatUGVModel/RLlibFlatUAVModel，已在 init_ray 中注册，无需覆盖
                    try:
                        algo = Algorithm.from_checkpoint(latest_ckpt)
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        raise RuntimeError(
                            f"[FATAL] {method_id} checkpoint 加载失败，已中止评估。\n"
                            f"  checkpoint: {latest_ckpt}\n"
                            f"  错误: {e}\n"
                            f"  请确认 checkpoint 完整性后重新运行。"
                        ) from e
                else:
                    raise RuntimeError(
                        f"[FATAL] 未找到 {method_id} 的训练 checkpoint，已中止评估。\n"
                        f"  查找路径: {ckpt_dir}\n"
                        f"  请先运行训练脚本生成 checkpoint 后重试。"
                    )
            
            for seed in TEST_SEEDS:
                # 原汁原味的真实环境物理推演
                if method_id in ("Heuristic_MACPP", "AG_CVG", "Eker_DP"):
                    # use_resume_scan=False: 断点续扫是 SA-HMARL 的专属创新，
                    # 所有规则基线对整块重新规划航点，确保对照公平性。
                    env = HierarchicalEnv(RandomMapEnv(grid_resolution=GRID_RES, seed=seed),
                                          use_resume_scan=False)
                    if method_id == "Heuristic_MACPP":
                        controller = HeuristicController(env)
                    elif method_id == "AG_CVG":
                        controller = AGCVGController(env)
                    elif method_id == "Eker_DP":
                        controller = EkerDPController(env)
                    obs, _ = env.reset(seed=seed)
                    step_count, deadhead_dist, total_scan_dist = 0, 0.0, 0.0
                    prev_c = {a: (env.uavs[a]['x'], env.uavs[a]['y']) for a in ['uav_0', 'uav_1']}
                    while True:
                        actions = controller.get_actions(obs)
                        obs, _, terms, truncs, _ = env.step(actions)
                        step_count += 1
                        for a in ['uav_0', 'uav_1']:
                            u = env.uavs[a]
                            d = math.hypot(u['x'] - prev_c[a][0], u['y'] - prev_c[a][1])
                            if not u.get('is_busy') or u.get('is_returning') or u.get('is_swapping'):
                                deadhead_dist += d
                            else:  # is_busy 且非返航/换电 → 主动扫图模式
                                total_scan_dist += d
                            prev_c[a] = (u['x'], u['y'])
                        if any(terms.values()) or any(truncs.values()):
                            break
                    # 简化版冗余扫描距离 = 总扫图飞行距离 − 理论最小覆盖里程
                    _cov_area = float(np.sum(env.coverage_grid)) * GRID_RES * GRID_RES
                    redundant_scan_dist = max(0.0, total_scan_dist - _cov_area / UAV_SCAN_WIDTH)
                else:
                    if method_id == "Our_SA_HMARL":
                        from experiments.my_method.HierarchicalEnvV2 import HierarchicalEnv as HierarchicalEnvV2
                        from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
                        env = ParallelPettingZooEnv(HierarchicalEnvV2(RandomMapEnv(grid_resolution=GRID_RES, seed=seed)))
                    elif method_id == "Standard_H_MARL":
                        env = env_creator_standard({"seed": seed})
                    else:
                        env = env_creator_flat({"seed": seed})
                        
                    obs, _ = env.reset(seed=seed)
                    # 安全获取底层原始环境，因为 petting zoo 经常有多层封装
                    base_env = env
                    for _ in range(5):
                        if hasattr(base_env, "uavs"): break
                        if hasattr(base_env, "par_env"): base_env = base_env.par_env
                        elif hasattr(base_env, "env"): base_env = base_env.env
                        elif hasattr(base_env, "unwrapped") and base_env.unwrapped != base_env: base_env = base_env.unwrapped
                        else: break
                    
                    prev_c = {a: (base_env.uavs[a]['x'], base_env.uavs[a]['y']) for a in ['uav_0', 'uav_1'] if hasattr(base_env, 'uavs') and a in base_env.uavs}
                    step_count, deadhead_dist, total_scan_dist = 0, 0.0, 0.0
                    while True:
                        actions = {}
                        for agent_id, agent_obs in obs.items():
                            # MAPPO Flat 同样使用 ugv_policy / uav_policy 分开训练
                            # 不可覆盖为 "default_policy"（已废弃的旧版单策略名称）
                            policy_id = "ugv_policy" if "ugv" in agent_id else "uav_policy"
                            actions[agent_id] = algo.compute_single_action(agent_obs, policy_id=policy_id, explore=False)
                            
                        obs, _, terms, truncs, _ = env.step(actions)
                        step_count += 1
                        
                        for a in ['uav_0', 'uav_1']:
                            if not hasattr(base_env, 'uavs'): continue
                            u = base_env.uavs.get(a)
                            if (not u) or (a not in prev_c): continue
                            d = math.hypot(u['x'] - prev_c[a][0], u['y'] - prev_c[a][1])
                            if not u.get('is_busy') or u.get('is_returning') or u.get('is_swapping'):
                                deadhead_dist += d
                            else:  # is_busy 且非返航/换电 → 主动扫图模式
                                total_scan_dist += d
                            prev_c[a] = (u['x'], u['y'])
                            
                        if any(terms.values()) or any(truncs.values()):
                            break
                    # 简化版冗余扫描距离 = 总扫图飞行距离 − 理论最小覆盖里程
                    if hasattr(base_env, 'coverage_grid'):
                        _cov_area = float(np.sum(base_env.coverage_grid)) * GRID_RES * GRID_RES
                        redundant_scan_dist = max(0.0, total_scan_dist - _cov_area / UAV_SCAN_WIDTH)
                    else:
                        redundant_scan_dist = 0.0
                        
                print(f"    Seed {seed}: Makespan = {step_count}, "
                      f"Deadhead = {deadhead_dist:.1f}, "
                      f"RedundantScan = {redundant_scan_dist:.1f}, "
                      f"TotalWasted = {deadhead_dist + redundant_scan_dist:.1f}")
                records.append({
                    "Algorithm": method_label,
                    "Seed": seed,
                    "Global Makespan": step_count,
                    "Deadhead Distance (m)": deadhead_dist,
                    "Redundant Scan (m)": redundant_scan_dist,
                    "Total Wasted (m)": deadhead_dist + redundant_scan_dist,
                })
                
        df = pd.DataFrame(records)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        csv_p = os.path.join(RESULTS_DIR, "performance_metrics.csv")
        df.to_csv(csv_p, index=False)
        print(f"\n[✓] 量化评估数据已保存至: {csv_p}")
        return df

    def _format_ax(self, ax):
        """统一隐藏上右边框，开启下沉虚线网格"""
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', color='gray', alpha=0.3, zorder=0)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)

    def plot_academic_charts(self, df):
        print("\n[*] 正在渲染 SCI 学术级矢量图表 (严谨双侧误差棒版)...")
        set_academic_style()
        
        algos = list(PALETTE.keys())
        colors = list(PALETTE.values())
        
        # ==========================================
        # 1. 绝对防重叠 Bar Plot - Global Makespan
        # ==========================================
        fig1, ax1 = plt.subplots(figsize=(7, 5.5))
        
        # 使用 Pandas 聚合 + Matplotlib 手绘，实现像素级坐标控制
        grouped = df.groupby("Algorithm")["Global Makespan"].agg(['mean', 'std']).reindex(algos)
        x_pos = np.arange(len(algos))
        
        # 绘制主柱子，设置一定的透明度 (alpha=0.85) 让下方的误差棒能透视，避免突兀
        bars = ax1.bar(
            x_pos, grouped['mean'], color=colors, edgecolor='black', 
            linewidth=1.2, width=0.55, zorder=3, alpha=0.85
        )
        
        # 为每个柱子添加黑白打印友好的纹理 (SCI 顶刊标准)
        for i, bar in enumerate(bars):
            bar.set_hatch(HATCHES[i % len(HATCHES)])
            
        # [严谨学术修复] 恢复对称的双侧误差棒 (yerr=grouped['std'])
        # 加粗线宽 (elinewidth=1.5)、使用深炭灰 (#2b2b2b) 配合 zorder=4 确保误差棒锐利且不与背景杂糅
        ax1.errorbar(
            x_pos, grouped['mean'], yerr=grouped['std'], 
            fmt='none', ecolor='#2b2b2b', capsize=5, capthick=1.5, elinewidth=1.5, zorder=4
        )

        # 智能动态标签：精准计算“误差棒最高点”，让数值永远悬浮于双侧误差棒的绝对最上方
        y_max = (grouped['mean'] + grouped['std']).max()
        ax1.set_ylim(0, y_max * 1.2) # 为顶部文字留出 20% 呼吸空间
        
        for i, (bar, std) in enumerate(zip(bars, grouped['std'])):
            height = bar.get_height()
            # 动态偏置 3%，确保文字稳稳悬浮在上侧误差棒的正上方，绝对不重叠
            label_y = height + std + (y_max * 0.03) 
            ax1.text(bar.get_x() + bar.get_width()/2, label_y, 
                     f"{int(height):,}", ha='center', va='bottom', 
                     fontweight='bold', fontsize=11, color='#111111')

        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(algos, fontweight='bold')
        ax1.set_ylabel("Execution Steps (Makespan)", fontweight='bold')
        ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        
        self._format_ax(ax1)
        plt.tight_layout()
        path1 = os.path.join(PLOTS_DIR, "Fig_5_2_Makespan_Comparison.pdf")
        plt.savefig(path1, dpi=600, transparent=True)
        plt.close(fig1)
        
        # ==========================================
        # 2. 高级 Box + Stripplot - Deadhead Distance
        # ==========================================
        fig2, ax2 = plt.subplots(figsize=(7, 5.5))
        
        # 绘制高级学术箱线图 (关闭 fliers 离群点，交由散点图绘制真实分布)
        sns.boxplot(
            data=df, x="Algorithm", y="Deadhead Distance (m)", 
            order=algos, hue="Algorithm", palette=PALETTE, showmeans=True, width=0.45,
            fliersize=0, zorder=2, ax=ax2, legend=False,
            # 白色冷峻小菱形，取代暴发户审美的黄色大星星
            meanprops={"marker":"D", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":6}, 
            boxprops={'edgecolor':'black', 'linewidth':1.2, 'alpha': 0.85},
            whiskerprops={'color':'black', 'linewidth':1.2},
            capprops={'color':'black', 'linewidth':1.2},
            medianprops={'color':'#2b2b2b', 'linewidth':1.5, 'linestyle': '-'}
        )
        
        # 底层散点分布：使用半透明压暗处理，缩小点径，作为背景衬托以展现真实方差
        sns.stripplot(
            data=df, x="Algorithm", y="Deadhead Distance (m)", 
            order=algos, hue="Algorithm", palette=PALETTE, alpha=0.4, jitter=0.15, 
            size=4.5, ax=ax2, zorder=1, legend=False
        )
        
        ax2.set_ylabel("Deadhead Distance (meters)", fontweight='bold')
        ax2.set_xlabel("")
        ax2.set_xticks(range(len(algos)))
        ax2.set_xticklabels(algos, fontweight='bold')
        ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        
        self._format_ax(ax2)
        plt.tight_layout()
        path2 = os.path.join(PLOTS_DIR, "Fig_5_3_Deadhead_Comparison.pdf")
        plt.savefig(path2, dpi=600, transparent=True)
        plt.close(fig2)
        
        # ==========================================
        # 3. Box + Stripplot - Redundant Scan Distance
        # ==========================================
        if "Redundant Scan (m)" in df.columns:
            fig3, ax3 = plt.subplots(figsize=(7, 5.5))
            sns.boxplot(
                data=df, x="Algorithm", y="Redundant Scan (m)",
                order=algos, hue="Algorithm", palette=PALETTE, showmeans=True, width=0.45,
                fliersize=0, zorder=2, ax=ax3, legend=False,
                meanprops={"marker":"D", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":6},
                boxprops={'edgecolor':'black', 'linewidth':1.2, 'alpha': 0.85},
                whiskerprops={'color':'black', 'linewidth':1.2},
                capprops={'color':'black', 'linewidth':1.2},
                medianprops={'color':'#2b2b2b', 'linewidth':1.5},
            )
            sns.stripplot(
                data=df, x="Algorithm", y="Redundant Scan (m)",
                order=algos, hue="Algorithm", palette=PALETTE, alpha=0.4, jitter=0.15,
                size=4.5, ax=ax3, zorder=1, legend=False
            )
            ax3.set_ylabel("Redundant Scan Distance (meters)", fontweight='bold')
            ax3.set_xlabel("")
            ax3.set_xticks(range(len(algos)))
            ax3.set_xticklabels(algos, fontweight='bold')
            ax3.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
            self._format_ax(ax3)
            plt.tight_layout()
            path3 = os.path.join(PLOTS_DIR, "Fig_5_4_RedundantScan_Comparison.pdf")
            plt.savefig(path3, dpi=600, transparent=True)
            plt.close(fig3)

        # ==========================================
        # 4. Stacked Bar - Total Wasted Flight 构成分解
        # ==========================================
        if "Total Wasted (m)" in df.columns:
            fig4, ax4 = plt.subplots(figsize=(7, 5.5))
            dh_means = df.groupby("Algorithm")["Deadhead Distance (m)"].mean().reindex(algos)
            rs_means = df.groupby("Algorithm")["Redundant Scan (m)"].mean().reindex(algos)
            x_pos = np.arange(len(algos))
            bars_dh = ax4.bar(
                x_pos, dh_means, width=0.55, label="Deadhead (Return-to-Recharge)",
                color=colors, edgecolor='black', linewidth=1.2, alpha=0.85, hatch='//',
            )
            bars_rs = ax4.bar(
                x_pos, rs_means, width=0.55, bottom=dh_means,
                label="Redundant Scan (Re-coverage)",
                color=colors, edgecolor='black', linewidth=1.2, alpha=0.45, hatch='xx',
            )
            y_max = (dh_means + rs_means).max()
            ax4.set_ylim(0, y_max * 1.25)
            for i, (dh, rs) in enumerate(zip(dh_means, rs_means)):
                total = dh + rs
                ax4.text(x_pos[i], total + y_max * 0.02,
                         f"{int(total):,}", ha='center', va='bottom',
                         fontweight='bold', fontsize=10, color='#111111')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(algos, fontweight='bold')
            ax4.set_ylabel("Total Wasted Flight Distance (meters)", fontweight='bold')
            ax4.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
            ax4.legend(loc='upper left', framealpha=0.9, fontsize=10)
            self._format_ax(ax4)
            plt.tight_layout()
            path4 = os.path.join(PLOTS_DIR, "Fig_5_5_TotalWasted_Stacked.pdf")
            plt.savefig(path4, dpi=600, transparent=True)
            plt.close(fig4)

        print(f"[✓] 渲染完毕！顶级学术 PDF 矢量图表已保存至: {PLOTS_DIR}")


if __name__ == "__main__":
    evaluator = PerformanceEvaluator()
    df_results = evaluator.run_eval_loop()
    evaluator.plot_academic_charts(df_results)
    if evaluator.ray_initialized:
        ray.shutdown()