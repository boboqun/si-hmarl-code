import os
import sys
import argparse
import math
import numpy as np
import pandas as pd
import ray
import csv

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SYS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

if SYS_ROOT not in sys.path:
    sys.path.insert(0, SYS_ROOT)

# removed mock import that caused SyntaxError
from ray.tune.registry import register_env
from ray.rllib.algorithms.algorithm import Algorithm
from experiments.my_method.env_defs import RandomMapEnv, GRID_RES
from experiments.scripts.fine_grid_audit import attach_auditor   # [audit] 多尺度细网格覆盖审计

TEST_SEEDS = list(range(1001, 1031))  # 30 seeds default, overridable via --n_seeds
RESULTS_DIR = os.path.join(SYS_ROOT, "experiments", "results")
AUDIT_CSV = os.path.join(RESULTS_DIR, "fine_grid_audit_multiscale.csv")  # [audit] 各尺度细网格覆盖输出

# 从 evaluate_performance 复用环境挂载工厂
def env_creator_my_method(config):
    from experiments.my_method.HierarchicalEnvV2 import HierarchicalEnv
    return HierarchicalEnv(RandomMapEnv(grid_resolution=GRID_RES, seed=config.get("seed", 42)))

def env_creator_standard(config):
    from experiments.baselines.standard_hmarl.StandardEnv import StandardEnv
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    return ParallelPettingZooEnv(StandardEnv(RandomMapEnv(grid_resolution=GRID_RES, seed=config.get("seed", 42))))

def env_creator_flat(config):
    from experiments.baselines.mappo_flat.FlatEnv import FlatEnv
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    return ParallelPettingZooEnv(FlatEnv(RandomMapEnv(grid_resolution=GRID_RES, seed=config.get("seed", 42))))

def _find_latest_checkpoint(path):
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale_mult", type=float, required=True)
    parser.add_argument("--scale_name", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--n_seeds", type=int, default=30, help="Number of seeds")
    args = parser.parse_args()

    seeds = list(range(1001, 1001 + args.n_seeds))

    # [audit] 初始化细网格审计输出（含表头，仅当文件不存在时写表头）
    if not os.path.exists(AUDIT_CSV):
        with open(AUDIT_CSV, 'w', newline='', encoding='utf-8') as _af:
            csv.writer(_af).writerow(["method", "scale", "seed", "coarse_cov", "fine_cov", "success"])

    # Register under ALL possible env names that checkpoints might store
    register_env("hierarchical_coverage_env", env_creator_my_method)
    register_env("hierarchical_coverage_v2_env", env_creator_my_method)
    register_env("standard_coverage_env", env_creator_standard)
    register_env("standard_coverage_fair_env", env_creator_standard)
    register_env("flat_coverage_env", env_creator_flat)
    register_env("flat_coverage_fair_env", env_creator_flat)

    from ray.rllib.models import ModelCatalog
    from experiments.my_method.enjoy_v2 import RLlibUGVModel_v2, RLlibUAVModel_v2
    from experiments.baselines.standard_hmarl.train_standard_fair import RLlibUGVModel as StandardRLlibUGVModel, RLlibUAVModel as StandardRLlibUAVModel
    from experiments.baselines.mappo_flat.flat_models import RLlibFlatUAVModel, RLlibFlatUGVModel
    
    ModelCatalog.register_custom_model("RLlibUGVModel_v2", RLlibUGVModel_v2)
    ModelCatalog.register_custom_model("RLlibUAVModel_v2", RLlibUAVModel_v2)
    ModelCatalog.register_custom_model("StandardRLlibUGVModel", StandardRLlibUGVModel)
    ModelCatalog.register_custom_model("StandardRLlibUAVModel", StandardRLlibUAVModel)
    ModelCatalog.register_custom_model("RLlibFlatUAVModel", RLlibFlatUAVModel)
    ModelCatalog.register_custom_model("RLlibFlatUGVModel", RLlibFlatUGVModel)

    ray.init(ignore_reinit_error=True, log_to_driver=False)

    methods = [
        ("Our_SA_HMARL", "Our SA-HMARL"),
        ("Standard_H_MARL", "Standard H-MARL"),
        ("MAPPO_Flat", "MAPPO Flat"),
        ("Heuristic_MACPP", "Heuristic MACPP"),
        ("AG_CVG", "AG-CVG"),
        ("Eker_DP", "Eker DP"),
        ("Porcelli_CACPP", "Porcelli CACPP"),
    ]

    completed_methods = set()
    if os.path.exists(args.csv_path):
        df_csv = pd.read_csv(args.csv_path)
        for _, row in df_csv.iterrows():
            completed_methods.add((row['Algorithm'], row['Scale']))

    for short_method, disp_label in methods:
        if (disp_label, args.scale_name) in completed_methods:
            print(f"\n  [{disp_label}] 在 {args.scale_name} 尺度已记录完成，跳过。")
            continue
            
        print(f"\n  [{disp_label}] 评估中 (尺度: {args.scale_name}) ...")
        algo = None
        is_mock = False

        if short_method in ("Heuristic_MACPP", "AG_CVG", "Eker_DP", "Porcelli_CACPP"):
            try:
                from experiments.my_method.HierarchicalEnvV2 import HierarchicalEnv
                if short_method == "Heuristic_MACPP":
                    from experiments.baselines.heuristic_macpp.heuristic_policy import HeuristicController
                elif short_method == "AG_CVG":
                    from experiments.baselines.ag_cvg.ag_cvg_policy import AGCVGController
                elif short_method == "Eker_DP":
                    from experiments.baselines.eker_dp.eker_dp_policy import EkerDPController
                elif short_method == "Porcelli_CACPP":
                    from experiments.baselines.porcelli_cacpp.porcelli_policy import PorcelliCACPPController
            except ImportError:
                is_mock = True
        else:
            if short_method == "Our_SA_HMARL":
                latest_ckpt = os.path.join(RESULTS_DIR, "sa_hmarl_v2", "checkpoints", "latest")
            elif short_method == "Standard_H_MARL":
                latest_ckpt = os.path.join(
                    SYS_ROOT, "experiments", "baselines", "standard_hmarl",
                    "checkpoints_standard_fair", "latest"
                )
            elif short_method == "MAPPO_Flat":
                latest_ckpt = os.path.join(
                    SYS_ROOT, "experiments", "baselines", "mappo_flat",
                    "checkpoints_flat_fair", "milestones", "iter_01000"
                )
            else:
                short_id = short_method.lower().replace("standard_h_marl", "standard_hmarl")
                ckpt_dir = os.path.join(RESULTS_DIR, short_id, "checkpoints")
                latest_ckpt = _find_latest_checkpoint(ckpt_dir)
            
            if latest_ckpt:
                try:
                    # Switch generic model names to match what this checkpoint expects
                    if short_method == "Our_SA_HMARL":
                        ModelCatalog.register_custom_model("RLlibUGVModel", RLlibUGVModel_v2)
                        ModelCatalog.register_custom_model("RLlibUAVModel", RLlibUAVModel_v2)
                    elif short_method == "Standard_H_MARL":
                        ModelCatalog.register_custom_model("RLlibUGVModel", StandardRLlibUGVModel)
                        ModelCatalog.register_custom_model("RLlibUAVModel", StandardRLlibUAVModel)
                    algo = Algorithm.from_checkpoint(latest_ckpt)
                except Exception as e:
                    print(f"    [!] 模型还原失败 {e}")
                    is_mock = True
            else:
                is_mock = True
                
        seeds_success = 0
        seeds_makespan = []
        
        # 增加容忍上限，Heuristic MACPP 在 2x (4000x4000) 下可能需要非常庞大的物理仿真步数才能全局跑完
        MAX_STEPS_THRESHOLD = max(3000, int(15000 * (args.scale_mult ** 2.2)))
        
        for seed in seeds:
            if is_mock:
                print(f"    Seed {seed}: 跳过 (权重不存在)")
                continue

            step_count = 0
            success = False
            auditor = None
            cov_env = None
            
            if short_method in ("Heuristic_MACPP", "AG_CVG", "Eker_DP", "Porcelli_CACPP"):
                env = HierarchicalEnv(RandomMapEnv(grid_resolution=GRID_RES, seed=seed), use_resume_scan=False)
                cov_env = env
                auditor = attach_auditor(env, 50.0)   # [audit] 监听 env 扫幅，重算 50m 细网格覆盖
                if short_method == "Heuristic_MACPP":
                    controller = HeuristicController(env)
                elif short_method == "AG_CVG":
                    controller = AGCVGController(env)
                elif short_method == "Eker_DP":
                    controller = EkerDPController(env)
                elif short_method == "Porcelli_CACPP":
                    controller = PorcelliCACPPController(env)
                obs, _ = env.reset(seed=seed)
                
                while step_count < MAX_STEPS_THRESHOLD:
                    actions = controller.get_actions(obs)
                    obs, _, terms, truncs, _ = env.step(actions)
                    step_count += 1
                    
                    if any(terms.values()):
                        if env._compute_coverage_ratio() >= 0.98:
                            success = True
                        break
                    if any(truncs.values()):
                        break
            else:
                from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
                if short_method == "Our_SA_HMARL":
                    from experiments.my_method.HierarchicalEnvV2 import HierarchicalEnv
                    raw_env = HierarchicalEnv(RandomMapEnv(grid_resolution=GRID_RES, seed=seed))
                    env = ParallelPettingZooEnv(raw_env)
                elif short_method == "Standard_H_MARL":
                    from experiments.baselines.standard_hmarl.StandardEnv import StandardEnv
                    raw_env = StandardEnv(RandomMapEnv(grid_resolution=GRID_RES, seed=seed))
                    env = ParallelPettingZooEnv(raw_env)
                elif short_method == "MAPPO_Flat":
                    from experiments.baselines.mappo_flat.FlatEnv import FlatEnv
                    raw_env = FlatEnv(RandomMapEnv(grid_resolution=GRID_RES, seed=seed))
                    env = ParallelPettingZooEnv(raw_env)
                    
                obs, _ = env.reset(seed=seed)
                
                cov_env = raw_env
                auditor = attach_auditor(raw_env, 50.0) if (hasattr(raw_env, 'uavs') and hasattr(raw_env, 'map_env') and hasattr(raw_env, '_fill_uav_swath')) else None
                while step_count < MAX_STEPS_THRESHOLD:
                    actions = {}
                    for agent_id, agent_obs in obs.items():
                        policy_id = "ugv_policy" if "ugv" in agent_id else "uav_policy"
                        actions[agent_id] = algo.compute_single_action(agent_obs, policy_id=policy_id, explore=False)
                        
                    obs, _, terms, truncs, _ = env.step(actions)
                    step_count += 1
                    
                    if any(terms.values()):
                        cov = raw_env._compute_coverage_ratio()
                        if cov >= 0.98:
                            success = True
                        print(f"      -> Terminated! Cov: {cov:.3f}")
                        break
                    if any(truncs.values()):
                        cov = raw_env._compute_coverage_ratio()
                        print(f"      -> Truncated! (Battery/Time). Cov: {cov:.3f}")
                        break
                        
            print(f"    Seed {seed}: 成功={success}, 步长={step_count}")
            # [audit] 记录本 seed 的粗/细网格覆盖（coarse=env 20x20；fine=固定 50m 物理网格）
            _fine = auditor.coverage_ratio() if auditor is not None else float('nan')
            _coarse = cov_env._compute_coverage_ratio() if cov_env is not None else float('nan')
            with open(AUDIT_CSV, 'a', newline='', encoding='utf-8') as _af:
                csv.writer(_af).writerow([disp_label, args.scale_name, seed, _coarse, _fine, success])
            if success:
                seeds_success += 1
                seeds_makespan.append(step_count)

        # 统计本尺度的结果
        sr_percent = (seeds_success / len(seeds)) * 100 if not is_mock else 0.0
        # 这里记录绝对均值，外部 pandas 会基于 1x (5x5) 进行归一化 multiplier
        avg_makespan = np.mean(seeds_makespan) if len(seeds_makespan) > 0 else float('nan')
        
        with open(args.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([disp_label, args.scale_name, sr_percent, avg_makespan])

    ray.shutdown()

if __name__ == "__main__":
    main()
