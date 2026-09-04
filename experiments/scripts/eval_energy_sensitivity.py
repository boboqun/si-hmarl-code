"""
E1 — Mode-dependent energy sensitivity (eval-only, no retraining).
==================================================================
回应审稿意见 E1：当前能量模型对扫描态/返航态用相同的每步功耗 P0。本脚本在 **评测期**
让返航态功耗变为扫描态的 ratio 倍（return-to-scan power ratio），检验各方法的
success/makespan/deadhead 排序是否在中等功耗比下保持稳定。

机制（已在 HierarchicalEnvV2 中加 return_power_ratio 钩子，默认 1.0=原恒定模型）：
  - 返航态每拍扣电 = ratio（扫描/空闲态仍 = 1.0）；
  - 安全返航阈值 dynamic_low 的预留按 ratio 放大（E_safe ∝ P_return），
    使各方法在“知道返航更贵”的前提下公平预留——隔离“效率排序”而非“是否预留够”。
这是 eval-only：用现有 checkpoint，不重训。策略是在 ratio=1.0 下训练/设计的，
因此这同时是一种对“能量模型失配”的鲁棒性检验。

覆盖范围：SA-HMARL（RLlib checkpoint）+ 规划类基线（Porcelli/AG-CVG/Eker/Heuristic，
纯 Python 控制器）。三者都跑在打了钩子的 HierarchicalEnv 上，故都受 ratio 影响。
注：Standard H-MARL 用的是单独的 StandardEnv，没有此钩子；如需纳入，须在
StandardEnv 中加同样的 return_power_ratio（drain + dynamic_low），再在本脚本加分支。

用法：
  # 冒烟（1 seed，仅 ratio=1.0 与 2.0，验证可运行）
  python eval_energy_sensitivity.py --smoke
  # 正式（10 seeds × 4 ratios，全部方法）
  python eval_energy_sensitivity.py
  # 只跑规划基线（不加载 RLlib）
  python eval_energy_sensitivity.py --methods Porcelli AG-CVG Eker Heuristic
"""
import os, sys, math, time, argparse, csv
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.my_method.env_defs import RandomMapEnv, GRID_RES
from experiments.my_method.HierarchicalEnvV2 import HierarchicalEnv

RESULTS_DIR    = os.path.join(PROJECT_ROOT, "experiments", "results")
UAV_SCAN_WIDTH = 50.0
SUCCESS_COV    = 0.999          # 与正文一致：100% 覆盖判成功（记录原始 coverage 供查近失败）

# ── 规划类基线（纯 Python 控制器，跑在 HierarchicalEnv 上）────────────────
def _planning_controllers():
    from experiments.baselines.porcelli_cacpp.porcelli_policy import PorcelliCACPPController
    from experiments.baselines.ag_cvg.ag_cvg_policy import AGCVGController
    from experiments.baselines.eker_dp.eker_dp_policy import EkerDPController
    from experiments.baselines.heuristic_macpp.heuristic_policy import HeuristicController
    return {
        "Porcelli":  PorcelliCACPPController,
        "AG-CVG":    AGCVGController,
        "Eker":      EkerDPController,
        "Heuristic": HeuristicController,
    }

# ── RLlib 方法的 checkpoint 路径（镜像 run_scalability_evaluator_subprocess.py）──
RLLIB_CKPT = {
    "SA-HMARL": os.path.join(RESULTS_DIR, "sa_hmarl_v2", "checkpoints", "latest"),
}


def _accumulate_metrics(raw_env, prev, deadhead_ref):
    """按 deadhead 约定（非作业/返航/换电时的位移计入 deadhead）累计。"""
    for a in ['uav_0', 'uav_1']:
        u = raw_env.uavs[a]
        d = math.hypot(u['x'] - prev[a][0], u['y'] - prev[a][1])
        if u.get('is_returning') or u.get('is_swapping') or (not u.get('is_busy')):
            deadhead_ref[0] += d
        prev[a] = (u['x'], u['y'])


def run_planning(controller_cls, ratio, seed, max_steps):
    env = HierarchicalEnv(RandomMapEnv(grid_resolution=GRID_RES, seed=seed),
                          use_resume_scan=False, return_power_ratio=ratio)
    ctrl = controller_cls(env)
    obs, _ = env.reset(seed=seed)
    prev = {a: (env.uavs[a]['x'], env.uavs[a]['y']) for a in ['uav_0', 'uav_1']}
    deadhead = [0.0]; step = 0; outcome = "timeout"
    while True:
        obs, _, terms, truncs, _ = env.step(ctrl.get_actions(obs)); step += 1
        _accumulate_metrics(env, prev, deadhead)
        if any(terms.values()):
            outcome = "complete" if env._compute_coverage_ratio() >= SUCCESS_COV else "crash"
            break
        if any(truncs.values()) or (max_steps and step >= max_steps):
            break
    cov = env._compute_coverage_ratio()
    return dict(makespan=step, deadhead=deadhead[0], coverage=cov,
                success=(cov >= SUCCESS_COV), outcome=outcome)


def run_rllib(algo, ratio, seed, max_steps):
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    raw_env = HierarchicalEnv(RandomMapEnv(grid_resolution=GRID_RES, seed=seed),
                              return_power_ratio=ratio)
    env = ParallelPettingZooEnv(raw_env)
    obs, _ = env.reset(seed=seed)
    prev = {a: (raw_env.uavs[a]['x'], raw_env.uavs[a]['y']) for a in ['uav_0', 'uav_1']}
    deadhead = [0.0]; step = 0; outcome = "timeout"
    while True:
        actions = {aid: algo.compute_single_action(
                       o, policy_id="ugv_policy" if "ugv" in aid else "uav_policy",
                       explore=False)
                   for aid, o in obs.items()}
        obs, _, terms, truncs, _ = env.step(actions); step += 1
        _accumulate_metrics(raw_env, prev, deadhead)
        if any(terms.values()):
            outcome = "complete" if raw_env._compute_coverage_ratio() >= SUCCESS_COV else "crash"
            break
        if any(truncs.values()) or (max_steps and step >= max_steps):
            break
    cov = raw_env._compute_coverage_ratio()
    return dict(makespan=step, deadhead=deadhead[0], coverage=cov,
                success=(cov >= SUCCESS_COV), outcome=outcome)


def load_rllib_algo(method):
    """镜像 run_scalability_evaluator_subprocess.py 的模型注册 + checkpoint 还原。"""
    import ray
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.models import ModelCatalog
    from experiments.my_method.enjoy_v2 import RLlibUGVModel_v2, RLlibUAVModel_v2
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    ModelCatalog.register_custom_model("RLlibUGVModel_v2", RLlibUGVModel_v2)
    ModelCatalog.register_custom_model("RLlibUAVModel_v2", RLlibUAVModel_v2)
    ModelCatalog.register_custom_model("RLlibUGVModel", RLlibUGVModel_v2)
    ModelCatalog.register_custom_model("RLlibUAVModel", RLlibUAVModel_v2)
    ckpt = RLLIB_CKPT[method]
    if not os.path.exists(ckpt):
        print(f"  [!] checkpoint 不存在: {ckpt} → 跳过 {method}")
        return None
    return Algorithm.from_checkpoint(ckpt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--ratios", type=float, nargs="+", default=[1.0, 1.2, 1.5, 2.0])
    ap.add_argument("--methods", nargs="+",
                    default=["SA-HMARL", "Porcelli", "AG-CVG", "Eker", "Heuristic"])
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=30000)
    args = ap.parse_args()

    seeds  = [1001] if args.smoke else list(range(1001, 1001 + args.n_seeds))
    ratios = [1.0, 2.0] if args.smoke else args.ratios
    planning = _planning_controllers()

    rows = []
    for method in args.methods:
        algo = load_rllib_algo(method) if method in RLLIB_CKPT else None
        if method in RLLIB_CKPT and algo is None:
            continue
        for ratio in ratios:
            print(f"\n=== {method} | return_power_ratio={ratio} ===")
            for seed in seeds:
                t0 = time.time()
                if method in RLLIB_CKPT:
                    r = run_rllib(algo, ratio, seed, args.max_steps)
                else:
                    r = run_planning(planning[method], ratio, seed, args.max_steps)
                print(f"  seed {seed}: {r['outcome']:9s} mk={r['makespan']:6d} "
                      f"cov={r['coverage']:.3f} dh={r['deadhead']:8.0f} ({time.time()-t0:.1f}s)")
                rows.append(dict(method=method, ratio=ratio, seed=seed, **r))

    import pandas as pd
    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, "energy_sensitivity_results.csv")
    df.to_csv(out, index=False)
    print("\n" + "=" * 78 + "\nSUMMARY (mean over seeds; makespan/deadhead over completed only)\n" + "=" * 78)
    for method in df['method'].unique():
        for ratio in sorted(df[df.method == method]['ratio'].unique()):
            sub = df[(df.method == method) & (df.ratio == ratio)]
            done = sub[sub.success]
            sr = 100.0 * len(done) / len(sub)
            mk = f"{done['makespan'].mean():.0f}" if len(done) else "-"
            dh = f"{done['deadhead'].mean():.0f}" if len(done) else "-"
            ncrash = int((sub.outcome == "crash").sum()); ntime = int((sub.outcome == "timeout").sum())
            print(f"  {method:10s} ratio={ratio:<4} SR={sr:5.0f}%  makespan={mk:>7}  "
                  f"deadhead={dh:>8}  crash={ncrash} timeout={ntime}")
    print(f"\nSaved: {out}")
    print("解读：若 SA-HMARL 在 ratio=1.2/1.5 下相对各基线的 makespan/deadhead 排序保持，")
    print("即可写“中等模式相关功耗比下排序稳定；极端比值才触及已知硬件能量边界”。")


if __name__ == "__main__":
    main()
