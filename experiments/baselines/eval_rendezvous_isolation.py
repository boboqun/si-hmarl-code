"""
M4 — Rendezvous-model isolation experiment.
============================================
目的：把"会合模型(continuous vs node)"的贡献，从"调度/学习策略"的贡献中分离出来，
直接回应审稿意见 M4（外部基线是否因被强加节点约束才输）。

关键事实（已用代码核实）：
  - HierarchicalEnvV2 的 rendezvous_mode 默认就是 'continuous'（连续中途对接：
    返航 UAV 实时追踪 UGV 的连续坐标 car(x,y)，物理距离 <= UAV_RECHARGE_DIST 即在
    路段任意位置换电，见 _step_uav）。
  - 现有外部基线（AG-CVG / Eker / Porcelli）的评测脚本 eval_new_baselines.py
    并未传 rendezvous_mode，因此它们**本来就用连续会合**。
  - 'node' 模式（UAV 飞到固定路网节点、UGV 也到达该节点附近才换电）仅被
    SA-HMARL 自身的消融(dual_node/flat_node)使用。

因此本脚本把同一个外部基线策略分别放进 continuous 和 node 两种会合模型里跑，
对比二者，即可"在相同规划/调度下"量化会合模型本身值多少。

用法：
  # 快速冒烟（1 seed，限步，仅验证可运行 + 记录换电位置）
  python eval_rendezvous_isolation.py --smoke
  # 正式评测（10 seeds，不限步）
  python eval_rendezvous_isolation.py
"""
import sys, os, math, time, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pandas as pd

from experiments.my_method.env_defs import RandomMapEnv, GRID_RES
from experiments.my_method.HierarchicalEnvV2 import HierarchicalEnv
from experiments.baselines.porcelli_cacpp.porcelli_policy import PorcelliCACPPController
from experiments.baselines.ag_cvg.ag_cvg_policy import AGCVGController
from experiments.baselines.eker_dp.eker_dp_policy import EkerDPController
from experiments.my_method.HierarchicalEnvV2 import UAV_RECHARGE_DIST as RECHARGE_DIST

UAV_SCAN_WIDTH = 50.0

METHODS = {
    "Porcelli_CACPP": ("Porcelli-CACPP", PorcelliCACPPController),
    "AG_CVG":         ("AG-CVG",         AGCVGController),
    "Eker_DP":        ("Eker DP",        EkerDPController),
}
MODES = ["continuous", "node"]


def _nearest_node_dist(env, x, y):
    best = float('inf')
    for n in env.G.nodes():
        d = math.hypot(float(env.G.nodes[n]['x']) - x, float(env.G.nodes[n]['y']) - y)
        if d < best:
            best = d
    return best


def run_episode(controller_class, mode, seed, max_steps=None):
    env = HierarchicalEnv(RandomMapEnv(grid_resolution=GRID_RES, seed=seed),
                          use_resume_scan=False, rendezvous_mode=mode)
    controller = controller_class(env)
    obs, _ = env.reset(seed=seed)

    step = 0
    deadhead = 0.0
    scan = 0.0
    prev = {a: (env.uavs[a]['x'], env.uavs[a]['y']) for a in ['uav_0', 'uav_1']}
    was_swap = {a: False for a in ['uav_0', 'uav_1']}
    swap_node_dists = []   # 每次换电触发时，UGV 距最近路网节点的距离（>RECHARGE_DIST 即"边内连续点"）

    while True:
        actions = controller.get_actions(obs)
        obs, _, terms, truncs, _ = env.step(actions)
        step += 1

        for a in ['uav_0', 'uav_1']:
            u = env.uavs[a]
            d = math.hypot(u['x'] - prev[a][0], u['y'] - prev[a][1])
            if u.get('is_returning') or u.get('is_swapping') or (not u.get('is_busy')):
                deadhead += d
            else:
                scan += d
            prev[a] = (u['x'], u['y'])
            # 记录换电触发瞬间（is_swapping 上升沿）的 UGV 位置相对最近节点的距离
            if u.get('is_swapping') and not was_swap[a]:
                swap_node_dists.append(_nearest_node_dist(env, env.car['x'], env.car['y']))
            was_swap[a] = u.get('is_swapping', False)

        if any(terms.values()) or any(truncs.values()):
            break
        if max_steps is not None and step >= max_steps:
            break

    cov = env._compute_coverage_ratio()
    success = cov >= 0.999
    cov_area = float(np.sum(env.coverage_grid)) * GRID_RES * GRID_RES
    redundant = max(0.0, scan - cov_area / UAV_SCAN_WIDTH)
    n_swaps = len(swap_node_dists)
    mid_edge = sum(1 for d in swap_node_dists if d > RECHARGE_DIST)  # 真正发生在边内连续点的换电次数
    return dict(makespan=step, deadhead=deadhead, redundant=redundant, coverage=cov,
                success=success, n_swaps=n_swaps, mid_edge_swaps=mid_edge,
                mean_swap_node_dist=(float(np.mean(swap_node_dists)) if swap_node_dists else 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1 seed, capped steps, execution check")
    ap.add_argument("--methods", nargs="+", default=["Porcelli_CACPP"],
                    help="subset of: " + ", ".join(METHODS))
    args = ap.parse_args()

    seeds = [1001] if args.smoke else list(range(1001, 1011))
    max_steps = 2500 if args.smoke else None

    rows = []
    for mid in args.methods:
        label, cls = METHODS[mid]
        for mode in MODES:
            print(f"\n=== {label} | rendezvous={mode} ===")
            for seed in seeds:
                t0 = time.time()
                r = run_episode(cls, mode, seed, max_steps=max_steps)
                print(f"  seed {seed}: makespan={r['makespan']} success={r['success']} "
                      f"cov={r['coverage']:.3f} deadhead={r['deadhead']:.0f} "
                      f"swaps={r['n_swaps']} mid_edge={r['mid_edge_swaps']} "
                      f"mean_swap_node_dist={r['mean_swap_node_dist']:.1f}m ({time.time()-t0:.1f}s)")
                rows.append(dict(method=label, mode=mode, seed=seed, **r))

    df = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    print("SUMMARY (mean over seeds)")
    print("=" * 78)
    agg = df.groupby(["method", "mode"]).agg(
        makespan=("makespan", "mean"), deadhead=("deadhead", "mean"),
        success=("success", "mean"), mid_edge_swaps=("mid_edge_swaps", "mean"),
        mean_swap_node_dist=("mean_swap_node_dist", "mean")).round(1)
    print(agg.to_string())

    out = os.path.join(os.path.dirname(__file__), "rendezvous_isolation_results.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    print("\n解读：同一基线在 continuous 与 node 下的差距 = 会合模型本身的贡献；")
    print("continuous 行的 mid_edge_swaps>0 / mean_swap_node_dist>2m 证明换电确实发生在路段连续位置。")


if __name__ == "__main__":
    main()
