"""
M1/E3 — Post-hoc fine-grid coverage audit.
==========================================
回应审稿意见：终止判据按固定 20x20 粗网格判 100%，在大尺度下每个 cell 物理上很大
（8km 时 400m），可能掩盖“是否真的物理覆盖到位”。本审计**独立地**用固定物理 cell
（默认 50m，= 扫幅宽度）从 UAV 实际扫描轨迹重算覆盖率，复刻 env 的 _fill_uav_swath
扫幅几何，给出 “fine-grid coverage = 100% / 99.x%” 以正面回应该质疑。

要点：
  - FineGridAuditor 与 env 无关：从 env.map_env 读地图范围，自己维护固定 50m 物理网格；
  - 仅在 UAV “扫描态”（is_busy 且非 returning/swapping）时按 25m 半宽铺扫幅，
    与 env 内 _fill_uav_swath 的标记时机一致；
  - 同时打印 env 自带的粗 20x20 覆盖率，与细网格覆盖率对照。

诚实声明：审计结果可能不是 100%（若末块细扫未跑完即按粗网格终止）。**如实报告**，
即便是 99.x% 也照写——这正是审计的意义。

用法：
  # (A) 多尺度审计（推荐）：已集成进 run_scalability_evaluator_subprocess.py。
  #     用现有多尺度管线即可，2km/4km/8km 全部方法的细网格覆盖会自动写到
  #     experiments/results/fine_grid_audit_multiscale.csv（含 coarse_cov 对照）：
  #         python experiments/scripts/generate_scalability_data.py --scales 1.0 2.0 4.0
  #
  # (B) 单尺度快速 sanity check（规划基线 @ 2km，本文件自带）：
  #         python fine_grid_audit.py --methods Porcelli Heuristic --seeds 1001 1002
  #
  # 任意 eval 循环手动集成只需 1 行（monkey-patch env 扫幅，自动镜像，无需逐 step 调用）：
  #     auditor = attach_auditor(raw_env, fine_res=50.0)   # episode 末取 auditor.coverage_ratio()
"""
import os, sys, math, argparse
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.my_method.env_defs import RandomMapEnv, GRID_RES
from experiments.my_method.HierarchicalEnvV2 import HierarchicalEnv

UAV_SWATH = 50.0


def _point_in_convex_polygon(px, py, verts):
    """与 HierarchicalEnvV2._point_in_convex_polygon 一致：叉积同号即在凸多边形内。"""
    sign = 0
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if cross != 0:
            s = 1 if cross > 0 else -1
            if sign == 0:
                sign = s
            elif s != sign:
                return False
    return True


class FineGridAuditor:
    """独立的固定物理分辨率覆盖审计器（复刻 env 扫幅标记几何）。"""
    def __init__(self, env, fine_res=50.0, swath=UAV_SWATH):
        m = env.map_env
        self.min_x, self.min_y = m.min_x, m.min_y
        self.max_x, self.max_y = m.max_x, m.max_y
        self.res = float(fine_res)
        self.half_width = swath / 2.0
        self.cols = max(1, int(math.ceil((self.max_x - self.min_x) / self.res)))
        self.rows = max(1, int(math.ceil((self.max_y - self.min_y) / self.res)))
        self.grid = np.zeros((self.rows, self.cols), dtype=np.uint8)
        # Lazy init: _prev is set on first update() call, so auditor works before env.reset()
        self._prev = None
        self._env_ref = env

    def _fill_swath(self, x0, y0, x1, y1, half_width=None):
        """向量化标记：cell 中心落入以 (x0,y0)->(x1,y1) 为中轴、半宽 hw 的矩形扫幅内即置 1。
        与 env._fill_uav_swath 的矩形几何等价（沿轨 u∈[0,L] 且横向 |w|≤hw），但用 numpy
        批量判定 bbox 内所有 cell，避免逐 cell 的 Python 循环——大尺度(8km/50m=160x160)下显著提速。"""
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length < 1e-9:
            return
        hw = self.half_width if half_width is None else half_width
        c0 = max(0, int((min(x0, x1) - hw - self.min_x) // self.res))
        c1 = min(self.cols - 1, int((max(x0, x1) + hw - self.min_x) // self.res))
        r0 = max(0, int((min(y0, y1) - hw - self.min_y) // self.res))
        r1 = min(self.rows - 1, int((max(y0, y1) + hw - self.min_y) // self.res))
        if c1 < c0 or r1 < r0:
            return
        cx = self.min_x + (np.arange(c0, c1 + 1) + 0.5) * self.res
        cy = self.min_y + (np.arange(r0, r1 + 1) + 0.5) * self.res
        PX, PY = np.meshgrid(cx, cy)                       # (nrows, ncols)
        u = ((PX - x0) * dx + (PY - y0) * dy) / length     # 沿轨距离 ∈ [0, length]
        w = ((PX - x0) * dy - (PY - y0) * dx) / length     # 横向有符号距离
        mask = (u >= 0.0) & (u <= length) & (np.abs(w) <= hw)
        self.grid[r0:r1 + 1, c0:c1 + 1][mask] = 1

    def update(self, env):
        """每步调用：仅对处于扫描态的 UAV 铺扫幅（与 env 标记时机一致）。"""
        # Lazy init _prev on first call (after env.reset() has populated uavs)
        if self._prev is None:
            self._prev = {a: (env.uavs[a]['x'], env.uavs[a]['y']) for a in ['uav_0', 'uav_1']}
            return
        for a in ['uav_0', 'uav_1']:
            u = env.uavs[a]
            x0, y0 = self._prev[a]
            x1, y1 = u['x'], u['y']
            scanning = u.get('is_busy') and not u.get('is_returning') and not u.get('is_swapping')
            if scanning:
                self._fill_swath(x0, y0, x1, y1)
            self._prev[a] = (x1, y1)

    def coverage_ratio(self):
        return float(self.grid.sum()) / float(self.grid.size)

    def mark(self, x0, y0, x1, y1, half_width):
        """按 env 实际传入的 half_width 标记细网格（供 attach_auditor 的 monkey-patch 调用，精确镜像 env）。"""
        self._fill_swath(x0, y0, x1, y1, half_width)


def attach_auditor(env, fine_res=50.0):
    """把 FineGridAuditor 挂到 env 上：monkey-patch env._fill_uav_swath，使其在 env 自身标记
    粗网格的同时，按相同参数标记一份固定物理分辨率(fine_res)的细网格。返回 auditor。
    只需在 episode 开始（env 构造后）调用一次，无需逐 step 调用；多尺度下自动按 env 地图范围建网格。"""
    auditor = FineGridAuditor(env, fine_res=fine_res)
    _orig = env._fill_uav_swath

    def _patched(x0, y0, x1, y1, half_width):
        _orig(x0, y0, x1, y1, half_width)
        auditor.mark(x0, y0, x1, y1, half_width)

    env._fill_uav_swath = _patched
    return auditor


# ── 自带可运行示例：规划基线 @ 2km ───────────────────────────────────────
def _controllers():
    from experiments.baselines.porcelli_cacpp.porcelli_policy import PorcelliCACPPController
    from experiments.baselines.ag_cvg.ag_cvg_policy import AGCVGController
    from experiments.baselines.eker_dp.eker_dp_policy import EkerDPController
    from experiments.baselines.heuristic_macpp.heuristic_policy import HeuristicController
    from experiments.baselines.predictive_mpc.predictive_policy import PredictiveMPCController
    return {"Porcelli": PorcelliCACPPController, "AG-CVG": AGCVGController,
            "Eker": EkerDPController, "Heuristic": HeuristicController,
            "MPC-Predictive": PredictiveMPCController}

def load_rllib_algo():
    import ray
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.models import ModelCatalog
    from experiments.my_method.enjoy_v2 import RLlibUGVModel_v2, RLlibUAVModel_v2
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    ModelCatalog.register_custom_model("RLlibUGVModel_v2", RLlibUGVModel_v2)
    ModelCatalog.register_custom_model("RLlibUAVModel_v2", RLlibUAVModel_v2)
    ModelCatalog.register_custom_model("RLlibUGVModel", RLlibUGVModel_v2)
    ModelCatalog.register_custom_model("RLlibUAVModel", RLlibUAVModel_v2)
    ckpt = os.path.join(PROJECT_ROOT, "experiments", "results", "sa_hmarl_v2", "checkpoints", "latest")
    return Algorithm.from_checkpoint(ckpt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["Porcelli"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1001])
    ap.add_argument("--fine_res", type=float, default=50.0)
    ap.add_argument("--max_steps", type=int, default=30000)
    args = ap.parse_args()
    ctrls = _controllers()
    algo = None
    if "SA-HMARL" in args.methods:
        algo = load_rllib_algo()

    print(f"{'method':10s} {'seed':>5s} {'coarse_cov':>11s} {'fine_cov':>9s} {'fine_res':>8s} {'makespan':>9s}")
    results = []
    import pandas as pd
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    for method in args.methods:
        for seed in args.seeds:
            # SA-HMARL 用其正式配置 use_resume_scan=True；规划基线沿用 False（与主实验一致）
            raw_env = HierarchicalEnv(RandomMapEnv(grid_resolution=GRID_RES, seed=seed),
                                      use_resume_scan=(method == "SA-HMARL"))
            auditor = FineGridAuditor(raw_env, fine_res=args.fine_res, swath=UAV_SWATH)
            step = 0
            
            if method == "SA-HMARL":
                env = ParallelPettingZooEnv(raw_env)
                obs, _ = env.reset(seed=seed)
                while True:
                    actions = {aid: algo.compute_single_action(
                                   o, policy_id="ugv_policy" if "ugv" in aid else "uav_policy",
                                   explore=False)
                               for aid, o in obs.items()}
                    obs, _, terms, truncs, _ = env.step(actions); step += 1
                    auditor.update(raw_env)
                    if any(terms.values()) or any(truncs.values()) or step >= args.max_steps:
                        break
            else:
                obs, _ = raw_env.reset(seed=seed)
                ctrl = ctrls[method](raw_env)
                while True:
                    obs, _, terms, truncs, _ = raw_env.step(ctrl.get_actions(obs)); step += 1
                    auditor.update(raw_env)
                    if any(terms.values()) or any(truncs.values()) or step >= args.max_steps:
                        break
                        
            coarse = raw_env._compute_coverage_ratio()
            fine = auditor.coverage_ratio()
            print(f"{method:10s} {seed:>5d} {coarse:>11.4f} {fine:>9.4f} "
                  f"{args.fine_res:>7.0f}m {step:>9d}")
            results.append({
                "method": method,
                "seed": seed,
                "coarse_cov": coarse,
                "fine_cov": fine,
                "fine_res_m": args.fine_res,
                "makespan": step
            })
            
    df = pd.DataFrame(results)
    out_path = os.path.join(PROJECT_ROOT, "experiments", "results", "fine_grid_audit_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved results to: {out_path}")
    print("如实报告 fine_cov。若 fine_cov≈1.0 → 物理覆盖确实完整，可在 scalability 段写")
    print("“a post-hoc audit at fixed 50 m physical cells confirms ~100% realized coverage”。")
    print("若 fine_cov 明显<1.0 → 说明粗网格终止确实高估了大尺度覆盖，需如实写明并讨论。")


if __name__ == "__main__":
    main()
