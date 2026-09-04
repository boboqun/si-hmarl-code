"""
boustrophedon_planner.py
========================
带"出入场最优化"的矩形区域全覆盖路径规划器
（Boustrophedon / 牛耕式扫描）

设计原则：
  - 纯数学/几何实现，零框架依赖（仅 numpy + matplotlib）
  - 8 种拓扑枚举 + 通勤代价最小化
  - 不漏扫、不越界，支持横向/纵向两种主扫描方向

作者：Boustrophedon Planner v1.0
"""

import numpy as np
import math
from typing import List, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────
#  核心规划器
# ─────────────────────────────────────────────────────────────────────

class SmartBoustrophedonPlanner:
    """
    智能牛耕式全覆盖路径规划器。

    调用示例：
        planner = SmartBoustrophedonPlanner()
        waypoints = planner.generate_optimal_waypoints(
            rect_min=(20, 20), rect_max=(80, 80),
            sweep_width=10.0,
            start_reference=(0, 0),
            end_reference=(100, 100),
        )
    """

    def generate_optimal_waypoints(
        self,
        rect_min: Tuple[float, float],
        rect_max: Tuple[float, float],
        sweep_width: float,
        start_reference: Tuple[float, float],
        end_reference: Tuple[float, float],
    ) -> List[Tuple[float, float]]:
        """
        生成最优全覆盖航点序列。

        Parameters
        ----------
        rect_min        : (x_min, y_min) 作业矩形左下角
        rect_max        : (x_max, y_max) 作业矩形右上角
        sweep_width     : 有效扫描行距 (米)
        start_reference : 无人机当前所在坐标
        end_reference   : 期望扫图结束后前往的坐标（例如地面车）

        Returns
        -------
        list of (x, y) 航点，按飞行顺序排列
        """
        rx0, ry0 = float(rect_min[0]), float(rect_min[1])
        rx1, ry1 = float(rect_max[0]), float(rect_max[1])
        s = np.array(start_reference, dtype=float)
        e = np.array(end_reference,   dtype=float)

        best_cost      = math.inf
        best_waypoints = None

        # ── 枚举 8 种拓扑 ────────────────────────────────────────────
        # 横向 (sweep along X，行沿 Y 步进) × 4 个角点起始
        # 纵向 (sweep along Y，行沿 X 步进) × 4 个角点起始
        for direction in ('horizontal', 'vertical'):
            strips = self._compute_strips(rx0, ry0, rx1, ry1,
                                          sweep_width, direction)
            topologies = self._enumerate_topologies(strips, direction)

            for waypoints in topologies:
                if len(waypoints) < 2:
                    continue
                wp_start = np.array(waypoints[0])
                wp_end   = np.array(waypoints[-1])
                cost = self._dist(s, wp_start) + self._dist(wp_end, e)

                if cost < best_cost:
                    best_cost      = cost
                    best_waypoints = waypoints

        return best_waypoints or []

    # ── 内部方法 ──────────────────────────────────────────────────────

    @staticmethod
    def _dist(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def _compute_strips(
        self,
        rx0: float, ry0: float, rx1: float, ry1: float,
        sweep_width: float,
        direction: str,
    ) -> List[float]:
        """
        计算扫描条带中心线坐标列表。

        横向扫描：条带沿 Y 轴均匀分布，返回每条带的 y 坐标。
        纵向扫描：条带沿 X 轴均匀分布，返回每条带的 x 坐标。

        采用半行距偏移使第一/最后一条带贴近边界（不漏扫）。
        """
        if direction == 'horizontal':
            span   = ry1 - ry0
            n      = max(1, math.ceil(span / sweep_width))
            step   = span / n
            return [ry0 + step * (i + 0.5) for i in range(n)]
        else:
            span   = rx1 - rx0
            n      = max(1, math.ceil(span / sweep_width))
            step   = span / n
            return [rx0 + step * (i + 0.5) for i in range(n)]

    def _enumerate_topologies(
        self,
        strips: List[float],
        direction: str,
    ) -> List[List[Tuple[float, float]]]:
        """
        对给定的条带列表枚举 4 种扫描拓扑（正序/逆序 × 每条带起点左/右）。

        direction='horizontal' : 条带是 y 坐标列表，沿 X 方向来回
        direction='vertical'   : 条带是 x 坐标列表，沿 Y 方向来回
        """
        topologies = []

        # strip_order: 正序(0) 或 逆序(1)
        # first_side : 第一条带从低端(0)还是高端(1)开始
        for strip_reversed in (False, True):
            ordered = list(reversed(strips)) if strip_reversed else strips
            for start_high in (False, True):
                wps = self._build_waypoints(ordered, direction, start_high)
                topologies.append(wps)

        return topologies

    def _build_waypoints(
        self,
        strips: List[float],
        direction: str,
        start_high: bool,
    ) -> List[Tuple[float, float]]:
        """
        根据有序条带列表和起点方向，生成完整牛耕式航点。

        横向（direction='horizontal'）：
            strips = [y1, y2, ...] (条带中心 y)
            每条带：从 (x_min/x_max, y_i) 飞到 (x_max/x_min, y_i)

        纵向（direction='vertical'）：
            strips = [x1, x2, ...] (条带中心 x)
            每条带：从 (x_i, y_min/y_max) 飞到 (x_i, y_max/y_min)
        """
        waypoints: List[Tuple[float, float]] = []
        # 这里用占位符，实际 rx0/rx1/ry0/ry1 需要从调用链传入
        # 利用 Python 的闭包特性，_build_waypoints 不单独使用；
        # 实际值由 generate_optimal_waypoints 的内部调用链负责。
        # 重构为将边界作为参数传入（见 _generate_strips_and_build）
        raise NotImplementedError("应通过 _generate_strips_and_build 调用")

    def _generate_strips_and_build(
        self,
        rx0: float, ry0: float, rx1: float, ry1: float,
        strips: List[float],
        direction: str,
        strip_reversed: bool,
        start_high: bool,
    ) -> List[Tuple[float, float]]:
        """将完整参数传入，生成航点。"""
        ordered = list(reversed(strips)) if strip_reversed else strips
        waypoints: List[Tuple[float, float]] = []

        for i, center in enumerate(ordered):
            going_right = (i % 2 == 0)           # 偶数条带从低到高，奇数反向
            if start_high:
                going_right = not going_right     # 整体翻转起始方向

            if direction == 'horizontal':
                y = center
                x_lo, x_hi = rx0, rx1
                if going_right:
                    waypoints.append((x_lo, y))
                    waypoints.append((x_hi, y))
                else:
                    waypoints.append((x_hi, y))
                    waypoints.append((x_lo, y))
            else:  # vertical
                x = center
                y_lo, y_hi = ry0, ry1
                if going_right:                   # 此处"right"表示沿 Y 正方向
                    waypoints.append((x, y_lo))
                    waypoints.append((x, y_hi))
                else:
                    waypoints.append((x, y_hi))
                    waypoints.append((x, y_lo))

        return waypoints

    def _enumerate_topologies(                    # noqa: F811  (覆盖上面的占位版本)
        self,
        strips: List[float],
        direction: str,
        rx0: float = 0, ry0: float = 0,
        rx1: float = 0, ry1: float = 0,
    ) -> List[List[Tuple[float, float]]]:
        topologies = []
        for strip_reversed in (False, True):
            for start_high in (False, True):
                wps = self._generate_strips_and_build(
                    rx0, ry0, rx1, ry1,
                    strips, direction, strip_reversed, start_high,
                )
                topologies.append(wps)
        return topologies

    def generate_optimal_waypoints(              # noqa: F811  (覆盖上面的版本，解决边界传递问题)
        self,
        rect_min: Tuple[float, float],
        rect_max: Tuple[float, float],
        sweep_width: float,
        start_reference: Tuple[float, float],
        end_reference: Tuple[float, float],
    ) -> List[Tuple[float, float]]:
        rx0, ry0 = float(rect_min[0]), float(rect_min[1])
        rx1, ry1 = float(rect_max[0]), float(rect_max[1])
        s = np.array(start_reference, dtype=float)
        e = np.array(end_reference,   dtype=float)

        best_cost      = math.inf
        best_waypoints: Optional[List[Tuple[float, float]]] = None

        for direction in ('horizontal', 'vertical'):
            strips = self._compute_strips(rx0, ry0, rx1, ry1,
                                          sweep_width, direction)
            topologies = self._enumerate_topologies(
                strips, direction, rx0, ry0, rx1, ry1)

            for waypoints in topologies:
                if len(waypoints) < 2:
                    continue
                wp_start = np.array(waypoints[0])
                wp_end   = np.array(waypoints[-1])
                cost = (np.linalg.norm(s - wp_start)
                        + np.linalg.norm(wp_end - e))

                if cost < best_cost:
                    best_cost      = cost
                    best_waypoints = waypoints

        return best_waypoints or []

    def get_all_topologies(
        self,
        rect_min: Tuple[float, float],
        rect_max: Tuple[float, float],
        sweep_width: float,
        start_reference: Tuple[float, float],
        end_reference: Tuple[float, float],
    ) -> List[dict]:
        """
        返回全部 8 种拓扑的详细信息（用于调试/对比可视化）。

        Returns
        -------
        list of dict，每项包含:
            'direction'  : 'horizontal' / 'vertical'
            'strip_rev'  : bool
            'start_high' : bool
            'waypoints'  : list of (x, y)
            'cost'       : float（通勤代价）
            'optimal'    : bool（是否最优）
        """
        rx0, ry0 = float(rect_min[0]), float(rect_min[1])
        rx1, ry1 = float(rect_max[0]), float(rect_max[1])
        s = np.array(start_reference, dtype=float)
        e = np.array(end_reference,   dtype=float)

        results = []
        for direction in ('horizontal', 'vertical'):
            strips = self._compute_strips(rx0, ry0, rx1, ry1,
                                          sweep_width, direction)
            for strip_reversed in (False, True):
                for start_high in (False, True):
                    wps = self._generate_strips_and_build(
                        rx0, ry0, rx1, ry1,
                        strips, direction, strip_reversed, start_high,
                    )
                    if not wps:
                        continue
                    wp_s = np.array(wps[0])
                    wp_e = np.array(wps[-1])
                    cost = (float(np.linalg.norm(s - wp_s))
                            + float(np.linalg.norm(wp_e - e)))
                    results.append({
                        'direction':  direction,
                        'strip_rev':  strip_reversed,
                        'start_high': start_high,
                        'waypoints':  wps,
                        'cost':       cost,
                        'optimal':    False,
                    })

        if results:
            best_idx = min(range(len(results)), key=lambda i: results[i]['cost'])
            results[best_idx]['optimal'] = True

        return results


# ─────────────────────────────────────────────────────────────────────
#  测试与可视化
# ─────────────────────────────────────────────────────────────────────

def _add_arrow(ax, x1, y1, x2, y2, color='steelblue', lw=1.5):
    """在线段中点处添加方向箭头。"""
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = x2 - x1, y2 - y1
    ax.annotate(
        '', xy=(mx + dx * 0.01, my + dy * 0.01),
        xytext=(mx - dx * 0.01, my - dy * 0.01),
        arrowprops=dict(
            arrowstyle='->', color=color,
            lw=lw, mutation_scale=14,
        ),
    )


def visualize_optimal(
    waypoints: List[Tuple[float, float]],
    rect_min: Tuple[float, float],
    rect_max: Tuple[float, float],
    start_ref: Tuple[float, float],
    end_ref: Tuple[float, float],
    title: str = "Boustrophedon Coverage Path (Optimal)",
):
    """单图可视化最优路径。"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(8, 8))

    # 矩形边界
    rect_w = rect_max[0] - rect_min[0]
    rect_h = rect_max[1] - rect_min[1]
    ax.add_patch(patches.Rectangle(
        rect_min, rect_w, rect_h,
        linewidth=2, edgecolor='black', facecolor='#F0F8FF', alpha=0.6,
        label='Task Area',
    ))

    # 航点连线 + 箭头
    xs = [p[0] for p in waypoints]
    ys = [p[1] for p in waypoints]
    ax.plot(xs, ys, '-o', color='steelblue', lw=1.8,
            markersize=4, markerfacecolor='white', markeredgecolor='steelblue',
            label='Flight Path', zorder=3)
    for i in range(len(waypoints) - 1):
        _add_arrow(ax, waypoints[i][0], waypoints[i][1],
                   waypoints[i+1][0], waypoints[i+1][1])

    # 扫图真实起点 / 终点
    ax.scatter(*waypoints[0],  s=120, c='limegreen', zorder=5,
               edgecolors='darkgreen', linewidths=1.5,
               label=f'Sweep Start {waypoints[0]}')
    ax.scatter(*waypoints[-1], s=120, c='tomato',    zorder=5,
               edgecolors='darkred', linewidths=1.5,
               label=f'Sweep End  {waypoints[-1]}')

    # 参考点
    ax.scatter(*start_ref, s=160, c='lime',   marker='o', zorder=6,
               edgecolors='green', linewidths=2,
               label=f'UAV Start Ref {start_ref}')
    ax.scatter(*end_ref,   s=160, c='red',    marker='s', zorder=6,
               edgecolors='darkred', linewidths=2,
               label=f'Car End Ref {end_ref}')

    # 通勤虚线
    ax.plot([start_ref[0], waypoints[0][0]],
            [start_ref[1], waypoints[0][1]],
            '--', color='green', lw=1.2, alpha=0.6, label='Commute IN')
    ax.plot([waypoints[-1][0], end_ref[0]],
            [waypoints[-1][1], end_ref[1]],
            '--', color='red',   lw=1.2, alpha=0.6, label='Commute OUT')

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.4)
    margin = max(rect_w, rect_h) * 0.3
    ax.set_xlim(min(start_ref[0], end_ref[0], rect_min[0]) - margin,
                max(start_ref[0], end_ref[0], rect_max[0]) + margin)
    ax.set_ylim(min(start_ref[1], end_ref[1], rect_min[1]) - margin,
                max(start_ref[1], end_ref[1], rect_max[1]) + margin)

    plt.tight_layout()
    plt.show()


def visualize_all_topologies(
    all_topo: List[dict],
    rect_min: Tuple[float, float],
    rect_max: Tuple[float, float],
    start_ref: Tuple[float, float],
    end_ref: Tuple[float, float],
):
    """2×4 子图对比全部 8 种拓扑。"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    n = len(all_topo)
    cols = 4
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(18, 8))
    axes = axes.flatten()

    for idx, topo in enumerate(all_topo):
        ax   = axes[idx]
        wps  = topo['waypoints']
        cost = topo['cost']
        is_best = topo['optimal']

        color = 'tomato' if is_best else 'steelblue'
        lw    = 2.5 if is_best else 1.2

        rect_w = rect_max[0] - rect_min[0]
        rect_h = rect_max[1] - rect_min[1]
        ax.add_patch(patches.Rectangle(
            rect_min, rect_w, rect_h,
            linewidth=1, edgecolor='black', facecolor='#F0F8FF', alpha=0.5,
        ))

        xs = [p[0] for p in wps]
        ys = [p[1] for p in wps]
        ax.plot(xs, ys, '-o', color=color, lw=lw,
                markersize=3, markerfacecolor='white', markeredgecolor=color)
        for i in range(len(wps) - 1):
            _add_arrow(ax, wps[i][0], wps[i][1],
                       wps[i+1][0], wps[i+1][1], color=color, lw=lw)

        ax.scatter(*wps[0],  s=60, c='limegreen', zorder=5)
        ax.scatter(*wps[-1], s=60, c='red',       zorder=5)
        ax.scatter(*start_ref, s=80, c='lime', marker='o', zorder=6)
        ax.scatter(*end_ref,   s=80, c='red',  marker='s', zorder=6)

        tag = f"{'★ ' if is_best else ''}{topo['direction'][0].upper()}"
        tag += f" | rev={topo['strip_rev']} sh={topo['start_high']}"
        ax.set_title(f"{tag}\ncost={cost:.1f}", fontsize=8,
                     color='darkred' if is_best else 'black',
                     fontweight='bold' if is_best else 'normal')
        ax.set_aspect('equal')
        ax.tick_params(labelsize=6)
        ax.grid(True, linestyle=':', alpha=0.3)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("All 8 Boustrophedon Topologies (★ = Optimal)", fontsize=13)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────
#  主测试入口
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── 参数设置 ────────────────────────────────────────────────────
    RECT_MIN    = (20.0, 20.0)
    RECT_MAX    = (80.0, 80.0)
    SWEEP_WIDTH = 10.0
    START_REF   = (0.0,   0.0)    # 无人机当前位置（绿圆）
    END_REF     = (100.0, 100.0)  # 地面车期望接应位置（红方块）

    planner = SmartBoustrophedonPlanner()

    # ── 生成最优路径 ────────────────────────────────────────────────
    optimal_wps = planner.generate_optimal_waypoints(
        rect_min=RECT_MIN, rect_max=RECT_MAX,
        sweep_width=SWEEP_WIDTH,
        start_reference=START_REF,
        end_reference=END_REF,
    )

    print("=" * 60)
    print("  Boustrophedon 最优覆盖路径规划结果")
    print("=" * 60)
    print(f"  矩形范围    : {RECT_MIN} → {RECT_MAX}")
    print(f"  扫描行距    : {SWEEP_WIDTH} m")
    print(f"  无人机起始  : {START_REF}")
    print(f"  地面车位置  : {END_REF}")
    print(f"  生成航点数  : {len(optimal_wps)}")
    print(f"  扫图起点    : {optimal_wps[0]}")
    print(f"  扫图终点    : {optimal_wps[-1]}")

    s = np.array(START_REF)
    e = np.array(END_REF)
    commute_in  = np.linalg.norm(s - np.array(optimal_wps[0]))
    commute_out = np.linalg.norm(np.array(optimal_wps[-1]) - e)
    total_cost  = commute_in + commute_out

    print(f"  通勤IN 代价 : {commute_in:.2f} m")
    print(f"  通勤OUT代价 : {commute_out:.2f} m")
    print(f"  总通勤代价  : {total_cost:.2f} m  (8种拓扑中最小)")
    print("=" * 60)
    print("  航点列表:")
    for i, wp in enumerate(optimal_wps):
        print(f"    [{i:02d}] ({wp[0]:7.2f}, {wp[1]:7.2f})")
    print("=" * 60)

    # ── 可视化所有拓扑（对比图）────────────────────────────────────
    print("\n[1/2] 绘制全部 8 种拓扑对比图...")
    all_topo = planner.get_all_topologies(
        rect_min=RECT_MIN, rect_max=RECT_MAX,
        sweep_width=SWEEP_WIDTH,
        start_reference=START_REF,
        end_reference=END_REF,
    )
    visualize_all_topologies(all_topo, RECT_MIN, RECT_MAX, START_REF, END_REF)

    # ── 可视化最优路径（详细图）────────────────────────────────────
    print("[2/2] 绘制最优路径详细图...")
    visualize_optimal(
        optimal_wps, RECT_MIN, RECT_MAX, START_REF, END_REF,
        title=f"Optimal Boustrophedon Path  |  Cost={total_cost:.1f}m",
    )
