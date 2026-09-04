import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'my_method'))
from boustrophedon_planner import SmartBoustrophedonPlanner, _add_arrow

def save_fig2():
    RECT_MIN = (20.0, 20.0)
    RECT_MAX = (80.0, 80.0)
    SWEEP_WIDTH = 12.0
    START_REF = (0.0, 10.0)
    END_REF = (95.0, 95.0)

    planner = SmartBoustrophedonPlanner()
    all_topo = planner.get_all_topologies(RECT_MIN, RECT_MAX, SWEEP_WIDTH, START_REF, END_REF)
    
    n = len(all_topo)
    cols = 4
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 8))
    axes = axes.flatten()

    for idx, topo in enumerate(all_topo):
        ax = axes[idx]
        wps = topo['waypoints']
        cost = topo['cost']
        is_best = topo['optimal']

        color = 'tomato' if is_best else 'steelblue'
        lw = 2.5 if is_best else 1.2

        rect_w = RECT_MAX[0] - RECT_MIN[0]
        rect_h = RECT_MAX[1] - RECT_MIN[1]
        
        # Draw bounding box
        ax.add_patch(patches.Rectangle(
            RECT_MIN, rect_w, rect_h,
            linewidth=1.5, edgecolor='black', linestyle='--', facecolor='#F0F8FF', alpha=0.5,
        ))

        xs = [p[0] for p in wps]
        ys = [p[1] for p in wps]
        ax.plot(xs, ys, '-o', color=color, lw=lw,
                markersize=3, markerfacecolor='white', markeredgecolor=color)
        for i in range(len(wps) - 1):
            _add_arrow(ax, wps[i][0], wps[i][1], wps[i+1][0], wps[i+1][1], color=color, lw=lw)

        ax.scatter(*wps[0], s=60, c='limegreen', zorder=5)
        ax.scatter(*wps[-1], s=60, c='red', zorder=5)
        ax.scatter(*START_REF, s=100, c='lime', marker='o', zorder=6, edgecolors='black')
        ax.scatter(*END_REF, s=100, c='red', marker='s', zorder=6, edgecolors='black')
        
        # Draw commute lines
        ax.plot([START_REF[0], wps[0][0]], [START_REF[1], wps[0][1]], '--', color='green', alpha=0.6)
        ax.plot([wps[-1][0], END_REF[0]], [wps[-1][1], END_REF[1]], '--', color='red', alpha=0.6)

        tag = f"{'★ Optimal: ' if is_best else ''}{topo['direction'].capitalize()}"
        tag += f"\nRev={topo['strip_rev']} | StartHigh={topo['start_high']}"
        ax.set_title(f"{tag}\nCost = {cost:.1f} m", fontsize=10,
                     color='darkred' if is_best else 'black',
                     fontweight='bold' if is_best else 'normal')
        ax.set_aspect('equal')
        ax.axis('off')

    plt.tight_layout()
    os.makedirs('../../paper_figures', exist_ok=True)
    plt.rcParams['pdf.fonttype'] = 42
    plt.savefig('../../paper_figures/fig2_boustrophedon.png', dpi=300, bbox_inches='tight')
    plt.savefig('../../paper_figures/fig2_boustrophedon.pdf', bbox_inches='tight')
    print("Saved ../../paper_figures/fig2_boustrophedon.png")

if __name__ == '__main__':
    save_fig2()
