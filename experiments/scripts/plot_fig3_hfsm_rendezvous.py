import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from matplotlib.path import Path

def setup_academic_style():
    sns.set_theme(style="white", context="paper", font_scale=1.4)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["text.usetex"] = False  # Avoid requiring latex installation, use mathtext
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["axes.linewidth"] = 1.2
    plt.rcParams["xtick.major.width"] = 1.2
    plt.rcParams["ytick.major.width"] = 1.2

def draw_fancy_arrow(ax, start, end, color="#2C3E50", rad=0.2, text="", text_offset=(0,0)):
    ax.annotate("",
                xy=end, xycoords='data',
                xytext=start, textcoords='data',
                arrowprops=dict(arrowstyle="-|>,head_length=0.8,head_width=0.3",
                                lw=1.5, color=color,
                                connectionstyle=f"arc3,rad={rad}"))
    if text:
        mx = (start[0] + end[0]) / 2 + text_offset[0]
        my = (start[1] + end[1]) / 2 + text_offset[1]
        ax.text(mx, my, text, ha='center', va='center', fontsize=11, color=color, 
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1))

def draw_hfsm(ax):
    ax.set_title("(a) Hybrid Finite State Machine", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#BDC3C7')
    
    # State Nodes
    states = {
        'Work': (0.25, 0.75),
        'Approach': (0.75, 0.75),
        'Swap': (0.5, 0.25)
    }
    
    # Academic Colors
    color_fill = "#F8F9FA"
    color_edge = "#343A40"
    
    for name, pos in states.items():
        # Shadow
        shadow = patches.Circle((pos[0]+0.01, pos[1]-0.01), 0.15, facecolor='gray', alpha=0.2, zorder=1)
        ax.add_patch(shadow)
        # Node
        circle = patches.Circle(pos, 0.15, facecolor=color_fill, edgecolor=color_edge, linewidth=2, zorder=3)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], name, ha='center', va='center', fontsize=13, fontweight='bold', color=color_edge, zorder=4)
        
    # Transitions
    draw_fancy_arrow(ax, (0.43, 0.75), (0.56, 0.75), color="#E74C3C", rad=0.15, text="Energy < $E_{low}$", text_offset=(0, 0.12))
    draw_fancy_arrow(ax, (0.7, 0.58), (0.58, 0.38), color="#27AE60", rad=0.2, text="Distance $\leq$ Tolerance", text_offset=(0.14, 0.1))
    draw_fancy_arrow(ax, (0.42, 0.38), (0.3, 0.58), color="#2980B9", rad=0.2, text="Energy == $E_{max}$", text_offset=(-0.14, 0.1))

def draw_rendezvous(ax):
    ax.set_title("(b) Continuous Rendezvous Mechanism", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    # Grid
    ax.grid(True, linestyle=':', alpha=0.6, color='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#BDC3C7')
    
    # Road Network segment
    v_a = np.array([2, 3])
    v_b = np.array([8, 6])
    
    # Draw Road shadow and line
    ax.plot([v_a[0], v_b[0]], [v_a[1], v_b[1]], color="#BDC3C7", lw=12, solid_capstyle='round', zorder=1)
    ax.plot([v_a[0], v_b[0]], [v_a[1], v_b[1]], color="#7F8C8D", lw=3, zorder=2, label='Road Segment $\mathcal{E}$')
    
    # Nodes
    ax.plot(*v_a, 'o', markersize=12, color="#2C3E50", zorder=4)
    ax.plot(*v_b, 'o', markersize=12, color="#2C3E50", zorder=4)
    ax.text(v_a[0]-0.6, v_a[1]-0.4, "Node $v_a$", fontsize=12, color="#2C3E50")
    ax.text(v_b[0]+0.4, v_b[1]-0.2, "Node $v_b$", fontsize=12, color="#2C3E50")
    
    # UGV position (Exact perpendicular projection for mathematical visual accuracy)
    lam = 0.6
    p_ugv = (1 - lam) * v_a + lam * v_b
    ax.plot(*p_ugv, 's', markersize=14, color="#E67E22", markeredgecolor='black', zorder=5, label='UGV: $p_{ugv}(\lambda)$')
    ax.text(p_ugv[0]+0.3, p_ugv[1]-0.6, "$p_{ugv}(\lambda)$", fontsize=12, fontweight='bold', color="#D35400")
    
    # UAV position
    p_uav = np.array([4, 8])
    ax.plot(*p_uav, '^', markersize=16, color="#3498DB", markeredgecolor='black', zorder=5, label='UAV: $p_{uav}(t_{rth})$')
    ax.text(p_uav[0]-1.0, p_uav[1]+0.4, "$p_{uav}(t_{rth})$", fontsize=12, fontweight='bold', color="#2980B9")
    
    # Mathematical projection line (Continuous)
    ax.plot([p_uav[0], p_ugv[0]], [p_uav[1], p_ugv[1]], '-', color="#27AE60", lw=2.5, zorder=3)
    
    # Discrete paths
    ax.plot([p_uav[0], v_a[0]], [p_uav[1], v_a[1]], '--', color="#E74C3C", lw=1.5, alpha=0.8, zorder=3)
    ax.plot([p_uav[0], v_b[0]], [p_uav[1], v_b[1]], '--', color="#E74C3C", lw=1.5, alpha=0.8, zorder=3)
    
    # Annotations for distance
    mid_cont = (p_uav + p_ugv) / 2
    ax.text(mid_cont[0]+0.2, mid_cont[1]+0.2, "$D_{continuous}$", color="#27AE60", fontsize=12, fontweight='bold', 
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5))
            
    mid_da = (p_uav + v_a) / 2
    ax.text(mid_da[0]-0.8, mid_da[1], "$D_{discrete}^{(a)}$", color="#C0392B", fontsize=11,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5))
            
    mid_db = (p_uav + v_b) / 2
    ax.text(mid_db[0]+0.3, mid_db[1]+0.3, "$D_{discrete}^{(b)}$", color="#C0392B", fontsize=11,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5))
            
    ax.legend(loc='lower left', fontsize=10, framealpha=0.9, edgecolor='#BDC3C7')

def save_fig3():
    setup_academic_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), gridspec_kw={'width_ratios': [1, 1.2]})
    draw_hfsm(ax1)
    draw_rendezvous(ax2)
    plt.tight_layout(pad=2.0)
    
    # Get paper_figures path robustly relative to the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    paper_figures_dir = os.path.abspath(os.path.join(script_dir, '..', '..', 'paper_figures'))
    os.makedirs(paper_figures_dir, exist_ok=True)
    
    save_path = os.path.join(paper_figures_dir, 'fig3_hfsm_rendezvous.png')
    plt.rcParams['pdf.fonttype'] = 42
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved optimized figure to {save_path}")

if __name__ == '__main__':
    save_fig3()
