"""
plot_scalability_envelope.py — Fig_5_6 (fixed-interface scale-transfer envelope),
recreated to fix the Type-3 DejaVu fonts in the old PDF and to match the journal
style of the other figures. (a) SI-HMARL makespan vs map scale (log-y, mean +/- std);
(b) coverage-complete success rate vs scale, with the reliable / transition /
breakdown regions shaded.

Values are the paper's transfer table (tab:scalability). Hardcoded so it
regenerates anywhere.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import apply_plot_style, format_ax

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'paper_figures', 'Ch5_Scalability')
os.makedirs(OUT_DIR, exist_ok=True)

HERO = "#2878B5"

# makespan defined only where coverage completes (up to 7 km)
SCALE_MS = [0.5, 1, 2, 3, 4, 5, 6, 7]
MS_MEAN  = [596, 1893, 6472, 15259, 22918, 39646, 51939, 82562]
MS_STD   = [18, 39, 214, 321, 616, 1752, 2658, 11238]
# success across the full envelope
SCALE_SR = [0.5, 1, 2, 3, 4, 5, 6, 7, 8]
SUCCESS  = [100, 100, 100, 100, 100, 100, 100, 70, 0]


def main():
    apply_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.0))

    # (a) makespan vs scale, log-y
    ax1.errorbar(SCALE_MS, MS_MEAN, yerr=MS_STD, fmt='o-', color=HERO, mfc=HERO,
                 mec='black', mew=0.8, markersize=6, capsize=4, capthick=1.2,
                 elinewidth=1.2, linewidth=1.6, zorder=3, label='SI-HMARL')
    ax1.axvspan(1.9, 2.1, color='#cccccc', alpha=0.5, zorder=0)
    ax1.text(2, MS_MEAN[2] * 0.30, 'train\nscale', ha='center', va='top',
             fontsize=6.5, style='italic', color='#666666')
    ax1.set_yscale('log')
    ax1.set_xlabel("Map scale (km)", fontweight='bold')
    ax1.set_ylabel("Makespan (ticks)", fontweight='bold')
    ax1.set_title("(a) Makespan vs. scale", fontweight='bold')
    ax1.set_xticks(SCALE_SR)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: format(int(v), ',')))
    format_ax(ax1)

    # (b) success vs scale, region-shaded
    ax2.axvspan(0.3, 6.5, color='#659266', alpha=0.12, zorder=0)
    ax2.axvspan(6.5, 7.5, color='#E68A5C', alpha=0.14, zorder=0)
    ax2.axvspan(7.5, 8.3, color='#C25E5E', alpha=0.14, zorder=0)
    ax2.plot(SCALE_SR, SUCCESS, 's-', color=HERO, mfc=HERO, mec='black', mew=0.8,
             markersize=6, linewidth=1.6, zorder=3)
    for x, y in zip(SCALE_SR, SUCCESS):
        if y in (70, 0) or x == 6:
            ax2.text(x, y + 4, f"{y}%", ha='center', va='bottom', fontsize=7,
                     fontweight='bold', color='#111111')
    ax2.text(3.4, 8, 'reliable', ha='center', fontsize=6.5, style='italic', color='#4a6b4a')
    ax2.text(7, 40, 'transition', ha='center', fontsize=6.5, style='italic', color='#9c5a36', rotation=90)
    ax2.text(8, 40, 'breakdown', ha='center', fontsize=6.5, style='italic', color='#8a3b3b', rotation=90)
    ax2.set_ylim(-5, 112)
    ax2.set_xlabel("Map scale (km)", fontweight='bold')
    ax2.set_ylabel("Success rate (%)", fontweight='bold')
    ax2.set_title("(b) Coverage-complete success", fontweight='bold')
    ax2.set_xticks(SCALE_SR)
    format_ax(ax2)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "Fig_5_6_Scalability_Envelope.pdf"), facecolor='white')
    fig.savefig(os.path.join(OUT_DIR, "Fig_5_6_Scalability_Envelope.png"), facecolor='white')
    plt.close(fig)
    print("Saved Fig_5_6_Scalability_Envelope.{pdf,png}")


if __name__ == "__main__":
    main()
