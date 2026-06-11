"""
08_regenerate_pipeline.py — Fig 1 analytical pipeline (JIE-compliant)
====================================================================
Recreates the pipeline schematic at 600 dpi, sans-serif, with no in-figure
title, and with terminology consistent with the stylised sensitivity-analysis
framing (Stage 3). Self-contained (no data inputs).
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

from config import FIG_DIR as FIGS

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})

NAVY = "#2c3e50"
BOX_EDGE = "#34495e"
BOX_FILL = "#f4f6f7"
HILITE_FILL = "#d6e4f0"

fig, ax = plt.subplots(figsize=(7.2, 9.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 20); ax.axis("off")
CX = 5.6
BW, BH = 5.4, 1.0


def box(cx, cy, text, w=BW, h=BH, fill=BOX_FILL, bold=False, edge=BOX_EDGE, lw=1.3):
    p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                       boxstyle="round,pad=0.04,rounding_size=0.12",
                       linewidth=lw, edgecolor=edge, facecolor=fill, zorder=2)
    ax.add_patch(p)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=11,
            fontweight="bold" if bold else "normal", color=NAVY, zorder=3)


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                 mutation_scale=16, color="#7f8c8d", linewidth=1.6, zorder=1))


def stage(cy, num, label):
    ax.add_patch(Circle((0.7, cy), 0.32, color=NAVY, zorder=4))
    ax.text(0.7, cy, str(num), ha="center", va="center", color="white",
            fontsize=12, fontweight="bold", zorder=5)
    ax.text(1.3, cy, label, ha="left", va="center", fontsize=13, fontweight="bold", color=NAVY)


stage(19.3, 1, "Tensor construction")
box(CX, 18.3, "GLORIA MRIO (Loop060)\n164 regions · 120 sectors · 6,130 satellites · 33 years", h=1.1)
box(CX, 16.8, "Material satellite extraction (TQ matrices)")
arrow(CX, 17.75, CX, 17.32)
box(CX, 15.4, "Raw tensor   31 × 120 × 367 × 33")
arrow(CX, 16.3, CX, 15.92)
box(CX, 14.0, "Aggregation  +  log(1 + x) transform")
arrow(CX, 14.9, CX, 14.52)
box(CX, 12.6, "Analysis tensor   31 × 120 × 15 × 33")
arrow(CX, 13.5, CX, 13.12)

stage(11.3, 2, "NTF decomposition")
arrow(CX, 12.1, CX, 10.92)
box(CX, 10.4, "Rank selection   K = 2 … 10")
arrow(CX, 9.9, CX, 9.52)
box(CX, 9.0, "NTF  (K = 6)    R² = 0.75", fill=HILITE_FILL, bold=True, lw=1.8)
arrow(CX, 8.5, CX, 8.12)
box(CX, 7.6, "Component loadings")

stage(6.3, 3, "Sensitivity analysis and mapping")
arrow(CX, 7.1, CX, 6.0)
LX, RX = 3.0, 8.2
ax.add_patch(FancyArrowPatch((CX, 6.0), (LX, 5.3), arrowstyle="-|>", mutation_scale=16,
             color="#7f8c8d", linewidth=1.6, zorder=1))
ax.add_patch(FancyArrowPatch((CX, 6.0), (RX, 5.3), arrowstyle="-|>", mutation_scale=16,
             color="#7f8c8d", linewidth=1.6, zorder=1))
box(LX, 4.8, "Geospatial\nmapping", w=3.0, h=1.1)
box(RX, 4.8, "Stylised scenario\nsensitivity analysis", w=3.4, h=1.1)
arrow(RX, 4.25, RX, 3.62)
box(RX, 3.1, "Shock sweep +\ntrend bootstrap", w=3.4, h=1.1)

fig.savefig(FIGS / "fig01_pipeline.png", dpi=600, bbox_inches="tight")
plt.close(fig)
print("saved fig01_pipeline.png at 600 dpi")
