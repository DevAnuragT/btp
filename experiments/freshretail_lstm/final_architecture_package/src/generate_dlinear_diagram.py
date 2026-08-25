import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set clean styling parameters
plt.rcParams.update({
    'font.sans-serif': 'DejaVu Sans',
    'font.family': 'sans-serif',
    'savefig.dpi': 300
})

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis("off")

# Helper function to draw rounded boxes
def draw_box(ax, x, y, w, h, title, subtitle="", bg_color="#fff3e0", border_color="#e65100", title_color="#e65100", fontsize=9.5):
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.25",
        facecolor=bg_color,
        edgecolor=border_color,
        linewidth=1.5,
        zorder=3
    )
    ax.add_patch(rect)
    if subtitle:
        ax.text(x + w/2, y + h/2 + 0.15, title, ha="center", va="center", fontsize=fontsize, fontweight="bold", color=title_color, zorder=4)
        ax.text(x + w/2, y + h/2 - 0.22, subtitle, ha="center", va="center", fontsize=fontsize-1.5, color="#37474f", zorder=4)
    else:
        ax.text(x + w/2, y + h/2, title, ha="center", va="center", fontsize=fontsize, fontweight="bold", color=title_color, zorder=4)

# Helper function to draw arrows
def draw_arrow(ax, x1, y1, x2, y2, label="", color="#37474f", lw=1.8, style="->"):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle=style, color=color, lw=lw, mutation_scale=14),
        zorder=2
    )
    if label:
        mid_x, mid_y = (x1 + x2)/2, (y1 + y2)/2
        ax.text(mid_x, mid_y + 0.2, label, ha="center", va="center", fontsize=8.5, fontweight="bold", color="#e65100", zorder=5)

# --- 1. TITLE & SUBTITLE ---
ax.text(8, 9.5, "Hourly Slot-Specific & Multi-Kernel DLinear Architecture", ha="center", va="center", fontsize=14, fontweight="bold", color="#e65100")
ax.text(8, 9.15, "Linear Series Decomposition with 24 Dedicated Hourly Projection Heads", ha="center", va="center", fontsize=10.5, style="italic", color="#455a64")

# --- 2. INPUT SEQUENCE MATRIX ---
draw_box(ax, 0.5, 4.0, 2.5, 2.0, "Input Sequence X", "L=14 Days × d_in=10 Feats\n[Sales, Stockouts, Discounts]", bg_color="#e8eaf6", border_color="#303f9f", title_color="#1a237e")

# Arrow from Input splitting into Trend and Seasonal pathways
draw_arrow(ax, 3.0, 5.0, 4.0, 6.8, "Upper Pathway (Trend)")
draw_arrow(ax, 3.0, 5.0, 4.0, 3.2, "Lower Pathway (Seasonal)")

# --- 3. TREND DECOMPOSITION PATHWAY (TOP) ---
draw_box(ax, 4.1, 6.0, 3.2, 1.6, "Multi-Kernel AvgPooling", "Moving Average Filters\nk ∈ {3, 5, 7} Pooling Kernels", bg_color="#ffe0b2", border_color="#e65100", title_color="#e65100")

draw_arrow(ax, 7.3, 6.8, 8.3, 6.8)

draw_box(ax, 8.4, 6.0, 2.8, 1.6, "Trend Component X_trend", "X_trend = AvgPool(X)\nCaptures Multi-Day Velocity", bg_color="#fff3e0", border_color="#f57f17", title_color="#f57f17")

# --- 4. SEASONAL DECOMPOSITION PATHWAY (BOTTOM) ---
draw_box(ax, 4.1, 2.4, 3.2, 1.6, "Seasonal Subtraction", "X_seasonal = X - X_trend\nIsolates High-Freq Residuals", bg_color="#e0f2f1", border_color="#00695c", title_color="#00695c")

draw_arrow(ax, 7.3, 3.2, 8.3, 3.2)

draw_box(ax, 8.4, 2.4, 2.8, 1.6, "Seasonal Component X_seas", "High-Frequency Shocks\nIntraday & Weekly Cycles", bg_color="#e1f5fe", border_color="#0277bd", title_color="#0277bd")

# --- 5. 24 DEDICATED HOURLY SLOT LINEAR HEADS (RIGHT) ---
draw_arrow(ax, 11.2, 6.8, 12.0, 5.5)
draw_arrow(ax, 11.2, 3.2, 12.0, 4.5)

draw_box(ax, 12.1, 4.0, 3.4, 2.0, "24 Dedicated Hourly Heads", "y_h = W_trend,h X_trend + W_seas,h X_seas\n(24 Independent Projections W_1...W_24)", bg_color="#d1c4e9", border_color="#4a148c", title_color="#4a148c")

draw_arrow(ax, 13.8, 4.0, 13.8, 2.6)

# --- 6. OUTPUT PREDICTION VEC ---
draw_box(ax, 12.5, 1.2, 2.6, 1.4, "Output Vector Y_hat", "Y_hat = σ(Logits) ∈ [0, 1]^{24}\n24 Binary Hourly Stockout Probs", bg_color="#c8e6c9", border_color="#1b5e20", title_color="#1b5e20")

# High-level summary box on bottom left
summary_box = patches.FancyBboxPatch(
    (0.5, 0.8), 3.5, 2.4,
    boxstyle="round,pad=0.25",
    facecolor="#f5f5f5",
    edgecolor="#9e9e9e",
    linewidth=1.2,
    zorder=2
)
ax.add_patch(summary_box)
ax.text(2.25, 2.9, "DLinear Architectural Summary", ha="center", va="center", fontsize=9, fontweight="bold", color="#212121")
summary_text = (
    "• Single-Layer Linear Weight Efficiency\n"
    "• Trend Pooling: k=3, 5, 7 kernels\n"
    "• 24 Dedicated Hourly Linear Heads\n"
    "• Eliminates inter-hour gradient noise\n"
    "• F1-Score: 0.8058 (Slot) / 0.8131 (MK)\n"
    "• Parameters: 6,744 (Slot) / 13,539 (MK)"
)
ax.text(2.25, 1.8, summary_text, ha="center", va="center", fontsize=7.8, color="#424242", linespacing=1.4)

out_path = "docs/images/dlinear_architecture_diagram.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"\nSuccessfully generated DLinear architecture diagram at: {out_path}")
