import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set clean styling parameters
plt.rcParams.update({
    'font.sans-serif': 'DejaVu Sans',
    'font.family': 'sans-serif',
    'savefig.dpi': 300
})

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9.5)
ax.axis("off")

# Helper function to draw rounded cards with soft shadows
def draw_card(ax, x, y, w, h, title, text_lines=[], bg_color="#ffffff", border_color="#1e88e5", title_color="#0d47a1", title_size=11, text_size=9.0, shadow=True):
    if shadow:
        shadow_rect = patches.FancyBboxPatch(
            (x + 0.08, y - 0.08), w, h,
            boxstyle="round,pad=0.2",
            facecolor="#d5d5d5",
            edgecolor="none",
            alpha=0.5,
            zorder=1
        )
        ax.add_patch(shadow_rect)
        
    card = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.2",
        facecolor=bg_color,
        edgecolor=border_color,
        linewidth=1.8,
        zorder=2
    )
    ax.add_patch(card)
    
    # Title
    ax.text(x + w/2, y + h - 0.38, title, ha="center", va="top", fontsize=title_size, fontweight="bold", color=title_color, zorder=3)
    
    # Text lines
    if text_lines:
        line_y = y + h - 0.85
        for line in text_lines:
            ax.text(x + w/2, line_y, line, ha="center", va="top", fontsize=text_size, color="#263238", zorder=3)
            line_y -= 0.35

# Helper function to draw sharp stylized arrows
def draw_arrow(ax, start, end, color="#37474f", lw=2.2, rad=0.0):
    connection = f"arc3,rad={rad}" if rad != 0.0 else "arc3"
    ax.annotate(
        "", xy=end, xytext=start,
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            mutation_scale=16,
            connectionstyle=connection
        ),
        zorder=4
    )

# --- 1. MAIN TITLE BANNER ---
title_box = patches.FancyBboxPatch(
    (0.4, 8.4), 15.2, 0.9,
    boxstyle="round,pad=0.1",
    facecolor="#0d47a1",
    edgecolor="none",
    zorder=2
)
ax.add_patch(title_box)
ax.text(8.0, 8.95, "Multi-Kernel & Slot DLinear Core Architecture Schematic", ha="center", va="center", fontsize=14, fontweight="bold", color="white", zorder=3)
ax.text(8.0, 8.62, "Series Decomposition → Multi-Scale Trend & Seasonal Pathways → GELU Bottleneck → 24 Hourly Slot Heads", ha="center", va="center", fontsize=9.5, style="italic", color="#e3f2fd", zorder=3)

# --- 2. INPUT SEQUENCE ---
draw_card(
    ax, 0.4, 3.2, 2.5, 4.5,
    "Input Sequence X",
    [
        "L = 14 Days (336 hrs)",
        "C = 10 Features",
        "",
        "• Hourly Sales Volume",
        "• Stockout Flags",
        "• Price & Discounts",
        "• Day-of-Week Encoding"
    ],
    bg_color="#e8eaf6", border_color="#3f51b5", title_color="#1a237e", title_size=11
)

# Flow split arrows from Input (x end at 4.1 to give clear gap)
draw_arrow(ax, (2.9, 6.2), (4.1, 6.8), color="#e65100", lw=2.5)
ax.text(3.35, 6.75, "Trend Path", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#e65100", zorder=5)

draw_arrow(ax, (2.9, 4.6), (4.1, 2.5), color="#00695c", lw=2.5)
ax.text(3.35, 3.3, "Seasonal Path", ha="center", va="top", fontsize=9.5, fontweight="bold", color="#00695c", zorder=5)

# --- 3. UPPER PATHWAY: MULTI-KERNEL TREND STREAM ---
draw_card(
    ax, 4.1, 5.5, 3.3, 2.6,
    "Multi-Kernel Avg Pooling",
    [
        "Trend Kernels: k ∈ {3, 5, 7}",
        "X_trend = ∑ w_k · AvgPool_k(X)",
        "",
        "Captures Short & Multi-Day",
        "Demand Velocity"
    ],
    bg_color="#fff3e0", border_color="#e65100", title_color="#e65100"
)

draw_arrow(ax, (7.4, 6.8), (8.0, 6.8), color="#e65100", lw=2.2)

draw_card(
    ax, 8.0, 5.5, 3.3, 2.6,
    "Trend GELU Bottleneck",
    [
        "Non-Linear Activation",
        "H_trend = GELU(W_t1 · X_trend)",
        "",
        "Models Non-Linear",
        "Stock Depletion Rate"
    ],
    bg_color="#ffe0b2", border_color="#f57c00", title_color="#bf360c"
)

# --- 4. LOWER PATHWAY: SEASONAL RESIDUAL STREAM ---
draw_card(
    ax, 4.1, 1.2, 3.3, 2.6,
    "Seasonal Decomposition",
    [
        "Residual High-Frequency Shocks",
        "X_seas = X - X_trend",
        "",
        "Isolates Hourly & Intraday",
        "Demand Spikes"
    ],
    bg_color="#e0f2f1", border_color="#00695c", title_color="#00695c"
)

draw_arrow(ax, (7.4, 2.5), (8.0, 2.5), color="#00695c", lw=2.2)

draw_card(
    ax, 8.0, 1.2, 3.3, 2.6,
    "Seasonal GELU Bottleneck",
    [
        "Non-Linear Activation",
        "H_seas = GELU(W_s1 · X_seas)",
        "",
        "Preserves Complex",
        "Intraday Shocks"
    ],
    bg_color="#b2dfdb", border_color="#00796b", title_color="#004d40"
)

# Merge arrows into 24 Hourly Slot Projections
draw_arrow(ax, (11.3, 6.8), (12.0, 5.6), color="#e65100", lw=2.2)
draw_arrow(ax, (11.3, 2.5), (12.0, 3.8), color="#00695c", lw=2.2)

# --- 5. 24 DEDICATED HOURLY SLOT HEADS ---
draw_card(
    ax, 12.0, 3.0, 3.6, 3.3,
    "24 Hourly Slot Heads",
    [
        "W_1, W_2, ..., W_24 Projections",
        "Logits_h = W_t,h H_t + W_s,h H_s",
        "",
        "• Dedicated weights per hour h",
        "• Eliminates inter-hour",
        "  gradient noise & conflict",
        "• Rush hour specialization"
    ],
    bg_color="#f3e5f5", border_color="#7b1fa2", title_color="#4a148c", title_size=10.5, text_size=8.5
)

draw_arrow(ax, (13.8, 3.0), (13.8, 1.9), color="#4a148c", lw=2.5)

# --- 6. OUTPUT PREDICTION VECTOR ---
draw_card(
    ax, 12.0, 0.3, 3.6, 1.6,
    "24-Hour Stockout Predictions",
    [
        "Y_hat = Sigmoid(Logits) ∈ [0, 1]^24",
        "Hourly Binary Stockout Probabilities",
        "(6 AM to 10 PM Operational Horizon)"
    ],
    bg_color="#e8f5e9", border_color="#2e7d32", title_color="#1b5e20", title_size=10, text_size=8.2
)

plt.tight_layout()

for out_path in ["docs/images/dlinear_architecture_diagram.png", "images/dlinear_architecture_diagram.png"]:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")

plt.close()
print("Successfully regenerated perfectly aligned DLinear architecture diagram.")
