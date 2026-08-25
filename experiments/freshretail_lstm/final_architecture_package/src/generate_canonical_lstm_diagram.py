import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set clean styling parameters with high legibility
plt.rcParams.update({
    'font.sans-serif': 'DejaVu Sans',
    'font.family': 'sans-serif',
    'savefig.dpi': 300
})

fig, ax = plt.subplots(figsize=(15, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis("off")

# Helper function to draw rounded layer boxes
def draw_layer_box(ax, x, y, w, h, text, bg_color="#fff59d", border_color="#fbc02d", text_color="#212121", fontsize=9.5):
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.15",
        facecolor=bg_color,
        edgecolor=border_color,
        linewidth=1.5,
        zorder=4
    )
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize, fontweight="bold", color=text_color, zorder=5)

# Helper function to draw point-wise operation circles
def draw_op_circle(ax, x, y, symbol, bg_color="#ff80ab", border_color="#c2185b", text_color="#ffffff", radius=0.32):
    circle = plt.Circle((x, y), radius, facecolor=bg_color, edgecolor=border_color, linewidth=1.5, zorder=4)
    ax.add_patch(circle)
    ax.text(x, y, symbol, ha="center", va="center", fontsize=11, fontweight="bold", color=text_color, zorder=5)

# Helper function for data routing lines with optional arrow
def draw_line(ax, points, color="#37474f", lw=2.0, style="-", arrow=False):
    x_val = [p[0] for p in points]
    y_val = [p[1] for p in points]
    ax.plot(x_val, y_val, color=color, linewidth=lw, linestyle=style, zorder=2)
    if arrow:
        x1, y1 = points[-2]
        x2, y2 = points[-1]
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color=color, lw=lw, mutation_scale=14),
            zorder=3
        )

# --- 1. TITLE & SUBTITLE ---
ax.text(8, 9.6, "Standard Canonical Plain LSTM Cell Architecture", ha="center", va="center", fontsize=15, fontweight="bold", color="#0d47a1")
ax.text(8, 9.25, "Olah / PyTorch Textbook Vector Routing for 24-Hour Stock Status Prediction", ha="center", va="center", fontsize=10.5, style="italic", color="#455a64")

# --- 2. MAIN LSTM CELL CONTAINER BOX ---
cell_bg = patches.FancyBboxPatch(
    (2.8, 1.6), 9.6, 6.4,
    boxstyle="round,pad=0.4",
    facecolor="#f1f8e9",
    edgecolor="#33691e",
    linewidth=2.2,
    zorder=1
)
ax.add_patch(cell_bg)
ax.text(7.6, 7.6, "Standard LSTM Cell (Time Step t)", ha="center", va="center", fontsize=11, fontweight="bold", color="#33691e")

# --- 3. MAIN FLOW LINES (Cell State Top & Hidden State Bottom) ---
# A. Top Cell State Line: C_{t-1} -> C_t (y = 7.0)
draw_line(ax, [(0.5, 7.0), (14.5, 7.0)], color="#c2185b", lw=2.5, arrow=True)
ax.text(1.1, 7.4, "Cell State\nC_{t-1}", ha="center", va="center", fontsize=9, fontweight="bold", color="#c2185b")
ax.text(13.9, 7.4, "Cell State\nC_t", ha="center", va="center", fontsize=9, fontweight="bold", color="#c2185b")

# B. Bottom Hidden State Line: h_{t-1} -> h_t (y = 2.2)
draw_line(ax, [(0.5, 2.2), (14.5, 2.2)], color="#0d47a1", lw=2.5, arrow=True)
ax.text(1.1, 1.7, "Hidden State\nh_{t-1}", ha="center", va="center", fontsize=9, fontweight="bold", color="#0d47a1")
ax.text(13.9, 1.7, "Hidden State\nh_t", ha="center", va="center", fontsize=9, fontweight="bold", color="#0d47a1")

# C. Vertical Input Line: x_t -> Concatenation
draw_line(ax, [(1.8, 0.7), (1.8, 2.2)], color="#2e7d32", lw=2.5, arrow=True)
ax.text(1.8, 0.35, "Input Day x_t\n(13 Features)", ha="center", va="center", fontsize=9, fontweight="bold", color="#2e7d32")

# --- 4. POINT-WISE OPERATIONS ON CELL STATE LINE ---
# Multiplier 1 (Forget Gate) at x=4.2, y=7.0
draw_op_circle(ax, 4.2, 7.0, "×", bg_color="#ff80ab", border_color="#c2185b")
# Adder (Input Gate update) at x=8.2, y=7.0
draw_op_circle(ax, 8.2, 7.0, "+", bg_color="#80d8ff", border_color="#0091ea")

# tanh Output Gate scaling box at x=11.2, y=5.3
draw_layer_box(ax, 10.6, 5.0, 1.2, 0.6, "tanh", bg_color="#80deea", border_color="#00838f", text_color="#004d40", fontsize=9)
draw_line(ax, [(11.2, 7.0), (11.2, 5.6)], color="#c2185b", lw=1.8, arrow=True)

# Multiplier 2 (Output Gate Scaling) at x=11.2, y=3.8
draw_op_circle(ax, 11.2, 3.8, "×", bg_color="#ff80ab", border_color="#c2185b")
draw_line(ax, [(11.2, 5.0), (11.2, 4.12)], color="#00838f", lw=1.8, arrow=True)
draw_line(ax, [(11.2, 3.48), (11.2, 2.2)], color="#0d47a1", lw=2.0)

# --- 5. FOUR INTERNAL PARALLEL GATING COLUMNS ---
# Column 1: Forget Gate (x=4.2, y=3.0)
draw_layer_box(ax, 3.7, 2.7, 1.0, 0.65, "σ", bg_color="#fff59d", border_color="#fbc02d")
ax.text(4.2, 4.0, "Forget Gate\n(f_t)", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#f57f17")
draw_line(ax, [(4.2, 3.35), (4.2, 6.68)], color="#f57f17", lw=1.8, arrow=True)

# Column 2: Input Gate (x=6.2, y=3.0)
draw_layer_box(ax, 5.7, 2.7, 1.0, 0.65, "σ", bg_color="#fff59d", border_color="#fbc02d")
ax.text(6.2, 4.0, "Input Gate\n(i_t)", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#f57f17")

# Column 3: Candidate Cell State (x=8.2, y=3.0)
draw_layer_box(ax, 7.7, 2.7, 1.0, 0.65, "tanh", bg_color="#80deea", border_color="#00838f", text_color="#004d40", fontsize=8.5)
ax.text(8.2, 4.0, "Candidate\n(C~_t)", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#00695c")

# Multiplier for Candidate & Input Gate at x=7.2, y=5.3
draw_op_circle(ax, 7.2, 5.3, "×", bg_color="#ff80ab", border_color="#c2185b", radius=0.3)
draw_line(ax, [(6.2, 3.35), (6.2, 5.3), (6.9, 5.3)], color="#f57f17", lw=1.8, arrow=True)
draw_line(ax, [(8.2, 3.35), (8.2, 5.3), (7.5, 5.3)], color="#00695c", lw=1.8, arrow=True)
draw_line(ax, [(7.2, 5.6), (7.2, 7.0), (7.88, 7.0)], color="#c2185b", lw=1.8, arrow=True)

# Column 4: Output Gate (x=10.0, y=3.0)
draw_layer_box(ax, 9.5, 2.7, 1.0, 0.65, "σ", bg_color="#fff59d", border_color="#fbc02d")
ax.text(10.0, 4.0, "Output Gate\n(o_t)", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#f57f17")
draw_line(ax, [(10.0, 3.35), (10.0, 3.8), (10.88, 3.8)], color="#f57f17", lw=1.8, arrow=True)

# Connect Concatenated Vector [h_{t-1}, x_t] to all 4 gate columns
draw_line(ax, [(2.8, 2.2), (4.2, 2.2), (4.2, 2.7)], color="#0d47a1", lw=1.8, arrow=True)
draw_line(ax, [(4.2, 2.2), (6.2, 2.2), (6.2, 2.7)], color="#0d47a1", lw=1.8, arrow=True)
draw_line(ax, [(6.2, 2.2), (8.2, 2.2), (8.2, 2.7)], color="#0d47a1", lw=1.8, arrow=True)
draw_line(ax, [(8.2, 2.2), (10.0, 2.2), (10.0, 2.7)], color="#0d47a1", lw=1.8, arrow=True)

# --- 6. 24-HOUR LINEAR PROJECTION HEAD (RIGHT SIDE OUTSIDE CELL) ---
draw_line(ax, [(14.5, 2.2), (15.1, 2.2), (15.1, 4.2)], color="#0d47a1", lw=2.0, arrow=True)

draw_layer_box(ax, 14.1, 4.2, 1.8, 1.0, "Linear Head\nW_out ∈ R^{96×24}", bg_color="#fff3e0", border_color="#e65100", text_color="#e65100", fontsize=8.5)

draw_line(ax, [(15.0, 5.2), (15.0, 5.9)], color="#e65100", lw=1.8, arrow=True)

draw_layer_box(ax, 14.1, 5.9, 1.8, 0.7, "Sigmoid σ\nProbabilities", bg_color="#e8f5e9", border_color="#2e7d32", text_color="#1b5e20", fontsize=8.5)

draw_line(ax, [(15.0, 6.6), (15.0, 7.3)], color="#1b5e20", lw=1.8, arrow=True)

draw_layer_box(ax, 13.9, 7.3, 2.0, 1.2, "Target Y ∈ [0, 1]^{24}\n(24 Binary Hours)", bg_color="#c8e6c9", border_color="#1b5e20", text_color="#1b5e20", fontsize=8.5)

# --- 7. LEGEND BOX FOR TEXTBOOK SYMBOLS (TOP-LEFT) ---
legend_box = patches.FancyBboxPatch(
    (0.4, 7.8), 2.2, 1.5,
    boxstyle="round,pad=0.15",
    facecolor="#ffffff",
    edgecolor="#bdbdbd",
    linewidth=1.2
)
ax.add_patch(legend_box)
ax.text(1.5, 9.0, "Textbook Legend", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#424242")
ax.text(1.5, 8.4, "σ : Sigmoid Layer\ntanh : Tanh Layer\n× : Pointwise Mult\n+ : Pointwise Add", ha="center", va="center", fontsize=7.5, color="#616161", linespacing=1.35)

plt.tight_layout()

out_path = "docs/images/plain_lstm_architecture_diagram.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"\nSuccessfully generated perfectly non-overlapping canonical LSTM diagram at: {out_path}")
