import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set clean styling parameters
plt.rcParams.update({
    'font.sans-serif': 'DejaVu Sans',
    'font.family': 'sans-serif',
    'savefig.dpi': 300
})

fig, ax = plt.subplots(figsize=(13, 8))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis("off")

# Helper function to draw rounded boxes
def draw_box(ax, x, y, w, h, title, subtitle="", bg_color="#e3f2fd", border_color="#0d47a1", title_color="#0d47a1", fontsize=10):
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
        ax.text(x + w/2, y + h/2 - 0.25, subtitle, ha="center", va="center", fontsize=fontsize-1.5, color="#37474f", zorder=4)
    else:
        ax.text(x + w/2, y + h/2, title, ha="center", va="center", fontsize=fontsize, fontweight="bold", color=title_color, zorder=4)

# Helper function to draw arrows
def draw_arrow(ax, x1, y1, x2, y2, label="", color="#37474f", lw=1.5, style="->"):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle=style, color=color, lw=lw, mutation_scale=12),
        zorder=2
    )
    if label:
        mid_x, mid_y = (x1 + x2)/2, (y1 + y2)/2
        ax.text(mid_x, mid_y + 0.2, label, ha="center", va="center", fontsize=8.5, fontweight="bold", color="#1a237e", zorder=5)

# --- 1. TITLE & SUBTITLE ---
ax.text(8, 9.5, "Detailed Plain Standard LSTM Architecture Diagram", ha="center", va="center", fontsize=14, fontweight="bold", color="#0d47a1")
ax.text(8, 9.15, "Next-Day 24-Hour Stock Status Prediction (FreshRetailNet-50K)", ha="center", va="center", fontsize=10.5, style="italic", color="#455a64")

# --- 2. INPUT SEQUENCE STAGE ---
draw_box(ax, 0.5, 5.8, 2.5, 2.2, "Input Feature Matrix X", "L=14 Days × d_in=13 Feats\n[Sales, Stockout, Discount...]", bg_color="#e8eaf6", border_color="#303f9f", title_color="#1a237e")

draw_arrow(ax, 3.25, 6.9, 4.2, 6.9, "x_1, ..., x_t, ..., x_14")

# --- 3. RECURRENT LSTM UNROLLING STAGE ---
# Draw sequence steps t-1, t, 14
draw_box(ax, 4.4, 6.0, 1.8, 1.8, "LSTM Cell (t-1)", "h_{t-1}, C_{t-1}", bg_color="#e3f2fd", border_color="#1565c0", title_color="#0d47a1")
draw_arrow(ax, 6.45, 6.9, 7.35, 6.9, "h_{t-1}")

draw_box(ax, 7.6, 6.0, 1.8, 1.8, "LSTM Cell (t)", "h_t, C_t", bg_color="#bbdefb", border_color="#0d47a1", title_color="#0d47a1")
draw_arrow(ax, 9.65, 6.9, 10.55, 6.9, "h_t")

draw_box(ax, 10.8, 6.0, 1.8, 1.8, "LSTM Cell (L=14)", "Final State h_14", bg_color="#90caf9", border_color="#0d47a1", title_color="#0d47a1")

# --- 4. DETAILED LSTM CELL INTERNAL GATING DIAGRAM (EXPANDED AT BOTTOM) ---
cell_box = patches.FancyBboxPatch(
    (3.5, 1.0), 9.0, 4.0,
    boxstyle="round,pad=0.3",
    facecolor="#ffffff",
    edgecolor="#0d47a1",
    linewidth=2.0,
    linestyle="--",
    zorder=1
)
ax.add_patch(cell_box)
ax.text(8.0, 4.7, "Internal Mechanics of Plain Standard LSTM Cell (at step t)", ha="center", va="center", fontsize=11, fontweight="bold", color="#0d47a1")

# Internal Gates Inside the Expanded Box
draw_box(ax, 3.8, 2.8, 1.7, 1.3, "Forget Gate f_t", "f_t = σ(W_f · [h_{t-1}, x_t] + b_f)", bg_color="#ffe0b2", border_color="#e65100", title_color="#e65100", fontsize=8.5)
draw_box(ax, 5.8, 2.8, 1.7, 1.3, "Input Gate i_t", "i_t = σ(W_i · [h_{t-1}, x_t] + b_i)", bg_color="#c8e6c9", border_color="#2e7d32", title_color="#2e7d32", fontsize=8.5)
draw_box(ax, 7.8, 2.8, 1.8, 1.3, "Candidate C~_t", "C~_t = tanh(W_c · [h_{t-1}, x_t])", bg_color="#d1c4e9", border_color="#4a148c", title_color="#4a148c", fontsize=8.5)
draw_box(ax, 9.9, 2.8, 1.7, 1.3, "Output Gate o_t", "o_t = σ(W_o · [h_{t-1}, x_t] + b_o)", bg_color="#ffecb3", border_color="#ff6f00", title_color="#ff6f00", fontsize=8.5)

# State Updates at bottom of Cell
draw_box(ax, 4.8, 1.3, 3.0, 1.1, "Cell State Update C_t", "C_t = f_t ⊙ C_{t-1} + i_t ⊙ C~_t", bg_color="#e0f2f1", border_color="#00695c", title_color="#00695c", fontsize=8.5)
draw_box(ax, 8.2, 1.3, 3.0, 1.1, "Hidden State Update h_t", "h_t = o_t ⊙ tanh(C_t)", bg_color="#e1f5fe", border_color="#0277bd", title_color="#0277bd", fontsize=8.5)

# Connect top cell (t) to expanded mechanics box
draw_arrow(ax, 8.5, 5.9, 8.5, 5.1, style="-|>", color="#0d47a1", lw=1.5)

# --- 5. LINEAR PROJECTION HEAD & OUTPUT STAGE ---
draw_arrow(ax, 12.85, 6.9, 13.5, 6.9, "h_14 ∈ R^96")

draw_box(ax, 13.6, 5.8, 2.0, 2.2, "Linear Projection", "W_out ∈ R^{96 × 24}\nb_out ∈ R^{24}", bg_color="#fff9c4", border_color="#f57f17", title_color="#f57f17")

draw_arrow(ax, 14.6, 5.7, 14.6, 4.5, "Logits z ∈ R^24")

draw_box(ax, 13.3, 3.3, 2.6, 1.0, "Sigmoid Activation σ(z)", "Y_hat ∈ [0, 1]^{24} Probabilities", bg_color="#dcedc8", border_color="#33691e", title_color="#33691e")

draw_arrow(ax, 14.6, 3.2, 14.6, 2.2)

draw_box(ax, 13.3, 0.9, 2.6, 1.2, "Target Vector Y", "24 Binary Hourly Stockouts\n(6 AM to 10 PM)", bg_color="#c8e6c9", border_color="#1b5e20", title_color="#1b5e20")

# Highlight Summary Legend at Top Right
rect_sum = patches.FancyBboxPatch(
    (0.5, 0.9), 2.6, 3.8,
    boxstyle="round,pad=0.25",
    facecolor="#f5f5f5",
    edgecolor="#9e9e9e",
    linewidth=1.2,
    zorder=2
)
ax.add_patch(rect_sum)
ax.text(1.8, 4.4, "Plain LSTM Spec", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#212121")
summary_text = (
    "• Hidden Size h: 96 or 32\n"
    "• Sequence Window L: 14 Days\n"
    "• Input Feats d_in: 13 or 6\n"
    "• Parameters: 44,952 (h=96)\n"
    "• Peak F1-Score: 0.7852\n"
    "• Output: 24 Binary Hours\n"
    "• Loss: BCEWithLogitsLoss"
)
ax.text(1.8, 2.6, summary_text, ha="center", va="center", fontsize=8, color="#424242", linespacing=1.6)

out_path = "docs/images/plain_lstm_architecture_diagram.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"\nSuccessfully generated detailed plain LSTM architecture diagram at: {out_path}")
