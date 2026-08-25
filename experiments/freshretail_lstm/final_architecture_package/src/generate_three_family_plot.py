import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'font.sans-serif': 'DejaVu Sans',
    'font.family': 'sans-serif',
    'figure.titlesize': 13,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'savefig.dpi': 300
})

combined_family_data = [
    # --- Normal LSTM Family ---
    {"model_name": "Ultra Minimal LSTM (h=8)", "family": "Normal LSTM Family", "parameters": 728, "f1_score": 0.6468},
    {"model_name": "Ultra Minimal LSTM (h=16)", "family": "Normal LSTM Family", "parameters": 1944, "f1_score": 0.7407},
    {"model_name": "Ultra Short-Window LSTM (seq=7)", "family": "Normal LSTM Family", "parameters": 1944, "f1_score": 0.7593},
    {"model_name": "Compact CIFG LSTM (h=32)", "family": "Normal LSTM Family", "parameters": 4536, "f1_score": 0.7650},
    {"model_name": "Compact Standard LSTM (h=32)", "family": "Normal LSTM Family", "parameters": 6808, "f1_score": 0.7526},
    {"model_name": "Compact Standard LSTM (h=64)", "family": "Normal LSTM Family", "parameters": 21784, "f1_score": 0.7712},
    {"model_name": "Short-Window Standard LSTM (seq=7)", "family": "Normal LSTM Family", "parameters": 21784, "f1_score": 0.7761},
    {"model_name": "Baseline Standard LSTM (h=96)", "family": "Normal LSTM Family", "parameters": 44952, "f1_score": 0.7852},
    {"model_name": "Feature-Reduced Standard LSTM (h=64)", "family": "Normal LSTM Family", "parameters": 19992, "f1_score": 0.7880},
    {"model_name": "Compact Feature-Reduced LSTM (h=32)", "family": "Normal LSTM Family", "parameters": 5912, "f1_score": 0.7897},

    # --- RNN Family ---
    {"model_name": "Deep Multi-Layer RNN (h=96)", "family": "RNN Family", "parameters": 68424, "f1_score": 0.7480},
    {"model_name": "1D Conv-BiGRU Self-Attn", "family": "RNN Family", "parameters": 52056, "f1_score": 0.7620},
    {"model_name": "Vanilla Simple RNN (h=64)", "family": "RNN Family", "parameters": 14744, "f1_score": 0.7750},

    # --- Transformer Family ---
    {"model_name": "TFT-Lite VSN Transformer", "family": "Transformer Family", "parameters": 288390, "f1_score": 0.7490},
    {"model_name": "PatchTST Linear Transformer", "family": "Transformer Family", "parameters": 285624, "f1_score": 0.7540},
    {"model_name": "Hourly Query Transformer", "family": "Transformer Family", "parameters": 281345, "f1_score": 0.7685}
]

df_comb = pd.DataFrame(combined_family_data)

fig, ax = plt.subplots(figsize=(10.5, 6.2))

family_palette = {
    "Normal LSTM Family": "#1f77b4",   # Vivid Blue for Normal LSTM
    "Transformer Family": "#ff7f0e",   # Bright Orange for Transformers
    "RNN Family": "#2ca02c"            # Forest Green for RNNs
}

family_markers = {
    "Normal LSTM Family": "o",
    "Transformer Family": "s",
    "RNN Family": "^"
}

# Plot clean scatter points
sns.scatterplot(
    data=df_comb,
    x="parameters",
    y="f1_score",
    hue="family",
    style="family",
    markers=family_markers,
    s=190,
    palette=family_palette,
    edgecolor="white",
    linewidth=1.2,
    ax=ax,
    zorder=4
)

# Connect model points per family with clean dashed trend lines
for family, color in family_palette.items():
    sub = df_comb[df_comb["family"] == family].sort_values("parameters")
    ax.plot(sub["parameters"], sub["f1_score"], linestyle="--", color=color, alpha=0.7, linewidth=2.0, zorder=3)

# Add clean horizontal reference lines for family peak scores
ax.axhline(0.7897, color="#1f77b4", linestyle=":", linewidth=1.4, alpha=0.8)
ax.axhline(0.7750, color="#2ca02c", linestyle=":", linewidth=1.2, alpha=0.7)
ax.axhline(0.7685, color="#ff7f0e", linestyle=":", linewidth=1.2, alpha=0.7)

# ---------------------------------------------------------
# EXPLICIT NON-OVERLAPPING CALLOUT ANNOTATIONS
# Positioned in dedicated empty whitespace areas
# ---------------------------------------------------------

# 1. Peak Normal LSTM (Positioned in Top-Left empty space: x=800, y=0.803)
ax.annotate(
    "Peak Normal LSTM (h=32)\nF1 = 0.7897 (5.9k Params)",
    xy=(5912, 0.7897),
    xytext=(750, 0.803),
    fontsize=9.5,
    fontweight="bold",
    color="#0d47a1",
    arrowprops=dict(arrowstyle="->", color="#0d47a1", lw=1.5, connectionstyle="arc3,rad=-0.1"),
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#e3f2fd", edgecolor="#0d47a1", lw=1)
)

# 2. Peak RNN (Positioned in Top-Right empty space: x=30000, y=0.803)
ax.annotate(
    "Vanilla Simple RNN (h=64)\nF1 = 0.7750 (14.7k Params)",
    xy=(14744, 0.7750),
    xytext=(25000, 0.803),
    fontsize=8.5,
    fontweight="bold",
    color="#1b5e20",
    arrowprops=dict(arrowstyle="->", color="#1b5e20", lw=1.2, connectionstyle="arc3,rad=0.1"),
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#e8f5e9", edgecolor="#1b5e20", lw=0.8)
)

# 3. Peak Transformer (Positioned in Bottom-Right empty space: x=20000, y=0.680)
ax.annotate(
    "Hourly Query Transformer\nF1 = 0.7685 (281k Params)",
    xy=(281345, 0.7685),
    xytext=(25000, 0.680),
    fontsize=8.5,
    fontweight="bold",
    color="#e65100",
    arrowprops=dict(arrowstyle="->", color="#e65100", lw=1.2, connectionstyle="arc3,rad=-0.15"),
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff3e0", edgecolor="#e65100", lw=0.8)
)

ax.set_xscale("log")
ax.set_xlim(500, 450000)
ax.set_ylim(0.62, 0.825)

ax.set_xlabel("Total Trainable Parameters (Log Scale)", fontweight="bold", labelpad=8)
ax.set_ylabel("Hour-Level F1-Score (Primary Metric)", fontweight="bold", labelpad=8)
ax.set_title("Hourly Stockout Prediction: Parameter-Efficient Normal LSTM Outperforms Transformers & RNNs", fontweight="bold", pad=12)

# Place legend in Bottom-Left empty space (x=700, y=0.65 zone)
ax.legend(title="Architecture Family", loc="lower left", frameon=True, framealpha=0.95, facecolor="white", edgecolor="#cccccc")

plt.tight_layout()

out_plot_path = "docs/images/hourly_f1_vs_params_by_family.png"
plt.savefig(out_plot_path, dpi=300)
plt.close()

# Also copy clean version to normal_lstm_variations_comparison.png
plt.figure(figsize=(10.5, 6.2))
ax2 = sns.scatterplot(
    data=df_comb,
    x="parameters",
    y="f1_score",
    hue="family",
    style="family",
    markers=family_markers,
    s=190,
    palette=family_palette,
    edgecolor="white",
    linewidth=1.2,
    zorder=4
)
for family, color in family_palette.items():
    sub = df_comb[df_comb["family"] == family].sort_values("parameters")
    ax2.plot(sub["parameters"], sub["f1_score"], linestyle="--", color=color, alpha=0.7, linewidth=2.0, zorder=3)
ax2.axhline(0.7897, color="#1f77b4", linestyle=":", linewidth=1.4, alpha=0.8)
ax2.annotate(
    "Peak Normal LSTM (h=32)\nF1 = 0.7897 (5.9k Params)",
    xy=(5912, 0.7897),
    xytext=(750, 0.803),
    fontsize=9.5,
    fontweight="bold",
    color="#0d47a1",
    arrowprops=dict(arrowstyle="->", color="#0d47a1", lw=1.5, connectionstyle="arc3,rad=-0.1"),
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#e3f2fd", edgecolor="#0d47a1", lw=1)
)
ax2.set_xscale("log")
ax2.set_xlim(500, 450000)
ax2.set_ylim(0.62, 0.825)
ax2.set_xlabel("Total Trainable Parameters (Log Scale)", fontweight="bold", labelpad=8)
ax2.set_ylabel("Hour-Level F1-Score (Primary Metric)", fontweight="bold", labelpad=8)
ax2.set_title("Hourly Stockout Prediction: Parameter-Efficient Normal LSTM Outperforms Transformers & RNNs", fontweight="bold", pad=12)
plt.legend(title="Architecture Family", loc="lower left", frameon=True, framealpha=0.95, facecolor="white", edgecolor="#cccccc")
plt.tight_layout()
plt.savefig("docs/images/normal_lstm_variations_comparison.png", dpi=300)
plt.close()

print(f"\nSuccessfully generated perfectly non-overlapping plot at: {out_plot_path}")
