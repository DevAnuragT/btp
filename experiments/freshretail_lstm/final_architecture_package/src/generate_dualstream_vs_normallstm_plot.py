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

# Data: Dual-Stream LSTM models compared against the Best Normal LSTM Baseline
# (Removed Hierarchical Dual-Stream ResNet and Compact Decoupled Dual-Stream BiLSTM)
data = [
    {"architecture": "Best Normal LSTM (h=32 Baseline)", "type": "Normal LSTM (Baseline)", "parameters": 5912, "f1_score": 0.7897, "gain": "+0.0000"},
    {"architecture": "Dual-Stream Standard LSTM", "type": "Dual-Stream Architecture", "parameters": 4536, "f1_score": 0.7879, "gain": "-0.0018"},
    {"architecture": "Dual-Stream ResNet Shortcut", "type": "Dual-Stream Architecture", "parameters": 4728, "f1_score": 0.7914, "gain": "+0.0017"},
    {"architecture": "Dual-Stream Inventory Shortcut", "type": "Dual-Stream Architecture (Top)", "parameters": 4600, "f1_score": 0.7926, "gain": "+0.0029"}
]

df = pd.DataFrame(data)

fig, ax = plt.subplots(figsize=(9, 6))

bar_colors = [
    "#78909c", # Muted Gray for Baseline Normal LSTM
    "#42a5f5", # Light Blue for Basic Dual Stream
    "#26a69a", # Teal for Dual Stream Shortcut
    "#1e88e5"  # Vivid Blue for Top Dual Stream Inventory Shortcut
]

bars = ax.bar(df["architecture"], df["f1_score"], color=bar_colors, width=0.48, edgecolor="black", linewidth=1.1, zorder=3)

# Add baseline horizontal line at 0.7897 (Best Normal LSTM)
ax.axhline(0.7897, color="#d32f2f", linestyle="--", linewidth=2.0, label="Best Normal LSTM Baseline (F1 = 0.7897)", zorder=4)

for bar, (_, row) in zip(bars, df.iterrows()):
    y_val = row["f1_score"]
    gain = row["gain"]
    
    if row["type"] == "Normal LSTM (Baseline)":
        text_str = f"F1 = {y_val:.4f}\n(Baseline)"
        color = "#37474f"
    else:
        text_str = f"F1 = {y_val:.4f}\n({gain})"
        color = "#0d47a1" if y_val == 0.7926 else "#1b5e20"
        
    ax.text(
        bar.get_x() + bar.get_width()/2.0,
        y_val + 0.0006,
        text_str,
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color=color
    )

ax.set_ylim(0.780, 0.800)
ax.set_ylabel("Hour-Level F1-Score (Primary Metric)", fontweight="bold", labelpad=10)
ax.set_title("Architectural Impact: Dual-Stream LSTMs vs Best Normal LSTM Baseline", fontweight="bold", pad=14)

plt.xticks(rotation=10, ha="right", fontweight="bold", fontsize=9.5)

# Callout annotation for Dual-Stream Inventory Shortcut
ax.annotate(
    "Dual-Stream Feature Decoupling Gain\nF1 = 0.7926 (+0.0029 Gain | 4.6k Params)",
    xy=(3, 0.7926),
    xytext=(1.2, 0.7965),
    fontsize=9,
    fontweight="bold",
    color="#0d47a1",
    arrowprops=dict(arrowstyle="->", color="#0d47a1", lw=1.4, connectionstyle="arc3,rad=-0.1"),
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#e3f2fd", edgecolor="#0d47a1", lw=1.0),
    zorder=5
)

ax.legend(loc="upper left", frameon=True, framealpha=0.95, facecolor="white", edgecolor="#cccccc")
plt.tight_layout()

out_plot_path = "docs/images/dualstream_vs_normal_lstm_comparison.png"
plt.savefig(out_plot_path, dpi=300)
plt.close()

print(f"\nSuccessfully updated Dual-Stream comparison plot (removed 2 models) at: {out_plot_path}")
df.to_csv("experiments/freshretail_lstm/final_architecture_package/dualstream_vs_normal_lstm_summary.csv", index=False)
