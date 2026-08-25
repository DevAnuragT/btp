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

# Data: 4 LSTM variations + DLinear models compared against Best Normal LSTM Baseline (0.7897)
data = [
    {"architecture": "Dual-Stream Standard LSTM", "category": "LSTM Family", "parameters": 4536, "f1_score": 0.7879, "gain": "-0.0018"},
    {"architecture": "Best Normal LSTM (h=32 Baseline)", "category": "Normal LSTM Baseline", "parameters": 5912, "f1_score": 0.7897, "gain": "Baseline"},
    {"architecture": "Dual-Stream ResNet Shortcut", "category": "LSTM Family", "parameters": 4728, "f1_score": 0.7914, "gain": "+0.0017"},
    {"architecture": "Dual-Stream Inventory Shortcut", "category": "LSTM Family", "parameters": 4600, "f1_score": 0.7926, "gain": "+0.0029"},
    {"architecture": "Standard DLinear", "category": "Linear Family", "parameters": 6768, "f1_score": 0.7932, "gain": "+0.0035"},
    {"architecture": "Hourly Slot-Specific DLinear 🏆", "category": "Linear Family (24 Heads)", "parameters": 6744, "f1_score": 0.8058, "gain": "+0.0161"},
    {"architecture": "Multi-Kernel DLinear", "category": "Linear Family (Multi-Scale)", "parameters": 13539, "f1_score": 0.8131, "gain": "+0.0234"}
]

df = pd.DataFrame(data)

fig, ax = plt.subplots(figsize=(11, 6.5))

bar_colors = [
    "#42a5f5", # Light Blue (Dual-Stream Standard)
    "#78909c", # Muted Gray (Best Normal LSTM Baseline)
    "#26a69a", # Teal (Dual-Stream ResNet)
    "#1e88e5", # Vivid Blue (Dual-Stream Inventory)
    "#ff9800", # Orange (Standard DLinear)
    "#e65100", # Deep Amber/Orange (Hourly Slot DLinear)
    "#bf360c"  # Dark Red/Orange (Multi-Kernel DLinear)
]

bars = ax.bar(df["architecture"], df["f1_score"], color=bar_colors, width=0.55, edgecolor="black", linewidth=1.1, zorder=3)

# Horizontal dashed baseline STRICTLY at Best Normal LSTM Baseline (0.7897)
ax.axhline(0.7897, color="#d32f2f", linestyle="--", linewidth=2.2, label="Best Normal LSTM Baseline (F1 = 0.7897)", zorder=4)

for bar, (_, row) in zip(bars, df.iterrows()):
    y_val = row["f1_score"]
    gain = row["gain"]
    params = row["parameters"]
    
    if row["architecture"] == "Best Normal LSTM (h=32 Baseline)":
        text_str = f"F1 = {y_val:.4f}\n(BASELINE)"
        color = "#37474f"
        fontweight = "bold"
    else:
        text_str = f"F1 = {y_val:.4f}\n({gain})"
        color = "#bf360c" if "Multi-Kernel" in row["architecture"] else ("#e65100" if "DLinear" in row["architecture"] else "#0d47a1")
        fontweight = "bold"
        
    ax.text(
        bar.get_x() + bar.get_width()/2.0,
        y_val + 0.0006,
        text_str,
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight=fontweight,
        color=color
    )

ax.set_ylim(0.780, 0.825)
ax.set_ylabel("Hour-Level F1-Score (Primary Metric)", fontweight="bold", labelpad=10)
ax.set_title("Empirical Benchmark: DLinear Models & LSTM Variants vs Best Normal LSTM Baseline (F1 = 0.7897)\n(Hourly Slot-Specific DLinear Gains +0.0161 F1 Boost Over Best Normal LSTM with 24 Dedicated Heads)", fontweight="bold", pad=14)

plt.xticks(rotation=15, ha="right", fontweight="bold", fontsize=9.5)

# Callout annotation for Hourly Slot-Specific DLinear vs Best Normal LSTM Baseline
ax.annotate(
    "Hourly Slot-Specific DLinear (+0.0161 F1 Gain vs Baseline)\n24 Dedicated Hourly Linear Heads (W₁, W₂, ..., W₂₄)\nF1 = 0.8058 | 88.02% Recall | 6.7k Params",
    xy=(5, 0.8058),
    xytext=(2.2, 0.817),
    fontsize=9.5,
    fontweight="bold",
    color="#e65100",
    arrowprops=dict(arrowstyle="->", color="#e65100", lw=1.5, connectionstyle="arc3,rad=-0.12"),
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff3e0", edgecolor="#e65100", lw=1.2),
    zorder=5
)

ax.legend(loc="upper left", frameon=True, framealpha=0.95, facecolor="white", edgecolor="#cccccc")
plt.tight_layout()

out_plot_path = "docs/images/lstm_vs_dlinear_comparison.png"
plt.savefig(out_plot_path, dpi=300)
plt.close()

print(f"\nSuccessfully generated plot with Best Normal LSTM as baseline at: {out_plot_path}")
df.to_csv("experiments/freshretail_lstm/final_architecture_package/lstm_vs_dlinear_summary.csv", index=False)
