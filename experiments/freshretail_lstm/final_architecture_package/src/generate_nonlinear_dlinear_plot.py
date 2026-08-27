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

# Benchmark data: Linear DLinear vs Non-Linear DLinear Models
data = [
    {"architecture": "Best Normal LSTM (Baseline)", "category": "LSTM Baseline", "parameters": 5912, "f1_score": 0.7897, "recall": "81.29%", "type": "Baseline"},
    {"architecture": "Standard Linear DLinear", "category": "Linear DLinear", "parameters": 6768, "f1_score": 0.7932, "recall": "87.44%", "type": "Linear"},
    {"architecture": "Non-Linear DLinear (GELU)", "category": "Non-Linear DLinear", "parameters": 21168, "f1_score": 0.8083, "recall": "91.37%", "type": "Non-Linear"},
    {"architecture": "Hourly Slot Linear DLinear", "category": "Linear DLinear", "parameters": 6744, "f1_score": 0.8058, "recall": "88.02%", "type": "Linear"},
    {"architecture": "Hourly Slot Non-Linear DLinear", "category": "Non-Linear DLinear", "parameters": 109104, "f1_score": 0.8050, "recall": "86.70%", "type": "Non-Linear"},
    {"architecture": "Multi-Kernel Linear DLinear", "category": "Linear DLinear", "parameters": 13539, "f1_score": 0.8131, "recall": "85.22%", "type": "Linear"},
    {"architecture": "Multi-Kernel Non-Linear DLinear [Peak]", "category": "Non-Linear DLinear (Peak)", "parameters": 42339, "f1_score": 0.8156, "recall": "87.13%", "type": "Non-Linear (Top)"}
]

df = pd.DataFrame(data)

fig, ax = plt.subplots(figsize=(11, 6.5))

bar_colors = [
    "#78909c", # Baseline Normal LSTM
    "#ff9800", # Standard Linear DLinear
    "#f57c00", # Non-Linear DLinear GELU
    "#e65100", # Hourly Slot Linear DLinear
    "#d84315", # Hourly Slot Non-Linear DLinear
    "#bf360c", # Multi-Kernel Linear DLinear
    "#8d6e63"  # Multi-Kernel Non-Linear DLinear (Top)
]

bars = ax.bar(df["architecture"], df["f1_score"], color=bar_colors, width=0.55, edgecolor="black", linewidth=1.1, zorder=3)

# Horizontal dashed baseline at Best Normal LSTM (0.7897)
ax.axhline(0.7897, color="#d32f2f", linestyle="--", linewidth=2.2, label="Best Normal LSTM Baseline (F1 = 0.7897)", zorder=4)

for bar, (_, row) in zip(bars, df.iterrows()):
    y_val = row["f1_score"]
    rec = row["recall"]
    params = row["parameters"]
    
    if "Multi-Kernel Non-Linear" in row["architecture"]:
        text_str = f"F1 = {y_val:.4f}\n(PEAK | {params:,} p)"
        color = "#bf360c"
        fontweight = "bold"
    elif "GELU" in row["architecture"]:
        text_str = f"F1 = {y_val:.4f}\n({rec} Rec)"
        color = "#e65100"
        fontweight = "bold"
    else:
        text_str = f"F1 = {y_val:.4f}"
        color = "#37474f" if "Baseline" in row["architecture"] else "#e65100"
        fontweight = "bold"
        
    ax.text(
        bar.get_x() + bar.get_width()/2.0,
        y_val + 0.0006,
        text_str,
        ha="center",
        va="bottom",
        fontsize=8.5,
        fontweight=fontweight,
        color=color
    )

ax.set_ylim(0.780, 0.825)
ax.set_ylabel("Hour-Level F1-Score (Primary Metric)", fontweight="bold", labelpad=10)
ax.set_title("Empirical Benchmark: Impact of Non-Linearity on DLinear Models\n(Multi-Kernel Non-Linear DLinear achieves Peak 0.8156 F1 | Non-Linear GELU DLinear reaches 91.37% Recall)", fontweight="bold", pad=14)

plt.xticks(rotation=15, ha="right", fontweight="bold", fontsize=9)

# Callout annotation for Non-Linear GELU DLinear Record Recall
ax.annotate(
    "Non-Linear GELU Bottleneck\n91.37% Stockout Recall (Highest in Repo!)\nF1 = 0.8083 (+0.0151 vs Linear DLinear)",
    xy=(2, 0.8083),
    xytext=(0.8, 0.817),
    fontsize=9,
    fontweight="bold",
    color="#e65100",
    arrowprops=dict(arrowstyle="->", color="#e65100", lw=1.5, connectionstyle="arc3,rad=-0.12"),
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff3e0", edgecolor="#e65100", lw=1.2),
    zorder=5
)

# Callout annotation for Multi-Kernel Non-Linear DLinear Peak F1
ax.annotate(
    "Multi-Kernel Non-Linear DLinear\nNew Peak F1 = 0.8156 | Lowest 3.75h MAE\n(+0.0259 Gain vs Normal LSTM Baseline)",
    xy=(6, 0.8156),
    xytext=(3.5, 0.821),
    fontsize=9,
    fontweight="bold",
    color="#bf360c",
    arrowprops=dict(arrowstyle="->", color="#bf360c", lw=1.5, connectionstyle="arc3,rad=-0.12"),
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#fbe9e7", edgecolor="#bf360c", lw=1.2),
    zorder=5
)

ax.legend(loc="upper left", frameon=True, framealpha=0.95, facecolor="white", edgecolor="#cccccc")
plt.tight_layout()

out_plot_path = "docs/images/nonlinear_dlinear_comparison.png"
plt.savefig(out_plot_path, dpi=300)
plt.close()

print(f"\nSuccessfully generated Non-Linear DLinear comparison plot at: {out_plot_path}")
