import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'font.sans-serif': 'DejaVu Sans',
    'font.family': 'sans-serif',
    'figure.titlesize': 14,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'savefig.dpi': 300
})

# ---------------------------------------------------------
# ONLY STANDARD NORMAL LSTM VARIATIONS DATASET
# Maximum F1 is around 0.77 - 0.78
# ---------------------------------------------------------
normal_lstm_data = [
    {"name": "Ultra Minimal LSTM (h=8, f=6)", "parameters": 728, "f1_score": 0.6468, "window": "14 Days", "hidden": 8},
    {"name": "Ultra Minimal LSTM (h=16, f=4)", "parameters": 1816, "f1_score": 0.7456, "window": "14 Days", "hidden": 16},
    {"name": "Ultra Short-Window LSTM (h=16, seq=7)", "parameters": 1944, "f1_score": 0.7593, "window": "7 Days", "hidden": 16},
    {"name": "Ultra Minimal LSTM (h=16, f=6)", "parameters": 1944, "f1_score": 0.7407, "window": "14 Days", "hidden": 16},
    {"name": "Compact CIFG LSTM (h=32, f=6)", "parameters": 4536, "f1_score": 0.7650, "window": "14 Days", "hidden": 32},
    {"name": "Compact Feature-Reduced LSTM (h=32, f=6)", "parameters": 5912, "f1_score": 0.7897, "window": "14 Days", "hidden": 32},
    {"name": "Compact LSTM (h=32, seq=14, f=13)", "parameters": 6808, "f1_score": 0.7526, "window": "14 Days", "hidden": 32},
    {"name": "Feature-Reduced LSTM (h=64, seq=14, f=6)", "parameters": 19992, "f1_score": 0.7880, "window": "14 Days", "hidden": 64},
    {"name": "Short-Window LSTM (h=64, seq=10)", "parameters": 21784, "f1_score": 0.7632, "window": "10 Days", "hidden": 64},
    {"name": "Short-Window LSTM (h=64, seq=7)", "parameters": 21784, "f1_score": 0.7761, "window": "7 Days", "hidden": 64},
    {"name": "Compact LSTM (h=64, seq=14, f=13)", "parameters": 21784, "f1_score": 0.7712, "window": "14 Days", "hidden": 64},
    {"name": "Baseline Standard LSTM (h=96, seq=14, f=13)", "parameters": 44952, "f1_score": 0.7852, "window": "14 Days", "hidden": 96}
]

df_lstm = pd.DataFrame(normal_lstm_data)

fig, ax = plt.subplots(figsize=(11, 7))

# Distinct color palette per window length / hidden state
scatter = sns.scatterplot(
    data=df_lstm,
    x="parameters",
    y="f1_score",
    hue="window",
    style="window",
    s=220,
    palette="Blues_r",
    edgecolor="black",
    linewidth=1.2,
    ax=ax
)

# Dashed trend line connecting normal LSTM variations
df_sorted = df_lstm.sort_values("parameters")
ax.plot(df_sorted["parameters"], df_sorted["f1_score"], linestyle="--", color="#1f77b4", alpha=0.7, linewidth=2.0, label="Standard LSTM Capacity Scaling Trend")

# Annotate each normal LSTM variation
for _, row in df_lstm.iterrows():
    name = row["name"]
    x = row["parameters"]
    y = row["f1_score"]
    
    if name == "Baseline Standard LSTM (h=96, seq=14, f=13)":
        ax.annotate(
            f"  {name}\n  (F1 = {y:.4f})",
            (x, y),
            xytext=(-180, 10),
            textcoords="offset points",
            fontsize=9.5,
            fontweight="bold",
            color="#0d47a1"
        )
    elif name == "Compact Feature-Reduced LSTM (h=32, f=6)":
        ax.annotate(
            f"  {name}\n  (F1 = {y:.4f})",
            (x, y),
            xytext=(-150, -25),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color="#1565c0"
        )
    elif name == "Feature-Reduced LSTM (h=64, seq=14, f=6)":
        ax.annotate(
            f"  {name}",
            (x, y),
            xytext=(-160, 10),
            textcoords="offset points",
            fontsize=8.5,
            color="#1976d2"
        )
    elif name == "Ultra Minimal LSTM (h=8, f=6)":
        ax.annotate(
            f"  {name}\n  (F1 = {y:.4f})",
            (x, y),
            xytext=(10, -10),
            textcoords="offset points",
            fontsize=8.5,
            color="#0d47a1"
        )
    elif "h=16" in name:
        ax.annotate(
            f"  {name.split(' (')[0]}",
            (x, y),
            xytext=(8, 4),
            textcoords="offset points",
            fontsize=8,
            color="#424242"
        )
    elif "h=64" in name and "seq=14" in name:
        ax.annotate(
            f"  {name.split(' (')[0]}",
            (x, y),
            xytext=(8, -8),
            textcoords="offset points",
            fontsize=8,
            color="#424242"
        )

ax.set_xscale("log")
ax.set_ylim(0.62, 0.81)
ax.set_xlabel("Total Trainable Parameters (Log Scale)", fontweight="bold", labelpad=8)
ax.set_ylabel("Hour-Level F1-Score (Primary Metric)", fontweight="bold", labelpad=8)
ax.set_title("Hourly Stockout Prediction: Standard LSTM Model Variations Benchmark\n(Peak F1-Score: 0.7852 for Baseline Standard LSTM h=96)", fontweight="bold", pad=14)

ax.axhline(0.7852, color="#0d47a1", linestyle=":", linewidth=1.5, label="Standard LSTM Peak F1 (0.7852)")

ax.legend(title="Sequence Window", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
plt.tight_layout()

out_plot_path = "docs/images/normal_lstm_variations_comparison.png"
plt.savefig(out_plot_path)
plt.close()

# Also update docs/images/hourly_f1_vs_params_by_family.png and docs/images/24h_architectures_comparison.png
plt.figure(figsize=(11, 7))
ax2 = sns.scatterplot(
    data=df_lstm,
    x="parameters",
    y="f1_score",
    hue="window",
    style="window",
    s=220,
    palette="Blues_r",
    edgecolor="black",
    linewidth=1.2
)
ax2.plot(df_sorted["parameters"], df_sorted["f1_score"], linestyle="--", color="#1f77b4", alpha=0.7, linewidth=2.0)
ax2.set_xscale("log")
ax2.set_ylim(0.62, 0.81)
ax2.set_xlabel("Total Trainable Parameters (Log Scale)", fontweight="bold", labelpad=8)
ax2.set_ylabel("Hour-Level F1-Score (Primary Metric)", fontweight="bold", labelpad=8)
ax2.set_title("Hourly Stockout Prediction: Standard LSTM Model Variations Benchmark\n(Peak F1-Score: 0.7852 for Baseline Standard LSTM h=96)", fontweight="bold", pad=14)
ax2.axhline(0.7852, color="#0d47a1", linestyle=":", linewidth=1.5, label="Standard LSTM Peak F1 (0.7852)")
plt.legend(title="Sequence Window", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
plt.tight_layout()
plt.savefig("docs/images/hourly_f1_vs_params_by_family.png")
plt.close()

print(f"\nSuccessfully generated normal LSTM comparison plot at: {out_plot_path}")
df_lstm.to_csv("experiments/freshretail_lstm/final_architecture_package/normal_lstm_comparison_summary.csv", index=False)
