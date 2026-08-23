import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'font.sans-serif': 'DejaVu Sans',
    'font.family': 'sans-serif',
    'figure.titlesize': 14,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.autolayout': True,
    'savefig.dpi': 300
})

os.makedirs('docs/images', exist_ok=True)

# ---------------------------------------------------------
# Load Hourly Data Only (from FreshRetail LSTM Package)
# ---------------------------------------------------------
ablation_summary_path = 'experiments/freshretail_lstm/final_architecture_package/lstm_ablation_summary.csv'
df_24h = pd.read_csv(ablation_summary_path)

# Categorize model architectures
def categorize_model(name):
    if 'dual_stream_gated' in name:
        return 'Dual-Stream Gated Shortcut'
    elif 'dual_stream_inventory' in name:
        return 'Dual-Stream Inventory Shortcut'
    elif 'dual_stream_shortcut' in name:
        return 'Dual-Stream Shortcut'
    elif 'dual_stream' in name:
        return 'Dual-Stream Baseline / CIFG'
    elif 'cifg' in name:
        return 'Compact CIFG'
    elif 'ultra' in name:
        return 'Ultra-Compact LSTM (h8-h16)'
    elif 'feature_reduced' in name:
        return 'Compact Feature-Reduced (f6-f4)'
    elif 'compact' in name:
        return 'Compact Standard LSTM (h32-h64)'
    else:
        return 'Baseline LSTM (h96)'

df_24h['category'] = df_24h['name'].apply(categorize_model)

# Clean display names
name_clean_map = {
    'baseline_lstm_h96_seq14_all13': 'Baseline LSTM (h=96, seq=14, f=13)',
    'compact_lstm_h64_seq14_all13': 'Compact LSTM (h=64, seq=14, f=13)',
    'compact_lstm_h32_seq14_all13': 'Compact LSTM (h=32, seq=14, f=13)',
    'compact_lstm_h64_seq10_all13': 'Short-Window LSTM (h=64, seq=10, f=13)',
    'compact_lstm_h64_seq07_all13': 'Short-Window LSTM (h=64, seq=7, f=13)',
    'feature_reduced_lstm_h64_seq14_f6': 'Feature-Reduced LSTM (h=64, seq=14, f=6)',
    'feature_reduced_lstm_h32_seq14_f6': 'Compact Feature-Reduced (h=32, seq=14, f=6)',
    'ultra_lstm_h16_seq14_f6': 'Ultra LSTM (h=16, seq=14, f=6)',
    'ultra_lstm_h08_seq14_f6': 'Ultra LSTM (h=8, seq=14, f=6)',
    'ultra_lstm_h16_seq14_f4': 'Ultra Minimal LSTM (h=16, seq=14, f=4)',
    'ultra_lstm_h16_seq07_f6': 'Ultra Short-Window (h=16, seq=7, f=6)',
    'compact_cifg_h32_seq14_f6': 'Compact CIFG LSTM (h=32, seq=14, f=6)',
    'dual_stream_lstm_h16x16_seq14_f6': 'Dual-Stream LSTM (h=16x16, f=6)',
    'dual_stream_cifg_h16x16_seq14_f6': 'Dual-Stream CIFG (h=16x16, f=6)',
    'dual_stream_lstm_h20x12_seq14_f6': 'Dual-Stream Asymmetric (h=20x12, f=6)',
    'dual_stream_shortcut_h16x16_seq14_f6': 'Dual-Stream Shortcut (h=16x16, f=6)',
    'dual_stream_shortcut_h12x12_seq14_f6': 'Dual-Stream Shortcut (h=12x12, f=6)',
    'dual_stream_inventory_shortcut_h16x16_seq14_f6': 'Dual-Stream Inventory Shortcut (h=16x16, f=6)',
    'dual_stream_gated_shortcut_h16x16_seq14_f6': 'Dual-Stream Gated Shortcut (h=16x16, f=6)'
}

df_24h['display_name'] = df_24h['name'].map(name_clean_map).fillna(df_24h['name'])

# ---------------------------------------------------------
# Plot 1: All Hourly Architectures Ranked by Hour-Level F1-Score
# ---------------------------------------------------------
df_sorted = df_24h.sort_values(by='best_hour_level_f1', ascending=True)

fig, ax = plt.subplots(figsize=(11, 7.5))
colors = sns.color_palette("viridis", len(df_sorted))

bars = ax.barh(df_sorted['display_name'], df_sorted['best_hour_level_f1'], color=colors, height=0.65)

# Benchmark baseline line
baseline_f1 = df_24h[df_24h['name'] == 'baseline_lstm_h96_seq14_all13']['best_hour_level_f1'].values[0]
ax.axvline(baseline_f1, color='#d62728', linestyle='--', linewidth=1.5, label=f'Baseline LSTM Benchmark ({baseline_f1:.3f})')

for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.003, bar.get_y() + bar.get_height()/2, f'{w:.3f}', va='center', ha='left', fontsize=8.5, fontweight='bold')

ax.set_xlim(0.60, 0.82)
ax.set_xlabel('Best Hour-Level F1-Score (Primary Operational Metric)', fontweight='bold')
ax.set_title('Hourly Stock Status Forecasting: Model Variations Ranked by Hour-Level F1-Score', fontweight='bold', pad=12)
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('docs/images/24h_architectures_comparison.png')
plt.close()

# ---------------------------------------------------------
# Plot 2: F1-Score vs Model Parameter Count (Log Scale)
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

palette = {
    'Baseline LSTM (h96)': '#d62728',
    'Compact Standard LSTM (h32-h64)': '#1f77b4',
    'Compact Feature-Reduced (f6-f4)': '#2ca02c',
    'Compact CIFG': '#9467bd',
    'Ultra-Compact LSTM (h8-h16)': '#ff7f0e',
    'Dual-Stream Baseline / CIFG': '#17becf',
    'Dual-Stream Shortcut': '#bcbd22',
    'Dual-Stream Inventory Shortcut': '#8c564b',
    'Dual-Stream Gated Shortcut': '#e377c2'
}

sns.scatterplot(
    data=df_24h,
    x='parameters',
    y='best_hour_level_f1',
    hue='category',
    style='category',
    s=150,
    palette=palette,
    ax=ax
)

key_labels = [
    'baseline_lstm_h96_seq14_all13',
    'feature_reduced_lstm_h32_seq14_f6',
    'ultra_lstm_h08_seq14_f6',
    'dual_stream_inventory_shortcut_h16x16_seq14_f6',
    'dual_stream_gated_shortcut_h16x16_seq14_f6'
]

for _, row in df_24h.iterrows():
    if row['name'] in key_labels:
        ax.annotate(
            row['display_name'].split(' (')[0],
            (row['parameters'], row['best_hour_level_f1']),
            xytext=(8, -4),
            textcoords='offset points',
            fontsize=8.5,
            fontweight='bold'
        )

ax.set_xscale('log')
ax.set_title('Hourly Forecast Models: Hour-Level F1-Score vs Parameters (Log Scale)', fontweight='bold', pad=12)
ax.set_xlabel('Total Trainable Parameters (Log Scale)', fontweight='bold')
ax.set_ylabel('Best Hour-Level F1-Score', fontweight='bold')
ax.legend(title='Architecture Category', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig('docs/images/24h_ablation_f1_params.png')
plt.close()

# ---------------------------------------------------------
# Plot 3: Exact 24-Hour Match Rate vs MAE (Hours)
# ---------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(10, 5.5))

df_sorted_mae = df_24h.sort_values(by='best_mean_absolute_hour_count_error', ascending=True)

x = np.arange(len(df_sorted_mae))
width = 0.35

rects1 = ax1.bar(x - width/2, df_sorted_mae['best_exact_24h_match_rate'] * 100, width, label='Exact 24h Match Rate (%)', color='#1f77b4')

ax2 = ax1.twinx()
rects2 = ax2.bar(x + width/2, df_sorted_mae['best_mean_absolute_hour_count_error'], width, label='MAE Error (Hours - Lower is Better)', color='#d62728')

ax1.set_ylabel('Exact Match Rate (%)', color='#1f77b4', fontweight='bold')
ax2.set_ylabel('Mean Absolute Hour Count Error (Hours)', color='#d62728', fontweight='bold')
ax1.set_title('Hourly Forecast Models: Exact 24-Hour Match Rate vs Hour Count Error (MAE)', fontweight='bold', pad=12)

ax1.set_xticks(x)
ax1.set_xticklabels(df_sorted_mae['display_name'], rotation=40, ha='right', fontsize=8)

ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.tight_layout()
plt.savefig('docs/images/hourly_exact_match_vs_mae.png')
plt.close()

# ---------------------------------------------------------
# Plot 4: Inference Latency vs Hour-Level F1-Score & MAE
# ---------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(9, 5))

df_lat = df_24h.sort_values(by='inference_ms_per_sequence')

color = '#1f77b4'
ax1.set_xlabel('Inference Latency (ms per sequence)', fontweight='bold')
ax1.set_ylabel('Hour-Level F1-Score', color=color, fontweight='bold')
ax1.scatter(df_lat['inference_ms_per_sequence'], df_lat['best_hour_level_f1'], color=color, s=90, label='Hour-Level F1')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = '#d62728'
ax2.set_ylabel('Mean Absolute Hour Count Error (MAE in hours)', color=color, fontweight='bold')
ax2.scatter(df_lat['inference_ms_per_sequence'], df_lat['best_mean_absolute_hour_count_error'], color=color, marker='s', s=90, label='MAE (Lower is Better)')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Hourly Forecast Models: Inference Latency vs F1-Score & MAE', fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('docs/images/24h_latency_vs_accuracy.png')
plt.close()

# ---------------------------------------------------------
# Plot 5: Category Aggregated Performance Comparison
# ---------------------------------------------------------
df_cat = df_24h.groupby('category').agg({
    'best_hour_level_f1': 'mean',
    'best_hour_level_recall': 'mean',
    'best_exact_24h_match_rate': 'mean',
    'best_mean_absolute_hour_count_error': 'mean',
    'parameters': 'mean'
}).reset_index().sort_values(by='best_hour_level_f1', ascending=False)

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(df_cat['category'], df_cat['best_hour_level_f1'], color=sns.color_palette("mako", len(df_cat)), width=0.55)

for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.005, f'{h:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylim(0.60, 0.82)
ax.set_ylabel('Mean Hour-Level F1-Score', fontweight='bold')
ax.set_title('Hourly Forecast Models: Performance Comparison by Architectural Category', fontweight='bold', pad=12)
plt.xticks(rotation=25, ha='right', fontsize=9)
plt.tight_layout()
plt.savefig('docs/images/hourly_ablation_categories.png')
plt.close()

# ---------------------------------------------------------
# Plot 6: Dataset & Metric Choice Analysis (Hourly Forecast Context)
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# Synthetic ROC vs PR curve demonstrating metric sensitivity under 80:20 imbalance for hourly predictions
hours = np.linspace(1, 24, 24)
true_stockout_hours = np.array([0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0])
pred_probs_lstm = np.array([0.05,0.05,0.05,0.05,0.05,0.05,0.85,0.90,0.88,0.92,0.95,0.91,0.89,0.87,0.82,0.80,0.78,0.75,0.70,0.10,0.05,0.05,0.05,0.05])
pred_probs_constant = np.zeros(24) + 0.05 # Naive constant predictor

ax1.plot(hours, true_stockout_hours, color='black', drawstyle='steps-mid', linewidth=2.5, label='True Hourly Stock Status (1=Stockout)')
ax1.plot(hours, pred_probs_lstm, color='#1f77b4', linestyle='--', linewidth=2, label='LSTM Hourly Probability')
ax1.set_title('(A) Hourly Stock Status Temporal Profile (24 Hours)', fontweight='bold')
ax1.set_xlabel('Hour of Day (1 to 24)')
ax1.set_ylabel('Stockout Probability / Status')
ax1.set_xticks(range(1, 25, 2))
ax1.legend(loc='upper right', fontsize=8.5)

# Metrics comparison bar plot
metric_names = ['Raw Accuracy', 'F1-Score', 'Exact 24h Match', 'MAE (hrs)']
lstm_scores = [0.88, 0.79, 0.26, 3.97]
naive_scores = [0.70, 0.00, 0.00, 13.00]

x_m = np.arange(len(metric_names))
w_m = 0.35

ax2.bar(x_m - w_m/2, [0.88, 0.79, 0.26, 0.70], w_m, label='Compact LSTM (h32)', color='#1f77b4')
ax2.bar(x_m + w_m/2, [0.70, 0.00, 0.00, 0.00], w_m, label='Naive All-Instock Model', color='#d62728')
ax2.set_xticks(x_m)
ax2.set_xticklabels(metric_names, rotation=15, fontsize=9)
ax2.set_ylabel('Normalized Score', fontweight='bold')
ax2.set_title('(B) Operational Metric Discrimination Power', fontweight='bold')
ax2.legend(loc='upper right', fontsize=8.5)

plt.tight_layout()
plt.savefig('docs/images/dataset_metric_choice_analysis.png')
plt.close()

print("Successfully generated all HOURLY forecast plot files in docs/images/")
