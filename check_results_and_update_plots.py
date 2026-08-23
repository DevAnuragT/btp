import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess

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

ablation_summary_path = 'experiments/freshretail_lstm/final_architecture_package/lstm_ablation_summary.csv'
new_res_path = 'experiments/freshretail_lstm/final_architecture_package/new_architectures_summary.json'

df_24h = pd.read_csv(ablation_summary_path)

# Map existing names
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
df_24h['is_new'] = False

# Add new architectures if JSON exists
if os.path.exists(new_res_path):
    with open(new_res_path) as f:
        new_res = json.load(f)
    
    new_rows = []
    for item in new_res:
        new_rows.append({
            'name': item['name'],
            'display_name': item['display_name'],
            'best_hour_level_f1': item['best_hour_level_f1'],
            'best_hour_level_accuracy': item['best_hour_level_accuracy'],
            'best_hour_level_precision': item['best_hour_level_precision'],
            'best_hour_level_recall': item['best_hour_level_recall'],
            'best_exact_24h_match_rate': item['best_exact_24h_match_rate'],
            'best_mean_absolute_hour_count_error': item['best_mean_absolute_hour_count_error'],
            'inference_ms_per_sequence': item['inference_ms_per_sequence'],
            'parameters': item['parameters'],
            'is_new': True
        })
    df_new = pd.DataFrame(new_rows)
    df_combined = pd.concat([df_24h, df_new], ignore_index=True)
else:
    df_combined = df_24h.copy()

# Plot 1: All Hourly Model Variations Ranked by F1
df_sorted = df_combined.sort_values(by='best_hour_level_f1', ascending=True)

fig, ax = plt.subplots(figsize=(11, 8.5))
bar_colors = ['#2ca02c' if is_n else '#1f77b4' for is_n in df_sorted['is_new']]

bars = ax.barh(df_sorted['display_name'], df_sorted['best_hour_level_f1'], color=bar_colors, height=0.65)

baseline_f1 = df_24h[df_24h['name'] == 'baseline_lstm_h96_seq14_all13']['best_hour_level_f1'].values[0]
ax.axvline(baseline_f1, color='#d62728', linestyle='--', linewidth=1.5, label=f'Baseline LSTM ({baseline_f1:.3f})')

for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.003, bar.get_y() + bar.get_height()/2, f'{w:.3f}', va='center', ha='left', fontsize=8, fontweight='bold')

ax.set_xlim(0.60, 0.88)
ax.set_xlabel('Best Hour-Level F1-Score (Primary Metric)', fontweight='bold')
ax.set_title('Hourly Stock Status Forecasting: Model Variations (Green = Tweaked/New Architectures)', fontweight='bold', pad=12)
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('docs/images/24h_architectures_comparison.png')
plt.close()

# Plot 2: F1 vs Parameters (Log scale)
fig, ax = plt.subplots(figsize=(10, 6))

sns.scatterplot(
    data=df_combined,
    x='parameters',
    y='best_hour_level_f1',
    hue='is_new',
    style='is_new',
    s=160,
    palette={False: '#1f77b4', True: '#2ca02c'},
    ax=ax
)

for _, row in df_combined.iterrows():
    if row['is_new'] or row['name'] in ['baseline_lstm_h96_seq14_all13', 'feature_reduced_lstm_h32_seq14_f6']:
        ax.annotate(
            row['display_name'].split(' (')[0],
            (row['parameters'], row['best_hour_level_f1']),
            xytext=(8, -4),
            textcoords='offset points',
            fontsize=8.5,
            fontweight='bold'
        )

ax.set_xscale('log')
ax.set_title('Hourly Models: Hour-Level F1-Score vs Parameters (Green = Tweaked Architectures)', fontweight='bold', pad=12)
ax.set_xlabel('Total Trainable Parameters (Log Scale)', fontweight='bold')
ax.set_ylabel('Best Hour-Level F1-Score', fontweight='bold')
plt.tight_layout()
plt.savefig('docs/images/24h_ablation_f1_params.png')
plt.close()

print("Successfully updated plots with new architectures!")
