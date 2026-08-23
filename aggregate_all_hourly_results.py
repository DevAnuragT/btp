import os
import json
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
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.autolayout': True,
    'savefig.dpi': 300
})

ablation_summary_path = 'experiments/freshretail_lstm/final_architecture_package/lstm_ablation_summary.csv'
new_res_path = 'experiments/freshretail_lstm/final_architecture_package/new_architectures_summary.json'
sota_res_path = 'experiments/freshretail_lstm/final_architecture_package/sota_architectures_summary.json'
ssm_res_path = 'experiments/freshretail_lstm/final_architecture_package/mamba_ssm_summary.json'
dlinear_res_path = 'experiments/freshretail_lstm/final_architecture_package/dlinear_tuning_summary.json'

df_24h = pd.read_csv(ablation_summary_path)

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
df_24h['category'] = 'Baseline & Ablation'

with open(sota_res_path) as f:
    sota_data = json.load(f)
for item in sota_data: item['category'] = 'SOTA Model'

with open(new_res_path) as f:
    new_data = json.load(f)
for item in new_data: item['category'] = 'Tweaked Architecture'

with open(ssm_res_path) as f:
    ssm_data = json.load(f)
for item in ssm_data: item['category'] = 'State Sequence Model (SSM/BiLSTM)'

with open(dlinear_res_path) as f:
    dlinear_data = json.load(f)
for item in dlinear_data: item['category'] = 'Enhanced DLinear Variant'

df_sota = pd.DataFrame(sota_data)
df_new = pd.DataFrame(new_data)
df_ssm = pd.DataFrame(ssm_data)
df_dlinear = pd.DataFrame(dlinear_data)

df_all = pd.concat([df_24h, df_new, df_sota, df_ssm, df_dlinear], ignore_index=True)
df_all = df_all.drop_duplicates(subset=['name'], keep='last')
df_all.to_csv('experiments/freshretail_lstm/final_architecture_package/master_hourly_models_benchmark.csv', index=False)

# ---------------------------------------------------------
# Plot 1: All Hourly Model Variations Ranked by F1-Score
# ---------------------------------------------------------
df_sorted = df_all.sort_values(by='best_hour_level_f1', ascending=True)

fig, ax = plt.subplots(figsize=(12, 12))

palette = {
    'Baseline & Ablation': '#1f77b4',
    'Tweaked Architecture': '#2ca02c',
    'SOTA Model': '#ff7f0e',
    'State Sequence Model (SSM/BiLSTM)': '#9467bd',
    'Enhanced DLinear Variant': '#e377c2'
}

colors = [palette.get(cat, '#1f77b4') for cat in df_sorted['category']]
bars = ax.barh(df_sorted['display_name'], df_sorted['best_hour_level_f1'], color=colors, height=0.68)

baseline_f1 = df_24h[df_24h['name'] == 'baseline_lstm_h96_seq14_all13']['best_hour_level_f1'].values[0]
ax.axvline(baseline_f1, color='#d62728', linestyle='--', linewidth=1.5, label=f'Baseline LSTM ({baseline_f1:.3f})')

for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.003, bar.get_y() + bar.get_height()/2, f'{w:.3f}', va='center', ha='left', fontsize=8, fontweight='bold')

ax.set_xlim(0.60, 0.86)
ax.set_xlabel('Best Hour-Level F1-Score (Primary Metric)', fontweight='bold')
ax.set_title('Hourly Stock Status Prediction: Master Rankings (Pink = Enhanced DLinear, Purple = SSM, Orange = SOTA)', fontweight='bold', pad=12)
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('docs/images/24h_architectures_comparison.png')
plt.close()

# ---------------------------------------------------------
# Plot 2: F1-Score vs Parameters (Log Scale) across SOTA Models
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

sns.scatterplot(
    data=df_all,
    x='parameters',
    y='best_hour_level_f1',
    hue='category',
    style='category',
    s=160,
    palette=palette,
    ax=ax
)

top_labels = ['baseline_lstm_h96_seq14_all13', 'dlinear_multikernel', 'dlinear_gated_residual', 'deep_bilstm_resnet', 'mamba_ssm_selective', 'nhits_hierarchical', 'tft_lite_vsn', 'ultra_sota_super_blend', 'dlinear_decomposition', 'hierarchical_dualstream_resnet', 'hourly_query_transformer']

for _, row in df_all.iterrows():
    if row['name'] in top_labels:
        ax.annotate(
            row['display_name'].split(' (')[0].replace('★ ', '').replace('🏆 ', ''),
            (row['parameters'], row['best_hour_level_f1']),
            xytext=(8, -4),
            textcoords='offset points',
            fontsize=8.5,
            fontweight='bold'
        )

ax.set_xscale('log')
ax.set_title('Master Hourly SOTA Models: F1-Score vs Parameters (Log Scale)', fontweight='bold', pad=12)
ax.set_xlabel('Total Trainable Parameters (Log Scale)', fontweight='bold')
ax.set_ylabel('Best Hour-Level F1-Score', fontweight='bold')
ax.legend(title='Architecture Category', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig('docs/images/24h_ablation_f1_params.png')
plt.close()

print("Successfully updated Master SOTA plots including Enhanced DLinear in docs/images/")
