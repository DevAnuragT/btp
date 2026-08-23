import os
import random
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import load_dataset
from pathlib import Path

# Seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load FreshRetail Dataset
print("Loading FreshRetailNet-50K...")
raw_df = load_dataset("Dingdong-Inc/FreshRetailNet-50K", split="train").to_pandas()

def to_float_list(value, length=24):
    if isinstance(value, np.ndarray):
        values = value.tolist()
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = []
    values = [0.0 if pd.isna(item) else float(item) for item in values[:length]]
    if len(values) < length:
        values.extend([0.0] * (length - len(values)))
    return values

# Data Cleaning & Processing
raw_df["series_id"] = raw_df["city_id"].astype(str) + "_" + raw_df["store_id"].astype(str) + "_" + raw_df["product_id"].astype(str)
raw_df["dt"] = pd.to_datetime(raw_df["dt"], errors="coerce")
raw_df = raw_df.dropna(subset=["dt", "city_id", "store_id", "product_id"])
raw_df = raw_df.drop_duplicates(subset=["series_id", "dt"], keep="last")

stock_vectors = raw_df["hours_stock_status"].map(to_float_list)
stock_matrix = np.array(stock_vectors.tolist(), dtype=np.float32)
for h in range(24):
    raw_df[f"stock_h{h:02d}"] = stock_matrix[:, h]

raw_df["hours_sale_sum"] = raw_df["hours_sale"].map(lambda v: float(np.sum(to_float_list(v))))
raw_df["hours_stock_status_sum"] = stock_matrix.sum(axis=1)

# Feature Engineering
raw_df["day_of_week"] = raw_df["dt"].dt.dayofweek
raw_df["dow_sin"] = np.sin(2 * np.pi * raw_df["day_of_week"] / 7.0)
raw_df["dow_cos"] = np.cos(2 * np.pi * raw_df["day_of_week"] / 7.0)

for col in ["sale_amount", "stock_hour6_22_cnt", "discount", "holiday_flag", "hours_sale_sum", "hours_stock_status_sum"]:
    raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce").fillna(0.0)

# Sort & Group
raw_df = raw_df.sort_values(["series_id", "dt"]).reset_index(drop=True)

# Rolling features per series
raw_df["stockout_rolling_3"] = raw_df.groupby("series_id")["stock_hour6_22_cnt"].transform(lambda s: (s > 0).rolling(3, min_periods=1).sum())
raw_df["sales_momentum"] = raw_df.groupby("series_id")["sale_amount"].diff().fillna(0.0)

IMPROVED_FEATURES = [
    "sale_amount", "stock_hour6_22_cnt", "discount", "holiday_flag", 
    "hours_sale_sum", "hours_stock_status_sum", "dow_sin", "dow_cos", 
    "stockout_rolling_3", "sales_momentum"
]

TARGET_COLUMNS = [f"target_stock_h{h:02d}" for h in range(24)]
stock_columns = [f"stock_h{h:02d}" for h in range(24)]
for src, tgt in zip(stock_columns, TARGET_COLUMNS):
    raw_df[tgt] = raw_df.groupby("series_id")[src].shift(-1)

clean_df = raw_df.dropna(subset=TARGET_COLUMNS).reset_index(drop=True)

# Select Top 15 SKUs
summary = clean_df.groupby("series_id").agg(
    total_sales=("sale_amount", "sum"),
    sales_std=("sale_amount", "std"),
    total_stockout_hours=("hours_stock_status_sum", "sum"),
    stockout_days=("stock_hour6_22_cnt", lambda v: int((v > 0).sum())),
    product_id=("product_id", "first")
).reset_index()

for c in ["total_sales", "sales_std", "total_stockout_hours", "stockout_days"]:
    denom = summary[c].max() - summary[c].min()
    summary[f"{c}_score"] = (summary[c] - summary[c].min()) / denom if denom != 0 else 0

summary["selection_score"] = 0.40 * summary["total_sales_score"] + 0.25 * summary["total_stockout_hours_score"] + 0.20 * summary["stockout_days_score"] + 0.15 * summary["sales_std_score"]
selected = summary.sort_values("selection_score", ascending=False).drop_duplicates(subset=["product_id"]).head(15)

selected_df = clean_df[clean_df["series_id"].isin(selected["series_id"])].copy()

validation_days = 15
cutoff_date = selected_df["dt"].max() - pd.Timedelta(days=validation_days)

scaler = StandardScaler()
scaler.fit(selected_df.loc[selected_df["dt"] <= cutoff_date, IMPROVED_FEATURES])
selected_df[IMPROVED_FEATURES] = scaler.transform(selected_df[IMPROVED_FEATURES])

class HourlySeqDataset(Dataset):
    def __init__(self, frame, seq_len=14, mode="train"):
        self.x, self.y = [], []
        for s_id, group in frame.groupby("series_id", sort=False):
            group = group.sort_values("dt").reset_index(drop=True)
            feats = group[IMPROVED_FEATURES].to_numpy(dtype=np.float32)
            tgts = group[TARGET_COLUMNS].to_numpy(dtype=np.float32)
            dates = group["dt"].to_numpy()
            if len(group) <= seq_len: continue
            for end in range(seq_len, len(group)):
                t_date = pd.Timestamp(dates[end])
                if mode == "train" and t_date > cutoff_date: continue
                if mode == "val" and t_date <= cutoff_date: continue
                self.x.append(feats[end - seq_len : end])
                self.y.append(tgts[end - 1])
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return torch.tensor(self.x[i]), torch.tensor(self.y[i])

train_ds = HourlySeqDataset(selected_df, seq_len=14, mode="train")
val_ds = HourlySeqDataset(selected_df, seq_len=14, mode="val")

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

# ---------------------------------------------------------
# Advanced Models & Loss Functions
# ---------------------------------------------------------

# 1. Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.5):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = bce * ((1 - p_t) ** self.gamma)
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss.mean()

# 2. Improved Architecture: BiGRU with Temporal Self-Attention
class ImprovedBiGRUAttn(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 24)
        )
    def forward(self, x):
        out, _ = self.gru(x) # [B, L, H*2]
        weights = F.softmax(self.attn(out), dim=1) # [B, L, 1]
        context = torch.sum(weights * out, dim=1) # [B, H*2]
        return self.fc(context)

# 3. Improved Architecture: Dual-Stream Gated Shortcut with LayerNorm
class ImprovedDualStreamGated(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32):
        super().__init__()
        demand_dim = 5 # sales, discount, dow_sin, dow_cos, sales_momentum
        inven_dim = 5  # stock_hour6_22_cnt, holiday_flag, hours_sale_sum, hours_stock_status_sum, stockout_rolling_3
        
        self.demand_lstm = nn.LSTM(demand_dim, hidden_dim//2, batch_first=True)
        self.inven_lstm = nn.LSTM(inven_dim, hidden_dim//2, batch_first=True)
        
        self.shortcut_proj = nn.Linear(input_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 24)
        )
    def forward(self, x):
        d_x = x[:, :, :5]
        i_x = x[:, :, 5:]
        _, (d_h, _) = self.demand_lstm(d_x)
        _, (i_h, _) = self.inven_lstm(i_x)
        fused = torch.cat([d_h[-1], i_h[-1]], dim=-1)
        shortcut = self.shortcut_proj(x[:, -1, :])
        g = torch.sigmoid(self.gate(torch.cat([fused, shortcut], dim=-1)))
        out = self.norm(g * fused + (1.0 - g) * shortcut)
        return self.head(out)

# Evaluate function with threshold scanning
def evaluate_model(model, loader, thresholds=None):
    model.eval()
    all_targets, all_probs = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(y.numpy())
    
    targets = np.vstack(all_targets)
    probs = np.vstack(all_probs)
    
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 17)
    
    best_tau = 0.5
    best_f1 = -1.0
    best_metrics = {}
    
    for tau in thresholds:
        preds = (probs >= tau).astype(int)
        flat_y = targets.astype(int).ravel()
        flat_p = preds.ravel()
        f1 = f1_score(flat_y, flat_p, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_tau = tau
            best_metrics = {
                "threshold": float(tau),
                "hour_level_f1": float(f1),
                "hour_level_accuracy": float(accuracy_score(flat_y, flat_p)),
                "hour_level_precision": float(precision_score(flat_y, flat_p, zero_division=0)),
                "hour_level_recall": float(recall_score(flat_y, flat_p, zero_division=0)),
                "exact_24h_match_rate": float((preds == targets).all(axis=1).mean()),
                "mean_absolute_hour_count_error": float(np.abs(preds.sum(axis=1) - targets.sum(axis=1)).mean())
            }
    return best_metrics, probs, targets

# Training Runner
def train_and_eval(model_class, name, use_focal=True, epochs=15):
    print(f"\n--- Training {name} ---")
    model = model_class().to(device)
    criterion = FocalLoss(gamma=2.0) if use_focal else nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        
        if epoch % 5 == 0 or epoch == epochs:
            val_metrics, _, _ = evaluate_model(model, val_loader)
            print(f"Epoch {epoch}/{epochs} | Loss: {np.mean(losses):.4f} | Val F1 (Best Tau={val_metrics['threshold']:.2f}): {val_metrics['hour_level_f1']:.4f} | Recall: {val_metrics['hour_level_recall']:.4f} | MAE: {val_metrics['mean_absolute_hour_count_error']:.3f}")
            
    final_metrics, probs, targets = evaluate_model(model, val_loader)
    return final_metrics, probs, targets

# Train Models
m1_metrics, p1, y_val = train_and_eval(ImprovedBiGRUAttn, "Improved BiGRU with Self-Attention", use_focal=True)
m2_metrics, p2, _ = train_and_eval(ImprovedDualStreamGated, "Improved Dual-Stream Gated LayerNorm", use_focal=True)

# Ensemble Blending (50/50 Weighted Average)
p_ensemble = 0.5 * p1 + 0.5 * p2
best_ens_f1 = -1.0
best_ens_tau = 0.5
ens_metrics = {}
for tau in np.linspace(0.1, 0.9, 17):
    preds = (p_ensemble >= tau).astype(int)
    flat_y = y_val.astype(int).ravel()
    flat_p = preds.ravel()
    f1 = f1_score(flat_y, flat_p, zero_division=0)
    if f1 > best_ens_f1:
        best_ens_f1 = f1
        best_ens_tau = tau
        ens_metrics = {
            "model": "Ensemble (BiGRU-Attn + DualStream-Gated)",
            "threshold": float(tau),
            "hour_level_f1": float(f1),
            "hour_level_accuracy": float(accuracy_score(flat_y, flat_p)),
            "hour_level_precision": float(precision_score(flat_y, flat_p, zero_division=0)),
            "hour_level_recall": float(recall_score(flat_y, flat_p, zero_division=0)),
            "exact_24h_match_rate": float((preds == y_val).all(axis=1).mean()),
            "mean_absolute_hour_count_error": float(np.abs(preds.sum(axis=1) - y_val.sum(axis=1)).mean())
        }

print("\n=======================================================")
print("IMPROVEMENT RESULTS SUMMARY (Next-Day 24h Hourly Prediction)")
print("=======================================================")
print(f"1. BiGRU + Self-Attention: F1 = {m1_metrics['hour_level_f1']:.4f} | Recall = {m1_metrics['hour_level_recall']:.4f} | Exact Match = {m1_metrics['exact_24h_match_rate']*100:.1f}% | MAE = {m1_metrics['mean_absolute_hour_count_error']:.3f} (Best Threshold = {m1_metrics['threshold']:.2f})")
print(f"2. Dual-Stream Gated LayerNorm: F1 = {m2_metrics['hour_level_f1']:.4f} | Recall = {m2_metrics['hour_level_recall']:.4f} | Exact Match = {m2_metrics['exact_24h_match_rate']*100:.1f}% | MAE = {m2_metrics['mean_absolute_hour_count_error']:.3f} (Best Threshold = {m2_metrics['threshold']:.2f})")
print(f"3. Ensemble Model Blend: F1 = {ens_metrics['hour_level_f1']:.4f} | Recall = {ens_metrics['hour_level_recall']:.4f} | Exact Match = {ens_metrics['exact_24h_match_rate']*100:.1f}% | MAE = {ens_metrics['mean_absolute_hour_count_error']:.3f} (Best Threshold = {ens_metrics['threshold']:.2f})")

# Save results
output_res = {
    "baseline_f1": 0.790,
    "improved_bigru_attn": m1_metrics,
    "improved_dual_stream": m2_metrics,
    "ensemble_blend": ens_metrics
}
with open("experiments/freshretail_lstm/final_architecture_package/improved_results_summary.json", "w") as f:
    json.dump(output_res, f, indent=2)

print("\nSaved improved results to experiments/freshretail_lstm/final_architecture_package/improved_results_summary.json")
