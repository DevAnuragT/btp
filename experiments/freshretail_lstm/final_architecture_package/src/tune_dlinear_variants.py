import os
import random
import json
import time
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
print("Loading FreshRetailNet-50K dataset...")
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

# Data Preprocessing
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

raw_df = raw_df.sort_values(["series_id", "dt"]).reset_index(drop=True)

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

# Focal Loss
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

# Moving Average Block for DLinear
class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: [B, L, C]
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x_padded = torch.cat([front, x, end], dim=1)
        x_avg = self.avg(x_padded.permute(0, 2, 1)).permute(0, 2, 1)
        return x_avg

class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

# ---------------------------------------------------------
# DLINEAR VARIATIONS FOR F1-SCORE OPTIMIZATION
# ---------------------------------------------------------

# Variant 1: Baseline Standard DLinear
class BaselineDLinear(nn.Module):
    def __init__(self, seq_len=14, input_dim=10, kernel_size=5):
        super().__init__()
        self.decomp = SeriesDecomp(kernel_size)
        self.Linear_Seasonal = nn.Linear(seq_len * input_dim, 24)
        self.Linear_Trend = nn.Linear(seq_len * input_dim, 24)

    def forward(self, x):
        seasonal_init, trend_init = self.decomp(x)
        seasonal_flat = seasonal_init.reshape(x.size(0), -1)
        trend_flat = trend_init.reshape(x.size(0), -1)
        logits = self.Linear_Seasonal(seasonal_flat) + self.Linear_Trend(trend_flat)
        return logits

# Variant 2: Multi-Kernel DLinear (Extracts Short k=3, Mid k=5, Long k=7 trends)
class MultiKernelDLinear(nn.Module):
    def __init__(self, seq_len=14, input_dim=10):
        super().__init__()
        self.decomp3 = SeriesDecomp(3)
        self.decomp5 = SeriesDecomp(5)
        self.decomp7 = SeriesDecomp(7)

        self.Linear_Seasonal = nn.Linear(seq_len * input_dim, 24)
        self.Linear_Trend3 = nn.Linear(seq_len * input_dim, 24)
        self.Linear_Trend5 = nn.Linear(seq_len * input_dim, 24)
        self.Linear_Trend7 = nn.Linear(seq_len * input_dim, 24)
        
        # Softmax weighting across trend scales
        self.scale_weights = nn.Parameter(torch.ones(3) / 3.0)

    def forward(self, x):
        B = x.size(0)
        s3, t3 = self.decomp3(x)
        _, t5 = self.decomp5(x)
        _, t7 = self.decomp7(x)

        s_flat = s3.reshape(B, -1)
        t3_flat = t3.reshape(B, -1)
        t5_flat = t5.reshape(B, -1)
        t7_flat = t7.reshape(B, -1)

        w = F.softmax(self.scale_weights, dim=0)
        trend_out = w[0] * self.Linear_Trend3(t3_flat) + w[1] * self.Linear_Trend5(t5_flat) + w[2] * self.Linear_Trend7(t7_flat)
        seasonal_out = self.Linear_Seasonal(s_flat)

        return trend_out + seasonal_out

# Variant 3: Channel-Independent + Feature Gated DLinear (`ChannelGatedDLinear`)
class ChannelGatedDLinear(nn.Module):
    def __init__(self, seq_len=14, input_dim=10, kernel_size=5):
        super().__init__()
        self.decomp = SeriesDecomp(kernel_size)
        self.input_dim = input_dim
        
        # Individual linear projections per input channel (Feature-wise independence)
        self.seasonal_channels = nn.ModuleList([nn.Linear(seq_len, 24) for _ in range(input_dim)])
        self.trend_channels = nn.ModuleList([nn.Linear(seq_len, 24) for _ in range(input_dim)])
        
        # Softmax feature importance gating weights
        self.feature_gate = nn.Parameter(torch.ones(input_dim) / input_dim)

    def forward(self, x):
        # x: [B, L, C]
        B = x.size(0)
        s_init, t_init = self.decomp(x)
        
        weights = F.softmax(self.feature_gate, dim=0)
        out = 0.0
        for i in range(self.input_dim):
            s_c = s_init[:, :, i] # [B, L]
            t_c = t_init[:, :, i] # [B, L]
            out_c = self.seasonal_channels[i](s_c) + self.trend_channels[i](t_c) # [B, 24]
            out = out + weights[i] * out_c

        return out

# Variant 4: Hourly Slot-Specific DLinear (`HourlySlotDLinear`)
class HourlySlotDLinear(nn.Module):
    """
    Learns 24 explicit hourly slot linear heads.
    Each target hour h in [1..24] has its own dedicated trend and seasonal linear weights.
    """
    def __init__(self, seq_len=14, input_dim=10, kernel_size=5):
        super().__init__()
        self.decomp = SeriesDecomp(kernel_size)
        # 24 separate linear projections for each hour
        self.hourly_seasonal = nn.Parameter(torch.randn(24, seq_len * input_dim) * 0.01)
        self.hourly_trend = nn.Parameter(torch.randn(24, seq_len * input_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(24))

    def forward(self, x):
        B = x.size(0)
        s_init, t_init = self.decomp(x)
        s_flat = s_init.reshape(B, -1) # [B, L*C]
        t_flat = t_init.reshape(B, -1) # [B, L*C]

        # Explicit einsum mapping per hour slot: logits[b, h] = s_flat[b, i] * s_weights[h, i]
        s_logits = torch.einsum('bi,hi->bh', s_flat, self.hourly_seasonal)
        t_logits = torch.einsum('bi,hi->bh', t_flat, self.hourly_trend)
        
        return s_logits + t_logits + self.bias

# Variant 5: Gated Residual Non-Linear DLinear (`GatedResidualDLinear`)
class GatedResidualDLinear(nn.Module):
    """
    Combines DLinear decomposition with a Gated Linear Unit (GLU) residual shortcut
    to capture non-linear demand thresholds without overparameterization.
    """
    def __init__(self, seq_len=14, input_dim=10, kernel_size=5):
        super().__init__()
        self.decomp = SeriesDecomp(kernel_size)
        self.Linear_Seasonal = nn.Linear(seq_len * input_dim, 24)
        self.Linear_Trend = nn.Linear(seq_len * input_dim, 24)
        
        # Gated non-linear shortcut
        self.glu_proj = nn.Sequential(
            nn.Linear(seq_len * input_dim, 48),
            nn.SiLU(),
            nn.Linear(48, 24)
        )
        self.glu_gate = nn.Sequential(
            nn.Linear(seq_len * input_dim, 24),
            nn.Sigmoid()
        )

    def forward(self, x):
        B = x.size(0)
        s_init, t_init = self.decomp(x)
        s_flat = s_init.reshape(B, -1)
        t_flat = t_init.reshape(B, -1)
        
        dlinear_out = self.Linear_Seasonal(s_flat) + self.Linear_Trend(t_flat)
        glu_residual = self.glu_proj(s_flat) * self.glu_gate(t_flat)
        
        return dlinear_out + glu_residual

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate_model(model, loader):
    model.eval()
    all_targets, all_probs = [], []
    
    start_time = time.time()
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(y.numpy())
    end_time = time.time()
    
    targets = np.vstack(all_targets)
    probs = np.vstack(all_probs)
    
    total_seqs = len(targets)
    latency_ms = ((end_time - start_time) / total_seqs) * 1000.0
    
    best_tau = 0.5
    best_f1 = -1.0
    best_metrics = {}
    
    for tau in np.linspace(0.1, 0.9, 17):
        preds = (probs >= tau).astype(int)
        flat_y = targets.astype(int).ravel()
        flat_p = preds.ravel()
        f1 = f1_score(flat_y, flat_p, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_tau = tau
            best_metrics = {
                "threshold": float(tau),
                "best_hour_level_f1": float(f1),
                "best_hour_level_accuracy": float(accuracy_score(flat_y, flat_p)),
                "best_hour_level_precision": float(precision_score(flat_y, flat_p, zero_division=0)),
                "best_hour_level_recall": float(recall_score(flat_y, flat_p, zero_division=0)),
                "best_exact_24h_match_rate": float((preds == targets).all(axis=1).mean()),
                "best_mean_absolute_hour_count_error": float(np.abs(preds.sum(axis=1) - targets.sum(axis=1)).mean()),
                "inference_ms_per_sequence": float(latency_ms),
                "parameters": count_parameters(model)
            }
    return best_metrics, probs, targets

dlinear_variants = [
    ("dlinear_baseline", BaselineDLinear, "Baseline DLinear (Standard k=5)"),
    ("dlinear_multikernel", MultiKernelDLinear, "Multi-Kernel DLinear (k=3,5,7 Multi-Scale)"),
    ("dlinear_channel_gated", ChannelGatedDLinear, "Channel-Gated DLinear (Feature Gating)"),
    ("dlinear_hourly_slot", HourlySlotDLinear, "Hourly Slot-Specific DLinear (24 Heads)"),
    ("dlinear_gated_residual", GatedResidualDLinear, "Gated Residual DLinear (GLU Non-Linear Shortcut)")
]

results = []

print("\nStarting Empirical Benchmarking of DLinear Enhancements...")
for name_id, cls, label in dlinear_variants:
    print(f"\nTraining {label} ({name_id})...", flush=True)
    model = cls().to(device)
    criterion = FocalLoss(gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    
    for ep in range(1, 21):
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
        
        if ep == 20:
            m, probs, targets = evaluate_model(model, val_loader)
            m["name"] = name_id
            m["display_name"] = label
            results.append(m)
            print(f"DONE! F1={m['best_hour_level_f1']:.4f} | Recall={m['best_hour_level_recall']:.4f} | Exact Match={m['best_exact_24h_match_rate']*100:.1f}% | MAE={m['best_mean_absolute_hour_count_error']:.3f} | Params={m['parameters']} | Latency={m['inference_ms_per_sequence']:.4f}ms", flush=True)

out_file = Path("experiments/freshretail_lstm/final_architecture_package/dlinear_tuning_summary.json")
out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"\nSaved DLinear Tuning results to {out_file}", flush=True)
