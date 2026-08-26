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
print(f"=== Non-Linear DLinear Experiments ===")
print(f"Using compute device: {device}\n")

# Load Local Parquet Dataset from Repository
local_train_path = Path("data/train.parquet")
local_top15_path = Path("data/top15_train.parquet")

if local_train_path.exists():
    print(f"Loading local dataset from repository: {local_train_path}")
    raw_df = pd.read_parquet(local_train_path)
elif local_top15_path.exists():
    print(f"Loading local pre-filtered dataset from repository: {local_top15_path}")
    raw_df = pd.read_parquet(local_top15_path)
else:
    print("Local parquet not found. Downloading via HuggingFace datasets...")
    from datasets import load_dataset
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

# Focal Loss for handles class imbalance
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

# Moving Average Series Decomposition
class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
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

# =========================================================
# NON-LINEAR DLINEAR VARIANTS
# =========================================================

# 1. Non-Linear DLinear with GELU Activation Bottleneck
class NonLinearDLinear_GELU(nn.Module):
    def __init__(self, seq_len=14, input_dim=10, hidden_dim=64, kernel_size=5):
        super().__init__()
        self.decomp = SeriesDecomp(kernel_size)
        in_flat = seq_len * input_dim
        
        self.seasonal_mlp = nn.Sequential(
            nn.Linear(in_flat, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 24)
        )
        self.trend_mlp = nn.Sequential(
            nn.Linear(in_flat, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 24)
        )

    def forward(self, x):
        seasonal_init, trend_init = self.decomp(x)
        seasonal_flat = seasonal_init.reshape(x.size(0), -1)
        trend_flat = trend_init.reshape(x.size(0), -1)
        logits = self.seasonal_mlp(seasonal_flat) + self.trend_mlp(trend_flat)
        return logits

# 2. Non-Linear DLinear with SwiGLU Gating
class SwiGLUGatedDLinear(nn.Module):
    def __init__(self, seq_len=14, input_dim=10, hidden_dim=64, kernel_size=5):
        super().__init__()
        self.decomp = SeriesDecomp(kernel_size)
        in_flat = seq_len * input_dim
        
        self.w_s1 = nn.Linear(in_flat, hidden_dim)
        self.w_s2 = nn.Linear(in_flat, hidden_dim)
        self.w_s3 = nn.Linear(hidden_dim, 24)

        self.w_t1 = nn.Linear(in_flat, hidden_dim)
        self.w_t2 = nn.Linear(in_flat, hidden_dim)
        self.w_t3 = nn.Linear(hidden_dim, 24)

    def forward(self, x):
        seasonal_init, trend_init = self.decomp(x)
        s_flat = seasonal_init.reshape(x.size(0), -1)
        t_flat = trend_init.reshape(x.size(0), -1)

        s_hidden = self.w_s1(s_flat) * F.silu(self.w_s2(s_flat))
        s_out = self.w_s3(s_hidden)

        t_hidden = self.w_t1(t_flat) * F.silu(self.w_t2(t_flat))
        t_out = self.w_t3(t_hidden)

        return s_out + t_out

# 3. Non-Linear Hourly Slot-Specific DLinear (24 Non-Linear Slot Heads)
class HourlySlot_NonLinearDLinear(nn.Module):
    def __init__(self, seq_len=14, input_dim=10, hidden_dim=16, kernel_size=5):
        super().__init__()
        self.decomp = SeriesDecomp(kernel_size)
        in_flat = seq_len * input_dim
        
        self.seasonal_slot_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_flat, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(24)
        ])
        
        self.trend_slot_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_flat, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(24)
        ])

    def forward(self, x):
        B = x.size(0)
        s_init, t_init = self.decomp(x)
        s_flat = s_init.reshape(B, -1)
        t_flat = t_init.reshape(B, -1)

        logits = []
        for h in range(24):
            s_h = self.seasonal_slot_mlps[h](s_flat)
            t_h = self.trend_slot_mlps[h](t_flat)
            logits.append(s_h + t_h)

        return torch.cat(logits, dim=1)

# 4. Multi-Kernel Non-Linear DLinear (Multi-Scale Non-Linear Decomposition)
class MultiKernel_NonLinearDLinear(nn.Module):
    def __init__(self, seq_len=14, input_dim=10, hidden_dim=64):
        super().__init__()
        self.decomp3 = SeriesDecomp(3)
        self.decomp5 = SeriesDecomp(5)
        self.decomp7 = SeriesDecomp(7)

        in_flat = seq_len * input_dim

        self.seasonal_mlp = nn.Sequential(
            nn.Linear(in_flat, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 24)
        )
        self.trend3_mlp = nn.Sequential(
            nn.Linear(in_flat, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 24)
        )
        self.trend5_mlp = nn.Sequential(
            nn.Linear(in_flat, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 24)
        )
        self.trend7_mlp = nn.Sequential(
            nn.Linear(in_flat, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 24)
        )

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
        trend_out = w[0] * self.trend3_mlp(t3_flat) + w[1] * self.trend5_mlp(t5_flat) + w[2] * self.trend7_mlp(t7_flat)
        seasonal_out = self.seasonal_mlp(s_flat)

        return trend_out + seasonal_out

# 5. Multi-Kernel Hourly-Slot Non-Linear DLinear
class MultiKernel_Slot_NonLinearDLinear(nn.Module):
    def __init__(self, seq_len=14, input_dim=10, hidden_dim=16):
        super().__init__()
        self.decomp3 = SeriesDecomp(3)
        self.decomp5 = SeriesDecomp(5)
        self.decomp7 = SeriesDecomp(7)
        in_flat = seq_len * input_dim

        self.seasonal_slot_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(in_flat, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)) for _ in range(24)
        ])
        self.trend3_slot_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(in_flat, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)) for _ in range(24)
        ])
        self.trend5_slot_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(in_flat, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)) for _ in range(24)
        ])
        self.trend7_slot_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(in_flat, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)) for _ in range(24)
        ])
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
        logits = []
        for h in range(24):
            s_h = self.seasonal_slot_mlps[h](s_flat)
            t3_h = self.trend3_slot_mlps[h](t3_flat)
            t5_h = self.trend5_slot_mlps[h](t5_flat)
            t7_h = self.trend7_slot_mlps[h](t7_flat)
            trend_h = w[0] * t3_h + w[1] * t5_h + w[2] * t7_h
            logits.append(s_h + trend_h)

        return torch.cat(logits, dim=1)

# Training & Evaluation Function
def train_and_eval(model_class, model_kwargs, display_name, epochs=30):
    set_seed(42)
    model = model_class(**model_kwargs).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    criterion = FocalLoss(gamma=2.0, alpha=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_f1 = 0.0
    best_metrics = {}

    for epoch in range(1, epochs + 1):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                probs = torch.sigmoid(logits)
                all_preds.append(probs.cpu().numpy())
                all_targets.append(by.cpu().numpy())

        preds_arr = np.vstack(all_preds)
        targets_arr = np.vstack(all_targets)

        # Threshold Search
        for tau in np.arange(0.35, 0.65, 0.05):
            binary_preds = (preds_arr >= tau).astype(int)
            f1 = f1_score(targets_arr.ravel(), binary_preds.ravel(), zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                rec = recall_score(targets_arr.ravel(), binary_preds.ravel(), zero_division=0)
                prec = precision_score(targets_arr.ravel(), binary_preds.ravel(), zero_division=0)
                acc = accuracy_score(targets_arr.ravel(), binary_preds.ravel())
                exact_match = np.mean(np.all(binary_preds == targets_arr, axis=1))
                mae = np.mean(np.abs(binary_preds.sum(axis=1) - targets_arr.sum(axis=1)))
                
                best_metrics = {
                    "display_name": display_name,
                    "parameters": params,
                    "best_f1": f1,
                    "recall": rec,
                    "precision": prec,
                    "accuracy": acc,
                    "exact_match": exact_match,
                    "mae_hours": mae,
                    "best_tau": tau
                }

    print(f"[{display_name}] Params: {params:,} | Best F1: {best_metrics['best_f1']:.4f} | Recall: {best_metrics['recall']:.4f} | MAE: {best_metrics['mae_hours']:.2f}h")
    return best_metrics

# Execute Experiment Suite
models_to_test = [
    (NonLinearDLinear_GELU, {"seq_len": 14, "input_dim": 10, "hidden_dim": 64}, "Non-Linear DLinear (GELU Bottleneck)"),
    (SwiGLUGatedDLinear, {"seq_len": 14, "input_dim": 10, "hidden_dim": 64}, "SwiGLU Gated DLinear"),
    (HourlySlot_NonLinearDLinear, {"seq_len": 14, "input_dim": 10, "hidden_dim": 16}, "Non-Linear Hourly Slot DLinear"),
    (MultiKernel_NonLinearDLinear, {"seq_len": 14, "input_dim": 10, "hidden_dim": 64}, "Multi-Kernel Non-Linear DLinear"),
    (MultiKernel_Slot_NonLinearDLinear, {"seq_len": 14, "input_dim": 10, "hidden_dim": 16}, "Multi-Kernel Non-Linear Slot DLinear")
]

results = []
print("\n--- Running Non-Linear DLinear Experiment Suite ---\n")
for m_cls, m_kwargs, name in models_to_test:
    res = train_and_eval(m_cls, m_kwargs, name, epochs=30)
    results.append(res)

df_res = pd.DataFrame(results)
out_csv = "experiments/freshretail_lstm/final_architecture_package/nonlinear_dlinear_results.csv"
df_res.to_csv(out_csv, index=False)
print(f"\nSaved Non-Linear DLinear benchmark results to: {out_csv}\n")
