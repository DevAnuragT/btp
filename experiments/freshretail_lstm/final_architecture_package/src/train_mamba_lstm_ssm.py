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

# ---------------------------------------------------------
# ADVANCED STATE SEQUENCE & DEEP LSTM ARCHITECTURES
# ---------------------------------------------------------

# 1. Selective Mamba-SSM (Structured State Space Model)
class SelectiveMambaSSMBlock(nn.Module):
    """
    Selective State Space Model (Mamba-style SSM).
    Input-dependent step-size Delta(x_t) and continuous-to-discrete state transition matrices.
    """
    def __init__(self, d_model=64, state_dim=16):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        
        # State matrices A, B, C
        self.A_log = nn.Parameter(torch.log(torch.arange(1, state_dim + 1, dtype=torch.float32).repeat(d_model, 1)))
        self.B_proj = nn.Linear(d_model, state_dim, bias=False)
        self.C_proj = nn.Linear(d_model, state_dim, bias=False)
        self.delta_proj = nn.Linear(d_model, d_model, bias=True)

        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.out_proj = nn.Linear(d_model, d_model)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)

    def forward(self, x):
        # x: [B, L, D]
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_val, z_gate = torch.chunk(xz, 2, dim=-1)

        x_conv = F.silu(self.conv1d(x_val.transpose(1, 2)).transpose(1, 2))
        
        # Input-dependent Delta(x_t)
        delta = F.softplus(self.delta_proj(x_conv)) # [B, L, D]
        A = -torch.exp(self.A_log) # [D, N]
        B_t = self.B_proj(x_conv)   # [B, L, N]
        C_t = self.C_proj(x_conv)   # [B, L, N]

        # Recurrent State-Space discretization scan
        h = torch.zeros(B, D, self.state_dim, device=x.device)
        ys = []
        for t in range(L):
            # Discretize A_bar = exp(delta * A)
            d_t = delta[:, t, :].unsqueeze(-1) # [B, D, 1]
            A_bar = torch.exp(d_t * A.unsqueeze(0)) # [B, D, N]
            B_bar = d_t * B_t[:, t, :].unsqueeze(1) # [B, D, N]
            x_t = x_conv[:, t, :].unsqueeze(-1)     # [B, D, 1]

            # State recurrence: h_t = A_bar * h_{t-1} + B_bar * x_t
            h = A_bar * h + B_bar * x_t             # [B, D, N]
            
            # Output: y_t = C_t * h_t
            y_t = torch.sum(h * C_t[:, t, :].unsqueeze(1), dim=-1) # [B, D]
            ys.append(y_t.unsqueeze(1))

        y_ssm = torch.cat(ys, dim=1) # [B, L, D]
        y_gated = y_ssm * F.silu(z_gate)
        return self.out_proj(y_gated)

class MambaSSMModel(nn.Module):
    def __init__(self, input_dim=10, d_model=64, state_dim=16):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, d_model)
        self.ssm1 = SelectiveMambaSSMBlock(d_model=d_model, state_dim=state_dim)
        self.ssm2 = SelectiveMambaSSMBlock(d_model=d_model, state_dim=state_dim)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 24 hourly slot cross-attention queries
        self.hourly_queries = nn.Parameter(torch.randn(24, d_model))
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        # x: [B, L, C]
        B = x.size(0)
        h = self.in_proj(x)
        h = h + self.ssm1(self.norm1(h))
        h = h + self.ssm2(self.norm2(h))

        # Hourly slot cross-attention
        queries = self.hourly_queries.unsqueeze(0).expand(B, -1, -1) # [B, 24, D]
        attn_out, _ = self.cross_attn(queries, h, h) # [B, 24, D]
        logits = self.head(attn_out).squeeze(-1) # [B, 24]
        return logits

# 2. Deep Residual BiLSTM with Gated Linear Projection & Cross-Attention (`MambaBiLSTM-ResNet`)
class ResidualBiLSTMBlock(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.lstm = nn.LSTM(d_model, d_model // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.gate_proj = nn.Linear(d_model, d_model * 2)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        # x: [B, L, D]
        res = x
        lstm_out, _ = self.lstm(x) # [B, L, D]
        val, gate = torch.chunk(self.gate_proj(lstm_out), 2, dim=-1)
        gated = val * torch.sigmoid(gate)
        return self.norm(res + gated)

class DeepBiLSTMResNetModel(nn.Module):
    def __init__(self, input_dim=10, d_model=64):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, d_model)
        self.b1 = ResidualBiLSTMBlock(d_model=d_model)
        self.b2 = ResidualBiLSTMBlock(d_model=d_model)
        self.b3 = ResidualBiLSTMBlock(d_model=d_model)
        
        self.hourly_queries = nn.Parameter(torch.randn(24, d_model))
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        B = x.size(0)
        h = self.in_proj(x)
        h = self.b1(h)
        h = self.b2(h)
        h = self.b3(h)

        queries = self.hourly_queries.unsqueeze(0).expand(B, -1, -1)
        attn_out, _ = self.cross_attn(queries, h, h)
        logits = self.head(attn_out).squeeze(-1)
        return logits

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

new_state_models = [
    ("mamba_ssm_selective", MambaSSMModel, "Mamba-SSM (Selective State Space Model)"),
    ("deep_bilstm_resnet", DeepBiLSTMResNetModel, "Deep BiLSTM-ResNet (Gated Cross-Attn)")
]

results = []

print("\nStarting Training of Advanced State Sequence & LSTM Models...")
for name_id, cls, label in new_state_models:
    print(f"\nTraining {label} ({name_id})...", flush=True)
    model = cls().to(device)
    criterion = FocalLoss(gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    
    for ep in range(1, 16):
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
        
        if ep == 15:
            m, probs, targets = evaluate_model(model, val_loader)
            m["name"] = name_id
            m["display_name"] = label
            results.append(m)
            print(f"DONE! F1={m['best_hour_level_f1']:.4f} | Recall={m['best_hour_level_recall']:.4f} | Exact Match={m['best_exact_24h_match_rate']*100:.1f}% | MAE={m['best_mean_absolute_hour_count_error']:.3f} | Params={m['parameters']} | Latency={m['inference_ms_per_sequence']:.4f}ms", flush=True)

out_file = Path("experiments/freshretail_lstm/final_architecture_package/mamba_ssm_summary.json")
out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"\nSaved State Sequence Model results to {out_file}", flush=True)
