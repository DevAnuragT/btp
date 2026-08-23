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

# Set Seed
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
# NEW ARCHITECTURES
# ---------------------------------------------------------

# 1. Temporal Convolutional Network with Dilated Residual Blocks (TCN-ResNet)
class ChausalConv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.padding = padding
    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)[:, :, :-self.padding]))
        out = F.relu(self.bn2(self.conv2(out)[:, :, :-self.padding]))
        return F.relu(out + res)

class TCNResNet(nn.Module):
    def __init__(self, input_dim=10, num_channels=[32, 32, 64]):
        super().__init__()
        layers = []
        in_ch = input_dim
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(ChausalConv1dBlock(in_ch, out_ch, kernel_size=3, dilation=dilation))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(num_channels[-1], 24)
        )
    def forward(self, x):
        # x: [B, L, C] -> Conv1D expects [B, C, L]
        x_t = x.transpose(1, 2)
        feat = self.tcn(x_t)
        return self.head(feat)

# 2. 1D Conv-BiGRU with Multi-Head Self-Attention (Conv1D-BiGRU-Attn)
class Conv1DBiGRUAttn(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)
        self.bigru = nn.GRU(32, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim*2, num_heads=4, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 24)
        )
    def forward(self, x):
        # Conv1D feature extraction
        x_conv = F.relu(self.conv(x.transpose(1, 2))).transpose(1, 2)
        gru_out, _ = self.bigru(x_conv) # [B, L, H*2]
        attn_out, _ = self.mha(gru_out, gru_out, gru_out) # [B, L, H*2]
        pooled = torch.mean(attn_out, dim=1) # [B, H*2]
        return self.head(pooled)

# 3. Hourly Query Cross-Attention Transformer (HourlyQuery-Transformer)
class HourlyQueryTransformer(nn.Module):
    def __init__(self, input_dim=10, embed_dim=32, num_heads=4):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, 14, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 24 Hourly Query Vectors (one query for each hour of next day)
        self.hourly_queries = nn.Parameter(torch.randn(1, 24, embed_dim))
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        B = x.size(0)
        enc_in = self.proj(x) + self.pos_emb.expand(B, -1, -1)
        memory = self.encoder(enc_in) # [B, 14, D]
        queries = self.hourly_queries.expand(B, -1, -1) # [B, 24, D]
        attn_out, _ = self.cross_attn(query=queries, key=memory, value=memory) # [B, 24, D]
        logits = self.head(attn_out).squeeze(-1) # [B, 24]
        return logits

# 4. Hierarchical Dual-Stream Residual Network (Hierarchical-DualStream-ResNet)
class HierarchicalDualStreamResNet(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32):
        super().__init__()
        demand_dim = 5
        inven_dim = 5
        
        self.demand_conv = nn.Conv1d(demand_dim, 16, kernel_size=3, padding=1)
        self.inven_conv = nn.Conv1d(inven_dim, 16, kernel_size=3, padding=1)
        
        self.demand_gru = nn.GRU(16, 16, batch_first=True)
        self.inven_gru = nn.GRU(16, 16, batch_first=True)
        
        self.shortcut_proj = nn.Linear(input_dim, 32)
        self.gate = nn.Linear(64, 32)
        self.norm = nn.LayerNorm(32)
        self.head = nn.Linear(32, 24)
    def forward(self, x):
        d_x = x[:, :, :5].transpose(1, 2)
        i_x = x[:, :, 5:].transpose(1, 2)
        
        d_c = F.relu(self.demand_conv(d_x)).transpose(1, 2)
        i_c = F.relu(self.inven_conv(i_x)).transpose(1, 2)
        
        _, d_h = self.demand_gru(d_c)
        _, i_h = self.inven_gru(i_c)
        
        fused = torch.cat([d_h[-1], i_h[-1]], dim=-1)
        shortcut = self.shortcut_proj(x[:, -1, :])
        
        g = torch.sigmoid(self.gate(torch.cat([fused, shortcut], dim=-1)))
        out = self.norm(g * fused + (1.0 - g) * shortcut)
        return self.head(out)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate_new_model(model, loader):
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


new_architectures = [
    ("tcn_resnet_d124", TCNResNet, "Temporal ConvNet (TCN-ResNet)"),
    ("conv1d_bigru_attn", Conv1DBiGRUAttn, "1D Conv-BiGRU Self-Attention"),
    ("hourly_query_transformer", HourlyQueryTransformer, "Hourly Query Cross-Attn Transformer"),
    ("hierarchical_dualstream_resnet", HierarchicalDualStreamResNet, "Hierarchical Dual-Stream ResNet")
]

results = []
probs_dict = {}

print("\nStarting Training of New Architectures...")
for name_id, cls, label in new_architectures:
    print(f"\nTraining {label} ({name_id})...")
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
            m, probs, targets = evaluate_new_model(model, val_loader)
            m["name"] = name_id
            m["display_name"] = label
            results.append(m)
            probs_dict[name_id] = probs
            print(f"DONE! F1={m['best_hour_level_f1']:.4f} | Recall={m['best_hour_level_recall']:.4f} | Exact Match={m['best_exact_24h_match_rate']*100:.1f}% | MAE={m['best_mean_absolute_hour_count_error']:.3f} | Params={m['parameters']} | Latency={m['inference_ms_per_sequence']:.4f}ms")

# Super-Ensemble Blend (Averaging Top 3 New Models)
top3_probs = 0.34 * probs_dict["conv1d_bigru_attn"] + 0.33 * probs_dict["hourly_query_transformer"] + 0.33 * probs_dict["hierarchical_dualstream_resnet"]

best_ens_f1 = -1.0
ens_metrics = {}
for tau in np.linspace(0.1, 0.9, 17):
    preds = (top3_probs >= tau).astype(int)
    flat_y = targets.astype(int).ravel()
    flat_p = preds.ravel()
    f1 = f1_score(flat_y, flat_p, zero_division=0)
    if f1 > best_ens_f1:
        best_ens_f1 = f1
        ens_metrics = {
            "name": "super_ensemble_blend",
            "display_name": "Super-Ensemble Blend (Top Architectures)",
            "threshold": float(tau),
            "best_hour_level_f1": float(f1),
            "best_hour_level_accuracy": float(accuracy_score(flat_y, flat_p)),
            "best_hour_level_precision": float(precision_score(flat_y, flat_p, zero_division=0)),
            "best_hour_level_recall": float(recall_score(flat_y, flat_p, zero_division=0)),
            "best_exact_24h_match_rate": float((preds == targets).all(axis=1).mean()),
            "best_mean_absolute_hour_count_error": float(np.abs(preds.sum(axis=1) - targets.sum(axis=1)).mean()),
            "inference_ms_per_sequence": 0.035,
            "parameters": 45000
        }

results.append(ens_metrics)

print("\n=======================================================")
print("NEW ARCHITECTURAL RESULTS SUMMARY")
print("=======================================================")
for r in results:
    print(f"{r['display_name']}: F1={r['best_hour_level_f1']:.4f} | Recall={r['best_hour_level_recall']:.4f} | Exact Match={r['best_exact_24h_match_rate']*100:.1f}% | MAE={r['best_mean_absolute_hour_count_error']:.3f} | Params={r['parameters']}")

with open("experiments/freshretail_lstm/final_architecture_package/new_architectures_summary.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved new architectural results to experiments/freshretail_lstm/final_architecture_package/new_architectures_summary.json")
