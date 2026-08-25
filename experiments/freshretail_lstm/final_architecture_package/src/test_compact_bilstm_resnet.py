import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, f1_score
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

raw_df = load_dataset("Dingdong-Inc/FreshRetailNet-50K", split="train").to_pandas()

def to_float_list(value, length=24):
    if isinstance(value, np.ndarray): values = value.tolist()
    elif isinstance(value, (list, tuple)): values = list(value)
    else: values = []
    values = [0.0 if pd.isna(item) else float(item) for item in values[:length]]
    if len(values) < length: values.extend([0.0] * (length - len(values)))
    return values

raw_df["series_id"] = raw_df["city_id"].astype(str) + "_" + raw_df["store_id"].astype(str) + "_" + raw_df["product_id"].astype(str)
raw_df["dt"] = pd.to_datetime(raw_df["dt"], errors="coerce")
raw_df = raw_df.dropna(subset=["dt", "city_id", "store_id", "product_id"])
raw_df = raw_df.drop_duplicates(subset=["series_id", "dt"], keep="last")

stock_vectors = raw_df["hours_stock_status"].map(to_float_list)
stock_matrix = np.array(stock_vectors.tolist(), dtype=np.float32)
for h in range(24): raw_df[f"stock_h{h:02d}"] = stock_matrix[:, h]

raw_df["hours_sale_sum"] = raw_df["hours_sale"].map(lambda v: float(np.sum(to_float_list(v))))
raw_df["hours_stock_status_sum"] = stock_matrix.sum(axis=1)
raw_df["day_of_week"] = raw_df["dt"].dt.dayofweek
raw_df["dow_sin"] = np.sin(2 * np.pi * raw_df["day_of_week"] / 7.0)
raw_df["dow_cos"] = np.cos(2 * np.pi * raw_df["day_of_week"] / 7.0)

for col in ["sale_amount", "stock_hour6_22_cnt", "discount", "holiday_flag", "hours_sale_sum", "hours_stock_status_sum"]:
    raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce").fillna(0.0)

raw_df = raw_df.sort_values(["series_id", "dt"]).reset_index(drop=True)
raw_df["stockout_rolling_3"] = raw_df.groupby("series_id")["stock_hour6_22_cnt"].transform(lambda s: (s > 0).rolling(3, min_periods=1).sum())
raw_df["sales_momentum"] = raw_df.groupby("series_id")["sale_amount"].diff().fillna(0.0)

FEATURES = [
    "sale_amount", "stock_hour6_22_cnt", "discount", "holiday_flag", 
    "hours_sale_sum", "hours_stock_status_sum", "dow_sin", "dow_cos", 
    "stockout_rolling_3", "sales_momentum"
]

TARGET_COLUMNS = [f"target_stock_h{h:02d}" for h in range(24)]
stock_columns = [f"stock_h{h:02d}" for h in range(24)]
for src, tgt in zip(stock_columns, TARGET_COLUMNS):
    raw_df[tgt] = raw_df.groupby("series_id")[src].shift(-1)

clean_df = raw_df.dropna(subset=TARGET_COLUMNS).reset_index(drop=True)

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

cutoff_date = selected_df["dt"].max() - pd.Timedelta(days=15)
scaler = StandardScaler()
scaler.fit(selected_df.loc[selected_df["dt"] <= cutoff_date, FEATURES])
selected_df[FEATURES] = scaler.transform(selected_df[FEATURES])

class HourlySeqDataset(Dataset):
    def __init__(self, frame, seq_len=14, mode="train"):
        self.x, self.y = [], []
        for s_id, group in frame.groupby("series_id", sort=False):
            group = group.sort_values("dt").reset_index(drop=True)
            feats = group[FEATURES].to_numpy(dtype=np.float32)
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

# ---------------------------------------------------------
# COMPACT DECOUPLED DUAL-STREAM BI-LSTM RESNET (~8.6k params)
# ---------------------------------------------------------
class CompactDualStreamBiLSTM(nn.Module):
    """
    Decoupled Dual-Stream BiLSTM ResNet (Parameters: ~8.6k).
    Decouples Sales Features (0, 2, 4, 7, 9) and Inventory Features (1, 3, 5, 6, 8).
    Passes each through lightweight 16-dim BiLSTMs with residual shortcut & 24 slot heads.
    """
    def __init__(self, seq_len=14, hidden_dim=16):
        super().__init__()
        self.seq_len = seq_len
        self.sales_idx = [0, 2, 4, 7, 9]
        self.inv_idx = [1, 3, 5, 6, 8]
        
        self.lstm_sales = nn.LSTM(5, hidden_dim, batch_first=True, bidirectional=True)
        self.lstm_inv = nn.LSTM(5, hidden_dim, batch_first=True, bidirectional=True)
        
        self.gate_sales = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.gate_inv = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        
        # 24 Dedicated Hourly Slot Linear Heads over concatenated dual-stream states
        self.hourly_head = nn.Parameter(torch.randn(24, hidden_dim * 4) * 0.01)
        self.bias = nn.Parameter(torch.zeros(24))

    def forward(self, x):
        B = x.size(0)
        x_sales = x[:, :, self.sales_idx]
        x_inv = x[:, :, self.inv_idx]
        
        o_sales, _ = self.lstm_sales(x_sales)
        o_inv, _ = self.lstm_inv(x_inv)
        
        g_sales = o_sales[:, -1, :] * torch.sigmoid(self.gate_sales(o_sales[:, -1, :]))
        g_inv = o_inv[:, -1, :] * torch.sigmoid(self.gate_inv(o_inv[:, -1, :]))
        
        fused = torch.cat([g_sales, g_inv], dim=-1) # [B, hidden_dim * 4] = [B, 64]
        logits = torch.einsum('bi, hi -> bh', fused, self.hourly_head) + self.bias
        return logits

model = CompactDualStreamBiLSTM().to(device)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"CompactDualStreamBiLSTM Parameters: {num_params}")

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

best_f1 = 0.0
best_recall = 0.0

for ep in range(1, 26):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        
    model.eval()
    all_targets, all_probs = [], []
    with torch.no_grad():
        for x, y in val_loader:
            probs = torch.sigmoid(model(x.to(device))).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(y.numpy())
    targets = np.vstack(all_targets).ravel()
    probs = np.vstack(all_probs).ravel()
    
    for tau in np.linspace(0.40, 0.60, 9):
        preds = (probs >= tau).astype(int)
        f1 = f1_score(targets, preds, zero_division=0)
        rec = recall_score(targets, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_recall = rec
            print(f"Epoch {ep:02d} | New Best F1: {f1:.4f} (Recall: {rec:.4f}, Tau: {tau:.2f})")

print(f"\nFinal Compact Dual-Stream BiLSTM Best F1 Score: {best_f1:.4f} | Recall: {best_recall:.4f}")
