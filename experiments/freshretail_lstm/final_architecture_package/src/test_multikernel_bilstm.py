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

# Multi-Kernel BiLSTM-ResNet Model (~21k params)
class MultiKernelBiLSTM(nn.Module):
    """
    Multi-Scale Moving Average Kernel BiLSTM with 24 Hourly Slot Heads
    """
    def __init__(self, seq_len=14, input_dim=10, hidden_dim=24):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        
        # Multi-scale 1D depthwise convolutions (k=3, 5, 7)
        self.conv3 = nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1, groups=input_dim)
        self.conv5 = nn.Conv1d(input_dim, input_dim, kernel_size=5, padding=2, groups=input_dim)
        self.conv7 = nn.Conv1d(input_dim, input_dim, kernel_size=7, padding=3, groups=input_dim)
        
        # BiLSTM over fused multi-scale features
        self.bilstm = nn.LSTM(input_dim * 3, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        
        # 24 Dedicated Hourly Linear Slot Heads
        self.hourly_head = nn.Parameter(torch.randn(24, seq_len * hidden_dim * 2) * 0.01)
        self.bias = nn.Parameter(torch.zeros(24))

    def forward(self, x):
        # x: [B, L, C]
        B = x.size(0)
        x_t = x.transpose(1, 2)
        c3 = self.conv3(x_t).transpose(1, 2)
        c5 = self.conv5(x_t).transpose(1, 2)
        c7 = self.conv7(x_t).transpose(1, 2)
        
        fused = torch.cat([c3, c5, c7], dim=-1) # [B, L, C*3]
        lstm_out, _ = self.bilstm(fused)       # [B, L, hidden*2]
        
        gated = lstm_out * torch.sigmoid(self.gate(lstm_out))
        flat_feats = gated.reshape(B, -1)     # [B, L * hidden * 2]
        
        logits = torch.einsum('bi, hi -> bh', flat_feats, self.hourly_head) + self.bias
        return logits

model = MultiKernelBiLSTM().to(device)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"MultiKernelBiLSTM Parameters: {num_params}")

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

best_f1 = 0.0
best_recall = 0.0

for ep in range(1, 21):
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
    
    for tau in np.linspace(0.35, 0.65, 13):
        preds = (probs >= tau).astype(int)
        f1 = f1_score(targets, preds, zero_division=0)
        rec = recall_score(targets, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_recall = rec
            print(f"Epoch {ep:02d} | New Best F1: {f1:.4f} (Recall: {rec:.4f}, Tau: {tau:.2f})")

print(f"\nFinal MultiKernelBiLSTM Best F1 Score: {best_f1:.4f} | Recall: {best_recall:.4f}")
