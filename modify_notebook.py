import json

path = "notebooks/04_rnn_attention_models.ipynb"
with open(path, "r") as f:
    nb = json.load(f)

# Update Config Cell (Cell 2)
cell_2_src = """@dataclass
class Config:
    history_len: int = 28
    forecast_len: int = 7
    val_days: int = 15
    batch_size: int = 256
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 128
    embed_dim: int = 32
    num_layers: int = 2
    dropout: float = 0.2
    patience: int = 4
    seed: int = 42
    use_pos_weight: bool = True

CFG = Config()

STATIC_CAT_COLS = ["city_id", "store_id", "management_group_id", "first_category_id", "second_category_id", "third_category_id", "product_id"]
TEMPORAL_CAT_COLS = ["holiday_flag", "activity_flag"]
CAT_COLS = STATIC_CAT_COLS + TEMPORAL_CAT_COLS
HIST_NUM_COLS = ["discount", "precpt", "avg_temperature", "avg_humidity", "avg_wind_level", "sale_amount", "stockout", "stockout_streak", "stockout_rolling_7", "sale_amount_lag1"]
FUTURE_NUM_COLS = ["discount", "precpt", "avg_temperature", "avg_humidity", "avg_wind_level"]
TARGET_COL = "stockout"

def seed_everything(seed: int):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(CFG.seed)
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
"""
nb["cells"][2]["source"] = [line + "\n" if i < len(cell_2_src.split("\n"))-1 else line for i, line in enumerate(cell_2_src.split("\n"))]

# Update Data Loading Cell (Cell 4)
new_funcs = """def add_engineered_features(df):
    df = df.sort_values(["store_id", "product_id", "dt"]).reset_index(drop=True)
    def streak(s):
        groups = (s != s.shift()).cumsum()
        return s.groupby(groups).cumcount() + 1
    df["stockout_streak"] = df.groupby(["store_id", "product_id"])[TARGET_COL].transform(streak)
    df.loc[df[TARGET_COL] == 0, "stockout_streak"] = 0
    df["stockout_rolling_7"] = df.groupby(["store_id", "product_id"])[TARGET_COL].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df["sale_amount_lag1"] = df.groupby(["store_id", "product_id"])["sale_amount"].shift(1).fillna(0)
    return df

def fit_apply_preprocessors(train_df, test_df, train_end):
    combined = pd.concat([train_df.assign(source="train"), test_df.assign(source="test")], ignore_index=True)
    combined = add_engineered_features(combined)
    fit_rows = combined[(combined["source"] == "train") & (combined["dt"] <= train_end)].copy()
    cat_encoder = CategoryEncoder(CAT_COLS).fit(fit_rows)
    scale_cols = sorted((set(HIST_NUM_COLS + FUTURE_NUM_COLS) - {TARGET_COL}))
    scaler = StandardScaler().fit(fit_rows[scale_cols].fillna(0.0))
    combined = cat_encoder.transform(combined)
    combined[scale_cols] = scaler.transform(combined[scale_cols].fillna(0.0))
    return combined, cat_encoder
"""

# Extract old source, find and replace fit_apply_preprocessors
old_src = "".join(nb["cells"][4]["source"])
import re
new_src = re.sub(r'def fit_apply_preprocessors.*?return combined, cat_encoder\n', new_funcs, old_src, flags=re.DOTALL)
nb["cells"][4]["source"] = [line + "\n" if i < len(new_src.split("\n"))-1 else line for i, line in enumerate(new_src.split("\n"))]

with open(path, "w") as f:
    json.dump(nb, f, indent=1)
