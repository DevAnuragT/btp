from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / ".hf_cache"
os.environ.setdefault("HF_HOME", str(CACHE_DIR))

from datasets import load_dataset


DATASET_NAME = "Dingdong-Inc/FreshRetailNet-50K"
SERIES_COLUMNS = ["city_id", "store_id", "product_id"]
FEATURE_COLUMNS = [
    "sale_amount",
    "stock_hour6_22_cnt",
    "discount",
    "holiday_flag",
    "activity_flag",
    "precpt",
    "avg_temperature",
    "avg_humidity",
    "avg_wind_level",
    "hours_sale_sum",
    "hours_sale_max",
    "hours_stock_status_sum",
    "hours_stock_status_mean",
]
TARGET_COLUMNS = [f"target_stock_h{hour:02d}" for hour in range(24)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-series", type=int, default=15)
    parser.add_argument("--sequence-length", type=int, default=14)
    parser.add_argument("--validation-days", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_float_list(value: object, length: int = 24) -> list[float]:
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


def add_series_id(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["series_id"] = (
        data["city_id"].astype(str)
        + "_"
        + data["store_id"].astype(str)
        + "_"
        + data["product_id"].astype(str)
    )
    return data


def clean_frame(data: pd.DataFrame) -> pd.DataFrame:
    data = add_series_id(data)
    data["dt"] = pd.to_datetime(data["dt"], errors="coerce")
    data = data.dropna(subset=["dt", "city_id", "store_id", "product_id"])
    data = data.drop_duplicates(subset=["series_id", "dt"], keep="last")

    stock_vectors = data["hours_stock_status"].map(to_float_list)
    stock_matrix = np.array(stock_vectors.tolist(), dtype=np.float32)
    for hour in range(24):
        data[f"stock_h{hour:02d}"] = stock_matrix[:, hour]

    data["hours_sale_sum"] = data["hours_sale"].map(lambda value: float(np.sum(to_float_list(value))))
    data["hours_sale_max"] = data["hours_sale"].map(lambda value: float(np.max(to_float_list(value))))
    data["hours_stock_status_sum"] = stock_matrix.sum(axis=1)
    data["hours_stock_status_mean"] = data["hours_stock_status_sum"] / 24.0

    for column in FEATURE_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0.0)

    data = data.sort_values(["series_id", "dt"]).reset_index(drop=True)
    return data


def select_top_series(data: pd.DataFrame, top_series: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        data.groupby("series_id")
        .agg(
            city_id=("city_id", "first"),
            store_id=("store_id", "first"),
            product_id=("product_id", "first"),
            days=("dt", "nunique"),
            first_date=("dt", "min"),
            last_date=("dt", "max"),
            total_sales=("sale_amount", "sum"),
            mean_sales=("sale_amount", "mean"),
            sales_std=("sale_amount", "std"),
            total_stockout_hours=("hours_stock_status_sum", "sum"),
            stockout_days=("stock_hour6_22_cnt", lambda value: int((value > 0).sum())),
        )
        .reset_index()
    )
    summary["sales_std"] = summary["sales_std"].fillna(0.0)
    complete_days = int(summary["days"].max())
    summary = summary[summary["days"] >= complete_days].copy()

    for column in ["total_sales", "sales_std", "total_stockout_hours", "stockout_days"]:
        denominator = summary[column].max() - summary[column].min()
        if denominator == 0:
            summary[f"{column}_score"] = 0.0
        else:
            summary[f"{column}_score"] = (summary[column] - summary[column].min()) / denominator

    summary["selection_score"] = (
        0.40 * summary["total_sales_score"]
        + 0.25 * summary["total_stockout_hours_score"]
        + 0.20 * summary["stockout_days_score"]
        + 0.15 * summary["sales_std_score"]
    )
    selected = (
        summary.sort_values("selection_score", ascending=False)
        .drop_duplicates(subset=["product_id"], keep="first")
        .head(top_series)
    )
    filtered = data[data["series_id"].isin(selected["series_id"])].copy()
    return filtered, selected


def add_next_day_targets(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    stock_columns = [f"stock_h{hour:02d}" for hour in range(24)]
    for source, target in zip(stock_columns, TARGET_COLUMNS):
        data[target] = data.groupby("series_id")[source].shift(-1)
    return data.dropna(subset=TARGET_COLUMNS).reset_index(drop=True)


class Stock24SequenceDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        sequence_length: int,
        cutoff_date: pd.Timestamp,
        mode: str,
    ):
        self.x: list[np.ndarray] = []
        self.y: list[np.ndarray] = []
        self.meta: list[dict[str, object]] = []
        for series_id, group in frame.groupby("series_id", sort=False):
            group = group.sort_values("dt").reset_index(drop=True)
            features = group[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
            targets = group[TARGET_COLUMNS].to_numpy(dtype=np.float32)
            dates = group["dt"].to_numpy()
            if len(group) <= sequence_length:
                continue
            for end in range(sequence_length, len(group)):
                target_date = pd.Timestamp(dates[end])
                if mode == "train" and target_date > cutoff_date:
                    continue
                if mode == "validation" and target_date <= cutoff_date:
                    continue
                self.x.append(features[end - sequence_length : end])
                self.y.append(targets[end - 1])
                self.meta.append({"series_id": series_id, "target_date": str(target_date.date())})

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.x[index], dtype=torch.float32),
            torch.tensor(self.y[index], dtype=torch.float32),
        )


class CIFGLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_linear = nn.Linear(input_size + hidden_size, hidden_size)
        self.candidate_linear = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_linear = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(
        self,
        x_t: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = state
        combined = torch.cat([x_t, h_prev], dim=-1)
        input_gate = torch.sigmoid(self.input_linear(combined))
        forget_gate = 1.0 - input_gate
        candidate = torch.tanh(self.candidate_linear(combined))
        output_gate = torch.sigmoid(self.output_linear(combined))
        c_t = forget_gate * c_prev + input_gate * candidate
        h_t = output_gate * torch.tanh(c_t)
        return h_t, c_t


class CIFGLSTMLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = CIFGLSTMCell(input_size, hidden_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch_size, sequence_length, _ = x.shape
        h_t = x.new_zeros(batch_size, self.hidden_size)
        c_t = x.new_zeros(batch_size, self.hidden_size)
        outputs = []
        for step in range(sequence_length):
            h_t, c_t = self.cell(x[:, step, :], (h_t, c_t))
            outputs.append(h_t.unsqueeze(1))
        return torch.cat(outputs, dim=1), (h_t, c_t)


class Stock24LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(hidden_size, 24))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        return self.head(hidden[-1])


class Stock24CIFGLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.lstm = CIFGLSTMLayer(input_size=input_size, hidden_size=hidden_size)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(hidden_size, 24))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        return self.head(hidden)


class Stock24DualStreamLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        demand_indices: list[int],
        inventory_indices: list[int],
        demand_hidden_size: int | None = None,
        inventory_hidden_size: int | None = None,
    ):
        super().__init__()
        if not demand_indices or not inventory_indices:
            raise ValueError("Dual-stream LSTM requires non-empty demand and inventory features.")
        if demand_hidden_size is None and inventory_hidden_size is None:
            if hidden_size % 2 != 0:
                raise ValueError("Dual-stream LSTM requires an even hidden_size.")
            demand_hidden_size = hidden_size // 2
            inventory_hidden_size = hidden_size // 2
        elif demand_hidden_size is None or inventory_hidden_size is None:
            raise ValueError("Dual-stream LSTM requires both branch hidden sizes when using a custom split.")
        if demand_hidden_size + inventory_hidden_size != hidden_size:
            raise ValueError("Dual-stream hidden sizes must sum to hidden_size.")

        self.demand_indices = demand_indices
        self.inventory_indices = inventory_indices
        self.demand_lstm = nn.LSTM(
            input_size=len(demand_indices),
            hidden_size=demand_hidden_size,
            batch_first=True,
        )
        self.inventory_lstm = nn.LSTM(
            input_size=len(inventory_indices),
            hidden_size=inventory_hidden_size,
            batch_first=True,
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.2),
        )
        self.head = nn.Linear(hidden_size, 24)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        demand_x = x[:, :, self.demand_indices]
        inventory_x = x[:, :, self.inventory_indices]
        _, (demand_hidden, _) = self.demand_lstm(demand_x)
        _, (inventory_hidden, _) = self.inventory_lstm(inventory_x)
        fused = torch.cat([demand_hidden[-1], inventory_hidden[-1]], dim=-1)
        return self.head(self.fuse(fused))


class Stock24DualStreamCIFGLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        demand_hidden_size: int,
        inventory_hidden_size: int,
        demand_indices: list[int],
        inventory_indices: list[int],
    ):
        super().__init__()
        if not demand_indices or not inventory_indices:
            raise ValueError("Dual-stream CIFG LSTM requires non-empty demand and inventory features.")

        total_hidden = demand_hidden_size + inventory_hidden_size
        self.demand_indices = demand_indices
        self.inventory_indices = inventory_indices
        self.demand_lstm = CIFGLSTMLayer(
            input_size=len(demand_indices),
            hidden_size=demand_hidden_size,
        )
        self.inventory_lstm = CIFGLSTMLayer(
            input_size=len(inventory_indices),
            hidden_size=inventory_hidden_size,
        )
        self.fuse = nn.Sequential(
            nn.Linear(total_hidden, total_hidden),
            nn.Tanh(),
            nn.Dropout(0.2),
        )
        self.head = nn.Linear(total_hidden, 24)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        demand_x = x[:, :, self.demand_indices]
        inventory_x = x[:, :, self.inventory_indices]
        _, (demand_hidden, _) = self.demand_lstm(demand_x)
        _, (inventory_hidden, _) = self.inventory_lstm(inventory_x)
        fused = torch.cat([demand_hidden, inventory_hidden], dim=-1)
        return self.head(self.fuse(fused))


class Stock24DualStreamShortcutLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        demand_indices: list[int],
        inventory_indices: list[int],
        demand_hidden_size: int | None = None,
        inventory_hidden_size: int | None = None,
    ):
        super().__init__()
        if not demand_indices or not inventory_indices:
            raise ValueError("Dual-stream shortcut LSTM requires non-empty demand and inventory features.")
        if demand_hidden_size is None and inventory_hidden_size is None:
            if hidden_size % 2 != 0:
                raise ValueError("Dual-stream shortcut LSTM requires an even hidden_size.")
            demand_hidden_size = hidden_size // 2
            inventory_hidden_size = hidden_size // 2
        elif demand_hidden_size is None or inventory_hidden_size is None:
            raise ValueError("Dual-stream shortcut LSTM requires both branch hidden sizes when using a custom split.")
        if demand_hidden_size + inventory_hidden_size != hidden_size:
            raise ValueError("Dual-stream shortcut hidden sizes must sum to hidden_size.")

        self.demand_indices = demand_indices
        self.inventory_indices = inventory_indices
        self.demand_lstm = nn.LSTM(
            input_size=len(demand_indices),
            hidden_size=demand_hidden_size,
            batch_first=True,
        )
        self.inventory_lstm = nn.LSTM(
            input_size=len(inventory_indices),
            hidden_size=inventory_hidden_size,
            batch_first=True,
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden_size + input_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.2),
        )
        self.head = nn.Linear(hidden_size, 24)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        demand_x = x[:, :, self.demand_indices]
        inventory_x = x[:, :, self.inventory_indices]
        _, (demand_hidden, _) = self.demand_lstm(demand_x)
        _, (inventory_hidden, _) = self.inventory_lstm(inventory_x)
        fused = torch.cat([demand_hidden[-1], inventory_hidden[-1]], dim=-1)
        last_day_features = x[:, -1, :]
        shortcut_fused = torch.cat([fused, last_day_features], dim=-1)
        return self.head(self.fuse(shortcut_fused))


class Stock24DualStreamInventoryShortcutLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        demand_indices: list[int],
        inventory_indices: list[int],
        demand_hidden_size: int | None = None,
        inventory_hidden_size: int | None = None,
    ):
        super().__init__()
        if not demand_indices or not inventory_indices:
            raise ValueError("Inventory shortcut LSTM requires non-empty demand and inventory features.")
        if demand_hidden_size is None and inventory_hidden_size is None:
            if hidden_size % 2 != 0:
                raise ValueError("Inventory shortcut LSTM requires an even hidden_size.")
            demand_hidden_size = hidden_size // 2
            inventory_hidden_size = hidden_size // 2
        elif demand_hidden_size is None or inventory_hidden_size is None:
            raise ValueError("Inventory shortcut LSTM requires both branch hidden sizes when using a custom split.")
        if demand_hidden_size + inventory_hidden_size != hidden_size:
            raise ValueError("Inventory shortcut hidden sizes must sum to hidden_size.")

        self.demand_indices = demand_indices
        self.inventory_indices = inventory_indices
        self.demand_lstm = nn.LSTM(
            input_size=len(demand_indices),
            hidden_size=demand_hidden_size,
            batch_first=True,
        )
        self.inventory_lstm = nn.LSTM(
            input_size=len(inventory_indices),
            hidden_size=inventory_hidden_size,
            batch_first=True,
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden_size + len(inventory_indices), hidden_size),
            nn.Tanh(),
            nn.Dropout(0.2),
        )
        self.head = nn.Linear(hidden_size, 24)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        demand_x = x[:, :, self.demand_indices]
        inventory_x = x[:, :, self.inventory_indices]
        _, (demand_hidden, _) = self.demand_lstm(demand_x)
        _, (inventory_hidden, _) = self.inventory_lstm(inventory_x)
        fused = torch.cat([demand_hidden[-1], inventory_hidden[-1]], dim=-1)
        last_inventory = x[:, -1, self.inventory_indices]
        shortcut_fused = torch.cat([fused, last_inventory], dim=-1)
        return self.head(self.fuse(shortcut_fused))


class Stock24DualStreamInventoryEmbeddingLSTM(Stock24DualStreamInventoryShortcutLSTM):
    """Inventory-shortcut model augmented with static store/product context."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        demand_indices: list[int],
        inventory_indices: list[int],
        store_count: int,
        product_count: int,
        embedding_size: int = 4,
        demand_hidden_size: int | None = None,
        inventory_hidden_size: int | None = None,
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            demand_indices=demand_indices,
            inventory_indices=inventory_indices,
            demand_hidden_size=demand_hidden_size,
            inventory_hidden_size=inventory_hidden_size,
        )
        self.store_embedding = nn.Embedding(store_count, embedding_size)
        self.product_embedding = nn.Embedding(product_count, embedding_size)
        self.static_embedding_size = embedding_size * 2
        self.fuse = nn.Sequential(
            nn.Linear(hidden_size + len(inventory_indices) + self.static_embedding_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.2),
        )

    def forward(
        self,
        x: torch.Tensor,
        store_indices: torch.Tensor,
        product_indices: torch.Tensor,
    ) -> torch.Tensor:
        demand_x = x[:, :, self.demand_indices]
        inventory_x = x[:, :, self.inventory_indices]
        _, (demand_hidden, _) = self.demand_lstm(demand_x)
        _, (inventory_hidden, _) = self.inventory_lstm(inventory_x)
        fused = torch.cat([demand_hidden[-1], inventory_hidden[-1]], dim=-1)
        last_inventory = x[:, -1, self.inventory_indices]
        static_context = torch.cat(
            [self.store_embedding(store_indices), self.product_embedding(product_indices)],
            dim=-1,
        )
        shortcut_fused = torch.cat([fused, last_inventory, static_context], dim=-1)
        return self.head(self.fuse(shortcut_fused))


class Stock24DualStreamGatedShortcutLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        demand_indices: list[int],
        inventory_indices: list[int],
        demand_hidden_size: int | None = None,
        inventory_hidden_size: int | None = None,
    ):
        super().__init__()
        if not demand_indices or not inventory_indices:
            raise ValueError("Gated shortcut LSTM requires non-empty demand and inventory features.")
        if demand_hidden_size is None and inventory_hidden_size is None:
            if hidden_size % 2 != 0:
                raise ValueError("Gated shortcut LSTM requires an even hidden_size.")
            demand_hidden_size = hidden_size // 2
            inventory_hidden_size = hidden_size // 2
        elif demand_hidden_size is None or inventory_hidden_size is None:
            raise ValueError("Gated shortcut LSTM requires both branch hidden sizes when using a custom split.")
        if demand_hidden_size + inventory_hidden_size != hidden_size:
            raise ValueError("Gated shortcut hidden sizes must sum to hidden_size.")

        self.demand_indices = demand_indices
        self.inventory_indices = inventory_indices
        self.demand_lstm = nn.LSTM(
            input_size=len(demand_indices),
            hidden_size=demand_hidden_size,
            batch_first=True,
        )
        self.inventory_lstm = nn.LSTM(
            input_size=len(inventory_indices),
            hidden_size=inventory_hidden_size,
            batch_first=True,
        )
        self.shortcut_proj = nn.Linear(input_size, hidden_size)
        self.gate = nn.Linear(hidden_size * 2, hidden_size)
        self.post = nn.Sequential(
            nn.Tanh(),
            nn.Dropout(0.2),
        )
        self.head = nn.Linear(hidden_size, 24)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        demand_x = x[:, :, self.demand_indices]
        inventory_x = x[:, :, self.inventory_indices]
        _, (demand_hidden, _) = self.demand_lstm(demand_x)
        _, (inventory_hidden, _) = self.inventory_lstm(inventory_x)
        memory_fused = torch.cat([demand_hidden[-1], inventory_hidden[-1]], dim=-1)
        shortcut = self.shortcut_proj(x[:, -1, :])
        gate = torch.sigmoid(self.gate(torch.cat([memory_fused, shortcut], dim=-1)))
        fused = gate * memory_fused + (1.0 - gate) * shortcut
        return self.head(self.post(fused))


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    losses = []
    for batch in loader:
        if len(batch) == 2:
            features, labels = batch
            model_inputs = (features.to(device),)
        else:
            features, store_indices, product_indices, labels = batch
            model_inputs = (
                features.to(device),
                store_indices.to(device),
                product_indices.to(device),
            )
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(*model_inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels = []
    probabilities = []
    for batch in loader:
        if len(batch) == 2:
            features, target = batch
            model_inputs = (features.to(device),)
        else:
            features, store_indices, product_indices, target = batch
            model_inputs = (
                features.to(device),
                store_indices.to(device),
                product_indices.to(device),
            )
        logits = model(*model_inputs)
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(target.numpy())
    return np.vstack(labels), np.vstack(probabilities)


def compute_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)
    flat_labels = labels.astype(int).ravel()
    flat_predictions = predictions.ravel()
    return {
        "hour_level_accuracy": float(accuracy_score(flat_labels, flat_predictions)),
        "hour_level_precision": float(precision_score(flat_labels, flat_predictions, zero_division=0)),
        "hour_level_recall": float(recall_score(flat_labels, flat_predictions, zero_division=0)),
        "hour_level_f1": float(f1_score(flat_labels, flat_predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(flat_labels, probabilities.ravel())),
        "exact_24h_match_rate": float((predictions == labels).all(axis=1).mean()),
        "mean_absolute_hour_count_error": float(
            np.abs(predictions.sum(axis=1) - labels.sum(axis=1)).mean()
        ),
    }


def save_prediction_sample(
    dataset: Stock24SequenceDataset,
    labels: np.ndarray,
    probabilities: np.ndarray,
    path: Path,
    limit: int = 60,
) -> None:
    predictions = (probabilities >= 0.5).astype(int)
    rows = []
    for index, meta in enumerate(dataset.meta[:limit]):
        rows.append(
            {
                **meta,
                "actual_stockout_hours": int(labels[index].sum()),
                "predicted_stockout_hours": int(predictions[index].sum()),
                "actual_24h_vector": labels[index].astype(int).tolist(),
                "predicted_24h_vector": predictions[index].astype(int).tolist(),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    RESULTS_DIR.mkdir(exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    raw = load_dataset(DATASET_NAME, split="train").to_pandas()
    clean = clean_frame(raw)
    selected_frame, selected_summary = select_top_series(clean, args.top_series)
    selected_frame = add_next_day_targets(selected_frame)
    cutoff_date = selected_frame["dt"].max() - pd.Timedelta(days=args.validation_days)

    scaler = StandardScaler()
    scaler.fit(selected_frame.loc[selected_frame["dt"] <= cutoff_date, FEATURE_COLUMNS])
    selected_frame[FEATURE_COLUMNS] = scaler.transform(selected_frame[FEATURE_COLUMNS])

    train_dataset = Stock24SequenceDataset(
        selected_frame, args.sequence_length, cutoff_date, "train"
    )
    validation_dataset = Stock24SequenceDataset(
        selected_frame, args.sequence_length, cutoff_date, "validation"
    )
    if not train_dataset or not validation_dataset:
        raise ValueError("No sequences created. Reduce sequence length or validation days.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)

    positive_hours = float(np.vstack(train_dataset.y).sum())
    total_hours = float(np.vstack(train_dataset.y).size)
    negative_hours = total_hours - positive_hours
    pos_weight = torch.tensor([negative_hours / max(positive_hours, 1.0)] * 24, device=device)

    model = Stock24LSTM(len(FEATURE_COLUMNS), args.hidden_size).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    history = []
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        labels, probabilities = predict(model, validation_loader, device)
        metrics = compute_metrics(labels, probabilities)
        metrics["epoch"] = epoch
        metrics["train_loss"] = loss
        history.append(metrics)
        print(
            f"epoch={epoch} loss={loss:.4f} "
            f"hour_f1={metrics['hour_level_f1']:.4f} "
            f"hour_recall={metrics['hour_level_recall']:.4f} "
            f"count_mae={metrics['mean_absolute_hour_count_error']:.3f}"
        )

    selected_summary_path = RESULTS_DIR / "top_sku_series.csv"
    prediction_path = RESULTS_DIR / "top_sku_24h_predictions_sample.csv"
    metrics_path = RESULTS_DIR / "top_sku_24h_lstm_metrics.json"

    selected_summary.to_csv(selected_summary_path, index=False)
    labels, probabilities = predict(model, validation_loader, device)
    save_prediction_sample(validation_dataset, labels, probabilities, prediction_path)

    output = {
        "dataset": DATASET_NAME,
        "task": "Predict next-day 24-hour stock status vector for statistically selected SKU-store series.",
        "selection": {
            "unit": "distinct product_id SKU, represented by its highest-scoring city-store-product series",
            "top_series": args.top_series,
            "score": "0.40 total_sales + 0.25 total_stockout_hours + 0.20 stockout_days + 0.15 sales_std, min-max normalized",
        },
        "split": {
            "available_days": int(selected_frame["dt"].nunique()),
            "train_until": str(cutoff_date.date()),
            "validation_days": args.validation_days,
        },
        "features": FEATURE_COLUMNS,
        "target": "24 binary values from next day's hours_stock_status",
        "sequence_length": args.sequence_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_size": args.hidden_size,
        "seed": args.seed,
        "device": str(device),
        "train_sequences": len(train_dataset),
        "validation_sequences": len(validation_dataset),
        "train_positive_hour_rate": positive_hours / total_hours,
        "history": history,
        "final_metrics": history[-1],
        "artifacts": {
            "selected_series": str(selected_summary_path.relative_to(ROOT)),
            "prediction_sample": str(prediction_path.relative_to(ROOT)),
        },
    }
    metrics_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved selected SKU-series to {selected_summary_path}")
    print(f"Saved prediction sample to {prediction_path}")


if __name__ == "__main__":
    main()
