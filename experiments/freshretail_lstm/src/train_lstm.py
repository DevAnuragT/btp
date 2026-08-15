from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
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
BASE_FEATURES = [
    "sale_amount",
    "stock_hour6_22_cnt",
    "discount",
    "holiday_flag",
    "activity_flag",
    "precpt",
    "avg_temperature",
    "avg_humidity",
    "avg_wind_level",
]
DERIVED_FEATURES = [
    "hours_sale_sum",
    "hours_sale_max",
    "hours_stock_status_sum",
    "hours_stock_status_mean",
]
FEATURE_COLUMNS = BASE_FEATURES + DERIVED_FEATURES
TARGET_COLUMN = "next_day_stockout"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-series", type=int, default=500)
    parser.add_argument("--sequence-length", type=int, default=14)
    parser.add_argument("--validation-days", type=int, default=14)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def list_sum(value: object) -> float:
    if isinstance(value, (list, tuple, np.ndarray)):
        return float(np.nansum(value))
    return 0.0


def list_max(value: object) -> float:
    if isinstance(value, (list, tuple, np.ndarray)) and len(value):
        return float(np.nanmax(value))
    return 0.0


def load_frame(max_series: int) -> pd.DataFrame:
    dataset = load_dataset(DATASET_NAME, split="train")
    data = dataset.to_pandas()
    data["series_id"] = (
        data["city_id"].astype(str)
        + "_"
        + data["store_id"].astype(str)
        + "_"
        + data["product_id"].astype(str)
    )
    selected_series = sorted(data["series_id"].unique())[:max_series]
    data = data[data["series_id"].isin(selected_series)].copy()
    return prepare_frame(data)


def prepare_frame(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["dt"] = pd.to_datetime(data["dt"])
    data["hours_sale_sum"] = data["hours_sale"].map(list_sum)
    data["hours_sale_max"] = data["hours_sale"].map(list_max)
    data["hours_stock_status_sum"] = data["hours_stock_status"].map(list_sum)
    data["hours_stock_status_mean"] = data["hours_stock_status_sum"] / 24.0
    data = data.sort_values(["series_id", "dt"]).reset_index(drop=True)
    data[TARGET_COLUMN] = (
        data.groupby("series_id")["stock_hour6_22_cnt"].shift(-1).fillna(0) > 0
    ).astype(int)
    for column in FEATURE_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0.0)
    return data


class SequenceDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        sequence_length: int,
        cutoff_date: pd.Timestamp,
        mode: str,
    ):
        self.x: list[np.ndarray] = []
        self.y: list[int] = []
        for _, group in frame.groupby("series_id", sort=False):
            group = group.sort_values("dt").reset_index(drop=True)
            values = group[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
            targets = group[TARGET_COLUMN].to_numpy(dtype=np.float32)
            dates = group["dt"].to_numpy()
            if len(group) <= sequence_length:
                continue
            for end in range(sequence_length, len(group)):
                target_date = pd.Timestamp(dates[end])
                if mode == "train" and target_date > cutoff_date:
                    continue
                if mode == "validation" and target_date <= cutoff_date:
                    continue
                self.x.append(values[end - sequence_length : end])
                self.y.append(int(targets[end - 1]))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.x[index], dtype=torch.float32),
            torch.tensor(self.y[index], dtype=torch.float32),
        )


class StockoutLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        return self.classifier(hidden[-1]).squeeze(1)


def scale_features(train: pd.DataFrame, validation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scaler = StandardScaler()
    train = train.copy()
    validation = validation.copy()
    train[FEATURE_COLUMNS] = scaler.fit_transform(train[FEATURE_COLUMNS])
    validation[FEATURE_COLUMNS] = scaler.transform(validation[FEATURE_COLUMNS])
    return train, validation


def train_one_epoch(
    model: StockoutLSTM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    losses = []
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def predict(model: StockoutLSTM, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities = []
    labels_out = []
    for features, labels in loader:
        logits = model(features.to(device))
        probabilities.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        labels_out.extend(labels.numpy().tolist())
    return np.array(labels_out), np.array(probabilities)


def metric_dict(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
    }
    if len(np.unique(labels)) > 1:
        metrics["roc_auc"] = roc_auc_score(labels, probabilities)
        metrics["average_precision"] = average_precision_score(labels, probabilities)
    else:
        metrics["roc_auc"] = float("nan")
        metrics["average_precision"] = float("nan")
    return {key: float(value) for key, value in metrics.items()}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    RESULTS_DIR.mkdir(exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    frame = load_frame(args.max_series)
    cutoff_date = frame["dt"].max() - pd.Timedelta(days=args.validation_days)
    scaler = StandardScaler()
    scaler.fit(frame.loc[frame["dt"] <= cutoff_date, FEATURE_COLUMNS])
    frame = frame.copy()
    frame[FEATURE_COLUMNS] = scaler.transform(frame[FEATURE_COLUMNS])

    train_dataset = SequenceDataset(frame, args.sequence_length, cutoff_date, "train")
    eval_dataset = SequenceDataset(frame, args.sequence_length, cutoff_date, "validation")
    if not train_dataset or not eval_dataset:
        raise ValueError("No sequences were created. Lower --sequence-length or increase --max-series.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    positives = float(sum(train_dataset.y))
    negatives = float(len(train_dataset.y) - positives)
    pos_weight = torch.tensor([negatives / max(positives, 1.0)], device=device)

    model = StockoutLSTM(len(FEATURE_COLUMNS), args.hidden_size).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    history = []
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        labels, probabilities = predict(model, eval_loader, device)
        metrics = metric_dict(labels, probabilities)
        metrics["epoch"] = epoch
        metrics["train_loss"] = loss
        history.append(metrics)
        print(
            f"epoch={epoch} loss={loss:.4f} "
            f"f1={metrics['f1']:.4f} recall={metrics['recall']:.4f} "
            f"roc_auc={metrics['roc_auc']:.4f}"
        )

    output = {
        "dataset": DATASET_NAME,
        "task": "predict next-day stockout from previous sequence window",
        "target_definition": "next_day_stockout = next stock_hour6_22_cnt > 0",
        "features": FEATURE_COLUMNS,
        "series_columns": SERIES_COLUMNS,
        "max_series": args.max_series,
        "sequence_length": args.sequence_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_size": args.hidden_size,
        "validation_days": args.validation_days,
        "cutoff_date": str(cutoff_date.date()),
        "seed": args.seed,
        "device": str(device),
        "train_sequences": len(train_dataset),
        "eval_sequences": len(eval_dataset),
        "train_positive_rate": positives / len(train_dataset),
        "history": history,
        "final_metrics": history[-1],
    }
    path = RESULTS_DIR / "lstm_metrics.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved metrics to {path}")


if __name__ == "__main__":
    main()
