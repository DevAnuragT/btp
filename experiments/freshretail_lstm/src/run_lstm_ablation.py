from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

from train_top_sku_24h_lstm import (
    DATASET_NAME,
    FEATURE_COLUMNS,
    RESULTS_DIR,
    TARGET_COLUMNS,
    Stock24CIFGLSTM,
    Stock24DualStreamGatedShortcutLSTM,
    Stock24DualStreamInventoryEmbeddingLSTM,
    Stock24DualStreamInventoryShortcutLSTM,
    Stock24DualStreamLSTM,
    Stock24DualStreamCIFGLSTM,
    Stock24DualStreamShortcutLSTM,
    Stock24LSTM,
    Stock24SequenceDataset,
    add_next_day_targets,
    clean_frame,
    compute_metrics,
    load_dataset,
    predict,
    select_top_series,
    set_seed,
    train_epoch,
)


REDUCED_FEATURES = [
    "sale_amount",
    "stock_hour6_22_cnt",
    "discount",
    "holiday_flag",
    "hours_sale_sum",
    "hours_stock_status_sum",
]

DEMAND_FEATURES = [
    "sale_amount",
    "discount",
    "holiday_flag",
    "hours_sale_sum",
]

INVENTORY_FEATURES = [
    "stock_hour6_22_cnt",
    "hours_stock_status_sum",
]

MINIMAL_FEATURES = [
    "sale_amount",
    "stock_hour6_22_cnt",
    "hours_sale_sum",
    "hours_stock_status_sum",
]


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    hidden_size: int
    sequence_length: int
    feature_set: str
    model_type: str = "baseline"
    demand_hidden_size: int | None = None
    inventory_hidden_size: int | None = None
    embedding_size: int = 4
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3


CONFIGS = [
    ExperimentConfig("baseline_lstm_h96_seq14_all13", 96, 14, "all"),
    ExperimentConfig("compact_lstm_h64_seq14_all13", 64, 14, "all"),
    ExperimentConfig("compact_lstm_h32_seq14_all13", 32, 14, "all"),
    ExperimentConfig("compact_lstm_h64_seq10_all13", 64, 10, "all"),
    ExperimentConfig("compact_lstm_h64_seq07_all13", 64, 7, "all"),
    ExperimentConfig("feature_reduced_lstm_h64_seq14_f6", 64, 14, "reduced"),
    ExperimentConfig("feature_reduced_lstm_h32_seq14_f6", 32, 14, "reduced"),
    ExperimentConfig("ultra_lstm_h16_seq14_f6", 16, 14, "reduced"),
    ExperimentConfig("ultra_lstm_h08_seq14_f6", 8, 14, "reduced"),
    ExperimentConfig("ultra_lstm_h16_seq14_f4", 16, 14, "minimal"),
    ExperimentConfig("ultra_lstm_h16_seq07_f6", 16, 7, "reduced"),
    ExperimentConfig("compact_cifg_h32_seq14_f6", 32, 14, "reduced", model_type="cifg"),
    ExperimentConfig(
        "dual_stream_lstm_h16x16_seq14_f6",
        32,
        14,
        "reduced",
        model_type="dual_stream",
        demand_hidden_size=16,
        inventory_hidden_size=16,
    ),
    ExperimentConfig(
        "dual_stream_cifg_h16x16_seq14_f6",
        32,
        14,
        "reduced",
        model_type="dual_stream_cifg",
        demand_hidden_size=16,
        inventory_hidden_size=16,
    ),
    ExperimentConfig(
        "dual_stream_lstm_h20x12_seq14_f6",
        32,
        14,
        "reduced",
        model_type="dual_stream",
        demand_hidden_size=20,
        inventory_hidden_size=12,
    ),
    ExperimentConfig(
        "dual_stream_shortcut_h16x16_seq14_f6",
        32,
        14,
        "reduced",
        model_type="dual_stream_shortcut",
        demand_hidden_size=16,
        inventory_hidden_size=16,
    ),
    ExperimentConfig(
        "dual_stream_shortcut_h12x12_seq14_f6",
        24,
        14,
        "reduced",
        model_type="dual_stream_shortcut",
        demand_hidden_size=12,
        inventory_hidden_size=12,
    ),
    ExperimentConfig(
        "dual_stream_inventory_shortcut_h16x16_seq14_f6",
        32,
        14,
        "reduced",
        model_type="dual_stream_inventory_shortcut",
        demand_hidden_size=16,
        inventory_hidden_size=16,
    ),
    ExperimentConfig(
        "dual_stream_gated_shortcut_h16x16_seq14_f6",
        32,
        14,
        "reduced",
        model_type="dual_stream_gated_shortcut",
        demand_hidden_size=16,
        inventory_hidden_size=16,
    ),
    ExperimentConfig(
        "dual_stream_inventory_embedding_h16x16_e4_seq14_f6",
        32,
        14,
        "reduced",
        model_type="dual_stream_inventory_embedding",
        demand_hidden_size=16,
        inventory_hidden_size=16,
        embedding_size=4,
    ),
]


def selected_features(feature_set: str) -> list[str]:
    if feature_set == "all":
        return FEATURE_COLUMNS
    if feature_set == "reduced":
        return REDUCED_FEATURES
    if feature_set == "minimal":
        return MINIMAL_FEATURES
    raise ValueError(f"Unknown feature set: {feature_set}")


class FeatureSubsetSequenceDataset(Stock24SequenceDataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        feature_columns: list[str],
        sequence_length: int,
        cutoff_date: pd.Timestamp,
        mode: str,
    ):
        self.x: list[np.ndarray] = []
        self.y: list[np.ndarray] = []
        self.meta: list[dict[str, object]] = []
        for series_id, group in frame.groupby("series_id", sort=False):
            group = group.sort_values("dt").reset_index(drop=True)
            features = group[feature_columns].to_numpy(dtype=np.float32)
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


class StaticEmbeddingSequenceDataset(FeatureSubsetSequenceDataset):
    """Return categorical store and product indices with each sequence."""

    def __init__(
        self,
        frame: pd.DataFrame,
        feature_columns: list[str],
        sequence_length: int,
        cutoff_date: pd.Timestamp,
        mode: str,
    ):
        store_values = sorted(frame["store_id"].astype(str).unique())
        product_values = sorted(frame["product_id"].astype(str).unique())
        self.store_to_index = {value: index for index, value in enumerate(store_values)}
        self.product_to_index = {value: index for index, value in enumerate(product_values)}
        self.x: list[np.ndarray] = []
        self.y: list[np.ndarray] = []
        self.store_indices: list[int] = []
        self.product_indices: list[int] = []
        self.meta: list[dict[str, object]] = []
        for series_id, group in frame.groupby("series_id", sort=False):
            group = group.sort_values("dt").reset_index(drop=True)
            features = group[feature_columns].to_numpy(dtype=np.float32)
            targets = group[TARGET_COLUMNS].to_numpy(dtype=np.float32)
            dates = group["dt"].to_numpy()
            if len(group) <= sequence_length:
                continue
            store_index = self.store_to_index[str(group["store_id"].iloc[0])]
            product_index = self.product_to_index[str(group["product_id"].iloc[0])]
            for end in range(sequence_length, len(group)):
                target_date = pd.Timestamp(dates[end])
                if mode == "train" and target_date > cutoff_date:
                    continue
                if mode == "validation" and target_date <= cutoff_date:
                    continue
                self.x.append(features[end - sequence_length : end])
                self.y.append(targets[end - 1])
                self.store_indices.append(store_index)
                self.product_indices.append(product_index)
                self.meta.append({"series_id": series_id, "target_date": str(target_date.date())})

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return (
            torch.tensor(self.x[index], dtype=torch.float32),
            torch.tensor(self.store_indices[index], dtype=torch.long),
            torch.tensor(self.product_indices[index], dtype=torch.long),
            torch.tensor(self.y[index], dtype=torch.float32),
        )


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_model(
    config: ExperimentConfig,
    features: list[str],
    store_count: int,
    product_count: int,
) -> nn.Module:
    if config.model_type == "baseline":
        return Stock24LSTM(len(features), config.hidden_size)
    if config.model_type == "cifg":
        return Stock24CIFGLSTM(len(features), config.hidden_size)
    if config.model_type in {
        "dual_stream",
        "dual_stream_cifg",
        "dual_stream_shortcut",
        "dual_stream_inventory_shortcut",
        "dual_stream_gated_shortcut",
        "dual_stream_inventory_embedding",
    }:
        missing = [
            feature
            for feature in [*DEMAND_FEATURES, *INVENTORY_FEATURES]
            if feature not in features
        ]
        if missing:
            raise ValueError(
                f"Model {config.name} requires reduced dual-stream features, missing: {missing}"
            )
        demand_indices = [features.index(feature) for feature in DEMAND_FEATURES]
        inventory_indices = [features.index(feature) for feature in INVENTORY_FEATURES]
    if config.model_type == "dual_stream":
        return Stock24DualStreamLSTM(
            input_size=len(features),
            hidden_size=config.hidden_size,
            demand_indices=demand_indices,
            inventory_indices=inventory_indices,
            demand_hidden_size=config.demand_hidden_size,
            inventory_hidden_size=config.inventory_hidden_size,
        )
    if config.demand_hidden_size is None or config.inventory_hidden_size is None:
        raise ValueError(f"Dual-stream configuration requires branch hidden sizes: {config.name}")
    if config.model_type == "dual_stream_cifg":
        return Stock24DualStreamCIFGLSTM(
            input_size=len(features),
            demand_hidden_size=config.demand_hidden_size,
            inventory_hidden_size=config.inventory_hidden_size,
            demand_indices=demand_indices,
            inventory_indices=inventory_indices,
        )
    if config.model_type == "dual_stream_shortcut":
        return Stock24DualStreamShortcutLSTM(
            input_size=len(features),
            hidden_size=config.hidden_size,
            demand_indices=demand_indices,
            inventory_indices=inventory_indices,
            demand_hidden_size=config.demand_hidden_size,
            inventory_hidden_size=config.inventory_hidden_size,
        )
    if config.model_type == "dual_stream_inventory_shortcut":
        return Stock24DualStreamInventoryShortcutLSTM(
            input_size=len(features),
            hidden_size=config.hidden_size,
            demand_indices=demand_indices,
            inventory_indices=inventory_indices,
            demand_hidden_size=config.demand_hidden_size,
            inventory_hidden_size=config.inventory_hidden_size,
        )
    if config.model_type == "dual_stream_inventory_embedding":
        return Stock24DualStreamInventoryEmbeddingLSTM(
            input_size=len(features),
            hidden_size=config.hidden_size,
            demand_indices=demand_indices,
            inventory_indices=inventory_indices,
            store_count=store_count,
            product_count=product_count,
            embedding_size=config.embedding_size,
            demand_hidden_size=config.demand_hidden_size,
            inventory_hidden_size=config.inventory_hidden_size,
        )
    if config.model_type == "dual_stream_gated_shortcut":
        return Stock24DualStreamGatedShortcutLSTM(
            input_size=len(features),
            hidden_size=config.hidden_size,
            demand_indices=demand_indices,
            inventory_indices=inventory_indices,
            demand_hidden_size=config.demand_hidden_size,
            inventory_hidden_size=config.inventory_hidden_size,
        )
    raise ValueError(f"Unknown model_type: {config.model_type}")


def prepare_data(top_series: int, validation_days: int) -> tuple[pd.DataFrame, pd.Timestamp, pd.DataFrame]:
    raw = load_dataset(DATASET_NAME, split="train").to_pandas()
    clean = clean_frame(raw)
    selected_frame, selected_summary = select_top_series(clean, top_series)
    selected_frame = add_next_day_targets(selected_frame)
    cutoff_date = selected_frame["dt"].max() - pd.Timedelta(days=validation_days)
    return selected_frame, cutoff_date, selected_summary


def run_config(
    base_frame: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    config: ExperimentConfig,
    device: torch.device,
    seed: int,
) -> dict[str, object]:
    set_seed(seed)
    features = selected_features(config.feature_set)
    frame = base_frame.copy()
    scaler = StandardScaler()
    scaler.fit(frame.loc[frame["dt"] <= cutoff_date, features])
    frame[features] = scaler.transform(frame[features])

    dataset_class = (
        StaticEmbeddingSequenceDataset
        if config.model_type == "dual_stream_inventory_embedding"
        else FeatureSubsetSequenceDataset
    )
    train_dataset = dataset_class(frame, features, config.sequence_length, cutoff_date, "train")
    validation_dataset = dataset_class(frame, features, config.sequence_length, cutoff_date, "validation")
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False)

    train_targets = np.vstack(train_dataset.y)
    positive_hours = float(train_targets.sum())
    total_hours = float(train_targets.size)
    negative_hours = total_hours - positive_hours
    pos_weight = torch.tensor([negative_hours / max(positive_hours, 1.0)] * 24, device=device)

    model = build_model(
        config,
        features,
        store_count=frame["store_id"].astype(str).nunique(),
        product_count=frame["product_id"].astype(str).nunique(),
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    parameter_count = count_parameters(model)

    history = []
    train_start = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        labels, probabilities = predict(model, validation_loader, device)
        metrics = compute_metrics(labels, probabilities)
        metrics["epoch"] = epoch
        metrics["train_loss"] = loss
        history.append(metrics)
    train_seconds = time.perf_counter() - train_start

    inference_start = time.perf_counter()
    labels, probabilities = predict(model, validation_loader, device)
    inference_seconds = time.perf_counter() - inference_start
    final_metrics = compute_metrics(labels, probabilities)
    best_metrics = max(history, key=lambda item: item["hour_level_f1"])

    return {
        **asdict(config),
        "features": len(features),
        "feature_names": features,
        "parameters": parameter_count,
        "train_sequences": len(train_dataset),
        "validation_sequences": len(validation_dataset),
        "train_positive_hour_rate": positive_hours / total_hours,
        "train_seconds": train_seconds,
        "inference_seconds": inference_seconds,
        "inference_ms_per_sequence": (inference_seconds / len(validation_dataset)) * 1000,
        "best_epoch": int(best_metrics["epoch"]),
        "best_hour_level_accuracy": best_metrics["hour_level_accuracy"],
        "best_hour_level_precision": best_metrics["hour_level_precision"],
        "best_hour_level_recall": best_metrics["hour_level_recall"],
        "best_hour_level_f1": best_metrics["hour_level_f1"],
        "best_pr_auc": best_metrics["pr_auc"],
        "best_exact_24h_match_rate": best_metrics["exact_24h_match_rate"],
        "best_mean_absolute_hour_count_error": best_metrics[
            "mean_absolute_hour_count_error"
        ],
        "final_epoch": len(history),
        "final_hour_level_accuracy": final_metrics["hour_level_accuracy"],
        "final_hour_level_precision": final_metrics["hour_level_precision"],
        "final_hour_level_recall": final_metrics["hour_level_recall"],
        "final_hour_level_f1": final_metrics["hour_level_f1"],
        "final_pr_auc": final_metrics["pr_auc"],
        "final_exact_24h_match_rate": final_metrics["exact_24h_match_rate"],
        "final_mean_absolute_hour_count_error": final_metrics[
            "mean_absolute_hour_count_error"
        ],
        "history": history,
    }


def main() -> None:
    seed = 42
    top_series = 15
    validation_days = 15
    set_seed(seed)
    RESULTS_DIR.mkdir(exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    frame, cutoff_date, selected_summary = prepare_data(top_series, validation_days)
    selected_summary.to_csv(RESULTS_DIR / "ablation_top_sku_series.csv", index=False)

    results = []
    for index, config in enumerate(CONFIGS, start=1):
        print(f"[{index}/{len(CONFIGS)}] running {config.name}")
        result = run_config(frame, cutoff_date, config, device, seed)
        results.append(result)
        print(
            f"  best_f1={result['best_hour_level_f1']:.4f} "
            f"params={result['parameters']} "
            f"train_s={result['train_seconds']:.2f}"
        )

    json_path = RESULTS_DIR / "lstm_ablation_results.json"
    csv_path = RESULTS_DIR / "lstm_ablation_summary.csv"
    json_path.write_text(
        json.dumps(
            {
                "dataset": DATASET_NAME,
                "task": "LSTM complexity ablation for next-day 24-hour stock-status prediction.",
                "top_series": top_series,
                "validation_days": validation_days,
                "train_until": str(cutoff_date.date()),
                "device": str(device),
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    summary = pd.DataFrame([{k: v for k, v in row.items() if k != "history"} for row in results])
    summary.to_csv(csv_path, index=False)
    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
