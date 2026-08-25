import json
import numpy as np
import pandas as pd
import torch
from torch import nn

from run_lstm_ablation import (
    ExperimentConfig,
    prepare_data,
    run_config,
    RESULTS_DIR,
    DEMAND_FEATURES,
    INVENTORY_FEATURES
)
import run_lstm_ablation

class PRSimpleRNNBase(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 24)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.rnn(x)
        return self.fc(hidden[-1])

class PRStackedRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, 24)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.rnn(x)
        return self.fc(hidden[-1])

class PRDropoutDenseRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32, nonlinearity='tanh'):
        super().__init__()
        # PyTorch RNN dropout is only applied between layers.
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2, nonlinearity=nonlinearity)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 24)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.rnn(x)
        return self.fc(hidden[-1])

class PRDualStreamRNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        demand_indices: list[int],
        inventory_indices: list[int],
    ):
        super().__init__()
        self.demand_indices = demand_indices
        self.inventory_indices = inventory_indices

        self.demand_rnn = nn.RNN(len(demand_indices), 16, batch_first=True)
        self.inventory_rnn = nn.RNN(len(inventory_indices), 16, batch_first=True)
        self.full_rnn = nn.RNN(input_size, 32, batch_first=True)

        self.rnn_dropout = nn.Dropout(0.10)

        self.fc = nn.Sequential(
            nn.Linear(16 + 16 + 32, 32),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, 24)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        demand_x = x[:, :, self.demand_indices]
        inventory_x = x[:, :, self.inventory_indices]

        _, demand_hidden = self.demand_rnn(demand_x)
        _, inventory_hidden = self.inventory_rnn(inventory_x)
        _, full_hidden = self.full_rnn(x)

        combined = torch.cat([
            demand_hidden[-1], 
            inventory_hidden[-1], 
            full_hidden[-1]
        ], dim=-1)

        combined = self.rnn_dropout(combined)
        return self.fc(combined)

original_build_model = run_lstm_ablation.build_model

def custom_build_model(config, features, **kwargs):
    if config.model_type == "pr_var1":
        return PRSimpleRNNBase(len(features), 32)
    elif config.model_type == "pr_var2":
        return PRSimpleRNNBase(len(features), 64)
    elif config.model_type == "pr_var3":
        return PRStackedRNN(len(features), 32)
    elif config.model_type == "pr_var4":
        return PRDropoutDenseRNN(len(features), 32, nonlinearity='tanh')
    elif config.model_type == "pr_var5":
        return PRDropoutDenseRNN(len(features), 32, nonlinearity='relu')
    elif config.model_type == "pr_var6":
        demand_indices = [features.index(feature) for feature in DEMAND_FEATURES if feature in features]
        inventory_indices = [features.index(feature) for feature in INVENTORY_FEATURES if feature in features]
        return PRDualStreamRNN(len(features), demand_indices, inventory_indices)
    
    return original_build_model(config, features, **kwargs)

run_lstm_ablation.build_model = custom_build_model

def get_param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    variations = [
        {"id": "pr_var6", "name": "Dual-Stream SimpleRNN 24H"},
    ]

    seeds = [42, 100, 2024, 777, 999]
    top_series = 15
    validation_days = 15
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print("Preparing data...")
    frame, cutoff_date, features = prepare_data(top_series, validation_days)

    final_results = []
    
    for var in variations:
        print(f"\n=============================================")
        print(f"Evaluating {var['name']}...")
        print(f"=============================================")
        
        config = ExperimentConfig(
            name=var['id'],
            hidden_size=0,
            sequence_length=14,
            feature_set="all",
            model_type=var['id'],
            epochs=10,
            batch_size=64,
            learning_rate=1e-3,
        )

        dummy_model = custom_build_model(config, features)
        params = get_param_count(dummy_model)
        print(f"Parameters: {params}")

        var_results = []
        for seed in seeds:
            print(f"--- Running seed {seed} ---")
            result = run_config(frame, cutoff_date, config, device, seed)
            result["seed"] = seed
            var_results.append(result)
            print(f"  Seed {seed} best_f1={result['best_hour_level_f1']:.4f} pr_auc={result.get('best_pr_auc', 0.0):.4f}")

        f1_scores = [r["best_hour_level_f1"] for r in var_results]
        pr_aucs = [r.get("best_pr_auc", 0.0) for r in var_results]
        
        mean_f1 = float(np.mean(f1_scores))
        std_f1 = float(np.std(f1_scores))
        mean_prauc = float(np.mean(pr_aucs))
        std_prauc = float(np.std(pr_aucs))
        
        final_results.append({
            "variation": var['name'],
            "params": params,
            "mean_f1": mean_f1,
            "std_f1": std_f1,
            "mean_prauc": mean_prauc,
            "std_prauc": std_prauc,
        })

    RESULTS_DIR.mkdir(exist_ok=True)
    json_path = RESULTS_DIR / "rnn_var6_prauc.json"
    json_path.write_text(json.dumps(final_results, indent=2), encoding="utf-8")
    
    print("\n\n=== FINAL SUMMARY ===")
    for res in final_results:
        print(f"{res['variation']} ({res['params']} params): F1 {res['mean_f1']:.4f} +/- {res['std_f1']:.4f}, PR-AUC {res['mean_prauc']:.4f} +/- {res['std_prauc']:.4f}")

if __name__ == "__main__":
    main()
