import json
import numpy as np
import torch

from run_lstm_ablation import (
    CONFIGS,
    prepare_data,
    run_config,
    RESULTS_DIR,
)

def main():
    target_config_name = "dual_stream_inventory_shortcut_h16x16_seq14_f6"
    config = next((c for c in CONFIGS if c.name == target_config_name), None)
    if not config:
        raise ValueError(f"Config {target_config_name} not found.")

    seeds = [42, 100, 2024, 777, 999]
    top_series = 15
    validation_days = 15
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print("Preparing data...")
    frame, cutoff_date, _ = prepare_data(top_series, validation_days)

    results = []
    print(f"Running validation for {config.name} over {len(seeds)} seeds...")
    
    for seed in seeds:
        print(f"--- Running seed {seed} ---")
        result = run_config(frame, cutoff_date, config, device, seed)
        result["seed"] = seed
        results.append(result)
        print(f"  Seed {seed} best_f1={result['best_hour_level_f1']:.4f} pr_auc={result.get('best_pr_auc', 0.0):.4f}")

    # Compute aggregate metrics
    f1_scores = [r["best_hour_level_f1"] for r in results]
    recall_scores = [r["best_hour_level_recall"] for r in results]
    pr_aucs = [r.get("best_pr_auc", 0.0) for r in results]
    
    print("\n=== Validation Results ===")
    print(f"F1 Scores: {[round(x, 4) for x in f1_scores]}")
    print(f"Mean F1: {np.mean(f1_scores):.4f} +/- {np.std(f1_scores):.4f}")
    print(f"Recall Scores: {[round(x, 4) for x in recall_scores]}")
    print(f"Mean Recall: {np.mean(recall_scores):.4f} +/- {np.std(recall_scores):.4f}")
    print(f"Mean PR-AUC: {np.mean(pr_aucs):.4f} +/- {np.std(pr_aucs):.4f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    json_path = RESULTS_DIR / "seed_validation_results.json"
    json_path.write_text(
        json.dumps(
            {
                "config_name": config.name,
                "seeds": seeds,
                "results": results,
                "summary": {
                    "mean_f1": float(np.mean(f1_scores)),
                    "std_f1": float(np.std(f1_scores)),
                    "mean_recall": float(np.mean(recall_scores)),
                    "std_recall": float(np.std(recall_scores))
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {json_path}")

if __name__ == "__main__":
    main()
