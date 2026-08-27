# RNN vs LSTM Model Comparison

We evaluated the teammate's **Dual-Stream SimpleRNN 24H** (from PR #6) against our best **Dual-Stream Inventory Shortcut LSTM** using the exact same data preparation, top 15 SKUs, chronological test split, and evaluated both across 5 random seeds to ensure fairness and robustness.

## Metrics Comparison (5-Seed Average)

| Metric | Our Best LSTM (`dual_stream_inventory_shortcut`) | Teammate's PR #6 RNN (`pr_dual_stream_rnn_24h`) |
| :--- | :--- | :--- |
| **Mean Hour-Level F1** | **0.7941 ± 0.0040** | 0.7782 ± 0.0109 |
| **Mean Hour-Level Recall** | **0.7938 ± 0.0044** | 0.7762 ± 0.0292 |
| **Stability (F1 Range)** | **0.789 - 0.799** | 0.763 - 0.788 |

### Seed-by-Seed F1 Breakdown
* **Seed 42:** LSTM (0.7947) vs RNN (0.7850)
* **Seed 100:** LSTM (0.7992) vs RNN (0.7879)
* **Seed 2024:** LSTM (0.7895) vs RNN (0.7662)
* **Seed 777:** LSTM (0.7936) vs RNN (0.7637)
* **Seed 999:** LSTM (0.7937) vs RNN (0.7880)

## Key Takeaways

1. **Better Performance:** Our LSTM model consistently outperforms the SimpleRNN in predicting exact hour-level stockouts, achieving an average F1 score of **0.7941** compared to the RNN's **0.7782**.
2. **Superior Stability:** The RNN model shows significant variance depending on the random initialization (standard deviation of `±0.0109`), dipping as low as `0.763` on some seeds. Our LSTM is extremely robust with minimal variance (`±0.0040`), consistently hovering around the `0.794` mark regardless of the seed.
3. **Architecture Justification:** The addition of the LSTM cell's gating mechanisms and our specific shortcut connection from the inventory branch successfully combat the vanishing gradient problem present in SimpleRNNs, allowing the model to capture the complex temporal dependencies over the 14-day lookback window much more effectively.

We can conclude that the **Dual-Stream Inventory Shortcut LSTM** is undeniably the superior architecture for this 24-hour stockout prediction task.
