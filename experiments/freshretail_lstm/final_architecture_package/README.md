# Final Architecture Package

This folder collects the final documents and key references for the custom LSTM architecture work on FreshRetailNet-50K.

## What is included

- `lstm_architecture_explainer.docx`
  - Beginner-friendly explainer with simple diagrams
  - Explains what changes were tried and why

- `lstm_architecture_mentor_summary.docx`
  - Shorter mentor-facing technical summary
  - Focuses on architecture decisions, results, and final recommendation

- `lstm_ablation_summary.csv`
  - Compact summary of all architecture experiments and metrics

- `src/`
  - Snapshot copies of the main model code used for the final architecture work
  - Includes `train_top_sku_24h_lstm.py` and `run_lstm_ablation.py`

## Main code files

The working source code remains in the experiment source folder:

- `../src/train_top_sku_24h_lstm.py`
- `../src/run_lstm_ablation.py`

For convenience, snapshot copies are also included inside:

- `src/train_top_sku_24h_lstm.py`
- `src/run_lstm_ablation.py`

These files now contain:

- standard compact LSTM baseline
- CIFG-LSTM
- dual-stream LSTM
- dual-stream shortcut variants
- final best model: dual-stream LSTM with inventory-only shortcut

## Final recommended model

Recommended final model:

- `dual_stream_inventory_shortcut_h16x16_seq14_f6`

Why:

- best F1 among tested variants
- fewer parameters than the earlier compact baseline
- architecture is more aligned with stockout behavior

## Lightweight backup model

Compact alternative:

- `dual_stream_shortcut_h12x12_seq14_f6`

Why:

- nearly baseline-level F1
- much smaller parameter count
