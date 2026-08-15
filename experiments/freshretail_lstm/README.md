# FreshRetailNet-50K LSTM Experiment

This experiment starts the time-series part of our BTP work using
FreshRetailNet-50K.

FreshRetailNet-50K is useful for our project because it is not just a static
SKU snapshot. It contains daily store-product time series with hourly sales and
hourly stock-status annotations. That lets us train a model that learns from a
sequence of recent days instead of one row at a time.

Sources:

- Dataset: https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K
- Paper: https://arxiv.org/abs/2505.16319
- Baseline code: https://github.com/Dingdong-Inc/frn-50k-baseline

## What We Predict

Initial target:

> Given the last `N` days for one store-product pair, predict whether the next
> day has any stockout signal.

Operationally:

```text
target = 1 if next_day.stock_hour6_22_cnt > 0 else 0
```

This is a clean first LSTM task because:

- input is sequential,
- target is inventory/stockout related,
- the model can learn patterns before stockout events.

## Setup

From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run a Quick LSTM Baseline

```bash
python src/train_lstm.py \
  --max-series 500 \
  --sequence-length 14 \
  --epochs 3 \
  --batch-size 128
```

The script downloads the dataset through Hugging Face `datasets`, builds
sequences, trains a small LSTM, and writes metrics to `results/lstm_metrics.json`.

## Why LSTM Here?

LSTM means Long Short-Term Memory. It is a recurrent neural network designed for
ordered data. In our case, each SKU-store pair has a timeline:

```text
day 1 -> day 2 -> day 3 -> ... -> day t
```

A normal Random Forest or XGBoost model sees one row at a time unless we manually
create lag features. An LSTM directly receives a window like the last 14 days and
learns which parts of that history matter.

The "memory" part is the key. It uses gates to decide:

- what old information to keep,
- what new information to add,
- what information to forget,
- what to output for prediction.

For stockout prediction, that is useful because risk may build over time:

- sales rising for several days,
- frequent recent stockout hours,
- discount/activity causing demand pressure,
- weather or holiday effects.

## Current Project Story

1. We reproduced the Ali et al. (2024/C1) snapshot backorder baseline.
2. We found that C1 is useful but not genuinely time-series.
3. The mentor asked for a time-series dataset.
4. FreshRetailNet-50K fills that gap because it has store-product-date records
   and stockout annotations.
5. This folder starts the deep-learning/time-series branch using LSTM.

