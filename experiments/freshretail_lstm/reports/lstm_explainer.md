# LSTM Explainer for Our BTP

## What LSTM Is

LSTM stands for Long Short-Term Memory. It is a type of recurrent neural network
for sequence data.

In plain terms, an LSTM reads a timeline step by step. For our project, that
timeline is a store-product sequence:

```text
day 1 -> day 2 -> day 3 -> ... -> today
```

At each step it receives features like sales, stockout hours, discount,
holiday flag, and weather. It keeps an internal memory of what happened earlier
and uses that memory to predict what comes next.

## Why Not Just Random Forest?

Random Forest and XGBoost are strong for tabular rows, and we already used them
for the Ali/C1 replication. But the C1 data is a snapshot. It does not truly
model "what happened before today".

FreshRetailNet-50K gives us actual time order. LSTM is useful because stockout
risk is often temporal:

- demand may rise for several days,
- the product may repeatedly run out in recent hours,
- discounts or activities may create demand spikes,
- weather/holiday effects may change demand over time.

The LSTM can look at a recent window, such as the last 14 days, and learn from
the pattern.

## What Happened in the Project So Far

1. We started by reading the primary stockout paper and Ali et al. (2024/C1).
2. We replicated the Ali/C1 Kaggle backorder baseline.
3. That replication showed C1 is useful, but its data is a static SKU snapshot.
4. The mentor asked for a time-series dataset.
5. We identified FreshRetailNet-50K because it has store-product time series and
   explicit stockout annotations.
6. This experiment begins the time-series modeling stage with LSTM.

## First LSTM Task

Input:

```text
last N days of features for one store-product pair
```

Output:

```text
whether the next day has stockout hours
```

Target:

```text
next_day_stockout = 1 if next day's stock_hour6_22_cnt > 0 else 0
```

## First Smoke-Test Result

We ran a deliberately small smoke test to verify the pipeline:

```bash
python src/train_lstm.py \
  --max-series 100 \
  --sequence-length 14 \
  --validation-days 14 \
  --epochs 1 \
  --batch-size 128
```

Setup:

- Dataset split used: FreshRetailNet train split.
- Validation design: chronological holdout from the last 14 days.
- Store-product series used: 100.
- Training sequences: 6,200.
- Validation sequences: 1,400.
- Device: Apple `mps`.

Metrics after 1 epoch:

- Accuracy: 0.588
- Precision: 0.551
- Recall: 0.638
- F1: 0.592
- ROC-AUC: 0.625
- Average precision: 0.594

This is not a final research result. It only proves that data loading, sequence
construction, class-weighted loss, LSTM training, and validation metrics are
working end-to-end.

## Why This Is a Good BTP Step

This gives us a clean bridge:

- C1 replication proves we understand the literature baseline.
- FreshRetailNet gives the time-series data the mentor asked for.
- LSTM gives us a real sequence model.
- Later, we can compare LSTM with XGBoost using lag features.
- After that, we can add explainability: SHAP for XGBoost and maybe attention or
  perturbation-based explanations for the LSTM.
