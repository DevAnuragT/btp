# Top-SKU 24-Hour Stock Prediction with LSTM

## Goal

Use FreshRetailNet-50K for time-series stock prediction:

> Select statistically important SKUs, train on roughly 2.5 months of data, and
> predict the next 15 days of 24-hour stock status using LSTM.

## Dataset Used

- Dataset: `Dingdong-Inc/FreshRetailNet-50K`
- Split used: `train`
- Date range in raw train split: 2024-03-28 to 2024-06-25
- After next-day target creation: 89 usable days

The official `eval` split has only around 7 days per series, so it is too short
for a 14-day LSTM input window. Therefore, we used the train split and created a
chronological holdout:

- Train period: up to 2024-06-09
- Validation/prediction period: last 15 days

## Cleaning

The script performs these cleaning steps:

1. Creates `series_id = city_id + store_id + product_id`.
2. Converts `dt` to datetime.
3. Removes rows with missing date/store/product identifiers.
4. Removes duplicate `(series_id, dt)` rows.
5. Converts `hours_stock_status` into 24 binary hourly stock-status columns.
6. Converts `hours_sale` into summary features:
   - total hourly sales
   - max hourly sales
7. Fills numeric missing values with `0.0`.
8. Standardizes features using only the training period.

## Top SKU Selection

Stockout is store-specific, so a SKU is represented by its best city-store-product
series. We enforce distinct `product_id` values so the top list is not dominated
by one product across many stores.

Selection score:

```text
0.40 * total_sales
+ 0.25 * total_stockout_hours
+ 0.20 * stockout_days
+ 0.15 * sales_std
```

Each component is min-max normalized before scoring.

Selected top 15 distinct SKUs:

| Rank | series_id | product_id | total_sales | stockout_days | selection_score |
|---:|---|---:|---:|---:|---:|
| 1 | 4_11_267 | 267 | 1610.20 | 90 | 0.880 |
| 2 | 13_627_589 | 589 | 1590.00 | 74 | 0.747 |
| 3 | 16_519_151 | 151 | 58.00 | 89 | 0.458 |
| 4 | 13_702_580 | 580 | 737.90 | 76 | 0.454 |
| 5 | 0_18_300 | 300 | 855.07 | 53 | 0.434 |
| 6 | 0_822_834 | 834 | 170.50 | 85 | 0.431 |
| 7 | 0_255_4 | 4 | 612.70 | 64 | 0.425 |
| 8 | 0_237_540 | 540 | 878.10 | 49 | 0.419 |
| 9 | 12_230_757 | 757 | 32.70 | 84 | 0.418 |
| 10 | 3_712_333 | 333 | 42.26 | 81 | 0.411 |
| 11 | 0_593_129 | 129 | 60.80 | 87 | 0.410 |
| 12 | 0_471_215 | 215 | 355.80 | 79 | 0.401 |
| 13 | 0_750_90 | 90 | 44.00 | 81 | 0.393 |
| 14 | 3_854_157 | 157 | 197.76 | 85 | 0.391 |
| 15 | 0_822_70 | 70 | 585.20 | 61 | 0.389 |

## LSTM Setup

Input:

```text
last 14 days of features for one selected SKU-store series
```

Target:

```text
next day's 24 hourly stock-status values
```

So each prediction is a 24-value binary vector:

```text
[hour_0, hour_1, ..., hour_23]
```

Model:

- One LSTM layer
- Hidden size: 96
- Dropout: 0.2
- Output layer: 24 sigmoid logits
- Loss: `BCEWithLogitsLoss`
- Positive-class weighting used because stock-status labels are imbalanced

Run command:

```bash
python src/train_top_sku_24h_lstm.py \
  --top-series 15 \
  --sequence-length 14 \
  --validation-days 15 \
  --epochs 10 \
  --batch-size 64
```

Training samples:

- Train sequences: 900
- Validation sequences: 225
- Training positive-hour rate: 0.609

## Results

Best validation epoch: 7

| Metric | Value |
|---|---:|
| Hour-level accuracy | 0.791 |
| Hour-level precision | 0.767 |
| Hour-level recall | 0.804 |
| Hour-level F1 | 0.785 |
| Exact 24-hour vector match | 0.302 |
| Mean absolute stockout-hour count error | 4.671 hours |

Final epoch: 10

| Metric | Value |
|---|---:|
| Hour-level accuracy | 0.783 |
| Hour-level precision | 0.774 |
| Hour-level recall | 0.769 |
| Hour-level F1 | 0.772 |
| Exact 24-hour vector match | 0.271 |
| Mean absolute stockout-hour count error | 4.889 hours |

## Interpretation

The LSTM learned useful temporal patterns. It predicts individual hourly
stock-status labels reasonably well, with best hour-level F1 around 0.785.

The exact 24-hour vector match is lower because this is a stricter metric: all
24 hourly labels must be correct for a day to count as an exact match.

The mean absolute count error means that, on average, the model's predicted
number of stockout hours differs from the true count by around 4.7 hours.

## What This Means for the Project

This is now aligned with the mentor's direction:

- We are using a true time-series dataset.
- We selected top SKUs statistically.
- We trained on about 2.5 months.
- We predicted the last 15 days.
- We moved from binary stockout prediction to 24-hour stock-status prediction.

Next research step:

1. Add an XGBoost baseline using lag features for the same selected SKUs.
2. Compare LSTM vs XGBoost on the same 15-day holdout.
3. Add explainability:
   - SHAP for XGBoost.
   - Feature ablation or perturbation importance for LSTM.

