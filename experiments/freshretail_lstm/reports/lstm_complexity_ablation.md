# LSTM Complexity Ablation

## Why We Ran This

Mentor's question:

> If RNN gives similar accuracy, why use a much more complex LSTM for a short
> sequence window?

So we tested whether the LSTM can be simplified while preserving performance.
The goal is not to defend a large LSTM blindly. The goal is to find the simplest
LSTM variant that still performs well.

## Fixed Setup

- Dataset: FreshRetailNet-50K
- Selected items: top 15 distinct SKUs
- Task: next-day 24-hour stock-status prediction
- Train period: first ~2.5 months
- Validation period: last 15 days
- Output: 24 binary hourly stock-status predictions
- Metrics:
  - hour-level F1
  - hour-level recall
  - exact 24-hour match
  - stockout-hour count error
  - trainable parameters
  - training time

## Experiments

| Experiment | Hidden Size | Window | Features | Params | Best F1 | Recall | Exact 24h Match |
|---|---:|---:|---:|---:|---:|---:|---:|
| Standard LSTM | 96 | 14 | 13 | 44,952 | 0.785 | 0.804 | 0.302 |
| Compact LSTM | 64 | 14 | 13 | 21,784 | 0.771 | 0.775 | 0.227 |
| Compact LSTM | 32 | 14 | 13 | 6,808 | 0.753 | 0.811 | 0.138 |
| Short-window LSTM | 64 | 10 | 13 | 21,784 | 0.763 | 0.756 | 0.200 |
| Short-window LSTM | 64 | 7 | 13 | 21,784 | 0.776 | 0.774 | 0.302 |
| Feature-reduced LSTM | 64 | 14 | 6 | 19,992 | 0.788 | 0.816 | 0.284 |
| Feature-reduced LSTM | 32 | 14 | 6 | 5,912 | 0.790 | 0.813 | 0.213 |

## Main Finding

The best F1 came from:

```text
Feature-reduced LSTM
hidden size = 32
sequence length = 14
features = 6
```

It achieved:

- F1: 0.790
- Recall: 0.813
- Parameters: 5,912

Compared with the standard LSTM:

- Standard LSTM params: 44,952
- Compact feature-reduced LSTM params: 5,912
- Parameter reduction: ~86.8%
- F1 slightly improved: 0.785 -> 0.790

## Interpretation

This directly answers the complexity concern.

For this short 14-day prediction problem, the full LSTM is probably unnecessary.
A compact LSTM with fewer hidden units and fewer features gives similar or
slightly better performance with much lower complexity.

The most useful simplification was feature reduction, not just reducing hidden
size. This suggests that the model mainly needs a small set of strong signals:

- `sale_amount`
- `stock_hour6_22_cnt`
- `discount`
- `holiday_flag`
- `hours_sale_sum`
- `hours_stock_status_sum`

Weather/activity features did not appear necessary in this first ablation.

## Suggested Mentor Response

We agree that a full LSTM may not be justified if RNN performs similarly.
Therefore, we ran a complexity ablation. The results show that a compact
feature-reduced LSTM can reduce parameters by about 87% while matching or
slightly improving the standard LSTM's F1-score. So our proposed LSTM direction
should not be a heavy standard LSTM, but a compact LSTM variant.

## Next Step

Compare this best compact LSTM directly against teammate's RNN under the same:

- selected top 15 SKUs,
- 14-day input window,
- 15-day validation period,
- 24-hour output target,
- metrics and parameter count.

If RNN still matches the compact LSTM with fewer parameters, then RNN should be
preferred. If compact LSTM improves recall/F1 enough, then it is justified.

