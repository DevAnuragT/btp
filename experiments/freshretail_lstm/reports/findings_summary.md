# FreshRetailNet LSTM Findings Summary

## Objective

We used FreshRetailNet-50K to perform time-series stock-status prediction.

Goal:

> Use past SKU-store behavior to predict the next day's 24-hour stock status.

This was done to move beyond the earlier static backorder dataset and satisfy
the requirement for a true time-series dataset.

## Dataset and Setup

- Dataset: FreshRetailNet-50K
- Unit of analysis: SKU-store time series
- Selected items: top 15 distinct SKUs
- Input window: previous 14 days
- Prediction target: next day's 24 hourly stock-status values
- Train period: around 2.5 months
- Validation period: last 15 days

Input shape to LSTM:

```text
batch_size x time_steps x features
```

For the main setup:

```text
batch_size x 14 x 13
```

Output shape:

```text
batch_size x 24
```

Each output value represents one hour of the next day.

## Initial LSTM Result

The first standard LSTM used:

- hidden size: 96
- sequence length: 14 days
- features: 13
- parameters: 44,952

Best validation result:

| Metric | Value |
|---|---:|
| Hour-level accuracy | 0.791 |
| Hour-level precision | 0.767 |
| Hour-level recall | 0.804 |
| Hour-level F1 | 0.785 |
| Exact 24-hour match | 0.302 |
| Mean stockout-hour count error | 4.67 hours |

## Why We Ran Ablation Experiments

Mentor's concern:

> If RNN gives similar accuracy, why use a more complex LSTM for such a small
> time window?

This is a valid concern because our input window is only 14 days. So we tested
whether LSTM complexity can be reduced without losing performance.

## Ablation Experiments

We tested:

1. Hidden-size reduction
2. Sequence-length reduction
3. Feature reduction

| Experiment | Hidden Size | Window | Features | Params | Best F1 | Recall |
|---|---:|---:|---:|---:|---:|---:|
| Standard LSTM | 96 | 14 | 13 | 44,952 | 0.785 | 0.804 |
| Compact LSTM | 64 | 14 | 13 | 21,784 | 0.771 | 0.775 |
| Compact LSTM | 32 | 14 | 13 | 6,808 | 0.753 | 0.811 |
| Short-window LSTM | 64 | 10 | 13 | 21,784 | 0.763 | 0.756 |
| Short-window LSTM | 64 | 7 | 13 | 21,784 | 0.776 | 0.774 |
| Feature-reduced LSTM | 64 | 14 | 6 | 19,992 | 0.788 | 0.816 |
| Feature-reduced LSTM | 32 | 14 | 6 | 5,912 | 0.790 | 0.813 |

## Best Finding

The best model was:

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

Compared to the standard LSTM:

| Model | Params | F1 |
|---|---:|---:|
| Standard LSTM | 44,952 | 0.785 |
| Compact feature-reduced LSTM | 5,912 | 0.790 |

Parameter reduction:

```text
~86.8%
```

## Interpretation

- A full-size LSTM is probably unnecessary for this short-window task.
- Reducing hidden size alone reduced complexity but also reduced F1.
- Reducing features worked better than only reducing hidden size.
- The compact feature-reduced LSTM gave slightly better F1 with far fewer
  parameters.
- This makes the LSTM direction more defensible, but only as a compact LSTM,
  not as a heavy standard LSTM.

## Selected Reduced Features

The best compact model used only six features:

1. `sale_amount`
2. `stock_hour6_22_cnt`
3. `discount`
4. `holiday_flag`
5. `hours_sale_sum`
6. `hours_stock_status_sum`

These features capture:

- demand level,
- previous stockout behavior,
- promotion/discount effects,
- holiday effect,
- hourly sales summary,
- hourly stock-status summary.

## Mentor-Facing Conclusion

We agree that if RNN gives similar accuracy, a standard LSTM is not justified.
Therefore, we tested compact LSTM variants.

The experiments show that a feature-reduced compact LSTM can reduce parameters
by about 87% while slightly improving F1 compared with the standard LSTM.

So the final LSTM proposal should not be:

> Use a full LSTM because LSTM is powerful.

It should be:

> Use a compact LSTM and compare it against RNN under the same setup. If compact
> LSTM gives better recall/F1 with acceptable complexity, it is justified;
> otherwise RNN should be preferred.

## Remaining Work

To make the final model decision, we still need the teammate's RNN results under
the same setup:

- same top 15 SKUs,
- same 14-day input window,
- same 15-day validation period,
- same 24-hour output target,
- same metrics,
- parameter count and training time.

Then we can conclude whether compact LSTM is better than RNN or whether RNN is
the more efficient choice.

