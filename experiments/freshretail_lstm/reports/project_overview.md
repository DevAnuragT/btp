# Our Project So Far

We started by studying the paper **“Maximising Supply for C1”** and tried to understand and replicate its research approach. The paper mainly works with supply/backorder-style prediction, but our mentor suggested using a real time-series dataset for stockout prediction.

We selected the **FreshRetailNet-50K** dataset because it contains store-level product data with hourly sales and stock-status information.

## What the dataset contains

Each record represents:

```text
City + Store + Product + Day
```

For example:

```text
City 4, Store 11, Product 267
```

The dataset contains:

- daily sales;
- hourly sales;
- hourly stock-status values;
- discount information;
- holidays and activities;
- weather conditions;
- stockout-related hourly counts.

This is **store-level stockout data**, not warehouse-level data. We cannot directly know whether a warehouse was out of stock.

## Our prediction problem

We decided to predict the next day’s stockout status for every hour.

The model receives:

```text
Previous 14 days of data
```

and predicts:

```text
The next day’s 24 hourly stock-status values
```

So the model output looks like:

```text
[hour 0, hour 1, hour 2, ..., hour 23]
```

This is a daily time-series model with hourly predictions.

## Data preparation

We:

- cleaned missing and duplicate records;
- created a unique `series_id` using city, store, and product;
- converted the hourly arrays into separate columns;
- created features such as total sales, maximum hourly sales, stockout hours, discount, weather, and activity;
- selected the top 15 statistically important store-product series;
- used approximately 2.5 months for training;
- used the final 15 days for validation.

The selected series were chosen using sales activity, number of stockout hours, number of stockout days, and sales variation.

## Why we used LSTM

LSTM is a type of recurrent neural network designed for sequential data.

It reads the timeline in order:

```text
Day 1 → Day 2 → Day 3 → ... → Day 14
```

It uses gates to decide:

- what old information to remember;
- what information to forget;
- what new information to use;
- what information to pass to the output.

This is useful because stockout risk may build over multiple days due to increasing sales, repeated stockouts, discounts, or activities.

## Experiments we performed

We did not use only the standard LSTM. We experimented with simpler and more problem-specific architectures:

- reduced feature LSTM;
- smaller hidden sizes;
- compact CIFG-LSTM;
- separate demand and inventory processing streams;
- shortcut connections from recent features;
- inventory-focused shortcut connections;
- gated shortcut connections.

The goal was to reduce unnecessary complexity while improving or maintaining performance.

## Best result

Our best model was:

```text
Dual-stream inventory-shortcut LSTM
Hidden sizes: 16 and 16
Parameters: 4,600
Best F1 score: 0.7926
```

The model separates:

- demand-related features, such as sales;
- inventory-related features, such as stockout information.

It then gives more direct access to recent inventory information through a shortcut connection.

## Main conclusion

The modified LSTM performed slightly better than the original reduced-feature LSTM while using relatively few parameters.

This suggests that the architecture can be made more suitable for this particular stockout problem instead of blindly using a standard LSTM.

However, we should not claim that it is universally better yet. We need more validation, repeated runs with different random seeds, and comparison with the teammate’s RNN and simpler baselines.

## Current limitation

Our data tells us about stockouts at individual stores. It does not tell us:

- warehouse inventory;
- replenishment quantity;
- delivery delays;
- supplier lead time;
- the actual reason for a stockout.

Therefore, our current research claim should be:

> We predict hourly stockout status for selected store-product series using a daily LSTM-based time-series model.
