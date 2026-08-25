# Beginner Guide: Our Stockout Prediction Project

This guide explains our dataset, prediction task, LSTM, architectural experiments, final LSTM, and DLinear.

## 1. Project in one sentence

We use FreshRetailNet-50K to predict the stock status of a product at a store for every hour of the next day, using the previous 14 days of information.

```text
Previous 14 days of one store-product series
                    |
                    v
              LSTM or DLinear
                    |
                    v
Next day's 24 hourly stock-status predictions
```

## 2. What exactly are we predicting?

Suppose product `267` is sold in store `11`. We want to answer:

> Based on the recent history of this product in this store, which hours tomorrow are likely to have a stockout?

The output contains 24 binary decisions:

```text
[hour_00, hour_01, hour_02, ..., hour_23]
```

Each output is interpreted as:

```text
0 = no stockout predicted
1 = stockout predicted
```

This is a **one-day-ahead, multi-output hourly prediction** task.

## 3. What is a SKU and what is one series?

SKU means **Stock Keeping Unit**. It identifies a product that a business tracks.

The same product can behave differently in different stores, so our actual time series is:

```text
city_id + store_id + product_id
```

We call this combination a `series_id`.

Example:

```text
series_id = 4_11_267
city_id   = 4
store_id  = 11
product_id = 267
```

This means product `267` in store `11` in city `4`.

## 4. What data do we have?

FreshRetailNet-50K is mainly organized as one row per:

```text
city + store + product + day
```

Important fields include daily sales, hourly sales, hourly stock-status values, stockout-related counts, discount, holiday/activity flags, and weather variables.

The 24-hour arrays are expanded into columns such as:

```text
stock_h00, stock_h01, ..., stock_h23
```

This is **store-level** stockout data. It is not warehouse-level inventory data. We do not know warehouse quantity, delivery quantity, supplier lead time, or the exact cause of a store stockout.

## 5. Why is this a time-series problem?

A time series is a sequence ordered by time:

```text
Day 1 -> Day 2 -> Day 3 -> ... -> Day 90
```

The order matters. Recent sales, previous stockouts, discounts, and activities may help predict tomorrow. This differs from treating every row as an unrelated table row.

## 6. Day-wise or hour-wise?

Our task is a useful hybrid:

```text
Time processing: day-wise
Prediction detail: hour-wise
```

The model receives 14 previous daily records and predicts 24 values for the following day. It is not a fully hourly model with 168 separate previous hourly steps. We chose the daily formulation because the dataset is organized around daily records and each series has limited history.

## 7. Data preparation

We:

1. converted dates to a consistent datetime format;
2. created `series_id` from city, store, and product;
3. removed missing identifiers and invalid dates;
4. removed duplicate store-product-date rows;
5. expanded the 24-hour arrays into separate columns;
6. filled missing numeric values with `0.0`;
7. sorted every series chronologically.

For each day, the target comes from the following day:

```text
Today's features      -> input
Tomorrow's 24 labels  -> target
```

We selected 15 important store-product series using total sales, total stockout hours, number of stockout days, and sales variation. Earlier days were used for training and the final 15 days for validation.

The validation is rolling one-day-ahead evaluation: for every validation date, the model uses the previous 14 available days and predicts the next day.

For a stricter production study, series selection should use training data only. Selecting using the entire period can create a small amount of selection leakage.

## 8. What are the features?

The full model used these 13 features:

```text
sale_amount
stock_hour6_22_cnt
discount
holiday_flag
activity_flag
precpt
avg_temperature
avg_humidity
avg_wind_level
hours_sale_sum
hours_sale_max
hours_stock_status_sum
hours_stock_status_mean
```

The best compact experiment used six:

```text
sale_amount
stock_hour6_22_cnt
discount
holiday_flag
hours_sale_sum
hours_stock_status_sum
```

Raw `store_id` and `product_id` are not ordinary numeric features. ID `267` is not mathematically larger than ID `100` in a meaningful way.

## 9. Input shape

The sequence length is 14:

```text
Input shape = [batch_size, 14 days, number of features]
```

For the six-feature model:

```text
[batch_size, 14, 6] -> [batch_size, 24]
```

The 24 values are the next day's hourly predictions.

## 10. What is an RNN?

RNN means **Recurrent Neural Network**. It processes a sequence one step at a time.

```text
h_t = activation(current input x_t + previous hidden state h_(t-1))
```

For our data:

```text
Day 1 -> hidden state
Day 2 -> updated hidden state
...
Day 14 -> final hidden state -> prediction
```

### Vanishing gradients

During training, the gradient travels backward through all time steps. In a basic RNN, repeated multiplication can make the gradient extremely small. This is the **vanishing gradient problem**. Early days then receive almost no learning signal, so the RNN struggles to remember older information.

## 11. What is an LSTM?

LSTM means **Long Short-Term Memory**. It is a special RNN with an explicit memory cell and gates.

It keeps:

```text
h_t = hidden state exposed to the next step
c_t = cell state carrying long-term memory
```

The main gates are the forget gate, input gate, and output gate. There is also a candidate memory update.

## 12. How does one LSTM cell work?

At time `t`, the cell combines the current input and previous hidden state:

```text
[h_(t-1), x_t]
```

Forget gate:

```text
f_t = sigmoid(W_f [h_(t-1), x_t] + b_f)
```

It decides what old memory to keep: `0` means forget and `1` means keep.

Input gate:

```text
i_t = sigmoid(W_i [h_(t-1), x_t] + b_i)
```

It controls how much new information enters memory.

Candidate memory:

```text
g_t = tanh(W_g [h_(t-1), x_t] + b_g)
```

Cell update:

```text
c_t = f_t * c_(t-1) + i_t * g_t
```

Output gate and hidden state:

```text
o_t = sigmoid(W_o [h_(t-1), x_t] + b_o)
h_t = o_t * tanh(c_t)
```

The additive cell-state update gives gradients a better path through time than a basic RNN has. LSTM does not eliminate every training problem, but it helps preserve useful information.

## 13. What is hidden size?

Hidden size is the number of values in the hidden state. If it is 32:

```text
h_t = [h_1, h_2, ..., h_32]
```

A larger hidden size can represent more patterns, but it also causes more parameters, computation, and overfitting risk. Since our window is only 14 days, a large LSTM may be unnecessary.

## 14. Why does an LSTM have many parameters?

A normal LSTM learns four transformations: forget, input, candidate, and output. Its approximate parameter count is:

```text
4 * hidden_size * (input_size + hidden_size + 1)
```

The factor 4 comes from the four transformations. This is why reducing hidden size can reduce complexity quickly.

## 15. Our first standard LSTM

The first model used:

```text
Input: 14 days x 13 features
LSTM hidden size: 96
Output: 24 logits
Loss: BCEWithLogitsLoss
Optimizer: Adam
```

Its data flow was:

```text
[batch, 14, 13] -> LSTM -> final hidden state [batch, 96]
                                      |
                              linear prediction head
                                      |
                                24 logits [batch, 24]
```

## 16. Why sigmoid and binary cross-entropy?

Each hour is a binary prediction. The model first produces one raw logit per hour. Sigmoid converts each logit into a probability:

```text
probability = sigmoid(logit)
```

Then we use a threshold:

```text
probability >= 0.5 -> stockout
probability < 0.5  -> no stockout
```

Binary cross-entropy measures how wrong these predictions are. `BCEWithLogitsLoss` combines the sigmoid and loss calculation in a numerically stable way. Positive-class weighting gives stockout errors appropriate importance when classes are imbalanced.

## 17. What does 24-output prediction mean?

The model predicts all 24 hours together:

```text
output[0]  -> hour 00
output[1]  -> hour 01
...
output[23] -> hour 23
```

This is multi-output or multi-label prediction. It is not an autoregressive process where hour 1 is fed back before predicting hour 2.

## 18. Complete model progression

We did not jump directly to the final architecture. The project progressed through several stages. The important thing is that not all stages used exactly the same target, horizon, or metric, so their scores must not be compared as if they were one single leaderboard.

### Stage 0: Original C1/backorder research direction

We first studied and attempted to replicate the approach from the C1 supply/backorder paper. That work was useful for understanding supply prediction, but it was not a genuine store-level hourly time series. The mentor then asked us to move to FreshRetailNet-50K so that our model would learn from ordered historical observations.

### Stage 1: Simple and tabular baselines

Before the deeper sequence models, the notebook also defined simple baselines such as persistence/last-value prediction and an `MLPBaseline`. These provide sanity checks: if a neural network cannot beat a simple historical rule, its extra complexity is not justified.

The first FreshRetailNet baseline was a **PyTorch Tabular Transformer**. In the code, its main name is `seq2seq_transformer`, although this early pipeline was not the same as our later 24-output store-product LSTM pipeline.

The model used:

- categorical embeddings for city, store, product, category, holiday, and activity fields;
- numeric historical variables such as discount, weather, sales, and stockout;
- projected numeric and categorical representations;
- Transformer encoder-decoder layers with positional information;
- a prediction head for the selected forecast horizon.

The configuration used a hidden dimension of 96, embedding dimension of 32, two layers, four attention heads, and dropout of 0.15.

The early tabular baseline achieved approximately:

```text
Accuracy: 55.83%
AUC:      0.5257
```

This result was close to random discrimination. It showed that treating the problem as a simple tabular classification problem was not enough. We needed chronological windows and a more explicit time-series formulation.

The `0.59` number remembered from the early work is associated with the original RNN evaluation context, not the `0.5257` AUC of this Transformer baseline. These should not be reported as the same metric.

### Stage 2: Vanilla RNN and vanilla LSTM

We then defined a common short-horizon task:

```text
Input:  previous 14 daily records
Output: next day's 24 hourly stock-status labels
```

The **vanilla RNN** used a standard `nn.RNN` followed by a linear layer producing 24 outputs. The **vanilla LSTM** used `nn.LSTM` followed by the same type of 24-output head.

The vanilla LSTM baseline used 13 features, hidden size 96, one recurrent layer, and approximately 44,952 trainable parameters. Its best single-run result was:

```text
Hour-level F1: 0.785
Recall:        0.804
Exact 24h:     0.302
```

This established the reference LSTM against which compact and modified versions were compared.

### Stage 3: SimpleRNN variations

The teammate's RNN branch tested six progressively different SimpleRNN designs:

| Variation | Technical design | Parameters | Mean F1 |
|---|---|---:|---:|
| 1 | One SimpleRNN layer, 32 hidden units, 24-output head | 2,360 | 0.7703 |
| 2 | One SimpleRNN layer, 64 hidden units, 24-output head | 6,744 | 0.7836 |
| 3 | Two stacked SimpleRNN layers, 32 units each | 4,472 | 0.7760 |
| 4 | Two-layer tanh SimpleRNN, dropout, dense ReLU head | 5,528 | 0.7650 |
| 5 | Two-layer ReLU SimpleRNN, dropout, dense head | 5,528 | 0.7681 |
| 6 | Dual-stream SimpleRNN with demand, inventory, and full-feature branches | 5,016 | 0.7782 |

Variation 6 was the teammate's final RNN design. It used:

- demand RNN: 16 hidden units;
- inventory RNN: 16 hidden units;
- full-feature RNN: 32 hidden units;
- concatenation of all three final hidden states;
- dropout;
- a dense layer followed by a 24-output head.

The original RNN experiment reported an F1 close to `0.59` under a broader and noisier evaluation setup. When the RNN was evaluated through the same Top-15 chronological pipeline as our LSTM, its mean hour-level F1 was `0.7782 +/- 0.0109` across five seeds.

This distinction matters: evaluation on thousands of sparse or inactive series is not directly comparable to evaluation on the selected Top-15 active series.

### Stage 4: LSTM complexity ablation

The mentor's question was whether a large LSTM was justified for only a 14-day window. We therefore changed one factor at a time:

| Experiment | Hidden size | Window | Features | Parameters | F1 |
|---|---:|---:|---:|---:|---:|
| Standard LSTM | 96 | 14 | 13 | 44,952 | 0.785 |
| Compact LSTM | 64 | 14 | 13 | 21,784 | 0.771 |
| Compact LSTM | 32 | 14 | 13 | 6,808 | 0.753 |
| Short-window LSTM | 64 | 10 | 13 | 21,784 | 0.763 |
| Short-window LSTM | 64 | 7 | 13 | 21,784 | 0.776 |
| Feature-reduced LSTM | 64 | 14 | 6 | 19,992 | 0.788 |
| Feature-reduced LSTM | 32 | 14 | 6 | 5,912 | 0.790 |

The main result was that feature reduction worked better than simply shrinking the hidden state. The six-feature h32 model reduced parameters by approximately 86.8% while slightly improving F1 over the full h96 baseline.

### Stage 5: LSTM cell and information-flow modifications

After the basic ablation, we changed the architecture itself:

- **CIFG-LSTM:** input and forget gates were coupled using `forget = 1 - input`; this reduced parameters but performed worse.
- **Dual-stream LSTM:** demand and inventory features were processed by separate recurrent branches.
- **Dual-stream CIFG-LSTM:** combined separate branches with the coupled-gate cell; it also underperformed.
- **Unequal branch sizes:** tested demand hidden size 20 and inventory hidden size 12.
- **General shortcut:** passed recent features directly to the fusion layer.
- **Inventory shortcut:** passed only the latest inventory features directly to fusion; this was strongest.
- **Gated shortcut:** learned how much to use from recurrent memory versus the shortcut.
- **Static ID embeddings:** represented store/product identities with learned vectors; they did not consistently improve F1 on only 15 series.

### Stage 6: Dual-stream inventory-shortcut LSTM

The strongest LSTM design was:

```text
Demand features    -> LSTM(16) -------+
                                      |
Inventory features -> LSTM(16) -------+-> fusion -> 24 outputs
                                      |
Latest inventory features ------------+
```

It achieved approximately:

```text
Single-seed F1: 0.7926
Parameters:     4,600
```

In the later five-seed comparison against the teammate's RNN, it achieved mean F1 of approximately `0.7941 +/- 0.0040`, compared with the RNN's `0.7782 +/- 0.0109`.

The LSTM had both better average F1 and lower variation across seeds. This is the main evidence supporting LSTM over the SimpleRNN for our short-horizon task.

### Stage 7: DLinear comparison

After understanding and modifying LSTM, we compared it with DLinear as the selected lightweight alternative. We tested several DLinear variants for the same next-day, 24-hour stock-status task.

We did not stop at one DLinear implementation. We tested several ways to specialize decomposition for our 24-hour stock-status target:

| DLinear variant | Technical change | Parameters | F1 |
|---|---|---:|---:|
| Baseline DLinear, standard kernel `k=5` | Basic moving-average trend/residual split | 6,768 | 0.8054 |
| Multi-Kernel DLinear, `k=3,5,7` | Uses multiple moving-average scales and combines them | 13,539 | **0.8131** |
| Channel-Gated DLinear | Learns which feature channels deserve more weight | 7,210 | 0.7992 |
| Hourly Slot-Specific DLinear | Uses separate output heads for the 24 hourly slots | 6,744 | 0.8058 |
| Gated Residual DLinear | Adds a GLU-style nonlinear residual shortcut | 18,096 | 0.8124 |

The current concluding candidate is therefore the **Multi-Kernel DLinear**:

```text
F1:        0.8131
Parameters: 13,539
```

It improves over the best LSTM comparison of approximately `0.7941` while remaining far smaller than the Transformer-family models. The standard DLinear result of `0.7932` is useful as the plain DLinear baseline; the multi-kernel result is the improved DLinear architecture.

The LSTM remains important as our architectural research contribution. Multi-Kernel DLinear becomes the practical concluding model for this dataset because it gives the best balance of F1 and complexity among the selected final candidates.

## Final-report standardized protocol

The final BTP report should use one consistent protocol for comparing the selected models. This is the version to describe in the methodology chapter. Earlier ablation numbers should be labelled as ablations because some used the original 13-feature or reduced six-feature input and a fixed `0.5` threshold.

### Standardized 10-dimensional input

The final-report input vector is:

```text
1.  sale_amount
2.  stock_hour6_22_cnt
3.  discount
4.  holiday_flag
5.  hours_sale_sum
6.  hours_stock_status_sum
7.  dow_sin
8.  dow_cos
9.  stockout_rolling_3
10. sales_momentum
```

The last four features add temporal context:

- `dow_sin` and `dow_cos` represent day-of-week cyclically, so Sunday and Monday are close rather than far apart numerically;
- `stockout_rolling_3` summarizes recent stockout frequency over three days;
- `sales_momentum` measures the recent change in sales.

### Top-15 selection score

After min-max normalization of its components:

```text
score = 0.40 * sales
      + 0.25 * stockout_hours
      + 0.20 * stockout_days
      + 0.15 * sales_std
```

This selects active, variable, and stockout-relevant store-product series for the controlled experiment.

### Focal Loss

The final standardized framework uses Focal Loss rather than ordinary binary cross-entropy:

```text
L = -alpha_t * (1 - p_t)^gamma * log(p_t)
```

The planned settings are `gamma = 2.0` and `alpha = 0.5`. Focal Loss reduces the influence of easy majority-class examples and focuses training on difficult stockout transitions.

### Threshold scanning

The model outputs probabilities, but a binary decision requires a threshold. Instead of assuming `0.5` is always optimal, the standardized framework tests thresholds from `0.10` to `0.90` in steps of `0.05`. The threshold that maximizes validation hour-level F1 is selected and then used for the final holdout evaluation.

### Fixed comparison requirements

For the final LSTM-versus-DLinear comparison, keep these identical:

- Top-15 series;
- 14-day history window;
- 24-hour target vector;
- chronological final-15-day holdout;
- 10-dimensional feature vector;
- scaler fitted only on training data;
- Focal Loss settings;
- threshold-selection procedure;
- hour-level F1 as the primary metric.

### Common experimental controls

For the main 24-hour comparison, we tried to keep the evaluation fair:

- same FreshRetailNet-50K source;
- same Top-15 selected store-product series;
- same 14-day input window;
- same final 15-day chronological holdout;
- same 24-output target;
- Adam optimizer with learning rate `1e-3` for the PyTorch sequence models;
- batch size `64` and approximately 10 training epochs;
- standardization fitted using the training period;
- positive-class weighting for the binary stockout loss in the LSTM pipeline;
- multiple random seeds for the final RNN-versus-LSTM comparison;
- hour-level F1 as the primary metric.

Some other branches use different loss functions, thresholds, horizons, or feature engineering. Their results must be labelled separately rather than mixed into the main table.

## 19. Detailed architectural experiments

Our mentor asked:

> If the time window is small, why use an unnecessarily complex LSTM?

We tested the following changes.

### Feature reduction

We reduced 13 features to six important features. The feature-reduced h32 model had approximately 5,912 parameters and F1 around 0.790, compared with approximately 44,952 parameters and F1 around 0.785 for the first full model.

### Smaller hidden sizes

We tested hidden sizes 64, 32, 16, and 8. Very small sizes saved computation but lost too much predictive capacity. h32 was a better compact balance.

### CIFG-LSTM

CIFG means **Coupled Input Forget Gate**. It uses:

```text
forget gate = 1 - input gate
```

This saves parameters, but it forces two decisions to be linked. It performed worse in our experiment, so we did not select it.

### Dual-stream LSTM

We split the six features into two logical groups:

```text
Demand stream: sale_amount, discount, holiday_flag, hours_sale_sum
Inventory stream: stock_hour6_22_cnt, hours_stock_status_sum
```

Each group gets its own small LSTM:

```text
Demand features    -> Demand LSTM    -> demand memory
Inventory features -> Inventory LSTM -> inventory memory
                                          |
                              concatenate both memories
                                          |
                                      prediction head
```

The motivation is that demand pressure and inventory status are related, but they are not the same type of signal.

### Shortcut connection

The most recent day may be more important than older days. We pass recent information directly to the fusion layer:

```text
LSTM memory + latest-day features -> fusion -> output
```

This prevents the LSTM from having to preserve every recent detail only inside its hidden state.

### Inventory shortcut

The best modification passes the latest inventory features directly to the output path:

```text
demand memory
inventory memory
latest inventory features
             |
             v
        fusion layer
             |
        24 predictions
```

This is task-specific because recent stockout information is directly relevant to tomorrow's stockout status.

### Gated shortcut

We also tested a learnable gate:

```text
gate = sigmoid(learned function of memory and shortcut)
final = gate * memory + (1 - gate) * shortcut
```

It lets the model decide how much to trust memory versus recent information, but it added parameters and did not beat the inventory-shortcut model.

### Store/product embeddings

We mapped store and product IDs to learned vectors and added them at the fusion layer. This was the correct way to use IDs categorically, but it did not consistently improve F1 on only 15 selected series. It also increased parameters, so we did not select it.

## 20. Our final LSTM architecture

```text
14 days x 6 features
        |
        +--> Demand LSTM, hidden size 16
        |
        +--> Inventory LSTM, hidden size 16
                         |
        demand memory + inventory memory
                         |
        latest inventory features shortcut
                         |
                    fusion layer
                         |
                    linear head
                         |
                   24 hourly logits
```

The selected LSTM has two specialized recurrent streams, hidden size 16 in each stream, six input features, direct use of latest inventory information, and 24 output values. It has approximately 4,600 trainable parameters.

The best single-seed result was:

```text
Hour-level F1: 0.7926
Parameters: 4,600
```

The later five-seed comparison report records mean F1 of approximately `0.7941 +/- 0.0040` for this LSTM comparison. Always report the run configuration and metric definition with the number.

## 21. Evaluation metrics

### Accuracy

The fraction of all hourly decisions that are correct. It can be misleading when classes are imbalanced.

### Precision

Among predicted stockout hours, how many were truly stockout hours?

```text
precision = true positives / (true positives + false positives)
```

### Recall

Among actual stockout hours, how many did the model detect?

```text
recall = true positives / (true positives + false negatives)
```

### F1

F1 combines precision and recall:

```text
F1 = 2 * precision * recall / (precision + recall)
```

F1 is useful because missed stockouts and false alarms both matter.

### Exact 24-hour match

A day counts as correct only when all 24 predictions are correct. This is much stricter than hour-level F1.

### Mean stockout-hour count error

This measures the average difference between predicted and true numbers of stockout hours per day.

## 22. Why is DLinear important?

LSTM was our main architecture investigation, but the final system should be selected based on performance and complexity, not on the model name.

Our latest comparison uses DLinear as the concluding model because it is lightweight and has higher F1 than the LSTM on the same research task. The current repository does not contain a DLinear result artifact, so its exact F1 number must be copied from the final DLinear run rather than invented here.

## 23. What is DLinear?

DLinear means **Decomposition Linear**. It is a lightweight time-series forecasting model.

Its main idea is:

> Split a time series into a slowly changing trend and a remaining short-term component. Forecast both components with simple linear mappings and add them together.

Unlike an LSTM, DLinear does not repeatedly process the sequence with gates and hidden states.

## 24. How DLinear decomposes the sequence

Let the input sequence be `x`.

First, a moving average estimates the trend:

```text
trend = moving_average(x)
```

The remaining part is:

```text
seasonal_or_residual = x - trend
```

Two linear mappings forecast the parts:

```text
trend forecast    = linear_trend(trend)
residual forecast = linear_residual(residual)
```

Finally:

```text
final forecast = trend forecast + residual forecast
```

For our task, the final output is converted into 24 hourly stock-status decisions.

## 25. DLinear diagram

```text
Previous daily sequence
          |
          v
Moving-average decomposition
          |
          +------------------+
          |                  |
          v                  v
       Trend             Residual/seasonal
          |                  |
          v                  v
   Linear projection   Linear projection
          |                  |
          +--------+---------+
                   |
                   v
          Add both forecasts
                   |
                   v
          Next 24 hourly outputs
```

## 26. Why is DLinear lightweight?

LSTM repeatedly applies several gate transformations at every time step and maintains hidden and cell states. DLinear mainly uses decomposition and linear projections.

Therefore DLinear generally has fewer parameters, lower memory use, faster training, faster inference, simpler debugging, and lower overfitting risk on a short dataset.

This is important here because each selected series has limited history and the input window is only 14 days.

## 27. Why can DLinear beat LSTM?

More expressive does not always mean better. DLinear can win when:

- the input window is short;
- the useful structure is mostly trend and recent variation;
- the dataset is small;
- the LSTM has unnecessary parameters;
- recurrent gates do not learn additional useful nonlinear structure.

This is why our concluding model can be simpler than our architectural research model.

## 28. LSTM versus DLinear

| Property | Final LSTM | DLinear |
|---|---|---|
| Main idea | Recurrent memory with gates | Trend/residual decomposition plus linear mappings |
| Input | Previous 14 daily records | Previous 14 daily records or equivalent window |
| Output | 24 hourly predictions | 24 hourly predictions |
| Complexity | Higher | Lower |
| Memory mechanism | Hidden state and cell state | No recurrent hidden-state memory |
| Strength | Nonlinear temporal dependencies | Efficient simple temporal structure |
| Weakness | More parameters and training complexity | May miss complex nonlinear patterns |
| Role in our project | Main LSTM architecture contribution | Lightweight concluding model |

## 29. How to explain our final decision

Use this explanation:

> We first investigated LSTM because stockout prediction is a temporal problem. We tested feature reduction, smaller hidden sizes, separate demand and inventory streams, shortcut connections, CIFG gates, gated shortcuts, and categorical embeddings. The best LSTM was a dual-stream inventory-shortcut model with about 4,600 parameters. We then compared it with DLinear. Since DLinear achieved higher F1 with lower complexity on the same task, we selected DLinear as the concluding model while retaining the modified LSTM as our main architectural research contribution.

This shows that we compared alternatives instead of assuming that a more complex neural network must be better.

## 30. What did we contribute?

Our work includes:

1. moving toward a real time-series formulation;
2. defining a store-product next-day stockout task;
3. predicting 24 hourly values instead of only one daily label;
4. studying how much LSTM complexity is necessary;
5. separating demand and inventory signals;
6. adding a task-specific inventory shortcut;
7. testing categorical embeddings correctly;
8. comparing the modified LSTM with simpler alternatives;
9. selecting DLinear as the lightweight final model because it performed better on the current task.

## 31. Limitations to state honestly

- We have store-level stock status, not warehouse inventory.
- We do not know exact inventory quantity or the cause of a stockout.
- Each series has limited history.
- The first experiment uses only 15 selected series.
- Results can vary with random seed, so mean and standard deviation are better than one run.
- Exact 24-hour match is stricter than hour-level F1.
- The current validation is rolling one-day-ahead prediction, not a single prediction made 15 days into the future without updated observations.
- Series selection should ideally be performed using training data only.
- The exact DLinear F1 must be reported from its final saved result artifact.

## 32. Final takeaway

```text
FreshRetailNet-50K
        |
Clean store-product daily time series
        |
Previous 14 days
        |
Predict next day's 24 hourly stock statuses
        |
Investigate compact, task-specific LSTM architectures
        |
Best LSTM: dual-stream inventory shortcut
        |
Compare with simpler forecasting model
        |
Concluding model: DLinear
```

The LSTM work taught us how memory, gates, feature grouping, and shortcuts can be designed for stockout prediction. The DLinear result teaches an equally important lesson: a lightweight model can be the better final choice when it achieves higher F1 with much lower complexity.
