#set page(
  paper: "a4",
  margin: (x: 2cm, y: 2.5cm),
  header: align(right, text(size: 9pt, fill: gray)[Standardized Experimental Framework & Research Specification]),
  footer: [
    #align(center)[#text(size: 9pt, fill: gray)[Page #context counter(page).display()]]
  ]
)

#set text(
  size: 10pt,
  lang: "en"
)

#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.1")

// Title Section
#align(center)[
  #v(0.5cm)
  #text(size: 20pt, weight: "bold", fill: rgb("#111827"))[Standardized Experimental Framework] \
  #v(0.3cm)
  #text(size: 13pt, weight: "medium", fill: rgb("#374151"))[Next-Day 24-Hour Hourly Stock Status Forecasting: Task Definition, Dataset Standardization, Metrics & Architectural Protocols] \
  #v(0.4cm)
  #text(size: 10pt, style: "italic", fill: rgb("#4b5563"))[
    Authors: Anurag Thakur, Samyak, Vibhor Kumar \
    Date: August 2026 | Master Technical & Experimental Standardization Document
  ]
  #v(0.5cm)
]

#rect(width: 100%, fill: rgb("#f3f4f6"), radius: 4pt, inset: 12pt)[
  #text(weight: "bold", size: 11pt)[Executive Objective] \
  #v(4pt)
  This document establishes the single authoritative, standardized experimental specification for all research, dataset processing, model development, metric evaluation, and architectural comparisons across our Next-Day 24-Hour Stock Status Prediction initiatives.

  Every experiment in this repository must strictly adhere to the unit of analysis, statistical SKU selection formula, chronological train/validation splitting, loss functions, dynamic thresholding, and objective metric hierarchy defined herein.
]

#v(0.5cm)

== Problem Formulation & Operational Context

=== Business Context & Research Motivation
In high-frequency fresh retail supply chains (e.g., Dingdong Fresh, instant grocery delivery), stockout events lead to severe operational damage: lost revenue, unfulfilled customer carts, and reduced long-term customer retention. 

Traditional inventory forecasting models predict coarse *daily aggregated demand*, which fails to capture intra-day inventory depletion patterns. A store may appear "in-stock" on a daily aggregate basis even if it suffered a critical 4-hour stockout during peak evening sales (5 PM – 9 PM).

To solve this, our framework operates at *hourly granularity*, predicting the 24-hour operational stock status vector for the target day $t+1$.

=== Task Definition
Given a sequence of $N$ historical days ($N in {7, 10, 14}$), the objective is to predict the binary stock-status matrix for all active operational hours of the subsequent day ($t+1$):
$ X_(t-N+1:t) in RR^(N times D) => \hat(Y)_(t+1) in {0, 1}^24 $
where $D$ represents the input feature dimension, and $y_h = 1$ indicates a stockout state during hour $h in [1..24]$ while $y_h = 0$ indicates in-stock status.

== Dataset Specification & Preprocessing Standards

=== Primary Dataset: FreshRetailNet-50K
All standardized experiments utilize the benchmark *FreshRetailNet-50K* dataset (Dingdong Inc.), containing daily and hourly sales, inventory counts, promotion discounts, holiday indicators, and store metadata across thousands of SKUs over multiple months.

=== Unit of Analysis Definition
Stockout events are localized to store replenishment cycles and regional supply chains. Therefore, the unit of analysis is defined at the strict *store-product time-series level*:
$ "series_id" = "city_id" + "store_id" + "product_id" $

Grouping by product alone or store alone is explicitly prohibited, as inventory levels differ across store locations for the exact same product.

=== Statistical Top 15 SKU Selection Formula
To eliminate non-moving items and focus evaluation on high-impact, non-trivial demand patterns, SKUs are statistically selected using a weighted multi-factor score:
$ "Score" = 0.40 dot "Sales" + 0.25 dot "StockoutHours" + 0.20 dot "StockoutDays" + 0.15 dot "SalesStd" $

where all 4 components are min-max normalized across all unique product IDs:
1. *"Sales"*: Total historical sales volume across all stores.
2. *"StockoutHours"*: Cumulative count of historical stockout hours.
3. *"StockoutDays"*: Total days with at least one stockout hour.
4. *"SalesStd"*: Standard deviation of daily sales (measuring demand volatility).

The top 15 products with the highest composite selection scores are retained for standardized benchmark evaluation.

=== Feature Standardization Protocol
Each time step $t$ is represented by a 10-dimensional standardized feature vector:
1. `sale_amount`: Total daily sales quantity (standardized).
2. `stock_hour6_22_cnt`: Count of operational stockout hours during 6 AM – 10 PM (standardized).
3. `discount`: Average promotional discount rate $[0, 1]$.
4. `holiday_flag`: Binary indicator for national/statutory holidays $\{0, 1\}$.
5. `hours_sale_sum`: Sum of intra-day hourly sales (standardized).
6. `hours_stock_status_sum`: Sum of intra-day hourly stockout status (standardized).
7. `dow_sin`: Cyclical day-of-week sine transformation $sin(2 pi dot "dow" / 7)$.
8. `dow_cos`: Cyclical day-of-week cosine transformation $cos(2 pi dot "dow" / 7)$.
9. `stockout_rolling_3`: 3-day rolling window stockout day count.
10. `sales_momentum`: 1-day sales velocity change $Delta "Sales"_t = "Sales"_t - "Sales"_(t-1)$.

=== Chronological Train / Validation Data Splitting
To prevent data leakage and simulate true deployment conditions:
- *Training Set*: Chronological historical data up to cutoff date $T_("train") = T_("max") - 15 " days"$ (~2.5 months of sequence data).
- *Validation Set*: Final 15 chronological days ($T_("max") - 15 " days" < t \le T_("max")$).
- *Scaler Fitting*: `StandardScaler` must be fitted *strictly on the training split* and applied to transform both train and validation sets.

== Metric Hierarchy & Objective Measurement Rationale

=== Primary Operational Metric: Hour-Level F1-Score
In retail datasets, operational hours are dominated by in-stock status (60:40 to 80:20 in-stock ratio). Under such class imbalance:
- Raw Accuracy is *misleading*: A trivial model predicting 100% in-stock achieves 80% accuracy while missing *100% of stockout events*.
- *Hour-Level F1-Score* is our *Primary Objective Metric*:
  $ F_1 = 2 dot ("Precision" dot "Recall") / ("Precision" + "Recall") $
  It measures the harmonic mean of stockout detection precision and recall without being inflated by true negative in-stock hours.

=== Secondary Metric Evaluation Protocol
1. *Hour-Level Recall*: Percentage of actual stockout hours correctly detected ($"TP" / ("TP" + "FN")$). Crucial for preventing unfulfilled customer orders.
2. *Exact 24-Hour Match Rate (%)*: Percentage of evaluation days where the model correctly predicts *all 24 hourly stockout statuses perfectly*:
   $ "ExactMatch" = 1/M sum_(i=1)^M bb(I)(\hat(bold(y))_i = bold(y)_i) $
3. *Mean Absolute Hour Count Error (MAE in hours)*: Mean absolute difference between predicted total stockout hours and actual total stockout hours per day:
   $ "MAE"_("hours") = 1/M sum_(i=1)^M | sum_(h=1)^24 \hat(y)_(i, h) - sum_(h=1)^24 y_(i, h) | $
4. *Inference Latency (ms/seq)*: Average CPU/GPU prediction time per sequence (milliseconds).
5. *Parameter Count*: Total trainable parameters.

== Training Protocol & Optimization Standards

=== Loss Function Standard: Focal Loss
Standard Binary Cross Entropy (BCE) fails under class imbalance because easy in-stock negative examples dominate the loss gradient. All models are optimized using *Focal Loss* ($gamma = 2.0, alpha = 0.5$):
$ cal(L)_("Focal") = -alpha_t (1 - p_t)^gamma log(p_t) $
where $p_t = p$ if $y=1$ else $1-p$. This dynamically down-weights easy non-stockout examples and forces the network to learn hard stockout transitions.

=== Dynamic Threshold Optimization ($tau^*$)
Fixed classification thresholds ($tau = 0.50$) are suboptimal under imbalanced Focal Loss outputs. Every model evaluation scans thresholds $tau in [0.10, 0.90]$ with step size $0.05$ on the validation set to select the optimal threshold $tau^*$ that maximizes the Hour-Level F1-Score:
$ tau^* = arg max_(tau in [0.1, 0.9]) F_1(bold(y), sigma(\hat(bold(z))) >= tau) $

== Standardized Model Architecture Taxonomy

All models evaluated in this repository fall into 5 standardized architectural families:

1. *Baseline Recurrent Models*: Standard LSTMs (hidden sizes $h in {16, 32, 64, 96}$) and Coupled Input-Forget Gate LSTMs (CIFG).
2. *Ablation & Compact Variations*: Window length ablations ($N in {7, 10, 14}$) and feature dimension reductions ($F in {4, 6, 10, 13}$).
3. *Architectural Tweaks*:
   - *Dual-Stream Decoupled Networks*: Parallel Conv1D+GRU streams separating demand and inventory.
   - *Direct Inventory Residual Shortcuts*: Direct linear shortcuts projecting inventory status to output logits.
   - *Dynamic Sigmoid Gated Fusion*: $alpha = sigma(W_g ["Demand" || "Inventory"] + b_g)$.
4. *State-of-the-Art (SOTA) Time-Series Models*:
   - *DLinear*: Trend-Seasonal linear decomposition ($6,768$ params, $0.3326$ ms/seq, F1 = $0.7932$).
   - *N-HiTS*: Multi-rate sampling with 3 hierarchical interpolation blocks ($0.8049$ F1).
   - *WaveNet-GLU*: Dilated causal convolutions ($d=1, 2, 4, 8$) with Gated Linear Units ($31.1%$ Exact Match).
   - *TFT-Lite*: Softmax Variable Selection Networks (VSN) + Multi-Head Self-Attention ($0.8023$ F1, $4.018$ hrs MAE).
   - *PatchTST*: Sub-series patching ($P=16, S=8$) with Channel Independence ($0.8041$ F1).
   - *Hourly Query Cross-Attention Transformer*: 24 learnable hourly slot queries ($0.8056$ F1, $90.08%$ Recall).
5. *Super-Ensemble Stacking*: Weighted probability averaging across top models (*0.8096* F1, *28.9%* Exact Match, *4.124 hrs* MAE).

== Summary & Compliance Requirements

- Any new model added to the codebase MUST be trained using `train_sota_hourly_models.py` or equivalent standardized pipelines.
- Results MUST be saved to JSON/CSV format containing all 8 standardized metrics (`best_hour_level_f1`, `best_hour_level_recall`, `best_hour_level_precision`, `best_hour_level_accuracy`, `best_exact_24h_match_rate`, `best_mean_absolute_hour_count_error`, `inference_ms_per_sequence`, `parameters`).
- Reference compiled documentation artifacts:
  - PDF: [docs/standardized_experiment_framework.pdf](standardized_experiment_framework.pdf)
  - Markdown: [docs/standardized_experiment_framework.md](standardized_experiment_framework.md)
