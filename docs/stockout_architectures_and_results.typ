#set page(
  paper: "a4",
  margin: (x: 2cm, y: 2.5cm),
  header: align(right, text(size: 9pt, fill: gray)[Next-Day 24-Hour Stock Status Prediction: Architecture Analysis & Empirical SOTA Benchmarks]),
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
  #text(size: 20pt, weight: "bold", fill: rgb("#111827"))[Next-Day 24-Hour Stock Status Prediction] \
  #v(0.3cm)
  #text(size: 13pt, weight: "medium", fill: rgb("#374151"))[Repository Architecture Analysis, Empirical SOTA Benchmarks, and Model Performance] \
  #v(0.4cm)
  #text(size: 10pt, style: "italic", fill: rgb("#4b5563"))[
    Authors: Anurag Thakur, Samyak, Vibhor Kumar \
    Date: August 2026 | Master Technical Report & Empirical Architecture Package
  ]
  #v(0.5cm)
]

#rect(width: 100%, fill: rgb("#f3f4f6"), radius: 4pt, inset: 12pt)[
  #text(weight: "bold", size: 11pt)[Executive Summary] \
  #v(4pt)
  This report focuses *exclusively on next-day 24-hour hourly stock status prediction models* trained on the *FreshRetailNet-50K* dataset. The objective is to forecast hourly stock status (active operational hours 6 AM to 10 PM, represented as 24 binary hourly values) for individual SKU-store time series (`series_id = city_id + store_id + product_id`) using recent sequence history.

  We present empirical training and validation benchmarks comparing *Plain Standard LSTMs vs Advanced Model Families*:
  1. *Plain Standard LSTM*: Standard 96-hidden-unit LSTM architecture achieving *0.7852 F1-Score* (44.9k parameters).
  2. *Detailed Mechanics*: Recurrent unrolling over $L=14$ historical days, gating mechanisms (forget $f_t$, input $i_t$, candidate $tilde(C)_t$, output $o_t$), cell state $C_t$ updates, and 24-dim linear projection head.
]

#v(0.5cm)

== Problem Formulation & Dataset Analysis

=== Unit of Analysis & Target Definition
The task moves beyond coarse daily aggregation to fine-grained hourly stock status forecasting:
- *Input*: Recent $N$-day historical sequence ($N in {7, 10, 14}$ days) of SKU-store daily metrics.
- *Target*: 24 binary hourly stock-status values for the subsequent day ($t+1$).
- *Unit of Analysis*: SKU-store time series created by combining `series_id = city_id + store_id + product_id`.

=== Top 15 SKU Statistical Selection
To evaluate models on operationally critical demand patterns, SKUs were statistically selected using:
$ "Score" = 0.40 dot "Sales" + 0.25 dot "StockoutHours" + 0.20 dot "StockoutDays" + 0.15 dot "SalesStd" $

#figure(
  image("images/pipeline_overview.png", width: 95%),
  caption: [End-to-End Next-Day 24-Hour Stock Status Prediction Framework]
)

== Plain Standard LSTM Architecture & Internal Mechanics

The Plain Standard LSTM maps a historical $L=14$ day sequence vector into 24 binary hourly stockout predictions:

#figure(
  image("images/plain_lstm_architecture_diagram.png", width: 95%),
  caption: [Detailed Plain Standard LSTM Architecture Diagram showing Recurrent Unrolling, Internal Cell Gating Equations, and 24-Hour Linear Projection Head]
)

=== Mathematical Formulations & Gating Mechanics
For each daily sequence step $t in {1, 2, dots, L}$:
1. *Forget Gate*: $f_t = sigma(W_f dot [h_(t-1), x_t] + b_f)$
2. *Input Gate*: $i_t = sigma(W_i dot [h_(t-1), x_t] + b_i)$
3. *Candidate Cell State*: $tilde(C)_t = tanh(W_c dot [h_(t-1), x_t] + b_c)$
4. *Cell State Update*: $C_t = f_t dot C_(t-1) + i_t dot tilde(C)_t$
5. *Output Gate*: $o_t = sigma(W_o dot [h_(t-1), x_t] + b_o)$
6. *Hidden State Update*: $h_t = o_t dot tanh(C_t)$
7. *24-Hour Linear Projection*: $hat(Y) = sigma(W_"out" dot h_L + b_"out") in [0, 1]^(24)$

== Architectural Benchmark: Normal LSTMs vs DLinear Variations

To evaluate linear decomposition against recurrent sequence modeling, we benchmarked standard Normal LSTMs directly against Standard DLinear and Hourly Slot-Specific DLinear:

#figure(
  image("images/lstm_vs_dlinear_comparison.png", width: 95%),
  caption: [Empirical F1-Score Comparison: LSTM Variations vs Standard DLinear (F1 = 0.7932) and Hourly Slot-Specific DLinear (F1 = 0.8058)]
)

== Empirical Validation Benchmark Table Across All Models

All models were evaluated on identical 24-hour stock status validation splits.

#table(
  columns: (2.4fr, 1.2fr, 0.6fr, 0.6fr, 0.9fr, 0.9fr, 0.9fr),
  align: (left, center, center, center, center, center, center),
  stroke: 0.5pt + luma(150),
  fill: (col, row) => if row == 0 { rgb("#e5e7eb") } else if row == 1 { rgb("#d1fae5") } else if row >= 2 and row <= 3 { rgb("#fef3c7") } else { none },
  [*Architecture Name*], [*Architecture Category*], [*Seq*], [*Feats*], [*Params*], [*F1-Score*], [*Recall*],
  [*Multi-Kernel DLinear*], [Linear (Multi-Scale)], [14], [10], [*13,539*], [*0.8131*], [*0.8522*],
  [*Hourly Slot-Specific DLinear*], [Linear (24 Heads)], [14], [10], [*6,744*], [*0.8058*], [*0.8802*],
  [*Standard DLinear*], [Linear (Decomposition)], [14], [10], [*6,768*], [*0.7932*], [*0.8744*],
  [*Baseline Standard LSTM*], [Plain Standard LSTM], [14], [13], [44,952], [*0.7852*], [0.8040]
)

#figure(
  image("images/hourly_f1_vs_params_by_family.png", width: 95%),
  caption: [Overall Model Architecture Family Benchmark Scatter Plot]
)

== Strategic Recommendations & Conclusion

1. *Baseline Recurrent Model*: Deploy *Baseline Standard LSTM (h=96)* (0.7852 F1-Score, 44.9k params).
2. *Best Linear Projection*: Deploy *Hourly Slot-Specific DLinear* (0.8058 F1-Score, 88.02% Recall, 6.7k params).
3. *Compiled Report*: Refer to [docs/stockout_architectures_and_results.pdf](stockout_architectures_and_results.pdf) for the updated publication-ready PDF documentation.
