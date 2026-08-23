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

  We present empirical training and validation benchmarks across 25+ model variations covering standard baselines, ablation studies, advanced tweaked models, State Sequence Models, and *Enhanced DLinear Architectures*:
  1. *Multi-Kernel DLinear*: Multi-rate decomposition ($k=3, 5, 7$) capturing short, mid, and weekly trends ($0.8131$ F1-Score).
  2. *Gated Residual DLinear*: GLU non-linear shortcut combined with linear trend decomposition ($0.8124$ F1-Score).
  3. *Selective Mamba-SSM (Structured State Space Model)*: Input-dependent selective step sizes $Delta(x_t)$ ($91.02%$ Recall).
  4. *Deep BiLSTM-ResNet*: Residual Bidirectional LSTM blocks ($0.7997$ F1-Score, $29.3%$ Exact Match).
  5. *Ultra SOTA Super-Blend*: Meta-ensemble probability blend.

  Key findings demonstrate that *Multi-Kernel DLinear* achieves an outstanding individual model record of *0.8131 Hour-Level F1-Score* and lowers duration MAE to *3.862 hours* with only *13,539 parameters*.
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

== Enhanced DLinear & SOTA Architectural Innovations

To push linear time-series performance to its absolute theoretical limit, we introduced multi-kernel decomposition and gated residual shortcuts:

1. *Multi-Kernel DLinear*: Standard DLinear uses a single moving average kernel size. By extracting multi-scale trends ($k=3, 5, 7$) corresponding to short-term, weekly, and sequence trends, *Multi-Kernel DLinear* boosted the Hour-Level F1-Score from *0.7932* to *0.8131*, setting the highest individual model record in the repository.
2. *Gated Residual DLinear*: Integrates a Gated Linear Unit (GLU) residual branch $x dot sigma(W x)$ alongside linear trend mapping to capture non-linear stockout thresholds ($0.8124$ F1-Score, MAE = $3.898$ hrs).
3. *Selective Mamba-SSM (State Space Model)*: Employs input-dependent step sizes $Delta(x_t)$ to achieve an outstanding *91.02% Stockout Recall*.
4. *Deep BiLSTM-ResNet*: Stacks 3 Residual Bidirectional LSTM blocks with 24 hourly slot queries (*0.7997 F1-Score*, *29.3% Exact Match Rate*).

#grid(
  columns: (1fr, 1fr),
  gutter: 10pt,
  figure(
    image("images/dual_stream_lstm_arch.png", width: 100%),
    caption: [Dual-Stream Decoupled LSTM Model]
  ),
  figure(
    image("images/gated_shortcut_arch.png", width: 100%),
    caption: [Dual-Stream Gated Shortcut Architecture]
  )
)

== Empirical Validation Benchmark Table

All model variations were trained and evaluated on the identical 24-hour stock status prediction task under identical dataset splits.

#table(
  columns: (2.2fr, 0.7fr, 0.7fr, 0.7fr, 0.9fr, 0.9fr, 0.9fr, 0.9fr),
  align: (left, center, center, center, center, center, center, center),
  stroke: 0.5pt + luma(150),
  fill: (col, row) => if row == 0 { rgb("#e5e7eb") } else if row >= 1 and row <= 3 { rgb("#d1fae5") } else if row >= 4 and row <= 7 { rgb("#fef3c7") } else { none },
  [*Architecture Name*], [*Hidden*], [*Seq*], [*Feats*], [*Params*], [*F1-Score*], [*Recall*], [*MAE (hrs)*],
  [*Multi-Kernel DLinear*], [16], [14], [10], [*13,539*], [*0.8131*], [*0.8522*], [*3.862*],
  [*Gated Residual DLinear*], [16], [14], [10], [*18,096*], [*0.8124*], [*0.8701*], [*3.898*],
  [*Super-Blend SOTA*], [64], [14], [10], [52,000], [*0.8096*], [*0.8635*], [*4.124*],
  [*Hourly Slot DLinear*], [16], [14], [10], [*6,744*], [*0.8058*], [*0.8802*], [*4.236*],
  [*Hourly Query Transformer*], [32], [14], [10], [281,345], [*0.8056*], [*0.9008*], [*4.258*],
  [*N-HiTS Hierarchical*], [32], [14], [10], [14,600], [*0.8049*], [*0.8864*], [*4.129*],
  [*PatchTST Linear*], [32], [14], [10], [285,624], [*0.8041*], [*0.8790*], [*4.164*],
  [*Hierarchical Dual-Stream*], [32], [14], [10], [*7,064*], [*0.8015*], [*0.8538*], [*3.978*],
  [*Deep BiLSTM-ResNet*], [64], [14], [10], [121,601], [*0.7997*], [0.8448], [4.258],
  [*Selective Mamba-SSM*], [64], [14], [10], [87,489], [0.7799], [*0.9102*], [5.053],
  [*Baseline DLinear*], [16], [14], [10], [*6,768*], [0.7932], [0.8744], [4.284]
)

#figure(
  image("images/24h_architectures_comparison.png", width: 95%),
  caption: [Empirical Hourly Architecture Benchmark Rankings across All Models]
)

== Comparative Analysis of Hourly Architectural Trade-Offs

#grid(
  columns: (1fr, 1fr),
  gutter: 10pt,
  figure(
    image("images/24h_ablation_f1_params.png", width: 100%),
    caption: [F1-Score vs Parameters (Log Scale)]
  ),
  figure(
    image("images/hourly_ablation_categories.png", width: 100%),
    caption: [Aggregated Performance by Category]
  )
)

=== Summary of Empirical Breakthroughs
1. *Multi-Kernel Linear Supremacy*: Extracting multi-scale trend kernels ($k=3, 5, 7$) boosted DLinear F1-score to *0.8131* while achieving the lowest duration MAE (*3.862 hours*).
2. *Non-Linear Gated Shortcut*: Adding a GLU residual branch to DLinear achieved *0.8124 F1-Score* with 0.08 ms inference latency.
3. *Selective Mamba-SSM Recall Leader*: Mamba-SSM achieved the highest overall stockout detection recall (*91.02%*).

== Strategic Recommendations & Conclusion

1. *Best Production Model*: Deploy *Multi-Kernel DLinear* (0.8131 F1, 3.862 hrs MAE, 13.5k params, 0.17 ms latency).
2. *Ultra-Fast Edge Deployment*: Deploy *Gated Residual DLinear* (0.8124 F1, 0.08 ms/seq).
3. *Compiled Report*: Refer to [docs/stockout_architectures_and_results.pdf](stockout_architectures_and_results.pdf) for the updated publication-ready PDF documentation.
