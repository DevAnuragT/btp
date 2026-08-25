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
  This report focuses *exclusively on next-day 24-hour hourly stock status prediction models* trained on the *FreshRetailNet-50K* dataset.

  We present empirical training and validation benchmarks comparing *Linear vs. Non-Linear DLinear Variations*:
  1. *Linear DLinear Baselines*: Standard DLinear achieves *0.7932 F1-Score* (6.7k params) and Hourly Slot-Specific DLinear achieves *0.8058 F1-Score*.
  2. *Non-Linear DLinear Innovations*: Introducing GELU activation bottlenecks boosts F1-Score to *0.8083* and Stockout Recall to *91.37%*.
  3. *Multi-Kernel Non-Linear DLinear*: Multi-scale non-linear trend decomposition ($k=3, 5, 7$) reaches *0.8156 F1-Score* and the lowest daily stockout duration error of *3.75 hours MAE*.
]

#v(0.5cm)

== Problem Formulation & Dataset Analysis

=== Unit of Analysis & Target Definition
The task moves beyond coarse daily aggregation to fine-grained hourly stock status forecasting:
- *Input*: Recent $N$-day historical sequence ($N in {7, 10, 14}$ days) of SKU-store daily metrics.
- *Target*: 24 binary hourly stock-status values for the subsequent day ($t+1$).
- *Unit of Analysis*: SKU-store time series created by combining `series_id = city_id + store_id + product_id`.

== Architectural Impact: Linear vs. Non-Linear DLinear Variations

To evaluate the effect of non-linearity on series decomposition models, we benchmarked GELU, SwiGLU, and Multi-Kernel non-linear DLinear variants against standard linear models:

#figure(
  image("images/nonlinear_dlinear_comparison.png", width: 95%),
  caption: [Empirical F1-Score & Stockout Recall Comparison: Linear DLinear vs Non-Linear DLinear Variations]
)

=== Key Empirical Discoveries
1. *Non-Linear GELU Bottleneck*: Adding a GELU activation layer between sequence decomposition and logit projection boosts F1-Score from *0.7932* to *0.8083* (+0.0151 gain) and sets a new record for *Stockout Recall at 91.37%*.
2. *Multi-Kernel Non-Linear Peak*: `Multi-Kernel Non-Linear DLinear` reaches *0.8156 F1-Score* (+0.0259 gain over Best Normal LSTM baseline) and achieves the lowest daily error of *3.75 hours MAE*.

== Master Empirical Validation Benchmark Table Across All Models

All models were evaluated on identical 24-hour stock status validation splits.

#table(
  columns: (2.4fr, 1.2fr, 0.6fr, 0.6fr, 0.9fr, 0.9fr, 0.9fr),
  align: (left, center, center, center, center, center, center),
  stroke: 0.5pt + luma(150),
  fill: (col, row) => if row == 0 { rgb("#e5e7eb") } else if row == 1 { rgb("#d1fae5") } else if row >= 2 and row <= 3 { rgb("#fef3c7") } else { none },
  [*Architecture Name*], [*Architecture Category*], [*Seq*], [*Feats*], [*Params*], [*F1-Score*], [*Recall*],
  [*Multi-Kernel Non-Linear DLinear*], [Non-Linear DLinear], [14], [10], [*42,339*], [*0.8156*], [87.13%],
  [*Multi-Kernel Linear DLinear*], [Linear (Multi-Scale)], [14], [10], [*13,539*], [*0.8131*], [85.22%],
  [*Non-Linear GELU DLinear*], [Non-Linear DLinear], [14], [10], [*21,168*], [*0.8083*], [*91.37%*],
  [*Hourly Slot-Specific DLinear*], [Linear (24 Heads)], [14], [10], [*6,744*], [*0.8058*], [88.02%],
  [*Standard Linear DLinear*], [Linear (Decomposition)], [14], [10], [*6,768*], [0.7932], [87.44%],
  [*Best Normal LSTM Baseline*], [Plain Standard LSTM], [14], [13], [5,912], [0.7897], [81.29%]
)

== Strategic Recommendations & Conclusion

1. *Highest F1 & Lowest MAE*: Deploy *Multi-Kernel Non-Linear DLinear* (0.8156 F1-Score, 3.75 hrs MAE, 42.3k params).
2. *Highest Stockout Recall*: Deploy *Non-Linear GELU DLinear* (0.8083 F1-Score, 91.37% Recall, 21.1k params).
3. *Compiled Report*: Refer to [docs/stockout_architectures_and_results.pdf](stockout_architectures_and_results.pdf) for the updated publication-ready PDF documentation.
