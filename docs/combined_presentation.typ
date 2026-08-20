#import "@preview/touying:0.5.3": *
#import themes.metropolis: *

#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  config-info(
    title: [Supply Chain Stockout Prediction],
    subtitle: [End-to-End Deep Learning Approaches],
    author: [Anurag 2023IMG008 \ Samyak 2023IMG044 \ Vibhor 2023IMG054],
    date: [May to July 2026],
  ),
)

#title-slide()

== 1. The Stockout Prediction Problem

- *Business Objective*: Accurately predict future out-of-stock events to optimize supply chain operations and prevent lost sales.
- *The Challenge*: Moving from simple static backorder prediction to dynamic, time-series stockout forecasting.
- *Core Task*: Predict whether a specific product (SKU) at a specific store will experience a stockout in the future (next 24 hours vs. next 7 days).

== 2. Dataset & Signal Extraction

- *Dataset Selection*: FreshRetailNet-50K
  - Provides true time-series daily records at the SKU-Store level over 97 days.
  - Contains hourly stock-status tracking (predicting stockouts during active hours: 6 AM to 10 PM).
- *Targeting High-Impact SKUs*:
  - Evaluated the Top 15 SKUs statistically based on high sales volume and high stockout variance.
  - Focuses the models on the most challenging, non-trivial demand patterns.

== 3. Establishing a Baseline

- *Model*: PyTorch Tabular Transformer
- *Approach*: Treats daily features (like sales, discounts, weather) as a sequence of categorical and numeric embeddings.
- *Baseline Results*: 
  - Accuracy: 55.83% | AUC: 0.5257 
- *Key Takeaway*:
  - Barely outperformed random guessing (0.50). 
  - Proved that treating stockout prediction as a simple classification problem is insufficient. Sequential time-series modeling and feature engineering are strictly required.

== 4. Short-Term Forecasting (Next 24 Hours)

- *Objective*: Predict next-day hourly stock status using recent history (7 to 14 days).
- *RNN Experimentation*:
  - Tested various RNN architectures; a SimpleRNN (32 units) achieved *~75.88% Accuracy*. 
  - Complex architectures degraded due to noise and underfitting.
- *LSTM Ablation*:
  - Standard LSTM (44k params) achieved 0.785 F1-score.
  - A *Compact Feature-Reduced LSTM* (6 features, 5.9k params) outperformed it with a *0.790 F1-score*, reducing parameters by 87%.

== 4.1. Exploring the Ultra-Compact LSTM Limit

- *Goal*: Test the lower bound of LSTM complexity to see if even smaller models are viable.
- *Ultra-Compact Variants*:
  - *h16 (6 features)*: 1,944 params | F1: 0.741 (Drops F1 significantly)
  - *h8 (6 features)*: 728 params | F1: 0.647 | Recall: 0.924 (Recall-heavy, poor balanced F1)
  - *h16 (4 features)*: 1,816 params | F1: 0.746
- *Findings*:
  - Shrinking beyond `h32` trades off too much F1-score for parameter reduction. 
  - The `h32` model (5,912 params) remains the optimal *practical minimum*.
  - *Recommendation*: Use `h32` as the primary LSTM, retaining `h16` only as an ultra-light fallback.

== 5. Long-Term Forecasting (7-Day Horizon)

- *Objective*: Predict 7-day future stockouts using a 28-day historical window.
- *The Breakthrough*: 
  - Engineered time-aware features: `stockout_streak` and `stockout_rolling_7`.
- *Model Rankings*:
  - *🥇 LSTM Seq2Seq*: Achieved Best pure ranking (PR-AUC 0.5613), crossing the naive persistence baseline.
  - *🥈 GRU Attention*: Achieved Highest F1-Score (0.5201).
- *Conclusion*: Sequence-to-Sequence models with engineered temporal features are highly effective for extended stockout forecasting horizons.

== 6. Explaining the Stockout Triggers (SHAP)

- Supply chain managers need to trust the model. We used *SHAP* to pinpoint exactly which features (e.g., temperature spikes or recent promotions) triggered a short-term stockout alert.
#align(center)[
  #image("../samyak/download.png", height: 60%)
]

== 7. Explaining Temporal Importance (Attention)

- *Temporal Attention*: Visualizes *when* the LSTM Seq2Seq model looks back into the 28-day history (X-axis) to predict a future stockout day (Y-axis).
#align(center)[
  #image("images/attention_heatmap.png", height: 65%)
]

