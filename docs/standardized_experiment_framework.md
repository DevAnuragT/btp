# Standardized Experimental Framework & Technical Specification

## 1. Executive Objective & Context
This document defines the single authoritative, standardized experimental framework for all **Next-Day 24-Hour Hourly Stock Status Forecasting** experiments conducted in this repository.

### Business Context & Research Motivation
In high-frequency fresh retail supply chains (e.g., Dingdong Fresh, instant grocery delivery), stockouts cause severe operational damage: lost revenue, unfulfilled customer carts, and reduced long-term customer retention. 

Traditional inventory models forecast coarse **daily aggregated demand**, which fails to capture intra-day inventory depletion patterns. A store may appear "in-stock" on a daily aggregate basis even if it suffered a critical 4-hour stockout during peak evening sales (5 PM – 9 PM).

To solve this, our framework operates at **hourly granularity**, predicting the 24-hour operational stock status vector for the target day $t+1$.

---

## 2. Task Definition & Data Standards

### Task Definition
Given an $N$-day historical sequence ($N \in \{7, 10, 14\}$), the model forecasts 24 binary hourly stock-status indicators for the target day $t+1$:
$$X_{t-N+1:t} \in \mathbb{R}^{N \times D} \implies \hat{Y}_{t+1} \in \{0, 1\}^{24}$$
where $y_h = 1$ denotes a stockout state during hour $h \in [1..24]$, and $y_h = 0$ denotes in-stock status.

### Primary Dataset & Unit of Analysis
- **Dataset**: `FreshRetailNet-50K` (Dingdong Inc.).
- **Unit of Analysis**:
  $$\text{series\_id} = \text{city\_id} + \text{store\_id} + \text{product\_id}$$
  Grouping by product alone or store alone is strictly prohibited as inventory levels differ across physical store locations.

### Statistical Top 15 SKU Selection Formula
SKUs are statistically selected using a weighted multi-factor score:
$$\text{Score} = 0.40 \cdot \text{Sales} + 0.25 \cdot \text{StockoutHours} + 0.20 \cdot \text{StockoutDays} + 0.15 \cdot \text{SalesStd}$$
All 4 components are min-max normalized across all unique product IDs.

### Standardized 10-Dimensional Input Feature Vector
1. `sale_amount`: Total daily sales quantity (standardized).
2. `stock_hour6_22_cnt`: Count of operational stockout hours during 6 AM – 10 PM (standardized).
3. `discount`: Average promotional discount rate $[0, 1]$.
4. `holiday_flag`: Binary indicator for national/statutory holidays $\{0, 1\}$.
5. `hours_sale_sum`: Sum of intra-day hourly sales (standardized).
6. `hours_stock_status_sum`: Sum of intra-day hourly stockout status (standardized).
7. `dow_sin`: Cyclical day-of-week sine transformation $\sin(2\pi \cdot \text{dow} / 7)$.
8. `dow_cos`: Cyclical day-of-week cosine transformation $\cos(2\pi \cdot \text{dow} / 7)$.
9. `stockout_rolling_3`: 3-day rolling window stockout day count.
10. `sales_momentum`: 1-day sales velocity change $\Delta \text{Sales}_t = \text{Sales}_t - \text{Sales}_{t-1}$.

### Chronological Train / Validation Split
- **Training Set**: Historical sequence data up to $T_{\text{max}} - 15 \text{ days}$ (~2.5 months).
- **Validation Set**: Final 15 chronological days ($T_{\text{max}} - 15 \text{ days} < t \le T_{\text{max}}$).
- **Scaler**: `StandardScaler` fitted strictly on the training set.

---

## 3. Metric Hierarchy & Evaluation Rationale

### Primary Objective Metric: Hour-Level F1-Score
In retail datasets, operational hours are dominated by in-stock status (60:40 to 80:20 in-stock dominance).
- Raw Accuracy is **misleading**: A trivial model predicting 100% in-stock achieves 80% accuracy while missing **100% of stockout events**.
- **Hour-Level F1-Score** is our **Primary Metric**:
  $$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

### Secondary Evaluation Metrics
1. **Hour-Level Recall**: Percentage of actual stockout hours correctly detected ($\text{TP} / (\text{TP} + \text{FN})$).
2. **Exact 24-Hour Match Rate (%)**: Percentage of days where the model correctly predicts all 24 hourly stockout statuses perfectly.
3. **Mean Absolute Hour Count Error (MAE in hours)**: Mean absolute difference between predicted total stockout hours and actual total stockout hours per day.
4. **Inference Latency (ms/seq)**: Average CPU/GPU prediction time per sequence (milliseconds).
5. **Parameter Count**: Total trainable parameters.

---

## 4. Optimization & Loss Standards

### Focal Loss ($\gamma = 2.0, \alpha = 0.5$)
All models are optimized using Focal Loss to down-weight easy in-stock negative examples and focus learning on hard stockout transitions:
$$\mathcal{L}_{\text{Focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

### Dynamic Threshold Scanning ($\tau^*$)
Scanning thresholds $\tau \in [0.10, 0.90]$ with step size $0.05$ on the validation split to select $\tau^*$ that maximizes the Hour-Level F1-Score:
$$\tau^* = \arg\max_{\tau \in [0.1, 0.9]} F_1(\mathbf{y}, \sigma(\hat{\mathbf{z}}) \ge \tau)$$

---

## 5. Standardized Architectural Taxonomy & Empirical Results

| Category | Representative Architecture | Parameters | Hour-Level F1 | Recall | Exact Match | MAE (hrs) | Latency (ms) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline Recurrent** | Baseline LSTM ($h=96$) | 44,952 | 0.7850 | 80.4% | 30.2% | 4.670 | 0.0065 |
| **Ablation & Compact** | Compact Feature-Reduced ($h=32$) | 5,912 | 0.7900 | 81.3% | 21.3% | 4.320 | 0.0041 |
| **Tweaked Architecture** | Hierarchical Dual-Stream ResNet | **7,064** | **0.8015** | 85.4% | 20.9% | **3.978** | **0.7129** |
| **SOTA Linear** | DLinear (Decomposition Linear) | **6,768** | **0.7932** | 87.4% | 24.4% | 4.284 | **0.3326** |
| **SOTA Hierarchical** | N-HiTS (Hierarchical Interpolation) | 14,600 | **0.8049** | 88.6% | 19.1% | 4.129 | 0.1973 |
| **SOTA Dilated Conv** | WaveNet-GLU (Dilated Convs) | 19,960 | **0.7960** | 85.3% | **31.1%** | 4.400 | 0.1442 |
| **SOTA Transformer** | TFT-Lite (Variable Selection) | 288,390 | **0.8023** | 85.4% | 25.3% | 4.018 | 0.2930 |
| **SOTA Transformer** | PatchTST (Patch Linear) | 285,624 | **0.8041** | 87.9% | 24.4% | 4.164 | 2.6389 |
| **SOTA Cross-Attn** | Hourly Query Transformer | 281,345 | **0.8056** | **90.1%** | 27.1% | 4.258 | 7.3837 |
| **Super-Ensemble** | **Ultra SOTA Super-Blend** | 52,000 | **0.8096** | **86.4%** | **28.9%** | **4.124** | 0.0420 |

---

## 6. Artifact Compliance & Standardized File Paths

- **PDF Specification**: [docs/standardized_experiment_framework.pdf](standardized_experiment_framework.pdf)
- **Markdown Specification**: [docs/standardized_experiment_framework.md](standardized_experiment_framework.md)
- **Master Benchmark CSV**: [master_hourly_models_benchmark.csv](../experiments/freshretail_lstm/final_architecture_package/master_hourly_models_benchmark.csv)
- **SOTA Results JSON**: [sota_architectures_summary.json](../experiments/freshretail_lstm/final_architecture_package/sota_architectures_summary.json)
