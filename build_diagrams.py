import os
import subprocess

diagrams = {
    "docs/images/pipeline_overview.mmd": """
graph TD
    subgraph "Data Processing & Pipeline"
        A["FreshRetailNet-50K / DataCo Raw Data"] --> B["Data Cleaning & Encoding"]
        B --> C["Top 15 SKU Selection Score (Sales + Stockout Hours + Variance)"]
        C --> D["Feature Engineering (stockout_streak, stockout_rolling_7)"]
        D --> E["Sliding Window Generator (Chronological Train/Test Split)"]
    end

    subgraph "Dual Modeling Tracks"
        E --> F1["Track 1: 7-Day Long-Term Horizon (28-Day History -> 7-Day Binary)"]
        E --> F2["Track 2: 24-Hour Short-Term Horizon (14-Day History -> 24-Hour Status)"]
    end

    subgraph "Architectures & Evaluation"
        F1 --> G1["LSTM Seq2Seq, GRU Seq2Seq, BiLSTM, Transformers, TFT, PatchTST"]
        F2 --> G2["Baseline LSTM, Compact LSTM, Dual-Stream, Gated Shortcuts"]
        G1 --> H1["Evaluation: PR-AUC (Primary), ROC-AUC, F1, Horizon Decay"]
        G2 --> H2["Evaluation: F1-Score (Primary), Exact 24h Match, MAE, Latency"]
    end

    subgraph "Explainability Engine"
        H1 --> I["Captum Integrated Gradients & Temporal Attention Heatmaps"]
        H2 --> I
    end
""",

    "docs/images/lstm_seq2seq_arch.mmd": """
graph LR
    subgraph "Input Sequence (28 Days)"
        X1["Day t-27"] 
        X2["Day t-1"]
        FE["Engineered Features: stockout_streak, stockout_rolling_7, sale_amount_lag1"]
    end

    subgraph "Encoder RNN / LSTM"
        FE --> X1
        FE --> X2
        X1 --> Enc1["LSTM Cell 1"]
        Enc1 --> Enc2["LSTM Cell ..."]
        Enc2 --> EncN["LSTM Cell 28"]
    end

    subgraph "Context Vector"
        EncN --> C["Latent State Vector (h_t, c_t)"]
    end

    subgraph "Decoder & Projection"
        C --> Dec1["Decoder Cell Day t+1"]
        Dec1 --> Dec2["Decoder Cell ..."]
        Dec2 --> Dec7["Decoder Cell Day t+7"]
        Dec1 --> Out1["Pr(Stockout t+1)"]
        Dec2 --> Out2["Pr(Stockout t+...)"]
        Dec7 --> Out7["Pr(Stockout t+7)"]
    end
""",

    "docs/images/dual_stream_lstm_arch.mmd": """
graph TD
    subgraph "Multi-Modal Inputs (14 Days)"
        InD["Demand Features (sale_amount, hours_sale_sum, discount, holiday_flag)"]
        InI["Inventory Features (stock_hour6_22_cnt, hours_stock_status_sum)"]
    end

    subgraph "Decoupled Recurrent Streams"
        InD --> L_Demand["Demand LSTM Subnetwork (h_d = 16 units)"]
        InI --> L_Inven["Inventory LSTM Subnetwork (h_i = 16 units)"]
    end

    subgraph "Fusion & Gating Layer"
        L_Demand --> Conc["Feature Concatenation [h_d || h_i] (32 dim)"]
        L_Inven --> Conc
        Conc --> Dense["Fusion FC + BatchNorm + ReLU"]
    end

    subgraph "Hourly Output Head"
        Dense --> OutHead["Linear Projection Layer (32 -> 24 units)"]
        OutHead --> Sigm["Sigmoid Activation"]
        Sigm --> Y24["24-Hour Stock Status Predictions [H1, H2, ..., H24]"]
    end
""",

    "docs/images/gated_shortcut_arch.mmd": """
graph TD
    subgraph "Stream Inputs"
        D_in["Demand Sequence Input"]
        I_in["Inventory Sequence Input"]
    end

    subgraph "Processing Branches"
        D_in --> L_Demand["Demand Stream LSTM"]
        I_in --> L_Inven["Inventory Stream LSTM"]
        I_in --> DirectCut["Direct Inventory Residual Shortcut"]
    end

    subgraph "Dynamic Gating Mechanism"
        L_Demand --> Concat["Concatenate [h_demand, h_inven]"]
        L_Inven --> Concat
        Concat --> GateLinear["Gating Linear Layer W_g"]
        GateLinear --> GateSig["Sigmoid Gate α = σ(W_g x + b_g)"]
        
        Concat --> MainFeat["Main Feature Transformation"]
    end

    subgraph "Residual Gated Fusion"
        MainFeat --> Mult1["α * Main Features"]
        GateSig --> Mult1
        DirectCut --> Add1["(1 - α) * Direct Shortcut"]
        GateSig --> Add1
        Mult1 --> Combine["Gated Residual Addition"]
        Add1 --> Combine
    end

    subgraph "Output Projection"
        Combine --> Head["24-Hour Output Projection Head"]
        Head --> Y["24-Hour Stock Status Output"]
    end
""",

    "docs/images/explainability_flow.mmd": """
graph TD
    subgraph "Model Predictions"
        M["Trained Seq2Seq / RNN Model"] --> P["Stockout Probabilities P(Y)"]
    end

    subgraph "Temporal Attention Track"
        M --> Attn["Attention Matrix A[t_hist, t_fut]"]
        Attn --> AttnHeat["Temporal Attention Heatmap (Identifies WHEN model looks)"]
    end

    subgraph "Feature Attribution Track (Captum)"
        M --> IG["Integrated Gradients Engine"]
        IG --> Base["Baseline X_0 (Zero Vector)"]
        IG --> Grad["Riemann Sum Gradient Integration"]
        Grad --> IGHeat["Feature Importance Heatmap (Identifies WHICH features drive prediction)"]
        Grad --> IGBar["Aggregated Feature Importance Barplot"]
    end
"""
}

for filepath, content in diagrams.items():
    with open(filepath, "w") as f:
        f.write(content.strip())
    
    svg_filepath = filepath.replace(".mmd", ".svg")
    png_filepath = filepath.replace(".mmd", ".png")
    
    print(f"Compiling {filepath}...")
    
    # Compile SVG
    res_svg = subprocess.run(
        ['npx', '@mermaid-js/mermaid-cli', '-i', filepath, '-o', svg_filepath],
        capture_output=True, text=True
    )
    if res_svg.returncode != 0:
        print(f"Error compiling SVG for {filepath}: {res_svg.stderr}")
    else:
        print(f"Generated {svg_filepath}")

    # Compile PNG
    res_png = subprocess.run(
        ['npx', '@mermaid-js/mermaid-cli', '-i', filepath, '-o', png_filepath],
        capture_output=True, text=True
    )
    if res_png.returncode != 0:
        print(f"Error compiling PNG for {filepath}: {res_png.stderr}")
    else:
        print(f"Generated {png_filepath}")

print("Completed all diagram compilations!")
