import json

path = "notebooks/04_rnn_attention_models.ipynb"
with open(path, "r") as f:
    nb = json.load(f)

# 1. Update the Model Cell (Cell 6) to include new architectures
model_cell_index = 6
model_src = "".join(nb["cells"][model_cell_index]["source"])

new_models_str = """
class BiRNNSeq2Seq(nn.Module):
    def __init__(self, rnn_type="lstm"):
        super().__init__()
        self.hist_cat_emb = CatEmbedder([CAT_ENCODER.sizes[c] for c in CAT_COLS], CFG.embed_dim)
        self.future_cat_emb = CatEmbedder([CAT_ENCODER.sizes[c] for c in TEMPORAL_CAT_COLS], CFG.embed_dim)
        
        hist_input_dim = len(HIST_NUM_COLS) + self.hist_cat_emb.out_dim
        future_input_dim = len(FUTURE_NUM_COLS) + self.future_cat_emb.out_dim
        
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.encoder = rnn_cls(hist_input_dim, CFG.hidden_dim, num_layers=CFG.num_layers, 
                               batch_first=True, dropout=CFG.dropout, bidirectional=True)
        self.decoder = rnn_cls(future_input_dim, CFG.hidden_dim, num_layers=CFG.num_layers, 
                               batch_first=True, dropout=CFG.dropout, bidirectional=True)
        
        self.fc = nn.Linear(CFG.hidden_dim * 2, 1)  # * 2 because bidirectional
        
    def forward(self, batch):
        hist_num = batch["hist_num"]
        hist_cat = self.hist_cat_emb(batch["hist_cat"])
        hist_feat = torch.cat([hist_num, hist_cat], dim=-1)
        
        future_num = batch["future_num"]
        future_cat = self.future_cat_emb(batch["future_cat"])
        future_feat = torch.cat([future_num, future_cat], dim=-1)
        
        _, hidden = self.encoder(hist_feat)
        
        dec_out, _ = self.decoder(future_feat, hidden)
        
        out = self.fc(dec_out).squeeze(-1)
        return out

class CNN_LSTM_Seq2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.hist_cat_emb = CatEmbedder([CAT_ENCODER.sizes[c] for c in CAT_COLS], CFG.embed_dim)
        self.future_cat_emb = CatEmbedder([CAT_ENCODER.sizes[c] for c in TEMPORAL_CAT_COLS], CFG.embed_dim)
        
        hist_input_dim = len(HIST_NUM_COLS) + self.hist_cat_emb.out_dim
        future_input_dim = len(FUTURE_NUM_COLS) + self.future_cat_emb.out_dim
        
        # 1D CNN to extract local temporal features (kernel size 3)
        self.conv1d = nn.Conv1d(in_channels=hist_input_dim, out_channels=CFG.hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
        self.encoder = nn.LSTM(CFG.hidden_dim, CFG.hidden_dim, num_layers=CFG.num_layers, 
                               batch_first=True, dropout=CFG.dropout)
        self.decoder = nn.LSTM(future_input_dim, CFG.hidden_dim, num_layers=CFG.num_layers, 
                               batch_first=True, dropout=CFG.dropout)
        
        self.fc = nn.Linear(CFG.hidden_dim, 1)
        
    def forward(self, batch):
        hist_num = batch["hist_num"]
        hist_cat = self.hist_cat_emb(batch["hist_cat"])
        hist_feat = torch.cat([hist_num, hist_cat], dim=-1)
        
        # CNN expects (batch, channels, seq_len)
        hist_feat = hist_feat.transpose(1, 2)
        conv_feat = self.relu(self.conv1d(hist_feat))
        conv_feat = conv_feat.transpose(1, 2)  # back to (batch, seq_len, channels)
        
        _, hidden = self.encoder(conv_feat)
        
        future_num = batch["future_num"]
        future_cat = self.future_cat_emb(batch["future_cat"])
        future_feat = torch.cat([future_num, future_cat], dim=-1)
        
        dec_out, _ = self.decoder(future_feat, hidden)
        
        out = self.fc(dec_out).squeeze(-1)
        return out

model_factories = {
    "lstm_seq2seq": lambda: RNNSeq2Seq("lstm"),
    "gru_seq2seq": lambda: RNNSeq2Seq("gru"),
    "bilstm_seq2seq": lambda: BiRNNSeq2Seq("lstm"),
    "cnn_lstm_seq2seq": lambda: CNN_LSTM_Seq2Seq(),
    "gru_attention": lambda: RNNAttentionSeq2Seq("gru"),
    "lstm_attention": lambda: RNNAttentionSeq2Seq("lstm")
}
"""

import re
# Replace the old model_factories with the new models and updated factories
model_src = re.sub(r'model_factories = \{.*?\}', new_models_str, model_src, flags=re.DOTALL)
nb["cells"][model_cell_index]["source"] = [line + "\n" if i < len(model_src.split("\n"))-1 else line for i, line in enumerate(model_src.split("\n"))]

# 2. Fix the cudnn error in the captum cell
for cell in nb.get("cells", []):
    if "def plot_integrated_gradients" in "".join(cell.get("source", [])):
        source = cell["source"]
        for i, line in enumerate(source):
            if "attributions, delta = ig.attribute(inputs=hist_num" in line:
                source.insert(i, "    with torch.backends.cudnn.flags(enabled=False):\n")
                # Indent the next 3 lines which are part of the ig.attribute call
                for j in range(3):
                    source[i+1+j] = "    " + source[i+1+j]
                break

with open(path, "w") as f:
    json.dump(nb, f, indent=1)
