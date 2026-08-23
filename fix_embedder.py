import json
import re
path = "notebooks/04_rnn_attention_models.ipynb"
with open(path, "r") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        src_str = "".join(cell["source"])
        if "class BiRNNSeq2Seq" in src_str:
            new_birstm = """class BiRNNSeq2Seq(nn.Module):
    def __init__(self, rnn_type="lstm"):
        super().__init__()
        self.cat_embed = CatEmbedder([cat_encoder.sizes[c] for c in CAT_COLS], CFG.embed_dim)
        
        hist_input_dim = len(HIST_NUM_COLS) + self.cat_embed.out_dim
        future_input_dim = len(FUTURE_NUM_COLS) + self.cat_embed.out_dim
        
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.encoder = rnn_cls(hist_input_dim, CFG.hidden_dim, num_layers=CFG.num_layers, 
                               batch_first=True, dropout=CFG.dropout, bidirectional=True)
        self.decoder = rnn_cls(future_input_dim, CFG.hidden_dim, num_layers=CFG.num_layers, 
                               batch_first=True, dropout=CFG.dropout, bidirectional=True)
        
        self.fc = nn.Linear(CFG.hidden_dim * 2, 1)  # * 2 because bidirectional
        
    def forward(self, batch):
        hist_num = batch["hist_num"]
        hist_cat = self.cat_embed(batch["hist_cat"])
        hist_feat = torch.cat([hist_num, hist_cat], dim=-1)
        
        future_num = batch["future_num"]
        future_cat = self.cat_embed(batch["future_cat"])
        future_feat = torch.cat([future_num, future_cat], dim=-1)
        
        _, hidden = self.encoder(hist_feat)
        
        dec_out, _ = self.decoder(future_feat, hidden)
        
        out = self.fc(dec_out).squeeze(-1)
        return out
"""
            new_cnn = """class CNN_LSTM_Seq2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.cat_embed = CatEmbedder([cat_encoder.sizes[c] for c in CAT_COLS], CFG.embed_dim)
        
        hist_input_dim = len(HIST_NUM_COLS) + self.cat_embed.out_dim
        future_input_dim = len(FUTURE_NUM_COLS) + self.cat_embed.out_dim
        
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
        hist_cat = self.cat_embed(batch["hist_cat"])
        hist_feat = torch.cat([hist_num, hist_cat], dim=-1)
        
        # CNN expects (batch, channels, seq_len)
        hist_feat = hist_feat.transpose(1, 2)
        conv_feat = self.relu(self.conv1d(hist_feat))
        conv_feat = conv_feat.transpose(1, 2)  # back to (batch, seq_len, channels)
        
        _, hidden = self.encoder(conv_feat)
        
        future_num = batch["future_num"]
        future_cat = self.cat_embed(batch["future_cat"])
        future_feat = torch.cat([future_num, future_cat], dim=-1)
        
        dec_out, _ = self.decoder(future_feat, hidden)
        
        out = self.fc(dec_out).squeeze(-1)
        return out
"""
            src_str = re.sub(r'class BiRNNSeq2Seq.*?return out\n', new_birstm, src_str, flags=re.DOTALL)
            src_str = re.sub(r'class CNN_LSTM_Seq2Seq.*?return out\n', new_cnn, src_str, flags=re.DOTALL)
            cell["source"] = [line + "\n" if i < len(src_str.split("\n"))-1 else line for i, line in enumerate(src_str.split("\n"))]

with open(path, "w") as f:
    json.dump(nb, f, indent=1)
