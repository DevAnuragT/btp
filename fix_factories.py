import json
import re

path = "notebooks/04_rnn_attention_models.ipynb"
with open(path, "r") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        src_str = "".join(cell["source"])
        if "model_factories = {" in src_str:
            fixed_factories = """cat_sizes = [CAT_ENCODER.sizes[c] for c in CAT_COLS]
model_factories = {
    "lstm_seq2seq": lambda: RNNSeq2Seq(len(HIST_NUM_COLS), len(FUTURE_NUM_COLS), cat_sizes, CFG, "lstm"),
    "gru_seq2seq": lambda: RNNSeq2Seq(len(HIST_NUM_COLS), len(FUTURE_NUM_COLS), cat_sizes, CFG, "gru"),
    "bilstm_seq2seq": lambda: BiRNNSeq2Seq("lstm"),
    "cnn_lstm_seq2seq": lambda: CNN_LSTM_Seq2Seq(),
    "gru_attention": lambda: RNNAttentionSeq2Seq(len(HIST_NUM_COLS), len(FUTURE_NUM_COLS), cat_sizes, CFG, "gru"),
    "lstm_attention": lambda: RNNAttentionSeq2Seq(len(HIST_NUM_COLS), len(FUTURE_NUM_COLS), cat_sizes, CFG, "lstm")
}"""
            new_src_str = re.sub(r'model_factories = \{.*?\}', fixed_factories, src_str, flags=re.DOTALL)
            cell["source"] = [line + "\n" if i < len(new_src_str.split("\n"))-1 else line for i, line in enumerate(new_src_str.split("\n"))]

with open(path, "w") as f:
    json.dump(nb, f, indent=1)
