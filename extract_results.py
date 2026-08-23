import json
import base64
import os

path = "notebooks/04_rnn_attention_models.ipynb"
artifact_dir = "/Users/vibhorkumar/.gemini/antigravity-cli/brain/be8a7b9c-5aa5-4454-8414-6aeb43de9444"

with open(path, "r") as f:
    nb = json.load(f)

img_count = 0
for i, cell in enumerate(nb.get('cells', [])):
    if cell.get('cell_type') == 'code' and cell.get('outputs'):
        for output in cell['outputs']:
            # Extract text outputs (metrics)
            if output.get('output_type') == 'stream':
                text = ''.join(output.get('text', []))
                if "TEST PR-AUC:" in text:
                    print(text.strip().split('\n')[-1])
            
            # Extract image outputs
            if 'data' in output and 'image/png' in output['data']:
                img_data = output['data']['image/png']
                img_bytes = base64.b64decode(img_data)
                
                # Guess what the image is based on the cell code or order
                src = "".join(cell.get('source', [])).lower()
                img_name = f"plot_{img_count}.png"
                if "attention" in src and "heatmap" in src:
                    img_name = "attention_heatmap.png"
                elif "integratedgradients" in src or "integrated_gradients" in src:
                    if img_count == 1:
                        img_name = "ig_heatmap.png"
                    else:
                        img_name = "ig_barplot.png"
                elif "shap" in src:
                    img_name = "shap_summary.png"
                
                img_path = os.path.join(artifact_dir, img_name)
                with open(img_path, "wb") as img_file:
                    img_file.write(img_bytes)
                print(f"Saved image to {img_path}")
                img_count += 1
