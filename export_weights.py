# Run in your Kaggle/Colab notebook AFTER training (or after load_model('FFD.keras')).
# Creates: manifest.json + weights_0.bin, weights_1.bin ...  (float16, <25MB each)
import json, numpy as np

CHUNK = 20 * 1024 * 1024  # bytes per file (GitHub web upload limit is 25MB)

ws = model.get_weights()   # order: conv kernel, conv bias, ... dense kernel, dense bias
shapes = [list(w.shape) for w in ws]
blob = b"".join(w.astype("<f2").tobytes() for w in ws)

files = []
for i in range(0, len(blob), CHUNK):
    name = f"weights_{i // CHUNK}.bin"
    open(name, "wb").write(blob[i:i + CHUNK])
    files.append(name)

json.dump({"shapes": shapes, "files": files, "total_bytes": len(blob)}, open("manifest.json", "w"))
print("Created manifest.json and", files)
try:
    from google.colab import files as f; [f.download(x) for x in ["manifest.json"] + files]
except Exception:
    pass  # on Kaggle: download from the Output tab
