"""Package checkpoints/best.pt for Hugging Face: model.safetensors + config.json + tokenizer.json + model.py + generate.py.

    py -3.12 export_hf.py            -> writes hf_export/ (fp32 weights, no optimizer state)
"""
import os, json, shutil, torch
from safetensors.torch import save_file

H = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(H, "hf_export")
ck = torch.load(os.path.join(H, "checkpoints", "best.pt"), map_location="cpu", weights_only=False)
sd = {k: v.contiguous() for k, v in ck["model"].items() if k != "head.weight"}   # head is tied to tok.weight
os.makedirs(OUT, exist_ok=True)
save_file(sd, os.path.join(OUT, "model.safetensors"), metadata={"format": "pt"})
cfg = dict(ck["cfg"]); cfg.update({"architecture": "tiny-gpt (decoder-only, tied embeddings)", "step": ck["step"],
                                   "val_loss": round(ck["val_loss"], 4)})
json.dump(cfg, open(os.path.join(OUT, "config.json"), "w"), indent=2)
for f in ("tokenizer.json", "model.py", "generate.py"): shutil.copy(os.path.join(H, f), os.path.join(OUT, f))
n = sum(v.numel() for v in sd.values())
print(f"exported step {ck['step']} (val loss {ck['val_loss']:.3f}), {n / 1e6:.1f}M params -> {OUT}")
