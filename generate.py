"""Sample from LM 1. Works from this repo (checkpoints/best.pt) or a Hugging Face download (model.safetensors).

    py -3.12 generate.py "Once upon a time" --tokens 120 --temperature 0.8
"""
import os, json, argparse, torch
from tokenizers import Tokenizer
from model import GPT, Config

H = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("prompt"); ap.add_argument("--tokens", type=int, default=120)
ap.add_argument("--temperature", type=float, default=0.8); ap.add_argument("--top_k", type=int, default=50)
ap.add_argument("--seed", type=int, default=None)
a = ap.parse_args()
if a.seed is not None: torch.manual_seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
if os.path.exists(os.path.join(H, "model.safetensors")):
    from safetensors.torch import load_file
    c = json.load(open(os.path.join(H, "config.json")))
    model = GPT(Config(**{k: c[k] for k in ("vocab", "block", "n_layer", "n_head", "n_embd")}))
    model.load_state_dict(load_file(os.path.join(H, "model.safetensors")), strict=False)  # head.weight is tied
else:
    ck = torch.load(os.path.join(H, "checkpoints", "best.pt"), map_location="cpu", weights_only=False)
    model = GPT(Config(**ck["cfg"])); model.load_state_dict(ck["model"])
model.to(dev).eval()
tok = Tokenizer.from_file(os.path.join(H, "tokenizer.json")); eot = tok.token_to_id("<|endoftext|>")
idx = torch.tensor([tok.encode(a.prompt).ids], device=dev)
print(tok.decode(model.generate(idx, a.tokens, temperature=a.temperature, top_k=a.top_k, eot=eot)[0].tolist()))
