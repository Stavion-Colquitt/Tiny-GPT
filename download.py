"""Download one slice of each openly licensed source into data/raw (about 5.7 GB). Resumable; safe to re-run.

  FineWeb-Edu sample-10BT shard 0      ODC-By          HuggingFaceFW/fineweb-edu
  Cosmopedia v2 shard 0 (SmolLM)       ODC-By          HuggingFaceTB/smollm-corpus
  TinyStories V2 (GPT-4) train + valid CDLA-Sharing-1.0 roneneldan/TinyStories
  Simple English Wikipedia 2023-11-01  CC BY-SA 3.0    wikimedia/wikipedia
"""
import os, time
from huggingface_hub import hf_hub_download

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
FILES = [
    ("HuggingFaceFW/fineweb-edu", "sample/10BT/000_00000.parquet"),
    ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2/train-00000-of-00104.parquet"),
    ("wikimedia/wikipedia", "20231101.simple/train-00000-of-00001.parquet"),
    ("roneneldan/TinyStories", "TinyStoriesV2-GPT4-valid.txt"),
    ("roneneldan/TinyStories", "TinyStoriesV2-GPT4-train.txt"),
]

for repo, path in FILES:
    t = time.time()
    p = hf_hub_download(repo, path, repo_type="dataset", local_dir=RAW)
    print(f"{repo}/{path}: {os.path.getsize(p) / 1e9:.2f} GB in {time.time() - t:.0f} s", flush=True)
print("DOWNLOADS DONE", flush=True)
