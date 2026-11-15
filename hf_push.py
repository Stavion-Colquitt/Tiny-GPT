"""Create Stavion-Colquitt/Tiny-GPT-LM1 (public) and upload hf_export/. Uses the huggingface_hub login already on this PC."""
import os
from huggingface_hub import HfApi
api = HfApi(); repo = "Stavion-Colquitt/Tiny-GPT-LM1"
api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_export")
info = api.upload_folder(folder_path=d, repo_id=repo, repo_type="model",
                         commit_message="Tiny GPT (LM 1): weights, tokenizer, model code and card (val ppl 18.6)")
print("uploaded:", info.commit_url if hasattr(info, "commit_url") else info)
print(sorted(f for f in api.list_repo_files(repo)))
