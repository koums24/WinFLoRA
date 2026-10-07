# One-off: fetch TinyLlama (safetensors only) and AG News into the paths the training script expects.
import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # Src/

from datasets import load_dataset
from huggingface_hub import snapshot_download

snapshot_download("TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                  local_dir="models/TinyLlama-1.1B-Chat-v1.0",
                  allow_patterns=["*.json", "*.safetensors", "tokenizer.model"])
d = load_dataset("ag_news")
d.save_to_disk("datasets/ag_news_dataset")
print(d)
