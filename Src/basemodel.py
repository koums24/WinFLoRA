# no_lora_baseline.py
# -*- coding: utf-8 -*-
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import math
import json
import copy
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, Subset
from datasets import load_dataset, load_from_disk
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils import shuffle
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, set_seed


# =========================
# Configuration
# =========================
# Number of clients and data size per client
NUM_CLIENTS        = 4
CLIENT_DATA_SIZE   = 5000          # Number of samples per client (sliced from the global training set)
RANDOM_SEED        = 42
MODEL_NAME         = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Training and Evaluation
NUM_EPOCHS         = 2
LR                 = 2e-4
BATCH_TRAIN        = 2
BATCH_EVAL         = 4
MAX_LEN            = 512

# Save Directories
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()
ART_DIR    = BASE_DIR / "no_lora_baseline" / "artifacts"
CKPT_DIR   = BASE_DIR / "no_lora_baseline" / "checkpoints"
DATA_CACHE = BASE_DIR / "datasets" / "ag_news_dataset"
ART_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

set_seed(RANDOM_SEED)


# =========================
# Data & Model Definition
# =========================
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = torch.tensor(labels, dtype=torch.long)
        # Pre-encode to save repeated encoding overhead in Trainer
        texts = [f"Classify this text: {t}" for t in texts]
        enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        self.input_ids = enc.input_ids
        self.attention_mask = enc.attention_mask

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


class TinyLlamaForNewsClassification(nn.Module):
  
    def __init__(self, base_model, num_labels, seed=0):
        super().__init__()
        self.base_model = base_model
        hidden = base_model.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden, num_labels)
        torch.manual_seed(seed)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        last = out.hidden_states[-1]                       # [B, T, H]
        mask = attention_mask.unsqueeze(-1).float()        # [B, T, 1]
        sum_hidden = (last * mask).sum(dim=1)              # [B, H]
        denom = mask.sum(dim=1).clamp(min=1e-9)            # [B, 1]
        pooled = sum_hidden / denom                        # [B, H]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "micro_f1": f1_score(labels, preds, average="micro"),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
    }


# =========================
# Utility Functions
# =========================
def freeze_backbone(model: TinyLlamaForNewsClassification):
    for p in model.base_model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True


def save_classifier_only(client_id: int, model: TinyLlamaForNewsClassification):
    d = ART_DIR / f"client_{client_id+1}"
    d.mkdir(parents=True, exist_ok=True)
    # Save only the classifier head parameters (weights + bias)
    cls = {k.replace("classifier.", ""): v.cpu()
           for k, v in model.state_dict().items() if k.startswith("classifier.")}
    torch.save(cls, d / "classifier.pt")
    # meta (optional)
    meta = {"model_name": MODEL_NAME}
    (d / "meta.json").write_text(json.dumps(meta, indent=2))


def load_classifier_only(client_id: int):
    p = ART_DIR / f"client_{client_id+1}" / "classifier.pt"
    return torch.load(p, map_location="cpu") if p.exists() else None


def avg_classifiers(classifier_list):
    """
    classifier_list: List[dict], each dict: {"weight": tensor[H, C], "bias": tensor[C]}.
    Takes a simple average.
    """
    out = copy.deepcopy(classifier_list[0])
    for k in out.keys():
        for i in range(1, len(classifier_list)):
            out[k] += classifier_list[i][k]
        out[k] /= float(len(classifier_list))
    return out


def frobenius_of_classifier(cls_dict):
    w = cls_dict["weight"].float().cpu()
    b = cls_dict["bias"].float().cpu()
    wf = torch.linalg.matrix_norm(w, ord="fro").item()
    bf = torch.linalg.vector_norm(b).item()
    return wf, bf, math.sqrt(wf*wf + bf*bf)


# =========================
# Main Process
# =========================
def main():
    # --------- Data ---------
    if DATA_CACHE.exists():
        print(f"[DATA] Load AG News from cache: {DATA_CACHE}")
        dataset = load_from_disk(DATA_CACHE)
    else:
        print(f"[DATA] Download AG News and save to cache: {DATA_CACHE}")
        dataset = load_dataset("ag_news")
        DATA_CACHE.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(DATA_CACHE)

    train_data = dataset["train"]
    texts = [item["text"] for item in train_data]
    labels = [item["label"] for item in train_data]
    texts, labels = shuffle(texts, labels, random_state=RANDOM_SEED)

    num_classes = 4
    # Global split: train/validation (mutually exclusive)
    all_idx = np.arange(len(texts))
    tr_idx, val_idx = train_test_split(
        all_idx, test_size=0.2, shuffle=True, random_state=RANDOM_SEED+1
    )
    train_texts  = [texts[i] for i in tr_idx]
    train_labels = [labels[i] for i in tr_idx]
    val_texts    = [texts[i] for i in val_idx]
    val_labels   = [labels[i] for i in val_idx]

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Global validation set (sub-sampling can speed it up)
    total_val_ds = NewsDataset(val_texts, val_labels, tokenizer, max_length=MAX_LEN)
    rng = np.random.default_rng(RANDOM_SEED)
    sub_idx = rng.choice(len(total_val_ds), size=max(1000, len(total_val_ds)//5), replace=False)
    global_val_ds = Subset(total_val_ds, sub_idx)
    print(f"[INFO] Global val subset = {len(global_val_ds)}")

    # Split the global training set by client
    client_data = []
    for i in range(NUM_CLIENTS):
        s = i * CLIENT_DATA_SIZE
        e = (i + 1) * CLIENT_DATA_SIZE
        c_texts  = train_texts[s:e]
        c_labels = train_labels[s:e]
        client_data.append((c_texts, c_labels))
    for i, (ct, _) in enumerate(client_data):
        h = hashlib.md5("".join(ct).encode()).hexdigest()
        print(f"[INFO] Client {i+1}: {len(ct)} samples, hash={h[:10]}...")

    # --------- Train each client (only the classifier head) ---------
    per_client_val = []      # Record validation metrics for each client
    saved_classifiers = []   # Record the classifier head for each client

    for cid, (c_texts, c_labels) in enumerate(client_data):
        # Re-split train/val within the client
        n = len(c_texts)
        idx = np.arange(n)
        tr_i, va_i = train_test_split(
            idx, test_size=0.2, shuffle=True, random_state=1000+cid
        )
        c_tr_texts  = [c_texts[i]  for i in tr_i]
        c_tr_labels = [c_labels[i] for i in tr_i]
        c_va_texts  = [c_texts[i]  for i in va_i]
        c_va_labels = [c_labels[i] for i in va_i]

        train_ds = NewsDataset(c_tr_texts, c_tr_labels, tokenizer, max_length=MAX_LEN)
        val_ds   = NewsDataset(c_va_texts, c_va_labels, tokenizer, max_length=MAX_LEN)

        base = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        model = TinyLlamaForNewsClassification(base, num_classes, seed=cid)
        freeze_backbone(model)  # Linear probing

        args = TrainingArguments(
            output_dir=str(CKPT_DIR / f"client_{cid+1}"),
            evaluation_strategy="no",   # Do not evaluate during training
            learning_rate=LR,
            per_device_train_batch_size=BATCH_TRAIN,
            per_device_eval_batch_size=BATCH_EVAL,
            num_train_epochs=NUM_EPOCHS,
            weight_decay=0.01,
            logging_strategy="no",
            save_strategy="no",
            report_to=[],
            no_cuda=False,              # Will automatically use GPU if available
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
        )
        print(f"\n[TRAIN] Client {cid+1} start ...")
        trainer.train()

        # Evaluate (on client's own validation set)
        res = trainer.evaluate()
        per_client_val.append((cid, res))
        print(f"[EVAL] Client {cid+1} on its val: {res}")

        # Save classifier head
        save_classifier_only(cid, model)
        cls = {k.replace("classifier.", ""): v.detach().cpu()
               for k, v in model.state_dict().items() if k.startswith("classifier.")}
        saved_classifiers.append(cls)

        # Release GPU memory
        del trainer, model, base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --------- Build global model: frozen backbone + simple average of classifier heads ---------
    print("\n[AGG] Averaging classifiers across clients ...")
    global_cls = avg_classifiers(saved_classifiers)
    wf, bf, tot = frobenius_of_classifier(global_cls)
    print(f"[AGG] ||W||_F={wf:.3f}, ||b||={bf:.3f}, total={tot:.3f}")

    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    global_model = TinyLlamaForNewsClassification(base, num_classes, seed=0)
    # Freeze backbone and load the averaged classifier head
    freeze_backbone(global_model)
    global_model.classifier.load_state_dict(global_cls)

    # Evaluate on the global validation set
    eval_args = TrainingArguments(
        output_dir=str(CKPT_DIR / "global_model"),
        evaluation_strategy="no",
        per_device_eval_batch_size=BATCH_EVAL,
        logging_strategy="no",
        report_to=[],
        no_cuda=False,
    )
    eval_trainer = Trainer(
        model=global_model,
        args=eval_args,
        eval_dataset=global_val_ds,
        compute_metrics=compute_metrics,
    )
    print(f"\n[EVAL] Global averaged classifier on global val ...")
    g_res = eval_trainer.evaluate()
    print("[RESULT] Global (avg classifier) on global val:", g_res)

    # Also check: apply the globally averaged classifier head to each client's validation set
    for cid, (c_texts, c_labels) in enumerate(client_data):
        n = len(c_texts)
        idx = np.arange(n)
        _, va_i = train_test_split(idx, test_size=0.2, shuffle=True, random_state=1000+cid)
        c_va_texts  = [c_texts[i]  for i in va_i]
        c_va_labels = [c_labels[i] for i in va_i]
        c_val_ds = NewsDataset(c_va_texts, c_va_labels, tokenizer, max_length=MAX_LEN)
        res = eval_trainer.evaluate(eval_dataset=c_val_ds, metric_key_prefix=f"global_on_client{cid+1}")
        print(f"[RESULT] Global (avg classifier) on Client {cid+1} val:", res)

    print("\n[SUMMARY] Per-client (own val) results:")
    for cid, r in per_client_val:
        print(f"  - Client {cid+1}: {r}")


if __name__ == "__main__":
    main()