"""
Accuracy impact of the attacks in attack_sim.py on the aggregated global model.

Rebuilds the global model exactly like tiny_pca_greedy_agnews.py (one round):
  delta-W = (lora_alpha/r) * sum_i w_i B_i A_i   (= stacking sqrt(w_i)B_i, sqrt(w_i)A_i, r*N, alpha*N)
  classifier = sum_i w_i * classifier_i
and evaluates it on a fixed 600-sample subset (seed 0) of the repo's 2400-sample global
validation subset. The base model is loaded once; q/v weights are reset per configuration.

Usage:
  python eval_attacks.py [--n_eval 600] [--out attack_results/eval_attacks.json]
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from datasets import load_from_disk
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from transformers import AutoModelForCausalLM, AutoTokenizer

import attack_sim as A
from dataset_model import NewsDataset, TinyLlamaForNewsClassification

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "TinyLlama-1.1B-Chat-v1.0"
DATA_DIR = BASE_DIR / "datasets" / "ag_news_dataset"
NUM_CLASSES = 4
LORA_SCALE = 32 / 16          # lora_alpha / r (unchanged by stacking: alpha*N / r*N)


# ---------------------------------------------------------------------------
# Global validation subset (copied from tiny_pca_greedy_agnews.py)
# ---------------------------------------------------------------------------
def global_val_texts_labels():
    train_data = load_from_disk(str(DATA_DIR))["train"]
    texts, labels = shuffle([x["text"] for x in train_data], [int(x["label"]) for x in train_data],
                            random_state=42)
    _, global_val_idx = train_test_split(np.arange(len(texts)), test_size=0.2, random_state=43,
                                         shuffle=True, stratify=None)
    vt = [texts[i] for i in global_val_idx]
    vl = [labels[i] for i in global_val_idx]
    val_idx = np.random.default_rng(42).choice(len(vt), size=int(len(vt) * 0.1), replace=False)
    return [vt[i] for i in val_idx], [vl[i] for i in val_idx]


def load_classifiers(artifact_dir, n):
    out = []
    for cid in range(n):
        d = sorted(Path(artifact_dir).glob(f"client_{cid}_*"))[0]
        out.append({k: v.float() for k, v in torch.load(d / "classifier.pt", map_location="cpu").items()})
    return out


# ---------------------------------------------------------------------------
# Global model (base loaded once, LoRA merged per configuration)
# ---------------------------------------------------------------------------
class GlobalModel:
    def __init__(self, device):
        base = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), local_files_only=True,
                                                    torch_dtype=torch.float32)
        self.model = TinyLlamaForNewsClassification(base, NUM_CLASSES, client_id=0).to(device).eval()
        self.device = device
        self.mods = {n: m for n, m in self.model.base_model.named_modules()
                     if n.endswith("q_proj") or n.endswith("v_proj")}
        self.W0 = {n: m.weight.detach().clone() for n, m in self.mods.items()}

    @torch.no_grad()
    def set_global(self, loras, classifiers, weights):
        w = torch.tensor(np.asarray(weights), dtype=torch.float32, device=self.device)
        for n, m in self.mods.items():
            m.weight.copy_(self.W0[n])
        for Ak, Bk in A.ab_pairs(loras[0]):
            name = Ak.replace("base_model.model.", "", 1).replace(".lora_A.default.weight", "")
            dW = sum(w[i] * (lp[Bk].to(self.device) @ lp[Ak].to(self.device)) for i, lp in enumerate(loras))
            self.mods[name].weight.add_(LORA_SCALE * dW)
        cls = {k: sum(float(weights[i]) * c[k] for i, c in enumerate(classifiers)) for k in classifiers[0]}
        self.model.classifier.load_state_dict(cls)

    @torch.no_grad()
    def accuracy(self, ds, bs, right_pad):
        correct, nonfinite = 0, 0
        for s in range(0, len(ds), bs):
            ids = ds.input_ids[s:s + bs]
            am = ds.attention_mask[s:s + bs]
            y = ds.labels[s:s + bs].to(self.device)
            if right_pad:                       # drop all-pad tail columns (no effect on causal mean-pool)
                L = int(am.sum(1).max())
                ids, am = ids[:, :L], am[:, :L]
            ids, am = ids.to(self.device), am.to(self.device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"):
                logits = self.model(input_ids=ids, attention_mask=am)["logits"].float()
            if not torch.isfinite(logits).all():  # fp16 overflow -> redo this batch in fp32
                nonfinite += 1
                logits = self.model(input_ids=ids, attention_mask=am)["logits"].float()
            correct += (logits.argmax(-1) == y).sum().item()
        return correct / len(ds), nonfinite


# ---------------------------------------------------------------------------
# Configurations (uploads + weights), built with attack_sim helpers
# ---------------------------------------------------------------------------
def build_configs(clean, classifiers, sigma_max, noise_seed, s_a1):
    N = len(clean)
    # training rounds sigmas to 3 decimals (CLIENT_SIGMA); keep that so 'honest' matches the repo run
    honest = np.round(A.honest_sigmas(N, sigma_max, noise_seed), 3)
    cfgs = {}

    def add(name, est, uploads, cls, sig_true, uniform=False):
        cfgs[name] = (est, uploads, cls, sig_true, uniform)

    add("uniform_clean", None, clean, classifiers, np.zeros(N), uniform=True)
    add("uniform_honest", None, lambda: A.noisy_round(clean, honest), classifiers, honest, uniform=True)
    for est in ("repo", "paper"):
        add(f"honest_{est}", est, lambda: A.noisy_round(clean, honest), classifiers, honest)

        sig = honest.copy(); sig[0] = sigma_max
        add(f"A1_s1_{est}", est, lambda sig=sig: A.noisy_round(clean, sig), classifiers, sig)

        def a1(sig=sig):
            up = A.noisy_round(clean, sig)
            up[0] = A.rescale(up[0], s_a1)
            return up
        add(f"A1_s{s_a1}_{est}", est, a1, classifiers, sig)

        sig3 = honest.copy(); sig3[0] = sig3[1] = sigma_max
        def a3(sig3=sig3):
            up = A.noisy_round(clean, sig3)
            up[1] = {k: v.clone() for k, v in up[0].items()}     # Sybil twin: identical upload
            return up
        cls3 = list(classifiers); cls3[1] = classifiers[0]        # ...including the classifier head
        add(f"A3_sybil_{est}", est, a3, cls3, sig3)

    # Controls at a moderate attacker sigma (~median honest): at sigma_max a single attacker already
    # collapses accuracy even at its fair (low) weight, so these isolate the effect of weight capture.
    sig_mid = 0.05
    for est in ("repo", "paper"):
        sigm = honest.copy(); sigm[0] = sig_mid
        add(f"mid_A1_s1_{est}", est, lambda sigm=sigm: A.noisy_round(clean, sigm), classifiers, sigm)

        def a1m(sigm=sigm):
            up = A.noisy_round(clean, sigm)
            up[0] = A.rescale(up[0], s_a1)
            return up
        add(f"mid_A1_s{s_a1}_{est}", est, a1m, classifiers, sigm)

        sig3m = honest.copy(); sig3m[0] = sig3m[1] = sig_mid
        add(f"mid_honestpair_{est}", est, lambda sig3m=sig3m: A.noisy_round(clean, sig3m), classifiers, sig3m)

        def a3m(sig3m=sig3m):
            up = A.noisy_round(clean, sig3m)
            up[1] = {k: v.clone() for k, v in up[0].items()}
            return up
        cls3m = list(classifiers); cls3m[1] = classifiers[0]
        add(f"mid_A3_sybil_{est}", est, a3m, cls3m, sig3m)
    return cfgs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifact_dir", default="noisy_tiny_agnews/artifacts")
    p.add_argument("--num_clients", type=int, default=10)
    p.add_argument("--sigma_max", type=float, default=0.1)
    p.add_argument("--noise_seed", type=int, default=2025)
    p.add_argument("--tau", type=float, default=0.8)
    p.add_argument("--a1_scale", type=float, default=128)
    p.add_argument("--n_eval", type=int, default=600)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--out", default="attack_results/eval_attacks.json")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    res = json.loads(out.read_text()) if out.exists() else {}
    res.setdefault("config", {k: v for k, v in vars(args).items()})
    res.setdefault("runs", {})

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    texts, labels = global_val_texts_labels()
    sub = np.random.default_rng(0).choice(len(texts), size=min(args.n_eval, len(texts)), replace=False)
    ds = NewsDataset([texts[i] for i in sub], [labels[i] for i in sub], tok)
    res["config"].update(eval_set=f"{len(sub)} of {len(texts)} global-val samples (rng seed 0)",
                         padding_side=tok.padding_side)
    print(f"device={device}  eval n={len(ds)}  padding_side={tok.padding_side}", flush=True)

    clean = A.load_artifact_loras(args.artifact_dir, args.num_clients)
    classifiers = load_classifiers(args.artifact_dir, args.num_clients)
    gm = GlobalModel(device)

    for name, (est, uploads, cls, sig_true, uniform) in build_configs(
            clean, classifiers, args.sigma_max, args.noise_seed, args.a1_scale).items():
        if name in res["runs"]:
            print(f"[skip] {name}: acc={res['runs'][name]['accuracy']:.4f}", flush=True)
            continue
        t0 = time.time()
        loras = uploads() if callable(uploads) else uploads
        if uniform:
            w, sh = np.full(len(loras), 1.0 / len(loras)), None
        else:
            A.ESTIMATOR = est
            rank_k = 1 if est == "repo" else len(loras) - 2
            sh, w = A.server_weights(loras, rank_k=rank_k, tau=args.tau)
        gm.set_global(loras, cls, w)
        acc, nonfinite = gm.accuracy(ds, args.batch_size, right_pad=tok.padding_side == "right")
        res["runs"][name] = {
            "estimator": est, "accuracy": acc, "n": len(ds), "fp32_fallback_batches": nonfinite,
            "weights": [float(x) for x in w],
            "sigma_hat": None if sh is None else [float(x) for x in sh],
            "sigma_true": [float(x) for x in sig_true],
            "seconds": round(time.time() - t0, 1),
        }
        out.write_text(json.dumps(res, indent=2))
        print(f"[done] {name}: acc={acc:.4f}  w0={w[0]:.3f} w1={w[1]:.3f}  "
              f"fallback={nonfinite}  ({time.time() - t0:.0f}s)", flush=True)
        del loras
        torch.cuda.empty_cache()
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
