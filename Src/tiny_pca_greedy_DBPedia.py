from itertools import cycle
import os
import re

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import copy
import hashlib
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import torch, math
import torch.nn as nn
from torch.utils.data import Dataset,Subset
from datasets import load_dataset, load_from_disk, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np, hashlib, json
import gc
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from transformers.trainer_utils import get_last_checkpoint

from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from rpca import RPCA, RPCA_weights, build_rpca_M_from_B, build_rpca_M_from_A, noise_by_residual_pca_B_loo,noise_by_residual_pca_A_loo
from dataset_model import NewsDataset, TinyLlamaForNewsClassification
from options import get_args
from clients_greedy import GreedyNoiseSelector
from utils import printargs, dirichlet_partition_indices, surpassed_percentage, compute_utilities, to_jsonable_metrics,frobenius_of_lora_AB, add_noise_to_lora, measure_noise_ratio_for_client

args = get_args()
printargs(args)
# === Define run config ===
num_clients        = args.num_clients
RUN_SERIAL         = args.run_serial   # serial or processpool
LOAD_ARTIFACTS     = args.load_artifacts
# add noise to dataset
NOISY_SET = False
noisy_proportion   = args.noisy_proportion
num_noisy          = int(num_clients * noisy_proportion)
noise_data_rate    = args.noise_data_rate
noise_mean         = args.noise_mean
noise_std          = args.noise_std
# add noise to LoRA A/B
ADD_LORA_NOISE     = args.add_lora_noise 
NOISE_SEED         = args.noise_seed
# ===== Per-client LoRA noise config (pre-generated) =====
SIGMA_MIN, SIGMA_MAX = args.sigma_min, args.sigma_max
rng_sigma = np.random.default_rng(NOISE_SEED)
  
weight_mode = args.weight_mode # Weights Allocation Mode 

# ===== cache configuration ===== 
try:
    BASE_DIR = Path(__file__).resolve().parent   
except NameError:                               
    BASE_DIR = Path.cwd()
ART_DIR = BASE_DIR / "noisy_tiny_DBP" / "artifacts"
RESULTS_DIR = BASE_DIR / "noisy_tiny_DBP" / "checkpoints"
ART_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
   

 
# === Load model and tokenizer ===
from huggingface_hub import snapshot_download

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
shared_model_path = BASE_DIR / "models" / "TinyLlama-1.1B-Chat-v1.0"
model_name = str(shared_model_path)

if (not shared_model_path.exists()) or (not any(shared_model_path.iterdir())):
    print(f"[INFO] Downloading {model_id} to {shared_model_path} ...")
    shared_model_path.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_id,
        local_dir=str(shared_model_path),
        local_dir_use_symlinks=False, 
    )

print(f"Loading TinyLlama model from local path: {shared_model_path}")
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
model     = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
# === Load or download  AGnews dataset ===

shared_dataset_path = BASE_DIR / "datasets" / "dbpedia_dataset"
if os.path.exists(shared_dataset_path):
    print(f"Loading DBPedia dataset from local path: {shared_dataset_path}")
    dataset = load_from_disk(shared_dataset_path)
else:
    print(f"Downloading DBPedia dataset and saving to local path: {shared_dataset_path}")
    dataset = load_dataset("dbpedia_14", split="train")
    dataset.save_to_disk(shared_dataset_path)
    print(f"Dataset saved to {shared_dataset_path}")

print("Shuffling dataset...")
dataset = dataset.shuffle(seed=42)

total_train_size = int(len(dataset)* 0.1)
eval_fraction = 0.001
total_eval_size = int(len(dataset) * eval_fraction)

train_dataset_slice = dataset.select(range(total_train_size))
eval_start = total_train_size

df_train = pd.DataFrame({
    'text': train_dataset_slice['content'],
    'label': train_dataset_slice['label']
})

print(f"Train set: {len(df_train)} samples")
num_classes = 14  # DBpedia has 14 classes
# Shuffle train set
df_train = shuffle(df_train, random_state=42).reset_index(drop=True)

# Use train set for client distribution
texts_original = df_train['text'].tolist()
labels_original = df_train['label'].tolist()

# ===== Split Train/Val=====
all_idx = np.arange(len(texts_original))
global_train_idx, global_val_idx = train_test_split( all_idx, test_size=0.2, random_state=42, shuffle=True, stratify=None)

global_train_texts  = [texts_original[i]  for i in global_train_idx]
global_train_labels = [labels_original[i] for i in global_train_idx]
global_val_texts    = [texts_original[i]  for i in global_val_idx]
global_val_labels   = [labels_original[i] for i in global_val_idx]
total_global_train_dataset = NewsDataset(global_train_texts, global_train_labels, tokenizer)
total_global_val_dataset   = NewsDataset(global_val_texts,   global_val_labels,   tokenizer)
val_idx = np.random.default_rng(42).choice(len(total_global_val_dataset), size=int(len(total_global_val_dataset)* 0.1), replace=False) # use 0.2 as validation dataset
global_val_dataset = Subset(total_global_val_dataset, val_idx)
print(f"[INFO] Global split -> total train: {len(global_train_texts)}, used validation dataset: {len(global_val_dataset)}")
iid = False

if iid:  
    # client_data_size = len(texts_original) // num_clients
    client_data_size = 500   
    client_data = []
    for i in range(num_clients):
        start_idx = i * client_data_size
        end_idx   = (i + 1) * client_data_size
        client_texts  = global_train_texts[start_idx:end_idx]   # split from global training dataset
        client_labels = global_train_labels[start_idx:end_idx]
        client_data.append((client_texts, client_labels))      
else:
    dir_alpha   = 0.5
    min_size    = 1    
    seed=42

    client_train_indices = dirichlet_partition_indices(labels=global_train_labels, num_clients=num_clients, alpha=dir_alpha, seed=seed)
    rng = np.random.default_rng(seed)
    client_data = []
    for idxs in client_train_indices:
        client_texts  = [global_train_texts[i]  for i in idxs]
        client_labels = [global_train_labels[i] for i in idxs]
        client_data.append((client_texts, client_labels))

    client_data_size = 500
    for i, (client_texts, client_labels) in enumerate(client_data):
        if len(client_texts) > client_data_size:
            sel = rng.choice(len(client_texts), size=client_data_size, replace=False)
        else:
            extra = rng.choice(len(client_texts), size=client_data_size - len(client_texts), replace=True)
            sel = np.concatenate([np.arange(len(client_texts)), extra])
        client_data[i] = ([client_texts[j] for j in sel], [client_labels[j] for j in sel])
    
splits = []  # [(train_idx, val_idx)] for each client
for cid, (client_texts, client_labels) in enumerate(client_data):
    n = len(client_texts)
    idx = np.arange(n)
    labels_arr = np.asarray(client_labels)
    strat = labels_arr if (len(np.unique(labels_arr)) > 1 and
                        all((labels_arr == c).sum() >= 2 for c in np.unique(labels_arr))) else None

    train_idx, val_idx = train_test_split(
        idx, test_size=0.2, shuffle=True, random_state=cid + 1000, stratify=strat
    )
    splits.append((train_idx, val_idx))

# each client’s validation dataset
client_val_datasets = []
for cid, ((client_texts, client_labels), (train_idx, val_idx)) in enumerate(zip(client_data, splits)):
    val_texts  = [client_texts[i]  for i in val_idx]
    val_labels = [client_labels[i] for i in val_idx]
    client_val_datasets.append(NewsDataset(val_texts, val_labels, tokenizer))

for i, (client_texts, _) in enumerate(client_data):
    text_hash = hashlib.md5("".join(client_texts).encode()).hexdigest()
    print(f"Client {i+1} Data: {len(client_texts)} samples, Hash: {text_hash}")

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

# =========== LoRA Configuration =======
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
    init_lora_weights=True,
)

# ========= save checkpoints========
def _client_dir(cid: int) -> str:
    d = os.path.join(ART_DIR, f"client_{cid}_dbpedia")
    os.makedirs(d, exist_ok=True)
    return d

def save_client_artifacts(client_id, lora_params, classifier_params, result, client_texts, lora_cfg):
    d = _client_dir(client_id)
    # save LoRA A/B + multihead
    torch.save(lora_params, os.path.join(d, "lora.pt"))
    torch.save(classifier_params, os.path.join(d, "classifier.pt"))
    
    meta = {
        "model_name": model_name,
        "num_labels": num_classes,
        "data_hash": hashlib.md5("".join(client_texts).encode()).hexdigest(),
        "lora_r": int(lora_cfg.r),
        "lora_alpha": int(lora_cfg.lora_alpha),
        "target_modules": list(lora_cfg.target_modules),
        "metrics": to_jsonable_metrics(result),
    }
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

def load_client_artifacts(client_id, expected_texts, expected_lora_cfg):
    d = _client_dir(client_id)
    try:
        with open(os.path.join(d, "meta.json"), "r") as f:
            meta = json.load(f)
        ok = True
        ok &= meta["model_name"] == model_name
        ok &= meta["num_labels"] == num_classes
        ok &= meta["lora_r"] == int(expected_lora_cfg.r)
        ok &= meta["lora_alpha"] == int(expected_lora_cfg.lora_alpha)
        ok &= set(meta["target_modules"]) == set(expected_lora_cfg.target_modules)
        # ok &= meta["data_hash"] == hashlib.md5("".join(expected_texts).encode()).hexdigest()
        if not ok:
            return None
        lora_params = torch.load(os.path.join(d, "lora.pt"), map_location="cpu")
        classifier_params = torch.load(os.path.join(d, "classifier.pt"), map_location="cpu")
        result = meta["metrics"]
        return lora_params, classifier_params, result
    except Exception:
        return None
    
# ===========================================================================
# Clients' trainning (with checkpoint) 
# ===========================================================================
def train_client(client_id, client_texts, client_labels, train_idx, val_idx, gpu_id):

    # cuda setting for process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
   
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  
        torch.cuda.empty_cache()

    torch.manual_seed(client_id + 1000)
    np.random.seed(client_id + 1000)
    
    print(f"\n=== Client {client_id+1} Starting Training on "
          f"{'GPU ' + str(gpu_id) if torch.cuda.is_available() else 'CPU'} ===")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        model = TinyLlamaForNewsClassification(base_model, num_classes, client_id)

        # inject LoRA
        model.base_model = get_peft_model(model.base_model, lora_config)
        
        train_texts = [client_texts[i]  for i in train_idx]
        train_labels = [client_labels[i] for i in train_idx]
        val_texts   = [client_texts[i]  for i in val_idx]
        val_labels  = [client_labels[i] for i in val_idx]
        
        train_dataset = NewsDataset(train_texts, train_labels, tokenizer)
        val_dataset   = NewsDataset(val_texts,  val_labels,  tokenizer)

        val_text_hash = hashlib.md5("".join(val_texts).encode()).hexdigest()
        print(f"Client {client_id+1} Validation Set: {len(val_texts)} samples, "
              f"Label Counts: {val_dataset.label_counts}, Hash: {val_text_hash}")

        # training parameters (with checkpoint)
        training_args = TrainingArguments(
            output_dir=os.path.join(RESULTS_DIR, f"client_{client_id}"),
            eval_strategy="no",    
            eval_steps=200,
            save_strategy="no",
            save_total_limit=2,
            load_best_model_at_end=False,
            # metric_for_best_model="accuracy",
            # greater_is_better=True,
            learning_rate=2e-4,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=2,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_strategy="no",
            logging_steps=5,
            report_to=[],                   
            no_cuda=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )
        
        os.makedirs(training_args.output_dir, exist_ok=True)
        
        # continue from checkpoint
        # last_ckpt = get_last_checkpoint(training_args.output_dir) if os.path.isdir(training_args.output_dir) else None
        # trainer.train(resume_from_checkpoint=last_ckpt)
        trainer.train()

        # merge LoRA, for evaluation
        merged_model = copy.deepcopy(model)
        merged_model.base_model = merged_model.base_model.merge_and_unload()
        eval_trainer = Trainer(
            model=merged_model,
            args=training_args,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )
        result = eval_trainer.evaluate()
        print(f"Client {client_id+1} Results: {result}")

        lora_params = {k: v.detach().cpu().clone()
               for k, v in model.base_model.state_dict().items()
               if "lora_" in k}
        classifier_params = {k: v.detach().cpu().clone()
                            for k, v in model.classifier.state_dict().items()}
        # save artifacts and checkpoints
        save_client_artifacts(
            client_id=client_id,
            lora_params=lora_params,
            classifier_params=classifier_params,
            result=result,
            client_texts=client_texts,
            lora_cfg=lora_config,
        )
        print("Artifacts saved to:", _client_dir(client_id), "->", os.listdir(_client_dir(client_id)))
        print("Results saved to:", training_args.output_dir, "->", os.listdir(training_args.output_dir))
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return lora_params, classifier_params, result

    except Exception as e:
        print(f"Client {client_id+1} Failed with Error: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None, None, None

Utility_mean_list, Global_acc_list, Global_average_list, Sigma_mean_list,Acc_mean_list =[],[],[],[],[]

# ===========================================================================
# main 
# ===========================================================================
if __name__ == "__main__":
    
    # initialize clients' noise
    T = args.times
    
    rng_gamma = np.random.default_rng(args.gamma_seed)
    r = rng_gamma.normal(loc=5, scale=(10-1)/6, size=num_clients)
    alpha = np.ones_like(r) # alpha = 1
    # beta = 1.0 / r  # gamma = 1 / r ∈ (0.1, 1]
    beta = np.round(beta, 3)

    sigmas = rng_sigma.normal(loc=SIGMA_MAX/2, scale=SIGMA_MAX/6, size=num_clients)
    sigmas = np.clip(sigmas, SIGMA_MIN, SIGMA_MAX)

    selector = GreedyNoiseSelector(
        sigma_max=SIGMA_MAX,
        alpha=alpha,
        beta=beta,
        sigma_init=sigmas,
        c=1.0,
        use_ema=True,
        rho=0.1
    )
    
    print("Fixed (alpha, beta) per client:")
    for i, (a, b) in enumerate(selector.get_alphas_betas()):
        print(f"  client {i}: alpha={a:.3f}, beta={b:.3f}")
        
    for t in range(T):
        print(f"\n============================ Round t={t} ================================= ")

        sigmas = selector.get_sigma()
        CLIENT_SIGMA = {i: (round(float(sigmas[i]), 3), round(float(sigmas[i]), 3)) for i in range(num_clients)}
        print("[LoRA noise per client] (A, B):", CLIENT_SIGMA)     

        multiprocessing.set_start_method("spawn", force=True)
        print("[Devive Check] torch.cuda.is_available() =", torch.cuda.is_available())
        print("[Devive Check] torch.cuda.device_count() =", torch.cuda.device_count())
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible:
            vis_list = [s.strip() for s in visible.split(",") if s.strip() != ""]
        else:
            vis_list = [str(i) for i in range(torch.cuda.device_count())]
        
        gpu_nums = len(vis_list)
        # device_cycle = cycle(vis_list)
        print(f"\n gpu_nums: {gpu_nums}")
        # gpu_nums = args.gpu_nums
        client_models = []      # clients' LoRA A/B
        client_classifiers = [] 
        client_results = []     # (client_id, eval_result)

        # 1) first load artifacts（not retraining） 
        to_train = []  # [(client_id, texts, labels)]
        for cid, (client_texts, client_labels) in enumerate(client_data):
            if LOAD_ARTIFACTS:
                loaded = load_client_artifacts(cid, client_texts, lora_config)
            else:
                loaded = None
            if loaded is not None:
                lora_params, classifier_params, result = loaded
                client_models.append(lora_params)             # clean LoRA
                client_classifiers.append(classifier_params)
                client_results.append((cid + 1, result))
                print(f"[CACHE] Loaded artifacts for client {cid+1}")
            else:
                to_train.append((cid, client_texts, client_labels))
        
        # 2) train lost client (and save）
        if to_train:
            print(f"{len(to_train)} clients missing artifacts -> training start")
            
            if RUN_SERIAL:
                # avoid OOM
                for cid, client_texts, client_labels in to_train:
                    gpu_id = 0
                    print(f"[SERIAL] Training client {cid+1} on GPU {gpu_id}")
                    train_idx, val_idx = splits[cid]
                    lora_params, classifier_params, result = train_client(cid, client_texts, client_labels, train_idx, val_idx, gpu_id)

                    if lora_params is not None and classifier_params is not None and result is not None:
                        client_models.append(lora_params)
                        client_classifiers.append(classifier_params)
                        client_results.append((cid + 1, result))
                    else:
                        print(f"Client {cid + 1} Result is None - Training Failed")

                    del lora_params, classifier_params, result # release storage
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            else:
                # processpool
                from concurrent.futures import ProcessPoolExecutor
                with ProcessPoolExecutor(max_workers=gpu_nums) as executor:
                    futures = []
                    for cid, client_texts, client_labels in to_train:
                        # gpu_id = cid % gpu_nums
                        gpu_id = cid % gpu_nums
                        train_idx, val_idx = splits[cid]
                        futures.append(executor.submit(train_client, cid, client_texts, client_labels, train_idx, val_idx, gpu_id)) # client_texts, client_labels include train and val, train_idx and val_idx are index

                    for i, future in enumerate(futures):
                        try:
                            lora_params, classifier_params, result = future.result()
                            if lora_params is not None and classifier_params is not None and result is not None:
                                client_models.append(lora_params) # original clean lora
                                client_classifiers.append(classifier_params)
                                # to_train[i][0] is the original cid
                                client_results.append((to_train[i][0] + 1, result))
                            else:
                                print(f"Client {to_train[i][0] + 1} Result is None - Training Failed")
                        except Exception as e:
                            print(f"Client {to_train[i][0] + 1} Failed to Return Result: {str(e)}")
        else:
            print("All client artifacts found. Skipping training.")

        # print("\n=== All Client Results ===")
        # for client_id, result in client_results:
        #     print(f"Client {client_id} Results: {result}")

        num_successful_clients = len(client_models)
        if num_successful_clients != num_clients:
            print(f"Warning: Only {num_successful_clients} out of {num_clients} clients completed successfully")
        else:
            print(f"All {num_clients} clients completed successfully")

        if num_successful_clients == 0:
            print("No clients completed successfully. Aborting aggregation.")
            raise SystemExit(1)

    # ========= 3) Add Noise to client's LoRA =========
        client_models_clean = [ {k: v.detach().cpu().clone() for k, v in lp.items()}
                            for lp in client_models ]
        if ADD_LORA_NOISE:
                client_models = []
                print("\n=== ADD_LORA_NOISE = True, Injecting noise to clients' LoRA (post-training) ===")
                for cid, clean_lp in enumerate(client_models_clean):
                    sigmaA, sigmaB = CLIENT_SIGMA[cid]
                    noisy_lp = add_noise_to_lora(clean_lp, sigmaA, sigmaB, seed=1234 + cid)
                    ratio = measure_noise_ratio_for_client(
                        clean_lp, noisy_lp, lora_alpha=lora_config.lora_alpha, r=lora_config.r
                    )
                    print(f"[NOISE][Client {cid+1}] σA={sigmaA}, σB={sigmaB} -> ||ΔW_noise||/||ΔW|| ≈ {ratio:.2%}")
                    client_models.append(noisy_lp)
        else:
            print("\n=== ADD_LORA_NOISE = False, using clean LoRA for aggregation ===")
            client_models = client_models_clean
            
    # ========= 4) Aggregating LoRA via RPCA-based Weights =========
        print("\n=== Aggregating LoRA via RPCA-based Weights ===")  
        #------------ RPCA and estimate noise ----------
        sigma_true_std = [float(CLIENT_SIGMA[cid][0]) for cid in range(num_successful_clients)] # for print 
        if weight_mode == 'rpca':
            # estimate by only B
            sigma_true_std = RPCA_weights(CLIENT_SIGMA, client_models, num_successful_clients)
        elif weight_mode == 'pca':
            key_sets_B = [{k for k in mp.keys() if "lora_B" in k} for mp in client_models]
            b_keys = sorted(set.intersection(*key_sets_B))
            key_sets_A = [{k for k in mp.keys() if "lora_A" in k} for mp in client_models]
            a_keys = sorted(set.intersection(*key_sets_A))
            
            # noise estimate
            sigma_hat_B_loo = noise_by_residual_pca_B_loo(client_models, b_keys, rank_k=1)  # np.ndarray, shape=(C,)
            eps = 1e-8
            weights = (1.0 / (sigma_hat_B_loo + eps)) ** args.tau
            weights = (weights / weights.sum()).astype(float) # calculate weights
            for i in range(num_successful_clients):
                st = "N/A" if (sigma_true_std[i] is None) else f"{sigma_true_std[i]:.3f}"
                print(f"Client {i+1}: sigma_hat_B_loo={sigma_hat_B_loo[i]:.3f}, sigma_true={st}")             
            print("Weights by residual-PCA-LOO on B:", ", ".join(f"{w:.3f}" for w in weights)) 
        else:
            #average weights
            weights = [1.0 / num_successful_clients] * num_successful_clients
            print(f"Client Weights via Average: {weights}")
    
        # ========= 5) Global Model and Stacking LoRA  =========
        APPLY_GLOBAL_LORA = True
        if APPLY_GLOBAL_LORA:
            global_r = lora_config.r * num_successful_clients
            
            stack_lora_config = LoraConfig(
                task_type=lora_config.task_type,
                r=global_r,
                lora_alpha=lora_config.lora_alpha * num_successful_clients, # keep scaling factor
                lora_dropout=lora_config.lora_dropout,
                target_modules=lora_config.target_modules,
                init_lora_weights=True,
            )
            
            def deltaW_fro_hash(lora_params: dict):
                A_keys = sorted([k for k in lora_params if "lora_A" in k])
                B_keys = sorted([k for k in lora_params if "lora_B" in k])
                s_all = 0.0
                md5 = hashlib.md5()
                for Ak, Bk in zip(A_keys, B_keys):
                    A = lora_params[Ak].float()
                    B = lora_params[Bk].float()
                    DW = B @ A
                    s_all += (DW*DW).sum().item()
                    md5.update(DW.detach().cpu().numpy().tobytes())
                return (s_all ** 0.5, md5.hexdigest()[:12])

            # print("\n[CHECK] Artifacts actually loaded for aggregation (without scaling):")
            # for cid, lp in enumerate(client_models):
            #     fro, hh = deltaW_fro_hash(lp)
            #     print(f"  client{cid+1}: ||ΔW||_F={fro:.3f}, hash={hh}")
        
            # global model for aggragting LoRA
            global_model = AutoModelForCausalLM.from_pretrained(model_name)
            global_model = TinyLlamaForNewsClassification(global_model, num_classes, client_id=0)
            global_model.base_model = get_peft_model(global_model.base_model, stack_lora_config)
            global_state_dict = global_model.base_model.state_dict()

            # test and print norm of client's LoRA
            # for cid, lp in enumerate(client_models):    # client_models: List[dict]
            #     s = frobenius_of_lora_AB(lp)
                # print(f"[CHECK][client {cid}] ||A||_F={s['fro_A']:.3f}, ||B||_F={s['fro_B']:.3f}, total={s['fro_all']:.3f}")
            
            # A B stack
            sample_lora_params = client_models[0]
            lora_B_keys = [k for k in sample_lora_params.keys() if "lora_B" in k]
            lora_A_keys = [k for k in sample_lora_params.keys() if "lora_A" in k]
            if len(lora_B_keys) != len(lora_A_keys):
                raise ValueError("Mismatch between number of LoRA A and B matrices")

            for B_key, A_key in zip(lora_B_keys, lora_A_keys):
                # print(f"\nStacking for {B_key} and {A_key}")
                stacked_B = None
                stacked_A = None
                for i, (model_params, weight) in enumerate(zip(client_models, weights)):
                    B_i = model_params[B_key] * (weight ** 0.5)
                    A_i = model_params[A_key] * (weight ** 0.5)
                    if i == 0:
                        stacked_B = B_i
                        stacked_A = A_i
                    else:
                        stacked_B = torch.cat((stacked_B, B_i), dim=1)
                        stacked_A = torch.cat((stacked_A, A_i), dim=0)
                if stacked_B.shape[1] != stacked_A.shape[0]:
                    raise ValueError(f"Dimension mismatch: stacked_B {stacked_B.shape}, stacked_A {stacked_A.shape}")
                global_state_dict[B_key] = stacked_B
                global_state_dict[A_key] = stacked_A

            missing, unexpected = global_model.base_model.load_state_dict(global_state_dict, strict=False)
            # check whether injecting LoRA
            print("[CHECK] missing:", missing)
            print("[CHECK] unexpected:", unexpected)
            assert all("lora_" not in m for m in missing), "LoRA keys were NOT loaded!"
            num_lora_tensors = sum(1 for k in global_model.base_model.state_dict().keys() if "lora_" in k)
            print(f"[CHECK] #LoRA tensors in global model: {num_lora_tensors}")
            if num_lora_tensors == 0: print("LoRA is not injected")
            global_model.base_model = global_model.base_model.merge_and_unload() 

        else:
            # not injecting lora, use base model
            global_model = AutoModelForCausalLM.from_pretrained(model_name)
            global_model = TinyLlamaForNewsClassification(global_model, num_classes, client_id=0)
            
        # classifier by average
        avg_classifier_dict = copy.deepcopy(client_classifiers[0])
        for key in avg_classifier_dict.keys():
            avg_classifier_dict[key] = sum(weights[i] * client_classifiers[i][key] for i in range(num_successful_clients))
        global_model.classifier.load_state_dict(avg_classifier_dict)
    
    # ===========================================================================
    # Evaluation 
    # ===========================================================================
        # ===== evaluate global model on global validation dataset
        global_training_args = TrainingArguments(
            output_dir="./results/global_model",
            eval_strategy="no",
            per_device_eval_batch_size=16,
            logging_dir="./logs",
            report_to=[],
            no_cuda=False,
        )
        trainer = Trainer(
            model=global_model,
            args=global_training_args,
            eval_dataset=global_val_dataset,
            compute_metrics=compute_metrics,
        )
        print(f"[INFO] Evaluate device is {trainer.args.device}")
        results = trainer.evaluate()
        print("\n=== Global Model Results on Original/Validation Dataset ===")
        print(results)

        global_metrics = ["eval_accuracy", "eval_micro_f1", "eval_macro_f1", "eval_precision", "eval_recall"]
        global_metric_values = [float(results[m]) for m in global_metrics if m in results]
        if global_metric_values:
            global_avg = np.mean(global_metric_values)
            print(f"\n=== Global Model Metrics Average ===\n{global_avg:.4f}")
        global_accuracy = float(global_metric_values[0])
        print(f"\n=== Global Model Accuracy  ===\n{global_accuracy:.3f}")
        
        # evaluate global model on clients dataset
        client_results_global_model = []
        for cid, val_ds in enumerate(client_val_datasets, start=1):
            res = trainer.evaluate(eval_dataset=val_ds) 
            client_results_global_model.append((cid, res)) 
            # print(f"\n=== Global model on Client {cid+1}'s validation set ===")
            # print(res)
            
        metrics = ["eval_accuracy", "eval_micro_f1", "eval_macro_f1", "eval_precision", "eval_recall"]
        metric_values_agg = {m: [] for m in metrics}
        for _, res in client_results_global_model:
            for m in metrics:
                metric_values_agg[m].append(res.get(m, np.nan))

        print("\n=== Individual Client Metrics by Category after aggregation  ===")
        for m in metrics:
            arr = metric_values_agg[m]
            avg = np.nanmean(arr) if len(arr) > 0 else float("nan")
        
            print(f"{m} Array: {[round(x, 4) if isinstance(x, (int, float, np.floating)) else x for x in arr]}")
            print(f"{m} Average: {avg:.4f}\n")

        print("\n=== Per-Client Metric Averages after aggregation ===")
        for cid, res in client_results_global_model:   
            acc   = float(res.get("eval_accuracy",  np.nan))
            micro = float(res.get("eval_micro_f1",  np.nan))
            macro = float(res.get("eval_macro_f1",  np.nan))
            prec  = float(res.get("eval_precision", np.nan))
            rec   = float(res.get("eval_recall",    np.nan))
            avg   = np.nanmean([acc, micro, macro, prec, rec])
            print(f"Client {cid}: Accuracy={acc:.4f}, Micro F1={micro:.4f}, "
                f"Macro F1={macro:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, "
                f"Average={avg:.4f}")
            
        # ======== Client evaluation on one test client========
        client_id_for_test = 0
        client_model = AutoModelForCausalLM.from_pretrained(model_name)
        client_model = TinyLlamaForNewsClassification(client_model, num_classes, client_id=client_id_for_test)
        client_model.base_model = get_peft_model(client_model.base_model, lora_config)
        state_dict = client_model.state_dict()

        for key in client_models[client_id_for_test]:
            state_dict[key] = client_models[client_id_for_test][key]
        for key in client_classifiers[client_id_for_test]:
            state_dict["classifier." + key] = client_classifiers[client_id_for_test][key]

        client_model.load_state_dict(state_dict, strict=False)
        client_model.base_model = client_model.base_model.merge_and_unload()

        client_trainer = Trainer(
            model=client_model,
            args=global_training_args,
            eval_dataset=global_val_dataset,
            compute_metrics=compute_metrics,
        )
        # client_results_eval = client_trainer.evaluate()
        # print("\n=== Global Test Data Results on Client 1's Model ===")
        # print(client_results_eval)

        # ===== each client's metrics
        def _as_float(x): 
            try:
                return float(x) # JSON metrics to float
            except Exception:
                return 0.0
            
        print("\n=== Individual Client Metrics by Category after training (not aggregated) ===")
        metric_arrays = {metric: [] for metric in metrics}
        for client_id, result in client_results:
            for metric in metrics:
                metric_arrays[metric].append(_as_float(result.get(metric, 0.0)))
        for metric in metrics:
            array = metric_arrays[metric]
            avg = np.mean(array) if array else 0.0
            print(f"{metric} Array: {array}")
            print(f"{metric} Average: {avg:.4f}\n")

        print("\n=== Per-Client Metric Averages after training (not aggregated) ===")
        for client_id, result in client_results:
            metric_values = [_as_float(result.get(metric, 0.0)) for metric in metrics]
            client_avg = np.mean(metric_values) if metric_values else 0.0
            print(f"Client {client_id}: "
                f"Accuracy={_as_float(result.get('eval_accuracy', 0.0)):.4f}, "
                f"Micro F1={_as_float(result.get('eval_micro_f1', 0.0)):.4f}, "
                f"Macro F1={_as_float(result.get('eval_macro_f1', 0.0)):.4f}, "
                f"Precision={_as_float(result.get('eval_precision', 0.0)):.4f}, "
                f"Recall={_as_float(result.get('eval_recall', 0.0)):.4f}, "
                f"Average={client_avg:.4f}")
            
        del global_model, trainer, client_trainer
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

     
        utility_acc_t = metric_values_agg.get('eval_accuracy')
        utilities, norm_utilities = compute_utilities(alpha, beta, utility_acc_t, sigmas, SIGMA_MAX)
        norm_utilities_mean = float(np.nanmean(np.asarray(norm_utilities, dtype=float).reshape(-1)))
        Sigma_mean = float(np.nanmean(np.asarray(sigmas, dtype=float).reshape(-1)))
        Acc_mean = float(np.nanmean(np.asarray(utility_acc_t, dtype=float).reshape(-1)))

        sigma_next = selector.update_and_select_next(t=t, acc_t=utility_acc_t)
       
        print(f"  acc_{t}   = {[f'{x:.4f}' for x in utility_acc_t]}")
        print(f"  sigma_{t} = {[f'{x:.3f}' for x in sigmas]}")
        print(f"->sigma_{t+1}= {[f'{x:.3f}' for x in sigma_next]}")
        print(f"  Utility_{t} = {[f'{x:.3f}' for x in norm_utilities]}")
        print(f"  Utilities_mean_{t} = {norm_utilities_mean:.3f}")
        print(f"  Sigma_mean_{t} = {Sigma_mean:.3f}")
        print(f"  Acc_mean_{t} = {Acc_mean:.3f}")

        Utility_mean_list.append(round(norm_utilities_mean,4))
        Global_acc_list.append(round(global_accuracy,4))
        Global_average_list.append(round(global_avg,4))
        Sigma_mean_list.append(round(Sigma_mean,4))
        Acc_mean_list.append(round(Acc_mean,4))
       
    print(f"Global_acc_list= {Global_acc_list}")
    print(f"Global_average_list= {Global_average_list}")
    print(f"Utility_mean_list= {Utility_mean_list}")
    print(f"Acc_mean_list= {Acc_mean_list}")
    print(f"Sigma_mean_list= {Sigma_mean_list}")