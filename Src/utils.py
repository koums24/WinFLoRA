import numpy as np
import torch

def printargs(args):
    fields = [
        "add_lora_noise",
        "load_artifacts",
        "weight_mode",
        "tau",
        "noise_seed",
        "gamma_seed",
        "sigma_max",
    ]
    
    def _get(a, k, default=None):
        if isinstance(a, dict):
            return a.get(k, default)
        return getattr(a, k, default)
    key_width = max(len(k) for k in fields)

    print("=== Arguments ===")
    for k in fields:
        v = _get(args, k, default=None)
        print(f"{k:<{key_width}} : {repr(v)}")
        

def dirichlet_partition_indices(labels, num_clients, alpha=0.3, seed=42):
        labels = np.asarray(labels)
        classes = np.unique(labels)
        rng = np.random.default_rng(seed)

        while True:
            client_indices = [[] for _ in range(num_clients)]
            for k in classes:
                k_idx = np.where(labels == k)[0]
                rng.shuffle(k_idx)
                p = rng.dirichlet(np.full(num_clients, alpha))       # 各客户端占比
                counts = rng.multinomial(len(k_idx), p)              # 对应该类的样本数
                s = 0
                for cid, c in enumerate(counts):
                    if c > 0:
                        client_indices[cid].extend(k_idx[s:s+c].tolist())
                        s += c
            if min(len(x) for x in client_indices) >= 1:
                break
        for cid in range(num_clients):
            rng.shuffle(client_indices[cid])
        return client_indices
    
def surpassed_percentage(weights):
    w = np.asarray(weights, dtype=float)
    N = w.size
    if N <= 1:
        return np.zeros_like(w, dtype=float)
    vals, inv, counts = np.unique(w, return_inverse=True, return_counts=True)  
    
    counts_less_group = np.cumsum(counts) - counts  # shape=(#groups,)
    less = counts_less_group[inv]
  
    return less / (N - 1)

def compute_utilities(alpha, beta, acc_prev, sigmas, sigma_max):
  
    a  = np.asarray(alpha,  dtype=float).reshape(-1)
    b  = np.asarray(beta,   dtype=float).reshape(-1)
    ap = np.asarray(acc_prev, dtype=float).reshape(-1)
    sg = np.asarray(sigmas, dtype=float).reshape(-1)
   
    utility = a * ap + b * sg/sigma_max
    norm_utility = (a * ap + b * sg/sigma_max) / (a+b)
    return utility, norm_utility

def to_jsonable_metrics(result: dict) -> dict:
    out = {}
    for k, v in result.items():
        if isinstance(v, (int, float)):
            out[k] = float(v)
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        else:
            try:
                out[k] = float(v)
            except Exception:
                out[k] = str(v)
    return out

def frobenius_of_lora_AB(lora_params: dict):
    import math, torch
    A_sq = B_sq = 0.0
    for k, t in lora_params.items():
        v = t.detach().float().cpu()
        s = (v * v).sum().item()
        if "lora_A" in k:
            A_sq += s
        elif "lora_B" in k:
            B_sq += s
    return {
        "fro_A": math.sqrt(A_sq),
        "fro_B": math.sqrt(B_sq),
        "fro_all": math.sqrt(A_sq + B_sq),
    }
    
def add_noise_to_lora(lora_params: dict, sigmaA: float, sigmaB: float, seed: int = None):
    """对单个 client 的 LoRA 参数 (A/B) 加高斯噪声，返回新的字典（不改原始字典）。"""
    g = torch.Generator(device='cpu')
    if seed is not None:
        g.manual_seed(int(seed))
    out = {}
    for k, v in lora_params.items():
        t = v.detach().cpu()
        if "lora_A" in k:
            eps = torch.normal(0.0, sigmaA, size=t.shape, generator=g, dtype=t.dtype)
            out[k] = t + eps
        elif "lora_B" in k:
            eps = torch.normal(0.0, sigmaB, size=t.shape, generator=g, dtype=t.dtype)
            out[k] = t + eps
        else:
            out[k] = t.clone()
    return out

def measure_noise_ratio_for_client(lora_clean: dict, lora_noisy: dict, lora_alpha: float, r: int):
    import math, re
    def nat_key(s: str):
        return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]

    A_keys = sorted([k for k in lora_clean if "lora_A" in k], key=nat_key)
    B_keys = sorted([k for k in lora_clean if "lora_B" in k], key=nat_key)
    assert len(A_keys) == len(B_keys)
    scale = float(lora_alpha) / float(r)

    num_sq, den_sq = 0.0, 0.0
    for Ak, Bk in zip(A_keys, B_keys):
        A  = lora_clean[Ak].float()
        B  = lora_clean[Bk].float()
        A_ = lora_noisy[Ak].float()
        B_ = lora_noisy[Bk].float()
        DW  = scale * (B  @ A)
        DW_ = scale * (B_ @ A_)
        E   = DW_ - DW
        nf = torch.linalg.matrix_norm(E,  ord='fro').item()
        sf = torch.linalg.matrix_norm(DW, ord='fro').item()
        num_sq += nf*nf
        den_sq += sf*sf
    return math.sqrt(num_sq) / (math.sqrt(den_sq) + 1e-12)