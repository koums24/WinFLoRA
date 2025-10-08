# rpca.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
import re

class RPCA:
    def __init__(self,
                 lam: Optional[float] = None,
                 mu: float = 1.0,
                 max_iter: int = 500,
                 tol: float = 1e-6,
                 verbose: bool = False) -> None:
        self.lam = lam
        self.mu = mu
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

        # fitted
        self.L_: Optional[np.ndarray] = None
        self.S_: Optional[np.ndarray] = None
        self.rel_err_: Optional[float] = None
        self.n_iter_: Optional[int] = None

    # ---------- public API ----------
    def fit(self, M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run PCP-ADMM and return (L, S).
        M: shape (d, n)
        """
        M = np.asarray(M, dtype=np.float64, order="F")  # Fortran-order helps SVD a bit
        m, n = M.shape
        lam = self.lam if self.lam is not None else 1.0 / np.sqrt(max(m, n))
        mu = float(self.mu)
        assert mu > 0.0

        L = np.zeros_like(M)
        S = np.zeros_like(M)
        Y = np.zeros_like(M)

        Mnrm = np.linalg.norm(M, "fro") + 1e-12

        for k in range(self.max_iter):
            L = self._svt(M - S + (1.0 / mu) * Y, tau=1.0 / mu) # L-update: singular value thresholding
            T = M - L + (1.0 / mu) * Y
            S = self._soft_threshold(T, tau=lam / mu)  # S-update: elementwise soft-thresholding
            Y = Y + mu * (M - L - S) # dual update

            rel_err = np.linalg.norm(M - L - S, "fro") / Mnrm
            if self.verbose and (k % 50 == 0 or rel_err < self.tol):
                print(f"[RPCA] iter={k:4d}, rel_err={rel_err:.3e}")
            if rel_err < self.tol:
                self.n_iter_ = k + 1
                self.rel_err_ = float(rel_err)
                break
        else:
            self.n_iter_ = self.max_iter
            self.rel_err_ = float(rel_err)

        self.L_, self.S_ = L, S
        return L, S

    def estimate_noise(self, axis: int = 0) -> np.ndarray:
        if self.S_ is None:
            raise RuntimeError("Call fit(M) before estimate_noise().")
        return np.linalg.norm(self.S_, axis=axis)

    def weights_from_noise(self, sigma_hat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        sigma_hat = np.asarray(sigma_hat, dtype=np.float64)
        if np.allclose(sigma_hat, 0):
            return np.ones_like(sigma_hat) / sigma_hat.size
        inv = 1.0 / (sigma_hat + eps)
        return inv / inv.sum()

    # ---------- static helpers ----------
    @staticmethod
    def _soft_threshold(X: np.ndarray, tau: float) -> np.ndarray:
        return np.sign(X) * np.maximum(np.abs(X) - tau, 0.0)

    @staticmethod
    def _svt(X: np.ndarray, tau: float) -> np.ndarray:
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        s_thr = np.maximum(s - tau, 0.0)
        return (U * s_thr) @ Vt
    
def _nat_key(s: str):
    """Natural sort: 'layer.10' will come after 'layer.2'."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]

def _to_numpy(x: Any) -> np.ndarray:
    """Convert torch.Tensor / np.ndarray / array-like to a row-major np.ndarray."""
    try:
        import torch  
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().contiguous().numpy()
    except Exception:
        pass
    return np.asarray(x)

def build_rpca_M_from_B(client_models: List[Dict[str, Any]],
                        dtype: np.dtype = np.float32) -> np.ndarray:
    if not client_models:
        raise ValueError("client_models is empty.")

    # Get common 'lora_B' keys from all clients to ensure alignment
    key_sets = []
    for p in client_models:
        key_sets.append({k for k in p.keys() if "lora_B" in k})
    common_keys = set.intersection(*key_sets)
    if not common_keys:
        raise ValueError("No 'lora_B' keys found in client_models.")

    b_key_order = sorted(common_keys, key=_nat_key)

    # Flatten and concatenate for each client in a consistent order
    cols = []
    for idx, params in enumerate(client_models):
        missing = [k for k in b_key_order if k not in params]
        if missing:
            raise KeyError(f"Client {idx} missing keys: {missing}")
        flat_parts = []
        for k in b_key_order:
            B = _to_numpy(params[k])        # Expected shape: (out, r)
            flat_parts.append(B.reshape(-1))
        cols.append(np.concatenate(flat_parts, axis=0))

    M = np.stack(cols, axis=1)
    if dtype is not None:
        M = M.astype(dtype, copy=False)
    return M

def build_rpca_M_from_A(client_models: List[Dict[str, Any]],
                        dtype: np.dtype = np.float32) -> np.ndarray:
    if not client_models:
        raise ValueError("client_models is empty.")

    # Get common 'lora_A' keys from all clients to ensure alignment
    key_sets = [{k for k in p.keys() if "lora_A" in k} for p in client_models]
    common_keys = set.intersection(*key_sets) if key_sets else set()
    if not common_keys:
        raise ValueError("No 'lora_A' keys found in client_models.")

    a_key_order = sorted(common_keys, key=_nat_key)

    # Flatten and concatenate for each client in a consistent order
    cols = []
    for idx, params in enumerate(client_models):
        missing = [k for k in a_key_order if k not in params]
        if missing:
            raise KeyError(f"Client {idx} missing keys: {missing}")
        flat_parts = []
        for k in a_key_order:
            A = _to_numpy(params[k])      # Expected shape: (r, in)
            flat_parts.append(A.reshape(-1))
        cols.append(np.concatenate(flat_parts, axis=0))

    M = np.stack(cols, axis=1)
    if dtype is not None:
        M = M.astype(dtype, copy=False)
    return M

def build_rpca_M_from_delta(client_models: List[Dict[str, Any]],
                            dtype: np.dtype = np.float32) -> np.ndarray:
    if not client_models:
        raise ValueError("client_models is empty.")

    def _base_from_key(k: str) -> str:
        # Normalize '...lora_A...' and '...lora_B...' to a common base key for pairing
        return re.sub(r"lora_[AB]\b", "lora", k)

    bases_per_client = []
    a_maps, b_maps = [], []

    # For each client, build base->A_key / base->B_key maps and collect valid bases
    for p in client_models:
        A_keys = [k for k in p.keys() if "lora_A" in k]
        B_keys = [k for k in p.keys() if "lora_B" in k]
        a_map = {_base_from_key(k): k for k in A_keys}
        b_map = {_base_from_key(k): k for k in B_keys}
        bases = set(a_map.keys()) & set(b_map.keys())
        if not bases:
            raise ValueError("Found client without aligned A/B pairs.")
        bases_per_client.append(bases)
        a_maps.append(a_map)
        b_maps.append(b_map)

    # Keep only common bases that exist in all clients
    common_bases = set.intersection(*bases_per_client)
    if not common_bases:
        raise ValueError("No common A/B bases across clients.")
    bases_order = sorted(list(common_bases), key=_nat_key)

    # For each client, compute ΔW = B @ A, flatten, and concatenate
    cols = []
    for i, p in enumerate(client_models):
        parts = []
        a_map, b_map = a_maps[i], b_maps[i]
        for base in bases_order:
            A = _to_numpy(p[a_map[base]])  # (r, in)
            B = _to_numpy(p[b_map[base]])  # (out, r)
            if B.shape[1] != A.shape[0]:
                raise ValueError(
                    f"Shape mismatch at client {i}, base {base}: "
                    f"B{B.shape} vs A{A.shape} (B.shape[1] must equal A.shape[0])"
                )
            DW = (B @ A).reshape(-1)
            parts.append(DW)
        cols.append(np.concatenate(parts, axis=0))

    M = np.stack(cols, axis=1)
    if dtype is not None:
        M = M.astype(dtype, copy=False)
    return M

def noise_by_residual_pca_B_loo(client_models, b_keys, rank_k=1):
    # Construct X of shape (d, C)
    cols = []
    for mp in client_models:
        flat = np.concatenate([_to_numpy(mp[k]).reshape(-1).astype(np.float32) for k in b_keys], 0)
        cols.append(flat)
    X = np.stack(cols, 1)                      # d x C
    d, C = X.shape
    sig = np.zeros(C, dtype=np.float64)

    for i in range(C):
        # 1) Estimate subspace using all columns except i
        Xm = np.delete(X, i, axis=1)           # d x (C-1)
        Xm = Xm - Xm.mean(axis=1, keepdims=True)   # Row-wise centering (without i)
        U, s, Vt = np.linalg.svd(Xm, full_matrices=False)
        k_eff = max(0, min(rank_k, U.shape[1]-1))
        if k_eff > 0:
            Uk = U[:, :k_eff]
            # 2) Project x_i to the residual space after centering with the same mean
            mu = Xm.mean(axis=1, keepdims=True)
            xi = (X[:, [i]] - mu)
            ri = xi - Uk @ (Uk.T @ xi)
            denom = max(d - k_eff, 1)
        else:
            mu = Xm.mean(axis=1, keepdims=True)
            ri = (X[:, [i]] - mu)
            denom = d
        sig[i] = float(np.linalg.norm(ri)**2 / denom)
    return np.sqrt(sig)  # sigma_hat_i

def noise_by_residual_pca_A_loo(client_models, a_keys, rank_k=1):
    dtype=np.float32
    cols = []
    for mp in client_models:
        flat = np.concatenate([_to_numpy(mp[k]).astype(dtype).reshape(-1) for k in a_keys], axis=0)
        cols.append(flat)
    X = np.stack(cols, axis=1).astype(dtype)  # d x C

    d, C = X.shape
    sigma = np.zeros(C, dtype=np.float64)

    for i in range(C):
        Xm = np.delete(X, i, axis=1)                 # d x (C-1)
        Xm = Xm - Xm.mean(axis=1, keepdims=True)     # Row-wise centering (without i)
        U, s, Vt = np.linalg.svd(Xm, full_matrices=False)
        k_eff = max(0, min(rank_k, U.shape[1]-1))

        mu = Xm.mean(axis=1, keepdims=True)
        xi = (X[:, [i]] - mu)                        # Center i with the same mean
        if k_eff > 0:
            Uk = U[:, :k_eff]
            ri = xi - Uk @ (Uk.T @ xi)
            denom = max(d - k_eff, 1)
        else:
            ri = xi
            denom = d

        sigma[i] = float(np.linalg.norm(ri)**2 / denom)

    return np.sqrt(sigma)  # (C,)

def RPCA_weights(CLIENT_SIGMA, client_models, num_successful_clients):
    M_B = build_rpca_M_from_B(client_models)  # M.shape = (d_B, num_successful_clients)
    rpca_B = RPCA(lam=None, mu=1.0, max_iter=500, tol=1e-6, verbose=False)
    L_rpca_B, S_rpca_B = rpca_B.fit(M_B) 
    sigma_hat_B = rpca_B.estimate_noise(axis=0)                 # ||S_:i||_2
    weights_B = rpca_B.weights_from_noise(sigma_hat_B).tolist()   # 1/(sigma_hat+eps)
    weights_B = [round(float(w), 3) for w in weights_B]
            
    print("\n=== Noise per Client (RPCA estimate vs. true if available) ===")
    sigma_true_std = [float(CLIENT_SIGMA[cid][0]) for cid in range(num_successful_clients)]
    for i in range(num_successful_clients):
        st = "N/A" if (sigma_true_std[i] is None) else f"{sigma_true_std[i]:.3f}"
        print(f"Client {i+1}: sigma_hat={sigma_hat_B[i]:.3f}, sigma_true={st}, weight={weights_B[i]:.4f}")

    print(f"Client Weights based on estimated noise by B: [{', '.join(f'{w:.3f}' for w in weights_B)}]")
            
            #estimate by only A
    M_A = build_rpca_M_from_A(client_models)
    rpca_A = RPCA(lam=None, mu=1.0, max_iter=500, tol=1e-6, verbose=False)
    L_rpca_A, S_rpca_A = rpca_A.fit(M_A)
    sigma_hat_A = rpca_A.estimate_noise(axis=0)                 # ||S_:i||_2
    weights_A = rpca_A.weights_from_noise(sigma_hat_A).tolist()   # 1/(sigma_hat+eps)
    weights_A = [round(float(w), 3) for w in weights_A]
    for i in range(num_successful_clients):
        st = "N/A" if (sigma_true_std[i] is None) else f"{sigma_true_std[i]:.3f}"
        print(f"Client {i+1}: sigma_hat={sigma_hat_A[i]:.3f}, sigma_true={st}, weight={weights_A[i]:.4f}")

    print(f"Client Weights based on estimated noise by A: [{', '.join(f'{w:.3f}' for w in weights_A)}]")
    weights = weights_B
            # weights = [0.2,0.4,0.2,0.2]
    print(f"Client Weights based on estimated noise: [{', '.join(f'{w:.3f}' for w in weights)}]")
    return sigma_true_std


def _to_numpy(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)