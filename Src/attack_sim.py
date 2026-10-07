"""
Attacks on WinFLoRA's noise-aware aggregation (server-side only, no LLM training needed).

  A1  Rescaling attack   : upload (B/s, s*A). Delta W = BA is unchanged, but the
                           B-only LOO-PCA noise estimate shrinks -> attacker gets top weight.
  A2  Seed-reuse denoise : add_noise_to_lora(seed=1234+cid) draws the same z every round,
                           so two uploads with different sigma reveal the clean adapter.
  A3  Sybil collusion    : two clients upload identical updates; each one's twin lies in
                           its leave-one-out subspace -> residual ~ 0 -> weight blows up.

Clean adapters come from either cached real artifacts (--source artifacts) or a synthetic
generator with TinyLlama's LoRA shapes (--source synthetic).

Usage:
  python attack_sim.py --source synthetic
  python attack_sim.py --source artifacts --artifact_dir noisy_tiny_agnews/artifacts
"""
import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import torch

from rpca import noise_by_residual_pca_B_loo
from utils import add_noise_to_lora

EPS = 1e-8


# ---------------------------------------------------------------------------
# Clean adapters
# ---------------------------------------------------------------------------
def make_synthetic_loras(num_clients, seed=0, n_layers=22, r=16, d_model=2048, d_kv=256):
    """PEFT-style LoRA dicts with TinyLlama q_proj/v_proj shapes.

    B = shared low-rank task signal + client-specific deviation (non-IID);
    A = per-client random init (as in the repo: torch.manual_seed(cid+1000)) + small shared drift.
    """
    rng = np.random.default_rng(seed)
    shared = {}
    for l in range(n_layers):
        for proj, d_out in (("q_proj", d_model), ("v_proj", d_kv)):
            shared[(l, proj, "B")] = rng.normal(0, 4e-3, size=(d_out, r))
            shared[(l, proj, "A")] = rng.normal(0, 2e-3, size=(r, d_model))
    clients = []
    for cid in range(num_clients):
        crng = np.random.default_rng(1000 + cid)
        lp = {}
        for l in range(n_layers):
            for proj, d_out in (("q_proj", d_model), ("v_proj", d_kv)):
                pre = f"base_model.model.model.layers.{l}.self_attn.{proj}"
                A = crng.uniform(-1 / np.sqrt(d_model), 1 / np.sqrt(d_model), size=(r, d_model)) \
                    + shared[(l, proj, "A")]
                B = shared[(l, proj, "B")] + crng.normal(0, 2e-3, size=(d_out, r))
                lp[f"{pre}.lora_A.default.weight"] = torch.tensor(A, dtype=torch.float32)
                lp[f"{pre}.lora_B.default.weight"] = torch.tensor(B, dtype=torch.float32)
        clients.append(lp)
    return clients


def load_artifact_loras(artifact_dir, num_clients):
    clients = []
    for cid in range(num_clients):
        d = Path(artifact_dir)
        cand = sorted(d.glob(f"client_{cid}_*"))
        if not cand:
            raise FileNotFoundError(f"no artifact folder for client {cid} in {d}")
        lp = torch.load(cand[0] / "lora.pt", map_location="cpu")
        clients.append({k: v.float() for k, v in lp.items()})
    return clients


# ---------------------------------------------------------------------------
# Server side (mirrors tiny_pca_greedy_agnews.py, weight_mode='pca')
# ---------------------------------------------------------------------------
def nat_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def loo_pca_paper(client_models, b_keys, rank_k):
    """Algorithm 1 as written in the paper: x_i is centred with the LOO mean before projecting.
    (The repo's noise_by_residual_pca_B_loo recomputes mu from the already-centred matrix,
    so mu ~ 0 and x_i is effectively not centred.)"""
    X = np.stack([np.concatenate([m[k].numpy().reshape(-1).astype(np.float64) for k in b_keys])
                  for m in client_models], 1)
    d, C = X.shape
    sig = np.zeros(C)
    for i in range(C):
        Xm = np.delete(X, i, axis=1)
        mu = Xm.mean(axis=1, keepdims=True)
        U, _, _ = np.linalg.svd(Xm - mu, full_matrices=False)
        K = max(0, min(rank_k, U.shape[1] - 1))
        xi = X[:, [i]] - mu
        ri = xi - U[:, :K] @ (U[:, :K].T @ xi) if K > 0 else xi
        sig[i] = np.linalg.norm(ri) ** 2 / max(d - K, 1)
    return np.sqrt(sig)


ESTIMATOR = "repo"   # set from --estimator


def server_weights(client_models, rank_k=1, tau=0.8):
    b_keys = sorted(set.intersection(*[{k for k in m if "lora_B" in k} for m in client_models]))
    if ESTIMATOR == "paper":
        sigma_hat = loo_pca_paper(client_models, b_keys, rank_k)
    else:
        sigma_hat = noise_by_residual_pca_B_loo(client_models, b_keys, rank_k=rank_k)
    w = (1.0 / (sigma_hat + EPS)) ** tau
    return sigma_hat, w / w.sum()


def ab_pairs(lp):
    A_keys = sorted([k for k in lp if "lora_A" in k], key=nat_key)
    B_keys = sorted([k for k in lp if "lora_B" in k], key=nat_key)
    return list(zip(A_keys, B_keys))


def aggregate_error(noisy_models, clean_models, weights, device):
    """|| sum_i w_i B_i A_i (noisy) - mean_i B_i A_i (clean) ||_F / || mean clean ||_F"""
    num, den = 0.0, 0.0
    w = torch.tensor(weights, dtype=torch.float32, device=device)
    for Ak, Bk in ab_pairs(clean_models[0]):
        Bn = torch.cat([m[Bk].to(device) * w[i].sqrt() for i, m in enumerate(noisy_models)], 1)
        An = torch.cat([m[Ak].to(device) * w[i].sqrt() for i, m in enumerate(noisy_models)], 0)
        Bc = torch.cat([m[Bk].to(device) for m in clean_models], 1) / np.sqrt(len(clean_models))
        Ac = torch.cat([m[Ak].to(device) for m in clean_models], 0) / np.sqrt(len(clean_models))
        target = Bc @ Ac
        num += ((Bn @ An - target) ** 2).sum().item()
        den += (target ** 2).sum().item()
    return float(np.sqrt(num / den))


def deltaW_noise_ratio(noisy, clean, device):
    num, den = 0.0, 0.0
    for Ak, Bk in ab_pairs(clean):
        Wc = clean[Bk].to(device) @ clean[Ak].to(device)
        Wn = noisy[Bk].to(device) @ noisy[Ak].to(device)
        num += ((Wn - Wc) ** 2).sum().item()
        den += (Wc ** 2).sum().item()
    return float(np.sqrt(num / den))


def honest_sigmas(n, sigma_max, seed):
    # same initial sigma draw as the training script
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(sigma_max / 2, sigma_max / 6, size=n), 0.0, sigma_max)


def noisy_round(clean_models, sigmas):
    return [add_noise_to_lora(lp, float(s), float(s), seed=1234 + cid)
            for cid, (lp, s) in enumerate(zip(clean_models, sigmas))]


def rescale(lp, s):
    return {k: (v / s if "lora_B" in k else v * s if "lora_A" in k else v) for k, v in lp.items()}


# ---------------------------------------------------------------------------
# Attacks
# ---------------------------------------------------------------------------
def attack1_rescaling(clean, sigmas, sigma_max, scales, device, rank_k):
    att = 0
    out = []
    for s in scales:
        sig = sigmas.copy()
        sig[att] = sigma_max                       # attacker uses the MOST noise allowed
        noisy = noisy_round(clean, sig)
        noisy[att] = rescale(noisy[att], s)        # ...then rescales B/s, A*s
        sh, w = server_weights(noisy, rank_k=rank_k)
        out.append({
            "scale": s,
            "attacker_sigma_true": float(sig[att]),
            "attacker_sigma_hat": float(sh[att]),
            "attacker_weight": float(w[att]),
            "attacker_rank": int((w > w[att]).sum()) + 1,
            "uniform_weight": 1.0 / len(clean),
            "attacker_deltaW_noise_ratio": deltaW_noise_ratio(noisy[att], clean[att], device),
            "aggregate_rel_error": aggregate_error(noisy, clean, w, device),
        })
        print(f"[A1] s={s:>6}: sigma_hat={sh[att]:.4f}  w_att={w[att]:.3f} (rank {out[-1]['attacker_rank']})"
              f"  ||dW noise||/||dW||={out[-1]['attacker_deltaW_noise_ratio']:.2f}"
              f"  agg_err={out[-1]['aggregate_rel_error']:.3f}")
    # reference: an honest client with zero noise
    sig = sigmas.copy(); sig[att] = 0.0
    sh, w = server_weights(noisy_round(clean, sig), rank_k=rank_k)
    ref = {"honest_clean_attacker_weight": float(w[att]), "honest_clean_sigma_hat": float(sh[att])}
    print(f"[A1] reference: honest sigma=0 client gets w={w[att]:.3f}")
    return {"sweep": out, "reference": ref}


def flat(lp, keys):
    return torch.cat([lp[k].reshape(-1).double() for k in keys])


def attack2_seed_reuse(clean, sigma_pairs):
    cid = 0
    keys = sorted([k for k in clean[cid] if "lora_" in k], key=nat_key)
    c = flat(clean[cid], keys)
    res = []
    for (s1, s2) in sigma_pairs:
        x1 = flat(add_noise_to_lora(clean[cid], s1, s1, seed=1234 + cid), keys)  # round t
        x2 = flat(add_noise_to_lora(clean[cid], s2, s2, seed=1234 + cid), keys)  # round t'
        # (a) server knows/estimates sigmas: exact linear solve
        c_known = (s2 * x1 - s1 * x2) / (s2 - s1)
        # (b) sigma-free: noise direction u = (x1-x2)/||x1-x2||; remove it from x1
        u = (x1 - x2) / (x1 - x2).norm()
        c_free = x1 - (x1 @ u) * u
        # control: fresh noise every round (what a correct implementation would do)
        y1 = flat(add_noise_to_lora(clean[cid], s1, s1, seed=99991), keys)
        y2 = flat(add_noise_to_lora(clean[cid], s2, s2, seed=77777), keys)
        uc = (y1 - y2) / (y1 - y2).norm()
        c_ctrl = y1 - (y1 @ uc) * uc
        rel = lambda v: float((v - c).norm() / c.norm())
        r = {"sigma_t": s1, "sigma_t2": s2,
             "single_upload_err": rel(x1),
             "recovered_err_known_sigma": rel(c_known),
             "recovered_err_sigma_free": rel(c_free),
             "control_fresh_noise_err": rel(c_ctrl)}
        res.append(r)
        print(f"[A2] sigma {s1}->{s2}: upload err={r['single_upload_err']:.3f}  "
              f"recovered (known sigma)={r['recovered_err_known_sigma']:.2e}  "
              f"(sigma-free)={r['recovered_err_sigma_free']:.2e}  "
              f"control w/ fresh noise={r['control_fresh_noise_err']:.3f}")
    return res


def attack3_sybil(clean, sigmas, att_sigmas, rank_ks):
    a, b = 0, 1
    out = []
    for rk in rank_ks:
        for s_att in att_sigmas:
            sig = sigmas.copy()
            sig[a] = sig[b] = s_att
            noisy = noisy_round(clean, sig)
            noisy[b] = {k: v.clone() for k, v in noisy[a].items()}   # Sybil: identical upload
            sh, w = server_weights(noisy, rank_k=rk)
            # baseline: same two clients, honest independent uploads at the same sigma
            noisy_h = noisy_round(clean, sig)
            _, w_h = server_weights(noisy_h, rank_k=rk)
            r = {"rank_k": rk, "attacker_sigma": s_att,
                 "colluders_weight": float(w[a] + w[b]),
                 "honest_pair_weight": float(w_h[a] + w_h[b]),
                 "fair_share": 2.0 / len(clean),
                 "sigma_hat_colluder": float(sh[a])}
            out.append(r)
            print(f"[A3] K={rk} sigma_att={s_att:.3f}: colluders' weight={r['colluders_weight']:.3f}"
                  f"  (same pair honest: {r['honest_pair_weight']:.3f}, fair 2/N={r['fair_share']:.2f})")
    return out


# ---------------------------------------------------------------------------
def plots(results, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    sw = results["A1"]["sweep"]
    ax[0].semilogx([r["scale"] for r in sw], [r["attacker_weight"] for r in sw], "o-", label="attacker (σ=σmax, rescaled)")
    ax[0].axhline(results["A1"]["reference"]["honest_clean_attacker_weight"], ls="--", c="g", label="honest σ=0 client")
    ax[0].axhline(sw[0]["uniform_weight"], ls=":", c="gray", label="uniform 1/N")
    ax[0].set_xlabel("rescale factor s  (upload B/s, sA)"); ax[0].set_ylabel("aggregation weight")
    ax[0].set_title("A1: rescaling attack"); ax[0].legend(fontsize=8)
    for rk in sorted({r["rank_k"] for r in results["A3"]}):
        rows = [r for r in results["A3"] if r["rank_k"] == rk]
        ax[1].plot([r["attacker_sigma"] for r in rows], [r["colluders_weight"] for r in rows], "o-", label=f"Sybil pair, K={rk}")
        ax[1].plot([r["attacker_sigma"] for r in rows], [r["honest_pair_weight"] for r in rows], "x--", label=f"honest pair, K={rk}")
    ax[1].axhline(results["A3"][0]["fair_share"], ls=":", c="gray", label="fair 2/N")
    ax[1].set_xlabel("colluders' noise σ"); ax[1].set_ylabel("combined weight")
    ax[1].set_title("A3: Sybil collusion"); ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "attacks.png", dpi=150)
    print(f"plot -> {out_dir / 'attacks.png'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["synthetic", "artifacts"], default="synthetic")
    p.add_argument("--artifact_dir", default="noisy_tiny_agnews/artifacts")
    p.add_argument("--num_clients", type=int, default=10)
    p.add_argument("--sigma_max", type=float, default=0.1)   # repo default
    p.add_argument("--noise_seed", type=int, default=2025)
    p.add_argument("--estimator", choices=["repo", "paper"], default="repo",
                   help="repo: rpca.noise_by_residual_pca_B_loo; paper: Algorithm 1 with correct centring")
    p.add_argument("--rank_k", type=int, default=None, help="default: 1 for repo, N-2 for paper")
    p.add_argument("--out", default="attack_results")
    args = p.parse_args()
    global ESTIMATOR
    ESTIMATOR = args.estimator
    if args.rank_k is None:
        args.rank_k = 1 if args.estimator == "repo" else args.num_clients - 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out) / f"{args.source}_{args.estimator}"
    out_dir.mkdir(parents=True, exist_ok=True)

    clean = (make_synthetic_loras(args.num_clients) if args.source == "synthetic"
             else load_artifact_loras(args.artifact_dir, args.num_clients))
    sigmas = honest_sigmas(args.num_clients, args.sigma_max, args.noise_seed)
    print(f"device={device}  source={args.source}  N={args.num_clients}  honest sigmas={np.round(sigmas, 3).tolist()}")

    grid = [0.01 * args.sigma_max, 0.1 * args.sigma_max, 0.5 * args.sigma_max, args.sigma_max]  # INA action set
    results = {
        "config": vars(args),
        "honest_sigmas": sigmas.tolist(),
        "A1": attack1_rescaling(clean, sigmas, args.sigma_max, [1, 2, 4, 8, 16, 32, 64, 128], device, args.rank_k),
        "A2": attack2_seed_reuse(clean, [(grid[2], grid[1]), (grid[3], grid[2]), (grid[3], grid[0])]),
        "A3": attack3_sybil(clean, sigmas, [0.0] + grid[1:] + [2 * args.sigma_max],
                            sorted({args.rank_k, args.num_clients - 2})),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"results -> {out_dir / 'results.json'}")
    plots(results, out_dir)


if __name__ == "__main__":
    main()
