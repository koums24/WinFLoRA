import math
import random
from typing import List, Dict, Tuple
import numpy as np
from utils import compute_utilities


class GreedyNoiseSelector:

    def __init__(
        self,
        sigma_max: float,
        alpha: List[float],      # 每个 client 的 alpha（长度 = num_clients）
        beta:  List[float],      # 每个 client 的 beta  （长度 = num_clients）
        sigma_init: List[float], # 外部传入的首轮 sigma_t
        c: float = 0.05,          # ε_t = min(1, c / sqrt(t))
        use_ema: bool = True,    # 非平稳环境推荐 True
        rho: float = 0.1         # EMA 系数（0.05~0.2 常用）
    ):
        assert sigma_max > 0, "sigma_max must be > 0"

        # 候选噪声集合（离散臂）
        # self.S_base = [0.01, 0.2, 0.4, 0.6, 0.8, 1.0]
        self.S_base = [0.01, 0.1, 0.5, 1.0]

        self.sigma_max = float(sigma_max)
        self.S = [s * self.sigma_max for s in self.S_base]

        # 固定 alpha/beta（逐个归一化，确保 α+β=1 且非负）
        alpha = np.asarray(alpha, dtype=float).reshape(-1)
        beta  = np.asarray(beta,  dtype=float).reshape(-1)
        assert len(alpha) == len(beta), "alpha and beta must have the same length"
        self.num_clients = len(alpha)
        self.alpha: List[float] = alpha
        self.beta:  List[float] = beta

        # 首轮 sigma：外部传入并存成属性（不强制贴齐网格；计账时会贴齐）
        sigma_init = np.asarray(sigma_init, dtype=float).reshape(-1)
        if len(sigma_init) != self.num_clients:
            raise ValueError("sigma_init length must equal number of clients")
        self.sigma = sigma_init.tolist()

        # 经验均值与选择次数
        self.mu: List[Dict[float, float]] = [{s: 0.0 for s in self.S} for _ in range(self.num_clients)]
        self.N:  List[Dict[float, int]]   = [{s: 0   for s in self.S} for _ in range(self.num_clients)]

        # 策略超参数
        self.c = float(c)
        self.use_ema = bool(use_ema)
        self.rho = float(rho)
        
    def select_ucb_arm(self, mu, N, t, arms, c=1.0):
            # 先试未尝试过的臂
            untried = [s for s in arms if N[s] == 0]
            if untried:
                return random.choice(untried)

            # 避免 log(0)
            t = max(2, int(t))
            best_s, best_idx = None, float("-inf")
            for s in arms:
                idx = mu[s] + c * math.sqrt(math.log(t) / N[s])
                if idx > best_idx:
                    best_s, best_idx = s, idx
            return best_s
        
    # ----------------- 核心接口 -----------------
    def update_and_select_next(
        self,
        t: int,
        acc_t                 # list 或 1D np.array（建议已在 [0,1]）
    ) -> List[float]:
    
        acc_t = np.asarray(acc_t, dtype=float).reshape(-1)
        assert len(acc_t) == self.num_clients, "acc_t length mismatch"

        eps = min(1.0, self.c / max(1.0, math.sqrt(max(1, t))))
        sigma_next = np.zeros(self.num_clients, dtype=float)

        for i in range(self.num_clients):
            # 1) 贴齐本轮使用的 σ 到候选网格（计账用）
            s_prev = self._snap_to_grid(float(self.sigma[i]))

            # 2) 当期效用：U_obs = α * G + β * P
            U_obs, norm_utility = compute_utilities(self.alpha[i], self.beta[i], acc_t[i], s_prev, self.sigma_max)
            # G_it = self._clip01(float(acc_t[i]))      # 若未归一化请先处理
            # P_it = s_prev / self.sigma_max
            # U_obs = self.alpha[i] * G_it + self.beta[i] * P_it

            # 3) 更新经验均值
            if self.use_ema:
                self.mu[i][s_prev] = (1.0 - self.rho) * self.mu[i][s_prev] + self.rho * U_obs
                self.N[i][s_prev] += 1
            else:
                self.N[i][s_prev] += 1
                n = self.N[i][s_prev]
                self.mu[i][s_prev] += (U_obs - self.mu[i][s_prev]) / n
            
            # 4）UCB
            # t_i: 该 client 的累计尝试次数（或全局轮数），至少从 1 开始
            t_i = max(1, sum(self.N[i].values()))
            # c = self.c / (self.alpha[i]/self.beta[i])
            c = self.c
            s_next = self.select_ucb_arm(self.mu[i], self.N[i], t_i, self.S, c=c/5)
            
            
            sigma_next[i] = s_next

        # 写回属性，并返回
        self.sigma = sigma_next.tolist()
        return self.sigma

    def get_alphas_betas(self) -> List[Tuple[float, float]]:
        return list(zip(self.alpha, self.beta))

    def get_sigma(self) -> List[float]:
        return list(self.sigma)

    def set_sigma(self, sigma_new: List[float]) -> None:
        sigma_new = np.asarray(sigma_new, dtype=float).reshape(-1)
        if len(sigma_new) != self.num_clients:
            raise ValueError("sigma_new length must equal number of clients")
        self.sigma = sigma_new.tolist()

    
    # ----------------- 工具函数 -----------------

    def _snap_to_grid(self, sigma: float) -> float:
        best_s, best_d = self.S[0], float("inf")
        for s in self.S:
            d = abs(s - sigma)
            if d < best_d:
                best_s, best_d = s, d
        return best_s

    @staticmethod
    def _clip01(x: float) -> float:
        return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)