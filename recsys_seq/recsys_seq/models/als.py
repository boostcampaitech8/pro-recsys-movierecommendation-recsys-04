from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix

class ImplicitALS:
    """Implicit MF via ALS (Hu, Koren, Volinsky 스타일)."""
    def __init__(self, factors: int = 64, reg: float = 0.01, alpha: float = 40.0, iters: int = 10):
        self.factors = factors
        self.reg = reg
        self.alpha = alpha
        self.iters = iters
        self.X: csr_matrix | None = None
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None

    def fit(self, X: csr_matrix, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.X = X.tocsr()
        n_users, n_items = self.X.shape
        self.user_factors = 0.01 * rng.standard_normal((n_users, self.factors)).astype(np.float32)
        self.item_factors = 0.01 * rng.standard_normal((n_items, self.factors)).astype(np.float32)

        for _ in range(self.iters):
            self._als_step_users()
            self._als_step_items()

    def _als_step_users(self):
        assert self.X is not None and self.user_factors is not None and self.item_factors is not None
        Y = self.item_factors
        YtY = Y.T @ Y
        I = np.eye(self.factors, dtype=np.float32)
        X = self.X
        for u in range(X.shape[0]):
            start, end = X.indptr[u], X.indptr[u + 1]
            items = X.indices[start:end]
            if len(items) == 0:
                continue
            Cu = 1.0 + self.alpha
            Yu = Y[items]
            A = YtY + (Cu - 1.0) * (Yu.T @ Yu) + self.reg * I
            b = Cu * Yu.sum(axis=0)
            self.user_factors[u] = np.linalg.solve(A, b).astype(np.float32)

    def _als_step_items(self):
        assert self.X is not None and self.user_factors is not None and self.item_factors is not None
        X = self.X.tocsc()
        U = self.user_factors
        UtU = U.T @ U
        I = np.eye(self.factors, dtype=np.float32)
        for i in range(X.shape[1]):
            start, end = X.indptr[i], X.indptr[i + 1]
            users = X.indices[start:end]
            if len(users) == 0:
                continue
            Ci = 1.0 + self.alpha
            Ui = U[users]
            A = UtU + (Ci - 1.0) * (Ui.T @ Ui) + self.reg * I
            b = Ci * Ui.sum(axis=0)
            self.item_factors[i] = np.linalg.solve(A, b).astype(np.float32)

    def recommend_with_scores(self, user: int, k: int = 10, seen: set[int] | None = None):
        assert self.X is not None and self.user_factors is not None and self.item_factors is not None
        seen_items = set(self.X[user].indices.tolist()) if seen is None else seen
        scores = (self.user_factors[user] @ self.item_factors.T).astype(np.float32)
        if seen_items:
            scores[list(seen_items)] = -np.inf
        k_eff = min(k, len(scores))
        top_items = np.argpartition(-scores, kth=k_eff - 1)[:k_eff]
        top_items = top_items[np.argsort(-scores[top_items])]
        return top_items.tolist(), scores[top_items].tolist()

    def recommend(self, user: int, k: int = 10, seen: set[int] | None = None) -> list[int]:
        items, _ = self.recommend_with_scores(user, k=k, seen=seen)
        return items
