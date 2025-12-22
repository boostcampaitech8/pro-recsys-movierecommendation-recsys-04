from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize


class ItemKNN:
    """
    Memory-based Item-item CF (implicit)
    - similarity: cosine on item vectors
    - score(u, j) = sum_{i in seen(u)} sim(j, i)
    """
    def __init__(self, topk_sim: int = 200):
        self.topk_sim = topk_sim
        self.X: csr_matrix | None = None          # user-item
        self.item_sim: csr_matrix | None = None   # item-item (sparse)

    def fit(self, X: csr_matrix):
        self.X = X.tocsr()

        # cosine similarity = normalize then dot
        Xt = self.X.T.tocsr()
        Xt_norm = normalize(Xt, norm="l2", axis=1, copy=True)
        sim = Xt_norm @ Xt_norm.T  # (n_items, n_items) sparse

        # self-sim 제거
        sim.setdiag(0.0)
        sim.eliminate_zeros()

        # topk_sim으로 sparsify
        self.item_sim = _keep_topk_per_row(sim.tocsr(), self.topk_sim)

    def recommend_with_scores(
        self, user: int, k: int = 10, seen: set[int] | None = None
    ) -> tuple[list[int], list[float]]:
        assert self.X is not None and self.item_sim is not None

        user_row = self.X[user]  # 1 x n_items
        seen_items = set(user_row.indices.tolist()) if seen is None else seen

        # scores = user_row * item_sim^T  (1 x n_items)
        scores = (user_row @ self.item_sim.T).toarray().ravel()

        # seen 제거
        if seen_items:
            scores[list(seen_items)] = -np.inf

        k_eff = min(k, len(scores))
        top_items = np.argpartition(-scores, kth=k_eff - 1)[:k_eff]
        top_items = top_items[np.argsort(-scores[top_items])]

        top_scores = scores[top_items].astype(np.float32)
        return top_items.tolist(), top_scores.tolist()

    def recommend(self, user: int, k: int = 10, seen: set[int] | None = None) -> list[int]:
        items, _ = self.recommend_with_scores(user=user, k=k, seen=seen)
        return items


class UserKNN:
    """
    Memory-based User-user CF (implicit)
    - similarity: cosine on user vectors
    - score(u, j) = sum_{v in neighbors(u)} sim(u, v) * X[v, j]
    """
    def __init__(self, topk_sim: int = 200):
        self.topk_sim = topk_sim
        self.X: csr_matrix | None = None          # user-item
        self.user_sim: csr_matrix | None = None   # user-user (sparse)

    def fit(self, X: csr_matrix):
        self.X = X.tocsr()

        X_norm = normalize(self.X, norm="l2", axis=1, copy=True)
        sim = X_norm @ X_norm.T   # (n_users, n_users) sparse
        sim.setdiag(0.0)
        sim.eliminate_zeros()

        self.user_sim = _keep_topk_per_row(sim.tocsr(), self.topk_sim)

    def recommend_with_scores(
        self, user: int, k: int = 10, seen: set[int] | None = None
    ) -> tuple[list[int], list[float]]:
        assert self.X is not None and self.user_sim is not None

        seen_items = set(self.X[user].indices.tolist()) if seen is None else seen

        # scores = user_sim[user] * X  (1 x n_items)
        scores = (self.user_sim[user] @ self.X).toarray().ravel()

        if seen_items:
            scores[list(seen_items)] = -np.inf

        k_eff = min(k, len(scores))
        top_items = np.argpartition(-scores, kth=k_eff - 1)[:k_eff]
        top_items = top_items[np.argsort(-scores[top_items])]

        top_scores = scores[top_items].astype(np.float32)
        return top_items.tolist(), top_scores.tolist()

    def recommend(self, user: int, k: int = 10, seen: set[int] | None = None) -> list[int]:
        items, _ = self.recommend_with_scores(user=user, k=k, seen=seen)
        return items


class ImplicitALS:
    """
    Implicit MF via ALS (Hu, Koren, Volinsky 스타일)
    - X: implicit user-item (0/1)
    - optimize: sum_u,i c_ui (p_ui - x_u^T y_i)^2 + reg (||x_u||^2 + ||y_i||^2)
      where p_ui = 1 if interacted else 0
            c_ui = 1 + alpha * r_ui  (r_ui=1)
    """
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

        # 초기화
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
            Yu = Y[items]  # (nnz, f)

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

    def recommend_with_scores(
        self, user: int, k: int = 10, seen: set[int] | None = None
    ) -> tuple[list[int], list[float]]:
        assert self.X is not None and self.user_factors is not None and self.item_factors is not None

        seen_items = set(self.X[user].indices.tolist()) if seen is None else seen
        scores = (self.user_factors[user] @ self.item_factors.T).astype(np.float32)  # (n_items,)

        if seen_items:
            scores[list(seen_items)] = -np.inf

        k_eff = min(k, len(scores))
        top_items = np.argpartition(-scores, kth=k_eff - 1)[:k_eff]
        top_items = top_items[np.argsort(-scores[top_items])]
        top_scores = scores[top_items]

        return top_items.tolist(), top_scores.tolist()

    def recommend(self, user: int, k: int = 10, seen: set[int] | None = None) -> list[int]:
        items, _ = self.recommend_with_scores(user=user, k=k, seen=seen)
        return items


def _keep_topk_per_row(M: csr_matrix, k: int) -> csr_matrix:
    """
    sparse matrix row-wise top-k 유지 (나머지 0)
    """
    M = M.tocsr()
    new_data = []
    new_indices = []
    new_indptr = [0]

    for r in range(M.shape[0]):
        start, end = M.indptr[r], M.indptr[r + 1]
        inds = M.indices[start:end]
        vals = M.data[start:end]

        if len(vals) > k:
            topk_idx = np.argpartition(-vals, kth=k - 1)[:k]
            inds = inds[topk_idx]
            vals = vals[topk_idx]

            order = np.argsort(-vals)
            inds = inds[order]
            vals = vals[order]

        new_data.extend(vals.tolist())
        new_indices.extend(inds.tolist())
        new_indptr.append(len(new_data))

    return csr_matrix(
        (
            np.array(new_data, dtype=np.float32),
            np.array(new_indices, dtype=np.int32),
            np.array(new_indptr, dtype=np.int32),
        ),
        shape=M.shape,
    )
