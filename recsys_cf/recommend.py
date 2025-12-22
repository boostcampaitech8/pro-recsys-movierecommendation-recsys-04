from __future__ import annotations

from typing import Dict, List, Tuple
from scipy.sparse import csr_matrix


def recommend_topk_for_user(model, X_train: csr_matrix, user: int, k: int = 10) -> list[int]:
    seen = set(X_train[user].indices.tolist())
    return model.recommend(user, k=k, seen=seen)


def recommend_all_users(model, X_train: csr_matrix, users: list[int], k: int = 10) -> Dict[int, list[int]]:
    out = {}
    for u in users:
        out[u] = recommend_topk_for_user(model, X_train, u, k)
    return out
