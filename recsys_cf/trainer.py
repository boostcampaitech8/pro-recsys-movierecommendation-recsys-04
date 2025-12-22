from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
from scipy.sparse import csr_matrix

from metrics import recall_at_k, ndcg_at_k


@dataclass
class EvalResult:
    recall: float
    ndcg: float


class Trainer:
    """
    - model.fit(X_train)
    - valid: user별 마지막 item 1개를 정답으로 평가
    """
    def __init__(self, k_eval: int = 10):
        self.k_eval = k_eval

    def fit(self, model, X_train: csr_matrix, seed: int = 42):
        # 모델별 시그니처 차이 처리
        if hasattr(model, "fit"):
            # ImplicitALS는 seed 받도록 해둠
            try:
                model.fit(X_train, seed=seed)
            except TypeError:
                model.fit(X_train)
        return model

    def evaluate_last_item(self, model, X_train: csr_matrix, valid_pairs: np.ndarray) -> EvalResult:
        """
        valid_pairs: shape (n_valid, 2) with (u, true_item)
        """
        recalls = []
        ndcgs = []

        for u, true_i in valid_pairs:
            seen = set(X_train[u].indices.tolist())
            recs = model.recommend(int(u), k=self.k_eval, seen=seen)
            gt = {int(true_i)}

            recalls.append(recall_at_k(recs, gt, self.k_eval))
            ndcgs.append(ndcg_at_k(recs, gt, self.k_eval))

        return EvalResult(float(np.mean(recalls)), float(np.mean(ndcgs)))
