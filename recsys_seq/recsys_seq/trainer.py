from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix

from .metrics import recall_at_k, ndcg_at_k

@dataclass
class EvalResult:
    recall: float
    ndcg: float

class Trainer:
    """간단 Trainer: fit + last-item 평가.
    - CF/MF: model.fit(X_train)
    - SASRec: model.fit_sequences(train_df, n_users, n_items, sideinfo)
    """
    def __init__(self, k_eval: int = 10):
        self.k_eval = k_eval

    def fit(self, model, X_train: csr_matrix, seed: int = 42, **kwargs):
        if hasattr(model, "fit"):
            try:
                model.fit(X_train, seed=seed, **kwargs)
            except TypeError:
                model.fit(X_train, **kwargs)
        elif hasattr(model, "fit_sequences"):
            model.fit_sequences(**kwargs)
        else:
            raise ValueError("Model has neither fit nor fit_sequences.")
        return model

    def evaluate_last_item(self, model, X_train: csr_matrix, valid_pairs: np.ndarray) -> EvalResult:
        recalls, ndcgs = [], []
        for u, true_i in valid_pairs:
            seen = set(X_train[int(u)].indices.tolist())
            recs = model.recommend(int(u), k=self.k_eval, seen=seen)
            gt = {int(true_i)}
            recalls.append(recall_at_k(recs, gt, self.k_eval))
            ndcgs.append(ndcg_at_k(recs, gt, self.k_eval))
        return EvalResult(float(np.mean(recalls)), float(np.mean(ndcgs)))
