from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import normalize

from .als import ImplicitALS
from ..data_utils import SideInfo

def build_item_content_matrix(sideinfo: SideInfo) -> csr_matrix:
    """item -> (genres/directors/writers) multi-hot sparse matrix."""
    n_items = len(sideinfo.item_genres)
    # feature offsets
    off_g = 0
    off_d = sideinfo.n_genres
    off_w = sideinfo.n_genres + sideinfo.n_directors
    n_feat = sideinfo.n_genres + sideinfo.n_directors + sideinfo.n_writers

    rows = []
    cols = []
    data = []

    def add_features(item_idx: int, feats: List[int], offset: int):
        for f in feats:
            if f <= 0:
                continue
            rows.append(item_idx)
            cols.append(offset + f)
            data.append(1.0)

    for i in range(n_items):
        add_features(i, sideinfo.item_genres[i], off_g)
        add_features(i, sideinfo.item_directors[i], off_d)
        add_features(i, sideinfo.item_writers[i], off_w)

    M = csr_matrix((np.array(data, dtype=np.float32), (np.array(rows), np.array(cols))), shape=(n_items, n_feat))
    M.sum_duplicates()
    return normalize(M, norm="l2", axis=1)

@dataclass
class HybridConfig:
    # ALS
    factors: int = 64
    reg: float = 0.01
    alpha: float = 40.0
    iters: int = 10
    # hybrid
    cand_k: int = 200      # ALS 후보 수
    w_als: float = 0.7     # ALS score weight
    w_content: float = 0.3 # content cosine weight

class HybridALSContent:
    """Two-stage Hybrid:
    1) ALS로 candidate 생성
    2) content cosine으로 re-rank
    """
    def __init__(self, sideinfo: SideInfo, cfg: HybridConfig):
        self.sideinfo = sideinfo
        self.cfg = cfg
        self.als = ImplicitALS(factors=cfg.factors, reg=cfg.reg, alpha=cfg.alpha, iters=cfg.iters)
        self.item_content: csr_matrix | None = None
        self.X_train: csr_matrix | None = None

    def fit(self, X: csr_matrix, seed: int = 42):
        self.X_train = X.tocsr()
        self.als.fit(self.X_train, seed=seed)
        self.item_content = build_item_content_matrix(self.sideinfo)
        return self

    def _user_profile(self, user: int) -> csr_matrix:
        assert self.X_train is not None and self.item_content is not None
        seen_items = self.X_train[user].indices
        if len(seen_items) == 0:
            # zero vector
            return csr_matrix((1, self.item_content.shape[1]), dtype=np.float32)
        prof = self.item_content[seen_items].sum(axis=0)
        prof = csr_matrix(prof)
        return normalize(prof, norm="l2", axis=1)

    def recommend_with_scores(self, user: int, k: int = 10, seen: Optional[set[int]] = None):
        assert self.item_content is not None and self.X_train is not None
        if seen is None:
            seen = set(self.X_train[user].indices.tolist())

        cand_items, cand_scores = self.als.recommend_with_scores(user, k=self.cfg.cand_k, seen=seen)
        if len(cand_items) == 0:
            return [], []

        # content similarity
        user_prof = self._user_profile(user)  # 1 x F
        C = self.item_content[cand_items]     # K x F
        content_sim = (C @ user_prof.T).toarray().ravel().astype(np.float32)  # cosine since normalized

        als_scores = np.array(cand_scores, dtype=np.float32)

        # normalize als scores to [0,1] robustly
        finite = np.isfinite(als_scores)
        if finite.any():
            mn, mx = als_scores[finite].min(), als_scores[finite].max()
            if mx > mn:
                als_norm = (als_scores - mn) / (mx - mn)
            else:
                als_norm = np.zeros_like(als_scores)
        else:
            als_norm = np.zeros_like(als_scores)

        final = self.cfg.w_als * als_norm + self.cfg.w_content * content_sim

        k_eff = min(k, len(final))
        top = np.argpartition(-final, kth=k_eff - 1)[:k_eff]
        top = top[np.argsort(-final[top])]
        return [cand_items[i] for i in top], final[top].astype(np.float32).tolist()

    def recommend(self, user: int, k: int = 10, seen: Optional[set[int]] = None) -> list[int]:
        items, _ = self.recommend_with_scores(user, k=k, seen=seen)
        return items
