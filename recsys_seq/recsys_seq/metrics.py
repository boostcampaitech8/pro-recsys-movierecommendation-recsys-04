from __future__ import annotations
import numpy as np

def recall_at_k(recommended: list[int], ground_truth: set[int], k: int) -> float:
    if not ground_truth:
        return 0.0
    topk = recommended[:k]
    hit = sum(1 for x in topk if x in ground_truth)
    return hit / float(len(ground_truth))

def ndcg_at_k(recommended: list[int], ground_truth: set[int], k: int) -> float:
    """Binary relevance NDCG."""
    topk = recommended[:k]
    dcg = 0.0
    for rank, item in enumerate(topk, start=1):
        if item in ground_truth:
            dcg += 1.0 / np.log2(rank + 1)
    ideal_hits = min(len(ground_truth), k)
    idcg = sum(1.0 / np.log2(r + 1) for r in range(1, ideal_hits + 1))
    return (dcg / idcg) if idcg > 0 else 0.0
