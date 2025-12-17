import numpy as np


def recall_at_k(actual_list, pred_list, k: int = 10):
    """
    actual_list: list of lists (each user's ground truth items)
    pred_list:   list of lists (each user's recommended items)
    """
    recalls = []
    for actual, pred in zip(actual_list, pred_list):
        actual_set = set(actual)
        pred_k = pred[:k]
        hit = len(actual_set.intersection(pred_k))
        denom = min(k, len(actual_set)) if len(actual_set) > 0 else 1
        recalls.append(hit / denom)
    return float(np.mean(recalls))
