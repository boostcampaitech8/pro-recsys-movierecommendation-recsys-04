import numpy as np


def recall_at_k(actual_list, pred_list, k=10):
    recalls = []

    for actual, pred in zip(actual_list, pred_list):
        if len(actual) == 0:
            continue

        actual_set = set(actual)
        pred_k = pred[:k]

        hits = len(actual_set.intersection(pred_k))
        denom = min(k, len(actual))
        recalls.append(hits / denom)

    return float(np.mean(recalls)) if recalls else 0.0
