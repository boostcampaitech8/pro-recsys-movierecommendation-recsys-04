import torch
import numpy as np


@torch.no_grad()
def recommend_topk(model, train_mat, topk, device):
    model.eval()
    num_users = train_mat.shape[0]
    all_recs = []

    for u in range(num_users):
        x = torch.from_numpy(
            train_mat[u].toarray()
        ).float().to(device)

        logits, _, _ = model(x)
        scores = logits.squeeze().cpu().numpy()

        seen = train_mat[u].indices
        scores[seen] = -1e9

        topk_items = np.argsort(scores)[-topk:][::-1]
        all_recs.append(topk_items)

    return np.array(all_recs)
