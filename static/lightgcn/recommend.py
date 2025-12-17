import numpy as np
import torch


@torch.no_grad()
def recommend_topk(model, norm_adj, train_mat, topk, device):
    model.eval()
    user_emb, item_emb = model(norm_adj.to(device))
    scores = (user_emb @ item_emb.T).cpu().numpy()

    for u in range(train_mat.shape[0]):
        seen = train_mat.indices[train_mat.indptr[u]: train_mat.indptr[u + 1]]
        scores[u, seen] = -1e9

    idx = np.argpartition(scores, -topk, axis=1)[:, -topk:]
    part = scores[np.arange(scores.shape[0])[:, None], idx]
    order = np.argsort(part, axis=1)[:, ::-1]
    return idx[np.arange(idx.shape[0])[:, None], order]
