import numpy as np
import torch


@torch.no_grad()
def recommend_topk(
    model,
    train_mat,
    item_attr_mat,
    item_attr_mask,
    topk,
    device,
    user_offset,
    item_offset,
    chunk_size=2048,
):
    model.eval()
    num_users, num_items = train_mat.shape
    rec = np.zeros((num_users, topk), dtype=np.int64)

    item_attr_mat = item_attr_mat.to(device)
    item_attr_mask = item_attr_mask.to(device)

    for u in range(num_users):
        seen = train_mat.indices[train_mat.indptr[u]: train_mat.indptr[u + 1]]
        scores_all = []

        for start in range(0, num_items, chunk_size):
            end = min(start + chunk_size, num_items)
            items = torch.arange(start, end, device=device)

            B = items.size(0)
            feat_idx = torch.cat(
                [
                    torch.full((B, 1), u + user_offset, device=device),
                    (items + item_offset).view(B, 1),
                    item_attr_mat[items],
                ],
                dim=1,
            )

            feat_mask = torch.cat(
                [torch.ones((B, 2), device=device), item_attr_mask[items]],
                dim=1,
            )

            scores = model(feat_idx, feat_mask).cpu().numpy()

            if len(seen) > 0:
                in_chunk = (seen >= start) & (seen < end)
                scores[seen[in_chunk] - start] = -1e9

            scores_all.append(scores)

        scores_all = np.concatenate(scores_all)
        rec[u] = np.argsort(scores_all)[::-1][:topk]

    return rec
