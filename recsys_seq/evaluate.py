import argparse
import pandas as pd
import numpy as np

from recsys_seq.data_utils import (
    load_interactions,
    make_id_mappings,
    apply_id_mappings,
    split_last_item_per_user,
)

def recall_ndcg_at_k(pred_items, true_item, k):
    if true_item in pred_items[:k]:
        recall = 1.0
        rank = pred_items.index(true_item)
        ndcg = 1.0 / np.log2(rank + 2)
    else:
        recall = 0.0
        ndcg = 0.0
    return recall, ndcg

def main(args):
    df = load_interactions(args.train_path)
    user2idx, item2idx, idx2user, idx2item = make_id_mappings(df)
    df_ui = apply_id_mappings(df, user2idx, item2idx)

    _, valid_df = split_last_item_per_user(df_ui)
    gt = { idx2user[u]: idx2item[i] for u, i in valid_df[["u","i"]].to_numpy() }

    scores = pd.read_parquet(args.scores_path)
    user_preds = (
        scores.sort_values(["user","rank"])
        .groupby("user")["item"]
        .apply(list).to_dict()
    )

    recalls, ndcgs = [], []
    for user, true_item in gt.items():
        if user not in user_preds:
            continue
        r, n = recall_ndcg_at_k(user_preds[user], true_item, args.k)
        recalls.append(r); ndcgs.append(n)

    print(f"Recall@{args.k}: {np.mean(recalls):.6f}")
    print(f"NDCG@{args.k}:  {np.mean(ndcgs):.6f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_path", type=str, required=True)
    p.add_argument("--scores_path", type=str, required=True)
    p.add_argument("--k", type=int, default=10)
    main(p.parse_args())
