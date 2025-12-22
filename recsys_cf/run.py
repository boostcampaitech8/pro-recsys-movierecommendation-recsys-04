from __future__ import annotations

import argparse
import os
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from data_utils import (
    load_interactions,
    make_id_mappings,
    apply_id_mappings,
    split_last_item_per_user,
    build_implicit_matrix,
)
from model import ItemKNN, UserKNN, ImplicitALS
from trainer import Trainer


BASE_EXP_DIR = "/data/ephemeral/home/minyou/experiments"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_path", type=str, required=True)
    p.add_argument("--model", type=str, choices=["itemcf", "usercf", "als"], required=True)

    p.add_argument("--k_eval", type=int, default=10)
    p.add_argument("--k_rec", type=int, default=10)
    p.add_argument("--topk_sim", type=int, default=200)

    # ALS params
    p.add_argument("--factors", type=int, default=64)
    p.add_argument("--reg", type=float, default=0.01)
    p.add_argument("--alpha", type=float, default=40.0)
    p.add_argument("--iters", type=int, default=10)

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    # ---------------- experiment dir ----------------
    exp_dir = os.path.join(BASE_EXP_DIR, args.model)
    os.makedirs(exp_dir, exist_ok=True)

    submission_path = os.path.join(exp_dir, "submission.csv")
    scores_path = os.path.join(exp_dir, "scores.parquet")
    config_path = os.path.join(exp_dir, "config.yaml")

    # ---------------- save config ----------------
    config = vars(args)
    config["timestamp"] = datetime.now().isoformat()

    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    # ---------------- data ----------------
    df = load_interactions(args.train_path)
    user2idx, item2idx, idx2user, idx2item = make_id_mappings(df)
    df_ui = apply_id_mappings(df, user2idx, item2idx)

    # train_df, valid_df = split_last_item_per_user(df_ui)

    n_users = len(user2idx)
    n_items = len(item2idx)

    X_train = build_implicit_matrix(df_ui, n_users, n_items)

    # ---------------- model ----------------
    if args.model == "itemcf":
        model = ItemKNN(topk_sim=args.topk_sim)
    elif args.model == "usercf":
        model = UserKNN(topk_sim=args.topk_sim)
    else:
        model = ImplicitALS(
            factors=args.factors,
            reg=args.reg,
            alpha=args.alpha,
            iters=args.iters,
        )

    trainer = Trainer(k_eval=args.k_eval)
    trainer.fit(model, X_train, seed=args.seed)

    # ---------------- generate scores & submission ----------------
    score_rows = []
    sub_rows = []

    for u in tqdm(range(n_users), desc=f"Generate recs [{args.model}]"):
        seen = set(X_train[u].indices.tolist())

        # 넉넉히 top-100 score 저장 (앙상블 대비)
        rec_items, rec_scores = model.recommend_with_scores(
            u, k=100, seen=seen
        )

        for rank, (i, s) in enumerate(zip(rec_items, rec_scores), start=1):
            score_rows.append({
                "user": idx2user[u],
                "item": idx2item[i],
                "score": float(s),
                "rank": rank,
            })

        # submission은 top-10만
        for i in rec_items[:args.k_rec]:
            sub_rows.append({
                "user": idx2user[u],
                "item": idx2item[i],
            })

    # ---------------- save ----------------
    pd.DataFrame(sub_rows).to_csv(submission_path, index=False)
    pd.DataFrame(score_rows).to_parquet(scores_path, index=False)

    print(f"[DONE] {args.model}")
    print(f" - submission : {submission_path}")
    print(f" - scores     : {scores_path}")
    print(f" - config     : {config_path}")


if __name__ == "__main__":
    main()
