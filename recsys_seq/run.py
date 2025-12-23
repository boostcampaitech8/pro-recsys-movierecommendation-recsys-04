from __future__ import annotations

import argparse
import os
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from recsys_seq.data_utils import (
    load_interactions,
    make_id_mappings,
    apply_id_mappings,
    split_last_item_per_user,
    build_implicit_matrix,
    load_item_sideinfo,
)
from recsys_seq.trainer import Trainer

from recsys_seq.models.als import ImplicitALS
from recsys_seq.models.sasrec_content import SASRecWithContent, SASRecConfig
from recsys_seq.models.hybrid import HybridALSContent, HybridConfig

BASE_EXP_DIR = "./experiments"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_path", type=str, required=True, help=".../train_ratings.csv")
    p.add_argument("--meta_dir", type=str, required=True, help="dir containing genres.tsv/directors.tsv/writers.tsv")
    p.add_argument("--model", type=str, choices=["als", "sasrec_content", "hybrid"], required=True)

    p.add_argument("--k_eval", type=int, default=10)
    p.add_argument("--k_rec", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    # ALS params
    p.add_argument("--factors", type=int, default=64)
    p.add_argument("--reg", type=float, default=0.01)
    p.add_argument("--alpha", type=float, default=40.0)
    p.add_argument("--iters", type=int, default=10)

    # SASRec params
    p.add_argument("--max_seq_len", type=int, default=50)
    p.add_argument("--embed_dim", type=int, default=64)
    p.add_argument("--num_heads", type=int, default=2)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_neg", type=int, default=50)
    p.add_argument("--device", type=str, default="cpu")

    # content truncation
    p.add_argument("--max_genres", type=int, default=5)
    p.add_argument("--max_directors", type=int, default=5)
    p.add_argument("--max_writers", type=int, default=5)

    # Hybrid params
    p.add_argument("--cand_k", type=int, default=200)
    p.add_argument("--w_als", type=float, default=0.7)
    p.add_argument("--w_content", type=float, default=0.3)

    return p.parse_args()

def build_user_sequences(df_ui: pd.DataFrame, n_users: int) -> list[list[int]]:
    # df_ui must be sorted by (u, time)
    user_seqs = [[] for _ in range(n_users)]
    for u, i in df_ui[["u","i"]].to_numpy():
        user_seqs[int(u)].append(int(i))
    return user_seqs

def main():
    args = parse_args()
    os.makedirs(BASE_EXP_DIR, exist_ok=True)

    exp_dir = os.path.join(BASE_EXP_DIR, args.model)
    os.makedirs(exp_dir, exist_ok=True)

    submission_path = os.path.join(exp_dir, "submission.csv")
    scores_path = os.path.join(exp_dir, "scores.parquet")
    config_path = os.path.join(exp_dir, "config.yaml")

    config = vars(args)
    config["timestamp"] = datetime.now().isoformat()
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    # ---------------- data ----------------
    df = load_interactions(args.train_path)
    user2idx, item2idx, idx2user, idx2item = make_id_mappings(df)
    df_ui = apply_id_mappings(df, user2idx, item2idx)
    train_df, valid_df = split_last_item_per_user(df_ui)

    n_users = len(user2idx)
    n_items = len(item2idx)

    X_train = build_implicit_matrix(train_df, n_users, n_items)

    # sideinfo (item index 기준)
    sideinfo = load_item_sideinfo(args.meta_dir, item2idx=item2idx)

    trainer = Trainer(k_eval=args.k_eval)

    # ---------------- model ----------------
    if args.model == "als":
        model = ImplicitALS(factors=args.factors, reg=args.reg, alpha=args.alpha, iters=args.iters)
        trainer.fit(model, X_train, seed=args.seed)

    elif args.model == "hybrid":
        hcfg = HybridConfig(
            factors=args.factors, reg=args.reg, alpha=args.alpha, iters=args.iters,
            cand_k=args.cand_k, w_als=args.w_als, w_content=args.w_content
        )
        model = HybridALSContent(sideinfo=sideinfo, cfg=hcfg)
        trainer.fit(model, X_train, seed=args.seed)

    else:
        scfg = SASRecConfig(
            max_seq_len=args.max_seq_len,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_neg=args.num_neg,
            max_genres=args.max_genres,
            max_directors=args.max_directors,
            max_writers=args.max_writers,
            device=args.device,
        )
        model = SASRecWithContent(n_items=n_items, sideinfo=sideinfo, cfg=scfg)

        # build sequences from train_df (last item removed)
        user_seqs = build_user_sequences(train_df.sort_values(["u","time"]), n_users)
        model.fit_sequences(user_seqs=user_seqs, seed=args.seed)

        # stash sequences for inference
        model.user_seqs = user_seqs

    # ---------------- evaluate (last-item) ----------------
    valid_pairs = valid_df[["u","i"]].to_numpy()
    if args.model == "sasrec_content":
        # SASRec needs user_seq in recommend
        # quick wrapper with bound sequence
        class _Wrap:
            def __init__(self, m): self.m=m
            def recommend(self,u,k=10,seen=None):
                return self.m.recommend(u, self.m.user_seqs[u], k=k, seen=seen)
            def recommend_with_scores(self,u,k=10,seen=None):
                return self.m.recommend_with_scores(u, self.m.user_seqs[u], k=k, seen=seen)
        eval_model = _Wrap(model)
    else:
        eval_model = model

    res = trainer.evaluate_last_item(eval_model, X_train, valid_pairs)
    print(f"[EVAL] Recall@{args.k_eval}: {res.recall:.6f} | NDCG@{args.k_eval}: {res.ndcg:.6f}")

    # ---------------- generate scores & submission ----------------
    score_rows = []
    sub_rows = []

    for u in tqdm(range(n_users), desc=f"Generate recs [{args.model}]"):
        seen = set(X_train[u].indices.tolist())

        if args.model == "sasrec_content":
            rec_items, rec_scores = model.recommend_with_scores(u, model.user_seqs[u], k=100, seen=seen)
        else:
            rec_items, rec_scores = model.recommend_with_scores(u, k=100, seen=seen)

        for rank, (i, s) in enumerate(zip(rec_items, rec_scores), start=1):
            score_rows.append({
                "user": idx2user[u],
                "item": idx2item[i],
                "score": float(s),
                "rank": rank,
            })

        for i in rec_items[:args.k_rec]:
            sub_rows.append({"user": idx2user[u], "item": idx2item[i]})

    pd.DataFrame(sub_rows).to_csv(submission_path, index=False)
    pd.DataFrame(score_rows).to_parquet(scores_path, index=False)

    print(f"[DONE] {args.model}")
    print(f" - submission : {submission_path}")
    print(f" - scores     : {scores_path}")
    print(f" - config     : {config_path}")

if __name__ == "__main__":
    main()
