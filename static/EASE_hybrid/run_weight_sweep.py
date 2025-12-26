import os
import argparse
import itertools
import numpy as np
import pandas as pd

from data_utils import (
    read_interactions,
    encode_ids,
    build_user_item_matrix,
    load_metadata_matrix,
    train_valid_split_random,
    set_seed,
)
from metrics import recall_at_k
from model import EASE
from trainer import EASETrainer
from recommend import recommend_topk
from utils import build_valid_lists


def main():
    parser = argparse.ArgumentParser()

    # ===============================
    # Paths
    # ===============================
    parser.add_argument("--data_dir", default="/data/ephemeral/home/Seung/data/train/")
    parser.add_argument("--output_dir", default="/data/ephemeral/home/Seung/output/EASE_Hybrid/")

    # ===============================
    # Params
    # ===============================
    parser.add_argument("--lambda_reg", type=float, default=500.0)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--eval_topk", type=int, default=10)
    parser.add_argument("--submit_topk", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--sweep", action="store_true")

    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ===============================
    # Logging
    # ===============================
    log_path = os.path.join(args.output_dir, "metadata_weight_sweep.csv")
    if not os.path.exists(log_path):
        pd.DataFrame(
            columns=[
                "w_genres",
                "w_directors",
                "w_writers",
                "w_years",
                "recall@10",
            ]
        ).to_csv(log_path, index=False)

    # ===============================
    # Load Data
    # ===============================
    df = read_interactions(args.data_dir)
    df_enc, u2i, i2u, it2i, i2it = encode_ids(df)

    num_users = len(u2i)
    num_items = len(it2i)

    if args.valid_ratio > 0:
        train_df, valid_gt = train_valid_split_random(
            df_enc, num_users, args.valid_ratio, args.seed
        )
    else:
        train_df = df_enc
        valid_gt = {u: [] for u in range(num_users)}

    train_mat = build_user_item_matrix(train_df, num_users, num_items)
    feat_mats = load_metadata_matrix(args.data_dir, it2i, num_items)

    # ===============================
    # Metadata weight sweep (independent)
    # ===============================
    if args.sweep:
        BASE = 35.0  # 네가 이미 검증한 최적 scale

        values = [0.0, BASE]  # ON / OFF
        grid = list(itertools.product(values, repeat=4))

        for wg, wd, ww, wy in grid:
            meta_weights = {
                "genres": wg,
                "directors": wd,
                "writers": ww,
                "years": wy,
            }

            print("=" * 80)
            print("🧪 Running meta sweep:")
            print(meta_weights)
            print("=" * 80)

            model = EASE(lambda_reg=args.lambda_reg, meta_weights=meta_weights)
            trainer = EASETrainer(model, train_mat, feat_mats=feat_mats)
            trainer.train()

            score_mat = trainer.predict()
            rec = recommend_topk(score_mat, train_mat, args.submit_topk, True)

            actual, pred = build_valid_lists(valid_gt, rec)
            recall = recall_at_k(actual, pred, args.eval_topk)

            print(f"📊 Recall@{args.eval_topk}: {recall:.6f}")

            pd.DataFrame(
                [[wg, wd, ww, wy, recall]],
                columns=[
                    "w_genres",
                    "w_directors",
                    "w_writers",
                    "w_years",
                    "recall@10",
                ],
            ).to_csv(log_path, mode="a", header=False, index=False)

        print("✅ Metadata independent sweep finished.")
        return


if __name__ == "__main__":
    main()
