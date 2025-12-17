import os
import json
import argparse
import pandas as pd
import torch

from data_utils import (
    read_interactions,
    encode_ids,
    build_user_item_matrix,
    build_user_positives,
    train_valid_split_random,
    set_seed,
)
from metrics import recall_at_k

from model import FactorizationMachine
from trainer import FMTrainer
from recommend import recommend_topk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data/ephemeral/home/Seung/data/train",
        help="Path to interaction data directory",
    )

    parser.add_argument(
        "--attr_json",
        type=str,
        default="/data/ephemeral/home/Seung/data/train/Ml_item2attributes.json",
        help="Path to item-to-attributes json file",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/ephemeral/home/Seung/output/FM",
        help="Directory to save checkpoints and submission",
    )
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--steps_per_epoch", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--early_stop_patience", type=int, default=15)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # ---------- load ----------
    df = read_interactions(args.data_dir)
    df_enc, u2i, i2u, it2i, i2it = encode_ids(df)

    num_users = len(u2i)
    num_items = len(it2i)

    # ---------- split ----------
    train_df, valid_gt = train_valid_split_random(
        df_enc,
        num_users=num_users,
        min_interactions=5,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
    )

    train_mat = build_user_item_matrix(train_df, num_users, num_items)
    user_pos = build_user_positives(train_mat)

    # ---------- attributes ----------
    with open(args.attr_json, "r") as f:
        item2attrs = json.load(f)

    max_attrs = max(len(v) for v in item2attrs.values())
    item_attr_mat = torch.zeros((num_items, max_attrs), dtype=torch.long)
    item_attr_mask = torch.zeros((num_items, max_attrs))

    attr_offset = 1 + num_users + num_items
    for raw_item, attrs in item2attrs.items():
        if int(raw_item) not in it2i:
            continue
        i = it2i[int(raw_item)]
        for j, a in enumerate(attrs):
            item_attr_mat[i, j] = attr_offset + int(a)
            item_attr_mask[i, j] = 1.0

    # attribute id 전체에서 max 구하기
    all_attrs = set()
    for attrs in item2attrs.values():
        all_attrs.update(attrs)

    max_attr_id = max(all_attrs)

    num_features = 1 + num_users + num_items + max_attr_id + 1


    model = FactorizationMachine(num_features, args.embed_dim)

    trainer = FMTrainer(
        model=model,
        user_pos=user_pos,
        valid_user_pos=valid_gt,
        num_items=num_items,
        item_attr_mat=item_attr_mat,
        item_attr_mask=item_attr_mask,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        ckpt_path=os.path.join(args.output_dir, "best.pt"),
        early_stop_patience=args.early_stop_patience,
        user_offset=1,
        item_offset=1 + num_users,
    )

    trainer.train(args.epochs, args.batch_size, args.steps_per_epoch)

    ckpt = torch.load(os.path.join(args.output_dir, "best.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"✅ Best model loaded (epoch {ckpt['epoch']})")

    rec = recommend_topk(
        model,
        train_mat,
        item_attr_mat,
        item_attr_mask,
        args.topk,
        device,
        user_offset=1,
        item_offset=1 + num_users,
    )

    rows = []
    for u in range(num_users):
        for i in rec[u]:
            rows.append((i2u[u], i2it[i]))

    pd.DataFrame(rows, columns=["user", "item"]).to_csv(
        os.path.join(args.output_dir, "submission.csv"),
        index=False,
    )
    print("✅ submission.csv 생성 완료")


if __name__ == "__main__":
    main()
