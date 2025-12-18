import os
import argparse
import torch
import pandas as pd
from data_utils import (read_interactions, encode_ids, build_user_item_matrix, train_valid_split_random, set_seed)
from model import MultiVAE
from trainer import MultiVAETrainer
from recommend import recommend_topk

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/data/ephemeral/home/Seung/data/train/")
    parser.add_argument("--output_dir", default="/data/ephemeral/home/Seung/output/MultVAE/")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01) # 가중치 감쇠 추가
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--valid_ratio", type=float, default=0.1) # 0.1 추천
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stop_patience", type=int, default=20)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, "best.pt")

    df = read_interactions(args.data_dir)
    df_enc, u2i, i2u, it2i, i2it = encode_ids(df)
    num_users, num_items = len(u2i), len(it2i)

    if args.valid_ratio > 0:
        train_df, valid_gt = train_valid_split_random(df_enc, num_users, valid_ratio=args.valid_ratio)
    else:
        train_df, valid_gt = df_enc, {u: [] for u in range(num_users)}

    train_mat = build_user_item_matrix(train_df, num_users, num_items)
    model = MultiVAE(num_items)

    trainer = MultiVAETrainer(
        model, train_mat, valid_gt, num_items, args.lr, args.weight_decay,
        device, ckpt_path, early_stop_patience=args.early_stop_patience
    )
    trainer.train(args.epochs, args.batch_size, args.topk)

    # Best Load
    best = torch.load(ckpt_path)
    model.load_state_dict(best["model_state"])
    
    # Final Recommend (Batch)
    rec = recommend_topk(model, train_mat, args.topk, device, args.batch_size)

    rows = []
    for u_idx in range(num_users):
        for it in rec[u_idx]:
            rows.append((i2u[u_idx], i2it[int(it)]))
    pd.DataFrame(rows, columns=["user", "item"]).to_csv(os.path.join(args.output_dir, "submission.csv"), index=False)
    print("✅ All processes finished!")

if __name__ == "__main__":
    main()