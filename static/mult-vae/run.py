import os
import argparse
import torch
import pandas as pd

from data_utils import (
    read_interactions,
    encode_ids,
    build_user_item_matrix,
    train_valid_split_random,
    set_seed,
)
from metrics import recall_at_k
from model import MultiVAE
from trainer import MultiVAETrainer
from recommend import recommend_topk


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="/data/ephemeral/home/Seung/data/train/")
    parser.add_argument("--output_dir", default="/data/ephemeral/home/Seung/output/MultVAE/")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--valid_ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stop_patience", type=int, default=500)
    parser.add_argument("--no_cuda", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, "best.pt")

    # ---------- load ----------
    df = read_interactions(args.data_dir)
    df_enc, u2i, i2u, it2i, i2it = encode_ids(df)

    num_users = len(u2i)
    num_items = len(it2i)

    # ---------- split ----------
    if args.valid_ratio > 0:
        train_df, valid_gt = train_valid_split_random(
            df_enc,
            num_users=num_users,
            valid_ratio=args.valid_ratio,
            seed=args.seed,
        )
        print("🧪 Validation ON")
    else:
        train_df = df_enc
        valid_gt = {u: [] for u in range(num_users)}
        print("🧪 Validation OFF")

    train_mat = build_user_item_matrix(train_df, num_users, num_items)

    # ---------- model ----------
    model = MultiVAE(num_items)

    trainer = MultiVAETrainer(
        model,
        train_mat,
        num_items,
        args.lr,
        args.weight_decay,
        device,
        ckpt_path,
        early_stop_patience=args.early_stop_patience,
    )

    trainer.train(args.epochs, args.batch_size)

    # ---------- load best ----------
    best = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(best["model_state"])
    print(f"✅ Best model loaded (epoch {best['epoch']})")

    # ---------- recommend ----------
    rec = recommend_topk(model, train_mat, args.topk, device)

    # ---------- validation ----------
    if args.valid_ratio > 0:
        actual, pred = [], []
        for u in range(num_users):
            if len(valid_gt[u]) == 0:
                continue
            actual.append(valid_gt[u])
            pred.append(rec[u].tolist())
        val = recall_at_k(actual, pred, k=args.topk)
        print(f"📊 Validation Recall@{args.topk}: {val:.6f}")

    # ---------- submission ----------
    rows = []
    for u_idx in range(num_users):
        for it in rec[u_idx]:
            rows.append((i2u[u_idx], i2it[int(it)]))

    pd.DataFrame(rows, columns=["user", "item"]).to_csv(
        os.path.join(args.output_dir, "submission.csv"),
        index=False,
    )
    print("✅ submission.csv 생성 완료")


if __name__ == "__main__":
    main()
