import os
import argparse
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix, diags

from data_utils import (
    read_interactions,
    encode_ids,
    build_user_item_matrix,
    build_user_positives,
    set_seed,
    train_valid_split_random,
)
from metrics import recall_at_k
from model import LightGCN
from trainer import LightGCNTrainer
from recommend import recommend_topk


def build_norm_adj(train_mat):
    num_users, num_items = train_mat.shape
    R = train_mat.tocsr()

    rows = np.concatenate([R.nonzero()[0], R.nonzero()[1] + num_users])
    cols = np.concatenate([R.nonzero()[1] + num_users, R.nonzero()[0]])
    data = np.ones(len(rows), dtype=np.float32)

    A = csr_matrix(
        (data, (rows, cols)),
        shape=(num_users + num_items, num_users + num_items),
    )

    deg = np.array(A.sum(axis=1)).squeeze()
    deg[deg == 0.0] = 1.0
    D_inv = diags(1.0 / np.sqrt(deg))
    norm_A = D_inv @ A @ D_inv
    norm_A = norm_A.tocoo()

    indices = torch.from_numpy(
        np.vstack((norm_A.row, norm_A.col))
    ).long()
    values = torch.from_numpy(norm_A.data).float()

    return torch.sparse_coo_tensor(
        indices, values, norm_A.shape
    ).coalesce()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="/data/ephemeral/home/Seung/data/train/")
    parser.add_argument("--output_dir", default="/data/ephemeral/home/Seung/output/LightGCN/")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--steps_per_epoch", type=int, default=200)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--valid_ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stop_patience", type=int, default=20)
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
            min_interactions=5,
            valid_ratio=args.valid_ratio,
            seed=args.seed,
        )
        print(f"🧪 Validation ON (valid_ratio={args.valid_ratio})")
    else:
        train_df = df_enc
        valid_gt = {u: [] for u in range(num_users)}
        print("🧪 Validation OFF")

    # ---------- graph ----------
    train_mat = build_user_item_matrix(train_df, num_users, num_items)
    user_pos = build_user_positives(train_mat)
    norm_adj = build_norm_adj(train_mat).to(device)

    # ---------- model ----------
    model = LightGCN(num_users, num_items, args.embed_dim, args.num_layers)

    trainer = LightGCNTrainer(
        model,
        norm_adj,
        user_pos,
        num_items,
        args.lr,
        args.weight_decay,
        device,
        ckpt_path=ckpt_path,
        early_stop_patience=args.early_stop_patience,
    )

    # ---------- train ----------
    trainer.train(args.epochs, args.batch_size, args.steps_per_epoch)

    # ---------- load best ----------
    best_ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    print(f"✅ Best model loaded (epoch {best_ckpt['epoch']})")

    # ---------- recommend ----------
    rec = recommend_topk(
        model,
        norm_adj,
        train_mat,
        args.topk,
        device,
    )

    # ---------- validation ----------
    if args.valid_ratio > 0:
        actual_list, pred_list = [], []
        for u in range(num_users):
            if len(valid_gt[u]) == 0:
                continue
            actual_list.append(valid_gt[u])
            pred_list.append(rec[u].tolist())

        val_recall = recall_at_k(actual_list, pred_list, k=args.topk)
        print(f"📊 Validation Recall@{args.topk}: {val_recall:.6f}")

    # ---------- submission ----------
    rows = []
    for u_idx in range(rec.shape[0]):
        for j in range(args.topk):
            rows.append((i2u[u_idx], i2it[int(rec[u_idx, j])]))

    pd.DataFrame(rows, columns=["user", "item"]).to_csv(
        os.path.join(args.output_dir, "submission.csv"),
        index=False,
    )

    print("✅ submission.csv 생성 완료")


if __name__ == "__main__":
    main()
