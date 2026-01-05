import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader

from model import BERT4Rec
from trainer import BERT4RecTrainer
from data_utils import (
    read_interactions,
    encode_ids,
    build_user_sequences,
    train_valid_split_last,
    set_seed,
)


class BERT4RecMLMDataset(Dataset):
    """
    Trainer가 기대하는 batch dict:
      - input_ids: (B, L) long
      - labels:    (B, L) long  (masked position만 정답 token id, 나머지는 -100)
      - pad_mask:  (B, L) bool  (PAD 위치 True)

    Token convention:
      PAD = 0
      item token = item_idx + 1  (1..num_items)
      MASK = num_items + 1
    """

    def __init__(self, train_seqs, num_items, max_len=200, mask_prob=0.2, seed=42, always_mask_last=True):
        self.train_seqs = train_seqs
        self.num_items = num_items
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.always_mask_last = always_mask_last

        self.PAD = 0
        self.MASK = num_items + 1

        import numpy as np
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.train_seqs)

    def __getitem__(self, idx):
        seq = self.train_seqs[idx]  # list of item_idx (0..num_items-1)

        # shift to token ids (1..num_items)
        tokens = [it + 1 for it in seq][-self.max_len:]

        # left pad
        if len(tokens) < self.max_len:
            tokens = [self.PAD] * (self.max_len - len(tokens)) + tokens

        input_ids = tokens[:]
        labels = [-100] * self.max_len

        # mask candidates: non-PAD positions
        cand_pos = [i for i, t in enumerate(tokens) if t != self.PAD]
        mask_pos = set()

        if len(cand_pos) > 0:
            # (1) always include last real token position -> "last-item prediction" 강화
            if self.always_mask_last:
                last_real = cand_pos[-1]
                mask_pos.add(last_real)

            # (2) random masks (keep total around mask_prob)
            num_mask = max(1, int(len(cand_pos) * self.mask_prob))
            remain = num_mask - len(mask_pos)
            if remain > 0:
                pool = [p for p in cand_pos if p not in mask_pos]
                if len(pool) > 0:
                    extra = self.rng.choice(pool, size=min(remain, len(pool)), replace=False).tolist()
                    mask_pos.update(extra)

        for p in mask_pos:
            labels[p] = tokens[p]
            input_ids[p] = self.MASK

        pad_mask = [t == self.PAD for t in input_ids]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "pad_mask": torch.tensor(pad_mask, dtype=torch.bool),
        }


def build_train_seen(train_seqs):
    """
    recommend.py에서 seen masking 할 때 "shifted token id(1..num_items)"를 기대함. :contentReference[oaicite:4]{index=4}
    train_seqs는 item_idx이므로 +1 해서 저장.
    """
    train_seen = {}
    for u, seq in enumerate(train_seqs):
        train_seen[u] = set((it + 1) for it in seq)
    return train_seen


def main():
    parser = argparse.ArgumentParser()

    # ===============================
    # Paths
    # ===============================
    parser.add_argument("--data_dir", default="/data/ephemeral/home/Seung/data/train/")
    parser.add_argument("--data_file", default="train_ratings.csv")
    parser.add_argument("--output_dir", default="/data/ephemeral/home/Seung/output/BERT4Rec/")

    # ===============================
    # Training
    # ===============================
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--mask_prob", type=float, default=0.2)
    parser.add_argument("--always_mask_last", action="store_true")  # last-item 항상 mask (추천)

    # ===============================
    # Eval / Submit
    # ===============================
    parser.add_argument("--eval_topk", type=int, default=10)
    parser.add_argument("--submit_topk", type=int, default=10)
    parser.add_argument("--no_valid", action="store_true")

    # ===============================
    # Model
    # ===============================
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    # ===============================
    # S3Rec embedding (optional)
    # ===============================
    parser.add_argument("--s3rec_emb", type=str, default=None)

    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("🚀 BERT4Rec Fine-tune (share embedding + last-item masking option)")
    print("=" * 60)

    # ===============================
    # Load & encode
    # ===============================
    df = read_interactions(args.data_dir, args.data_file)
    df_enc, user2idx, idx2user, item2idx, idx2item = encode_ids(df)

    num_users = len(user2idx)
    num_items = len(item2idx)
    print(f"Users: {num_users}, Items: {num_items}")

    # sequences: list length = num_users, each is list[item_idx]
    seqs = build_user_sequences(df_enc, num_users)

    # ===============================
    # Split
    # ===============================
    if args.no_valid:
        train_seqs = seqs
        valid_gt = {u: [] for u in range(num_users)}
        print("🧪 Validation OFF (full data)")
    else:
        train_seqs, valid_gt = train_valid_split_last(seqs, min_interactions=3, n_valid=1)
        print("🧪 Validation ON (per-user last holdout)")

    # ===============================
    # Dataloader
    # ===============================
    dataset = BERT4RecMLMDataset(
        train_seqs=train_seqs,
        num_items=num_items,
        max_len=args.max_len,
        mask_prob=args.mask_prob,
        seed=args.seed,
        always_mask_last=args.always_mask_last,
    )

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    train_seen = build_train_seen(train_seqs)

    # trainer가 기대하는 valid_data tuple :contentReference[oaicite:5]{index=5}
    valid_data = (valid_gt, train_seqs, idx2user, idx2item)

    # ===============================
    # Model
    # ===============================
    model = BERT4Rec(
        num_items=num_items,
        max_len=args.max_len,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    # S3Rec emb load (optional)
    if args.s3rec_emb is not None:
        print("📥 Loading S3Rec item embedding:", args.s3rec_emb)
        s3_emb = torch.load(args.s3rec_emb, map_location="cpu")
        model.load_s3rec_item_embedding(s3_emb)
        print("✅ Loaded S3Rec item embedding into BERT4Rec")

    # ===============================
    # Train
    # ===============================
    trainer = BERT4RecTrainer(
        model=model,
        train_loader=train_loader,
        valid_data=valid_data,
        train_seen=train_seen,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        eval_topk=args.eval_topk,
        output_dir=args.output_dir,
        use_amp=True,
    )

    best_ckpt = trainer.train(epochs=args.epochs, log_every=200)

    # ===============================
    # Submission
    # ===============================
    # no_valid이면 valid_recall은 0.0이 정상이고, 그냥 best_ckpt(사실상 마지막 갱신)로 제출 생성
    trainer.model.load_state_dict(torch.load(best_ckpt, map_location=device))
    trainer.model = trainer.model.to(device)
    trainer.predict_submission(submit_topk=args.submit_topk, out_name="submission.csv")


if __name__ == "__main__":
    main()
