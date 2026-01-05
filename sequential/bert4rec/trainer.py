import os
import time
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

from metrics import recall_at_k
from recommend import predict_topk_for_users


class BERT4RecTrainer:
    def __init__(
        self,
        model,
        train_loader,
        valid_data,
        train_seen,
        device,
        lr=1e-3,
        weight_decay=0.0,
        eval_topk=10,
        output_dir="./output/BERT4Rec/",
        use_amp=True,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_gt, self.train_seqs, self.idx2user, self.idx2item = valid_data
        self.train_seen = train_seen
        self.device = device

        self.eval_topk = eval_topk
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        self.use_amp = use_amp and (device.type == "cuda")
        self.scaler = GradScaler(enabled=self.use_amp)

    def train(self, epochs=10, log_every=200):
        best_recall = -1.0
        best_path = os.path.join(self.output_dir, "best.pt")

        for ep in range(1, epochs + 1):
            t0 = time.time()
            self.model.train()

            total_loss = 0.0
            for step, batch in enumerate(self.train_loader, start=1):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                pad_mask = batch["pad_mask"].to(self.device)

                self.optim.zero_grad(set_to_none=True)

                with autocast(enabled=self.use_amp):
                    logits = self.model(input_ids, pad_mask)
                    loss = self.criterion(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1)
                    )

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()

                total_loss += float(loss.item())

                if step % log_every == 0:
                    avg = total_loss / step
                    print(f"[BERT4Rec][Epoch {ep}] step {step}/{len(self.train_loader)} | loss={avg:.4f}")

            avg_loss = total_loss / max(1, len(self.train_loader))
            dt = time.time() - t0
            print(f"✅ [BERT4Rec] Epoch {ep} finished | loss={avg_loss:.4f} | {dt:.1f}s")

            val_recall = self.evaluate()
            print(f"📊 Validation Recall@{self.eval_topk}: {val_recall:.6f}")

            if val_recall > best_recall:
                best_recall = val_recall
                torch.save(self.model.state_dict(), best_path)
                print(f"🏆 Best updated! saved: {best_path}")

        print(f"🎯 Best Recall@{self.eval_topk}: {best_recall:.6f}")
        return best_path

    @torch.no_grad()
    def evaluate(self, batch_users=1024):
        self.model.eval()

        users = [u for u, gt in self.valid_gt.items() if len(gt) > 0]
        if len(users) == 0:
            return 0.0

        preds = []
        actuals = []

        for i in range(0, len(users), batch_users):
            ub = users[i:i + batch_users]
            topk_items = predict_topk_for_users(
                model=self.model,
                user_ids=ub,
                train_seqs=self.train_seqs,
                train_seen=self.train_seen,
                topk=self.eval_topk,
                device=self.device,
            )
            for u, rec in zip(ub, topk_items):
                actuals.append(self.valid_gt[u])
                preds.append(rec)

        return recall_at_k(actuals, preds, k=self.eval_topk)

    @torch.no_grad()
    def predict_submission(self, submit_topk=10, batch_users=1024, out_name="submission.csv"):
        import pandas as pd

        self.model.eval()
        num_users = len(self.train_seqs)
        users = list(range(num_users))

        rows = []
        for i in range(0, num_users, batch_users):
            ub = users[i:i + batch_users]
            topk_items = predict_topk_for_users(
                model=self.model,
                user_ids=ub,
                train_seqs=self.train_seqs,
                train_seen=self.train_seen,
                topk=submit_topk,
                device=self.device,
            )
            for u, rec in zip(ub, topk_items):
                user_raw = self.idx2user[u]
                for it in rec:
                    item_raw = self.idx2item[it]
                    rows.append((user_raw, item_raw))

        out_path = os.path.join(self.output_dir, out_name)
        pd.DataFrame(rows, columns=["user", "item"]).to_csv(out_path, index=False)
        print("✅ Submission saved:", out_path)
        return out_path
