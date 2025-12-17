import numpy as np
import torch
from torch.optim import Adam


def bpr_loss(pos, neg):
    return -torch.log(torch.sigmoid(pos - neg) + 1e-12).mean()


class FMTrainer:
    def __init__(
        self,
        model,
        user_pos,
        valid_user_pos,
        num_items,
        item_attr_mat,
        item_attr_mask,
        device,
        lr,
        weight_decay,
        ckpt_path,
        early_stop_patience,
        user_offset,
        item_offset,
    ):
        self.model = model.to(device)
        self.device = device

        self.user_pos = user_pos
        self.user_pos_set = [set(x) for x in user_pos]
        self.valid_user_pos = valid_user_pos
        self.num_items = num_items

        self.item_attr_mat = item_attr_mat.to(device)
        self.item_attr_mask = item_attr_mask.to(device)

        self.user_offset = user_offset
        self.item_offset = item_offset

        self.optim = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        self.best_valid_loss = float("inf")
        self.no_improve = 0
        self.early_stop_patience = early_stop_patience
        self.ckpt_path = ckpt_path

    def sample(self, batch_size):
        users = np.random.randint(0, len(self.user_pos), size=batch_size)
        pos, neg = [], []

        for u in users:
            p = np.random.choice(self.user_pos[u])
            while True:
                n = np.random.randint(0, self.num_items)
                if n not in self.user_pos_set[u]:
                    break
            pos.append(p)
            neg.append(n)

        return (
            torch.tensor(users, device=self.device),
            torch.tensor(pos, device=self.device),
            torch.tensor(neg, device=self.device),
        )

    def build_feat(self, users, items):
        B = users.size(0)
        attrs = self.item_attr_mat[items]
        attr_mask = self.item_attr_mask[items]

        feat_idx = torch.cat(
            [
                (users + self.user_offset).view(B, 1),
                (items + self.item_offset).view(B, 1),
                attrs,
            ],
            dim=1,
        )

        feat_mask = torch.cat(
            [torch.ones((B, 2), device=self.device), attr_mask],
            dim=1,
        )

        return feat_idx, feat_mask

    @torch.no_grad()
    def compute_valid_loss(self, num_samples=3000):
        self.model.eval()
        losses = []
        users = list(self.valid_user_pos.keys())

        for _ in range(num_samples):
            u = np.random.choice(users)
            if len(self.valid_user_pos[u]) == 0:
                continue

            p = np.random.choice(self.valid_user_pos[u])
            while True:
                n = np.random.randint(0, self.num_items)
                if n not in self.user_pos_set[u]:
                    break

            u = torch.tensor([u], device=self.device)
            p = torch.tensor([p], device=self.device)
            n = torch.tensor([n], device=self.device)

            p_idx, p_mask = self.build_feat(u, p)
            n_idx, n_mask = self.build_feat(u, n)

            loss = bpr_loss(
                self.model(p_idx, p_mask),
                self.model(n_idx, n_mask),
            )
            losses.append(loss.item())

        return float(np.mean(losses))

    def train(self, epochs, batch_size, steps_per_epoch):
        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0

            for _ in range(steps_per_epoch):
                u, p, n = self.sample(batch_size)
                p_idx, p_mask = self.build_feat(u, p)
                n_idx, n_mask = self.build_feat(u, n)

                loss = bpr_loss(
                    self.model(p_idx, p_mask),
                    self.model(n_idx, n_mask),
                )

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                total_loss += loss.item()

            train_loss = total_loss / steps_per_epoch
            valid_loss = self.compute_valid_loss()

            print(
                f"[Epoch {epoch}] "
                f"train_loss={train_loss:.6f} | "
                f"valid_loss={valid_loss:.6f}"
            )

            if valid_loss < self.best_valid_loss - 1e-5:
                self.best_valid_loss = valid_loss
                self.no_improve = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": self.model.state_dict(),
                        "best_valid_loss": valid_loss,
                    },
                    self.ckpt_path,
                )
                print("💾 Best model updated (valid loss)")
            else:
                self.no_improve += 1
                if self.no_improve >= self.early_stop_patience:
                    print("🛑 Early stopping triggered")
                    break
