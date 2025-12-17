import torch
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam


class MultiVAETrainer:
    def __init__(
        self,
        model,
        train_mat,
        num_items,
        lr,
        weight_decay,
        device,
        ckpt_path,
        early_stop_patience=20,
        kl_max_weight=0.2,
        kl_anneal_steps=20000,
    ):
        self.model = model.to(device)
        self.train_mat = train_mat
        self.num_items = num_items
        self.device = device
        self.ckpt_path = ckpt_path

        self.optim = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        self.early_stop_patience = early_stop_patience
        self.best_loss = float("inf")
        self.no_improve = 0

        self.kl_max_weight = kl_max_weight
        self.kl_anneal_steps = kl_anneal_steps
        self.global_step = 0

    def _kl_weight(self):
        return min(
            self.kl_max_weight,
            self.kl_max_weight * self.global_step / self.kl_anneal_steps,
        )

    def train(self, epochs, batch_size):
        num_users = self.train_mat.shape[0]

        for epoch in range(1, epochs + 1):
            self.model.train()
            perm = np.random.permutation(num_users)
            total_loss = 0.0

            for start in range(0, num_users, batch_size):
                batch_users = perm[start : start + batch_size]
                x = torch.from_numpy(
                    self.train_mat[batch_users].toarray()
                ).float().to(self.device)

                logits, mu, logvar = self.model(x)

                log_softmax = F.log_softmax(logits, dim=1)
                recon = -(log_softmax * x).sum(dim=1).mean()

                kl = -0.5 * torch.sum(
                    1 + logvar - mu.pow(2) - logvar.exp(), dim=1
                ).mean()

                loss = recon + self._kl_weight() * kl

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                total_loss += loss.item()
                self.global_step += 1

            avg_loss = total_loss / (num_users // batch_size + 1)
            print(f"[Epoch {epoch}] loss={avg_loss:.6f}")

            if avg_loss < self.best_loss - 1e-5:
                self.best_loss = avg_loss
                self.no_improve = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": self.model.state_dict(),
                        "best_loss": self.best_loss,
                    },
                    self.ckpt_path,
                )
                print("💾 Best model updated")
            else:
                self.no_improve += 1
                if self.no_improve >= self.early_stop_patience:
                    print("🛑 Early stopping triggered")
                    break
