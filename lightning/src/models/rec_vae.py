import logging
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.metrics import recall_at_k

log = logging.getLogger(__name__)


class RecVAE(L.LightningModule):
    """
    RecVAE (EMA composite-prior + alternating updates) Lightning implementation.
    Enhanced with separate encoder/decoder dimensions and multi-layer support.

    Key differences vs MultiVAE:
      - 2 optimizers: encoder / decoder
      - manual optimization (alternating updates)
      - composite prior KL: alpha * KL(q||N(0,I)) + (1-alpha) * KL(q||q_old)
      - q_old is tracked via EMA of posterior stats (mu, logvar)
      - Separate encoder_dims and decoder_dims for flexible architecture
    """

    def __init__(
        self,
        num_items: int,
        # ✅ 새로운 파라미터: encoder/decoder/latent 분리
        encoder_dims: list[int] = None,      # e.g., [2048, 1024, 512]
        decoder_dims: list[int] = None,      # e.g., [512, 1024, 2048]
        latent_dim: int = 300,               # z의 차원
        # 기존 호환성을 위한 fallback
        hidden_dims: tuple = None,           # deprecated, but kept for compatibility
        dropout: float = 0.5,
        # optim
        lr_encoder: float = 1e-3,
        lr_decoder: float = 1e-3,
        weight_decay: float = 0.0,
        # RecVAE controls
        alpha: float = 0.5,
        ema_decay: float = 0.999,
        enc_steps_per_iter: int = 2,
        dec_steps_per_iter: int = 1,
        encoder_input_corruption: float = 0.5,
        grad_clip_val: float | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_items = num_items
        self.dropout = float(dropout)

        self.lr_encoder = float(lr_encoder)
        self.lr_decoder = float(lr_decoder)
        self.weight_decay = float(weight_decay)

        self.alpha = float(alpha)
        self.ema_decay = float(ema_decay)
        self.enc_steps_per_iter = int(enc_steps_per_iter)
        self.dec_steps_per_iter = int(dec_steps_per_iter)
        self.encoder_input_corruption = float(encoder_input_corruption)
        self.grad_clip_val = grad_clip_val

        # ✅ 파라미터 처리: encoder_dims/decoder_dims 우선, 없으면 hidden_dims 사용
        if encoder_dims is not None:
            self.encoder_dims = list(encoder_dims)
        elif hidden_dims is not None:
            # 기존 방식 호환: hidden_dims를 encoder로 사용
            self.encoder_dims = list(hidden_dims)
        else:
            # 기본값
            self.encoder_dims = [2048, 1024, 512]

        if decoder_dims is not None:
            self.decoder_dims = list(decoder_dims)
        elif hidden_dims is not None:
            # 기존 방식: encoder_dims를 역순으로
            self.decoder_dims = list(reversed(self.encoder_dims))
        else:
            # 기본값
            self.decoder_dims = [512, 1024, 2048]

        self.latent_dim = int(latent_dim)

        # ========== Encoder ==========
        encoder_layers = []
        input_dim = num_items
        
        for hidden_dim in self.encoder_dims:
            encoder_layers.append(nn.Linear(input_dim, hidden_dim))
            encoder_layers.append(nn.ELU())
            encoder_layers.append(nn.Dropout(self.dropout))
            input_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space: encoder의 마지막 dim -> latent_dim
        encoder_output_dim = self.encoder_dims[-1] if self.encoder_dims else num_items
        self.mu = nn.Linear(encoder_output_dim, self.latent_dim)
        self.logvar = nn.Linear(encoder_output_dim, self.latent_dim)

        # ========== Decoder ==========
        decoder_layers = []
        input_dim = self.latent_dim
        
        for hidden_dim in self.decoder_dims:
            decoder_layers.append(nn.Linear(input_dim, hidden_dim))
            decoder_layers.append(nn.ELU())
            decoder_layers.append(nn.Dropout(self.dropout))
            input_dim = hidden_dim
        
        # 마지막 layer: decoder의 마지막 dim -> num_items
        decoder_layers.append(nn.Linear(input_dim, num_items))
        
        self.decoder = nn.Sequential(*decoder_layers)

        self._init_weights()

        # ===== EMA buffers for old posterior stats (q_old) =====
        self.register_buffer("old_mu_ema", torch.zeros(self.latent_dim))
        self.register_buffer("old_logvar_ema", torch.zeros(self.latent_dim))
        self.register_buffer("ema_inited", torch.tensor(False))

        # Lightning: manual optimization
        self.automatic_optimization = False

        log.info(f"RecVAE initialized with {num_items} items")
        log.info(f"  Encoder dims: {num_items} -> {' -> '.join(map(str, self.encoder_dims))} -> {self.latent_dim}")
        log.info(f"  Decoder dims: {self.latent_dim} -> {' -> '.join(map(str, self.decoder_dims))} -> {num_items}")
        log.info(
            f"  alpha={self.alpha}, ema_decay={self.ema_decay}, "
            f"enc_steps={self.enc_steps_per_iter}, dec_steps={self.dec_steps_per_iter}, "
            f"enc_corruption={self.encoder_input_corruption}"
        )

    def _init_weights(self):
        """He initialization for ELU"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ---------------------------
    # core: encode / reparam / decode
    # ---------------------------
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters"""
        x = F.normalize(x, p=2, dim=1)
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # inference

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to reconstruction logits"""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar

    # ---------------------------
    # losses
    # ---------------------------
    @staticmethod
    def recon_multinomial_ce(logits: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Multinomial log-likelihood (reconstruction loss)"""
        log_softmax = F.log_softmax(logits, dim=1)
        return -(log_softmax * x).sum(dim=1).mean()

    @staticmethod
    def kl_normal_to_std(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL(q||N(0,I)) for diagonal Gaussian"""
        return (-0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)).mean()

    @staticmethod
    def kl_normal_to_normal(
        mu_q: torch.Tensor,
        logvar_q: torch.Tensor,
        mu_p: torch.Tensor,
        logvar_p: torch.Tensor,
    ) -> torch.Tensor:
        """KL(N(mu_q, var_q) || N(mu_p, var_p)) for diagonal Gaussians"""
        var_q = logvar_q.exp()
        var_p = logvar_p.exp()
        kl = 0.5 * torch.sum(
            (logvar_p - logvar_q) + (var_q + (mu_q - mu_p).pow(2)) / var_p - 1.0,
            dim=1,
        )
        return kl.mean()

    def composite_prior_kl(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Composite-prior KL: alpha * KL(q||N(0,I)) + (1-alpha) * KL(q||q_old)
        """
        kl_std = self.kl_normal_to_std(mu, logvar)

        if not bool(self.ema_inited.item()):
            return kl_std

        kl_old = self.kl_normal_to_normal(mu, logvar, self.old_mu_ema, self.old_logvar_ema)
        return self.alpha * kl_std + (1.0 - self.alpha) * kl_old

    @torch.no_grad()
    def update_old_posterior_ema(self, mu: torch.Tensor, logvar: torch.Tensor):
        """Update EMA buffers using batch mean stats"""
        batch_mu = mu.detach().mean(dim=0)
        batch_logvar = logvar.detach().mean(dim=0)

        if not bool(self.ema_inited.item()):
            self.old_mu_ema.copy_(batch_mu)
            self.old_logvar_ema.copy_(batch_logvar)
            self.ema_inited.copy_(torch.tensor(True, device=self.device))
            return

        d = self.ema_decay
        self.old_mu_ema.mul_(d).add_(batch_mu * (1.0 - d))
        self.old_logvar_ema.mul_(d).add_(batch_logvar * (1.0 - d))

    # ---------------------------
    # training / validation
    # ---------------------------
    def training_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        opt_enc, opt_dec = self.optimizers()
        sched_enc, sched_dec = self.lr_schedulers()

        # ====== (A) Encoder updates ======
        self.decoder.eval()
        for p in self.decoder.parameters():
            p.requires_grad = False
        for p in list(self.encoder.parameters()) + list(self.mu.parameters()) + list(self.logvar.parameters()):
            p.requires_grad = True

        enc_loss_last = None
        enc_recon_last = None
        enc_kl_last = None

        for _ in range(self.enc_steps_per_iter):
            opt_enc.zero_grad(set_to_none=True)
            x_corrupt = F.dropout(x, p=self.encoder_input_corruption, training=True)
            logits, mu, logvar = self(x_corrupt)
            recon = self.recon_multinomial_ce(logits, x)
            kl = self.composite_prior_kl(mu, logvar)
            loss = recon + kl
            self.manual_backward(loss)
            opt_enc.step()

            enc_loss_last = loss
            enc_recon_last = recon
            enc_kl_last = kl

            self.update_old_posterior_ema(mu, logvar)

        self.log("train/enc_recon", enc_recon_last, on_epoch=True)
        self.log("train/enc_kl", enc_kl_last, on_epoch=True)

        # ====== (B) Decoder updates ======
        self.decoder.train()
        for p in self.decoder.parameters():
            p.requires_grad = True
        for p in list(self.encoder.parameters()) + list(self.mu.parameters()) + list(self.logvar.parameters()):
            p.requires_grad = False

        dec_loss_last = None
        dec_recon_last = None
        dec_kl_last = None

        for _ in range(self.dec_steps_per_iter):
            opt_dec.zero_grad(set_to_none=True)
            logits, mu, logvar = self(x)
            recon = self.recon_multinomial_ce(logits, x)
            kl = self.composite_prior_kl(mu, logvar)
            loss = recon + kl
            self.manual_backward(loss)
            opt_dec.step()

            dec_loss_last = loss
            dec_recon_last = recon
            dec_kl_last = kl

            self.update_old_posterior_ema(mu, logvar)

        self.log("train/dec_recon", dec_recon_last, on_epoch=True)
        self.log("train/dec_kl", dec_kl_last, on_epoch=True)

        # 모든 파라미터 그레디언트 복구
        for p in self.parameters():
            p.requires_grad = True

        # 스케줄러 업데이트
        if self.trainer.is_last_batch:
            val_loss = self.trainer.callback_metrics.get("val_loss")
            if val_loss is not None:
                sched_enc.step(val_loss)
                sched_dec.step(val_loss)

        # 로깅
        self.log("train/enc_loss", enc_loss_last, prog_bar=False)
        self.log("train/dec_loss", dec_loss_last, prog_bar=True)

        return dec_loss_last

    def validation_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        logits, mu, logvar = self(x)

        recon = self.recon_multinomial_ce(logits, x)
        kl = self.composite_prior_kl(mu, logvar)
        loss = recon + kl

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        datamodule = self.trainer.datamodule
        valid_gt = datamodule.get_validation_ground_truth()
        train_mat = datamodule.get_train_matrix()
        x = torch.FloatTensor(train_mat.toarray()).to(self.device)

        with torch.no_grad():
            logits, _, _ = self(x)

        scores = logits.clone()
        scores[x > 0] = -torch.inf

        K = 10
        topk = scores.topk(K, dim=1).indices

        hits = []
        for u, gt_items in valid_gt.items():
            if len(gt_items) == 0:
                continue
            pred_items = set(topk[u].tolist())
            gt_items = set(gt_items)
            hits.append(len(pred_items & gt_items) > 0)

        recall_at_10 = sum(hits) / len(hits) if hits else 0.0
        self.log("val_recall@10", recall_at_10, prog_bar=True)

    def configure_optimizers(self):
        enc_params = (
            list(self.encoder.parameters()) 
            + list(self.mu.parameters()) 
            + list(self.logvar.parameters())
        )
        dec_params = list(self.decoder.parameters())

        opt_enc = torch.optim.Adam(
            enc_params, 
            lr=self.lr_encoder, 
            weight_decay=self.weight_decay
        )
        opt_dec = torch.optim.Adam(
            dec_params, 
            lr=self.lr_decoder, 
            weight_decay=self.weight_decay
        )

        sched_enc = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt_enc, mode="min", factor=0.7, patience=20
        )
        sched_dec = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt_dec, mode="min", factor=0.7, patience=20
        )

        return [opt_enc, opt_dec], [
            {"scheduler": sched_enc, "monitor": "val_loss"},
            {"scheduler": sched_dec, "monitor": "val_loss"}
        ]
