"""
RecVAE Training Script with PyTorch Lightning (EMA Composite Prior)
Updated to support separate encoder_dims, decoder_dims, and latent_dim

Usage:
    python train_rec_vae.py
    python train_rec_vae.py model.encoder_dims=[2048,1024,512] model.latent_dim=300
    python train_rec_vae.py model.hidden_dims=[400,200] training.lr_encoder=0.001  # legacy support
"""

import os
import numpy as np
import pandas as pd
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import TensorBoardLogger
import torch

from src.data.recsys_data import RecSysDataModule
from src.models.rec_vae import RecVAE
from src.utils import get_directories

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="rec_vae")
def main(cfg: DictConfig):
    # Hydra output dir
    hydra_cfg = HydraConfig.get()
    log.info(f"Hydra output directory: {hydra_cfg.runtime.output_dir}")

    # Print config
    log.info("Configuration:")
    log.info(OmegaConf.to_yaml(cfg))

    # Seed
    L.seed_everything(cfg.seed, workers=True)

    # Float32 matmul precision
    torch.set_float32_matmul_precision(cfg.float32_matmul_precision)

    # ------------------------
    # DataModule
    # ------------------------
    log.info("Initializing DataModule...")
    datamodule = RecSysDataModule(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        valid_ratio=cfg.data.valid_ratio,
        min_interactions=cfg.data.min_interactions,
        seed=cfg.data.seed,
        data_file=cfg.data.data_file,
        split_strategy=cfg.data.split_strategy,
        temporal_split_ratio=cfg.data.temporal_split_ratio,
    )

    datamodule.setup()
    log.info(f"Number of users: {datamodule.num_users}")
    log.info(f"Number of items: {datamodule.num_items}")

    # ------------------------
    # Model (RecVAE) - Updated Parameter Handling
    # ------------------------
    log.info("Initializing RecVAE model...")

    recvae_cfg = cfg.training.recvae

    # ✅ 모델 파라미터 동적 구성
    model_params = {
        'num_items': datamodule.num_items,
        'dropout': cfg.model.dropout,
        # optimizer
        'lr_encoder': cfg.training.lr_encoder,
        'lr_decoder': cfg.training.lr_decoder,
        'weight_decay': cfg.training.weight_decay,
        # RecVAE 핵심
        'alpha': recvae_cfg.alpha,
        'ema_decay': recvae_cfg.ema_decay,
        'enc_steps_per_iter': recvae_cfg.enc_steps_per_iter,
        'dec_steps_per_iter': recvae_cfg.dec_steps_per_iter,
        'encoder_input_corruption': recvae_cfg.encoder_input_corruption,
    }

    # ✅ 새로운 파라미터 우선 사용 (encoder_dims, decoder_dims, latent_dim)
    if 'encoder_dims' in cfg.model and cfg.model.encoder_dims is not None:
        model_params['encoder_dims'] = list(cfg.model.encoder_dims)
        log.info(f"Using encoder_dims: {model_params['encoder_dims']}")
    
    if 'decoder_dims' in cfg.model and cfg.model.decoder_dims is not None:
        model_params['decoder_dims'] = list(cfg.model.decoder_dims)
        log.info(f"Using decoder_dims: {model_params['decoder_dims']}")
    
    if 'latent_dim' in cfg.model and cfg.model.latent_dim is not None:
        model_params['latent_dim'] = int(cfg.model.latent_dim)
        log.info(f"Using latent_dim: {model_params['latent_dim']}")

    # ✅ 기존 hidden_dims 호환성 유지
    if 'hidden_dims' in cfg.model and cfg.model.hidden_dims is not None:
        model_params['hidden_dims'] = tuple(cfg.model.hidden_dims)
        log.info(f"Using hidden_dims (legacy): {model_params['hidden_dims']}")

    # 모델 생성
    model = RecVAE(**model_params)

    # ------------------------
    # Logging / Checkpoints
    # ------------------------
    checkpoint_dir, tensorboard_dir = get_directories(cfg, stage="fit")
    log.info(f"Checkpoint directory: {checkpoint_dir}")
    log.info(f"TensorBoard directory: {tensorboard_dir}")

    logger = TensorBoardLogger(
        save_dir=tensorboard_dir,
        name="",
    )

    callbacks = []

    # Save best model
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="rec-vae-{epoch:02d}-{val_loss:.4f}",
        monitor=cfg.checkpoint.monitor,
        mode=cfg.checkpoint.mode,
        save_top_k=cfg.checkpoint.save_top_k,
        save_last=False,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    # Early stopping
    if cfg.training.early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor=cfg.checkpoint.monitor,
                patience=cfg.training.early_stopping_patience,
                mode=cfg.checkpoint.mode,
                verbose=True,
            )
        )

    # LR monitor
    callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    # ------------------------
    # Trainer
    # ------------------------
    log.info("Initializing Trainer...")
    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        devices="auto",
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        val_check_interval=1.0,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # ------------------------
    # Train
    # ------------------------
    log.info("Starting RecVAE training...")
    trainer.fit(model, datamodule=datamodule)

    log.info("Training completed!")
    log.info(f"Best model path: {checkpoint_callback.best_model_path}")
    log.info(
        f"Best {cfg.checkpoint.monitor}: {checkpoint_callback.best_model_score:.4f}"
    )


if __name__ == "__main__":
    main()
