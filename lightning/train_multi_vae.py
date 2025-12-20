import os
import logging
import hydra
from omegaconf import DictConfig
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
import torch
import pandas as pd
import numpy as np

from src.data.recsys_data import RecSysDataModule
from src.models.multi_vae import MultiVAE
from src.utils.recommend import recommend_topk
from src.utils.metrics import recall_at_k


@hydra.main(version_base=None, config_path="configs", config_name="multi_vae")
def main(cfg: DictConfig):
    # 로그 포맷 설정
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level),
        format=cfg.logging.format,
        datefmt=cfg.logging.datefmt,
        force=True,
    )

    # 시드 고정
    L.seed_everything(cfg.seed)

    # Float32 matmul precision 설정
    torch.set_float32_matmul_precision(cfg.float32_matmul_precision)

    # DataModule 초기화
    logging.info("Initializing RecSys DataModule...")
    data_module = RecSysDataModule(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        valid_ratio=cfg.data.valid_ratio,
        min_interactions=cfg.data.min_interactions,
        seed=cfg.seed,
    )
    data_module.setup()

    logging.info(f"Number of users: {data_module.num_users}")
    logging.info(f"Number of items: {data_module.num_items}")

    # 모델 초기화
    logging.info("Initializing MultiVAE model...")
    model = MultiVAE(
        num_items=data_module.num_items,
        hidden_dims=cfg.model.hidden_dims,  # [ 600, 200]
        dropout=cfg.model.dropout,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        kl_max_weight=cfg.training.kl_max_weight,
        kl_anneal_steps=cfg.training.kl_anneal_steps,
    )

    # TensorBoard Logger
    logger = TensorBoardLogger(
        save_dir=cfg.logging.save_dir,
        name=cfg.logging.name,
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.logging.checkpoint_dir,
        filename="multi-vae-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=cfg.logging.save_top_k,
        save_last=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=20,
        mode="min",
        verbose=True,
    )

    # Trainer 설정
    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        devices=cfg.trainer.devices,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        val_check_interval=cfg.trainer.val_check_interval,
        enable_progress_bar=cfg.trainer.enable_progress_bar,
        enable_model_summary=cfg.trainer.enable_model_summary,
        default_root_dir=cfg.logging.default_root_dir,
    )

    # 학습
    logging.info("Starting training...")
    trainer.fit(model, data_module)

    # Best 체크포인트 로드
    logging.info("Loading best checkpoint...")
    best_model = MultiVAE.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        num_items=data_module.num_items,
    )

    # Top-K 추천 생성
    logging.info(f"Generating Top-{cfg.recommend.topk} recommendations...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_mat = data_module.get_train_matrix()

    recommendations = recommend_topk(
        best_model,
        train_mat,
        k=cfg.recommend.topk,
        device=device,
        batch_size=cfg.data.batch_size,
    )

    # Validation Recall@K 계산
    valid_gt = data_module.get_validation_ground_truth()
    valid_gt_list = [valid_gt[u] for u in range(data_module.num_users)]
    pred_list = [rec.tolist() for rec in recommendations]

    recall = recall_at_k(valid_gt_list, pred_list, k=cfg.recommend.topk)
    logging.info(f"Validation Recall@{cfg.recommend.topk}: {recall:.4f}")

    # Submission 파일 생성
    logging.info("Creating submission file...")
    output_dir = os.path.join(cfg.logging.default_root_dir, "submissions")
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for u_idx in range(data_module.num_users):
        for item_idx in recommendations[u_idx]:
            user_id = data_module.idx2user[u_idx]
            item_id = data_module.idx2item[int(item_idx)]
            rows.append((user_id, item_id))

    submission_df = pd.DataFrame(rows, columns=["user", "item"])
    submission_path = os.path.join(output_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    logging.info(f"Submission saved to: {submission_path}")
    logging.info("✅ All processes finished!")


if __name__ == "__main__":
    main()
