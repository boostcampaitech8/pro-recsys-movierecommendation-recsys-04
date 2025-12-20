import os
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
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

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="multi_vae")
def main(cfg: DictConfig):
    # 설정 출력
    log.info("=" * 80)
    log.info("Configuration:")
    log.info(OmegaConf.to_yaml(cfg))
    log.info("=" * 80)

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
        data_file=cfg.data.data_file,
        split_strategy=cfg.data.split_strategy,
        temporal_split_ratio=cfg.data.get("temporal_split_ratio", 0.8),
    )

    # 모델 초기화를 위해 setup 호출 (num_users, num_items 필요)
    data_module.setup()

    log.info(f"Number of users: {data_module.num_users}")
    log.info(f"Number of items: {data_module.num_items}")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.checkpoint.checkpoint_dir,
        # filename="multi-vae-{epoch:02d}-{val_loss:.4f}",
        filename="multi-vae-epoch=469-val_loss=888.5256.ckpt",
        monitor=cfg.checkpoint.monitor,
        mode=cfg.checkpoint.mode,
        save_top_k=cfg.checkpoint.save_top_k,
        save_last=True,
    )

    # Best 체크포인트 로드
    log.info("Loading best checkpoint...")
    best_model = MultiVAE.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        num_items=data_module.num_items,
        weights_only=False,
    )

    # Top-K 추천 생성
    log.info(f"Generating Top-{cfg.recommend.topk} recommendations...")
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
    log.info(f"Validation Recall@{cfg.recommend.topk}: {recall:.4f}")

    # Submission 파일 생성
    log.info("Creating submission file...")
    submission_dir = os.path.join(cfg.trainer.default_root_dir, "submissions")
    os.makedirs(submission_dir, exist_ok=True)

    rows = []
    for u_idx in range(data_module.num_users):
        for item_idx in recommendations[u_idx]:
            user_id = data_module.idx2user[u_idx]
            item_id = data_module.idx2item[int(item_idx)]
            rows.append((user_id, item_id))

    submission_df = pd.DataFrame(rows, columns=["user", "item"])
    submission_path = os.path.join(submission_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    log.info(f"Submission saved to: {submission_path}")
    log.info("✅ All processes finished!")


if __name__ == "__main__":
    main()
