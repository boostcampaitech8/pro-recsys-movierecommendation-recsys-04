"""
Offline Evaluation Script for Ensemble (Valid Recall@10)

- Input  : train sequence
- GT     : valid (next item)
- Exclude: train only
- Metric : Recall@10
"""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import torch
from tqdm import tqdm

from src.models.bert4rec import BERT4Rec
from src.data.bert4rec_data import BERT4RecDataModule

log = logging.getLogger(__name__)


# =========================
# Utility: auto 탐색
# =========================
def find_latest_run(model_name: str) -> Path:
    base = Path("saved/hydra_logs") / model_name
    runs = list(base.glob("*/*"))
    if not runs:
        raise FileNotFoundError(f"No runs found for model: {model_name}")
    return max(runs, key=lambda p: p.stat().st_mtime)


def find_latest_checkpoint(model_name: str) -> Path:
    run = find_latest_run(model_name)
    ckpts = list((run / "checkpoints").glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found for {model_name}")
    return ckpts[0]


def find_latest_submission_csv(model_name: str) -> Path:
    run = find_latest_run(model_name)
    csvs = list((run / "submissions").glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No submission csv found for {model_name}")
    return max(csvs, key=lambda p: p.stat().st_mtime)


# =========================
# Ratio Logic
# =========================
def calculate_static_limit(item_count: int, k: int = 10) -> int:
    ratio = item_count / 100
    if ratio >= k:
        return k - 1
    return int((ratio / (ratio + 1)) * k)


# =========================
# Main
# =========================
@hydra.main(version_base=None, config_path="configs", config_name="ensemble_user_ratio")
def main(cfg: DictConfig):

    log.info("Configuration:")
    log.info(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================
    # DataModule
    # =========================
    datamodule = BERT4RecDataModule(
        data_dir=cfg.data.data_dir,
        data_file=cfg.data.data_file,
        batch_size=cfg.inference.batch_size,
        max_len=cfg.model.max_len,
        mask_prob=0.0,
        min_interactions=cfg.data.min_interactions,
        seed=cfg.data.seed,
        num_workers=cfg.data.num_workers,
    )
    datamodule.setup()

    # =========================
    # Load BERT
    # =========================
    bert_ckpt = find_latest_checkpoint(cfg.ensemble.bert_model_name)
    log.info(f"Using BERT checkpoint: {bert_ckpt}")

    bert_model = BERT4Rec.load_from_checkpoint(bert_ckpt)
    bert_model.eval().to(device)

    # =========================
    # Load EASE
    # =========================
    ease_csv = find_latest_submission_csv(cfg.ensemble.ease_model_name)
    log.info(f"Using EASE csv: {ease_csv}")

    ease_df = pd.read_csv(ease_csv)
    ease_grouped = ease_df.groupby("user")["item"].apply(list).to_dict()

    # =========================
    # User interaction count
    # =========================
    train_df = pd.read_csv(Path(cfg.data.data_dir).expanduser() / cfg.data.data_file)
    user_item_count = train_df.groupby("user")["item"].count().to_dict()

    # =========================
    # Evaluation
    # =========================
    hits = 0
    total = 0

    for user_idx in tqdm(datamodule.user_train.keys(), desc="Evaluating"):
        user_id = datamodule.idx2user[user_idx]

        train_seq = datamodule.user_train[user_idx]
        gt_item_idx = datamodule.user_valid[user_idx]
        gt_item = datamodule.idx2item[gt_item_idx]

        exclude = set(train_seq)

        # ----- BERT -----
        bert_items = bert_model.predict(
            [train_seq],
            topk=cfg.ensemble.bert_topk,
            exclude_items=[exclude],
        )[0]
        bert_items = [
            datamodule.idx2item[i] for i in bert_items if 1 <= i <= datamodule.num_items
        ]

        # ----- EASE -----
        ease_items = ease_grouped.get(user_id, [])

        # ----- Ratio -----
        count = user_item_count.get(user_id, 0)
        static_k = calculate_static_limit(count, k=cfg.inference.topk)

        merged = []

        for it in ease_items:
            if len(merged) >= static_k:
                break
            merged.append(it)

        for it in bert_items:
            if len(merged) >= cfg.inference.topk:
                break
            if it not in merged:
                merged.append(it)

        if gt_item in merged:
            hits += 1
        total += 1

    recall_at_10 = hits / total
    log.info(f"Offline Valid Recall@10 = {recall_at_10:.5f}")


if __name__ == "__main__":
    main()
