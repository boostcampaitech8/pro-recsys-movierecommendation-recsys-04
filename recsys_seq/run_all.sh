#!/bin/bash
set -e

echo "=============================="
echo " Recommender System Training "
echo "=============================="

DATA_PATH="/data/ephemeral/home/minyou/data/raw/train/train_ratings.csv"
META_DIR="/data/ephemeral/home/minyou/data/raw/train"


echo "[2/3] SASRec + Content-augmented item embedding"
python run.py --train_path "$DATA_PATH" --meta_dir "$META_DIR" --model sasrec_content --device cpu

echo "[3/3] Hybrid (ALS candidate + content rerank)"
python run.py --train_path "$DATA_PATH" --meta_dir "$META_DIR" --model hybrid

echo "=============================="
echo " All models finished "
echo "=============================="
