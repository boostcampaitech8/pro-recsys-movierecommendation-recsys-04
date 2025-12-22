#!/bin/bash
set -e

echo "=============================="
echo " Recommender System Training "
echo "=============================="

# ✅ 변수 선언 (공백 절대 없음)
DATA_PATH="/data/ephemeral/home/minyou/data/raw/train/train_ratings.csv"

echo "[1/3] Item-based CF"
python run.py --train_path "$DATA_PATH" --model itemcf

echo "[2/3] User-based CF"
python run.py --train_path "$DATA_PATH" --model usercf

echo "[3/3] Matrix Factorization (ALS)"
python run.py --train_path "$DATA_PATH" --model als

echo "=============================="
echo " All models finished "
echo "=============================="
