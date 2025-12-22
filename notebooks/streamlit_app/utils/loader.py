import os
import json
import pandas as pd
import streamlit as st


@st.cache_data
def load_csv_with_sampling(path: str, sample_size: int = None):
    """CSV를 로드하고 필요 시 샘플링해서 반환."""
    df = pd.read_csv(path)

    if sample_size is not None and len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)

    return df


@st.cache_data
def load_tsv(path: str):
    """TSV 파일 로딩 (크지 않으므로 샘플링 불필요)"""
    return pd.read_csv(path, sep="\t")


@st.cache_data
def load_json(path: str):
    """JSON 파일 로딩 (캐싱만)"""
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def load_all_data(base_path: str, sample_size: int = 500_000):
    """
    base_path 아래 데이터셋을 로드해서 dict로 반환.
    ratings는 크므로 sample_size만큼 샘플링.
    나머지는 그대로 로드.
    """
    data = {}

    # 매우 큰 파일 → 샘플링
    ratings_path = os.path.join(base_path, "train_ratings.csv")
    data["ratings"] = load_csv_with_sampling(ratings_path, sample_size=sample_size)

    # tsv들 (상대적으로 작음 → 전체 로드)
    data["titles"] = load_tsv(os.path.join(base_path, "titles.tsv"))
    data["years"] = load_tsv(os.path.join(base_path, "years.tsv"))
    data["genres"] = load_tsv(os.path.join(base_path, "genres.tsv"))
    data["directors"] = load_tsv(os.path.join(base_path, "directors.tsv"))
    data["writers"] = load_tsv(os.path.join(base_path, "writers.tsv"))

    # JSON mapping
    json_path = os.path.join(base_path, "Ml_item2attributes.json")
    data["item2attributes"] = load_json(json_path)

    return data
