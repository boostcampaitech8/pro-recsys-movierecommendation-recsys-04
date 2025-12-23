from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

def load_interactions(path: str) -> pd.DataFrame:
    """train_ratings.csv: user,item,time"""
    df = pd.read_csv(path)
    # sequence 학습/last-item split에 정렬이 중요
    df = df.sort_values(["user", "time"]).reset_index(drop=True)
    return df

def make_id_mappings(df: pd.DataFrame):
    users = df["user"].unique()
    items = df["item"].unique()
    user2idx = {u: i for i, u in enumerate(users)}
    item2idx = {it: i for i, it in enumerate(items)}
    idx2user = {i: u for u, i in user2idx.items()}
    idx2item = {i: it for it, i in item2idx.items()}
    return user2idx, item2idx, idx2user, idx2item

def apply_id_mappings(df: pd.DataFrame, user2idx, item2idx) -> pd.DataFrame:
    out = df.copy()
    out["u"] = out["user"].map(user2idx).astype(np.int64)
    out["i"] = out["item"].map(item2idx).astype(np.int64)
    return out

def split_last_item_per_user(df_ui: pd.DataFrame):
    """user별 마지막 interaction을 validation target으로 분리."""
    last_idx = df_ui.groupby("u")["time"].idxmax()
    valid_df = df_ui.loc[last_idx].copy()
    train_df = df_ui.drop(index=last_idx).copy()
    train_users = set(train_df["u"].unique())
    valid_df = valid_df[valid_df["u"].isin(train_users)].copy()
    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True)

def build_implicit_matrix(df_ui: pd.DataFrame, n_users: int, n_items: int) -> csr_matrix:
    rows = df_ui["u"].to_numpy()
    cols = df_ui["i"].to_numpy()
    data = np.ones(len(df_ui), dtype=np.float32)
    mat = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
    mat.sum_duplicates()
    return mat

# ---------------- side information ----------------

@dataclass
class SideInfo:
    # item index 기준 (0..n_items-1)
    item_genres: List[List[int]]
    item_directors: List[List[int]]
    item_writers: List[List[int]]
    n_genres: int
    n_directors: int
    n_writers: int

def _load_tsv_list(path: str, item_col: str="item", value_col: str="genre") -> Dict[int, List[str]]:
    df = pd.read_csv(path, sep="\t")
    # item은 원본 item id
    g = df.groupby(item_col)[value_col].apply(list).to_dict()
    return g

def load_item_sideinfo(
    base_dir: str,
    item2idx: Dict[int, int],
    genres_tsv: str = "genres.tsv",
    directors_tsv: str = "directors.tsv",
    writers_tsv: str = "writers.tsv",
) -> SideInfo:
    """TSV(원본 item id) -> model item index에 맞춰 sideinfo list를 만든다."""
    genres_map = _load_tsv_list(os.path.join(base_dir, genres_tsv), value_col="genre")
    directors_map = _load_tsv_list(os.path.join(base_dir, directors_tsv), value_col="director")
    writers_map = _load_tsv_list(os.path.join(base_dir, writers_tsv), value_col="writer")

    # feature vocab 만들기
    all_genres = sorted({x for lst in genres_map.values() for x in lst})
    all_directors = sorted({x for lst in directors_map.values() for x in lst})
    all_writers = sorted({x for lst in writers_map.values() for x in lst})

    genre2idx = {g:i+1 for i,g in enumerate(all_genres)}        # 0은 PAD
    director2idx = {d:i+1 for i,d in enumerate(all_directors)}  # 0은 PAD
    writer2idx = {w:i+1 for i,w in enumerate(all_writers)}      # 0은 PAD

    n_items = len(item2idx)
    item_genres = [[] for _ in range(n_items)]
    item_directors = [[] for _ in range(n_items)]
    item_writers = [[] for _ in range(n_items)]

    for raw_item, iidx in item2idx.items():
        gs = genres_map.get(raw_item, [])
        ds = directors_map.get(raw_item, [])
        ws = writers_map.get(raw_item, [])
        item_genres[iidx] = [genre2idx[g] for g in gs if g in genre2idx]
        item_directors[iidx] = [director2idx[d] for d in ds if d in director2idx]
        item_writers[iidx] = [writer2idx[w] for w in ws if w in writer2idx]

    return SideInfo(
        item_genres=item_genres,
        item_directors=item_directors,
        item_writers=item_writers,
        n_genres=len(genre2idx)+1,
        n_directors=len(director2idx)+1,
        n_writers=len(writer2idx)+1,
    )
