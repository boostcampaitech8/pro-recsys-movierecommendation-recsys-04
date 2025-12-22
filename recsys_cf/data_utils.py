from __future__ import annotations

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix


def load_interactions(path: str) -> pd.DataFrame:
    """
    train_ratings.csv: user,item,time
    """
    df = pd.read_csv(path)
    # 정렬이 중요 (sequence / last-item split)
    df = df.sort_values(["user", "time"]).reset_index(drop=True)
    return df


def make_id_mappings(df: pd.DataFrame):
    """
    원본 user/item id -> 연속 index
    """
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
    """
    user별 마지막 interaction을 validation target으로 분리.
    df_ui는 이미 (user,time) 정렬된 상태가 바람직.
    반환: train_df, valid_df
      - train_df: 마지막 1개 제거된 interaction
      - valid_df: user별 마지막 1개 (정답)
    """
    # 각 user 그룹의 마지막 인덱스
    last_idx = df_ui.groupby("u")["time"].idxmax()
    valid_df = df_ui.loc[last_idx].copy()
    train_df = df_ui.drop(index=last_idx).copy()

    # 혹시 train이 비는 유저가 생길 수 있음(상호작용 1개 유저)
    # 그런 유저는 valid에서도 제외(평가/추천이 의미없음)
    train_users = set(train_df["u"].unique())
    valid_df = valid_df[valid_df["u"].isin(train_users)].copy()

    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True)


def build_implicit_matrix(df_ui: pd.DataFrame, n_users: int, n_items: int) -> csr_matrix:
    """
    implicit feedback matrix (u,i) = 1
    """
    rows = df_ui["u"].to_numpy()
    cols = df_ui["i"].to_numpy()
    data = np.ones(len(df_ui), dtype=np.float32)
    mat = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
    mat.sum_duplicates()
    return mat
