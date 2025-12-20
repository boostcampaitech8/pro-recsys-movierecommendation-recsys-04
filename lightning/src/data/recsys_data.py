import os
import random
import numpy as np
import pandas as pd
import lightning as L
import torch
from scipy import sparse
from torch.utils.data import DataLoader, TensorDataset


class RecSysDataModule(L.LightningDataModule):
    """
    RecSys 데이터 모듈 (Collaborative Filtering)

    Hydra 설정 사용:
        data:
            data_dir: "~/data/train/"
            batch_size: 512
            valid_ratio: 0.1
            min_interactions: 5
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 512,
        valid_ratio: float = 0.1,
        min_interactions: int = 5,
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = os.path.expanduser(data_dir)
        self.batch_size = batch_size
        self.valid_ratio = valid_ratio
        self.min_interactions = min_interactions
        self.seed = seed

        # 인코딩 매핑 (setup 후 사용 가능)
        self.user2idx = None  # user_id -> user_idx
        self.idx2user = None  # user_idx -> user_id
        self.item2idx = None  # item_id -> item_idx
        self.idx2item = None  # item_idx -> item_id

        self.num_users = None
        self.num_items = None

        self.train_mat = None  # CSR sparse matrix (num_users, num_items)
        self.valid_gt = None  # dict: {user_idx: [item_idx, ...]}

    def prepare_data(self):
        """데이터 다운로드 및 전처리 (단일 프로세스에서만 실행)"""
        # 이미 로컬에 있는 데이터 사용
        pass

    def setup(self, stage: str = None):
        """데이터 로드 및 분할"""
        # 1. 상호작용 데이터 읽기
        df = self._read_interactions()

        # 2. ID 인코딩 (user_id, item_id -> 연속적인 인덱스)
        df_enc = self._encode_ids(df)

        # 3. Train/Valid 분할 (랜덤 마스킹)
        if self.valid_ratio > 0:
            train_df, self.valid_gt = self._train_valid_split(df_enc)
        else:
            train_df = df_enc
            self.valid_gt = {u: [] for u in range(self.num_users)}

        # 4. Sparse Matrix 생성 (CSR 형식)
        self.train_mat = self._build_user_item_matrix(train_df)

    def _read_interactions(self):
        """train_ratings.csv 읽기 및 필터링"""
        file_path = os.path.join(self.data_dir, "train_ratings.csv")
        df = pd.read_csv(file_path)

        # 최소 상호작용 수 필터링
        if self.min_interactions > 0:
            user_counts = df["user"].value_counts()
            valid_users = user_counts[user_counts >= self.min_interactions].index
            df = df[df["user"].isin(valid_users)]

        return df

    def _encode_ids(self, df):
        """user_id, item_id를 0부터 시작하는 연속적인 인덱스로 변환"""
        unique_users = sorted(df["user"].unique())
        unique_items = sorted(df["item"].unique())

        self.user2idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.idx2user = {idx: uid for uid, idx in self.user2idx.items()}
        self.item2idx = {iid: idx for idx, iid in enumerate(unique_items)}
        self.idx2item = {idx: iid for iid, idx in self.item2idx.items()}

        self.num_users = len(self.user2idx)
        self.num_items = len(self.item2idx)

        df_enc = df.copy()
        df_enc["user"] = df["user"].map(self.user2idx)
        df_enc["item"] = df["item"].map(self.item2idx)

        return df_enc

    def _train_valid_split(self, df_enc):
        """각 유저별로 랜덤하게 valid_ratio 비율만큼 validation set으로 분할"""
        random.seed(self.seed)
        np.random.seed(self.seed)

        train_rows = []
        valid_gt = {u: [] for u in range(self.num_users)}

        for u_idx in range(self.num_users):
            user_items = df_enc[df_enc["user"] == u_idx]["item"].tolist()
            n_items = len(user_items)

            if n_items == 0:
                continue

            # validation 개수 계산
            n_valid = max(1, int(n_items * self.valid_ratio))

            # 랜덤 샘플링
            valid_items = random.sample(user_items, n_valid)
            train_items = [it for it in user_items if it not in valid_items]

            # validation ground truth 저장
            valid_gt[u_idx] = valid_items

            # train 데이터 저장
            for it in train_items:
                train_rows.append({"user": u_idx, "item": it})

        train_df = pd.DataFrame(train_rows)
        return train_df, valid_gt

    def _build_user_item_matrix(self, df):
        """DataFrame -> Sparse CSR Matrix (num_users, num_items)"""
        rows = df["user"].values
        cols = df["item"].values
        data = np.ones(len(df))

        mat = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(self.num_users, self.num_items),
            dtype=np.float32,
        )

        return mat

    def train_dataloader(self):
        """학습용 DataLoader: CSR matrix를 dense tensor로 변환하여 배치 생성"""
        # CSR -> Dense Tensor
        train_dense = torch.FloatTensor(self.train_mat.toarray())
        dataset = TensorDataset(train_dense)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self):
        """검증용 DataLoader: 동일한 데이터 사용 (validation은 메트릭 계산으로 평가)"""
        train_dense = torch.FloatTensor(self.train_mat.toarray())
        dataset = TensorDataset(train_dense)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    def get_validation_ground_truth(self):
        """Validation ground truth 반환 (메트릭 계산용)"""
        return self.valid_gt

    def get_train_matrix(self):
        """학습 데이터 행렬 반환 (추천 생성 시 사용)"""
        return self.train_mat
