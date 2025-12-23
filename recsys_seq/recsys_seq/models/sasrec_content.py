from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ..data_utils import SideInfo

def _pad_trunc(lst: List[int], max_len: int) -> List[int]:
    if len(lst) >= max_len:
        return lst[:max_len]
    return lst + [0] * (max_len - len(lst))

class ItemEmbeddingWithContent(nn.Module):
    def __init__(
        self,
        n_items: int,
        n_genres: int,
        n_directors: int,
        n_writers: int,
        embed_dim: int,
        max_genres: int = 5,
        max_directors: int = 5,
        max_writers: int = 5,
    ):
        super().__init__()
        self.max_genres = max_genres
        self.max_directors = max_directors
        self.max_writers = max_writers

        self.item_emb = nn.Embedding(n_items, embed_dim, padding_idx=0)
        self.genre_emb = nn.Embedding(n_genres, embed_dim, padding_idx=0)
        self.director_emb = nn.Embedding(n_directors, embed_dim, padding_idx=0)
        self.writer_emb = nn.Embedding(n_writers, embed_dim, padding_idx=0)

        self.proj = nn.Linear(embed_dim * 4, embed_dim)

    def forward(
        self,
        item_ids: torch.Tensor,        # (B,T)
        genre_ids: torch.Tensor,       # (B,T,G)
        director_ids: torch.Tensor,    # (B,T,D)
        writer_ids: torch.Tensor,      # (B,T,W)
    ) -> torch.Tensor:
        item_e = self.item_emb(item_ids)  # (B,T,E)
        genre_e = self.genre_emb(genre_ids).mean(dim=2)
        director_e = self.director_emb(director_ids).mean(dim=2)
        writer_e = self.writer_emb(writer_ids).mean(dim=2)
        x = torch.cat([item_e, genre_e, director_e, writer_e], dim=-1)
        return self.proj(x)

class SASRecEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_layers: int, dropout: float):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        # key_padding_mask: True for PAD positions
        return self.encoder(x, src_key_padding_mask=key_padding_mask)

class SASRecTrainDataset(Dataset):
    def __init__(
        self,
        user_seqs: List[List[int]],
        sideinfo: SideInfo,
        max_seq_len: int,
        max_genres: int,
        max_directors: int,
        max_writers: int,
        n_items: int,
        num_neg: int,
        seed: int = 42,
    ):
        self.user_seqs = user_seqs
        self.sideinfo = sideinfo
        self.max_seq_len = max_seq_len
        self.max_genres = max_genres
        self.max_directors = max_directors
        self.max_writers = max_writers
        self.n_items = n_items
        self.num_neg = num_neg
        self.rng = np.random.default_rng(seed)

        # (u, t) pairs: predict seq[t] from prefix <t
        self.samples = []
        for u, seq in enumerate(user_seqs):
            if len(seq) < 2:
                continue
            for t in range(1, len(seq)):
                self.samples.append((u, t))

        # precompute seen sets for negative sampling
        self.user_seen = [set(s) for s in user_seqs]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        u, t = self.samples[idx]
        seq = self.user_seqs[u]

        target = seq[t]
        prefix = seq[:t]

        # last max_seq_len items, left-pad with 0
        prefix = prefix[-self.max_seq_len:]
        pad_len = self.max_seq_len - len(prefix)
        items = [0] * pad_len + prefix  # (T,)
        items = np.array(items, dtype=np.int64)

        # build side feature tensors per position
        genres = []
        directors = []
        writers = []
        for it in items.tolist():
            if it == 0:
                genres.append([0]*self.max_genres)
                directors.append([0]*self.max_directors)
                writers.append([0]*self.max_writers)
            else:
                genres.append(_pad_trunc(self.sideinfo.item_genres[it], self.max_genres))
                directors.append(_pad_trunc(self.sideinfo.item_directors[it], self.max_directors))
                writers.append(_pad_trunc(self.sideinfo.item_writers[it], self.max_writers))

        genres = np.array(genres, dtype=np.int64)
        directors = np.array(directors, dtype=np.int64)
        writers = np.array(writers, dtype=np.int64)

        # negatives
        negs = []
        seen = self.user_seen[u]
        while len(negs) < self.num_neg:
            j = int(self.rng.integers(1, self.n_items))
            if j not in seen:
                negs.append(j)
        negs = np.array(negs, dtype=np.int64)

        return items, genres, directors, writers, np.int64(target), negs

@dataclass
class SASRecConfig:
    max_seq_len: int = 50
    embed_dim: int = 64
    num_heads: int = 2
    num_layers: int = 2
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 256
    epochs: int = 3
    num_neg: int = 50
    max_genres: int = 5
    max_directors: int = 5
    max_writers: int = 5
    device: str = "cpu"

class SASRecWithContent:
    """SASRec + item side-info를 item embedding에 흡수한 버전."""
    def __init__(self, n_items: int, sideinfo: SideInfo, cfg: SASRecConfig):
        self.n_items = n_items
        self.sideinfo = sideinfo
        self.cfg = cfg

        self.embedding = ItemEmbeddingWithContent(
            n_items=n_items,
            n_genres=sideinfo.n_genres,
            n_directors=sideinfo.n_directors,
            n_writers=sideinfo.n_writers,
            embed_dim=cfg.embed_dim,
            max_genres=cfg.max_genres,
            max_directors=cfg.max_directors,
            max_writers=cfg.max_writers,
        )
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.embed_dim)
        self.encoder = SASRecEncoder(cfg.embed_dim, cfg.num_heads, cfg.num_layers, cfg.dropout)

        self.to(cfg.device)

    def to(self, device: str):
        self.cfg.device = device
        self.embedding.to(device)
        self.pos_emb.to(device)
        self.encoder.to(device)

    def fit_sequences(self, user_seqs: List[List[int]], seed: int = 42):
        """user_seqs: item index sequence per user (item idx 기준, 1..n_items-1; 0 pad)"""
        cfg = self.cfg
        ds = SASRecTrainDataset(
            user_seqs=user_seqs,
            sideinfo=self.sideinfo,
            max_seq_len=cfg.max_seq_len,
            max_genres=cfg.max_genres,
            max_directors=cfg.max_directors,
            max_writers=cfg.max_writers,
            n_items=self.n_items,
            num_neg=cfg.num_neg,
            seed=seed,
        )
        dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

        params = list(self.embedding.parameters()) + list(self.pos_emb.parameters()) + list(self.encoder.parameters())
        opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

        for epoch in range(1, cfg.epochs + 1):
            self.embedding.train(); self.pos_emb.train(); self.encoder.train()
            losses = []
            for items, genres, directors, writers, target, negs in dl:
                items = items.to(cfg.device)
                genres = genres.to(cfg.device)
                directors = directors.to(cfg.device)
                writers = writers.to(cfg.device)
                target = target.to(cfg.device)
                negs = negs.to(cfg.device)

                # embeddings
                seq_emb = self.embedding(items, genres, directors, writers)  # (B,T,E)
                # add pos
                pos = torch.arange(cfg.max_seq_len, device=cfg.device).unsqueeze(0).expand(items.size(0), -1)
                seq_emb = seq_emb + self.pos_emb(pos)

                # mask pad positions
                key_padding_mask = (items == 0)  # (B,T)
                h = self.encoder(seq_emb, key_padding_mask=key_padding_mask)  # (B,T,E)
                # take last position (predict next from last token in prefix)
                last_h = h[:, -1, :]  # (B,E)

                # positive score
                pos_e = self.embedding.item_emb(target)  # (B,E)
                pos_score = (last_h * pos_e).sum(dim=-1)  # (B,)

                # negative scores
                neg_e = self.embedding.item_emb(negs)  # (B,neg,E)
                neg_score = torch.einsum("be,bne->bn", last_h, neg_e)  # (B,neg)

                # sampled softmax loss
                logits = torch.cat([pos_score.unsqueeze(1), neg_score], dim=1)  # (B,1+neg)
                labels = torch.zeros(items.size(0), dtype=torch.long, device=cfg.device)
                loss = nn.CrossEntropyLoss()(logits, labels)

                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss.detach().cpu().item()))

            print(f"[SASRec] epoch {epoch}/{cfg.epochs} loss={np.mean(losses):.4f}")

    @torch.no_grad()
    def recommend_with_scores(self, user: int, user_seq: List[int], k: int = 10, seen: Optional[set[int]] = None):
        cfg = self.cfg
        self.embedding.eval(); self.pos_emb.eval(); self.encoder.eval()

        seq = user_seq[-cfg.max_seq_len:]
        pad_len = cfg.max_seq_len - len(seq)
        items = [0]*pad_len + seq

        genres = []
        directors = []
        writers = []
        for it in items:
            if it == 0:
                genres.append([0]*cfg.max_genres)
                directors.append([0]*cfg.max_directors)
                writers.append([0]*cfg.max_writers)
            else:
                genres.append(_pad_trunc(self.sideinfo.item_genres[it], cfg.max_genres))
                directors.append(_pad_trunc(self.sideinfo.item_directors[it], cfg.max_directors))
                writers.append(_pad_trunc(self.sideinfo.item_writers[it], cfg.max_writers))

        items_t = torch.tensor(items, dtype=torch.long, device=cfg.device).unsqueeze(0)
        genres_t = torch.tensor(genres, dtype=torch.long, device=cfg.device).unsqueeze(0)
        directors_t = torch.tensor(directors, dtype=torch.long, device=cfg.device).unsqueeze(0)
        writers_t = torch.tensor(writers, dtype=torch.long, device=cfg.device).unsqueeze(0)

        seq_emb = self.embedding(items_t, genres_t, directors_t, writers_t)
        pos = torch.arange(cfg.max_seq_len, device=cfg.device).unsqueeze(0)
        seq_emb = seq_emb + self.pos_emb(pos)

        key_padding_mask = (items_t == 0)
        h = self.encoder(seq_emb, key_padding_mask=key_padding_mask)
        last_h = h[:, -1, :]  # (1,E)

        # score all items (1..n_items-1). 0은 PAD
        item_mat = self.embedding.item_emb.weight  # (n_items, E)
        scores = (last_h @ item_mat.T).squeeze(0).detach().cpu().numpy().astype(np.float32)

        # remove PAD and seen
        scores[0] = -np.inf
        if seen:
            scores[list(seen)] = -np.inf

        k_eff = min(k, len(scores))
        top_items = np.argpartition(-scores, kth=k_eff - 1)[:k_eff]
        top_items = top_items[np.argsort(-scores[top_items])]
        return top_items.tolist(), scores[top_items].tolist()

    def recommend(self, user: int, user_seq: List[int], k: int = 10, seen: Optional[set[int]] = None) -> List[int]:
        items, _ = self.recommend_with_scores(user, user_seq, k=k, seen=seen)
        return items
