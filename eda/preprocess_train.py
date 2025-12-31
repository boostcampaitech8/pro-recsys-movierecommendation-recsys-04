# ============================================================
# Full-data EASE Training Script (Final)
# - No validation
# - BM25 weighting
# - Optional popularity penalty at inference
# ============================================================

import numpy as np
import pandas as pd
from scipy import sparse
from pathlib import Path

# ======================
# Config
# ======================
DATA_PATH = "/data/ephemeral/home/Seung/data/train/train_ratings.csv"   # 경로 확인
OUT_PATH = "/data/ephemeral/home/Seung/pro-recsys-movierecommendation-recsys-04/eda/submission_ease_full.csv"

LAMBDA = 500.0        # validation에서 고른 값
K = 10
USE_POP_PENALTY = False
POP_ALPHA = 0.15      # validation에서 괜찮았던 값

SEED = 42
np.random.seed(SEED)

# ======================
# Utils
# ======================
def bm25_weight(X, K1=1.2, B=0.75):
    X = X.tocsr().astype(np.float32)
    N = X.shape[0]

    # item DF
    df = np.diff(X.tocsc().indptr)
    idf = np.log((N - df + 0.5) / (df + 0.5))
    idf = np.maximum(idf, 0).astype(np.float32)

    # user length
    row_sum = np.array(X.sum(axis=1)).ravel()
    avg_len = row_sum.mean()

    X = X.tocoo()
    denom = X.data + K1 * (1 - B + B * row_sum[X.row] / (avg_len + 1e-8))
    data = X.data * (K1 + 1) / (denom + 1e-8)
    data *= idf[X.col]

    return sparse.csr_matrix((data, (X.row, X.col)), shape=X.shape)

def fit_ease(X, lam):
    print("▶ Computing G = XᵀX")
    G = (X.T @ X).toarray().astype(np.float32)

    print("▶ Adding lambda to diagonal")
    diag = np.diag_indices_from(G)
    G[diag] += lam

    print("▶ Inverting matrix (this may take time)")
    P = np.linalg.inv(G)

    print("▶ Computing B matrix")
    B = -P / np.diag(P)
    B[diag] = 0.0
    return B

def predict_topk(X, B, k=10, pop_penalty=None, alpha=0.0):
    X = X.tocsr()

    scores = np.asarray(X @ B, dtype=np.float32)

    if pop_penalty is not None:
        scores -= alpha * pop_penalty[None, :]

    # remove seen
    for u in range(X.shape[0]):
        scores[u, X[u].indices] = -1e9

    # top-k
    topk = np.argpartition(-scores, kth=k-1, axis=1)[:, :k]
    topk_scores = scores[np.arange(scores.shape[0])[:, None], topk]
    order = np.argsort(-topk_scores, axis=1)
    topk = topk[np.arange(topk.shape[0])[:, None], order]

    return topk.astype(np.int32)

# ======================
# 1. Load data
# ======================
print("▶ Loading train data")
train = pd.read_csv(DATA_PATH)
print("shape:", train.shape)
print(train.head())

# ======================
# 2. Encode IDs
# ======================
print("▶ Encoding user / item ids")
user2idx = {u:i for i,u in enumerate(sorted(train["user"].unique()))}
item2idx = {i:j for j,i in enumerate(sorted(train["item"].unique()))}

train["u"] = train["user"].map(user2idx).astype(np.int32)
train["i"] = train["item"].map(item2idx).astype(np.int32)

U = len(user2idx)
I = len(item2idx)
print(f"U={U:,}, I={I:,}")

# ======================
# 3. Interaction matrix
# ======================
print("▶ Building interaction matrix")
rows = train["u"].to_numpy()
cols = train["i"].to_numpy()
data = np.ones(len(train), dtype=np.float32)

X = sparse.csr_matrix((data, (rows, cols)), shape=(U, I))
print("X shape:", X.shape, "nnz:", X.nnz)

# ======================
# 4. BM25 weighting
# ======================
print("▶ Applying BM25 weighting")
X_bm25 = bm25_weight(X)
print("BM25 nnz:", X_bm25.nnz)

# ======================
# 5. Popularity (optional)
# ======================
pop_penalty = None
if USE_POP_PENALTY:
    print("▶ Computing popularity penalty")
    item_pop = np.array(X.sum(axis=0)).ravel().astype(np.float32) + 1.0
    pop_penalty = np.log(item_pop)
    print("pop_penalty shape:", pop_penalty.shape)

# ======================
# 6. Train EASE
# ======================
print("▶ Training EASE model")
B = fit_ease(X_bm25, lam=LAMBDA)
print("B shape:", B.shape)

# ======================
# 7. Predict Top-K
# ======================
print("▶ Predicting top-k items")
topk = predict_topk(
    X_bm25,
    B,
    k=K,
    pop_penalty=pop_penalty,
    alpha=POP_ALPHA if USE_POP_PENALTY else 0.0
)

print("topk shape:", topk.shape)
print("sample:", topk[0])

# ======================
# 8. Build submission
# ======================
print("▶ Building submission file")
idx2user = {v:k for k,v in user2idx.items()}
idx2item = {v:k for k,v in item2idx.items()}

rows = []
for u in range(U):
    user_id = idx2user[u]
    for i in topk[u]:
        rows.append((user_id, idx2item[int(i)]))

submission = pd.DataFrame(rows, columns=["user", "item"])
print(submission.head(20))

submission.to_csv(OUT_PATH, index=False)
print(f"✅ Saved submission to {OUT_PATH}")
