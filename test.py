import pandas as pd

# 파일 경로
lgn = pd.read_csv("/data/ephemeral/home/Seung/output/LightGCN/LightGCN_200epochs/submission.csv")
vae = pd.read_csv("/data/ephemeral/home/Seung/output/MultVAE/valid_ratio 0.1 epoch141/submission.csv")

# dtype 통일
lgn["user"] = lgn["user"].astype(int)
lgn["item"] = lgn["item"].astype(int)
vae["user"] = vae["user"].astype(int)
vae["item"] = vae["item"].astype(int)

K = 10

# user별 topK set
lgn_topk = lgn.groupby("user")["item"].apply(lambda x: set(x.head(K)))
vae_topk = vae.groupby("user")["item"].apply(lambda x: set(x.head(K)))

# 공통 user
common_users = lgn_topk.index.intersection(vae_topk.index)

overlaps = []
for u in common_users:
    overlaps.append(len(lgn_topk[u] & vae_topk[u]))

import numpy as np
overlaps = np.array(overlaps)

print("=== Overlap stats (per user, out of 10) ===")
print("Mean overlap:", overlaps.mean())
print("Median overlap:", np.median(overlaps))
print("Overlap distribution:")
print(pd.Series(overlaps).value_counts().sort_index())
