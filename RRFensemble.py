import pandas as pd
import numpy as np

path_lgn = "/data/ephemeral/home/Seung/output/LightGCN/LightGCN_200epochs/submission.csv"
path_vae = "/data/ephemeral/home/Seung/output/MultVAE/novalid multvae/submission.csv"  # 0.1368 나온 그 파일!

lgn = pd.read_csv(path_lgn)
vae = pd.read_csv(path_vae)

for df in (lgn, vae):
    df["user"] = df["user"].astype(int)
    df["item"] = df["item"].astype(int)
    df.sort_values(["user"], inplace=True)

K = 10

# rank
lgn["rank"] = lgn.groupby("user").cumcount() + 1
vae["rank"] = vae.groupby("user").cumcount() + 1

# RRF score: 1 / (c + rank)
c = 60  # 보통 10~100 사이, 60 많이 씀
lgn["score_lgn"] = 1.0 / (c + lgn["rank"])
vae["score_vae"] = 1.0 / (c + vae["rank"])

df = pd.merge(
    lgn[["user","item","score_lgn"]],
    vae[["user","item","score_vae"]],
    on=["user","item"],
    how="outer"
).fillna(0)

# 가중치 (MultVAE 중심 추천)
w_lgn, w_vae = 0.2, 0.8
df["final"] = w_lgn*df["score_lgn"] + w_vae*df["score_vae"]

submit = (df.sort_values(["user","final"], ascending=[True, False])
            .groupby("user")
            .head(K))

out_path = "/data/ephemeral/home/Seung/output/ensemble_rrf.csv"
submit[["user","item"]].to_csv(out_path, index=False)
print("saved:", out_path)
