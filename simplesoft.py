import pandas as pd
import os

# 파일 경로
path_a = "/data/ephemeral/home/Seung/output/LightGCN/LightGCN_200epochs/submission.csv"
path_b = "/data/ephemeral/home/Seung/output/MultVAE/novalid epoch 200/submission.csv"

# 로드
a = pd.read_csv(path_a)
b = pd.read_csv(path_b)

print("Loaded A:", a.shape)
print("Loaded B:", b.shape)

# dtype 통일 (매우 중요)
a["user"] = a["user"].astype(int)
a["item"] = a["item"].astype(int)
b["user"] = b["user"].astype(int)
b["item"] = b["item"].astype(int)


# rank 부여
a["rank"] = a.groupby("user").cumcount() + 1
b["rank"] = b.groupby("user").cumcount() + 1

# rank → score
K = 10
a["score_a"] = K + 1 - a["rank"]
b["score_b"] = K + 1 - b["rank"]

# merge
df = pd.merge(
    a[["user", "item", "score_a"]],
    b[["user", "item", "score_b"]],
    on=["user", "item"],
    how="outer"
).fillna(0)

print("Merged shape:", df.shape)

# soft voting
w1, w2 = 0.5, 0.5
df["final_score"] = w1 * df["score_a"] + w2 * df["score_b"]

# user별 top10
submit = (
    df.sort_values(["user", "final_score"], ascending=[True, False])
      .groupby("user")
      .head(10)
)

print("Submit shape:", submit.shape)

# 저장 경로 명시
out_path = "/data/ephemeral/home/Seung/output/ensemble_submit.csv"
submit[["user", "item"]].to_csv(out_path, index=False)

print("Saved to:", out_path)
print("File exists:", os.path.exists(out_path))
