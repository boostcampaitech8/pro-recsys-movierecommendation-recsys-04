import pandas as pd
import os

# ===============================
# 1️⃣ 파일 경로
# ===============================
# 파일 경로
path_a = "/data/ephemeral/home/Seung/output/MultVAE/novalid epoch 200/submission.csv"
path_b = "/data/ephemeral/home/Seung/output/EASE/submission.csv"
out_path = "/data/ephemeral/home/Seung/output/ensemble_submit.csv"

# ===============================
# 2️⃣ 로드
# ===============================
a = pd.read_csv(path_a)
b = pd.read_csv(path_b)

# dtype 통일
for df in (a, b):
    df["user"] = df["user"].astype(int)
    df["item"] = df["item"].astype(int)

# ===============================
# 3️⃣ rank 부여
# ⚠️ 핵심: sort ❌, row 순서 그대로 사용
# ===============================
a["rank_a"] = a.groupby("user", sort=False).cumcount() + 1
b["rank_b"] = b.groupby("user", sort=False).cumcount() + 1

# ===============================
# 4️⃣ rank → score
# ===============================
K = 10
a["score_a"] = K + 1 - a["rank_a"]
b["score_b"] = K + 1 - b["rank_b"]

# ===============================
# 5️⃣ merge (후보 풀 생성)
# ===============================
df = pd.merge(
    a[["user", "item", "score_a"]],
    b[["user", "item", "score_b"]],
    on=["user", "item"],
    how="outer"
).fillna(0)

# ===============================
# 6️⃣ 앙상블 score
# ===============================
w1, w2 = 0.2, 0.8   # 필요하면 조정
df["final_score"] = w1 * df["score_a"] + w2 * df["score_b"]

# ===============================
# 7️⃣ 🔥 결정론적 Top-10 선택 (핵심)
# ===============================
submit = (
    df.sort_values(
        ["user", "final_score", "score_a", "score_b", "item"],
        ascending=[True, False, False, False, True]
    )
    .groupby("user", sort=False)
    .head(10)
)

# ===============================
# 8️⃣ 저장
# ===============================
submit[["user", "item"]].to_csv(out_path, index=False)
print("Saved to:", out_path)
print("File exists:", os.path.exists(out_path))
