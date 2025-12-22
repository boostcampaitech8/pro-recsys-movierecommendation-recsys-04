import pandas as pd
import os

# ===============================
# 1️⃣ 설정
# ===============================

# ⚠️ paths 순서 중요
# score_0 → MultVAE
# score_1 → EASE
paths = [
    "/data/ephemeral/home/Seung/output/MultVAE/submission.csv",
    "/data/ephemeral/home/Seung/output/EASE/submission_100.csv",
]

out_path = "/data/ephemeral/home/Seung/output/ensemble_submit.csv"

TOP_K = 10
CANDIDATE_K = 100
MIN_OVERLAP = 2   # MultVAE + EASE 둘 다 뽑은 경우만 우선

# 모델 가중치 (public 기준)
MODEL_WEIGHTS = {
    "score_0": 1.0,   # MultVAE
    "score_1": 1.2,   # EASE
}

# ===============================
# 2️⃣ 로드 + rank → score
# ===============================
dfs = []

for i, path in enumerate(paths):
    df = pd.read_csv(path)
    df["user"] = df["user"].astype(int)
    df["item"] = df["item"].astype(int)

    # row 순서 그대로 rank 부여 (절대 sort 금지)
    df[f"rank_{i}"] = df.groupby("user", sort=False).cumcount() + 1

    # top-100 후보만 유지
    df = df[df[f"rank_{i}"] <= CANDIDATE_K]

    # rank → score (클수록 좋음)
    df[f"score_{i}"] = CANDIDATE_K + 1 - df[f"rank_{i}"]

    dfs.append(df)

# ===============================
# 3️⃣ 후보 풀 merge
# ===============================
base = dfs[0][["user", "item", "score_0"]]

for i in range(1, len(dfs)):
    base = pd.merge(
        base,
        dfs[i][["user", "item", f"score_{i}"]],
        on=["user", "item"],
        how="outer",
    )

base = base.fillna(0)

score_cols = [c for c in base.columns if c.startswith("score_")]

# ===============================
# 4️⃣ overlap + weighted score
# ===============================

# 몇 개 모델에서 등장했는지
base["overlap_cnt"] = (base[score_cols] > 0).sum(axis=1)

# 가중치 점수 (fallback용)
base["final_score"] = 0.0
for col, w in MODEL_WEIGHTS.items():
    base["final_score"] += w * base[col]

# ===============================
# 5️⃣ user별 Top-10 구성
# ===============================
results = []

for user, g in base.groupby("user", sort=False):

    # 1️⃣ consensus 우선 (둘 다 뽑은 아이템)
    primary = g[g["overlap_cnt"] >= MIN_OVERLAP].sort_values(
        ["final_score", "item"],
        ascending=[False, True],
    )

    selected = primary.head(TOP_K)

    # 2️⃣ 부족하면 weighted fallback
    if len(selected) < TOP_K:
        remain = g[~g.index.isin(selected.index)].sort_values(
            ["final_score", "item"],
            ascending=[False, True],
        )
        selected = pd.concat(
            [selected, remain.head(TOP_K - len(selected))],
            axis=0,
        )

    results.append(selected)

submit = pd.concat(results, axis=0)

# ===============================
# 6️⃣ 저장
# ===============================
submit[["user", "item"]].to_csv(out_path, index=False)

print("✅ Ensemble submission saved to:", out_path)
print("📄 File exists:", os.path.exists(out_path))
print("🎯 Models: MultVAE + EASE")
print("⚖️ Weights:", MODEL_WEIGHTS)
print("🔁 MIN_OVERLAP =", MIN_OVERLAP)
