import pandas as pd

# ===============================
# 설정
# ===============================
TRAIN_PATH = "train_ratings.csv"
EASE_PATH = "ease.csv"
BERT_PATH = "bert.csv"

OUT_ALL_EASE = "submit_all_ease.csv"
OUT_LIGHT_BERT = "submit_light_bert_heavy_ease.csv"

LIGHT_THRESHOLD = 100  # light user 기준

# ===============================
# 1. train 데이터로 user interaction 수 계산
# ===============================
print("▶ Loading train_ratings.csv")
train = pd.read_csv(TRAIN_PATH)

user_cnt = (
    train.groupby("user")
    .size()
    .reset_index(name="cnt")
)

light_users = set(user_cnt[user_cnt["cnt"] <= LIGHT_THRESHOLD]["user"])

print(f"Light users : {len(light_users)}")
print(f"Total users : {user_cnt.shape[0]}")

# ===============================
# 2. 추천 결과 로드
# ===============================
print("▶ Loading EASE / BERT predictions")
ease = pd.read_csv(EASE_PATH)
bert = pd.read_csv(BERT_PATH)

# sanity check
assert set(ease["user"]) == set(bert["user"]), "❌ EASE/BERT user mismatch"
assert ease.groupby("user").size().nunique() == 1, "❌ EASE not 10 items per user"
assert bert.groupby("user").size().nunique() == 1, "❌ BERT not 10 items per user"

# ===============================
# 3. 제출 A: All-EASE (baseline)
# ===============================
ease.to_csv(OUT_ALL_EASE, index=False)
print(f"✅ Saved: {OUT_ALL_EASE}")

# ===============================
# 4. 제출 B: Light-BERT + Heavy-EASE
# ===============================
rows = []

for user, g in ease.groupby("user"):
    if user in light_users:
        # light user → BERT
        rows.append(
            bert[bert["user"] == user]
        )
    else:
        # heavy user → EASE 그대로
        rows.append(g)

submit_light_bert = (
    pd.concat(rows)
    .sort_values("user")
    .reset_index(drop=True)
)

submit_light_bert.to_csv(OUT_LIGHT_BERT, index=False)
print(f"✅ Saved: {OUT_LIGHT_BERT}")

print("🎉 Done")
