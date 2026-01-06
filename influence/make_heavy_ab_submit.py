import pandas as pd

# ===============================
# 설정
# ===============================
TRAIN_PATH = "train_ratings.csv"
EASE_PATH = "ease.csv"
BERT_PATH = "bert.csv"

OUT_HEAVY_BERT = "submit_heavy_bert_light_ease.csv"

LIGHT_THRESHOLD = 100

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
heavy_users = set(user_cnt[user_cnt["cnt"] > LIGHT_THRESHOLD]["user"])

print(f"Light users : {len(light_users)}")
print(f"Heavy users : {len(heavy_users)}")

# ===============================
# 2. 추천 결과 로드
# ===============================
print("▶ Loading EASE / BERT predictions")
ease = pd.read_csv(EASE_PATH)
bert = pd.read_csv(BERT_PATH)

assert set(ease["user"]) == set(bert["user"]), "❌ user mismatch"

# ===============================
# 3. Heavy-BERT + Light-EASE 제출 생성
# ===============================
rows = []

for user, g in ease.groupby("user"):
    if user in heavy_users:
        # heavy → BERT
        rows.append(
            bert[bert["user"] == user]
        )
    else:
        # light → EASE 유지
        rows.append(g)

submit_heavy_bert = (
    pd.concat(rows)
    .sort_values("user")
    .reset_index(drop=True)
)

submit_heavy_bert.to_csv(OUT_HEAVY_BERT, index=False)
print(f"✅ Saved: {OUT_HEAVY_BERT}")

print("🎉 Done")
