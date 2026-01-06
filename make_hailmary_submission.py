import os
import random
import pandas as pd

# =========================
# 설정
# =========================
LIGHT_TH = 5   # interaction <= 5 → light user
random.seed(42)

# =========================
# 데이터 로드
# =========================
train = pd.read_csv("train_ratings.csv")

# 컬럼명 맞추기 (필요시 수정)
train = train.rename(columns={
    "user_id": "user",
    "item_id": "item",
    "timestamp": "time"
})

user_cnt = train.groupby("user").size().to_dict()

def load_submission(path):
    df = pd.read_csv(path)
    return (
        df.sort_values(["user", "score"], ascending=[True, False])
          .groupby("user")["item"]
          .apply(list)
          .to_dict()
    )

ease_top = load_submission("sub_ease.csv")

bert_top = {}
if os.path.exists("sub_bert.csv"):
    bert_top = load_submission("sub_bert.csv")

# =========================
# helper
# =========================
def ease_tail(lst, k1=3, k2=10):
    return lst[k1:k2]

# =========================
# submission 생성
# =========================
final_rows = []

for user, ease_list in ease_top.items():
    cnt = user_cnt.get(user, 0)

    # Heavy user
    if cnt > LIGHT_TH:
        top10 = ease_list[:10]

    # Light user
    else:
        base = ease_list[:7]
        used = set(base)

        candidates = []
        candidates += ease_tail(ease_list)

        if user in bert_top:
            candidates += bert_top[user][:5]

        candidates = [i for i in candidates if i not in used]

        inject = []
        for item in candidates:
            if len(inject) == 3:
                break
            inject.append(item)

        top10 = base + inject

        for item in ease_list:
            if len(top10) == 10:
                break
            if item not in top10:
                top10.append(item)

    for rank, item in enumerate(top10):
        final_rows.append([user, item, 10 - rank])

# =========================
# 저장
# =========================
final_df = pd.DataFrame(final_rows, columns=["user", "item", "score"])

# 제출용 파일은 score 제거
submit_df = final_df[["user", "item"]]

submit_df.to_csv("submission_final_hailmary.csv", index=False)

print("✅ submission_final_hailmary.csv (score 제거, 제출용) 생성 완료")
