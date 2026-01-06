import os
import pandas as pd

# =========================
# Path 설정
# =========================
TRAIN_PATH = "train_ratings.csv"
EASE_PATH  = "sub_ease.csv"
BERT_PATH  = "sub_bert.csv"
OUT_PATH   = "submission_final_policy.csv"

# =========================
# 하이퍼파라미터
# =========================
K = 10
LIGHT_TH = 100   # interaction <= 100 => light user

# =========================
# Utils
# =========================
def normalize_cols(df):
    if "user_id" in df.columns:
        df = df.rename(columns={"user_id": "user"})
    if "item_id" in df.columns:
        df = df.rename(columns={"item_id": "item"})
    if "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "time"})
    return df

def load_ranked_list(path):
    df = pd.read_csv(path)
    df = normalize_cols(df)

    # score 없으면 파일 순서를 랭킹으로 사용
    if "score" in df.columns:
        df = df.sort_values(["user", "score"], ascending=[True, False])
    else:
        df["rank"] = df.groupby("user").cumcount()
        df = df.sort_values(["user", "rank"], ascending=[True, True])

    return df.groupby("user")["item"].apply(list).to_dict()

def fill_unique(base, candidates, k):
    out = list(base)
    used = set(out)
    for it in candidates:
        if len(out) >= k:
            break
        if it in used:
            continue
        out.append(it)
        used.add(it)
    return out

# =========================
# Main
# =========================
def main():
    # train 로드 → interaction count
    train = normalize_cols(pd.read_csv(TRAIN_PATH))
    user_cnt = train.groupby("user").size().to_dict()

    # 예측 로드
    ease_top = load_ranked_list(EASE_PATH)
    bert_top = load_ranked_list(BERT_PATH)

    rows = []

    for user, ease_list in ease_top.items():
        cnt = user_cnt.get(user, 0)

        ease = ease_list
        bert = bert_top.get(user, [])

        final10 = []

        # =========================
        # Light user
        # =========================
        if cnt <= LIGHT_TH:
            # 1) Top 1~3: BERT Top 1~3
            final10 = fill_unique([], bert[:3], K)

            # 2) Top 4~10: EASE Top 1~7 (BERT Top1~3와 겹치면 제외)
            final10 = fill_unique(final10, ease[:7], K)

            # 3) 부족분: BERT Top 4~10 중 EASE와 안 겹치는 것
            bert_exclusive = [b for b in bert[3:10] if b not in set(ease[:7])]
            final10 = fill_unique(final10, bert_exclusive, K)

            # 4) 그래도 부족하면 EASE로 마무리
            final10 = fill_unique(final10, ease, K)

        # =========================
        # Heavy user
        # =========================
        else:
            # Top 1~9: EASE
            final10 = fill_unique([], ease[:9], 9)

            # Top 10: BERT 중 EASE와 안 겹치는 가장 상위 1개
            bert_exclusive = [b for b in bert if b not in set(final10)]
            final10 = fill_unique(final10, bert_exclusive[:1], K)

            # 그래도 부족하면 EASE로 마무리
            final10 = fill_unique(final10, ease, K)

        # 저장 (user,item 포맷)
        for item in final10[:K]:
            rows.append([user, item])

    out = pd.DataFrame(rows, columns=["user", "item"])
    out.to_csv(OUT_PATH, index=False)
    print(f"✅ 제출 파일 생성 완료: {OUT_PATH} (rows={len(out)})")

# =========================
if __name__ == "__main__":
    main()
