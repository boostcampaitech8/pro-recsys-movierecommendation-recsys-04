import os
import pandas as pd

# =========================
# 설정 (여기만 조절하면 됨)
# =========================
TRAIN_PATH = "train_ratings.csv"
EASE_PATH  = "sub_ease.csv"      # EASE top10(또는 topN)
BERT_PATH  = "sub_bert.csv"      # BERT top10(또는 topN) - 없으면 자동으로 BERT=0처럼 동작

OUT_PATH   = "submission_final_vA.csv"

K = 10

LIGHT_TH = 5          # interaction <= 5 => light user
LIGHT_BASE_EASE = 7   # light 유저에서 EASE로 먼저 채울 개수
LIGHT_BERT_INJECT = 2 # light 유저에서 BERT로 주입할 개수(1~2 추천)

# ease tail에서 가져올 후보 범위 (EASE가 top10만 있으면 tail이 거의 없으니, 가능하면 top30 이상 파일 권장)
EASE_TAIL_START = 7
EASE_TAIL_END   = 50  # 넉넉히

# bert에서 가져올 후보 범위
BERT_CAND_END   = 20  # 넉넉히


# =========================
# 유틸
# =========================
def normalize_train_columns(train: pd.DataFrame) -> pd.DataFrame:
    # 흔한 컬럼명들 통일
    rename_map = {}
    if "user_id" in train.columns: rename_map["user_id"] = "user"
    if "item_id" in train.columns: rename_map["item_id"] = "item"
    if "timestamp" in train.columns: rename_map["timestamp"] = "time"
    return train.rename(columns=rename_map)

def load_ranked_list(path: str) -> dict:
    """
    CSV columns: user, item, score (score 없으면 item 순서대로 점수 생성해도 됨)
    반환: {user: [item1, item2, ...]} (score 내림차순)
    """
    df = pd.read_csv(path)

    # score 없으면 user별로 현재 순서를 score로 만들어줌
    if "score" not in df.columns:
        df["score"] = 0
        df["score"] = df.groupby("user").cumcount(ascending=True)
        # score 큰게 상위가 되도록 뒤집기
        df["score"] = -df["score"]

    df = df.sort_values(["user", "score"], ascending=[True, False])
    return df.groupby("user")["item"].apply(list).to_dict()

def fill_unique(base_list, candidates, k):
    """base_list에 candidates를 순서대로 넣되 중복 없이 k개 채움"""
    out = list(base_list)
    used = set(out)
    for it in candidates:
        if len(out) >= k:
            break
        if it in used:
            continue
        out.append(it)
        used.add(it)
    return out


def main():
    # 1) train 로드 → user interaction count
    train = pd.read_csv(TRAIN_PATH)
    train = normalize_train_columns(train)

    if "user" not in train.columns or "item" not in train.columns:
        raise ValueError(f"train 컬럼이 예상과 달라. 현재 컬럼: {list(train.columns)}")

    user_cnt = train.groupby("user").size().to_dict()

    # 2) 예측 리스트 로드
    ease_top = load_ranked_list(EASE_PATH)

    bert_top = {}
    if os.path.exists(BERT_PATH):
        bert_top = load_ranked_list(BERT_PATH)
        print("✅ BERT 파일 감지: 사용함")
    else:
        print("⚠️ BERT 파일 없음: BERT 주입 없이 동작(=BERT 0)")

    # 3) 유저별 Top-10 생성
    rows = []
    # for문 전에
    sample_user = next(iter(ease_top))
    print(f"[DEBUG] sample user ease_list length = {len(ease_top[sample_user])}")

    for user, ease_list in ease_top.items():
        cnt = user_cnt.get(user, 0)

        # -------------------------
        # Heavy: EASE Top-10 그대로
        # -------------------------
        if cnt > LIGHT_TH:
            top10 = fill_unique([], ease_list[:200], K)

        # -------------------------
        # Light: EASE 7 + BERT 2 + EASE tail로 마무리
        # -------------------------
        else:
            base = fill_unique([], ease_list[:LIGHT_BASE_EASE], LIGHT_BASE_EASE)

            bert_candidates = bert_top.get(user, [])[:BERT_CAND_END]

            # 🔴 디버깅 로그 (여기!)
            print(f"[DEBUG] user={user} | cnt={cnt}")
            print(f"  ease_base(7) = {base}")
            print(f"  bert_candidates(top5) = {bert_candidates[:5]}")

            # base와 중복 제거하면서 BERT 2개만 주입
            after_bert = fill_unique(base, bert_candidates, min(K, LIGHT_BASE_EASE + LIGHT_BERT_INJECT))

            # EASE tail 후보로 나머지 채움 (EASE 파일이 top10만이면 tail이 없어서 효과 제한)
            ease_tail = ease_list[EASE_TAIL_START:EASE_TAIL_END]
            top10 = fill_unique(after_bert, ease_tail, K)

            # 그래도 부족하면 EASE 앞부분으로 마무리
            if len(top10) < K:
                top10 = fill_unique(top10, ease_list, K)

        # 저장(점수는 10..1로)
        for rank, item in enumerate(top10[:K]):
            rows.append([user, item, K - rank])

    out = pd.DataFrame(rows, columns=["user", "item", "score"])

    # score 제거 (제출용)
    submit = out[["user", "item"]]

    submit.to_csv(OUT_PATH, index=False)
    print(f"✅ 제출용 파일 생성 완료: {OUT_PATH}")

    print(f"✅ 생성 완료: {OUT_PATH} (rows={len(out)})")


if __name__ == "__main__":
    main()
