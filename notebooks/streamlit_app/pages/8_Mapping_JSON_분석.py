import streamlit as st
import pandas as pd            # ← 반드시 필요!
import plotly.express as px    # ← 반드시 필요!
from collections import Counter

from utils.loader import load_all_data

st.title("8️⃣ Ml_item2attributes.json 분석")

@st.cache_data
def get_data(base_path: str):
    return load_all_data(base_path)

# 데이터 경로 체크
if "data_path" not in st.session_state:
    st.error("⚠ 먼저 `app.py`에서 train 데이터 경로를 설정하세요.")
    st.stop()

data_path = st.session_state["data_path"]
data = get_data(data_path)
item2attr = data["item2attributes"]

# JSON 존재 확인
if item2attr is None:
    st.error("❌ Ml_item2attributes.json 파일을 찾을 수 없습니다.")
    st.stop()

# ============================
# 0) JSON 샘플 보기
# ============================
st.subheader("📌 JSON 구조 예시")
sample_items = list(item2attr.items())[:10]
st.json(dict(sample_items))

# ============================
# 전처리: DataFrame 변환
# ============================
df = pd.DataFrame([
    {"item": int(k), "attrs": v, "attr_count": len(v)}
    for k, v in item2attr.items()
])

# ============================
# (1) Attribute ID 빈도 분포
# ============================
st.markdown("## 🔹 (1) Attribute ID 빈도 분포")

# attribute ID frequency 계산
all_attrs = []
for v in item2attr.values():
    all_attrs.extend(v)

attr_freq = pd.DataFrame(
    Counter(all_attrs).most_common(),
    columns=["attr_id", "freq"]
)

# Plot: frequency top 30
fig_attr_freq = px.bar(
    attr_freq.head(30),
    x="attr_id",
    y="freq",
    title="Attribute ID Frequency (Top 30)"
)
st.plotly_chart(fig_attr_freq, use_container_width=True)

st.markdown(
    """
    - Attribute ID는 `genre` 또는 전처리된 `attribute index`  
    - 특정 attribute가 압도적으로 많다면 장르 imbalance와 동일한 의미  
    """
)

# ============================
# (2) Item당 Attribute 개수 분포
# ============================
st.markdown("## 🔹 (2) Item당 Attribute 개수 분포")

fig_attr_cnt = px.histogram(
    df,
    x="attr_count",
    nbins=10,
    title="Item당 Attribute 개수 분포"
)
st.plotly_chart(fig_attr_cnt, use_container_width=True)

st.markdown(
    """
    - 대부분 아이템은 1~3개의 attribute를 가짐  
    - attribute 수가 너무 많으면: 잡음(noise)  
    - attribute 수가 너무 적으면: 정보 부족(sparse)  
    → 모델 feature engineering 시 고려해야 함
    """
)

# ============================
# (3) Attribute 조합 패턴
# ============================
st.markdown("## 🔹 (3) Attribute 조합 패턴")

df["attr_combo"] = df["attrs"].apply(lambda x: "|".join(map(str, sorted(x))))
combo_counts = df["attr_combo"].value_counts().reset_index()
combo_counts.columns = ["combo", "count"]

st.dataframe(combo_counts.head(20))

st.markdown(
    """
    **해석**  
    - attribute 조합은 사실상 **장르 조합**과 동일  
    - FISM / LightGCN / MF+Side 같은 모델에서  
      item feature embedding 시 중요한 signal  
    """
)
