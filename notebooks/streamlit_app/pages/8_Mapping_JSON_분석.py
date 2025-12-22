import streamlit as st
from utils.loader import load_all_data

@st.cache_data
def get_data(base_path: str):
    return load_all_data(base_path)

if "data_path" not in st.session_state:
    st.error("⚠ 먼저 `app.py`에서 train 데이터 경로를 설정하세요.")
    st.stop()

data_path = st.session_state["data_path"]
data = get_data(data_path)
item2attr = data["item2attributes"]

if item2attr is None:
    st.error("❌ Ml_item2attributes.json 파일을 찾을 수 없습니다.")
    st.stop()

st.subheader("📌 JSON 구조 예시")
sample_items = list(item2attr.items())[:10]
st.json(dict(sample_items))

# ====== 전처리 ====== #
df = pd.DataFrame([
    {"item": int(k), "attrs": v, "attr_count": len(v)}
    for k, v in item2attr.items()
])

# ====== 1) Attribute 빈도 ====== #
st.markdown("## 🔹 (1) Attribute ID 빈도 분포")

from collections import Counter

all_attrs = []
for v in item2attr.values():
    all_attrs.extend(v)

attr_freq = pd.DataFrame(
    Counter(all_attrs).most_common(),
    columns=["attr_id", "freq"]
)

fig_attr_freq = px.bar(
    attr_freq.head(30),
    x="attr_id",
    y="freq",
    title="Attribute ID Frequency (Top 30)"
)
st.plotly_chart(fig_attr_freq, use_container_width=True)

st.markdown(
    """
    - attribute ID는 전처리에서 **factorize**로 만들어진 genre index
    - 특정 attribute가 많이 등장한다면 genre imbalance와 연결됨  
    """
)

# ====== 2) Item당 attribute 개수 ====== #
st.markdown("## 🔹 (2) Item당 Attribute 개수 분포")

fig_attr_cnt = px.histogram(
    df,
    x="attr_count",
    nbins=10,
    title="Item당 attribute 개수 분포",
)
st.plotly_chart(fig_attr_cnt, use_container_width=True)

st.markdown(
    """
    - 대부분 아이템은 1~3개의 attribute를 가짐  
    - attribute 수가 너무 많거나(잡음) 너무 적으면(sparse) 모델에서 영향이 다름  
    """
)

# ====== 3) 조합 패턴 ====== #
st.markdown("## 🔹 (3) Attribute 조합 패턴")

df["attr_combo"] = df["attrs"].apply(lambda x: "|".join(map(str, sorted(x))))
combo_counts = df["attr_combo"].value_counts().reset_index()
combo_counts.columns = ["combo", "count"]

st.dataframe(combo_counts.head(20))

st.markdown(
    """
    **해석**  
    - attribute 조합은 사실상 **장르 조합**과 동일  
    - FISM / LightGCN / MF+side 등에서 item feature를 embedding할 때  
      attribute 조합이 중요한 역할을 할 수 있음
    """
)