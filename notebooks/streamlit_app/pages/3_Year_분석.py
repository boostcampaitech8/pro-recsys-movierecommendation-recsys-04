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
import streamlit as st
import plotly.express as px
import pandas as pd
from utils.loader import load_all_data


st.title("3️⃣ Year (개봉년도) 분석")

@st.cache_data
def get_data(base_path: str):
    return load_all_data(base_path)

if "data_path" not in st.session_state:
    st.error("⚠ 먼저 `app.py`에서 train 데이터 경로를 설정하세요.")
    st.stop()

data_path = st.session_state["data_path"]
data = get_data(data_path)

years = data["years"]       # item, year
ratings = data["ratings"]   # user, item, time

st.subheader("📌 years.tsv 샘플")
st.dataframe(years.head())

# 연도 분포
st.markdown("## 🔹 (1) 연도별 아이템 수")

items_per_year = years.groupby("year")["item"].nunique().reset_index(name="n_items")

fig1 = px.bar(
    items_per_year,
    x="year",
    y="n_items",
    title="연도별 아이템 개수",
)
st.plotly_chart(fig1, use_container_width=True)

st.markdown(
    """
    - 1920~1950년대처럼 **아주 오래된 연도**에는 아이템 수가 적고,  
      최근 연도일수록 아이템 수가 많아지는 패턴이 흔함.  
    - 이 분포는 `age_of_item` 같은 피처를 만들 때, 오래된 아이템에 대한 smoothing 필요성을 보여줌.
    """
)

# 연도별 popularity (ratings와 결합)
st.markdown("## 🔹 (2) 연도별 Popularity (Interaction 기준)")

merged = ratings.merge(years, on="item", how="left")
year_pop = merged.groupby("year")["user"].count().reset_index(name="n_interactions")

fig2 = px.line(
    year_pop.sort_values("year"),
    x="year",
    y="n_interactions",
    title="연도별 Interaction 수",
)
st.plotly_chart(fig2, use_container_width=True)

st.markdown(
    """
    **해석 포인트**  
    - 같은 연도라도 **아이템 수 대비 interaction**을 보면  
      “시대별 인기”와 “추억 보정(향수)” 같은 효과를 추정할 수 있음.  
    - 나중에 **신작/구작 선호도(latent preference)** 를 모델이 학습하게 만들지  
      혹은 명시적으로 feature로 넣을지 결정하는 근거로 활용 가능.
    """
)
