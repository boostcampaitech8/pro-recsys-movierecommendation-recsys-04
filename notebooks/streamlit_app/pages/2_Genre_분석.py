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


st.title("2️⃣ Genre 분석")

@st.cache_data
def get_data(base_path: str):
    return load_all_data(base_path)

if "data_path" not in st.session_state:
    st.error("⚠ 먼저 `app.py`에서 train 데이터 경로를 설정하세요.")
    st.stop()

data_path = st.session_state["data_path"]
data = get_data(data_path)

genres = data["genres"]       # item, genre
ratings = data["ratings"]     # user, item, time

st.subheader("📌 데이터 샘플")
st.dataframe(genres.head())

# 1) 장르별 item 수
st.markdown("## 🔹 (1) 장르별 item 수 분포")

items_per_genre = genres.groupby("genre")["item"].nunique().reset_index(name="n_items").sort_values("n_items", ascending=False)

fig1 = px.bar(
    items_per_genre,
    x="genre",
    y="n_items",
    title="장르별 아이템 개수",
)
st.plotly_chart(fig1, use_container_width=True)

st.markdown(
    """
    - 특정 장르(예: Drama, Comedy 등)가 아이템 수를 많이 차지하면 **장르 imbalance**가 존재.  
    - 이 경우 추천 모델이 자주 등장하는 장르 쪽으로 편향되기 쉬움.
    """
)

# 2) 장르 조합 빈도 (아이템 단위)
st.markdown("## 🔹 (2) 장르 조합 패턴 & item당 장르 개수")

item_genre_list = genres.groupby("item")["genre"].apply(list)

# item당 장르 개수
genre_count_per_item = item_genre_list.apply(len)

fig2 = px.histogram(
    genre_count_per_item,
    x=genre_count_per_item,
    nbins=10,
    labels={"x": "장르 개수"},
    title="아이템당 장르 개수 분포",
)
st.plotly_chart(fig2, use_container_width=True)

# 장르 조합 상위 패턴
combo = item_genre_list.apply(lambda g: "|".join(sorted(set(g))))
combo_counts = combo.value_counts().reset_index()
combo_counts.columns = ["genre_combo", "n_items"]

st.subheader("장르 조합 Top 20")
st.dataframe(combo_counts.head(20))

st.markdown(
    """
    - `Action|Thriller`, `Drama|Romance` 같은 패턴이 얼마나 자주 등장하는지로  
      **장르 공존 구조(co-occurrence)** 를 확인할 수 있음.  
    - 나중에 genre embedding, attribute factorization을 할 때 중요한 정보가 됨.
    """
)

# 3) 장르별 popularity (ratings와 결합)
st.markdown("## 🔹 (3) 장르별 Popularity (Interaction 기준)")

merged = ratings.merge(genres, on="item", how="left")  # user, item, time, genre
genre_popularity = merged.groupby("genre")["user"].count().reset_index(name="n_interactions")
genre_popularity = genre_popularity.sort_values("n_interactions", ascending=False)

fig3 = px.bar(
    genre_popularity,
    x="genre",
    y="n_interactions",
    title="장르별 Interaction 수",
)
st.plotly_chart(fig3, use_container_width=True)

st.markdown(
    """
    **해석**  
    - 장르별 **아이템 수**와 **interaction 수**를 비교하면,  
      “많이 만들어지지만 별로 안 보는 장르” vs “적게 만들어져도 끊임없이 보는 장르” 등을 구분할 수 있음.  
    - 향후 추천 시스템에서 장르별 prior(가중치)를 줄지, popularity를 보정할지 판단하는 근거가 됨.
    """
)
