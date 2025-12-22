import streamlit as st
import plotly.express as px   # ← 반드시 필요!
from utils.loader import load_all_data

@st.cache_data
def get_data(base_path: str):
    return load_all_data(base_path)

# 데이터 경로 체크
if "data_path" not in st.session_state:
    st.error("⚠ 먼저 `app.py`에서 train 데이터 경로를 설정하세요.")
    st.stop()

data_path = st.session_state["data_path"]
data = get_data(data_path)

ratings = data["ratings"]      # user, item, time
genres = data["genres"]        # item, genre
years = data["years"]

st.subheader("📌 데이터 샘플 (ratings + genre + year 조인 전)")
st.dataframe(ratings.head())

# ========= 1) User Favorite Genre ========= #
st.markdown("## 🔹 (1) User Favorite Genre 분석")

# user, item, time + genre merge
merged = ratings.merge(genres, on="item", how="left")

# 유저-장르별 interaction count
user_genre_counts = (
    merged.groupby(["user", "genre"])["item"]
    .count()
    .reset_index(name="cnt")
)

# 각 user의 가장 많이 본 genre
fav_genre = (
    user_genre_counts.sort_values(["user", "cnt"], ascending=[True, False])
    .groupby("user")
    .first()
    .reset_index()
)

# Plot
fig_fav = px.histogram(
    fav_genre,
    x="genre",
    title="User Favorite Genre 분포",
)
st.plotly_chart(fig_fav, use_container_width=True)

st.markdown(
    """
    **해석**  
    - 유저가 가장 많이 소비한 장르를 계산하면  
      장르 선호 기반의 **유저 취향 군집화**,  
      **개인화 추천(genre prior)** 에 활용됨.
    """
)

# ========= 2) Genre Popularity ========= #
st.markdown("## 🔹 (2) 장르 Popularity vs. Item Popularity")

genre_pop = merged.groupby("genre")["user"].count().reset_index(name="n_interactions")

fig_pop = px.bar(
    genre_pop.sort_values("n_interactions", ascending=False),
    x="genre",
    y="n_interactions",
    title="장르별 Interaction Popularity",
)
st.plotly_chart(fig_pop, use_container_width=True)

st.markdown(
    """
    - 인기 장르를 확인하면 모델이 자동으로 학습하는  
      **popularity bias / genre bias**를 이해할 수 있음.
    """
)

# ========= 3) Year Popularity ========= #
st.markdown("## 🔹 (3) 연도별 Popularity")

merged_year = ratings.merge(years, on="item", how="left")
year_pop = (
    merged_year.groupby("year")["user"]
    .count()
    .reset_index(name="n_interactions")
)

fig_year = px.line(
    year_pop.sort_values("year"),
    x="year",
    y="n_interactions",
    title="연도별 Interaction 수",
)
st.plotly_chart(fig_year, use_container_width=True)

st.markdown(
    """
    **해석**  
    - 특정 연대나 최근 연도에 interaction이 몰릴 경우  
      **신작 선호**, **temporal popularity drift** 등을 고려해야 함.
    """
)
