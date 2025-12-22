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
from utils.loader import load_all_data


st.title("5️⃣ Director / Writer Cardinality 분석")

@st.cache_data
def get_data(base_path: str):
    return load_all_data(base_path)

if "data_path" not in st.session_state:
    st.error("⚠ 먼저 `app.py`에서 train 데이터 경로를 설정하세요.")
    st.stop()

data_path = st.session_state["data_path"]
data = get_data(data_path)

directors = data["directors"]   # item, director
writers = data["writers"]       # item, writer

st.subheader("📌 directors.tsv 샘플")
st.dataframe(directors.head())

st.subheader("📌 writers.tsv 샘플")
st.dataframe(writers.head())

# 감독별 영화 수 분포
st.markdown("## 🔹 (1) 감독별 영화 수 분포")

dir_counts = directors.groupby("director")["item"].nunique()
fig1 = px.histogram(
    dir_counts,
    x=dir_counts,
    nbins=50,
    labels={"x": "감독별 아이템 수"},
    title="감독별 아이템 수 분포 (Cardinality)",
    log_y=True,
)
st.plotly_chart(fig1, use_container_width=True)

one_movie_dir_ratio = (dir_counts == 1).mean() * 100
st.write(f"✔ 한 편만 연출한 감독 비율: **{one_movie_dir_ratio:.2f}%**")

# 작가별 영화 수 분포
st.markdown("## 🔹 (2) 작가별 영화 수 분포")

writer_counts = writers.groupby("writer")["item"].nunique()
fig2 = px.histogram(
    writer_counts,
    x=writer_counts,
    nbins=50,
    labels={"x": "작가별 아이템 수"},
    title="작가별 아이템 수 분포 (Cardinality)",
    log_y=True,
)
st.plotly_chart(fig2, use_container_width=True)

one_movie_writer_ratio = (writer_counts == 1).mean() * 100
st.write(f"✔ 한 편만 쓴 작가 비율: **{one_movie_writer_ratio:.2f}%**")

st.markdown(
    """
    **해석**  
    - 감독/작가 모두 대부분이 **극단적인 long-tail** 구조를 가질 경우,  
      one-hot 혹은 단순 factorize만으로는 의미 있는 표현을 얻기 어렵고  
      모델에 그대로 넣을 경우 **노이즈**가 될 수 있음.  
    - 충분한 관측이 있는 소수의 감독/작가만 별도 feature로 쓰거나,  
      아예 **다른 content feature(genre, year 등)** 를 우선시하는 전략을 고려할 수 있음.
    """
)
