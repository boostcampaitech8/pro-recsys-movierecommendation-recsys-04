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


st.title("6️⃣ Item Metadata Coverage 분석")

@st.cache_data
def get_data(base_path: str):
    return load_all_data(base_path)

if "data_path" not in st.session_state:
    st.error("⚠ 먼저 `app.py`에서 train 데이터 경로를 설정하세요.")
    st.stop()

data_path = st.session_state["data_path"]
data = get_data(data_path)

titles = data["titles"]        # item, title
years = data["years"]          # item, year
genres = data["genres"]        # item, genre
directors = data["directors"]  # item, director
writers = data["writers"]      # item, writer
ratings = data["ratings"]      # user, item, time

# 모든 소스에서 등장하는 item id 모으기
all_items = pd.unique(
    pd.concat(
        [
            ratings["item"],
            titles["item"],
            years["item"],
            genres["item"],
            directors["item"],
            writers["item"],
        ],
        ignore_index=True,
    )
)

item_df = pd.DataFrame({"item": all_items})

item_df["has_title"] = item_df["item"].isin(titles["item"])
item_df["has_year"] = item_df["item"].isin(years["item"])
item_df["has_genre"] = item_df["item"].isin(genres["item"])
item_df["has_director"] = item_df["item"].isin(directors["item"])
item_df["has_writer"] = item_df["item"].isin(writers["item"])

st.subheader("📌 Item별 메타데이터 보유 여부 예시")
