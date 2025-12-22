import streamlit as st
import os

st.set_page_config(
    page_title="Movie EDA Dashboard",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Implicit Feedback EDA Dashboard")
st.write(
    """
    이 앱은 **implicit feedback 기반 영화 추천 데이터**에 대해  
    자동으로 EDA를 수행하는 대시보드입니다.
    """
)

default_path = st.session_state.get("data_path", "")

data_path = st.text_input(
    "📁 train 데이터 폴더 경로를 입력하세요 (예: /mnt/data/train)",
    value=default_path,
)

required_files = [
    "train_ratings.csv",
    "titles.tsv",
    "years.tsv",
    "genres.tsv",
    "directors.tsv",
    "writers.tsv",
    "Ml_item2attributes.json",
]

st.markdown("### ✅ 필요한 파일 목록")
st.code("\n".join(required_files), language="text")

if data_path:
    exists = {
        f: os.path.exists(os.path.join(data_path, f))
        for f in required_files
    }

    st.markdown("### 🔍 파일 존재 여부 확인")
    st.table(
        {
            "file": list(exists.keys()),
            "exists": ["✅" if v else "❌" for v in exists.values()],
        }
    )

    if all(exists.values()):
        st.success("✔ 모든 파일을 찾았습니다. 좌측 사이드바에서 각 EDA 페이지를 선택해 주세요.")
        st.session_state["data_path"] = data_path
    else:
        st.error("⚠ 일부 파일이 없습니다. 경로 또는 파일 구성을 다시 확인하세요.")

st.info("ℹ 상단 `app.py`에서 경로만 설정하면, 나머지 페이지는 자동으로 이 경로를 사용합니다.")
