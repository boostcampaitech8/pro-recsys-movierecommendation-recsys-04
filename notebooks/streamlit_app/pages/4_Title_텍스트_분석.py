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


st.title("4️⃣ Title 텍스트 분석")

@st.cache_data
def get_data(base_path: str):
    return load_all_data(base_path)

if "data_path" not in st.session_state:
    st.error("⚠ 먼저 `app.py`에서 train 데이터 경로를 설정하세요.")
    st.stop()

data_path = st.session_state["data_path"]
data = get_data(data_path)

titles = data["titles"]  # item, title

st.subheader("📌 titles.tsv 샘플")
st.dataframe(titles.head())

# 제목 길이
st.markdown("## 🔹 (1) 제목 길이 분포")

titles["title_len"] = titles["title"].astype(str).str.len()

fig1 = px.histogram(
    titles,
    x="title_len",
    nbins=50,
    title="제목 길이 분포",
)
st.plotly_chart(fig1, use_container_width=True)

st.markdown(
    """
    - 지나치게 긴 제목, 너무 짧은 제목 등의 outlier 확인.  
    - text 기반 feature를 만들 때 토크나이징/전처리 전략을 고민하는 데 도움 됨.
    """
)

# 중복 제목
st.markdown("## 🔹 (2) 중복 제목 존재 여부")

dup_titles = titles[titles.duplicated("title", keep=False)].sort_values("title")

st.write(f"중복 제목을 가진 아이템 수: {dup_titles.shape[0]}")
if not dup_titles.empty:
    st.dataframe(dup_titles.head(30))
    st.markdown(
        """
        - 동일 제목을 가진 서로 다른 item들이 존재하면,  
          title만으로 item을 구분하는 것은 위험할 수 있음.  
        - 따라서 title은 **content feature**일 뿐, 직접적인 ID로 쓰기에 부적절하다는 점을 확인.
        """
    )
else:
    st.write("중복 제목이 거의 없는 것으로 보입니다.")

st.markdown(
    """
    👉 결론적으로, title은 raw text 상태로는 쓰기 어렵고  
    **TF-IDF / embedding 등으로 변환해 content-based 보조 신호로 쓰는 게 적절**한 피처임.
    """
)
