import streamlit as st
import plotly.express as px
import pandas as pd
from utils.loader import load_all_data

st.title("1️⃣ Ratings Log (Implicit Feedback) 분석")

# ====== 데이터 로드 ======
@st.cache_data
def get_data(base_path: str):
    return load_all_data(base_path)

if "data_path" not in st.session_state:
    st.error("⚠ 먼저 `app.py`에서 train 데이터 경로를 설정하세요.")
    st.stop()

data_path = st.session_state["data_path"]
data = get_data(data_path)
ratings = data["ratings"]  # user, item, time

ratings["datetime"] = pd.to_datetime(ratings["time"], unit="s")


# ======================================================
# 🔹 SECTION 0 — 기본 정보
# ======================================================
st.markdown("### 📌 데이터 개요")
st.write(f"- 행 수: **{len(ratings):,} rows**")
st.write("- 컬럼: `user`, `item`, `time` (timestamp in seconds)")
st.dataframe(ratings.head())


# ======================================================
# 🔹 SECTION 1 — User / Item Interaction Count 분포
# ======================================================
st.markdown("## 🔹 (1) User별 Interaction Count")

user_counts = ratings.groupby("user")["item"].count().rename("interaction_count")
st.write(f"✔ 유저 수: {user_counts.shape[0]:,}")

fig_user = px.histogram(
    user_counts, x="interaction_count", nbins=50,
    title="User별 Interaction Count 분포 (log y)", log_y=True
)
st.plotly_chart(fig_user, use_container_width=True)

st.markdown("""
- 대부분 유저가 적은 수의 영화를 시청하는 **long-tail 구조**
- 극소수의 heavy user가 데이터에 강하게 영향 미침
""")


# ======================================================
# 🔹 SECTION 2 — Heavy User/Item 영향 분석
# ======================================================
st.markdown("## 🔹 (2) Heavy User / Heavy Item 분석")

top_percent = st.slider("Top N% (상위 사용자/아이템 비율)", 0.1, 10.0, 1.0, step=0.1)
n_ratio = top_percent / 100

# ---- HEAVY USER ----
sorted_user = user_counts.sort_values(ascending=False)
k_user = max(1, int(len(sorted_user) * n_ratio))
heavy_user_share = sorted_user.iloc[:k_user].sum() / sorted_user.sum()

st.metric(
    label=f"상위 {top_percent:.1f}% 유저가 차지하는 Interaction 비중",
    value=f"{heavy_user_share * 100:.2f}%"
)

# ---- HEAVY ITEM ----
item_counts = ratings.groupby("item")["user"].count().rename("interaction_count")
sorted_item = item_counts.sort_values(ascending=False)
k_item = max(1, int(len(sorted_item) * n_ratio))
heavy_item_share = sorted_item.iloc[:k_item].sum() / sorted_item.sum()

st.metric(
    label=f"상위 {top_percent:.1f}% 아이템이 차지하는 Interaction 비중",
    value=f"{heavy_item_share * 100:.2f}%"
)

st.markdown("""
### 📌 해석
- 상위 소수의 heavy user / 인기 아이템이 전체 데이터 대부분을 차지하는 **extreme long-tail** 구조.
- User-based CF는 heavy user에 과도하게 의존하고 sparse한 유저에게 불리.
- Item-based CF는 많이 소비된 아이템에서 안정적으로 학습됨.
""")


# ======================================================
# 🔹 SECTION 3 — Heavy User 샘플 Timeline 표시
# ======================================================
st.markdown("## 🔹 (3) Heavy User Activity Timeline (샘플)")

top_user_list = sorted_user.index[:10]  # 상위 10 heavy users
selected_user = st.selectbox("Heavy User 중 선택", top_user_list)

sample_user_times = ratings[ratings["user"] == selected_user].sort_values("datetime")

fig_timeline = px.scatter(
    sample_user_times,
    x="datetime", y="item",
    title=f"User {selected_user} — Interaction Timeline",
    opacity=0.6
)
st.plotly_chart(fig_timeline, use_container_width=True)


# ======================================================
# 🔹 SECTION 4 — 월별(연도 무관) Interaction Count
# ======================================================
st.markdown("## 🔹 (4) 월별 Interaction Count 추이 (연도 무관 Seasonality)")

ratings["month"] = ratings["datetime"].dt.month
month_counts = ratings.groupby("month")["user"].count().reset_index(name="interaction_count")

fig_month = px.line(
    month_counts, x="month", y="interaction_count",
    markers=True,
    title="월별 Interaction Count (Seasonality)"
)
st.plotly_chart(fig_month, use_container_width=True)

st.markdown("""
📌 연도와 상관없이 **월별 패턴(계절성)**을 확인할 수 있음  
예: 시즌별 이용량 증가/감소 → time-aware 모델에 활용 가능
""")


# ======================================================
# 🔹 SECTION 5 — User Timestamp Standard Deviation 분석
# ======================================================
st.markdown("## 🔹 (5) Timestamp 편차 분석")

# 전체 user std
user_time_std = (
    ratings.groupby("user")["time"]
    .std()
    .fillna(0)      # single-item users: std = 0
    .rename("time_std")
)

fig_std = px.histogram(
    user_time_std, x="time_std", nbins=50,
    title="전체 User Timestamp 편차 분포"
)
st.plotly_chart(fig_std, use_container_width=True)

st.markdown("""
- timestamp 편차가 크다는 것은 사용 기간이 길거나, 불규칙하게 오래 활동한 유저를 의미  
- 반대로 std=0이면 시청 기록이 1개뿐인 cold user
""")


# ======================================================
# 🔹 SECTION 6 — 특정 User Timestamp 편차 분석
# ======================================================
st.markdown("## 🔹 (6) 특정 User Timestamp Activity 분석")

input_user = st.number_input("User ID 입력", min_value=0, step=1)

if input_user in ratings["user"].unique():
    u_times = ratings[ratings["user"] == input_user]["time"].sort_values()
    u_std = u_times.std()

    st.metric(label=f"User {input_user} Timestamp Std", value=f"{u_std:.2f}")

    fig_user_timeline = px.scatter(
        ratings[ratings["user"] == input_user],
        x="datetime",
        y="item",
        title=f"User {input_user} — Timestamp Timeline",
        opacity=0.6
    )
    st.plotly_chart(fig_user_timeline, use_container_width=True)

else:
    st.info("해당 user ID는 데이터에 없습니다.")
