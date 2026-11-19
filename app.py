# =====================================================================
#  기상 조건으로 베이징 미세먼지(PM2.5) 농도 예측하기 - 웹페이지(app.py)
#  실행 방법 : streamlit run app.py
# =====================================================================
#  [웹 페이지 구현 설명 요약]
#   - 조건1 (독립변수 값 입력)  : st.number_input 6개로 기상 값을 입력받는다.
#   - 조건2 (예측 결과 출력)    : model.predict()로 예측한 값을 st.metric으로 보여주고
#                                 우리나라 미세먼지 예보 등급(좋음~매우나쁨)도 함께 출력한다.
#   - 조건3 (인공지능 분석 버튼): st.button을 누른 순간에만 예측 코드가 실행된다.
# =====================================================================

from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="베이징 미세먼지 예측 AI", page_icon="🌫️")

# 노트북(pm25_project.ipynb)에서 사용한 것과 같은 독립변수 6개
FEATURES = ["DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]
DATA_PATH = Path(__file__).parent / "beijing_pm25.csv"


# ---------------------------------------------------------------------
# 모델 학습 함수
#  - 노트북과 똑같은 방법(결측치 삭제 -> 8:2 분할 -> 선형회귀 학습)으로 모델을 만든다.
#  - @st.cache_resource 를 붙이면 버튼을 누를 때마다 다시 학습하지 않고
#    처음 한 번 학습한 모델을 계속 재사용하므로 웹페이지가 빨라진다.
# ---------------------------------------------------------------------
@st.cache_resource
def train_model():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["pm2.5"])  # 종속변수(pm2.5)가 비어 있는 행 삭제

    X = df[FEATURES]
    y = df["pm2.5"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, mse, r2


# ---------------------------------------------------------------------
# 예측값을 우리나라 PM2.5 예보 등급으로 바꿔 주는 함수 (단위: ㎍/㎥)
#   좋음 0~15 / 보통 16~35 / 나쁨 36~75 / 매우나쁨 76 이상
# ---------------------------------------------------------------------
def pm25_grade(value):
    if value <= 15:
        return "좋음", "😀", st.success
    elif value <= 35:
        return "보통", "🙂", st.info
    elif value <= 75:
        return "나쁨", "😷", st.warning
    else:
        return "매우나쁨", "🚨", st.error


model, mse, r2 = train_model()

st.title("🌫️ 베이징 미세먼지(PM2.5) 예측 AI")
st.write(
    "기온, 이슬점, 기압 같은 **날씨 정보**를 입력하면 "
    "선형회귀 모델이 베이징의 미세먼지(PM2.5) 농도를 예측합니다."
)
st.caption("학습 데이터: UCI Machine Learning Repository - Beijing PM2.5 (2010~2014년, 1시간 단위)")

# ---------------------------------------------------------------------
# [조건1] 독립변수 값 입력
#  - st.number_input(라벨, min_value, max_value, value=기본값, step=증가폭)
#  - 최솟값/최댓값은 학습 데이터에 실제로 있던 범위로 제한해서
#    말이 안 되는 값(예: 기압 0hPa)이 들어가지 않게 했다.
#  - 기본값은 학습 데이터의 중앙값(가장 흔한 날씨)으로 정했다.
#  - 두 칸(col1, col2)으로 나눠서 입력창을 보기 좋게 배치했다.
# ---------------------------------------------------------------------
st.subheader("① 날씨 정보 입력")
col1, col2 = st.columns(2)

with col1:
    temp = st.number_input(
        "기온 TEMP (℃)", min_value=-20.0, max_value=45.0, value=14.0, step=1.0
    )
    dewp = st.number_input(
        "이슬점 DEWP (℃)", min_value=-40.0, max_value=30.0, value=2.0, step=1.0,
        help="공기 중 수증기가 이슬로 맺히기 시작하는 온도입니다. 기온과 가까울수록 습한 날씨입니다.",
    )
    pres = st.number_input(
        "기압 PRES (hPa)", min_value=990.0, max_value=1050.0, value=1016.0, step=1.0
    )

with col2:
    iws = st.number_input(
        "누적 풍속 Iws (m/s)", min_value=0.0, max_value=570.0, value=5.4, step=1.0,
        help="같은 방향으로 바람이 계속 분 만큼 누적된 풍속입니다. 클수록 바람이 오래, 강하게 분 것입니다.",
    )
    is_snow = st.number_input(
        "누적 강설 시간 Is (시간)", min_value=0, max_value=30, value=0, step=1
    )
    ir_rain = st.number_input(
        "누적 강수 시간 Ir (시간)", min_value=0, max_value=40, value=0, step=1
    )

if dewp > temp:
    st.caption("⚠️ 이슬점은 보통 기온보다 높을 수 없어요. 값을 한 번 더 확인해 보세요.")

# ---------------------------------------------------------------------
# [조건3] 인공지능 분석 버튼
#  - st.button()은 버튼이 눌린 순간에만 True를 돌려준다.
#  - 그래서 if 문 안에 예측 코드를 넣으면, 값을 입력하는 동안에는 아무 일도
#    일어나지 않다가 버튼을 눌렀을 때만 인공지능이 분석을 시작한다.
# ---------------------------------------------------------------------
st.subheader("② 인공지능 분석")
if st.button("🤖 인공지능 분석", type="primary"):

    # -----------------------------------------------------------------
    # [조건2] 예측 결과 출력
    #  - 입력받은 값 6개를 학습 때와 같은 열 이름·순서의 DataFrame으로 만든다.
    #  - model.predict()는 결과를 배열로 돌려주므로 [0]으로 첫 번째 값을 꺼낸다.
    #  - 선형회귀는 직선 식이라 음수가 나올 수도 있는데, 농도는 0보다 작을 수
    #    없으므로 max(0, 값)으로 0 미만은 0으로 맞춘다.
    #  - st.metric으로 숫자를 크게 보여주고, 등급별 색깔 상자로 안내 문구를 출력한다.
    # -----------------------------------------------------------------
    input_data = pd.DataFrame(
        [[dewp, temp, pres, iws, is_snow, ir_rain]], columns=FEATURES
    )
    prediction = max(0.0, float(model.predict(input_data)[0]))
    grade, emoji, show_box = pm25_grade(prediction)

    st.subheader("③ 예측 결과")
    c1, c2 = st.columns(2)
    c1.metric("예측 PM2.5 농도", f"{prediction:.1f} ㎍/㎥")
    c2.metric("예보 등급", f"{emoji} {grade}")
    show_box(f"이 날씨에서 예상되는 미세먼지 등급은 **{grade}** 입니다.")

    with st.expander("입력한 값 다시 보기"):
        st.dataframe(input_data, hide_index=True)
else:
    st.info("날씨 정보를 입력하고 **🤖 인공지능 분석** 버튼을 눌러 주세요.")

# ---------------------------------------------------------------------
# 모델 성능 안내 (노트북의 조건5 결과와 같은 값)
# ---------------------------------------------------------------------
st.divider()
with st.expander("📊 이 AI 모델의 성능 보기"):
    m1, m2 = st.columns(2)
    m1.metric("R² (결정계수)", f"{r2:.3f}")
    m2.metric("MSE (평균제곱오차)", f"{mse:,.1f}")
    st.write(
        f"R²가 {r2:.2f}이므로 미세먼지 농도 변화의 약 {r2 * 100:.0f}%를 날씨만으로 설명합니다. "
        "미세먼지는 난방·자동차 같은 배출원과 이전 시간의 농도에도 크게 영향을 받기 때문에, "
        "예측값은 대략적인 경향으로 참고해 주세요."
    )
