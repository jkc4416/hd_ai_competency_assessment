# -*- coding: utf-8 -*-
"""
ARIMA 학습 시각화 가이드 (Intuition → Stationarity → Differencing → ACF/PACF → Identification → Estimation → Diagnostics → Forecast)
----------------------------------------------------------------------------------------------
이 스크립트는 다음 "학습 순서"를 따라 한 번에 실행 가능한 시각화 파이프라인을 제공합니다.

[학습 순서]
1) 직관 잡기: 비정상 시계열(랜덤워크+계절) 생성 및 관찰
2) 정상성 확인: ADF/KPSS 중 ADF로 단위근(비정상) 검정
3) 차분으로 정상화: 1차 차분(d=1), 필요 시 계절차분(옵션)
4) ACF/PACF로 p,q 초기 감 잡기(식별)
5) 후보 ARIMA(p,d,q) 적합 및 AIC/BIC 비교
6) 잔차 진단: 잔차 ACF, Ljung-Box(자기상관 없음 확인)
7) 예측: h-스텝 앞 예측 및 신뢰구간 시각화
8) p/d/q 변화 효과 체감: AR-only, MA-only, ARMA 비교 시뮬

필요 패키지: numpy, pandas, matplotlib, statsmodels
설치: pip install numpy pandas matplotlib statsmodels
주의: 모든 도표는 matplotlib 단일 플롯(figure당 하나)로 그립니다.
"""

# -------------------------------
# 0. 라이브러리 임포트
# Libraries Import
# -------------------------------
import os  # 운영체제 인터페이스 / Operating system interface
from pathlib import Path  # 경로 조작 / Path manipulation

# matplotlib 백엔드 설정 (비대화형 모드 / Non-interactive mode)
# CLI 환경에서 파일로 저장하기 위해 필요 / Required for saving files in CLI environment
import matplotlib

matplotlib.use("Agg")

import numpy as np  # 수치 연산 / Numerical operations
import pandas as pd  # 시계열/데이터프레임 / Time series and DataFrame
import matplotlib.pyplot as plt  # 시각화(한 Figure당 한 Plot 원칙) / Visualization
import matplotlib.font_manager as fm  # 폰트 관리 / Font management
from statsmodels.tsa.stattools import adfuller  # ADF 단위근 검정 / ADF unit root test
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # ACF/PACF 플롯
from statsmodels.tsa.arima.model import ARIMA  # ARIMA 적합 / ARIMA fitting
from statsmodels.stats.diagnostic import acorr_ljungbox  # Ljung-Box 검정
from statsmodels.tsa.statespace.sarimax import SARIMAX  # SARIMA 모델 / SARIMA model

# -------------------------------
# 한글 폰트 설정 / Korean Font Configuration
# -------------------------------
font_path = "/home/claude-dev-kcj/.fonts/NanumGothic-Regular.ttf"
if os.path.exists(font_path):
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = font_prop.get_name()
    plt.rcParams["axes.unicode_minus"] = (
        False  # 마이너스 기호 깨짐 방지 / Prevent minus sign corruption
    )

# -------------------------------
# 스크립트 디렉토리 설정 / Script Directory Configuration
# -------------------------------
SCRIPT_DIR = Path(__file__).parent  # 현재 스크립트가 있는 디렉토리 / Current script directory

# -------------------------------
# 출력 디렉토리 설정 / Output Directory Configuration
# -------------------------------
# 시각화 결과를 카테고리별로 분류하여 저장
# Organize visualization outputs by category
#
# 디렉토리 구조 / Directory structure:
# arima/
# ├── outputs/
# │   ├── main_analysis/       - 메인 ARIMA 분석 (Step 1-7)
# │   ├── model_comparison/    - AR/MA/ARMA 모형 비교
# │   └── unit_root_comparison/ - 단위근(랜덤워크 vs AR) 비교

OUTPUT_DIR = SCRIPT_DIR / "outputs"  # 출력 루트 디렉토리 / Output root directory
MAIN_ANALYSIS_DIR = (
    OUTPUT_DIR / "main_analysis"
)  # 메인 ARIMA 분석 결과 / Main ARIMA analysis results
MODEL_COMPARISON_DIR = OUTPUT_DIR / "model_comparison"  # AR/MA/ARMA 비교 / AR/MA/ARMA comparison
UNIT_ROOT_DIR = OUTPUT_DIR / "unit_root_comparison"  # 단위근 비교 / Unit root comparison

# 출력 디렉토리 생성 (존재하지 않으면 생성) / Create output directories if not exist
for dir_path in [OUTPUT_DIR, MAIN_ANALYSIS_DIR, MODEL_COMPARISON_DIR, UNIT_ROOT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# -------------------------------
# 1. 재현성 및 데이터 생성 / Reproducibility and Data Generation
# -------------------------------
# 랜덤 시드 고정으로 동일한 결과를 재현할 수 있도록 설정
# Fix random seed for reproducible results across different runs
np.random.seed(42)

# 시계열 데이터의 길이와 계절 주기 설정 / Set time series length and seasonal period
n = 400  # 시계열 관측치 개수 (데이터 포인트 수) / Number of observations (data points)
season = 24  # 계절 주기: 24일 (예: 월간 데이터의 연간 계절성) / Seasonal period: 24 days

# -------------------------------
# 1-1. 백색잡음 생성 / Generate White Noise
# -------------------------------
# 백색잡음(White Noise): 평균 0, 분산 1인 정규분포를 따르는 독립적인 확률 변수
# White noise: Independent random variables following N(0, 1) normal distribution
# 이는 ARIMA 모델의 혁신(innovation) 또는 충격(shock)을 나타냄
# This represents innovations or shocks in ARIMA models
eps = np.random.normal(loc=0, scale=1.0, size=n)

# -------------------------------
# 1-2. 랜덤워크 생성 / Generate Random Walk
# -------------------------------
# 랜덤워크(Random Walk): y_t = y_{t-1} + ε_t 형태의 비정상 시계열
# Random walk: Non-stationary time series with form y_t = y_{t-1} + ε_t
# 단위근(unit root)을 가지므로 비정상성(non-stationarity) 발생
# Contains unit root, causing non-stationarity
# cumsum()을 사용하여 백색잡음을 누적합하면 랜덤워크 생성
# Cumulative sum of white noise creates random walk process
rw = np.cumsum(eps)

# -------------------------------
# 1-3. 계절성 패턴 추가 / Add Seasonal Pattern
# -------------------------------
# 시간 인덱스 생성: 0부터 n-1까지 / Create time index: 0 to n-1
t = np.arange(n)

# 사인 함수로 주기적 계절성 생성 / Generate periodic seasonality using sine function
# sin(2πt/period) 형태로 주기가 season인 사인파 생성
# Sine wave with period = season using sin(2πt/period) form
# 진폭 2.0: 계절 효과의 크기 / Amplitude 2.0: magnitude of seasonal effect
seasonal = 2.0 * np.sin(2 * np.pi * t / season)

# -------------------------------
# 1-4. 최종 시계열 구성 / Construct Final Time Series
# -------------------------------
# 관측 시계열 = 추세(랜덤워크) + 계절성 + 관측 잡음
# Observed series = Trend (random walk) + Seasonality + Observation noise
# y_t = rw_t + seasonal_t + noise_t
# 관측 잡음: 평균 0, 표준편차 0.5인 정규분포 / Observation noise: N(0, 0.5²)
y = rw + seasonal + np.random.normal(0, 0.5, size=n)

# -------------------------------
# 1-5. 시계열 객체로 변환 / Convert to Time Series Object
# -------------------------------
# pandas Series로 변환하고 날짜 인덱스 부여 (일간 데이터)
# Convert to pandas Series with date index (daily frequency)
# 시작일: 2018-01-01, 빈도: 일간(D) / Start date: 2018-01-01, Frequency: Daily
ts = pd.Series(y, index=pd.date_range("2018-01-01", periods=n, freq="D"))

# -------------------------------
# 2. 원시계열 시각화 / Visualize Original Time Series
# -------------------------------
# 목적: 비정상성(추세, 계절성)을 시각적으로 확인
# Purpose: Visually confirm non-stationarity (trend, seasonality)
plt.figure(figsize=(9, 4))  # 그림 크기 설정: 가로 9인치, 세로 4인치 / Set figure size: 9x4 inches
plt.plot(ts)  # 시계열 플롯 / Plot time series
plt.title("[Step 1] 원시계열: 비정상(추세+계절) 직관 확인")  # 제목 / Title
plt.xlabel("Date")  # x축 레이블: 날짜 / x-axis label: Date
plt.ylabel("Value")  # y축 레이블: 값 / y-axis label: Value
plt.tight_layout()  # 레이아웃 자동 조정 (여백 최적화) / Auto-adjust layout
plt.savefig(
    MAIN_ANALYSIS_DIR / "step1_original_series.png", dpi=100, bbox_inches="tight"
)  # PNG 파일로 저장 / Save as PNG
plt.close()  # 메모리 해제 (figure 닫기) / Free memory (close figure)

# -------------------------------
# 3. 정상성 확인: ADF 검정 / Stationarity Check: ADF Test
# -------------------------------
# ADF(Augmented Dickey-Fuller) 검정: 단위근 존재 여부를 검정하는 통계적 방법
# ADF test: Statistical test for unit root presence
#
# 가설 / Hypotheses:
# - H0 (귀무가설): 단위근이 존재한다 (시계열이 비정상) / Unit root exists (non-stationary)
# - H1 (대립가설): 단위근이 없다 (시계열이 정상) / No unit root (stationary)
#
# 판단 기준 / Decision criteria:
# - p-value < 0.05: H0 기각 → 정상 시계열 / Reject H0 → Stationary
# - p-value >= 0.05: H0 채택 → 비정상 시계열 / Accept H0 → Non-stationary
#
# ADF 통계량이 임계값보다 작으면 정상성 지지
# ADF statistic smaller than critical value supports stationarity
adf_stat, adf_p, _, _, critical, _ = adfuller(ts)
print("[Step 2] ADF 원시계열")
print(f"  ADF 통계량: {adf_stat:.3f}, p-value: {adf_p:.4f}")
print(f"  임계값(1%/5%/10%): {critical}")

# -------------------------------
# 3-1. 원시계열 ACF/PACF 시각화 / Original Series ACF/PACF Visualization
# -------------------------------
# 목적: 비정상 시계열의 ACF/PACF 특성 확인
# Purpose: Observe ACF/PACF characteristics of non-stationary series
#
# 비정상 시계열의 ACF 특성 / Non-stationary series ACF characteristics:
# - 매우 느리게 감쇠 (천천히 0으로 수렴)
#   Very slow decay (slowly converges to 0)
# - 랜덤워크/추세가 있으면 높은 자기상관이 오래 유지
#   High autocorrelation persists with random walk/trend
#
# 정상 시계열의 ACF 특성 / Stationary series ACF characteristics:
# - 빠르게 0으로 수렴 또는 특정 lag에서 절단
#   Quickly converges to 0 or cuts off at specific lag
#
# 이 시각화를 통해 차분 전후의 ACF/PACF 패턴 변화를 직접 비교할 수 있습니다.
# This visualization enables direct comparison of ACF/PACF patterns before and after differencing.

# 원시계열 ACF 플롯 / Original series ACF plot
# ACF: 비정상 시계열에서는 느린 감쇠가 관찰됨
# ACF: Slow decay is observed in non-stationary series
plt.figure(figsize=(9, 4))
plot_acf(ts, lags=36)  # 36개 시차까지 ACF 계산 및 플롯 / Compute and plot ACF up to lag 36
plt.title("[Step 2-1] ACF (원시계열): 느린 감쇠 = 비정상 신호")
plt.tight_layout()
plt.savefig(MAIN_ANALYSIS_DIR / "step2_acf_original.png", dpi=100, bbox_inches="tight")
plt.close()

# 원시계열 PACF 플롯 / Original series PACF plot
# PACF: 비정상 시계열에서는 lag 1에서 높은 값, 이후 급격히 감소
# PACF: High value at lag 1 in non-stationary series, then drops sharply
plt.figure(figsize=(9, 4))
plot_pacf(ts, lags=36, method="yw")  # Yule-Walker 방법 사용 / Use Yule-Walker method
plt.title("[Step 2-1] PACF (원시계열)")
plt.tight_layout()
plt.savefig(MAIN_ANALYSIS_DIR / "step2_pacf_original.png", dpi=100, bbox_inches="tight")
plt.close()

# -------------------------------
# 4. 차분으로 정상화 / Differencing for Stationarity
# -------------------------------
# 차분(Differencing): 시계열의 추세를 제거하여 정상성을 달성하는 방법
# Differencing: Method to remove trend and achieve stationarity
#
# 1차 차분 공식 / First-order differencing formula:
# ∇y_t = y_t - y_{t-1}
#
# ARIMA(p,d,q)에서 d는 차분 차수를 의미
# In ARIMA(p,d,q), d represents the order of differencing
# d=0: 원시계열 사용 (이미 정상) / Use original series (already stationary)
# d=1: 1차 차분 적용 (추세 제거) / Apply 1st-order differencing (remove trend)
# d=2: 2차 차분 적용 (곡선 추세 제거) / Apply 2nd-order differencing (remove curved trend)
ts_d1 = (
    ts.diff().dropna()
)  # diff(): 1차 차분 계산, dropna(): 첫 번째 NaN 제거 / Compute 1st diff, remove first NaN

# 차분 후 시계열 시각화 / Visualize differenced series
plt.figure(figsize=(9, 4))
plt.plot(ts_d1)  # 1차 차분 시계열 플롯 / Plot 1st differenced series
plt.title("[Step 3] 1차 차분(d=1): 추세 제거 후 모습")
plt.xlabel("Date")
plt.ylabel("Diff(1)")  # 차분값 / Differenced value
plt.tight_layout()
plt.savefig(MAIN_ANALYSIS_DIR / "step3_differenced_series.png", dpi=100, bbox_inches="tight")
plt.close()

# 차분 후 ADF 검정: 정상성 달성 여부 확인 / ADF test after differencing: Check if stationarity achieved
adf_stat_d1, adf_p_d1, *_ = adfuller(ts_d1)
print("[Step 3] ADF 1차 차분")
print(f"  ADF 통계량: {adf_stat_d1:.3f}, p-value: {adf_p_d1:.4f}")
# 기대 결과: p-value < 0.05이면 정상성 달성 / Expected: p-value < 0.05 means stationarity achieved

# - (옵션) 계절 차분: season 주기 차분으로 계절성 제거
#   필요시 주석 해제하여 사용하세요.
# ts_sd = ts_d1.diff(season).dropna()
# plt.figure(figsize=(9, 4))
# plt.plot(ts_sd)
# plt.title("[옵션] 계절 차분: 계절성 제거")
# plt.xlabel("Date")
# plt.ylabel("Seasonal Diff")
# plt.tight_layout()
# plt.show()

# -------------------------------
# 5. ACF/PACF로 p,q 초기 감 잡기 / Identify p,q using ACF/PACF
# -------------------------------
# ACF (Autocorrelation Function, 자기상관함수):
# 시차(lag) k에서의 자기상관 계수를 시각화
# Visualizes autocorrelation coefficients at different lags
#
# PACF (Partial Autocorrelation Function, 편자기상관함수):
# 중간 시차의 영향을 제거한 순수 자기상관 계수
# Pure autocorrelation removing effects of intermediate lags
#
# ARIMA(p,d,q) 식별 규칙 / ARIMA(p,d,q) identification rules:
# 1. AR(p) 모형: PACF가 lag p에서 절단(cutoff), ACF는 지수 감쇠(decay)
#    AR(p) model: PACF cuts off at lag p, ACF decays exponentially
#
# 2. MA(q) 모형: ACF가 lag q에서 절단, PACF는 지수 감쇠
#    MA(q) model: ACF cuts off at lag q, PACF decays exponentially
#
# 3. ARMA(p,q): ACF와 PACF 모두 지수 감쇠 (혼합 패턴)
#    ARMA(p,q): Both ACF and PACF decay exponentially (mixed pattern)

# ACF 플롯: MA 차수(q) 식별에 유용 / ACF plot: Useful for identifying MA order (q)
plt.figure(figsize=(9, 4))
plot_acf(ts_d1, lags=36)  # 36개 시차까지 ACF 계산 및 플롯 / Compute and plot ACF up to lag 36
plt.title("[Step 4] ACF (1차 차분 데이터)")
plt.tight_layout()
plt.savefig(MAIN_ANALYSIS_DIR / "step4_acf_differenced.png", dpi=100, bbox_inches="tight")
plt.close()

# PACF 플롯: AR 차수(p) 식별에 유용 / PACF plot: Useful for identifying AR order (p)
plt.figure(figsize=(9, 4))
plot_pacf(ts_d1, lags=36, method="yw")  # Yule-Walker 방법 사용 / Use Yule-Walker method
plt.title("[Step 4] PACF (1차 차분 데이터)")
plt.tight_layout()
plt.savefig(MAIN_ANALYSIS_DIR / "step4_pacf_differenced.png", dpi=100, bbox_inches="tight")
plt.close()

# -------------------------------
# 6. 후보 ARIMA 적합 및 정보 기준 비교 / Fit Candidate ARIMA Models and Compare Information Criteria
# -------------------------------
# ARIMA(p, d, q) 모형 / ARIMA(p, d, q) model:
# - p: AR(AutoRegressive) 차수 / AR order
# - d: 차분(Differencing) 차수 / Differencing order
# - q: MA(Moving Average) 차수 / MA order
#
# 정보 기준 (Information Criteria):
# 모형의 복잡도와 적합도를 동시에 고려하여 최적 모형 선택
# Consider both model complexity and goodness-of-fit for optimal model selection
#
# 1. AIC (Akaike Information Criterion, 아카이케 정보 기준):
#    AIC = -2*log(L) + 2*k
#    L: 우도(likelihood), k: 파라미터 개수
#    작을수록 좋음 / Lower is better
#
# 2. BIC (Bayesian Information Criterion, 베이지안 정보 기준):
#    BIC = -2*log(L) + k*log(n)
#    n: 샘플 크기, BIC는 AIC보다 파라미터 페널티가 큼
#    Large sample size, stronger penalty than AIC
#    작을수록 좋음 / Lower is better

# 후보 모형 리스트: (p, d, q) 조합 / Candidate models: (p, d, q) combinations
candidates = [
    (1, 1, 0),  # ARIMA(1,1,0): AR(1) with 1st differencing
    (0, 1, 1),  # ARIMA(0,1,1): MA(1) with 1st differencing
    (1, 1, 1),  # ARIMA(1,1,1): ARMA(1,1) with 1st differencing
    (2, 1, 1),  # ARIMA(2,1,1): ARMA(2,1) with 1st differencing
    (1, 1, 2),
]  # ARIMA(1,1,2): ARMA(1,2) with 1st differencing
info_rows = []  # 결과 저장용 리스트 / List to store results

# 각 후보 모형을 적합하고 AIC/BIC 계산 / Fit each candidate model and compute AIC/BIC
for od in candidates:
    try:
        # ARIMA 모형 객체 생성 (원시계열 사용, 차분은 모형 내부에서 수행)
        # Create ARIMA model object (uses original series, differencing done internally)
        model = ARIMA(ts, order=od)

        # 모형 적합 (최대우도추정법 사용) / Fit model (using maximum likelihood estimation)
        res = model.fit()

        # 결과 저장 / Store results
        info_rows.append({"order": od, "AIC": res.aic, "BIC": res.bic})
        print(f"[Step 5] Fitted ARIMA{od}: AIC={res.aic:.2f}, BIC={res.bic:.2f}")
    except Exception as e:
        # 적합 실패 시 오류 메시지 출력 / Print error message if fitting fails
        print(f"[Step 5] ARIMA{od} 적합 실패: {e}")

# 결과를 DataFrame으로 변환하고 AIC 기준 오름차순 정렬
# Convert results to DataFrame and sort by AIC (ascending)
if len(info_rows) > 0:
    info_df = pd.DataFrame(info_rows).sort_values("AIC")
    print("\n[Step 5] 후보 비교 (AIC/BIC 낮을수록 우수)")
    print(info_df)

# -------------------------------
# 7. 최종 모형 선택 및 잔차 진단 / Final Model Selection and Residual Diagnostics
# -------------------------------
# AIC가 가장 낮은 모형을 최종 모형으로 선택
# Select model with lowest AIC as the final model
best_order = info_df.iloc[0]["order"] if len(info_rows) > 0 else (1, 1, 1)
print(f"\n[Step 6] 선택한 최종 모형: ARIMA{best_order}")

# 선택된 모형으로 재적합 / Refit with selected model
best_model = ARIMA(ts, order=tuple(best_order))
best_res = best_model.fit()

# 모형 요약 출력: 계수 추정값, 표준오차, p-value 등
# Print model summary: coefficient estimates, standard errors, p-values, etc.
print(best_res.summary())

# -------------------------------
# 7-1. 잔차 추출 및 시각화 / Extract and Visualize Residuals
# -------------------------------
# 잔차(Residuals): 관측값 - 예측값 = y_t - ŷ_t
# Residuals: Observed - Predicted = y_t - ŷ_t
# 잘 적합된 모형의 잔차는 백색잡음(평균 0, 일정한 분산, 독립)이어야 함
# Well-fitted model should have white noise residuals (mean 0, constant variance, independent)
resid = best_res.resid

# 잔차 시계열 플롯: 패턴이 없어야 함 / Residual time series plot: Should show no pattern
plt.figure(figsize=(9, 4))
plt.plot(resid)
plt.title("[Step 6] 잔차(innovation) 시계열")
plt.xlabel("Date")
plt.ylabel("Residual")
plt.tight_layout()
plt.savefig(MAIN_ANALYSIS_DIR / "step6_residuals.png", dpi=100, bbox_inches="tight")
plt.close()

# -------------------------------
# 7-2. 잔차 ACF 플롯 / Residual ACF Plot
# -------------------------------
# 잔차 ACF: 잔차의 자기상관 확인
# Residual ACF: Check autocorrelation in residuals
# 신뢰구간(파란색 밴드) 안에 있으면 자기상관 없음을 의미
# Values within confidence band (blue band) indicate no autocorrelation
plt.figure(figsize=(9, 4))
plot_acf(resid, lags=36)
plt.title("[Step 6] 잔차 ACF (독립성 확인)")
plt.tight_layout()
plt.savefig(MAIN_ANALYSIS_DIR / "step6_residuals_acf.png", dpi=100, bbox_inches="tight")
plt.close()

# -------------------------------
# 7-3. Ljung-Box 검정 / Ljung-Box Test
# -------------------------------
# Ljung-Box 검정: 잔차의 자기상관 여부를 통계적으로 검정
# Ljung-Box test: Statistical test for autocorrelation in residuals
#
# 가설 / Hypotheses:
# - H0: 잔차에 자기상관 없음 (모형이 적절) / No autocorrelation (model adequate)
# - H1: 잔차에 자기상관 있음 (모형 부적절) / Autocorrelation exists (model inadequate)
#
# 판단 기준 / Decision criteria:
# - p-value > 0.05: H0 채택, 잔차 독립적 (좋음) / Accept H0, residuals independent (good)
# - p-value <= 0.05: H0 기각, 잔차에 패턴 존재 (나쁨) / Reject H0, pattern exists (bad)
lb = acorr_ljungbox(resid, lags=[12, 24], return_df=True)
print("\n[Step 6] Ljung-Box 잔차 독립성 검정(라그 12, 24)")
print(lb)

# -------------------------------
# 8. 예측 (h-스텝 앞) 및 신뢰구간 / Forecasting and Confidence Intervals
# -------------------------------
# ARIMA 모형의 주요 목적 중 하나는 미래 값 예측
# One of main purposes of ARIMA is forecasting future values
#
# h-스텝 앞 예측 (h-step ahead forecast):
# 현재 시점에서 h 기간 후의 값을 예측
# Predict value h periods ahead from current time point
h = 24  # 예측 기간: 24단계(24일) 앞 / Forecast horizon: 24 steps (24 days) ahead

# get_forecast(): 예측값, 표준오차, 신뢰구간 계산
# get_forecast(): Compute forecasts, standard errors, and confidence intervals
forecast_res = best_res.get_forecast(steps=h)

# 예측값 (점 추정) / Point forecasts
f_mean = forecast_res.predicted_mean

# 95% 신뢰구간 (불확실성 표현) / 95% confidence intervals (uncertainty quantification)
# 신뢰구간이 넓을수록 예측 불확실성이 큼
# Wider interval indicates greater forecast uncertainty
f_ci = forecast_res.conf_int()

# 예측 결과 시각화 / Visualize forecast results
plt.figure(figsize=(10, 4))
plt.plot(ts, label="Observed")  # 관측값 (과거 데이터) / Observed values (historical data)
plt.plot(f_mean.index, f_mean, label="Forecast", color="red")  # 예측값 / Forecast
# 신뢰구간 음영 처리 / Shade confidence interval
plt.fill_between(
    f_ci.index, f_ci.iloc[:, 0], f_ci.iloc[:, 1], alpha=0.2, label="95% CI", color="red"
)
plt.title(f"[Step 7] {h}-스텝 앞 예측 및 95% 신뢰구간")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()  # 범례 표시 / Show legend
plt.tight_layout()
plt.savefig(MAIN_ANALYSIS_DIR / "step7_forecast.png", dpi=100, bbox_inches="tight")
plt.close()

# -------------------------------
# 8. SARIMA 모델로 계절성 캡처 / Capture Seasonality with SARIMA Model
# -------------------------------
# SARIMA(p,d,q)(P,D,Q)[s] 모델:
# - (p,d,q): 비계절 ARIMA 파라미터 / Non-seasonal ARIMA parameters
# - (P,D,Q): 계절 ARIMA 파라미터 / Seasonal ARIMA parameters
# - s: 계절 주기 (본 데이터에서 s=24) / Seasonal period (s=24 in this data)
#
# SARIMA vs ARIMA 차이 / Difference between SARIMA and ARIMA:
# - ARIMA: 계절 패턴 무시 → 예측이 상수로 수렴
#   Ignores seasonal pattern → Forecast converges to constant
# - SARIMA: 계절 패턴 명시적 모델링 → 주기적 예측 가능
#   Explicitly models seasonal pattern → Periodic forecast possible
#
# 왜 ARIMA 예측이 상수로 수렴하는가? / Why does ARIMA forecast converge to constant?
# -----------------------------------------------------------------
# ARIMA(1,1,0) 모델은 다음 형태를 따름 / ARIMA(1,1,0) follows this form:
#   ∇y_t = φ₁·∇y_{t-1} + ε_t  (where ∇ = 차분 연산자 / differencing operator)
#
# 장기 예측에서 / In long-horizon forecasting:
# 1. 차분 시계열의 예측이 0으로 수렴 (정상성 가정)
#    Differenced series forecast converges to 0 (stationarity assumption)
# 2. 따라서 원시계열 예측이 마지막 관측값 근처에서 상수로 수렴
#    Therefore original series forecast converges to constant near last observation
# 3. 계절 성분(sin 함수)이 차분으로 완전히 제거되지 않았지만 모형에서 무시됨
#    Seasonal component (sine) not fully removed by differencing, but ignored by model
#
# SARIMA가 필요한 이유 / Why SARIMA is needed:
# - SARIMA(p,d,q)(P,D,Q)[s]는 계절 차분과 계절 AR/MA 항을 포함
#   Includes seasonal differencing and seasonal AR/MA terms
# - 주기 s=24인 계절 패턴을 명시적으로 캡처
#   Explicitly captures seasonal pattern with period s=24
# - 더 현실적인 주기적 예측 가능
#   Enables more realistic periodic forecasting
print("\n" + "=" * 60)
print("[Step 8] SARIMA 모델로 계절성 캡처 / Capture Seasonality with SARIMA")
print("=" * 60)

# SARIMA(1,1,0)(1,0,1)[24] 모델 적합 / Fit SARIMA(1,1,0)(1,0,1)[24] model
# - 비계절 파라미터: (1,1,0) - AR(1) with 1차 차분 / AR(1) with 1st differencing
# - 계절 파라미터: (1,0,1) - 계절 AR(1), 계절 MA(1) / Seasonal AR(1), Seasonal MA(1)
# - 계절 주기: 24 / Seasonal period: 24
sarima_model = SARIMAX(
    ts,
    order=(1, 1, 0),
    seasonal_order=(1, 0, 1, season),  # season=24
    enforce_stationarity=False,  # 수치적 안정성을 위해 완화 / Relax for numerical stability
    enforce_invertibility=False,  # 수치적 안정성을 위해 완화 / Relax for numerical stability
)
sarima_res = sarima_model.fit(disp=False)  # disp=False: 최적화 과정 출력 억제

print("\n[SARIMA 모델 요약 / SARIMA Model Summary]")
print(sarima_res.summary())

# -------------------------------
# 8-1. AIC/BIC 비교 / AIC/BIC Comparison
# -------------------------------
# ARIMA와 SARIMA의 정보 기준 비교 / Compare information criteria between ARIMA and SARIMA
# 낮을수록 더 좋은 모형 / Lower is better
print("\n[AIC/BIC 비교 / AIC/BIC Comparison]")
print("-" * 50)
print(f"ARIMA{best_order}:              AIC={best_res.aic:.2f}, BIC={best_res.bic:.2f}")
print(f"SARIMA(1,1,0)(1,0,1)[{season}]: AIC={sarima_res.aic:.2f}, BIC={sarima_res.bic:.2f}")
print("-" * 50)

# AIC/BIC 차이 해석 / Interpret AIC/BIC difference
aic_diff = best_res.aic - sarima_res.aic
if aic_diff > 0:
    print(f"→ SARIMA가 AIC 기준 {aic_diff:.2f} 더 우수 / SARIMA better by {aic_diff:.2f} in AIC")
else:
    print(f"→ ARIMA가 AIC 기준 {-aic_diff:.2f} 더 우수 / ARIMA better by {-aic_diff:.2f} in AIC")
print("  (단, SARIMA는 계절 예측 능력에서 ARIMA보다 우수)")
print("  (However, SARIMA excels in seasonal forecasting capability)")

# -------------------------------
# 8-2. ARIMA vs SARIMA 예측 비교 시각화 / ARIMA vs SARIMA Forecast Comparison Visualization
# -------------------------------
# 예측 기간: 48 스텝 (2주기) / Forecast horizon: 48 steps (2 periods)
# 2주기를 예측하여 계절 패턴의 유지 여부를 명확히 비교
# Forecast 2 periods to clearly compare whether seasonal pattern is maintained
h_compare = 48  # 48 스텝 = 2 × 24 (2주기) / 48 steps = 2 × 24 (2 periods)

# ARIMA 예측 / ARIMA forecast
arima_forecast = best_res.get_forecast(steps=h_compare)
arima_f_mean = arima_forecast.predicted_mean
arima_f_ci = arima_forecast.conf_int()

# SARIMA 예측 / SARIMA forecast
sarima_forecast = sarima_res.get_forecast(steps=h_compare)
sarima_f_mean = sarima_forecast.predicted_mean
sarima_f_ci = sarima_forecast.conf_int()

# 비교 시각화: 2행 1열 서브플롯 / Comparison visualization: 2 rows, 1 column subplot
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

# 관측값의 마지막 100개만 표시 (시각적 명확성) / Show only last 100 observations (visual clarity)
ts_recent = ts[-100:]

# 상단: ARIMA 예측 / Top: ARIMA Forecast
axes[0].plot(ts_recent.index, ts_recent.values, "b-", linewidth=1, label="관측값 (Observed)")
axes[0].plot(arima_f_mean.index, arima_f_mean.values, "r-", linewidth=2, label="ARIMA 예측 (Forecast)")
axes[0].fill_between(
    arima_f_ci.index,
    arima_f_ci.iloc[:, 0],
    arima_f_ci.iloc[:, 1],
    alpha=0.2,
    color="red",
    label="95% 신뢰구간 (CI)",
)
axes[0].axhline(
    y=arima_f_mean.iloc[-1],
    color="gray",
    linestyle="--",
    linewidth=0.8,
    alpha=0.7,
)
axes[0].set_title(
    f"[비교] ARIMA{best_order}: 예측이 ~{arima_f_mean.iloc[-1]:.2f}로 수렴 (계절 패턴 무시)"
)
axes[0].set_ylabel("Value")
axes[0].legend(loc="upper left")
axes[0].grid(True, alpha=0.3)

# 하단: SARIMA 예측 / Bottom: SARIMA Forecast
axes[1].plot(ts_recent.index, ts_recent.values, "b-", linewidth=1, label="관측값 (Observed)")
axes[1].plot(
    sarima_f_mean.index, sarima_f_mean.values, "g-", linewidth=2, label="SARIMA 예측 (Forecast)"
)
axes[1].fill_between(
    sarima_f_ci.index,
    sarima_f_ci.iloc[:, 0],
    sarima_f_ci.iloc[:, 1],
    alpha=0.2,
    color="green",
    label="95% 신뢰구간 (CI)",
)
axes[1].set_title(
    f"[비교] SARIMA(1,1,0)(1,0,1)[{season}]: 계절 패턴 유지 (주기적 예측)"
)
axes[1].set_xlabel("Date")
axes[1].set_ylabel("Value")
axes[1].legend(loc="upper left")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(MAIN_ANALYSIS_DIR / "step8_sarima_comparison.png", dpi=100, bbox_inches="tight")
plt.close()

print("\n[Step 8] ARIMA vs SARIMA 비교 시각화 저장 완료")
print(f"  → {MAIN_ANALYSIS_DIR / 'step8_sarima_comparison.png'}")

# -------------------------------
# 8-3. 교육적 요약 / Educational Summary
# -------------------------------
print("\n[Step 8 교육적 요약 / Educational Summary]")
print("-" * 60)
print("| 모델              | 예측 특성           | 학습 포인트              |")
print("|-------------------|---------------------|--------------------------|")
print(f"| ARIMA{best_order}       | 상수 수렴           | 비계절 모델의 한계 이해  |")
print("| SARIMA(1,1,0)     | 주기적 패턴         | 계절 모델의 필요성 체감  |")
print("|   (1,0,1)[24]     |                     |                          |")
print("-" * 60)
print("\n핵심 인사이트 / Key Insights:")
print("1. 데이터에 계절성이 있으면 ARIMA만으로는 부족")
print("   If data has seasonality, ARIMA alone is insufficient")
print("2. SARIMA의 계절 파라미터 (P,D,Q)[s]가 계절 패턴을 캡처")
print("   SARIMA's seasonal parameters (P,D,Q)[s] capture seasonal pattern")
print("3. 실무에서는 ACF에서 주기적 스파이크가 보이면 SARIMA 고려")
print("   In practice, consider SARIMA if ACF shows periodic spikes")
print("=" * 60)

# -------------------------------
# 9. p/d/q 변화 효과 체감 시뮬레이션 (추가 학습) / Simulate AR/MA/ARMA Effects (Additional Learning)
# -------------------------------
# 목적: AR, MA, ARMA 모형의 특성을 시각적으로 이해
# Purpose: Visually understand characteristics of AR, MA, and ARMA models
#
# 각 모형의 ACF/PACF 패턴을 직접 시뮬레이션하여 확인
# Simulate and observe ACF/PACF patterns of each model type
m = 300  # 시뮬레이션 데이터 길이 / Simulation data length
e = np.random.normal(0, 1.0, size=m)  # 백색잡음 생성 / Generate white noise

# -------------------------------
# 9-1. AR(1) 모형 시뮬레이션 / AR(1) Model Simulation
# -------------------------------
# AR(1) 모형: AutoRegressive model of order 1
# 수식 / Equation: y_t = φ₁·y_{t-1} + ε_t
# 여기서 φ₁ = 0.7 (자기회귀 계수 / autoregressive coefficient)
#
# 특성 / Characteristics:
# - 과거 값이 현재 값에 직접 영향 (관성 효과)
# - Past values directly influence current value (inertia effect)
# - |φ₁| < 1이면 정상 시계열 / Stationary if |φ₁| < 1
# - φ₁이 양수이면 매끄러운 추세, 음수이면 진동
# - Positive φ₁ creates smooth trend, negative creates oscillation
ar = np.zeros(m)
for i in range(1, m):
    ar[i] = 0.7 * ar[i - 1] + e[i]  # AR(1) 프로세스 생성 / Generate AR(1) process
ar_series = pd.Series(ar)

# AR(1) 시계열 플롯: 값들이 서로 연결되어 매끄럽게 변화
# AR(1) time series plot: Values smoothly connected (inertia)
plt.figure(figsize=(9, 3.5))
plt.plot(ar_series)
plt.title("[추가] AR(1): 값의 관성(매끄러운 흐름)")
plt.tight_layout()
plt.savefig(MODEL_COMPARISON_DIR / "additional_ar1_series.png", dpi=100, bbox_inches="tight")
plt.close()

# AR(1) ACF: 지수적으로 감쇠 (기하급수적 감소)
# AR(1) ACF: Exponentially decays
plt.figure(figsize=(9, 3.5))
plot_acf(ar_series, lags=36)
plt.title("[추가] AR(1) ACF: 지수 감쇠")
plt.tight_layout()
plt.savefig(MODEL_COMPARISON_DIR / "additional_ar1_acf.png", dpi=100, bbox_inches="tight")
plt.close()

# AR(1) PACF: lag 1에서 절단 (lag 2 이상은 0에 가까움)
# AR(1) PACF: Cuts off at lag 1 (lags 2+ near zero)
# 이것이 AR(1) 식별의 핵심 패턴 / This is key pattern for AR(1) identification
plt.figure(figsize=(9, 3.5))
plot_pacf(ar_series, lags=36, method="yw")
plt.title("[추가] AR(1) PACF: 1에서 절단 경향")
plt.tight_layout()
plt.savefig(MODEL_COMPARISON_DIR / "additional_ar1_pacf.png", dpi=100, bbox_inches="tight")
plt.close()

# -------------------------------
# 9-2. MA(1) 모형 시뮬레이션 / MA(1) Model Simulation
# -------------------------------
# MA(1) 모형: Moving Average model of order 1
# 수식 / Equation: y_t = ε_t + θ₁·ε_{t-1}
# 여기서 θ₁ = 0.7 (이동평균 계수 / moving average coefficient)
#
# 특성 / Characteristics:
# - 현재 및 과거 1기의 충격(noise)만 영향 (단기 메모리)
# - Only current and 1-period lagged shocks affect value (short memory)
# - 충격의 영향이 1기 후에 완전히 소멸 (빠른 감쇠)
# - Shock effect completely dies out after 1 period (fast decay)
# - |θ₁| < 1이면 역변환 가능 (invertible)
ma = np.zeros(m)
for i in range(1, m):
    ma[i] = e[i] + 0.7 * e[i - 1]  # MA(1) 프로세스 생성 / Generate MA(1) process
ma_series = pd.Series(ma)

# MA(1) 시계열 플롯: AR보다 덜 매끄럽고 변동이 큼 (충격이 즉시 반영)
# MA(1) time series plot: Less smooth than AR, more volatile (shocks immediately reflected)
plt.figure(figsize=(9, 3.5))
plt.plot(ma_series)
plt.title("[추가] MA(1): 충격의 잔향(짧은 요동)")
plt.tight_layout()
plt.savefig(MODEL_COMPARISON_DIR / "additional_ma1_series.png", dpi=100, bbox_inches="tight")
plt.close()

# MA(1) ACF: lag 1에서 절단 (lag 2 이상은 0에 가까움)
# MA(1) ACF: Cuts off at lag 1 (lags 2+ near zero)
# 이것이 MA(1) 식별의 핵심 패턴 / This is key pattern for MA(1) identification
plt.figure(figsize=(9, 3.5))
plot_acf(ma_series, lags=36)
plt.title("[추가] MA(1) ACF: 1에서 절단 경향")
plt.tight_layout()
plt.savefig(MODEL_COMPARISON_DIR / "additional_ma1_acf.png", dpi=100, bbox_inches="tight")
plt.close()

# MA(1) PACF: 지수적으로 감쇠
# MA(1) PACF: Exponentially decays
plt.figure(figsize=(9, 3.5))
plot_pacf(ma_series, lags=36, method="yw")
plt.title("[추가] MA(1) PACF: 지수 감쇠")
plt.tight_layout()
plt.savefig(MODEL_COMPARISON_DIR / "additional_ma1_pacf.png", dpi=100, bbox_inches="tight")
plt.close()

# -------------------------------
# 9-3. ARMA(1,1) 모형 시뮬레이션 / ARMA(1,1) Model Simulation
# -------------------------------
# ARMA(1,1) 모형: Combined AutoRegressive and Moving Average model
# 수식 / Equation: y_t = φ₁·y_{t-1} + ε_t + θ₁·ε_{t-1}
# 여기서 φ₁ = 0.5, θ₁ = 0.5 / where φ₁ = 0.5, θ₁ = 0.5
#
# 특성 / Characteristics:
# - AR과 MA의 특성을 모두 포함 (관성 + 충격)
# - Combines both AR and MA characteristics (inertia + shock)
# - 과거 값과 과거 충격이 함께 영향
# - Both past values and past shocks influence current value
# - 더 복잡한 패턴을 모델링 가능
# - Can model more complex patterns
arma = np.zeros(m)
for i in range(1, m):
    arma[i] = 0.5 * arma[i - 1] + e[i] + 0.5 * e[i - 1]  # ARMA(1,1) 프로세스 생성
arma_series = pd.Series(arma)

# ARMA(1,1) 시계열 플롯: AR과 MA의 중간 특성
# ARMA(1,1) time series plot: Intermediate characteristics between AR and MA
plt.figure(figsize=(9, 3.5))
plt.plot(arma_series)
plt.title("[추가] ARMA(1,1): 관성 + 잔향 혼합")
plt.tight_layout()
plt.savefig(MODEL_COMPARISON_DIR / "additional_arma11_series.png", dpi=100, bbox_inches="tight")
plt.close()

# ARMA(1,1) ACF: 지수 감쇠 (절단되지 않음)
# ARMA(1,1) ACF: Exponentially decays (no cutoff)
plt.figure(figsize=(9, 3.5))
plot_acf(arma_series, lags=36)
plt.title("[추가] ARMA(1,1) ACF: 감쇠 패턴")
plt.tight_layout()
plt.savefig(MODEL_COMPARISON_DIR / "additional_arma11_acf.png", dpi=100, bbox_inches="tight")
plt.close()

# ARMA(1,1) PACF: 지수 감쇠 (절단되지 않음)
# ARMA(1,1) PACF: Exponentially decays (no cutoff)
# ACF와 PACF 모두 감쇠 → ARMA 모형 식별 패턴
# Both ACF and PACF decay → Pattern for identifying ARMA models
plt.figure(figsize=(9, 3.5))
plot_pacf(arma_series, lags=36, method="yw")
plt.title("[추가] ARMA(1,1) PACF: 감쇠 패턴")
plt.tight_layout()
plt.savefig(MODEL_COMPARISON_DIR / "additional_arma11_pacf.png", dpi=100, bbox_inches="tight")
plt.close()

# -------------------------------
# 9-4. 랜덤워크 vs 정상 AR(1) 비교 시뮬레이션 / Random Walk vs Stationary AR(1) Comparison
# -------------------------------
# 목적: 단위근(Unit Root)의 존재 여부가 시계열에 미치는 영향을 시각적으로 비교
# Purpose: Visually compare effects of unit root presence on time series
#
# 두 가지 모형 비교:
# 1. 랜덤워크 (Random Walk): y_t = y_{t-1} + ε_t (φ = 1, 단위근 존재)
#    - 비정상(Non-stationary) 시계열
#    - 분산이 시간에 따라 무한히 증가: Var(y_t) = t·σ²
#    - 평균으로 회귀하지 않음 (no mean reversion)
#    - ADF 검정: p-value > 0.05 (귀무가설 채택, 비정상)
#
# 2. 정상 AR(1): y_t = φ·y_{t-1} + ε_t (φ = 0.8, |φ| < 1)
#    - 정상(Stationary) 시계열
#    - 분산이 일정: Var(y_t) = σ²/(1-φ²)
#    - 평균으로 회귀 (mean reversion)
#    - ADF 검정: p-value < 0.05 (귀무가설 기각, 정상)

print("\n" + "=" * 60)
print("[추가 시뮬레이션] 랜덤워크(φ=1) vs 정상 AR(1)(φ=0.8) 비교")
print("=" * 60)

# 시뮬레이션 설정 / Simulation setup
n_sim = 500  # 시뮬레이션 길이 / Simulation length
np.random.seed(123)  # 재현성을 위한 시드 (다른 패턴 생성) / Different seed for variety
eps_sim = np.random.normal(0, 1, size=n_sim)  # 동일한 백색잡음 사용 / Same white noise

# -------------------------------
# 랜덤워크 생성 / Generate Random Walk
# -------------------------------
# 수식: y_t = y_{t-1} + ε_t (φ = 1)
# Equation: y_t = y_{t-1} + ε_t (φ = 1)
# 이는 y_t = Σ(ε_i) for i=1 to t (누적합)
# This equals y_t = Σ(ε_i) for i=1 to t (cumulative sum)
random_walk = np.cumsum(eps_sim)

# -------------------------------
# 정상 AR(1) 생성 / Generate Stationary AR(1)
# -------------------------------
# 수식: y_t = 0.8·y_{t-1} + ε_t (φ = 0.8)
# Equation: y_t = 0.8·y_{t-1} + ε_t (φ = 0.8)
# |φ| = 0.8 < 1 이므로 정상 시계열
# |φ| = 0.8 < 1, therefore stationary series
phi = 0.8  # AR(1) 계수 / AR(1) coefficient
ar1_stationary = np.zeros(n_sim)
for i in range(1, n_sim):
    ar1_stationary[i] = phi * ar1_stationary[i - 1] + eps_sim[i]

# -------------------------------
# 시각화 1: 시계열 비교 / Visualization 1: Time Series Comparison
# -------------------------------
# 랜덤워크는 무한히 발산하고, AR(1)은 평균 주위에서 진동
# Random walk diverges infinitely, AR(1) oscillates around mean
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# 랜덤워크 플롯 / Random walk plot
axes[0].plot(random_walk, color="red", linewidth=0.8)
axes[0].axhline(y=0, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
axes[0].set_title("[비교] 랜덤워크 (φ=1, 단위근): 평균으로 회귀하지 않음")
axes[0].set_ylabel("Value")
axes[0].legend(["Random Walk (Unit Root)", "Mean = 0"], loc="upper left")

# 정상 AR(1) 플롯 / Stationary AR(1) plot
axes[1].plot(ar1_stationary, color="blue", linewidth=0.8)
axes[1].axhline(y=0, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
axes[1].set_title("[비교] 정상 AR(1) (φ=0.8): 평균으로 회귀 (Mean Reversion)")
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Value")
axes[1].legend(["Stationary AR(1)", "Mean = 0"], loc="upper left")

plt.tight_layout()
plt.savefig(UNIT_ROOT_DIR / "comparison_rw_vs_ar1_series.png", dpi=100, bbox_inches="tight")
plt.close()

# -------------------------------
# 시각화 2: ACF 비교 / Visualization 2: ACF Comparison
# -------------------------------
# 랜덤워크 ACF: 매우 천천히 감쇠 (거의 1에서 시작하여 서서히 감소)
# Random walk ACF: Very slow decay (starts near 1, decreases slowly)
# 정상 AR(1) ACF: 빠르게 지수 감쇠 (φ^k 패턴)
# Stationary AR(1) ACF: Fast exponential decay (φ^k pattern)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 랜덤워크 ACF / Random walk ACF
plot_acf(random_walk, lags=40, ax=axes[0], title="")
axes[0].set_title("[비교] 랜덤워크 ACF: 매우 느린 감쇠 (비정상 신호)")
axes[0].set_xlabel("Lag")
axes[0].set_ylabel("Autocorrelation")

# 정상 AR(1) ACF / Stationary AR(1) ACF
plot_acf(ar1_stationary, lags=40, ax=axes[1], title="")
axes[1].set_title("[비교] 정상 AR(1) ACF: 빠른 지수 감쇠 (ρ_k = φ^k)")
axes[1].set_xlabel("Lag")
axes[1].set_ylabel("Autocorrelation")

plt.tight_layout()
plt.savefig(UNIT_ROOT_DIR / "comparison_rw_vs_ar1_acf.png", dpi=100, bbox_inches="tight")
plt.close()

# -------------------------------
# ADF 검정 비교 / ADF Test Comparison
# -------------------------------
# 랜덤워크: p-value > 0.05 예상 (비정상)
# Random walk: Expect p-value > 0.05 (non-stationary)
# AR(1): p-value < 0.05 예상 (정상)
# AR(1): Expect p-value < 0.05 (stationary)
adf_rw = adfuller(random_walk)
adf_ar1 = adfuller(ar1_stationary)

print("\n[ADF 검정 결과 비교]")
print("-" * 50)
print(f"랜덤워크 (φ=1):     ADF={adf_rw[0]:.3f}, p-value={adf_rw[1]:.4f}")
print("  → p-value > 0.05: 귀무가설 채택 (비정상, 단위근 존재)")
print(f"정상 AR(1) (φ=0.8): ADF={adf_ar1[0]:.3f}, p-value={adf_ar1[1]:.4f}")
print("  → p-value < 0.05: 귀무가설 기각 (정상, 단위근 없음)")
print("-" * 50)

# -------------------------------
# 시각화 3: 분산 증가 비교 / Visualization 3: Variance Growth Comparison
# -------------------------------
# 랜덤워크의 핵심 특성: 분산이 시간에 따라 선형 증가
# Key property of random walk: Variance grows linearly with time
# Var(y_t) = t·σ² for random walk
# Var(y_t) = σ²/(1-φ²) = 1/(1-0.64) ≈ 2.78 for AR(1) with φ=0.8
#
# 여러 번 시뮬레이션하여 분산의 시간적 변화 관찰
# Simulate multiple times to observe variance evolution over time
n_realizations = 100  # 시뮬레이션 반복 횟수 / Number of realizations
n_time = 300  # 각 시뮬레이션 길이 / Length of each simulation

# 저장 배열 초기화 / Initialize storage arrays
rw_realizations = np.zeros((n_realizations, n_time))
ar1_realizations = np.zeros((n_realizations, n_time))

# 여러 번 시뮬레이션 / Multiple simulations
for r in range(n_realizations):
    eps_r = np.random.normal(0, 1, size=n_time)
    # 랜덤워크 / Random walk
    rw_realizations[r, :] = np.cumsum(eps_r)
    # AR(1) / AR(1)
    for t in range(1, n_time):
        ar1_realizations[r, t] = phi * ar1_realizations[r, t - 1] + eps_r[t]

# 각 시점에서의 표본 분산 계산 / Compute sample variance at each time point
rw_variance = np.var(rw_realizations, axis=0)  # 시간별 분산 / Variance over time
ar1_variance = np.var(ar1_realizations, axis=0)

# 이론적 분산 / Theoretical variance
theoretical_rw_var = np.arange(1, n_time + 1) * 1.0  # Var(y_t) = t·σ²
theoretical_ar1_var = np.ones(n_time) * (1 / (1 - phi**2))  # Var(y_t) = σ²/(1-φ²)

# 분산 비교 플롯 / Variance comparison plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 랜덤워크 분산: 선형 증가 / Random walk variance: Linear growth
axes[0].plot(rw_variance, color="red", linewidth=1.5, label="표본 분산 (Sample Var)")
axes[0].plot(
    theoretical_rw_var, color="black", linestyle="--", linewidth=1.5, label="이론적 분산 (t·σ²)"
)
axes[0].set_title("[비교] 랜덤워크 분산: 시간에 따라 무한 증가")
axes[0].set_xlabel("Time (t)")
axes[0].set_ylabel("Variance")
axes[0].legend()
axes[0].set_ylim([0, max(rw_variance) * 1.1])

# AR(1) 분산: 일정 / AR(1) variance: Constant
axes[1].plot(ar1_variance, color="blue", linewidth=1.5, label="표본 분산 (Sample Var)")
axes[1].plot(
    theoretical_ar1_var,
    color="black",
    linestyle="--",
    linewidth=1.5,
    label=f"이론적 분산 (σ²/(1-φ²) ≈ {1/(1-phi**2):.2f})",
)
axes[1].set_title("[비교] 정상 AR(1) 분산: 시간과 무관하게 일정")
axes[1].set_xlabel("Time (t)")
axes[1].set_ylabel("Variance")
axes[1].legend()
axes[1].set_ylim([0, max(ar1_variance) * 2])

plt.tight_layout()
plt.savefig(UNIT_ROOT_DIR / "comparison_rw_vs_ar1_variance.png", dpi=100, bbox_inches="tight")
plt.close()

# -------------------------------
# 시각화 4: 여러 경로 시뮬레이션 비교 / Visualization 4: Multiple Path Comparison
# -------------------------------
# 5개의 다른 시드로 시뮬레이션하여 경로의 발산/수렴 패턴 확인
# Simulate with 5 different seeds to see divergence/convergence patterns
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for seed_offset in range(5):
    np.random.seed(200 + seed_offset)
    eps_path = np.random.normal(0, 1, size=300)

    # 랜덤워크 경로 / Random walk path
    rw_path = np.cumsum(eps_path)
    axes[0].plot(rw_path, linewidth=0.8, alpha=0.7)

    # AR(1) 경로 / AR(1) path
    ar1_path = np.zeros(300)
    for t in range(1, 300):
        ar1_path[t] = phi * ar1_path[t - 1] + eps_path[t]
    axes[1].plot(ar1_path, linewidth=0.8, alpha=0.7)

axes[0].axhline(y=0, color="black", linestyle="--", linewidth=1)
axes[0].set_title("[비교] 랜덤워크 5개 경로: 각 경로가 발산 (비정상)")
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Value")

axes[1].axhline(y=0, color="black", linestyle="--", linewidth=1)
axes[1].set_title("[비교] 정상 AR(1) 5개 경로: 평균 주위에 머무름 (정상)")
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Value")

plt.tight_layout()
plt.savefig(UNIT_ROOT_DIR / "comparison_rw_vs_ar1_paths.png", dpi=100, bbox_inches="tight")
plt.close()

print("\n[랜덤워크 vs 정상 AR(1) 비교 시뮬레이션 완료]")
print("생성된 파일 (outputs/unit_root_comparison/):")
print("  - comparison_rw_vs_ar1_series.png   : 시계열 비교")
print("  - comparison_rw_vs_ar1_acf.png      : ACF 비교")
print("  - comparison_rw_vs_ar1_variance.png : 분산 증가 비교")
print("  - comparison_rw_vs_ar1_paths.png    : 다중 경로 비교")

# 출력 디렉토리 구조 안내 / Output directory structure guide
print("\n" + "=" * 60)
print("[출력 파일 디렉토리 구조 / Output Directory Structure]")
print("=" * 60)
print("arima/outputs/")
print("├── main_analysis/        (10 files) - 메인 ARIMA/SARIMA 분석")
print("│   ├── step1_original_series.png")
print("│   ├── step2_acf_original.png")
print("│   ├── step2_pacf_original.png")
print("│   ├── step3_differenced_series.png")
print("│   ├── step4_acf_differenced.png")
print("│   ├── step4_pacf_differenced.png")
print("│   ├── step6_residuals.png")
print("│   ├── step6_residuals_acf.png")
print("│   ├── step7_forecast.png")
print("│   └── step8_sarima_comparison.png  ← NEW: ARIMA vs SARIMA 비교")
print("├── model_comparison/     (9 files) - AR/MA/ARMA 모형 비교")
print("│   ├── additional_ar1_*.png (3 files)")
print("│   ├── additional_ma1_*.png (3 files)")
print("│   └── additional_arma11_*.png (3 files)")
print("└── unit_root_comparison/ (4 files) - 단위근 비교")
print("    ├── comparison_rw_vs_ar1_series.png")
print("    ├── comparison_rw_vs_ar1_acf.png")
print("    ├── comparison_rw_vs_ar1_variance.png")
print("    └── comparison_rw_vs_ar1_paths.png")
print("=" * 60)

# -------------------------------
# 11. 학습 체크리스트 / Learning Checklist
# -------------------------------
# ARIMA/SARIMA 모델링의 전체 워크플로우를 요약한 체크리스트
# Checklist summarizing complete ARIMA/SARIMA modeling workflow
#
# 이 9단계를 순서대로 따르면 체계적인 시계열 분석 가능
# Following these 9 steps enables systematic time series analysis
print("\n[체크리스트 요약 / Checklist Summary]")
print("1) 원시계열 관찰: 추세/계절/이분산 유무 파악")
print("   Observe original series: Identify trend/seasonality/heteroskedasticity")
print("2) 정상성 검정: ADF로 단위근 확인 (p<0.05 목표)")
print("   Stationarity test: Check unit root with ADF (target p<0.05)")
print("3) 차분(d) 최소화: d=0/1 중심, 필요시 계절차분(D)")
print("   Minimize differencing order: Focus on d=0/1, seasonal differencing if needed")
print("4) ACF/PACF로 p,q 가늠: AR↔PACF 절단, MA↔ACF 절단(이상적)")
print("   Estimate p,q from ACF/PACF: AR↔PACF cutoff, MA↔ACF cutoff (ideal)")
print("5) 후보 적합: AIC/BIC와 잔차진단 함께 비교")
print("   Fit candidates: Compare using AIC/BIC and residual diagnostics")
print("6) 잔차 독립성: Ljung-Box p-value 충분히 크게")
print("   Residual independence: Ljung-Box p-value sufficiently large")
print("7) 예측 및 불확실성: 신뢰구간 확인")
print("   Forecast with uncertainty: Check confidence intervals")
print("8) 계절성 있으면 SARIMA 적용: SARIMA(p,d,q)(P,D,Q)[s] 비교")
print("   Apply SARIMA if seasonal: Compare SARIMA(p,d,q)(P,D,Q)[s]")
print("9) 필요시 확장: ARIMAX(외생), GARCH(이분산)")
print("   Extend if needed: ARIMAX(exogenous), GARCH(volatility)")
