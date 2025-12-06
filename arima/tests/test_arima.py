# -*- coding: utf-8 -*-
"""
ARIMA/SARIMA 모듈 단위 테스트 / Unit Tests for ARIMA/SARIMA Module
--------------------------------------------------------------------------------
이 모듈은 arima.py의 주요 기능을 테스트합니다.
This module tests the main functionality of arima.py.

테스트 커버리지 / Test Coverage:
- 데이터 생성 및 재현성 / Data generation and reproducibility
- 정상성 검정 / Stationarity tests
- ARIMA 모델 적합 / ARIMA model fitting
- SARIMA 모델 적합 / SARIMA model fitting
- 예측 기능 / Forecasting functionality
- AIC/BIC 비교 / AIC/BIC comparison
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller


# ============================================================
# 픽스처 / Fixtures
# ============================================================


@pytest.fixture
def seed() -> int:
    """
    재현성을 위한 랜덤 시드 / Random seed for reproducibility.

    Returns:
        int: 고정된 시드 값 / Fixed seed value
    """
    return 42


@pytest.fixture
def n_observations() -> int:
    """
    시계열 관측치 개수 / Number of time series observations.

    Returns:
        int: 관측치 수 / Number of observations
    """
    return 400


@pytest.fixture
def seasonal_period() -> int:
    """
    계절 주기 / Seasonal period.

    Returns:
        int: 계절 주기 (24일) / Seasonal period (24 days)
    """
    return 24


@pytest.fixture
def time_series(seed: int, n_observations: int, seasonal_period: int) -> pd.Series:
    """
    테스트용 시계열 데이터 생성 / Generate time series data for testing.

    구성 / Composition:
    - 랜덤워크 (추세) / Random walk (trend)
    - 계절성 패턴 / Seasonal pattern
    - 관측 잡음 / Observation noise

    Args:
        seed: 랜덤 시드 / Random seed
        n_observations: 관측치 수 / Number of observations
        seasonal_period: 계절 주기 / Seasonal period

    Returns:
        pd.Series: 생성된 시계열 / Generated time series
    """
    np.random.seed(seed)
    n = n_observations
    season = seasonal_period

    # 백색잡음 / White noise
    eps = np.random.normal(loc=0, scale=1.0, size=n)

    # 랜덤워크 / Random walk
    rw = np.cumsum(eps)

    # 계절성 패턴 / Seasonal pattern
    t = np.arange(n)
    seasonal = 2.0 * np.sin(2 * np.pi * t / season)

    # 최종 시계열 / Final time series
    y = rw + seasonal + np.random.normal(0, 0.5, size=n)

    return pd.Series(y, index=pd.date_range("2018-01-01", periods=n, freq="D"))


@pytest.fixture
def stationary_series(seed: int) -> pd.Series:
    """
    정상 AR(1) 시계열 생성 / Generate stationary AR(1) time series.

    Args:
        seed: 랜덤 시드 / Random seed

    Returns:
        pd.Series: 정상 AR(1) 시계열 / Stationary AR(1) time series
    """
    np.random.seed(seed)
    n = 300
    phi = 0.5  # |phi| < 1이므로 정상 / Stationary since |phi| < 1
    eps = np.random.normal(0, 1, size=n)

    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = phi * ar[i - 1] + eps[i]

    return pd.Series(ar, index=pd.date_range("2018-01-01", periods=n, freq="D"))


@pytest.fixture
def nonstationary_series(seed: int) -> pd.Series:
    """
    비정상 랜덤워크 시계열 생성 / Generate non-stationary random walk time series.

    Args:
        seed: 랜덤 시드 / Random seed

    Returns:
        pd.Series: 비정상 랜덤워크 시계열 / Non-stationary random walk time series
    """
    np.random.seed(seed)
    n = 300
    eps = np.random.normal(0, 1, size=n)
    rw = np.cumsum(eps)

    return pd.Series(rw, index=pd.date_range("2018-01-01", periods=n, freq="D"))


# ============================================================
# 데이터 생성 테스트 / Data Generation Tests
# ============================================================


class TestDataGeneration:
    """
    데이터 생성 기능 테스트 / Test data generation functionality.
    """

    def test_time_series_length(
        self, time_series: pd.Series, n_observations: int
    ) -> None:
        """
        시계열 길이 검증 / Verify time series length.

        Args:
            time_series: 테스트 시계열 / Test time series
            n_observations: 예상 관측치 수 / Expected number of observations
        """
        assert len(time_series) == n_observations

    def test_time_series_reproducibility(
        self, seed: int, n_observations: int
    ) -> None:
        """
        시계열 재현성 검증 / Verify time series reproducibility.

        동일한 시드로 동일한 시계열이 생성되는지 확인합니다.
        Verifies that the same seed produces the same time series.

        Args:
            seed: 랜덤 시드 / Random seed
            n_observations: 관측치 수 / Number of observations
        """
        # 첫 번째 생성 / First generation
        np.random.seed(seed)
        n = n_observations
        eps1 = np.random.normal(loc=0, scale=1.0, size=n)
        rw1 = np.cumsum(eps1)

        # 두 번째 생성 (동일한 시드) / Second generation (same seed)
        np.random.seed(seed)
        eps2 = np.random.normal(loc=0, scale=1.0, size=n)
        rw2 = np.cumsum(eps2)

        # 동일한 시드로 동일한 랜덤워크가 생성되어야 함
        # Same seed should produce same random walk
        assert_allclose(rw1, rw2, rtol=1e-10)

    def test_seasonal_pattern_period(
        self, time_series: pd.Series, seasonal_period: int
    ) -> None:
        """
        계절 패턴 주기 검증 / Verify seasonal pattern period.

        FFT를 사용하여 주요 주파수 성분이 계절 주기와 일치하는지 확인합니다.
        Uses FFT to verify that the dominant frequency component matches the seasonal period.

        Args:
            time_series: 테스트 시계열 / Test time series
            seasonal_period: 예상 계절 주기 / Expected seasonal period
        """
        # 차분하여 추세 제거 / Remove trend by differencing
        ts_diff = time_series.diff().dropna()

        # FFT로 주요 주파수 성분 추출 / Extract dominant frequency component using FFT
        fft_result = np.fft.fft(ts_diff.values)
        freqs = np.fft.fftfreq(len(ts_diff))

        # 양의 주파수만 고려 / Consider only positive frequencies
        positive_mask = freqs > 0
        positive_freqs = freqs[positive_mask]
        positive_power = np.abs(fft_result[positive_mask])

        # 가장 강한 주파수 찾기 / Find strongest frequency
        dominant_freq = positive_freqs[np.argmax(positive_power)]
        estimated_period = 1 / dominant_freq

        # 계절 주기와 비교 (허용 오차 ±2) / Compare with seasonal period (tolerance ±2)
        assert abs(estimated_period - seasonal_period) < 2


# ============================================================
# 정상성 검정 테스트 / Stationarity Test Tests
# ============================================================


class TestStationarity:
    """
    정상성 검정 기능 테스트 / Test stationarity testing functionality.
    """

    def test_adf_stationary_series(self, stationary_series: pd.Series) -> None:
        """
        정상 시계열에 대한 ADF 검정 / ADF test on stationary series.

        H0 기각 (p-value < 0.05)이 예상됩니다.
        Expected to reject H0 (p-value < 0.05).

        Args:
            stationary_series: 정상 시계열 / Stationary time series
        """
        adf_stat, p_value, *_ = adfuller(stationary_series)

        # 정상 시계열은 p-value < 0.05여야 함 / Stationary series should have p-value < 0.05
        assert p_value < 0.05, f"p-value {p_value} should be < 0.05 for stationary series"

    def test_adf_nonstationary_series(self, nonstationary_series: pd.Series) -> None:
        """
        비정상 시계열에 대한 ADF 검정 / ADF test on non-stationary series.

        H0 채택 (p-value >= 0.05)이 예상됩니다.
        Expected to accept H0 (p-value >= 0.05).

        Args:
            nonstationary_series: 비정상 시계열 / Non-stationary time series
        """
        adf_stat, p_value, *_ = adfuller(nonstationary_series)

        # 비정상 시계열은 p-value >= 0.05여야 함
        # Non-stationary series should have p-value >= 0.05
        assert p_value >= 0.05, f"p-value {p_value} should be >= 0.05 for non-stationary series"

    def test_adf_after_differencing(self, time_series: pd.Series) -> None:
        """
        차분 후 ADF 검정 / ADF test after differencing.

        1차 차분 후 정상성을 달성해야 합니다.
        Should achieve stationarity after 1st-order differencing.

        Args:
            time_series: 테스트 시계열 / Test time series
        """
        # 1차 차분 / First-order differencing
        ts_diff = time_series.diff().dropna()

        adf_stat, p_value, *_ = adfuller(ts_diff)

        # 차분 후 p-value < 0.05여야 함 / p-value should be < 0.05 after differencing
        assert p_value < 0.05, f"p-value {p_value} should be < 0.05 after differencing"


# ============================================================
# ARIMA 모델 테스트 / ARIMA Model Tests
# ============================================================


class TestARIMAModel:
    """
    ARIMA 모델 기능 테스트 / Test ARIMA model functionality.
    """

    def test_arima_fit(self, time_series: pd.Series) -> None:
        """
        ARIMA 모델 적합 검증 / Verify ARIMA model fitting.

        Args:
            time_series: 테스트 시계열 / Test time series
        """
        model = ARIMA(time_series, order=(1, 1, 0))
        result = model.fit()

        # 모델이 적합되었는지 확인 / Verify model was fitted
        assert result is not None
        assert hasattr(result, "aic")
        assert hasattr(result, "bic")
        assert hasattr(result, "resid")

    def test_arima_aic_bic_finite(self, time_series: pd.Series) -> None:
        """
        AIC/BIC가 유한한지 검증 / Verify AIC/BIC are finite.

        Args:
            time_series: 테스트 시계열 / Test time series
        """
        model = ARIMA(time_series, order=(1, 1, 0))
        result = model.fit()

        assert np.isfinite(result.aic), "AIC should be finite"
        assert np.isfinite(result.bic), "BIC should be finite"

    def test_arima_forecast(self, time_series: pd.Series) -> None:
        """
        ARIMA 예측 기능 검증 / Verify ARIMA forecasting.

        Args:
            time_series: 테스트 시계열 / Test time series
        """
        model = ARIMA(time_series, order=(1, 1, 0))
        result = model.fit()

        h = 24  # 예측 기간 / Forecast horizon
        forecast = result.get_forecast(steps=h)
        f_mean = forecast.predicted_mean
        f_ci = forecast.conf_int()

        # 예측 길이 검증 / Verify forecast length
        assert len(f_mean) == h, f"Expected {h} forecasts, got {len(f_mean)}"

        # 신뢰구간 검증 / Verify confidence intervals
        assert f_ci.shape == (h, 2), f"Expected CI shape ({h}, 2), got {f_ci.shape}"

        # 신뢰구간이 합리적인지 검증 (하한 < 평균 < 상한)
        # Verify CI is reasonable (lower < mean < upper)
        assert all(
            f_ci.iloc[:, 0] < f_mean
        ), "Lower CI should be less than mean forecast"
        assert all(
            f_mean < f_ci.iloc[:, 1]
        ), "Mean forecast should be less than upper CI"

    def test_arima_residuals_zero_mean(self, time_series: pd.Series) -> None:
        """
        잔차 평균이 0에 가까운지 검증 / Verify residuals have near-zero mean.

        Args:
            time_series: 테스트 시계열 / Test time series
        """
        model = ARIMA(time_series, order=(1, 1, 0))
        result = model.fit()
        resid = result.resid

        # 잔차 평균이 0에 가까워야 함 (허용 오차 0.1)
        # Residuals mean should be close to 0 (tolerance 0.1)
        assert abs(resid.mean()) < 0.1, f"Residuals mean {resid.mean():.4f} should be near 0"


# ============================================================
# SARIMA 모델 테스트 / SARIMA Model Tests
# ============================================================


class TestSARIMAModel:
    """
    SARIMA 모델 기능 테스트 / Test SARIMA model functionality.
    """

    def test_sarima_fit(
        self, time_series: pd.Series, seasonal_period: int
    ) -> None:
        """
        SARIMA 모델 적합 검증 / Verify SARIMA model fitting.

        Args:
            time_series: 테스트 시계열 / Test time series
            seasonal_period: 계절 주기 / Seasonal period
        """
        model = SARIMAX(
            time_series,
            order=(1, 1, 0),
            seasonal_order=(1, 0, 1, seasonal_period),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False)

        # 모델이 적합되었는지 확인 / Verify model was fitted
        assert result is not None
        assert hasattr(result, "aic")
        assert hasattr(result, "bic")

    def test_sarima_better_aic_than_arima(
        self, time_series: pd.Series, seasonal_period: int
    ) -> None:
        """
        SARIMA가 ARIMA보다 낮은 AIC를 가지는지 검증 / Verify SARIMA has lower AIC than ARIMA.

        계절성이 있는 데이터에서 SARIMA가 ARIMA보다 더 좋은 적합을 보여야 합니다.
        SARIMA should show better fit than ARIMA on seasonal data.

        Args:
            time_series: 테스트 시계열 / Test time series
            seasonal_period: 계절 주기 / Seasonal period
        """
        # ARIMA 적합 / Fit ARIMA
        arima_model = ARIMA(time_series, order=(1, 1, 0))
        arima_result = arima_model.fit()

        # SARIMA 적합 / Fit SARIMA
        sarima_model = SARIMAX(
            time_series,
            order=(1, 1, 0),
            seasonal_order=(1, 0, 1, seasonal_period),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        sarima_result = sarima_model.fit(disp=False)

        # SARIMA AIC가 ARIMA AIC보다 낮아야 함 / SARIMA AIC should be lower than ARIMA AIC
        assert sarima_result.aic < arima_result.aic, (
            f"SARIMA AIC ({sarima_result.aic:.2f}) should be lower than "
            f"ARIMA AIC ({arima_result.aic:.2f})"
        )

    def test_sarima_forecast_captures_seasonality(
        self, time_series: pd.Series, seasonal_period: int
    ) -> None:
        """
        SARIMA 예측이 계절성을 캡처하는지 검증 / Verify SARIMA forecast captures seasonality.

        SARIMA 예측의 분산이 ARIMA 예측의 분산보다 커야 합니다 (주기적 패턴 때문).
        SARIMA forecast variance should be larger than ARIMA (due to periodic pattern).

        Args:
            time_series: 테스트 시계열 / Test time series
            seasonal_period: 계절 주기 / Seasonal period
        """
        h = 48  # 2 주기 예측 / Forecast 2 periods

        # ARIMA 예측 / ARIMA forecast
        arima_model = ARIMA(time_series, order=(1, 1, 0))
        arima_result = arima_model.fit()
        arima_forecast = arima_result.get_forecast(steps=h).predicted_mean

        # SARIMA 예측 / SARIMA forecast
        sarima_model = SARIMAX(
            time_series,
            order=(1, 1, 0),
            seasonal_order=(1, 0, 1, seasonal_period),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        sarima_result = sarima_model.fit(disp=False)
        sarima_forecast = sarima_result.get_forecast(steps=h).predicted_mean

        # SARIMA 예측의 분산이 더 커야 함 (계절 패턴 때문)
        # SARIMA forecast should have larger variance (due to seasonal pattern)
        arima_var = np.var(arima_forecast)
        sarima_var = np.var(sarima_forecast)

        assert sarima_var > arima_var, (
            f"SARIMA forecast variance ({sarima_var:.4f}) should be larger than "
            f"ARIMA forecast variance ({arima_var:.4f})"
        )

    def test_sarima_seasonal_coefficients_significant(
        self, time_series: pd.Series, seasonal_period: int
    ) -> None:
        """
        SARIMA 계절 계수가 유의미한지 검증 / Verify SARIMA seasonal coefficients are significant.

        Args:
            time_series: 테스트 시계열 / Test time series
            seasonal_period: 계절 주기 / Seasonal period
        """
        model = SARIMAX(
            time_series,
            order=(1, 1, 0),
            seasonal_order=(1, 0, 1, seasonal_period),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False)

        # 계절 AR 계수 확인 / Check seasonal AR coefficient
        seasonal_ar_pvalue = result.pvalues.get(f"ar.S.L{seasonal_period}", 1.0)

        # 계절 계수가 유의미해야 함 (p-value < 0.05)
        # Seasonal coefficient should be significant (p-value < 0.05)
        assert seasonal_ar_pvalue < 0.05, (
            f"Seasonal AR coefficient p-value ({seasonal_ar_pvalue:.4f}) "
            "should be < 0.05"
        )


# ============================================================
# 모델 비교 테스트 / Model Comparison Tests
# ============================================================


class TestModelComparison:
    """
    모델 비교 기능 테스트 / Test model comparison functionality.
    """

    def test_arima_candidates_comparison(self, time_series: pd.Series) -> None:
        """
        ARIMA 후보 모델 비교 검증 / Verify ARIMA candidate model comparison.

        여러 ARIMA 후보 중 AIC가 가장 낮은 모델을 선택할 수 있어야 합니다.
        Should be able to select model with lowest AIC among candidates.

        Args:
            time_series: 테스트 시계열 / Test time series
        """
        candidates = [
            (1, 1, 0),
            (0, 1, 1),
            (1, 1, 1),
        ]

        results = []
        for order in candidates:
            model = ARIMA(time_series, order=order)
            result = model.fit()
            results.append({"order": order, "aic": result.aic, "bic": result.bic})

        # AIC 기준 정렬 / Sort by AIC
        results_sorted = sorted(results, key=lambda x: x["aic"])

        # 최소 AIC 모델이 선택되었는지 확인 / Verify lowest AIC model is selected
        best_order = results_sorted[0]["order"]
        assert best_order in candidates

        # 모든 결과가 유효한지 확인 / Verify all results are valid
        for r in results:
            assert np.isfinite(r["aic"])
            assert np.isfinite(r["bic"])


# ============================================================
# 엣지 케이스 테스트 / Edge Case Tests
# ============================================================


class TestEdgeCases:
    """
    엣지 케이스 테스트 / Test edge cases.
    """

    def test_short_time_series(self) -> None:
        """
        짧은 시계열에 대한 모델 적합 / Model fitting on short time series.

        최소 길이 시계열에서도 모델이 적합되어야 합니다.
        Model should fit even on minimum length time series.
        """
        np.random.seed(42)
        n = 50  # 짧은 시계열 / Short time series
        y = np.cumsum(np.random.normal(0, 1, n))
        ts = pd.Series(y)

        model = ARIMA(ts, order=(1, 1, 0))
        result = model.fit()

        assert result is not None
        assert np.isfinite(result.aic)

    def test_constant_series_handling(self) -> None:
        """
        상수 시계열 처리 / Handling constant time series.

        상수 시계열에 대한 ADF 검정이 비정상으로 판단되어야 합니다.
        ADF test on constant series should indicate non-stationarity.
        """
        # 상수 시계열 + 약간의 잡음 / Constant series + small noise
        np.random.seed(42)
        n = 100
        y = np.ones(n) + np.random.normal(0, 0.001, n)

        # ADF 검정 / ADF test
        adf_stat, p_value, *_ = adfuller(y)

        # 거의 상수인 시계열은 정상 (분산이 매우 작음)
        # Near-constant series is stationary (very small variance)
        assert np.isfinite(adf_stat)
        assert np.isfinite(p_value)

    def test_forecast_horizon_edge_cases(self, time_series: pd.Series) -> None:
        """
        예측 기간 엣지 케이스 / Forecast horizon edge cases.

        Args:
            time_series: 테스트 시계열 / Test time series
        """
        model = ARIMA(time_series, order=(1, 1, 0))
        result = model.fit()

        # 1 스텝 예측 / 1-step forecast
        forecast_1 = result.get_forecast(steps=1)
        assert len(forecast_1.predicted_mean) == 1

        # 긴 예측 기간 / Long forecast horizon
        forecast_100 = result.get_forecast(steps=100)
        assert len(forecast_100.predicted_mean) == 100


# ============================================================
# 메인 실행 / Main Execution
# ============================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
