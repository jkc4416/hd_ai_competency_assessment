"""
Data Preprocessing Utilities for Machine Learning.
머신러닝을 위한 데이터 전처리 유틸리티 모음.

This module provides a comprehensive collection of data preprocessing functions
commonly used in machine learning pipelines, including:
이 모듈은 머신러닝 파이프라인에서 자주 사용되는 데이터 전처리 함수들을 제공합니다:

1. Missing Value Handling / 결측치 처리
2. Outlier Detection & Removal / 이상치 탐지 및 제거
3. Feature Scaling / 피처 스케일링
4. Encoding (Label, One-Hot, Target) / 인코딩
5. Feature Engineering / 피처 엔지니어링
6. Data Splitting / 데이터 분할
7. Class Imbalance Handling / 클래스 불균형 처리
8. Time Series Preprocessing / 시계열 전처리
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

# ============================================================
# Logger Configuration / 로거 설정
# ============================================================
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


# ============================================================
# 1. Missing Value Handling / 결측치 처리
# ============================================================
def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze missing values in a DataFrame.
    데이터프레임의 결측치를 분석합니다.

    Args:
        df: Input DataFrame to analyze.
            분석할 입력 데이터프레임.

    Returns:
        DataFrame with missing value statistics including count and percentage.
        결측치 개수와 비율을 포함한 통계 데이터프레임.

    Examples:
        >>> df = pd.DataFrame({"A": [1, None, 3], "B": [4, 5, None]})
        >>> missing_stats = check_missing_values(df)
    """
    # Calculate missing count and percentage
    # 결측치 개수와 비율 계산
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100

    # Create summary DataFrame / 요약 데이터프레임 생성
    missing_df = pd.DataFrame(
        {
            "missing_count": missing_count,
            "missing_percentage": missing_pct.round(2),
            "dtype": df.dtypes,
        }
    )

    # Filter only columns with missing values and sort
    # 결측치가 있는 컬럼만 필터링하고 정렬
    missing_df = missing_df[missing_df["missing_count"] > 0].sort_values(
        "missing_count", ascending=False
    )

    logger.info(
        "Found %d columns with missing values / 결측치가 있는 컬럼 %d개 발견",
        len(missing_df),
        len(missing_df),
    )

    return missing_df


def fill_missing_values(
    df: pd.DataFrame,
    strategy: Literal["mean", "median", "mode", "constant", "knn"] = "mean",
    fill_value: float | str | None = None,
    n_neighbors: int = 5,
) -> pd.DataFrame:
    """
    Fill missing values using various strategies.
    다양한 전략을 사용하여 결측치를 채웁니다.

    Args:
        df: Input DataFrame with missing values.
            결측치가 있는 입력 데이터프레임.
        strategy: Imputation strategy. Options are:
            - "mean": Fill with column mean (numeric only)
            - "median": Fill with column median (numeric only)
            - "mode": Fill with most frequent value
            - "constant": Fill with a constant value
            - "knn": Fill using K-Nearest Neighbors
            대체 전략:
            - "mean": 컬럼 평균으로 채움 (수치형만)
            - "median": 컬럼 중앙값으로 채움 (수치형만)
            - "mode": 최빈값으로 채움
            - "constant": 상수값으로 채움
            - "knn": K-최근접 이웃 사용
        fill_value: Value to use when strategy is "constant".
            strategy가 "constant"일 때 사용할 값.
        n_neighbors: Number of neighbors for KNN imputation.
            KNN 대체 시 사용할 이웃 수.

    Returns:
        DataFrame with missing values filled.
        결측치가 채워진 데이터프레임.

    Examples:
        >>> df = pd.DataFrame({"A": [1, None, 3], "B": [4, 5, None]})
        >>> filled_df = fill_missing_values(df, strategy="mean")
    """
    df_filled = df.copy()

    # Separate numeric and categorical columns
    # 수치형과 범주형 컬럼 분리
    numeric_cols = df_filled.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_filled.select_dtypes(exclude=[np.number]).columns.tolist()

    if strategy == "knn":
        # KNN Imputer works only on numeric columns
        # KNN Imputer는 수치형 컬럼에서만 동작
        if numeric_cols:
            knn_imputer = KNNImputer(n_neighbors=n_neighbors)
            df_filled[numeric_cols] = knn_imputer.fit_transform(df_filled[numeric_cols])
            logger.info("Applied KNN imputation with %d neighbors", n_neighbors)

        # Fill categorical with mode / 범주형은 최빈값으로 채움
        for col in categorical_cols:
            if df_filled[col].isnull().any():
                mode_value = df_filled[col].mode()
                if len(mode_value) > 0:
                    df_filled[col] = df_filled[col].fillna(mode_value[0])

    elif strategy == "constant":
        # Fill with constant value / 상수값으로 채움
        df_filled = df_filled.fillna(fill_value)
        logger.info("Filled missing values with constant: %s", fill_value)

    elif strategy == "mode":
        # Fill with most frequent value / 최빈값으로 채움
        for col in df_filled.columns:
            if df_filled[col].isnull().any():
                mode_value = df_filled[col].mode()
                if len(mode_value) > 0:
                    df_filled[col] = df_filled[col].fillna(mode_value[0])
        logger.info("Filled missing values with mode")

    else:
        # Use SimpleImputer for mean/median
        # mean/median은 SimpleImputer 사용
        if numeric_cols:
            imputer = SimpleImputer(strategy=strategy)
            df_filled[numeric_cols] = imputer.fit_transform(df_filled[numeric_cols])
            logger.info("Applied %s imputation on numeric columns", strategy)

        # Fill categorical with mode / 범주형은 최빈값으로 채움
        for col in categorical_cols:
            if df_filled[col].isnull().any():
                mode_value = df_filled[col].mode()
                if len(mode_value) > 0:
                    df_filled[col] = df_filled[col].fillna(mode_value[0])

    return df_filled


# ============================================================
# 2. Outlier Detection & Removal / 이상치 탐지 및 제거
# ============================================================
def detect_outliers_iqr(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    threshold: float = 1.5,
) -> dict[str, pd.Index]:
    """
    Detect outliers using the IQR (Interquartile Range) method.
    IQR (사분위 범위) 방식으로 이상치를 탐지합니다.

    The IQR method identifies outliers as values below Q1 - threshold*IQR
    or above Q3 + threshold*IQR, where IQR = Q3 - Q1.
    IQR 방식은 Q1 - threshold*IQR 미만이거나 Q3 + threshold*IQR 초과인 값을
    이상치로 식별합니다. 여기서 IQR = Q3 - Q1입니다.

    Args:
        df: Input DataFrame.
            입력 데이터프레임.
        columns: List of columns to check. If None, checks all numeric columns.
            검사할 컬럼 리스트. None이면 모든 수치형 컬럼 검사.
        threshold: IQR multiplier (default 1.5). Use 3.0 for extreme outliers.
            IQR 배수 (기본값 1.5). 극단적 이상치는 3.0 사용.

    Returns:
        Dictionary mapping column names to outlier indices.
        컬럼명을 이상치 인덱스에 매핑하는 딕셔너리.

    Examples:
        >>> df = pd.DataFrame({"A": [1, 2, 3, 100], "B": [4, 5, 6, 7]})
        >>> outliers = detect_outliers_iqr(df, ["A"])
    """
    # Default to all numeric columns / 기본값: 모든 수치형 컬럼
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    outlier_dict: dict[str, pd.Index] = {}

    for col in columns:
        if col not in df.columns:
            logger.warning("Column '%s' not found in DataFrame", col)
            continue

        if not pd.api.types.is_numeric_dtype(df[col]):
            logger.warning("Column '%s' is not numeric, skipping", col)
            continue

        # Calculate IQR / IQR 계산
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        # Define boundaries / 경계값 정의
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        # Find outlier indices / 이상치 인덱스 찾기
        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_indices = df[outlier_mask].index

        if len(outlier_indices) > 0:
            outlier_dict[col] = outlier_indices
            logger.info(
                "Column '%s': %d outliers (lower=%.2f, upper=%.2f)",
                col,
                len(outlier_indices),
                lower_bound,
                upper_bound,
            )

    return outlier_dict


def detect_outliers_zscore(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    threshold: float = 3.0,
) -> dict[str, pd.Index]:
    """
    Detect outliers using the Z-score method.
    Z-점수 방식으로 이상치를 탐지합니다.

    Values with |z-score| > threshold are considered outliers.
    |z-점수| > threshold인 값을 이상치로 간주합니다.

    Args:
        df: Input DataFrame.
            입력 데이터프레임.
        columns: List of columns to check. If None, checks all numeric columns.
            검사할 컬럼 리스트. None이면 모든 수치형 컬럼 검사.
        threshold: Z-score threshold (default 3.0).
            Z-점수 임계값 (기본값 3.0).

    Returns:
        Dictionary mapping column names to outlier indices.
        컬럼명을 이상치 인덱스에 매핑하는 딕셔너리.

    Examples:
        >>> df = pd.DataFrame({"A": [1, 2, 3, 100], "B": [4, 5, 6, 7]})
        >>> outliers = detect_outliers_zscore(df, ["A"])
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    outlier_dict: dict[str, pd.Index] = {}

    for col in columns:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue

        # Calculate Z-score / Z-점수 계산
        mean_val = df[col].mean()
        std_val = df[col].std()

        if std_val == 0:
            continue

        z_scores = np.abs((df[col] - mean_val) / std_val)

        # Find outliers / 이상치 찾기
        outlier_mask = z_scores > threshold
        outlier_indices = df[outlier_mask].index

        if len(outlier_indices) > 0:
            outlier_dict[col] = outlier_indices
            logger.info(
                "Column '%s': %d outliers (z-score > %.1f)", col, len(outlier_indices), threshold
            )

    return outlier_dict


def remove_outliers(
    df: pd.DataFrame,
    outlier_dict: dict[str, pd.Index],
    method: Literal["drop", "clip", "nan"] = "drop",
) -> pd.DataFrame:
    """
    Remove or handle outliers based on the specified method.
    지정된 방법에 따라 이상치를 제거하거나 처리합니다.

    Args:
        df: Input DataFrame.
            입력 데이터프레임.
        outlier_dict: Dictionary of column names to outlier indices.
            컬럼명을 이상치 인덱스에 매핑하는 딕셔너리.
        method: How to handle outliers:
            - "drop": Remove rows containing outliers
            - "clip": Clip values to IQR boundaries
            - "nan": Replace outliers with NaN
            이상치 처리 방법:
            - "drop": 이상치가 포함된 행 제거
            - "clip": IQR 경계값으로 클리핑
            - "nan": 이상치를 NaN으로 대체

    Returns:
        DataFrame with outliers handled.
        이상치가 처리된 데이터프레임.

    Examples:
        >>> outliers = {"A": pd.Index([3])}
        >>> cleaned_df = remove_outliers(df, outliers, method="drop")
    """
    df_clean = df.copy()

    if method == "drop":
        # Collect all outlier indices / 모든 이상치 인덱스 수집
        all_outlier_indices: set[int] = set()
        for indices in outlier_dict.values():
            all_outlier_indices.update(indices)

        # Drop rows / 행 제거
        df_clean = df_clean.drop(index=list(all_outlier_indices))
        logger.info("Dropped %d rows containing outliers", len(all_outlier_indices))

    elif method == "clip":
        # Clip values to boundaries / 경계값으로 클리핑
        for col in outlier_dict:
            q1 = df_clean[col].quantile(0.25)
            q3 = df_clean[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
        logger.info("Clipped outliers in %d columns", len(outlier_dict))

    elif method == "nan":
        # Replace with NaN / NaN으로 대체
        for col, indices in outlier_dict.items():
            df_clean.loc[indices, col] = np.nan
        logger.info("Replaced outliers with NaN in %d columns", len(outlier_dict))

    return df_clean


# ============================================================
# 3. Feature Scaling / 피처 스케일링
# ============================================================
def scale_features(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: Literal["standard", "minmax", "robust"] = "standard",
    feature_range: tuple[float, float] = (0, 1),
) -> tuple[pd.DataFrame, StandardScaler | MinMaxScaler | RobustScaler]:
    """
    Scale numeric features using various methods.
    다양한 방법으로 수치형 피처를 스케일링합니다.

    Args:
        df: Input DataFrame.
            입력 데이터프레임.
        columns: Columns to scale. If None, scales all numeric columns.
            스케일링할 컬럼. None이면 모든 수치형 컬럼 스케일링.
        method: Scaling method:
            - "standard": Z-score normalization (mean=0, std=1)
            - "minmax": Min-Max scaling to [0, 1] range
            - "robust": Robust scaling using median and IQR
            스케일링 방법:
            - "standard": Z-점수 정규화 (평균=0, 표준편차=1)
            - "minmax": [0, 1] 범위로 Min-Max 스케일링
            - "robust": 중앙값과 IQR을 사용한 로버스트 스케일링
        feature_range: Range for MinMax scaling (default (0, 1)).
            MinMax 스케일링 범위 (기본값 (0, 1)).

    Returns:
        Tuple of (scaled DataFrame, fitted scaler object).
        (스케일링된 데이터프레임, 학습된 스케일러 객체) 튜플.

    Examples:
        >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        >>> scaled_df, scaler = scale_features(df, method="standard")
    """
    df_scaled = df.copy()

    if columns is None:
        columns = df_scaled.select_dtypes(include=[np.number]).columns.tolist()

    # Select scaler / 스케일러 선택
    scaler: StandardScaler | MinMaxScaler | RobustScaler
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler(feature_range=feature_range)
    else:  # robust
        scaler = RobustScaler()

    # Fit and transform / 학습 및 변환
    df_scaled[columns] = scaler.fit_transform(df_scaled[columns])

    logger.info("Applied %s scaling to %d columns", method, len(columns))

    return df_scaled, scaler


# ============================================================
# 4. Encoding / 인코딩
# ============================================================
def label_encode(
    df: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """
    Apply label encoding to categorical columns.
    범주형 컬럼에 레이블 인코딩을 적용합니다.

    Converts categorical values to integer codes (0, 1, 2, ...).
    범주형 값을 정수 코드 (0, 1, 2, ...)로 변환합니다.

    Args:
        df: Input DataFrame.
            입력 데이터프레임.
        columns: List of columns to encode.
            인코딩할 컬럼 리스트.

    Returns:
        Tuple of (encoded DataFrame, dictionary of fitted encoders).
        (인코딩된 데이터프레임, 학습된 인코더 딕셔너리) 튜플.

    Examples:
        >>> df = pd.DataFrame({"color": ["red", "blue", "red"]})
        >>> encoded_df, encoders = label_encode(df, ["color"])
    """
    df_encoded = df.copy()
    encoders: dict[str, LabelEncoder] = {}

    for col in columns:
        if col not in df_encoded.columns:
            logger.warning("Column '%s' not found, skipping", col)
            continue

        encoder = LabelEncoder()
        df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))
        encoders[col] = encoder

        logger.info("Label encoded '%s': %d classes", col, len(encoder.classes_))

    return df_encoded, encoders


def onehot_encode(
    df: pd.DataFrame,
    columns: list[str],
    drop_first: bool = False,
    sparse_output: bool = False,
) -> pd.DataFrame:
    """
    Apply one-hot encoding to categorical columns.
    범주형 컬럼에 원-핫 인코딩을 적용합니다.

    Creates binary columns for each category.
    각 범주에 대해 이진 컬럼을 생성합니다.

    Args:
        df: Input DataFrame.
            입력 데이터프레임.
        columns: List of columns to encode.
            인코딩할 컬럼 리스트.
        drop_first: Whether to drop the first category (avoid multicollinearity).
            첫 번째 범주를 제거할지 여부 (다중공선성 방지).
        sparse_output: Whether to return sparse matrix.
            희소 행렬 반환 여부.

    Returns:
        DataFrame with one-hot encoded columns.
        원-핫 인코딩된 컬럼이 포함된 데이터프레임.

    Examples:
        >>> df = pd.DataFrame({"color": ["red", "blue", "red"]})
        >>> encoded_df = onehot_encode(df, ["color"])
    """
    # Use pandas get_dummies for simplicity
    # 간편함을 위해 pandas get_dummies 사용
    df_encoded = pd.get_dummies(
        df,
        columns=columns,
        drop_first=drop_first,
        dtype=int,
    )

    new_cols = df_encoded.shape[1] - df.shape[1] + len(columns)
    logger.info("One-hot encoded %d columns, created %d new columns", len(columns), new_cols)

    return df_encoded


def target_encode(
    df: pd.DataFrame,
    column: str,
    target: str,
    smoothing: float = 1.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Apply target encoding (mean encoding) to a categorical column.
    범주형 컬럼에 타겟 인코딩 (평균 인코딩)을 적용합니다.

    Replaces categories with the mean of the target variable for that category.
    범주를 해당 범주의 타겟 변수 평균으로 대체합니다.

    Args:
        df: Input DataFrame.
            입력 데이터프레임.
        column: Column to encode.
            인코딩할 컬럼.
        target: Target column name.
            타겟 컬럼명.
        smoothing: Smoothing factor for regularization (higher = more global mean).
            정규화를 위한 스무딩 계수 (높을수록 전체 평균에 가까움).

    Returns:
        Tuple of (encoded DataFrame, encoding mapping dictionary).
        (인코딩된 데이터프레임, 인코딩 매핑 딕셔너리) 튜플.

    Examples:
        >>> df = pd.DataFrame({"cat": ["A", "B", "A"], "target": [1, 0, 1]})
        >>> encoded_df, mapping = target_encode(df, "cat", "target")
    """
    df_encoded = df.copy()

    # Calculate global mean / 전체 평균 계산
    global_mean = df[target].mean()

    # Calculate category statistics / 범주별 통계 계산
    category_stats = df.groupby(column)[target].agg(["mean", "count"])

    # Apply smoothing (regularization)
    # 스무딩 적용 (정규화)
    # smoothed_mean = (count * category_mean + smoothing * global_mean) / (count + smoothing)
    smoothed_means = (
        category_stats["count"] * category_stats["mean"] + smoothing * global_mean
    ) / (category_stats["count"] + smoothing)

    # Create mapping / 매핑 생성
    encoding_map = smoothed_means.to_dict()

    # Apply encoding / 인코딩 적용
    df_encoded[f"{column}_encoded"] = df_encoded[column].map(encoding_map)

    logger.info("Target encoded '%s' with %d categories", column, len(encoding_map))

    return df_encoded, encoding_map


# ============================================================
# 5. Feature Engineering / 피처 엔지니어링
# ============================================================
def create_polynomial_features(
    df: pd.DataFrame,
    columns: list[str],
    degree: int = 2,
    include_interaction: bool = True,
) -> pd.DataFrame:
    """
    Create polynomial and interaction features.
    다항식 및 상호작용 피처를 생성합니다.

    Args:
        df: Input DataFrame.
            입력 데이터프레임.
        columns: Columns to create polynomial features from.
            다항식 피처를 생성할 컬럼.
        degree: Maximum polynomial degree (default 2).
            최대 다항식 차수 (기본값 2).
        include_interaction: Whether to include interaction terms.
            상호작용 항 포함 여부.

    Returns:
        DataFrame with added polynomial features.
        다항식 피처가 추가된 데이터프레임.

    Examples:
        >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        >>> poly_df = create_polynomial_features(df, ["A", "B"], degree=2)
    """
    df_poly = df.copy()

    # Create squared terms / 제곱 항 생성
    for col in columns:
        for d in range(2, degree + 1):
            df_poly[f"{col}^{d}"] = df_poly[col] ** d

    # Create interaction terms / 상호작용 항 생성
    if include_interaction and len(columns) > 1:
        for i, col1 in enumerate(columns):
            for col2 in columns[i + 1 :]:
                df_poly[f"{col1}*{col2}"] = df_poly[col1] * df_poly[col2]

    new_features = df_poly.shape[1] - df.shape[1]
    logger.info("Created %d polynomial features", new_features)

    return df_poly


def create_datetime_features(
    df: pd.DataFrame,
    datetime_column: str,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """
    Extract datetime features from a datetime column.
    날짜/시간 컬럼에서 날짜/시간 피처를 추출합니다.

    Args:
        df: Input DataFrame.
            입력 데이터프레임.
        datetime_column: Name of the datetime column.
            날짜/시간 컬럼명.
        features: List of features to extract. Options:
            "year", "month", "day", "dayofweek", "hour", "minute",
            "quarter", "is_weekend", "is_month_start", "is_month_end"
            If None, extracts all features.
            추출할 피처 리스트. None이면 모든 피처 추출.

    Returns:
        DataFrame with datetime features added.
        날짜/시간 피처가 추가된 데이터프레임.

    Examples:
        >>> df = pd.DataFrame({"date": pd.to_datetime(["2024-01-15", "2024-02-20"])})
        >>> dt_df = create_datetime_features(df, "date")
    """
    df_dt = df.copy()

    # Convert to datetime if not already / 날짜/시간 형식으로 변환
    if not pd.api.types.is_datetime64_any_dtype(df_dt[datetime_column]):
        df_dt[datetime_column] = pd.to_datetime(df_dt[datetime_column])

    dt_col = df_dt[datetime_column]
    prefix = datetime_column

    # Default features / 기본 피처
    if features is None:
        features = [
            "year",
            "month",
            "day",
            "dayofweek",
            "hour",
            "quarter",
            "is_weekend",
        ]

    # Extract features / 피처 추출
    feature_map = {
        "year": dt_col.dt.year,
        "month": dt_col.dt.month,
        "day": dt_col.dt.day,
        "dayofweek": dt_col.dt.dayofweek,
        "hour": dt_col.dt.hour,
        "minute": dt_col.dt.minute,
        "quarter": dt_col.dt.quarter,
        "is_weekend": (dt_col.dt.dayofweek >= 5).astype(int),
        "is_month_start": dt_col.dt.is_month_start.astype(int),
        "is_month_end": dt_col.dt.is_month_end.astype(int),
    }

    for feat in features:
        if feat in feature_map:
            df_dt[f"{prefix}_{feat}"] = feature_map[feat]

    logger.info("Created %d datetime features from '%s'", len(features), datetime_column)

    return df_dt


def create_lag_features(
    df: pd.DataFrame,
    column: str,
    lags: list[int],
    group_column: str | None = None,
) -> pd.DataFrame:
    """
    Create lag features for time series data.
    시계열 데이터를 위한 래그 피처를 생성합니다.

    Args:
        df: Input DataFrame (must be sorted by time).
            입력 데이터프레임 (시간순 정렬 필요).
        column: Column to create lags from.
            래그를 생성할 컬럼.
        lags: List of lag periods (e.g., [1, 7, 30] for 1-day, 7-day, 30-day lags).
            래그 기간 리스트 (예: [1, 7, 30] - 1일, 7일, 30일 래그).
        group_column: Optional column to group by before creating lags.
            래그 생성 전 그룹화할 선택적 컬럼.

    Returns:
        DataFrame with lag features added.
        래그 피처가 추가된 데이터프레임.

    Examples:
        >>> df = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
        >>> lag_df = create_lag_features(df, "value", [1, 2])
    """
    df_lag = df.copy()

    for lag in lags:
        col_name = f"{column}_lag_{lag}"
        if group_column:
            df_lag[col_name] = df_lag.groupby(group_column)[column].shift(lag)
        else:
            df_lag[col_name] = df_lag[column].shift(lag)

    logger.info("Created %d lag features for '%s'", len(lags), column)

    return df_lag


def create_rolling_features(
    df: pd.DataFrame,
    column: str,
    windows: list[int],
    functions: list[str] | None = None,
    group_column: str | None = None,
) -> pd.DataFrame:
    """
    Create rolling window features for time series data.
    시계열 데이터를 위한 이동 윈도우 피처를 생성합니다.

    Args:
        df: Input DataFrame.
            입력 데이터프레임.
        column: Column to compute rolling statistics from.
            이동 통계를 계산할 컬럼.
        windows: List of window sizes.
            윈도우 크기 리스트.
        functions: List of aggregation functions (default: ["mean", "std", "min", "max"]).
            집계 함수 리스트 (기본값: ["mean", "std", "min", "max"]).
        group_column: Optional column to group by.
            그룹화할 선택적 컬럼.

    Returns:
        DataFrame with rolling features added.
        이동 윈도우 피처가 추가된 데이터프레임.

    Examples:
        >>> df = pd.DataFrame({"value": range(10)})
        >>> rolling_df = create_rolling_features(df, "value", [3, 7])
    """
    df_rolling = df.copy()

    if functions is None:
        functions = ["mean", "std", "min", "max"]

    for window in windows:
        for func in functions:
            col_name = f"{column}_rolling_{window}_{func}"

            if group_column:
                rolling = df_rolling.groupby(group_column)[column].transform(
                    lambda x: getattr(x.rolling(window=window, min_periods=1), func)()
                )
            else:
                rolling = getattr(df_rolling[column].rolling(window=window, min_periods=1), func)()

            df_rolling[col_name] = rolling

    total_features = len(windows) * len(functions)
    logger.info("Created %d rolling features for '%s'", total_features, column)

    return df_rolling


# ============================================================
# 6. Data Splitting / 데이터 분할
# ============================================================
def split_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    val_size: float | None = None,
    stratify: bool = True,
    random_state: int = 42,
) -> (
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
    | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]
):
    """
    Split data into train/test or train/val/test sets.
    데이터를 train/test 또는 train/val/test 세트로 분할합니다.

    Args:
        df: Input DataFrame.
            입력 데이터프레임.
        target_column: Name of the target column.
            타겟 컬럼명.
        test_size: Proportion for test set (default 0.2).
            테스트 세트 비율 (기본값 0.2).
        val_size: Proportion for validation set. If None, no validation set.
            검증 세트 비율. None이면 검증 세트 없음.
        stratify: Whether to use stratified sampling (for classification).
            층화 샘플링 사용 여부 (분류 문제용).
        random_state: Random seed for reproducibility.
            재현성을 위한 랜덤 시드.

    Returns:
        If val_size is None: (X_train, X_test, y_train, y_test)
        If val_size is set: (X_train, X_val, X_test, y_train, y_val, y_test)

    Examples:
        >>> X_train, X_test, y_train, y_test = split_data(df, "target")
        >>> X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, "target", val_size=0.1)
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    stratify_col = y if stratify else None

    if val_size is None:
        # Simple train/test split / 단순 train/test 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=stratify_col, random_state=random_state
        )
        logger.info("Split: Train=%d, Test=%d", len(X_train), len(X_test))
        return X_train, X_test, y_train, y_test

    else:
        # Train/val/test split / Train/val/test 분할
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, stratify=stratify_col, random_state=random_state
        )

        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        stratify_temp = y_temp if stratify else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, stratify=stratify_temp, random_state=random_state
        )

        logger.info("Split: Train=%d, Val=%d, Test=%d", len(X_train), len(X_val), len(X_test))
        return X_train, X_val, X_test, y_train, y_val, y_test


def split_time_series(
    df: pd.DataFrame,
    target_column: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split time series data chronologically (no shuffling).
    시계열 데이터를 시간순으로 분할합니다 (셔플 없음).

    Args:
        df: Input DataFrame (must be sorted by time).
            입력 데이터프레임 (시간순 정렬 필요).
        target_column: Name of the target column.
            타겟 컬럼명.
        train_ratio: Proportion for training set (default 0.7).
            학습 세트 비율 (기본값 0.7).
        val_ratio: Proportion for validation set (default 0.15).
            검증 세트 비율 (기본값 0.15).

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test) tuple.

    Examples:
        >>> X_train, X_val, X_test, y_train, y_val, y_test = split_time_series(df, "target")
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    # Split features and target / 피처와 타겟 분리
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Chronological split / 시간순 분할
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    logger.info(
        "Time series split: Train=%d (%.1f%%), Val=%d (%.1f%%), Test=%d (%.1f%%)",
        len(X_train),
        train_ratio * 100,
        len(X_val),
        val_ratio * 100,
        len(X_test),
        (1 - train_ratio - val_ratio) * 100,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
# 7. Class Imbalance Handling / 클래스 불균형 처리
# ============================================================
def check_class_imbalance(
    y: pd.Series | NDArray[np.int_],
) -> dict[str, float | dict]:
    """
    Analyze class distribution and imbalance.
    클래스 분포와 불균형을 분석합니다.

    Args:
        y: Target variable (array or Series).
            타겟 변수 (배열 또는 시리즈).

    Returns:
        Dictionary with class counts, percentages, and imbalance ratio.
        클래스 개수, 비율, 불균형 비율을 포함한 딕셔너리.

    Examples:
        >>> y = pd.Series([0, 0, 0, 1, 1])
        >>> stats = check_class_imbalance(y)
    """
    if isinstance(y, np.ndarray):
        y = pd.Series(y)

    # Count each class / 각 클래스 개수
    counts = y.value_counts().sort_index()
    total = len(y)

    # Calculate percentages / 비율 계산
    percentages = (counts / total * 100).round(2)

    # Calculate imbalance ratio / 불균형 비율 계산
    imbalance_ratio = float(counts.max() / counts.min()) if len(counts) > 1 else 1.0

    result = {
        "total_samples": total,
        "class_counts": counts.to_dict(),
        "class_percentages": percentages.to_dict(),
        "imbalance_ratio": round(imbalance_ratio, 2),
        "is_imbalanced": imbalance_ratio > 2.0,
    }

    logger.info(
        "Class imbalance ratio: %.2f (imbalanced=%s)",
        imbalance_ratio,
        result["is_imbalanced"],
    )

    return result


def undersample(
    X: pd.DataFrame,
    y: pd.Series,
    strategy: Literal["random", "tomek"] = "random",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Undersample the majority class.
    다수 클래스를 언더샘플링합니다.

    Args:
        X: Feature DataFrame.
            피처 데이터프레임.
        y: Target Series.
            타겟 시리즈.
        strategy: Undersampling strategy.
            언더샘플링 전략.
        random_state: Random seed.
            랜덤 시드.

    Returns:
        Resampled (X, y) tuple.
        리샘플링된 (X, y) 튜플.

    Examples:
        >>> X_resampled, y_resampled = undersample(X, y)
    """
    # Simple random undersampling / 단순 랜덤 언더샘플링
    np.random.seed(random_state)

    # Find minority class size / 소수 클래스 크기 찾기
    min_count = y.value_counts().min()

    # Sample from each class / 각 클래스에서 샘플링
    indices: list[int] = []
    for cls in y.unique():
        cls_indices = y[y == cls].index.tolist()
        if len(cls_indices) > min_count:
            sampled = np.random.choice(cls_indices, size=min_count, replace=False)
            indices.extend(sampled.tolist())
        else:
            indices.extend(cls_indices)

    X_resampled = X.loc[indices].reset_index(drop=True)
    y_resampled = y.loc[indices].reset_index(drop=True)

    logger.info(
        "Undersampled: %d -> %d samples (balanced to %d per class)",
        len(y),
        len(y_resampled),
        min_count,
    )

    return X_resampled, y_resampled


def oversample(
    X: pd.DataFrame,
    y: pd.Series,
    strategy: Literal["random", "smote"] = "random",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Oversample the minority class.
    소수 클래스를 오버샘플링합니다.

    Args:
        X: Feature DataFrame.
            피처 데이터프레임.
        y: Target Series.
            타겟 시리즈.
        strategy: Oversampling strategy ("random" or "smote").
            오버샘플링 전략 ("random" 또는 "smote").
        random_state: Random seed.
            랜덤 시드.

    Returns:
        Resampled (X, y) tuple.
        리샘플링된 (X, y) 튜플.

    Examples:
        >>> X_resampled, y_resampled = oversample(X, y, strategy="smote")
    """
    if strategy == "smote":
        try:
            from imblearn.over_sampling import SMOTE

            smote = SMOTE(random_state=random_state)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            logger.info("Applied SMOTE: %d -> %d samples", len(y), len(y_resampled))
            return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
        except ImportError:
            logger.warning("imblearn not installed, falling back to random oversampling")
            strategy = "random"

    # Random oversampling / 랜덤 오버샘플링
    np.random.seed(random_state)
    max_count = y.value_counts().max()

    X_list = [X]
    y_list = [y]

    for cls in y.unique():
        cls_indices = y[y == cls].index.tolist()
        if len(cls_indices) < max_count:
            n_samples = max_count - len(cls_indices)
            sampled_indices = np.random.choice(cls_indices, size=n_samples, replace=True)
            X_list.append(X.loc[sampled_indices])
            y_list.append(y.loc[sampled_indices])

    X_resampled = pd.concat(X_list, ignore_index=True)
    y_resampled = pd.concat(y_list, ignore_index=True)

    logger.info("Oversampled: %d -> %d samples", len(y), len(y_resampled))

    return X_resampled, y_resampled


# ============================================================
# 8. Convenience Functions / 편의 함수
# ============================================================
def preprocess_pipeline(
    df: pd.DataFrame,
    target_column: str,
    numeric_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
    fill_strategy: Literal["mean", "median", "mode", "constant", "knn"] = "median",
    scale_method: Literal["standard", "minmax", "robust"] = "standard",
    encode_method: Literal["onehot", "label"] = "onehot",
    handle_outliers: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Complete preprocessing pipeline combining multiple steps.
    여러 단계를 결합한 완전한 전처리 파이프라인.

    This function provides an end-to-end preprocessing workflow:
    이 함수는 엔드-투-엔드 전처리 워크플로우를 제공합니다:
    1. Missing value handling / 결측치 처리
    2. Outlier removal / 이상치 제거
    3. Feature scaling / 피처 스케일링
    4. Categorical encoding / 범주형 인코딩
    5. Train/test split / Train/test 분할

    Args:
        df: Input DataFrame.
        target_column: Name of the target column.
        numeric_columns: List of numeric columns (auto-detected if None).
        categorical_columns: List of categorical columns (auto-detected if None).
        fill_strategy: Strategy for filling missing values.
        scale_method: Scaling method ("standard", "minmax", "robust").
        encode_method: Encoding method ("onehot", "label").
        handle_outliers: Whether to remove outliers.
        test_size: Test set proportion.
        random_state: Random seed.

    Returns:
        Dictionary containing processed data and fitted transformers.
        처리된 데이터와 학습된 변환기를 포함한 딕셔너리.

    Examples:
        >>> result = preprocess_pipeline(df, "target")
        >>> X_train = result["X_train"]
    """
    df_processed = df.copy()

    # Auto-detect column types / 컬럼 타입 자동 감지
    if numeric_columns is None:
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_columns:
            numeric_columns.remove(target_column)

    if categorical_columns is None:
        categorical_columns = df_processed.select_dtypes(exclude=[np.number]).columns.tolist()

    logger.info("Starting preprocessing pipeline")
    logger.info("Numeric columns: %s", numeric_columns)
    logger.info("Categorical columns: %s", categorical_columns)

    # 1. Fill missing values / 결측치 채우기
    df_processed = fill_missing_values(df_processed, strategy=fill_strategy)

    # 2. Handle outliers / 이상치 처리
    if handle_outliers and numeric_columns:
        outliers = detect_outliers_iqr(df_processed, numeric_columns)
        df_processed = remove_outliers(df_processed, outliers, method="clip")

    # 3. Scale numeric features / 수치형 피처 스케일링
    scaler = None
    if numeric_columns:
        df_processed, scaler = scale_features(df_processed, numeric_columns, method=scale_method)

    # 4. Encode categorical features / 범주형 피처 인코딩
    encoders = None
    if categorical_columns:
        if encode_method == "onehot":
            df_processed = onehot_encode(df_processed, categorical_columns)
        else:
            df_processed, encoders = label_encode(df_processed, categorical_columns)

    # 5. Split data / 데이터 분할
    # Use train_test_split directly to avoid union type issues
    # 유니온 타입 이슈를 피하기 위해 train_test_split 직접 사용
    X = df_processed.drop(columns=[target_column])
    y = df_processed[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    logger.info("Preprocessing pipeline completed")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "encoders": encoders,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
    }


# ============================================================
# Module Exports / 모듈 내보내기
# ============================================================
__all__ = [
    # Missing values / 결측치
    "check_missing_values",
    "fill_missing_values",
    # Outliers / 이상치
    "detect_outliers_iqr",
    "detect_outliers_zscore",
    "remove_outliers",
    # Scaling / 스케일링
    "scale_features",
    # Encoding / 인코딩
    "label_encode",
    "onehot_encode",
    "target_encode",
    # Feature engineering / 피처 엔지니어링
    "create_polynomial_features",
    "create_datetime_features",
    "create_lag_features",
    "create_rolling_features",
    # Data splitting / 데이터 분할
    "split_data",
    "split_time_series",
    # Class imbalance / 클래스 불균형
    "check_class_imbalance",
    "undersample",
    "oversample",
    # Pipeline / 파이프라인
    "preprocess_pipeline",
]
