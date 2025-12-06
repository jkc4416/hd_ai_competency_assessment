"""
Utility functions for AI Competency Assessment 2025 Exam Solution.
AI 역량 평가 2025 시험 솔루션을 위한 유틸리티 함수들.

This module contains helper functions extracted from the exam solution notebook:
이 모듈은 시험 솔루션 노트북에서 추출한 헬퍼 함수들을 포함합니다:

1. manage_outliers_iqr: Outlier detection and removal using IQR method
   IQR 방식을 사용한 이상치 탐지 및 제거
2. identify_data_imbalance: Data imbalance analysis for classification
   분류를 위한 데이터 불균형 분석
3. evaluate_threshold: Threshold evaluation for binary classification
   이진 분류를 위한 임계값 평가
4. get_top_anomalies: Top anomaly identification
   상위 이상 샘플 식별
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Configure module logger
# 모듈 로거 설정
logger = logging.getLogger(__name__)


def manage_outliers_iqr(
    df: pd.DataFrame,
    columns: list[str],
    *,
    remove: bool = False,
) -> pd.DataFrame:
    """
    Detect and optionally remove outliers using the IQR method.
    IQR 방식을 사용하여 이상치를 탐지하고 선택적으로 제거합니다.

    Uses the 1.5 * IQR threshold to identify outliers in specified numeric columns.
    지정된 수치형 컬럼에서 1.5 * IQR 임계값을 사용하여 이상치를 식별합니다.

    Args:
        df: Original dataframe to process.
            처리할 원본 데이터프레임.
        columns: List of numeric column names to check for outliers.
            이상치를 확인할 수치형 컬럼 이름 리스트.
        remove: If True, removes rows containing outliers and returns cleaned dataframe.
            If False, only reports outliers and returns original dataframe. Defaults to False.
            True이면 이상치가 포함된 행을 제거한 데이터프레임을 반환합니다.
            False이면 이상치 정보만 출력하고 원본 데이터프레임을 반환합니다. 기본값은 False입니다.

    Returns:
        Cleaned dataframe with outliers removed if remove=True,
        otherwise returns the original dataframe.
        remove=True인 경우 이상치가 제거된 데이터프레임,
        그렇지 않으면 원본 데이터프레임을 반환합니다.

    Examples:
        >>> df = pd.DataFrame({"A": [1, 2, 3, 100], "B": [4, 5, 6, 7]})
        >>> cleaned_df = manage_outliers_iqr(df, ["A", "B"], remove=True)
    """
    df_cleaned = df.copy()
    outlier_indices: set[int] = set()

    logger.info("--- Outlier Detection Results (IQR method, 1.5 * IQR threshold) ---")
    logger.info("--- 이상치 탐지 결과 (IQR 방식, 1.5 * IQR 기준) ---")

    for col in columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            logger.warning(
                "Column '%s' is not numeric, skipping. / 컬럼 '%s'은(는) 수치형이 아니므로 건너뜁니다.",
                col,
                col,
            )
            continue

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        # Calculate outlier boundaries
        # 이상치 경계 계산
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Find outlier indices
        # 이상치 인덱스 찾기
        col_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
        outlier_indices.update(col_outliers)

        logger.info(
            "Column '%s': %d outliers found (lower: %.2f, upper: %.2f)",
            col,
            len(col_outliers),
            lower_bound,
            upper_bound,
        )
        logger.info(
            "컬럼 '%s': 이상치 %d개 발견 (하한: %.2f, 상한: %.2f)",
            col,
            len(col_outliers),
            lower_bound,
            upper_bound,
        )

    if remove:
        original_count = len(df)
        df_cleaned = df_cleaned.drop(index=list(outlier_indices))
        cleaned_count = len(df_cleaned)

        logger.info("--- Outlier Removal Results / 이상치 제거 결과 ---")
        logger.info("Original rows: %d / 원본 데이터 행 수: %d", original_count, original_count)
        logger.info(
            "Removed outlier rows: %d / 제거된 이상치 행 수: %d",
            original_count - cleaned_count,
            original_count - cleaned_count,
        )
        logger.info(
            "Rows after removal: %d / 이상치 제거 후 데이터 행 수: %d",
            cleaned_count,
            cleaned_count,
        )
        return df_cleaned

    logger.info(
        "remove=False, returning original dataframe. / "
        "remove=False로 설정되어 원본 데이터프레임을 반환합니다."
    )
    return df


def identify_data_imbalance(df: pd.DataFrame, target_column: str) -> dict:
    """
    Identify data imbalance in the specified target column.
    지정된 타겟 열에서 데이터 불균형을 식별합니다.

    Calculates class frequencies, percentages, and imbalance ratio to assess
    the severity of class imbalance in classification tasks.
    분류 작업에서 클래스 불균형의 심각도를 평가하기 위해 클래스별 빈도수, 백분율,
    불균형 비율을 계산합니다.

    Args:
        df: Pandas DataFrame to check for imbalance.
            불균형을 확인할 Pandas DataFrame.
        target_column: Name of the target column to analyze.
            분석할 타겟 열의 이름.

    Returns:
        Dictionary containing:
        - Total sample count / 총 샘플 수
        - Class frequencies / 클래스별 빈도수
        - Class percentages / 클래스별 백분율
        - Imbalance ratio (max/min) / 불균형 비율
        - Severity assessment / 심각도 평가

        다음을 포함하는 딕셔너리:
        - 총 샘플 수, 클래스별 빈도수, 백분율, 불균형 비율, 심각도 평가

    Examples:
        >>> df = pd.DataFrame({"target": [0, 0, 0, 1, 1, 1, 1, 1]})
        >>> result = identify_data_imbalance(df, "target")
        >>> print(result["imbalance_ratio"])
    """
    if target_column not in df.columns:
        error_msg = f"Column '{target_column}' does not exist in dataframe."
        logger.error(error_msg)
        return {"error": error_msg}

    # Calculate class frequencies
    # 클래스별 빈도수 계산
    counts = df[target_column].value_counts()
    total_samples = len(df)

    # Calculate class percentages
    # 클래스별 백분율 계산
    percentages = (counts / total_samples) * 100

    # Format results
    # 결과 포맷팅
    results: dict = {
        "total_samples": total_samples,
        "class_counts": counts.to_dict(),
        "class_percentages": {k: round(v, 2) for k, v in percentages.to_dict().items()},
        "description": (
            "A larger percentage difference between classes (typically >20-30%) "
            "indicates more severe imbalance. / "
            "클래스 간 백분율 차이가 클수록(일반적으로 20~30% 이상 차이) 불균형이 심각합니다."
        ),
    }

    # Calculate imbalance ratio
    # 불균형 지수 계산
    if len(percentages) > 1:
        imbalance_ratio = float(percentages.max() / percentages.min())
        results["imbalance_ratio"] = round(imbalance_ratio, 2)

        if imbalance_ratio > 2:
            results["severity"] = (
                "Medium or High (consider resampling) / " "중간 또는 높음 (리샘플링 고려 필요)"
            )
        else:
            results["severity"] = "Low / 낮음"
    else:
        results["severity"] = "Only one class exists / 클래스가 하나만 존재합니다."

    return results


def evaluate_threshold(
    y_true: NDArray[np.int_],
    y_prob: NDArray[np.float64],
    threshold: float,
) -> tuple[float, float, float, NDArray[np.int_]]:
    """
    Evaluate classification performance at a given threshold.
    주어진 임계값에서 분류 성능을 평가합니다.

    Converts predicted probabilities to binary predictions using the specified
    threshold and calculates precision, recall, and F1 score.
    지정된 임계값을 사용하여 예측 확률을 이진 예측으로 변환하고
    정밀도, 재현율, F1 점수를 계산합니다.

    Args:
        y_true: True binary labels (ground truth).
            실제 이진 레이블 (정답).
        y_prob: Predicted probabilities for the positive class.
            양성 클래스에 대한 예측 확률.
        threshold: Classification threshold value (0.0 to 1.0).
            분류 임계값 (0.0에서 1.0 사이).

    Returns:
        Tuple containing (precision, recall, f1_score, y_pred).
        (정밀도, 재현율, F1 점수, 예측 레이블)을 포함하는 튜플.

    Examples:
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> y_prob = np.array([0.2, 0.8, 0.6, 0.3, 0.9])
        >>> precision, recall, f1, y_pred = evaluate_threshold(y_true, y_prob, 0.5)
    """
    y_pred = (y_prob >= threshold).astype(int)
    precision = float(precision_score(y_true, y_pred))
    recall = float(recall_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))

    logger.debug(
        "Threshold %.2f: precision=%.4f, recall=%.4f, f1=%.4f",
        threshold,
        precision,
        recall,
        f1,
    )

    return precision, recall, f1, y_pred


def get_top_anomalies(
    df_validation: pd.DataFrame,
    prob_1d_array: NDArray[np.float64],
    select_columns: list[str],
    *,
    top_n: int = 30,
) -> pd.DataFrame:
    """
    Identify and return the top N anomalies from a validation dataset.
    검증 데이터셋에서 상위 N개의 이상 샘플을 식별하고 반환합니다.

    Filters samples in the top 5% of predicted probabilities (assuming higher
    probability indicates higher likelihood of being anomalous), sorts them
    by probability, and returns the top N samples.
    예측 확률 상위 5%에 해당하는 샘플을 필터링하고(확률이 높을수록 비정상일 가능성이
    높다고 가정), 확률 순으로 정렬하여 상위 N개 샘플을 반환합니다.

    Args:
        df_validation: Validation dataframe containing the data to analyze.
            분석할 데이터가 포함된 검증 데이터프레임.
        prob_1d_array: 1D array of predicted probabilities for each sample.
            각 샘플에 대한 예측 확률의 1차원 배열.
        select_columns: List of column names to include in the output.
            출력에 포함할 컬럼 이름 리스트.
        top_n: Number of top anomalies to return. Defaults to 30.
            반환할 상위 이상 샘플 수. 기본값은 30.

    Returns:
        DataFrame containing top N anomalies with selected columns and
        prediction probability ('y_pred_proba' column).
        선택된 컬럼과 예측 확률('y_pred_proba' 컬럼)을 포함하는
        상위 N개 이상 샘플의 DataFrame.

    Examples:
        >>> df = pd.DataFrame({"A": range(100), "B": range(100)})
        >>> probs = np.random.random(100)
        >>> top_anomalies = get_top_anomalies(df, probs, ["A", "B"], top_n=10)
    """
    # Create a copy to avoid modifying the original dataframe
    # 원본 데이터프레임 수정을 방지하기 위해 복사본 생성
    df_result = df_validation.copy()

    # Add prediction probabilities
    # 예측 확률 추가
    df_result["y_pred_proba"] = pd.Series(prob_1d_array, index=df_result.index)

    # Calculate 95th percentile threshold (top 5%)
    # 95 백분위수 임계값 계산 (상위 5%)
    threshold_value = df_result["y_pred_proba"].quantile(0.95)

    logger.info(
        "95th percentile threshold: %.4f / 95 백분위수 임계값: %.4f",
        threshold_value,
        threshold_value,
    )

    # Filter samples above threshold (top 5%)
    # 임계값 이상의 샘플 필터링 (상위 5%)
    anomalies_top_5_percent = df_result[df_result["y_pred_proba"] >= threshold_value]

    # Sort by probability in descending order
    # 확률 기준 내림차순 정렬
    sorted_anomalies = anomalies_top_5_percent.sort_values(
        by="y_pred_proba",
        ascending=False,
    )

    # Select top N samples
    # 상위 N개 샘플 선택
    top_n_anomalies = sorted_anomalies.head(top_n)

    # Ensure y_pred_proba is included in output columns
    # 출력 컬럼에 y_pred_proba가 포함되도록 보장
    final_columns = list(set(select_columns) | {"y_pred_proba"})

    logger.info(
        "Returning top %d anomalies out of %d samples. / "
        "%d개 샘플 중 상위 %d개 이상 샘플을 반환합니다.",
        len(top_n_anomalies),
        len(df_validation),
        len(df_validation),
        len(top_n_anomalies),
    )

    return top_n_anomalies[final_columns]


__all__ = [
    "manage_outliers_iqr",
    "identify_data_imbalance",
    "evaluate_threshold",
    "get_top_anomalies",
]
