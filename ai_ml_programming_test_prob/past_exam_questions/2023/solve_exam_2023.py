#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI 역량 평가 2023 문제 풀이 스크립트
AI Competency Assessment 2023 Exam Solution Script

이 스크립트는 2023 AI 역량 평가의 모든 문제(Q1-Q17)를 순차적으로 풀어냅니다.
This script solves all problems (Q1-Q17) of the 2023 AI Competency Assessment sequentially.

Usage:
    python solve_exam_2023.py

Author: AI Assistant
"""

from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from lightgbm.sklearn import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from xgboost import XGBRegressor

if TYPE_CHECKING:
    from numpy.typing import NDArray

# =============================================================================
# 로깅 설정 (Logging Configuration)
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# 상수 정의 (Constants Definition)
# =============================================================================
DATA_PATH = Path("dataset/")
RANDOM_SEED = 42
SEQ_LEN = 10  # LSTM 시퀀스 길이 (LSTM sequence length)

# LSTM 하이퍼파라미터 (LSTM Hyperparameters)
HIDDEN_DIM = 64
N_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001


def seed_everything(seed: int = RANDOM_SEED) -> None:
    """
    재현성을 위한 시드 설정 함수
    Set random seed for reproducibility

    Args:
        seed: 랜덤 시드 값 (Random seed value)
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


class LSTMModel(nn.Module):
    """
    LSTM 모델 클래스
    LSTM Model Class

    전력 소비량 예측을 위한 LSTM 신경망 모델입니다.
    LSTM neural network model for power consumption prediction.

    Args:
        input_dim: 입력 feature 차원 (Input feature dimension)
        hidden_dim: 은닉층 차원 (Hidden layer dimension)
        n_layers: LSTM 레이어 수 (Number of LSTM layers)
        output_dim: 출력 차원 (Output dimension)
        dropout: 드롭아웃 비율 (Dropout rate)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = HIDDEN_DIM,
        n_layers: int = N_LAYERS,
        output_dim: int = 1,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파 함수 (Forward pass function)

        Args:
            x: 입력 텐서 (batch_size, seq_len, input_dim)

        Returns:
            출력 텐서 (batch_size, output_dim)
        """
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


def create_sequences(
    features: NDArray[np.float64], targets: NDArray[np.float64], seq_len: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    시계열 시퀀스 데이터 생성 함수
    Create time series sequence data

    Args:
        features: 입력 feature 배열 (Input feature array)
        targets: 타겟 값 배열 (Target value array)
        seq_len: 시퀀스 길이 (Sequence length)

    Returns:
        X: (num_samples, seq_len, num_features) 형태의 배열
        y: (num_samples,) 형태의 배열
    """
    x_list: list[NDArray[np.float64]] = []
    y_list: list[float] = []
    for i in range(len(features) - seq_len):
        x_list.append(features[i : i + seq_len])
        y_list.append(targets[i + seq_len])
    return np.array(x_list), np.array(y_list)


def solve_q1() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Q1. CSV 파일들을 불러오고 shape 출력
    Q1. Load CSV files and print shapes

    Returns:
        train, test, building_info DataFrames
    """
    logger.info("=" * 60)
    logger.info("Q1: Loading CSV files and printing shapes")
    logger.info("=" * 60)

    train = pd.read_csv(DATA_PATH / "train.csv")
    test = pd.read_csv(DATA_PATH / "test.csv")
    building_info = pd.read_csv(DATA_PATH / "building_info.csv")

    logger.info(f"train.csv shape: {train.shape}")
    logger.info(f"test.csv shape: {test.shape}")
    logger.info(f"building_info.csv shape: {building_info.shape}")

    return train, test, building_info


def solve_q2(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """
    Q2. train, test 데이터의 결측치 확인 및 column별 결측치 개수 출력
    Q2. Check missing values in train/test data and print count per column

    Args:
        train: 학습 데이터 (Training data)
        test: 테스트 데이터 (Test data)
    """
    logger.info("=" * 60)
    logger.info("Q2: Checking missing values")
    logger.info("=" * 60)

    logger.info("Train 데이터 결측치 (Missing values in Train data):")
    for col, count in train.isnull().sum().items():
        if count > 0:
            logger.info(f"  {col}: {count}")

    logger.info("Test 데이터 결측치 (Missing values in Test data):")
    for col, count in test.isnull().sum().items():
        if count > 0:
            logger.info(f"  {col}: {count}")


def solve_q3(building_info: pd.DataFrame) -> pd.DataFrame:
    """
    Q3. building_info의 '-'를 0.0으로 변경하고 해당 column의 dtype을 float로 변환
    Q3. Replace '-' with 0.0 in building_info and convert dtype to float

    Args:
        building_info: 건물 정보 데이터 (Building info data)

    Returns:
        처리된 building_info DataFrame
    """
    logger.info("=" * 60)
    logger.info("Q3: Replacing '-' with 0.0 in building_info")
    logger.info("=" * 60)

    columns_with_dash = []
    for col in building_info.columns:
        if building_info[col].dtype == "object":
            if (building_info[col] == "-").any():
                columns_with_dash.append(col)

    logger.info(f"'-' 값이 있는 컬럼 (Columns with '-'): {columns_with_dash}")

    for col in columns_with_dash:
        building_info[col] = building_info[col].replace("-", 0.0).astype(float)

    logger.info("변환 완료 (Conversion completed)")
    return building_info


def solve_q4(
    train: pd.DataFrame, test: pd.DataFrame, building_info: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Q4. building_info를 train, test 데이터와 INNER JOIN으로 병합
    Q4. Merge building_info with train/test using INNER JOIN

    Args:
        train: 학습 데이터 (Training data)
        test: 테스트 데이터 (Test data)
        building_info: 건물 정보 데이터 (Building info data)

    Returns:
        병합된 train, test DataFrames
    """
    logger.info("=" * 60)
    logger.info("Q4: Merging building_info with train/test")
    logger.info("=" * 60)

    num_col = "num" if "num" in train.columns else "건물번호"

    train = pd.merge(train, building_info, on=num_col, how="inner")
    test = pd.merge(test, building_info, on=num_col, how="inner")

    logger.info(f"병합 후 train shape: {train.shape}")
    logger.info(f"병합 후 test shape: {test.shape}")

    return train, test


def solve_q5(train: pd.DataFrame) -> pd.Series:
    """
    Q5. 건물유형별 전력사용량 평균값 계산
    Q5. Calculate mean power consumption by building type

    Args:
        train: 학습 데이터 (Training data)

    Returns:
        건물유형별 평균 전력사용량 Series
    """
    logger.info("=" * 60)
    logger.info("Q5: Calculating mean power by building type")
    logger.info("=" * 60)

    building_type_col = "building_type" if "building_type" in train.columns else "건물유형"
    target_col = "target" if "target" in train.columns else "전력소비량(kWh)"

    mean_power_by_type = train.groupby(building_type_col)[target_col].mean()

    logger.info("건물유형별 전력사용량 평균 (Mean power by building type):")
    for btype, mean_val in mean_power_by_type.sort_values(ascending=False).items():
        logger.info(f"  {btype}: {mean_val:.2f} kWh")

    return mean_power_by_type


def solve_q6(train: pd.DataFrame) -> None:
    """
    Q6. 건물유형별 전력사용량을 시간에 따른 선그래프로 출력
    Q6. Plot power consumption by building type over time

    Args:
        train: 학습 데이터 (Training data)
    """
    logger.info("=" * 60)
    logger.info("Q6: Plotting power consumption by building type over time")
    logger.info("=" * 60)

    date_col = "date_time" if "date_time" in train.columns else "일시"
    building_type_col = "building_type" if "building_type" in train.columns else "건물유형"
    target_col = "target" if "target" in train.columns else "전력소비량(kWh)"

    train[date_col] = pd.to_datetime(train[date_col])

    power_by_type_time = (
        train.groupby([date_col, building_type_col])[target_col].mean().reset_index()
    )
    building_types = power_by_type_time[building_type_col].unique()

    fig, ax = plt.subplots(figsize=(16, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, len(building_types)))

    for i, btype in enumerate(building_types):
        data = power_by_type_time[power_by_type_time[building_type_col] == btype]
        ax.plot(data[date_col], data[target_col], label=btype, color=colors[i], alpha=0.8)

    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Power Consumption (kWh)", fontsize=12)
    ax.set_title("Power Consumption by Building Type over Time", fontsize=14)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("q6_power_consumption_plot.png", dpi=150)
    logger.info("Plot saved to q6_power_consumption_plot.png")
    plt.close()


def solve_q7(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Q7. 강수량은 0으로 채우고, 풍속/습도는 linear interpolation으로 채움
    Q7. Fill precipitation with 0, interpolate wind speed and humidity

    Args:
        train: 학습 데이터 (Training data)
        test: 테스트 데이터 (Test data)

    Returns:
        결측치 처리된 train, test DataFrames
    """
    logger.info("=" * 60)
    logger.info("Q7: Filling missing values")
    logger.info("=" * 60)

    precip_col = "precipitation" if "precipitation" in train.columns else "강수량(mm)"
    wind_col = "windspeed" if "windspeed" in train.columns else "풍속(m/s)"
    humid_col = "humidity" if "humidity" in train.columns else "습도(%)"

    # Train 데이터 처리 (Process train data)
    train[precip_col] = train[precip_col].fillna(0)
    train[wind_col] = train[wind_col].interpolate(method="linear")
    train[humid_col] = train[humid_col].interpolate(method="linear")

    # Test 데이터 처리 (Process test data)
    test[precip_col] = test[precip_col].fillna(0)
    test[wind_col] = test[wind_col].interpolate(method="linear")
    test[humid_col] = test[humid_col].interpolate(method="linear")

    logger.info("결측치 처리 완료 (Missing value handling completed)")
    return train, test


def solve_q8(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Q8. 건물유형 정보를 One-Hot Encoding
    Q8. One-Hot Encode building type information

    Args:
        train: 학습 데이터 (Training data)
        test: 테스트 데이터 (Test data)

    Returns:
        One-Hot Encoding된 train, test DataFrames
    """
    logger.info("=" * 60)
    logger.info("Q8: One-Hot Encoding building types")
    logger.info("=" * 60)

    building_type_col = "building_type" if "building_type" in train.columns else "건물유형"

    train = pd.get_dummies(train, columns=[building_type_col], prefix="building_type")
    test = pd.get_dummies(test, columns=[building_type_col], prefix="building_type")

    building_type_cols = [col for col in train.columns if col.startswith("building_type_")]
    logger.info(f"생성된 건물유형 컬럼 수: {len(building_type_cols)}")

    return train, test


def solve_q9(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Q9. TimeSeriesKMeans를 활용하여 시간당 전력사용량이 비슷한 건물 그룹 찾기
    Q9. Use TimeSeriesKMeans to find building groups with similar hourly power consumption

    Args:
        train: 학습 데이터 (Training data)
        test: 테스트 데이터 (Test data)

    Returns:
        cluster 컬럼이 추가된 train, test DataFrames
    """
    logger.info("=" * 60)
    logger.info("Q9: TimeSeriesKMeans clustering")
    logger.info("=" * 60)

    num_col = "num" if "num" in train.columns else "건물번호"
    date_col = "date_time" if "date_time" in train.columns else "일시"
    target_col = "target" if "target" in train.columns else "전력소비량(kWh)"

    # 건물별 시계열 데이터 생성 (Create time series data per building)
    building_power_matrix = train.pivot_table(
        index=num_col, columns=date_col, values=target_col, aggfunc="mean"
    ).values

    logger.info(f"건물별 전력사용량 행렬 shape: {building_power_matrix.shape}")

    # MinMaxScaler로 정규화 (Normalize with MinMaxScaler)
    scaler = MinMaxScaler()
    building_power_scaled = scaler.fit_transform(building_power_matrix.T).T

    # 최적의 K 찾기 (Find optimal K)
    best_k = 2
    best_score = -1.0

    logger.info("K별 Silhouette Score 계산 중...")
    for k in range(2, 11):
        km = TimeSeriesKMeans(n_clusters=k, metric="euclidean", random_state=RANDOM_SEED)
        labels = km.fit_predict(building_power_scaled)
        score = silhouette_score(building_power_scaled, labels, metric="euclidean")
        logger.info(f"K={k}: Silhouette Score = {score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k

    logger.info(f"최적의 K: {best_k}, Silhouette Score: {best_score:.4f}")

    # 최적의 K로 최종 클러스터링 (Final clustering with optimal K)
    final_km = TimeSeriesKMeans(n_clusters=best_k, metric="euclidean", random_state=RANDOM_SEED)
    final_labels = final_km.fit_predict(building_power_scaled)

    # 클러스터 결과를 DataFrame으로 생성 (Create DataFrame with cluster results)
    cluster_df = pd.DataFrame({num_col: range(1, 101), "cluster": final_labels})

    # train, test에 cluster 컬럼 추가 (Add cluster column to train, test)
    train = pd.merge(train, cluster_df, on=num_col, how="left")
    test = pd.merge(test, cluster_df, on=num_col, how="left")

    return train, test


def solve_q10(
    train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Q10. 학습에 필요없는 컬럼 제외 후 X, y 생성 및 8:2 분할
    Q10. Remove unnecessary columns, create X, y and split 80/20

    Args:
        train: 학습 데이터 (Training data)

    Returns:
        X, X_train, X_valid, y, y_train, y_valid
    """
    logger.info("=" * 60)
    logger.info("Q10: Creating X, y and splitting data")
    logger.info("=" * 60)

    num_col = "num" if "num" in train.columns else "건물번호"
    date_col = "date_time" if "date_time" in train.columns else "일시"
    target_col = "target" if "target" in train.columns else "전력소비량(kWh)"

    drop_cols = ["num_date_time", num_col, date_col]

    x = train.drop(columns=drop_cols + [target_col], errors="ignore")
    y = train[target_col]

    logger.info(f"X shape: {x.shape}")
    logger.info(f"y shape: {y.shape}")

    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_SEED, shuffle=True
    )

    logger.info(f"X_train shape: {x_train.shape}")
    logger.info(f"X_valid shape: {x_valid.shape}")

    return x, x_train, x_valid, y, y_train, y_valid


def solve_q11(
    x_train: pd.DataFrame, y_train: pd.Series, x_valid: pd.DataFrame
) -> tuple[XGBRegressor, NDArray[np.float64], NDArray[np.float64]]:
    """
    Q11. XGBoost 학습 및 예측
    Q11. Train XGBoost and make predictions

    Args:
        x_train: 학습 feature (Training features)
        y_train: 학습 타겟 (Training target)
        x_valid: 검증 feature (Validation features)

    Returns:
        학습된 모델, 학습 예측값, 검증 예측값
    """
    logger.info("=" * 60)
    logger.info("Q11: Training XGBoost model")
    logger.info("=" * 60)

    xgb_model = XGBRegressor(
        max_depth=10,
        n_estimators=200,
        learning_rate=0.1,
        colsample_bynode=0.5,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    logger.info("XGBoost 모델 학습 중...")
    xgb_model.fit(x_train, y_train)
    logger.info("학습 완료!")

    y_train_pred = xgb_model.predict(x_train)
    y_valid_pred = xgb_model.predict(x_valid)

    return xgb_model, y_train_pred, y_valid_pred


def solve_q12(
    y_train: pd.Series,
    y_train_pred: NDArray[np.float64],
    y_valid: pd.Series,
    y_valid_pred: NDArray[np.float64],
) -> None:
    """
    Q12. 모델 평가 - Baseline RMSE, P_train RMSE, P_valid RMSE 출력
    Q12. Model evaluation - Print Baseline RMSE, P_train RMSE, P_valid RMSE

    Args:
        y_train: 학습 타겟 (Training target)
        y_train_pred: 학습 예측값 (Training predictions)
        y_valid: 검증 타겟 (Validation target)
        y_valid_pred: 검증 예측값 (Validation predictions)
    """
    logger.info("=" * 60)
    logger.info("Q12: Model evaluation")
    logger.info("=" * 60)

    baseline_pred = np.full(len(y_train), y_train.mean())
    baseline_rmse = np.sqrt(mean_squared_error(y_train, baseline_pred))
    p_train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    p_valid_rmse = np.sqrt(mean_squared_error(y_valid, y_valid_pred))

    logger.info(f"Baseline RMSE (평균값 예측): {baseline_rmse:.4f}")
    logger.info(f"P_train RMSE (학습 데이터): {p_train_rmse:.4f}")
    logger.info(f"P_valid RMSE (검증 데이터): {p_valid_rmse:.4f}")


def solve_q13(
    x_train: pd.DataFrame, y_train: pd.Series, x_valid: pd.DataFrame, y_valid: pd.Series
) -> XGBRegressor:
    """
    Q13. GridSearchCV를 사용하여 XGBoost 하이퍼파라미터 튜닝
    Q13. Use GridSearchCV for XGBoost hyperparameter tuning

    Args:
        x_train: 학습 feature (Training features)
        y_train: 학습 타겟 (Training target)
        x_valid: 검증 feature (Validation features)
        y_valid: 검증 타겟 (Validation target)

    Returns:
        최적화된 XGBoost 모델
    """
    logger.info("=" * 60)
    logger.info("Q13: GridSearchCV hyperparameter tuning")
    logger.info("=" * 60)

    param_grid = {
        "max_depth": [6, 8, 10],
        "n_estimators": [200, 300, 400],
        "learning_rate": [0.05, 0.1, 0.15],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "subsample": [0.8, 0.9, 1.0],
    }

    xgb_base = XGBRegressor(random_state=RANDOM_SEED, n_jobs=-1)

    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        cv=3,
        scoring="neg_root_mean_squared_error",
        verbose=1,
        n_jobs=-1,
    )

    logger.info("GridSearchCV 수행 중... (시간이 다소 소요될 수 있습니다)")
    grid_search.fit(x_train, y_train)

    logger.info("최적 하이퍼파라미터:")
    for param, value in grid_search.best_params_.items():
        logger.info(f"  {param}: {value}")

    best_model = grid_search.best_estimator_

    y_train_pred_best = best_model.predict(x_train)
    y_valid_pred_best = best_model.predict(x_valid)

    p_train_rmse_best = np.sqrt(mean_squared_error(y_train, y_train_pred_best))
    p_valid_rmse_best = np.sqrt(mean_squared_error(y_valid, y_valid_pred_best))

    logger.info(f"GridSearchCV 후 P_train RMSE: {p_train_rmse_best:.4f}")
    logger.info(f"GridSearchCV 후 P_valid RMSE: {p_valid_rmse_best:.4f}")

    return best_model


def solve_q14(
    best_model: XGBRegressor, test: pd.DataFrame, x: pd.DataFrame
) -> pd.DataFrame:
    """
    Q14. Test 데이터를 best_estimator로 예측하고 submission 생성
    Q14. Predict test data with best_estimator and create submission

    Args:
        best_model: 최적화된 XGBoost 모델
        test: 테스트 데이터
        x: 학습에 사용된 feature DataFrame (컬럼 정렬용)

    Returns:
        예측값이 채워진 submission DataFrame
    """
    logger.info("=" * 60)
    logger.info("Q14: Predicting on test data")
    logger.info("=" * 60)

    num_col = "num" if "num" in test.columns else "건물번호"
    date_col = "date_time" if "date_time" in test.columns else "일시"
    drop_cols = ["num_date_time", num_col, date_col]

    submission = pd.read_csv(DATA_PATH / "sample_submission.csv")
    logger.info(f"sample_submission shape: {submission.shape}")

    x_test = test.drop(columns=drop_cols, errors="ignore")

    # 컬럼 정렬 (Align columns)
    missing_cols = set(x.columns) - set(x_test.columns)
    for col in missing_cols:
        x_test[col] = 0

    extra_cols = set(x_test.columns) - set(x.columns)
    x_test = x_test.drop(columns=list(extra_cols), errors="ignore")
    x_test = x_test[x.columns]

    test_predictions = best_model.predict(x_test)
    submission["answer"] = test_predictions

    logger.info(f"예측 완료! 예측값 shape: {test_predictions.shape}")

    return submission


def solve_q15_q16_q17(
    train: pd.DataFrame, test: pd.DataFrame, submission: pd.DataFrame
) -> pd.DataFrame:
    """
    Q15-Q17. LSTM 모델 구축, 학습, 예측
    Q15-Q17. Build, train, and predict with LSTM model

    Args:
        train: 학습 데이터
        test: 테스트 데이터
        submission: 제출 DataFrame

    Returns:
        answer2 컬럼이 추가된 submission DataFrame
    """
    logger.info("=" * 60)
    logger.info("Q15-Q17: LSTM model training and prediction")
    logger.info("=" * 60)

    num_col = "num" if "num" in train.columns else "건물번호"
    date_col = "date_time" if "date_time" in train.columns else "일시"
    target_col = "target" if "target" in train.columns else "전력소비량(kWh)"
    drop_cols = ["num_date_time", num_col, date_col]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"사용 디바이스: {device}")

    # Q15: 1번 건물 데이터 추출 및 시퀀스 생성
    train_b1 = train[train[num_col] == 1].copy()
    train_b1 = train_b1.sort_values(by=date_col).reset_index(drop=True)
    logger.info(f"1번 건물 데이터 shape: {train_b1.shape}")

    feature_cols = [col for col in train_b1.columns if col not in drop_cols + [target_col]]

    lstm_scaler = StandardScaler()
    train_b1_scaled = lstm_scaler.fit_transform(train_b1[feature_cols])

    # 날짜 기준 분할 (Split by date)
    train_end_date = pd.to_datetime("2022-08-10 23:00:00")
    valid_start_date = pd.to_datetime("2022-08-11 00:00:00")
    valid_end_date = pd.to_datetime("2022-08-16 23:00:00")

    train_mask = train_b1[date_col] <= train_end_date
    valid_mask = (train_b1[date_col] >= valid_start_date) & (train_b1[date_col] <= valid_end_date)

    train_features = train_b1_scaled[train_mask]
    train_targets = train_b1[target_col].values[train_mask]
    valid_features = train_b1_scaled[valid_mask]
    valid_targets = train_b1[target_col].values[valid_mask]

    # 시퀀스 생성 (Create sequences)
    x_lstm_train, y_lstm_train = create_sequences(train_features, train_targets, SEQ_LEN)

    all_features_for_valid = np.concatenate([train_features[-SEQ_LEN:], valid_features], axis=0)
    all_targets_for_valid = np.concatenate([train_targets[-SEQ_LEN:], valid_targets], axis=0)
    x_lstm_valid, y_lstm_valid = create_sequences(
        all_features_for_valid, all_targets_for_valid, SEQ_LEN
    )

    logger.info(f"X_lstm_train shape: {x_lstm_train.shape}")
    logger.info(f"X_lstm_valid shape: {x_lstm_valid.shape}")

    # Q16: LSTM 모델 학습
    input_dim = x_lstm_train.shape[2]

    x_train_tensor = torch.FloatTensor(x_lstm_train).to(device)
    y_train_tensor = torch.FloatTensor(y_lstm_train).unsqueeze(1).to(device)
    x_valid_tensor = torch.FloatTensor(x_lstm_valid).to(device)
    y_valid_tensor = torch.FloatTensor(y_lstm_valid).unsqueeze(1).to(device)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    lstm_model = LSTMModel(input_dim=input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE)

    logger.info("LSTM 학습 시작...")
    best_valid_loss = float("inf")

    for epoch in range(EPOCHS):
        lstm_model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = lstm_model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
        train_loss /= len(train_dataset)

        lstm_model.eval()
        with torch.no_grad():
            valid_outputs = lstm_model(x_valid_tensor)
            valid_loss = criterion(valid_outputs, y_valid_tensor).item()

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(lstm_model.state_dict(), "best_lstm_model.pth")

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch [{epoch+1}/{EPOCHS}] - Train: {train_loss:.4f}, Valid: {valid_loss:.4f}")

    # Best model 로드 및 평가 (Load best model and evaluate)
    lstm_model.load_state_dict(torch.load("best_lstm_model.pth", weights_only=True))

    lstm_model.eval()
    with torch.no_grad():
        y_train_pred_lstm = lstm_model(x_train_tensor).cpu().numpy().flatten()
        y_valid_pred_lstm = lstm_model(x_valid_tensor).cpu().numpy().flatten()

    lstm_train_rmse = np.sqrt(mean_squared_error(y_lstm_train, y_train_pred_lstm))
    lstm_valid_rmse = np.sqrt(mean_squared_error(y_lstm_valid, y_valid_pred_lstm))

    logger.info(f"LSTM Train RMSE: {lstm_train_rmse:.4f}")
    logger.info(f"LSTM Valid RMSE: {lstm_valid_rmse:.4f}")

    # Q17: Test 예측
    test_b1 = test[test[num_col] == 1].copy()
    test_b1 = test_b1.sort_values(by=date_col).reset_index(drop=True)

    test_b1_scaled = lstm_scaler.transform(test_b1[feature_cols])
    all_features_for_test = np.concatenate([train_b1_scaled[-SEQ_LEN:], test_b1_scaled], axis=0)

    x_lstm_test = []
    for i in range(len(test_b1_scaled)):
        x_lstm_test.append(all_features_for_test[i : i + SEQ_LEN])
    x_lstm_test = np.array(x_lstm_test)

    x_test_tensor = torch.FloatTensor(x_lstm_test).to(device)

    lstm_model.eval()
    with torch.no_grad():
        preds = lstm_model(x_test_tensor).cpu().numpy().flatten()

    submission["answer2"] = preds.tolist() + [0.0] * (len(submission) - len(preds))

    logger.info(f"LSTM 예측 완료! 예측값 shape: {preds.shape}")

    return submission


def main() -> None:
    """
    메인 실행 함수
    Main execution function
    """
    logger.info("=" * 60)
    logger.info("AI 역량 평가 2023 문제 풀이 시작")
    logger.info("AI Competency Assessment 2023 Exam Solution Start")
    logger.info("=" * 60)

    # 시드 설정 (Set seed)
    seed_everything(RANDOM_SEED)

    # Q1: 데이터 로드 (Load data)
    train, test, building_info = solve_q1()

    # Optional: 컬럼명 영문화 (Rename columns to English)
    train.columns = [
        "num_date_time",
        "num",
        "date_time",
        "temperature",
        "precipitation",
        "windspeed",
        "humidity",
        "target",
    ]
    test.columns = [
        "num_date_time",
        "num",
        "date_time",
        "temperature",
        "precipitation",
        "windspeed",
        "humidity",
    ]

    # Q2: 결측치 확인 (Check missing values)
    solve_q2(train, test)

    # Q3: building_info 전처리 (Preprocess building_info)
    building_info = solve_q3(building_info)

    # Q4: 데이터 병합 (Merge data)
    train, test = solve_q4(train, test, building_info)

    # Q5: 건물유형별 평균 전력사용량 (Mean power by building type)
    solve_q5(train)

    # Q6: 시각화 (Visualization)
    solve_q6(train)

    # Q7: 결측치 처리 (Handle missing values)
    train, test = solve_q7(train, test)

    # Q8: One-Hot Encoding
    train, test = solve_q8(train, test)

    # Q9: TimeSeriesKMeans 클러스터링 (Clustering)
    train, test = solve_q9(train, test)

    # Q10: X, y 생성 및 분할 (Create X, y and split)
    x, x_train, x_valid, y, y_train, y_valid = solve_q10(train)

    # Q11: XGBoost 학습 (Train XGBoost)
    xgb_model, y_train_pred, y_valid_pred = solve_q11(x_train, y_train, x_valid)

    # Q12: 모델 평가 (Model evaluation)
    solve_q12(y_train, y_train_pred, y_valid, y_valid_pred)

    # Q13: GridSearchCV 튜닝 (Hyperparameter tuning)
    best_model = solve_q13(x_train, y_train, x_valid, y_valid)

    # Q14: Test 예측 (Test prediction)
    submission = solve_q14(best_model, test, x)

    # Q15-Q17: LSTM 모델 (LSTM model)
    submission = solve_q15_q16_q17(train, test, submission)

    # 제출 파일 생성 (Create submission file)
    clock = int(time.time())
    submission_path = f"submission_script_{clock}.csv"
    submission.to_csv(submission_path, index=False)
    logger.info(f"제출 파일 생성 완료: {submission_path}")

    logger.info("=" * 60)
    logger.info("모든 문제 풀이 완료!")
    logger.info("All problems solved!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
