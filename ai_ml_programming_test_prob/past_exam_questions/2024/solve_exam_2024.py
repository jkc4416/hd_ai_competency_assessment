"""
AI Competency Assessment 2024 - Transformer Oil Temperature (OT) Prediction
AI 역량 평가 2024 - 변압기 오일 온도(OT) 예측

This module provides solutions for the 2024 AI competency assessment exam.
이 모듈은 2024년 AI 역량 평가 시험 문제에 대한 솔루션을 제공합니다.

Problem: Predict transformer oil temperature (OT) using ETDataset.
문제: ETDataset을 사용하여 변압기 오일 온도(OT)를 예측합니다.

Author: AI Assistant
Date: 2024
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from numpy.typing import NDArray

# ==============================================================================
# Logging Configuration | 로깅 설정
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output | 깔끔한 출력을 위해 경고 숨김
warnings.filterwarnings("ignore")

# Optuna verbosity | Optuna 로그 레벨 설정
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==============================================================================
# Constants | 상수 정의
# ==============================================================================
# Data is recorded at 15-minute intervals | 데이터는 15분 간격으로 기록됨
INTERVALS_PER_HOUR = 4  # 15분 간격이므로 1시간 = 4개 데이터 포인트
SEQUENCE_LENGTH = 24 * INTERVALS_PER_HOUR  # 24시간 = 96개 시퀀스 (Q7 요구사항)

# Random seed for reproducibility | 재현성을 위한 랜덤 시드
RANDOM_SEED = 42

# Device configuration for PyTorch | PyTorch 디바이스 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# Q1: Load Data and Check Missing Values
# Q1: 데이터 로드 및 결측치 확인
# ==============================================================================
def load_and_check_data(
    train_path: Path | str,
    test_path: Path | str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and test CSV files, print shapes, and check for missing values.
    train.csv와 test.csv를 불러오고, shape을 출력하며, 결측치를 확인합니다.

    Args:
        train_path: Path to train.csv file. | train.csv 파일 경로.
        test_path: Path to test.csv file. | test.csv 파일 경로.

    Returns:
        Tuple of (train_df, test_df). | (train_df, test_df) 튜플.

    Raises:
        FileNotFoundError: If the specified CSV files do not exist.
                          지정된 CSV 파일이 존재하지 않는 경우.
    """
    train_path = Path(train_path)
    test_path = Path(test_path)

    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    # Load CSV files | CSV 파일 로드
    train = pd.read_csv(train_path, encoding="utf-8")
    test = pd.read_csv(test_path, encoding="utf-8")

    # Print shapes | Shape 출력
    logger.info("=" * 60)
    logger.info("Q1: Data Loading and Missing Value Check")
    logger.info("Q1: 데이터 로드 및 결측치 확인")
    logger.info("=" * 60)

    print("\n[Data Shape | 데이터 Shape]")
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")

    # Check missing values | 결측치 확인
    print("\n[Train Missing Values | Train 결측치]")
    train_missing = train.isnull().sum()
    print(train_missing)

    print("\n[Test Missing Values | Test 결측치]")
    test_missing = test.isnull().sum()
    print(test_missing)

    # Summary | 요약
    total_train_missing = train_missing.sum()
    total_test_missing = test_missing.sum()
    print("\n[Summary | 요약]")
    print(f"Total missing in train: {total_train_missing}")
    print(f"Total missing in test: {total_test_missing}")

    return train, test


# ==============================================================================
# Q2: Create Cyclic Features from Date
# Q2: 날짜로부터 순환 특성 생성
# ==============================================================================
def create_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create hour, dayofweek, month features and apply sin/cos transformation.
    hour, dayofweek, month 특성을 생성하고 sin/cos 변환을 적용합니다.

    Cyclic features help the model understand periodicity in time series data.
    순환 특성은 모델이 시계열 데이터의 주기성을 이해하는 데 도움을 줍니다.

    Args:
        df: DataFrame with 'date' column. | 'date' 열이 있는 DataFrame.

    Returns:
        DataFrame with cyclic features added. | 순환 특성이 추가된 DataFrame.
    """
    df = df.copy()

    # Ensure date column is datetime | date 열이 datetime인지 확인
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    # Extract time components | 시간 구성요소 추출
    df["hour"] = df["date"].dt.hour
    df["dayofweek"] = df["date"].dt.dayofweek  # 0=Monday, 6=Sunday
    df["month"] = df["date"].dt.month

    # Cyclic transformation for hour (24-hour cycle)
    # 시간에 대한 순환 변환 (24시간 주기)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Cyclic transformation for dayofweek (7-day cycle)
    # 요일에 대한 순환 변환 (7일 주기)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    # Cyclic transformation for month (12-month cycle)
    # 월에 대한 순환 변환 (12개월 주기)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    logger.info("Q2: Cyclic features created successfully")
    logger.info("Q2: 순환 특성 생성 완료")
    print("\n[Cyclic Features Created | 생성된 순환 특성]")
    print("- hour, hour_sin, hour_cos")
    print("- dayofweek, dayofweek_sin, dayofweek_cos")
    print("- month, month_sin, month_cos")

    return df


# ==============================================================================
# Q3: Create Lag Features for OT
# Q3: OT에 대한 지연 특성 생성
# ==============================================================================
def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create lag features for OT (1 hour, 2 hours, 3 hours before).
    OT에 대한 지연 특성을 생성합니다 (1시간 전, 2시간 전, 3시간 전).

    Note: Data is recorded at 15-minute intervals.
    참고: 데이터는 15분 간격으로 기록되어 있습니다.
    - 1 hour lag = shift(4) | 1시간 전 = shift(4)
    - 2 hour lag = shift(8) | 2시간 전 = shift(8)
    - 3 hour lag = shift(12) | 3시간 전 = shift(12)

    Args:
        df: DataFrame with 'OT' column. | 'OT' 열이 있는 DataFrame.

    Returns:
        DataFrame with lag features added. | 지연 특성이 추가된 DataFrame.
    """
    df = df.copy()

    # Create lag features (15-minute intervals, so 4 steps = 1 hour)
    # 지연 특성 생성 (15분 간격이므로 4 스텝 = 1시간)
    df["OT_lag_1h"] = df["OT"].shift(INTERVALS_PER_HOUR * 1)  # 1시간 전
    df["OT_lag_2h"] = df["OT"].shift(INTERVALS_PER_HOUR * 2)  # 2시간 전
    df["OT_lag_3h"] = df["OT"].shift(INTERVALS_PER_HOUR * 3)  # 3시간 전

    logger.info("Q3: Lag features created successfully")
    logger.info("Q3: 지연 특성 생성 완료")
    print("\n[Lag Features Created | 생성된 지연 특성]")
    print(f"- OT_lag_1h: OT value from 1 hour ago (shift={INTERVALS_PER_HOUR})")
    print(f"- OT_lag_2h: OT value from 2 hours ago (shift={INTERVALS_PER_HOUR * 2})")
    print(f"- OT_lag_3h: OT value from 3 hours ago (shift={INTERVALS_PER_HOUR * 3})")
    print(f"\nNaN rows created by lag: {df['OT_lag_3h'].isna().sum()}")

    return df


# ==============================================================================
# Q4: Prepare Data and Split Train/Validation
# Q4: 데이터 준비 및 훈련/검증 분할
# ==============================================================================
def prepare_and_split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.75,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Remove 'date' column, create X and y, and split data by time order (3:1 ratio).
    'date' 열을 제거하고, X와 y를 생성하며, 시간 순서에 따라 3:1 비율로 분할합니다.

    Important: For time series data, we do NOT shuffle the data.
    중요: 시계열 데이터에서는 데이터를 섞지 않습니다.

    Args:
        df: Preprocessed DataFrame. | 전처리된 DataFrame.
        train_ratio: Ratio of training data (default: 0.75 = 3:1).
                    훈련 데이터 비율 (기본값: 0.75 = 3:1).

    Returns:
        Tuple of (X_train, X_val, y_train, y_val).

    Raises:
        ValueError: If required columns are missing. | 필수 열이 없는 경우.
    """
    df = df.copy()

    # Remove date column | date 열 제거
    if "date" in df.columns:
        df = df.drop(columns=["date"])

    # Drop rows with NaN values (caused by lag features)
    # NaN 값이 있는 행 제거 (지연 특성으로 인해 발생)
    initial_len = len(df)
    df = df.dropna()
    dropped_rows = initial_len - len(df)
    logger.info(f"Dropped {dropped_rows} rows with NaN values | NaN 행 {dropped_rows}개 제거")

    # Create X (features) and y (target) | X (특성)와 y (타겟) 생성
    if "OT" not in df.columns:
        raise ValueError("'OT' column is required but not found | 'OT' 열이 필요하지만 없습니다")

    y = df["OT"]
    X = df.drop(columns=["OT"])

    # Split by time order (no shuffle for time series)
    # 시간 순서에 따라 분할 (시계열이므로 섞지 않음)
    split_idx = int(len(X) * train_ratio)
    X_train = X.iloc[:split_idx]
    X_val = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_val = y.iloc[split_idx:]

    logger.info("Q4: Data prepared and split successfully")
    logger.info("Q4: 데이터 준비 및 분할 완료")
    print("\n[Data Split | 데이터 분할]")
    print(f"Training set size: {len(X_train)} ({train_ratio * 100:.0f}%)")
    print(f"Validation set size: {len(X_val)} ({(1 - train_ratio) * 100:.0f}%)")
    print(f"Feature columns: {list(X.columns)}")

    return X_train, X_val, y_train, y_val


# ==============================================================================
# Q5: Train LightGBM Baseline Model
# Q5: LightGBM 기준 모델 학습
# ==============================================================================
def train_lightgbm_baseline(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
) -> tuple[lgb.LGBMRegressor, float]:
    """
    Train LightGBM model with specified hyperparameters and calculate RMSE.
    지정된 하이퍼파라미터로 LightGBM 모델을 학습하고 RMSE를 계산합니다.

    Hyperparameters (as specified in the problem):
    하이퍼파라미터 (문제에서 지정됨):
    - num_leaves=31
    - n_estimators=100
    - learning_rate=0.05

    Args:
        X_train: Training features. | 훈련 특성.
        X_val: Validation features. | 검증 특성.
        y_train: Training target. | 훈련 타겟.
        y_val: Validation target. | 검증 타겟.

    Returns:
        Tuple of (trained model, validation RMSE). | (학습된 모델, 검증 RMSE) 튜플.
    """
    # Create model with specified hyperparameters | 지정된 하이퍼파라미터로 모델 생성
    lgb_model = lgb.LGBMRegressor(
        num_leaves=31,
        n_estimators=100,
        learning_rate=0.05,
        random_state=RANDOM_SEED,
        verbosity=-1,  # Suppress LightGBM warnings | LightGBM 경고 숨김
    )

    # Train model | 모델 학습
    lgb_model.fit(X_train, y_train)

    # Predict and calculate RMSE | 예측 및 RMSE 계산
    y_pred = lgb_model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    logger.info("Q5: LightGBM baseline model trained successfully")
    logger.info("Q5: LightGBM 기준 모델 학습 완료")
    print("\n[LightGBM Baseline Results | LightGBM 기준 결과]")
    print("Hyperparameters: num_leaves=31, n_estimators=100, learning_rate=0.05")
    print(f"Validation RMSE: {rmse:.6f}")

    return lgb_model, rmse


# ==============================================================================
# Q6: Optuna Hyperparameter Tuning for LightGBM
# Q6: LightGBM에 대한 Optuna 하이퍼파라미터 튜닝
# ==============================================================================
def optuna_lightgbm_tuning(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    n_trials: int = 50,
) -> tuple[lgb.LGBMRegressor, float, dict[str, Any]]:
    """
    Use Optuna to tune LightGBM hyperparameters and achieve RMSE < 0.5.
    Optuna를 사용하여 LightGBM 하이퍼파라미터를 튜닝하고 RMSE < 0.5를 달성합니다.

    Args:
        X_train: Training features. | 훈련 특성.
        X_val: Validation features. | 검증 특성.
        y_train: Training target. | 훈련 타겟.
        y_val: Validation target. | 검증 타겟.
        n_trials: Number of optimization trials (default: 50).
                 최적화 시도 횟수 (기본값: 50).

    Returns:
        Tuple of (best model, best RMSE, best params).
        (최적 모델, 최적 RMSE, 최적 파라미터) 튜플.
    """

    def objective(trial: optuna.Trial) -> float:
        """
        Optuna objective function for LightGBM hyperparameter optimization.
        LightGBM 하이퍼파라미터 최적화를 위한 Optuna 목적 함수.
        """
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": RANDOM_SEED,
            "verbosity": -1,
        }

        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))

        return rmse

    # Create study and optimize | 스터디 생성 및 최적화
    logger.info("Q6: Starting Optuna hyperparameter tuning...")
    logger.info("Q6: Optuna 하이퍼파라미터 튜닝 시작...")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Get best trial results | 최적 시도 결과 가져오기
    best_trial = study.best_trial
    best_params = best_trial.params
    best_params["random_state"] = RANDOM_SEED
    best_params["verbosity"] = -1

    # Train best model | 최적 모델 학습
    best_model = lgb.LGBMRegressor(**best_params)
    best_model.fit(X_train, y_train)

    y_pred_best = best_model.predict(X_val)
    rmse_best = np.sqrt(mean_squared_error(y_val, y_pred_best))

    logger.info("Q6: Optuna tuning completed successfully")
    logger.info("Q6: Optuna 튜닝 완료")

    print("\n[Optuna Tuning Results | Optuna 튜닝 결과]")
    print(f"Best Trial RMSE: {best_trial.value:.6f}")
    print("Best Parameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    print(f"\nOptuna Tuned LightGBM Validation RMSE: {rmse_best:.6f}")

    if rmse_best < 0.5:
        print("✓ Target achieved: RMSE < 0.5 | 목표 달성: RMSE < 0.5")
    else:
        print("✗ Target not achieved: RMSE >= 0.5 | 목표 미달성: RMSE >= 0.5")

    return best_model, rmse_best, best_params


# ==============================================================================
# Q7: Prepare Data for GRU Model
# Q7: GRU 모델을 위한 데이터 준비
# ==============================================================================
def prepare_gru_data(
    df: pd.DataFrame,
    sequence_length: int = SEQUENCE_LENGTH,
    train_ratio: float = 0.75,
) -> tuple[DataLoader, DataLoader, MinMaxScaler, int]:
    """
    Normalize OT and create sequences for GRU model.
    OT를 정규화하고 GRU 모델을 위한 시퀀스를 생성합니다.

    The sequence length is set to 24 hours = 96 time steps (15-min intervals).
    시퀀스 길이는 24시간 = 96 타임 스텝으로 설정됩니다 (15분 간격).

    Args:
        df: DataFrame with 'OT' column. | 'OT' 열이 있는 DataFrame.
        sequence_length: Length of input sequences (default: 96 for 24 hours).
                        입력 시퀀스 길이 (기본값: 24시간의 경우 96).
        train_ratio: Ratio of training data (default: 0.75).
                    훈련 데이터 비율 (기본값: 0.75).

    Returns:
        Tuple of (train_loader, val_loader, scaler, feature_dim).
        (train_loader, val_loader, scaler, feature_dim) 튜플.
    """
    df = df.copy()

    # Get feature columns (exclude date and target)
    # 특성 열 가져오기 (date와 타겟 제외)
    feature_cols = [col for col in df.columns if col not in ["date", "OT"]]
    feature_cols.append("OT")  # OT를 마지막에 추가 (타겟으로 사용)

    # Select relevant columns | 관련 열 선택
    data = df[feature_cols].values

    # Normalize data using MinMaxScaler | MinMaxScaler로 데이터 정규화
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Create sequences | 시퀀스 생성
    X_sequences: list[NDArray[np.float64]] = []
    y_sequences: list[float] = []

    for i in range(sequence_length, len(data_scaled)):
        # Input: sequence of all features | 입력: 모든 특성의 시퀀스
        X_sequences.append(data_scaled[i - sequence_length : i])
        # Output: OT value at current time (last column)
        # 출력: 현재 시점의 OT 값 (마지막 열)
        y_sequences.append(data_scaled[i, -1])

    X_array = np.array(X_sequences)
    y_array = np.array(y_sequences)

    # Split by time order | 시간 순서에 따라 분할
    split_idx = int(len(X_array) * train_ratio)

    X_train = X_array[:split_idx]
    X_val = X_array[split_idx:]
    y_train = y_array[:split_idx]
    y_val = y_array[split_idx:]

    # Convert to PyTorch tensors | PyTorch 텐서로 변환
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)

    # Create DataLoaders | DataLoader 생성
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    feature_dim = X_array.shape[2]  # Number of features | 특성 수

    logger.info("Q7: GRU data preparation completed")
    logger.info("Q7: GRU 데이터 준비 완료")
    print("\n[GRU Data Preparation | GRU 데이터 준비]")
    print(f"Sequence length: {sequence_length} (24 hours = {sequence_length} time steps)")
    print(f"Feature dimension: {feature_dim}")
    print(f"Training sequences: {len(X_train)}")
    print(f"Validation sequences: {len(X_val)}")
    print(f"Batch size: {batch_size}")

    return train_loader, val_loader, scaler, feature_dim


# ==============================================================================
# Q8: GRU Model Definition and Training
# Q8: GRU 모델 정의 및 학습
# ==============================================================================
class GRUModel(nn.Module):
    """
    GRU-based time series prediction model.
    GRU 기반 시계열 예측 모델.

    Architecture:
    - GRU layer with hidden_size=64, num_layers=2
    - Fully connected output layer

    Attributes:
        hidden_size: Number of hidden units in GRU. | GRU의 히든 유닛 수.
        num_layers: Number of GRU layers. | GRU 레이어 수.
        gru: GRU layer. | GRU 레이어.
        fc: Fully connected output layer. | 완전 연결 출력 레이어.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
    ) -> None:
        """
        Initialize GRU model.
        GRU 모델 초기화.

        Args:
            input_size: Number of input features. | 입력 특성 수.
            hidden_size: Number of hidden units (default: 64). | 히든 유닛 수 (기본값: 64).
            num_layers: Number of GRU layers (default: 2). | GRU 레이어 수 (기본값: 2).
            output_size: Number of output units (default: 1). | 출력 유닛 수 (기본값: 1).
            dropout: Dropout rate (default: 0.2). | 드롭아웃 비율 (기본값: 0.2).
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # GRU layer | GRU 레이어
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Fully connected layer | 완전 연결 레이어
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GRU model.
        GRU 모델의 순전파.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size).
               형상이 (batch, seq_len, input_size)인 입력 텐서.

        Returns:
            Output tensor of shape (batch, output_size).
            형상이 (batch, output_size)인 출력 텐서.
        """
        # GRU forward pass | GRU 순전파
        # out: (batch, seq_len, hidden_size)
        # h_n: (num_layers, batch, hidden_size)
        out, _ = self.gru(x)

        # Use last time step output | 마지막 타임 스텝 출력 사용
        out = self.fc(out[:, -1, :])

        return out


def train_gru_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    feature_dim: int,
    scaler: MinMaxScaler,
    epochs: int = 20,
) -> tuple[GRUModel, float]:
    """
    Train GRU model and calculate validation RMSE.
    GRU 모델을 학습하고 검증 RMSE를 계산합니다.

    Training settings (as specified in the problem):
    학습 설정 (문제에서 지정됨):
    - Epochs: 20
    - Loss function: MSELoss
    - Optimizer: Adam with lr=0.001

    Args:
        train_loader: Training data loader. | 훈련 데이터 로더.
        val_loader: Validation data loader. | 검증 데이터 로더.
        feature_dim: Number of input features. | 입력 특성 수.
        scaler: MinMaxScaler for inverse transformation. | 역변환용 MinMaxScaler.
        epochs: Number of training epochs (default: 20). | 학습 에포크 수 (기본값: 20).

    Returns:
        Tuple of (trained model, validation RMSE). | (학습된 모델, 검증 RMSE) 튜플.
    """
    # Set random seed for reproducibility | 재현성을 위한 랜덤 시드 설정
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)

    # Initialize model | 모델 초기화
    model = GRUModel(input_size=feature_dim).to(DEVICE)

    # Loss function and optimizer | 손실 함수 및 옵티마이저
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    logger.info("Q8: Starting GRU model training...")
    logger.info("Q8: GRU 모델 학습 시작...")

    # Training loop | 학습 루프
    for epoch in tqdm(range(epochs), desc="Training GRU"):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            # Forward pass | 순전파
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass | 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.6f}")

    # Validation | 검증
    model.eval()
    val_predictions: list[torch.Tensor] = []
    val_targets: list[torch.Tensor] = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(DEVICE)
            outputs = model(X_batch)
            val_predictions.append(outputs.cpu())
            val_targets.append(y_batch)

    # Concatenate predictions | 예측 결합
    val_pred_tensor = torch.cat(val_predictions).numpy()
    val_target_tensor = torch.cat(val_targets).numpy()

    # Inverse transform to get actual values | 실제 값으로 역변환
    # Create dummy array for inverse transform (OT is the last column)
    # 역변환을 위한 더미 배열 생성 (OT는 마지막 열)
    n_features = scaler.n_features_in_
    dummy_pred = np.zeros((len(val_pred_tensor), n_features))
    dummy_target = np.zeros((len(val_target_tensor), n_features))
    dummy_pred[:, -1] = val_pred_tensor.flatten()
    dummy_target[:, -1] = val_target_tensor.flatten()

    val_pred_actual = scaler.inverse_transform(dummy_pred)[:, -1]
    val_target_actual = scaler.inverse_transform(dummy_target)[:, -1]

    # Calculate RMSE | RMSE 계산
    rmse = np.sqrt(mean_squared_error(val_target_actual, val_pred_actual))

    logger.info("Q8: GRU model training completed")
    logger.info("Q8: GRU 모델 학습 완료")
    print("\n[GRU Model Results | GRU 모델 결과]")
    print(f"Epochs: {epochs}")
    print("Loss function: MSELoss")
    print("Optimizer: Adam (lr=0.001)")
    print(f"Validation RMSE: {rmse:.6f}")

    return model, rmse


# ==============================================================================
# Q9: Preprocess Test Data
# Q9: 테스트 데이터 전처리
# ==============================================================================
def preprocess_test_data(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply the same preprocessing to test data as was applied to training data.
    학습 데이터에 적용된 것과 동일한 전처리를 테스트 데이터에 적용합니다.

    Note: Since test data doesn't have OT column, we need to handle lag features
    differently. We'll use the last OT values from training data.
    참고: 테스트 데이터에는 OT 열이 없으므로 지연 특성을 다르게 처리해야 합니다.
    학습 데이터의 마지막 OT 값을 사용합니다.

    Args:
        test_df: Test DataFrame. | 테스트 DataFrame.
        train_df: Training DataFrame (for lag values). | 훈련 DataFrame (지연 값용).

    Returns:
        Preprocessed test DataFrame. | 전처리된 테스트 DataFrame.
    """
    test = test_df.copy()
    train = train_df.copy()

    # Q2: Create cyclic features | Q2: 순환 특성 생성
    test["date"] = pd.to_datetime(test["date"])

    test["hour"] = test["date"].dt.hour
    test["dayofweek"] = test["date"].dt.dayofweek
    test["month"] = test["date"].dt.month

    test["hour_sin"] = np.sin(2 * np.pi * test["hour"] / 24)
    test["hour_cos"] = np.cos(2 * np.pi * test["hour"] / 24)
    test["dayofweek_sin"] = np.sin(2 * np.pi * test["dayofweek"] / 7)
    test["dayofweek_cos"] = np.cos(2 * np.pi * test["dayofweek"] / 7)
    test["month_sin"] = np.sin(2 * np.pi * test["month"] / 12)
    test["month_cos"] = np.cos(2 * np.pi * test["month"] / 12)

    # Q3: For lag features, we need to use predictions or carry forward from train
    # Q3: 지연 특성의 경우, 예측값을 사용하거나 학습 데이터에서 이월해야 합니다
    # For initial test rows, use last values from training data
    # 초기 테스트 행의 경우, 학습 데이터의 마지막 값을 사용합니다

    # Initialize lag columns with NaN | NaN으로 지연 열 초기화
    test["OT_lag_1h"] = np.nan
    test["OT_lag_2h"] = np.nan
    test["OT_lag_3h"] = np.nan

    # Fill initial rows with training data's last OT values
    # 초기 행을 학습 데이터의 마지막 OT 값으로 채우기
    # For simplicity, we use the mean of recent OT values as a placeholder
    # 단순화를 위해 최근 OT 값의 평균을 플레이스홀더로 사용
    mean_recent_ot = train["OT"].tail(INTERVALS_PER_HOUR * 24).mean()
    test["OT_lag_1h"] = mean_recent_ot
    test["OT_lag_2h"] = mean_recent_ot
    test["OT_lag_3h"] = mean_recent_ot

    # Remove date column | date 열 제거
    test = test.drop(columns=["date"])

    logger.info("Q9: Test data preprocessing completed")
    logger.info("Q9: 테스트 데이터 전처리 완료")
    print("\n[Test Data Preprocessing | 테스트 데이터 전처리]")
    print(f"Test data shape after preprocessing: {test.shape}")
    print(f"Columns: {list(test.columns)}")

    return test


# ==============================================================================
# Q10: Ensemble Predictions
# Q10: 앙상블 예측
# ==============================================================================
def ensemble_predictions(
    lgb_model: lgb.LGBMRegressor,
    gru_model: GRUModel,
    X_val_lgb: pd.DataFrame,
    val_loader: DataLoader,
    y_val: pd.Series,
    scaler: MinMaxScaler,
) -> tuple[NDArray[np.float64], float]:
    """
    Ensemble LightGBM and GRU predictions using average.
    LightGBM과 GRU 예측을 평균을 사용하여 앙상블합니다.

    Args:
        lgb_model: Trained LightGBM model. | 학습된 LightGBM 모델.
        gru_model: Trained GRU model. | 학습된 GRU 모델.
        X_val_lgb: Validation features for LightGBM. | LightGBM용 검증 특성.
        val_loader: Validation data loader for GRU. | GRU용 검증 데이터 로더.
        y_val: Validation target values. | 검증 타겟 값.
        scaler: MinMaxScaler for GRU inverse transformation.
               GRU 역변환용 MinMaxScaler.

    Returns:
        Tuple of (ensemble predictions, ensemble RMSE).
        (앙상블 예측, 앙상블 RMSE) 튜플.
    """
    # LightGBM predictions | LightGBM 예측
    lgb_pred = lgb_model.predict(X_val_lgb)

    # GRU predictions | GRU 예측
    gru_model.eval()
    gru_predictions: list[torch.Tensor] = []

    with torch.no_grad():
        for X_batch, _ in val_loader:
            X_batch = X_batch.to(DEVICE)
            outputs = gru_model(X_batch)
            gru_predictions.append(outputs.cpu())

    gru_pred_scaled = torch.cat(gru_predictions).numpy().flatten()

    # Inverse transform GRU predictions | GRU 예측 역변환
    n_features = scaler.n_features_in_
    dummy = np.zeros((len(gru_pred_scaled), n_features))
    dummy[:, -1] = gru_pred_scaled
    gru_pred = scaler.inverse_transform(dummy)[:, -1]

    # Align predictions (GRU has fewer predictions due to sequence creation)
    # 예측 정렬 (GRU는 시퀀스 생성으로 인해 예측이 더 적음)
    min_len = min(len(lgb_pred), len(gru_pred))

    # Take last min_len predictions for alignment | 정렬을 위해 마지막 min_len 예측 사용
    lgb_pred_aligned = lgb_pred[-min_len:]
    gru_pred_aligned = gru_pred[-min_len:]
    y_val_aligned = y_val.values[-min_len:]

    # Ensemble: Average of two models | 앙상블: 두 모델의 평균
    ensemble_pred = (lgb_pred_aligned + gru_pred_aligned) / 2

    # Calculate individual and ensemble RMSE | 개별 및 앙상블 RMSE 계산
    lgb_rmse = np.sqrt(mean_squared_error(y_val_aligned, lgb_pred_aligned))
    gru_rmse = np.sqrt(mean_squared_error(y_val_aligned, gru_pred_aligned))
    ensemble_rmse = np.sqrt(mean_squared_error(y_val_aligned, ensemble_pred))

    logger.info("Q10: Ensemble predictions completed")
    logger.info("Q10: 앙상블 예측 완료")
    print("\n[Ensemble Results | 앙상블 결과]")
    print(f"LightGBM RMSE: {lgb_rmse:.6f}")
    print(f"GRU RMSE: {gru_rmse:.6f}")
    print(f"Ensemble RMSE (Average): {ensemble_rmse:.6f}")

    if ensemble_rmse < lgb_rmse and ensemble_rmse < gru_rmse:
        print("✓ Ensemble outperforms individual models | 앙상블이 개별 모델보다 우수함")
    else:
        print("Note: Ensemble did not outperform all individual models")
        print("참고: 앙상블이 모든 개별 모델보다 우수하지 않음")

    return ensemble_pred, ensemble_rmse


# ==============================================================================
# Main Execution
# ==============================================================================
def main() -> None:
    """
    Main function to execute all questions (Q1-Q10).
    모든 질문(Q1-Q10)을 실행하는 메인 함수.
    """
    print("=" * 70)
    print("AI Competency Assessment 2024 - Transformer OT Prediction")
    print("AI 역량 평가 2024 - 변압기 오일 온도 예측")
    print("=" * 70)

    # Define data paths | 데이터 경로 정의
    base_path = Path(__file__).parent / "dataset"
    train_path = base_path / "train.csv"
    test_path = base_path / "test.csv"

    # ========================================================================
    # Q1: Load and check data | Q1: 데이터 로드 및 확인
    # ========================================================================
    train, test = load_and_check_data(train_path, test_path)

    # ========================================================================
    # Q2: Create cyclic features | Q2: 순환 특성 생성
    # ========================================================================
    train["date"] = pd.to_datetime(train["date"])
    test["date"] = pd.to_datetime(test["date"])
    train = create_cyclic_features(train)

    # ========================================================================
    # Q3: Create lag features | Q3: 지연 특성 생성
    # ========================================================================
    train = create_lag_features(train)

    # ========================================================================
    # Q4: Prepare and split data | Q4: 데이터 준비 및 분할
    # ========================================================================
    X_train, X_val, y_train, y_val = prepare_and_split_data(train)

    # ========================================================================
    # Q5: Train LightGBM baseline | Q5: LightGBM 기준 모델 학습
    # ========================================================================
    lgb_baseline_model, baseline_rmse = train_lightgbm_baseline(X_train, X_val, y_train, y_val)

    # ========================================================================
    # Q6: Optuna hyperparameter tuning | Q6: Optuna 하이퍼파라미터 튜닝
    # ========================================================================
    best_lgb_model, best_lgb_rmse, best_params = optuna_lightgbm_tuning(
        X_train, X_val, y_train, y_val, n_trials=50
    )

    # ========================================================================
    # Q7 & Q8: GRU model | Q7 & Q8: GRU 모델
    # ========================================================================
    # Reload train data for GRU (need original data with date)
    # GRU를 위해 train 데이터 다시 로드 (date가 있는 원본 데이터 필요)
    train_gru = pd.read_csv(train_path, encoding="utf-8")
    train_gru["date"] = pd.to_datetime(train_gru["date"])
    train_gru = create_cyclic_features(train_gru)

    train_loader, val_loader, scaler, feature_dim = prepare_gru_data(train_gru)
    gru_model, gru_rmse = train_gru_model(train_loader, val_loader, feature_dim, scaler)

    # ========================================================================
    # Q9: Preprocess test data | Q9: 테스트 데이터 전처리
    # ========================================================================
    test_preprocessed = preprocess_test_data(test, train_gru)

    # ========================================================================
    # Q10: Ensemble predictions | Q10: 앙상블 예측
    # ========================================================================
    ensemble_pred, ensemble_rmse = ensemble_predictions(
        best_lgb_model, gru_model, X_val, val_loader, y_val, scaler
    )

    # ========================================================================
    # Final Summary | 최종 요약
    # ========================================================================
    print("\n" + "=" * 70)
    print("Final Summary | 최종 요약")
    print("=" * 70)
    print(f"Q5 - LightGBM Baseline RMSE: {baseline_rmse:.6f}")
    print(f"Q6 - Optuna Tuned LightGBM RMSE: {best_lgb_rmse:.6f}")
    print(f"Q8 - GRU Model RMSE: {gru_rmse:.6f}")
    print(f"Q10 - Ensemble RMSE: {ensemble_rmse:.6f}")
    print("=" * 70)

    # Make predictions on test data using ensemble method
    # 앙상블 방법을 사용하여 테스트 데이터에 대한 예측 수행
    test_pred_lgb = best_lgb_model.predict(test_preprocessed)
    print(f"\nTest predictions made: {len(test_pred_lgb)} samples")
    print(f"Test prediction range: [{test_pred_lgb.min():.4f}, {test_pred_lgb.max():.4f}]")


if __name__ == "__main__":
    main()
