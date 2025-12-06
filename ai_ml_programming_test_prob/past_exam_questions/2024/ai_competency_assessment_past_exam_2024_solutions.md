# AI Competency Assessment 2024 - Solutions
# AI 역량 평가 2024 - 솔루션

---

## 문제 정의 | Problem Definition

주어진 ETDataset을 사용하여 전력 변압기의 **오일 온도(OT)**를 예측하는 문제를 풉니다.

주어진 데이터는 총 3개의 CSV 파일입니다. 각 CSV 파일에 대한 설명은 아래에 기술되어 있습니다.

각 시간대별로 예측한 OT와 실제 OT 사이의 RMSE(Root Mean Squared Error) 값을 성능 지표로 사용합니다.

해당 문제는 머신러닝 및 딥러닝 예측 모델을 만드는 과정을 코드로 구현하는 것을 평가합니다.

### 데이터 설명 | Data Description

1. **train.csv**
   - 학습에 사용되는 데이터
   - 기간: 2016년 7월 1일 0시 ~ 2017년 12월 31일 23시 45분
   - 열: date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT

2. **test.csv**
   - 예측에 사용되는 데이터
   - 기간: 2018년 1월 1일 0시 ~ 2018년 6월 30일 23시 45분
   - OT 열 제외, 나머지 열은 train.csv와 동일

3. **submission.csv**
   - 실제 예측값을 기록하는 파일
   - OT 열에 예측한 OT 값을 기록

---

## 패키지 설치 | Package Installation

```python
!pip install optuna
```

---

## 라이브러리 임포트 | Import Libraries

```python
# 사용할 라이브러리 불러오기 | Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.auto import tqdm
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 디바이스 설정 | Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 재현성을 위한 시드 설정 | Set seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# 데이터는 15분 간격으로 기록됨 | Data is recorded at 15-minute intervals
INTERVALS_PER_HOUR = 4

# 기본 데이터 경로 설정 (로컬 환경용) | Default data path (for local environment)
DATA_PATH = 'dataset/'
```

---

## Google Colab 환경 설정 | Google Colab Setup

```python
# Google Colab 환경 설정 | Google Colab environment setup
# 로컬 환경에서는 이 셀을 건너뛰세요 | Skip this cell in local environment

try:
    from google.colab import drive
    drive.mount('/content/drive')
    # Colab에서 데이터 경로 설정 | Set data path for Colab
    DATA_PATH = '/content/drive/MyDrive/your_path_here/dataset/'
    IN_COLAB = True
    print("Running in Google Colab")
except ImportError:
    DATA_PATH = 'dataset/'
    IN_COLAB = False
    print("Running in local environment")

print(f"Data path: {DATA_PATH}")
```

---

## Q1. 데이터 로드 및 결측치 확인 | Load Data and Check Missing Values

> train.csv와 test.csv를 불러오고, 각 데이터의 shape을 출력하세요. 또한, 결측치가 있는지 확인하고 각 열별로 결측치의 개수를 출력하세요.

```python
# A1. 데이터 로드 및 결측치 확인 | Load data and check missing values

# CSV 파일 로드 | Load CSV files
train = pd.read_csv(DATA_PATH + 'train.csv')
test = pd.read_csv(DATA_PATH + 'test.csv')

# 데이터 shape 출력 | Print data shapes
print("=" * 50)
print("[Data Shape | 데이터 Shape]")
print("=" * 50)
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# 결측치 확인 | Check missing values
print("\n" + "=" * 50)
print("[Train Missing Values | Train 결측치]")
print("=" * 50)
print(train.isnull().sum())

print("\n" + "=" * 50)
print("[Test Missing Values | Test 결측치]")
print("=" * 50)
print(test.isnull().sum())

# 결측치 요약 | Missing values summary
print("\n" + "=" * 50)
print("[Summary | 요약]")
print("=" * 50)
print(f"Total missing in train: {train.isnull().sum().sum()}")
print(f"Total missing in test: {test.isnull().sum().sum()}")
```

### 예상 출력 | Expected Output

```
==================================================
[Data Shape | 데이터 Shape]
==================================================
Train shape: (52704, 8)
Test shape: (16976, 7)

Total missing in train: 0
Total missing in test: 0
```

---

## Q2. 순환 특성 생성 | Create Cyclic Features

> 'date' 열을 사용하여 'hour', 'dayofweek', 'month' 특성을 생성하고, 'hour'와 'dayofweek'에 대해 sin과 cos 변환을 적용하여 cyclic feature를 만드세요.

```python
# A2. 순환 특성 생성 | Create cyclic features

# date 열을 datetime으로 변환 | Convert date column to datetime
train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])

# 시간 특성 추출 | Extract time features
train['hour'] = train['date'].dt.hour
train['dayofweek'] = train['date'].dt.dayofweek  # 0=Monday, 6=Sunday
train['month'] = train['date'].dt.month

test['hour'] = test['date'].dt.hour
test['dayofweek'] = test['date'].dt.dayofweek
test['month'] = test['date'].dt.month

# 순환 변환 적용 (sin/cos) | Apply cyclic transformation (sin/cos)
# hour: 24시간 주기 | 24-hour cycle
train['hour_sin'] = np.sin(2 * np.pi * train['hour'] / 24)
train['hour_cos'] = np.cos(2 * np.pi * train['hour'] / 24)
test['hour_sin'] = np.sin(2 * np.pi * test['hour'] / 24)
test['hour_cos'] = np.cos(2 * np.pi * test['hour'] / 24)

# dayofweek: 7일 주기 | 7-day cycle
train['dayofweek_sin'] = np.sin(2 * np.pi * train['dayofweek'] / 7)
train['dayofweek_cos'] = np.cos(2 * np.pi * train['dayofweek'] / 7)
test['dayofweek_sin'] = np.sin(2 * np.pi * test['dayofweek'] / 7)
test['dayofweek_cos'] = np.cos(2 * np.pi * test['dayofweek'] / 7)

# month: 12개월 주기 | 12-month cycle
train['month_sin'] = np.sin(2 * np.pi * train['month'] / 12)
train['month_cos'] = np.cos(2 * np.pi * train['month'] / 12)
test['month_sin'] = np.sin(2 * np.pi * test['month'] / 12)
test['month_cos'] = np.cos(2 * np.pi * test['month'] / 12)

print("[Cyclic Features Created | 생성된 순환 특성]")
print("- hour, hour_sin, hour_cos (24시간 주기)")
print("- dayofweek, dayofweek_sin, dayofweek_cos (7일 주기)")
print("- month, month_sin, month_cos (12개월 주기)")
print(f"\nTrain shape after feature engineering: {train.shape}")
print(f"Test shape after feature engineering: {test.shape}")
```

### 순환 변환 공식 | Cyclic Transformation Formula

$$
\text{feature\_sin} = \sin\left(\frac{2\pi \times \text{feature}}{\text{period}}\right)
$$

$$
\text{feature\_cos} = \cos\left(\frac{2\pi \times \text{feature}}{\text{period}}\right)
$$

---

## Q3. 지연 특성 생성 | Create Lag Features

> 'OT' 열에 대해 1시간 전, 2시간 전, 3시간 전의 값을 나타내는 lag 특성을 생성하세요.

```python
# A3. 지연 특성 생성 | Create lag features

# 데이터가 15분 간격이므로 1시간 = 4 스텝
# Data is at 15-minute intervals, so 1 hour = 4 steps
# 1시간 전 = shift(4), 2시간 전 = shift(8), 3시간 전 = shift(12)

train['OT_lag_1h'] = train['OT'].shift(INTERVALS_PER_HOUR * 1)  # 1시간 전 | 1 hour ago
train['OT_lag_2h'] = train['OT'].shift(INTERVALS_PER_HOUR * 2)  # 2시간 전 | 2 hours ago
train['OT_lag_3h'] = train['OT'].shift(INTERVALS_PER_HOUR * 3)  # 3시간 전 | 3 hours ago

print("[Lag Features Created | 생성된 지연 특성]")
print(f"- OT_lag_1h: OT value from 1 hour ago (shift={INTERVALS_PER_HOUR})")
print(f"- OT_lag_2h: OT value from 2 hours ago (shift={INTERVALS_PER_HOUR * 2})")
print(f"- OT_lag_3h: OT value from 3 hours ago (shift={INTERVALS_PER_HOUR * 3})")
print(f"\nNaN rows created by lag: {train['OT_lag_3h'].isna().sum()}")
print(f"Train shape: {train.shape}")
```

### 참고 | Note

- 데이터는 **15분 간격**으로 기록됨
- 1시간 = 4 타임 스텝
- `shift(4)` = 1시간 전, `shift(8)` = 2시간 전, `shift(12)` = 3시간 전

---

## Q4. 데이터 분할 | Split Data

> 불필요한 열인 'date'를 제거하고, 특성 행렬 X와 목표 변수 y를 생성하여 데이터를 시간 순서에 따라 3:1 비율로 훈련 세트와 검증 세트로 분할하세요.

```python
# A4. 데이터 준비 및 분할 | Prepare and split data

# date 열 제거 | Remove date column
train_processed = train.drop(columns=['date'])

# NaN 값이 있는 행 제거 (lag 특성으로 인해 발생)
# Drop rows with NaN values (caused by lag features)
initial_len = len(train_processed)
train_processed = train_processed.dropna()
dropped_rows = initial_len - len(train_processed)
print(f"Dropped {dropped_rows} rows with NaN values | NaN 행 {dropped_rows}개 제거")

# 특성 행렬 X와 목표 변수 y 생성 | Create feature matrix X and target variable y
y = train_processed['OT']
X = train_processed.drop(columns=['OT'])

# 시간 순서에 따라 3:1 비율로 분할 (시계열이므로 shuffle=False)
# Split by time order in 3:1 ratio (no shuffle for time series)
train_ratio = 0.75
split_idx = int(len(X) * train_ratio)

X_train = X.iloc[:split_idx]
X_val = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_val = y.iloc[split_idx:]

print("\n[Data Split | 데이터 분할]")
print(f"Training set size: {len(X_train)} ({train_ratio * 100:.0f}%)")
print(f"Validation set size: {len(X_val)} ({(1 - train_ratio) * 100:.0f}%)")
print(f"\nFeature columns ({len(X.columns)}): {list(X.columns)}")
```

### 중요 | Important

- 시계열 데이터이므로 **shuffle=False** (시간 순서 유지)
- 3:1 비율 = 75% 훈련, 25% 검증

---

## Q5. LightGBM 기준 모델 | LightGBM Baseline Model

> LightGBM을 사용하여 모델을 학습한 후, 검증 세트에 대한 RMSE를 계산하세요. 하이퍼파라미터는 num_leaves=31, n_estimators=100, learning_rate=0.05로 설정하세요.

```python
# A5. LightGBM 기준 모델 학습 | Train LightGBM baseline model

# 지정된 하이퍼파라미터로 모델 생성 | Create model with specified hyperparameters
lgb_model = lgb.LGBMRegressor(
    num_leaves=31,
    n_estimators=100,
    learning_rate=0.05,
    random_state=RANDOM_SEED,
    verbosity=-1
)

# 모델 학습 | Train model
lgb_model.fit(X_train, y_train)

# 예측 및 RMSE 계산 | Predict and calculate RMSE
y_pred = lgb_model.predict(X_val)
rmse_baseline = np.sqrt(mean_squared_error(y_val, y_pred))

print("[LightGBM Baseline Results | LightGBM 기준 결과]")
print(f"Hyperparameters: num_leaves=31, n_estimators=100, learning_rate=0.05")
print(f"Validation RMSE: {rmse_baseline:.6f}")
```

### 예상 결과 | Expected Result

```
Validation RMSE: ~0.77
```

---

## Q6. Optuna 하이퍼파라미터 튜닝 | Optuna Hyperparameter Tuning

> optuna를 사용하여 LightGBM의 하이퍼파라미터를 튜닝하고, 최적의 모델을 이용하여 검증 세트에 대한 RMSE를 계산하세요. RMSE를 0.5 이하로 낮추는 것을 목표로 합니다.

```python
# A6. Optuna 하이퍼파라미터 튜닝 | Optuna hyperparameter tuning

def objective(trial):
    """
    Optuna 목적 함수 | Optuna objective function
    LightGBM 하이퍼파라미터 최적화 | LightGBM hyperparameter optimization
    """
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': RANDOM_SEED,
        'verbosity': -1
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    return rmse

# Optuna 스터디 생성 및 최적화 | Create study and optimize
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print("\nBest Trial:")
trial = study.best_trial
print(f"  RMSE: {trial.value:.6f}")
print("  Best Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# 최적의 하이퍼파라미터로 모델 재학습 | Retrain model with best hyperparameters
best_params = trial.params
best_params['random_state'] = RANDOM_SEED
best_params['verbosity'] = -1

best_model = lgb.LGBMRegressor(**best_params)
best_model.fit(X_train, y_train)

y_pred_best = best_model.predict(X_val)
rmse_best = np.sqrt(mean_squared_error(y_val, y_pred_best))
print(f"\nOptuna Tuned LightGBM Validation RMSE: {rmse_best:.6f}")

if rmse_best < 0.5:
    print("✓ Target achieved: RMSE < 0.5 | 목표 달성: RMSE < 0.5")
else:
    print("✗ Target not achieved: RMSE >= 0.5 | 목표 미달성: RMSE >= 0.5")
```

### 튜닝 파라미터 범위 | Tuning Parameter Ranges

| Parameter | Range |
|-----------|-------|
| num_leaves | 20 ~ 150 |
| n_estimators | 100 ~ 500 |
| learning_rate | 0.01 ~ 0.3 (log scale) |
| max_depth | 3 ~ 15 |
| min_child_samples | 5 ~ 100 |
| subsample | 0.5 ~ 1.0 |
| colsample_bytree | 0.5 ~ 1.0 |
| reg_alpha | 1e-8 ~ 10.0 (log scale) |
| reg_lambda | 1e-8 ~ 10.0 (log scale) |

---

## Q7. GRU 데이터 준비 | Prepare GRU Data

> PyTorch를 사용하여 GRU 기반의 시계열 예측 모델을 구축하기 위해, OT 열을 정규화(Min-Max Scaling)하고 시계열 형태로 변환하세요. 입력 시퀀스의 길이는 24시간으로 설정하세요.

```python
# A7. GRU 데이터 준비 | Prepare data for GRU

# 시퀀스 길이: 24시간 = 96 타임 스텝 (15분 간격)
# Sequence length: 24 hours = 96 time steps (15-minute intervals)
SEQUENCE_LENGTH = 24 * INTERVALS_PER_HOUR  # 96

# 원본 train 데이터를 다시 로드하여 GRU용으로 사용
# Reload original train data for GRU (to avoid conflicts with modified data)
train_gru = pd.read_csv(DATA_PATH + 'train.csv')
train_gru['date'] = pd.to_datetime(train_gru['date'])

# 시간 특성 추출 | Extract time features
train_gru['hour'] = train_gru['date'].dt.hour
train_gru['dayofweek'] = train_gru['date'].dt.dayofweek
train_gru['month'] = train_gru['date'].dt.month

# 순환 변환 | Cyclic transformation
train_gru['hour_sin'] = np.sin(2 * np.pi * train_gru['hour'] / 24)
train_gru['hour_cos'] = np.cos(2 * np.pi * train_gru['hour'] / 24)
train_gru['dayofweek_sin'] = np.sin(2 * np.pi * train_gru['dayofweek'] / 7)
train_gru['dayofweek_cos'] = np.cos(2 * np.pi * train_gru['dayofweek'] / 7)
train_gru['month_sin'] = np.sin(2 * np.pi * train_gru['month'] / 12)
train_gru['month_cos'] = np.cos(2 * np.pi * train_gru['month'] / 12)

# 특성 열 선택 (date 제외, OT는 마지막에)
# Select feature columns (exclude date, OT at the end)
feature_cols = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL',
                'hour', 'dayofweek', 'month',
                'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos',
                'month_sin', 'month_cos', 'OT']

data = train_gru[feature_cols].values

# MinMaxScaler로 정규화 | Normalize with MinMaxScaler
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# 시퀀스 생성 | Create sequences
X_sequences = []
y_sequences = []

for i in range(SEQUENCE_LENGTH, len(data_scaled)):
    X_sequences.append(data_scaled[i - SEQUENCE_LENGTH:i])  # 입력: 24시간 시퀀스
    y_sequences.append(data_scaled[i, -1])  # 출력: 현재 OT 값 (마지막 열)

X_array = np.array(X_sequences)
y_array = np.array(y_sequences)

# 시간 순서에 따라 분할 | Split by time order
split_idx_gru = int(len(X_array) * 0.75)

X_train_gru = X_array[:split_idx_gru]
X_val_gru = X_array[split_idx_gru:]
y_train_gru = y_array[:split_idx_gru]
y_val_gru = y_array[split_idx_gru:]

# PyTorch 텐서로 변환 | Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_gru)
y_train_tensor = torch.FloatTensor(y_train_gru).unsqueeze(1)
X_val_tensor = torch.FloatTensor(X_val_gru)
y_val_tensor = torch.FloatTensor(y_val_gru).unsqueeze(1)

# DataLoader 생성 | Create DataLoaders
batch_size = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

feature_dim = X_array.shape[2]

print("[GRU Data Preparation | GRU 데이터 준비]")
print(f"Sequence length: {SEQUENCE_LENGTH} (24 hours = {SEQUENCE_LENGTH} time steps)")
print(f"Feature dimension: {feature_dim}")
print(f"Training sequences: {len(X_train_gru)}")
print(f"Validation sequences: {len(X_val_gru)}")
print(f"Batch size: {batch_size}")
```

### 시퀀스 구조 | Sequence Structure

- **입력 형상**: `(batch_size, sequence_length, features)` = `(64, 96, 16)`
- **출력 형상**: `(batch_size, 1)` = `(64, 1)`

---

## Q8. GRU 모델 정의 및 학습 | Define and Train GRU Model

> GRU 모델을 정의하고 학습한 후, 검증 세트에 대한 RMSE를 계산하세요. 에포크는 20번으로 설정하고, 손실 함수는 MSELoss, 옵티마이저는 Adam을 사용하세요.

```python
# A8. GRU 모델 정의 및 학습 | Define and train GRU model

# GRU 모델 정의 | GRU model definition
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        """
        GRU 기반 시계열 예측 모델 | GRU-based time series prediction model

        Args:
            input_size: 입력 특성 수 | Number of input features
            hidden_size: 히든 유닛 수 | Number of hidden units
            num_layers: GRU 레이어 수 | Number of GRU layers
            output_size: 출력 크기 | Output size
            dropout: 드롭아웃 비율 | Dropout rate
        """
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # GRU 순전파 | GRU forward pass
        out, _ = self.gru(x)
        # 마지막 타임 스텝 출력 사용 | Use last time step output
        out = self.fc(out[:, -1, :])
        return out

# 모델 초기화 | Initialize model
model = GRU(input_size=feature_dim).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"Model architecture:\n{model}")
print(f"\nDevice: {device}")

# 모델 학습 | Train model
epochs = 20
for epoch in tqdm(range(epochs), desc="Training GRU"):
    model.train()
    train_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # 순전파 | Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # 역전파 | Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.6f}")

# 모델 검증 | Validate model
model.eval()
val_predictions = []
val_targets = []

with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        val_predictions.append(outputs.cpu())
        val_targets.append(y_batch)

# 예측 결합 | Concatenate predictions
val_pred_tensor = torch.cat(val_predictions).numpy()
val_target_tensor = torch.cat(val_targets).numpy()

# 역변환하여 실제 값 얻기 | Inverse transform to get actual values
n_features = scaler.n_features_in_
dummy_pred = np.zeros((len(val_pred_tensor), n_features))
dummy_target = np.zeros((len(val_target_tensor), n_features))
dummy_pred[:, -1] = val_pred_tensor.flatten()
dummy_target[:, -1] = val_target_tensor.flatten()

val_pred_actual = scaler.inverse_transform(dummy_pred)[:, -1]
val_target_actual = scaler.inverse_transform(dummy_target)[:, -1]

# RMSE 계산 | Calculate RMSE
gru_rmse = np.sqrt(mean_squared_error(val_target_actual, val_pred_actual))

print(f"\n[GRU Model Results | GRU 모델 결과]")
print(f"Epochs: {epochs}")
print(f"Loss function: MSELoss")
print(f"Optimizer: Adam (lr=0.001)")
print(f"Validation RMSE: {gru_rmse:.6f}")
```

### GRU 모델 아키텍처 | GRU Model Architecture

```
GRU(
  (gru): GRU(16, 64, num_layers=2, batch_first=True, dropout=0.2)
  (fc): Linear(in_features=64, out_features=1, bias=True)
)
```

---

## Q9. 테스트 데이터 전처리 | Preprocess Test Data

> 전처리가 완료된 test 데이터를 생성하세요. train 데이터에서 사용했던 전처리를 동일하게 적용하세요.

```python
# A9. 테스트 데이터 전처리 | Preprocess test data

# 테스트 데이터에 동일한 전처리 적용 | Apply same preprocessing to test data
test_processed = test.copy()

# 이미 Q2에서 순환 특성이 생성되어 있음 | Cyclic features already created in Q2
# 지연 특성 처리 - 테스트 데이터에는 OT가 없으므로 train의 마지막 값 사용
# Lag features - use train's last OT values since test doesn't have OT

# train 데이터의 최근 OT 평균을 플레이스홀더로 사용
# Use mean of recent OT values from train as placeholder
mean_recent_ot = train['OT'].tail(INTERVALS_PER_HOUR * 24).mean()

test_processed['OT_lag_1h'] = mean_recent_ot
test_processed['OT_lag_2h'] = mean_recent_ot
test_processed['OT_lag_3h'] = mean_recent_ot

# date 열 제거 | Remove date column
test_processed = test_processed.drop(columns=['date'])

print("[Test Data Preprocessing | 테스트 데이터 전처리]")
print(f"Test data shape after preprocessing: {test_processed.shape}")
print(f"Columns: {list(test_processed.columns)}")

# X_train과 동일한 열 순서로 정렬 | Align column order with X_train
test_final = test_processed[X_train.columns]
print(f"\nFinal test shape (aligned with training): {test_final.shape}")
```

---

## Q10. 앙상블 예측 | Ensemble Predictions

> 마지막으로, LightGBM 모델과 GRU 모델의 예측값을 앙상블하여 검증 세트에 대한 RMSE를 계산하세요. 앙상블 방법으로 두 모델의 예측값의 평균을 사용하세요.

```python
# A10. 앙상블 예측 | Ensemble predictions

# LightGBM 예측 | LightGBM predictions
lgb_pred = best_model.predict(X_val)

# GRU 예측 (이미 Q8에서 계산됨) | GRU predictions (already computed in Q8)
# val_pred_actual 변수에 저장되어 있음

# 예측 정렬 (GRU는 시퀀스 생성으로 인해 예측이 더 적음)
# Align predictions (GRU has fewer predictions due to sequence creation)
min_len = min(len(lgb_pred), len(val_pred_actual))

# 마지막 min_len 예측 사용 | Use last min_len predictions
lgb_pred_aligned = lgb_pred[-min_len:]
gru_pred_aligned = val_pred_actual[-min_len:]
y_val_aligned = y_val.values[-min_len:]

# 앙상블: 두 모델의 평균 | Ensemble: Average of two models
ensemble_pred = (lgb_pred_aligned + gru_pred_aligned) / 2

# 개별 및 앙상블 RMSE 계산 | Calculate individual and ensemble RMSE
lgb_rmse_final = np.sqrt(mean_squared_error(y_val_aligned, lgb_pred_aligned))
gru_rmse_final = np.sqrt(mean_squared_error(y_val_aligned, gru_pred_aligned))
ensemble_rmse = np.sqrt(mean_squared_error(y_val_aligned, ensemble_pred))

print("[Ensemble Results | 앙상블 결과]")
print(f"LightGBM RMSE: {lgb_rmse_final:.6f}")
print(f"GRU RMSE: {gru_rmse_final:.6f}")
print(f"Ensemble RMSE (Average): {ensemble_rmse:.6f}")

if ensemble_rmse < lgb_rmse_final and ensemble_rmse < gru_rmse_final:
    print("\n✓ Ensemble outperforms individual models | 앙상블이 개별 모델보다 우수함")
else:
    print("\nNote: Ensemble did not outperform all individual models")
    print("참고: 앙상블이 모든 개별 모델보다 우수하지 않음")

# 최종 요약 | Final Summary
print("\n" + "=" * 60)
print("Final Summary | 최종 요약")
print("=" * 60)
print(f"Q5 - LightGBM Baseline RMSE: {rmse_baseline:.6f}")
print(f"Q6 - Optuna Tuned LightGBM RMSE: {rmse_best:.6f}")
print(f"Q8 - GRU Model RMSE: {gru_rmse:.6f}")
print(f"Q10 - Ensemble RMSE: {ensemble_rmse:.6f}")
print("=" * 60)
```

### 앙상블 공식 | Ensemble Formula

$$
\text{Ensemble Prediction} = \frac{\text{LightGBM Prediction} + \text{GRU Prediction}}{2}
$$

---

## 최종 결과 요약 | Final Results Summary

| Question | Task | Metric |
|----------|------|--------|
| Q1 | 데이터 로드 | Train: (52704, 8), Test: (16976, 7) |
| Q2 | 순환 특성 | hour/dayofweek/month + sin/cos |
| Q3 | 지연 특성 | OT_lag_1h, OT_lag_2h, OT_lag_3h |
| Q4 | 데이터 분할 | 75% train, 25% validation |
| Q5 | LightGBM Baseline | RMSE: ~0.77 |
| Q6 | Optuna Tuning | RMSE: ~0.74 |
| Q7 | GRU 데이터 준비 | Sequence length: 96 |
| Q8 | GRU 모델 | RMSE: varies |
| Q9 | 테스트 전처리 | Shape: (16976, 18) |
| Q10 | 앙상블 | Average of LightGBM + GRU |

---

## 주요 학습 포인트 | Key Learning Points

1. **순환 특성 (Cyclic Features)**: sin/cos 변환으로 시간적 주기성 표현
2. **지연 특성 (Lag Features)**: 과거 값을 특성으로 사용하여 시계열 패턴 학습
3. **시계열 분할**: shuffle 없이 시간 순서 유지
4. **Optuna**: 자동 하이퍼파라미터 최적화
5. **GRU**: 시퀀스 데이터를 위한 순환 신경망
6. **앙상블**: 다양한 모델의 예측을 결합하여 성능 향상
