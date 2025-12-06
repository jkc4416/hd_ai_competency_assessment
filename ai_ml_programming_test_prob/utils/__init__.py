"""
Utilities package for AI/ML programming tests.
AI/ML 프로그래밍 테스트를 위한 유틸리티 패키지.

This package contains:
이 패키지는 다음을 포함합니다:
    - models: Neural network architectures (MLP, CNN, LSTM, GRU, Transformer)
      모델: 신경망 아키텍처 (MLP, CNN, LSTM, GRU, Transformer)
    - preprocessing_utils: Data preprocessing utilities
      전처리 유틸리티: 데이터 전처리 유틸리티
"""

from .preprocessing_utils import (
    fill_missing_values,
    detect_outliers_iqr,
    detect_outliers_zscore,
    remove_outliers,
    cap_outliers,
    scale_features,
    encode_labels,
    encode_onehot,
    encode_target,
    create_polynomial_features,
    extract_datetime_features,
    create_lag_features,
    create_rolling_features,
    split_data,
    split_time_series,
    handle_imbalanced_data,
    preprocess_pipeline,
)

from .models import (
    MLP,
    CNN,
    LSTMModel,
    GRUModel,
    TransformerModel,
    PositionalEncoding,
    count_parameters,
    get_device,
    print_model_summary,
    setup_logging,
)

__all__ = [
    # Preprocessing utilities / 전처리 유틸리티
    "fill_missing_values",
    "detect_outliers_iqr",
    "detect_outliers_zscore",
    "remove_outliers",
    "cap_outliers",
    "scale_features",
    "encode_labels",
    "encode_onehot",
    "encode_target",
    "create_polynomial_features",
    "extract_datetime_features",
    "create_lag_features",
    "create_rolling_features",
    "split_data",
    "split_time_series",
    "handle_imbalanced_data",
    "preprocess_pipeline",
    # Models / 모델
    "MLP",
    "CNN",
    "LSTMModel",
    "GRUModel",
    "TransformerModel",
    "PositionalEncoding",
    # Model utilities / 모델 유틸리티
    "count_parameters",
    "get_device",
    "print_model_summary",
    "setup_logging",
]
