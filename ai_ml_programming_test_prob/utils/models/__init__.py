"""
Neural Network Models Package.
신경망 모델 패키지.

This package provides basic neural network architectures for deep learning:
이 패키지는 딥러닝을 위한 기본 신경망 아키텍처를 제공합니다:

Models / 모델:
    - MLP: Multi-Layer Perceptron for tabular data
      MLP: 테이블 데이터를 위한 다층 퍼셉트론
    - CNN: Convolutional Neural Network for image data
      CNN: 이미지 데이터를 위한 합성곱 신경망
    - LSTMModel: Long Short-Term Memory for sequences
      LSTMModel: 시퀀스를 위한 장단기 메모리
    - GRUModel: Gated Recurrent Unit for sequences
      GRUModel: 시퀀스를 위한 게이트 순환 유닛
    - TransformerModel: Attention-based architecture
      TransformerModel: 어텐션 기반 아키텍처

Utilities / 유틸리티:
    - count_parameters: Count trainable parameters
      count_parameters: 학습 가능한 파라미터 수 계산
    - get_device: Get best available device (CUDA/MPS/CPU)
      get_device: 최적의 장치 가져오기 (CUDA/MPS/CPU)

Usage / 사용법:
    >>> from ai_ml_programming_test_prob.models import MLP, CNN, LSTMModel, GRUModel, TransformerModel
    >>> model = MLP(input_dim=20, hidden_dims=[128, 64], output_dim=10)

Run individual model demos / 개별 모델 데모 실행:
    python -m ai_ml_programming_test_prob.models.mlp
    python -m ai_ml_programming_test_prob.models.cnn
    python -m ai_ml_programming_test_prob.models.lstm
    python -m ai_ml_programming_test_prob.models.gru
    python -m ai_ml_programming_test_prob.models.transformer
"""

# Utilities / 유틸리티
from .base import count_parameters, get_device, print_model_summary, setup_logging

# Models / 모델
from .cnn import CNN
from .gru import GRUModel
from .lstm import LSTMModel
from .mlp import MLP
from .transformer import PositionalEncoding, TransformerModel

__all__ = [
    # Utilities / 유틸리티
    "setup_logging",
    "count_parameters",
    "get_device",
    "print_model_summary",
    # Models / 모델
    "MLP",
    "CNN",
    "LSTMModel",
    "GRUModel",
    "PositionalEncoding",
    "TransformerModel",
]
