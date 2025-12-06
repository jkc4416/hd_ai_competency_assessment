"""
MLP (Multi-Layer Perceptron) Model.
MLP (다층 퍼셉트론) 모델.

This module implements a basic feedforward neural network with customizable
hidden layers, suitable for tabular data classification and regression tasks.
이 모듈은 테이블 데이터 분류 및 회귀 작업에 적합한 커스터마이징 가능한
은닉층을 가진 기본 순방향 신경망을 구현합니다.

Architecture / 아키텍처:
    Input -> [Linear -> BatchNorm -> Activation -> Dropout] x N -> Output

Usage / 사용법:
    python -m ai_ml_programming_test_prob.models.mlp
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from .base import count_parameters, get_device, logger, setup_logging


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (Fully Connected Neural Network).
    다층 퍼셉트론 (완전 연결 신경망).

    A basic feedforward neural network with customizable hidden layers.
    Use for tabular data classification/regression.
    커스터마이징 가능한 은닉층을 가진 기본 순방향 신경망입니다.
    테이블 데이터 분류/회귀에 사용합니다.

    Architecture / 아키텍처:
        Input -> [Linear -> BatchNorm -> ReLU -> Dropout] x N -> Output

    Attributes:
        input_dim: Number of input features / 입력 피처 수
        hidden_dims: List of hidden layer dimensions / 은닉층 차원 리스트
        output_dim: Number of output classes/values / 출력 클래스/값 수
        dropout: Dropout probability / 드롭아웃 확률

    Examples:
        >>> model = MLP(input_dim=20, hidden_dims=[128, 64], output_dim=10)
        >>> x = torch.randn(32, 20)
        >>> output = model(x)
        >>> print(output.shape)  # torch.Size([32, 10])
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float = 0.2,
        use_batch_norm: bool = True,
        activation: Literal["relu", "leaky_relu", "gelu"] = "relu",
    ) -> None:
        """
        Initialize the MLP model.
        MLP 모델을 초기화합니다.

        Args:
            input_dim: Dimension of input features.
                입력 피처의 차원.
            hidden_dims: List of hidden layer dimensions (e.g., [128, 64, 32]).
                은닉층 차원 리스트 (예: [128, 64, 32]).
            output_dim: Dimension of output (num_classes for classification).
                출력 차원 (분류의 경우 클래스 수).
            dropout: Dropout probability for regularization (default 0.2).
                정규화를 위한 드롭아웃 확률 (기본값 0.2).
            use_batch_norm: Whether to use batch normalization (default True).
                배치 정규화 사용 여부 (기본값 True).
            activation: Activation function type ("relu", "leaky_relu", "gelu").
                활성화 함수 종류 ("relu", "leaky_relu", "gelu").
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        # Build layers dynamically / 동적으로 레이어 구축
        layers: list[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            # Linear layer / 선형 레이어
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Batch normalization (optional) / 배치 정규화 (선택적)
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Activation function / 활성화 함수
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU(0.1))
            else:  # gelu
                layers.append(nn.GELU())

            # Dropout for regularization / 정규화를 위한 드롭아웃
            layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # Hidden layers / 은닉층
        self.hidden_layers = nn.Sequential(*layers)

        # Output layer (no activation, handled by loss function)
        # 출력층 (활성화 없음, 손실 함수에서 처리)
        self.output_layer = nn.Linear(prev_dim, output_dim)

        # Initialize weights / 가중치 초기화
        self._init_weights()

        logger.info(
            "Created MLP: input=%d, hidden=%s, output=%d",
            input_dim,
            hidden_dims,
            output_dim,
        )

    def _init_weights(self) -> None:
        """
        Initialize weights using Xavier/Glorot initialization.
        Xavier/Glorot 초기화를 사용하여 가중치를 초기화합니다.

        Xavier initialization helps maintain stable gradients during training
        by keeping the variance of activations consistent across layers.
        Xavier 초기화는 레이어 간 활성화의 분산을 일정하게 유지하여
        학습 중 안정적인 기울기를 유지하는 데 도움이 됩니다.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        네트워크를 통한 순전파.

        Args:
            x: Input tensor of shape (batch_size, input_dim).
                형상이 (batch_size, input_dim)인 입력 텐서.

        Returns:
            Output tensor of shape (batch_size, output_dim).
            형상이 (batch_size, output_dim)인 출력 텐서.
        """
        # Pass through hidden layers / 은닉층 통과
        x = self.hidden_layers(x)

        # Output layer / 출력층
        x = self.output_layer(x)

        return x


def main() -> None:
    """
    Demonstrate MLP model with sample data.
    샘플 데이터로 MLP 모델을 시연합니다.

    This function shows how to:
    이 함수는 다음을 보여줍니다:
    1. Initialize the MLP model / MLP 모델 초기화
    2. Create sample input tensor / 샘플 입력 텐서 생성
    3. Perform forward pass / 순전파 수행
    4. Check output shapes / 출력 형상 확인
    """
    setup_logging()

    print("=" * 60)
    print("MLP (Multi-Layer Perceptron) Demo / 다층 퍼셉트론 데모")
    print("=" * 60)

    # Get device / 장치 가져오기
    device = get_device()
    print(f"\nUsing device / 사용 장치: {device}")

    # Configuration / 설정
    batch_size = 32
    input_dim = 20  # Number of input features / 입력 피처 수
    hidden_dims = [128, 64, 32]  # Hidden layer dimensions / 은닉층 차원
    output_dim = 10  # Number of classes / 클래스 수

    print("\nConfiguration / 설정:")
    print(f"  - Input dimension / 입력 차원: {input_dim}")
    print(f"  - Hidden dimensions / 은닉층 차원: {hidden_dims}")
    print(f"  - Output dimension / 출력 차원: {output_dim}")
    print(f"  - Batch size / 배치 크기: {batch_size}")

    # Create model / 모델 생성
    print("\n" + "-" * 60)
    print("Creating MLP model... / MLP 모델 생성 중...")
    print("-" * 60)

    model = MLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        dropout=0.2,
        use_batch_norm=True,
        activation="relu",
    ).to(device)

    # Print model architecture / 모델 아키텍처 출력
    print("\nModel Architecture / 모델 아키텍처:")
    print(model)

    # Create sample input / 샘플 입력 생성
    print("\n" + "-" * 60)
    print("Forward Pass Demo / 순전파 데모")
    print("-" * 60)

    # Shape: (batch_size, input_dim) = (32, 20)
    x = torch.randn(batch_size, input_dim, device=device)
    print(f"\nInput shape / 입력 형상: {tuple(x.shape)}")

    # Forward pass / 순전파
    model.eval()  # Set to evaluation mode / 평가 모드로 설정
    with torch.no_grad():
        output = model(x)

    print(f"Output shape / 출력 형상: {tuple(output.shape)}")

    # Model summary / 모델 요약
    print("\n" + "-" * 60)
    print("Model Summary / 모델 요약")
    print("-" * 60)
    print(f"Total parameters / 총 파라미터: {count_parameters(model):,}")

    # Example: Classification prediction
    # 예시: 분류 예측
    print("\n" + "-" * 60)
    print("Example Prediction / 예측 예시")
    print("-" * 60)

    # Apply softmax for classification probabilities
    # 분류 확률을 위한 소프트맥스 적용
    probs = torch.softmax(output, dim=1)
    pred_classes = torch.argmax(probs, dim=1)

    print(f"Predicted classes (first 5) / 예측 클래스 (처음 5개): {pred_classes[:5].tolist()}")
    print(
        f"Max probabilities (first 5) / 최대 확률 (처음 5개): {probs.max(dim=1).values[:5].tolist()}"
    )

    print("\n" + "=" * 60)
    print("Demo completed! / 데모 완료!")
    print("=" * 60)


__all__ = ["MLP"]


if __name__ == "__main__":
    main()
