"""
CNN (Convolutional Neural Network) Model.
CNN (합성곱 신경망) 모델.

This module implements a basic CNN architecture for image classification tasks.
이 모듈은 이미지 분류 작업을 위한 기본 CNN 아키텍처를 구현합니다.

Architecture / 아키텍처:
    [Conv2d -> BatchNorm -> ReLU -> MaxPool -> Dropout] x N -> AdaptivePool -> FC layers

Usage / 사용법:
    python -m ai_ml_programming_test_prob.models.cnn
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import count_parameters, get_device, logger, setup_logging


class CNN(nn.Module):
    """
    Convolutional Neural Network for image classification.
    이미지 분류를 위한 합성곱 신경망.

    A basic CNN architecture with convolutional, pooling, and fully connected layers.
    Use for image classification tasks (e.g., MNIST, CIFAR-10).
    합성곱, 풀링, 완전 연결 레이어를 갖춘 기본 CNN 아키텍처입니다.
    이미지 분류 작업에 사용합니다 (예: MNIST, CIFAR-10).

    Architecture / 아키텍처:
        [Conv2d -> BatchNorm -> ReLU -> MaxPool] x N -> Flatten -> FC layers

    Key Concepts / 핵심 개념:
        - Convolution: Extracts local features using learnable filters
          합성곱: 학습 가능한 필터를 사용하여 지역 특징 추출
        - Pooling: Reduces spatial dimensions, provides translation invariance
          풀링: 공간 차원 축소, 이동 불변성 제공
        - FC layers: Combines features for final classification
          완전 연결 레이어: 최종 분류를 위해 특징 결합

    Attributes:
        in_channels: Number of input channels (1 for grayscale, 3 for RGB)
            입력 채널 수 (흑백 1, RGB 3)
        num_classes: Number of output classes / 출력 클래스 수
        conv_channels: List of convolutional channel dimensions
            합성곱 채널 차원 리스트

    Examples:
        >>> model = CNN(in_channels=3, num_classes=10)
        >>> x = torch.randn(32, 3, 32, 32)  # CIFAR-10 size
        >>> output = model(x)
        >>> print(output.shape)  # torch.Size([32, 10])
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        conv_channels: list[int] | None = None,
        fc_dims: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.25,
    ) -> None:
        """
        Initialize the CNN model.
        CNN 모델을 초기화합니다.

        Args:
            in_channels: Number of input image channels (1=grayscale, 3=RGB).
                입력 이미지 채널 수 (1=흑백, 3=RGB).
            num_classes: Number of output classes.
                출력 클래스 수.
            conv_channels: List of conv layer channel dimensions (default [32, 64, 128]).
                합성곱 레이어 채널 차원 리스트 (기본값 [32, 64, 128]).
            fc_dims: List of fully connected layer dimensions (default [256]).
                완전 연결 레이어 차원 리스트 (기본값 [256]).
            kernel_size: Kernel size for conv layers (default 3).
                합성곱 레이어 커널 크기 (기본값 3).
            dropout: Dropout probability (default 0.25).
                드롭아웃 확률 (기본값 0.25).
        """
        super().__init__()

        # Default channel configurations / 기본 채널 설정
        if conv_channels is None:
            conv_channels = [32, 64, 128]
        if fc_dims is None:
            fc_dims = [256]

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv_channels = conv_channels

        # Build convolutional layers / 합성곱 레이어 구축
        conv_layers: list[nn.Module] = []
        prev_channels = in_channels
        padding = kernel_size // 2  # Same padding / 동일 패딩

        for channels in conv_channels:
            # Conv block: Conv -> BatchNorm -> ReLU -> MaxPool -> Dropout
            # 합성곱 블록: 합성곱 -> 배치정규화 -> ReLU -> 최대풀링 -> 드롭아웃
            conv_layers.extend(
                [
                    # Convolution layer / 합성곱 레이어
                    nn.Conv2d(prev_channels, channels, kernel_size, padding=padding),
                    # Batch normalization for stable training / 안정적인 학습을 위한 배치 정규화
                    nn.BatchNorm2d(channels),
                    # ReLU activation / ReLU 활성화
                    nn.ReLU(inplace=True),
                    # Max pooling (halves spatial dimensions) / 최대 풀링 (공간 차원 절반으로 줄임)
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    # Dropout for regularization / 정규화를 위한 드롭아웃
                    nn.Dropout2d(dropout),
                ]
            )
            prev_channels = channels

        self.conv_layers = nn.Sequential(*conv_layers)

        # Adaptive pooling to fixed size (ensures compatibility with any input size)
        # 고정 크기로 적응적 풀링 (모든 입력 크기와 호환성 보장)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Calculate flattened dimension / 평탄화 차원 계산
        flatten_dim = conv_channels[-1] * 4 * 4

        # Build fully connected layers / 완전 연결 레이어 구축
        fc_layers: list[nn.Module] = []
        prev_dim = flatten_dim

        for fc_dim in fc_dims:
            fc_layers.extend(
                [
                    nn.Linear(prev_dim, fc_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = fc_dim

        self.fc_layers = nn.Sequential(*fc_layers)

        # Output layer / 출력층
        self.output_layer = nn.Linear(prev_dim, num_classes)

        # Initialize weights / 가중치 초기화
        self._init_weights()

        logger.info(
            "Created CNN: channels=%d, conv=%s, fc=%s, classes=%d",
            in_channels,
            conv_channels,
            fc_dims,
            num_classes,
        )

    def _init_weights(self) -> None:
        """
        Initialize weights using Kaiming initialization for conv layers.
        합성곱 레이어에 Kaiming 초기화를 사용하여 가중치를 초기화합니다.

        Kaiming initialization is optimal for ReLU activation functions,
        maintaining stable variance across layers.
        Kaiming 초기화는 ReLU 활성화 함수에 최적이며,
        레이어 간 안정적인 분산을 유지합니다.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Kaiming initialization for conv layers with ReLU
                # ReLU가 있는 합성곱 레이어를 위한 Kaiming 초기화
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.Linear):
                # Xavier initialization for fully connected layers
                # 완전 연결 레이어를 위한 Xavier 초기화
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN.
        CNN을 통한 순전파.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).
                형상이 (batch_size, channels, height, width)인 입력 텐서.

        Returns:
            Output tensor of shape (batch_size, num_classes).
            형상이 (batch_size, num_classes)인 출력 텐서.
        """
        # Convolutional layers / 합성곱 레이어
        x = self.conv_layers(x)

        # Adaptive pooling / 적응적 풀링
        x = self.adaptive_pool(x)

        # Flatten: (batch, channels, 4, 4) -> (batch, channels * 16)
        # 평탄화: (batch, channels, 4, 4) -> (batch, channels * 16)
        x = x.view(x.size(0), -1)

        # Fully connected layers / 완전 연결 레이어
        x = self.fc_layers(x)

        # Output / 출력
        x = self.output_layer(x)

        return x


def main() -> None:
    """
    Demonstrate CNN model with sample image data.
    샘플 이미지 데이터로 CNN 모델을 시연합니다.

    This function shows how to:
    이 함수는 다음을 보여줍니다:
    1. Initialize the CNN model / CNN 모델 초기화
    2. Create sample image tensors / 샘플 이미지 텐서 생성
    3. Perform forward pass / 순전파 수행
    4. Check output shapes / 출력 형상 확인
    """
    setup_logging()

    print("=" * 60)
    print("CNN (Convolutional Neural Network) Demo / 합성곱 신경망 데모")
    print("=" * 60)

    # Get device / 장치 가져오기
    device = get_device()
    print(f"\nUsing device / 사용 장치: {device}")

    # Configuration / 설정
    batch_size = 32
    in_channels = 3  # RGB image / RGB 이미지
    image_size = 32  # 32x32 image (CIFAR-10 size)
    num_classes = 10

    print("\nConfiguration / 설정:")
    print(f"  - Input channels / 입력 채널: {in_channels} (RGB)")
    print(f"  - Image size / 이미지 크기: {image_size}x{image_size}")
    print(f"  - Number of classes / 클래스 수: {num_classes}")
    print(f"  - Batch size / 배치 크기: {batch_size}")

    # Create model / 모델 생성
    print("\n" + "-" * 60)
    print("Creating CNN model... / CNN 모델 생성 중...")
    print("-" * 60)

    model = CNN(
        in_channels=in_channels,
        num_classes=num_classes,
        conv_channels=[32, 64, 128],
        fc_dims=[256],
        dropout=0.25,
    ).to(device)

    # Print model architecture / 모델 아키텍처 출력
    print("\nModel Architecture / 모델 아키텍처:")
    print(model)

    # Create sample input / 샘플 입력 생성
    print("\n" + "-" * 60)
    print("Forward Pass Demo / 순전파 데모")
    print("-" * 60)

    # Shape: (batch_size, channels, height, width) = (32, 3, 32, 32)
    x = torch.randn(batch_size, in_channels, image_size, image_size, device=device)
    print(f"\nInput shape / 입력 형상: {tuple(x.shape)}")
    print("  - batch_size=32, channels=3, height=32, width=32")

    # Forward pass / 순전파
    model.eval()
    with torch.no_grad():
        output = model(x)

    print(f"Output shape / 출력 형상: {tuple(output.shape)}")

    # Model summary / 모델 요약
    print("\n" + "-" * 60)
    print("Model Summary / 모델 요약")
    print("-" * 60)
    print(f"Total parameters / 총 파라미터: {count_parameters(model):,}")

    # Feature map dimensions through layers
    # 레이어별 특징 맵 차원
    print("\nFeature map progression / 특징 맵 변화:")
    print(f"  Input:  {in_channels} x {image_size} x {image_size}")
    print("  Conv1:  32 x 16 x 16 (after MaxPool)")
    print("  Conv2:  64 x 8 x 8 (after MaxPool)")
    print("  Conv3:  128 x 4 x 4 (after MaxPool)")
    print("  AdaptivePool: 128 x 4 x 4")
    print("  Flatten: 2048")
    print(f"  FC: 256 -> {num_classes}")

    # Example: Classification prediction
    # 예시: 분류 예측
    print("\n" + "-" * 60)
    print("Example Prediction / 예측 예시")
    print("-" * 60)

    probs = torch.softmax(output, dim=1)
    pred_classes = torch.argmax(probs, dim=1)

    print(f"Predicted classes (first 5) / 예측 클래스 (처음 5개): {pred_classes[:5].tolist()}")

    print("\n" + "=" * 60)
    print("Demo completed! / 데모 완료!")
    print("=" * 60)


__all__ = ["CNN"]


if __name__ == "__main__":
    main()
