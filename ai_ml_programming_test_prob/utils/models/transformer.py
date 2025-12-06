"""
Transformer Model.
트랜스포머 모델.

This module implements a Transformer Encoder for sequence classification/regression.
Based on "Attention Is All You Need" (Vaswani et al., 2017).
이 모듈은 시퀀스 분류/회귀를 위한 트랜스포머 인코더를 구현합니다.
"Attention Is All You Need" (Vaswani et al., 2017) 기반.

Architecture / 아키텍처:
    Input -> Linear Projection -> Positional Encoding -> Transformer Encoder -> FC -> Output

Usage / 사용법:
    python -m ai_ml_programming_test_prob.models.transformer
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn

from .base import count_parameters, get_device, logger, setup_logging


class PositionalEncoding(nn.Module):
    """
    Positional Encoding for Transformer.
    트랜스포머를 위한 위치 인코딩.

    Adds positional information to input embeddings since Transformer
    has no inherent notion of sequence order (unlike RNNs).
    트랜스포머는 RNN과 달리 본질적으로 시퀀스 순서 개념이 없으므로
    입력 임베딩에 위치 정보를 추가합니다.

    Formula / 공식:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Why sin/cos? / 왜 sin/cos를 사용하는가?
        - Allows model to learn relative positions
          모델이 상대적 위치를 학습할 수 있게 함
        - PE(pos+k) can be represented as linear function of PE(pos)
          PE(pos+k)를 PE(pos)의 선형 함수로 표현 가능

    Attributes:
        d_model: Dimension of the model / 모델 차원
        max_len: Maximum sequence length / 최대 시퀀스 길이
        dropout: Dropout probability / 드롭아웃 확률
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        """
        Initialize positional encoding.
        위치 인코딩을 초기화합니다.

        Args:
            d_model: Dimension of the model / 모델 차원.
            max_len: Maximum sequence length (default 5000).
                최대 시퀀스 길이 (기본값 5000).
            dropout: Dropout probability (default 0.1).
                드롭아웃 확률 (기본값 0.1).
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix / 위치 인코딩 행렬 생성
        pe = torch.zeros(max_len, d_model)

        # Position indices: [0, 1, 2, ..., max_len-1]
        # 위치 인덱스: [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Division term for frequency scaling / 주파수 스케일링을 위한 나눗셈 항
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sin to even indices, cos to odd indices
        # 짝수 인덱스에 sin, 홀수 인덱스에 cos 적용
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension and register as buffer (not a parameter)
        # 배치 차원 추가 및 버퍼로 등록 (파라미터 아님)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        입력에 위치 인코딩을 추가합니다.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
                형상이 (batch_size, seq_len, d_model)인 입력 텐서.

        Returns:
            Tensor with positional encoding added.
            위치 인코딩이 추가된 텐서.
        """
        # self.pe is a registered buffer, cast to Tensor for type checker
        # self.pe는 등록된 버퍼이며, 타입 체커를 위해 Tensor로 캐스팅
        pe: torch.Tensor = self.pe  # type: ignore[assignment]
        x = x + pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """
    Transformer Encoder for sequence classification/regression.
    시퀀스 분류/회귀를 위한 트랜스포머 인코더.

    Based on "Attention Is All You Need" (Vaswani et al., 2017).
    "Attention Is All You Need" (Vaswani et al., 2017) 기반.

    Key Components / 핵심 구성 요소:
        1. Multi-Head Self-Attention / 멀티헤드 셀프 어텐션
           - Allows model to attend to different positions
             모델이 서로 다른 위치에 주의를 기울일 수 있게 함
           - Multiple attention heads capture different patterns
             여러 어텐션 헤드가 서로 다른 패턴 포착

        2. Position-wise Feed-Forward Networks / 위치별 순방향 네트워크
           - Two linear transformations with ReLU activation
             ReLU 활성화를 사용한 두 개의 선형 변환
           - Applied independently to each position
             각 위치에 독립적으로 적용

        3. Layer Normalization & Residual Connections / 레이어 정규화 및 잔차 연결
           - Stabilizes training / 학습 안정화
           - Enables deeper networks / 더 깊은 네트워크 가능

    Advantages over RNNs / RNN 대비 장점:
        - Parallel computation (no sequential dependency)
          병렬 계산 (순차적 의존성 없음)
        - Direct connections between any positions (no vanishing gradients)
          모든 위치 간 직접 연결 (기울기 소실 없음)
        - Better at capturing long-range dependencies
          장거리 의존성 포착에 우수

    Attributes:
        input_dim: Input feature dimension / 입력 피처 차원
        d_model: Transformer model dimension / 트랜스포머 모델 차원
        nhead: Number of attention heads / 어텐션 헤드 수
        num_layers: Number of encoder layers / 인코더 레이어 수
        output_dim: Output dimension / 출력 차원

    Examples:
        >>> model = TransformerModel(input_dim=10, d_model=64, nhead=4, output_dim=1)
        >>> x = torch.randn(32, 50, 10)  # (batch, seq_len, features)
        >>> output = model(x)
        >>> print(output.shape)  # torch.Size([32, 1])
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        output_dim: int = 1,
        dropout: float = 0.1,
        max_len: int = 5000,
        task_type: Literal["regression", "classification"] = "regression",
    ) -> None:
        """
        Initialize the Transformer model.
        트랜스포머 모델을 초기화합니다.

        Args:
            input_dim: Number of input features.
                입력 피처 수.
            d_model: Dimension of the transformer model (default 128).
                트랜스포머 모델 차원 (기본값 128).
            nhead: Number of attention heads (default 8). Must divide d_model evenly.
                어텐션 헤드 수 (기본값 8). d_model을 균등하게 나눠야 함.
            num_layers: Number of transformer encoder layers (default 4).
                트랜스포머 인코더 레이어 수 (기본값 4).
            dim_feedforward: Dimension of feedforward network (default 256).
                순방향 네트워크 차원 (기본값 256).
            output_dim: Output dimension.
                출력 차원.
            dropout: Dropout probability (default 0.1).
                드롭아웃 확률 (기본값 0.1).
            max_len: Maximum sequence length for positional encoding.
                위치 인코딩을 위한 최대 시퀀스 길이.
            task_type: Type of task ("regression" or "classification").
                작업 유형 ("regression" 또는 "classification").
        """
        super().__init__()

        # Validate nhead divides d_model / nhead가 d_model을 나누는지 검증
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.input_dim = input_dim
        self.d_model = d_model
        self.output_dim = output_dim
        self.task_type = task_type

        # Input projection (if input_dim != d_model)
        # 입력 투영 (input_dim != d_model인 경우)
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding / 위치 인코딩
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        # Transformer encoder layers / 트랜스포머 인코더 레이어
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",  # GELU activation (used in BERT, GPT)
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Output layers / 출력 레이어
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim),
        )

        # Initialize weights / 가중치 초기화
        self._init_weights()

        logger.info(
            "Created Transformer: input=%d, d_model=%d, nhead=%d, layers=%d, output=%d",
            input_dim,
            d_model,
            nhead,
            num_layers,
            output_dim,
        )

    def _init_weights(self) -> None:
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass through the Transformer.
        트랜스포머를 통한 순전파.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim).
                형상이 (batch_size, seq_len, input_dim)인 입력 텐서.
            src_mask: Optional attention mask of shape (seq_len, seq_len).
                형상이 (seq_len, seq_len)인 선택적 어텐션 마스크.
                Used for causal masking in autoregressive models.
                자기회귀 모델에서 인과 마스킹에 사용.
            src_key_padding_mask: Optional padding mask of shape (batch_size, seq_len).
                형상이 (batch_size, seq_len)인 선택적 패딩 마스크.
                True for positions that should be ignored (padding).
                무시해야 할 위치(패딩)에 True.

        Returns:
            Output tensor of shape (batch_size, output_dim).
            형상이 (batch_size, output_dim)인 출력 텐서.
        """
        # Project input to d_model dimension / d_model 차원으로 입력 투영
        x = self.input_projection(x)

        # Add positional encoding / 위치 인코딩 추가
        x = self.pos_encoder(x)

        # Pass through transformer encoder / 트랜스포머 인코더 통과
        x = self.transformer_encoder(x, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        # Global average pooling over sequence dimension
        # 시퀀스 차원에 대한 전역 평균 풀링
        # Alternative: Use [CLS] token or last position
        # 대안: [CLS] 토큰 또는 마지막 위치 사용
        x = x.mean(dim=1)

        # Output layers / 출력 레이어
        x = self.fc(x)

        return x


def main() -> None:
    """
    Demonstrate Transformer model with sample sequence data.
    샘플 시퀀스 데이터로 트랜스포머 모델을 시연합니다.

    This function shows how to:
    이 함수는 다음을 보여줍니다:
    1. Initialize the Transformer model / 트랜스포머 모델 초기화
    2. Understand attention mechanism / 어텐션 메커니즘 이해
    3. Perform forward pass / 순전파 수행
    4. Use masking for padding / 패딩을 위한 마스킹 사용
    """
    setup_logging()

    print("=" * 60)
    print("Transformer Demo / 트랜스포머 데모")
    print("=" * 60)

    # Get device / 장치 가져오기
    device = get_device()
    print(f"\nUsing device / 사용 장치: {device}")

    # Configuration / 설정
    batch_size = 32
    seq_len = 50
    input_dim = 10
    d_model = 64  # Must be divisible by nhead / nhead로 나눠져야 함
    nhead = 4  # Number of attention heads / 어텐션 헤드 수
    num_layers = 4
    output_dim = 1

    print("\nConfiguration / 설정:")
    print(f"  - Sequence length / 시퀀스 길이: {seq_len}")
    print(f"  - Input dimension / 입력 차원: {input_dim}")
    print(f"  - Model dimension / 모델 차원: {d_model}")
    print(f"  - Attention heads / 어텐션 헤드: {nhead}")
    print(f"  - Head dimension / 헤드 차원: {d_model // nhead}")
    print(f"  - Encoder layers / 인코더 레이어: {num_layers}")
    print(f"  - Batch size / 배치 크기: {batch_size}")

    # Create model / 모델 생성
    print("\n" + "-" * 60)
    print("Creating Transformer model... / 트랜스포머 모델 생성 중...")
    print("-" * 60)

    model = TransformerModel(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=128,
        output_dim=output_dim,
        dropout=0.1,
    ).to(device)

    # Print model architecture / 모델 아키텍처 출력
    print("\nModel Architecture / 모델 아키텍처:")
    print(model)

    # Create sample input / 샘플 입력 생성
    print("\n" + "-" * 60)
    print("Forward Pass Demo / 순전파 데모")
    print("-" * 60)

    # Shape: (batch_size, seq_len, input_dim) = (32, 50, 10)
    x = torch.randn(batch_size, seq_len, input_dim, device=device)
    print(f"\nInput shape / 입력 형상: {tuple(x.shape)}")
    print("  - batch_size=32, seq_len=50, input_dim=10")

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

    # Attention mechanism explanation / 어텐션 메커니즘 설명
    print("\n" + "-" * 60)
    print("Attention Mechanism / 어텐션 메커니즘")
    print("-" * 60)
    print("Self-Attention Formula / 셀프 어텐션 공식:")
    print("  Attention(Q, K, V) = softmax(QK^T / √d_k) · V")
    print("")
    print("Multi-Head Attention / 멀티헤드 어텐션:")
    print(f"  - Number of heads / 헤드 수: {nhead}")
    print(f"  - Head dimension / 헤드 차원: d_k = d_model / nhead = {d_model // nhead}")
    print("  - Each head learns different attention patterns")
    print("    각 헤드는 서로 다른 어텐션 패턴을 학습")

    # Transformer vs RNN comparison / Transformer vs RNN 비교
    print("\n" + "-" * 60)
    print("Transformer vs RNN / 트랜스포머 vs RNN 비교")
    print("-" * 60)
    print("Transformer advantages / 트랜스포머 장점:")
    print("  1. Parallel computation / 병렬 계산")
    print("     - All positions processed simultaneously")
    print("       모든 위치가 동시에 처리됨")
    print("  2. Long-range dependencies / 장거리 의존성")
    print("     - Direct connections between any positions")
    print("       모든 위치 간 직접 연결")
    print("  3. No vanishing gradients / 기울기 소실 없음")
    print("     - Path length O(1) vs O(n) for RNN")
    print("       경로 길이 O(1) vs RNN의 O(n)")

    # Example: Using padding mask / 예시: 패딩 마스크 사용
    print("\n" + "-" * 60)
    print("Padding Mask Example / 패딩 마스크 예시")
    print("-" * 60)

    # Create variable length sequences (some padded)
    # 가변 길이 시퀀스 생성 (일부 패딩됨)
    # True = ignore this position (padding)
    # True = 이 위치 무시 (패딩)
    padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    padding_mask[:, 40:] = True  # Last 10 positions are padding / 마지막 10개 위치가 패딩

    with torch.no_grad():
        output_with_mask = model(x, src_key_padding_mask=padding_mask)

    print(f"Output with padding mask / 패딩 마스크 적용 출력: {tuple(output_with_mask.shape)}")
    print("  Note: Padding positions are ignored in attention")
    print("  참고: 패딩 위치는 어텐션에서 무시됨")

    # Example prediction / 예시 예측
    print("\n" + "-" * 60)
    print("Example Prediction / 예측 예시")
    print("-" * 60)
    print(f"Predicted values (first 5) / 예측값 (처음 5개): {output[:5, 0].tolist()}")

    print("\n" + "=" * 60)
    print("Demo completed! / 데모 완료!")
    print("=" * 60)


__all__ = ["PositionalEncoding", "TransformerModel"]


if __name__ == "__main__":
    main()
