"""
GRU (Gated Recurrent Unit) Model.
GRU (게이트 순환 유닛) 모델.

This module implements a GRU network for sequence modeling tasks.
GRU is a simplified version of LSTM with fewer parameters and faster training.
이 모듈은 시퀀스 모델링 작업을 위한 GRU 네트워크를 구현합니다.
GRU는 LSTM의 단순화된 버전으로 파라미터가 더 적고 학습이 빠릅니다.

Architecture / 아키텍처:
    Input -> GRU (stacked) -> FC layers -> Output

Usage / 사용법:
    python -m ai_ml_programming_test_prob.models.gru
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from .base import count_parameters, get_device, logger, setup_logging


class GRUModel(nn.Module):
    """
    GRU (Gated Recurrent Unit) Network for sequence modeling.
    시퀀스 모델링을 위한 GRU (게이트 순환 유닛) 네트워크.

    Simplified version of LSTM with fewer parameters.
    파라미터가 더 적은 LSTM의 단순화된 버전입니다.

    Differences from LSTM / LSTM과의 차이점:
        - No separate cell state (only hidden state)
          별도의 셀 상태 없음 (은닉 상태만 존재)
        - Fewer gates (reset, update) vs LSTM (forget, input, output, cell)
          더 적은 게이트 (리셋, 업데이트) vs LSTM (망각, 입력, 출력, 셀)
        - Generally faster training with similar performance
          일반적으로 비슷한 성능으로 더 빠른 학습
        - ~25% fewer parameters than LSTM
          LSTM보다 약 25% 적은 파라미터

    GRU Gates / GRU 게이트:
        - Reset gate (r_t): Controls how much past information to forget
          리셋 게이트: 과거 정보를 얼마나 잊을지 제어
        - Update gate (z_t): Controls how much new information to add
          업데이트 게이트: 새로운 정보를 얼마나 추가할지 제어

    Architecture / 아키텍처:
        Input(seq) -> GRU(stacked layers) -> Last hidden -> FC -> Output

    Attributes:
        input_dim: Dimension of input features per time step
            시간 스텝당 입력 피처 차원
        hidden_dim: GRU hidden state dimension
            GRU 은닉 상태 차원
        num_layers: Number of stacked GRU layers
            스택된 GRU 레이어 수
        output_dim: Output dimension
            출력 차원

    Examples:
        >>> model = GRUModel(input_dim=10, hidden_dim=64, output_dim=5)
        >>> x = torch.randn(32, 50, 10)  # (batch, seq_len, features)
        >>> output, h_n = model(x)
        >>> print(output.shape)  # torch.Size([32, 5])
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
        task_type: Literal["regression", "classification"] = "regression",
    ) -> None:
        """
        Initialize the GRU model.
        GRU 모델을 초기화합니다.

        Args:
            input_dim: Number of input features per time step.
                시간 스텝당 입력 피처 수.
            hidden_dim: GRU hidden state dimension (default 64).
                GRU 은닉 상태 차원 (기본값 64).
            num_layers: Number of stacked GRU layers (default 2).
                스택된 GRU 레이어 수 (기본값 2).
            output_dim: Output dimension.
                출력 차원.
            dropout: Dropout probability between GRU layers (default 0.2).
                GRU 레이어 사이의 드롭아웃 확률 (기본값 0.2).
            bidirectional: Whether to use bidirectional GRU (default False).
                양방향 GRU 사용 여부 (기본값 False).
            task_type: Type of task ("regression" or "classification").
                작업 유형 ("regression" 또는 "classification").
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        self.task_type = task_type

        # Direction multiplier / 방향 배수
        self.num_directions = 2 if bidirectional else 1

        # GRU layer / GRU 레이어
        # Unlike LSTM, GRU only has hidden state (no cell state)
        # LSTM과 달리 GRU는 은닉 상태만 있음 (셀 상태 없음)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Fully connected layers / 완전 연결 레이어
        fc_input_dim = hidden_dim * self.num_directions
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        logger.info(
            "Created GRU: input=%d, hidden=%d, layers=%d, output=%d, bidirectional=%s",
            input_dim,
            hidden_dim,
            num_layers,
            output_dim,
            bidirectional,
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the GRU.
        GRU를 통한 순전파.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim).
                형상이 (batch_size, seq_len, input_dim)인 입력 텐서.
            hidden: Optional initial hidden state h_0.
                선택적 초기 은닉 상태 h_0.
                If None, initializes with zeros.
                None이면 0으로 초기화.

        Returns:
            Tuple of (output, h_n):
                - output: Final output of shape (batch_size, output_dim)
                  형상이 (batch_size, output_dim)인 최종 출력
                - h_n: Final hidden state of shape (num_layers * num_directions, batch, hidden_dim)
                  최종 은닉 상태

        Note:
            Unlike LSTM, GRU only returns hidden state (no cell state).
            LSTM과 달리 GRU는 은닉 상태만 반환합니다 (셀 상태 없음).
        """
        batch_size = x.size(0)

        # Initialize hidden state if not provided
        # 제공되지 않으면 은닉 상태 초기화
        if hidden is None:
            hidden = torch.zeros(
                self.num_layers * self.num_directions,
                batch_size,
                self.hidden_dim,
                device=x.device,
            )

        # GRU forward pass / GRU 순전파
        # gru_out shape: (batch_size, seq_len, hidden_dim * num_directions)
        gru_out, h_n = self.gru(x, hidden)

        # Use the last time step output / 마지막 시간 스텝 출력 사용
        last_output = gru_out[:, -1, :]

        # Fully connected layers / 완전 연결 레이어
        output = self.fc(last_output)

        return output, h_n


def main() -> None:
    """
    Demonstrate GRU model with sample sequence data.
    샘플 시퀀스 데이터로 GRU 모델을 시연합니다.

    This function shows how to:
    이 함수는 다음을 보여줍니다:
    1. Initialize the GRU model / GRU 모델 초기화
    2. Compare with LSTM (parameter count) / LSTM과 비교 (파라미터 수)
    3. Perform forward pass / 순전파 수행
    4. Access hidden states / 은닉 상태 접근
    """
    setup_logging()

    print("=" * 60)
    print("GRU (Gated Recurrent Unit) Demo / 게이트 순환 유닛 데모")
    print("=" * 60)

    # Get device / 장치 가져오기
    device = get_device()
    print(f"\nUsing device / 사용 장치: {device}")

    # Configuration / 설정
    batch_size = 32
    seq_len = 50
    input_dim = 10
    hidden_dim = 64
    num_layers = 2
    output_dim = 5  # Classification with 5 classes / 5개 클래스 분류

    print("\nConfiguration / 설정:")
    print(f"  - Sequence length / 시퀀스 길이: {seq_len}")
    print(f"  - Input dimension / 입력 차원: {input_dim}")
    print(f"  - Hidden dimension / 은닉 차원: {hidden_dim}")
    print(f"  - Number of layers / 레이어 수: {num_layers}")
    print(f"  - Output dimension / 출력 차원: {output_dim}")
    print(f"  - Batch size / 배치 크기: {batch_size}")

    # Create model / 모델 생성
    print("\n" + "-" * 60)
    print("Creating GRU model... / GRU 모델 생성 중...")
    print("-" * 60)

    model = GRUModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=output_dim,
        dropout=0.2,
        bidirectional=False,
        task_type="classification",
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
        output, h_n = model(x)

    print(f"Output shape / 출력 형상: {tuple(output.shape)}")
    print(f"Hidden state (h_n) shape / 은닉 상태 형상: {tuple(h_n.shape)}")
    print("  Note: GRU has no cell state (unlike LSTM)")
    print("  참고: GRU는 셀 상태가 없습니다 (LSTM과 다름)")

    # Model summary / 모델 요약
    print("\n" + "-" * 60)
    print("Model Summary / 모델 요약")
    print("-" * 60)
    print(f"Total parameters / 총 파라미터: {count_parameters(model):,}")

    # Compare GRU vs LSTM / GRU vs LSTM 비교
    print("\n" + "-" * 60)
    print("GRU vs LSTM Comparison / GRU vs LSTM 비교")
    print("-" * 60)

    # Create equivalent LSTM for comparison
    # 비교를 위한 동등한 LSTM 생성
    from .lstm import LSTMModel

    lstm_model = LSTMModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=output_dim,
        dropout=0.2,
        bidirectional=False,
    ).to(device)

    gru_params = count_parameters(model)
    lstm_params = count_parameters(lstm_model)
    reduction = (1 - gru_params / lstm_params) * 100

    print(f"GRU parameters / GRU 파라미터:   {gru_params:,}")
    print(f"LSTM parameters / LSTM 파라미터: {lstm_params:,}")
    print(f"Parameter reduction / 파라미터 감소: {reduction:.1f}%")

    print("\nKey differences / 주요 차이점:")
    print("  GRU:")
    print("    - 2 gates (reset, update) / 2개 게이트 (리셋, 업데이트)")
    print("    - 1 state (hidden) / 1개 상태 (은닉)")
    print("    - Faster training / 더 빠른 학습")
    print("  LSTM:")
    print("    - 3 gates (forget, input, output) / 3개 게이트 (망각, 입력, 출력)")
    print("    - 2 states (hidden, cell) / 2개 상태 (은닉, 셀)")
    print("    - Better for long sequences / 긴 시퀀스에 더 적합")

    # GRU gate explanation / GRU 게이트 설명
    print("\nGRU Gates / GRU 게이트:")
    print("  - Reset gate: r_t = σ(W_r · [h_{t-1}, x_t])")
    print("    리셋 게이트: 과거 은닉 상태를 얼마나 잊을지 결정")
    print("  - Update gate: z_t = σ(W_z · [h_{t-1}, x_t])")
    print("    업데이트 게이트: 새로운 후보 상태를 얼마나 반영할지 결정")

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


__all__ = ["GRUModel"]


if __name__ == "__main__":
    main()
