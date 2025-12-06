"""
LSTM (Long Short-Term Memory) Model.
LSTM (장단기 메모리) 모델.

This module implements an LSTM network for sequence modeling tasks such as
time series forecasting, sequence classification, and natural language processing.
이 모듈은 시계열 예측, 시퀀스 분류, 자연어 처리와 같은 시퀀스 모델링 작업을
위한 LSTM 네트워크를 구현합니다.

Architecture / 아키텍처:
    Input -> LSTM (stacked) -> FC layers -> Output

Usage / 사용법:
    python -m ai_ml_programming_test_prob.models.lstm
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from .base import count_parameters, get_device, logger, setup_logging


class LSTMModel(nn.Module):
    """
    LSTM (Long Short-Term Memory) Network for sequence modeling.
    시퀀스 모델링을 위한 LSTM (장단기 메모리) 네트워크.

    Designed for time series forecasting, sequence classification, and NLP tasks.
    시계열 예측, 시퀀스 분류, NLP 작업에 적합합니다.

    Key features of LSTM / LSTM의 핵심 특징:
        - Cell state (C_t): Long-term memory that flows through the network
          셀 상태 (C_t): 네트워크를 통해 흐르는 장기 기억
        - Forget gate: Controls what information to discard from cell state
          망각 게이트: 셀 상태에서 버릴 정보 제어
        - Input gate: Controls what new information to store in cell state
          입력 게이트: 셀 상태에 저장할 새 정보 제어
        - Output gate: Controls what information to output based on cell state
          출력 게이트: 셀 상태를 기반으로 출력할 정보 제어

    Architecture / 아키텍처:
        Input(seq) -> LSTM(stacked layers) -> Last hidden -> FC -> Output

    Attributes:
        input_dim: Dimension of input features per time step
            시간 스텝당 입력 피처 차원
        hidden_dim: LSTM hidden state dimension
            LSTM 은닉 상태 차원
        num_layers: Number of stacked LSTM layers
            스택된 LSTM 레이어 수
        output_dim: Output dimension
            출력 차원

    Examples:
        >>> model = LSTMModel(input_dim=10, hidden_dim=64, output_dim=1)
        >>> x = torch.randn(32, 50, 10)  # (batch, seq_len, features)
        >>> output, (h_n, c_n) = model(x)
        >>> print(output.shape)  # torch.Size([32, 1])
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
        Initialize the LSTM model.
        LSTM 모델을 초기화합니다.

        Args:
            input_dim: Number of input features per time step.
                시간 스텝당 입력 피처 수.
            hidden_dim: LSTM hidden state dimension (default 64).
                LSTM 은닉 상태 차원 (기본값 64).
            num_layers: Number of stacked LSTM layers (default 2).
                스택된 LSTM 레이어 수 (기본값 2).
            output_dim: Output dimension (1 for regression, num_classes for classification).
                출력 차원 (회귀는 1, 분류는 클래스 수).
            dropout: Dropout probability between LSTM layers (default 0.2).
                LSTM 레이어 사이의 드롭아웃 확률 (기본값 0.2).
            bidirectional: Whether to use bidirectional LSTM (default False).
                양방향 LSTM 사용 여부 (기본값 False).
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

        # Direction multiplier for hidden dim / 은닉 차원을 위한 방향 배수
        # Bidirectional LSTM has 2x hidden states (forward + backward)
        # 양방향 LSTM은 2배의 은닉 상태를 가짐 (순방향 + 역방향)
        self.num_directions = 2 if bidirectional else 1

        # LSTM layer / LSTM 레이어
        # batch_first=True: input shape (batch, seq_len, input_dim)
        # batch_first=True: 입력 형상 (batch, seq_len, input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,  # Dropout between layers only
            bidirectional=bidirectional,
        )

        # Fully connected layers after LSTM / LSTM 이후 완전 연결 레이어
        fc_input_dim = hidden_dim * self.num_directions
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        logger.info(
            "Created LSTM: input=%d, hidden=%d, layers=%d, output=%d, bidirectional=%s",
            input_dim,
            hidden_dim,
            num_layers,
            output_dim,
            bidirectional,
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the LSTM.
        LSTM을 통한 순전파.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim).
                형상이 (batch_size, seq_len, input_dim)인 입력 텐서.
            hidden: Optional initial hidden state (h_0, c_0).
                선택적 초기 은닉 상태 (h_0, c_0).
                If None, initializes with zeros.
                None이면 0으로 초기화.

        Returns:
            Tuple of (output, (h_n, c_n)):
                - output: Final output of shape (batch_size, output_dim)
                  형상이 (batch_size, output_dim)인 최종 출력
                - h_n: Final hidden state of shape (num_layers * num_directions, batch, hidden_dim)
                  최종 은닉 상태
                - c_n: Final cell state of shape (num_layers * num_directions, batch, hidden_dim)
                  최종 셀 상태
        """
        batch_size = x.size(0)

        # Initialize hidden state if not provided / 제공되지 않으면 은닉 상태 초기화
        if hidden is None:
            # h_0: hidden state / 은닉 상태
            h_0 = torch.zeros(
                self.num_layers * self.num_directions,
                batch_size,
                self.hidden_dim,
                device=x.device,
            )
            # c_0: cell state (long-term memory) / 셀 상태 (장기 기억)
            c_0 = torch.zeros(
                self.num_layers * self.num_directions,
                batch_size,
                self.hidden_dim,
                device=x.device,
            )
            hidden = (h_0, c_0)

        # LSTM forward pass / LSTM 순전파
        # lstm_out shape: (batch_size, seq_len, hidden_dim * num_directions)
        # lstm_out 형상: (batch_size, seq_len, hidden_dim * num_directions)
        lstm_out, (h_n, c_n) = self.lstm(x, hidden)

        # Use the last time step output / 마지막 시간 스텝 출력 사용
        # For bidirectional, this contains both forward and backward last outputs
        # 양방향의 경우, 순방향과 역방향 마지막 출력을 모두 포함
        last_output = lstm_out[:, -1, :]

        # Fully connected layers / 완전 연결 레이어
        output = self.fc(last_output)

        return output, (h_n, c_n)


def main() -> None:
    """
    Demonstrate LSTM model with sample sequence data.
    샘플 시퀀스 데이터로 LSTM 모델을 시연합니다.

    This function shows how to:
    이 함수는 다음을 보여줍니다:
    1. Initialize the LSTM model / LSTM 모델 초기화
    2. Create sample sequence tensors / 샘플 시퀀스 텐서 생성
    3. Perform forward pass / 순전파 수행
    4. Access hidden and cell states / 은닉 상태와 셀 상태 접근
    """
    setup_logging()

    print("=" * 60)
    print("LSTM (Long Short-Term Memory) Demo / 장단기 메모리 데모")
    print("=" * 60)

    # Get device / 장치 가져오기
    device = get_device()
    print(f"\nUsing device / 사용 장치: {device}")

    # Configuration / 설정
    batch_size = 32
    seq_len = 50  # Sequence length / 시퀀스 길이
    input_dim = 10  # Features per time step / 시간 스텝당 피처 수
    hidden_dim = 64  # LSTM hidden dimension / LSTM 은닉 차원
    num_layers = 2  # Stacked LSTM layers / 스택된 LSTM 레이어 수
    output_dim = 1  # Regression output / 회귀 출력

    print("\nConfiguration / 설정:")
    print(f"  - Sequence length / 시퀀스 길이: {seq_len}")
    print(f"  - Input dimension / 입력 차원: {input_dim}")
    print(f"  - Hidden dimension / 은닉 차원: {hidden_dim}")
    print(f"  - Number of layers / 레이어 수: {num_layers}")
    print("  - Bidirectional / 양방향: True")
    print(f"  - Batch size / 배치 크기: {batch_size}")

    # Create model (bidirectional) / 모델 생성 (양방향)
    print("\n" + "-" * 60)
    print("Creating Bidirectional LSTM model... / 양방향 LSTM 모델 생성 중...")
    print("-" * 60)

    model = LSTMModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=output_dim,
        dropout=0.2,
        bidirectional=True,
        task_type="regression",
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
        output, (h_n, c_n) = model(x)

    print(f"Output shape / 출력 형상: {tuple(output.shape)}")
    print(f"Hidden state (h_n) shape / 은닉 상태 형상: {tuple(h_n.shape)}")
    print(f"Cell state (c_n) shape / 셀 상태 형상: {tuple(c_n.shape)}")

    # Explain hidden state dimensions
    # 은닉 상태 차원 설명
    print("\nHidden state dimensions / 은닉 상태 차원 설명:")
    print("  - Shape: (num_layers * num_directions, batch, hidden_dim)")
    print("  - (2 * 2, 32, 64) = (4, 32, 64)")

    # Model summary / 모델 요약
    print("\n" + "-" * 60)
    print("Model Summary / 모델 요약")
    print("-" * 60)
    print(f"Total parameters / 총 파라미터: {count_parameters(model):,}")

    # LSTM gate explanation / LSTM 게이트 설명
    print("\nLSTM Gates / LSTM 게이트:")
    print("  - Forget gate: f_t = σ(W_f · [h_{t-1}, x_t] + b_f)")
    print("    망각 게이트: 이전 정보 중 버릴 것 결정")
    print("  - Input gate: i_t = σ(W_i · [h_{t-1}, x_t] + b_i)")
    print("    입력 게이트: 새로운 정보 중 저장할 것 결정")
    print("  - Output gate: o_t = σ(W_o · [h_{t-1}, x_t] + b_o)")
    print("    출력 게이트: 셀 상태 중 출력할 것 결정")

    # Example: Time series prediction
    # 예시: 시계열 예측
    print("\n" + "-" * 60)
    print("Example Prediction / 예측 예시")
    print("-" * 60)
    print(f"Predicted values (first 5) / 예측값 (처음 5개): {output[:5, 0].tolist()}")

    print("\n" + "=" * 60)
    print("Demo completed! / 데모 완료!")
    print("=" * 60)


__all__ = ["LSTMModel"]


if __name__ == "__main__":
    main()
