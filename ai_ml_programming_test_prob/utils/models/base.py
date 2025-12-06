"""
Base utilities for neural network models.
신경망 모델을 위한 기본 유틸리티.

This module provides common utility functions used across all model architectures.
이 모듈은 모든 모델 아키텍처에서 사용되는 공통 유틸리티 함수를 제공합니다.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

# ============================================================
# Logger Configuration / 로거 설정
# ============================================================
logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """
    Set up logging configuration for all model modules.
    모든 모델 모듈을 위한 로깅 설정을 구성합니다.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    모델의 학습 가능한 파라미터 수를 계산합니다.

    Args:
        model: PyTorch model.
            PyTorch 모델.

    Returns:
        Number of trainable parameters.
        학습 가능한 파라미터 수.

    Examples:
        >>> model = nn.Linear(10, 5)
        >>> params = count_parameters(model)
        >>> print(f"Parameters: {params}")
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> torch.device:
    """
    Get the best available device (CUDA, MPS, or CPU).
    사용 가능한 최적의 장치를 가져옵니다 (CUDA, MPS, 또는 CPU).

    Priority order / 우선순위:
    1. CUDA (NVIDIA GPU)
    2. MPS (Apple Silicon)
    3. CPU (fallback)

    Returns:
        torch.device: Best available device.
        가장 적합한 장치.

    Examples:
        >>> device = get_device()
        >>> model = model.to(device)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def print_model_summary(model: nn.Module, model_name: str) -> None:
    """
    Print a summary of the model architecture.
    모델 아키텍처의 요약을 출력합니다.

    Args:
        model: PyTorch model to summarize.
            요약할 PyTorch 모델.
        model_name: Name of the model for display.
            표시할 모델 이름.

    Examples:
        >>> model = nn.Linear(10, 5)
        >>> print_model_summary(model, "Linear")
    """
    print(f"\n{'=' * 60}")
    print(f"{model_name} Model Summary / 모델 요약")
    print(f"{'=' * 60}")
    print(f"Total parameters / 총 파라미터: {count_parameters(model):,}")
    print(f"{'=' * 60}\n")


__all__ = [
    "logger",
    "setup_logging",
    "count_parameters",
    "get_device",
    "print_model_summary",
]
