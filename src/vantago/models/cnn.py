"""Convolutional policy network for 19x19 Go board tensors."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

from vantago.data.encoding import CHANNEL_COUNT, SUPPORTED_LABEL_COUNT
from vantago.replay import SUPPORTED_BOARD_SIZE


class CnnPolicyNetworkError(ValueError):
    """Raised when the CNN policy network is misconfigured or misused."""


class CnnPolicyNetwork(nn.Module):
    """A small CNN that maps encoded board tensors to policy logits."""

    def __init__(self, *, hidden_channels: int = 64) -> None:
        super().__init__()
        if hidden_channels < 1:
            msg = f"hidden_channels must be positive, got {hidden_channels}"
            raise CnnPolicyNetworkError(msg)

        self.feature_extractor = nn.Sequential(
            _conv_block(CHANNEL_COUNT, hidden_channels),
            _conv_block(hidden_channels, hidden_channels),
            _conv_block(hidden_channels, hidden_channels),
        )
        self.policy_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_input_shape(x)
        features = self.feature_extractor(x)
        logits = self.policy_head(features)
        return cast(torch.Tensor, logits.flatten(start_dim=1))


def _conv_block(input_channels: int, output_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        ),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(),
    )


def _validate_input_shape(x: torch.Tensor) -> None:
    expected_shape = (
        CHANNEL_COUNT,
        SUPPORTED_BOARD_SIZE,
        SUPPORTED_BOARD_SIZE,
    )
    if x.ndim != 4 or tuple(x.shape[1:]) != expected_shape:
        msg = (
            "x must have shape "
            f"[batch, {CHANNEL_COUNT}, {SUPPORTED_BOARD_SIZE}, "
            f"{SUPPORTED_BOARD_SIZE}], got {tuple(x.shape)}"
        )
        raise CnnPolicyNetworkError(msg)
    if int(x.shape[0]) < 1:
        msg = f"batch dimension must be positive, got {int(x.shape[0])}"
        raise CnnPolicyNetworkError(msg)
    if SUPPORTED_BOARD_SIZE * SUPPORTED_BOARD_SIZE != SUPPORTED_LABEL_COUNT:
        msg = (
            "configured board size does not match supported label count: "
            f"{SUPPORTED_BOARD_SIZE}x{SUPPORTED_BOARD_SIZE} vs "
            f"{SUPPORTED_LABEL_COUNT}"
        )
        raise CnnPolicyNetworkError(msg)
