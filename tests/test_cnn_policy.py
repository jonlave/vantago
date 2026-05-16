from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from vantago.models import CnnPolicyNetwork, CnnPolicyNetworkError

BOARD_SIZE = 19
CHANNEL_COUNT = 5
POINT_COUNT = BOARD_SIZE * BOARD_SIZE


def test_cnn_policy_network_default_maps_single_board_to_policy_logits() -> None:
    model = CnnPolicyNetwork()
    logits = model(torch.zeros((1, CHANNEL_COUNT, BOARD_SIZE, BOARD_SIZE)))

    assert logits.shape == (1, POINT_COUNT)
    assert torch.isfinite(logits).all()


def test_cnn_policy_network_maps_training_batch_to_policy_logits() -> None:
    model = CnnPolicyNetwork(hidden_channels=8)
    logits = model(torch.zeros((8, CHANNEL_COUNT, BOARD_SIZE, BOARD_SIZE)))

    assert logits.shape == (8, POINT_COUNT)
    assert torch.isfinite(logits).all()


def test_cnn_policy_network_uses_three_padded_conv_blocks_and_policy_head() -> None:
    model = CnnPolicyNetwork(hidden_channels=64)

    feature_blocks = list(model.feature_extractor.children())
    feature_convs: list[nn.Conv2d] = []

    assert len(feature_blocks) == 3
    for block in feature_blocks:
        assert isinstance(block, nn.Sequential)
        assert len(block) == 3
        conv, batch_norm, activation = block
        assert isinstance(conv, nn.Conv2d)
        assert isinstance(batch_norm, nn.BatchNorm2d)
        assert isinstance(activation, nn.ReLU)
        feature_convs.append(conv)

    assert len(feature_convs) == 3
    assert all(conv.kernel_size == (3, 3) for conv in feature_convs)
    assert all(conv.padding == (1, 1) for conv in feature_convs)
    assert all(conv.out_channels == 64 for conv in feature_convs)
    assert feature_convs[0].in_channels == CHANNEL_COUNT
    assert all(conv.in_channels == 64 for conv in feature_convs[1:])
    assert isinstance(model.policy_head, nn.Conv2d)
    assert model.policy_head.kernel_size == (1, 1)
    assert model.policy_head.out_channels == 1


def test_cnn_policy_network_rejects_invalid_hidden_channels() -> None:
    with pytest.raises(CnnPolicyNetworkError, match="hidden_channels"):
        CnnPolicyNetwork(hidden_channels=0)


@pytest.mark.parametrize(
    "shape",
    [
        (CHANNEL_COUNT, BOARD_SIZE, BOARD_SIZE),
        (1, CHANNEL_COUNT - 1, BOARD_SIZE, BOARD_SIZE),
        (1, CHANNEL_COUNT, BOARD_SIZE - 1, BOARD_SIZE),
        (1, CHANNEL_COUNT, BOARD_SIZE, BOARD_SIZE - 1),
        (0, CHANNEL_COUNT, BOARD_SIZE, BOARD_SIZE),
    ],
)
def test_cnn_policy_network_rejects_invalid_input_shapes(
    shape: tuple[int, ...],
) -> None:
    model = CnnPolicyNetwork(hidden_channels=8)

    with pytest.raises(CnnPolicyNetworkError, match="shape|batch"):
        model(torch.zeros(shape))
