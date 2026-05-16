"""Reusable policy model architectures."""

from vantago.models.cnn import CnnPolicyNetwork, CnnPolicyNetworkError

__all__ = [
    "CnnPolicyNetwork",
    "CnnPolicyNetworkError",
]
