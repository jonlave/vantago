"""Data APIs for supervised Go policy records."""

from vantago.data.artifacts import (
    ProcessedDatasetArtifact,
    ProcessedDatasetBuildResult,
    ProcessedDatasetError,
    ProcessedDatasetFailure,
    ProcessedDatasetInspection,
    ProcessedDatasetSkipCount,
    inspect_processed_dataset,
    load_processed_dataset,
    write_processed_dataset,
)
from vantago.data.encoding import (
    BoolArray,
    Float32Array,
    PositionEncodingError,
    PositionRecord,
    decode_label,
    encode_board_tensor,
    encode_label,
    encode_legal_mask,
    encode_replay_steps,
)

__all__ = [
    "BoolArray",
    "Float32Array",
    "ProcessedDatasetArtifact",
    "ProcessedDatasetBuildResult",
    "ProcessedDatasetError",
    "ProcessedDatasetFailure",
    "ProcessedDatasetInspection",
    "ProcessedDatasetSkipCount",
    "PositionEncodingError",
    "PositionRecord",
    "decode_label",
    "encode_board_tensor",
    "encode_label",
    "encode_legal_mask",
    "encode_replay_steps",
    "inspect_processed_dataset",
    "load_processed_dataset",
    "write_processed_dataset",
]
