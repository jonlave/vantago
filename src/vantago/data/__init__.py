"""Data APIs for supervised Go policy records."""

from importlib import import_module
from typing import TYPE_CHECKING

from vantago.data.artifacts import (
    ProcessedDatasetArtifact,
    ProcessedDatasetBuildResult,
    ProcessedDatasetError,
    ProcessedDatasetFailure,
    ProcessedDatasetInspection,
    ProcessedDatasetMetadataArtifact,
    ProcessedDatasetSkipCount,
    inspect_processed_dataset,
    load_processed_dataset,
    load_processed_dataset_metadata,
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
from vantago.data.splits import (
    DatasetSplitBuildResult,
    DatasetSplitError,
    DatasetSplitManifest,
    load_dataset_split_manifest,
    write_dataset_split_manifest,
)

if TYPE_CHECKING:
    from vantago.data.torch_loading import (
        PolicyBatch,
        PolicyDatasetItem,
        PolicyMetadataBatch,
        ProcessedPolicyDataset,
        ProcessedPolicyMetadataDataset,
        load_policy_dataloaders,
        load_policy_dataset,
        load_policy_datasets,
        load_policy_metadata_datasets,
    )

_TORCH_LOADING_EXPORTS = frozenset(
    {
        "PolicyBatch",
        "PolicyDatasetItem",
        "PolicyMetadataBatch",
        "ProcessedPolicyDataset",
        "ProcessedPolicyMetadataDataset",
        "load_policy_dataloaders",
        "load_policy_dataset",
        "load_policy_datasets",
        "load_policy_metadata_datasets",
    }
)

__all__ = [
    "BoolArray",
    "DatasetSplitBuildResult",
    "DatasetSplitError",
    "DatasetSplitManifest",
    "Float32Array",
    "ProcessedDatasetArtifact",
    "ProcessedDatasetBuildResult",
    "ProcessedDatasetError",
    "ProcessedDatasetFailure",
    "ProcessedDatasetInspection",
    "ProcessedDatasetMetadataArtifact",
    "ProcessedDatasetSkipCount",
    "PositionEncodingError",
    "PositionRecord",
    "PolicyBatch",
    "PolicyDatasetItem",
    "PolicyMetadataBatch",
    "ProcessedPolicyDataset",
    "ProcessedPolicyMetadataDataset",
    "decode_label",
    "encode_board_tensor",
    "encode_label",
    "encode_legal_mask",
    "encode_replay_steps",
    "inspect_processed_dataset",
    "load_dataset_split_manifest",
    "load_policy_dataloaders",
    "load_policy_dataset",
    "load_policy_datasets",
    "load_policy_metadata_datasets",
    "load_processed_dataset",
    "load_processed_dataset_metadata",
    "write_dataset_split_manifest",
    "write_processed_dataset",
]


def __getattr__(name: str) -> object:
    if name in _TORCH_LOADING_EXPORTS:
        module = import_module("vantago.data.torch_loading")
        value = getattr(module, name)
        globals()[name] = value
        return value
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
