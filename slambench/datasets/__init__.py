"""Dataset adapters. Register new adapters in DATASETS."""

from .base import DatasetAdapter
from .lunapolaris import LunaPolarisAdapter
from .omnilrs import OmniLrsAdapter
from .subtmrs import SubtMrsAdapter
from .tumvi import TumViAdapter

DATASETS = {
    LunaPolarisAdapter.name: LunaPolarisAdapter,
    OmniLrsAdapter.name: OmniLrsAdapter,
    TumViAdapter.name: TumViAdapter,
    SubtMrsAdapter.name: SubtMrsAdapter,
}

__all__ = ["DATASETS", "DatasetAdapter", "LunaPolarisAdapter", "OmniLrsAdapter", "SubtMrsAdapter", "TumViAdapter"]
