"""Dataset adapters. Register new adapters in DATASETS."""

from .base import DatasetAdapter
from .lunapolaris import LunaPolarisAdapter
from .tumvi import TumViAdapter

DATASETS = {
    LunaPolarisAdapter.name: LunaPolarisAdapter,
    TumViAdapter.name: TumViAdapter,
}

__all__ = ["DATASETS", "DatasetAdapter", "LunaPolarisAdapter", "TumViAdapter"]
