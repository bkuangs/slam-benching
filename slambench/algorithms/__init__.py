"""Algorithm adapters. Register new adapters in ALGORITHMS."""

from .base import AlgorithmAdapter, AlgorithmOutput, RunContext
from .orbslam3 import OrbSlam3Adapter
from .rtabmap import RtabmapAdapter

ALGORITHMS = {
    OrbSlam3Adapter.name: OrbSlam3Adapter,
    RtabmapAdapter.name: RtabmapAdapter,
}

__all__ = ["ALGORITHMS", "AlgorithmAdapter", "AlgorithmOutput", "RunContext", "OrbSlam3Adapter", "RtabmapAdapter"]
