"""Algorithm adapters. Register new adapters in ALGORITHMS."""

from .base import AlgorithmAdapter, AlgorithmOutput, RunContext
from .fastlio import FastLioAdapter
from .orbslam3 import OrbSlam3Adapter
from .rtabmap import RtabmapAdapter

ALGORITHMS = {
    FastLioAdapter.name: FastLioAdapter,
    OrbSlam3Adapter.name: OrbSlam3Adapter,
    RtabmapAdapter.name: RtabmapAdapter,
}

__all__ = ["ALGORITHMS", "AlgorithmAdapter", "AlgorithmOutput", "RunContext", "FastLioAdapter", "OrbSlam3Adapter", "RtabmapAdapter"]
