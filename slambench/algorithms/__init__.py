"""Algorithm adapters. Register new adapters in ALGORITHMS."""

from .base import AlgorithmAdapter, AlgorithmOutput, RunContext
from .fastlio import FastLioAdapter
from .lac import LacAdapter
from .orbslam3 import OrbSlam3Adapter
from .rtabmap import RtabmapAdapter
from .spslam3 import SpSlam3Adapter

ALGORITHMS = {
    FastLioAdapter.name: FastLioAdapter,
    LacAdapter.name: LacAdapter,
    OrbSlam3Adapter.name: OrbSlam3Adapter,
    RtabmapAdapter.name: RtabmapAdapter,
    SpSlam3Adapter.name: SpSlam3Adapter,
}

__all__ = ["ALGORITHMS", "AlgorithmAdapter", "AlgorithmOutput", "RunContext", "FastLioAdapter", "LacAdapter", "OrbSlam3Adapter", "RtabmapAdapter", "SpSlam3Adapter"]
