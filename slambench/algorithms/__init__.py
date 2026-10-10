"""Algorithm adapters. Register new adapters in ALGORITHMS."""

from .base import AlgorithmAdapter, AlgorithmOutput, RunContext
from .orbslam3 import OrbSlam3Adapter

ALGORITHMS = {
    OrbSlam3Adapter.name: OrbSlam3Adapter,
}

__all__ = ["ALGORITHMS", "AlgorithmAdapter", "AlgorithmOutput", "RunContext", "OrbSlam3Adapter"]
