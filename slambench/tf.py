"""Minimal static transform tree (no ROS dependency)."""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from .trajectory import invert_T


class StaticTfTree:
    def __init__(self):
        # (parent, child) -> T_parent_child (pose of child in parent)
        self._edges: Dict[Tuple[str, str], np.ndarray] = {}
        self._adj: Dict[str, List[str]] = {}

    def add(self, parent: str, child: str, T_parent_child: np.ndarray) -> None:
        parent, child = parent.lstrip("/"), child.lstrip("/")
        self._edges[(parent, child)] = np.asarray(T_parent_child, dtype=float)
        self._adj.setdefault(parent, []).append(child)
        self._adj.setdefault(child, []).append(parent)

    @property
    def frames(self) -> List[str]:
        return sorted(self._adj)

    def path(self, a: str, b: str) -> Optional[List[str]]:
        a, b = a.lstrip("/"), b.lstrip("/")
        if a not in self._adj or b not in self._adj:
            return None
        prev = {a: None}
        q = deque([a])
        while q:
            n = q.popleft()
            if n == b:
                break
            for m in self._adj[n]:
                if m not in prev:
                    prev[m] = n
                    q.append(m)
        if b not in prev:
            return None
        out = [b]
        while prev[out[-1]] is not None:
            out.append(prev[out[-1]])
        return out[::-1]

    def lookup(self, a: str, b: str) -> Optional[np.ndarray]:
        """T_a_b: pose of frame b expressed in frame a."""
        p = self.path(a, b)
        if p is None:
            return None
        T = np.eye(4)
        for u, v in zip(p, p[1:]):
            if (u, v) in self._edges:
                T = T @ self._edges[(u, v)]
            else:
                T = T @ invert_T(self._edges[(v, u)])
        return T
