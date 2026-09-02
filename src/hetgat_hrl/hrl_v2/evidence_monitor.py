from __future__ import annotations

from collections import defaultdict
from typing import Dict


class EvidenceMonitor:
    def __init__(self):
        self._counts: Dict[str, float] = defaultdict(float)

    def inc(self, name: str, value: float = 1.0) -> None:
        self._counts[str(name)] += float(value)

    def set(self, name: str, value: float) -> None:
        self._counts[str(name)] = float(value)

    def get(self, name: str, default: float = 0.0) -> float:
        return float(self._counts.get(str(name), default))

    def to_metrics(self) -> Dict[str, float]:
        return dict(self._counts)
