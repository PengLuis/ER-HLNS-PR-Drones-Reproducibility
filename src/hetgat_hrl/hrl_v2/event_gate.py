from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class EventImpact:
    event_type: str
    affected_agents: List[str] = field(default_factory=list)
    affected_tasks: List[str] = field(default_factory=list)
    impact_positive: bool = False
    reason: str = ""


class EventGate:
    def __init__(self):
        self.event_detected_count = 0
        self.event_impact_positive_count = 0
        self.event_impact_none_count = 0
        self.weak_event_suppressed_count = 0
        self.forced_switch_blocked_count = 0

    def classify_path_blocked(
        self,
        blocked_edge: Tuple[int, int],
        truck_remaining_paths: Dict[str, Iterable[int]],
    ) -> EventImpact:
        self.event_detected_count += 1
        edge = (int(blocked_edge[0]), int(blocked_edge[1]))
        rev = (edge[1], edge[0])
        affected: List[str] = []
        for aid, path in truck_remaining_paths.items():
            nodes = [int(x) for x in path]
            pairs = list(zip(nodes[:-1], nodes[1:]))
            if edge in pairs or rev in pairs:
                affected.append(str(aid))
        if affected:
            self.event_impact_positive_count += 1
            return EventImpact("path_blocked", affected_agents=affected, impact_positive=True, reason="edge_on_current_path")
        self.event_impact_none_count += 1
        return EventImpact("path_blocked", impact_positive=False, reason="edge_not_on_current_path")

    def suppress_weak_event(self, event_type: str) -> EventImpact:
        self.event_detected_count += 1
        self.weak_event_suppressed_count += 1
        self.event_impact_none_count += 1
        return EventImpact(str(event_type), impact_positive=False, reason="weak_event_suppressed")
