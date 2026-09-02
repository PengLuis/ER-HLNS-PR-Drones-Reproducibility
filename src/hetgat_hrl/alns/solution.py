from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Tuple, Union

StableId = Union[int, str]
AgentType = str
SequencePairs = Tuple[Tuple[StableId, Tuple[StableId, ...]], ...]


def _stable_id_key(value: StableId) -> tuple[str, str]:
    return (type(value).__name__, str(value))


def _normalize_sequence_pairs(value: Mapping[StableId, Iterable[StableId]] | Iterable[tuple[StableId, Iterable[StableId]]]) -> SequencePairs:
    items = value.items() if isinstance(value, Mapping) else value
    normalized = []
    for agent_id, seq in items:
        normalized.append((agent_id, tuple(seq)))
    return tuple(sorted(normalized, key=lambda item: _stable_id_key(item[0])))


def _stable_payload(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_stable_payload(v) for v in value]
    if isinstance(value, list):
        return [_stable_payload(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _stable_payload(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "to_stable_dict"):
        return value.to_stable_dict()
    return str(value)


@dataclass(frozen=True)
class SupportBinding:
    __slots__ = ("uav_id", "truck_id", "task_id", "launch_anchor", "recovery_anchor")

    uav_id: StableId
    truck_id: StableId
    task_id: StableId
    launch_anchor: StableId
    recovery_anchor: StableId

    def to_stable_dict(self) -> dict[str, Any]:
        return {
            "launch_anchor": self.launch_anchor,
            "recovery_anchor": self.recovery_anchor,
            "task_id": self.task_id,
            "truck_id": self.truck_id,
            "uav_id": self.uav_id,
        }


@dataclass(frozen=True)
class SortiePlan:
    __slots__ = (
        "uav_id",
        "task_id",
        "launch_anchor",
        "recovery_anchor",
        "estimated_launch_step",
        "estimated_service_step",
        "estimated_recovery_step",
    )

    uav_id: StableId
    task_id: StableId
    launch_anchor: StableId
    recovery_anchor: StableId
    estimated_launch_step: Optional[int]
    estimated_service_step: Optional[int]
    estimated_recovery_step: Optional[int]

    def to_stable_dict(self) -> dict[str, Any]:
        return {
            "estimated_launch_step": self.estimated_launch_step,
            "estimated_recovery_step": self.estimated_recovery_step,
            "estimated_service_step": self.estimated_service_step,
            "launch_anchor": self.launch_anchor,
            "recovery_anchor": self.recovery_anchor,
            "task_id": self.task_id,
            "uav_id": self.uav_id,
        }


class ALNSSolution:
    __slots__ = ("truck_sequences", "uav_sequences", "support_bindings", "sortie_plans")

    def __init__(
        self,
        truck_sequences: Mapping[StableId, Iterable[StableId]] | SequencePairs,
        uav_sequences: Mapping[StableId, Iterable[StableId]] | SequencePairs,
        support_bindings: tuple[SupportBinding, ...] = (),
        sortie_plans: tuple[SortiePlan, ...] = (),
    ) -> None:
        object.__setattr__(self, "truck_sequences", _normalize_sequence_pairs(truck_sequences))
        object.__setattr__(self, "uav_sequences", _normalize_sequence_pairs(uav_sequences))
        object.__setattr__(
            self,
            "support_bindings",
            tuple(sorted(tuple(support_bindings), key=lambda b: json.dumps(b.to_stable_dict(), sort_keys=True))),
        )
        object.__setattr__(
            self,
            "sortie_plans",
            tuple(sorted(tuple(sortie_plans), key=lambda p: json.dumps(p.to_stable_dict(), sort_keys=True))),
        )

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("ALNSSolution is immutable")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ALNSSolution) and self.to_stable_dict() == other.to_stable_dict()

    def __hash__(self) -> int:
        return hash((self.truck_sequences, self.uav_sequences, self.support_bindings, self.sortie_plans))

    def sequence_for(self, agent_id: StableId, agent_type: AgentType) -> tuple[StableId, ...]:
        pairs = self.truck_sequences if str(agent_type).lower() == "truck" else self.uav_sequences
        for aid, seq in pairs:
            if aid == agent_id:
                return tuple(seq)
        return ()

    def first_task_for(self, agent_id: StableId, agent_type: AgentType) -> StableId | None:
        seq = self.sequence_for(agent_id, agent_type)
        return seq[0] if seq else None

    def with_sequence(self, agent_id: StableId, agent_type: AgentType, sequence: Iterable[StableId]) -> "ALNSSolution":
        truck = dict(self.truck_sequences)
        uav = dict(self.uav_sequences)
        target = truck if str(agent_type).lower() == "truck" else uav
        seq_tuple = tuple(sequence)
        if seq_tuple:
            target[agent_id] = seq_tuple
        else:
            target.pop(agent_id, None)
        return ALNSSolution(
            truck_sequences=truck,
            uav_sequences=uav,
            support_bindings=self.support_bindings,
            sortie_plans=self.sortie_plans,
        )

    def without_task(self, task_id: StableId) -> "ALNSSolution":
        truck = {aid: tuple(x for x in seq if x != task_id) for aid, seq in self.truck_sequences}
        uav = {aid: tuple(x for x in seq if x != task_id) for aid, seq in self.uav_sequences}
        return ALNSSolution(
            truck_sequences={aid: seq for aid, seq in truck.items() if seq},
            uav_sequences={aid: seq for aid, seq in uav.items() if seq},
            support_bindings=tuple(b for b in self.support_bindings if b.task_id != task_id),
            sortie_plans=tuple(p for p in self.sortie_plans if p.task_id != task_id),
        )

    def to_stable_dict(self) -> dict[str, Any]:
        return {
            "sortie_plans": [p.to_stable_dict() for p in self.sortie_plans],
            "support_bindings": [b.to_stable_dict() for b in self.support_bindings],
            "truck_sequences": [[aid, list(seq)] for aid, seq in self.truck_sequences],
            "uav_sequences": [[aid, list(seq)] for aid, seq in self.uav_sequences],
        }

    def digest(self) -> str:
        text = json.dumps(_stable_payload(self.to_stable_dict()), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
