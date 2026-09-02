from __future__ import annotations

from typing import Mapping, Optional

from hetgat_hrl.alns.solution import ALNSSolution, StableId


class SolutionAdapterContext:
    __slots__ = ("agent_types", "sequence_length")

    def __init__(self, agent_types: Mapping[StableId, str], sequence_length: int = 1) -> None:
        object.__setattr__(self, "agent_types", dict(agent_types))
        object.__setattr__(self, "sequence_length", int(sequence_length))
        if int(self.sequence_length) != 1:
            raise ValueError("legacy_k1 adapter requires sequence_length == 1")

    def __setattr__(self, name: str, value) -> None:
        raise AttributeError("SolutionAdapterContext is immutable")


def legacy_goals_to_solution(
    goals: Mapping[StableId, StableId | None],
    context: SolutionAdapterContext,
) -> ALNSSolution:
    truck_sequences: dict[StableId, tuple[StableId, ...]] = {}
    uav_sequences: dict[StableId, tuple[StableId, ...]] = {}
    for agent_id, goal_id in sorted(goals.items(), key=lambda kv: str(kv[0])):
        agent_type = str(context.agent_types.get(agent_id, "")).lower()
        if agent_type not in {"truck", "uav"}:
            continue
        seq: tuple[StableId, ...] = () if goal_id is None else (goal_id,)
        if agent_type == "truck":
            truck_sequences[agent_id] = seq
        else:
            uav_sequences[agent_id] = seq
    return ALNSSolution(truck_sequences=truck_sequences, uav_sequences=uav_sequences)


def solution_to_legacy_goals(solution: ALNSSolution) -> dict[StableId, Optional[StableId]]:
    goals: dict[StableId, Optional[StableId]] = {}
    for agent_id, seq in solution.truck_sequences:
        goals[agent_id] = seq[0] if seq else None
    for agent_id, seq in solution.uav_sequences:
        goals[agent_id] = seq[0] if seq else None
    return dict(sorted(goals.items(), key=lambda kv: str(kv[0])))


def env_adapter_context(env, sequence_length: int = 1) -> SolutionAdapterContext:
    agent_types = {}
    for aid, st in getattr(getattr(env, "state", None), "agents", {}).items():
        kind = getattr(getattr(st, "kind", None), "value", getattr(st, "kind", ""))
        agent_types[str(aid)] = str(kind).lower()
    return SolutionAdapterContext(agent_types=agent_types, sequence_length=int(sequence_length))
