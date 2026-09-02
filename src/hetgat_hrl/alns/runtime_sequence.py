from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping

from hetgat_hrl.alns.solution import StableId


@dataclass(frozen=True)
class AgentSequenceRuntime:
    agent_id: StableId
    agent_type: str
    planned_sequence: tuple[StableId, ...]
    current_head_task: StableId | None
    tail_tasks: tuple[StableId, ...]
    solution_digest: str | None
    created_step: int
    last_validated_step: int
    source_replan_reason: str | None
    status: str
    tail_created_step: int | None = None

    def with_sequence(
        self,
        planned_sequence: Iterable[StableId],
        *,
        solution_digest: str | None = None,
        created_step: int | None = None,
        last_validated_step: int | None = None,
        source_replan_reason: str | None = None,
        status: str | None = None,
        tail_created_step: int | None = None,
    ) -> "AgentSequenceRuntime":
        seq = tuple(planned_sequence)[:2]
        head = seq[0] if seq else None
        tail = seq[1:] if len(seq) >= 2 else ()
        return AgentSequenceRuntime(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            planned_sequence=seq,
            current_head_task=head,
            tail_tasks=tail,
            solution_digest=self.solution_digest if solution_digest is None else solution_digest,
            created_step=self.created_step if created_step is None else int(created_step),
            last_validated_step=self.last_validated_step if last_validated_step is None else int(last_validated_step),
            source_replan_reason=self.source_replan_reason if source_replan_reason is None else source_replan_reason,
            status=self.status if status is None else status,
            tail_created_step=self.tail_created_step if tail_created_step is None else tail_created_step,
        )

    def to_stable_dict(self) -> dict[str, Any]:
        return {
            "agent_id": str(self.agent_id),
            "agent_type": str(self.agent_type),
            "planned_sequence": [str(x) for x in self.planned_sequence],
            "current_head_task": None if self.current_head_task is None else str(self.current_head_task),
            "tail_tasks": [str(x) for x in self.tail_tasks],
            "solution_digest": self.solution_digest,
            "created_step": int(self.created_step),
            "last_validated_step": int(self.last_validated_step),
            "source_replan_reason": self.source_replan_reason,
            "status": self.status,
            "tail_created_step": self.tail_created_step,
        }


@dataclass
class SequenceRuntimeState:
    by_agent: Dict[str, AgentSequenceRuntime] = field(default_factory=dict)
    tail_lifetime_steps: list[int] = field(default_factory=list)
    tail_invalidation_reason_counts: Dict[str, int] = field(default_factory=dict)
    tail_replacement_reason_counts: Dict[str, int] = field(default_factory=dict)

    def clear(self) -> None:
        self.by_agent.clear()
        self.tail_lifetime_steps.clear()
        self.tail_invalidation_reason_counts.clear()
        self.tail_replacement_reason_counts.clear()

    def runtime_for(self, agent_id: StableId) -> AgentSequenceRuntime | None:
        return self.by_agent.get(str(agent_id), None)

    def set_runtime(self, runtime: AgentSequenceRuntime) -> None:
        self.by_agent[str(runtime.agent_id)] = runtime

    def remove_runtime(self, agent_id: StableId) -> AgentSequenceRuntime | None:
        return self.by_agent.pop(str(agent_id), None)

    def export_rows(self) -> list[dict[str, Any]]:
        return [runtime.to_stable_dict() for _, runtime in sorted(self.by_agent.items(), key=lambda kv: kv[0])]

    def register_tail_lifetime(self, start_step: int | None, end_step: int) -> None:
        if start_step is None:
            return
        self.tail_lifetime_steps.append(int(max(int(end_step) - int(start_step), 0)))

    def increment_reason_count(self, counter_name: str, reason_codes: Iterable[str]) -> None:
        counter = (
            self.tail_invalidation_reason_counts
            if counter_name == "invalidation"
            else self.tail_replacement_reason_counts
        )
        for reason in reason_codes:
            counter[str(reason)] = int(counter.get(str(reason), 0) + 1)

    def summary_rows(self) -> list[dict[str, Any]]:
        lifetimes = list(self.tail_lifetime_steps)
        avg = float(sum(lifetimes) / max(len(lifetimes), 1))
        med = float(statistics.median(lifetimes)) if lifetimes else 0.0
        max_v = int(max(lifetimes)) if lifetimes else 0
        rows = [
            {
                "metric": "average_tail_lifetime_steps",
                "value": avg,
            },
            {
                "metric": "median_tail_lifetime_steps",
                "value": med,
            },
            {
                "metric": "maximum_tail_lifetime_steps",
                "value": max_v,
            },
        ]
        for reason, count in sorted(self.tail_invalidation_reason_counts.items()):
            rows.append({"metric": f"tail_invalidation_reason:{reason}", "value": int(count)})
        for reason, count in sorted(self.tail_replacement_reason_counts.items()):
            rows.append({"metric": f"tail_replacement_reason:{reason}", "value": int(count)})
        return rows


@dataclass(frozen=True)
class TailValidationResult:
    valid: bool
    reason_codes: tuple[str, ...]
    promotable: bool


def runtime_from_solution(
    *,
    agent_id: StableId,
    agent_type: str,
    sequence: Iterable[StableId],
    solution_digest: str | None,
    step: int,
    source_replan_reason: str | None,
    status: str,
    existing: AgentSequenceRuntime | None = None,
) -> AgentSequenceRuntime:
    seq = tuple(sequence)[:2]
    head = seq[0] if seq else None
    tail = seq[1:] if len(seq) >= 2 else ()
    return AgentSequenceRuntime(
        agent_id=agent_id,
        agent_type=str(agent_type),
        planned_sequence=seq,
        current_head_task=head,
        tail_tasks=tail,
        solution_digest=solution_digest,
        created_step=int(step if existing is None else existing.created_step),
        last_validated_step=int(step),
        source_replan_reason=source_replan_reason,
        status=str(status),
        tail_created_step=(
            int(step)
            if tail and (existing is None or tuple(existing.tail_tasks) != tail)
            else (existing.tail_created_step if existing is not None else None)
        ),
    )


def stable_sequence_map(
    rows: Mapping[StableId, Iterable[StableId]] | Iterable[tuple[StableId, Iterable[StableId]]],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    items = rows.items() if isinstance(rows, Mapping) else rows
    normalized = [(str(agent_id), tuple(str(task_id) for task_id in sequence)) for agent_id, sequence in items]
    return tuple(sorted(normalized, key=lambda item: item[0]))
