from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

from hetgat_hrl.core.mdp_spec import TaskKind, TaskStatus


@dataclass
class Commitment:
    agent_id: str
    task_id: str
    kind: str
    start_step: int
    status: str = "active"
    reason: str = ""


class CommitmentManager:
    def __init__(self):
        self._active: Dict[str, Commitment] = {}

    def hold(self, commitment: Commitment) -> None:
        self._active[str(commitment.agent_id)] = commitment

    def release(self, agent_id: str, reason: str = "") -> Optional[Commitment]:
        c = self._active.pop(str(agent_id), None)
        if c is not None:
            c.status = "released"
            c.reason = reason
        return c

    def expire(self, step: int, max_age_steps: int = 30) -> int:
        expired = 0
        for aid, c in list(self._active.items()):
            if int(step) - int(c.start_step) > int(max_age_steps):
                c.status = "expired"
                self._active.pop(aid, None)
                expired += 1
        return expired

    def abort(self, agent_id: str, reason: str = "") -> Optional[Commitment]:
        c = self._active.pop(str(agent_id), None)
        if c is not None:
            c.status = "aborted"
            c.reason = reason
        return c

    def active_for_agent(self, agent_id: str) -> Optional[Commitment]:
        return self._active.get(str(agent_id))

    @staticmethod
    def residual_followup_candidates(tasks: Iterable[object]) -> list[str]:
        out: list[str] = []
        for task in tasks:
            if getattr(task, "kind", None) != TaskKind.EMERGENCY:
                continue
            if getattr(task, "status", None) != TaskStatus.PENDING:
                continue
            if getattr(task, "first_service_step", None) is None:
                continue
            remaining = max(float(getattr(task, "remaining_demand_kg", 0.0)), float(getattr(task, "demand_left", 0.0)))
            if remaining > 1e-9:
                out.append(str(getattr(task, "task_id", "")))
        return out
