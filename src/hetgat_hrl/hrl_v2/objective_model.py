from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from hetgat_hrl.core.mdp_spec import TaskKind


@dataclass(frozen=True)
class ObjectiveChoice:
    task_id: Optional[str]
    reason: str
    score: float


class ObjectiveModel:
    def score_task(self, task, full_sortie_feasible: bool = False, near_completion: bool = False, continue_commitment: bool = False) -> float:
        if continue_commitment:
            return 1000.0
        if near_completion:
            return 800.0
        if getattr(task, "kind", None) == TaskKind.EMERGENCY:
            return 650.0 if full_sortie_feasible else -100.0
        return 300.0

    def choose_best(self, tasks: Iterable[object], *, prefer_emergency: bool = False) -> ObjectiveChoice:
        best_task = None
        best_score = float("-inf")
        best_reason = "low_cost"
        for task in tasks:
            is_tc = getattr(task, "kind", None) == TaskKind.EMERGENCY
            score = self.score_task(task, full_sortie_feasible=is_tc and prefer_emergency)
            if score > best_score:
                best_task = task
                best_score = score
                best_reason = "tc_delivery" if is_tc else "routine_completion"
        return ObjectiveChoice(
            task_id=None if best_task is None else str(getattr(best_task, "task_id", "")),
            reason=best_reason,
            score=float(best_score if best_task is not None else 0.0),
        )
