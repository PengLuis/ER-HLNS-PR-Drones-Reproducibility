from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from hetgat_hrl.alns.objective import ObjectiveEvaluation, is_better
from hetgat_hrl.alns.solution import ALNSSolution, StableId


@dataclass(frozen=True)
class LocalSearchBudget:
    max_moves: int = 5
    max_exact_checks: int = 5
    max_time_ms: int = 20


@dataclass(frozen=True)
class LocalSearchResult:
    solution: ALNSSolution
    evaluation: ObjectiveEvaluation
    accepted_moves: int
    attempted_moves: int
    exact_checks: int
    runtime_ms: float
    ledger_rows: tuple[dict[str, Any], ...]


def _sequence_items(solution: ALNSSolution) -> list[tuple[StableId, str, tuple[StableId, ...]]]:
    rows = [(aid, "truck", tuple(seq)) for aid, seq in solution.truck_sequences]
    rows.extend((aid, "uav", tuple(seq)) for aid, seq in solution.uav_sequences)
    return rows


class LocalSearchRefinerV2:
    """Small-budget strict-improvement refiner for K2 solutions."""

    move_names = (
        "relocate",
        "swap",
        "tail_exchange",
        "support_binding_refinement",
        "recovery_anchor_refinement",
    )

    def __init__(self, budget: LocalSearchBudget | None = None, *, disabled_moves: Iterable[str] = ()) -> None:
        self.budget = budget or LocalSearchBudget()
        self.disabled_moves = frozenset(str(x).strip().lower() for x in disabled_moves if str(x).strip())

    def refine(
        self,
        solution: ALNSSolution,
        evaluation: ObjectiveEvaluation,
        *,
        evaluate: Callable[[ALNSSolution], ObjectiveEvaluation],
        exact_feasible: Callable[[ALNSSolution], bool],
    ) -> LocalSearchResult:
        start = time.perf_counter()
        current = solution
        current_eval = evaluation
        attempted = 0
        accepted = 0
        exact_checks = 0
        rows: list[dict[str, Any]] = []
        for move_type, candidate in self._candidate_moves(current):
            if str(move_type).lower() in self.disabled_moves:
                continue
            if attempted >= int(self.budget.max_moves) or exact_checks >= int(self.budget.max_exact_checks):
                break
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if elapsed_ms >= float(self.budget.max_time_ms):
                break
            attempted += 1
            exact_checks += 1
            feasible = bool(exact_feasible(candidate))
            candidate_eval = evaluate(candidate) if feasible else current_eval
            improved = bool(feasible and candidate_eval.feasible and is_better(candidate_eval, current_eval))
            row = {
                "move_type": str(move_type),
                "candidate_digest": candidate.digest(),
                "feasible": bool(feasible and candidate_eval.feasible),
                "objective_delta": float(candidate_eval.breakdown.total_cost - current_eval.breakdown.total_cost) if feasible else 0.0,
                "accepted": bool(improved),
                "exact_checks": int(exact_checks),
                "runtime_ms": float((time.perf_counter() - start) * 1000.0),
                "failure_reason": "" if feasible else "EXACT_FEASIBILITY_REJECT",
            }
            rows.append(row)
            if improved:
                current = candidate
                current_eval = candidate_eval
                accepted += 1
        return LocalSearchResult(
            solution=current,
            evaluation=current_eval,
            accepted_moves=int(accepted),
            attempted_moves=int(attempted),
            exact_checks=int(exact_checks),
            runtime_ms=float((time.perf_counter() - start) * 1000.0),
            ledger_rows=tuple(rows),
        )

    def _candidate_moves(self, solution: ALNSSolution) -> Iterable[tuple[str, ALNSSolution]]:
        items = _sequence_items(solution)
        for aid, atype, seq in items:
            if len(seq) >= 2:
                yield "relocate", solution.with_sequence(aid, atype, (seq[1], seq[0]))
                yield "tail_exchange", solution.with_sequence(aid, atype, (seq[0],))
        for i, (aid_a, atype_a, seq_a) in enumerate(items):
            for aid_b, atype_b, seq_b in items[i + 1:]:
                if not seq_a or not seq_b:
                    continue
                new_a = (seq_b[0],) + tuple(seq_a[1:])
                new_b = (seq_a[0],) + tuple(seq_b[1:])
                yield "swap", solution.with_sequence(aid_a, atype_a, new_a).with_sequence(aid_b, atype_b, new_b)
        yield "support_binding_refinement", ALNSSolution(
            truck_sequences=solution.truck_sequences,
            uav_sequences=solution.uav_sequences,
            support_bindings=solution.support_bindings,
            sortie_plans=solution.sortie_plans,
        )
        yield "recovery_anchor_refinement", ALNSSolution(
            truck_sequences=solution.truck_sequences,
            uav_sequences=solution.uav_sequences,
            support_bindings=solution.support_bindings,
            sortie_plans=solution.sortie_plans,
        )
