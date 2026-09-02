from __future__ import annotations

import time
from typing import Dict, Optional

from hetgat_hrl.alns.sequence import construct_k2_solution
from hetgat_hrl.alns.tabu_search import TabuSearchK2Planner, solution_to_goals


class VariableNeighborhoodSearchK2Planner(TabuSearchK2Planner):
    """Deterministic K2 variable-neighborhood-search baseline.

    Unlike ALNS it uses no destroy/repair operators or adaptive weights.  It
    cycles through relocate, swap, tail-replacement and reassignment
    neighborhoods, restarting from the first neighborhood after improvement.
    """

    neighborhoods = (
        "task_relocate",
        "task_swap",
        "tail_replacement",
        "agent_reassignment",
    )

    def _alns_optimize_goals(
        self, env, base_goals: Dict[str, Optional[str]]
    ) -> Dict[str, Optional[str]]:
        started = time.perf_counter()
        self.alns_diagnostics.replan_count += 1
        current_goals = self._repair_goals(env, dict(base_goals))
        current_solution = construct_k2_solution(env, current_goals)
        current_solution, current_eval, _ = self._objective_safe_k2_solution(
            env, current_solution
        )
        best_solution, best_eval = current_solution, current_eval
        k = 0

        for _ in range(int(self.alns_iterations)):
            self.alns_iteration_count_total += 1
            self.alns_diagnostics.iteration_count += 1
            move = str(self.neighborhoods[k])
            candidates = list(
                self._tabu_neighbors(
                    env, current_solution, allowed_moves={move}
                )
            )
            self.alns_diagnostics.repair_attempt_count += int(
                max(len(candidates), 1)
            )
            self.alns_diagnostics.repair_feasible_count += int(len(candidates))
            if not candidates:
                self.alns_diagnostics.noop_iteration_count += 1
                k = (k + 1) % len(self.neighborhoods)
                continue
            _name, _attrs, candidate, candidate_eval = min(
                candidates,
                key=lambda row: float(row[3].breakdown.total_cost),
            )
            if float(candidate_eval.breakdown.total_cost) < float(
                current_eval.breakdown.total_cost
            ) - 1e-12:
                current_solution, current_eval = candidate, candidate_eval
                self.alns_accepted_count_total += 1
                self.alns_improvement_count_total += 1
                self.alns_diagnostics.accepted_count += 1
                self.alns_diagnostics.accepted_improving_count += 1
                self.alns_diagnostics.improvement_count += 1
                if float(current_eval.breakdown.total_cost) < float(
                    best_eval.breakdown.total_cost
                ) - 1e-12:
                    best_solution, best_eval = current_solution, current_eval
                k = 0
            else:
                k = (k + 1) % len(self.neighborhoods)

        self.alns_diagnostics.wall_clock_time_s += float(
            time.perf_counter() - started
        )
        goals = solution_to_goals(best_solution)
        return {
            str(k): (None if v is None else str(v))
            for k, v in self._restore_protected_goals(
                env, current_goals, goals
            ).items()
        }
