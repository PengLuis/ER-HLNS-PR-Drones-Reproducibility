"""Minimal, contamination-auditable Dynamic Replanning ALNS baseline.

This module deliberately owns only a generic trigger and the canonical ALNS
operator pool.  It does not import or inherit the ER-HLNS/ER-ALNS planners.
The rolling planner remains the common assignment and execution guardrail;
UAV/truck physics are therefore supplied by the environment, not weakened by
this benchmark implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from hetgat_hrl.alns.canonical_operators import (
    greedy_insertion,
    random_removal,
    regret_2_insertion,
    regret_3_insertion,
    related_removal,
    sequence_segment_removal,
    worst_cost_removal,
)
from hetgat_hrl.alns.objective import should_accept_minimization
from hetgat_hrl.alns.sequence import construct_k2_solution, evaluate_k2_solution
from hetgat_hrl.alns.solution import ALNSSolution
from hetgat_hrl.core.mdp_spec import TaskStatus
from hetgat_hrl.hrl.rolling_planner import EventTriggeredRollingPlanner


CANONICAL_DESTROY_OPERATORS = (
    "random_removal",
    "worst_cost_removal",
    "related_removal",
    "sequence_segment_removal",
)
CANONICAL_REPAIR_OPERATORS = (
    "greedy_insertion",
    "regret_2_insertion",
    "regret_3_insertion",
)


@dataclass
class DynamicALNSDiagnostics:
    """Small metric surface consumed by the existing experiment runner."""

    replan_count: int = 0
    iteration_count: int = 0
    operator_attempt_count: int = 0
    objective_evaluation_count: int = 0
    feasibility_evaluation_count: int = 0
    accepted_count: int = 0
    improvement_count: int = 0
    destroyed_assignment_count: int = 0
    repair_attempt_count: int = 0
    repair_feasible_count: int = 0
    wall_clock_time_s: float = 0.0
    initial_destroy_weights: Dict[str, float] = field(default_factory=dict)
    initial_repair_weights: Dict[str, float] = field(default_factory=dict)
    final_destroy_weights: Dict[str, float] = field(default_factory=dict)
    final_repair_weights: Dict[str, float] = field(default_factory=dict)

    def to_flat_dict(self) -> Dict[str, float]:
        return {
            "alns_replan_count": float(self.replan_count),
            "alns_iteration_count": float(self.iteration_count),
            "alns_operator_attempt_count": float(self.operator_attempt_count),
            "alns_objective_evaluation_count": float(self.objective_evaluation_count),
            "alns_feasibility_evaluation_count": float(self.feasibility_evaluation_count),
            "alns_accepted_count": float(self.accepted_count),
            "alns_improvement_count": float(self.improvement_count),
            "alns_destroyed_assignment_count": float(self.destroyed_assignment_count),
            "alns_repair_attempt_count": float(self.repair_attempt_count),
            "alns_repair_feasible_count": float(self.repair_feasible_count),
            "alns_wall_clock_time_s": float(self.wall_clock_time_s),
            "alns_iterations_per_replan": float(self.iteration_count / max(self.replan_count, 1)),
        }


class DynamicReplanningALNSPlanner(EventTriggeredRollingPlanner):
    """Canonical ALNS rerun after a material dynamic-state change.

    The trigger intentionally has no ranking, severity, support, or critical
    task policy: only a changed current-goal feasibility, a newly pending task,
    or a committed goal becoming unreachable causes a replan.
    """

    algorithm_id = "dynamic_replanning_alns"
    contamination_features = {
        "er_destroy_operators": False,
        "er_repair_operators": False,
        "support_rebind": False,
        "critical_recovery_repair": False,
        "event_specific_adaptive_horizon": False,
        "er_event_ranking": False,
        "predictive_recovery_anchor": False,
    }

    def __init__(
        self,
        decision_interval: int = 5,
        seed: int = 0,
        iterations: int = 4,
        use_risk_term: bool = True,
        use_rth_repair: bool = True,
    ) -> None:
        # The inherited rolling assignment is the shared feasibility/safety
        # guardrail.  Its event policy is bypassed by _should_refresh below.
        super().__init__(
            decision_interval=decision_interval,
            seed=seed,
            use_risk_term=use_risk_term,
            use_rth_repair=use_rth_repair,
            use_event_trigger=False,
        )
        self.alns_iterations = int(max(iterations, 1))
        self.rng = np.random.default_rng(int(seed) + 15485863)
        self.alns_diagnostics = DynamicALNSDiagnostics(
            initial_destroy_weights={name: 1.0 for name in CANONICAL_DESTROY_OPERATORS},
            initial_repair_weights={name: 1.0 for name in CANONICAL_REPAIR_OPERATORS},
        )
        self.canonical_operator_records: List[Dict[str, Any]] = []
        self.dynamic_trigger_records: List[Dict[str, Any]] = []
        self._dynamic_last_road_version: Optional[int] = None
        self._dynamic_last_pending: Tuple[str, ...] = ()
        self._dynamic_last_goal_distances: Dict[str, float] = {}
        self._dynamic_last_goal_signature: Dict[str, Optional[str]] = {}

    def _episode_reset_if_needed(self, env) -> None:
        previous_step = int(getattr(self, "_last_seen_step", -1))
        super()._episode_reset_if_needed(env)
        if int(env.state.step_index) == 0 and previous_step > 0:
            self._dynamic_last_road_version = None
            self._dynamic_last_pending = ()
            self._dynamic_last_goal_distances = {}
            self._dynamic_last_goal_signature = {}

    @staticmethod
    def _pending_signature(env) -> Tuple[str, ...]:
        return tuple(sorted(
            str(t.task_id)
            for t in env.state.tasks.values()
            if t.status == TaskStatus.PENDING
        ))

    @staticmethod
    def _goal_distance(env, aid: str, gid: Optional[str]) -> float:
        if gid is None:
            return float("nan")
        task = env.state.tasks.get(str(gid))
        if task is not None:
            try:
                return float(env._agent_distance_to_task(str(aid), task))
            except Exception:
                return float("inf")
        # A virtual truck goal is a valid UAV recovery target; it is not a
        # task-feasibility trigger and is therefore represented as unknown.
        return float("nan")

    def _should_refresh(self, env) -> bool:
        step = int(env.state.step_index)
        road_version = int(getattr(env.topology, "_road_version", 0))
        pending = self._pending_signature(env)
        goals = dict(self.state.goals)
        distances = {
            str(aid): self._goal_distance(env, str(aid), gid)
            for aid, gid in goals.items()
            if gid is not None
        }
        if step == 0 and not self._dynamic_trigger_records_seen_initial():
            trigger, reason = "initial", "initial_plan"
        else:
            old_pending = set(self._dynamic_last_pending)
            new_pending = set(pending)
            released = sorted(new_pending - old_pending)
            unreachable = []
            for aid, gid in goals.items():
                if gid is None:
                    continue
                task = env.state.tasks.get(str(gid))
                if task is not None and task.status != TaskStatus.PENDING:
                    unreachable.append(aid)
                elif not math.isfinite(float(distances.get(aid, float("nan")))):
                    unreachable.append(aid)
            changed_goal = False
            if self._dynamic_last_road_version is not None and road_version != self._dynamic_last_road_version:
                for aid, distance in distances.items():
                    old = self._dynamic_last_goal_distances.get(aid, float("nan"))
                    if (math.isfinite(float(old)) and not math.isfinite(float(distance))) or (
                        math.isfinite(float(old)) and math.isfinite(float(distance))
                        and abs(float(distance) - float(old)) > 1e-6
                    ):
                        changed_goal = True
                        break
            if unreachable:
                trigger, reason = "goal_unreachable", "committed_goal_unreachable"
            elif released:
                trigger, reason = "task_release", "new_task_released"
            elif changed_goal:
                trigger, reason = "road_change", "blocked_road_changed_current_feasibility"
            elif step - int(self.state.step_last_refresh) >= int(self.decision_interval):
                # Generic fallback: bounded periodic check, without ranking or
                # event severity logic.  This keeps a completely quiet world
                # from holding stale assignments forever.
                trigger, reason = "interval_fallback", "dynamic_check_interval"
            else:
                trigger, reason = "none", ""
        self._dynamic_last_road_version = road_version
        self._dynamic_last_pending = pending
        self._dynamic_last_goal_distances = distances
        self._dynamic_last_goal_signature = goals
        self._last_refresh_flags = {"refresh": trigger != "none"}
        self._last_refresh_flags["dynamic_trigger"] = trigger
        self._last_refresh_flags["dynamic_trigger_reason"] = reason
        if trigger == "none":
            return False
        self._last_refresh_flags[f"dynamic_trigger_{trigger}"] = True
        self.dynamic_trigger_records.append({
            "step": step,
            "trigger_type": trigger,
            "reason": reason,
            "road_version": road_version,
            "pending_task_count": len(pending),
            "goal_count": sum(gid is not None for gid in goals.values()),
        })
        return True

    def _dynamic_trigger_records_seen_initial(self) -> bool:
        return bool(self.dynamic_trigger_records)

    @staticmethod
    def _destroy(name: str, env, solution: ALNSSolution, rng):
        return {
            "random_removal": random_removal,
            "worst_cost_removal": worst_cost_removal,
            "related_removal": related_removal,
            "sequence_segment_removal": sequence_segment_removal,
        }[name](env, solution, rng)

    @staticmethod
    def _repair(name: str, env, solution: ALNSSolution, removed: Iterable[Any], rng):
        return {
            "greedy_insertion": greedy_insertion,
            "regret_2_insertion": regret_2_insertion,
            "regret_3_insertion": regret_3_insertion,
        }[name](env, solution, removed, rng)

    def _plan_once(self, env) -> Dict[str, Optional[str]]:
        base_goals = super()._plan_once(env)
        current = construct_k2_solution(env, base_goals)
        current_eval = evaluate_k2_solution(env, current)
        self.alns_diagnostics.replan_count += 1
        self.alns_diagnostics.objective_evaluation_count += 1
        self.alns_diagnostics.feasibility_evaluation_count += 1
        best = current
        best_eval = current_eval
        evaluation_budget = int(
            max(
                getattr(
                    env.cfg,
                    "hrl_route_plan_alns_objective_evaluation_budget",
                    0,
                ),
                0,
            )
        )
        temperature = float(max(getattr(env.cfg, "alns_sa_initial_temperature", 1.0), 1e-6))
        start = __import__("time").perf_counter()
        for _ in range(self.alns_iterations):
            if (
                evaluation_budget > 0
                and self.alns_diagnostics.objective_evaluation_count >= evaluation_budget
            ):
                break
            destroy_name = CANONICAL_DESTROY_OPERATORS[int(self.rng.integers(0, len(CANONICAL_DESTROY_OPERATORS)))]
            repair_name = CANONICAL_REPAIR_OPERATORS[int(self.rng.integers(0, len(CANONICAL_REPAIR_OPERATORS)))]
            self.alns_diagnostics.iteration_count += 1
            self.alns_diagnostics.operator_attempt_count += 2
            destroyed = self._destroy(destroy_name, env, current, self.rng)
            removed = tuple(destroyed.removed_items)
            self.alns_diagnostics.destroyed_assignment_count += len(removed)
            if not removed:
                continue
            self.alns_diagnostics.repair_attempt_count += 1
            repaired = self._repair(repair_name, env, destroyed.partial_solution, removed, self.rng)
            self.alns_diagnostics.repair_feasible_count += int(bool(repaired.feasible))
            candidate = repaired.candidate_solution
            if (
                evaluation_budget > 0
                and self.alns_diagnostics.objective_evaluation_count >= evaluation_budget
            ):
                break
            candidate_eval = evaluate_k2_solution(env, candidate)
            self.alns_diagnostics.objective_evaluation_count += 1
            self.alns_diagnostics.feasibility_evaluation_count += 1
            delta = float(candidate_eval.breakdown.total_cost - current_eval.breakdown.total_cost)
            accepted = bool(
                (candidate_eval.feasible and not current_eval.feasible)
                or (candidate_eval.feasible == current_eval.feasible and should_accept_minimization(delta, temperature, self.rng))
            )
            improved = bool(accepted and candidate_eval.breakdown.total_cost < best_eval.breakdown.total_cost)
            self.canonical_operator_records.append({
                "step": int(env.state.step_index),
                "phase": "destroy_repair",
                "destroy_operator": destroy_name,
                "repair_operator": repair_name,
                "removed_count": len(removed),
                "accepted": accepted,
                "improved": improved,
                "feasible": bool(candidate_eval.feasible),
                "reason_codes": list(getattr(repaired, "reason_codes", ())),
            })
            if accepted:
                current, current_eval = candidate, candidate_eval
                self.alns_diagnostics.accepted_count += 1
            if improved:
                best, best_eval = candidate, candidate_eval
                self.alns_diagnostics.improvement_count += 1
            temperature *= float(np.clip(getattr(env.cfg, "alns_sa_cooling_rate", 0.95), 0.80, 0.9999))
        self.alns_diagnostics.wall_clock_time_s += float(__import__("time").perf_counter() - start)
        return {
            str(aid): str(gid) if gid is not None else None
            for aid, gid in self._solution_to_goals(best).items()
        }

    @staticmethod
    def _solution_to_goals(solution: ALNSSolution) -> Dict[str, Optional[str]]:
        goals: Dict[str, Optional[str]] = {}
        for aid, seq in tuple(solution.truck_sequences) + tuple(solution.uav_sequences):
            goals[str(aid)] = str(seq[0]) if seq else None
        return goals

    def get_alns_diagnostics(self) -> DynamicALNSDiagnostics:
        return self.alns_diagnostics

    def export_canonical_operator_records(self) -> List[Dict[str, Any]]:
        return list(self.canonical_operator_records)

    def export_event_trigger_records(self) -> List[Dict[str, Any]]:
        return list(self.dynamic_trigger_records)

    def export_dynamic_trigger_records(self) -> List[Dict[str, Any]]:
        return list(self.dynamic_trigger_records)
