from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import numpy as np

from hetgat_hrl.alns.adapters import solution_to_legacy_goals
from hetgat_hrl.alns.solution import ALNSSolution, StableId
from hetgat_hrl.core.mdp_spec import AgentKind, TaskKind, TaskStatus


class ObjectiveWeights:
    __slots__ = ("unserved_timecritical", "lifeline_loss", "unserved_routine", "travel", "energy", "switching", "support")

    def __init__(
        self,
        unserved_timecritical: float = 4.0,
        lifeline_loss: float = 1.0,
        unserved_routine: float = 1.0,
        travel: float = 1.0 / 5000.0,
        energy: float = 1.0,
        switching: float = 0.15,
        support: float = 0.25,
    ) -> None:
        object.__setattr__(self, "unserved_timecritical", float(unserved_timecritical))
        object.__setattr__(self, "lifeline_loss", float(lifeline_loss))
        object.__setattr__(self, "unserved_routine", float(unserved_routine))
        object.__setattr__(self, "travel", float(travel))
        object.__setattr__(self, "energy", float(energy))
        object.__setattr__(self, "switching", float(switching))
        object.__setattr__(self, "support", float(support))

    def __setattr__(self, name: str, value) -> None:
        raise AttributeError("ObjectiveWeights is immutable")


@dataclass(frozen=True)
class ObjectiveBreakdown:
    __slots__ = (
        "hard_violation_cost",
        "unserved_timecritical_cost",
        "lifeline_loss_cost",
        "unserved_routine_cost",
        "travel_cost",
        "energy_cost",
        "switching_cost",
        "support_cost",
        "total_cost",
    )

    hard_violation_cost: float
    unserved_timecritical_cost: float
    lifeline_loss_cost: float
    unserved_routine_cost: float
    travel_cost: float
    energy_cost: float
    switching_cost: float
    support_cost: float
    total_cost: float

    def component_sum(self) -> float:
        return float(
            self.hard_violation_cost
            + self.unserved_timecritical_cost
            + self.lifeline_loss_cost
            + self.unserved_routine_cost
            + self.travel_cost
            + self.energy_cost
            + self.switching_cost
            + self.support_cost
        )

    def is_decomposed(self, tol: float = 1e-9) -> bool:
        return bool(abs(float(self.total_cost) - self.component_sum()) <= float(tol))


class ObjectiveEvaluation:
    __slots__ = ("feasible", "breakdown", "infeasibility_reasons")

    def __init__(
        self,
        feasible: bool,
        breakdown: ObjectiveBreakdown,
        infeasibility_reasons: tuple[str, ...] = (),
    ) -> None:
        object.__setattr__(self, "feasible", bool(feasible))
        object.__setattr__(self, "breakdown", breakdown)
        object.__setattr__(self, "infeasibility_reasons", tuple(infeasibility_reasons))

    def __setattr__(self, name: str, value) -> None:
        raise AttributeError("ObjectiveEvaluation is immutable")


def is_better(lhs: ObjectiveEvaluation, rhs: ObjectiveEvaluation) -> bool:
    if bool(lhs.feasible) and not bool(rhs.feasible):
        return True
    if not bool(lhs.feasible) and bool(rhs.feasible):
        return False
    return float(lhs.breakdown.total_cost) < float(rhs.breakdown.total_cost)


def minimization_delta(current_cost: float, candidate_cost: float) -> float:
    return float(candidate_cost) - float(current_cost)


def minimization_acceptance_probability(delta: float, temperature: float) -> float:
    if float(delta) <= 0.0:
        return 1.0
    temp = float(max(float(temperature), 1e-12))
    return float(np.clip(math.exp(-float(delta) / temp), 0.0, 1.0))


def should_accept_minimization(delta: float, temperature: float, rng) -> bool:
    prob = minimization_acceptance_probability(delta, temperature)
    if prob >= 1.0:
        return True
    return bool(float(rng.random()) < prob)


def _task_lifeline_loss(task) -> float:
    cur = float(getattr(task, "lifeline_current", getattr(task, "lifeline", 100.0)))
    init = float(max(getattr(task, "lifeline_init", getattr(task, "lifeline_initial", 100.0)), 1e-9))
    return float(np.clip(1.0 - cur / init, 0.0, 1.0))


def _goal_task(env, goal_id: StableId):
    return getattr(env.state, "tasks", {}).get(str(goal_id), None)


def _agent_distance_to_goal(env, agent_id: StableId, goal_id: StableId) -> float:
    task = _goal_task(env, goal_id)
    if task is not None and hasattr(env, "_agent_distance_to_task"):
        try:
            return float(env._agent_distance_to_task(str(agent_id), task))
        except Exception:
            return 0.0
    st = getattr(env.state, "agents", {}).get(str(agent_id), None)
    target = getattr(env.state, "agents", {}).get(str(goal_id), None)
    if st is not None and target is not None and hasattr(env, "_decision_shortest_path_distance"):
        try:
            return float(env._decision_shortest_path_distance(int(st.node), int(target.node)))
        except Exception:
            return 0.0
    return 0.0


def _uav_energy_cost(env, agent_id: StableId, goal_id: StableId, distance_m: float) -> float:
    st = getattr(env.state, "agents", {}).get(str(agent_id), None)
    task = _goal_task(env, goal_id)
    if st is None or getattr(st, "kind", None) != AgentKind.UAV or task is None:
        return 0.0
    xy = getattr(st, "pos", None)
    if hasattr(env, "_uav_energy_cost_fraction"):
        try:
            return float(max(env._uav_energy_cost_fraction(str(agent_id), float(distance_m), xy), 0.0))
        except Exception:
            return 0.0
    return float(max(distance_m, 0.0) / 10000.0)


def _assigned_pending_task_ids(env, solution: ALNSSolution) -> set[str]:
    out: set[str] = set()
    for _aid, seq in tuple(solution.truck_sequences) + tuple(solution.uav_sequences):
        if not seq:
            continue
        task = _goal_task(env, seq[0])
        if task is not None and getattr(task, "status", None) == TaskStatus.PENDING:
            out.add(str(task.task_id))
    return out


def evaluate_solution(
    env,
    solution: ALNSSolution,
    *,
    weights: ObjectiveWeights = ObjectiveWeights(),
    previous_goals: Optional[dict[StableId, StableId | None]] = None,
    hard_feasibility_checker: Optional[Callable[[object, dict[StableId, StableId | None]], bool]] = None,
    infeasibility_reasons: Iterable[str] = (),
) -> ObjectiveEvaluation:
    legacy_goals = solution_to_legacy_goals(solution)
    reasons = tuple(str(x) for x in infeasibility_reasons)
    feasible = True
    if hard_feasibility_checker is not None:
        try:
            feasible = bool(hard_feasibility_checker(env, legacy_goals))
        except Exception as exc:
            feasible = False
            reasons = (*reasons, f"checker_error:{type(exc).__name__}")

    assigned = _assigned_pending_task_ids(env, solution)
    pending_tasks = [
        t for t in getattr(env.state, "tasks", {}).values()
        if getattr(t, "status", None) == TaskStatus.PENDING
    ]
    pending_tc = [t for t in pending_tasks if getattr(t, "kind", None) == TaskKind.EMERGENCY]
    pending_routine = [t for t in pending_tasks if getattr(t, "kind", None) == TaskKind.NORMAL]
    unserved_tc = sum(1 for t in pending_tc if str(t.task_id) not in assigned)
    unserved_routine = sum(1 for t in pending_routine if str(t.task_id) not in assigned)
    lifeline_loss = sum(_task_lifeline_loss(t) for t in pending_tc if str(t.task_id) not in assigned)

    travel = 0.0
    energy = 0.0
    switching = 0.0
    for aid, seq in tuple(solution.truck_sequences) + tuple(solution.uav_sequences):
        if not seq:
            continue
        goal_id = seq[0]
        dist = float(max(_agent_distance_to_goal(env, aid, goal_id), 0.0))
        travel += dist
        energy += _uav_energy_cost(env, aid, goal_id, dist)
        if previous_goals is not None and previous_goals.get(aid, None) not in {None, goal_id}:
            switching += 1.0

    support = float(len(solution.support_bindings) + len(solution.sortie_plans))
    hard = 0.0 if feasible else 1.0
    total = float(
        hard
        + weights.unserved_timecritical * float(unserved_tc)
        + weights.lifeline_loss * float(lifeline_loss)
        + weights.unserved_routine * float(unserved_routine)
        + weights.travel * float(travel)
        + weights.energy * float(energy)
        + weights.switching * float(switching)
        + weights.support * float(support)
    )
    breakdown = ObjectiveBreakdown(
        hard_violation_cost=float(hard),
        unserved_timecritical_cost=float(weights.unserved_timecritical * float(unserved_tc)),
        lifeline_loss_cost=float(weights.lifeline_loss * float(lifeline_loss)),
        unserved_routine_cost=float(weights.unserved_routine * float(unserved_routine)),
        travel_cost=float(weights.travel * float(travel)),
        energy_cost=float(weights.energy * float(energy)),
        switching_cost=float(weights.switching * float(switching)),
        support_cost=float(weights.support * float(support)),
        total_cost=float(total),
    )
    return ObjectiveEvaluation(feasible=bool(feasible), breakdown=breakdown, infeasibility_reasons=reasons)
