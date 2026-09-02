from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

import numpy as np

from hetgat_hrl.alns.adapters import SolutionAdapterContext, legacy_goals_to_solution
from hetgat_hrl.alns.objective import ObjectiveBreakdown, ObjectiveEvaluation, ObjectiveWeights, is_better
from hetgat_hrl.alns.solution import ALNSSolution, StableId
from hetgat_hrl.core.mdp_spec import AgentKind, TaskKind, TaskStatus


REASON_FIRST_TASK_INVALID = "FIRST_TASK_INVALID"
REASON_SECOND_TASK_INVALID = "SECOND_TASK_INVALID"
REASON_TASK_NOT_ACTIVE = "TASK_NOT_ACTIVE"
REASON_AGENT_TYPE_INCOMPATIBLE = "AGENT_TYPE_INCOMPATIBLE"
REASON_DUPLICATE_EXCLUSIVE_TASK = "DUPLICATE_EXCLUSIVE_TASK"
REASON_TARGET_UNREACHABLE = "TARGET_UNREACHABLE"
REASON_INSUFFICIENT_UAV_ENERGY = "INSUFFICIENT_UAV_ENERGY"
REASON_SEQUENCE_TOO_LONG = "SEQUENCE_TOO_LONG"


@dataclass(frozen=True)
class SequencePlanningState:
    __slots__ = (
        "agent_id",
        "agent_type",
        "current_node",
        "available_step",
        "remaining_inventory",
        "remaining_payload",
        "remaining_energy",
        "support_truck_id",
        "launch_anchor",
        "recovery_anchor",
    )

    agent_id: StableId
    agent_type: str
    current_node: Optional[int]
    available_step: int
    remaining_inventory: Any
    remaining_payload: Optional[float]
    remaining_energy: Optional[float]
    support_truck_id: Optional[StableId]
    launch_anchor: Optional[int]
    recovery_anchor: Optional[int]


@dataclass(frozen=True)
class SequenceFeasibility:
    __slots__ = ("feasible", "first_task_feasible", "second_task_feasible", "reason_codes")

    feasible: bool
    first_task_feasible: bool
    second_task_feasible: bool
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class SequenceCost:
    __slots__ = (
        "agent_id",
        "sequence",
        "eta_first",
        "eta_second",
        "completion_first",
        "completion_second",
        "travel_cost_first",
        "travel_cost_second",
        "energy_cost_first",
        "energy_cost_second",
        "lifeline_loss_first",
        "lifeline_loss_second",
        "service_time_first",
        "service_time_second",
    )

    agent_id: StableId
    sequence: tuple[StableId, ...]
    eta_first: Optional[float]
    eta_second: Optional[float]
    completion_first: Optional[float]
    completion_second: Optional[float]
    travel_cost_first: float
    travel_cost_second: float
    energy_cost_first: float
    energy_cost_second: float
    lifeline_loss_first: float
    lifeline_loss_second: float
    service_time_first: float
    service_time_second: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": str(self.agent_id),
            "sequence": [str(x) for x in self.sequence],
            "eta_first": self.eta_first,
            "eta_second": self.eta_second,
            "completion_first": self.completion_first,
            "completion_second": self.completion_second,
            "travel_cost_first": self.travel_cost_first,
            "travel_cost_second": self.travel_cost_second,
            "energy_cost_first": self.energy_cost_first,
            "energy_cost_second": self.energy_cost_second,
            "lifeline_loss_first": self.lifeline_loss_first,
            "lifeline_loss_second": self.lifeline_loss_second,
            "service_time_first": self.service_time_first,
            "service_time_second": self.service_time_second,
        }


def _agent_kind_value(st) -> str:
    return str(getattr(getattr(st, "kind", ""), "value", getattr(st, "kind", ""))).lower()


def _task_kind(task) -> Any:
    return getattr(task, "kind", None)


def _task_active(task) -> bool:
    return bool(task is not None and getattr(task, "status", None) == TaskStatus.PENDING)


def _task_node(task) -> Optional[int]:
    node = getattr(task, "demand_node", None)
    if node is None:
        return None
    try:
        return int(node)
    except Exception:
        return None


def _service_steps(env, task) -> float:
    if task is None:
        return 0.0
    if getattr(task, "kind", None) == TaskKind.EMERGENCY:
        return float(max(getattr(env.cfg, "unload_rounds_emergency", 1), 1))
    return float(max(getattr(env.cfg, "unload_rounds_normal", 1), 1))


def _distance_between_nodes(env, a: Optional[int], b: Optional[int]) -> float:
    if a is None or b is None:
        return float("inf")
    if int(a) == int(b):
        return 0.0
    if hasattr(env, "_decision_shortest_path_distance"):
        try:
            return float(env._decision_shortest_path_distance(int(a), int(b)))
        except Exception:
            return float("inf")
    return float(abs(int(a) - int(b)))


def _agent_node(env, agent_id: StableId) -> Optional[int]:
    st = getattr(env.state, "agents", {}).get(str(agent_id), None)
    node = getattr(st, "node", None)
    if node is None:
        return None
    try:
        return int(node)
    except Exception:
        return None


def _speed_mps(env, agent_type: str) -> float:
    if str(agent_type).lower() == "uav":
        return float(max(getattr(env.cfg, "uav_max_speed_mps", 1.0), 1e-6))
    return float(max(getattr(env.cfg, "truck_speed_mps", 1.0), 1e-6))


def _distance_to_steps(env, distance_m: float, agent_type: str) -> float:
    if not np.isfinite(float(distance_m)):
        return float("inf")
    dt = float(max(getattr(env.cfg, "dt_seconds", getattr(env.cfg, "dt", 1.0)), 1e-6))
    return float(max(distance_m, 0.0) / _speed_mps(env, agent_type) / dt)


def _lifeline_loss_at(task, completion_step: Optional[float]) -> float:
    if task is None or completion_step is None or not np.isfinite(float(completion_step)):
        return 1.0
    cur = float(getattr(task, "lifeline_current", getattr(task, "lifeline", 100.0)))
    init = float(max(getattr(task, "lifeline_init", getattr(task, "lifeline_initial", 100.0)), 1e-9))
    decay = float(max(getattr(task, "lifeline_decay_per_step", getattr(task, "lifeline_decay", 0.0)), 0.0))
    projected = max(cur - decay * max(float(completion_step), 0.0), 0.0)
    return float(np.clip(1.0 - projected / init, 0.0, 1.0))


def initial_planning_state(env, agent_id: StableId) -> SequencePlanningState:
    st = getattr(env.state, "agents", {}).get(str(agent_id), None)
    agent_type = _agent_kind_value(st)
    return SequencePlanningState(
        agent_id=agent_id,
        agent_type=agent_type,
        current_node=_agent_node(env, agent_id),
        available_step=int(getattr(env.state, "step_index", 0)),
        remaining_inventory={
            "bulk": float(getattr(st, "bulk_inventory_kg", getattr(st, "inventory_kg", 0.0))),
            "timecritical": float(getattr(st, "timecritical_inventory_kg", 0.0)),
        },
        remaining_payload=float(getattr(st, "payload_kg", 0.0)) if hasattr(st, "payload_kg") else None,
        remaining_energy=float(getattr(st, "battery", 0.0)) if hasattr(st, "battery") else None,
        support_truck_id=getattr(st, "follow_target", None),
        launch_anchor=_agent_node(env, agent_id),
        recovery_anchor=None,
    )


def _derive_after_task(env, state: SequencePlanningState, task) -> SequencePlanningState:
    node = _task_node(task)
    travel = _distance_between_nodes(env, state.current_node, node)
    eta = float(state.available_step) + _distance_to_steps(env, travel, state.agent_type)
    available = int(np.ceil(eta + _service_steps(env, task))) if np.isfinite(eta) else int(state.available_step)
    energy = state.remaining_energy
    if state.agent_type == "uav" and energy is not None:
        energy = float(max(float(energy) - _uav_energy_for_segment(env, state.agent_id, travel), 0.0))
    return SequencePlanningState(
        agent_id=state.agent_id,
        agent_type=state.agent_type,
        current_node=node,
        available_step=available,
        remaining_inventory=state.remaining_inventory,
        remaining_payload=state.remaining_payload,
        remaining_energy=energy,
        support_truck_id=state.support_truck_id,
        launch_anchor=state.launch_anchor,
        recovery_anchor=state.recovery_anchor,
    )


def _uav_energy_for_segment(env, agent_id: StableId, distance_m: float) -> float:
    if not np.isfinite(float(distance_m)):
        return float("inf")
    st = getattr(env.state, "agents", {}).get(str(agent_id), None)
    if hasattr(env, "_uav_energy_cost_fraction"):
        try:
            return float(max(env._uav_energy_cost_fraction(str(agent_id), float(max(distance_m, 0.0)), getattr(st, "pos", None)), 0.0))
        except Exception:
            pass
    return float(max(distance_m, 0.0) * float(getattr(env.cfg, "uav_flight_discharge_per_m", 0.0)))


def _agent_can_serve_task(env, agent_id: StableId, task, *, planning_state: Optional[SequencePlanningState] = None) -> bool:
    st = getattr(env.state, "agents", {}).get(str(agent_id), None)
    if st is None or not _task_active(task):
        return False
    kind = getattr(st, "kind", None)
    if kind == AgentKind.TRUCK:
        if hasattr(env, "_decision_shortest_path_distance") and planning_state is not None:
            return bool(np.isfinite(_distance_between_nodes(env, planning_state.current_node, _task_node(task))))
        return True
    if kind == AgentKind.UAV:
        if getattr(task, "kind", None) != TaskKind.EMERGENCY:
            return False
        if planning_state is not None and planning_state.remaining_energy is not None:
            d = _distance_between_nodes(env, planning_state.current_node, _task_node(task))
            reserve = float(max(getattr(env.cfg, "uav_return_margin_fraction", 0.0), 0.0))
            return bool(float(planning_state.remaining_energy) + 1e-9 >= _uav_energy_for_segment(env, agent_id, d) + reserve)
        return True
    return False


def evaluate_sequence_cost(env, agent_id: StableId, sequence: Iterable[StableId]) -> SequenceCost:
    seq = tuple(sequence)[:2]
    state = initial_planning_state(env, agent_id)
    first = getattr(env.state, "tasks", {}).get(str(seq[0]), None) if len(seq) >= 1 else None
    second = getattr(env.state, "tasks", {}).get(str(seq[1]), None) if len(seq) >= 2 else None
    first_node = _task_node(first)
    d1 = _distance_between_nodes(env, state.current_node, first_node) if first is not None else 0.0
    eta1 = float(state.available_step) + _distance_to_steps(env, d1, state.agent_type) if first is not None else None
    s1 = _service_steps(env, first)
    c1 = None if eta1 is None else float(eta1 + s1)
    e1 = _uav_energy_for_segment(env, agent_id, d1) if state.agent_type == "uav" and first is not None else 0.0
    loss1 = _lifeline_loss_at(first, c1) if first is not None else 0.0

    after_first = _derive_after_task(env, state, first) if first is not None else state
    second_node = _task_node(second)
    d2 = _distance_between_nodes(env, after_first.current_node, second_node) if second is not None else 0.0
    eta2 = float(after_first.available_step) + _distance_to_steps(env, d2, state.agent_type) if second is not None else None
    s2 = _service_steps(env, second)
    c2 = None if eta2 is None else float(eta2 + s2)
    e2 = _uav_energy_for_segment(env, agent_id, d2) if state.agent_type == "uav" and second is not None else 0.0
    loss2 = _lifeline_loss_at(second, c2) if second is not None else 0.0
    return SequenceCost(
        agent_id=agent_id,
        sequence=seq,
        eta_first=eta1,
        eta_second=eta2,
        completion_first=c1,
        completion_second=c2,
        travel_cost_first=float(d1 if np.isfinite(d1) else 1e9),
        travel_cost_second=float(d2 if np.isfinite(d2) else 1e9),
        energy_cost_first=float(e1 if np.isfinite(e1) else 1e9),
        energy_cost_second=float(e2 if np.isfinite(e2) else 1e9),
        lifeline_loss_first=float(loss1),
        lifeline_loss_second=float(loss2),
        service_time_first=float(s1),
        service_time_second=float(s2),
    )


def evaluate_sequence_feasibility(
    env,
    agent_id: StableId,
    sequence: Iterable[StableId],
    *,
    exclusive_claims: Optional[Mapping[str, StableId]] = None,
) -> SequenceFeasibility:
    seq = tuple(sequence)
    reasons: list[str] = []
    if len(seq) > 2:
        reasons.append(REASON_SEQUENCE_TOO_LONG)
    if len(set(str(x) for x in seq)) != len(seq):
        reasons.append(REASON_DUPLICATE_EXCLUSIVE_TASK)
    st = getattr(env.state, "agents", {}).get(str(agent_id), None)
    first_ok = True
    second_ok = True
    state = initial_planning_state(env, agent_id)
    for idx, tid in enumerate(seq[:2]):
        task = getattr(env.state, "tasks", {}).get(str(tid), None)
        ok = True
        if not _task_active(task):
            reasons.append(REASON_TASK_NOT_ACTIVE)
            ok = False
        owner = None if exclusive_claims is None else exclusive_claims.get(str(tid), None)
        if owner is not None and str(owner) != str(agent_id):
            reasons.append(REASON_DUPLICATE_EXCLUSIVE_TASK)
            ok = False
        if not _agent_can_serve_task(env, agent_id, task, planning_state=state):
            reasons.append(REASON_AGENT_TYPE_INCOMPATIBLE)
            ok = False
        d = _distance_between_nodes(env, state.current_node, _task_node(task))
        if not np.isfinite(float(d)):
            reasons.append(REASON_TARGET_UNREACHABLE)
            ok = False
        if _agent_kind_value(st) == "uav" and state.remaining_energy is not None:
            reserve = float(max(getattr(env.cfg, "uav_return_margin_fraction", 0.0), 0.0))
            if float(state.remaining_energy) + 1e-9 < _uav_energy_for_segment(env, agent_id, d) + reserve:
                reasons.append(REASON_INSUFFICIENT_UAV_ENERGY)
                ok = False
        if idx == 0:
            first_ok = ok
        else:
            second_ok = ok
        if ok:
            state = _derive_after_task(env, state, task)
    if len(seq) == 0:
        first_ok = True
        second_ok = True
    elif len(seq) == 1:
        second_ok = True
    reasons_tuple = tuple(sorted(set(reasons)))
    return SequenceFeasibility(
        feasible=bool(first_ok and second_ok and not reasons_tuple),
        first_task_feasible=bool(first_ok),
        second_task_feasible=bool(second_ok),
        reason_codes=reasons_tuple,
    )


def _sequence_pairs(solution: ALNSSolution) -> list[tuple[StableId, str, tuple[StableId, ...]]]:
    out = [(aid, "truck", tuple(seq)) for aid, seq in solution.truck_sequences]
    out.extend((aid, "uav", tuple(seq)) for aid, seq in solution.uav_sequences)
    return out


def sequence_claims(solution: ALNSSolution) -> dict[str, StableId]:
    claims: dict[str, StableId] = {}
    for aid, _atype, seq in _sequence_pairs(solution):
        for tid in seq:
            claims.setdefault(str(tid), aid)
    return claims


def construct_k2_solution(env, goals: Mapping[StableId, StableId | None]) -> ALNSSolution:
    agent_types = {aid: _agent_kind_value(st) for aid, st in getattr(env.state, "agents", {}).items()}
    k1 = legacy_goals_to_solution(goals, SolutionAdapterContext(agent_types=agent_types, sequence_length=1))
    used = {str(tid) for _aid, _atype, seq in _sequence_pairs(k1) for tid in seq}
    truck: dict[StableId, tuple[StableId, ...]] = dict(k1.truck_sequences)
    uav: dict[StableId, tuple[StableId, ...]] = dict(k1.uav_sequences)
    init_mode = str(getattr(env.cfg, "alns_initialization_mode", "objective_greedy")).strip().lower() or "objective_greedy"

    def _init_priority(aid: StableId, atype: str, task, ev: ObjectiveEvaluation) -> tuple[float, float, float, float, float, str]:
        if task is None:
            return (0.0, 0.0, 0.0, float("inf"), float(ev.breakdown.total_cost), "")
        is_tc = 1.0 if getattr(task, "kind", None) == TaskKind.EMERGENCY else 0.0
        cur = float(getattr(task, "lifeline_current", getattr(task, "lifeline", 100.0)))
        init = float(max(getattr(task, "lifeline_init", getattr(task, "lifeline_initial", 100.0)), 1e-9))
        urgency = 1.0 - float(np.clip(cur / init, 0.0, 1.0))
        deadline = float(getattr(task, "deadline_step", getattr(env.state, "step_index", 0)) - getattr(env.state, "step_index", 0))
        cost = evaluate_sequence_cost(env, aid, tuple(list((truck if atype == "truck" else uav).get(aid, ())) + [str(getattr(task, "task_id", ""))])[:2])
        energy_penalty = float(cost.energy_cost_second if len(cost.sequence) >= 2 else cost.energy_cost_first)
        if atype != "uav":
            energy_penalty = 0.0
        if init_mode == "critical_first":
            return (-is_tc, -urgency, float(deadline), energy_penalty, float(ev.breakdown.total_cost), str(getattr(task, "task_id", "")))
        return (0.0, 0.0, 0.0, 0.0, float(ev.breakdown.total_cost), str(getattr(task, "task_id", "")))

    for aid, atype in sorted(agent_types.items(), key=lambda kv: str(kv[0])):
        cur = tuple((truck if atype == "truck" else uav).get(aid, ()))
        if len(cur) >= 2:
            continue
        best_tid = None
        best_eval = None
        best_key = None
        for task in getattr(env.state, "tasks", {}).values():
            tid = str(getattr(task, "task_id", ""))
            if not tid or tid in used or tid in {str(x) for x in cur}:
                continue
            trial_seq = tuple(list(cur) + [tid])
            feas = evaluate_sequence_feasibility(env, aid, trial_seq)
            if not feas.feasible:
                continue
            trial_solution = ALNSSolution(
                truck_sequences={**truck, **({aid: trial_seq} if atype == "truck" else {})},
                uav_sequences={**uav, **({aid: trial_seq} if atype == "uav" else {})},
            )
            ev = evaluate_k2_solution(env, trial_solution)
            key = _init_priority(aid, atype, task, ev)
            if best_eval is None or is_better(ev, best_eval) or (best_key is not None and key < best_key):
                best_tid = tid
                best_eval = ev
                best_key = key
        if best_tid is not None:
            new_seq = tuple(list(cur) + [best_tid])[:2]
            if atype == "truck":
                truck[aid] = new_seq
            else:
                uav[aid] = new_seq
            used.add(str(best_tid))
    return ALNSSolution(truck_sequences=truck, uav_sequences=uav)


def evaluate_k2_solution(
    env,
    solution: ALNSSolution,
    *,
    weights: ObjectiveWeights = ObjectiveWeights(),
    hard_feasible: bool = True,
    infeasibility_reasons: Iterable[str] = (),
) -> ObjectiveEvaluation:
    reasons = list(infeasibility_reasons)
    claims: dict[str, StableId] = {}
    feasible = bool(hard_feasible)
    unserved_tc = 0
    unserved_routine = 0
    travel = 0.0
    energy = 0.0
    life = 0.0
    for aid, _atype, seq in _sequence_pairs(solution):
        for tid in seq:
            if str(tid) in claims and str(claims[str(tid)]) != str(aid):
                feasible = False
                reasons.append(REASON_DUPLICATE_EXCLUSIVE_TASK)
            claims[str(tid)] = aid
        feas = evaluate_sequence_feasibility(env, aid, seq, exclusive_claims=claims)
        if not feas.feasible:
            feasible = False
            reasons.extend(feas.reason_codes)
        cost = evaluate_sequence_cost(env, aid, seq)
        travel += cost.travel_cost_first + cost.travel_cost_second
        energy += cost.energy_cost_first + cost.energy_cost_second
        life += cost.lifeline_loss_first + cost.lifeline_loss_second
    pending = [t for t in getattr(env.state, "tasks", {}).values() if _task_active(t)]
    for task in pending:
        if str(getattr(task, "task_id", "")) in claims:
            continue
        if getattr(task, "kind", None) == TaskKind.EMERGENCY:
            unserved_tc += 1
            life += _lifeline_loss_at(task, float(getattr(env.state, "step_index", 0)))
        else:
            unserved_routine += 1
    support = float(len(solution.support_bindings) + len(solution.sortie_plans))
    hard = 0.0 if feasible else 1.0
    total = float(
        hard
        + weights.unserved_timecritical * unserved_tc
        + weights.lifeline_loss * life
        + weights.unserved_routine * unserved_routine
        + weights.travel * travel
        + weights.energy * energy
        + weights.support * support
    )
    return ObjectiveEvaluation(
        feasible=bool(feasible),
        breakdown=ObjectiveBreakdown(
            hard_violation_cost=float(hard),
            unserved_timecritical_cost=float(weights.unserved_timecritical * unserved_tc),
            lifeline_loss_cost=float(weights.lifeline_loss * life),
            unserved_routine_cost=float(weights.unserved_routine * unserved_routine),
            travel_cost=float(weights.travel * travel),
            energy_cost=float(weights.energy * energy),
            switching_cost=0.0,
            support_cost=float(weights.support * support),
            total_cost=float(total),
        ),
        infeasibility_reasons=tuple(sorted(set(str(x) for x in reasons))),
    )


def promote_completed_or_invalid_tail(env, solution: ALNSSolution) -> tuple[ALNSSolution, dict[str, int]]:
    stats = {"tail_promoted": 0, "tail_dropped": 0, "tail_invalidated": 0}
    truck = {}
    uav = {}
    for aid, atype, seq in _sequence_pairs(solution):
        new_seq = tuple(seq)
        if new_seq:
            first = getattr(env.state, "tasks", {}).get(str(new_seq[0]), None)
            if first is None or getattr(first, "status", None) != TaskStatus.PENDING:
                new_seq = new_seq[1:]
                if new_seq:
                    stats["tail_promoted"] += 1
                else:
                    stats["tail_dropped"] += 1
        if len(new_seq) >= 2:
            second = getattr(env.state, "tasks", {}).get(str(new_seq[1]), None)
            if second is None or getattr(second, "status", None) != TaskStatus.PENDING:
                new_seq = (new_seq[0],)
                stats["tail_invalidated"] += 1
        if atype == "truck":
            truck[aid] = new_seq
        else:
            uav[aid] = new_seq
    return ALNSSolution(truck_sequences=truck, uav_sequences=uav), stats
