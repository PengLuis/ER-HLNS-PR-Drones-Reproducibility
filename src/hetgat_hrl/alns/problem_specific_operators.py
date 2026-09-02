from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Optional

import numpy as np

from hetgat_hrl.alns.canonical_operators import (
    FAIL_AGENT_STATE_CONFLICT,
    FAIL_DUPLICATE_EXCLUSIVE_TASK,
    FAIL_NO_FEASIBLE_INSERTION_POSITION,
    FAIL_NO_SEQUENCE_CAPACITY,
    FAIL_RECOVERY_INFEASIBLE,
    FAIL_ROAD_UNREACHABLE,
    FAIL_SUPPORT_BINDING_REQUIRED,
    FAIL_UAV_ENERGY_INSUFFICIENT,
    DestroyResult,
    InsertedItem,
    InsertOption,
    RemovedItem,
    RepairResult,
    _distance,
    _insertion_options,
    _is_removable,
    _lifeline_ratio,
    _map_sequence_reasons,
    _partial_result,
    _remove_position,
    _repair_result,
    _sequence_items,
    _solution_agents,
    _support_required,
    _task,
    _task_ids,
    _task_node,
)
from hetgat_hrl.alns.objective import ObjectiveWeights
from hetgat_hrl.alns.sequence import evaluate_k2_solution, evaluate_sequence_cost, evaluate_sequence_feasibility
from hetgat_hrl.alns.solution import ALNSSolution, SortiePlan, StableId, SupportBinding
from hetgat_hrl.core.mdp_spec import AgentKind, TaskKind, TaskStatus


REASON_ROAD_DISRUPTION_REMOVAL = "ROAD_DISRUPTION_REMOVAL"
REASON_CRITICAL_TASK_REASSIGNMENT_REMOVAL = "CRITICAL_TASK_REASSIGNMENT_REMOVAL"
REASON_SUPPORT_CONFLICT_REMOVAL = "SUPPORT_CONFLICT_REMOVAL"
REASON_SYNCHRONIZATION_RISK_REMOVAL = "SYNCHRONIZATION_RISK_REMOVAL"


def _agent_node(env, agent_id: StableId) -> Optional[int]:
    st = getattr(env.state, "agents", {}).get(str(agent_id), None)
    node = getattr(st, "node", None)
    if node is None:
        return None
    try:
        return int(node)
    except Exception:
        return None


def _segment_origin(env, solution: ALNSSolution, item: RemovedItem) -> Optional[int]:
    if int(item.sequence_position) <= 0:
        return _agent_node(env, item.agent_id)
    seq = solution.sequence_for(item.agent_id, item.agent_type)
    prev_task = _task(env, seq[int(item.sequence_position) - 1])
    return _task_node(prev_task)


def _is_timecritical(task) -> bool:
    return bool(getattr(task, "kind", None) == TaskKind.EMERGENCY or str(getattr(task, "task_class", "")).lower().find("time") >= 0)


def _critical_priority(task) -> float:
    if task is None:
        return 0.0
    urgency = 1.0 - _lifeline_ratio(task)
    decay = float(max(getattr(task, "lifeline_decay_rate", getattr(task, "lifeline_decay_per_step", 0.0)), 0.0))
    return float((3.0 if _is_timecritical(task) else 0.0) + urgency + decay)


def _assigned_task_ids(solution: ALNSSolution) -> set[str]:
    return {str(tid) for item in _sequence_items(solution) for tid in (item.task_id,)}


def _pending_tasks(env) -> list[Any]:
    return [
        task
        for task in getattr(env.state, "tasks", {}).values()
        if getattr(task, "status", None) == TaskStatus.PENDING
    ]


def _solution_with_support(
    solution: ALNSSolution,
    *,
    binding: SupportBinding,
    sortie: SortiePlan,
    uav_id: StableId,
    task_id: StableId,
) -> ALNSSolution:
    truck = dict(solution.truck_sequences)
    uav = dict(solution.uav_sequences)
    uav[str(uav_id)] = tuple(list(uav.get(str(uav_id), ())) + [task_id])[:2]
    bindings = tuple(b for b in solution.support_bindings if not (str(b.uav_id) == str(uav_id) or str(b.task_id) == str(task_id)))
    sorties = tuple(p for p in solution.sortie_plans if not (str(p.uav_id) == str(uav_id) or str(p.task_id) == str(task_id)))
    return ALNSSolution(
        truck_sequences=truck,
        uav_sequences=uav,
        support_bindings=(*bindings, binding),
        sortie_plans=(*sorties, sortie),
    )


def road_disruption_removal(env, solution: ALNSSolution, rng=None, *, max_remove: int = 2) -> DestroyResult:
    scored: list[tuple[float, str, RemovedItem]] = []
    for item in _sequence_items(solution):
        if not _is_removable(env, solution, item):
            continue
        task = _task(env, item.task_id)
        origin = _segment_origin(env, solution, item)
        dist = _distance(env, origin, _task_node(task))
        if not np.isfinite(float(dist)):
            scored.append((float("inf"), str(item.task_id), item))
        elif float(dist) > float(max(getattr(env.cfg, "road_disruption_removal_distance_m", 5000.0), 1.0)):
            scored.append((float(dist), str(item.task_id), item))
    if not scored:
        return _partial_result(solution, (), REASON_ROAD_DISRUPTION_REMOVAL)
    scored.sort(key=lambda row: (-float(row[0]) if np.isfinite(row[0]) else float("-inf"), row[1]))
    cand = solution
    removed: list[RemovedItem] = []
    for _score, _tid, item in scored[: max(int(max_remove), 1)]:
        cand, rem = _remove_position(cand, item, REASON_ROAD_DISRUPTION_REMOVAL)
        removed.append(rem)
    return _partial_result(cand, removed, REASON_ROAD_DISRUPTION_REMOVAL, {"affected_segments": len(scored)})


def critical_task_reassignment_removal(env, solution: ALNSSolution, rng=None, *, max_remove: int = 1) -> DestroyResult:
    assigned = _assigned_task_ids(solution)
    uncovered = [
        task
        for task in _pending_tasks(env)
        if str(getattr(task, "task_id", "")) not in assigned and _critical_priority(task) >= 2.0
    ]
    if not uncovered:
        return _partial_result(solution, (), REASON_CRITICAL_TASK_REASSIGNMENT_REMOVAL)
    scored: list[tuple[float, str, RemovedItem]] = []
    for item in _sequence_items(solution):
        if not _is_removable(env, solution, item):
            continue
        task = _task(env, item.task_id)
        score = float(_critical_priority(task) + (0.25 if int(item.sequence_position) == 1 else 0.0))
        scored.append((score, str(item.task_id), item))
    if not scored:
        return _partial_result(solution, (), REASON_CRITICAL_TASK_REASSIGNMENT_REMOVAL)
    scored.sort(key=lambda row: (float(row[0]), row[1]))
    cand = solution
    removed: list[RemovedItem] = []
    for _score, _tid, item in scored[: max(int(max_remove), 1)]:
        cand, rem = _remove_position(cand, item, REASON_CRITICAL_TASK_REASSIGNMENT_REMOVAL)
        removed.append(rem)
    return _partial_result(cand, removed, REASON_CRITICAL_TASK_REASSIGNMENT_REMOVAL, {"uncovered_critical": len(uncovered)})


def support_conflict_removal(env, solution: ALNSSolution, rng=None, *, max_remove: int = 1) -> DestroyResult:
    truck_claims: dict[str, list[SupportBinding]] = {}
    for binding in solution.support_bindings:
        truck_claims.setdefault(str(binding.truck_id), []).append(binding)
    conflicted_tasks = {
        str(binding.task_id)
        for bindings in truck_claims.values()
        if len(bindings) > 1
        for binding in bindings
    }
    for binding in solution.support_bindings:
        task = _task(env, binding.task_id)
        if task is None or getattr(task, "status", None) != TaskStatus.PENDING:
            conflicted_tasks.add(str(binding.task_id))
    scored = [item for item in _sequence_items(solution) if str(item.task_id) in conflicted_tasks and _is_removable(env, solution, item)]
    if not scored:
        scored = [
            item
            for item in _sequence_items(solution)
            if item.agent_type == "truck" and _is_removable(env, solution, item) and not _is_timecritical(_task(env, item.task_id))
        ]
    if not scored:
        return _partial_result(solution, (), REASON_SUPPORT_CONFLICT_REMOVAL)
    cand = solution
    removed: list[RemovedItem] = []
    for item in scored[: max(int(max_remove), 1)]:
        cand, rem = _remove_position(cand, item, REASON_SUPPORT_CONFLICT_REMOVAL)
        removed.append(rem)
    return _partial_result(cand, removed, REASON_SUPPORT_CONFLICT_REMOVAL, {"conflicted_tasks": sorted(conflicted_tasks)})


def synchronization_risk_removal(env, solution: ALNSSolution, rng=None, *, max_remove: int = 1) -> DestroyResult:
    risky: list[tuple[float, str, RemovedItem]] = []
    for item in _sequence_items(solution):
        if not _is_removable(env, solution, item):
            continue
        st = getattr(env.state, "agents", {}).get(str(item.agent_id), None)
        task = _task(env, item.task_id)
        risk = 0.0
        if item.agent_type == "uav":
            cost = evaluate_sequence_cost(env, item.agent_id, solution.sequence_for(item.agent_id, item.agent_type))
            energy = float(getattr(st, "battery", 0.0))
            reserve = float(max(getattr(env.cfg, "uav_return_margin_fraction", 0.0), 0.0))
            if energy + 1e-9 < cost.energy_cost_first + cost.energy_cost_second + reserve:
                risk += 10.0
        for binding in solution.support_bindings:
            if str(binding.task_id) == str(item.task_id):
                launch_d = _distance(env, _agent_node(env, binding.truck_id), int(binding.launch_anchor))
                rec_d = _distance(env, _task_node(task), int(binding.recovery_anchor))
                if not np.isfinite(launch_d) or not np.isfinite(rec_d):
                    risk += 10.0
        if risk > 0.0:
            risky.append((risk, str(item.task_id), item))
    if not risky:
        return _partial_result(solution, (), REASON_SYNCHRONIZATION_RISK_REMOVAL)
    risky.sort(key=lambda row: (-float(row[0]), row[1]))
    cand = solution
    removed: list[RemovedItem] = []
    for _risk, _tid, item in risky[: max(int(max_remove), 1)]:
        cand, rem = _remove_position(cand, item, REASON_SYNCHRONIZATION_RISK_REMOVAL)
        removed.append(rem)
    return _partial_result(cand, removed, REASON_SYNCHRONIZATION_RISK_REMOVAL)


def _prioritized_task_ids(env, partial_solution: ALNSSolution, removed_items: Iterable[RemovedItem | StableId]) -> list[StableId]:
    requested = list(_task_ids(removed_items))
    assigned = _assigned_task_ids(partial_solution)
    for task in _pending_tasks(env):
        tid = str(getattr(task, "task_id", ""))
        if tid and tid not in assigned and _critical_priority(task) >= 2.0 and tid not in {str(x) for x in requested}:
            requested.append(tid)
    requested.sort(key=lambda tid: (-_critical_priority(_task(env, tid)), str(tid)))
    return requested


def _critical_recovery_candidate_ids(
    env,
    partial_solution: ALNSSolution,
    removed_items: Iterable[RemovedItem | StableId],
) -> list[StableId]:
    assigned = _assigned_task_ids(partial_solution)
    requested: list[StableId] = []
    seen: set[str] = set()
    max_tasks = int(max(getattr(env.cfg, "alns_critical_recovery_repair_max_tasks", 3), 0))
    min_priority = float(max(getattr(env.cfg, "alns_critical_recovery_repair_min_priority", 0.0), 0.0))
    head_tasks = {
        str(solution_task)
        for aid, atype in _solution_agents(env, partial_solution)
        for solution_task in partial_solution.sequence_for(aid, atype)[:1]
    }

    def _add(tid: StableId) -> None:
        text = str(tid)
        if not text or text in seen:
            return
        task = _task(env, tid)
        if task is None or getattr(task, "status", None) != TaskStatus.PENDING:
            return
        if not (_is_timecritical(task) or _critical_priority(task) >= min_priority):
            return
        if text in head_tasks:
            return
        requested.append(tid)
        seen.add(text)

    for item in removed_items:
        tid = item.task_id if isinstance(item, RemovedItem) else item
        _add(tid)

    for aid, atype in _solution_agents(env, partial_solution):
        seq = partial_solution.sequence_for(aid, atype)
        if len(seq) > 1:
            _add(seq[1])

    for task in _pending_tasks(env):
        tid = str(getattr(task, "task_id", ""))
        if tid and tid not in assigned:
            _add(tid)

    requested.sort(
        key=lambda tid: (
            -float(_critical_priority(_task(env, tid))),
            float(_lifeline_ratio(_task(env, tid))),
            -float(getattr(_task(env, tid), "urgency_score", 0.0)),
            str(tid),
        )
    )
    return requested[:max_tasks]


def _removed_agent_pairs(removed_items: Iterable[RemovedItem | StableId]) -> dict[str, set[str]]:
    pairs: dict[str, set[str]] = {}
    for item in removed_items:
        if not isinstance(item, RemovedItem):
            continue
        pairs.setdefault(str(item.task_id), set()).add(str(item.agent_id))
    return pairs


def _critical_recovery_cost(
    env,
    option: InsertOption,
    *,
    blocked_agents: set[str],
    prefer_truck: bool,
) -> float:
    task = _task(env, option.task_id)
    cost = evaluate_sequence_cost(env, option.agent_id, option.solution.sequence_for(option.agent_id, option.agent_type))
    priority_bonus = float(_critical_priority(task))
    life_ratio = float(_lifeline_ratio(task))
    penalty = 0.0
    if prefer_truck and str(option.agent_type) == "uav":
        penalty += 0.30
    if str(option.agent_id) in blocked_agents:
        penalty += 1.50
    if str(option.agent_type) == "uav":
        st = getattr(env.state, "agents", {}).get(str(option.agent_id), None)
        energy = float(getattr(st, "battery", 0.0))
        penalty += max(float(cost.energy_cost_first + cost.energy_cost_second) - energy, 0.0)
    return float(option.cost_delta + penalty - 0.10 * priority_bonus + 0.05 * life_ratio)


def _critical_recovery_reorder_options(
    env,
    solution: ALNSSolution,
    task_id: StableId,
    *,
    blocked_agents: set[str],
    prefer_truck: bool,
) -> tuple[tuple[InsertOption, ...], tuple[str, ...], dict[str, int]]:
    task = _task(env, task_id)
    if task is None or getattr(task, "status", None) != TaskStatus.PENDING:
        return (), (FAIL_AGENT_STATE_CONFLICT,), {"critical_recovery_rejected_infeasible": 1}
    base_eval = evaluate_k2_solution(env, solution, weights=ObjectiveWeights())
    options: list[InsertOption] = []
    failures: Counter[str] = Counter()
    diagnostics = {
        "critical_recovery_rejected_infeasible": 0,
        "critical_recovery_rejected_no_slot": 0,
        "critical_recovery_rejected_duplicate_claim": 0,
        "critical_recovery_avoided_failed_agent": 0,
    }
    task_priority = float(_critical_priority(task))
    assigned = _assigned_task_ids(solution)
    for aid, atype in _solution_agents(env, solution):
        seq = solution.sequence_for(aid, atype)
        if len(seq) < 2:
            diagnostics["critical_recovery_rejected_no_slot"] += 1
            failures[FAIL_NO_SEQUENCE_CAPACITY] += 1
            continue
        if str(aid) in blocked_agents:
            diagnostics["critical_recovery_avoided_failed_agent"] += 1
            continue
        victim_task = _task(env, seq[1])
        victim_priority = float(_critical_priority(victim_task))
        if _is_timecritical(victim_task) and victim_priority + 1e-9 >= task_priority:
            diagnostics["critical_recovery_rejected_no_slot"] += 1
            continue
        if str(task_id) in assigned and str(task_id) != str(seq[1]):
            diagnostics["critical_recovery_rejected_duplicate_claim"] += 1
            failures[FAIL_DUPLICATE_EXCLUSIVE_TASK] += 1
            continue
        trial = solution.without_task(seq[1]).with_sequence(aid, atype, (seq[0], task_id))
        seq_feas = evaluate_sequence_feasibility(env, aid, trial.sequence_for(aid, atype))
        trial_eval = evaluate_k2_solution(env, trial, weights=ObjectiveWeights())
        if not seq_feas.feasible or not trial_eval.feasible:
            diagnostics["critical_recovery_rejected_infeasible"] += 1
            for reason in _map_sequence_reasons(tuple(seq_feas.reason_codes) + tuple(trial_eval.infeasibility_reasons)):
                failures[reason] += 1
            continue
        options.append(
            InsertOption(
                solution=trial,
                agent_id=aid,
                agent_type=atype,
                task_id=task_id,
                sequence_position=1,
                cost_delta=float(trial_eval.breakdown.total_cost - base_eval.breakdown.total_cost),
            )
        )
    ranked = tuple(
        sorted(
            options,
            key=lambda opt: (
                _critical_recovery_cost(env, opt, blocked_agents=blocked_agents, prefer_truck=prefer_truck),
                str(opt.agent_id),
            ),
        )
    )
    return ranked, tuple(sorted(failures.keys())), diagnostics


def _result_from_steps(env, solution: ALNSSolution, inserted: list[InsertedItem], failures: Counter[str], diagnostics: dict[str, Any]) -> RepairResult:
    diagnostics["failure_reason_counts"] = dict(failures)
    diagnostics["feasibility_rate"] = float(
        diagnostics.get("feasible_candidates", 0)
        / max(diagnostics.get("feasible_candidates", 0) + diagnostics.get("infeasible_candidates", 0), 1)
    )
    diagnostics["_env"] = env
    diagnostics["_evaluation"] = evaluate_k2_solution(env, solution)
    return _repair_result(solution, inserted, failures, diagnostics)


def critical_first_insertion(env, partial_solution: ALNSSolution, removed_items: Iterable[RemovedItem | StableId], rng=None) -> RepairResult:
    cand = partial_solution
    inserted: list[InsertedItem] = []
    failures: Counter[str] = Counter()
    diagnostics: dict[str, Any] = {"attempts": 0, "feasible_candidates": 0, "infeasible_candidates": 0}
    for task_id in _prioritized_task_ids(env, cand, removed_items):
        options, reasons = _insertion_options(env, cand, task_id, weights=ObjectiveWeights())
        diagnostics["attempts"] += 1
        diagnostics["feasible_candidates"] += len(options)
        if not options:
            diagnostics["infeasible_candidates"] += 1
            for reason in reasons or (FAIL_NO_FEASIBLE_INSERTION_POSITION,):
                failures[reason] += 1
            continue
        best = options[0]
        cand = best.solution
        inserted.append(InsertedItem(best.agent_id, best.agent_type, best.task_id, best.sequence_position, best.cost_delta))
    return _result_from_steps(env, cand, inserted, failures, diagnostics)


def _risk_augmented_cost(env, option: InsertOption) -> float:
    task = _task(env, option.task_id)
    cost = evaluate_sequence_cost(env, option.agent_id, option.solution.sequence_for(option.agent_id, option.agent_type))
    road_risk = 1.0 if not np.isfinite(_distance(env, _agent_node(env, option.agent_id), _task_node(task))) else 0.0
    energy_margin = 0.0
    if option.agent_type == "uav":
        st = getattr(env.state, "agents", {}).get(str(option.agent_id), None)
        energy_margin = -float(getattr(st, "battery", 0.0)) + cost.energy_cost_first + cost.energy_cost_second
    switching = 0.1 if int(option.sequence_position) == 0 else 0.0
    support = 0.25 if _support_required(task) else 0.0
    lateness = max(float(cost.completion_first or 0.0) - float(getattr(task, "deadline_step", getattr(task, "deadline", 1e9))), 0.0) / 100.0
    life = float(cost.lifeline_loss_first + cost.lifeline_loss_second)
    return float(option.cost_delta + road_risk + max(energy_margin, 0.0) + switching + support + lateness + life)


def risk_aware_insertion(env, partial_solution: ALNSSolution, removed_items: Iterable[RemovedItem | StableId], rng=None) -> RepairResult:
    cand = partial_solution
    inserted: list[InsertedItem] = []
    failures: Counter[str] = Counter()
    diagnostics: dict[str, Any] = {"attempts": 0, "feasible_candidates": 0, "infeasible_candidates": 0, "risk_terms": []}
    for task_id in _task_ids(removed_items):
        options, reasons = _insertion_options(env, cand, task_id, weights=ObjectiveWeights())
        diagnostics["attempts"] += 1
        diagnostics["feasible_candidates"] += len(options)
        if not options:
            diagnostics["infeasible_candidates"] += 1
            for reason in reasons or (FAIL_NO_FEASIBLE_INSERTION_POSITION,):
                failures[reason] += 1
            continue
        ranked = sorted(options, key=lambda opt: (_risk_augmented_cost(env, opt), str(opt.agent_id), int(opt.sequence_position)))
        best = ranked[0]
        cand = best.solution
        inserted.append(InsertedItem(best.agent_id, best.agent_type, best.task_id, best.sequence_position, _risk_augmented_cost(env, best)))
        diagnostics["risk_terms"].append({"task_id": str(task_id), "risk_augmented_cost": float(_risk_augmented_cost(env, best))})
    return _result_from_steps(env, cand, inserted, failures, diagnostics)


def _first_available(env, kind: AgentKind) -> Optional[str]:
    for aid, st in sorted(getattr(env.state, "agents", {}).items(), key=lambda kv: str(kv[0])):
        if getattr(st, "kind", None) == kind and not bool(getattr(st, "crashed", False)):
            return str(aid)
    return None


def synchronized_insertion(env, partial_solution: ALNSSolution, removed_items: Iterable[RemovedItem | StableId], rng=None) -> RepairResult:
    cand = partial_solution
    inserted: list[InsertedItem] = []
    failures: Counter[str] = Counter()
    diagnostics: dict[str, Any] = {"attempts": 0, "feasible_candidates": 0, "infeasible_candidates": 0, "support_coordination": []}
    for task_id in _task_ids(removed_items):
        diagnostics["attempts"] += 1
        task = _task(env, task_id)
        if task is None or getattr(task, "status", None) != TaskStatus.PENDING or not _is_timecritical(task):
            failures[FAIL_SUPPORT_BINDING_REQUIRED] += 1
            diagnostics["infeasible_candidates"] += 1
            continue
        uav_id = _first_available(env, AgentKind.UAV)
        truck_id = _first_available(env, AgentKind.TRUCK)
        if uav_id is None or truck_id is None:
            failures[FAIL_SUPPORT_BINDING_REQUIRED] += 1
            diagnostics["infeasible_candidates"] += 1
            continue
        uav_seq = cand.sequence_for(uav_id, "uav")
        if len(uav_seq) >= 2 or str(task_id) in _assigned_task_ids(cand):
            failures[FAIL_DUPLICATE_EXCLUSIVE_TASK if str(task_id) in _assigned_task_ids(cand) else FAIL_NO_SEQUENCE_CAPACITY] += 1
            diagnostics["infeasible_candidates"] += 1
            continue
        launch = _agent_node(env, truck_id)
        recovery = _task_node(task)
        if launch is None or recovery is None:
            failures[FAIL_RECOVERY_INFEASIBLE] += 1
            diagnostics["infeasible_candidates"] += 1
            continue
        trial_seq = tuple(list(uav_seq) + [task_id])[:2]
        seq_feas = evaluate_sequence_feasibility(env, uav_id, trial_seq)
        if not seq_feas.feasible:
            for reason in seq_feas.reason_codes:
                if str(reason).find("ENERGY") >= 0:
                    failures[FAIL_UAV_ENERGY_INSUFFICIENT] += 1
                elif str(reason).find("UNREACH") >= 0:
                    failures[FAIL_ROAD_UNREACHABLE] += 1
                else:
                    failures[FAIL_NO_FEASIBLE_INSERTION_POSITION] += 1
            diagnostics["infeasible_candidates"] += 1
            continue
        cost = evaluate_sequence_cost(env, uav_id, trial_seq)
        binding = SupportBinding(uav_id=uav_id, truck_id=truck_id, task_id=task_id, launch_anchor=launch, recovery_anchor=recovery)
        sortie = SortiePlan(
            uav_id=uav_id,
            task_id=task_id,
            launch_anchor=launch,
            recovery_anchor=recovery,
            estimated_launch_step=int(getattr(env.state, "step_index", 0)),
            estimated_service_step=None if cost.eta_first is None else int(np.ceil(float(cost.eta_first))),
            estimated_recovery_step=None if cost.completion_first is None else int(np.ceil(float(cost.completion_first))),
        )
        cand = _solution_with_support(cand, binding=binding, sortie=sortie, uav_id=uav_id, task_id=task_id)
        diagnostics["feasible_candidates"] += 1
        diagnostics["support_coordination"].append(
            {
                "truck_id": str(truck_id),
                "uav_id": str(uav_id),
                "task_id": str(task_id),
                "launch_anchor": int(launch),
                "recovery_anchor": int(recovery),
            }
        )
        inserted.append(InsertedItem(uav_id, "uav", task_id, len(uav_seq), 0.0))
    return _result_from_steps(env, cand, inserted, failures, diagnostics)


def feasibility_restoration_insertion(env, partial_solution: ALNSSolution, removed_items: Iterable[RemovedItem | StableId], rng=None) -> RepairResult:
    cand = partial_solution
    failures: Counter[str] = Counter()
    inserted: list[InsertedItem] = []
    diagnostics: dict[str, Any] = {"attempts": 0, "feasible_candidates": 0, "infeasible_candidates": 0, "restored_sequences": 0}
    for aid, atype in _solution_agents(env, cand):
        seq = cand.sequence_for(aid, atype)
        if len(seq) >= 2 and not evaluate_sequence_feasibility(env, aid, seq).feasible:
            cand = cand.with_sequence(aid, atype, (seq[0],))
            diagnostics["restored_sequences"] += 1
    for task_id in _task_ids(removed_items):
        options, reasons = _insertion_options(env, cand, task_id, weights=ObjectiveWeights())
        diagnostics["attempts"] += 1
        diagnostics["feasible_candidates"] += len(options)
        if not options:
            diagnostics["infeasible_candidates"] += 1
            for reason in reasons or (FAIL_NO_FEASIBLE_INSERTION_POSITION,):
                failures[reason] += 1
            continue
        best = options[0]
        cand = best.solution
        inserted.append(InsertedItem(best.agent_id, best.agent_type, best.task_id, best.sequence_position, best.cost_delta))
    return _result_from_steps(env, cand, inserted, failures, diagnostics)


def critical_recovery_repair_insertion(
    env,
    partial_solution: ALNSSolution,
    removed_items: Iterable[RemovedItem | StableId],
    rng=None,
) -> RepairResult:
    del rng
    cand = partial_solution
    inserted: list[InsertedItem] = []
    failures: Counter[str] = Counter()
    blocked_pairs = _removed_agent_pairs(removed_items)
    prefer_truck = bool(getattr(env.cfg, "alns_critical_recovery_repair_prefer_truck", True))
    avoid_failed_agent = bool(getattr(env.cfg, "alns_critical_recovery_repair_avoid_failed_agent", True))
    diagnostics: dict[str, Any] = {
        "attempts": 0,
        "feasible_candidates": 0,
        "infeasible_candidates": 0,
        "critical_recovery_enabled": bool(getattr(env.cfg, "alns_critical_recovery_repair_enabled", False)),
        "critical_recovery_candidates": 0,
        "critical_recovery_attempts": 0,
        "critical_recovery_direct_insertions": 0,
        "critical_recovery_safe_reorders": 0,
        "critical_recovery_rejected_infeasible": 0,
        "critical_recovery_rejected_no_slot": 0,
        "critical_recovery_rejected_duplicate_claim": 0,
        "critical_recovery_avoided_failed_agent": 0,
        "critical_recovery_task_ids": [],
        "failure_reason_counts": {},
    }
    for task_id in _critical_recovery_candidate_ids(env, cand, removed_items):
        blocked_agents = blocked_pairs.get(str(task_id), set()) if avoid_failed_agent else set()
        diagnostics["critical_recovery_candidates"] += 1
        diagnostics["critical_recovery_attempts"] += 1
        diagnostics["attempts"] += 1
        options, reasons = _insertion_options(env, cand, task_id, weights=ObjectiveWeights())
        ranked_options = tuple(
            sorted(
                (
                    opt
                    for opt in options
                    if str(opt.agent_id) not in blocked_agents
                ),
                key=lambda opt: (
                    _critical_recovery_cost(env, opt, blocked_agents=blocked_agents, prefer_truck=prefer_truck),
                    str(opt.agent_id),
                    int(opt.sequence_position),
                ),
            )
        )
        if blocked_agents and options and not ranked_options:
            diagnostics["critical_recovery_avoided_failed_agent"] += 1
        diagnostics["feasible_candidates"] += len(ranked_options)
        if ranked_options:
            best = ranked_options[0]
            cand = best.solution
            inserted.append(InsertedItem(best.agent_id, best.agent_type, best.task_id, best.sequence_position, best.cost_delta))
            diagnostics["critical_recovery_direct_insertions"] += 1
            diagnostics["critical_recovery_task_ids"].append(str(task_id))
            continue
        diagnostics["infeasible_candidates"] += 1
        reorder_options, reorder_reasons, reorder_diag = _critical_recovery_reorder_options(
            env,
            cand,
            task_id,
            blocked_agents=blocked_agents,
            prefer_truck=prefer_truck,
        )
        for key in (
            "critical_recovery_rejected_infeasible",
            "critical_recovery_rejected_no_slot",
            "critical_recovery_rejected_duplicate_claim",
            "critical_recovery_avoided_failed_agent",
        ):
            diagnostics[key] += int(reorder_diag.get(key, 0))
        if reorder_options:
            best = reorder_options[0]
            cand = best.solution
            inserted.append(InsertedItem(best.agent_id, best.agent_type, best.task_id, best.sequence_position, best.cost_delta))
            diagnostics["critical_recovery_safe_reorders"] += 1
            diagnostics["critical_recovery_task_ids"].append(str(task_id))
            continue
        for reason in (reorder_reasons or reasons or (FAIL_NO_FEASIBLE_INSERTION_POSITION,)):
            failures[reason] += 1
    diagnostics["critical_recovery_task_ids"] = tuple(diagnostics["critical_recovery_task_ids"])
    return _result_from_steps(env, cand, inserted, failures, diagnostics)


ER_DESTROY_POOL = (
    road_disruption_removal,
    critical_task_reassignment_removal,
    support_conflict_removal,
    synchronization_risk_removal,
)

ER_REPAIR_POOL = (
    critical_first_insertion,
    risk_aware_insertion,
    synchronized_insertion,
    feasibility_restoration_insertion,
    critical_recovery_repair_insertion,
)

ER_DESTROY_NAMES = tuple(fn.__name__ for fn in ER_DESTROY_POOL)
ER_REPAIR_NAMES = tuple(fn.__name__ for fn in ER_REPAIR_POOL)
