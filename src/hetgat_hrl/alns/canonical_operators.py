from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

import numpy as np

from hetgat_hrl.alns.objective import ObjectiveWeights
from hetgat_hrl.alns.sequence import (
    REASON_AGENT_TYPE_INCOMPATIBLE,
    REASON_DUPLICATE_EXCLUSIVE_TASK,
    REASON_INSUFFICIENT_UAV_ENERGY,
    REASON_TARGET_UNREACHABLE,
    evaluate_k2_solution,
    evaluate_sequence_feasibility,
)
from hetgat_hrl.alns.solution import ALNSSolution, StableId
from hetgat_hrl.core.mdp_spec import AgentKind, TaskStatus


REASON_RANDOM_REMOVAL = "RANDOM_REMOVAL"
REASON_WORST_COST_REMOVAL = "WORST_COST_REMOVAL"
REASON_RELATED_REMOVAL = "RELATED_REMOVAL"
REASON_SEQUENCE_SEGMENT_REMOVAL = "SEQUENCE_SEGMENT_REMOVAL"

FAIL_NO_SEQUENCE_CAPACITY = "NO_SEQUENCE_CAPACITY"
FAIL_DUPLICATE_EXCLUSIVE_TASK = "DUPLICATE_EXCLUSIVE_TASK"
FAIL_ROAD_UNREACHABLE = "ROAD_UNREACHABLE"
FAIL_INVENTORY_INSUFFICIENT = "INVENTORY_INSUFFICIENT"
FAIL_UAV_ENERGY_INSUFFICIENT = "UAV_ENERGY_INSUFFICIENT"
FAIL_SUPPORT_BINDING_REQUIRED = "SUPPORT_BINDING_REQUIRED"
FAIL_RECOVERY_INFEASIBLE = "RECOVERY_INFEASIBLE"
FAIL_AGENT_STATE_CONFLICT = "AGENT_STATE_CONFLICT"
FAIL_NO_FEASIBLE_INSERTION_POSITION = "NO_FEASIBLE_INSERTION_POSITION"


@dataclass(frozen=True)
class RemovedItem:
    __slots__ = ("agent_id", "agent_type", "task_id", "sequence_position", "marginal_cost", "reason_code")

    agent_id: StableId
    agent_type: str
    task_id: StableId
    sequence_position: int
    marginal_cost: Optional[float]
    reason_code: str


@dataclass(frozen=True)
class DestroyResult:
    __slots__ = (
        "partial_solution",
        "removed_items",
        "feasible",
        "reason_codes",
        "diagnostics",
    )

    partial_solution: ALNSSolution
    removed_items: tuple[RemovedItem, ...]
    feasible: bool
    reason_codes: tuple[str, ...]
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class InsertedItem:
    __slots__ = ("agent_id", "agent_type", "task_id", "sequence_position", "insertion_cost")

    agent_id: StableId
    agent_type: str
    task_id: StableId
    sequence_position: int
    insertion_cost: float


@dataclass(frozen=True)
class RepairResult:
    __slots__ = (
        "candidate_solution",
        "inserted_items",
        "feasible",
        "reason_codes",
        "diagnostics",
    )

    candidate_solution: ALNSSolution
    inserted_items: tuple[InsertedItem, ...]
    feasible: bool
    reason_codes: tuple[str, ...]
    diagnostics: Mapping[str, Any]


class RelatedRemovalWeights:
    __slots__ = ("distance", "deadline", "category", "agent_type", "support")

    def __init__(
        self,
        distance: float = 1.0,
        deadline: float = 0.75,
        category: float = 0.5,
        agent_type: float = 0.4,
        support: float = 0.25,
    ) -> None:
        object.__setattr__(self, "distance", float(distance))
        object.__setattr__(self, "deadline", float(deadline))
        object.__setattr__(self, "category", float(category))
        object.__setattr__(self, "agent_type", float(agent_type))
        object.__setattr__(self, "support", float(support))

    def __setattr__(self, name: str, value) -> None:
        raise AttributeError("RelatedRemovalWeights is immutable")


@dataclass(frozen=True)
class InsertOption:
    __slots__ = ("solution", "agent_id", "agent_type", "task_id", "sequence_position", "cost_delta")

    solution: ALNSSolution
    agent_id: StableId
    agent_type: str
    task_id: StableId
    sequence_position: int
    cost_delta: float


def _agent_type(env, agent_id: StableId) -> str:
    st = getattr(env.state, "agents", {}).get(str(agent_id), None)
    kind = getattr(st, "kind", None)
    if kind == AgentKind.UAV:
        return "uav"
    return "truck"


def _solution_agents(env, solution: ALNSSolution) -> tuple[tuple[StableId, str], ...]:
    pairs: dict[str, tuple[StableId, str]] = {}
    for aid, _seq in solution.truck_sequences:
        pairs[str(aid)] = (aid, "truck")
    for aid, _seq in solution.uav_sequences:
        pairs[str(aid)] = (aid, "uav")
    for aid in sorted(getattr(env.state, "agents", {}).keys(), key=str):
        pairs.setdefault(str(aid), (aid, _agent_type(env, aid)))
    return tuple(pairs[k] for k in sorted(pairs.keys(), key=str))


def _task(env, task_id: StableId):
    return getattr(env.state, "tasks", {}).get(str(task_id), None)


def _task_node(task) -> Optional[int]:
    node = getattr(task, "demand_node", None)
    if node is None:
        return None
    try:
        return int(node)
    except Exception:
        return None


def _distance(env, a: Optional[int], b: Optional[int]) -> float:
    if a is None or b is None:
        return float("inf")
    if hasattr(env, "_decision_shortest_path_distance"):
        try:
            return float(env._decision_shortest_path_distance(int(a), int(b)))
        except Exception:
            return float("inf")
    return float(abs(int(a) - int(b)))


def _lifeline_ratio(task) -> float:
    cur = float(getattr(task, "lifeline_current", getattr(task, "lifeline", 100.0)))
    init = float(max(getattr(task, "lifeline_init", getattr(task, "lifeline_initial", 100.0)), 1e-9))
    return float(np.clip(cur / init, 0.0, 1.0))


def _support_required(task) -> bool:
    return bool(getattr(task, "requires_support", False) or getattr(task, "support_required", False))


def _sequence_items(solution: ALNSSolution) -> tuple[RemovedItem, ...]:
    out: list[RemovedItem] = []
    for aid, seq in solution.truck_sequences:
        for pos, tid in enumerate(seq):
            out.append(RemovedItem(aid, "truck", tid, int(pos), None, ""))
    for aid, seq in solution.uav_sequences:
        for pos, tid in enumerate(seq):
            out.append(RemovedItem(aid, "uav", tid, int(pos), None, ""))
    return tuple(out)


def _is_removable(env, solution: ALNSSolution, item: RemovedItem) -> bool:
    st = getattr(env.state, "agents", {}).get(str(item.agent_id), None)
    if st is None:
        return False
    if bool(getattr(st, "crashed", False)):
        return False
    task = _task(env, item.task_id)
    if task is None or getattr(task, "status", None) != TaskStatus.PENDING:
        return False
    if item.sequence_position == 0 and bool(getattr(st, "airborne", False)):
        return False
    seq = solution.sequence_for(item.agent_id, item.agent_type)
    trial = tuple(t for idx, t in enumerate(seq) if idx != item.sequence_position)
    feas = evaluate_sequence_feasibility(env, item.agent_id, trial)
    return bool(feas.feasible)


def _remove_position(solution: ALNSSolution, item: RemovedItem, reason_code: str, marginal_cost: Optional[float] = None) -> tuple[ALNSSolution, RemovedItem]:
    seq = solution.sequence_for(item.agent_id, item.agent_type)
    new_seq = tuple(t for idx, t in enumerate(seq) if idx != item.sequence_position)
    removed = RemovedItem(
        agent_id=item.agent_id,
        agent_type=item.agent_type,
        task_id=item.task_id,
        sequence_position=int(item.sequence_position),
        marginal_cost=None if marginal_cost is None else float(marginal_cost),
        reason_code=reason_code,
    )
    return solution.with_sequence(item.agent_id, item.agent_type, new_seq), removed


def _partial_result(solution: ALNSSolution, removed: Iterable[RemovedItem], reason: str, diagnostics: Optional[dict[str, Any]] = None) -> DestroyResult:
    ev = evaluate_k2_solution(None, solution) if False else None
    del ev
    removed_tuple = tuple(removed)
    return DestroyResult(
        partial_solution=solution,
        removed_items=removed_tuple,
        feasible=True,
        reason_codes=(reason,) if removed_tuple else ("NO_REMOVABLE_ITEM",),
        diagnostics=diagnostics or {},
    )


def random_removal(env, solution: ALNSSolution, rng, *, max_remove: int = 1) -> DestroyResult:
    removable = [item for item in _sequence_items(solution) if _is_removable(env, solution, item)]
    if not removable:
        return _partial_result(solution, (), REASON_RANDOM_REMOVAL)
    order = np.asarray(len(removable))
    del order
    indices = list(range(len(removable)))
    rng.shuffle(indices)
    cand = solution
    removed: list[RemovedItem] = []
    for idx in indices[: max(int(max_remove), 1)]:
        item = removable[int(idx)]
        cand, rem = _remove_position(cand, item, REASON_RANDOM_REMOVAL)
        removed.append(rem)
    return _partial_result(cand, removed, REASON_RANDOM_REMOVAL)


def worst_cost_removal(env, solution: ALNSSolution, rng=None, *, max_remove: int = 1, weights: ObjectiveWeights = ObjectiveWeights()) -> DestroyResult:
    before = evaluate_k2_solution(env, solution, weights=weights).breakdown.total_cost
    scored: list[tuple[float, str, RemovedItem, ALNSSolution, float]] = []
    for item in _sequence_items(solution):
        if not _is_removable(env, solution, item):
            continue
        trial, _ = _remove_position(solution, item, REASON_WORST_COST_REMOVAL)
        after = evaluate_k2_solution(env, trial, weights=weights).breakdown.total_cost
        marginal = float(before - after)
        scored.append((marginal, str(item.task_id), item, trial, after))
    if not scored:
        return _partial_result(solution, (), REASON_WORST_COST_REMOVAL, {"objective_before": float(before)})
    scored.sort(key=lambda row: (-float(row[0]), row[1]))
    cand = solution
    removed: list[RemovedItem] = []
    diagnostics = {"objective_before": float(before), "removed": []}
    for marginal, _tid, item, _trial, after in scored[: max(int(max_remove), 1)]:
        cand, rem = _remove_position(cand, item, REASON_WORST_COST_REMOVAL, marginal)
        removed.append(rem)
        diagnostics["removed"].append(
            {
                "task_id": str(item.task_id),
                "objective_after_removal": float(after),
                "marginal_cost": float(marginal),
            }
        )
    return _partial_result(cand, removed, REASON_WORST_COST_REMOVAL, diagnostics)


def _relatedness(env, left: RemovedItem, right: RemovedItem, weights: RelatedRemovalWeights) -> float:
    lt = _task(env, left.task_id)
    rt = _task(env, right.task_id)
    ln = _task_node(lt)
    rn = _task_node(rt)
    d = _distance(env, ln, rn)
    d_norm = 1.0 if not np.isfinite(d) else float(np.clip(d / 10000.0, 0.0, 1.0))
    ldiff = abs(_lifeline_ratio(lt) - _lifeline_ratio(rt)) if lt is not None and rt is not None else 1.0
    category = 0.0 if getattr(lt, "kind", None) == getattr(rt, "kind", None) else 1.0
    agent_type = 0.0 if left.agent_type == right.agent_type else 1.0
    support = 0.0 if _support_required(lt) == _support_required(rt) else 1.0
    return float(
        weights.distance * d_norm
        + weights.deadline * float(ldiff)
        + weights.category * category
        + weights.agent_type * agent_type
        + weights.support * support
    )


def related_removal(
    env,
    solution: ALNSSolution,
    rng,
    *,
    max_remove: int = 2,
    weights: RelatedRemovalWeights = RelatedRemovalWeights(),
) -> DestroyResult:
    removable = [item for item in _sequence_items(solution) if _is_removable(env, solution, item)]
    if not removable:
        return _partial_result(solution, (), REASON_RELATED_REMOVAL)
    seed = removable[int(rng.integers(0, len(removable)))]
    ranked = sorted(removable, key=lambda item: (_relatedness(env, seed, item, weights), str(item.task_id)))
    cand = solution
    removed: list[RemovedItem] = []
    for item in ranked[: max(int(max_remove), 1)]:
        cand, rem = _remove_position(cand, item, REASON_RELATED_REMOVAL)
        removed.append(rem)
    return _partial_result(
        cand,
        removed,
        REASON_RELATED_REMOVAL,
        {"seed_task": str(seed.task_id), "removed_count": int(len(removed))},
    )


def sequence_segment_removal(env, solution: ALNSSolution, rng, *, max_remove: int = 2) -> DestroyResult:
    candidates = []
    for aid, atype in _solution_agents(env, solution):
        seq = solution.sequence_for(aid, atype)
        if not seq:
            continue
        removable_positions = [
            pos
            for pos, tid in enumerate(seq)
            if _is_removable(env, solution, RemovedItem(aid, atype, tid, pos, None, ""))
        ]
        if removable_positions:
            candidates.append((aid, atype, seq, tuple(removable_positions)))
    if not candidates:
        return _partial_result(solution, (), REASON_SEQUENCE_SEGMENT_REMOVAL)
    aid, atype, seq, positions = candidates[int(rng.integers(0, len(candidates)))]
    if len(seq) >= 2 and set(positions) == {0, 1} and int(max_remove) >= 2:
        chosen = (0, 1)
        new_seq: tuple[StableId, ...] = ()
    elif 1 in positions:
        chosen = (1,)
        new_seq = (seq[0],)
    else:
        chosen = (positions[0],)
        new_seq = tuple(t for pos, t in enumerate(seq) if pos not in chosen)
    removed = tuple(
        RemovedItem(aid, atype, seq[pos], int(pos), None, REASON_SEQUENCE_SEGMENT_REMOVAL)
        for pos in chosen
    )
    cand = solution.with_sequence(aid, atype, new_seq)
    return _partial_result(
        cand,
        removed,
        REASON_SEQUENCE_SEGMENT_REMOVAL,
        {
            "agent_id": str(aid),
            "agent_type": atype,
            "sequence_before": [str(x) for x in seq],
            "sequence_after": [str(x) for x in new_seq],
        },
    )


def _map_sequence_reasons(reason_codes: Iterable[str]) -> tuple[str, ...]:
    mapped: list[str] = []
    for reason in reason_codes:
        if reason == REASON_DUPLICATE_EXCLUSIVE_TASK:
            mapped.append(FAIL_DUPLICATE_EXCLUSIVE_TASK)
        elif reason == REASON_TARGET_UNREACHABLE:
            mapped.append(FAIL_ROAD_UNREACHABLE)
        elif reason == REASON_INSUFFICIENT_UAV_ENERGY:
            mapped.append(FAIL_UAV_ENERGY_INSUFFICIENT)
        elif reason == REASON_AGENT_TYPE_INCOMPATIBLE:
            mapped.append(FAIL_AGENT_STATE_CONFLICT)
        else:
            mapped.append(FAIL_NO_FEASIBLE_INSERTION_POSITION)
    return tuple(sorted(set(mapped)))


def _task_ids(items: Iterable[RemovedItem | StableId]) -> tuple[StableId, ...]:
    out: list[StableId] = []
    for item in items:
        out.append(item.task_id if isinstance(item, RemovedItem) else item)
    return tuple(out)


def _insert_sequence(seq: tuple[StableId, ...], task_id: StableId, position: int) -> Optional[tuple[StableId, ...]]:
    if str(task_id) in {str(x) for x in seq}:
        return None
    if len(seq) >= 2:
        return None
    pos = int(max(min(position, len(seq)), 0))
    return tuple(seq[:pos] + (task_id,) + seq[pos:])


def _insertion_options(
    env,
    solution: ALNSSolution,
    task_id: StableId,
    *,
    weights: ObjectiveWeights,
) -> tuple[tuple[InsertOption, ...], tuple[str, ...]]:
    task = _task(env, task_id)
    if task is None or getattr(task, "status", None) != TaskStatus.PENDING:
        return (), (FAIL_AGENT_STATE_CONFLICT,)
    base_eval = evaluate_k2_solution(env, solution, weights=weights)
    options: list[InsertOption] = []
    failures: Counter[str] = Counter()
    for aid, atype in _solution_agents(env, solution):
        seq = solution.sequence_for(aid, atype)
        if len(seq) >= 2:
            failures[FAIL_NO_SEQUENCE_CAPACITY] += 1
            continue
        for pos in range(len(seq) + 1):
            new_seq = _insert_sequence(seq, task_id, pos)
            if new_seq is None:
                failures[FAIL_DUPLICATE_EXCLUSIVE_TASK] += 1
                continue
            trial = solution.with_sequence(aid, atype, new_seq)
            seq_feas = evaluate_sequence_feasibility(env, aid, new_seq)
            trial_eval = evaluate_k2_solution(env, trial, weights=weights)
            if not seq_feas.feasible or not trial_eval.feasible:
                for reason in _map_sequence_reasons(tuple(seq_feas.reason_codes) + tuple(trial_eval.infeasibility_reasons)):
                    failures[reason] += 1
                continue
            options.append(
                InsertOption(
                    solution=trial,
                    agent_id=aid,
                    agent_type=atype,
                    task_id=task_id,
                    sequence_position=int(pos),
                    cost_delta=float(trial_eval.breakdown.total_cost - base_eval.breakdown.total_cost),
                )
            )
    if not options and not failures:
        failures[FAIL_NO_FEASIBLE_INSERTION_POSITION] += 1
    return tuple(sorted(options, key=lambda opt: (float(opt.cost_delta), str(opt.agent_id), int(opt.sequence_position)))), tuple(
        sorted(failures.keys())
    )


def _repair_result(solution: ALNSSolution, inserted: Iterable[InsertedItem], failures: Counter[str], diagnostics: dict[str, Any]) -> RepairResult:
    ev = diagnostics.get("_evaluation")
    feasible = bool(getattr(ev, "feasible", evaluate_k2_solution(diagnostics["_env"], solution).feasible if "_env" in diagnostics else True))
    reason_codes = tuple(sorted(set(failures.keys())))
    clean_diagnostics = {k: v for k, v in diagnostics.items() if not str(k).startswith("_")}
    return RepairResult(
        candidate_solution=solution,
        inserted_items=tuple(inserted),
        feasible=bool(feasible),
        reason_codes=reason_codes,
        diagnostics=clean_diagnostics,
    )


def greedy_insertion(
    env,
    partial_solution: ALNSSolution,
    removed_items: Iterable[RemovedItem | StableId],
    rng=None,
    *,
    weights: ObjectiveWeights = ObjectiveWeights(),
) -> RepairResult:
    cand = partial_solution
    inserted: list[InsertedItem] = []
    failures: Counter[str] = Counter()
    diagnostics = {"attempts": 0, "feasible_candidates": 0, "infeasible_candidates": 0, "failure_reason_counts": {}}
    for task_id in _task_ids(removed_items):
        options, reasons = _insertion_options(env, cand, task_id, weights=weights)
        diagnostics["attempts"] += 1
        diagnostics["feasible_candidates"] += len(options)
        if not options:
            diagnostics["infeasible_candidates"] += 1
            for r in reasons or (FAIL_NO_FEASIBLE_INSERTION_POSITION,):
                failures[r] += 1
            continue
        best = options[0]
        cand = best.solution
        inserted.append(InsertedItem(best.agent_id, best.agent_type, best.task_id, best.sequence_position, best.cost_delta))
    diagnostics["failure_reason_counts"] = dict(failures)
    diagnostics["feasibility_rate"] = float(
        diagnostics["feasible_candidates"]
        / max(diagnostics["feasible_candidates"] + diagnostics["infeasible_candidates"], 1)
    )
    diagnostics["_env"] = env
    diagnostics["_evaluation"] = evaluate_k2_solution(env, cand, weights=weights)
    return _repair_result(cand, inserted, failures, diagnostics)


def _regret_insertion(
    env,
    partial_solution: ALNSSolution,
    removed_items: Iterable[RemovedItem | StableId],
    *,
    regret_k: int,
    weights: ObjectiveWeights,
) -> RepairResult:
    cand = partial_solution
    remaining = list(_task_ids(removed_items))
    inserted: list[InsertedItem] = []
    failures: Counter[str] = Counter()
    diagnostics = {
        "attempts": 0,
        "feasible_candidates": 0,
        "infeasible_candidates": 0,
        "failure_reason_counts": {},
        "regret_k": int(regret_k),
        "degenerate_regret_count": 0,
    }
    while remaining:
        ranked: list[tuple[float, str, InsertOption, tuple[str, ...], int]] = []
        failed_tasks: dict[str, tuple[str, ...]] = {}
        for task_id in remaining:
            options, reasons = _insertion_options(env, cand, task_id, weights=weights)
            diagnostics["attempts"] += 1
            diagnostics["feasible_candidates"] += len(options)
            if not options:
                diagnostics["infeasible_candidates"] += 1
                failed_tasks[str(task_id)] = reasons or (FAIL_NO_FEASIBLE_INSERTION_POSITION,)
                continue
            costs = [float(opt.cost_delta) for opt in options]
            best = options[0]
            if len(costs) < regret_k:
                diagnostics["degenerate_regret_count"] += 1
                regret = float(1e6 + (regret_k - len(costs)) * 1e3 - costs[0])
            elif regret_k == 2:
                regret = float(costs[1] - costs[0])
            else:
                regret = float((costs[1] - costs[0]) + (costs[2] - costs[0]))
            ranked.append((regret, str(task_id), best, reasons, len(options)))
        if not ranked:
            for reasons in failed_tasks.values():
                for r in reasons:
                    failures[r] += 1
            break
        ranked.sort(key=lambda row: (-float(row[0]), row[1]))
        _regret, tid, best, _reasons, _num = ranked[0]
        cand = best.solution
        inserted.append(InsertedItem(best.agent_id, best.agent_type, best.task_id, best.sequence_position, best.cost_delta))
        remaining = [task_id for task_id in remaining if str(task_id) != tid]
    diagnostics["failure_reason_counts"] = dict(failures)
    diagnostics["feasibility_rate"] = float(
        diagnostics["feasible_candidates"]
        / max(diagnostics["feasible_candidates"] + diagnostics["infeasible_candidates"], 1)
    )
    diagnostics["_env"] = env
    diagnostics["_evaluation"] = evaluate_k2_solution(env, cand, weights=weights)
    return _repair_result(cand, inserted, failures, diagnostics)


def regret_2_insertion(
    env,
    partial_solution: ALNSSolution,
    removed_items: Iterable[RemovedItem | StableId],
    rng=None,
    *,
    weights: ObjectiveWeights = ObjectiveWeights(),
) -> RepairResult:
    return _regret_insertion(env, partial_solution, removed_items, regret_k=2, weights=weights)


def regret_3_insertion(
    env,
    partial_solution: ALNSSolution,
    removed_items: Iterable[RemovedItem | StableId],
    rng=None,
    *,
    weights: ObjectiveWeights = ObjectiveWeights(),
) -> RepairResult:
    return _regret_insertion(env, partial_solution, removed_items, regret_k=3, weights=weights)


CANONICAL_DESTROY_POOL = (
    random_removal,
    worst_cost_removal,
    related_removal,
    sequence_segment_removal,
)

CANONICAL_REPAIR_POOL = (
    greedy_insertion,
    regret_2_insertion,
    regret_3_insertion,
)

CANONICAL_DESTROY_NAMES = tuple(fn.__name__ for fn in CANONICAL_DESTROY_POOL)
CANONICAL_REPAIR_NAMES = tuple(fn.__name__ for fn in CANONICAL_REPAIR_POOL)
