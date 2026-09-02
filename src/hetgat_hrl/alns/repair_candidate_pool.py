from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from hetgat_hrl.alns.canonical_operators import (
    InsertedItem,
    RemovedItem,
    RepairResult,
    _insertion_options,
    _repair_result,
    _task,
    _task_ids,
)
from hetgat_hrl.alns.objective import ObjectiveWeights
from hetgat_hrl.alns.sequence import evaluate_k2_solution
from hetgat_hrl.alns.solution import ALNSSolution, StableId


FEATURE_COLUMNS = (
    "sequence_position",
    "deadline_slack",
    "lifeline_value",
    "payload",
    "battery_reserve",
    "recovery_reserve",
    "support_conflict",
    "road_damage_probability",
    "road_blocked",
    "weather_severity",
    "wind_speed",
    "rain_intensity",
    "visibility",
    "temperature",
    "travel_estimate",
)


@dataclass(frozen=True)
class RepairCandidate:
    candidate_id: str
    solution: ALNSSolution
    task_id: StableId
    agent_id: StableId
    position: int
    operator_name: str
    features: tuple[float, ...]
    rank_before: int
    insertion_cost: float

    def to_feature_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in zip(FEATURE_COLUMNS, self.features)}


@dataclass(frozen=True)
class RepairCandidatePool:
    pool_id: str
    candidates: tuple[RepairCandidate, ...]
    operator_name: str
    raw_candidate_count: int
    duplicate_digest_count: int
    failure_reason_counts: dict[str, int]


def _deadline_slack(env: Any, task: Any) -> float:
    step = int(getattr(getattr(env, "state", None), "step_index", 0))
    return float(getattr(task, "deadline_step", step) - step) if task is not None else 0.0


def _payload(task: Any) -> float:
    return float(getattr(task, "demand_kg", getattr(task, "payload_kg", 0.0)) or 0.0) if task is not None else 0.0


def _lifeline(task: Any) -> float:
    return float(getattr(task, "lifeline_value", getattr(task, "priority", 0.0)) or 0.0) if task is not None else 0.0


def _battery_reserve(env: Any, agent_id: StableId) -> float:
    st = getattr(getattr(env, "state", None), "agents", {}).get(str(agent_id), None)
    return float(getattr(st, "battery", 0.0) or 0.0)


def _weather_features(env: Any) -> tuple[float, float, float, float, float]:
    physical = getattr(env, "physical_v2", None)
    ledger = getattr(physical, "weather_ledger", []) if physical is not None else []
    if ledger:
        w = ledger[-1]
        severity = 1.0 if bool(getattr(w, "no_fly_status", False)) else 0.0
        return (
            float(severity),
            float(getattr(w, "wind_speed", 0.0)),
            float(getattr(w, "rain_intensity", 0.0)),
            float(getattr(w, "visibility", 10.0)),
            float(getattr(w, "temperature", 20.0)),
        )
    return (0.0, 0.0, 0.0, 10.0, 20.0)


def _features(env: Any, cand: RepairCandidate | None, *, task: Any, agent_id: StableId, position: int, insertion_cost: float) -> tuple[float, ...]:
    del cand
    weather_severity, wind, rain, visibility, temp = _weather_features(env)
    blocked = float(getattr(getattr(getattr(env, "state", None), "hazard", None), "blocked_ratio", 0.0))
    return (
        float(position),
        _deadline_slack(env, task),
        _lifeline(task),
        _payload(task),
        _battery_reserve(env, agent_id),
        float(getattr(env, "physical_v2_minimum_energy_reserve_seen", 0.0) or 0.0),
        0.0,
        float(blocked),
        float(blocked > 0.0),
        weather_severity,
        wind,
        rain,
        visibility,
        temp,
        float(max(insertion_cost, 0.0)),
    )


def _candidate_digest(solution: ALNSSolution, task_id: StableId, agent_id: StableId, position: int, operator_name: str) -> str:
    payload = {
        "solution": solution.to_stable_dict(),
        "task_id": str(task_id),
        "agent_id": str(agent_id),
        "position": int(position),
        "operator_name": str(operator_name),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")).hexdigest()


def enumerate_repair_candidate_pool(
    env: Any,
    partial_solution: ALNSSolution,
    removed_items: Iterable[RemovedItem | StableId],
    operator_name: str,
    *,
    pool_size: int,
    weights: ObjectiveWeights = ObjectiveWeights(),
) -> RepairCandidatePool:
    candidates: list[RepairCandidate] = []
    failures: dict[str, int] = {}
    seen: set[str] = set()
    duplicate = 0
    raw = 0
    for task_id in _task_ids(removed_items):
        options, reasons = _insertion_options(env, partial_solution, task_id, weights=weights)
        if not options:
            for reason in reasons:
                failures[str(reason)] = int(failures.get(str(reason), 0) + 1)
            continue
        for opt in options:
            raw += 1
            digest = _candidate_digest(opt.solution, opt.task_id, opt.agent_id, int(opt.sequence_position), operator_name)
            if digest in seen:
                duplicate += 1
                continue
            seen.add(digest)
            task = _task(env, opt.task_id)
            candidates.append(
                RepairCandidate(
                    candidate_id=digest,
                    solution=opt.solution,
                    task_id=opt.task_id,
                    agent_id=opt.agent_id,
                    position=int(opt.sequence_position),
                    operator_name=str(operator_name),
                    features=_features(
                        env,
                        None,
                        task=task,
                        agent_id=opt.agent_id,
                        position=int(opt.sequence_position),
                        insertion_cost=float(opt.cost_delta),
                    ),
                    rank_before=len(candidates),
                    insertion_cost=float(opt.cost_delta),
                )
            )
    candidates.sort(key=lambda c: (float(c.insertion_cost), str(c.agent_id), int(c.position), str(c.task_id)))
    candidates = [
        RepairCandidate(
            candidate_id=c.candidate_id,
            solution=c.solution,
            task_id=c.task_id,
            agent_id=c.agent_id,
            position=c.position,
            operator_name=c.operator_name,
            features=c.features,
            rank_before=i,
            insertion_cost=c.insertion_cost,
        )
        for i, c in enumerate(candidates[: max(int(pool_size), 0)])
    ]
    pool_payload = {
        "operator_name": str(operator_name),
        "candidate_ids": [c.candidate_id for c in candidates],
        "step": int(getattr(getattr(env, "state", None), "step_index", 0)),
    }
    pool_id = hashlib.sha256(json.dumps(pool_payload, sort_keys=True).encode("utf-8")).hexdigest()
    return RepairCandidatePool(pool_id, tuple(candidates), str(operator_name), int(raw), int(duplicate), failures)


def repair_result_from_candidate(env: Any, candidate: RepairCandidate, pool: RepairCandidatePool) -> RepairResult:
    inserted = (InsertedItem(candidate.agent_id, "uav" if "uav" in str(candidate.agent_id).lower() else "truck", candidate.task_id, candidate.position, candidate.insertion_cost),)
    diagnostics = {
        "attempts": int(len(pool.candidates)),
        "feasible_candidates": 1,
        "infeasible_candidates": 0,
        "failure_reason_counts": dict(pool.failure_reason_counts),
        "pool_id": str(pool.pool_id),
        "pool_size": int(len(pool.candidates)),
        "_env": env,
        "_evaluation": evaluate_k2_solution(env, candidate.solution),
    }
    return _repair_result(candidate.solution, inserted, {}, diagnostics)


def select_ranker_candidates(
    candidates: Sequence[RepairCandidate],
    scores: Sequence[float],
    *,
    exact_check_budget: int,
    exploration_count: int,
    rng,
) -> tuple[RepairCandidate, ...]:
    budget = int(max(exact_check_budget, 0))
    explore = int(max(min(exploration_count, budget), 0))
    top_count = int(max(budget - explore, 0))
    paired = list(zip(candidates, scores, range(len(candidates))))
    paired.sort(key=lambda row: (-float(row[1]), int(row[2])))
    selected: list[RepairCandidate] = [c for c, _s, _i in paired[:top_count]]
    remaining = [c for c, _s, _i in paired[top_count:] if c not in selected]
    if explore > 0 and remaining:
        order = list(range(len(remaining)))
        try:
            rng.shuffle(order)
        except Exception:
            order = order
        for idx in order[:explore]:
            selected.append(remaining[int(idx)])
    return tuple(selected[:budget])
