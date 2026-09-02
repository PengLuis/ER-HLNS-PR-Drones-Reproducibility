from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class CandidateLearningRecord:
    scenario: str
    step: int
    agent_type: str
    task_type: str
    sequence_position: int
    operator: str
    road_disruption_features: dict[str, float]
    weather_features: dict[str, float]
    deadline_slack: float
    lifeline_value: float
    payload: float
    battery_reserve: float
    recovery_reserve: float
    support_conflict: bool
    objective_before: float
    objective_after: float
    delta_objective: float
    feasible: bool
    failure_reason: str
    accepted: bool
    improved: bool
    runtime: float


class CandidateLearningDatasetRecorder:
    def __init__(self) -> None:
        self.records: list[CandidateLearningRecord] = []

    def record(self, record: CandidateLearningRecord) -> None:
        self.records.append(record)

    def schema(self) -> dict[str, str]:
        return {k: str(v) for k, v in CandidateLearningRecord.__annotations__.items()}

    def export_schema(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.schema(), indent=2, sort_keys=True), encoding="utf-8")

    def export_dataset(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [asdict(r) for r in self.records]
        if path.suffix.lower() == ".parquet":
            try:
                import pandas as pd

                pd.DataFrame(rows).to_parquet(path, index=False)
                return
            except Exception:
                path = path.with_suffix(".jsonl")
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")


class FeasibilityCandidateRanker:
    def __init__(self, model: Any | None = None, feature_columns: Sequence[str] | None = None) -> None:
        self.model = model
        self.feature_columns = tuple(feature_columns or ())

    def _feature_row(self, candidate: Any) -> list[float]:
        if isinstance(candidate, dict):
            return [float(candidate.get(c, 0.0) or 0.0) for c in self.feature_columns]
        return [float(getattr(candidate, c, 0.0) or 0.0) for c in self.feature_columns]

    def score_candidates(self, candidates: Sequence[Any], context: Any | None = None) -> list[float]:
        if self.model is None or not self.feature_columns:
            return [float(getattr(c, "predicted_feasibility", 0.0) if not isinstance(c, dict) else c.get("predicted_feasibility", 0.0)) for c in candidates]
        try:
            x = [self._feature_row(c) for c in candidates]
            if hasattr(self.model, "predict_proba"):
                return [float(v[1]) for v in self.model.predict_proba(x)]
            return [float(v) for v in self.model.predict(x)]
        except Exception:
            return [0.0 for _ in candidates]

    def rank_candidates(self, candidates: Sequence[Any], context: Any | None = None) -> list[Any]:
        try:
            scores = self.score_candidates(candidates, context)
            paired = list(zip(candidates, scores, range(len(candidates))))
            return [c for c, _s, _i in sorted(paired, key=lambda x: (-float(x[1]), int(x[2])))]
        except Exception:
            return list(candidates)


def candidate_record_to_flat_features(record: CandidateLearningRecord) -> dict[str, float]:
    road = record.road_disruption_features
    weather = record.weather_features
    return {
        "sequence_position": float(record.sequence_position),
        "deadline_slack": float(record.deadline_slack),
        "lifeline_value": float(record.lifeline_value),
        "payload": float(record.payload),
        "battery_reserve": float(record.battery_reserve),
        "recovery_reserve": float(record.recovery_reserve),
        "support_conflict": float(bool(record.support_conflict)),
        "road_damage_probability": float(road.get("damage_probability", 0.0)),
        "road_blocked": float(road.get("blocked", 0.0)),
        "weather_severity": float(weather.get("severity", 0.0)),
        "wind_speed": float(weather.get("wind_speed", 0.0)),
        "rain_intensity": float(weather.get("rain_intensity", 0.0)),
        "visibility": float(weather.get("visibility", 10.0)),
        "temperature": float(weather.get("temperature", 20.0)),
        "travel_estimate": float(road.get("travel_estimate", 0.0)),
    }


def nonfinite_feature_count(row: dict[str, Any]) -> int:
    out = 0
    for value in row.values():
        try:
            if not math.isfinite(float(value)):
                out += 1
        except Exception:
            continue
    return out


def ranker_active_select(
    candidates: Sequence[Any],
    ranker: FeasibilityCandidateRanker,
    *,
    exact_feasibility,
    top_m: int,
    exploration_count: int = 1,
) -> tuple[list[Any], dict[str, int]]:
    """Rank candidates, then apply exact feasibility to a fixed evaluation budget."""
    budget = int(max(0, int(top_m)) + max(0, int(exploration_count)))
    ranked = ranker.rank_candidates(candidates)
    selected = list(ranked[: max(0, int(top_m))])
    for cand in candidates:
        if len(selected) >= budget:
            break
        if cand not in selected:
            selected.append(cand)
    feasible = [cand for cand in selected if bool(exact_feasibility(cand))]
    return feasible, {
        "candidate_count": int(len(candidates)),
        "ranked_count": int(len(ranked)),
        "exact_feasibility_budget": int(budget),
        "exact_feasibility_evaluations": int(len(selected)),
        "feasible_count": int(len(feasible)),
    }


def ranker_shadow_scores(candidates: Sequence[Any], ranker: FeasibilityCandidateRanker) -> tuple[list[float], float]:
    start = time.perf_counter()
    scores = ranker.score_candidates(candidates)
    return scores, float(time.perf_counter() - start)


class ContextualOperatorController:
    def select(self, context: dict[str, float], fallback: tuple[str, str] = ("random_removal", "greedy_insertion")) -> dict[str, Any]:
        try:
            road = float(context.get("road_damage_ratio", 0.0))
            weather = float(context.get("weather_severity", 0.0))
            support = float(context.get("support_conflict", 0.0))
            stagnation = float(context.get("stagnation", 0.0))
            if support > 0.5:
                destroy, repair = "support_conflict_removal", "synchronized_insertion"
            elif road > 0.35 or weather > 0.6:
                destroy, repair = "road_disruption_removal", "risk_aware_insertion"
            elif stagnation > 0.5:
                destroy, repair = "worst_cost_removal", "greedy_insertion"
            else:
                destroy, repair = fallback
            return {"destroy_operator": destroy, "repair_operator": repair, "destroy_degree": 0.25, "local_search_budget": 0}
        except Exception:
            return {"destroy_operator": fallback[0], "repair_operator": fallback[1], "destroy_degree": 0.2, "local_search_budget": 0}


class AdaptiveHorizonController:
    def choose_k(self, context: dict[str, float]) -> int:
        uncertainty = max(float(context.get("road_damage_ratio", 0.0)), float(context.get("weather_severity", 0.0)))
        stability = float(context.get("recent_feasible_rate", 0.0)) - float(context.get("stagnation", 0.0))
        if uncertainty >= 0.65:
            return 1
        if stability >= 0.75:
            return 3
        return 2


def timed_candidate_record(**kwargs: Any) -> CandidateLearningRecord:
    kwargs.setdefault("runtime", time.perf_counter())
    return CandidateLearningRecord(**kwargs)
