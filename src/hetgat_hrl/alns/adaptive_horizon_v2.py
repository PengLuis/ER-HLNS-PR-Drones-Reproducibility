from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class AdaptiveHorizonFeatures:
    road_damage_ratio: float = 0.0
    blocked_edge_ratio: float = 0.0
    degraded_edge_ratio: float = 0.0
    weather_severity: float = 0.0
    no_fly_ratio: float = 0.0
    wind_speed: float = 0.0
    rain_intensity: float = 0.0
    visibility: float = 1.0
    battery_reserve: float = 1.0
    recovery_reserve: float = 1.0
    recent_feasible_repair_rate: float = 1.0
    recent_shield_intervention_rate: float = 0.0
    recent_stagnation: float = 0.0
    support_conflict_rate: float = 0.0
    critical_task_ratio: float = 0.0
    scenario_scale: float = 1.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AdaptiveHorizonFeatures":
        data = {}
        for field in cls.__dataclass_fields__:
            raw = value.get(field, getattr(cls(), field))
            try:
                data[field] = float(raw)
            except Exception:
                data[field] = float(getattr(cls(), field))
        return cls(**data)

    def to_dict(self) -> dict[str, float]:
        return {field: float(getattr(self, field)) for field in self.__dataclass_fields__}


@dataclass(frozen=True)
class AdaptiveHorizonDecision:
    chosen_k: int
    reason_codes: tuple[str, ...]
    confidence: float
    risk_score: float
    features: AdaptiveHorizonFeatures

    def to_record(self) -> dict[str, Any]:
        return {
            "chosen_K": int(self.chosen_k),
            "reason_codes": "|".join(self.reason_codes),
            "confidence": float(self.confidence),
            "risk_score": float(self.risk_score),
            "risk_features": self.features.to_dict(),
        }


class AdaptiveHorizonControllerV2:
    """Rule-based K=1/2 controller for V2 diagnostics and active variants."""

    allowed_values = (1, 2)

    def __init__(
        self,
        *,
        high_risk_threshold: float = 0.62,
        low_repair_rate_threshold: float = 0.45,
        low_reserve_threshold: float = 0.25,
    ) -> None:
        self.high_risk_threshold = float(high_risk_threshold)
        self.low_repair_rate_threshold = float(low_repair_rate_threshold)
        self.low_reserve_threshold = float(low_reserve_threshold)

    def decide(self, features: AdaptiveHorizonFeatures | Mapping[str, Any]) -> AdaptiveHorizonDecision:
        f = features if isinstance(features, AdaptiveHorizonFeatures) else AdaptiveHorizonFeatures.from_mapping(features)
        weather = max(float(f.weather_severity), min(float(f.wind_speed) / 25.0, 1.0), float(f.rain_intensity), 1.0 - float(f.visibility))
        road = max(float(f.road_damage_ratio), float(f.blocked_edge_ratio), 0.5 * float(f.degraded_edge_ratio))
        reserve_pressure = max(1.0 - float(f.battery_reserve), 1.0 - float(f.recovery_reserve))
        repair_pressure = 1.0 - float(f.recent_feasible_repair_rate)
        safety_pressure = max(float(f.recent_shield_intervention_rate), float(f.no_fly_ratio))
        coordination_pressure = max(float(f.support_conflict_rate), float(f.recent_stagnation))
        mission_pressure = 0.5 * float(f.critical_task_ratio) + 0.2 * min(float(f.scenario_scale), 3.0) / 3.0
        risk_score = float(
            np.clip(
                0.24 * road
                + 0.20 * weather
                + 0.18 * reserve_pressure
                + 0.16 * repair_pressure
                + 0.12 * safety_pressure
                + 0.06 * coordination_pressure
                + 0.04 * mission_pressure,
                0.0,
                1.0,
            )
        )
        reasons: list[str] = []
        if road >= 0.40:
            reasons.append("ROAD_UNCERTAINTY_HIGH")
        if weather >= 0.55:
            reasons.append("WEATHER_RISK_HIGH")
        if reserve_pressure >= 1.0 - self.low_reserve_threshold:
            reasons.append("RECOVERY_RESERVE_LOW")
        if float(f.recent_feasible_repair_rate) <= self.low_repair_rate_threshold:
            reasons.append("RECENT_REPAIR_RATE_LOW")
        if safety_pressure >= 0.35:
            reasons.append("SHIELD_PRESSURE_HIGH")
        if coordination_pressure >= 0.50:
            reasons.append("COORDINATION_STAGNATION")
        chosen = 1 if risk_score >= self.high_risk_threshold or reasons else 2
        if not reasons:
            reasons.append("STABLE_USE_K2")
        confidence = float(np.clip(abs(risk_score - self.high_risk_threshold) / max(self.high_risk_threshold, 1e-9), 0.10, 1.0))
        return AdaptiveHorizonDecision(
            chosen_k=int(chosen),
            reason_codes=tuple(reasons),
            confidence=confidence,
            risk_score=risk_score,
            features=f,
        )
