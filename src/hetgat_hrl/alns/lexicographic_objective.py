from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LexicographicObjective:
    safety_recoverability: float
    critical_lifeline_service: float
    overall_completion: float
    operational_cost: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (
            float(self.safety_recoverability),
            float(self.critical_lifeline_service),
            float(self.overall_completion),
            float(self.operational_cost),
        )


def lexicographic_is_better(lhs: LexicographicObjective, rhs: LexicographicObjective, tolerances: tuple[float, float, float, float] = (1e-9, 1e-9, 1e-9, 1e-9)) -> bool:
    for left, right, tol in zip(lhs.as_tuple(), rhs.as_tuple(), tolerances):
        if left < right - tol:
            return True
        if left > right + tol:
            return False
    return False


def shadow_compare_lexicographic(weighted_current: float, weighted_candidate: float, current: LexicographicObjective, candidate: LexicographicObjective) -> dict[str, object]:
    weighted_prefers_candidate = float(weighted_candidate) < float(weighted_current)
    lex_prefers_candidate = lexicographic_is_better(candidate, current)
    return {
        "weighted_current": float(weighted_current),
        "weighted_candidate": float(weighted_candidate),
        "weighted_prefers_candidate": bool(weighted_prefers_candidate),
        "lexicographic_prefers_candidate": bool(lex_prefers_candidate),
        "ranking_agreement": bool(weighted_prefers_candidate == lex_prefers_candidate),
    }
