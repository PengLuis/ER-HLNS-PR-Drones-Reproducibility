from __future__ import annotations

from typing import Mapping

from hetgat_hrl.alns.solution import StableId


def legacy_goal_feasible_with_checker(env, goals: Mapping[StableId, StableId | None], checker) -> bool:
    return bool(checker(env, dict(goals)))
