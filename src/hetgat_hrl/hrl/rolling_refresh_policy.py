from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class RefreshDecision:
    refresh: bool
    flags: Dict[str, Any]


def build_no_event_refresh_decision(
    *,
    since_last: int,
    decision_interval: int,
    has_goals: bool,
    hard_reason_totals: Dict[str, Any],
    hard_seen_count_total: int,
    hard_actionable_count_total: int,
    hard_deferred_count_total: int,
    hard_immediate_refresh_count_total: int,
) -> RefreshDecision:
    do_refresh = bool(since_last >= decision_interval or (not has_goals))
    flags: Dict[str, Any] = {
        "interval": bool(since_last >= decision_interval),
        "empty_goals": bool(not has_goals),
        "risk_spike": False,
        "resolution": False,
        "arrival": False,
        "goal_invalid": False,
        "normal_stall": False,
        "high_priority_uncovered": False,
        "truck_dead_end": False,
        "truck_idle": False,
        "uav_emergency": False,
        "uav_idle": False,
        "map_update": False,
        "map_update_light": False,
        "map_update_hard_seen": False,
        "map_update_hard_actionable": False,
        "map_update_hard_deferred": False,
        "map_update_hard_immediate_refresh": False,
        "map_update_hard_seen_step": 0,
        "map_update_hard_actionable_step": 0,
        "map_update_hard_deferred_step": 0,
        "map_update_hard_immediate_refresh_step": 0,
        "map_update_hard_seen_count_total": int(hard_seen_count_total),
        "map_update_hard_actionable_count_total": int(hard_actionable_count_total),
        "map_update_hard_deferred_count_total": int(hard_deferred_count_total),
        "map_update_hard_immediate_refresh_count_total": int(hard_immediate_refresh_count_total),
        "map_update_hard_reason_path_blocked_step": 0,
        "map_update_hard_reason_goal_unreachable_step": 0,
        "map_update_hard_reason_ranking_changed_step": 0,
        "map_update_hard_reason_dead_end_step": 0,
        "map_update_hard_reason_recovery_path_fractured_step": 0,
        "map_update_hard_reason_path_blocked_total": int(hard_reason_totals.get("path_blocked", 0)),
        "map_update_hard_reason_goal_unreachable_total": int(hard_reason_totals.get("goal_unreachable", 0)),
        "map_update_hard_reason_ranking_changed_total": int(hard_reason_totals.get("ranking_changed", 0)),
        "map_update_hard_reason_dead_end_total": int(hard_reason_totals.get("dead_end", 0)),
        "map_update_hard_reason_recovery_path_fractured_total": int(
            hard_reason_totals.get("recovery_path_fractured", 0)
        ),
        "high_value_event": False,
        "low_value_event": False,
        "low_value_event_streak": 0,
        "cooldown_blocked": False,
        "event_budget_blocked": False,
        "event_replans_in_window": 0,
        "event_first_enabled": False,
        "no_event_fallback_refresh": False,
        "hard_event_refresh": False,
        "hard_reason_assigned_but_not_progressing": False,
        "refresh": bool(do_refresh),
    }
    return RefreshDecision(refresh=do_refresh, flags=flags)
