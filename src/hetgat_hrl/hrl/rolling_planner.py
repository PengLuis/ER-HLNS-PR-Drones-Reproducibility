from __future__ import annotations

import heapq
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
except Exception:  # pragma: no cover - scipy optional
    linear_sum_assignment = None

from hetgat_hrl.core.mdp_spec import AgentKind, TaskKind, TaskStatus
from hetgat_hrl.core.runtime_constants import DEPOT_DOCK_ID
from hetgat_hrl.core.algorithm_profile import (
    er_hlns_coordination_active,
    er_hlns_route_plan_active,
)
from hetgat_hrl.hrl.hierarchical_route_plan import HierarchicalRoutePlanManager
from hetgat_hrl.hrl.rolling_refresh_policy import build_no_event_refresh_decision
from hetgat_hrl.hrl.rolling_planner_runtime import init_planner_runtime_state
from hetgat_hrl.hrl.rolling_planner_types import RollingPlannerState, RollingPlannerWeights


class EventTriggeredRollingPlanner:
    """
    Deterministic non-RL rolling planner.
    - Event-triggered refresh (interval/risk/resolution/arrival/dead-end/emergency)
    - Greedy assignment in configurable order (UAV-first or Truck-first)
    - Score-based selection over existing state/env fields only
    """

    def __init__(
        self,
        decision_interval: int = 5,
        seed: int = 0,
        weights: Optional[RollingPlannerWeights] = None,
        use_risk_term: bool = True,
        use_rth_repair: bool = True,
        use_event_trigger: bool = True,
        use_keep_goal_bonus: bool = True,
        rth_safety_factor: Optional[float] = None,
        service_battery_buffer: float = 0.02,
        max_uav_wind_mps: Optional[float] = 14.0,
        max_uav_rainfall_mmh: Optional[float] = 24.0,
        max_uav_node_risk: Optional[float] = 1.50,
        replan_cooldown_steps: int = 4,
        min_goal_hold_steps: int = 8,
        switch_margin: float = 0.12,
        short_sortie_max_distance_m: float = 650.0,
        short_sortie_min_battery: float = 0.40,
        stable_goal_before_takeoff_steps: int = 4,
        recovery_trigger_factor: float = 1.20,
        assignment_order: str = "uav_first",
    ) -> None:
        self.decision_interval = max(int(decision_interval), 1)
        self.state = RollingPlannerState()
        self.weights = weights if weights is not None else RollingPlannerWeights()
        self.use_risk_term = bool(use_risk_term)
        self.use_rth_repair = bool(use_rth_repair)
        self.use_event_trigger = bool(use_event_trigger)
        self.use_keep_goal_bonus = bool(use_keep_goal_bonus)
        self.rth_safety_factor = None if rth_safety_factor is None else float(max(rth_safety_factor, 0.0))
        self.service_battery_buffer = float(max(service_battery_buffer, 0.0))
        self.max_uav_wind_mps = None if max_uav_wind_mps is None else float(max(max_uav_wind_mps, 0.0))
        self.max_uav_rainfall_mmh = (
            None if max_uav_rainfall_mmh is None else float(max(max_uav_rainfall_mmh, 0.0))
        )
        self.max_uav_node_risk = None if max_uav_node_risk is None else float(max(max_uav_node_risk, 0.0))
        self.replan_cooldown_steps = int(max(replan_cooldown_steps, 0))
        self.min_goal_hold_steps = int(max(min_goal_hold_steps, 0))
        self.switch_margin = float(max(switch_margin, 0.0))
        self.short_sortie_max_distance_m = float(max(short_sortie_max_distance_m, 1.0))
        self.short_sortie_min_battery = float(np.clip(short_sortie_min_battery, 0.0, 1.0))
        self.stable_goal_before_takeoff_steps = int(max(stable_goal_before_takeoff_steps, 0))
        self.recovery_trigger_factor = float(max(recovery_trigger_factor, 1.0))
        ord_norm = str(assignment_order).strip().lower()
        if ord_norm not in {"uav_first", "truck_first"}:
            raise ValueError(f"unsupported assignment_order={assignment_order!r}")
        self.assignment_order = ord_norm
        self.rng = np.random.default_rng(seed)

        init_planner_runtime_state(self)
        # V2 is an additive planning path.  The complete legacy planner state
        # above remains available when the configuration switch is disabled.
        self._route_plan_v2 = HierarchicalRoutePlanManager(seed=seed)

    @staticmethod
    def _legacy_sortie_cap_enabled(env) -> bool:
        return bool(
            str(getattr(env.cfg, "physical_environment_version", "v1")).lower() != "v2"
            and getattr(env.cfg, "uav_enforce_max_sortie_limit", False)
        )

    @staticmethod
    def _er_hlns_coordination_active(env) -> bool:
        """Whether ER-HLNS-owned support/anchor/rescue helpers may run.

        The shared planner is also used by fixed-period comparators.  A bound
        comparator profile therefore needs an explicit capability boundary;
        unbound legacy fixtures keep the historical enabled behavior through
        :func:`er_hlns_coordination_active`.
        """

        return bool(er_hlns_coordination_active(env))

    def _reset_goal_switch_breakdown(self) -> None:
        self.goal_switch_count_total = 0
        self.goal_switch_candidate_count_total = 0
        self.goal_switch_accepted_count_total = 0
        self.goal_switch_rejected_by_threshold_count_total = 0
        self.goal_switch_forced_count_total = 0
        self.goal_switch_accepted_by_score_count_total = 0
        self.goal_switch_accepted_by_eta_count_total = 0
        self.goal_switch_forced_reason_completed_total = 0
        self.goal_switch_forced_reason_failed_total = 0
        self.goal_switch_forced_reason_infeasible_total = 0
        self.goal_switch_forced_reason_dead_end_total = 0
        self.goal_switch_forced_reason_uav_recovery_total = 0
        self.goal_switch_forced_reason_stall_total = 0
        self.cluster_primary_reject_count_total = 0
        self.cluster_primary_switch_count_total = 0
        self.same_task_cooldown_reject_count_total = 0
        self._last_switch_decision_by_aid.clear()
        self.uav_reject_cache_hit_count_total = 0
        self.uav_reject_cache_insert_count_total = 0
        self.uav_reject_cache_clear_count_total = 0
        self.uav_reject_cache_reason_insufficient_recovery_margin_total = 0
        self.uav_reject_cache_reason_corridor_total = 0
        self.uav_reject_cache_reason_comm_block_total = 0
        self.uav_reject_cache_reason_energy_infeasible_total = 0
        self.uav_reject_cache_reason_no_recovery_total = 0
        self.uav_task_selected_count_total = 0
        self._uav_reject_cache.clear()
        self._uav_reject_state_sig.clear()
        self.low_value_refresh_candidate_count_total = 0
        self.low_value_refresh_allowed_count_total = 0
        self.low_value_refresh_blocked_by_ablation_count_total = 0
        self.map_ranking_refresh_candidate_count_total = 0
        self.map_ranking_refresh_allowed_count_total = 0
        self.map_ranking_refresh_blocked_by_ablation_count_total = 0
        self.tc_global_assignment_called_count_total = 0
        self.tc_global_assignment_skipped_by_ablation_count_total = 0
        self.tc_assignment_epoch_applied_count_total = 0
        self.support_chain_candidate_count_total = 0
        self.support_chain_applied_count_total = 0
        self.support_chain_blocked_by_ablation_count_total = 0
        self.region_commitment_setup_count_total = 0
        self.region_commitment_local_candidate_count_total = 0
        self.region_commitment_cross_filtered_count_total = 0
        self.region_commitment_cross_override_count_total = 0
        self.region_commitment_outlier_task_count_total = 0
        self.region_commitment_outlier_filtered_count_total = 0
        self.region_commitment_outlier_override_count_total = 0
        self.region_commitment_auto_enabled_count_total = 0
        self.region_commitment_auto_disabled_count_total = 0
        self.cluster_primary_candidate_count_total = 0
        self.cluster_primary_applied_count_total = 0
        self.cluster_primary_blocked_by_ablation_count_total = 0
        self.task_reservation_applied_count_total = 0
        self.task_reservation_blocked_by_ablation_count_total = 0
        self.event_scoring_bonus_applied_count_total = 0
        self.event_scoring_bonus_blocked_by_ablation_count_total = 0
        self.normal_protection_candidate_count_total = 0
        self.normal_protection_applied_count_total = 0
        self.normal_protection_blocked_by_ablation_count_total = 0
        self.uav_emergency_commit_hold_count_total = 0
        self.uav_emergency_commit_break_hard_invalid_count_total = 0
        self.uav_emergency_commit_prevented_switch_count_total = 0
        self.uav_emergency_commit_followed_by_launch_count_total = 0
        self.uav_emergency_commit_followed_by_delivery_count_total = 0
        self.truck_routine_stuck_candidate_count_total = 0
        self.truck_routine_stuck_escape_count_total = 0
        self.truck_routine_stuck_escape_blocked_no_alt_count_total = 0
        self.truck_routine_stuck_escape_blocked_insufficient_gain_count_total = 0
        self.truck_routine_stuck_escape_followed_by_service_count_total = 0
        self.truck_routine_stuck_escape_followed_by_completion_count_total = 0
        self.routine_localize_eta_check_count_total = 0
        self.routine_localize_keep_current_count_total = 0
        self.routine_localize_escape_by_eta_worse_count_total = 0
        self.routine_localize_escape_followed_by_service_count_total = 0
        self.routine_localize_escape_followed_by_completion_count_total = 0
        self.uav_task_reserved_count_total = 0
        self.uav_task_reservation_release_count_total = 0
        self.uav_task_airborne_committed_count_total = 0
        self.uav_task_reserved_to_launch_count_total = 0
        self.uav_task_reserved_to_completion_count_total = 0
        self.uav_task_reservation_stale_count_total = 0
        self.uav_airborne_goal_switch_blocked_count_total = 0
        self.uav_airborne_safety_abort_count_total = 0
        self.uav_airborne_task_completed_count_total = 0
        self.truck_uav_assist_candidate_count_total = 0
        self.truck_uav_assist_accepted_count_total = 0
        self.truck_uav_assist_rejected_extra_distance_count_total = 0
        self.truck_uav_assist_rejected_normal_service_count_total = 0
        self.truck_uav_assist_launch_success_count_total = 0
        self.truck_uav_assist_followed_by_emergency_completion_count_total = 0
        self.truck_uav_assist_extra_distance_m_total = 0.0
        self._uav_task_reservation_state_by_task.clear()
        self._uav_task_reservation_by_uav.clear()
        self._region_commitment_signature = None
        self._region_centers_xy.clear()
        self._task_region_by_task.clear()
        self._region_task_distance_m.clear()
        self._region_outlier_task_ids.clear()
        self._region_commitment_effective_k = 0
        self._region_commitment_enabled_effective = False
        self._region_commitment_auto_score = 0.0
        self._region_commitment_separation_score = 0.0
        self._region_commitment_load_balance_score = 0.0
        self._region_commitment_coverage_score = 0.0
        self._region_commitment_strength = 0.0
        self._agent_home_region.clear()
        self._uav_intent_signal_by_uav.clear()
        self._truck_assist_waypoint_by_truck.clear()
        self._truck_assist_pending_windows = []
        self._far_routine_bootstrap_force_step.clear()
        self.harmful_switch_proxy_count_total = 0
        self.missed_switch_proxy_count_total = 0
        self._uav_commit_hold_pending = []
        self._truck_routine_escape_pending = []
        self._routine_localize_escape_pending = []
        self._switch_decision_ledger_rows = []
        self._switch_decision_pending_windows = []
        self._switch_decision_seq = 0
        self._switch_goal_distance_history_by_agent.clear()
        self._task_recent_prev_goal.clear()
        self._task_recent_switch_step.clear()
        self.task_aba_switch_blocked_count_total = 0
        self.comm_blackout_commit_hold_count_total = 0
        self.refresh_total_count = 0
        self.fixed_interval_refresh_count = 0
        self.event_refresh_count = 0
        self.no_event_fallback_refresh_count = 0
        self.initial_refresh_count = 0
        self.empty_goal_refresh_count = 0
        self.steps_since_last_refresh_sum = 0
        self.steps_since_last_refresh_max = 0
        self._episode_first_refresh_done = False
        self.event_refresh_reason_arrival_count_total = 0
        self.event_refresh_reason_resolution_count_total = 0
        self.event_refresh_reason_uav_idle_count_total = 0
        self.event_refresh_reason_truck_idle_count_total = 0
        self.event_refresh_reason_map_update_light_count_total = 0
        self.event_refresh_reason_map_update_hard_count_total = 0
        self.event_refresh_reason_goal_invalid_count_total = 0
        self.event_refresh_reason_path_blocked_count_total = 0
        self.event_refresh_reason_goal_unreachable_count_total = 0
        self.event_refresh_reason_uav_safety_count_total = 0
        self.event_refresh_reason_truck_dead_end_count_total = 0
        self.event_refresh_reason_high_priority_uncovered_count_total = 0
        self.event_refresh_reason_normal_stall_count_total = 0
        self.event_refresh_no_goal_change_count_total = 0
        self.event_refresh_goal_change_count_total = 0
        self.event_refresh_to_launch_count_total = 0
        self.event_refresh_to_completion_count_total = 0
        self.event_refresh_followed_by_reject_count_total = 0
        self.event_refresh_followed_by_stall_count_total = 0
        self.hard_event_refresh_count_total = 0
        self.hard_event_reason_goal_invalid_count_total = 0
        self.hard_event_reason_current_goal_unreachable_count_total = 0
        self.hard_event_reason_path_blocked_count_total = 0
        self.hard_event_reason_uav_safety_count_total = 0
        self.hard_event_reason_uav_recovery_count_total = 0
        self.hard_event_reason_truck_dead_end_count_total = 0
        self.hard_event_reason_high_priority_uncovered_count_total = 0
        self.hard_event_reason_normal_stall_count_total = 0
        self.hard_event_reason_assigned_but_not_progressing_count_total = 0
        self.hard_event_reason_goal_completed_count_total = 0
        self.hard_event_reason_goal_failed_count_total = 0
        self.goal_invalid_reason_task_completed_total = 0
        self.goal_invalid_reason_task_failed_total = 0
        self.goal_invalid_reason_task_missing_total = 0
        self.goal_invalid_reason_truck_unreachable_total = 0
        self.goal_invalid_reason_uav_energy_infeasible_total = 0
        self.goal_invalid_reason_uav_recovery_margin_total = 0
        self.goal_invalid_reason_uav_corridor_total = 0
        self.goal_invalid_reason_uav_comm_block_total = 0
        self.goal_invalid_reason_uav_not_loaded_total = 0
        self.goal_invalid_reason_uav_not_docked_total = 0
        self.goal_invalid_reason_soft_reject_cache_total = 0
        self.suspect_soft_as_hard_count_total = 0
        self.hard_event_refresh_no_goal_change_count_total = 0
        self.hard_event_refresh_goal_change_count_total = 0
        self.hard_event_refresh_to_launch_count_total = 0
        self.hard_event_refresh_to_completion_count_total = 0
        self.hard_event_refresh_followed_by_reject_count_total = 0
        self.hard_event_refresh_followed_by_stall_count_total = 0
        self.normal_stall_candidate_count_total = 0
        self.normal_stall_blocked_by_persist_count_total = 0
        self.normal_stall_blocked_by_cooldown_count_total = 0
        self.normal_stall_local_correction_count_total = 0
        self.normal_stall_global_refresh_count_total = 0
        self.goal_invalid_hard_count_total = 0
        self.goal_invalid_soft_count_total = 0
        self.goal_invalid_soft_suppressed_count_total = 0
        self.goal_invalid_soft_escalated_count_total = 0
        self.uav_recovery_hard_count_total = 0
        self.uav_recovery_soft_count_total = 0
        self.uav_recovery_soft_suppressed_count_total = 0
        self.uav_recovery_local_action_count_total = 0
        self.uav_recovery_global_refresh_count_total = 0
        self.truck_dead_end_candidate_count_total = 0
        self.truck_dead_end_blocked_by_persist_count_total = 0
        self.truck_dead_end_blocked_by_cooldown_count_total = 0
        self.truck_dead_end_local_path_repair_count_total = 0
        self.truck_dead_end_local_goal_reassign_count_total = 0
        self.truck_dead_end_global_refresh_count_total = 0
        self.truck_dead_end_noop_count_total = 0
        self.truck_dead_end_routine_localized_count_total = 0
        self.truck_dead_end_emergency_kept_hard_count_total = 0
        self.truck_dead_end_support_kept_hard_count_total = 0
        self.truck_dead_end_recovery_kept_hard_count_total = 0
        self.truck_dead_end_local_repair_no_goal_change_count_total = 0
        self.truck_dead_end_global_refresh_no_goal_change_count_total = 0
        self.path_blocked_candidate_count_total = 0
        self.path_blocked_nonimpact_suppressed_count_total = 0
        self.path_blocked_impacted_current_path_count_total = 0
        self.path_blocked_impacted_goal_reachability_count_total = 0
        self.path_blocked_impacted_recovery_count_total = 0
        self.path_blocked_local_path_repair_count_total = 0
        self.path_blocked_local_goal_reassign_count_total = 0
        self.path_blocked_global_refresh_count_total = 0
        self.path_blocked_noop_count_total = 0
        self.path_blocked_routine_localized_count_total = 0
        self.path_blocked_emergency_kept_hard_count_total = 0
        self.path_blocked_recovery_kept_hard_count_total = 0
        self.path_blocked_support_kept_hard_count_total = 0
        self.path_blocked_goal_unreachable_kept_hard_count_total = 0
        self.path_blocked_local_repair_no_goal_change_count_total = 0
        self.path_blocked_global_refresh_no_goal_change_count_total = 0
        self.erc_event_detected_count_total = 0
        self.erc_event_gate_pass_count_total = 0
        self.erc_event_gate_reject_count_total = 0
        self.erc_local_correction_count_total = 0
        self.erc_global_replan_count_total = 0
        self.committed_goal_hold_count_total = 0
        self.committed_goal_broken_count_total = 0
        self.committed_goal_broken_reason_hard_invalid_count_total = 0
        self.committed_goal_broken_reason_stall_count_total = 0
        self.committed_goal_broken_reason_tc_gain_count_total = 0
        self.airborne_uav_goal_lock_count_total = 0
        self.path_blocked_local_agent_count_total = 0
        self.high_priority_event_rejected_no_launchable_uav_count_total = 0
        self.committed_goal.clear()
        self.committed_goal_step.clear()
        self.committed_goal_progress.clear()
        self.committed_goal_status.clear()
        self.committed_goal_feasibility.clear()
        self._last_goal_invalid_record = None
        self._last_uav_emergency_record = None
        self._last_truck_dead_end_record = None
        self._last_high_priority_uncovered_record = None
        self._last_normal_stall_record = None
        self._last_goal_terminal_status = ""
        self._last_hard_event_offenders = []
        self._hard_event_offender_stats.clear()
        self._hard_event_reason_outcome_stats.clear()
        self._normal_stall_cooldown_until_by_truck.clear()
        self._truck_dead_end_persist_by_truck.clear()
        self._truck_dead_end_cooldown_until_by_truck.clear()
        self._debug_truck_dead_end_localize_by_truck.clear()
        self._debug_truck_path_blocked_localize_by_truck.clear()
        self._soft_invalid_repeat.clear()
        self._soft_invalid_cooldown_until.clear()
        self._noop_event_cooldown_until_step_by_reason.clear()
        self._active_event_refresh_window = None

    def _record_switch_decision(self, aid: str, decision: str, reason: str = "") -> None:
        self._last_switch_decision_by_aid[str(aid)] = {"decision": str(decision), "reason": str(reason)}

    def _switch_task_kind_name(self, task) -> str:
        if task is None:
            return ""
        if task.kind == TaskKind.EMERGENCY:
            return "emergency"
        if task.kind == TaskKind.NORMAL:
            return "normal"
        return str(getattr(task.kind, "value", str(task.kind))).lower()

    def _switch_goal_type_name(self, env, goal_id: Optional[str]) -> str:
        if goal_id is None:
            return "none"
        t = env.state.tasks.get(str(goal_id), None)
        if t is not None:
            return "task"
        a = env.state.agents.get(str(goal_id), None)
        if a is not None:
            return "agent"
        return "unknown"

    def _switch_goal_status_name(self, env, goal_id: Optional[str]) -> str:
        if goal_id is None:
            return "none"
        t = env.state.tasks.get(str(goal_id), None)
        if t is not None:
            return str(getattr(t.status, "name", str(t.status))).lower()
        a = env.state.agents.get(str(goal_id), None)
        if a is not None:
            return "alive" if not bool(getattr(a, "crashed", False)) else "crashed"
        return "missing"

    def _switch_goal_distance(self, env, aid: str, goal_id: Optional[str]) -> float:
        if goal_id is None:
            return float("nan")
        t = env.state.tasks.get(str(goal_id), None)
        if t is not None:
            try:
                return float(env._agent_distance_to_task(str(aid), t))
            except Exception:
                return float("nan")
        ag = env.state.agents.get(str(goal_id), None)
        if ag is not None:
            ax, ay = self._agent_xy(env, str(aid))
            gx, gy = self._agent_xy(env, str(goal_id))
            return float(np.hypot(float(gx) - float(ax), float(gy) - float(ay)))
        return float("nan")

    def _switch_goal_eta(self, env, aid: str, goal_id: Optional[str]) -> float:
        d = self._switch_goal_distance(env, str(aid), goal_id)
        if not np.isfinite(d):
            return float("nan")
        st = env.state.agents.get(str(aid), None)
        if st is None:
            return float("nan")
        speed = float(getattr(st, "speed", 0.0))
        dt = float(max(getattr(env.cfg, "dt_seconds", 1.0), 1e-6))
        if speed <= 1e-9:
            return float("inf")
        return float(d / max(speed * dt, 1e-6))

    def _switch_total_uav_reject_count(self, env) -> int:
        return int(
            int(getattr(env, "uav_task_reject_below_launch_min_count", 0))
            + int(getattr(env, "uav_task_reject_not_loaded_count", 0))
            + int(getattr(env, "uav_task_reject_recovery_margin_count", 0))
            + int(getattr(env, "uav_task_reject_horizon_count", 0))
            + int(getattr(env, "uav_task_reject_comm_block_count", 0))
            + int(getattr(env, "uav_task_reject_corridor_count", 0))
        )

    def _has_launchable_or_near_launchable_uav(self, env) -> bool:
        pending_emg = [t for t in env.state.tasks.values() if t.status == TaskStatus.PENDING and t.kind == TaskKind.EMERGENCY]
        if not pending_emg:
            return False
        docked_fn = getattr(env, "_uav_docked_task_actionable_now", None)
        for aid, st in env.state.agents.items():
            if st.kind != AgentKind.UAV or bool(getattr(st, "crashed", False)):
                continue
            for task in pending_emg:
                actionable = False
                if callable(docked_fn):
                    try:
                        actionable = bool(docked_fn(str(aid), task))
                    except Exception:
                        actionable = False
                if actionable:
                    return True
                try:
                    if bool(self._uav_task_feasible(env, str(aid), task)):
                        return True
                except Exception:
                    pass
        return False

    def _commitment_local_correction_mode(self, env) -> bool:
        # Use existing generic commitment + UAV emergency hold as the execution-baseline switch.
        return bool(
            self.use_event_trigger
            and bool(getattr(env.cfg, "execution_commitment_enabled", True))
            and bool(getattr(env.cfg, "hrl_uav_emergency_commit_hold_enabled", False))
        )

    def _switch_event_reason_compact(self) -> str:
        flags = self._last_refresh_flags if isinstance(self._last_refresh_flags, dict) else {}
        out: List[str] = []
        if bool(flags.get("arrival", False)):
            out.append("arrival")
        if bool(flags.get("resolution", False)):
            out.append("resolution")
        if bool(flags.get("uav_idle", False)):
            out.append("uav_idle")
        if bool(flags.get("truck_idle", False)):
            out.append("truck_idle")
        if bool(flags.get("map_update", False)):
            out.append("map_update")
        if bool(flags.get("map_update_hard", False)):
            out.append("map_update_hard")
        if bool(flags.get("high_priority_uncovered", False)):
            out.append("high_priority_uncovered")
        return "|".join(out)

    def _switch_hard_event_reason_compact(self) -> str:
        flags = self._last_refresh_flags if isinstance(self._last_refresh_flags, dict) else {}
        out: List[str] = []
        if bool(flags.get("hard_reason_goal_invalid", False)):
            out.append("goal_invalid")
        if bool(flags.get("hard_reason_goal_unreachable", False)):
            out.append("goal_unreachable")
        if bool(flags.get("hard_reason_path_blocked", False)):
            out.append("path_blocked")
        if bool(flags.get("hard_reason_uav_safety", False)):
            out.append("uav_safety")
        if bool(flags.get("hard_reason_uav_recovery", False)):
            out.append("uav_recovery")
        if bool(flags.get("hard_reason_truck_dead_end", False)):
            out.append("truck_dead_end")
        if bool(flags.get("hard_reason_high_priority_uncovered", False)):
            out.append("high_priority_uncovered")
        if bool(flags.get("hard_reason_normal_stall", False)):
            out.append("normal_stall")
        if bool(flags.get("hard_reason_assigned_but_not_progressing", False)):
            out.append("assigned_but_not_progressing")
        if bool(flags.get("hard_reason_goal_completed", False)):
            out.append("goal_completed")
        if bool(flags.get("hard_reason_goal_failed", False)):
            out.append("goal_failed")
        return "|".join(out)

    def _switch_update_goal_distance_history(self, env) -> None:
        step_now = int(getattr(env.state, "step_index", 0))
        for aid, gid in self.state.goals.items():
            if gid is None:
                continue
            d = self._switch_goal_distance(env, str(aid), str(gid))
            if not np.isfinite(d):
                continue
            hist = self._switch_goal_distance_history_by_agent.get(str(aid), [])
            hist.append((int(step_now), str(gid), float(d)))
            if len(hist) > 24:
                hist = hist[-24:]
            self._switch_goal_distance_history_by_agent[str(aid)] = hist

    def _switch_goal_progress_recent(self, env, aid: str, goal_id: Optional[str], window: int) -> float:
        if goal_id is None:
            return float("nan")
        hist = list(self._switch_goal_distance_history_by_agent.get(str(aid), []))
        if not hist:
            return float("nan")
        step_now = int(getattr(env.state, "step_index", 0))
        kept = [h for h in hist if str(h[1]) == str(goal_id) and int(h[0]) >= (step_now - int(max(window, 1)))]
        if len(kept) < 2:
            return float("nan")
        d0 = float(kept[0][2])
        d1 = float(kept[-1][2])
        return float(d0 - d1)

    def _switch_update_outcome_windows(self, env, force_finalize: bool = False) -> None:
        if not self._switch_decision_pending_windows:
            return
        step_now = int(getattr(env.state, "step_index", 0))
        keep: List[Dict[str, Any]] = []
        for rec in list(self._switch_decision_pending_windows):
            aid = str(rec.get("agent_id", ""))
            settled_goal = rec.get("settled_goal_id", None)
            cur_now = self.state.goals.get(str(aid), None)
            if settled_goal is not None and str(cur_now) == str(settled_goal):
                rec["after_switch_same_goal_retained_steps"] = int(rec.get("after_switch_same_goal_retained_steps", 0)) + 1
            else:
                rec["after_switch_goal_changed_again"] = 1
            d_now = self._switch_goal_distance(env, str(aid), settled_goal)
            d_prev = float(rec.get("_prev_dist", float("nan")))
            if np.isfinite(d_now) and np.isfinite(d_prev) and d_now >= d_prev - 1e-6:
                rec["after_switch_stall_count"] = int(rec.get("after_switch_stall_count", 0)) + 1
            rec["_prev_dist"] = d_now
            if (not bool(force_finalize)) and int(step_now) - int(rec.get("step", step_now)) < int(rec.get("outcome_window_steps", 10)):
                keep.append(rec)
                continue

            launch_delta = int(getattr(env, "uav_launch_count_total", 0)) - int(rec.get("_launch_base", 0))
            delivery_delta = int(getattr(env, "uav_delivery_count_total", 0)) - int(rec.get("_delivery_base", 0))
            reject_delta = int(self._switch_total_uav_reject_count(env)) - int(rec.get("_reject_base", 0))
            start_dist = float(rec.get("_start_dist", float("nan")))
            rec["after_switch_distance_progress"] = (
                float(start_dist - d_now) if np.isfinite(start_dist) and np.isfinite(d_now) else float("nan")
            )
            rec["after_switch_uav_launch"] = int(max(launch_delta, 0))
            rec["after_switch_uav_delivery"] = int(max(delivery_delta, 0))
            rec["after_switch_reject_count"] = int(max(reject_delta, 0))
            t = env.state.tasks.get(str(settled_goal), None) if settled_goal is not None else None
            service_started = 0
            completed = 0
            if t is not None:
                if t.first_service_step is not None and int(t.first_service_step) >= int(rec.get("step", 0)):
                    service_started = 1
                if t.delivered_step is not None and int(t.delivered_step) >= int(rec.get("step", 0)):
                    completed = 1
            rec["after_switch_service_started"] = int(service_started)
            rec["after_switch_task_completed"] = int(completed)

            # Error labels.
            accepted = int(rec.get("switch_accepted", 0)) == 1
            rejected = int(rec.get("switch_rejected", 0)) == 1
            cur_prog3 = float(rec.get("current_goal_progress_last_3_steps", float("nan")))
            cur_prog5 = float(rec.get("current_goal_progress_last_5_steps", float("nan")))
            near_service = int(rec.get("current_goal_near_service_radius", 0)) == 1
            cur_serv = int(rec.get("current_goal_service_started", 0)) == 1
            cur_launchable = int(rec.get("uav_current_launchable", 0)) == 1
            cur_airborne = int(rec.get("uav_current_airborne", 0)) == 1
            has_positive_outcome = bool(
                int(rec.get("after_switch_service_started", 0)) > 0
                or int(rec.get("after_switch_task_completed", 0)) > 0
                or int(rec.get("after_switch_uav_launch", 0)) > 0
                or int(rec.get("after_switch_uav_delivery", 0)) > 0
            )
            harmful = 0
            if accepted:
                positive_before = bool(
                    (np.isfinite(cur_prog3) and cur_prog3 > 0.0)
                    or near_service
                    or cur_serv
                    or cur_launchable
                    or cur_airborne
                )
                if positive_before and (not has_positive_outcome):
                    if int(rec.get("after_switch_goal_changed_again", 0)) > 0 or int(rec.get("after_switch_reject_count", 0)) > 0 or int(rec.get("after_switch_stall_count", 0)) > 0:
                        harmful = 1
            missed = 0
            if rejected:
                eta_gain = float(rec.get("eta_gain", float("nan")))
                eta_thr = float(rec.get("switch_eta_threshold", float("nan")))
                prop_feas = int(rec.get("proposed_uav_recovery_feasible", 0)) == 1 or int(rec.get("proposed_goal_feasible", 0)) == 1
                blocked = int(rec.get("current_goal_path_blocked", 0)) == 1 or int(rec.get("current_goal_unreachable", 0)) == 1
                stuck = bool(np.isfinite(cur_prog5) and cur_prog5 <= 1e-6)
                better_alt = bool(np.isfinite(eta_gain) and np.isfinite(eta_thr) and eta_gain > eta_thr and prop_feas)
                if (blocked or stuck or better_alt) and (not has_positive_outcome):
                    missed = 1
            rec["harmful_switch"] = int(harmful)
            rec["missed_switch"] = int(missed)
            if harmful:
                self.harmful_switch_proxy_count_total = int(self.harmful_switch_proxy_count_total) + 1
            if missed:
                self.missed_switch_proxy_count_total = int(self.missed_switch_proxy_count_total) + 1

            # Drop private fields and emit finalized ledger row.
            for pk in (
                "_launch_base",
                "_delivery_base",
                "_reject_base",
                "_prev_dist",
                "_start_dist",
                "settled_goal_id",
            ):
                rec.pop(pk, None)
            self._switch_decision_ledger_rows.append(dict(rec))
        self._switch_decision_pending_windows = keep

    def _export_switch_decision_ledger_rows(self, env) -> List[Dict[str, Any]]:
        self._update_progress_aware_pending(env, force_finalize=True)
        self._switch_update_outcome_windows(env, force_finalize=True)
        rows = list(self._switch_decision_ledger_rows)
        self._switch_decision_ledger_rows = []
        self._switch_decision_pending_windows = []
        return rows

    def _update_progress_aware_pending(self, env, force_finalize: bool = False) -> None:
        step_now = int(getattr(env.state, "step_index", 0))
        keep_u: List[Dict[str, Any]] = []
        for rec in list(self._uav_commit_hold_pending):
            if (not bool(force_finalize)) and step_now - int(rec.get("step", 0)) < 10:
                keep_u.append(rec)
                continue
            d_launch = int(getattr(env, "uav_launch_count_total", 0)) - int(rec.get("launch_base", 0))
            d_deliv = int(getattr(env, "uav_delivery_count_total", 0)) - int(rec.get("delivery_base", 0))
            if d_launch > 0:
                self.uav_emergency_commit_followed_by_launch_count_total = int(self.uav_emergency_commit_followed_by_launch_count_total) + 1
            if d_deliv > 0:
                self.uav_emergency_commit_followed_by_delivery_count_total = int(self.uav_emergency_commit_followed_by_delivery_count_total) + 1
        self._uav_commit_hold_pending = keep_u

        keep_t: List[Dict[str, Any]] = []
        for rec in list(self._truck_routine_escape_pending):
            tid = str(rec.get("task_id", ""))
            t = env.state.tasks.get(tid, None)
            if (not bool(force_finalize)) and step_now - int(rec.get("step", 0)) < 10:
                keep_t.append(rec)
                continue
            if t is not None:
                if t.first_service_step is not None and int(t.first_service_step) >= int(rec.get("step", 0)):
                    self.truck_routine_stuck_escape_followed_by_service_count_total = int(self.truck_routine_stuck_escape_followed_by_service_count_total) + 1
                if t.delivered_step is not None and int(t.delivered_step) >= int(rec.get("step", 0)):
                    self.truck_routine_stuck_escape_followed_by_completion_count_total = int(self.truck_routine_stuck_escape_followed_by_completion_count_total) + 1
        self._truck_routine_escape_pending = keep_t

        keep_l: List[Dict[str, Any]] = []
        for rec in list(self._routine_localize_escape_pending):
            tid = str(rec.get("task_id", ""))
            t = env.state.tasks.get(tid, None)
            if (not bool(force_finalize)) and step_now - int(rec.get("step", 0)) < 10:
                keep_l.append(rec)
                continue
            if t is not None:
                if t.first_service_step is not None and int(t.first_service_step) >= int(rec.get("step", 0)):
                    self.routine_localize_escape_followed_by_service_count_total = int(self.routine_localize_escape_followed_by_service_count_total) + 1
                if t.delivered_step is not None and int(t.delivered_step) >= int(rec.get("step", 0)):
                    self.routine_localize_escape_followed_by_completion_count_total = int(self.routine_localize_escape_followed_by_completion_count_total) + 1
        self._routine_localize_escape_pending = keep_l

    def _best_alternative_routine_for_truck(self, env, aid: str, exclude_tid: str) -> Tuple[Optional[str], float, float]:
        best_tid: Optional[str] = None
        best_eta = float("inf")
        best_score = -1e18
        st = env.state.agents.get(str(aid), None)
        if st is None:
            return None, float("inf"), -1e18
        speed = float(max(getattr(st, "speed", 0.0), 1e-6))
        dt = float(max(getattr(env.cfg, "dt_seconds", 1.0), 1e-6))
        for t in env.state.tasks.values():
            if t.status != TaskStatus.PENDING or t.kind != TaskKind.NORMAL:
                continue
            tid = str(t.task_id)
            if tid == str(exclude_tid):
                continue
            if not self._truck_task_valid(env, str(aid), tid):
                continue
            if not self._truck_task_reachable(env, str(aid), t):
                continue
            d = float(self._truck_task_distance(env, str(aid), t))
            if not np.isfinite(d):
                continue
            eta = float(d / max(speed * dt, 1e-6))
            sc = float(self._score_goal_for_agent(env, str(aid), tid))
            if (eta + 1e-9 < best_eta) or (abs(eta - best_eta) <= 1e-9 and sc > best_score):
                best_tid = tid
                best_eta = eta
                best_score = sc
        return best_tid, float(best_eta), float(best_score)

    def _truck_routine_stuck_escape_goal(self, env, aid: str, keep_current: str, cur_score: float) -> Optional[str]:
        if not bool(getattr(env.cfg, "hrl_truck_routine_stuck_escape_enabled", True)):
            return None
        st = env.state.agents.get(str(aid), None)
        cur_task = env.state.tasks.get(str(keep_current), None)
        if st is None or st.kind != AgentKind.TRUCK or cur_task is None:
            return None
        if cur_task.kind != TaskKind.NORMAL or cur_task.status != TaskStatus.PENDING:
            return None
        if bool(cur_task.in_service_by == str(aid) and int(getattr(cur_task, "service_remaining", 0)) > 0):
            return None
        self.truck_routine_stuck_candidate_count_total = int(self.truck_routine_stuck_candidate_count_total) + 1
        persist_steps = int(max(getattr(env.cfg, "hrl_truck_routine_stuck_persist_steps", 5), 1))
        eps_m = float(max(getattr(env.cfg, "hrl_truck_routine_progress_epsilon_m", 30.0), 0.0))
        p_recent = float(self._switch_goal_progress_recent(env, str(aid), str(keep_current), persist_steps))
        if np.isfinite(p_recent) and p_recent > eps_m:
            return None
        if cur_task.first_service_step is not None:
            return None
        best_tid, best_eta, best_score = self._best_alternative_routine_for_truck(env, str(aid), str(cur_task.task_id))
        if best_tid is None:
            self.truck_routine_stuck_escape_blocked_no_alt_count_total = int(self.truck_routine_stuck_escape_blocked_no_alt_count_total) + 1
            return None
        cur_eta = float(self._switch_goal_eta(env, str(aid), str(keep_current)))
        eta_gain = float(cur_eta - best_eta) if np.isfinite(cur_eta) and np.isfinite(best_eta) else float("-inf")
        score_gain = float(best_score - cur_score) if np.isfinite(best_score) and np.isfinite(cur_score) else float("-inf")
        min_eta = float(max(getattr(env.cfg, "hrl_truck_routine_escape_min_eta_gain_steps", 6), 0.0))
        min_score = float(max(getattr(env.cfg, "hrl_truck_routine_escape_min_score_gain", 0.12), 0.0))
        allow_any_alt = bool(getattr(env.cfg, "hrl_truck_routine_escape_allow_any_alt_when_stuck", False))
        if allow_any_alt:
            min_any_step = int(max(getattr(env.cfg, "hrl_truck_routine_escape_allow_any_alt_min_step", 0), 0))
            allow_any_alt = bool(int(getattr(env.state, "step_index", 0)) >= min_any_step)
        if (not allow_any_alt) and eta_gain < min_eta and score_gain < min_score:
            self.truck_routine_stuck_escape_blocked_insufficient_gain_count_total = int(self.truck_routine_stuck_escape_blocked_insufficient_gain_count_total) + 1
            return None
        self.truck_routine_stuck_escape_count_total = int(self.truck_routine_stuck_escape_count_total) + 1
        self._truck_routine_escape_pending.append({"step": int(getattr(env.state, "step_index", 0)), "task_id": str(best_tid)})
        return str(best_tid)

    def _uav_reject_cache_window_steps(self, env) -> int:
        return int(max(getattr(env.cfg, "uav_reject_cache_window_steps", 20), 1))

    def _uav_reject_cache_min_repeat(self, env) -> int:
        return int(max(getattr(env.cfg, "uav_reject_cache_min_repeat", 2), 1))

    def _uav_reject_cache_ttl_steps(self, env) -> int:
        return int(max(getattr(env.cfg, "uav_reject_cache_ttl_steps", 30), 1))

    def _uav_reject_cache_ttl_for_reason(self, env, reason: str) -> int:
        ttl = int(self._uav_reject_cache_ttl_steps(env))
        rr = self._normalize_uav_reject_reason(reason)
        if rr in {"insufficient_recovery_margin", "no_recovery"}:
            return int(max(min(ttl, 4), 1))
        return int(ttl)

    def _normalize_uav_reject_reason(self, reason: str) -> str:
        r = str(reason).strip().lower()
        if r in {"insufficient_recovery_margin", "recovery_margin"}:
            return "insufficient_recovery_margin"
        if r in {"corridor", "corridor_blocked"}:
            return "corridor"
        if r in {"comm_block", "comm_degraded"}:
            return "comm_block"
        if r in {"below_launch_min", "energy_infeasible", "horizon"}:
            return "energy_infeasible"
        if r in {"no_truck_for_return", "rendezvous_launch_disabled", "not_loaded", "no_recovery"}:
            return "no_recovery"
        return ""

    def _uav_reject_state_signature(self, env, aid: str) -> Tuple[float, str, int, int]:
        st = env.state.agents.get(str(aid), None)
        batt = float(getattr(st, "battery", 0.0)) if st is not None else 0.0
        follow = str(getattr(st, "follow_target", "")) if st is not None and getattr(st, "follow_target", None) is not None else ""
        comm = int(bool(getattr(env, "comm_blocked", {}).get(str(aid), False))) if isinstance(getattr(env, "comm_blocked", {}), dict) else 0
        bver = int(self._blocked_edge_version(env))
        return (round(batt, 3), follow, comm, bver)

    def _clear_uav_reject_cache_by_agent(self, aid: str) -> None:
        stale = [k for k in self._uav_reject_cache.keys() if str(k[0]) == str(aid)]
        if not stale:
            return
        for k in stale:
            self._uav_reject_cache.pop(k, None)
        self.uav_reject_cache_clear_count_total += int(len(stale))

    def _maybe_clear_uav_reject_cache(self, env, aid: str) -> None:
        sig = self._uav_reject_state_signature(env, str(aid))
        prev = self._uav_reject_state_sig.get(str(aid), None)
        if prev is None:
            self._uav_reject_state_sig[str(aid)] = sig
            return
        if sig != prev:
            self._clear_uav_reject_cache_by_agent(str(aid))
        self._uav_reject_state_sig[str(aid)] = sig

    def _pending_emergency_task_count(self, env) -> int:
        return int(sum(1 for t in env.state.tasks.values() if t.status == TaskStatus.PENDING and t.kind == TaskKind.EMERGENCY))

    def _uav_reject_cache_blocked(self, env, aid: str, task_id: str) -> bool:
        step_now = int(getattr(env.state, "step_index", 0))
        # If this is effectively the only pending emergency, do not block by cache.
        if self._pending_emergency_task_count(env) <= 1:
            return False
        chain = self._support_bound_chain_info_for_uav(env, str(aid))
        if chain is not None and str(chain.get("task_id", "")) == str(task_id):
            return False
        for reason in ("insufficient_recovery_margin", "corridor", "comm_block", "energy_infeasible", "no_recovery"):
            rec = self._uav_reject_cache.get((str(aid), str(task_id), reason), None)
            if not rec:
                continue
            until = int(rec.get("blocked_until", -1))
            if step_now <= until:
                self.uav_reject_cache_hit_count_total += 1
                if reason == "insufficient_recovery_margin":
                    self.uav_reject_cache_reason_insufficient_recovery_margin_total += 1
                elif reason == "corridor":
                    self.uav_reject_cache_reason_corridor_total += 1
                elif reason == "comm_block":
                    self.uav_reject_cache_reason_comm_block_total += 1
                elif reason == "energy_infeasible":
                    self.uav_reject_cache_reason_energy_infeasible_total += 1
                elif reason == "no_recovery":
                    self.uav_reject_cache_reason_no_recovery_total += 1
                return True
        return False

    def _record_uav_reject_cache(self, env, aid: str, task_id: str, reason: str) -> None:
        rr = self._normalize_uav_reject_reason(reason)
        if not rr:
            return
        key = (str(aid), str(task_id), rr)
        step_now = int(getattr(env.state, "step_index", 0))
        win = int(self._uav_reject_cache_window_steps(env))
        rep = int(self._uav_reject_cache_min_repeat(env))
        ttl = int(self._uav_reject_cache_ttl_for_reason(env, rr))
        rec = self._uav_reject_cache.get(key, {"steps": [], "blocked_until": -1})
        steps = [int(x) for x in list(rec.get("steps", [])) if int(x) >= (step_now - win)]
        steps.append(step_now)
        rec["steps"] = steps
        if len(steps) >= rep:
            rec["blocked_until"] = int(step_now + ttl)
        self._uav_reject_cache[key] = rec
        self.uav_reject_cache_insert_count_total += 1
    def _distance_norm_m(self, env) -> float:
        return float(
            max(
                getattr(
                    env.cfg,
                    "distance_norm_m",
                    getattr(env.cfg, "pbrs_distance_norm_m", 3000.0),
                ),
                1e-6,
            )
        )

    def _ensure_step_eval_caches(self, env) -> int:
        step_now = int(getattr(env.state, "step_index", 0))
        if self._planner_eval_cache_step != step_now:
            self._planner_eval_cache_step = step_now
            self._truck_task_distance_cache.clear()
            self._truck_task_serviceable_cache.clear()
            self._truck_nearest_reachable_cache.clear()
            self._support_anchor_gain_cache.clear()
            self._truck_support_candidate_cache.clear()
            self._task_high_pressure_cache.clear()
        return step_now

    def _uav_task_reservation_enabled(self, env) -> bool:
        return bool(
            self._er_hlns_coordination_active(env)
            and getattr(env.cfg, "hrl_uav_task_reservation_enabled", True)
        )

    def _supported_sortie_joint_enabled(self, env) -> bool:
        return bool(getattr(env.cfg, "hrl_supported_sortie_joint_enabled", True))

    def _dynamic_task_pressure_enabled(self, env) -> bool:
        return bool(getattr(env.cfg, "hrl_dynamic_task_pressure_enabled", True))

    def _support_conversion_gate_enabled(self, env) -> bool:
        return bool(getattr(env.cfg, "hrl_support_conversion_gate_enabled", False))

    def _truck_normal_commit_guard2_enabled(self, env) -> bool:
        return bool(getattr(env.cfg, "hrl_truck_normal_commit_guard2_enabled", False))

    def _uav_ride_stall_release_enabled(self, env) -> bool:
        return bool(getattr(env.cfg, "hrl_uav_ride_stall_release_enabled", False))

    def _large_map_active(self, env) -> bool:
        map_size = float(max(getattr(env.cfg, "map_size_m", 0.0), 0.0))
        min_map = float(max(getattr(env.cfg, "hrl_large_map_active_min_map_size_m", 9000.0), 0.0))
        return bool(map_size >= min_map)

    def _timecritical_pressure_active(self, env) -> bool:
        pending_tc = int(
            sum(
                1
                for t in env.state.tasks.values()
                if t.status == TaskStatus.PENDING and self._is_timecritical_lightweight_task(t)
            )
        )
        min_tc = int(max(getattr(env.cfg, "hrl_timecritical_pressure_min_pending", 2), 0))
        return bool(pending_tc >= min_tc)

    def _direct_ready_pressure_active(self, env) -> bool:
        pending_em = int(
            sum(1 for t in env.state.tasks.values() if t.status == TaskStatus.PENDING and t.kind == TaskKind.EMERGENCY)
        )
        min_em = int(max(getattr(env.cfg, "hrl_direct_ready_pressure_min_pending", 2), 0))
        return bool(pending_em >= min_em)

    def _execution_switch_pressure_active(self, env) -> bool:
        threshold = int(max(getattr(env.cfg, "hrl_execution_switch_pressure_goal_switch_threshold", 24), 0))
        return bool(int(self.goal_switch_count_total) >= threshold or self._direct_ready_pressure_active(env))

    def _support_need_active(self, env) -> bool:
        return bool(self._timecritical_pressure_active(env) or self._direct_ready_pressure_active(env))

    def _tc_global_assignment_runtime_enabled(self, env) -> bool:
        if not bool(getattr(env.cfg, "erc_ablate_tc_global_assignment", False)):
            return True
        if not bool(getattr(env.cfg, "hrl_tc_global_assignment_adaptive_escape_enabled", True)):
            return False
        map_size = float(max(getattr(env.cfg, "map_size_m", 0.0), 0.0))
        min_map = float(max(getattr(env.cfg, "hrl_tc_global_assignment_escape_min_map_size_m", 12000.0), 0.0))
        if map_size < min_map:
            return False
        if str(getattr(env.cfg, "scenario", "")).upper() != "C":
            return False
        if self._timecritical_pressure_active(env):
            return True
        cover_thr = float(np.clip(getattr(env.cfg, "hrl_tc_global_assignment_escape_low_cover_threshold", 0.35), 0.0, 1.0))
        ratio_thr = float(np.clip(getattr(env.cfg, "hrl_tc_global_assignment_escape_max_lifeline_ratio", 0.55), 0.0, 1.0))
        for task in env.state.tasks.values():
            if not self._task_planner_active(task):
                continue
            if not self._is_timecritical_lightweight_task(task):
                continue
            ratio = float(self._task_lifeline_ratio(task))
            cover = float(np.clip(self._uav_emergency_cover_fraction(env, task), 0.0, 1.0))
            if ratio <= ratio_thr and cover <= cover_thr:
                return True
        return False

    def _hard_impact_event_active(self, env) -> bool:
        flags = self._last_refresh_flags if isinstance(self._last_refresh_flags, dict) else {}
        return bool(
            bool(flags.get("goal_invalid", False))
            or bool(flags.get("truck_dead_end", False))
            or bool(flags.get("uav_emergency", False))
            or bool(flags.get("high_priority_uncovered", False))
            or (bool(flags.get("map_update_hard", False)) and int(self._last_map_update_critical_count) > 0)
        )

    def _event_bonus_gain(self, env) -> float:
        if not bool(self.use_event_trigger):
            return 0.0
        if bool(getattr(env.cfg, "erc_ablate_event_scoring_bonus", False)):
            return 0.0
        conditional_enabled = bool(getattr(env.cfg, "hrl_event_bonus_conditional_enabled", True))
        base_gain = float(np.clip(getattr(env.cfg, "hrl_event_bonus_base_gain", 0.42), 0.0, 1.0))
        hard_gain = float(max(getattr(env.cfg, "hrl_event_bonus_hard_gain", 1.0), base_gain))
        if not conditional_enabled:
            return hard_gain
        return hard_gain if self._hard_impact_event_active(env) else base_gain

    def _direct_ready_timecritical_count(self, env) -> int:
        cnt = 0
        for task in env.state.tasks.values():
            if task.status != TaskStatus.PENDING or (not self._is_timecritical_lightweight_task(task)):
                continue
            ready = False
            for uid, st in env.state.agents.items():
                if st.kind != AgentKind.UAV or bool(getattr(st, "crashed", False)):
                    continue
                docked_actionable_fn = getattr(env, "_uav_docked_task_actionable_now", None)
                if callable(docked_actionable_fn):
                    try:
                        if bool(docked_actionable_fn(str(uid), task)):
                            ready = True
                            break
                    except Exception:
                        pass
                if bool(self._uav_task_feasible(env, str(uid), task)):
                    ready = True
                    break
            if ready:
                cnt += 1
        return int(cnt)

    def _truck_has_near_complete_bulk_goal(self, env, aid: str) -> bool:
        gid = self.state.goals.get(str(aid), None)
        if gid is None:
            return False
        task = env.state.tasks.get(str(gid), None)
        if task is None or task.status != TaskStatus.PENDING or task.kind != TaskKind.NORMAL:
            return False
        d = float(self._truck_task_distance(env, str(aid), task))
        if not np.isfinite(d):
            return False
        near_m = float(max(getattr(env.cfg, "hrl_support_chain_disable_if_bulk_near_m", 900.0), 0.0))
        return bool(near_m > 0.0 and d <= near_m)

    def _support_chain_condition_enabled(self, env, aid: str, task, support_gain: float) -> bool:
        if task is None or task.kind != TaskKind.EMERGENCY or task.status != TaskStatus.PENDING:
            return False
        # Generic, non-RC predicates only.
        pending_tc = int(
            sum(
                1
                for t in env.state.tasks.values()
                if t.status == TaskStatus.PENDING and self._is_timecritical_lightweight_task(t)
            )
        )
        if pending_tc < 2:
            return False
        critical_escape = False
        escape_scenarios_raw = getattr(env.cfg, "hrl_support_chain_critical_escape_scenarios", "C")
        if isinstance(escape_scenarios_raw, str):
            escape_scenarios = {
                x.strip().upper()
                for x in escape_scenarios_raw.replace(";", ",").split(",")
                if x.strip()
            }
        else:
            try:
                escape_scenarios = {str(x).strip().upper() for x in escape_scenarios_raw if str(x).strip()}
            except Exception:
                escape_scenarios = {"C"}
        scenario = str(getattr(env.cfg, "scenario", "")).upper()
        if (
            bool(getattr(env.cfg, "hrl_support_chain_critical_escape_enabled", True))
            and (not escape_scenarios or scenario in escape_scenarios or "ALL" in escape_scenarios)
            and self._is_timecritical_lightweight_task(task)
        ):
            ratio = float(self._task_lifeline_ratio(task))
            max_ratio = float(np.clip(getattr(env.cfg, "hrl_support_chain_critical_escape_max_lifeline_ratio", 0.55), 0.0, 1.0))
            cover = float(np.clip(self._uav_emergency_cover_fraction(env, task), 0.0, 1.0))
            cover_thr = float(np.clip(getattr(env.cfg, "hrl_support_chain_critical_escape_low_cover_threshold", 0.35), 0.0, 1.0))
            min_escape_gain = float(max(getattr(env.cfg, "hrl_support_chain_critical_escape_min_gain", 0.04), 0.0))
            critical_escape = bool(ratio <= max_ratio and cover <= cover_thr and float(max(support_gain, 0.0)) >= min_escape_gain)
        max_direct_ready = int(max(getattr(env.cfg, "hrl_support_chain_max_direct_ready_timecritical", 1), 0))
        if (not critical_escape) and int(self._direct_ready_timecritical_count(env)) > max_direct_ready:
            return False
        if float(max(support_gain, 0.0)) < float(max(getattr(env.cfg, "hrl_support_chain_min_gain_for_enable", 0.25), 0.0)):
            if not critical_escape:
                return False
        if (not critical_escape) and bool(self._execution_switch_pressure_active(env)):
            return False
        if bool(self._truck_has_near_complete_bulk_goal(env, str(aid))):
            return False
        return True

    def _support_conversion_quality(self, env) -> float:
        if not self._support_conversion_gate_enabled(env):
            return 1.0
        sup = float(max(getattr(env, "truck_forward_support_count_total", 0), 0.0))
        if sup <= 1e-9:
            return 1.0
        min_sup = float(max(getattr(env.cfg, "hrl_support_conversion_min_support_count", 8), 0))
        if sup < min_sup:
            return 1.0
        delivered = float(max(getattr(env, "uav_delivery_count_total", 0), 0.0))
        ratio = float(delivered / max(sup, 1.0))
        target = float(np.clip(getattr(env.cfg, "hrl_support_conversion_target_ratio", 0.45), 1e-6, 1.0))
        return float(np.clip(ratio / target, 0.0, 1.0))

    def _delivered_count(self, env) -> int:
        return int(
            sum(
                1
                for t in env.state.tasks.values()
                if t.status == TaskStatus.DELIVERED
            )
        )

    def _truck_has_direct_delivery_candidate(self, env, aid: str) -> bool:
        for task in env.state.tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
            if hasattr(env, "is_task_serviceable_by_agent") and (not bool(env.is_task_serviceable_by_agent(str(aid), task))):
                continue
            if not self._truck_task_reachable(env, str(aid), task):
                continue
            if task.kind == TaskKind.NORMAL:
                return True
            if task.kind == TaskKind.EMERGENCY and self._truck_emergency_relief_allowed(env, str(aid), task):
                return True
        return False

    def _scenario_tag(self, env) -> str:
        return str(getattr(env.cfg, "scenario", "")).upper().strip()

    def _real_case_active(self, env) -> bool:
        # Legacy alias kept for compatibility only.
        return bool(self._large_map_active(env) and self._timecritical_pressure_active(env))

    def _real_case_strong_mode(self, env) -> bool:
        # Legacy alias kept for compatibility only.
        return bool(self._execution_switch_pressure_active(env))

    def _rc_support_chain_lock_enabled(self, env) -> bool:
        return bool(getattr(env.cfg, "execution_commitment_enabled", True))

    def _rc_support_chain_override_margin(self, env) -> float:
        return float(max(getattr(env.cfg, "execution_commitment_override_lifeline_margin", 0.10), 0.0))

    def _rc_locked_support_chain_for_truck(self, env, aid: str) -> Optional[Dict[str, object]]:
        if not self._rc_support_chain_lock_enabled(env):
            return None
        info = self._support_bound_chain_info_for_truck(env, str(aid))
        if info is None:
            return None
        task = info.get("task", None)
        if task is None or task.status != TaskStatus.PENDING:
            return None
        if not self._is_timecritical_lightweight_task(task):
            return None
        return info

    def _rc_should_override_locked_support_chain(
        self,
        env,
        locked_task,
        candidate_task,
        aid: Optional[str] = None,
    ) -> bool:
        if not self._rc_support_chain_lock_enabled(env):
            return False
        if locked_task is None or candidate_task is None:
            return False
        if getattr(locked_task, "status", None) != TaskStatus.PENDING:
            return True
        if getattr(candidate_task, "status", None) != TaskStatus.PENDING:
            return False
        if not self._is_timecritical_lightweight_task(candidate_task):
            return False
        locked_ratio = float(np.clip(self._task_lifeline_ratio(locked_task), 0.0, 1.0))
        cand_ratio = float(np.clip(self._task_lifeline_ratio(candidate_task), 0.0, 1.0))
        margin = float(self._rc_support_chain_override_margin(env))
        if cand_ratio + margin < locked_ratio:
            return True
        if aid is not None:
            cand_tier = int(self._task_priority_tier(env, str(aid), candidate_task))
            locked_tier = int(self._task_priority_tier(env, str(aid), locked_task))
            if cand_tier > locked_tier and cand_ratio + 1e-6 < locked_ratio:
                return True
        return False

    def _rc_should_avoid_recent_loop_goal(self, env, aid: str, task_id: Optional[str]) -> bool:
        if (not bool(getattr(env.cfg, "truck_loop_break_enabled", False))) or (not self._execution_switch_pressure_active(env)):
            return False
        tid = str(task_id or "").strip()
        if not tid:
            return False
        window = int(max(getattr(env.cfg, "truck_loop_break_window_steps", 12), 0))
        if window <= 0:
            return False
        prev_goal = str(self._truck_recent_normal_prev_goal.get(str(aid), "")).strip()
        last_switch_step = int(self._truck_recent_normal_switch_step.get(str(aid), -10**9))
        now_step = int(getattr(env.state, "step_index", 0))
        return bool(prev_goal and tid == prev_goal and (now_step - last_switch_step) <= window)

    def _rc_best_truck_fallback_goal(self, env, aid: str, excluded_task_id: Optional[str] = None) -> Optional[str]:
        if not bool(getattr(env.cfg, "truck_force_nonnull_goal_enabled", False)):
            return None
        excluded = str(excluded_task_id or "").strip()
        best_norm_tid: Optional[str] = None
        best_norm_key: Optional[tuple] = None
        for task in env.state.tasks.values():
            if task.status != TaskStatus.PENDING or task.kind != TaskKind.NORMAL:
                continue
            tid = str(task.task_id)
            if excluded and tid == excluded:
                continue
            if self._rc_should_avoid_recent_loop_goal(env, str(aid), tid):
                continue
            if not self._truck_task_direct_serviceable(env, str(aid), task):
                continue
            if not self._truck_task_reachable(env, str(aid), task):
                continue
            d = float(self._truck_task_distance(env, str(aid), task))
            if not np.isfinite(d):
                continue
            urg = float(np.clip(self._norm_deadline_urgency(task, int(env.state.step_index)), 0.0, 1.0))
            key = (float(d - 220.0 * urg), float(d), tid)
            if best_norm_key is None or key < best_norm_key:
                best_norm_key = key
                best_norm_tid = tid
        if best_norm_tid is not None:
            return best_norm_tid

        best_support_tid: Optional[str] = None
        best_support_score = -1.0e9
        for task in env.state.tasks.values():
            if task.status != TaskStatus.PENDING or task.kind != TaskKind.EMERGENCY:
                continue
            tid = str(task.task_id)
            if excluded and tid == excluded:
                continue
            if self._rc_should_avoid_recent_loop_goal(env, str(aid), tid):
                continue
            if not self._truck_emergency_support_candidate(env, str(aid), task):
                continue
            gain_info = self._support_anchor_service_gain(env, str(aid), task)
            bind_info = self._support_bound_delivery_info(env, str(aid), task, gain_info=gain_info)
            gain = float(np.clip(float(gain_info.get("gain_score", 0.0)), 0.0, 1.0))
            new_serviceable = float(max(float(gain_info.get("new_serviceable_task_count", 0.0)), 0.0))
            tc_bind = float(bind_info.get("bound_timecritical", 0.0))
            if tc_bind <= 0.0 and new_serviceable < 1.0 and gain < 0.18:
                continue
            dist = float(self._truck_task_distance(env, str(aid), task))
            if not np.isfinite(dist):
                continue
            urg = float(np.clip(float(getattr(task, "urgency_score", self._norm_deadline_urgency(task, int(env.state.step_index)))), 0.0, 1.0))
            score = 3.0 * tc_bind + 1.6 * gain + 0.7 * urg - float(dist / max(self._distance_norm_m(env), 1.0))
            if score > best_support_score + 1e-9:
                best_support_score = score
                best_support_tid = tid
        return best_support_tid

    def _support_bind_horizon_steps(self, env) -> int:
        horizon = int(max(getattr(env.cfg, "hrl_support_bind_horizon_steps", 4), 0))
        map_size = float(max(getattr(env.cfg, "map_size_m", 5000.0), 0.0))
        if map_size >= 12000.0:
            horizon = int(max(horizon, int(max(getattr(env.cfg, "hrl_support_bind_horizon_steps_large_map", 8), 0))))
        return int(horizon)

    def _support_binding_is_strong_enough(self, env, task, bind_info: Dict[str, float], gain_info: Optional[Dict[str, float]] = None) -> bool:
        if task is None or task.status != TaskStatus.PENDING:
            return False
        if float(bind_info.get("bound_any", 0.0)) <= 0.0:
            return False
        if not self._support_need_active(env):
            return True

        gi = gain_info if isinstance(gain_info, dict) else {}
        gain_score = float(np.clip(float(gi.get("gain_score", 0.0)), 0.0, 1.0))
        new_serviceable = float(max(float(gi.get("new_serviceable_task_count", 0.0)), 0.0))
        relaxed_new = float(max(float(gi.get("new_relaxed_feasible_task_count", 0.0)), 0.0))
        eta_steps = float(bind_info.get("bound_eta_steps", float("inf")))
        horizon = int(self._support_bind_horizon_steps(env))
        ratio = float(np.clip(self._task_lifeline_ratio(task), 0.0, 1.0))
        critical = float(np.clip(getattr(env.cfg, "hrl_timecritical_lifeline_critical_ratio", 0.35), 0.0, 1.0))
        warning = float(np.clip(getattr(env.cfg, "hrl_timecritical_lifeline_warning_ratio", 0.55), 0.0, 1.0))
        urgency = float(np.clip(float(getattr(task, "urgency_score", self._norm_deadline_urgency(task, int(env.state.step_index)))), 0.0, 1.0))

        if self._is_timecritical_lightweight_task(task):
            post_support_d = float(gi.get("post_support_primary_distance_m", float("inf")))
            _, long_cap = self._uav_dispatch_distance_caps(env, task)
            dispatch_ready = bool(np.isfinite(post_support_d) and post_support_d <= float(0.82 * max(long_cap, 1.0)))
            fast_eta = bool(np.isfinite(eta_steps) and eta_steps <= float(horizon + 2))
            moderate_eta = bool(np.isfinite(eta_steps) and eta_steps <= float(horizon + 4))
            if ratio <= critical:
                return bool(moderate_eta and (dispatch_ready or gain_score >= 0.18 or relaxed_new >= 1.0))
            if ratio <= warning:
                return bool(fast_eta and (dispatch_ready or gain_score >= 0.22 or new_serviceable >= 1.0 or relaxed_new >= 1.0))
            return bool(urgency >= 0.85 and fast_eta and dispatch_ready and gain_score >= 0.30 and new_serviceable >= 1.0)

        if float(bind_info.get("bound_bulk", 0.0)) > 0.0:
            return bool(np.isfinite(eta_steps) and eta_steps <= float(horizon + 2) and gain_score >= 0.30 and new_serviceable >= 1.0)
        return False

    def _support_soft_clamp_blocks_task(self, env, aid: str, task, gain_info: Optional[Dict[str, float]] = None) -> bool:
        if task is None or task.kind != TaskKind.EMERGENCY:
            return False
        if not bool(getattr(env.cfg, "hrl_support_soft_clamp_enabled", False)):
            return False
        dist_m = float(self._truck_task_distance(env, str(aid), task))
        long_thr = float(max(getattr(env.cfg, "hrl_support_soft_clamp_long_distance_m", 2200.0), 0.0))
        if (not np.isfinite(dist_m)) or dist_m < long_thr:
            return False

        gi = gain_info if isinstance(gain_info, dict) else self._support_anchor_service_gain(env, str(aid), task)
        gain = float(np.clip(float(gi.get("gain_score", 0.0)), 0.0, 1.0))
        new_serviceable = float(max(float(gi.get("new_serviceable_task_count", 0.0)), 0.0))
        min_gain = float(np.clip(getattr(env.cfg, "hrl_support_soft_clamp_min_gain", 0.30), 0.0, 1.0))
        bindable_min = int(max(getattr(env.cfg, "hrl_support_soft_clamp_bindable_min_new_serviceable", 1), 0))

        has_binding = bool((new_serviceable >= float(bindable_min)) or (gain >= min_gain))
        if has_binding:
            return False

        need_direct = bool(getattr(env.cfg, "hrl_support_soft_clamp_require_direct_delivery_candidates", True))
        if need_direct and (not self._truck_has_direct_delivery_candidate(env, str(aid))):
            return False
        return True

    def _support_escape_hatch_allows(self, env, aid: str, task, gain_info: Optional[Dict[str, float]] = None) -> bool:
        if task is None or task.kind != TaskKind.EMERGENCY:
            return False
        if not bool(getattr(env.cfg, "hrl_support_escape_hatch_enabled", False)):
            return False

        pending_em = int(sum(1 for t in env.state.tasks.values() if t.status == TaskStatus.PENDING and t.kind == TaskKind.EMERGENCY))
        min_pending = int(max(getattr(env.cfg, "hrl_support_escape_hatch_min_pending_emergency", 6), 0))
        if pending_em < min_pending:
            return False

        gi = gain_info if isinstance(gain_info, dict) else self._support_anchor_service_gain(env, str(aid), task)
        gain = float(np.clip(float(gi.get("gain_score", 0.0)), 0.0, 1.0))
        new_serviceable = float(max(float(gi.get("new_serviceable_task_count", 0.0)), 0.0))
        min_gain = float(np.clip(getattr(env.cfg, "hrl_support_escape_hatch_min_gain", 0.32), 0.0, 1.0))
        urgency = float(np.clip(self._norm_deadline_urgency(task, int(env.state.step_index)), 0.0, 1.0))
        urg_thr = float(np.clip(getattr(env.cfg, "hrl_support_escape_hatch_min_urgency", 0.60), 0.0, 1.0))

        base_ok = bool((new_serviceable >= 1.0 or gain >= min_gain) and (urgency >= urg_thr or self._is_island_task(env, task)))
        if base_ok:
            return True

        # Controlled fallback for medium-scale C pockets: when a time-critical
        # task is already in warning/critical lifeline zone and UAV cover is low,
        # permit support chain entry even before strict bind is formed.
        if bool(getattr(env.cfg, "hrl_support_escape_hatch_allow_low_cover_timecritical", True)) and self._is_timecritical_lightweight_task(task):
            ratio = float(np.clip(self._task_lifeline_ratio(task), 0.0, 1.0))
            warning = float(np.clip(getattr(env.cfg, "hrl_timecritical_lifeline_warning_ratio", 0.55), 0.0, 1.0))
            low_cover_thr = float(np.clip(getattr(env.cfg, "hrl_support_escape_hatch_low_cover_threshold", 0.32), 0.0, 1.0))
            cover = float(np.clip(self._uav_emergency_cover_fraction(env, task), 0.0, 1.0))
            urgent_tc = float(np.clip(float(getattr(task, "urgency_score", urgency)), 0.0, 1.0))
            if ratio <= warning and cover < low_cover_thr and urgent_tc >= max(urg_thr - 0.10, 0.45):
                return True

        return False

    def _support_backoff_active(self, env, aid: str, task, gain_info: Optional[Dict[str, float]] = None) -> bool:
        if task is None or task.kind != TaskKind.EMERGENCY:
            return False
        if not bool(getattr(env.cfg, "hrl_support_no_gain_backoff_enabled", True)):
            return False
        now = int(env.state.step_index)
        until = int(self._support_backoff_until_step.get(str(aid), -1))
        if now > until:
            return False
        if self._support_escape_hatch_allows(env, str(aid), task, gain_info=gain_info):
            return False
        return True

    def _update_support_backoff_after_selection(self, env, aid: str, gain: float) -> None:
        if not bool(getattr(env.cfg, "hrl_support_no_gain_backoff_enabled", True)):
            return
        aid_s = str(aid)
        if float(gain) > 1e-9:
            self._support_no_gain_streak[aid_s] = 0
            self._support_backoff_until_step.pop(aid_s, None)
            return
        streak = int(self._support_no_gain_streak.get(aid_s, 0)) + 1
        self._support_no_gain_streak[aid_s] = int(streak)
        thr = int(max(getattr(env.cfg, "hrl_support_no_gain_streak_threshold", 3), 1))
        if streak >= thr:
            cooldown = int(max(getattr(env.cfg, "hrl_support_no_gain_cooldown_steps", 8), 0))
            self._support_backoff_until_step[aid_s] = int(env.state.step_index) + int(cooldown)


    def _uav_ride_stall_bonus_term(self, env, aid: str, task, dist_m: float) -> float:
        if not self._uav_ride_stall_release_enabled(env):
            return 0.0
        st = env.state.agents.get(str(aid), None)
        if st is None or st.kind != AgentKind.UAV or st.follow_target is None:
            return 0.0
        if task is None or task.kind != TaskKind.EMERGENCY or task.status != TaskStatus.PENDING:
            return 0.0
        if hasattr(env, "_uav_loaded") and (not bool(env._uav_loaded(str(aid)))):
            return 0.0
        if not self._uav_task_feasible(env, str(aid), task):
            return 0.0
        max_dist = float(max(getattr(env.cfg, "hrl_uav_ride_stall_max_dist_m", 1200.0), 1.0))
        if float(dist_m) > max_dist:
            return 0.0
        dwell = int(self._uav_docked_steps.get(str(aid), 0))
        trig = int(max(getattr(env.cfg, "hrl_uav_ride_stall_trigger_steps", 6), 0))
        if dwell < trig:
            return 0.0
        amp = float(max(getattr(env.cfg, "hrl_uav_ride_stall_bonus", 0.28), 0.0))
        sat = float(np.clip((float(dwell - trig + 1)) / float(max(trig, 1)), 0.0, 1.0))
        near = float(np.clip(1.0 - float(max(dist_m, 0.0)) / max_dist, 0.0, 1.0))
        return float(amp * (0.6 * sat + 0.4 * near))

    def _update_uav_docked_steps(self, env) -> None:
        next_map: Dict[str, int] = {}
        for aid, st in env.state.agents.items():
            if st.kind != AgentKind.UAV or bool(getattr(st, "crashed", False)):
                continue
            if st.follow_target is not None:
                next_map[str(aid)] = int(self._uav_docked_steps.get(str(aid), 0)) + 1
            else:
                next_map[str(aid)] = 0
        self._uav_docked_steps = next_map

    def _pending_task_pressure(self, env) -> Tuple[float, float]:
        # Dynamic pressure over pending NORMAL/EMERGENCY tasks using both count
        # and waiting age, then normalize to [0,1].
        if not self._dynamic_task_pressure_enabled(env):
            n_cnt = float(sum(1 for t in env.state.tasks.values() if t.status == TaskStatus.PENDING and t.kind == TaskKind.NORMAL))
            e_cnt = float(sum(1 for t in env.state.tasks.values() if t.status == TaskStatus.PENDING and t.kind == TaskKind.EMERGENCY))
            total = float(max(n_cnt + e_cnt, 1.0))
            return (float(np.clip(n_cnt / total, 0.0, 1.0)), float(np.clip(e_cnt / total, 0.0, 1.0)))
        step_now = int(env.state.step_index)
        horizon = float(max(getattr(env.cfg, "episode_horizon_steps", 200), 1))
        n_cnt = 0.0
        e_cnt = 0.0
        n_age = 0.0
        e_age = 0.0
        for task in env.state.tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
            created = int(getattr(task, "created_step", 0))
            age = float(max(step_now - created, 0)) / horizon
            if task.kind == TaskKind.NORMAL:
                n_cnt += 1.0
                n_age += age
            elif task.kind == TaskKind.EMERGENCY:
                e_cnt += 1.0
                e_age += age
        total = float(max(n_cnt + e_cnt, 1.0))
        n_ratio = float(n_cnt / total)
        e_ratio = float(e_cnt / total)
        n_age_mean = float(n_age / max(n_cnt, 1.0))
        e_age_mean = float(e_age / max(e_cnt, 1.0))
        n_sig = float(0.72 * n_ratio + 0.28 * np.clip(n_age_mean, 0.0, 1.0))
        e_sig = float(0.72 * e_ratio + 0.28 * np.clip(e_age_mean, 0.0, 1.0))
        den = float(max(n_sig + e_sig, 1e-9))
        return (float(np.clip(n_sig / den, 0.0, 1.0)), float(np.clip(e_sig / den, 0.0, 1.0)))

    def _rth_safety_factor(self, env) -> float:
        if self.rth_safety_factor is not None:
            return float(self.rth_safety_factor)
        return float(max(getattr(env.cfg, "rth_safety_factor", 1.2), 0.0))

    def _uav_task_reservation_ttl_steps(self, env) -> int:
        if not self._uav_task_reservation_enabled(env):
            return 0
        return int(max(getattr(env.cfg, "hrl_uav_task_reservation_ttl_steps", 8), 0))

    def _task_exclusive_contract_enabled(self, env) -> bool:
        return bool(
            self._er_hlns_coordination_active(env)
            and getattr(env.cfg, "hrl_task_exclusive_contract_enabled", True)
        )

    def _prune_task_contracts(self, env) -> None:
        """Release only resolved tasks or owners that are genuinely unavailable."""
        if not self._task_exclusive_contract_enabled(env):
            self._task_contract_by_task.clear()
            self._task_contract_by_agent.clear()
            return
        for tid, owner in list(self._task_contract_by_task.items()):
            task = env.state.tasks.get(str(tid), None)
            agent = env.state.agents.get(str(owner), None)
            if (
                task is None
                or task.status in (TaskStatus.DELIVERED, TaskStatus.FAILED)
                or agent is None
                or bool(getattr(agent, "crashed", False))
            ):
                self._task_contract_by_task.pop(str(tid), None)
                if self._task_contract_by_agent.get(str(owner)) == str(tid):
                    self._task_contract_by_agent.pop(str(owner), None)

    def _task_contract_owner(self, env, task_id: str) -> Optional[str]:
        self._prune_task_contracts(env)
        return self._task_contract_by_task.get(str(task_id), None)

    def _promote_globally_unreachable_normals(self, env) -> None:
        """Hand road-isolated bulk demand to UAVs as emergency relay demand.

        A normal task is promoted only when *no* live truck has a finite road
        path.  The former truck contract is released, while the task keeps its
        remaining demand so a UAV can deliver it over multiple reload sorties.
        """
        if (
            not self._er_hlns_coordination_active(env)
            or not bool(
                getattr(env.cfg, "hrl_unreachable_normal_uav_takeover_enabled", True)
            )
        ):
            return
        live_trucks = [
            st for st in env.state.agents.values()
            if st.kind == AgentKind.TRUCK and not bool(getattr(st, "crashed", False)) and st.node is not None
        ]
        if not live_trucks:
            return
        for task in env.state.tasks.values():
            if task.status != TaskStatus.PENDING or task.kind != TaskKind.NORMAL:
                continue
            reachable = False
            for truck in live_trucks:
                d = float(env._decision_shortest_path_distance(int(truck.node), int(task.demand_node)))
                if np.isfinite(d):
                    reachable = True
                    break
            if reachable:
                continue
            tid = str(task.task_id)
            old_owner = self._task_contract_by_task.pop(tid, None)
            if old_owner is not None and self._task_contract_by_agent.get(str(old_owner)) == tid:
                self._task_contract_by_agent.pop(str(old_owner), None)
            # Preserve the provenance for reporting, but make the task enter
            # the emergency/UAV candidate and launch pipeline immediately.
            setattr(task, "original_task_kind", "normal")
            setattr(task, "uav_takeover_from_unreachable_normal", True)
            task.kind = TaskKind.EMERGENCY
            task.urgency_score = float(max(float(getattr(task, "urgency_score", 0.0)), 0.95))

    def _apply_task_exclusive_contracts(self, env, goals: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
        """Make a task claim persistent and invisible to every other agent."""
        if not self._task_exclusive_contract_enabled(env):
            return goals
        self._prune_task_contracts(env)
        if self._er_hlns_coordination_active(env):
            self._promote_globally_unreachable_normals(env)
        out = dict(goals)
        # Existing contracts win every replan; their owner cannot be silently
        # replaced by a score fluctuation or a new global assignment.
        for tid, owner in self._task_contract_by_task.items():
            if owner in out:
                out[str(owner)] = str(tid)
            for aid, gid in list(out.items()):
                if str(aid) != str(owner) and gid is not None and str(gid) == str(tid):
                    out[str(aid)] = None
        # First new claimant owns the task. An agent may own one active task;
        # agent-id ordering makes simultaneous claims deterministic.
        for aid in sorted(out):
            gid = out.get(str(aid), None)
            task = env.state.tasks.get(str(gid), None) if gid is not None else None
            if task is None or task.status != TaskStatus.PENDING:
                continue
            prior = self._task_contract_by_task.get(str(task.task_id), None)
            if prior is not None and str(prior) != str(aid):
                out[str(aid)] = None
                continue
            prior_task = self._task_contract_by_agent.get(str(aid), None)
            if prior_task is not None and str(prior_task) != str(task.task_id):
                out[str(aid)] = str(prior_task)
                continue
            self._task_contract_by_task[str(task.task_id)] = str(aid)
            self._task_contract_by_agent[str(aid)] = str(task.task_id)
        return out

    def _uav_task_reservation_penalty(self, env) -> float:
        return float(max(getattr(env.cfg, "hrl_uav_task_reservation_penalty", 0.30), 0.0))

    def _uav_task_reservation_keep_bonus(self, env) -> float:
        return float(max(getattr(env.cfg, "hrl_uav_task_reservation_keep_bonus", 0.10), 0.0))

    def _prune_uav_task_reservations(self, env, step_now: int) -> None:
        ttl = self._uav_task_reservation_ttl_steps(env)
        if ttl <= 0:
            self._uav_task_reservations.clear()
            self._task_contract_by_task.clear()
            self._task_contract_by_agent.clear()
            return
        stale: List[str] = []
        for tid, (owner, step_mark) in self._uav_task_reservations.items():
            task = env.state.tasks.get(str(tid), None)
            if task is None or task.status != TaskStatus.PENDING or task.kind != TaskKind.EMERGENCY:
                stale.append(str(tid))
                continue
            if int(step_now) - int(step_mark) > int(ttl):
                stale.append(str(tid))
                continue
            ag = env.state.agents.get(str(owner), None)
            if ag is None or ag.kind != AgentKind.UAV or bool(getattr(ag, "crashed", False)):
                stale.append(str(tid))
                continue
            owner_goal = self.state.goals.get(str(owner), None)
            if owner_goal is not None and str(owner_goal) != str(tid) and (int(step_now) - int(step_mark)) >= 1:
                stale.append(str(tid))
                continue
        for tid in stale:
            self._uav_task_reservations.pop(str(tid), None)

    def _uav_task_reservation_term(self, env, aid: str, task_id: str) -> float:
        self.cluster_primary_candidate_count_total = int(self.cluster_primary_candidate_count_total) + 1
        if bool(getattr(env.cfg, "erc_ablate_cluster_primary_reservation", False)):
            self.cluster_primary_blocked_by_ablation_count_total = int(self.cluster_primary_blocked_by_ablation_count_total) + 1
            self.task_reservation_blocked_by_ablation_count_total = int(self.task_reservation_blocked_by_ablation_count_total) + 1
            return 0.0
        if not self._uav_task_reservations:
            return 0.0
        step_now = int(env.state.step_index)
        if self._er_hlns_coordination_active(env):
            self._prune_uav_task_reservations(env, step_now)
        rec = self._uav_task_reservations.get(str(task_id), None)
        if rec is None:
            return 0.0
        owner, _ = rec
        if str(owner) == str(aid):
            self.cluster_primary_applied_count_total = int(self.cluster_primary_applied_count_total) + 1
            return float(self._uav_task_reservation_keep_bonus(env))
        owner_state = env.state.agents.get(str(owner), None)
        if owner_state is None or owner_state.kind != AgentKind.UAV or bool(getattr(owner_state, "crashed", False)):
            return 0.0
        self.cluster_primary_applied_count_total = int(self.cluster_primary_applied_count_total) + 1
        return float(-self._uav_task_reservation_penalty(env))

    def _refresh_uav_task_reservations(self, env, goals: Dict[str, Optional[str]]) -> None:
        if bool(getattr(env.cfg, "erc_ablate_cluster_primary_reservation", False)):
            self.task_reservation_blocked_by_ablation_count_total = int(self.task_reservation_blocked_by_ablation_count_total) + int(len(goals))
            return
        step_now = int(env.state.step_index)
        if self._er_hlns_coordination_active(env):
            self._prune_uav_task_reservations(env, step_now)
        ttl = self._uav_task_reservation_ttl_steps(env)
        if ttl <= 0:
            return
        for aid, gid in goals.items():
            if gid is None:
                continue
            ag = env.state.agents.get(str(aid), None)
            if ag is None or ag.kind != AgentKind.UAV or bool(getattr(ag, "crashed", False)):
                continue
            task = env.state.tasks.get(str(gid), None)
            if task is None or task.status != TaskStatus.PENDING or task.kind != TaskKind.EMERGENCY:
                continue
            self._uav_task_reservations[str(task.task_id)] = (str(aid), int(step_now))
            self.task_reservation_applied_count_total = int(self.task_reservation_applied_count_total) + 1

    def _uav_task_reservation_exec_enabled(self, env) -> bool:
        # Reservation protects a physical pre-launch commitment; it must stay
        # active even when event-triggered replanning is disabled.
        return bool(
            self._er_hlns_coordination_active(env)
            and getattr(env.cfg, "hrl_uav_task_reservation_exec_enabled", True)
        )

    def _uav_task_reservation_stale_steps(self, env) -> int:
        return int(max(getattr(env.cfg, "hrl_uav_task_reservation_stale_steps", 24), 1))

    def _uav_task_reserved_by_other(self, aid: str, task_id: str) -> bool:
        rec = self._uav_task_reservation_state_by_task.get(str(task_id), None)
        if not isinstance(rec, dict):
            return False
        owner = str(rec.get("uav_id", ""))
        status = str(rec.get("status", ""))
        if owner == "" or owner == str(aid):
            return False
        return bool(status in {"reserved_prelaunch", "airborne_committed", "servicing"})

    def _uav_reservation_release(self, task_id: str, reason: str = "released") -> None:
        rec = self._uav_task_reservation_state_by_task.pop(str(task_id), None)
        if isinstance(rec, dict):
            owner = str(rec.get("uav_id", ""))
            if owner:
                if self._uav_task_reservation_by_uav.get(owner, "") == str(task_id):
                    self._uav_task_reservation_by_uav.pop(owner, None)
            self.uav_task_reservation_release_count_total = int(self.uav_task_reservation_release_count_total) + 1
            if str(reason) == "stale":
                self.uav_task_reservation_stale_count_total = int(self.uav_task_reservation_stale_count_total) + 1

    def _uav_reservation_assign(self, env, aid: str, task_id: str, status: str = "reserved_prelaunch") -> None:
        step_now = int(env.state.step_index)
        prev = self._uav_task_reservation_state_by_task.get(str(task_id), None)
        if isinstance(prev, dict):
            old_owner = str(prev.get("uav_id", ""))
            if old_owner and old_owner != str(aid):
                self._uav_task_reservation_by_uav.pop(str(old_owner), None)
        old_task = self._uav_task_reservation_by_uav.get(str(aid), "")
        if old_task and old_task != str(task_id):
            self._uav_reservation_release(str(old_task), reason="reassign")
        self._uav_task_reservation_by_uav[str(aid)] = str(task_id)
        if not isinstance(prev, dict):
            self.uav_task_reserved_count_total = int(self.uav_task_reserved_count_total) + 1
            prev = {
                "uav_id": str(aid),
                "reserved_step": int(step_now),
                "status": str(status),
                "launch_base": int(getattr(env, "uav_launch_count_total", 0)),
                "delivery_base": int(getattr(env, "uav_delivery_count_total", 0)),
            }
        prev["uav_id"] = str(aid)
        prev["status"] = str(status)
        prev["last_step"] = int(step_now)
        self._uav_task_reservation_state_by_task[str(task_id)] = prev

    def _uav_has_basic_launch_readiness(self, env, aid: str) -> bool:
        st = env.state.agents.get(str(aid), None)
        if st is None or st.kind != AgentKind.UAV or bool(getattr(st, "crashed", False)):
            return False
        if bool(getattr(st, "uav_needs_reload_flag", False)):
            return False
        loaded_fn = getattr(env, "_uav_loaded", None)
        if callable(loaded_fn):
            try:
                if not bool(loaded_fn(str(aid))):
                    return False
            except Exception:
                return False
        # Reservation is not a launch authorization. A loaded docked UAV may
        # reserve its task at any positive SOC and keep charging; the
        # environment later authorizes takeoff only after the complete
        # delivery-and-recovery energy check passes.
        return bool(float(getattr(st, "battery", 0.0)) > 1e-9)

    def _update_uav_reservation_states(self, env, goals: Dict[str, Optional[str]]) -> None:
        if not self._uav_task_reservation_exec_enabled(env):
            self._uav_task_reservation_state_by_task.clear()
            self._uav_task_reservation_by_uav.clear()
            self._uav_intent_signal_by_uav.clear()
            return
        step_now = int(env.state.step_index)
        stale_steps = int(self._uav_task_reservation_stale_steps(env))
        # Prune invalid/completed reservations and track completion transitions.
        for tid, rec in list(self._uav_task_reservation_state_by_task.items()):
            task = env.state.tasks.get(str(tid), None)
            status_prev = str(rec.get("status", ""))
            if task is None or task.status == TaskStatus.FAILED:
                if status_prev == "airborne_committed":
                    self.uav_airborne_safety_abort_count_total = int(self.uav_airborne_safety_abort_count_total) + 1
                self._uav_reservation_release(str(tid), reason="task_failed")
                continue
            if task.status == TaskStatus.DELIVERED:
                if status_prev in {"airborne_committed", "servicing"}:
                    self.uav_airborne_task_completed_count_total = int(self.uav_airborne_task_completed_count_total) + 1
                launch_delta = int(getattr(env, "uav_launch_count_total", 0)) - int(rec.get("launch_base", 0))
                delivery_delta = int(getattr(env, "uav_delivery_count_total", 0)) - int(rec.get("delivery_base", 0))
                if launch_delta > 0:
                    self.uav_task_reserved_to_launch_count_total = int(self.uav_task_reserved_to_launch_count_total) + 1
                if delivery_delta > 0:
                    self.uav_task_reserved_to_completion_count_total = int(self.uav_task_reserved_to_completion_count_total) + 1
                self._uav_reservation_release(str(tid), reason="completed")
                continue
            owner = str(rec.get("uav_id", ""))
            ag = env.state.agents.get(owner, None)
            if ag is None or ag.kind != AgentKind.UAV or bool(getattr(ag, "crashed", False)):
                self._uav_reservation_release(str(tid), reason="owner_invalid")
                continue
            if status_prev == "reserved_prelaunch":
                if int(step_now) - int(rec.get("reserved_step", step_now)) >= stale_steps:
                    self._uav_reservation_release(str(tid), reason="stale")
                    continue
            goal_owner = goals.get(str(owner), self.state.goals.get(str(owner), None))
            if str(goal_owner) != str(tid):
                if status_prev == "reserved_prelaunch" and int(step_now) - int(rec.get("reserved_step", step_now)) >= 2:
                    self._uav_reservation_release(str(tid), reason="goal_changed")
                    continue
            task_pending = bool(task.status == TaskStatus.PENDING)
            if task_pending and getattr(ag, "follow_target", None) is None:
                if status_prev != "airborne_committed":
                    self.uav_task_airborne_committed_count_total = int(self.uav_task_airborne_committed_count_total) + 1
                    self.uav_task_reserved_to_launch_count_total = int(self.uav_task_reserved_to_launch_count_total) + 1
                rec["status"] = "airborne_committed"
            if task_pending and bool(getattr(task, "first_service_step", None) is not None):
                rec["status"] = "servicing"
            rec["last_step"] = int(step_now)
            self._uav_task_reservation_state_by_task[str(tid)] = rec

    def _ensure_uav_emergency_reservations(self, env, goals: Dict[str, Optional[str]], used_tasks: set) -> None:
        if not self._uav_task_reservation_exec_enabled(env):
            return
        step_now = int(env.state.step_index)
        # Refresh states first.
        self._update_uav_reservation_states(env, goals)
        # Assign nearest unreserved emergency task for idle/prelaunch UAV.
        uav_ids = [str(aid) for aid, st in env.state.agents.items() if st.kind == AgentKind.UAV and not bool(getattr(st, "crashed", False))]
        # Keep deterministic order.
        uav_ids.sort()
        pending_emerg = [
            t for t in env.state.tasks.values()
            if t.kind == TaskKind.EMERGENCY and t.status == TaskStatus.PENDING
        ]
        if not pending_emerg:
            return
        pending_emerg.sort(key=lambda t: str(t.task_id))
        for aid in uav_ids:
            st = env.state.agents.get(str(aid), None)
            if st is None or st.kind != AgentKind.UAV:
                continue
            # airborne UAV is already committed.
            if getattr(st, "follow_target", None) is None:
                gid = goals.get(str(aid), self.state.goals.get(str(aid), None))
                task_cur = env.state.tasks.get(str(gid), None) if gid is not None else None
                if task_cur is not None and task_cur.kind == TaskKind.EMERGENCY and task_cur.status == TaskStatus.PENDING:
                    self._uav_reservation_assign(env, str(aid), str(task_cur.task_id), status="airborne_committed")
                continue
            # not executing, but require basic readiness for reservation.
            if not self._uav_has_basic_launch_readiness(env, str(aid)):
                continue
            cur_gid = goals.get(str(aid), None)
            cur_task = env.state.tasks.get(str(cur_gid), None) if cur_gid is not None else None
            if cur_task is not None and cur_task.kind == TaskKind.EMERGENCY and cur_task.status == TaskStatus.PENDING:
                self._uav_reservation_assign(env, str(aid), str(cur_task.task_id), status="reserved_prelaunch")
                # A retained docked commitment must also publish an assist
                # intent. Previously only a newly selected task did so, which
                # meant the truck-side route-preserving support layer never
                # saw the very tasks that were being held while charging.
                tx, ty = env._node_xy(int(cur_task.demand_node))
                self._uav_intent_signal_by_uav[str(aid)] = {
                    "uav_intent_task_id": str(cur_task.task_id),
                    "uav_intent_target_xy": (float(tx), float(ty)),
                    "uav_intent_nearest_launch_area": int(cur_task.demand_node),
                    "uav_intent_candidate_truck_id": str(getattr(st, "follow_target", "") or ""),
                    "step": int(step_now),
                }
                continue
            best_tid = None
            best_d = float("inf")
            best_urg = -1.0
            best_life = 2.0
            for task in pending_emerg:
                tid = str(task.task_id)
                if self._uav_task_reserved_by_other(str(aid), tid):
                    continue
                if tid in used_tasks:
                    continue
                d = float(env._agent_distance_to_task(str(aid), task))
                if not np.isfinite(d):
                    continue
                urg = float(np.clip(float(getattr(task, "urgency_score", self._norm_deadline_urgency(task, step_now))), 0.0, 1.0))
                life = float(self._task_lifeline_ratio(task)) if self._is_timecritical_lightweight_task(task) else 1.0
                if (d < best_d - 1e-9) or (abs(d - best_d) <= 1e-9 and (urg > best_urg + 1e-9 or (abs(urg - best_urg) <= 1e-9 and life < best_life - 1e-9))):
                    best_d = float(d)
                    best_urg = float(urg)
                    best_life = float(life)
                    best_tid = str(tid)
            if best_tid is None:
                continue
            goals[str(aid)] = str(best_tid)
            used_tasks.add(str(best_tid))
            self._uav_reservation_assign(env, str(aid), str(best_tid), status="reserved_prelaunch")
            # Build intent signal for truck-side assist heuristics.
            t = env.state.tasks.get(str(best_tid), None)
            if t is not None:
                tx, ty = env._node_xy(int(t.demand_node))
                cand_truck = str(getattr(st, "follow_target", "") or "")
                self._uav_intent_signal_by_uav[str(aid)] = {
                    "uav_intent_task_id": str(best_tid),
                    "uav_intent_target_xy": (float(tx), float(ty)),
                    "uav_intent_nearest_launch_area": int(t.demand_node),
                    "uav_intent_candidate_truck_id": cand_truck,
                    "step": int(step_now),
                }

    def _truck_uav_assist_enabled(self, env) -> bool:
        # Route-preserving support is an execution constraint, not an event
        # refresh policy.  Gate it by its own configuration only.
        return bool(getattr(env.cfg, "hrl_uav_assist_enabled", True))

    def _update_truck_uav_assist_waypoints(self, env, goals: Dict[str, Optional[str]]) -> None:
        self._truck_assist_waypoint_by_truck.clear()
        self._update_tc_support_required_assist_waypoints(env, goals)
        if not self._truck_uav_assist_enabled(env):
            return
        max_extra_m = float(max(getattr(env.cfg, "hrl_uav_assist_max_extra_distance_m", 600.0), 0.0))
        max_extra_ratio = float(max(getattr(env.cfg, "hrl_uav_assist_max_extra_ratio", 0.20), 0.0))
        min_reduction = float(max(getattr(env.cfg, "hrl_uav_assist_min_launch_distance_reduction_m", 400.0), 0.0))
        step_now = int(env.state.step_index)
        if not self._uav_intent_signal_by_uav:
            return
        truck_ids = [str(aid) for aid, st in env.state.agents.items() if st.kind == AgentKind.TRUCK and not bool(getattr(st, "crashed", False))]
        pending_normal_count = sum(
            1
            for task in env.state.tasks.values()
            if task.status == TaskStatus.PENDING and task.kind == TaskKind.NORMAL
        )
        used_assist_tasks: set = set()
        for tid in truck_ids:
            goal_id = goals.get(str(tid), None)
            goal_task = env.state.tasks.get(str(goal_id), None) if goal_id is not None else None
            st = env.state.agents.get(str(tid), None)
            if st is None:
                continue
            if bool(getattr(st, "service_timer", 0) > 0) or bool(getattr(st, "is_servicing", False)):
                self.truck_uav_assist_rejected_normal_service_count_total = int(self.truck_uav_assist_rejected_normal_service_count_total) + 1
                continue
            node_now = int(getattr(st, "node", -1))
            normal_goal_active = bool(goal_task is not None and goal_task.status == TaskStatus.PENDING and goal_task.kind == TaskKind.NORMAL)
            goal_node = int(getattr(goal_task, "demand_node", int(node_now))) if normal_goal_active else int(node_now)
            if node_now < 0 or (normal_goal_active and goal_node < 0):
                continue
            try:
                d0 = (
                    float(env._decision_shortest_path_distance(int(node_now), int(goal_node)))
                    if normal_goal_active
                    else 0.0
                )
            except Exception:
                continue
            if normal_goal_active and not np.isfinite(d0):
                continue
            has_serviceable_normal = False
            if (not normal_goal_active) and pending_normal_count > 0:
                for cand in env.state.tasks.values():
                    if cand.status != TaskStatus.PENDING or cand.kind != TaskKind.NORMAL:
                        continue
                    if hasattr(env, "is_task_serviceable_by_agent") and (not bool(env.is_task_serviceable_by_agent(str(tid), cand))):
                        continue
                    try:
                        d_cand = float(env._decision_shortest_path_distance(int(node_now), int(cand.demand_node)))
                    except Exception:
                        d_cand = float("inf")
                    if np.isfinite(d_cand):
                        has_serviceable_normal = True
                        break
            idle_support_mode = bool((not normal_goal_active) and (pending_normal_count <= 0 or not has_serviceable_normal))
            best_pick: Optional[Dict[str, Any]] = None
            # inspect UAV intents tied to this truck or nearby truck-follower UAV.
            for uid, sig in self._uav_intent_signal_by_uav.items():
                uag = env.state.agents.get(str(uid), None)
                if uag is None or uag.kind != AgentKind.UAV or bool(getattr(uag, "crashed", False)):
                    continue
                if str(getattr(uag, "follow_target", "") or "") != str(tid):
                    continue
                task_id = str(sig.get("uav_intent_task_id", ""))
                if task_id in used_assist_tasks:
                    continue
                task = env.state.tasks.get(task_id, None)
                if task is None or task.status != TaskStatus.PENDING or task.kind != TaskKind.EMERGENCY:
                    continue
                owner = self._uav_task_reservation_state_by_task.get(task_id, {})
                if str(owner.get("uav_id", "")) != str(uid):
                    continue
                if str(owner.get("status", "")) not in {"reserved_prelaunch"}:
                    continue
                self.truck_uav_assist_candidate_count_total = int(self.truck_uav_assist_candidate_count_total) + 1
                launch_node = int(getattr(task, "demand_node", -1))
                if launch_node < 0:
                    continue
                try:
                    d_to_launch = float(env._decision_shortest_path_distance(int(node_now), int(launch_node)))
                    d_launch_to_goal = (
                        float(env._decision_shortest_path_distance(int(launch_node), int(goal_node)))
                        if normal_goal_active
                        else 0.0
                    )
                except Exception:
                    continue
                if (not np.isfinite(d_to_launch)) or (not np.isfinite(d_launch_to_goal)):
                    continue
                d1 = float(d_to_launch + d_launch_to_goal)
                delta = float(d1 - d0)
                # Both absolute and relative detour limits are hard. A short
                # normal route may not absorb a large percentage detour, and a
                # long route may not absorb a large absolute detour.
                if normal_goal_active and (
                    delta > max_extra_m + 1e-9
                    or delta / max(d0, 1.0) > max_extra_ratio + 1e-9
                ):
                    self.truck_uav_assist_rejected_extra_distance_count_total = int(self.truck_uav_assist_rejected_extra_distance_count_total) + 1
                    continue
                # Approximate launch-distance reduction with truck->task approach.
                tx, ty = self._agent_xy(env, str(tid))
                ux, uy = self._agent_xy(env, str(uid))
                node_t = env.topology.nodes[int(task.demand_node)]
                du = float(np.hypot(float(node_t.x) - float(ux), float(node_t.y) - float(uy)))
                dt = float(np.hypot(float(node_t.x) - float(tx), float(node_t.y) - float(ty)))
                reduction = float(max(du - dt, 0.0))
                # When the UAV is docked on this truck, current UAV and truck
                # positions coincide; the useful gain is the truck's future
                # road progress toward the launch/task node, not a present
                # UAV-vs-truck position difference.
                effective_reduction = float(max(reduction, d_to_launch))
                if effective_reduction < min_reduction - 1e-9:
                    continue
                life = float(self._task_lifeline_ratio(task)) if self._is_timecritical_lightweight_task(task) else 1.0
                urg = float(np.clip(float(getattr(task, "urgency_score", self._norm_deadline_urgency(task, step_now))), 0.0, 1.0))
                if idle_support_mode:
                    score = float(urg + (1.0 - np.clip(life, 0.0, 1.0)) + effective_reduction / max(self._distance_norm_m(env), 1.0) - 0.15 * d_to_launch / max(self._distance_norm_m(env), 1.0))
                    better = bool((best_pick is None) or score > float(best_pick.get("score", -float("inf"))) + 1e-9)
                else:
                    score = float(-delta)
                    better = bool((best_pick is None) or (delta < float(best_pick.get("delta", float("inf"))) - 1e-9))
                if better:
                    best_pick = {
                        "uav_id": str(uid),
                        "task_id": str(task_id),
                        "launch_node": int(launch_node),
                        "delta": float(delta),
                        "score": float(score),
                        "idle_support": bool(idle_support_mode),
                    }
            if best_pick is None:
                continue
            self.truck_uav_assist_accepted_count_total = int(self.truck_uav_assist_accepted_count_total) + 1
            self.truck_uav_assist_extra_distance_m_total = float(self.truck_uav_assist_extra_distance_m_total) + float(best_pick.get("delta", 0.0))
            used_assist_tasks.add(str(best_pick.get("task_id", "")))
            self._truck_assist_waypoint_by_truck[str(tid)] = {
                "assist_waypoint_insert": True,
                "idle_support": bool(best_pick.get("idle_support", False)),
                "uav_id": str(best_pick.get("uav_id", "")),
                "task_id": str(best_pick.get("task_id", "")),
                "launch_node": int(best_pick.get("launch_node", -1)),
                "normal_goal_task_id": str(goal_id) if normal_goal_active else "",
                "step": int(step_now),
                "extra_distance_m": float(best_pick.get("delta", 0.0)),
            }
            self._truck_assist_pending_windows.append(
                {
                    "step": int(step_now),
                    "uav_id": str(best_pick.get("uav_id", "")),
                    "task_id": str(best_pick.get("task_id", "")),
                    "launch_base": int(getattr(env, "uav_launch_count_total", 0)),
                    "delivery_base": int(getattr(env, "uav_delivery_count_total", 0)),
                }
            )
        self._update_tc_support_required_assist_waypoints(env, goals)

    def _update_tc_support_required_assist_waypoints(self, env, goals: Dict[str, Optional[str]]) -> None:
        if not bool(getattr(env.cfg, "erc_tc_support_required_enabled", False)):
            return
        if not bool(getattr(env.cfg, "erc_tc_support_anchor_waypoint_enabled", False)):
            return
        step_now = int(getattr(env.state, "step_index", 0))
        for truck_id, anchor_node in list(self._support_bound_chain_anchor_node_by_truck.items()):
            info = self._support_bound_chain_info_for_truck(env, str(truck_id))
            if info is None or self._tc_support_chain_class.get(str(truck_id), "") != "support_required":
                self._support_bound_chain_anchor_node_by_truck.pop(str(truck_id), None)
                continue
            task_id = str(info.get("task_id", ""))
            uav_id = str(info.get("uav_id", ""))
            task = info.get("task", None)
            if task is None or not task_id or not uav_id:
                continue
            try:
                launch_node = int(anchor_node)
            except Exception:
                continue
            self._truck_assist_waypoint_by_truck[str(truck_id)] = {
                "assist_waypoint_insert": True,
                "idle_support": True,
                "uav_id": str(uav_id),
                "task_id": str(task_id),
                "launch_node": int(launch_node),
                "normal_goal_task_id": str(goals.get(str(truck_id), "") or ""),
                "step": int(step_now),
                "extra_distance_m": 0.0,
                "tc_support_required": True,
            }

    def _update_truck_assist_outcomes(self, env, force_finalize: bool = False) -> None:
        if not self._truck_assist_pending_windows:
            return
        now_step = int(env.state.step_index)
        keep: List[Dict[str, Any]] = []
        for rec in list(self._truck_assist_pending_windows):
            age = int(now_step) - int(rec.get("step", now_step))
            launch_delta = int(getattr(env, "uav_launch_count_total", 0)) - int(rec.get("launch_base", 0))
            delivery_delta = int(getattr(env, "uav_delivery_count_total", 0)) - int(rec.get("delivery_base", 0))
            if launch_delta > 0:
                self.truck_uav_assist_launch_success_count_total = int(self.truck_uav_assist_launch_success_count_total) + 1
            if delivery_delta > 0:
                self.truck_uav_assist_followed_by_emergency_completion_count_total = int(
                    self.truck_uav_assist_followed_by_emergency_completion_count_total
                ) + 1
            if force_finalize or launch_delta > 0 or delivery_delta > 0 or age >= 24:
                continue
            keep.append(rec)
        self._truck_assist_pending_windows = keep

    def _episode_reset_if_needed(self, env) -> None:
        step_now = int(env.state.step_index)
        if step_now == 0 and self._last_seen_step > 0:
            self.state = RollingPlannerState()
            self._reset_goal_switch_breakdown()
            self._event_window_start_step = 0
            self._event_replans_in_window = 0
            self._low_value_event_streak = 0
            self._normal_pending_unchanged_steps = 0
            self._last_pending_normal_count = -1
            self._delivery_stall_steps = 0
            self._last_delivered_count = -1
            self._last_island_serviceability = 1.0
            self.map_update_hard_seen_count_total = 0
            self.map_update_hard_actionable_count_total = 0
            self.map_update_hard_deferred_count_total = 0
            self.map_update_hard_immediate_refresh_count_total = 0
            for _k in list(self._map_update_hard_actionable_reasons_total.keys()):
                self._map_update_hard_actionable_reasons_total[_k] = 0
            self.map_update_hard_seen_count_total = 0
            self.map_update_hard_actionable_count_total = 0
            self.map_update_hard_deferred_count_total = 0
            self.map_update_hard_immediate_refresh_count_total = 0
            for _k in list(self._map_update_hard_actionable_reasons_total.keys()):
                self._map_update_hard_actionable_reasons_total[_k] = 0
            self._uav_task_reservations.clear()
            self._task_contract_by_task.clear()
            self._task_contract_by_agent.clear()
            self._uav_docked_steps.clear()
            self._initial_directional_plan_step = -1
            self._initial_directional_truck_sector.clear()
            self._initial_directional_uav_truck.clear()
            self._initial_directional_sector_stats.clear()
            self.uav_recovery_feasibility_eval_count_total = 0
            self.unique_agent_task_candidate_count_total = 0
            self.truck_support_selected_count_total = 0
            self.truck_support_improves_serviceability_count_total = 0
            self.truck_support_no_gain_count_total = 0
            self.support_selected_count_total = 0
            self.support_improves_serviceability_count_total = 0
            self.support_no_gain_count_total = 0
            self._uav_task_feasible_cache_step = -1
            self._uav_task_feasible_cache.clear()
            self._step_unique_agent_task_keys.clear()
            self._support_anchor_until_step.clear()
            self._support_anchor_gain.clear()
            self._support_anchor_task_id.clear()
            self._support_bound_chain_until_step.clear()
            self._support_bound_chain_task_id.clear()
            self._support_bound_chain_uav_id.clear()
            self._support_bound_chain_truck_by_uav.clear()
            self._support_bound_chain_anchor_node_by_truck.clear()
            self._support_bound_chain_latest_start_by_truck.clear()
            self._tc_support_chain_class.clear()
            self.tc_direct_feasible_count_total = 0
            self.tc_support_required_count_total = 0
            self.tc_truly_infeasible_count_total = 0
            self.tc_support_lock_created_count_total = 0
            self.tc_support_lock_to_dispatch_count_total = 0
            self._relaxed_chain_until_step.clear()
            self.timecritical_tier3_candidate_count_total = 0
            self.timecritical_tier3_selected_count_total = 0
            self.timecritical_tier2_candidate_count_total = 0
            self.timecritical_tier2_selected_count_total = 0
            self.timecritical_candidate_ignored_count_total = 0
            self.support_selected_with_bound_timecritical_delivery_count_total = 0
            self.support_selected_without_bound_timecritical_delivery_count_total = 0
            self.support_filtered_no_bound_timecritical_delivery_count_total = 0
            self.support_selected_with_bound_bulk_delivery_count_total = 0
            self.support_bound_dispatch_count_total = 0
            self.support_bound_recovery_redirect_count_total = 0
            self.support_no_gain_backoff_block_count_total = 0
            self.support_proxy_candidate_count_total = 0
            self.support_relay_reserved_count_total = 0
            self.truck_emergency_goal_assigned_count_total = 0
            self._support_relay_force_step.clear()
            self._support_no_gain_streak.clear()
            self._support_backoff_until_step.clear()
            self._task_last_goal_step.clear()
            self._task_goal_exposure_count.clear()
            self._uav_anchor_task_goal.clear()
            self._uav_transfer_target_truck.clear()
            self._uav_transfer_target_task.clear()
            self.uav_transfer_hint_issue_count_total = 0
            self.uav_transfer_hint_keep_count_total = 0
            self._truck_recent_normal_prev_goal.clear()
            self._truck_recent_normal_switch_step.clear()
            self._task_recent_prev_goal.clear()
            self._task_recent_switch_step.clear()
            self._uav_recent_truck_anchor_prev_goal.clear()
            self._uav_recent_truck_anchor_switch_step.clear()
            self._normal_task_unreachable_streak.clear()
            self.normal_unreachable_task_count_total = 0
            self.timecritical_tier3_candidate_count_total = 0
            self.timecritical_tier3_selected_count_total = 0
            self.timecritical_tier2_candidate_count_total = 0
            self.timecritical_tier2_selected_count_total = 0
            self.timecritical_candidate_ignored_count_total = 0
            self.support_selected_with_bound_timecritical_delivery_count_total = 0
            self.support_selected_without_bound_timecritical_delivery_count_total = 0
            self.support_filtered_no_bound_timecritical_delivery_count_total = 0
            self.support_selected_with_bound_bulk_delivery_count_total = 0
            self.support_bound_dispatch_count_total = 0
            self.support_bound_recovery_redirect_count_total = 0
            self.support_no_gain_backoff_block_count_total = 0
            self.support_proxy_candidate_count_total = 0
            self.support_relay_reserved_count_total = 0
            self.truck_emergency_goal_assigned_count_total = 0
            self._support_relay_force_step.clear()
            self._support_no_gain_streak.clear()
            self._support_backoff_until_step.clear()
            self._initial_directional_plan_step = -1
            self._initial_directional_truck_sector.clear()
            self._initial_directional_uav_truck.clear()
            self._initial_directional_sector_stats.clear()
            self.uav_recovery_feasibility_eval_count_total = 0
            self.unique_agent_task_candidate_count_total = 0
            self.truck_support_selected_count_total = 0
            self.truck_support_improves_serviceability_count_total = 0
            self.truck_support_no_gain_count_total = 0
            self.support_selected_count_total = 0
            self.support_improves_serviceability_count_total = 0
            self.support_no_gain_count_total = 0
            self._uav_task_feasible_cache_step = -1
            self._uav_task_feasible_cache.clear()
            self._step_unique_agent_task_keys.clear()
            self._support_anchor_until_step.clear()
            self._support_anchor_gain.clear()
            self._support_anchor_task_id.clear()
            self._support_bound_chain_until_step.clear()
            self._support_bound_chain_task_id.clear()
            self._support_bound_chain_uav_id.clear()
            self._support_bound_chain_truck_by_uav.clear()
            self._support_bound_chain_anchor_node_by_truck.clear()
            self._support_bound_chain_latest_start_by_truck.clear()
            self._tc_support_chain_class.clear()
            self._relaxed_chain_until_step.clear()
        if step_now < self._last_seen_step:
            self.state = RollingPlannerState()
            self._reset_goal_switch_breakdown()
            self._event_window_start_step = 0
            self._event_replans_in_window = 0
            self._low_value_event_streak = 0
            self._normal_pending_unchanged_steps = 0
            self._last_pending_normal_count = -1
            self._delivery_stall_steps = 0
            self._last_delivered_count = -1
            self._last_island_serviceability = 1.0
            self._uav_task_reservations.clear()
            self._uav_docked_steps.clear()
            self._support_bound_chain_until_step.clear()
            self._support_bound_chain_task_id.clear()
            self._support_bound_chain_uav_id.clear()
            self._support_bound_chain_truck_by_uav.clear()
            self._support_bound_chain_anchor_node_by_truck.clear()
            self._support_bound_chain_latest_start_by_truck.clear()
            self._tc_support_chain_class.clear()
            self._planner_eval_cache_step = -1
            self._truck_task_distance_cache.clear()
            self._truck_task_serviceable_cache.clear()
            self._truck_nearest_reachable_cache.clear()
            self._support_anchor_gain_cache.clear()
            self._truck_support_candidate_cache.clear()
            self._task_high_pressure_cache.clear()
            self._uav_anchor_task_goal.clear()
            self._uav_transfer_target_truck.clear()
            self._uav_transfer_target_task.clear()
            self.uav_transfer_hint_issue_count_total = 0
            self.uav_transfer_hint_keep_count_total = 0
            self.support_bound_dispatch_count_total = 0
            self.support_bound_recovery_redirect_count_total = 0
            self.support_no_gain_backoff_block_count_total = 0
            self.support_proxy_candidate_count_total = 0
            self.support_relay_reserved_count_total = 0
            self.truck_emergency_goal_assigned_count_total = 0
            self._support_relay_force_step.clear()
            self._support_no_gain_streak.clear()
            self._support_backoff_until_step.clear()
            self._task_last_goal_step.clear()
            self._task_goal_exposure_count.clear()
            self._truck_recent_normal_prev_goal.clear()
            self._truck_recent_normal_switch_step.clear()
            self._task_recent_prev_goal.clear()
            self._task_recent_switch_step.clear()
            self._uav_recent_truck_anchor_prev_goal.clear()
            self._uav_recent_truck_anchor_switch_step.clear()
            self._normal_task_unreachable_streak.clear()
            self.normal_unreachable_task_count_total = 0
        self._last_seen_step = step_now

    # --------------------------
    # Reused helper logic (from planner.py, neural-free)
    # --------------------------
    def _resolved_count(self, env) -> int:
        return int(
            sum(
                1
                for t in env.state.tasks.values()
                if t.status in (TaskStatus.DELIVERED, TaskStatus.FAILED)
            )
        )

    def _goal_invalid_reason_from_launch_reason(self, launch_reason: str) -> Tuple[str, bool]:
        rr = str(launch_reason).strip().lower()
        if rr in {"below_launch_min", "energy_infeasible", "horizon"}:
            return "uav_energy_infeasible", False
        if rr in {"insufficient_recovery_margin", "recovery_margin"}:
            return "uav_recovery_margin", True
        if rr in {"corridor", "corridor_blocked"}:
            return "uav_corridor", True
        if rr in {"comm_block", "comm_degraded"}:
            return "uav_comm_block", True
        if rr in {"not_loaded"}:
            return "uav_not_loaded", True
        if rr in {"not_docked"}:
            return "uav_not_docked", True
        if rr in {"cache_blocked"}:
            return "uav_soft_reject_cache", True
        if rr in {"rendezvous_launch_disabled", "no_truck_for_return", "no_recovery"}:
            return "uav_recovery_margin", True
        return "uav_energy_infeasible", False

    def _append_hard_event_offender(
        self,
        out: List[Dict[str, object]],
        reason: str,
        aid: Optional[str],
        task_id: Optional[str],
        step_now: int,
    ) -> None:
        out.append(
            {
                "reason": str(reason),
                "agent_id": "" if aid is None else str(aid),
                "task_id": "" if task_id is None else str(task_id),
                "step": int(step_now),
            }
        )

    def _truck_role_flags_for_dead_end(self, env, aid: str, task) -> Dict[str, bool]:
        aid_s = str(aid)
        is_emergency_goal = bool(task is not None and task.kind == TaskKind.EMERGENCY)
        has_bound_support_chain = bool(self._support_bound_chain_info_for_truck(env, aid_s) is not None)
        has_active_support_anchor = bool(str(self._support_anchor_task_id.get(aid_s, "")).strip())

        has_rendezvous_anchor = False
        has_recovery_anchor = False
        for uid, ust in env.state.agents.items():
            if ust.kind != AgentKind.UAV or bool(getattr(ust, "crashed", False)):
                continue
            if str(getattr(ust, "follow_target", "")) != aid_s:
                continue
            has_rendezvous_anchor = True
            if bool(self._uav_needs_recovery(env, str(uid))):
                has_recovery_anchor = True
                break

        return {
            "is_emergency_goal": bool(is_emergency_goal),
            "is_support_anchor": bool(has_bound_support_chain or has_active_support_anchor),
            "is_recovery_anchor": bool(has_recovery_anchor),
            "is_rendezvous_anchor": bool(has_rendezvous_anchor),
        }

    def _truck_dead_end_refresh(self, env) -> bool:
        """
        Localize truck dead-end handling:
        - detect persistent, truly non-progressable truck-goal pairs
        - prefer local path/goal correction
        - escalate to global hard refresh only when local correction fails and
          hrl_truck_dead_end_global_refresh_enabled is true.
        """
        self._last_truck_dead_end_record = None
        legal = env.legal_actions() if hasattr(env, "legal_actions") else {}
        step_now = int(getattr(env.state, "step_index", 0))

        # Keep per-truck persist state fresh.
        truck_ids = [
            str(aid)
            for aid, st in env.state.agents.items()
            if st.kind == AgentKind.TRUCK and (not bool(getattr(st, "crashed", False)))
        ]
        for tid in list(self._truck_dead_end_persist_by_truck.keys()):
            if tid not in truck_ids:
                self._truck_dead_end_persist_by_truck.pop(tid, None)

        persist_steps = int(max(getattr(env.cfg, "hrl_truck_dead_end_persist_steps", 3), 1))
        cooldown_steps = int(max(getattr(env.cfg, "hrl_truck_dead_end_cooldown_steps", 10), 0))
        local_first = bool(getattr(env.cfg, "hrl_truck_dead_end_local_first", True))
        allow_global = bool(getattr(env.cfg, "hrl_truck_dead_end_global_refresh_enabled", False))

        for aid in truck_ids:
            gid = self.state.goals.get(str(aid), None)
            if gid is None:
                self._truck_dead_end_persist_by_truck[str(aid)] = 0
                continue
            a = env.state.agents.get(str(aid), None)
            t = env.state.tasks.get(str(gid), None)
            if a is None or t is None or t.status != TaskStatus.PENDING:
                self._truck_dead_end_persist_by_truck[str(aid)] = 0
                continue
            # Skip when truck is already servicing.
            if bool(t.in_service_by == str(aid) and int(getattr(t, "service_remaining", 0)) > 0):
                self._truck_dead_end_persist_by_truck[str(aid)] = 0
                continue
            # Skip when already at service node.
            if a.node is not None and int(a.node) == int(t.demand_node):
                self._truck_dead_end_persist_by_truck[str(aid)] = 0
                continue

            nbs = [int(x) for x in legal.get(str(aid), {}).get("neighbors", [])]
            if a.node is None or not nbs:
                is_candidate = True
            else:
                cur_d = float("inf")
                if hasattr(env, "_decision_shortest_path_distance"):
                    cur_d = float(env._decision_shortest_path_distance(int(a.node), int(t.demand_node)))
                progressing = False
                has_unblocked = False
                for nb in nbs:
                    blocked = bool(hasattr(env, "_decision_is_blocked") and env._decision_is_blocked(int(a.node), int(nb)))
                    if blocked:
                        continue
                    has_unblocked = True
                    if hasattr(env, "_decision_shortest_path_distance"):
                        nb_d = float(env._decision_shortest_path_distance(int(nb), int(t.demand_node)))
                        if np.isfinite(nb_d) and (not np.isfinite(cur_d) or nb_d + 1e-6 < cur_d):
                            progressing = True
                            break
                # Candidate only when path is truly not progressing.
                is_candidate = bool((not has_unblocked) or (not progressing))

            if not is_candidate:
                self._truck_dead_end_persist_by_truck[str(aid)] = 0
                continue

            self.truck_dead_end_candidate_count_total = int(self.truck_dead_end_candidate_count_total) + 1
            streak = int(self._truck_dead_end_persist_by_truck.get(str(aid), 0) + 1)
            self._truck_dead_end_persist_by_truck[str(aid)] = streak
            if streak < persist_steps:
                self.truck_dead_end_blocked_by_persist_count_total = int(self.truck_dead_end_blocked_by_persist_count_total) + 1
                continue

            cool_until = int(self._truck_dead_end_cooldown_until_by_truck.get(str(aid), -1))
            if step_now <= cool_until:
                self.truck_dead_end_blocked_by_cooldown_count_total = int(self.truck_dead_end_blocked_by_cooldown_count_total) + 1
                continue

            role_flags = self._truck_role_flags_for_dead_end(env, str(aid), t)
            goal_unreachable = bool(not self._truck_task_reachable(env, str(aid), t))
            keep_hard = bool(
                role_flags.get("is_emergency_goal", False)
                or role_flags.get("is_support_anchor", False)
                or role_flags.get("is_recovery_anchor", False)
                or role_flags.get("is_rendezvous_anchor", False)
                or goal_unreachable
            )

            if keep_hard:
                if bool(role_flags.get("is_emergency_goal", False)):
                    self.truck_dead_end_emergency_kept_hard_count_total = int(
                        self.truck_dead_end_emergency_kept_hard_count_total
                    ) + 1
                if bool(role_flags.get("is_support_anchor", False)):
                    self.truck_dead_end_support_kept_hard_count_total = int(
                        self.truck_dead_end_support_kept_hard_count_total
                    ) + 1
                if bool(role_flags.get("is_recovery_anchor", False) or role_flags.get("is_rendezvous_anchor", False)):
                    self.truck_dead_end_recovery_kept_hard_count_total = int(
                        self.truck_dead_end_recovery_kept_hard_count_total
                    ) + 1
                self.truck_dead_end_global_refresh_count_total = int(self.truck_dead_end_global_refresh_count_total) + 1
                self._truck_dead_end_cooldown_until_by_truck[str(aid)] = int(step_now + cooldown_steps)
                self._last_truck_dead_end_record = {
                    "reason": "truck_dead_end",
                    "agent_id": str(aid),
                    "task_id": str(t.task_id),
                    "step": int(step_now),
                    "offenders": [
                        {
                            "reason": "truck_dead_end",
                            "agent_id": str(aid),
                            "task_id": str(t.task_id),
                            "step": int(step_now),
                            "current_goal_type": "normal" if t.kind == TaskKind.NORMAL else "emergency",
                            "proposed_goal_type": "unknown",
                            "task_status": str(getattr(t.status, "name", str(t.status))),
                            "battery": float("nan"),
                            "distance_to_goal": float("nan"),
                        }
                    ],
                }
                return True

            # Local-first handling (routine/NORMAL chain only).
            local_fixed = False
            if local_first:
                if not bool(getattr(env.cfg, "hrl_routine_localize_eta_exit_enabled", True)):
                    if bool(self._truck_task_reachable(env, str(aid), t)):
                        self.truck_dead_end_local_path_repair_count_total = int(self.truck_dead_end_local_path_repair_count_total) + 1
                        local_fixed = True
                    else:
                        if self._truck_dead_end_local_goal_reassign(env, str(aid), str(t.task_id)):
                            self.truck_dead_end_local_goal_reassign_count_total = int(self.truck_dead_end_local_goal_reassign_count_total) + 1
                            local_fixed = True
                    self._truck_dead_end_cooldown_until_by_truck[str(aid)] = int(step_now + cooldown_steps)
                    if local_fixed:
                        self.truck_dead_end_routine_localized_count_total = int(self.truck_dead_end_routine_localized_count_total) + 1
                        aid_s = str(aid)
                        self._debug_truck_dead_end_localize_by_truck[aid_s] = int(
                            self._debug_truck_dead_end_localize_by_truck.get(aid_s, 0) + 1
                        )
                        self.truck_dead_end_local_repair_no_goal_change_count_total = int(
                            self.truck_dead_end_local_repair_no_goal_change_count_total
                        ) + 1
                        self.truck_dead_end_noop_count_total = int(self.truck_dead_end_noop_count_total) + 1
                        return False
                # routine-localize ETA guard: keep current only when local ETA is not clearly worse than alternatives.
                eta_ratio = float(max(getattr(env.cfg, "hrl_routine_localize_max_eta_increase_ratio", 1.20), 1.0))
                eta_margin = float(max(getattr(env.cfg, "hrl_routine_localize_alt_eta_margin_steps", 6), 0.0))
                self.routine_localize_eta_check_count_total = int(self.routine_localize_eta_check_count_total) + 1
                cur_eta = float(self._switch_goal_eta(env, str(aid), str(t.task_id)))
                local_eta = float(cur_eta if bool(self._truck_task_reachable(env, str(aid), t)) else float("inf"))
                best_tid, best_eta, _best_sc = self._best_alternative_routine_for_truck(env, str(aid), str(t.task_id))
                keep_current_local = False
                if np.isfinite(local_eta) and np.isfinite(cur_eta) and local_eta <= float(cur_eta * eta_ratio):
                    keep_current_local = True
                if np.isfinite(local_eta) and np.isfinite(best_eta) and local_eta <= float(best_eta + eta_margin):
                    keep_current_local = True

                if keep_current_local:
                    self.routine_localize_keep_current_count_total = int(self.routine_localize_keep_current_count_total) + 1
                    if bool(self._truck_task_reachable(env, str(aid), t)):
                        # Keep same goal and let low-level route update resolve.
                        self.truck_dead_end_local_path_repair_count_total = int(self.truck_dead_end_local_path_repair_count_total) + 1
                        local_fixed = True
                    else:
                        if self._truck_dead_end_local_goal_reassign(env, str(aid), str(t.task_id)):
                            self.truck_dead_end_local_goal_reassign_count_total = int(self.truck_dead_end_local_goal_reassign_count_total) + 1
                            local_fixed = True
                else:
                    # ETA degrades too much; escape to better routine target when possible.
                    if best_tid is not None:
                        self.state.goals[str(aid)] = str(best_tid)
                        self.state.goal_assigned_step[str(aid)] = int(step_now)
                        self.routine_localize_escape_by_eta_worse_count_total = int(
                            self.routine_localize_escape_by_eta_worse_count_total
                        ) + 1
                        self._routine_localize_escape_pending.append({"step": int(step_now), "task_id": str(best_tid)})
                        self.truck_dead_end_local_goal_reassign_count_total = int(self.truck_dead_end_local_goal_reassign_count_total) + 1
                        local_fixed = True
                self._truck_dead_end_cooldown_until_by_truck[str(aid)] = int(step_now + cooldown_steps)
                if local_fixed:
                    self.truck_dead_end_routine_localized_count_total = int(self.truck_dead_end_routine_localized_count_total) + 1
                    aid_s = str(aid)
                    self._debug_truck_dead_end_localize_by_truck[aid_s] = int(
                        self._debug_truck_dead_end_localize_by_truck.get(aid_s, 0) + 1
                    )
                    self.truck_dead_end_local_repair_no_goal_change_count_total = int(
                        self.truck_dead_end_local_repair_no_goal_change_count_total
                    ) + 1
                    self.truck_dead_end_noop_count_total = int(self.truck_dead_end_noop_count_total) + 1
                    return False

            # Escalate only when configured.
            if not allow_global:
                self.truck_dead_end_noop_count_total = int(self.truck_dead_end_noop_count_total) + 1
                self._truck_dead_end_cooldown_until_by_truck[str(aid)] = int(step_now + cooldown_steps)
                return False

            self.truck_dead_end_global_refresh_count_total = int(self.truck_dead_end_global_refresh_count_total) + 1
            self._truck_dead_end_cooldown_until_by_truck[str(aid)] = int(step_now + cooldown_steps)
            self._last_truck_dead_end_record = {
                "reason": "truck_dead_end",
                "agent_id": str(aid),
                "task_id": str(t.task_id),
                "step": int(step_now),
                "offenders": [
                    {
                        "reason": "truck_dead_end",
                        "agent_id": str(aid),
                        "task_id": str(t.task_id),
                        "step": int(step_now),
                        "current_goal_type": "normal" if t.kind == TaskKind.NORMAL else "emergency",
                        "proposed_goal_type": "unknown",
                        "task_status": str(getattr(t.status, "name", str(t.status))),
                        "battery": float("nan"),
                        "distance_to_goal": float("nan"),
                    }
                ],
            }
            return True
        return False

    def _truck_dead_end_local_goal_reassign(self, env, aid: str, old_task_id: str) -> bool:
        """
        Local-only truck goal reassignment for dead-end correction.
        Prefer reachable NORMAL tasks; fallback to emergency relief when allowed.
        """
        best_tid: Optional[str] = None
        best_dist = float("inf")
        # Normal first.
        for t in env.state.tasks.values():
            if t.status != TaskStatus.PENDING or t.kind != TaskKind.NORMAL:
                continue
            tid = str(t.task_id)
            if tid == str(old_task_id):
                continue
            if not self._truck_task_valid(env, str(aid), tid):
                continue
            if not self._truck_task_reachable(env, str(aid), t):
                continue
            d = float(self._truck_task_distance(env, str(aid), t))
            if np.isfinite(d) and d + 1e-9 < best_dist:
                best_dist = float(d)
                best_tid = tid
        # Emergency relief fallback.
        if best_tid is None:
            for t in env.state.tasks.values():
                if t.status != TaskStatus.PENDING or t.kind != TaskKind.EMERGENCY:
                    continue
                tid = str(t.task_id)
                if tid == str(old_task_id):
                    continue
                if not self._truck_task_valid(env, str(aid), tid):
                    continue
                if not self._truck_task_reachable(env, str(aid), t):
                    continue
                if not bool(self._truck_emergency_relief_allowed(env, str(aid), t)):
                    continue
                d = float(self._truck_task_distance(env, str(aid), t))
                if np.isfinite(d) and d + 1e-9 < best_dist:
                    best_dist = float(d)
                    best_tid = tid
        if best_tid is None:
            return False
        self.state.goals[str(aid)] = str(best_tid)
        self.state.goal_assigned_step[str(aid)] = int(getattr(env.state, "step_index", 0))
        return True

    def _uav_emergency_refresh(self, env) -> bool:
        """
        Trigger replan if a UAV is still on a task-goal while battery is already
        in emergency zone, or the target becomes unsafe/infeasible.
        """
        self._last_uav_emergency_record = None
        thr = float(max(getattr(env.cfg, "uav_replan_emergency_battery_threshold", 0.30), 0.0))
        step_now = int(getattr(env.state, "step_index", 0))
        for aid, tid in self.state.goals.items():
            if tid is None:
                continue
            a = env.state.agents.get(str(aid), None)
            if a is None or a.kind != AgentKind.UAV or bool(a.crashed):
                continue
            goal_agent = env.state.agents.get(str(tid), None)
            if goal_agent is not None and goal_agent.kind == AgentKind.TRUCK:
                continue
            goal_task = env.state.tasks.get(str(tid), None)
            if goal_task is None or goal_task.status != TaskStatus.PENDING:
                continue
            if float(a.battery) < thr:
                self._last_uav_emergency_record = {
                    "reason": "uav_safety",
                    "agent_id": str(aid),
                    "task_id": str(goal_task.task_id),
                    "step": int(step_now),
                    "safety_type": "battery",
                    "offenders": [
                        {
                            "reason": "uav_safety",
                            "agent_id": str(aid),
                            "task_id": str(goal_task.task_id),
                            "step": int(step_now),
                            "current_goal_type": "normal" if goal_task.kind == TaskKind.NORMAL else "emergency",
                            "proposed_goal_type": "unknown",
                            "task_status": str(getattr(goal_task.status, "name", str(goal_task.status))),
                            "battery": float(getattr(a, "battery", float("nan"))),
                            "distance_to_goal": float("nan"),
                        }
                    ],
                }
                return True
            if not self._uav_task_feasible(env, str(aid), goal_task):
                self._last_uav_emergency_record = {
                    "reason": "uav_recovery",
                    "agent_id": str(aid),
                    "task_id": str(goal_task.task_id),
                    "step": int(step_now),
                    "safety_type": "feasibility",
                    "offenders": [
                        {
                            "reason": "uav_recovery",
                            "agent_id": str(aid),
                            "task_id": str(goal_task.task_id),
                            "step": int(step_now),
                            "current_goal_type": "normal" if goal_task.kind == TaskKind.NORMAL else "emergency",
                            "proposed_goal_type": "unknown",
                            "task_status": str(getattr(goal_task.status, "name", str(goal_task.status))),
                            "battery": float(getattr(a, "battery", float("nan"))),
                            "distance_to_goal": float("nan"),
                        }
                    ],
                }
                return True
        return False

    def _uav_idle_refresh(self, env) -> bool:
        """
        Trigger replan when a UAV has no current goal while feasible emergency
        tasks still exist. Prevents long idle windows under event-budget throttling.
        """
        for aid, ag in env.state.agents.items():
            if ag.kind != AgentKind.UAV or bool(getattr(ag, "crashed", False)):
                continue
            if self.state.goals.get(str(aid), None) is not None:
                continue
            if self._uav_stage_blocks_task_goal(env, str(aid)):
                continue
            for task in env.state.tasks.values():
                if task.kind != TaskKind.EMERGENCY or task.status != TaskStatus.PENDING:
                    continue
                if self._uav_task_feasible(env, str(aid), task):
                    return True
        return False

    def _goal_invalidated_refresh(self, env) -> bool:
        """
        High-value refresh: current assigned goal becomes invalid/infeasible.
        """
        self._last_goal_invalid_record = None
        step_now = int(getattr(env.state, "step_index", 0))
        for aid, goal_id in self.state.goals.items():
            if goal_id is None:
                continue
            st = env.state.agents.get(str(aid), None)
            if st is None or bool(getattr(st, "crashed", False)):
                continue
            task = env.state.tasks.get(str(goal_id), None)
            if task is not None:
                if task.status != TaskStatus.PENDING:
                    rr = "task_failed" if task.status == TaskStatus.FAILED else "task_completed"
                    self._last_goal_invalid_record = {
                        "reason": str(rr),
                        "agent_id": str(aid),
                        "task_id": str(task.task_id),
                        "step": int(step_now),
                        "suspect_soft_as_hard": False,
                        "offenders": [
                            {
                                "reason": "goal_invalid",
                                "agent_id": str(aid),
                                "task_id": str(task.task_id),
                                "step": int(step_now),
                                "current_goal_type": "normal" if task.kind == TaskKind.NORMAL else "emergency",
                                "proposed_goal_type": "unknown",
                                "task_status": str(getattr(task.status, "name", str(task.status))),
                                "battery": float(getattr(st, "battery", float("nan"))),
                                "distance_to_goal": float("nan"),
                            }
                        ],
                    }
                    return True
                if st.kind == AgentKind.TRUCK:
                    if (not self._truck_task_valid(env, str(aid), str(task.task_id))) or (not self._truck_task_reachable(env, str(aid), task)):
                        self._last_goal_invalid_record = {
                            "reason": "truck_unreachable",
                            "agent_id": str(aid),
                            "task_id": str(task.task_id),
                            "step": int(step_now),
                            "suspect_soft_as_hard": False,
                            "offenders": [
                                {
                                    "reason": "goal_invalid",
                                    "agent_id": str(aid),
                                    "task_id": str(task.task_id),
                                    "step": int(step_now),
                                    "current_goal_type": "normal" if task.kind == TaskKind.NORMAL else "emergency",
                                    "proposed_goal_type": "unknown",
                                    "task_status": str(getattr(task.status, "name", str(task.status))),
                                    "battery": float(getattr(st, "battery", float("nan"))),
                                    "distance_to_goal": float("nan"),
                                }
                            ],
                        }
                        return True
                elif st.kind == AgentKind.UAV:
                    if not self._uav_task_feasible(env, str(aid), task):
                        launch_reason = ""
                        if hasattr(env, "_uav_launch_gate_check"):
                            prev_eff = env._effective_goals.get(str(aid), None) if hasattr(env, "_effective_goals") else None
                            try:
                                if hasattr(env, "_effective_goals"):
                                    env._effective_goals[str(aid)] = str(task.task_id)
                                _ok, launch_reason, _ = env._uav_launch_gate_check(str(aid), task=task, count_reject=False)
                            except Exception:
                                launch_reason = ""
                            finally:
                                if hasattr(env, "_effective_goals"):
                                    env._effective_goals[str(aid)] = prev_eff
                        if self._uav_reject_cache_blocked(env, str(aid), str(task.task_id)):
                            reason_key = "uav_soft_reject_cache"
                            suspect_soft = True
                        else:
                            reason_key, suspect_soft = self._goal_invalid_reason_from_launch_reason(str(launch_reason))
                        self._last_goal_invalid_record = {
                            "reason": str(reason_key),
                            "agent_id": str(aid),
                            "task_id": str(task.task_id),
                            "step": int(step_now),
                            "suspect_soft_as_hard": bool(suspect_soft),
                            "offenders": [
                                {
                                    "reason": "goal_invalid",
                                    "agent_id": str(aid),
                                    "task_id": str(task.task_id),
                                    "step": int(step_now),
                                    "current_goal_type": "normal" if task.kind == TaskKind.NORMAL else "emergency",
                                    "proposed_goal_type": "unknown",
                                    "task_status": str(getattr(task.status, "name", str(task.status))),
                                    "battery": float(getattr(st, "battery", float("nan"))),
                                    "distance_to_goal": float("nan"),
                                }
                            ],
                        }
                        return True
                continue

            ag = env.state.agents.get(str(goal_id), None)
            # ``DEPOT_DOCK_ID`` is a valid UAV recovery/reload sentinel, not a
            # task or an agent entry.  It must not be treated as a missing goal
            # (which would trigger an unnecessary hard refresh).  Keep this
            # exception scoped to UAVs: a truck targeting the sentinel remains
            # invalid because depot recovery is UAV-specific.
            if st.kind == AgentKind.UAV and str(goal_id) == DEPOT_DOCK_ID:
                continue
            if st.kind == AgentKind.UAV and ag is not None and ag.kind == AgentKind.TRUCK:
                if bool(getattr(ag, "crashed", False)):
                    self._last_goal_invalid_record = {
                        "reason": "task_missing",
                        "agent_id": str(aid),
                        "task_id": str(goal_id),
                        "step": int(step_now),
                        "suspect_soft_as_hard": False,
                        "offenders": [
                            {
                                "reason": "goal_invalid",
                                "agent_id": str(aid),
                                "task_id": str(goal_id),
                                "step": int(step_now),
                                "current_goal_type": "unknown",
                                "proposed_goal_type": "unknown",
                                "task_status": "missing",
                                "battery": float(getattr(st, "battery", float("nan"))),
                                "distance_to_goal": float("nan"),
                            }
                        ],
                    }
                    return True
                continue
            self._last_goal_invalid_record = {
                "reason": "task_missing",
                "agent_id": str(aid),
                "task_id": str(goal_id),
                "step": int(step_now),
                "suspect_soft_as_hard": False,
                "offenders": [
                    {
                        "reason": "goal_invalid",
                        "agent_id": str(aid),
                        "task_id": str(goal_id),
                        "step": int(step_now),
                        "current_goal_type": "unknown",
                        "proposed_goal_type": "unknown",
                        "task_status": "missing",
                        "battery": float(getattr(st, "battery", float("nan"))),
                        "distance_to_goal": float("nan"),
                    }
                ],
            }
            return True
        return False

    def _high_priority_emergency_uncovered_refresh(self, env) -> bool:
        """
        High-value refresh: urgent emergency task is uncovered by current goals.
        """
        self._last_high_priority_uncovered_record = None
        step_now = int(env.state.step_index)
        slack_thr = int(max(getattr(env.cfg, "hrl_high_priority_emergency_slack_steps", 12), 1))

        goal_covered = set()
        for _, gid in self.state.goals.items():
            if gid is None:
                continue
            t = env.state.tasks.get(str(gid), None)
            if t is not None and t.kind == TaskKind.EMERGENCY and t.status == TaskStatus.PENDING:
                goal_covered.add(str(t.task_id))

        for task in env.state.tasks.values():
            if task.kind != TaskKind.EMERGENCY or task.status != TaskStatus.PENDING:
                continue
            slack = int(max(int(task.deadline_step) - step_now, 0))
            if slack > slack_thr:
                continue
            if str(task.task_id) in goal_covered:
                continue

            uav_feasible = any(
                self._uav_task_feasible(env, str(aid), task)
                for aid, ag in env.state.agents.items()
                if ag.kind == AgentKind.UAV and (not bool(getattr(ag, "crashed", False)))
            )
            truck_feasible = any(
                self._truck_task_valid(env, str(aid), str(task.task_id))
                and self._truck_task_reachable(env, str(aid), task)
                and bool(self._truck_emergency_relief_allowed(env, str(aid), task))
                for aid, ag in env.state.agents.items()
                if ag.kind == AgentKind.TRUCK and (not bool(getattr(ag, "crashed", False)))
            )
            if bool(uav_feasible or truck_feasible):
                self._last_high_priority_uncovered_record = {
                    "reason": "high_priority_uncovered",
                    "agent_id": "",
                    "task_id": str(task.task_id),
                    "step": int(step_now),
                    "offenders": [
                        {
                            "reason": "high_priority_uncovered",
                            "agent_id": "",
                            "task_id": str(task.task_id),
                            "step": int(step_now),
                            "current_goal_type": "emergency",
                            "proposed_goal_type": "emergency",
                            "task_status": str(getattr(task.status, "name", str(task.status))),
                            "battery": float("nan"),
                            "distance_to_goal": float("nan"),
                        }
                    ],
                }
                return True
        return False

    def _normal_stall_refresh(self, env) -> bool:
        """
        High-value refresh: normal backlog unchanged for a while and trucks are
        not effectively pursuing reachable normal tasks.
        """
        self._last_normal_stall_record = None
        step_now = int(getattr(env.state, "step_index", 0))
        pending_norm = int(self._pending_normal_task_count(env))
        if pending_norm <= 0:
            return False

        stall_steps = int(max(getattr(env.cfg, "hrl_normal_stall_min_persist_steps", getattr(env.cfg, "hrl_normal_stall_confirm_steps", 8)), 1))
        candidate: Optional[Tuple[str, str]] = None

        for aid, ag in env.state.agents.items():
            if ag.kind != AgentKind.TRUCK or bool(getattr(ag, "crashed", False)):
                continue
            gid = self.state.goals.get(str(aid), None)
            cur_task = env.state.tasks.get(str(gid), None) if gid is not None else None
            pursuing_normal = bool(
                cur_task is not None
                and cur_task.kind == TaskKind.NORMAL
                and cur_task.status == TaskStatus.PENDING
                and self._truck_task_reachable(env, str(aid), cur_task)
            )
            if pursuing_normal:
                continue
            for t in env.state.tasks.values():
                if t.kind != TaskKind.NORMAL or t.status != TaskStatus.PENDING:
                    continue
                if self._truck_task_valid(env, str(aid), str(t.task_id)) and self._truck_task_reachable(env, str(aid), t):
                    candidate = (str(aid), str(t.task_id))
                    break
            if candidate is not None:
                break
        if candidate is None:
            return False

        self.normal_stall_candidate_count_total = int(self.normal_stall_candidate_count_total) + 1
        if int(self._normal_pending_unchanged_steps) < stall_steps:
            self.normal_stall_blocked_by_persist_count_total = int(self.normal_stall_blocked_by_persist_count_total) + 1
            return False

        aid, tid = candidate
        cooldown_steps = int(max(getattr(env.cfg, "hrl_normal_stall_cooldown_steps", 12), 0))
        cool_until = int(self._normal_stall_cooldown_until_by_truck.get(str(aid), -1))
        if step_now <= cool_until:
            self.normal_stall_blocked_by_cooldown_count_total = int(self.normal_stall_blocked_by_cooldown_count_total) + 1
            return False

        self._last_normal_stall_record = {
            "reason": "normal_stall",
            "agent_id": str(aid),
            "task_id": str(tid),
            "step": int(step_now),
            "offenders": [
                {
                    "reason": "normal_stall",
                    "agent_id": str(aid),
                    "task_id": str(tid),
                    "step": int(step_now),
                    "current_goal_type": "normal",
                    "proposed_goal_type": "normal",
                    "task_status": "pending",
                    "battery": float("nan"),
                    "distance_to_goal": float("nan"),
                }
            ],
        }
        hard_enabled = bool(getattr(env.cfg, "hrl_normal_stall_hard_refresh_enabled", False))
        local_only = bool(getattr(env.cfg, "hrl_normal_stall_local_only", True))
        if (not hard_enabled) or local_only:
            self.normal_stall_local_correction_count_total = int(self.normal_stall_local_correction_count_total) + 1
            self._normal_stall_cooldown_until_by_truck[str(aid)] = int(step_now + cooldown_steps)
            return False

        self.normal_stall_global_refresh_count_total = int(self.normal_stall_global_refresh_count_total) + 1
        self._normal_stall_cooldown_until_by_truck[str(aid)] = int(step_now + cooldown_steps)
        return True

    def _should_refresh(self, env) -> bool:
        # Ablation: disable event-triggering -> fixed-interval replanning only.
        if not self.use_event_trigger:
            since_last = int(env.state.step_index - self.state.step_last_refresh)
            decision = build_no_event_refresh_decision(
                since_last=since_last,
                decision_interval=int(self.decision_interval),
                has_goals=bool(self.state.goals),
                hard_reason_totals=dict(self._map_update_hard_actionable_reasons_total),
                hard_seen_count_total=int(self.map_update_hard_seen_count_total),
                hard_actionable_count_total=int(self.map_update_hard_actionable_count_total),
                hard_deferred_count_total=int(self.map_update_hard_deferred_count_total),
                hard_immediate_refresh_count_total=int(self.map_update_hard_immediate_refresh_count_total),
            )
            self._last_refresh_flags = dict(decision.flags)
            return bool(decision.refresh)

        since_last = int(env.state.step_index - self.state.step_last_refresh)
        self._last_goal_invalid_record = None
        self._last_uav_emergency_record = None
        self._last_truck_dead_end_record = None
        self._last_high_priority_uncovered_record = None
        self._last_normal_stall_record = None
        self._last_goal_terminal_status = ""
        self._last_hard_event_offenders = []
        event_first_enabled = bool(getattr(env.cfg, "hrl_event_first_refresh_enabled", False))
        max_no_refresh_steps = int(max(getattr(env.cfg, "hrl_max_no_refresh_steps", 5), 1))
        by_interval = bool((not event_first_enabled) and (since_last >= self.decision_interval))
        risk_spike = bool(env.state.hazard.risk_spike)
        edge_only = bool(getattr(env.cfg, "hrl_risk_spike_edge_trigger_only", True))
        if edge_only:
            by_risk = bool(risk_spike and len(getattr(env.topology, "blocked_edges", set())) > 0)
        else:
            comm_any = bool(any(bool(v) for v in getattr(env, "comm_blocked", {}).values()))
            by_risk = bool(risk_spike or comm_any)

        resolved_now = self._resolved_count(env)
        by_resolution = resolved_now != self.state.resolved_tasks_last
        by_arrival = False
        by_goal_terminal = False
        goal_terminal_status = ""

        for aid, tid in self.state.goals.items():
            if tid is None:
                continue
            t = env.state.tasks.get(str(tid))
            a = env.state.agents[aid]
            # Virtual truck goal (for UAV recovery): arrival means already attached
            # or inside bind radius of that truck.
            if t is None:
                goal_agent = env.state.agents.get(str(tid), None)
                if (
                    a.kind == AgentKind.UAV
                    and goal_agent is not None
                    and goal_agent.kind == AgentKind.TRUCK
                ):
                    if a.follow_target is not None and str(a.follow_target) == str(tid):
                        by_arrival = True
                        break
                    ax, ay = self._agent_xy(env, str(aid))
                    tx, ty = self._agent_xy(env, str(tid))
                    d = float(((ax - tx) ** 2 + (ay - ty) ** 2) ** 0.5)
                    if d <= float(getattr(env.cfg, "uav_bind_radius_m", 50.0)):
                        by_arrival = True
                        break
                    continue
                by_arrival = True
                break

            if t.status in (TaskStatus.DELIVERED, TaskStatus.FAILED):
                by_arrival = True
                by_goal_terminal = True
                goal_terminal_status = "failed" if t.status == TaskStatus.FAILED else "completed"
                break
            if a.kind == AgentKind.TRUCK and a.node is not None:
                if int(a.node) == int(t.demand_node):
                    by_arrival = True
                    break
            if a.kind == AgentKind.UAV and a.pos_xy is not None:
                nx = env.topology.nodes[int(t.demand_node)]
                d = float(((a.pos_xy[0] - nx.x) ** 2 + (a.pos_xy[1] - nx.y) ** 2)
                          ** 0.5)
                if d <= float(env.cfg.uav_delivery_radius_m):
                    by_arrival = True
                    break

        pending_norm_now = int(self._pending_normal_task_count(env))
        if pending_norm_now > 0 and int(self._last_pending_normal_count) == pending_norm_now:
            self._normal_pending_unchanged_steps = int(self._normal_pending_unchanged_steps) + 1
        elif pending_norm_now > 0:
            self._normal_pending_unchanged_steps = 0
        else:
            self._normal_pending_unchanged_steps = 0
        self._last_pending_normal_count = int(pending_norm_now)

        delivered_now = int(self._delivered_count(env))
        if delivered_now == int(self._last_delivered_count):
            self._delivery_stall_steps = int(self._delivery_stall_steps) + 1
        else:
            self._delivery_stall_steps = 0
        self._last_delivered_count = int(delivered_now)

        island_serviceability_now = float(self._island_serviceability_ratio(env))
        island_serviceability_drop = bool(island_serviceability_now + 1e-9 < float(self._last_island_serviceability) - 0.05)
        self._last_island_serviceability = float(island_serviceability_now)

        by_truck_dead_end = self._truck_dead_end_refresh(env)
        by_uav_emergency = self._uav_emergency_refresh(env)
        by_uav_idle = self._uav_idle_refresh(env)
        by_goal_invalid = self._goal_invalidated_refresh(env)
        by_normal_stall = self._normal_stall_refresh(env)
        by_high_priority_uncovered = self._high_priority_emergency_uncovered_refresh(env)
        self._last_goal_terminal_status = str(goal_terminal_status)
        step_now_local = int(getattr(env.state, "step_index", 0))

        soft_invalid_reasons = {
            "uav_recovery_margin",
            "uav_corridor",
            "uav_comm_block",
            "uav_not_loaded",
            "uav_not_docked",
            "uav_soft_reject_cache",
            "uav_energy_infeasible",
        }
        if by_goal_invalid:
            rec = self._last_goal_invalid_record if isinstance(self._last_goal_invalid_record, dict) else {}
            rr = str(rec.get("reason", "")).strip().lower()
            aid_rr = str(rec.get("agent_id", ""))
            tid_rr = str(rec.get("task_id", ""))
            if rr in soft_invalid_reasons:
                self.goal_invalid_soft_count_total = int(self.goal_invalid_soft_count_total) + 1
                key = (aid_rr, tid_rr, rr)
                rep = int(self._soft_invalid_repeat.get(key, 0) + 1)
                self._soft_invalid_repeat[key] = rep
                retry_steps = int(max(getattr(env.cfg, "hrl_soft_invalid_retry_cooldown_steps", 10), 0))
                esc_after = int(max(getattr(env.cfg, "hrl_soft_invalid_escalate_after_count", 3), 1))
                cool_until = int(self._soft_invalid_cooldown_until.get(key, -1))
                in_cooldown = bool(step_now_local <= cool_until)
                suppress_soft = bool(not getattr(env.cfg, "hrl_soft_invalid_hard_refresh_enabled", False))
                if suppress_soft:
                    by_goal_invalid = False
                    self.goal_invalid_soft_suppressed_count_total = int(self.goal_invalid_soft_suppressed_count_total) + 1
                    if (not in_cooldown) and rep >= esc_after:
                        self.goal_invalid_soft_escalated_count_total = int(self.goal_invalid_soft_escalated_count_total) + 1
                    if retry_steps > 0:
                        self._soft_invalid_cooldown_until[key] = int(max(cool_until, step_now_local + retry_steps))
            else:
                self.goal_invalid_hard_count_total = int(self.goal_invalid_hard_count_total) + 1

        if by_uav_emergency:
            rec_u = self._last_uav_emergency_record if isinstance(self._last_uav_emergency_record, dict) else {}
            reason_u = str(rec_u.get("reason", "")).strip().lower()
            aid_u = str(rec_u.get("agent_id", ""))
            st_u = env.state.agents.get(str(aid_u), None)
            airborne_u = bool(st_u is not None and st_u.kind == AgentKind.UAV and st_u.follow_target is None)
            batt_u = float(getattr(st_u, "battery", 1.0)) if st_u is not None else 1.0
            thr_u = float(max(getattr(env.cfg, "uav_replan_emergency_battery_threshold", 0.30), 0.0))
            if reason_u == "uav_recovery":
                is_hard = bool(airborne_u and batt_u < thr_u)
                if is_hard:
                    self.uav_recovery_hard_count_total = int(self.uav_recovery_hard_count_total) + 1
                    self.uav_recovery_global_refresh_count_total = int(self.uav_recovery_global_refresh_count_total) + 1
                else:
                    self.uav_recovery_soft_count_total = int(self.uav_recovery_soft_count_total) + 1
                    if not bool(getattr(env.cfg, "hrl_soft_invalid_hard_refresh_enabled", False)):
                        by_uav_emergency = False
                        self.uav_recovery_soft_suppressed_count_total = int(self.uav_recovery_soft_suppressed_count_total) + 1
                        self.uav_recovery_local_action_count_total = int(self.uav_recovery_local_action_count_total) + 1
            else:
                self.uav_recovery_hard_count_total = int(self.uav_recovery_hard_count_total) + 1
                self.uav_recovery_global_refresh_count_total = int(self.uav_recovery_global_refresh_count_total) + 1

        # Per-truck idle refresh: if a truck has no current goal while reachable
        # pending tasks still exist, force replan to avoid drifting under low-level fallback.
        by_truck_idle = False
        for aid, st in env.state.agents.items():
            if st.kind != AgentKind.TRUCK or bool(getattr(st, "crashed", False)):
                continue
            if self.state.goals.get(str(aid), None) is not None:
                continue
            if self._truck_has_assignable_task(env, str(aid)):
                by_truck_idle = True
                break

        map_new_blocked = int(max(getattr(env, "_shared_map_new_blocked_step", 0), 0))
        map_unknown_hit = int(max(getattr(env, "_unknown_blocked_edge_hit_step", 0), 0))
        map_cleared = int(max(getattr(env, "_shared_map_cleared_step", 0), 0))
        map_event = bool(getattr(env, "_shared_map_update_event_step", False))
        any_goal_assigned = bool(any(g is not None for g in self.state.goals.values())) if bool(self.state.goals) else False

        map_signal = bool(
            getattr(env.cfg, "road_shared_replan_on_update", True)
            and (
                map_new_blocked > 0
                or map_unknown_hit > 0
                or (map_cleared > 0 and by_truck_dead_end)
                or (map_event and (not any_goal_assigned))
            )
        )
        map_update_hard_seen = bool(map_signal and (map_new_blocked > 0 or map_unknown_hit > 0))
        (
            map_update_hard_actionable,
            map_critical,
            map_impacted_count,
            map_critical_count,
            map_actionable_reasons,
        ) = self._map_update_replan_gate(
            env,
            bool(map_update_hard_seen),
            any_goal_assigned=bool(any_goal_assigned),
            map_event=bool(map_event),
            by_truck_dead_end=bool(by_truck_dead_end),
        )

        # Path-blocked impact gate (local-first).
        path_blocked_step = int(max(int(map_actionable_reasons.get("path_blocked", 0)), 0))
        goal_unreachable_step = int(max(int(map_actionable_reasons.get("goal_unreachable", 0)), 0))
        recovery_fractured_step = int(max(int(map_actionable_reasons.get("recovery_path_fractured", 0)), 0))
        if bool(map_update_hard_seen):
            self.path_blocked_candidate_count_total = int(self.path_blocked_candidate_count_total) + 1
        if path_blocked_step > 0:
            self.path_blocked_impacted_current_path_count_total = int(self.path_blocked_impacted_current_path_count_total) + int(path_blocked_step)
        if goal_unreachable_step > 0:
            self.path_blocked_impacted_goal_reachability_count_total = int(
                self.path_blocked_impacted_goal_reachability_count_total
            ) + int(goal_unreachable_step)
        if recovery_fractured_step > 0:
            self.path_blocked_impacted_recovery_count_total = int(self.path_blocked_impacted_recovery_count_total) + int(recovery_fractured_step)

        if bool(getattr(env.cfg, "hrl_path_blocked_impact_gate_enabled", True)) and bool(map_update_hard_seen):
            no_exec_impact = bool(path_blocked_step <= 0 and goal_unreachable_step <= 0 and recovery_fractured_step <= 0)
            if no_exec_impact:
                self.path_blocked_nonimpact_suppressed_count_total = int(self.path_blocked_nonimpact_suppressed_count_total) + 1
                self.path_blocked_noop_count_total = int(self.path_blocked_noop_count_total) + 1
                map_update_hard_actionable = False
                map_critical = False
            elif path_blocked_step > 0 and bool(getattr(env.cfg, "hrl_path_blocked_local_repair_first", True)):
                emergency_or_timecritical_active = False
                support_chain_active = False
                recovery_or_rendezvous_active = False
                for _aid, _gid in self.state.goals.items():
                    if _gid is None:
                        continue
                    _st = env.state.agents.get(str(_aid), None)
                    _task = env.state.tasks.get(str(_gid), None)
                    if _st is None:
                        continue
                    if _st.kind == AgentKind.TRUCK and _task is not None and _task.status == TaskStatus.PENDING:
                        if _task.kind == TaskKind.EMERGENCY:
                            emergency_or_timecritical_active = True
                        if self._support_bound_chain_info_for_truck(env, str(_aid)) is not None:
                            support_chain_active = True
                    if _st.kind == AgentKind.UAV:
                        if _task is None and env.state.agents.get(str(_gid), None) is not None:
                            _anchor = env.state.agents.get(str(_gid))
                            if _anchor is not None and _anchor.kind == AgentKind.TRUCK:
                                recovery_or_rendezvous_active = True
                        if bool(getattr(_st, "follow_target", None)):
                            recovery_or_rendezvous_active = True
                    if emergency_or_timecritical_active and support_chain_active and recovery_or_rendezvous_active:
                        break

                scope_routine_only = bool(
                    (goal_unreachable_step <= 0)
                    and (recovery_fractured_step <= 0)
                    and (not bool(by_high_priority_uncovered))
                    and (not emergency_or_timecritical_active)
                    and (not support_chain_active)
                    and (not recovery_or_rendezvous_active)
                )
                local_suppressed = False
                if not scope_routine_only:
                    if goal_unreachable_step > 0:
                        self.path_blocked_goal_unreachable_kept_hard_count_total = int(
                            self.path_blocked_goal_unreachable_kept_hard_count_total
                        ) + 1
                    if recovery_fractured_step > 0 or recovery_or_rendezvous_active:
                        self.path_blocked_recovery_kept_hard_count_total = int(
                            self.path_blocked_recovery_kept_hard_count_total
                        ) + 1
                    if emergency_or_timecritical_active or bool(by_high_priority_uncovered):
                        self.path_blocked_emergency_kept_hard_count_total = int(
                            self.path_blocked_emergency_kept_hard_count_total
                        ) + 1
                    if support_chain_active:
                        self.path_blocked_support_kept_hard_count_total = int(
                            self.path_blocked_support_kept_hard_count_total
                        ) + 1
                else:
                    if goal_unreachable_step <= 0:
                        self.path_blocked_local_path_repair_count_total = int(self.path_blocked_local_path_repair_count_total) + 1
                        eta_ratio = float(max(getattr(env.cfg, "hrl_routine_localize_max_eta_increase_ratio", 1.20), 1.0))
                        eta_margin = float(max(getattr(env.cfg, "hrl_routine_localize_alt_eta_margin_steps", 6), 0.0))
                        eta_exit_enabled = bool(getattr(env.cfg, "hrl_routine_localize_eta_exit_enabled", True))
                        escape_count = 0
                        for _aid, _gid in list(self.state.goals.items()):
                            _st = env.state.agents.get(str(_aid), None)
                            _t = env.state.tasks.get(str(_gid), None) if _gid is not None else None
                            if _st is None or _st.kind != AgentKind.TRUCK or _t is None:
                                continue
                            if _t.status != TaskStatus.PENDING or _t.kind != TaskKind.NORMAL:
                                continue
                            self.routine_localize_eta_check_count_total = int(self.routine_localize_eta_check_count_total) + 1
                            cur_eta = float(self._switch_goal_eta(env, str(_aid), str(_t.task_id)))
                            local_eta = float(cur_eta)
                            best_tid, best_eta, _best_sc = self._best_alternative_routine_for_truck(env, str(_aid), str(_t.task_id))
                            keep_current_local = False
                            if not eta_exit_enabled:
                                keep_current_local = True
                            if np.isfinite(local_eta) and np.isfinite(cur_eta) and local_eta <= float(cur_eta * eta_ratio):
                                keep_current_local = True
                            if np.isfinite(local_eta) and np.isfinite(best_eta) and local_eta <= float(best_eta + eta_margin):
                                keep_current_local = True
                            if keep_current_local:
                                self.routine_localize_keep_current_count_total = int(self.routine_localize_keep_current_count_total) + 1
                            else:
                                if best_tid is not None:
                                    self.state.goals[str(_aid)] = str(best_tid)
                                    self.state.goal_assigned_step[str(_aid)] = int(step_now_local)
                                    self.routine_localize_escape_by_eta_worse_count_total = int(
                                        self.routine_localize_escape_by_eta_worse_count_total
                                    ) + 1
                                    self._routine_localize_escape_pending.append({"step": int(step_now_local), "task_id": str(best_tid)})
                                    escape_count += 1
                            aid_s = str(_aid)
                            self._debug_truck_path_blocked_localize_by_truck[aid_s] = int(
                                self._debug_truck_path_blocked_localize_by_truck.get(aid_s, 0) + 1
                            )
                        if escape_count > 0:
                            self.path_blocked_local_goal_reassign_count_total = int(
                                self.path_blocked_local_goal_reassign_count_total
                            ) + int(escape_count)
                        local_suppressed = True
                    else:
                        reassign_count = 0
                        for aid, gid in list(self.state.goals.items()):
                            if gid is None:
                                continue
                            st = env.state.agents.get(str(aid), None)
                            t = env.state.tasks.get(str(gid), None)
                            if st is None or st.kind != AgentKind.TRUCK or t is None:
                                continue
                            if t.status != TaskStatus.PENDING:
                                continue
                            if bool(self._truck_task_reachable(env, str(aid), t)):
                                continue
                            if self._truck_dead_end_local_goal_reassign(env, str(aid), str(t.task_id)):
                                reassign_count += 1
                                aid_s = str(aid)
                                self._debug_truck_path_blocked_localize_by_truck[aid_s] = int(
                                    self._debug_truck_path_blocked_localize_by_truck.get(aid_s, 0) + 1
                                )
                        if reassign_count > 0:
                            self.path_blocked_local_goal_reassign_count_total = int(self.path_blocked_local_goal_reassign_count_total) + int(reassign_count)
                            local_suppressed = True
                    if local_suppressed:
                        self.path_blocked_routine_localized_count_total = int(self.path_blocked_routine_localized_count_total) + 1
                        self.path_blocked_local_repair_no_goal_change_count_total = int(
                            self.path_blocked_local_repair_no_goal_change_count_total
                        ) + 1
                        if not bool(getattr(env.cfg, "hrl_path_blocked_global_refresh_enabled", False)):
                            map_update_hard_actionable = False
                            map_critical = False
                            self.path_blocked_noop_count_total = int(self.path_blocked_noop_count_total) + 1
                # If local repair failed and global refresh disabled, suppress anyway.
                if scope_routine_only and (not local_suppressed) and (not bool(getattr(env.cfg, "hrl_path_blocked_global_refresh_enabled", False))):
                    map_update_hard_actionable = False
                    map_critical = False
                    self.path_blocked_noop_count_total = int(self.path_blocked_noop_count_total) + 1

        by_map_update_hard = bool(map_update_hard_seen and map_update_hard_actionable)
        by_map_update_hard_deferred = bool(map_update_hard_seen and (not map_update_hard_actionable))
        by_map_update_light = bool(map_signal and (not map_update_hard_seen))

        # Conditional H2 refresh: turn aggressive map-refresh into symptom-triggered refresh.
        cond_h2 = bool(getattr(env.cfg, "hrl_conditional_h2_refresh_enabled", False))
        new_info_thr = int(max(getattr(env.cfg, "hrl_conditional_h2_new_info_threshold", 2), 0))
        info_burst = bool((map_new_blocked + map_unknown_hit) >= new_info_thr)
        stall_thr = int(max(getattr(env.cfg, "hrl_conditional_h2_delivery_stall_steps", 10), 0))
        support_dist_thr = float(max(getattr(env.cfg, "hrl_conditional_h2_support_distance_threshold_m", 85000.0), 0.0))
        support_quality_max = float(np.clip(getattr(env.cfg, "hrl_conditional_h2_support_quality_max", 0.55), 0.0, 1.0))
        support_quality = float(self._support_conversion_quality(env))
        support_dist_total = float(max(getattr(env, "truck_forward_support_distance_total", 0.0), 0.0))
        conversion_stall = bool(
            int(self._delivery_stall_steps) >= stall_thr
            and support_quality <= support_quality_max
            and support_dist_total >= support_dist_thr
        )
        island_low_thr = float(np.clip(getattr(env.cfg, "hrl_conditional_h2_island_serviceability_low", 0.45), 0.0, 1.0))
        island_serviceability_low = bool(island_serviceability_now <= island_low_thr)
        route_blocked = bool(
            int(map_actionable_reasons.get("path_blocked", 0)) > 0
            or int(map_actionable_reasons.get("goal_unreachable", 0)) > 0
        )
        route_blocked_critical = bool(route_blocked and bool(map_critical))
        symptom_refresh = bool(
            info_burst
            or conversion_stall
            or island_serviceability_drop
            or island_serviceability_low
            or route_blocked_critical
        )
        noncritical_votes = int(info_burst) + int(conversion_stall) + int(island_serviceability_drop) + int(island_serviceability_low)
        noncritical_min_votes = int(
            max(
                getattr(
                    env.cfg,
                    "hrl_noncritical_map_update_min_votes",
                    2 if bool(self.use_event_trigger) else 1,
                ),
                1,
            )
        )
        noncritical_symptom_refresh = bool(noncritical_votes >= noncritical_min_votes)
        if cond_h2:
            by_map_update_light = bool(by_map_update_light and symptom_refresh)
            if bool(by_map_update_hard) and (not bool(map_critical)):
                by_map_update_hard = bool(noncritical_symptom_refresh)
                by_map_update_hard_deferred = bool(map_update_hard_seen and (not by_map_update_hard))

        # Backward-compatible key: map_update means actionable hard update that can refresh now.
        by_map_update = bool(by_map_update_hard)

        if bool(map_update_hard_seen):
            self.map_update_hard_seen_count_total = int(self.map_update_hard_seen_count_total) + 1
            if bool(map_update_hard_actionable):
                self.map_update_hard_actionable_count_total = int(self.map_update_hard_actionable_count_total) + 1
            else:
                self.map_update_hard_deferred_count_total = int(self.map_update_hard_deferred_count_total) + 1
        for _rk, _rv in dict(map_actionable_reasons or {}).items():
            if str(_rk) not in self._map_update_hard_actionable_reasons_total:
                continue
            self._map_update_hard_actionable_reasons_total[str(_rk)] = int(
                self._map_update_hard_actionable_reasons_total.get(str(_rk), 0) + int(max(int(_rv), 0))
            )

        self._last_map_update_impacted_count = int(map_impacted_count)
        self._last_map_update_critical_count = int(map_critical_count)

        active_cooldown = int(max(getattr(env.cfg, "hrl_replan_cooldown_steps", self.replan_cooldown_steps), 0))

        step_now = int(env.state.step_index)
        window_len = int(max(self.decision_interval, 1))
        if step_now < int(self._event_window_start_step) or (step_now - int(self._event_window_start_step)) >= window_len:
            self._event_window_start_step = int(step_now)
            self._event_replans_in_window = 0
        event_budget = int(max(getattr(env.cfg, "hrl_event_replan_budget_per_window", 2), 0))
        low_value_merge_steps = int(max(getattr(env.cfg, "hrl_low_value_event_merge_steps", self.decision_interval), 1))

        # Unified value-aware trigger split.
        # Critical hard map updates remain high-value; non-critical hard map updates
        # are merged through the low-value path to reduce net churn in dense maps.
        by_map_update_hard_critical = bool(by_map_update_hard and bool(map_critical))
        by_map_update_hard_noncritical = bool(by_map_update_hard and (not bool(map_critical)))
        # High-priority uncovered emergency should only trigger when execution is actionable.
        if bool(by_high_priority_uncovered) and (not self._has_launchable_or_near_launchable_uav(env)):
            by_high_priority_uncovered = False
            self.high_priority_event_rejected_no_launchable_uav_count_total = int(
                self.high_priority_event_rejected_no_launchable_uav_count_total
            ) + 1
        weak_reason_flags = {
            "arrival": bool(by_arrival),
            "resolution": bool(by_resolution),
            "uav_idle": bool(by_uav_idle),
            "truck_idle": bool((not bool(getattr(env.cfg, "hrl_truck_idle_hard_refresh_enabled", False))) and by_truck_idle),
            "map_update_light": bool(by_map_update_light),
            "ranking_changed": bool(int(map_actionable_reasons.get("ranking_changed", 0)) > 0),
            "noncritical_map_update": bool(by_map_update_hard_noncritical or by_map_update_hard_deferred),
        }
        hard_reason_flags = {
            "goal_invalid": bool(by_goal_invalid),
            "goal_terminal": bool(by_goal_terminal),
            "path_blocked": bool(int(map_actionable_reasons.get("path_blocked", 0)) > 0),
            "goal_unreachable": bool(int(map_actionable_reasons.get("goal_unreachable", 0)) > 0),
            "uav_safety": bool(by_uav_emergency),
            "truck_dead_end": bool(by_truck_dead_end),
            "high_priority_uncovered": bool(by_high_priority_uncovered),
            "normal_stall": bool(by_normal_stall),
            "assigned_but_not_progressing": bool(by_normal_stall and bool(any(g is not None for g in self.state.goals.values()))),
        }
        truck_idle_as_hard = bool(getattr(env.cfg, "hrl_truck_idle_hard_refresh_enabled", False))
        high_value_event = bool(
            by_risk
            or by_goal_invalid
            or by_goal_terminal
            or by_truck_dead_end
            or by_normal_stall
            or by_high_priority_uncovered
            or by_uav_emergency
            or (truck_idle_as_hard and by_truck_idle)
            or by_map_update_hard_critical
        )
        low_value_event = bool(
            by_arrival
            or by_resolution
            or by_uav_idle
            or ((not truck_idle_as_hard) and by_truck_idle)
            or by_map_update_light
            or by_map_update_hard_deferred
            or by_map_update_hard_noncritical
        )
        weak_reason_active = [k for k, v in weak_reason_flags.items() if bool(v)]
        weak_reason_effective = list(weak_reason_active)
        weak_blocked_by_admission = False
        weak_blocked_by_cooldown = False
        event_admission_enabled = bool(self._event_admission_gate_enabled(env))
        step_now_int = int(env.state.step_index)
        if event_admission_enabled and weak_reason_effective and (not high_value_event):
            weak_actionable = bool(
                by_goal_invalid
                or by_map_update_hard
                or bool(int(map_actionable_reasons.get("path_blocked", 0)) > 0)
                or bool(int(map_actionable_reasons.get("goal_unreachable", 0)) > 0)
                or bool(int(map_actionable_reasons.get("recovery_path_fractured", 0)) > 0)
                or bool(self._uncovered_low_lifeline_emergency_exists(env))
                or bool(since_last >= int(max(getattr(env.cfg, "hrl_max_no_refresh_steps", 5), 1)))
            )
            if not weak_actionable:
                weak_reason_effective = []
                weak_blocked_by_admission = True
            else:
                weak_reason_effective = [
                    rr
                    for rr in weak_reason_effective
                    if not self._event_reason_in_cooldown(env, rr, step_now_int)
                ]
                if weak_reason_active and (not weak_reason_effective):
                    weak_blocked_by_cooldown = True
        low_value_event = bool(len(weak_reason_effective) > 0)
        if low_value_event:
            self.low_value_refresh_candidate_count_total = int(self.low_value_refresh_candidate_count_total) + 1
        if bool(getattr(env.cfg, "erc_ablate_low_value_refresh", False)):
            if low_value_event:
                self.low_value_refresh_blocked_by_ablation_count_total = int(self.low_value_refresh_blocked_by_ablation_count_total) + 1
            low_value_event = False
        if low_value_event:
            self._low_value_event_streak = int(self._low_value_event_streak) + 1
        else:
            self._low_value_event_streak = 0

        cooldown_blocked = bool(since_last < active_cooldown and bool(self.state.goals))
        low_value_ready = bool(
            low_value_event
            and (not cooldown_blocked)
            and int(self._low_value_event_streak) >= low_value_merge_steps
        )
        if low_value_ready:
            self.low_value_refresh_allowed_count_total = int(self.low_value_refresh_allowed_count_total) + 1

        by_no_event_fallback = bool(
            event_first_enabled
            and bool(self.state.goals)
            and (not high_value_event)
            and (not low_value_ready)
            and since_last >= max_no_refresh_steps
        )

        event_detected = bool(
            by_risk
            or by_resolution
            or by_arrival
            or by_goal_invalid
            or by_goal_terminal
            or by_truck_dead_end
            or by_uav_emergency
            or by_uav_idle
            or by_truck_idle
            or by_normal_stall
            or by_high_priority_uncovered
            or by_map_update_hard
            or by_map_update_light
            or by_map_update_hard_deferred
        )
        if event_detected:
            self.erc_event_detected_count_total = int(self.erc_event_detected_count_total) + 1

        do_refresh = bool(
            (not self.state.goals)
            or by_interval
            or high_value_event
            or low_value_ready
            or by_no_event_fallback
        )

        # Execution commitment baseline + event evidence gate + local correction:
        # keep committed goals by default and avoid event-driven global replans.
        if self._commitment_local_correction_mode(env) and bool(self.state.goals):
            hard_invalid = bool(by_goal_invalid or by_goal_terminal)
            hard_safety = bool(by_uav_emergency and self.uav_recovery_hard_count_total > 0)
            map_hard_impact = bool(by_map_update_hard and (map_critical or int(goal_unreachable_step) > 0 or int(recovery_fractured_step) > 0))
            gate_pass = bool(hard_invalid or hard_safety or map_hard_impact or by_no_event_fallback or by_interval or by_risk)
            if gate_pass:
                self.erc_event_gate_pass_count_total = int(self.erc_event_gate_pass_count_total) + 1
            else:
                # Local correction path: suppress global refresh, keep current commitments.
                if event_detected:
                    self.erc_event_gate_reject_count_total = int(self.erc_event_gate_reject_count_total) + 1
                    self.erc_local_correction_count_total = int(self.erc_local_correction_count_total) + 1
                    if bool(by_truck_dead_end or by_normal_stall or by_map_update_hard or by_map_update_light):
                        self.path_blocked_local_agent_count_total = int(self.path_blocked_local_agent_count_total) + int(
                            max(int(map_impacted_count), 1)
                        )
                do_refresh = False

        event_budget_blocked = False
        if (
            self.use_event_trigger
            and (not by_interval)
            and (not by_no_event_fallback)
            and bool(do_refresh)
            and bool(self.state.goals)
        ):
            # Budget should only throttle low-value refreshes.
            if (not high_value_event) and int(self._event_replans_in_window) >= int(event_budget):
                do_refresh = False
                event_budget_blocked = True

        if bool(do_refresh):
            if by_interval:
                self._event_window_start_step = int(step_now)
                self._event_replans_in_window = 0
            elif bool(self.state.goals):
                self._event_replans_in_window = int(self._event_replans_in_window) + 1

        map_update_hard_immediate_refresh_step = bool(by_map_update_hard and do_refresh)
        if map_update_hard_immediate_refresh_step:
            self.map_update_hard_immediate_refresh_count_total = int(self.map_update_hard_immediate_refresh_count_total) + 1
            if int(max(int(map_actionable_reasons.get("path_blocked", 0)), 0)) > 0:
                self.path_blocked_global_refresh_count_total = int(self.path_blocked_global_refresh_count_total) + 1

        if map_signal and hasattr(env, "_shared_map_update_event_step"):
            env._shared_map_update_event_step = False

        self._last_refresh_flags = {
            "interval": bool(by_interval),
            "risk_spike": bool(by_risk),
            "resolution": bool(by_resolution),
            "arrival": bool(by_arrival),
            "goal_terminal": bool(by_goal_terminal),
            "goal_invalid": bool(by_goal_invalid),
            "normal_stall": bool(by_normal_stall),
            "high_priority_uncovered": bool(by_high_priority_uncovered),
            "truck_dead_end": bool(by_truck_dead_end),
            "truck_idle": bool(by_truck_idle),
            "uav_emergency": bool(by_uav_emergency),
            "uav_idle": bool(by_uav_idle),
            "map_update": bool(by_map_update),
            "map_update_light": bool(by_map_update_light),
            "map_update_hard_seen": bool(map_update_hard_seen),
            "map_update_hard_actionable": bool(by_map_update_hard),
            "map_update_hard_deferred": bool(by_map_update_hard_deferred),
            "map_update_hard_immediate_refresh": bool(map_update_hard_immediate_refresh_step),
            "map_update_hard_seen_step": int(1 if map_update_hard_seen else 0),
            "map_update_hard_actionable_step": int(1 if by_map_update_hard else 0),
            "map_update_hard_deferred_step": int(1 if by_map_update_hard_deferred else 0),
            "map_update_hard_immediate_refresh_step": int(1 if map_update_hard_immediate_refresh_step else 0),
            "map_update_hard_seen_count_total": int(self.map_update_hard_seen_count_total),
            "map_update_hard_actionable_count_total": int(self.map_update_hard_actionable_count_total),
            "map_update_hard_deferred_count_total": int(self.map_update_hard_deferred_count_total),
            "map_update_hard_immediate_refresh_count_total": int(self.map_update_hard_immediate_refresh_count_total),
            "map_update_hard_reason_path_blocked_step": int(map_actionable_reasons.get("path_blocked", 0)),
            "map_update_hard_reason_goal_unreachable_step": int(map_actionable_reasons.get("goal_unreachable", 0)),
            "map_update_hard_reason_ranking_changed_step": int(map_actionable_reasons.get("ranking_changed", 0)),
            "map_update_hard_reason_dead_end_step": int(map_actionable_reasons.get("dead_end", 0)),
            "map_update_hard_reason_recovery_path_fractured_step": int(map_actionable_reasons.get("recovery_path_fractured", 0)),
            "map_update_hard_reason_path_blocked_total": int(self._map_update_hard_actionable_reasons_total.get("path_blocked", 0)),
            "map_update_hard_reason_goal_unreachable_total": int(self._map_update_hard_actionable_reasons_total.get("goal_unreachable", 0)),
            "map_update_hard_reason_ranking_changed_total": int(self._map_update_hard_actionable_reasons_total.get("ranking_changed", 0)),
            "map_update_hard_reason_dead_end_total": int(self._map_update_hard_actionable_reasons_total.get("dead_end", 0)),
            "map_update_hard_reason_recovery_path_fractured_total": int(self._map_update_hard_actionable_reasons_total.get("recovery_path_fractured", 0)),
            "high_value_event": bool(high_value_event),
            "low_value_event": bool(low_value_event),
            "low_value_event_streak": int(self._low_value_event_streak),
            "map_critical": bool(map_critical),
            "map_impacted_count": int(map_impacted_count),
            "map_critical_count": int(map_critical_count),
            "symptom_refresh": bool(symptom_refresh),
            "info_burst": bool(info_burst),
            "conversion_stall": bool(conversion_stall),
            "island_serviceability_drop": bool(island_serviceability_drop),
            "island_serviceability_low": bool(island_serviceability_low),
            "route_blocked": bool(route_blocked),
            "delivery_stall_steps": int(self._delivery_stall_steps),
            "support_conversion_quality": float(support_quality),
            "active_cooldown": int(active_cooldown),
            "low_value_merge_steps": int(low_value_merge_steps),
            "empty_goals": bool(not self.state.goals),
            "cooldown_blocked": bool(cooldown_blocked),
            "event_budget_blocked": bool(event_budget_blocked),
            "event_replans_in_window": int(self._event_replans_in_window),
            "event_first_enabled": bool(event_first_enabled),
            "no_event_fallback_refresh": bool(by_no_event_fallback and do_refresh),
            "hard_event_refresh": bool(high_value_event),
            "event_admission_enabled": bool(event_admission_enabled),
            "weak_reason_active_count": int(len(weak_reason_active)),
            "weak_reason_effective_count": int(len(weak_reason_effective)),
            "weak_blocked_by_admission": bool(weak_blocked_by_admission),
            "weak_blocked_by_cooldown": bool(weak_blocked_by_cooldown),
            "weak_reason_arrival": bool("arrival" in weak_reason_effective),
            "weak_reason_resolution": bool("resolution" in weak_reason_effective),
            "weak_reason_uav_idle": bool("uav_idle" in weak_reason_effective),
            "weak_reason_truck_idle": bool("truck_idle" in weak_reason_effective),
            "weak_reason_map_update_light": bool("map_update_light" in weak_reason_effective),
            "weak_reason_ranking_changed": bool("ranking_changed" in weak_reason_effective),
            "weak_reason_noncritical_map_update": bool("noncritical_map_update" in weak_reason_effective),
            "hard_reason_goal_invalid": bool(hard_reason_flags.get("goal_invalid", False)),
            "hard_reason_path_blocked": bool(hard_reason_flags.get("path_blocked", False)),
            "hard_reason_goal_unreachable": bool(hard_reason_flags.get("goal_unreachable", False)),
            "hard_reason_uav_safety": bool(hard_reason_flags.get("uav_safety", False)),
            "hard_reason_truck_dead_end": bool(hard_reason_flags.get("truck_dead_end", False)),
            "hard_reason_high_priority_uncovered": bool(hard_reason_flags.get("high_priority_uncovered", False)),
            "hard_reason_normal_stall": bool(hard_reason_flags.get("normal_stall", False)),
            "hard_reason_assigned_but_not_progressing": bool(hard_reason_flags.get("assigned_but_not_progressing", False)),
            "refresh": bool(do_refresh),
        }
        return bool(do_refresh)

    def _region_commitment_active(self, env) -> bool:
        if not bool(getattr(env.cfg, "region_commitment_enabled", False)):
            return False
        min_size = float(max(getattr(env.cfg, "region_commitment_min_map_size_m", 9000.0), 0.0))
        if float(getattr(env.cfg, "map_size_m", 0.0)) < min_size:
            return False
        if self._region_commitment_signature is not None:
            return bool(self._region_commitment_enabled_effective and int(self._region_commitment_effective_k) > 1)
        return True

    def _task_xy(self, env, task) -> Tuple[float, float]:
        n = env.topology.nodes[int(task.demand_node)]
        return float(n.x), float(n.y)

    def _region_count(self, env, n_points: int) -> int:
        explicit = int(max(getattr(env.cfg, "region_commitment_count", 0), 0))
        if explicit > 0:
            return int(np.clip(explicit, 1, max(n_points, 1)))
        truck_count = int(
            sum(
                1
                for st in env.state.agents.values()
                if st.kind == AgentKind.TRUCK and not bool(getattr(st, "crashed", False))
            )
        )
        map_size = float(getattr(env.cfg, "map_size_m", 0.0))
        if map_size >= 18000.0:
            return int(min(max(truck_count, 1), 3, max(n_points, 1)))
        return int(min(max(truck_count, 1), 2, max(n_points, 1)))

    def _farthest_first_centers(self, pts: np.ndarray, k: int) -> np.ndarray:
        if len(pts) == 0:
            return np.zeros((0, 2), dtype=np.float64)
        center0 = np.mean(pts, axis=0)
        first = int(np.argmin(np.sum((pts - center0) ** 2, axis=1)))
        centers = [pts[first]]
        while len(centers) < int(k):
            cur = np.asarray(centers, dtype=np.float64)
            d2 = np.min(np.sum((pts[:, None, :] - cur[None, :, :]) ** 2, axis=2), axis=1)
            centers.append(pts[int(np.argmax(d2))])
        return np.asarray(centers, dtype=np.float64)

    def _cluster_points(self, pts: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        centers = self._farthest_first_centers(pts, k)
        labels = np.zeros((len(pts),), dtype=np.int64)
        for _ in range(10):
            d2 = np.sum((pts[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels = np.argmin(d2, axis=1)
            new_centers = centers.copy()
            for rid in range(k):
                mask = labels == rid
                if np.any(mask):
                    new_centers[rid] = np.mean(pts[mask], axis=0)
            if np.allclose(new_centers, centers):
                break
            centers = new_centers
        return centers, labels

    def _score_region_partition(
        self,
        env,
        task_pts: np.ndarray,
        centers: np.ndarray,
        labels: np.ndarray,
        tc_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        k = int(len(centers))
        if k <= 1 or len(task_pts) <= 1:
            return {"score": -1e9, "separation": 0.0, "balance": 0.0, "coverage": 0.0, "outliers": 0.0}
        map_size = float(max(getattr(env.cfg, "map_size_m", 1.0), 1.0))
        radii: List[float] = []
        counts: List[int] = []
        for rid in range(k):
            mask = labels == rid
            counts.append(int(np.sum(mask)))
            if np.any(mask):
                d = np.sqrt(np.sum((task_pts[mask] - centers[rid]) ** 2, axis=1))
                radii.append(float(np.mean(d)))
            else:
                radii.append(float(map_size))
        if min(counts) <= 0:
            return {"score": -1e9, "separation": 0.0, "balance": 0.0, "coverage": 0.0, "outliers": 0.0}
        center_d = []
        for i in range(k):
            for j in range(i + 1, k):
                center_d.append(float(np.linalg.norm(centers[i] - centers[j])))
        mean_inter = float(np.mean(center_d)) if center_d else 0.0
        mean_radius = float(max(np.mean(radii), 1.0))
        separation = float(np.clip((mean_inter / mean_radius - 1.0) / 3.0, 0.0, 1.0))
        balance = float(np.clip(min(counts) / max(max(counts), 1), 0.0, 1.0))

        trucks = [
            self._agent_xy(env, str(aid))
            for aid, st in env.state.agents.items()
            if st.kind == AgentKind.TRUCK and not bool(getattr(st, "crashed", False))
        ]
        if trucks:
            cov_vals = []
            for c in centers:
                nearest = min(float(np.hypot(float(c[0]) - tx, float(c[1]) - ty)) for tx, ty in trucks)
                cov_vals.append(float(1.0 - np.clip(nearest / max(map_size * 0.50, 1.0), 0.0, 1.0)))
            coverage = float(np.clip(np.mean(cov_vals), 0.0, 1.0))
        else:
            coverage = 0.0

        over_pen = float(max(getattr(env.cfg, "region_commitment_overpartition_penalty", 0.16), 0.0)) * float(max(k - 2, 0))
        outliers = 0
        if tc_mask is not None and len(tc_mask) == len(task_pts):
            ratio_thr = float(max(getattr(env.cfg, "region_commitment_outlier_distance_ratio", 0.22), 0.0))
            min_thr = float(max(getattr(env.cfg, "region_commitment_outlier_min_distance_m", 3200.0), 0.0))
            outlier_thr = float(max(min_thr, ratio_thr * map_size))
            for idx in range(len(task_pts)):
                if not bool(tc_mask[idx]):
                    continue
                rid = int(labels[idx])
                if rid < 0 or rid >= len(centers):
                    continue
                d = float(np.linalg.norm(task_pts[idx] - centers[rid]))
                if d >= outlier_thr:
                    outliers += 1
        score = float(0.50 * separation + 0.25 * balance + 0.25 * coverage - over_pen)
        return {"score": score, "separation": separation, "balance": balance, "coverage": coverage, "outliers": float(outliers)}

    def _select_region_partition(
        self,
        env,
        task_pts: np.ndarray,
        tc_mask: Optional[np.ndarray] = None,
    ) -> Tuple[int, np.ndarray, np.ndarray, Dict[str, float]]:
        if len(task_pts) <= 1:
            centers = task_pts.copy()
            labels = np.zeros((len(task_pts),), dtype=np.int64)
            return 1, centers, labels, {"score": 0.0, "separation": 0.0, "balance": 1.0, "coverage": 1.0, "outliers": 0.0}
        explicit = int(max(getattr(env.cfg, "region_commitment_count", 0), 0))
        if explicit > 0 and not bool(getattr(env.cfg, "region_commitment_auto_select_enabled", False)):
            k = int(np.clip(explicit, 1, len(task_pts)))
            centers, labels = self._cluster_points(task_pts, k)
            stats = self._score_region_partition(env, task_pts, centers, labels, tc_mask=tc_mask)
            return k, centers, labels, stats

        truck_count = int(
            sum(
                1
                for st in env.state.agents.values()
                if st.kind == AgentKind.TRUCK and not bool(getattr(st, "crashed", False))
            )
        )
        max_k_cfg = int(max(getattr(env.cfg, "region_commitment_auto_max_k", 3), 1))
        max_k = int(min(max_k_cfg, max(truck_count, 1), len(task_pts)))
        best_k = 1
        best_centers = np.mean(task_pts, axis=0, keepdims=True)
        best_labels = np.zeros((len(task_pts),), dtype=np.int64)
        best_stats = {"score": 0.0, "separation": 0.0, "balance": 1.0, "coverage": 1.0, "outliers": 0.0}
        for k in range(2, max_k + 1):
            centers, labels = self._cluster_points(task_pts, k)
            stats = self._score_region_partition(env, task_pts, centers, labels, tc_mask=tc_mask)
            if float(stats["score"]) > float(best_stats["score"]):
                best_k, best_centers, best_labels, best_stats = int(k), centers, labels, stats
        score_thr = float(max(getattr(env.cfg, "region_commitment_enable_score_threshold", 0.18), 0.0))
        sep_thr = float(max(getattr(env.cfg, "region_commitment_min_separation_score", 0.22), 0.0))
        bal_thr = float(max(getattr(env.cfg, "region_commitment_min_load_balance_score", 0.20), 0.0))
        unbalanced_sep_thr = float(max(getattr(env.cfg, "region_commitment_unbalanced_min_separation_score", 0.75), 0.0))
        unbalanced_outlier_min = int(max(getattr(env.cfg, "region_commitment_unbalanced_min_outlier_tasks", 3), 0))
        balance_ok = bool(float(best_stats["balance"]) >= bal_thr)
        unbalanced_but_useful = bool(
            float(best_stats["separation"]) >= unbalanced_sep_thr
            and int(float(best_stats.get("outliers", 0.0))) >= unbalanced_outlier_min
        )
        if (
            best_k <= 1
            or float(best_stats["score"]) < score_thr
            or float(best_stats["separation"]) < sep_thr
            or (not balance_ok and not unbalanced_but_useful)
        ):
            best_k = 1
            best_centers = np.mean(task_pts, axis=0, keepdims=True)
            best_labels = np.zeros((len(task_pts),), dtype=np.int64)
            best_stats = {
                "score": float(best_stats["score"]),
                "separation": float(best_stats["separation"]),
                "balance": float(best_stats["balance"]),
                "coverage": float(best_stats["coverage"]),
                "outliers": float(best_stats.get("outliers", 0.0)),
            }
        return int(best_k), best_centers, best_labels, best_stats

    def _ensure_region_commitment(self, env) -> None:
        if not self._region_commitment_active(env):
            return
        task_ids = sorted(str(t.task_id) for t in env.state.tasks.values())
        sig = (
            int(getattr(env.state, "episode_seed", getattr(env.cfg, "seed", 0))),
            int(len(task_ids)),
            int(sum(int(t.demand_node) for t in env.state.tasks.values())),
            float(getattr(env.cfg, "map_size_m", 0.0)),
        )
        if self._region_commitment_signature == sig and self._task_region_by_task:
            return

        pts_list: List[Tuple[str, float, float]] = []
        for t in env.state.tasks.values():
            try:
                x, y = self._task_xy(env, t)
            except Exception:
                continue
            pts_list.append((str(t.task_id), float(x), float(y)))
        if bool(getattr(env.cfg, "region_commitment_include_gateways", True)):
            meta = getattr(env.topology, "real_case_meta", {})
            gateways = meta.get("gateways", []) if isinstance(meta, dict) else []
            for idx, item in enumerate(gateways[:80] if isinstance(gateways, list) else []):
                try:
                    x = float(item.get("x", item.get("xy", [np.nan, np.nan])[0]))
                    y = float(item.get("y", item.get("xy", [np.nan, np.nan])[1]))
                except Exception:
                    continue
                if np.isfinite(x) and np.isfinite(y):
                    pts_list.append((f"__gw{idx}", x, y))
        if not pts_list:
            return

        pts = np.asarray([(x, y) for _, x, y in pts_list], dtype=np.float64)
        task_pts = np.asarray(
            [(x, y) for pid, x, y in pts_list if not str(pid).startswith("__gw")],
            dtype=np.float64,
        )
        task_tc_mask = np.asarray(
            [
                bool(
                    (str(pid) in env.state.tasks)
                    and self._is_timecritical_lightweight_task(env.state.tasks[str(pid)])
                )
                for pid, _, _ in pts_list
                if not str(pid).startswith("__gw")
            ],
            dtype=bool,
        )
        if len(task_pts) == 0:
            task_pts = pts
            task_tc_mask = None
        k, centers, task_labels, stats = self._select_region_partition(env, task_pts, tc_mask=task_tc_mask)

        labels = np.zeros((len(pts),), dtype=np.int64)
        task_i = 0
        for i, (pid, x, y) in enumerate(pts_list):
            if not str(pid).startswith("__gw"):
                labels[i] = int(task_labels[task_i])
                task_i += 1
            else:
                d2 = np.sum((centers - np.asarray([float(x), float(y)], dtype=np.float64)) ** 2, axis=1)
                labels[i] = int(np.argmin(d2))

        self._region_commitment_signature = sig
        self._region_commitment_effective_k = int(k)
        self._region_commitment_enabled_effective = bool(int(k) > 1)
        self._region_commitment_auto_score = float(stats.get("score", 0.0))
        self._region_commitment_separation_score = float(stats.get("separation", 0.0))
        self._region_commitment_load_balance_score = float(stats.get("balance", 0.0))
        self._region_commitment_coverage_score = float(stats.get("coverage", 0.0))
        if self._region_commitment_enabled_effective:
            bal_thr_eff = float(max(getattr(env.cfg, "region_commitment_min_load_balance_score", 0.20), 0.0))
            if self._region_commitment_load_balance_score < bal_thr_eff:
                self._region_commitment_strength = 1.0
            else:
                strength_min = float(np.clip(getattr(env.cfg, "region_commitment_strength_min", 0.35), 0.0, 1.0))
                self._region_commitment_strength = float(
                    np.clip(
                        strength_min
                        + 0.35 * self._region_commitment_separation_score
                        + 0.20 * self._region_commitment_load_balance_score
                        + 0.10 * self._region_commitment_coverage_score,
                        0.0,
                        1.0,
                    )
                )
        else:
            self._region_commitment_strength = 0.0
        if self._region_commitment_enabled_effective:
            self.region_commitment_auto_enabled_count_total = int(self.region_commitment_auto_enabled_count_total) + 1
        else:
            self.region_commitment_auto_disabled_count_total = int(self.region_commitment_auto_disabled_count_total) + 1
        self._region_centers_xy = {int(i): (float(c[0]), float(c[1])) for i, c in enumerate(centers)}
        self._task_region_by_task.clear()
        self._region_task_distance_m.clear()
        self._region_outlier_task_ids.clear()
        for (pid, _, _), rid in zip(pts_list, labels.tolist()):
            if not str(pid).startswith("__gw"):
                self._task_region_by_task[str(pid)] = int(rid)

        for t in env.state.tasks.values():
            tid = str(t.task_id)
            rid = self._task_region_by_task.get(tid, None)
            if rid is None or int(rid) not in self._region_centers_xy:
                continue
            tx, ty = self._task_xy(env, t)
            cx, cy = self._region_centers_xy[int(rid)]
            d_region = float(np.hypot(float(tx) - float(cx), float(ty) - float(cy)))
            self._region_task_distance_m[tid] = float(d_region)
            if self._is_timecritical_lightweight_task(t):
                map_size_m = float(max(getattr(env.cfg, "map_size_m", 0.0), 1.0))
                ratio_thr = float(max(getattr(env.cfg, "region_commitment_outlier_distance_ratio", 0.22), 0.0))
                min_thr = float(max(getattr(env.cfg, "region_commitment_outlier_min_distance_m", 3200.0), 0.0))
                if d_region >= max(min_thr, ratio_thr * map_size_m):
                    self._region_outlier_task_ids.add(tid)

        self.region_commitment_outlier_task_count_total = int(self.region_commitment_outlier_task_count_total) + int(len(self._region_outlier_task_ids))

        trucks = [
            str(aid)
            for aid, st in env.state.agents.items()
            if st.kind == AgentKind.TRUCK and not bool(getattr(st, "crashed", False))
        ]
        self._agent_home_region.clear()
        used_regions: set = set()
        for aid in sorted(trucks):
            ax, ay = self._agent_xy(env, aid)
            order = sorted(
                self._region_centers_xy.keys(),
                key=lambda rid: (float((ax - self._region_centers_xy[rid][0]) ** 2 + (ay - self._region_centers_xy[rid][1]) ** 2), int(rid)),
            )
            rid = next((r for r in order if r not in used_regions), order[0] if order else 0)
            self._agent_home_region[str(aid)] = int(rid)
            used_regions.add(int(rid))

        for aid, st in env.state.agents.items():
            if st.kind != AgentKind.UAV or bool(getattr(st, "crashed", False)):
                continue
            ft = str(st.follow_target) if st.follow_target is not None else ""
            if ft in self._agent_home_region:
                self._agent_home_region[str(aid)] = int(self._agent_home_region[ft])
                continue
            ax, ay = self._agent_xy(env, str(aid))
            if self._region_centers_xy:
                rid = min(
                    self._region_centers_xy.keys(),
                    key=lambda r: float((ax - self._region_centers_xy[r][0]) ** 2 + (ay - self._region_centers_xy[r][1]) ** 2),
                )
                self._agent_home_region[str(aid)] = int(rid)

        self.region_commitment_setup_count_total = int(self.region_commitment_setup_count_total) + 1

    def _task_region(self, env, task) -> Optional[int]:
        self._ensure_region_commitment(env)
        return self._task_region_by_task.get(str(getattr(task, "task_id", "")), None)

    def _agent_region(self, env, aid: str) -> Optional[int]:
        self._ensure_region_commitment(env)
        rid = self._agent_home_region.get(str(aid), None)
        st = env.state.agents.get(str(aid), None)
        if st is not None and st.kind == AgentKind.UAV and st.follow_target is not None:
            tr = self._agent_home_region.get(str(st.follow_target), None)
            if tr is not None:
                rid = int(tr)
                self._agent_home_region[str(aid)] = int(tr)
        return rid

    def _region_has_local_work(self, env, aid: str, region_id: Optional[int]) -> bool:
        if region_id is None:
            return False
        st = env.state.agents.get(str(aid), None)
        if st is None:
            return False
        for t in env.state.tasks.values():
            if not self._task_planner_active(t):
                continue
            if self._task_region(env, t) != int(region_id):
                continue
            if st.kind == AgentKind.UAV and t.kind != TaskKind.EMERGENCY:
                continue
            if st.kind == AgentKind.TRUCK:
                if not self._truck_task_serviceable_or_support_proxy(env, str(aid), t):
                    continue
                if not self._truck_task_reachable(env, str(aid), t):
                    continue
            return True
        return False

    def _region_has_local_routine_work(self, env, aid: str, region_id: Optional[int]) -> bool:
        return bool(np.isfinite(self._region_nearest_local_routine_distance_m(env, aid, region_id)))

    def _region_nearest_local_routine_distance_m(self, env, aid: str, region_id: Optional[int]) -> float:
        if region_id is None:
            return float("inf")
        st = env.state.agents.get(str(aid), None)
        if st is None or st.kind != AgentKind.TRUCK:
            return float("inf")
        best = float("inf")
        for t in env.state.tasks.values():
            if t.status != TaskStatus.PENDING or t.kind != TaskKind.NORMAL:
                continue
            if self._task_region(env, t) != int(region_id):
                continue
            if not self._truck_task_serviceable_or_support_proxy(env, str(aid), t):
                continue
            if not self._truck_task_reachable(env, str(aid), t):
                continue
            d = float(self._truck_task_distance(env, str(aid), t))
            if np.isfinite(d):
                best = min(best, d)
        return float(best)

    def _region_cross_override(self, env, aid: str, task) -> bool:
        if task is None:
            return False
        cur = self.state.goals.get(str(aid), None)
        if bool(getattr(env.cfg, "region_commitment_keep_current_goal_enabled", True)) and cur is not None and str(cur) == str(task.task_id):
            return True
        if self._is_timecritical_lightweight_task(task):
            ratio = float(np.clip(self._task_lifeline_ratio(task), 0.0, 1.0))
            thr = float(np.clip(getattr(env.cfg, "region_commitment_override_lifeline_ratio", 0.32), 0.0, 1.0))
            if ratio <= thr:
                return True
        return False

    def _region_outlier_task(self, env, task) -> bool:
        if not bool(getattr(env.cfg, "region_commitment_outlier_gate_enabled", False)):
            return False
        self._ensure_region_commitment(env)
        return bool(str(getattr(task, "task_id", "")) in self._region_outlier_task_ids)

    def _region_outlier_override(self, env, aid: str, task) -> bool:
        if task is None:
            return False
        cur = self.state.goals.get(str(aid), None)
        if bool(getattr(env.cfg, "region_commitment_keep_current_goal_enabled", True)) and cur is not None and str(cur) == str(task.task_id):
            return True
        if self._is_timecritical_lightweight_task(task):
            ratio = float(np.clip(self._task_lifeline_ratio(task), 0.0, 1.0))
            thr = float(np.clip(getattr(env.cfg, "region_commitment_outlier_override_lifeline_ratio", 0.40), 0.0, 1.0))
            if ratio <= thr:
                return True
        st = env.state.agents.get(str(aid), None)
        if st is not None and st.kind == AgentKind.UAV:
            try:
                if self._uav_task_feasible(env, str(aid), task):
                    return True
            except Exception:
                pass
        if st is not None and st.kind == AgentKind.TRUCK:
            if not bool(getattr(env.cfg, "region_commitment_outlier_require_support_gain", True)):
                return True
            try:
                gain = float(self._truck_support_serviceability_gain(env, str(aid), task))
            except Exception:
                gain = 0.0
            min_gain = float(max(getattr(env.cfg, "region_commitment_outlier_min_support_gain", 0.12), 0.0))
            if gain >= min_gain:
                return True
        return False

    def _region_task_allowed(self, env, aid: str, task) -> bool:
        if not self._region_commitment_active(env):
            return True
        ar = self._agent_region(env, str(aid))
        tr = self._task_region(env, task)
        if (
            self._region_commitment_enabled_effective
            and
            self._region_outlier_task(env, task)
            and ar is not None
            and self._region_has_local_work(env, str(aid), ar)
        ):
            if self._region_outlier_override(env, str(aid), task):
                self.region_commitment_outlier_override_count_total = int(self.region_commitment_outlier_override_count_total) + 1
            else:
                self.region_commitment_outlier_filtered_count_total = int(self.region_commitment_outlier_filtered_count_total) + 1
                return False
        if ar is None or tr is None or int(ar) == int(tr):
            if ar is not None and tr is not None:
                self.region_commitment_local_candidate_count_total = int(self.region_commitment_local_candidate_count_total) + 1
            return True
        if not self._region_has_local_work(env, str(aid), ar):
            self.region_commitment_cross_override_count_total = int(self.region_commitment_cross_override_count_total) + 1
            return True
        if self._region_cross_override(env, str(aid), task):
            self.region_commitment_cross_override_count_total = int(self.region_commitment_cross_override_count_total) + 1
            return True
        if bool(getattr(env.cfg, "region_commitment_cross_region_filter_enabled", True)):
            self.region_commitment_cross_filtered_count_total = int(self.region_commitment_cross_filtered_count_total) + 1
            return False
        return True

    def _region_score_adjustment(self, env, aid: str, task) -> float:
        if not self._region_commitment_active(env):
            return 0.0
        ar = self._agent_region(env, str(aid))
        tr = self._task_region(env, task)
        if ar is None or tr is None:
            return 0.0
        strength = float(np.clip(getattr(self, "_region_commitment_strength", 1.0), 0.0, 1.0))
        adj = 0.0
        if int(ar) == int(tr):
            adj += float(strength * max(getattr(env.cfg, "region_commitment_local_bonus", 0.28), 0.0))
        elif self._region_outlier_task(env, task) and not self._region_outlier_override(env, str(aid), task):
            adj -= float(strength * max(getattr(env.cfg, "region_commitment_outlier_penalty", 0.65), 0.0))
        elif not (self._region_cross_override(env, str(aid), task) or not self._region_has_local_work(env, str(aid), ar)):
            adj -= float(strength * max(getattr(env.cfg, "region_commitment_cross_region_penalty", 0.85), 0.0))

        if bool(getattr(env.cfg, "region_commitment_routine_guard_enabled", False)):
            st = env.state.agents.get(str(aid), None)
            if (
                st is not None
                and st.kind == AgentKind.TRUCK
                and task is not None
                and task.kind == TaskKind.EMERGENCY
                and task.status == TaskStatus.PENDING
            ):
                cur = self.state.goals.get(str(aid), None)
                near_dist = float(self._region_nearest_local_routine_distance_m(env, str(aid), ar))
                max_near_dist = float(max(getattr(env.cfg, "region_commitment_routine_guard_max_normal_dist_m", 1200.0), 0.0))
                if (cur is None or str(cur) != str(task.task_id)) and near_dist <= max_near_dist:
                    ratio = float(np.clip(self._task_lifeline_ratio(task), 0.0, 1.0))
                    bypass_ratio = float(
                        np.clip(getattr(env.cfg, "region_commitment_routine_guard_lifeline_ratio", 0.42), 0.0, 1.0)
                    )
                    if ratio > bypass_ratio:
                        base = float(max(getattr(env.cfg, "region_commitment_routine_guard_penalty", 0.36), 0.0))
                        try:
                            gain = float(np.clip(self._truck_support_serviceability_gain(env, str(aid), task), 0.0, 1.0))
                        except Exception:
                            gain = 0.0
                        relief = float(
                            np.clip(getattr(env.cfg, "region_commitment_routine_guard_support_gain_relief", 0.55), 0.0, 1.0)
                        )
                        adj -= float(strength * base * (1.0 - relief * gain))
        return float(adj)

    def _candidate_tasks(self, env, aid: str) -> List[object]:
        a = env.state.agents[aid]
        if a.kind == AgentKind.TRUCK and self._truck_stage_blocks_task_goal(env, str(aid)):
            return []
        self._ensure_region_commitment(env)
        tasks = [t for t in env.state.tasks.values() if self._task_planner_active(t)]
        if a.kind == AgentKind.UAV:
            tasks = [t for t in tasks if t.kind == TaskKind.EMERGENCY]
        if hasattr(env, "is_task_serviceable_by_agent") and a.kind == AgentKind.TRUCK:
            tasks = [t for t in tasks if self._truck_task_serviceable_or_support_proxy(env, str(aid), t)]
        elif hasattr(env, "is_task_serviceable_by_agent"):
            tasks = [t for t in tasks if bool(env.is_task_serviceable_by_agent(str(aid), t))]
        if a.kind == AgentKind.TRUCK:
            tasks = [t for t in tasks if self._truck_task_reachable(env, str(aid), t)]
            pending_norm, norm_reach_by_truck, any_reachable_normal = self._normal_reachability_snapshot(env)
            truck_has_normal_reachable = bool(norm_reach_by_truck.get(str(aid), True))
            hard_normal_first = bool(getattr(env.cfg, "hrl_truck_hard_normal_first_enabled", True))
            no_normal_mode = bool(
                int(pending_norm) > 0
                and bool(getattr(env.cfg, "hrl_truck_emergency_support_when_no_normal_enabled", True))
                and (not truck_has_normal_reachable)
            )
            if hard_normal_first and int(pending_norm) > 0 and truck_has_normal_reachable and any_reachable_normal:
                normal_tasks = [t for t in tasks if t.kind == TaskKind.NORMAL]
                if normal_tasks:
                    tasks = normal_tasks
            elif no_normal_mode:
                support_tasks = [
                    t
                    for t in tasks
                    if t.kind == TaskKind.EMERGENCY
                    and self._truck_supportworthy_emergency_task(env, str(aid), t)
                ]
                if support_tasks:
                    tasks = support_tasks
            tasks = [t for t in tasks if self._truck_emergency_relief_allowed(env, str(aid), t)]
        if self._region_commitment_active(env):
            tasks = [t for t in tasks if self._region_task_allowed(env, str(aid), t)]
        return tasks


    def _pending_normal_task_count(self, env) -> int:
        return int(
            sum(
                1
                for t in env.state.tasks.values()
                if t.status == TaskStatus.PENDING and t.kind == TaskKind.NORMAL
            )
        )

    def _count_trucks_assigned_to_normal(self, env) -> int:
        cnt = 0
        for aid, ag in env.state.agents.items():
            if ag.kind != AgentKind.TRUCK:
                continue
            gid = self.state.goals.get(str(aid), None)
            if gid is None:
                continue
            t = env.state.tasks.get(str(gid), None)
            if t is not None and t.status == TaskStatus.PENDING and t.kind == TaskKind.NORMAL:
                cnt += 1
        return int(cnt)

    def _truck_has_reachable_normal(self, env, aid: str) -> bool:
        fn = getattr(env, "_truck_has_reachable_serviceable_normal", None)
        if callable(fn):
            try:
                return bool(fn(str(aid)))
            except Exception:
                pass
        for t in env.state.tasks.values():
            if t.status != TaskStatus.PENDING or t.kind != TaskKind.NORMAL:
                continue
            if not self._truck_task_serviceable_or_support_proxy(env, str(aid), t):
                continue
            if self._truck_task_reachable(env, str(aid), t):
                return True
        return False

    def _normal_reachability_snapshot(self, env) -> Tuple[int, Dict[str, bool], bool]:
        step_now = int(getattr(env.state, "step_index", 0))
        if self._normal_reachability_cache_step == step_now and self._normal_reachability_cache is not None:
            return self._normal_reachability_cache
        pending_norm = int(self._pending_normal_task_count(env))
        per_truck: Dict[str, bool] = {}
        for aid, st in env.state.agents.items():
            if st.kind != AgentKind.TRUCK or bool(getattr(st, "crashed", False)):
                continue
            per_truck[str(aid)] = bool(self._truck_has_reachable_normal(env, str(aid)))
        any_reachable = bool(any(per_truck.values()))
        self._normal_reachability_cache_step = step_now
        self._normal_reachability_cache = (pending_norm, per_truck, any_reachable)
        return self._normal_reachability_cache

    def _docked_uav_sortie_chain_ready(self, env, aid: str, task) -> bool:
        if not self._large_map_active(env):
            return True
        st = env.state.agents.get(str(aid), None)
        if st is None or st.kind != AgentKind.UAV or st.follow_target is None:
            return True
        docked_actionable_fn = getattr(env, "_uav_docked_task_actionable_now", None)
        if callable(docked_actionable_fn):
            try:
                if bool(docked_actionable_fn(str(aid), task)):
                    return True
            except Exception:
                pass
        dist_m = float(env._agent_distance_to_task(str(aid), task))
        if not np.isfinite(dist_m):
            return False
        short_cap, long_cap = self._uav_dispatch_distance_caps(env, task)
        one_way_cap = float(long_cap)
        recovery_buf = float(max(getattr(env.cfg, "uav_recovery_distance_buffer_m", 600.0), 0.0))
        if hasattr(env, "_effective_recovery_buffer_for_sortie"):
            try:
                recovery_buf = float(env._effective_recovery_buffer_for_sortie(str(aid), task, launch_reason="rendezvous_safe"))
            except Exception:
                recovery_buf = float(max(getattr(env.cfg, "uav_recovery_distance_buffer_m", 600.0), 0.0))
        mission_chain_m = float(2.0 * max(dist_m, 0.0) + max(recovery_buf, 0.0))
        if self._legacy_sortie_cap_enabled(env):
            sortie_cap = float(max(getattr(env.cfg, "uav_max_sortie_m", one_way_cap), one_way_cap))
            sortie_ok = bool(mission_chain_m <= sortie_cap * 0.92)
        else:
            sortie_ok = True
        return bool(dist_m <= one_way_cap and sortie_ok)

    def _uav_task_effectively_covering_now(self, env, aid: str, task) -> bool:
        if task is None or task.kind != TaskKind.EMERGENCY or task.status != TaskStatus.PENDING:
            return False
        st = env.state.agents.get(str(aid), None)
        if st is None or st.kind != AgentKind.UAV or bool(getattr(st, "crashed", False)):
            return False
        if not bool(self._uav_task_feasible(env, str(aid), task)):
            return False
        if st.follow_target is None:
            return True
        if not bool(self._docked_uav_sortie_chain_ready(env, str(aid), task)):
            return False
        goal_id = None
        if hasattr(env, "_effective_goals"):
            goal_id = env._effective_goals.get(str(aid), None)
        if goal_id is None:
            goal_id = self.state.goals.get(str(aid), None)
        if str(goal_id) != str(task.task_id):
            return False
        if not bool(self._goal_stable_for_takeoff(env, str(aid))):
            return False
        return bool(
            self._uav_task_short_sortie_safe(env, str(aid), task)
            or self._uav_task_clearly_safe_long_range(env, str(aid), task)
        )

    def _uav_emergency_cover_fraction(self, env, task) -> float:
        if task is None or task.kind != TaskKind.EMERGENCY:
            return 0.0
        all_uav_ids = [
            str(uid)
            for uid, ag in env.state.agents.items()
            if ag.kind == AgentKind.UAV and (not bool(getattr(ag, "crashed", False)))
        ]
        if not all_uav_ids:
            return 0.0
        max_eval_dist_m = float(max(getattr(env.cfg, "hrl_uav_cover_eval_max_distance_m", 2800.0), 0.0))
        uav_ids = []
        for uid in all_uav_ids:
            d = float(env._agent_distance_to_task(str(uid), task))
            if (not np.isfinite(d)):
                continue
            if max_eval_dist_m > 0.0 and d > max_eval_dist_m:
                continue
            uav_ids.append(str(uid))
        if not uav_ids:
            return 0.0
        feasible = sum(1 for uid in uav_ids if self._uav_task_effectively_covering_now(env, str(uid), task))
        return float(np.clip(float(feasible) / max(len(uav_ids), 1), 0.0, 1.0))

    def _truck_emergency_relief_allowed(self, env, aid: str, task) -> bool:
        if task is None or task.kind != TaskKind.EMERGENCY:
            return True
        chain = self._support_bound_chain_info_for_truck(env, str(aid))
        if (
            bool(getattr(env.cfg, "erc_tc_support_required_enabled", False))
            and chain is not None
            and str(chain.get("task_id", "")) == str(getattr(task, "task_id", ""))
        ):
            return True
        self.normal_protection_candidate_count_total = int(self.normal_protection_candidate_count_total) + 1
        if bool(getattr(env.cfg, "erc_ablate_normal_protection", False)):
            self.normal_protection_blocked_by_ablation_count_total = int(self.normal_protection_blocked_by_ablation_count_total) + 1
            return True
        self.normal_protection_applied_count_total = int(self.normal_protection_applied_count_total) + 1
        pending_norm, norm_reach_by_truck, any_reachable_normal = self._normal_reachability_snapshot(env)
        block_thr = int(max(getattr(env.cfg, "hrl_truck_emergency_min_pending_normal_to_block", 1), 0))
        if int(pending_norm) < block_thr:
            return True

        hard_recovery = False
        hard_recovery_fn = getattr(env, "_has_hard_recovery_uav", None)
        if callable(hard_recovery_fn):
            hard_recovery = bool(hard_recovery_fn())
        if hard_recovery:
            return True

        has_reachable_normal = bool(norm_reach_by_truck.get(str(aid), True))
        no_normal_mode_enabled = bool(getattr(env.cfg, "hrl_truck_emergency_support_when_no_normal_enabled", True))
        # If this truck has no reachable NORMAL while normal backlog exists,
        # allow dedicated emergency-support behavior.
        if no_normal_mode_enabled and (not has_reachable_normal):
            note_fn = getattr(env, "_note_truck_emergency_relief_gate", None)
            if callable(note_fn):
                note_fn(str(aid), task, blocked_by_normal_guard=False, override=True)
            return True

        uav_cover = float(self._uav_emergency_cover_fraction(env, task))
        urgency = float(self._norm_deadline_urgency(task, int(env.state.step_index)))
        force_cover_thr = float(np.clip(getattr(env.cfg, "hrl_truck_emergency_force_relief_uav_cover_threshold", 0.35), 0.0, 1.0))
        force_urg_thr = float(np.clip(getattr(env.cfg, "hrl_truck_emergency_force_relief_urgency_threshold", 0.72), 0.0, 1.0))

        # ERC-specific late-stage relief: when normal backlog is already low,
        # allow trucks to more aggressively absorb uncovered emergency tasks.
        if bool(self.use_event_trigger) and int(pending_norm) <= 2:
            force_cover_thr = float(np.clip(max(force_cover_thr, force_cover_thr + 0.12), 0.0, 1.0))

        high_pressure = bool(self._task_high_pressure(env, task))
        island_serviceability = float(self._island_serviceability_ratio(env))
        high_pressure_override = bool(
            high_pressure
            and (
                (uav_cover < force_cover_thr)
                or (self._is_island_task(env, task) and island_serviceability < 0.45)
            )
        )

        emergency_relief_override = bool((uav_cover < force_cover_thr) or (urgency >= force_urg_thr) or high_pressure_override)
        if emergency_relief_override:
            note_fn = getattr(env, "_note_truck_emergency_relief_gate", None)
            if callable(note_fn):
                note_fn(str(aid), task, blocked_by_normal_guard=False, override=True)
            return True

        # Strict mode: if normal backlog is still reachable by this truck, keep
        # emergency gating tighter to protect normal throughput.
        if bool(any_reachable_normal) and bool(has_reachable_normal):
            strict_cover_thr = float(
                np.clip(getattr(env.cfg, "hrl_truck_emergency_cover_threshold_when_normal_reachable", 0.30), 0.0, 1.0)
            )
            support_gain_min = float(
                np.clip(getattr(env.cfg, "hrl_truck_support_gain_min_when_normal_reachable", 0.45), 0.0, 1.0)
            )
            support_gain = float(np.clip(self._truck_support_serviceability_gain(env, str(aid), task), 0.0, 1.0))
            island_supportable = bool(self._is_island_task(env, task) and support_gain >= support_gain_min)
            urgent_low_cover = bool((uav_cover < strict_cover_thr) and (urgency >= 0.60))
            allowed = bool(island_supportable or urgent_low_cover)
        else:
            if self._is_island_task(env, task):
                support_score = float(self._truck_forward_support_score(env, str(aid), task))
                if support_score > 0.05:
                    return True
            cover_thr = float(np.clip(getattr(env.cfg, "hrl_truck_emergency_relief_uav_cover_threshold", 0.5), 0.0, 1.0))
            allowed = bool(uav_cover < cover_thr)

        if not allowed:
            note_fn = getattr(env, "_note_truck_emergency_relief_gate", None)
            if callable(note_fn):
                note_fn(str(aid), task, blocked_by_normal_guard=True, override=False)
        return bool(allowed)

    def _truck_supportworthy_emergency_task(self, env, aid: str, task) -> bool:
        if task is None or task.kind != TaskKind.EMERGENCY:
            return False
        if not bool(self._truck_emergency_relief_allowed(env, str(aid), task)):
            return False

        gain_info = self._support_anchor_service_gain(env, str(aid), task)
        support_gain = float(np.clip(float(gain_info.get("gain_score", 0.0)), 0.0, 1.0))
        if self._support_backoff_active(env, str(aid), task, gain_info=gain_info):
            self.support_no_gain_backoff_block_count_total = int(self.support_no_gain_backoff_block_count_total) + 1
            return False
        if self._support_soft_clamp_blocks_task(env, str(aid), task, gain_info=gain_info):
            if not self._support_escape_hatch_allows(env, str(aid), task, gain_info=gain_info):
                return False

        # Patch-2: support must bind short-horizon delivery (time-critical first).
        if bool(getattr(env.cfg, "hrl_support_requires_timecritical_binding", True)):
            bind_info = self._support_bound_delivery_info(env, str(aid), task, gain_info=gain_info)
            allow_bulk = bool(getattr(env.cfg, "hrl_support_fallback_allow_bulk_binding", False))
            if allow_bulk:
                if float(bind_info.get("bound_any", 0.0)) <= 0.0:
                    return False
            else:
                if float(bind_info.get("bound_timecritical", 0.0)) <= 0.0:
                    return False
            if not bool(self._support_binding_is_strong_enough(env, task, bind_info, gain_info=gain_info)):
                return False

        urgency = float(np.clip(self._norm_deadline_urgency(task, int(env.state.step_index)), 0.0, 1.0))
        min_gain = float(np.clip(getattr(env.cfg, "hrl_truck_no_normal_support_min_gain", 0.20), 0.0, 1.0))
        urgency_floor = float(np.clip(getattr(env.cfg, "hrl_truck_no_normal_support_urgency_floor", 0.55), 0.0, 1.0))
        return bool(
            self._is_island_task(env, task)
            or support_gain >= min_gain
            or urgency >= urgency_floor
            or self._support_escape_hatch_allows(env, str(aid), task, gain_info=gain_info)
        )

    def _ordered_agents(self, env) -> List[str]:
        # Deterministic assignment order ablation.
        if self.assignment_order == "truck_first":
            return sorted(
                env.state.agents.keys(),
                key=lambda aid: 0 if env.state.agents[aid].kind == AgentKind.TRUCK else 1,
            )
        return sorted(
            env.state.agents.keys(),
            key=lambda aid: 0 if env.state.agents[aid].kind == AgentKind.UAV else 1,
        )

    def _agent_xy(self, env, aid: str) -> Tuple[float, float]:
        st = env.state.agents[aid]
        if st.pos_xy is not None:
            return float(st.pos_xy[0]), float(st.pos_xy[1])
        return env._node_xy(int(st.node or 0))

    def _nearest_truck(self, env, aid: str) -> Tuple[Optional[str], float]:
        ax, ay = self._agent_xy(env, aid)
        best_id: Optional[str] = None
        best_d = float("inf")
        for tid, ts in env.state.agents.items():
            if ts.kind != AgentKind.TRUCK:
                continue
            tx, ty = self._agent_xy(env, str(tid))
            d = float(((ax - tx) ** 2 + (ay - ty) ** 2) ** 0.5)
            if d < best_d:
                best_d = d
                best_id = str(tid)
        return best_id, best_d

    def _required_rth_battery(self, env, aid: str, dist_to_truck: float) -> float:
        """Use the environment's canonical leg estimator for RTH prediction."""
        st = env.state.agents[aid]
        origin = self._agent_xy(env, str(aid))
        truck_id, _ = self._nearest_truck_from_xy(env, origin)
        destination = origin
        if truck_id is not None:
            destination = self._agent_xy(env, str(truck_id))
        if hasattr(env, "_uav_energy_cost_fraction"):
            destination = getattr(env, "_uav_extended_destination", lambda o, t, d: t)(
                origin, destination, float(max(dist_to_truck, 0.0))
            )
            return float(
                max(
                    env._uav_energy_cost_fraction(
                        str(aid),
                        float(max(dist_to_truck, 0.0)),
                        origin,
                        destination=destination,
                        payload_override=float(max(getattr(st, "payload_kg_current", 0.0), 0.0)),
                    ),
                    0.0,
                )
            )
        return 0.0

    def _nearest_truck_from_xy(self, env, xy: Tuple[float, float]) -> Tuple[Optional[str], float]:
        x, y = float(xy[0]), float(xy[1])
        best_id: Optional[str] = None
        best_d = float("inf")
        for tid, ts in env.state.agents.items():
            if ts.kind != AgentKind.TRUCK:
                continue
            tx, ty = self._agent_xy(env, str(tid))
            d = float(((x - tx) ** 2 + (y - ty) ** 2) ** 0.5)
            if d < best_d:
                best_d = d
                best_id = str(tid)
        return best_id, best_d

    def build_candidates_for_agent(
        self,
        env,
        aid: str,
        used_tasks: Optional[set] = None,
        enable_rth_mask: Optional[bool] = None,
    ) -> Tuple[List[str], List[List[float]], List[int]]:
        used = used_tasks if used_tasks is not None else set()
        enable_mask = bool(
            getattr(env.cfg, "enable_rth_mask", True) if enable_rth_mask is None else enable_rth_mask
        )
        st = env.state.agents[aid]
        dist_norm_m = self._distance_norm_m(env)

        tids: List[str] = []
        feats: List[List[float]] = []
        task_nodes: List[int] = []

        cands = self._candidate_tasks(env, aid)
        short_sortie_safe_ids: set = set()

        if st.kind == AgentKind.UAV and (not bool(st.crashed)) and self._uav_stage_blocks_task_goal(env, aid):
            near_tid, near_dist = self._nearest_truck(env, aid)
            if near_tid is not None and np.isfinite(near_dist):
                truck_state = env.state.agents.get(str(near_tid), None)
                truck_node = int(getattr(truck_state, "node", 0) or 0)
                truck_feat = [float(near_dist / dist_norm_m), 0.0]
                return [near_tid], [truck_feat], [truck_node]
            return [], [], []
        for t in cands:
            tid = str(t.task_id)
            if tid in used:
                continue
            dist_norm = float(env._agent_distance_to_task(aid, t) / dist_norm_m)
            emer = 1.0 if str(t.kind.value) == "emergency" else 0.0
            tids.append(tid)
            feats.append([dist_norm, emer])
            task_nodes.append(int(t.demand_node))
            if st.kind == AgentKind.UAV and self._uav_task_short_sortie_safe(env, aid, t):
                short_sortie_safe_ids.add(tid)

        # Keep candidate space aligned with current system behavior:
        # UAV may consider nearest-truck virtual target for recovery.
        if st.kind == AgentKind.UAV and (not bool(st.crashed)):
            near_tid, near_dist = self._nearest_truck(env, aid)
            if near_tid is not None and np.isfinite(near_dist):
                truck_state = env.state.agents.get(str(near_tid), None)
                truck_node = int(getattr(truck_state, "node", 0) or 0)

                force_rth = bool(self._uav_needs_recovery(env, aid))
                allow_truck_candidate = bool(force_rth or len(tids) == 0)
                if allow_truck_candidate:
                    truck_feat = [float(near_dist / dist_norm_m), 0.0]
                    tids = [near_tid] + tids
                    feats = [truck_feat] + feats
                    task_nodes = [truck_node] + task_nodes
                if enable_mask and force_rth:
                    # Preventive recovery is preferred, but allow short safe sorties
                    # once the current goal has remained stable long enough.
                    allow_short_sortie = bool(
                        self._goal_stable_for_takeoff(env, aid) and len(short_sortie_safe_ids) > 0
                    )
                    if allow_short_sortie:
                        keep_ids = [str(near_tid)] + [tid for tid in tids if tid in short_sortie_safe_ids]
                        keep_set = set(keep_ids)
                        keep_tids: List[str] = []
                        keep_feats: List[List[float]] = []
                        keep_nodes: List[int] = []
                        for tid, feat, tnode in zip(tids, feats, task_nodes):
                            if tid in keep_set:
                                keep_tids.append(tid)
                                keep_feats.append(feat)
                                keep_nodes.append(tnode)
                        tids, feats, task_nodes = keep_tids, keep_feats, keep_nodes
                    else:
                        truck_feat = [float(near_dist / dist_norm_m), 0.0]
                        tids = [near_tid]
                        feats = [truck_feat]
                        task_nodes = [truck_node]

        return tids, feats, task_nodes

    # --------------------------
    # Deterministic scoring logic
    # --------------------------
    @staticmethod
    def _norm_deadline_urgency(task, step_index: int) -> float:
        # Urgency rises as deadline slack shrinks: 1 / (1 + nonnegative slack).
        slack = float(max(int(task.deadline_step) - int(step_index), 0))
        return float(1.0 / (1.0 + slack))

    @staticmethod
    def _norm_eta_score(dist_m: float, speed_mps: float) -> float:
        # Smaller ETA -> larger score in (0,1].
        eta_s = float(max(dist_m, 0.0)) / float(max(speed_mps, 1e-6))
        return float(1.0 / (1.0 + eta_s))

    def _task_risk(self, env, task) -> float:
        # Risk proxy from local weather + topology slope around the task node.
        node = env.topology.nodes[int(task.demand_node)]
        weather = env.hazards.node_weather(int(task.demand_node))
        rain_norm = float(weather.rain / max(getattr(env.cfg, "base_rainfall_mmh", 1.0), 1e-6))
        wind_norm = float(weather.wind / max(getattr(env.cfg, "base_wind_mps", 1.0), 1e-6))
        quake = float(weather.quake)
        slope = float(getattr(node, "slope_norm", 0.0))
        return float(np.clip(0.30 * rain_norm + 0.25 * wind_norm + 0.30 * quake + 0.15 * slope, 0.0, 3.0))

    @staticmethod
    def _demand_score(task, env) -> float:
        # Demand term uses remaining demand normalized by configured demand scale.
        base = (
            float(getattr(env.cfg, "task_demand_emergency_units", 1.0))
            if task.kind == TaskKind.EMERGENCY
            else float(getattr(env.cfg, "task_demand_normal_units", 1.0))
        )
        return float(np.clip(float(getattr(task, "demand_left", 0.0)) / max(base, 1e-6), 0.0, 2.0))

    def _keep_goal_bonus(self, aid: str, candidate_id: str) -> float:
        # Inertia term to reduce unnecessary oscillations.
        if not self.use_keep_goal_bonus:
            return 0.0
        prev = self.state.goals.get(str(aid), None)
        if prev is None or str(prev) != str(candidate_id):
            return 0.0
        # Stronger keep-goal behavior than previous version to improve stability.
        return float(self.weights.keep_goal_bonus * 1.50)

    def _comm_degraded(self, env, aid: str) -> bool:
        comm_blocked = getattr(env, "comm_blocked", {})
        if isinstance(comm_blocked, dict):
            # Per-UAV comm degradation: do not globally degrade all UAVs
            # just because another agent is blocked.
            if bool(comm_blocked.get(str(aid), False)):
                return True
            if bool(comm_blocked.get("__all__", False)) or bool(comm_blocked.get("global", False)):
                return True
            return False
        return bool(comm_blocked)

    def _uav_long_range_block_m(self, env) -> float:
        bind_r = float(max(getattr(env.cfg, "uav_bind_radius_m", 50.0), 1.0))
        monitor_r = float(max(getattr(env.cfg, "uav_monitor_radius_m", bind_r), bind_r))
        return float(max(bind_r * 2.0, monitor_r))

    def _uav_comm_dispatch_limit_m(self, env, aid: str, task=None) -> float:
        limit = float(self._uav_long_range_block_m(env))
        if task is None or task.kind != TaskKind.EMERGENCY or task.status != TaskStatus.PENDING:
            return float(limit)

        near_dispatch = float(max(getattr(env.cfg, "hrl_uav_near_depot_direct_dispatch_radius_m", 700.0), 0.0))
        short_sortie = float(max(getattr(env.cfg, "uav_short_sortie_max_distance_m", 1200.0), 0.0))
        hp_cap = float(max(getattr(env.cfg, "uav_high_pressure_rendezvous_max_nearest_truck_m", 2200.0), 0.0))
        is_hp = bool(self._task_high_pressure(env, task))

        if is_hp:
            limit = max(limit, min(hp_cap, max(short_sortie, 1200.0)))
        else:
            limit = max(limit, near_dispatch, min(max(short_sortie, 1.0), 900.0))

        st = env.state.agents.get(str(aid), None)
        if st is not None and st.kind == AgentKind.UAV and st.follow_target is not None and is_hp:
            limit = max(limit, min(hp_cap, 1.15 * max(short_sortie, near_dispatch, 1.0)))

        return float(limit)

    def _uav_task_hard_risk_blocked(self, env, task) -> bool:
        if task.kind != TaskKind.EMERGENCY:
            return True
        w = env.hazards.node_weather(int(task.demand_node))
        if self.max_uav_wind_mps is not None and float(w.wind) > float(self.max_uav_wind_mps):
            return True
        if self.max_uav_rainfall_mmh is not None and float(w.rain) > float(self.max_uav_rainfall_mmh):
            return True
        if self.max_uav_node_risk is not None and self._task_risk(env, task) > float(self.max_uav_node_risk):
            return True
        return False

    def _uav_dispatch_distance_caps(self, env, task) -> Tuple[float, float]:
        short_cap = float(max(self.short_sortie_max_distance_m, 1.0))
        cfg_short = float(max(getattr(env.cfg, "uav_short_sortie_max_distance_m", short_cap), 1.0))
        short_cap = float(max(short_cap, cfg_short))
        long_cap = float(max(short_cap * 1.60, short_cap))

        if task is None or task.kind != TaskKind.EMERGENCY or task.status != TaskStatus.PENDING:
            return float(short_cap), float(long_cap)

        map_size = float(max(getattr(env.cfg, "map_size_m", 0.0), 0.0))
        if bool(self._task_high_pressure(env, task)) or bool(self._is_island_task(env, task)):
            if map_size >= 12000.0:
                if self._legacy_sortie_cap_enabled(env):
                    sortie_cap = float(max(getattr(env.cfg, "uav_max_sortie_m", short_cap), short_cap))
                    short_cap = float(max(short_cap, min(sortie_cap * 0.45, 3000.0)))
                    long_cap = float(max(long_cap, min(sortie_cap * 0.60, 3600.0)))
                else:
                    short_cap = float(max(short_cap, min(map_size * 0.30, 4200.0)))
                    long_cap = float(max(long_cap, min(map_size * 0.38, 5200.0)))
            elif map_size >= 5000.0:
                if self._legacy_sortie_cap_enabled(env):
                    sortie_cap = float(max(getattr(env.cfg, "uav_max_sortie_m", short_cap), short_cap))
                    short_cap = float(max(short_cap, min(sortie_cap * 0.35, 1800.0)))
                    long_cap = float(max(long_cap, min(sortie_cap * 0.50, 2400.0)))
                else:
                    short_cap = float(max(short_cap, min(map_size * 0.28, 3000.0)))
                    long_cap = float(max(long_cap, min(map_size * 0.36, 4200.0)))
        return float(short_cap), float(long_cap)

    def _uav_task_short_sortie_safe(self, env, aid: str, task) -> bool:
        if task is None or task.kind != TaskKind.EMERGENCY or task.status != TaskStatus.PENDING:
            return False
        dist_m = float(env._agent_distance_to_task(aid, task))
        short_cap, _ = self._uav_dispatch_distance_caps(env, task)
        if (not np.isfinite(dist_m)) or dist_m > float(short_cap):
            return False
        batt = float(getattr(env.state.agents[aid], "battery", 0.0))
        if batt < float(self.short_sortie_min_battery):
            return False
        # Short-sortie must still satisfy full mission feasibility.
        if not self._uav_task_feasible(env, aid, task):
            return False

        # Stricter-than-default weather/risk checks for short sortie dispatch.
        w = env.hazards.node_weather(int(task.demand_node))
        if self.max_uav_wind_mps is not None and float(w.wind) > float(self.max_uav_wind_mps) * 0.90:
            return False
        if self.max_uav_rainfall_mmh is not None and float(w.rain) > float(self.max_uav_rainfall_mmh) * 0.90:
            return False
        if self.max_uav_node_risk is not None and self._task_risk(env, task) > float(self.max_uav_node_risk) * 0.90:
            return False
        return True

    def _uav_task_clearly_safe_long_range(self, env, aid: str, task) -> bool:
        if task is None or task.kind != TaskKind.EMERGENCY or task.status != TaskStatus.PENDING:
            return False
        dist_m = float(env._agent_distance_to_task(aid, task))
        if not np.isfinite(dist_m):
            return False
        short_cap, long_cap = self._uav_dispatch_distance_caps(env, task)
        if dist_m <= float(short_cap):
            return False
        if dist_m > float(long_cap):
            return False
        if not self._uav_task_feasible(env, aid, task):
            return False
        batt = float(max(getattr(env.state.agents[aid], "battery", 0.0), 1e-6))
        req = float(self._uav_task_required_battery(env, aid, task))
        if (not np.isfinite(req)) or batt < float(req * 1.15):
            return False
        margin = float((batt - req) / batt)
        if margin < 0.20:
            return False
        w = env.hazards.node_weather(int(task.demand_node))
        if self.max_uav_wind_mps is not None and float(w.wind) > float(self.max_uav_wind_mps) * 0.80:
            return False
        if self.max_uav_rainfall_mmh is not None and float(w.rain) > float(self.max_uav_rainfall_mmh) * 0.80:
            return False
        if self.max_uav_node_risk is not None and self._task_risk(env, task) > float(self.max_uav_node_risk) * 0.80:
            return False
        return True

    def _uav_task_required_battery(self, env, aid: str, task) -> float:
        # Conservative full mission requirement:
        # battery >= (go_to_task + service_buffer + return_to_nearest_truck) * safety_factor
        d_go = float(env._agent_distance_to_task(aid, task))
        if not np.isfinite(d_go):
            return float("inf")
        n = env.topology.nodes[int(task.demand_node)]
        _, d_back = self._nearest_truck_from_xy(env, (float(n.x), float(n.y)))
        if not np.isfinite(d_back):
            return float("inf")
        recovery_buf = float(max(getattr(env.cfg, "uav_recovery_distance_buffer_m", 600.0), 0.0))
        if hasattr(env, "_uav_sortie_energy_requirement"):
            mission_batt = float(
                env._uav_sortie_energy_requirement(
                    str(aid),
                    task,
                    recovery_buffer_m=recovery_buf,
                )
            )
        else:
            mission_batt = self._required_rth_battery(env, aid, d_go + float(d_back) + recovery_buf)
        mission_batt = float(mission_batt + self.service_battery_buffer)
        return float(mission_batt * self._rth_safety_factor(env))

    def _uav_task_margin(self, env, aid: str, task) -> float:
        req = self._uav_task_required_battery(env, aid, task)
        batt = float(max(env.state.agents[aid].battery, 1e-6))
        if not np.isfinite(req):
            return -1.0
        return float((batt - req) / batt)

    def _uav_task_feasible(self, env, aid: str, task) -> bool:
        step_now = int(getattr(env.state, "step_index", 0))
        if step_now != int(self._uav_task_feasible_cache_step):
            self._uav_task_feasible_cache_step = int(step_now)
            self._uav_task_feasible_cache.clear()
        task_id = str(getattr(task, "task_id", ""))
        cache_key = (
            int(step_now),
            str(aid),
            str(task_id),
            int(self._blocked_edge_version(env)),
            tuple(self._anchor_state_signature(env)),
        )
        self._maybe_clear_uav_reject_cache(env, str(aid))
        if self._uav_reject_cache_blocked(env, str(aid), str(task_id)):
            self._uav_task_feasible_cache[cache_key] = False
            return False
        if cache_key in self._uav_task_feasible_cache:
            return bool(self._uav_task_feasible_cache[cache_key])

        self.uav_recovery_feasibility_eval_count_total = int(self.uav_recovery_feasibility_eval_count_total) + 1

        note_reject = getattr(env, "_note_uav_task_reject", None)
        if (not self._task_planner_active(task)) or task.kind != TaskKind.EMERGENCY:
            self._uav_task_feasible_cache[cache_key] = False
            return False
        if self._uav_task_hard_risk_blocked(env, task):
            if callable(note_reject):
                note_reject(str(aid), task, "corridor")
            self._uav_task_feasible_cache[cache_key] = False
            return False
        dist_m = float(env._agent_distance_to_task(aid, task))
        if not np.isfinite(dist_m):
            if callable(note_reject):
                note_reject(str(aid), task, "corridor")
            self._uav_task_feasible_cache[cache_key] = False
            return False
        st = env.state.agents.get(str(aid), None)

        # Communication degradation: keep conservative behavior, but use task-aware
        # distance limit to avoid over-blocking close/high-pressure emergency sorties.
        comm_limit = float(self._uav_comm_dispatch_limit_m(env, aid, task))
        if self._comm_degraded(env, aid) and dist_m > comm_limit:
            if callable(note_reject):
                note_reject(str(aid), task, "comm_block")
            self._record_uav_reject_cache(env, str(aid), str(task_id), "comm_block")
            self._uav_task_feasible_cache[cache_key] = False
            return False

        if st is None or st.kind != AgentKind.UAV:
            if callable(note_reject):
                note_reject(str(aid), task, "corridor")
            self._record_uav_reject_cache(env, str(aid), str(task_id), "corridor")
            self._uav_task_feasible_cache[cache_key] = False
            return False

        req = self._uav_task_required_battery(env, aid, task)
        batt = float(env.state.agents[aid].battery)

        launch_reason = ""
        launch_ok = False
        if st.follow_target is not None and hasattr(env, "_uav_launch_gate_check"):
            prev_effective = env._effective_goals.get(str(aid), None) if hasattr(env, "_effective_goals") else None
            try:
                if hasattr(env, "_effective_goals"):
                    env._effective_goals[str(aid)] = str(task.task_id)
                launch_ok, launch_reason, _ = env._uav_launch_gate_check(str(aid), task=task, count_reject=False)
            except Exception:
                launch_ok, launch_reason = False, ""
            finally:
                if hasattr(env, "_effective_goals"):
                    env._effective_goals[str(aid)] = prev_effective

        battery_ok = bool(np.isfinite(req) and batt >= req)
        # Keep planner feasibility aligned with env launch gate when UAV is docked;
        # this avoids selecting goals that later fail on recovery-margin launch check.
        if st.follow_target is not None and bool(getattr(env.cfg, "hrl_uav_docked_require_launch_gate_strict", True)):
            battery_ok = bool(battery_ok and launch_ok)
        corridor_ok = bool(launch_ok and (str(launch_reason) == "direct_safe" or str(launch_reason).startswith("rendezvous_safe")))
        if not (battery_ok or corridor_ok):
            if callable(note_reject):
                rr = str(launch_reason).strip().lower()
                if rr in {"below_launch_min", "not_loaded", "insufficient_recovery_margin", "rendezvous_launch_disabled", "no_truck_for_return"}:
                    note_reject(str(aid), task, rr)
                else:
                    note_reject(str(aid), task, "recovery_margin")
            self._record_uav_reject_cache(env, str(aid), str(task_id), str(launch_reason) if str(launch_reason) else "insufficient_recovery_margin")
            self._uav_task_feasible_cache[cache_key] = False
            return False

        # Horizon feasibility: avoid assigning missions that cannot reasonably
        # finish before deadline/episode end.
        max_steps = int(max(getattr(env.cfg, "max_steps", 0), 0))
        rem_episode = int(max(max_steps - step_now, 0))
        rem_deadline = int(max(int(getattr(task, "deadline_step", max_steps)) - step_now, 0))
        rem_steps = int(min(rem_episode, rem_deadline)) if rem_deadline > 0 else rem_episode

        node = env.topology.nodes[int(task.demand_node)]
        _, d_back = self._nearest_truck_from_xy(env, (float(node.x), float(node.y)))
        if not np.isfinite(d_back):
            if callable(note_reject):
                note_reject(str(aid), task, "corridor")
            self._uav_task_feasible_cache[cache_key] = False
            return False
        recovery_buf = float(max(getattr(env.cfg, "uav_recovery_distance_buffer_m", 600.0), 0.0))
        if hasattr(env, "_effective_recovery_buffer_for_sortie"):
            try:
                recovery_buf = float(env._effective_recovery_buffer_for_sortie(str(aid), task, launch_reason=str(launch_reason)))
            except Exception:
                recovery_buf = float(max(getattr(env.cfg, "uav_recovery_distance_buffer_m", 600.0), 0.0))

        if corridor_ok and str(launch_reason).startswith("rendezvous_safe"):
            bind_window = float(max(getattr(env.cfg, "uav_bind_radius_m", 170.0), 1.0))
            rendez_dist = float(max(0.75 * recovery_buf, bind_window))
            decision_interval = int(max(getattr(env.cfg, "decision_interval", 5), 1))
            dt = float(max(getattr(env.cfg, "dt_seconds", 1.0), 1e-6))
            truck_speed = float(max(getattr(env.cfg, "truck_speed_mps", 0.0), 0.0))
            truck_drift = float(truck_speed * dt * decision_interval)
            mission_dist = float(max(dist_m + rendez_dist + truck_drift, 0.0))
        else:
            mission_dist = float(max(dist_m + d_back + recovery_buf, 0.0))

        util = float(np.clip(getattr(env.cfg, "uav_launch_speed_utilization", 0.70), 0.1, 1.0))
        v_ref = float(max(getattr(env.cfg, "uav_max_speed_mps", 1.0) * util, 1e-6))
        dt = float(max(getattr(env.cfg, "dt_seconds", 1.0), 1e-6))
        est_steps = int(np.ceil(mission_dist / max(v_ref * dt, 1e-6)))
        horizon_buf = int(max(getattr(env.cfg, "uav_launch_min_horizon_buffer_steps", 4), 0))
        service_buf = int(max(getattr(env.cfg, "uav_service_horizon_steps", 1), 0))
        feasible_horizon = bool(rem_steps >= int(est_steps + horizon_buf + service_buf))
        if (not feasible_horizon) and callable(note_reject):
            note_reject(str(aid), task, "horizon")
        if not feasible_horizon:
            self._record_uav_reject_cache(env, str(aid), str(task_id), "energy_infeasible")

        out = bool(feasible_horizon)
        self._uav_task_feasible_cache[cache_key] = bool(out)
        return bool(out)

    def _uav_needs_recovery(self, env, aid: str) -> bool:
        st = env.state.agents[aid]
        if bool(st.crashed):
            return False
        near_tid, near_dist = self._nearest_truck(env, aid)
        if near_tid is None or not np.isfinite(near_dist):
            return False
        batt = float(getattr(st, "battery", 0.0))
        if (hasattr(env, "_uav_loaded") and (not bool(env._uav_loaded(aid)))) or float(getattr(st, "cargo", 0.0)) <= 0.0:
            # No payload means sortie is not service-capable yet: prefer docking/load.
            return True
        # Preventive recovery before battery enters emergency region.
        early_recovery_floor = float(
            np.clip(float(self.short_sortie_min_battery) + 0.08, 0.0, 0.95)
        )
        if batt <= early_recovery_floor:
            return True
        recovery_buf = float(max(getattr(env.cfg, "uav_recovery_distance_buffer_m", 600.0), 0.0))
        req = self._required_rth_battery(env, aid, float(near_dist) + recovery_buf)
        req = float((req + self.service_battery_buffer) * self._rth_safety_factor(env))
        req = float(req * self.recovery_trigger_factor)
        if batt < req:
            return True
        # Under communication degradation, use extra conservative recovery trigger.
        if self._comm_degraded(env, aid) and batt < float(req * 1.10):
            return True
        return False

    def _is_island_task(self, env, task) -> bool:
        if task is None or task.kind != TaskKind.EMERGENCY:
            return False
        fn = getattr(env, "_current_island_emergency_task_ids", None)
        if not callable(fn):
            return False
        try:
            island_ids = set(fn())
        except Exception:
            return False
        return bool(str(task.task_id) in island_ids)

    def _uav_has_feasible_island_task(self, env, aid: str, island_ids: set) -> bool:
        if not island_ids:
            return False
        for tid in sorted(island_ids):
            task = env.state.tasks.get(str(tid), None)
            if task is None or task.status != TaskStatus.PENDING or task.kind != TaskKind.EMERGENCY:
                continue
            if self._uav_task_feasible(env, str(aid), task):
                return True
        return False

    def _map_update_active(self) -> bool:
        return bool(self._last_refresh_flags.get("map_update", False))

    def _truck_stage_blocks_task_goal(self, env, aid: str) -> bool:
        st = env.state.agents.get(str(aid), None)
        if st is None or st.kind != AgentKind.TRUCK or bool(st.crashed):
            return True
        if int(getattr(st, "truck_replenish_timer", 0)) > 0:
            return True
        if hasattr(env, "_truck_requires_depot") and bool(env._truck_requires_depot(str(aid))):
            return True
        if st.node is None:
            return False
        if not hasattr(env, "_decision_neighbors"):
            return False
        neighbors = [int(x) for x in env._decision_neighbors(int(st.node))]
        if not neighbors:
            return False
        if bool(getattr(env.cfg, "truck_support_uav_recovery_enabled", True)) and hasattr(env, "_truck_recovery_support_target"):
            if env._truck_recovery_support_target(str(aid), neighbors) is not None:
                hard_recovery = True
                hard_fn = getattr(env, "_has_hard_recovery_uav", None)
                if callable(hard_fn):
                    hard_recovery = bool(hard_fn())
                if hard_recovery:
                    # Do not hard-lock truck support when reachable NORMAL tasks exist.
                    # This keeps truck-side delivery throughput from collapsing.
                    has_reachable_normal = False
                    for task in env.state.tasks.values():
                        if task.status != TaskStatus.PENDING or task.kind != TaskKind.NORMAL:
                            continue
                        if not self._truck_task_serviceable_or_support_proxy(env, str(aid), task):
                            continue
                        if self._truck_task_reachable(env, str(aid), task):
                            has_reachable_normal = True
                            break
                    if not has_reachable_normal:
                        return True
        # Island forward-support is now treated as a soft planner preference,
        # not a hard stage lock; otherwise truck task execution can be starved
        # when island tasks are present but currently not convertible.
        return False

    def _truck_task_distance(self, env, aid: str, task) -> float:
        self._ensure_step_eval_caches(env)
        task_id = str(getattr(task, "task_id", "<none>"))
        cache_key = (str(aid), task_id)
        cached = self._truck_task_distance_cache.get(cache_key, None)
        if cached is not None:
            return float(cached)
        st = env.state.agents.get(str(aid), None)
        if st is None or st.kind != AgentKind.TRUCK:
            out = float("inf")
        elif st.node is None:
            out = float(env._agent_distance_to_task(str(aid), task))
        elif hasattr(env, "_decision_shortest_path_distance"):
            out = float(env._decision_shortest_path_distance(int(st.node), int(task.demand_node)))
        else:
            out = float(env._agent_distance_to_task(str(aid), task))
        self._truck_task_distance_cache[cache_key] = float(out)
        return float(out)

    def _truck_task_reachable(self, env, aid: str, task) -> bool:
        d = float(self._truck_task_distance(env, str(aid), task))
        return bool(np.isfinite(d))


    def _truck_task_direct_serviceable(self, env, aid: str, task) -> bool:
        if task is None:
            return False
        if not hasattr(env, "is_task_serviceable_by_agent"):
            return True
        return bool(env.is_task_serviceable_by_agent(str(aid), task))

    def _truck_task_serviceable_or_support_proxy(self, env, aid: str, task) -> bool:
        if task is None:
            return False
        chain = self._support_bound_chain_info_for_truck(env, str(aid))
        if (
            bool(getattr(env.cfg, "erc_tc_support_required_enabled", False))
            and chain is not None
            and str(chain.get("task_id", "")) == str(getattr(task, "task_id", ""))
        ):
            return True
        self._ensure_step_eval_caches(env)
        cache_key = (str(aid), str(getattr(task, "task_id", "<none>")))
        cached = self._truck_task_serviceable_cache.get(cache_key, None)
        if cached is not None:
            return bool(cached)
        if (
            bool(getattr(env.cfg, "unreachable_bulk_watchlist_enabled", False))
            and task.kind == TaskKind.NORMAL
            and (not self._truck_task_reachable(env, str(aid), task))
            and self._large_map_active(env)
        ):
            out = False
        elif bool(self._truck_task_direct_serviceable(env, str(aid), task)):
            out = True
        else:
            out = bool(task.kind == TaskKind.EMERGENCY and self._truck_emergency_support_candidate(env, str(aid), task))
        self._truck_task_serviceable_cache[cache_key] = bool(out)
        return bool(out)

    def _truck_nearest_reachable_pending_distance(self, env, aid: str) -> float:
        self._ensure_step_eval_caches(env)
        aid_s = str(aid)
        cached = self._truck_nearest_reachable_cache.get(aid_s, None)
        if cached is not None:
            return float(cached)
        best = float("inf")
        for task in env.state.tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
            if not self._truck_task_serviceable_or_support_proxy(env, aid_s, task):
                continue
            d = float(self._truck_task_distance(env, aid_s, task))
            if np.isfinite(d) and d < best:
                best = float(d)
        self._truck_nearest_reachable_cache[aid_s] = float(best)
        return float(best)

    def _truck_locality_detour_ratio(self, env, aid: str, task_dist_m: float) -> float:
        nearest = float(self._truck_nearest_reachable_pending_distance(env, str(aid)))
        if not np.isfinite(task_dist_m) or not np.isfinite(nearest):
            return 0.0
        detour = float(max(task_dist_m - nearest, 0.0))
        norm = float(max(self._distance_norm_m(env), 1e-6))
        return float(np.clip(detour / norm, 0.0, 1.0))

    def _task_shared_map_block_pressure(self, env, task) -> float:
        if task is None:
            return 0.0
        node = int(task.demand_node)
        nbs = list(env.topology.adjacency.get(node, set()))
        if not nbs or not hasattr(env, "_decision_is_blocked"):
            return 0.0
        blocked = sum(1 for nb in nbs if bool(env._decision_is_blocked(node, int(nb))))
        return float(np.clip(float(blocked) / max(len(nbs), 1), 0.0, 1.0))

    def _goal_map_update_impact(self, env, aid: str, goal_id: Optional[str]) -> Tuple[bool, bool]:
        """
        Evaluate whether a map update materially affects an existing goal.
        Returns (impactful, critical).
        """
        if goal_id is None:
            return (False, False)
        st = env.state.agents.get(str(aid), None)
        if st is None:
            return (False, False)

        task = env.state.tasks.get(str(goal_id), None)
        if task is not None:
            if task.status != TaskStatus.PENDING:
                return (True, True)
            block_pressure = float(self._task_shared_map_block_pressure(env, task))
            high_thr = float(max(getattr(env.cfg, "hrl_map_update_block_pressure_threshold", 0.40), 0.0))
            critical_thr = float(max(getattr(env.cfg, "hrl_map_update_block_pressure_critical_threshold", 0.70), high_thr))
            if st.kind == AgentKind.TRUCK:
                if not bool(self._truck_task_reachable(env, str(aid), task)):
                    emergency_support_only = bool(
                        task.kind == TaskKind.EMERGENCY
                        and (not bool(getattr(env.cfg, "truck_can_serve_emergency_tasks", True)))
                    )
                    # Truck emergency tasks act as support/front-push goals in this paper setup;
                    # if such a support goal becomes unreachable after map updates, treat it as
                    # impactful but non-critical to allow deferred merge instead of immediate churn.
                    if emergency_support_only:
                        return (True, False)
                    return (True, True)
                if block_pressure >= critical_thr:
                    return (True, True)
                if block_pressure >= high_thr:
                    return (True, False)
                return (False, False)
            if st.kind == AgentKind.UAV:
                if not bool(self._uav_task_feasible(env, str(aid), task)):
                    return (True, True)
                if block_pressure >= critical_thr:
                    return (True, True)
                if block_pressure >= high_thr:
                    return (True, False)
                return (False, False)
            return (False, False)

        ag = env.state.agents.get(str(goal_id), None)
        if st.kind == AgentKind.UAV and ag is not None and ag.kind == AgentKind.TRUCK:
            if bool(getattr(ag, "crashed", False)):
                return (True, True)
            return (False, False)
        return (True, True)

    def _goal_eta_proxy_m(self, env, aid: str, goal_id: Optional[str]) -> float:
        if goal_id is None:
            return float("inf")
        st = env.state.agents.get(str(aid), None)
        if st is None:
            return float("inf")
        task = env.state.tasks.get(str(goal_id), None)
        if task is not None:
            if st.kind == AgentKind.TRUCK:
                return float(self._truck_task_distance(env, str(aid), task))
            return float(env._agent_distance_to_task(str(aid), task))
        ag = env.state.agents.get(str(goal_id), None)
        if st.kind == AgentKind.UAV and ag is not None and ag.kind == AgentKind.TRUCK:
            ax, ay = self._agent_xy(env, str(aid))
            tx, ty = self._agent_xy(env, str(goal_id))
            return float(((ax - tx) ** 2 + (ay - ty) ** 2) ** 0.5)
        return float("inf")

    def _map_update_ranking_changed_impact(self, env, aid: str, goal_id: Optional[str]) -> bool:
        self.map_ranking_refresh_candidate_count_total = int(self.map_ranking_refresh_candidate_count_total) + 1
        if bool(getattr(env.cfg, "erc_ablate_map_ranking_refresh", False)):
            self.map_ranking_refresh_blocked_by_ablation_count_total = int(self.map_ranking_refresh_blocked_by_ablation_count_total) + 1
            return False
        if goal_id is None:
            return False
        st = env.state.agents.get(str(aid), None)
        if st is None:
            return False
        cur_task = env.state.tasks.get(str(goal_id), None)
        if cur_task is None:
            return False

        candidates = self._candidate_tasks(env, str(aid))
        scored: List[Tuple[str, float]] = []
        for task in candidates:
            tid = str(task.task_id)
            sc = float(self._score_goal_for_agent(env, str(aid), tid))
            if np.isfinite(sc):
                scored.append((str(tid), float(sc)))
        if not scored:
            return False

        scored.sort(key=lambda x: x[1], reverse=True)
        top_k = int(max(getattr(env.cfg, "hrl_map_update_top_k", 2), 1))
        top_scored = scored[:top_k]
        top_ids = {tid for tid, _ in top_scored}
        if str(goal_id) in top_ids:
            return False

        best_tid, best_score = top_scored[0]
        cur_score = float(self._score_goal_for_agent(env, str(aid), str(goal_id)))
        if not np.isfinite(best_score):
            return False
        if not np.isfinite(cur_score):
            self.map_ranking_refresh_allowed_count_total = int(self.map_ranking_refresh_allowed_count_total) + 1
            return True

        if st.kind == AgentKind.UAV:
            score_margin = float(max(getattr(env.cfg, "hrl_uav_goal_switch_margin", self.switch_margin), 0.0))
        else:
            score_margin = float(max(getattr(env.cfg, "hrl_truck_goal_switch_margin", self.switch_margin), 0.0))
        if bool(self.use_event_trigger):
            # Value-aware ERC gate: ranking-changed should represent substantial
            # map-update value, not a tiny score jitter.
            margin_scale = float(max(getattr(env.cfg, "hrl_map_update_event_ranking_margin_scale", 1.35), 1.0))
            score_margin = float(score_margin * margin_scale)

        if float(best_score - cur_score) <= float(score_margin):
            return False

        cur_eta = float(self._goal_eta_proxy_m(env, str(aid), str(goal_id)))
        best_eta = float(self._goal_eta_proxy_m(env, str(aid), str(best_tid)))
        if (not np.isfinite(cur_eta)) or (not np.isfinite(best_eta)):
            self.map_ranking_refresh_allowed_count_total = int(self.map_ranking_refresh_allowed_count_total) + 1
            return True

        eta_jump_thr = float(max(getattr(env.cfg, "hrl_map_update_eta_jump_threshold", 0.15), 0.0))
        if bool(self.use_event_trigger):
            eta_jump_floor = float(max(getattr(env.cfg, "hrl_map_update_event_eta_jump_floor", 0.22), 0.0))
            eta_jump_thr = float(max(eta_jump_thr, eta_jump_floor))
        eta_gain = float((cur_eta - best_eta) / max(cur_eta, 1e-6))
        ok = bool(eta_gain >= eta_jump_thr)
        if ok:
            self.map_ranking_refresh_allowed_count_total = int(self.map_ranking_refresh_allowed_count_total) + 1
        return ok

    def _is_map_update_actionable_for_current_commitment(
        self,
        env,
        *,
        any_goal_assigned: bool,
        map_event: bool,
        by_truck_dead_end: bool,
    ) -> Tuple[bool, bool, int, int, Dict[str, int]]:
        reasons = {
            "path_blocked": 0,
            "goal_unreachable": 0,
            "ranking_changed": 0,
            "dead_end": 0,
            "recovery_path_fractured": 0,
        }

        if not bool(any_goal_assigned):
            # No committed goals: map update can wait for regular refresh.
            return (False, False, 0, 0, reasons)

        impacted_agents: set = set()
        critical_agents: set = set()

        for aid, gid in self.state.goals.items():
            if gid is None:
                continue
            aid_s = str(aid)
            gid_s = str(gid)
            st = env.state.agents.get(aid_s, None)
            task = env.state.tasks.get(gid_s, None)

            imp, crit = self._goal_map_update_impact(env, aid_s, gid_s)
            if imp:
                impacted_agents.add(aid_s)
            if crit:
                critical_agents.add(aid_s)

            if imp:
                if task is not None:
                    if st is not None and st.kind == AgentKind.TRUCK and (not bool(self._truck_task_reachable(env, aid_s, task))):
                        reasons["goal_unreachable"] += 1
                    elif st is not None and st.kind == AgentKind.UAV and (not bool(self._uav_task_feasible(env, aid_s, task))):
                        reasons["goal_unreachable"] += 1
                    else:
                        reasons["path_blocked"] += 1
                else:
                    ag = env.state.agents.get(gid_s, None)
                    if st is not None and st.kind == AgentKind.UAV and ag is not None and ag.kind == AgentKind.TRUCK and bool(getattr(ag, "crashed", False)):
                        reasons["recovery_path_fractured"] += 1
                    else:
                        reasons["goal_unreachable"] += 1

            allow_ranking_as_map_impact = bool(getattr(env.cfg, "hrl_map_update_allow_ranking_changed_impact", False))
            if allow_ranking_as_map_impact and self._map_update_ranking_changed_impact(env, aid_s, gid_s):
                impacted_agents.add(aid_s)
                reasons["ranking_changed"] += 1

        if bool(by_truck_dead_end):
            reasons["dead_end"] += 1

        # Hard map-impact immediate refresh should be path/goal/recovery only.
        # truck_dead_end remains a hard event trigger, but not a map-impact trigger.
        actionable = bool(len(impacted_agents) > 0)
        critical = bool(len(critical_agents) > 0)
        impacted_count = int(len(impacted_agents))
        critical_count = int(len(critical_agents))
        return (actionable, critical, impacted_count, critical_count, reasons)

    def _map_update_replan_gate(
        self,
        env,
        map_signal: bool,
        *,
        any_goal_assigned: bool,
        map_event: bool,
        by_truck_dead_end: bool,
    ) -> Tuple[bool, bool, int, int, Dict[str, int]]:
        """
        Return (allow_map_replan, map_critical, impacted_count, critical_count, reason_counts).
        """
        if not bool(map_signal):
            return (False, False, 0, 0, {
                "path_blocked": 0,
                "goal_unreachable": 0,
                "ranking_changed": 0,
                "dead_end": 0,
                "recovery_path_fractured": 0,
            })
        return self._is_map_update_actionable_for_current_commitment(
            env,
            any_goal_assigned=bool(any_goal_assigned),
            map_event=bool(map_event),
            by_truck_dead_end=bool(by_truck_dead_end),
        )

    def _uav_recovery_feasibility_score(self, env, task) -> float:
        if task is None or task.kind != TaskKind.EMERGENCY:
            return 0.0
        node = env.topology.nodes[int(task.demand_node)]
        _, d_back = self._nearest_truck_from_xy(env, (float(node.x), float(node.y)))
        if not np.isfinite(d_back):
            return 0.0
        return float(1.0 / (1.0 + float(d_back) / self._distance_norm_m(env)))

    def _island_serviceability_ratio(self, env) -> float:
        island_ids = set(getattr(env, "_current_island_emergency_task_ids", lambda: set())())
        pending = [
            env.state.tasks[str(tid)]
            for tid in sorted(island_ids)
            if str(tid) in env.state.tasks
            and env.state.tasks[str(tid)].status == TaskStatus.PENDING
            and env.state.tasks[str(tid)].kind == TaskKind.EMERGENCY
        ]
        if not pending:
            return 1.0
        serviceable = 0
        for task in pending:
            ok = False
            for uid, st in env.state.agents.items():
                if st.kind != AgentKind.UAV or bool(getattr(st, "crashed", False)):
                    continue
                if self._uav_task_feasible(env, str(uid), task):
                    ok = True
                    break
            if ok:
                serviceable += 1
        return float(np.clip(float(serviceable) / max(len(pending), 1), 0.0, 1.0))

    def _blocked_edge_version(self, env) -> int:
        return int(
            max(
                int(getattr(env, "_shared_map_update_count_total", 0)),
                int(getattr(env, "unknown_blocked_edge_hit_total", 0)),
                int(getattr(env, "_shared_map_new_blocked_step", 0)),
                int(getattr(env, "_shared_map_new_unblocked_step", 0)),
            )
        )

    def _anchor_state_signature(self, env) -> Tuple[Tuple[str, int], ...]:
        sig = []
        for tid, st in sorted(env.state.agents.items(), key=lambda kv: str(kv[0])):
            if st.kind != AgentKind.TRUCK or bool(getattr(st, "crashed", False)):
                continue
            if st.node is not None:
                pos_tag = int(st.node)
            else:
                xy = st.pos_xy if st.pos_xy is not None else (0.0, 0.0)
                pos_tag = int(round(float(xy[0]) / 10.0) * 100000 + round(float(xy[1]) / 10.0))
            sig.append((str(tid), int(pos_tag)))
        return tuple(sig)

    def _support_anchor_service_gain(self, env, aid: str, task) -> Dict[str, float]:
        self._ensure_step_eval_caches(env)
        cache_key = (str(aid), str(getattr(task, "task_id", "<none>")))
        cached = self._support_anchor_gain_cache.get(cache_key, None)
        if cached is not None:
            return dict(cached)
        out = {
            "new_serviceable_task_count": 0.0,
            "new_relaxed_feasible_task_count": 0.0,
            "improves_island_task_count": 0.0,
            "gain_score": 0.0,
            "post_support_primary_distance_m": float("inf"),
        }
        if task is None or task.kind != TaskKind.EMERGENCY or task.status != TaskStatus.PENDING:
            self._support_anchor_gain_cache[cache_key] = dict(out)
            return dict(out)
        st = env.state.agents.get(str(aid), None)
        if st is None or st.kind != AgentKind.TRUCK or st.node is None:
            self._support_anchor_gain_cache[cache_key] = dict(out)
            return dict(out)
        if not hasattr(env, "_decision_neighbors"):
            self._support_anchor_gain_cache[cache_key] = dict(out)
            return dict(out)

        neighbors = [int(x) for x in env._decision_neighbors(int(st.node))]
        if not neighbors:
            self._support_anchor_gain_cache[cache_key] = dict(out)
            return dict(out)

        task_node = int(task.demand_node)
        island_ids = set(getattr(env, "_current_island_emergency_task_ids", lambda: set())())
        is_island = bool(str(task.task_id) in island_ids)

        # Prefer island-specific support target when available; otherwise fall
        # back to generic best neighbor that reduces distance to this task.
        nb = None
        if is_island and hasattr(env, "_truck_island_forward_support_target"):
            try:
                nb = env._truck_island_forward_support_target(str(aid), neighbors, island_ids)
            except Exception:
                nb = None
        if nb is None:
            best_nb = None
            best_d = float("inf")
            for cand in neighbors:
                d = float(env._decision_shortest_path_distance(int(cand), task_node)) if hasattr(env, "_decision_shortest_path_distance") else float("inf")
                if np.isfinite(d) and d + 1e-9 < best_d:
                    best_d = float(d)
                    best_nb = int(cand)
            nb = best_nb
        if nb is None:
            self._support_anchor_gain_cache[cache_key] = dict(out)
            return dict(out)

        try:
            out["post_support_primary_distance_m"] = float(
                env._decision_shortest_path_distance(int(nb), int(task_node))
            ) if hasattr(env, "_decision_shortest_path_distance") else float("inf")
        except Exception:
            out["post_support_primary_distance_m"] = float("inf")

        hp_cap = float(max(getattr(env.cfg, "uav_high_pressure_rendezvous_max_nearest_truck_m", 2200.0), 0.0))
        multi_k = int(max(getattr(env.cfg, "hrl_support_gain_multi_task_k", 3), 1))

        truck_nodes = [
            int(ts.node)
            for _tid, ts in env.state.agents.items()
            if ts.kind == AgentKind.TRUCK and (not bool(getattr(ts, "crashed", False))) and ts.node is not None
        ]

        def _min_truck_dist_to_node(node_id: int) -> float:
            best = float("inf")
            for tnode in truck_nodes:
                d = float(env._decision_shortest_path_distance(int(tnode), int(node_id))) if hasattr(env, "_decision_shortest_path_distance") else float("inf")
                if np.isfinite(d) and d + 1e-9 < best:
                    best = float(d)
            return best

        def _post_min_dist_to_node(node_id: int) -> float:
            cur = _min_truck_dist_to_node(int(node_id))
            nb_d = float(env._decision_shortest_path_distance(int(nb), int(node_id))) if hasattr(env, "_decision_shortest_path_distance") else float("inf")
            outv = float(cur)
            if np.isfinite(nb_d):
                outv = float(min(outv, float(nb_d)))
            return outv

        pending_emerg = [
            t for t in env.state.tasks.values()
            if t.kind == TaskKind.EMERGENCY and t.status == TaskStatus.PENDING
        ]
        # Keep compute bounded: evaluate the anchor task + a few nearby emergencies.
        pending_emerg.sort(
            key=lambda t: float(env._decision_shortest_path_distance(int(t.demand_node), int(task_node)))
            if hasattr(env, "_decision_shortest_path_distance") else float("inf")
        )
        eval_tasks = pending_emerg[: int(max(multi_k, 1))]
        if all(str(getattr(t, "task_id", "")) != str(getattr(task, "task_id", "")) for t in eval_tasks):
            eval_tasks = [task] + eval_tasks[: max(multi_k - 1, 0)]

        newly_serviceable = 0.0
        newly_relaxed = 0.0
        newly_island = 0.0
        gain_norm_best = 0.0
        eta_rel_best = 0.0

        for cand in eval_tasks:
            node_id = int(cand.demand_node)
            cur_min = _min_truck_dist_to_node(node_id)
            post_min = _post_min_dist_to_node(node_id)

            local_new = 0.0
            if np.isfinite(cur_min) and np.isfinite(post_min):
                if cur_min > hp_cap and post_min <= hp_cap:
                    local_new = 1.0
            elif (not np.isfinite(cur_min)) and np.isfinite(post_min):
                local_new = 1.0

            gain_m = 0.0
            if np.isfinite(cur_min) and np.isfinite(post_min):
                gain_m = float(max(cur_min - post_min, 0.0))
            elif (not np.isfinite(cur_min)) and np.isfinite(post_min):
                gain_m = float(self._distance_norm_m(env))
            gain_norm = float(np.clip(gain_m / max(self._distance_norm_m(env), 1e-6), 0.0, 1.0))

            eta_rel_gain = 0.0
            if np.isfinite(cur_min) and np.isfinite(post_min) and cur_min > 1e-6:
                eta_rel_gain = float(np.clip((cur_min - post_min) / cur_min, 0.0, 1.0))
            if (not np.isfinite(cur_min)) and np.isfinite(post_min):
                eta_rel_gain = 1.0

            # Soft conversion credit: in large maps, support that substantially
            # reduces ETA for high-pressure tasks should count as actionable gain
            # even if it does not immediately cross the rendezvous distance cap.
            if local_new <= 0.0 and bool(self._task_high_pressure(env, cand)) and eta_rel_gain >= 0.18:
                local_new = 0.5

            if local_new > 0.0:
                newly_serviceable += float(local_new)
                if bool(self._task_high_pressure(env, cand)) or bool(self._is_island_task(env, cand)):
                    newly_relaxed += 1.0
                if bool(self._is_island_task(env, cand)):
                    newly_island += 1.0

            gain_norm_best = max(gain_norm_best, gain_norm)
            eta_rel_best = max(eta_rel_best, eta_rel_gain)

        out["new_serviceable_task_count"] = float(newly_serviceable)
        out["new_relaxed_feasible_task_count"] = float(newly_relaxed)
        out["improves_island_task_count"] = float(newly_island)

        count_gain = float(np.clip(newly_serviceable / max(float(multi_k), 1.0), 0.0, 1.0))
        relaxed_bonus = 0.15 if newly_relaxed > 0.0 else 0.0
        out["gain_score"] = float(np.clip(max(gain_norm_best, eta_rel_best, count_gain) + relaxed_bonus, 0.0, 1.0))
        self._support_anchor_gain_cache[cache_key] = dict(out)
        return dict(out)

    def _task_high_pressure(self, env, task) -> bool:
        if task is None or task.kind != TaskKind.EMERGENCY or task.status != TaskStatus.PENDING:
            return False
        self._ensure_step_eval_caches(env)
        task_id = str(getattr(task, "task_id", "<none>"))
        cached = self._task_high_pressure_cache.get(task_id, None)
        if cached is not None:
            return bool(cached)
        out = False
        fn = getattr(env, "_is_high_pressure_emergency_task", None)
        if callable(fn):
            try:
                if bool(fn(task)):
                    out = True
            except Exception:
                pass
        if not out:
            fn_island = getattr(env, "_is_high_pressure_island_task", None)
            if callable(fn_island):
                try:
                    if bool(fn_island(task)):
                        out = True
                except Exception:
                    pass
        if not out:
            out = bool(self._is_island_task(env, task))
        self._task_high_pressure_cache[task_id] = bool(out)
        return bool(out)

    def _truck_emergency_support_candidate(self, env, aid: str, task) -> bool:
        if task is None or task.kind != TaskKind.EMERGENCY or task.status != TaskStatus.PENDING:
            return False
        self._ensure_step_eval_caches(env)
        cache_key = (str(aid), str(getattr(task, "task_id", "<none>")))
        cached = self._truck_support_candidate_cache.get(cache_key, None)
        if cached is not None:
            return bool(cached)

        chain = self._support_bound_chain_info_for_truck(env, str(aid))
        if (
            chain is not None
            and self._tc_support_chain_class.get(str(aid), "") == "support_required"
            and str(chain.get("task_id", "")) == str(getattr(task, "task_id", ""))
        ):
            self._truck_support_candidate_cache[cache_key] = True
            return True

        d = float(self._truck_task_distance(env, str(aid), task))
        if not np.isfinite(d):
            self._truck_support_candidate_cache[cache_key] = False
            return False
        base_cap = float(max(getattr(env.cfg, "hrl_support_candidate_max_distance_m", 5000.0), 0.0))
        is_island = bool(self._is_island_task(env, task))
        is_hp = bool(self._task_high_pressure(env, task))
        cap = float(base_cap * (1.35 if (is_island or is_hp) else 1.0))
        if cap > 1e-9 and d > cap:
            self._truck_support_candidate_cache[cache_key] = False
            return False

        gain_info = self._support_anchor_service_gain(env, aid, task)
        actionable_gain = bool(self._support_anchor_actionable_gain(env, aid, task, gain_info=gain_info))

        if is_island or is_hp:
            out = bool(actionable_gain)
            self._truck_support_candidate_cache[cache_key] = bool(out)
            return bool(out)

        norm_pending_snapshot, norm_reach_by_truck, _any_reachable_normal = self._normal_reachability_snapshot(env)
        if int(norm_pending_snapshot) > 0 and (not bool(norm_reach_by_truck.get(str(aid), True))):
            if self._is_timecritical_lightweight_task(task):
                return True

        uav_cover = float(np.clip(self._uav_emergency_cover_fraction(env, task), 0.0, 1.0))
        urgency = float(np.clip(self._norm_deadline_urgency(task, int(env.state.step_index)), 0.0, 1.0))
        cover_thr = float(np.clip(getattr(env.cfg, "hrl_truck_emergency_force_relief_uav_cover_threshold", 0.35), 0.0, 1.0))
        urg_thr = float(np.clip(getattr(env.cfg, "hrl_truck_emergency_force_relief_urgency_threshold", 0.72), 0.0, 1.0))
        out = bool((uav_cover < cover_thr) or (urgency >= urg_thr))
        self._truck_support_candidate_cache[cache_key] = bool(out)
        return bool(out)

    def _truck_support_serviceability_gain(self, env, aid: str, task) -> float:
        gain_info = self._support_anchor_service_gain(env, aid, task)
        return float(np.clip(float(gain_info.get("gain_score", 0.0)), 0.0, 1.0))

    def _support_anchor_actionable_gain(self, env, aid: str, task, gain_info: Optional[Dict[str, float]] = None) -> bool:
        if task is None or task.kind != TaskKind.EMERGENCY or task.status != TaskStatus.PENDING:
            return False
        if (not self._support_need_active(env)) or (not bool(getattr(env.cfg, "hrl_support_require_actionable_gain", True))):
            return True
        gi = gain_info if isinstance(gain_info, dict) else self._support_anchor_service_gain(env, str(aid), task)
        gain_score = float(np.clip(float(gi.get("gain_score", 0.0)), 0.0, 1.0))
        new_serviceable = float(max(float(gi.get("new_serviceable_task_count", 0.0)), 0.0))
        new_relaxed = float(max(float(gi.get("new_relaxed_feasible_task_count", 0.0)), 0.0))
        improves_island = float(max(float(gi.get("improves_island_task_count", 0.0)), 0.0))
        post_dist = float(gi.get("post_support_primary_distance_m", float("inf")))
        min_gain = float(np.clip(getattr(env.cfg, "hrl_support_actionable_min_gain_score", 0.22), 0.0, 1.0))
        min_new = float(max(getattr(env.cfg, "hrl_support_actionable_min_new_serviceable", 0.5), 0.0))
        max_post_dist = float(max(getattr(env.cfg, "hrl_support_actionable_post_distance_m", 2200.0), 0.0))
        if new_serviceable >= min_new:
            return True
        if new_relaxed >= 1.0:
            return True
        if improves_island >= 1.0:
            return True
        if gain_score >= min_gain and np.isfinite(post_dist) and (max_post_dist <= 0.0 or post_dist <= max_post_dist):
            return True
        return False

    def _truck_forward_support_score(self, env, aid: str, task) -> float:
        if task is None or task.kind != TaskKind.EMERGENCY or (not self._is_island_task(env, task)):
            return 0.0
        # Only reward support moves that actually improve island emergency serviceability.
        gain = float(self._truck_support_serviceability_gain(env, aid, task))
        if gain <= 1e-9:
            return 0.0
        return float(np.clip(gain, 0.0, 1.0))

    def _supported_sortie_score(self, env, task) -> float:
        if task is None or task.kind != TaskKind.EMERGENCY:
            return 0.0
        node = env.topology.nodes[int(task.demand_node)]
        _, d_back = self._nearest_truck_from_xy(env, (float(node.x), float(node.y)))
        if not np.isfinite(d_back):
            return 0.0
        return float(1.0 / (1.0 + float(d_back) / self._distance_norm_m(env)))

    def _uav_stage_blocks_task_goal(self, env, aid: str) -> bool:
        st = env.state.agents.get(str(aid), None)
        if st is None or st.kind != AgentKind.UAV or bool(st.crashed):
            return True
        if st.follow_target is not None:
            dwell_rem = int(getattr(env, "_uav_post_bind_dwell_remaining", {}).get(str(aid), 0))
            if dwell_rem > 0:
                return True
            if bool(getattr(st, "uav_needs_reload_flag", False)):
                return True
            if bool(getattr(st, "replenish_timer", 0) > 0):
                return True
        if bool(getattr(st, "uav_needs_reload_flag", False)):
            return True
        if bool(getattr(env, "_uav_forced_rth_latch", {}).get(str(aid), False)):
            return True
        low_thr = float(np.clip(getattr(env.cfg, "uav_low_battery_goal_lock_threshold", 0.35), 0.0, 1.0))
        if float(getattr(st, "battery", 0.0)) < low_thr:
            return True
        if hasattr(env, "_uav_loaded") and (not bool(env._uav_loaded(str(aid)))):
            return True
        return False

    def _score_truck_task(self, env, aid: str, task) -> float:
        step_idx = int(env.state.step_index)
        dist_m = float(self._truck_task_distance(env, aid, task))
        if not np.isfinite(dist_m):
            return -1e9
        urgency = self._norm_deadline_urgency(task, step_idx)
        eta_score = self._norm_eta_score(dist_m, float(getattr(env.cfg, "truck_speed_mps", 8.0)))
        demand = self._demand_score(task, env)
        risk = self._task_risk(env, task) if self.use_risk_term else 0.0
        keep = self._keep_goal_bonus(aid, str(task.task_id))
        island_bonus = 1.0 if self._is_island_task(env, task) else 0.0
        map_bonus = 1.0 if self._map_update_active() else 0.0
        support_gain_info = self._support_anchor_service_gain(env, aid, task)
        support_gain = float(np.clip(float(support_gain_info.get("gain_score", 0.0)), 0.0, 1.0))
        support_bonus = float(np.clip(support_gain if (task is not None and task.kind == TaskKind.EMERGENCY and self._is_island_task(env, task)) else 0.0, 0.0, 1.0))
        recovery_bonus = self._uav_recovery_feasibility_score(env, task)
        supported_sortie_joint = self._truck_supported_sortie_joint_score(env, aid, task)
        direction_split_term = self._truck_directional_split_term(env, aid, task)
        late_sector_spread_term = self._truck_late_sector_spread_term(env, aid, task)
        initial_directional_term = self._truck_initial_directional_cover_term(env, aid, task)
        support_count = float(max(getattr(env, "truck_forward_support_count_total", 0), 0))
        truck_count = max(1.0, float(sum(1 for a in env.state.agents.values() if a.kind == AgentKind.TRUCK)))
        support_sat = float(np.clip(support_count / (3.0 * truck_count), 0.0, 1.0))
        conv_quality = float(self._support_conversion_quality(env))
        conv_penalty = float(max(getattr(env.cfg, "hrl_support_conversion_penalty_strength", 0.55), 0.0))
        island_bonus_eff = float(island_bonus * (1.0 - 0.35 * support_sat) * (1.0 - 0.25 * conv_penalty * (1.0 - conv_quality)))
        support_bonus_eff = float(support_bonus * (1.0 - 0.65 * support_sat) * (1.0 - conv_penalty * (1.0 - conv_quality)))
        pending_norm = sum(
            1
            for t in env.state.tasks.values()
            if t.status == TaskStatus.PENDING and t.kind == TaskKind.NORMAL
        )
        normal_pressure, emergency_pressure = self._pending_task_pressure(env)
        norm_pending_snapshot, norm_reach_by_truck, any_reachable_normal = self._normal_reachability_snapshot(env)
        pending_norm = int(norm_pending_snapshot)
        truck_has_normal_reachable = bool(norm_reach_by_truck.get(str(aid), True))
        support_scale_when_normal_reachable = float(
            np.clip(getattr(env.cfg, "hrl_truck_support_when_normal_reachable_scale", 0.45), 0.0, 1.0)
        )
        no_normal_reachable_mode = bool(
            int(pending_norm) > 0
            and bool(getattr(env.cfg, "hrl_truck_emergency_support_when_no_normal_enabled", True))
            and (not truck_has_normal_reachable)
        )
        if int(pending_norm) > 0 and truck_has_normal_reachable and any_reachable_normal:
            support_bonus_eff = float(support_bonus_eff * support_scale_when_normal_reachable)

        detour_ratio = float(self._truck_locality_detour_ratio(env, str(aid), float(dist_m)))
        locality_support_relief = float(np.clip(max(island_bonus, support_bonus, recovery_bonus), 0.0, 1.0))
        if self.use_event_trigger:
            if task.kind == TaskKind.NORMAL:
                locality_weight = float(0.12 + 0.10 * normal_pressure)
                locality_bonus = float((1.0 - detour_ratio) * (0.04 + 0.03 * normal_pressure))
            else:
                locality_weight = float((0.06 + 0.04 * normal_pressure) * (1.0 - 0.70 * locality_support_relief))
                locality_bonus = float((1.0 - detour_ratio) * (0.02 + 0.02 * locality_support_relief))
        else:
            locality_weight = 0.0
            locality_bonus = 0.0

        if task.kind == TaskKind.NORMAL:
            normal_event_bonus = 0.12 if self.use_event_trigger else 0.0
            task_type_bias = float(0.24 + 0.30 * normal_pressure + 0.08 * normal_pressure + normal_event_bonus)
        else:
            island_w = 0.12 if self.use_event_trigger else 0.14
            support_w = 0.08 if self.use_event_trigger else 0.12
            recovery_w = 0.06 if self.use_event_trigger else 0.10
            task_type_bias = float(
                -0.24 * normal_pressure
                + 0.10 * emergency_pressure * (0.60 * island_bonus_eff + 0.40 * support_bonus_eff)
                + island_w * island_bonus_eff
                + support_w * support_bonus_eff
                + recovery_w * recovery_bonus
            )

        emergency_diversion_penalty = 0.0
        normal_backlog_guard_penalty = 0.0
        support_ready_bonus = 0.0
        support_waste_penalty = 0.0
        no_normal_support_bonus = 0.0
        soft_clamp_penalty = 0.0
        timecritical_penalty_scale = 1.0
        timecritical_support_amp = 1.0
        timecritical_recovery_amp = 1.0
        timecritical_supported_sortie_amp = 1.0
        support_binding_priority_bonus = 0.0
        force_entry_bonus = 0.0
        if task.kind == TaskKind.EMERGENCY:
            uav_ids = [
                str(uid)
                for uid, ag in env.state.agents.items()
                if ag.kind == AgentKind.UAV and (not bool(getattr(ag, "crashed", False)))
            ]
            uav_cover = 0.0
            if uav_ids:
                feasible_cover = sum(1 for uid in uav_ids if self._uav_task_feasible(env, str(uid), task))
                uav_cover = float(np.clip(float(feasible_cover) / max(len(uav_ids), 1), 0.0, 1.0))
            tc_warning = float(np.clip(getattr(env.cfg, "hrl_timecritical_lifeline_warning_ratio", 0.55), 0.0, 1.0))
            tc_critical = float(np.clip(getattr(env.cfg, "hrl_timecritical_lifeline_critical_ratio", 0.35), 0.0, 1.0))
            if task.kind == TaskKind.EMERGENCY and self._is_timecritical_lightweight_task(task):
                tc_ratio = float(self._task_lifeline_ratio(task))
                if tc_ratio <= tc_critical:
                    timecritical_penalty_scale = float(np.clip(getattr(env.cfg, "hrl_truck_timecritical_penalty_scale_critical", 0.10), 0.0, 1.0))
                    timecritical_support_amp = float(max(getattr(env.cfg, "hrl_truck_timecritical_support_amp_critical", 1.45), 0.0))
                    timecritical_recovery_amp = float(max(getattr(env.cfg, "hrl_truck_timecritical_recovery_amp_critical", 1.35), 0.0))
                    timecritical_supported_sortie_amp = float(max(getattr(env.cfg, "hrl_truck_timecritical_supported_sortie_amp_critical", 1.30), 0.0))
                elif tc_ratio <= tc_warning:
                    timecritical_penalty_scale = float(np.clip(getattr(env.cfg, "hrl_truck_timecritical_penalty_scale_warning", 0.25), 0.0, 1.0))
                    timecritical_support_amp = float(max(getattr(env.cfg, "hrl_truck_timecritical_support_amp_warning", 1.25), 0.0))
                    timecritical_recovery_amp = float(max(getattr(env.cfg, "hrl_truck_timecritical_recovery_amp_warning", 1.15), 0.0))
                    timecritical_supported_sortie_amp = float(max(getattr(env.cfg, "hrl_truck_timecritical_supported_sortie_amp_warning", 1.12), 0.0))
                bind_info = self._support_bound_delivery_info(env, str(aid), task, gain_info=support_gain_info)
                if float(bind_info.get("bound_timecritical_critical", 0.0)) > 0.0:
                    support_binding_priority_bonus += float(max(getattr(env.cfg, "hrl_support_bind_bonus_critical", 0.45), 0.0))
                elif float(bind_info.get("bound_timecritical_warning", 0.0)) > 0.0:
                    support_binding_priority_bonus += float(max(getattr(env.cfg, "hrl_support_bind_bonus_warning", 0.25), 0.0))
                elif float(bind_info.get("bound_bulk", 0.0)) > 0.0:
                    support_binding_priority_bonus += float(max(getattr(env.cfg, "hrl_support_bind_bonus_bulk", 0.10), 0.0))
                if self._timecritical_force_entry_active(env, task):
                    force_entry_bonus += float(max(getattr(env.cfg, "hrl_timecritical_force_entry_truck_bonus", 0.20), 0.0))

            div_base = float(0.38 + 0.42 * normal_pressure) if self.use_event_trigger else float(0.22 + 0.26 * normal_pressure)
            emergency_diversion_penalty = float(
                div_base
                * uav_cover
                * (1.0 - 0.75 * island_bonus_eff - 0.65 * support_bonus_eff)
                * (1.0 - 0.35 * supported_sortie_joint)
                * (1.0 + 0.65 * conv_penalty * (1.0 - conv_quality))
            )
            hard_recovery_active = False
            hard_recovery_fn = getattr(env, "_has_hard_recovery_uav", None)
            if callable(hard_recovery_fn):
                hard_recovery_active = bool(hard_recovery_fn())
            if int(pending_norm) > 0 and (not hard_recovery_active):
                normal_backlog_guard_penalty = float(
                    (0.62 + 0.46 * uav_cover)
                    * normal_pressure
                    * (1.0 - 0.60 * island_bonus_eff - 0.50 * support_bonus_eff)
                    * (1.0 - 0.25 * supported_sortie_joint)
                )
                if truck_has_normal_reachable and any_reachable_normal:
                    normal_backlog_guard_penalty = float(
                        normal_backlog_guard_penalty
                        * (1.0 + 0.25 * (1.0 - support_scale_when_normal_reachable))
                    )
            low_cover = bool(uav_cover < float(np.clip(getattr(env.cfg, "hrl_truck_emergency_force_relief_uav_cover_threshold", 0.35), 0.0, 1.0)))
            if float(support_gain) > 1e-9 and (self._is_island_task(env, task) or low_cover):
                support_ready_bonus = float((0.14 + 0.10 * emergency_pressure) * float(np.clip(support_gain, 0.0, 1.0)))
                island_new_service_bonus = float(max(getattr(env.cfg, "hrl_serviceable_island_bonus", 0.25), 0.0))
                hp_new_service_bonus = float(max(getattr(env.cfg, "hrl_serviceable_high_pressure_emergency_bonus", 0.20), 0.0))
                if self._is_island_task(env, task):
                    support_ready_bonus += float(island_new_service_bonus * float(np.clip(support_gain, 0.0, 1.0)))
                elif self._task_high_pressure(env, task):
                    support_ready_bonus += float(hp_new_service_bonus * float(np.clip(support_gain, 0.0, 1.0)))
            if float(support_bonus) > 1e-9 and float(support_gain) <= 1e-9:
                support_waste_penalty = float(0.10 + 0.08 * normal_pressure)

            if self._support_soft_clamp_blocks_task(env, str(aid), task, gain_info=support_gain_info):
                if not self._support_escape_hatch_allows(env, str(aid), task, gain_info=support_gain_info):
                    soft_clamp_penalty = float(0.22 + 0.12 * normal_pressure)

            if int(pending_norm) > 0 and truck_has_normal_reachable and any_reachable_normal:
                support_ready_bonus = float(support_ready_bonus * support_scale_when_normal_reachable)
                support_waste_penalty = float(
                    support_waste_penalty + 0.06 * (1.0 - support_scale_when_normal_reachable)
                )

            if no_normal_reachable_mode:
                bonus_base = float(max(getattr(env.cfg, "hrl_truck_support_when_no_normal_bonus", 0.30), 0.0))
                support_signal = float(
                    np.clip(
                        max(
                            float(support_gain),
                            0.55 * float(recovery_bonus) + 0.45 * float(1.0 - uav_cover),
                            0.35 * float(urgency),
                        ),
                        0.0,
                        1.0,
                    )
                )
                no_normal_support_bonus = float(bonus_base * support_signal)
                emergency_diversion_penalty = float(0.35 * emergency_diversion_penalty)
                normal_backlog_guard_penalty = 0.0

            emergency_diversion_penalty = float(emergency_diversion_penalty * timecritical_penalty_scale)
            normal_backlog_guard_penalty = float(normal_backlog_guard_penalty * timecritical_penalty_scale)

        block_pressure = self._task_shared_map_block_pressure(env, task)
        map_penalty = 0.0
        map_relevance_bonus = 0.0
        if bool(map_bonus):
            if self.use_event_trigger:
                map_penalty = float(block_pressure * (0.30 + 0.20 * float(getattr(env, "_shared_map_new_blocked_step", 0) > 0)))
                map_relevance_bonus = float((1.0 - block_pressure) * (0.04 + 0.06 * support_bonus_eff + 0.04 * island_bonus_eff))
            else:
                map_penalty = float(block_pressure * (0.25 + 0.20 * float(getattr(env, "_shared_map_new_blocked_step", 0) > 0)))

        terminal_delivery_bonus = 0.0
        prev_goal = self.state.goals.get(str(aid), None)
        if prev_goal is not None and str(prev_goal) == str(task.task_id):
            d_prev = float(self._truck_task_distance(env, str(aid), task))
            if np.isfinite(d_prev):
                terminal_lock_dist = float(max(getattr(env.cfg, "truck_goal_terminal_lock_distance_m", 420.0), 1.0))
                terminal_delivery_bonus = float(np.clip(1.0 - d_prev / terminal_lock_dist, 0.0, 1.0))

        endgame_term = 0.0
        max_steps = int(max(getattr(env.cfg, "max_steps", 0), 0))
        rem_episode = int(max(max_steps - step_idx, 0))
        endgame_window = int(max(getattr(env.cfg, "hrl_endgame_window_steps", 70), 0))
        if endgame_window > 0:
            endgame_scale = float(np.clip((float(endgame_window) - float(rem_episode)) / float(max(endgame_window, 1)), 0.0, 1.0))
            if endgame_scale > 1e-9:
                far_thr = float(max(getattr(env.cfg, "hrl_endgame_far_distance_m", 2200.0), 1.0))
                far_ratio = float(np.clip((float(dist_m) - far_thr) / max(far_thr, 1.0), 0.0, 1.0))
                if task.kind == TaskKind.NORMAL:
                    nb = float(max(getattr(env.cfg, "hrl_endgame_truck_normal_bonus", 0.18), 0.0))
                    endgame_term = float(endgame_scale * nb * float(np.clip(eta_score, 0.0, 1.0)))
                else:
                    ep = float(max(getattr(env.cfg, "hrl_endgame_truck_far_emergency_penalty", 0.16), 0.0))
                    endgame_term = float(-endgame_scale * ep * far_ratio)

        ablate_event_bonus = bool(getattr(env.cfg, "erc_ablate_event_scoring_bonus", False))
        if ablate_event_bonus:
            self.event_scoring_bonus_blocked_by_ablation_count_total = int(self.event_scoring_bonus_blocked_by_ablation_count_total) + 1
        else:
            self.event_scoring_bonus_applied_count_total = int(self.event_scoring_bonus_applied_count_total) + 1
        if ablate_event_bonus:
            island_bonus = 0.0
            map_bonus = 0.0
        event_gain = float(self._event_bonus_gain(env))

        return float(
            self.weights.truck_urgency * urgency
            + self.weights.truck_eta * eta_score
            - self.weights.truck_risk * risk
            + self.weights.truck_demand * demand
            + 0.36 * island_bonus_eff
            + ((0.14 if self.use_event_trigger else 0.20) * timecritical_support_amp * support_bonus_eff)
            + ((0.10 if self.use_event_trigger else 0.16) * timecritical_recovery_amp * recovery_bonus)
            + ((0.12 if self.use_event_trigger else 0.08) * timecritical_supported_sortie_amp * supported_sortie_joint)
            + task_type_bias
            + support_binding_priority_bonus
            + force_entry_bonus
            - emergency_diversion_penalty
            - normal_backlog_guard_penalty
            + (0.10 if self.use_event_trigger else 0.12) * map_bonus
            + map_relevance_bonus
            - map_penalty
            + direction_split_term
            + late_sector_spread_term
            + initial_directional_term
            + timecritical_support_amp * support_ready_bonus
            - support_waste_penalty
            + no_normal_support_bonus
            - soft_clamp_penalty
            - locality_weight * detour_ratio
            + locality_bonus
            + (0.08 if self.use_event_trigger else 0.00) * terminal_delivery_bonus
            + endgame_term
            + event_gain
            * (
                0.12 * island_bonus_eff
                + 0.10 * map_bonus
                + 0.05 * support_bonus_eff
                + 0.02 * (1.0 if task.kind == TaskKind.EMERGENCY else 0.0)
                - 0.30 * normal_pressure * (1.0 if task.kind == TaskKind.EMERGENCY and (not bool(island_bonus_eff)) else 0.0)
            )
            + keep
        )

    def _truck_motion_heading_xy(self, env, truck_id: str) -> Optional[Tuple[float, float]]:
        ts = env.state.agents.get(str(truck_id), None)
        if ts is None or ts.kind != AgentKind.TRUCK:
            return None
        tr = getattr(ts, "transit", None)
        if tr is None:
            return None
        try:
            src, dst, _ = tr
            p0 = env._node_xy(int(src))
            p1 = env._node_xy(int(dst))
            vx = float(p1[0]) - float(p0[0])
            vy = float(p1[1]) - float(p0[1])
            norm = float(np.hypot(vx, vy))
            if norm <= 1e-9:
                return None
            return (float(vx / norm), float(vy / norm))
        except Exception:
            return None

    def _depot_xy(self, env) -> Tuple[float, float]:
        try:
            p = env._node_xy(0)
            return (float(p[0]), float(p[1]))
        except Exception:
            return (0.0, 0.0)

    def _initial_route_dispatch_enabled(self, env) -> bool:
        return bool(getattr(env.cfg, "hrl_initial_route_dispatch_enabled", True))

    def _depot_outlet_nodes(self, env) -> List[int]:
        if not self._initial_route_dispatch_enabled(env):
            return []
        try:
            nodes = list(env._decision_neighbors(0))
        except Exception:
            try:
                nodes = list(env.topology.neighbors(0))
            except Exception:
                nodes = []
        return sorted(int(n) for n in nodes if int(n) != 0)

    def _task_depot_outlet_bucket(self, env, task, outlets: Optional[List[int]] = None) -> Optional[int]:
        if task is None:
            return None
        outlet_nodes = list(outlets) if outlets is not None else self._depot_outlet_nodes(env)
        if len(outlet_nodes) < 2:
            return None
        try:
            task_node = int(task.demand_node)
        except Exception:
            return None
        best_idx: Optional[int] = None
        best_key: Tuple[float, float, int] = (float("inf"), float("inf"), 10**9)
        try:
            tx, ty = env._node_xy(int(task_node))
        except Exception:
            tx, ty = (0.0, 0.0)
        for idx, outlet in enumerate(outlet_nodes):
            try:
                sp = float(env._decision_shortest_path_distance(int(outlet), int(task_node)))
            except Exception:
                sp = float("inf")
            if not np.isfinite(sp):
                continue
            try:
                ox, oy = env._node_xy(int(outlet))
                dxy = float(np.hypot(float(tx) - float(ox), float(ty) - float(oy)))
            except Exception:
                dxy = float("inf")
            key = (float(sp), float(dxy), int(idx))
            if key < best_key:
                best_key = key
                best_idx = int(idx)
        return best_idx

    def _truck_route_lookahead_steps(self, env) -> int:
        base_steps = int(max(getattr(env.cfg, "hrl_truck_task_lookahead_steps", 3), 1))
        max_steps = int(max(getattr(env.cfg, "hrl_truck_task_lookahead_max_steps", base_steps), base_steps))
        phase = str(getattr(env.cfg, "phase", "")).strip().upper()
        map_size = float(max(getattr(env.cfg, "map_size_m", 0.0), 0.0))
        if phase in {"L", "R"} or map_size >= 10000.0:
            return int(min(max_steps, max(base_steps, 5)))
        return int(min(max_steps, max(base_steps, 3)))

    def _goal_target_node(self, env, goal_id: Optional[str]) -> Optional[int]:
        if goal_id is None:
            return None
        task = env.state.tasks.get(str(goal_id), None)
        if task is not None and task.status == TaskStatus.PENDING:
            try:
                return int(task.demand_node)
            except Exception:
                return None
        ag = env.state.agents.get(str(goal_id), None)
        if ag is None:
            return None
        try:
            return int(ag.node or 0)
        except Exception:
            return None

    def _predict_truck_node_after_steps(
        self,
        env,
        truck_id: str,
        *,
        steps: Optional[int] = None,
        toward_node: Optional[int] = None,
    ) -> Optional[int]:
        truck = env.state.agents.get(str(truck_id), None)
        if truck is None or truck.kind != AgentKind.TRUCK or bool(getattr(truck, "crashed", False)):
            return None
        cur_node = int(getattr(truck, "node", 0) or 0)
        if toward_node is None:
            goal_id = self.state.goals.get(str(truck_id), None)
            if goal_id is None and hasattr(env, "_effective_goals"):
                goal_id = env._effective_goals.get(str(truck_id), env._recommended_goals.get(str(truck_id), None))
            toward_node = self._goal_target_node(env, goal_id)
        if toward_node is None or int(toward_node) == int(cur_node):
            return int(cur_node)

        look_steps = int(max(steps if steps is not None else self._truck_route_lookahead_steps(env), 1))
        travel_budget_m = float(max(getattr(env.cfg, "truck_speed_mps", 0.0), 0.0)) * float(
            max(getattr(env.cfg, "dt_seconds", 0.0), 0.0)
        ) * float(look_steps)
        node_now = int(cur_node)
        remaining = float(max(travel_budget_m, 0.0))
        for _ in range(look_steps):
            try:
                cur_dist = float(env._decision_shortest_path_distance(int(node_now), int(toward_node)))
            except Exception:
                break
            best_nb: Optional[int] = None
            best_nb_dist = float(cur_dist)
            best_edge_len = float("inf")
            for nb in env._decision_neighbors(int(node_now)):
                try:
                    nb_dist = float(env._decision_shortest_path_distance(int(nb), int(toward_node)))
                    edge_len = float(env._decision_shortest_path_distance(int(node_now), int(nb)))
                except Exception:
                    continue
                if not np.isfinite(nb_dist) or not np.isfinite(edge_len):
                    continue
                if nb_dist + 1e-6 >= cur_dist:
                    continue
                cand = (float(nb_dist), float(edge_len), int(nb))
                best = (float(best_nb_dist), float(best_edge_len), int(best_nb) if best_nb is not None else 10**9)
                if cand < best:
                    best_nb = int(nb)
                    best_nb_dist = float(nb_dist)
                    best_edge_len = float(edge_len)
            if best_nb is None:
                break
            if np.isfinite(best_edge_len) and remaining > 1e-6 and best_edge_len > remaining + 1e-6:
                break
            node_now = int(best_nb)
            remaining = float(max(remaining - float(best_edge_len), 0.0))
            if int(node_now) == int(toward_node):
                break
        return int(node_now)

    def _truck_route_progress_to_task(self, env, truck_id: str, task) -> float:
        truck = env.state.agents.get(str(truck_id), None)
        if truck is None or truck.kind != AgentKind.TRUCK or task is None:
            return 0.0
        try:
            cur_node = int(getattr(truck, "node", 0) or 0)
            task_node = int(task.demand_node)
            cur_dist = float(env._decision_shortest_path_distance(int(cur_node), int(task_node)))
        except Exception:
            return 0.0
        if not np.isfinite(cur_dist) or cur_dist <= 1e-6:
            return 0.0
        toward_node = self._goal_target_node(env, self.state.goals.get(str(truck_id), None))
        if toward_node is None and hasattr(env, "_effective_goals"):
            toward_node = self._goal_target_node(
                env,
                env._effective_goals.get(str(truck_id), env._recommended_goals.get(str(truck_id), None)),
            )
        if toward_node is None:
            toward_node = int(task_node)
        future_node = self._predict_truck_node_after_steps(env, str(truck_id), toward_node=int(toward_node))
        if future_node is None:
            return 0.0
        try:
            future_dist = float(env._decision_shortest_path_distance(int(future_node), int(task_node)))
        except Exception:
            return 0.0
        if not np.isfinite(future_dist):
            return 0.0
        return float(np.clip((float(cur_dist) - float(future_dist)) / max(float(cur_dist), 1.0), -1.0, 1.0))

    def _uav_anchor_task(self, env, aid: str):
        current_goal = self.state.goals.get(str(aid), None)
        current_task = env.state.tasks.get(str(current_goal), None) if current_goal is not None else None
        if current_task is not None and current_task.kind == TaskKind.EMERGENCY and current_task.status == TaskStatus.PENDING:
            return current_task
        anchor_id = self._uav_anchor_task_goal.get(str(aid), "")
        anchor_task = env.state.tasks.get(str(anchor_id), None) if anchor_id else None
        if anchor_task is not None and anchor_task.kind == TaskKind.EMERGENCY and anchor_task.status == TaskStatus.PENDING:
            return anchor_task
        return None

    def _prune_uav_anchor_tasks(self, env) -> None:
        for aid, task_id in list(self._uav_anchor_task_goal.items()):
            task = env.state.tasks.get(str(task_id), None)
            if task is None or task.kind != TaskKind.EMERGENCY or task.status != TaskStatus.PENDING:
                self._uav_anchor_task_goal.pop(str(aid), None)

    def _uav_task_prelaunch_assignable(self, env, aid: str, task) -> bool:
        st = env.state.agents.get(str(aid), None)
        if st is None or st.kind != AgentKind.UAV or bool(getattr(st, "crashed", False)):
            return False
        if task is None or task.kind != TaskKind.EMERGENCY or task.status != TaskStatus.PENDING:
            return False
        if st.follow_target is None:
            return False
        if self._uav_task_feasible(env, str(aid), task):
            return True
        if not bool(getattr(env.cfg, "uav_docked_keep_task_goal_enabled", True)):
            return False
        if bool(getattr(st, "uav_needs_reload_flag", False)):
            return False
        loaded_fn = getattr(env, "_uav_loaded", None)
        if callable(loaded_fn):
            try:
                if not bool(loaded_fn(str(aid))):
                    return False
            except Exception:
                return False
        progress = float(self._truck_route_progress_to_task(env, str(st.follow_target), task))
        urgency = float(self._norm_deadline_urgency(task, int(env.state.step_index)))
        lifeline_ratio = float(self._task_lifeline_ratio(task)) if self._is_timecritical_lightweight_task(task) else 1.0
        assign_score = float(0.62 * max(progress, 0.0) + 0.24 * urgency + 0.14 * (1.0 - lifeline_ratio))
        min_score = float(max(getattr(env.cfg, "hrl_uav_docked_prelaunch_assign_min_score", 0.16), 0.0))
        return bool(assign_score >= min_score)

    def _refresh_uav_anchor_tasks(self, env, goals: Dict[str, Optional[str]]) -> None:
        self._prune_uav_anchor_tasks(env)
        for aid, gid in goals.items():
            st = env.state.agents.get(str(aid), None)
            if st is None or st.kind != AgentKind.UAV:
                continue
            task = env.state.tasks.get(str(gid), None) if gid is not None else None
            if task is not None and task.kind == TaskKind.EMERGENCY and task.status == TaskStatus.PENDING:
                self._uav_anchor_task_goal[str(aid)] = str(task.task_id)

    def _uav_task_transfer_enabled(self, env) -> bool:
        return bool(getattr(env.cfg, "hrl_uav_task_transfer_enabled", True))

    def _candidate_transfer_truck_for_uav_task(self, env, aid: str, task) -> Optional[str]:
        if not self._uav_task_transfer_enabled(env):
            return None
        st = env.state.agents.get(str(aid), None)
        if st is None or st.kind != AgentKind.UAV or bool(getattr(st, "crashed", False)):
            return None
        if st.follow_target is None:
            return None
        if task is None or task.kind != TaskKind.EMERGENCY or task.status != TaskStatus.PENDING:
            return None
        if bool(getattr(st, "uav_needs_reload_flag", False)):
            return None
        loaded_fn = getattr(env, "_uav_loaded", None)
        if callable(loaded_fn):
            try:
                if not bool(loaded_fn(str(aid))):
                    return None
            except Exception:
                return None
        docked_actionable_fn = getattr(env, "_uav_docked_task_actionable_now", None)
        if callable(docked_actionable_fn):
            try:
                if bool(docked_actionable_fn(str(aid), task)):
                    return None
            except Exception:
                pass

        current_tid = str(getattr(st, "follow_target", ""))
        current_agent = env.state.agents.get(str(current_tid), None)
        if current_agent is None or current_agent.kind != AgentKind.TRUCK or bool(getattr(current_agent, "crashed", False)):
            return None

        ax, ay = self._agent_xy(env, str(aid))
        max_target_dist = float(max(getattr(env.cfg, "hrl_uav_task_transfer_max_target_dist_m", 2600.0), 0.0))
        score_gain_min = float(max(getattr(env.cfg, "hrl_uav_task_transfer_score_gain_min", 0.22), 0.0))
        progress_gain_min = float(max(getattr(env.cfg, "hrl_uav_task_transfer_progress_gain_min", 0.18), 0.0))

        current_progress = float(self._truck_route_progress_to_task(env, current_tid, task))
        current_tx, current_ty = self._agent_xy(env, current_tid)
        current_dist = float(np.hypot(float(current_tx) - float(ax), float(current_ty) - float(ay)))
        current_score = float(self._score_uav_truck(env, str(aid), current_tid, current_dist))

        pred_current_node = self._predict_truck_node_after_steps(env, current_tid)
        current_future_actionable = False
        actionable_from_node_fn = getattr(env, "_uav_sortie_chain_actionable_from_truck_node", None)
        if callable(actionable_from_node_fn) and pred_current_node is not None:
            try:
                current_future_actionable = bool(actionable_from_node_fn(str(aid), int(pred_current_node), task))
            except Exception:
                current_future_actionable = False

        best_tid: Optional[str] = None
        best_score = float(current_score)
        for tid, ag in env.state.agents.items():
            if ag.kind != AgentKind.TRUCK or bool(getattr(ag, "crashed", False)):
                continue
            tid_s = str(tid)
            if tid_s == current_tid:
                continue
            slot_fn = getattr(env, "_truck_has_follow_slot", None)
            if callable(slot_fn):
                try:
                    if not bool(slot_fn(str(tid_s), exclude_aid=str(aid))):
                        continue
                except Exception:
                    pass
            tx, ty = self._agent_xy(env, tid_s)
            d = float(np.hypot(float(tx) - float(ax), float(ty) - float(ay)))
            if max_target_dist > 0.0 and d > max_target_dist:
                continue
            progress = float(self._truck_route_progress_to_task(env, tid_s, task))
            if progress + 1e-9 < current_progress + progress_gain_min:
                continue
            pred_node = self._predict_truck_node_after_steps(env, tid_s)
            future_actionable = False
            if callable(actionable_from_node_fn) and pred_node is not None:
                try:
                    future_actionable = bool(actionable_from_node_fn(str(aid), int(pred_node), task))
                except Exception:
                    future_actionable = False
            transfer_bonus = 0.0
            if future_actionable and (not current_future_actionable):
                transfer_bonus += 0.40
            sc = float(self._score_uav_truck(env, str(aid), tid_s, d) + transfer_bonus)
            if sc < current_score + score_gain_min - 1e-9:
                continue
            if sc > best_score + 1e-12:
                best_score = float(sc)
                best_tid = str(tid_s)
        return best_tid

    def _refresh_uav_transfer_hints(self, env, goals: Dict[str, Optional[str]]) -> None:
        self._uav_transfer_target_truck.clear()
        self._uav_transfer_target_task.clear()
        for aid, gid in goals.items():
            st = env.state.agents.get(str(aid), None)
            if st is None or st.kind != AgentKind.UAV:
                continue
            task = env.state.tasks.get(str(gid), None) if gid is not None else None
            if task is None or task.kind != TaskKind.EMERGENCY or task.status != TaskStatus.PENDING:
                continue
            best_tid = self._candidate_transfer_truck_for_uav_task(env, str(aid), task)
            if best_tid is None:
                continue
            self._uav_transfer_target_truck[str(aid)] = str(best_tid)
            self._uav_transfer_target_task[str(aid)] = str(task.task_id)
            self.uav_transfer_hint_issue_count_total += 1


    def _initial_directional_cover_enabled(self, env) -> bool:
        return bool(getattr(env.cfg, "hrl_initial_directional_cover_enabled", True))

    def _initial_directional_window_steps(self, env) -> int:
        return int(max(getattr(env.cfg, "hrl_initial_directional_window_steps", 10), 0))

    def _initial_directional_sector_count(self, env) -> int:
        base = int(max(getattr(env.cfg, "hrl_initial_directional_sector_count", 4), 2))
        outlets = self._depot_outlet_nodes(env)
        if len(outlets) >= 2:
            return int(max(base, len(outlets)))
        return int(base)

    def _in_initial_directional_phase(self, env) -> bool:
        if not self._initial_directional_cover_enabled(env):
            return False
        return int(getattr(env.state, "step_index", 0)) <= int(self._initial_directional_window_steps(env))

    def _task_sector_index(self, env, task, sectors: Optional[int] = None) -> int:
        if task is None:
            return -1
        outlets = self._depot_outlet_nodes(env)
        bucket = self._task_depot_outlet_bucket(env, task, outlets=outlets) if len(outlets) >= 2 else None
        if bucket is not None:
            return int(bucket)
        try:
            node = env.topology.nodes[int(task.demand_node)]
            depx, depy = self._depot_xy(env)
            dx = float(node.x) - float(depx)
            dy = float(node.y) - float(depy)
            if abs(dx) <= 1e-9 and abs(dy) <= 1e-9:
                return 0
            sc = int(sectors) if sectors is not None else int(self._initial_directional_sector_count(env))
            ang = float(np.arctan2(dy, dx))
            aa = (ang + 2.0 * np.pi) % (2.0 * np.pi)
            return int(np.floor(sc * aa / (2.0 * np.pi))) % max(sc, 1)
        except Exception:
            return -1

    def _build_initial_directional_plan(self, env) -> None:
        step_now = int(getattr(env.state, "step_index", 0))
        if not self._in_initial_directional_phase(env):
            self._initial_directional_plan_step = -1
            self._initial_directional_truck_sector.clear()
            self._initial_directional_uav_truck.clear()
            self._initial_directional_sector_stats.clear()
            return
        if self._initial_directional_plan_step >= 0:
            return

        sectors = int(self._initial_directional_sector_count(env))
        w_norm = float(max(getattr(env.cfg, "hrl_initial_directional_normal_weight", 1.0), 0.0))
        w_emer = float(max(getattr(env.cfg, "hrl_initial_directional_emergency_weight", 2.2), 0.0))
        w_urg = float(max(getattr(env.cfg, "hrl_initial_directional_urgency_weight", 0.35), 0.0))
        dup_pen = float(max(getattr(env.cfg, "hrl_initial_directional_duplicate_penalty", 0.60), 0.0))

        sector_stats: Dict[int, Dict[str, float]] = {
            int(i): {
                "normal_count": 0.0,
                "emergency_count": 0.0,
                "weighted": 0.0,
                "urgency_sum": 0.0,
            }
            for i in range(sectors)
        }
        pending_tasks = [t for t in env.state.tasks.values() if self._task_planner_active(t)]
        for task in pending_tasks:
            sidx = int(self._task_sector_index(env, task, sectors))
            if sidx < 0:
                continue
            urg = float(self._norm_deadline_urgency(task, step_now))
            if task.kind == TaskKind.EMERGENCY:
                sector_stats[sidx]["emergency_count"] += 1.0
                base_w = w_emer
            else:
                sector_stats[sidx]["normal_count"] += 1.0
                base_w = w_norm
            sector_stats[sidx]["urgency_sum"] += urg
            sector_stats[sidx]["weighted"] += float(base_w + w_urg * urg)

        trucks = [
            str(aid)
            for aid, st in sorted(env.state.agents.items(), key=lambda kv: str(kv[0]))
            if st.kind == AgentKind.TRUCK and (not bool(getattr(st, "crashed", False)))
        ]
        uavs = [
            str(aid)
            for aid, st in sorted(env.state.agents.items(), key=lambda kv: str(kv[0]))
            if st.kind == AgentKind.UAV and (not bool(getattr(st, "crashed", False)))
        ]

        truck_sector: Dict[str, int] = {}
        used_sector_count: Dict[int, int] = {int(i): 0 for i in range(sectors)}
        non_empty_sectors = [int(i) for i in range(sectors) if float(sector_stats.get(int(i), {}).get("weighted", 0.0)) > 1e-9]
        non_empty_sectors.sort(key=lambda s: float(sector_stats.get(int(s), {}).get("weighted", 0.0)), reverse=True)

        for tid_idx, tid in enumerate(trucks):
            preferred_sector = int(non_empty_sectors[tid_idx]) if tid_idx < len(non_empty_sectors) else None
            best_sector = int(preferred_sector) if preferred_sector is not None else 0
            best_score = -1e18

            # Stronger early directional split on large maps: if there are still
            # non-empty sectors not taken by any truck, evaluate those first.
            unused_non_empty = [
                int(sidx)
                for sidx in non_empty_sectors
                if int(used_sector_count.get(int(sidx), 0)) <= 0
            ]
            sector_iter: List[int] = list(unused_non_empty) if unused_non_empty else [int(sidx) for sidx in range(sectors)]

            for sidx in sector_iter:
                sec_w = float(sector_stats[int(sidx)]["weighted"])
                if sec_w <= 0.0:
                    continue
                svc_w = 0.0
                for task in pending_tasks:
                    if int(self._task_sector_index(env, task, sectors)) != int(sidx):
                        continue
                    if not self._truck_task_serviceable_or_support_proxy(env, str(tid), task):
                        continue
                    if not self._truck_task_reachable(env, str(tid), task):
                        continue
                    if task.kind == TaskKind.EMERGENCY:
                        svc_w += float(w_emer + 0.25 * w_urg * self._norm_deadline_urgency(task, step_now))
                    else:
                        svc_w += float(w_norm + 0.25 * w_urg * self._norm_deadline_urgency(task, step_now))
                if svc_w <= 0.0:
                    svc_w = 0.35 * sec_w
                dup = float(used_sector_count.get(int(sidx), 0))
                score = float(0.70 * svc_w + 0.30 * sec_w - dup_pen * dup * max(sec_w, 1.0))
                if preferred_sector is not None and int(sidx) == int(preferred_sector):
                    score += float(0.25 * max(sec_w, 1.0))
                if score > best_score + 1e-12:
                    best_score = float(score)
                    best_sector = int(sidx)
            truck_sector[str(tid)] = int(best_sector)
            used_sector_count[int(best_sector)] = int(used_sector_count.get(int(best_sector), 0) + 1)

        uav_truck: Dict[str, str] = {}
        sector_to_trucks: Dict[int, List[str]] = {int(i): [] for i in range(sectors)}
        for tid in trucks:
            sector_to_trucks[int(truck_sector.get(str(tid), 0))].append(str(tid))

        uav_count = int(len(uavs))
        emergency_total = float(sum(float(v.get("emergency_count", 0.0)) for v in sector_stats.values()))
        quota_by_sector: Dict[int, int] = {int(i): 0 for i in range(sectors)}
        if uav_count > 0:
            if emergency_total > 1e-9:
                raw = {
                    int(i): float(sector_stats.get(int(i), {}).get("emergency_count", 0.0)) / emergency_total
                    for i in range(sectors)
                }
                for i in range(sectors):
                    if float(raw[int(i)]) > 1e-9:
                        quota_by_sector[int(i)] = int(max(1, int(np.floor(uav_count * float(raw[int(i)])))))
                while int(sum(quota_by_sector.values())) > uav_count:
                    k = max(quota_by_sector.keys(), key=lambda kk: quota_by_sector[int(kk)])
                    if quota_by_sector[int(k)] <= 0:
                        break
                    quota_by_sector[int(k)] -= 1
                while int(sum(quota_by_sector.values())) < uav_count:
                    k = max(range(sectors), key=lambda kk: float(raw[int(kk)]) - float(quota_by_sector[int(kk)]) / max(uav_count, 1))
                    quota_by_sector[int(k)] += 1
            else:
                for i in range(uav_count):
                    quota_by_sector[int(i % sectors)] += 1

        assigned_by_sector: Dict[int, int] = {int(i): 0 for i in range(sectors)}
        per_truck_cap_cfg = int(
            max(
                getattr(
                    env.cfg,
                    "hrl_initial_directional_uav_per_truck_cap",
                    getattr(env.cfg, "uav_max_followers_per_truck", 2),
                ),
                0,
            )
        )
        if per_truck_cap_cfg <= 0:
            per_truck_cap_cfg = int(max(1, int(np.ceil(float(max(uav_count, 1)) / float(max(len(trucks), 1))))))
        assigned_by_truck: Dict[str, int] = {str(tid): 0 for tid in trucks}

        for uid in uavs:
            unmet = [int(s) for s in range(sectors) if assigned_by_sector[int(s)] < quota_by_sector.get(int(s), 0)]
            if unmet:
                choose_sector = max(unmet, key=lambda s: float(sector_stats.get(int(s), {}).get("emergency_count", 0.0)))
            else:
                choose_sector = max(range(sectors), key=lambda s: float(sector_stats.get(int(s), {}).get("emergency_count", 0.0)))

            trucks_in_sector = list(sector_to_trucks.get(int(choose_sector), []))
            if not trucks_in_sector:
                trucks_in_sector = list(trucks)
            trucks_available = [
                str(tid)
                for tid in trucks_in_sector
                if int(assigned_by_truck.get(str(tid), 0)) < int(per_truck_cap_cfg)
            ]
            if trucks_available:
                trucks_in_sector = trucks_available
            else:
                # Sector-local trucks are full: fallback to global trucks with remaining slots
                # instead of silently ignoring cap and over-concentrating UAV startup staging.
                global_available = [
                    str(tid)
                    for tid in trucks
                    if int(assigned_by_truck.get(str(tid), 0)) < int(per_truck_cap_cfg)
                ]
                if global_available:
                    trucks_in_sector = global_available

            best_tid: Optional[str] = None
            best_sc = -1e18
            for tid in trucks_in_sector:
                sidx = int(truck_sector.get(str(tid), 0))
                emer_pressure = float(sector_stats.get(int(sidx), {}).get("emergency_count", 0.0))
                sec_w = float(sector_stats.get(int(sidx), {}).get("weighted", 0.0))
                pull = float(self._truck_emergency_pull_score(env, str(tid)))
                bal = float(self._truck_follower_balance_score(env, str(uid), str(tid)))
                crowd_pen = 0.12 * float(assigned_by_sector.get(int(sidx), 0))
                sc = float(0.50 * emer_pressure + 0.22 * sec_w + 0.18 * pull + 0.10 * bal - crowd_pen)
                if sc > best_sc + 1e-12:
                    best_sc = float(sc)
                    best_tid = str(tid)
            if best_tid is not None:
                uav_truck[str(uid)] = str(best_tid)
                sidx = int(truck_sector.get(str(best_tid), 0))
                assigned_by_sector[int(sidx)] = int(assigned_by_sector.get(int(sidx), 0) + 1)
                assigned_by_truck[str(best_tid)] = int(assigned_by_truck.get(str(best_tid), 0) + 1)

        self._initial_directional_plan_step = int(step_now)
        self._initial_directional_truck_sector = dict(truck_sector)
        self._initial_directional_uav_truck = dict(uav_truck)
        self._initial_directional_sector_stats = dict(sector_stats)

    def _publish_runtime_sidechannels(self, env) -> None:
        try:
            env._uav_transfer_target_truck = dict(self._uav_transfer_target_truck)
            env._uav_transfer_target_task = dict(self._uav_transfer_target_task)
            env._planner_truck_assist_waypoint_by_truck = dict(self._truck_assist_waypoint_by_truck)
        except Exception:
            pass

    def _event_admission_gate_enabled(self, env) -> bool:
        return bool(getattr(env.cfg, "hrl_event_admission_gate_enabled", False))

    def _noop_event_cooldown_enabled(self, env) -> bool:
        return bool(getattr(env.cfg, "hrl_noop_event_cooldown_enabled", False))

    def _noop_event_cooldown_steps(self, env) -> int:
        return int(max(getattr(env.cfg, "hrl_noop_event_cooldown_steps", 10), 1))

    def _event_lifeline_threshold_ratio(self, env) -> float:
        return float(np.clip(getattr(env.cfg, "hrl_event_admission_lifeline_threshold_ratio", 0.40), 0.0, 1.0))

    def _weak_event_reason_names(self) -> set:
        return {
            "arrival",
            "resolution",
            "uav_idle",
            "truck_idle",
            "map_update_light",
            "ranking_changed",
            "noncritical_map_update",
        }

    def _hard_event_reason_names(self) -> set:
        return {
            "goal_invalid",
            "goal_terminal",
            "path_blocked",
            "goal_unreachable",
            "uav_safety",
            "truck_dead_end",
            "high_priority_uncovered",
            "normal_stall",
        }

    def _uncovered_low_lifeline_emergency_exists(self, env) -> bool:
        thr = float(self._event_lifeline_threshold_ratio(env))
        covered = set()
        for _, gid in self.state.goals.items():
            if gid is None:
                continue
            t = env.state.tasks.get(str(gid), None)
            if t is not None and t.status == TaskStatus.PENDING and t.kind == TaskKind.EMERGENCY:
                covered.add(str(t.task_id))
        for task in env.state.tasks.values():
            if task.kind != TaskKind.EMERGENCY or task.status != TaskStatus.PENDING:
                continue
            if str(task.task_id) in covered:
                continue
            if float(np.clip(self._task_lifeline_ratio(task), 0.0, 1.0)) <= thr:
                return True
        return False

    def _event_reason_in_cooldown(self, env, reason: str, step_now: int) -> bool:
        if not self._noop_event_cooldown_enabled(env):
            return False
        until = int(self._noop_event_cooldown_until_step_by_reason.get(str(reason), -1))
        return bool(step_now <= until)

    def _record_event_refresh_reason_counts(self, flags: Dict[str, bool]) -> None:
        if bool(flags.get("arrival", False)):
            self.event_refresh_reason_arrival_count_total += 1
        if bool(flags.get("resolution", False)):
            self.event_refresh_reason_resolution_count_total += 1
        if bool(flags.get("uav_idle", False)):
            self.event_refresh_reason_uav_idle_count_total += 1
        if bool(flags.get("truck_idle", False)):
            self.event_refresh_reason_truck_idle_count_total += 1
        if bool(flags.get("map_update_light", False)):
            self.event_refresh_reason_map_update_light_count_total += 1
        if bool(flags.get("map_update_hard_actionable", False)) or bool(flags.get("map_update_hard_immediate_refresh", False)):
            self.event_refresh_reason_map_update_hard_count_total += 1
        if bool(flags.get("goal_invalid", False)) or bool(flags.get("goal_terminal", False)):
            self.event_refresh_reason_goal_invalid_count_total += 1
        if bool(flags.get("hard_reason_path_blocked", False)) or int(flags.get("map_update_hard_reason_path_blocked_step", 0)) > 0:
            self.event_refresh_reason_path_blocked_count_total += 1
        if bool(flags.get("hard_reason_goal_unreachable", False)) or int(flags.get("map_update_hard_reason_goal_unreachable_step", 0)) > 0:
            self.event_refresh_reason_goal_unreachable_count_total += 1
        if bool(flags.get("hard_reason_uav_safety", False)) or bool(flags.get("uav_emergency", False)):
            self.event_refresh_reason_uav_safety_count_total += 1
        if bool(flags.get("hard_reason_truck_dead_end", False)) or bool(flags.get("truck_dead_end", False)):
            self.event_refresh_reason_truck_dead_end_count_total += 1
        if bool(flags.get("hard_reason_high_priority_uncovered", False)) or bool(flags.get("high_priority_uncovered", False)):
            self.event_refresh_reason_high_priority_uncovered_count_total += 1
        if bool(flags.get("hard_reason_normal_stall", False)) or bool(flags.get("normal_stall", False)):
            self.event_refresh_reason_normal_stall_count_total += 1

    def _finalize_active_event_refresh_window(self, env, step_now: int) -> None:
        window = self._active_event_refresh_window
        if not isinstance(window, dict):
            return
        start_step = int(window.get("step", step_now))
        if int(step_now) <= start_step:
            return
        launch0 = float(window.get("launch0", 0.0))
        comp0 = float(window.get("comp0", 0.0))
        rej0 = float(window.get("rej0", 0.0))
        comp_resolved0 = float(window.get("resolved0", 0.0))
        launch_now = float(getattr(env, "uav_launch_count_total", 0.0))
        comp_now = float(getattr(env, "uav_delivery_count_total", 0.0))
        rej_now = float(getattr(env, "uav_unsafe_launch_attempt_count_total", 0.0))
        comp_resolved_now = float(self._resolved_count(env))
        d_launch = float(max(launch_now - launch0, 0.0))
        d_comp = float(max(comp_now - comp0, 0.0))
        d_rej = float(max(rej_now - rej0, 0.0))
        d_resolved = float(max(comp_resolved_now - comp_resolved0, 0.0))
        if d_launch > 1e-9:
            self.event_refresh_to_launch_count_total += 1
            if bool(window.get("is_hard", False)):
                self.hard_event_refresh_to_launch_count_total += 1
        if d_comp > 1e-9:
            self.event_refresh_to_completion_count_total += 1
            if bool(window.get("is_hard", False)):
                self.hard_event_refresh_to_completion_count_total += 1
        if d_rej > 1e-9:
            self.event_refresh_followed_by_reject_count_total += 1
            if bool(window.get("is_hard", False)):
                self.hard_event_refresh_followed_by_reject_count_total += 1
        no_progress = bool(d_launch <= 1e-9 and d_comp <= 1e-9 and d_resolved <= 1e-9)
        if no_progress:
            self.event_refresh_followed_by_stall_count_total += 1
            if bool(window.get("is_hard", False)):
                self.hard_event_refresh_followed_by_stall_count_total += 1
        if bool(window.get("is_hard", False)):
            hard_reasons = [str(x) for x in list(window.get("hard_reasons", []))]
            no_goal_change = bool(window.get("no_goal_change", False))
            seen_reasons: set = set()
            for reason in hard_reasons:
                if not reason:
                    continue
                if reason in seen_reasons:
                    continue
                seen_reasons.add(reason)
                rrec = self._hard_event_reason_outcome_stats.get(
                    reason,
                    {
                        "hard_event_reason": str(reason),
                        "total_refresh_count": 0.0,
                        "no_goal_change_count": 0.0,
                        "goal_change_count": 0.0,
                        "followed_by_launch_count": 0.0,
                        "followed_by_completion_count": 0.0,
                        "followed_by_reject_count": 0.0,
                        "followed_by_stall_count": 0.0,
                    },
                )
                rrec["total_refresh_count"] = float(rrec.get("total_refresh_count", 0.0) + 1.0)
                if no_goal_change:
                    rrec["no_goal_change_count"] = float(rrec.get("no_goal_change_count", 0.0) + 1.0)
                else:
                    rrec["goal_change_count"] = float(rrec.get("goal_change_count", 0.0) + 1.0)
                if d_launch > 1e-9:
                    rrec["followed_by_launch_count"] = float(rrec.get("followed_by_launch_count", 0.0) + 1.0)
                if d_comp > 1e-9:
                    rrec["followed_by_completion_count"] = float(rrec.get("followed_by_completion_count", 0.0) + 1.0)
                if d_rej > 1e-9:
                    rrec["followed_by_reject_count"] = float(rrec.get("followed_by_reject_count", 0.0) + 1.0)
                if no_progress:
                    rrec["followed_by_stall_count"] = float(rrec.get("followed_by_stall_count", 0.0) + 1.0)
                self._hard_event_reason_outcome_stats[str(reason)] = rrec

            hard_offenders = list(window.get("hard_offenders", []))
            for off in hard_offenders:
                if not isinstance(off, dict):
                    continue
                reason = str(off.get("reason", "unknown"))
                aid = str(off.get("agent_id", ""))
                tid = str(off.get("task_id", ""))
                key = (aid, tid, reason)
                rec = self._hard_event_offender_stats.get(
                    key,
                    {
                        "reason": reason,
                        "agent_id": aid,
                        "task_id": tid,
                        "count": 0.0,
                        "no_goal_change_count": 0.0,
                        "goal_change_count": 0.0,
                        "first_step": float(start_step),
                        "last_step": float(start_step),
                        "launch_count_after_event": 0.0,
                        "completion_count_after_event": 0.0,
                        "reject_count_after_event": 0.0,
                        "goal_switch_after_event": 0.0,
                        "current_goal_type": str(off.get("current_goal_type", "")),
                        "proposed_goal_type": str(off.get("proposed_goal_type", "")),
                        "task_status": str(off.get("task_status", "")),
                        "distance_to_goal_sum": float(off.get("distance_to_goal", 0.0) or 0.0),
                        "distance_to_goal_count": 1.0 if np.isfinite(float(off.get("distance_to_goal", np.nan))) else 0.0,
                        "battery_sum": float(off.get("battery", 0.0) or 0.0),
                        "battery_count": 1.0 if np.isfinite(float(off.get("battery", np.nan))) else 0.0,
                    },
                )
                rec["count"] = float(rec.get("count", 0.0) + 1.0)
                if bool(window.get("no_goal_change", False)):
                    rec["no_goal_change_count"] = float(rec.get("no_goal_change_count", 0.0) + 1.0)
                else:
                    rec["goal_change_count"] = float(rec.get("goal_change_count", 0.0) + 1.0)
                rec["first_step"] = float(min(float(rec.get("first_step", start_step)), float(start_step)))
                rec["last_step"] = float(max(float(rec.get("last_step", start_step)), float(start_step)))
                if d_launch > 1e-9:
                    rec["launch_count_after_event"] = float(rec.get("launch_count_after_event", 0.0) + 1.0)
                if d_comp > 1e-9:
                    rec["completion_count_after_event"] = float(rec.get("completion_count_after_event", 0.0) + 1.0)
                if d_rej > 1e-9:
                    rec["reject_count_after_event"] = float(rec.get("reject_count_after_event", 0.0) + 1.0)
                d_switch = float(window.get("goal_switch_after_event", 0.0))
                if d_switch > 0.0:
                    rec["goal_switch_after_event"] = float(rec.get("goal_switch_after_event", 0.0) + d_switch)
                d_goal = float(off.get("distance_to_goal", np.nan))
                if np.isfinite(d_goal):
                    rec["distance_to_goal_sum"] = float(rec.get("distance_to_goal_sum", 0.0) + d_goal)
                    rec["distance_to_goal_count"] = float(rec.get("distance_to_goal_count", 0.0) + 1.0)
                d_batt = float(off.get("battery", np.nan))
                if np.isfinite(d_batt):
                    rec["battery_sum"] = float(rec.get("battery_sum", 0.0) + d_batt)
                    rec["battery_count"] = float(rec.get("battery_count", 0.0) + 1.0)
                if (not str(rec.get("current_goal_type", ""))) and str(off.get("current_goal_type", "")):
                    rec["current_goal_type"] = str(off.get("current_goal_type", ""))
                if (not str(rec.get("proposed_goal_type", ""))) and str(off.get("proposed_goal_type", "")):
                    rec["proposed_goal_type"] = str(off.get("proposed_goal_type", ""))
                if str(off.get("task_status", "")):
                    rec["task_status"] = str(off.get("task_status", ""))
                self._hard_event_offender_stats[key] = rec
        # Weak-event no-op cooldown (only when no-goal-change and no progress).
        weak_reasons = list(window.get("weak_reasons", []))
        no_goal_change = bool(window.get("no_goal_change", False))
        if weak_reasons and no_goal_change and no_progress and self._noop_event_cooldown_enabled(env):
            ttl = int(self._noop_event_cooldown_steps(env))
            until = int(start_step + ttl)
            for rr in weak_reasons:
                self._noop_event_cooldown_until_step_by_reason[str(rr)] = int(max(until, int(self._noop_event_cooldown_until_step_by_reason.get(str(rr), -1))))
        self._active_event_refresh_window = None

    def _start_event_refresh_window(
        self,
        env,
        step_now: int,
        weak_reasons: List[str],
        no_goal_change: bool,
        is_hard: bool = False,
        hard_reasons: Optional[List[str]] = None,
        hard_offenders: Optional[List[Dict[str, object]]] = None,
    ) -> None:
        self._active_event_refresh_window = {
            "step": int(step_now),
            "launch0": float(getattr(env, "uav_launch_count_total", 0.0)),
            "comp0": float(getattr(env, "uav_delivery_count_total", 0.0)),
            "rej0": float(getattr(env, "uav_unsafe_launch_attempt_count_total", 0.0)),
            "resolved0": float(self._resolved_count(env)),
            "weak_reasons": list(weak_reasons),
            "no_goal_change": bool(no_goal_change),
            "is_hard": bool(is_hard),
            "hard_reasons": list(hard_reasons or []),
            "hard_offenders": list(hard_offenders or []),
            "goal_switch_after_event": 0.0,
        }

    def _truck_initial_directional_cover_term(self, env, aid: str, task) -> float:
        if not self._in_initial_directional_phase(env):
            return 0.0
        self._build_initial_directional_plan(env)
        if not self._initial_directional_truck_sector:
            return 0.0
        assigned = self._initial_directional_truck_sector.get(str(aid), None)
        if assigned is None:
            return 0.0
        tsec = int(self._task_sector_index(env, task, self._initial_directional_sector_count(env)))
        if tsec < 0:
            return 0.0
        bonus = float(max(getattr(env.cfg, "hrl_initial_directional_task_bonus", 0.24), 0.0))
        mis = float(max(getattr(env.cfg, "hrl_initial_directional_task_mismatch_penalty", 0.08), 0.0))
        if int(tsec) == int(assigned):
            sw = float(self._initial_directional_sector_stats.get(int(tsec), {}).get("weighted", 0.0))
            all_w = [float(v.get("weighted", 0.0)) for v in self._initial_directional_sector_stats.values()]
            max_w = float(max(all_w)) if all_w else 0.0
            dens = float(np.clip(sw / max(max_w, 1e-6), 0.0, 1.0))
            return float(bonus * (0.65 + 0.35 * dens))
        # Circular sector distance penalty.
        sec_n = int(max(self._initial_directional_sector_count(env), 2))
        d = abs(int(tsec) - int(assigned))
        d = min(d, sec_n - d)
        return float(-mis * float(np.clip(d / max(sec_n / 2.0, 1.0), 0.0, 1.0)))

    def _uav_initial_truck_route_term(self, env, aid: str, truck_id: str) -> float:
        if not self._in_initial_directional_phase(env):
            return 0.0
        self._build_initial_directional_plan(env)
        pref = self._initial_directional_uav_truck.get(str(aid), "")
        if not pref:
            return 0.0
        bonus = float(max(getattr(env.cfg, "hrl_initial_directional_uav_truck_bonus", 0.26), 0.0))
        penalty = float(max(getattr(env.cfg, "hrl_initial_directional_uav_truck_mismatch_penalty", 0.10), 0.0))
        if str(pref) == str(truck_id):
            return float(bonus)
        return float(-penalty)

    def _uav_initial_task_direction_term(self, env, aid: str, task) -> float:
        if task is None or task.kind != TaskKind.EMERGENCY:
            return 0.0
        if not self._in_initial_directional_phase(env):
            return 0.0
        self._build_initial_directional_plan(env)
        pref_truck = self._initial_directional_uav_truck.get(str(aid), "")
        if not pref_truck:
            return 0.0
        pref_sector = self._initial_directional_truck_sector.get(str(pref_truck), None)
        if pref_sector is None:
            return 0.0
        tsec = int(self._task_sector_index(env, task, self._initial_directional_sector_count(env)))
        if tsec < 0:
            return 0.0
        bonus = float(max(getattr(env.cfg, "hrl_initial_directional_uav_task_bonus", 0.16), 0.0))
        if int(tsec) == int(pref_sector):
            emer_cnt = float(self._initial_directional_sector_stats.get(int(tsec), {}).get("emergency_count", 0.0))
            peak = 1.0
            if self._initial_directional_sector_stats:
                peak = max(float(v.get("emergency_count", 0.0)) for v in self._initial_directional_sector_stats.values())
                peak = max(peak, 1.0)
            return float(bonus * float(np.clip(0.65 + 0.35 * emer_cnt / peak, 0.0, 1.2)))
        return 0.0

    def _truck_directional_split_term(self, env, aid: str, task) -> float:
        """
        Early-step direction diversification:
        encourage different trucks to expand toward different sectors so they do
        not all chase the same first-wave tasks.
        """
        if not bool(getattr(env.cfg, "hrl_truck_directional_split_enabled", True)):
            return 0.0
        if int(env.state.step_index) > int(max(getattr(env.cfg, "hrl_truck_directional_split_steps", 24), 0)):
            return 0.0
        truck_ids = sorted(
            str(tid)
            for tid, st in env.state.agents.items()
            if st.kind == AgentKind.TRUCK and (not bool(getattr(st, "crashed", False)))
        )
        if len(truck_ids) <= 1 or str(aid) not in set(truck_ids):
            return 0.0
        try:
            idx = int(truck_ids.index(str(aid)))
        except ValueError:
            return 0.0

        depx, depy = self._depot_xy(env)
        node = env.topology.nodes[int(task.demand_node)]
        vx = float(node.x) - float(depx)
        vy = float(node.y) - float(depy)
        vnorm = float(np.hypot(vx, vy))
        if vnorm <= 1e-9:
            return 0.0
        ux, uy = float(vx / vnorm), float(vy / vnorm)

        n_tr = max(len(truck_ids), 1)
        theta = float(2.0 * np.pi * float(idx) / float(n_tr))
        ax, ay = float(np.cos(theta)), float(np.sin(theta))
        align = float(np.clip(ux * ax + uy * ay, -1.0, 1.0))

        # Only apply when this truck actually has pending work in its preferred
        # sector; otherwise avoid pushing it into empty space.
        has_sector_work = False
        for t in env.state.tasks.values():
            if t.status != TaskStatus.PENDING:
                continue
            if not self._truck_task_serviceable_or_support_proxy(env, str(aid), t):
                continue
            try:
                tn = env.topology.nodes[int(t.demand_node)]
                tvx = float(tn.x) - float(depx)
                tvy = float(tn.y) - float(depy)
                tnorm = float(np.hypot(tvx, tvy))
                if tnorm <= 1e-9:
                    continue
                tux, tuy = float(tvx / tnorm), float(tvy / tnorm)
                if float(tux * ax + tuy * ay) >= 0.20:
                    has_sector_work = True
                    break
            except Exception:
                continue
        if not bool(has_sector_work):
            return 0.0
        w = float(max(getattr(env.cfg, "hrl_truck_directional_split_bonus", 0.16), 0.0))
        return float(w * align)

    def _point_sector_index(self, env, xy: Tuple[float, float], sectors: Optional[int] = None) -> int:
        try:
            depx, depy = self._depot_xy(env)
            dx = float(xy[0]) - float(depx)
            dy = float(xy[1]) - float(depy)
            if abs(dx) <= 1e-9 and abs(dy) <= 1e-9:
                return 0
            sc = int(sectors) if sectors is not None else int(self._initial_directional_sector_count(env))
            ang = float(np.arctan2(dy, dx))
            aa = (ang + 2.0 * np.pi) % (2.0 * np.pi)
            return int(np.floor(sc * aa / (2.0 * np.pi))) % max(sc, 1)
        except Exception:
            return -1

    def _truck_late_sector_spread_term(self, env, aid: str, task) -> float:
        if task is None or task.kind != TaskKind.NORMAL:
            return 0.0
        if (not self._large_map_active(env)) or (not bool(getattr(env.cfg, "hrl_truck_late_sector_spread_enabled", True))):
            return 0.0
        map_size = float(max(getattr(env.cfg, "map_size_m", 0.0), 0.0))
        if map_size < float(max(getattr(env.cfg, "hrl_truck_late_sector_spread_min_map_size_m", 12000.0), 0.0)):
            return 0.0
        step_now = int(getattr(env.state, "step_index", 0))
        if step_now <= int(max(getattr(env.cfg, "hrl_truck_directional_split_steps", 24), 0)):
            return 0.0
        pending_norm, norm_reach_by_truck, any_reachable_normal = self._normal_reachability_snapshot(env)
        if int(pending_norm) <= 1 or (not any_reachable_normal) or (not bool(norm_reach_by_truck.get(str(aid), True))):
            return 0.0

        sectors = int(self._initial_directional_sector_count(env))
        task_sector = int(self._task_sector_index(env, task, sectors))
        if task_sector < 0:
            return 0.0

        occupied: set = set()
        for oid, ost in env.state.agents.items():
            if str(oid) == str(aid) or ost.kind != AgentKind.TRUCK or bool(getattr(ost, "crashed", False)):
                continue
            og = self.state.goals.get(str(oid), None)
            ot = env.state.tasks.get(str(og), None) if og is not None else None
            if self._task_planner_active(ot):
                occupied.add(int(self._task_sector_index(env, ot, sectors)))
                continue
            oxy = self._agent_xy(env, str(oid))
            occupied.add(int(self._point_sector_index(env, oxy, sectors)))

        alt_free_exists = False
        for cand in env.state.tasks.values():
            if not self._task_planner_active(cand) or cand.kind != TaskKind.NORMAL:
                continue
            if not self._truck_task_serviceable_or_support_proxy(env, str(aid), cand):
                continue
            if not self._truck_task_reachable(env, str(aid), cand):
                continue
            csec = int(self._task_sector_index(env, cand, sectors))
            if csec >= 0 and csec != task_sector and csec not in occupied:
                alt_free_exists = True
                break

        bonus = float(max(getattr(env.cfg, "hrl_truck_late_sector_spread_bonus", 0.12), 0.0))
        penalty = float(max(getattr(env.cfg, "hrl_truck_late_sector_convergence_penalty", 0.26), 0.0))
        if task_sector not in occupied:
            return float(bonus if alt_free_exists else 0.0)
        if alt_free_exists:
            return float(-penalty)
        return 0.0

    def _uav_task_locality_term(self, env, aid: str, task) -> float:
        """
        Prefer assigning an emergency task to the UAV that is actually closer to it.
        This reduces cross-assignment (near task given to another far UAV).
        """
        w = float(max(getattr(env.cfg, "hrl_uav_task_locality_weight", 0.22), 0.0))
        if w <= 1e-9:
            return 0.0
        d_self = float(env._agent_distance_to_task(str(aid), task))
        if (not np.isfinite(d_self)) or d_self < 0.0:
            return 0.0
        dists: List[float] = []
        for uid, us in env.state.agents.items():
            if us.kind != AgentKind.UAV or bool(getattr(us, "crashed", False)):
                continue
            if not self._uav_task_feasible(env, str(uid), task):
                continue
            du = float(env._agent_distance_to_task(str(uid), task))
            if np.isfinite(du):
                dists.append(float(max(du, 0.0)))
        if len(dists) <= 1:
            return 0.0
        d_mean = float(np.mean(np.asarray(dists, dtype=np.float64)))
        norm = float(max(self._distance_norm_m(env), 1e-6))
        adv = float(np.clip((d_mean - d_self) / norm, -1.0, 1.0))
        return float(w * adv)

    def _uav_near_depot_direct_dispatch_term(self, env, aid: str, task, dist_m: float) -> float:
        """
        Early episode flexibility:
        if emergency is close to depot and UAV is currently not docked, prefer
        direct dispatch before boarding a truck.
        """
        if task.kind != TaskKind.EMERGENCY:
            return 0.0
        step_now = int(env.state.step_index)
        step_win = int(max(getattr(env.cfg, "hrl_uav_near_depot_direct_dispatch_steps", 18), 0))
        if step_now > step_win:
            return 0.0
        st = env.state.agents.get(str(aid), None)
        if st is None or st.kind != AgentKind.UAV or st.follow_target is not None:
            return 0.0
        radius = float(max(getattr(env.cfg, "hrl_uav_near_depot_direct_dispatch_radius_m", 700.0), 1.0))
        depx, depy = self._depot_xy(env)
        node = env.topology.nodes[int(task.demand_node)]
        d_depot = float(np.hypot(float(node.x) - depx, float(node.y) - depy))
        if d_depot > radius:
            return 0.0
        bonus = float(max(getattr(env.cfg, "hrl_uav_near_depot_direct_dispatch_bonus", 0.30), 0.0))
        near = float(np.clip(1.0 - float(max(dist_m, 0.0)) / float(max(1.2 * radius, 1.0)), 0.0, 1.0))
        return float(bonus * near)

    def _truck_emergency_pull_score(self, env, truck_id: str) -> float:
        """
        Estimate how useful this truck is as a ride target for upcoming emergency
        coverage (distance + heading + urgency).
        """
        ts = env.state.agents.get(str(truck_id), None)
        if ts is None or ts.kind != AgentKind.TRUCK or bool(getattr(ts, "crashed", False)):
            return 0.0
        txy = self._agent_xy(env, str(truck_id))
        heading = self._truck_motion_heading_xy(env, str(truck_id))
        step_idx = int(env.state.step_index)
        norm = float(max(self._distance_norm_m(env), 1e-6))
        best = 0.0
        for task in env.state.tasks.values():
            if task.status != TaskStatus.PENDING or task.kind != TaskKind.EMERGENCY:
                continue
            node = env.topology.nodes[int(task.demand_node)]
            dx = float(node.x) - float(txy[0])
            dy = float(node.y) - float(txy[1])
            dist = float(np.hypot(dx, dy))
            if (not np.isfinite(dist)) or dist < 0.0:
                continue
            prox = float(1.0 / (1.0 + dist / norm))
            urg = float(self._norm_deadline_urgency(task, step_idx))
            align = 0.0
            if heading is not None and dist > 1e-9:
                ux, uy = float(dx / dist), float(dy / dist)
                align = float(np.clip((ux * float(heading[0]) + uy * float(heading[1]) + 1.0) * 0.5, 0.0, 1.0))
            score = float(0.55 * prox + 0.25 * align + 0.20 * urg)
            if score > best:
                best = float(score)
        return float(np.clip(best, 0.0, 1.0))

    def _truck_follower_balance_score(self, env, aid: str, truck_id: str) -> float:
        cap = int(max(getattr(env.cfg, "uav_max_followers_per_truck", 0), 0))
        if cap <= 0:
            return 0.0
        count_fn = getattr(env, "_truck_follower_count", None)
        slot_fn = getattr(env, "_truck_has_follow_slot", None)
        cnt = float(count_fn(str(truck_id), exclude_aid=str(aid))) if callable(count_fn) else 0.0
        bal = float(np.clip(1.0 - cnt / max(float(cap), 1.0), 0.0, 1.0))
        if callable(slot_fn):
            try:
                if not bool(slot_fn(str(truck_id), exclude_aid=str(aid))):
                    bal -= 0.60
            except Exception:
                pass
        return float(np.clip(bal, -1.0, 1.0))


    def _uav_supported_sortie_readiness_from_truck(self, env, aid: str, truck_id: str) -> float:
        # Joint readiness term: if this UAV follows this truck, estimate whether
        # truck position can quickly convert into a safe emergency sortie.
        if not self._supported_sortie_joint_enabled(env):
            return 0.0
        uav = env.state.agents.get(str(aid), None)
        truck = env.state.agents.get(str(truck_id), None)
        if uav is None or truck is None:
            return 0.0
        if uav.kind != AgentKind.UAV or truck.kind != AgentKind.TRUCK:
            return 0.0
        if bool(getattr(uav, "crashed", False)) or bool(getattr(truck, "crashed", False)):
            return 0.0
        if hasattr(env, "_uav_loaded") and (not bool(env._uav_loaded(str(aid)))):
            return 0.0

        tx, ty = self._agent_xy(env, str(truck_id))
        near_primary = float(max(getattr(env.cfg, "uav_docked_near_dispatch_radius_m", 900.0), 1.0))
        near_secondary = float(max(getattr(env.cfg, "uav_docked_heading_dispatch_radius_m", 1500.0), near_primary))
        best = 0.0
        for task in env.state.tasks.values():
            if task.kind != TaskKind.EMERGENCY or task.status != TaskStatus.PENDING:
                continue
            node = env.topology.nodes[int(task.demand_node)]
            d = float(np.hypot(float(node.x) - float(tx), float(node.y) - float(ty)))
            if (not np.isfinite(d)) or d < 0.0:
                continue
            near = float(np.clip(1.0 - d / max(near_secondary, 1.0), 0.0, 1.0))
            if d <= near_primary:
                near = float(min(1.0, near + 0.25))
            urg = float(self._norm_deadline_urgency(task, int(env.state.step_index)))
            sc = float(0.82 * near + 0.18 * urg)
            if sc > best:
                best = float(sc)
        return float(np.clip(best, 0.0, 1.0))

    def _truck_supported_sortie_joint_score(self, env, aid: str, task) -> float:
        # If a truck currently carries loaded UAV followers, reward emergency tasks
        # that are close enough to become high-probability supported sorties.
        if not self._supported_sortie_joint_enabled(env):
            return 0.0
        if task.kind != TaskKind.EMERGENCY:
            return 0.0
        truck = env.state.agents.get(str(aid), None)
        if truck is None or truck.kind != AgentKind.TRUCK or bool(getattr(truck, "crashed", False)):
            return 0.0
        d = float(self._truck_task_distance(env, str(aid), task))
        if not np.isfinite(d):
            return 0.0
        near_secondary = float(max(getattr(env.cfg, "uav_docked_heading_dispatch_radius_m", 1500.0), 1.0))
        near = float(np.clip(1.0 - d / near_secondary, 0.0, 1.0))

        followers = 0
        loaded_followers = 0
        for uid, us in env.state.agents.items():
            if us.kind != AgentKind.UAV or bool(getattr(us, "crashed", False)):
                continue
            if str(getattr(us, "follow_target", None)) != str(aid):
                continue
            followers += 1
            if hasattr(env, "_uav_loaded") and bool(env._uav_loaded(str(uid))):
                loaded_followers += 1
        if followers <= 0:
            return 0.0
        loaded_frac = float(np.clip(float(loaded_followers) / float(max(followers, 1)), 0.0, 1.0))
        return float(np.clip(loaded_frac * near, 0.0, 1.0))

    def _uav_docked_heading_aligned_with_task(self, env, aid: str, task) -> bool:
        st = env.state.agents.get(str(aid), None)
        if st is None or st.kind != AgentKind.UAV or st.follow_target is None:
            return False
        truck_id = str(st.follow_target)
        if truck_id.startswith("depot"):
            return False
        h = self._truck_motion_heading_xy(env, truck_id)
        if h is None:
            return False
        ax, ay = self._agent_xy(env, str(aid))
        node = env.topology.nodes[int(task.demand_node)]
        tx = float(node.x) - float(ax)
        ty = float(node.y) - float(ay)
        norm = float(np.hypot(tx, ty))
        if norm <= 1e-9:
            return True
        ux, uy = float(tx / norm), float(ty / norm)
        cosv = float(ux * float(h[0]) + uy * float(h[1]))
        cos_thr = float(np.clip(getattr(env.cfg, "uav_docked_heading_min_cosine", 0.05), -1.0, 1.0))
        return bool(cosv >= cos_thr)

    def _uav_nearest_feasible_emergency_distance(self, env, aid: str) -> float:
        best = float("inf")
        for t in env.state.tasks.values():
            if t.kind != TaskKind.EMERGENCY or t.status != TaskStatus.PENDING:
                continue
            if not self._uav_task_feasible(env, str(aid), t):
                continue
            d = float(env._agent_distance_to_task(str(aid), t))
            if np.isfinite(d) and d < best:
                best = float(d)
        return best

    def _task_class_label(self, task) -> str:
        cls = getattr(task, "task_class", None)
        if cls is None:
            return ""
        val = str(getattr(cls, "value", cls)).strip().lower()
        return val

    def _is_timecritical_lightweight_task(self, task) -> bool:
        if task is None:
            return False
        cls = self._task_class_label(task)
        if "time_critical_lightweight" in cls:
            return True
        # Backward-compatible fallback.
        return bool(getattr(task, "kind", None) == TaskKind.EMERGENCY)

    def _task_lifeline_ratio(self, task) -> float:
        if task is None:
            return 1.0
        life_init = float(max(float(getattr(task, "lifeline_init", 100.0)), 1e-6))
        life_cur = float(max(float(getattr(task, "lifeline_current", life_init)), 0.0))
        return float(np.clip(life_cur / life_init, 0.0, 1.0))

    def _task_planner_active(self, task) -> bool:
        if task is None:
            return False
        if getattr(task, "status", None) != TaskStatus.PENDING:
            return False
        # Planner-side guard: if a task has already exhausted lifeline in the
        # current decision slice, do not keep it in candidate pools even if a
        # viewer/trace snapshot still sees the old PENDING status for one beat.
        life_init = float(max(float(getattr(task, "lifeline_init", 0.0)), 0.0))
        life_cur = float(max(float(getattr(task, "lifeline_current", life_init)), 0.0))
        if life_init > 0.0 and life_cur <= 1e-9:
            return False
        if bool(getattr(task, "failed_due_to_lifeline_zero", False)):
            return False
        return True

    def _task_goal_gap_steps(self, env, task) -> int:
        if task is None:
            return 10**9
        last = int(self._task_last_goal_step.get(str(task.task_id), -10**9))
        return int(max(int(env.state.step_index) - last, 0))

    def _task_has_active_goal(self, env, task) -> bool:
        if task is None:
            return False
        tid = str(task.task_id)
        return any(str(gid) == tid for gid in self.state.goals.values() if gid is not None)

    def _timecritical_force_entry_active(self, env, task) -> bool:
        if not self._task_planner_active(task):
            return False
        if (not self._timecritical_pressure_active(env)) or (not bool(getattr(env.cfg, "hrl_timecritical_force_entry_enabled", True))):
            return False
        if not self._is_timecritical_lightweight_task(task):
            return False
        if self._task_has_active_goal(env, task):
            return False
        map_size = float(max(getattr(env.cfg, "map_size_m", 0.0), 0.0))
        min_map = float(max(getattr(env.cfg, "hrl_timecritical_force_entry_min_map_size_m", 12000.0), 0.0))
        if map_size < min_map:
            return False
        min_gap = int(max(getattr(env.cfg, "hrl_timecritical_force_entry_min_gap_steps", 12), 0))
        if self._task_goal_gap_steps(env, task) < min_gap:
            return False
        max_ratio = float(np.clip(getattr(env.cfg, "hrl_timecritical_force_entry_max_lifeline_ratio", 0.85), 0.0, 1.0))
        ratio = float(self._task_lifeline_ratio(task))
        urgent = float(np.clip(float(getattr(task, "urgency_score", self._norm_deadline_urgency(task, int(env.state.step_index)))), 0.0, 1.0))
        if ratio <= max_ratio:
            return True
        return bool(self._is_island_task(env, task) or self._task_high_pressure(env, task) or urgent >= 0.85)

    def _timecritical_far_exposure_active(self, env, task) -> bool:
        if not bool(getattr(env.cfg, "hrl_timecritical_far_exposure_enabled", True)):
            return False
        if not self._task_planner_active(task):
            return False
        if not self._is_timecritical_lightweight_task(task):
            return False
        if self._task_has_active_goal(env, task):
            return False
        map_size = float(max(getattr(env.cfg, "map_size_m", 0.0), 0.0))
        min_map = float(max(getattr(env.cfg, "hrl_timecritical_far_exposure_min_map_size_m", 9000.0), 0.0))
        if map_size < min_map:
            return False
        ratio = float(self._task_lifeline_ratio(task))
        max_ratio = float(np.clip(getattr(env.cfg, "hrl_timecritical_far_exposure_max_lifeline_ratio", 0.95), 0.0, 1.0))
        warning = float(np.clip(getattr(env.cfg, "hrl_timecritical_lifeline_warning_ratio", 0.55), 0.0, 1.0))
        critical = float(np.clip(getattr(env.cfg, "hrl_timecritical_lifeline_critical_ratio", 0.35), 0.0, 1.0))
        min_gap = int(max(getattr(env.cfg, "hrl_timecritical_far_exposure_min_gap_steps", 2), 0))
        gap_ok = bool(self._task_goal_gap_steps(env, task) >= min_gap)
        cover = float(np.clip(self._uav_emergency_cover_fraction(env, task), 0.0, 1.0))
        low_cover_thr = float(np.clip(getattr(env.cfg, "hrl_timecritical_far_exposure_low_cover_threshold", 0.35), 0.0, 1.0))
        low_cover = bool(cover <= low_cover_thr)
        urgent = float(np.clip(float(getattr(task, "urgency_score", self._norm_deadline_urgency(task, int(env.state.step_index)))), 0.0, 1.0))
        urgent_thr = float(np.clip(getattr(env.cfg, "hrl_timecritical_far_exposure_urgent_bypass_threshold", 0.88), 0.0, 1.0))
        urgent_bypass = bool(urgent >= urgent_thr)
        if (not urgent_bypass) and ratio > min(max_ratio, max(warning, 0.0)):
            return False
        return bool((gap_ok and low_cover and urgent >= 0.65) or ratio <= critical or urgent >= urgent_thr)

    def _task_priority_tier(self, env, aid: str, task) -> int:
        if not self._task_planner_active(task):
            return -1
        st = env.state.agents.get(str(aid), None)
        if self._is_timecritical_lightweight_task(task):
            ratio = float(self._task_lifeline_ratio(task))
            critical = float(np.clip(getattr(env.cfg, "hrl_timecritical_lifeline_critical_ratio", 0.35), 0.0, 1.0))
            warning = float(np.clip(getattr(env.cfg, "hrl_timecritical_lifeline_warning_ratio", 0.55), 0.0, 1.0))
            if ratio <= critical:
                base_tier = 3
            elif ratio <= warning:
                base_tier = 2
            else:
                base_tier = 1
            if self._timecritical_force_entry_active(env, task):
                base_tier = max(base_tier, 3 if ratio <= warning else 2)
            # Support priorities for truck emergency candidates:
            # critical-bound > warning-bound > bulk-bound > unbound.
            if st is not None and st.kind == AgentKind.TRUCK and task.kind == TaskKind.EMERGENCY:
                bind_info = self._support_bound_delivery_info(env, str(aid), task)
                if float(bind_info.get("bound_timecritical_critical", 0.0)) > 0.0:
                    return max(base_tier, 3)
                if float(bind_info.get("bound_timecritical_warning", 0.0)) > 0.0:
                    return max(base_tier, 2)
                if float(bind_info.get("bound_bulk", 0.0)) > 0.0:
                    return max(base_tier, 1)
                return max(base_tier, 0)
            return base_tier
        urgency = float(np.clip(self._norm_deadline_urgency(task, int(env.state.step_index)), 0.0, 1.0))
        if urgency >= 0.75:
            return 1
        return 0

    def _score_with_priority_tier(self, tier: int, tie_break_score: float) -> float:
        # Lexicographic proxy for linear assignment:
        # compare (priority_tier, tie_break_score) while limiting over-dominance
        # of tier under event-trigger mode in large maps.
        tier_band = 6.0 if bool(self.use_event_trigger) else 10.0
        return float(tier_band * float(int(tier)) + float(tie_break_score))

    def _support_bound_delivery_info(self, env, aid: str, task, gain_info: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        out = {
            "bound_timecritical": 0.0,
            "bound_timecritical_critical": 0.0,
            "bound_timecritical_warning": 0.0,
            "bound_bulk": 0.0,
            "bound_any": 0.0,
            "bound_timecritical_task_id": "",
            "bound_timecritical_uav_id": "",
            "bound_eta_steps": float("inf"),
        }
        if not self._task_planner_active(task):
            return out

        d_task = float(self._truck_task_distance(env, str(aid), task))
        cap = float(max(getattr(env.cfg, "hrl_support_candidate_max_distance_m", 5000.0), 0.0))
        if np.isfinite(d_task) and cap > 1e-9 and d_task > (1.35 * cap):
            if not (self._is_island_task(env, task) or self._task_high_pressure(env, task)):
                return out

        gi = gain_info if isinstance(gain_info, dict) else self._support_anchor_service_gain(env, str(aid), task)
        gain_score = float(np.clip(float(gi.get("gain_score", 0.0)), 0.0, 1.0))
        newly_serviceable = float(max(float(gi.get("new_serviceable_task_count", 0.0)), 0.0))
        relaxed_new = float(max(float(gi.get("new_relaxed_feasible_task_count", 0.0)), 0.0))

        horizon = int(self._support_bind_horizon_steps(env))

        life_cur = float(max(float(getattr(task, "lifeline_current", 100.0)), 0.0))
        life_decay = float(max(float(getattr(task, "lifeline_decay_rate", 0.0)), 0.0))
        gain_ok = bool((newly_serviceable >= 0.5) or (relaxed_new >= 1.0) or (gain_score >= 0.12))

        direct_bind_ok = False
        if task.kind == TaskKind.EMERGENCY:
            post_d = float(gi.get("post_support_primary_distance_m", float("inf")))
            d_m = float(post_d if np.isfinite(post_d) else self._truck_task_distance(env, str(aid), task))
            if np.isfinite(d_m):
                v = float(max(getattr(env.cfg, "truck_speed_mps", 1.0), 1e-6))
                dt = float(max(getattr(env.cfg, "dt_seconds", 1.0), 1e-6))
                eta_steps = int(np.ceil(max(d_m, 0.0) / max(v * dt, 1e-6)))
                life_after_eta = float(max(life_cur - float(eta_steps) * life_decay, 0.0))
                direct_bind_ok = bool(eta_steps <= int(max(horizon + 6, 1)) and life_after_eta > 0.0)

        best_uav: Optional[str] = None
        best_eta = float("inf")
        if self._is_timecritical_lightweight_task(task) and best_uav is None and (gain_ok or direct_bind_ok):
            # Fallback bind for high-pressure support chains: nearest UAV can be
            # temporarily bound and routed via support/recovery chain.
            if self._task_high_pressure(env, task) or self._is_island_task(env, task):
                util = float(np.clip(getattr(env.cfg, "uav_launch_speed_utilization", 0.70), 0.1, 1.0))
                v_ref = float(max(getattr(env.cfg, "uav_max_speed_mps", 1.0) * util, 1e-6))
                dt = float(max(getattr(env.cfg, "dt_seconds", 1.0), 1e-6))
                for uid, ust in env.state.agents.items():
                    if ust.kind != AgentKind.UAV or bool(getattr(ust, "crashed", False)):
                        continue
                    d = float(env._agent_distance_to_task(str(uid), task))
                    if not np.isfinite(d):
                        continue
                    eta_steps = float(np.ceil(max(d, 0.0) / max(v_ref * dt, 1e-6)))
                    if eta_steps + 1e-9 < best_eta:
                        best_eta = float(eta_steps)
                        best_uav = str(uid)

        if self._is_timecritical_lightweight_task(task):
            util = float(np.clip(getattr(env.cfg, "uav_launch_speed_utilization", 0.70), 0.1, 1.0))
            v_ref = float(max(getattr(env.cfg, "uav_max_speed_mps", 1.0) * util, 1e-6))
            dt = float(max(getattr(env.cfg, "dt_seconds", 1.0), 1e-6))
            for uid, ust in env.state.agents.items():
                if ust.kind != AgentKind.UAV or bool(getattr(ust, "crashed", False)):
                    continue
                d = float(env._agent_distance_to_task(str(uid), task))
                if not np.isfinite(d):
                    continue
                eta_steps = float(np.ceil(max(d, 0.0) / max(v_ref * dt, 1e-6)))
                eta_cap = float(max(horizon + 8, int(1.8 * max(horizon, 1))))
                if eta_steps > eta_cap:
                    continue
                feasible_now = bool(self._uav_task_effectively_covering_now(env, str(uid), task))
                if not feasible_now and bool(getattr(env.state.agents.get(str(uid), None), "follow_target", None)):
                    post_support_d = float(gi.get("post_support_primary_distance_m", float("inf")))
                    short_cap, long_cap = self._uav_dispatch_distance_caps(env, task)
                    recovery_buf = float(max(getattr(env.cfg, "uav_recovery_distance_buffer_m", 600.0), 0.0))
                    mission_chain_post = float(2.0 * max(post_support_d, 0.0) + recovery_buf)
                    if self._legacy_sortie_cap_enabled(env):
                        sortie_cap = float(max(getattr(env.cfg, "uav_max_sortie_m", long_cap), long_cap))
                        sortie_ok = bool(mission_chain_post <= sortie_cap * 0.92)
                    else:
                        sortie_ok = True
                    feasible_now = bool(np.isfinite(post_support_d) and post_support_d <= long_cap and sortie_ok)
                if not (feasible_now or gain_ok or direct_bind_ok):
                    continue
                if eta_steps + 1e-9 < best_eta:
                    best_eta = float(eta_steps)
                    best_uav = str(uid)

        if self._is_timecritical_lightweight_task(task):
            life_after_best_eta = float(max(life_cur - float(best_eta if np.isfinite(best_eta) else horizon) * life_decay, 0.0))
            post_support_d = float(gi.get("post_support_primary_distance_m", float("inf")))
            short_cap, long_cap = self._uav_dispatch_distance_caps(env, task)
            recovery_buf = float(max(getattr(env.cfg, "uav_recovery_distance_buffer_m", 600.0), 0.0))
            mission_chain_post = float(2.0 * max(post_support_d, 0.0) + recovery_buf)
            if self._legacy_sortie_cap_enabled(env):
                sortie_cap = float(max(getattr(env.cfg, "uav_max_sortie_m", long_cap), long_cap))
                sortie_ok = bool(mission_chain_post <= float(sortie_cap * 0.92))
            else:
                sortie_ok = True
            bound_chain_ready = bool(
                np.isfinite(post_support_d)
                and post_support_d <= float(long_cap)
                and sortie_ok
            )
            if (gain_ok or direct_bind_ok) and best_uav is not None and life_after_best_eta > 0.0 and bound_chain_ready:
                out["bound_timecritical"] = 1.0
                out["bound_timecritical_task_id"] = str(task.task_id)
                out["bound_timecritical_uav_id"] = str(best_uav)
                out["bound_eta_steps"] = float(best_eta)
                ratio = float(self._task_lifeline_ratio(task))
                critical = float(np.clip(getattr(env.cfg, "hrl_timecritical_lifeline_critical_ratio", 0.35), 0.0, 1.0))
                warning = float(np.clip(getattr(env.cfg, "hrl_timecritical_lifeline_warning_ratio", 0.55), 0.0, 1.0))
                if ratio <= critical:
                    out["bound_timecritical_critical"] = 1.0
                elif ratio <= warning:
                    out["bound_timecritical_warning"] = 1.0
        elif bool(getattr(env.cfg, "hrl_support_fallback_allow_bulk_binding", False)):
            if gain_ok or direct_bind_ok:
                out["bound_bulk"] = 1.0

        out["bound_any"] = 1.0 if (out["bound_timecritical"] > 0.0 or out["bound_bulk"] > 0.0) else 0.0
        return out

    def _tc_steps_remaining(self, env, task) -> float:
        if task is None:
            return 0.0
        deadline_rem = float(max(int(getattr(task, "deadline_step", env.cfg.max_steps)) - int(env.state.step_index), 0))
        life_cur = float(max(float(getattr(task, "lifeline_current", 0.0)), 0.0))
        life_decay = float(max(float(getattr(task, "lifeline_decay_rate", 0.0)), 0.0))
        if life_decay > 1e-9:
            return float(min(deadline_rem, life_cur / life_decay))
        return float(deadline_rem)

    def _uav_direct_feasible_for_tc(self, env, aid: str, task) -> bool:
        if task is None or not self._is_timecritical_lightweight_task(task):
            return False
        st = env.state.agents.get(str(aid), None)
        if st is None or st.kind != AgentKind.UAV or bool(getattr(st, "crashed", False)):
            return False
        if bool(self._uav_task_effectively_covering_now(env, str(aid), task)):
            return True
        docked_actionable_fn = getattr(env, "_uav_docked_task_actionable_now", None)
        if callable(docked_actionable_fn):
            try:
                if bool(docked_actionable_fn(str(aid), task)):
                    launch_gate_fn = getattr(env, "_uav_launch_gate_check", None)
                    if callable(launch_gate_fn):
                        try:
                            _ok, reason, force_recovery = launch_gate_fn(
                                str(aid),
                                task=task,
                                count_reject=False,
                            )
                            reason_s = str(reason)
                            if bool(force_recovery) and reason_s not in {"rendezvous_safe"}:
                                return False
                        except Exception:
                            pass
                    return True
            except Exception:
                pass
        return False

    def _tc_direct_feasible_any(self, env, task) -> bool:
        for uid, st in env.state.agents.items():
            if st.kind != AgentKind.UAV or bool(getattr(st, "crashed", False)):
                continue
            if self._uav_direct_feasible_for_tc(env, str(uid), task):
                return True
        return False

    def _uav_direct_feasible_alternative_tc(self, env, aid: str, exclude_task_id: str = "") -> Optional[str]:
        if not bool(getattr(env.cfg, "erc_tc_support_release_uav_for_direct_tc_enabled", True)):
            return None
        st = env.state.agents.get(str(aid), None)
        if st is None or st.kind != AgentKind.UAV or bool(getattr(st, "crashed", False)):
            return None
        if self._uav_needs_recovery(env, str(aid)):
            return None
        if bool(getattr(st, "uav_needs_reload_flag", False)):
            return None
        if hasattr(env, "_uav_loaded") and (not bool(env._uav_loaded(str(aid)))):
            return None
        best_tid: Optional[str] = None
        best_key: Tuple[float, float, str] = (float("inf"), float("inf"), "")
        for task in env.state.tasks.values():
            if not self._is_timecritical_lightweight_task(task) or task.status != TaskStatus.PENDING:
                continue
            tid = str(getattr(task, "task_id", ""))
            if tid == str(exclude_task_id):
                continue
            if not self._uav_direct_feasible_for_tc(env, str(aid), task):
                continue
            ratio = float(np.clip(self._task_lifeline_ratio(task), 0.0, 1.0))
            try:
                dist = float(env._agent_distance_to_task(str(aid), task))
            except Exception:
                dist = float("inf")
            key = (float(ratio), float(dist), str(tid))
            if key < best_key:
                best_key = key
                best_tid = str(tid)
        return best_tid

    def _tc_support_anchor_node(
        self,
        env,
        truck_id: str,
        task,
        ready_distance_m: float,
        max_setup_steps: float,
    ) -> Optional[Dict[str, float]]:
        truck = env.state.agents.get(str(truck_id), None)
        if truck is None or truck.kind != AgentKind.TRUCK or bool(getattr(truck, "crashed", False)):
            return None
        if truck.node is None or task is None:
            return None
        try:
            start_node = int(truck.node)
            task_node = int(task.demand_node)
        except Exception:
            return None
        try:
            tx, ty = env._node_xy(int(task_node))
        except Exception:
            node = env.topology.nodes.get(int(task_node), None)
            if node is None:
                return None
            tx, ty = float(node.x), float(node.y)
        speed = float(max(getattr(env.cfg, "truck_speed_mps", 1.0), 1e-6))
        dt = float(max(getattr(env.cfg, "dt_seconds", 1.0), 1e-6))
        budget_m = float(max(float(max_setup_steps), 1.0) * speed * dt)
        cur_node = int(start_node)
        truck_dist = 0.0
        try:
            cx, cy = env._node_xy(int(cur_node))
            cur_task_dist = float(np.hypot(float(cx) - float(tx), float(cy) - float(ty)))
        except Exception:
            return None
        if not np.isfinite(cur_task_dist):
            return None
        if cur_task_dist <= float(ready_distance_m) + 1e-9:
            return {
                "anchor_node": int(cur_node),
                "truck_to_anchor_m": 0.0,
                "anchor_to_task_m": float(cur_task_dist),
                "setup_steps": 0.0,
            }

        use_search_anchor = True
        min_balance = float(np.clip(getattr(env.cfg, "erc_tc_support_anchor_search_min_balance", 0.20), 0.0, 1.0))
        if self._region_commitment_active(env) and float(getattr(self, "_region_commitment_load_balance_score", 1.0)) < min_balance:
            use_search_anchor = False

        if not use_search_anchor:
            visited = {int(cur_node)}
            max_iter = int(max(min(float(max_setup_steps) + 4.0, float(len(getattr(env.topology, "nodes", {})) + 4)), 1))
            best_reachable_greedy: Tuple[float, float, int] = (float(cur_task_dist), 0.0, int(cur_node))
            for _ in range(max_iter):
                best_nb: Optional[int] = None
                best_nb_task_dist = float(cur_task_dist)
                best_edge_m = float("inf")
                try:
                    neighbors = list(env._decision_neighbors(int(cur_node)))
                except Exception:
                    neighbors = []
                for nb in neighbors:
                    nb_i = int(nb)
                    if nb_i in visited:
                        continue
                    try:
                        nx, ny = env._node_xy(int(nb_i))
                        nb_task_dist = float(np.hypot(float(nx) - float(tx), float(ny) - float(ty)))
                        edge_m = float(env.topology.edge_distance(int(cur_node), int(nb_i)))
                    except Exception:
                        continue
                    if not (np.isfinite(nb_task_dist) and np.isfinite(edge_m)):
                        continue
                    if nb_task_dist + 1e-9 >= cur_task_dist:
                        continue
                    cand = (float(nb_task_dist), float(edge_m), int(nb_i))
                    best_cur = (float(best_nb_task_dist), float(best_edge_m), int(best_nb) if best_nb is not None else 10**9)
                    if cand < best_cur:
                        best_nb = int(nb_i)
                        best_nb_task_dist = float(nb_task_dist)
                        best_edge_m = float(edge_m)
                if best_nb is None:
                    break
                if truck_dist + best_edge_m > budget_m + 1e-9:
                    break
                cur_node = int(best_nb)
                visited.add(int(cur_node))
                truck_dist = float(truck_dist + best_edge_m)
                cur_task_dist = float(best_nb_task_dist)
                best_reachable_greedy = (float(cur_task_dist), float(truck_dist), int(cur_node))
                if cur_task_dist <= float(ready_distance_m) + 1e-9:
                    break
            d_task_g, d_truck_g, node_id_g = best_reachable_greedy
            if d_task_g > float(ready_distance_m) + 1e-9:
                return None
            return {
                "anchor_node": int(node_id_g),
                "truck_to_anchor_m": float(d_truck_g),
                "anchor_to_task_m": float(d_task_g),
                "setup_steps": float(np.ceil(float(d_truck_g) / max(speed * dt, 1e-6))),
            }

        # Search all road nodes reachable within the support setup budget.
        # A local greedy walk can stop at a road-geometry local optimum that is
        # still unsafe for UAV recovery; bounded Dijkstra makes the launch anchor
        # reflect the best reachable recovery geometry, not only the first descent.
        pq: List[Tuple[float, int]] = [(0.0, int(cur_node))]
        best_dist_by_node: Dict[int, float] = {int(cur_node): 0.0}
        best_reachable: Tuple[float, float, int] = (float(cur_task_dist), 0.0, int(cur_node))
        best_ready: Optional[Tuple[float, float, int]] = None
        visit_cap = int(max(getattr(env.cfg, "erc_tc_support_anchor_search_node_cap", 320), 32))
        visited_count = 0
        while pq and visited_count < visit_cap:
            d_from_start, node_id_cur = heapq.heappop(pq)
            if d_from_start > budget_m + 1e-9:
                continue
            if d_from_start > best_dist_by_node.get(int(node_id_cur), float("inf")) + 1e-9:
                continue
            visited_count += 1
            try:
                nx, ny = env._node_xy(int(node_id_cur))
                node_task_dist = float(np.hypot(float(nx) - float(tx), float(ny) - float(ty)))
            except Exception:
                node_task_dist = float("inf")
            if np.isfinite(node_task_dist):
                reach_key = (float(node_task_dist), float(d_from_start), int(node_id_cur))
                if reach_key < best_reachable:
                    best_reachable = reach_key
                if node_task_dist <= float(ready_distance_m) + 1e-9:
                    # Primary objective: minimize UAV mission chain; secondary:
                    # avoid burning excessive truck setup time.
                    ready_key = (float(node_task_dist), float(d_from_start), int(node_id_cur))
                    if best_ready is None or ready_key < best_ready:
                        best_ready = ready_key
            try:
                neighbors = list(env._decision_neighbors(int(node_id_cur)))
            except Exception:
                neighbors = []
            for nb in neighbors:
                nb_i = int(nb)
                try:
                    edge_m = float(env.topology.edge_distance(int(node_id_cur), int(nb_i)))
                except Exception:
                    continue
                if not np.isfinite(edge_m) or edge_m < 0.0:
                    continue
                nd = float(d_from_start + edge_m)
                if nd > budget_m + 1e-9:
                    continue
                if nd + 1e-9 < best_dist_by_node.get(nb_i, float("inf")):
                    best_dist_by_node[nb_i] = float(nd)
                    heapq.heappush(pq, (float(nd), int(nb_i)))

        d_task, d_truck, node_id = best_ready if best_ready is not None else best_reachable
        if d_task > float(ready_distance_m) + 1e-9:
            return None
        return {
            "anchor_node": int(node_id),
            "truck_to_anchor_m": float(d_truck),
            "anchor_to_task_m": float(d_task),
            "setup_steps": float(np.ceil(float(d_truck) / max(speed * dt, 1e-6))),
        }

    def _tc_support_required_chain_candidate(self, env, truck_id: str, task) -> Optional[Dict[str, object]]:
        if task is None or (not self._is_timecritical_lightweight_task(task)) or task.status != TaskStatus.PENDING:
            return None
        map_complexity = str(getattr(env.cfg, "map_complexity", "")).upper().strip()
        scenario = str(getattr(env.cfg, "scenario", "")).upper().strip()
        if map_complexity not in {"L", "R"} and not (map_complexity == "M" and scenario == "C"):
            return None
        max_ratio = float(np.clip(getattr(env.cfg, "erc_tc_support_max_lifeline_ratio", 0.70), 0.0, 1.0))
        if scenario == "B":
            max_ratio = float(min(max_ratio, 1.0 if map_complexity in {"L", "R"} else 0.68))
        if float(np.clip(self._task_lifeline_ratio(task), 0.0, 1.0)) > max_ratio:
            return None
        gain_info = self._support_anchor_service_gain(env, str(truck_id), task)
        gain = float(np.clip(float(gain_info.get("gain_score", 0.0)), 0.0, 1.0))
        min_gain = float(np.clip(getattr(env.cfg, "erc_tc_support_min_gain_score", 0.10), 0.0, 1.0))
        post_d = float(gain_info.get("post_support_primary_distance_m", float("inf")))
        max_post = float(max(getattr(env.cfg, "erc_tc_support_post_distance_m", 3600.0), 0.0))
        high_urgency_post = float(max(getattr(env.cfg, "erc_tc_support_high_urgency_post_distance_m", 0.0), 0.0))
        high_urgency_thr = float(np.clip(getattr(env.cfg, "erc_tc_support_high_urgency_threshold", 0.88), 0.0, 1.0))
        if high_urgency_post > 0.0:
            task_urgency_now = float(
                np.clip(
                    float(getattr(task, "urgency_score", self._norm_deadline_urgency(task, int(env.state.step_index)))),
                    0.0,
                    1.0,
                )
            )
            if task_urgency_now >= high_urgency_thr:
                max_post = float(min(max_post, high_urgency_post))

        followers: List[str] = []
        require_follower = bool(getattr(env.cfg, "erc_tc_support_require_follower_uav", True))
        for uid, us in env.state.agents.items():
            if us.kind != AgentKind.UAV or bool(getattr(us, "crashed", False)):
                continue
            if require_follower and str(getattr(us, "follow_target", "")) != str(truck_id):
                continue
            if bool(getattr(us, "uav_needs_reload_flag", False)):
                continue
            if hasattr(env, "_uav_loaded") and (not bool(env._uav_loaded(str(uid)))):
                continue
            followers.append(str(uid))
        if not followers:
            return None

        short_cap, long_cap = self._uav_dispatch_distance_caps(env, task)
        recovery_buf = float(max(getattr(env.cfg, "uav_recovery_distance_buffer_m", 600.0), 0.0))
        if hasattr(env, "_effective_recovery_buffer_for_sortie"):
            try:
                recovery_buf = float(env._effective_recovery_buffer_for_sortie(str(followers[0]), task, launch_reason="rendezvous_safe"))
            except Exception:
                recovery_buf = float(max(getattr(env.cfg, "uav_recovery_distance_buffer_m", 600.0), 0.0))

        truck_d = float(self._truck_task_distance(env, str(truck_id), task))
        if not np.isfinite(truck_d):
            return None
        if self._legacy_sortie_cap_enabled(env):
            sortie_cap = float(max(getattr(env.cfg, "uav_max_sortie_m", long_cap), long_cap))
            max_sortie_oneway = float(max((float(sortie_cap) * 0.92 - max(recovery_buf, 0.0)) / 2.0, 1.0))
        else:
            max_sortie_oneway = float("inf")
        ready_distance_m = float(max(1.0, min(float(long_cap), float(max_sortie_oneway), float(max_post if max_post > 0 else long_cap))))
        if gain < min_gain and truck_d <= ready_distance_m:
            return None
        truck_speed = float(max(getattr(env.cfg, "truck_speed_mps", 1.0), 1e-6))
        uav_speed = float(max(getattr(env.cfg, "uav_max_speed_mps", 1.0), 1e-6))
        dt = float(max(getattr(env.cfg, "dt_seconds", 1.0), 1e-6))
        max_setup_steps = float(max(getattr(env.cfg, "erc_tc_support_max_setup_steps", 28), 1))
        anchor_enabled = bool(getattr(env.cfg, "erc_tc_support_anchor_waypoint_enabled", False))
        anchor: Optional[Dict[str, float]] = None
        if anchor_enabled:
            anchor = self._tc_support_anchor_node(
                env,
                str(truck_id),
                task,
                ready_distance_m=float(ready_distance_m),
                max_setup_steps=float(max_setup_steps),
            )
            if anchor is None:
                return None
            effective_ready_d = float(anchor.get("anchor_to_task_m", ready_distance_m))
            setup_steps = float(anchor.get("setup_steps", max_setup_steps))
            if setup_steps > max_setup_steps + 1e-9:
                return None
        else:
            if np.isfinite(post_d) and post_d <= ready_distance_m:
                effective_ready_d = float(post_d)
                setup_gain_m = float(max(truck_d - post_d, 0.0))
            else:
                effective_ready_d = float(ready_distance_m)
                setup_gain_m = float(max(truck_d - ready_distance_m, 0.0))
            setup_steps = float(np.ceil(setup_gain_m / max(truck_speed * dt, 1e-6)))
            setup_steps = float(min(setup_steps, max_setup_steps))
        service_steps = float(max(getattr(env.cfg, "uav_service_time_steps", getattr(env.cfg, "service_time_steps", 1)), 1))
        uav_steps = float(np.ceil(max(effective_ready_d, 0.0) / max(uav_speed * dt, 1e-6)) + service_steps)
        margin = float(max(getattr(env.cfg, "erc_tc_support_latest_start_margin_steps", 4), 0))
        remaining = float(self._tc_steps_remaining(env, task))
        eta_total = float(setup_steps + uav_steps + margin)
        latest_start_step = float(int(env.state.step_index) + max(remaining - eta_total, 0.0))
        if remaining + 1e-9 < eta_total:
            return None

        best_uid = min(
            followers,
            key=lambda uid: float(env._agent_distance_to_task(str(uid), task))
            if hasattr(env, "_agent_distance_to_task") else 0.0,
        )
        ratio = float(np.clip(self._task_lifeline_ratio(task), 0.0, 1.0))
        urgency = float(np.clip(float(getattr(task, "urgency_score", self._norm_deadline_urgency(task, int(env.state.step_index)))), 0.0, 1.0))
        score = float(3.0 * (1.0 - ratio) + 1.5 * gain + urgency - 0.03 * setup_steps)
        return {
            "class": "support_required",
            "truck_id": str(truck_id),
            "uav_id": str(best_uid),
            "task_id": str(task.task_id),
            "task": task,
            "gain": float(gain),
            "truck_distance_m": float(truck_d),
            "post_distance_m": float(post_d),
            "ready_distance_m": float(ready_distance_m),
            "anchor_node": int(anchor.get("anchor_node", int(task.demand_node))) if anchor is not None else int(task.demand_node),
            "truck_to_anchor_m": float(anchor.get("truck_to_anchor_m", 0.0)) if anchor is not None else float("nan"),
            "anchor_to_task_m": float(anchor.get("anchor_to_task_m", effective_ready_d)) if anchor is not None else float("nan"),
            "anchor_waypoint_enabled": bool(anchor_enabled),
            "setup_steps": float(setup_steps),
            "uav_steps": float(uav_steps),
            "eta_steps": float(eta_total),
            "latest_start_step": float(latest_start_step),
            "score": float(score),
        }

    def _tc_support_feasibility_class(self, env, task) -> Dict[str, object]:
        if task is None or (not self._is_timecritical_lightweight_task(task)) or task.status != TaskStatus.PENDING:
            return {"class": "truly_infeasible"}
        if self._tc_direct_feasible_any(env, task):
            return {"class": "direct_feasible", "task_id": str(task.task_id)}
        best: Optional[Dict[str, object]] = None
        for tid, st in env.state.agents.items():
            if st.kind != AgentKind.TRUCK or bool(getattr(st, "crashed", False)):
                continue
            cand = self._tc_support_required_chain_candidate(env, str(tid), task)
            if cand is None:
                continue
            if best is None or float(cand.get("score", -1e18)) > float(best.get("score", -1e18)):
                best = cand
        if best is not None:
            return best
        return {"class": "truly_infeasible", "task_id": str(task.task_id)}

    def _tc_uncovered_support_repair_candidate(
        self,
        env,
        task,
        active_trucks: set,
        active_uavs: set,
    ) -> Optional[Dict[str, object]]:
        if not bool(getattr(env.cfg, "erc_tc_uncovered_support_repair_enabled", False)):
            return None
        if task is None or (not self._is_timecritical_lightweight_task(task)) or task.status != TaskStatus.PENDING:
            return None
        map_complexity = str(getattr(env.cfg, "map_complexity", "")).upper().strip()
        if map_complexity not in {"M", "L", "R"}:
            return None
        if self._task_has_active_goal(env, task):
            return None
        step_now = int(env.state.step_index)
        min_step = int(max(getattr(env.cfg, "erc_tc_uncovered_support_repair_min_step", 18), 0))
        if step_now < min_step:
            return None
        min_gap = int(max(getattr(env.cfg, "erc_tc_uncovered_support_repair_min_gap_steps", 12), 0))
        if self._task_goal_gap_steps(env, task) < min_gap:
            return None
        ratio = float(np.clip(self._task_lifeline_ratio(task), 0.0, 1.0))
        max_ratio = float(np.clip(getattr(env.cfg, "erc_tc_uncovered_support_repair_max_lifeline_ratio", 0.82), 0.0, 1.0))
        if ratio > max_ratio:
            return None
        min_urgency = float(np.clip(getattr(env.cfg, "erc_tc_uncovered_support_repair_min_urgency", 0.0), 0.0, 1.0))
        if min_urgency > 0.0:
            urgency_now = float(
                np.clip(
                    float(getattr(task, "urgency_score", self._norm_deadline_urgency(task, int(env.state.step_index)))),
                    0.0,
                    1.0,
                )
            )
            if urgency_now < min_urgency:
                return None
        cover = float(np.clip(self._uav_emergency_cover_fraction(env, task), 0.0, 1.0))
        cover_thr = float(np.clip(getattr(env.cfg, "erc_tc_uncovered_support_repair_cover_threshold", 0.40), 0.0, 1.0))
        min_nearest = float(max(getattr(env.cfg, "erc_tc_uncovered_support_repair_min_nearest_truck_m", 5500.0), 0.0))
        nearest_truck_d = float("inf")
        for tid, st in env.state.agents.items():
            if st.kind != AgentKind.TRUCK or bool(getattr(st, "crashed", False)):
                continue
            d = float(self._truck_task_distance(env, str(tid), task))
            if np.isfinite(d):
                nearest_truck_d = min(nearest_truck_d, d)
        if not (cover <= cover_thr or (np.isfinite(nearest_truck_d) and nearest_truck_d >= min_nearest)):
            return None

        best: Optional[Dict[str, object]] = None
        for tid, st in env.state.agents.items():
            if st.kind != AgentKind.TRUCK or bool(getattr(st, "crashed", False)):
                continue
            if str(tid) in active_trucks:
                continue
            cand = self._tc_support_required_chain_candidate(env, str(tid), task)
            if cand is None:
                continue
            uid = str(cand.get("uav_id", ""))
            if uid in active_uavs:
                continue
            gap_bonus = float(np.clip(self._task_goal_gap_steps(env, task) / max(float(min_gap), 1.0), 0.0, 2.0))
            far_bonus = float(np.clip((nearest_truck_d - min_nearest) / max(min_nearest, 1.0), 0.0, 1.0)) if np.isfinite(nearest_truck_d) else 0.0
            cand = dict(cand)
            cand["score"] = float(cand.get("score", 0.0)) + 0.45 * (1.0 - ratio) + 0.25 * (1.0 - cover) + 0.12 * gap_bonus + 0.10 * far_bonus
            cand["uncovered_support_repair"] = True
            if best is None or float(cand.get("score", -1e18)) > float(best.get("score", -1e18)):
                best = cand
        return best

    def _tc_stale_assigned_support_repair_candidate(
        self,
        env,
        task,
        active_trucks: set,
        active_uavs: set,
    ) -> Optional[Dict[str, object]]:
        if not bool(getattr(env.cfg, "erc_tc_stale_assigned_support_repair_enabled", False)):
            return None
        if task is None or (not self._is_timecritical_lightweight_task(task)) or task.status != TaskStatus.PENDING:
            return None
        scenario = str(getattr(env.cfg, "scenario", "")).upper().strip()
        map_complexity = str(getattr(env.cfg, "map_complexity", "")).upper().strip()
        if not (scenario == "B" and map_complexity in {"L", "R"}):
            return None
        if getattr(task, "first_service_step", None) is not None:
            return None
        step_now = int(env.state.step_index)
        min_step = int(max(getattr(env.cfg, "erc_tc_stale_assigned_support_repair_min_step", 80), 0))
        if step_now < min_step:
            return None
        exposure = int(self._task_goal_exposure_count.get(str(task.task_id), 0))
        min_exposure = int(max(getattr(env.cfg, "erc_tc_stale_assigned_support_repair_min_exposure", 28), 0))
        if exposure < min_exposure:
            return None
        ratio = float(np.clip(self._task_lifeline_ratio(task), 0.0, 1.0))
        max_ratio = float(np.clip(getattr(env.cfg, "erc_tc_stale_assigned_support_repair_max_lifeline_ratio", 0.75), 0.0, 1.0))
        if ratio > max_ratio:
            return None
        min_nearest = float(max(getattr(env.cfg, "erc_tc_stale_assigned_support_repair_min_nearest_truck_m", 5200.0), 0.0))
        nearest_truck_d = float("inf")
        for tid, st in env.state.agents.items():
            if st.kind != AgentKind.TRUCK or bool(getattr(st, "crashed", False)):
                continue
            d = float(self._truck_task_distance(env, str(tid), task))
            if np.isfinite(d):
                nearest_truck_d = min(nearest_truck_d, d)
        if not (np.isfinite(nearest_truck_d) and nearest_truck_d >= min_nearest):
            return None

        best: Optional[Dict[str, object]] = None
        for tid, st in env.state.agents.items():
            if st.kind != AgentKind.TRUCK or bool(getattr(st, "crashed", False)):
                continue
            if str(tid) in active_trucks:
                continue
            cand = self._tc_support_required_chain_candidate(env, str(tid), task)
            if cand is None:
                continue
            uid = str(cand.get("uav_id", ""))
            if uid in active_uavs:
                continue
            cand = dict(cand)
            exposure_bonus = float(np.clip(exposure / max(float(min_exposure), 1.0), 0.0, 2.0))
            cand["score"] = float(cand.get("score", 0.0)) + 0.35 * exposure_bonus + 0.35 * (1.0 - ratio)
            cand["stale_assigned_support_repair"] = True
            if best is None or float(cand.get("score", -1e18)) > float(best.get("score", -1e18)):
                best = cand
        return best

    def _repair_stalled_routine_goal_ownership(
        self,
        env,
        goals: Dict[str, Optional[str]],
        used_tasks: set,
    ) -> None:
        if not bool(getattr(env.cfg, "erc_stalled_routine_ownership_repair_enabled", False)):
            return
        step_now = int(getattr(env.state, "step_index", 0))
        min_step = int(max(getattr(env.cfg, "erc_stalled_routine_ownership_min_step", 40), 0))
        if step_now < min_step:
            return
        exposure_thr = int(max(getattr(env.cfg, "erc_stalled_routine_ownership_exposure_steps", 48), 1))
        max_repairs = int(max(getattr(env.cfg, "erc_stalled_routine_ownership_max_repairs_per_step", 1), 0))
        if max_repairs <= 0:
            return
        pending_norm = [
            t
            for t in env.state.tasks.values()
            if t.kind == TaskKind.NORMAL and t.status == TaskStatus.PENDING
        ]
        completed_tc = sum(
            1
            for t in env.state.tasks.values()
            if self._is_timecritical_lightweight_task(t) and t.status == TaskStatus.DELIVERED
        )
        last_routine_mode = bool(
            len(pending_norm) <= int(max(getattr(env.cfg, "erc_last_routine_rescue_pending_threshold", 1), 1))
            and completed_tc >= int(max(getattr(env.cfg, "erc_last_routine_rescue_min_completed_tc", 7), 0))
        )
        speed = float(max(getattr(env.cfg, "truck_speed_mps", 1.0), 1e-6))
        dt = float(max(getattr(env.cfg, "dt_seconds", 1.0), 1e-6))
        candidates: List[Tuple[float, str, str]] = []
        for task in env.state.tasks.values():
            if task.kind != TaskKind.NORMAL or task.status != TaskStatus.PENDING:
                continue
            if getattr(task, "first_service_step", None) is not None:
                continue
            tid = str(task.task_id)
            exposure = int(self._task_goal_exposure_count.get(tid, 0))
            if exposure < exposure_thr and not last_routine_mode:
                continue
            best_truck = ""
            best_eta = float("inf")
            best_dist = float("inf")
            for aid, st in env.state.agents.items():
                if st.kind != AgentKind.TRUCK or bool(getattr(st, "crashed", False)):
                    continue
                if bool(getattr(st, "service_timer", 0) > 0) or bool(getattr(st, "is_servicing", False)):
                    continue
                if not self._truck_task_valid(env, str(aid), tid):
                    continue
                if not self._truck_task_reachable(env, str(aid), task):
                    continue
                d = float(self._truck_task_distance(env, str(aid), task))
                if not np.isfinite(d):
                    continue
                eta = float(d / max(speed * dt, 1e-6))
                cur_gid = goals.get(str(aid), None)
                cur_task = env.state.tasks.get(str(cur_gid), None) if cur_gid is not None else None
                if cur_task is not None and cur_task.kind == TaskKind.NORMAL and getattr(cur_task, "first_service_step", None) is not None:
                    continue
                if eta < best_eta - 1e-9 or (abs(eta - best_eta) <= 1e-9 and d < best_dist):
                    best_truck = str(aid)
                    best_eta = float(eta)
                    best_dist = float(d)
            if not best_truck:
                continue
            # Higher exposure and lower ETA first; in last-routine mode, make
            # the remaining routine dominate support churn until service starts.
            priority = float(exposure) - 0.01 * float(best_eta) + (1000.0 if last_routine_mode else 0.0)
            candidates.append((priority, str(best_truck), tid))
        if not candidates:
            return
        candidates.sort(key=lambda x: (-float(x[0]), str(x[2]), str(x[1])))
        repaired = 0
        claimed_trucks: set = set()
        for _priority, truck_id, task_id in candidates:
            if repaired >= max_repairs:
                break
            if truck_id in claimed_trucks:
                continue
            prev = goals.get(str(truck_id), None)
            if prev is not None and str(prev) in used_tasks:
                used_tasks.discard(str(prev))
            for aid, gid in list(goals.items()):
                if str(aid) == str(truck_id) or str(gid) != str(task_id):
                    continue
                st = env.state.agents.get(str(aid), None)
                if st is not None and st.kind == AgentKind.TRUCK:
                    goals[str(aid)] = None
            goals[str(truck_id)] = str(task_id)
            used_tasks.add(str(task_id))
            claimed_trucks.add(str(truck_id))
            repaired += 1

    def _repair_unassigned_reachable_routine(
        self,
        env,
        goals: Dict[str, Optional[str]],
        used_tasks: set,
    ) -> None:
        if not bool(getattr(env.cfg, "erc_unassigned_routine_repair_enabled", False)):
            return
        step_now = int(getattr(env.state, "step_index", 0))
        min_step = int(max(getattr(env.cfg, "erc_unassigned_routine_repair_min_step", 70), 0))
        if step_now < min_step:
            return
        max_repairs = int(max(getattr(env.cfg, "erc_unassigned_routine_repair_max_per_step", 1), 0))
        if max_repairs <= 0:
            return
        speed = float(max(getattr(env.cfg, "truck_speed_mps", 1.0), 1e-6))
        dt = float(max(getattr(env.cfg, "dt_seconds", 1.0), 1e-6))
        candidates: List[Tuple[float, str, str]] = []
        for task in env.state.tasks.values():
            if task.kind != TaskKind.NORMAL or task.status != TaskStatus.PENDING:
                continue
            if getattr(task, "first_service_step", None) is not None:
                continue
            tid = str(task.task_id)
            if tid in used_tasks:
                continue
            has_active_goal = bool(self._task_has_active_goal(env, task))
            if has_active_goal:
                # In large B, a route-plan contract can remain stamped on a
                # pending routine while the owner is held at an emergency
                # anchor and publishes no executable task goal.  Treat only
                # that narrow orphan case as repairable; a live goal or claim
                # remains protected exactly as before.
                b_orphan_rescue = bool(
                    getattr(env.cfg, "erc_b_orphaned_routine_rescue_enabled", False)
                    and str(getattr(env.cfg, "scenario", "")).upper() == "B"
                    and int(getattr(env.state, "step_index", 0))
                    >= int(max(getattr(env.cfg, "erc_b_orphaned_routine_rescue_min_step", 120), 0))
                    and float(self._task_lifeline_ratio(task))
                    <= float(np.clip(getattr(env.cfg, "erc_b_orphaned_routine_rescue_max_lifeline_ratio", 0.80), 0.0, 1.0))
                    and task.status == TaskStatus.PENDING
                    and getattr(task, "assigned_to", None) is None
                    and not any(str(gid) == tid for gid in goals.values() if gid is not None)
                )
                if not b_orphan_rescue:
                    continue
            best_truck = ""
            best_eta = float("inf")
            for aid, st in env.state.agents.items():
                if st.kind != AgentKind.TRUCK or bool(getattr(st, "crashed", False)):
                    continue
                if bool(getattr(st, "service_timer", 0) > 0) or bool(getattr(st, "is_servicing", False)):
                    continue
                if not self._truck_task_valid(env, str(aid), tid):
                    continue
                if not self._truck_task_reachable(env, str(aid), task):
                    continue
                d = float(self._truck_task_distance(env, str(aid), task))
                if not np.isfinite(d):
                    continue
                eta = float(d / max(speed * dt, 1e-6))
                if eta < best_eta:
                    best_eta = float(eta)
                    best_truck = str(aid)
            if not best_truck:
                continue
            urgency = float(1.0 - np.clip(self._task_lifeline_ratio(task), 0.0, 1.0))
            assigned_gap = float(self._task_goal_gap_steps(env, task))
            priority = float(2.0 * urgency + 0.01 * assigned_gap - 0.002 * best_eta)
            candidates.append((priority, best_truck, tid))
        if not candidates:
            return
        candidates.sort(key=lambda x: (-float(x[0]), str(x[2]), str(x[1])))
        repaired = 0
        claimed_trucks: set = set()
        for _priority, truck_id, task_id in candidates:
            if repaired >= max_repairs:
                break
            if truck_id in claimed_trucks:
                continue
            prev = goals.get(str(truck_id), None)
            if prev is not None and str(prev) in used_tasks:
                used_tasks.discard(str(prev))
            for aid, gid in list(goals.items()):
                if str(aid) == str(truck_id) or str(gid) != str(task_id):
                    continue
                st = env.state.agents.get(str(aid), None)
                if st is not None and st.kind == AgentKind.TRUCK:
                    goals[str(aid)] = None
            goals[str(truck_id)] = str(task_id)
            used_tasks.add(str(task_id))
            claimed_trucks.add(str(truck_id))
            repaired += 1

    def _apply_large_map_greedy_tc_fallback(
        self,
        env,
        goals: Dict[str, Optional[str]],
        used_tasks: set,
    ) -> None:
        if not bool(getattr(env.cfg, "erc_routine_progress_watchdog_enabled", False)):
            return
        if not bool(getattr(env.cfg, "erc_large_map_greedy_tc_fallback_enabled", True)):
            return
        map_complexity = str(getattr(env.cfg, "map_complexity", "")).upper().strip()
        if map_complexity not in {"L", "R"}:
            return
        scenario = str(getattr(env.cfg, "scenario", "")).upper().strip()
        if scenario not in {"B", "C"}:
            return
        if scenario == "B":
            step_now = int(getattr(env.state, "step_index", 0))
            min_step = int(max(getattr(env.cfg, "erc_large_map_greedy_tc_fallback_b_min_step", 80), 0))
            if step_now < min_step:
                return
            max_deliveries = int(max(getattr(env.cfg, "erc_large_map_greedy_tc_fallback_b_max_support_deliveries", 4), 0))
            if int(getattr(self, "support_authorized_to_delivery_count_total", 0)) > max_deliveries:
                return
            min_locks = int(max(getattr(env.cfg, "erc_large_map_greedy_tc_fallback_b_min_support_locks", 8), 0))
            if int(getattr(self, "tc_support_lock_created_count_total", 0)) < min_locks:
                return
        pending_tc = [
            t
            for t in env.state.tasks.values()
            if self._is_timecritical_lightweight_task(t) and t.status == TaskStatus.PENDING
        ]
        if len(pending_tc) < 4:
            return
        assigned_tc = {
            str(gid)
            for gid in goals.values()
            if gid is not None
            and str(gid) in env.state.tasks
            and self._is_timecritical_lightweight_task(env.state.tasks[str(gid)])
        }
        for aid in self._ordered_agents(env):
            st = env.state.agents.get(str(aid), None)
            if st is None or st.kind != AgentKind.UAV or bool(getattr(st, "crashed", False)):
                continue
            if bool(getattr(st, "airborne", False)) or bool(self._uav_needs_recovery(env, str(aid))):
                continue
            if bool(getattr(st, "uav_needs_reload_flag", False)):
                continue
            if hasattr(env, "_uav_loaded") and not bool(env._uav_loaded(str(aid))):
                continue
            cur_gid = goals.get(str(aid), None)
            cur_task = env.state.tasks.get(str(cur_gid), None) if cur_gid is not None else None
            if cur_task is not None and self._is_timecritical_lightweight_task(cur_task) and cur_task.status == TaskStatus.PENDING:
                replace_stale = False
                if bool(getattr(env.cfg, "erc_large_map_greedy_tc_fallback_replace_stale_enabled", False)):
                    assigned_step = int(self.state.goal_assigned_step.get(str(aid), step_now))
                    stale_steps = int(max(getattr(env.cfg, "erc_large_map_greedy_tc_fallback_stale_steps", 24), 0))
                    assigned_age = int(max(step_now - assigned_step, 0))
                    try:
                        cur_launchable = bool(self._uav_task_feasible(env, str(aid), cur_task))
                    except Exception:
                        cur_launchable = False
                    replace_stale = bool(assigned_age >= stale_steps and not cur_launchable)
                if not replace_stale:
                    continue
            best_task = None
            best_key = (float("inf"), float("inf"), "")
            for task in pending_tc:
                tid = str(task.task_id)
                if tid == str(cur_gid):
                    continue
                if tid in used_tasks or tid in assigned_tc:
                    continue
                d = float(env._agent_distance_to_task(str(aid), task)) if hasattr(env, "_agent_distance_to_task") else float("inf")
                if not np.isfinite(d):
                    continue
                try:
                    if not bool(self._uav_task_feasible(env, str(aid), task)):
                        continue
                except Exception:
                    continue
                urgency = 1.0 - float(np.clip(self._task_lifeline_ratio(task), 0.0, 1.0))
                key = (float(d), -float(urgency), tid)
                if key < best_key:
                    best_key = key
                    best_task = task
            if best_task is None:
                continue
            goals[str(aid)] = str(best_task.task_id)
            used_tasks.add(str(best_task.task_id))
            assigned_tc.add(str(best_task.task_id))

    def _apply_large_map_tc_coverage_intents(
        self,
        env,
        goals: Dict[str, Optional[str]],
        used_tasks: set,
    ) -> None:
        """Create early UAV-truck intent for uncovered large-map TC tasks.

        This is deliberately prelaunch-only. It may point a docked UAV and its
        truck toward a time-critical task, but actual takeoff still goes through
        the normal launch-quality and recovery feasibility gates.
        """
        if not bool(getattr(env.cfg, "erc_tc_coverage_intent_enabled", False)):
            return
        map_complexity = str(getattr(env.cfg, "map_complexity", "")).upper().strip()
        if map_complexity not in {"L", "R"}:
            return
        if not bool(getattr(env.cfg, "erc_routine_progress_watchdog_enabled", False)):
            return

        step_now = int(getattr(env.state, "step_index", 0))
        min_step = int(max(getattr(env.cfg, "erc_tc_coverage_intent_min_step", 0), 0))
        if step_now < min_step:
            return
        max_step = int(max(getattr(env.cfg, "erc_tc_coverage_intent_max_step", 160), 0))
        if max_step > 0 and step_now > max_step:
            return
        max_support_deliveries = int(getattr(env.cfg, "erc_tc_coverage_intent_max_support_deliveries", -1))
        if max_support_deliveries >= 0 and int(getattr(self, "support_authorized_to_delivery_count_total", 0)) > max_support_deliveries:
            return

        pending_tc = [
            t
            for t in env.state.tasks.values()
            if self._is_timecritical_lightweight_task(t) and t.status == TaskStatus.PENDING
        ]
        min_pending = int(max(getattr(env.cfg, "erc_tc_coverage_intent_min_pending", 4), 0))
        if len(pending_tc) < min_pending:
            return

        assigned_tc = {
            str(gid)
            for gid in goals.values()
            if gid is not None
            and str(gid) in env.state.tasks
            and self._is_timecritical_lightweight_task(env.state.tasks[str(gid)])
        }
        for task_id in self._support_bound_chain_task_id.values():
            if str(task_id) in env.state.tasks:
                task = env.state.tasks[str(task_id)]
                if self._is_timecritical_lightweight_task(task) and task.status == TaskStatus.PENDING:
                    assigned_tc.add(str(task_id))
        for task_id, rec in self._uav_task_reservation_state_by_task.items():
            task = env.state.tasks.get(str(task_id), None)
            if task is not None and self._is_timecritical_lightweight_task(task) and task.status == TaskStatus.PENDING:
                if str(rec.get("status", "")) in {"reserved_prelaunch", "airborne_committed", "servicing"}:
                    assigned_tc.add(str(task_id))

        cover_thr = float(np.clip(getattr(env.cfg, "erc_tc_coverage_intent_cover_threshold", 0.55), 0.0, 1.0))
        max_ratio = float(np.clip(getattr(env.cfg, "erc_tc_coverage_intent_max_lifeline_ratio", 0.92), 0.0, 1.0))
        min_gap = int(max(getattr(env.cfg, "erc_tc_coverage_intent_min_gap_steps", 4), 0))
        candidates: List[Tuple[float, str, str, int]] = []
        for task in pending_tc:
            tid = str(task.task_id)
            if tid in used_tasks or tid in assigned_tc:
                continue
            ratio = float(np.clip(self._task_lifeline_ratio(task), 0.0, 1.0))
            gap = int(self._task_goal_gap_steps(env, task))
            cover = float(np.clip(self._uav_emergency_cover_fraction(env, task), 0.0, 1.0))
            if ratio > max_ratio and gap < min_gap and cover >= cover_thr:
                continue
            urgency = float(np.clip(float(getattr(task, "urgency_score", self._norm_deadline_urgency(task, step_now))), 0.0, 1.0))
            for aid in self._ordered_agents(env):
                st = env.state.agents.get(str(aid), None)
                if st is None or st.kind != AgentKind.UAV or bool(getattr(st, "crashed", False)):
                    continue
                if getattr(st, "follow_target", None) is None:
                    continue
                truck_id = str(getattr(st, "follow_target", ""))
                truck = env.state.agents.get(truck_id, None)
                if truck is None or truck.kind != AgentKind.TRUCK or bool(getattr(truck, "crashed", False)):
                    continue
                if aid in self._support_bound_chain_truck_by_uav:
                    continue
                if self._uav_needs_recovery(env, str(aid)):
                    continue
                if bool(getattr(st, "uav_needs_reload_flag", False)):
                    continue
                loaded_fn = getattr(env, "_uav_loaded", None)
                if callable(loaded_fn):
                    try:
                        if not bool(loaded_fn(str(aid))):
                            continue
                    except Exception:
                        continue

                cur_gid = goals.get(str(aid), None)
                cur_task = env.state.tasks.get(str(cur_gid), None) if cur_gid is not None else None
                if cur_task is not None and self._is_timecritical_lightweight_task(cur_task) and cur_task.status == TaskStatus.PENDING:
                    cur_ratio = float(np.clip(self._task_lifeline_ratio(cur_task), 0.0, 1.0))
                    if cur_ratio <= ratio + 0.03:
                        continue

                d_uav = float(env._agent_distance_to_task(str(aid), task)) if hasattr(env, "_agent_distance_to_task") else float("inf")
                if not np.isfinite(d_uav):
                    continue
                try:
                    d_truck = float(env._decision_shortest_path_distance(int(getattr(truck, "node", -1)), int(task.demand_node)))
                except Exception:
                    d_truck = float("inf")
                if not np.isfinite(d_truck):
                    continue
                prelaunch = bool(self._uav_task_prelaunch_assignable(env, str(aid), task))
                direct = bool(self._uav_task_feasible(env, str(aid), task))
                truck_progress = float(max(self._truck_route_progress_to_task(env, truck_id, task), 0.0))
                dist_norm = float(max(self._distance_norm_m(env), 1.0))
                score = float(
                    0.48 * (1.0 - ratio)
                    + 0.26 * urgency
                    + 0.22 * (1.0 - cover)
                    + 0.16 * np.clip(gap / max(float(min_gap), 1.0), 0.0, 2.0)
                    + 0.22 * truck_progress
                    + (0.20 if direct else (0.10 if prelaunch else 0.0))
                    - 0.10 * np.clip(d_truck / dist_norm, 0.0, 2.0)
                )
                candidates.append((score, str(aid), tid, int(task.demand_node)))

        if not candidates:
            return
        candidates.sort(key=lambda x: (-float(x[0]), str(x[2]), str(x[1])))
        max_intents = int(max(getattr(env.cfg, "erc_tc_coverage_intent_max_per_step", 2), 0))
        claimed_uavs: set = set()
        claimed_trucks: set = set()
        claimed_tasks: set = set()
        applied = 0
        for _score, aid, tid, launch_node in candidates:
            if applied >= max_intents:
                break
            if aid in claimed_uavs or tid in claimed_tasks:
                continue
            st = env.state.agents.get(str(aid), None)
            if st is None or st.kind != AgentKind.UAV:
                continue
            truck_id = str(getattr(st, "follow_target", "") or "")
            if not truck_id or truck_id in claimed_trucks:
                continue
            task = env.state.tasks.get(str(tid), None)
            if task is None or task.status != TaskStatus.PENDING:
                continue
            prev = goals.get(str(aid), None)
            if prev is not None and str(prev) in used_tasks:
                used_tasks.discard(str(prev))
            goals[str(aid)] = str(tid)
            used_tasks.add(str(tid))
            self._uav_reservation_assign(env, str(aid), str(tid), status="reserved_prelaunch")
            self._uav_intent_signal_by_uav[str(aid)] = {
                "uav_intent_task_id": str(tid),
                "uav_intent_target_xy": tuple(float(v) for v in env._node_xy(int(task.demand_node))),
                "uav_intent_nearest_launch_area": int(launch_node),
                "uav_intent_candidate_truck_id": str(truck_id),
                "step": int(step_now),
                "coverage_intent": True,
            }
            claimed_uavs.add(str(aid))
            claimed_trucks.add(str(truck_id))
            claimed_tasks.add(str(tid))
            applied += 1

    def _apply_tc_support_required_locks(
        self,
        env,
        ordered_agents: List[str],
        goals: Dict[str, Optional[str]],
        used_tasks: set,
    ) -> None:
        if not bool(getattr(env.cfg, "erc_tc_support_required_enabled", False)):
            return

        step_now = int(env.state.step_index)
        max_active = int(max(getattr(env.cfg, "erc_tc_support_max_active_chains", 2), 0))
        map_complexity_for_cap = str(getattr(env.cfg, "map_complexity", "")).upper().strip()
        if self._region_commitment_active(env):
            max_active = int(min(max_active, 3 if map_complexity_for_cap in {"L", "R"} else 2))
        if str(getattr(env.cfg, "scenario", "")).upper().strip() == "B":
            b_dynamic_second = bool(
                bool(getattr(env.cfg, "erc_tc_support_b_dynamic_second_chain_enabled", False))
                and map_complexity_for_cap in {"L", "R"}
                and int(step_now) >= int(max(getattr(env.cfg, "erc_tc_support_b_second_chain_min_step", 70), 0))
                and int(getattr(self, "support_authorized_to_delivery_count_total", 0))
                <= int(max(getattr(env.cfg, "erc_tc_support_b_second_chain_max_deliveries", 4), 0))
            )
            max_active = int(min(max_active, 2 if b_dynamic_second else 1))
        lock_steps = int(max(getattr(env.cfg, "erc_tc_support_lock_steps", 18), 1))

        active_trucks: set = set()
        active_uavs: set = set()
        active_tasks: set = set()
        for truck_id in list(self._support_bound_chain_task_id.keys()):
            info = self._support_bound_chain_info_for_truck(env, str(truck_id))
            if info is None:
                self._tc_support_chain_class.pop(str(truck_id), None)
                self._support_bound_chain_anchor_node_by_truck.pop(str(truck_id), None)
                self._support_bound_chain_latest_start_by_truck.pop(str(truck_id), None)
                continue
            if self._tc_support_chain_class.get(str(truck_id), "") != "support_required":
                continue
            task = info.get("task", None)
            uav_id = str(info.get("uav_id", ""))
            task_id = str(info.get("task_id", ""))
            if task is None or task.status != TaskStatus.PENDING:
                self._support_bound_chain_anchor_node_by_truck.pop(str(truck_id), None)
                self._support_bound_chain_latest_start_by_truck.pop(str(truck_id), None)
                continue
            latest_start = int(self._support_bound_chain_latest_start_by_truck.get(str(truck_id), 10**9))
            if step_now > latest_start and not bool(self._docked_uav_sortie_chain_ready(env, str(uav_id), task)):
                self._support_bound_chain_until_step.pop(str(truck_id), None)
                self._support_bound_chain_task_id.pop(str(truck_id), None)
                self._support_bound_chain_uav_id.pop(str(truck_id), None)
                self._support_bound_chain_truck_by_uav.pop(str(uav_id), None)
                self._support_bound_chain_anchor_node_by_truck.pop(str(truck_id), None)
                self._support_bound_chain_latest_start_by_truck.pop(str(truck_id), None)
                self._tc_support_chain_class.pop(str(truck_id), None)
                continue
            goals[str(truck_id)] = str(task_id)
            uav_st = env.state.agents.get(str(uav_id), None)
            if uav_st is not None and uav_st.kind == AgentKind.UAV:
                if bool(self._docked_uav_sortie_chain_ready(env, str(uav_id), task)):
                    goals[str(uav_id)] = str(task_id)
                    self.tc_support_lock_to_dispatch_count_total = int(self.tc_support_lock_to_dispatch_count_total) + 1
                else:
                    alt_tid = self._uav_direct_feasible_alternative_tc(env, str(uav_id), exclude_task_id=str(task_id))
                    if alt_tid is not None:
                        goals[str(uav_id)] = str(alt_tid)
                        used_tasks.add(str(alt_tid))
                    else:
                        goals[str(uav_id)] = str(truck_id)
            used_tasks.add(str(task_id))
            active_trucks.add(str(truck_id))
            active_uavs.add(str(uav_id))
            active_tasks.add(str(task_id))

        if len(active_trucks) >= max_active:
            return

        candidates: List[Dict[str, object]] = []
        for task in env.state.tasks.values():
            if not self._is_timecritical_lightweight_task(task) or task.status != TaskStatus.PENDING:
                continue
            if str(task.task_id) in active_tasks:
                continue
            info = self._tc_support_feasibility_class(env, task)
            cls = str(info.get("class", "truly_infeasible"))
            if cls == "direct_feasible":
                self.tc_direct_feasible_count_total = int(self.tc_direct_feasible_count_total) + 1
                repair = self._tc_uncovered_support_repair_candidate(env, task, active_trucks, active_uavs)
                if repair is None:
                    repair = self._tc_stale_assigned_support_repair_candidate(env, task, active_trucks, active_uavs)
                if repair is not None:
                    self.tc_support_required_count_total = int(self.tc_support_required_count_total) + 1
                    candidates.append(repair)
                continue
            if cls == "support_required":
                self.tc_support_required_count_total = int(self.tc_support_required_count_total) + 1
                candidates.append(info)
                repair = self._tc_stale_assigned_support_repair_candidate(env, task, active_trucks, active_uavs)
                if repair is not None:
                    candidates.append(repair)
            else:
                self.tc_truly_infeasible_count_total = int(self.tc_truly_infeasible_count_total) + 1

        candidates.sort(key=lambda x: (-float(x.get("score", -1e18)), str(x.get("task_id", ""))))
        for info in candidates:
            if len(active_trucks) >= max_active:
                break
            truck_id = str(info.get("truck_id", ""))
            uav_id = str(info.get("uav_id", ""))
            task_id = str(info.get("task_id", ""))
            task = info.get("task", None)
            if not truck_id or not uav_id or not task_id or task is None:
                continue
            if truck_id in active_trucks or uav_id in active_uavs or task_id in active_tasks:
                continue
            if truck_id not in goals or uav_id not in goals:
                continue
            eta_steps = float(info.get("eta_steps", lock_steps))
            setup_steps = float(info.get("setup_steps", 0.0))
            truck_distance_m = float(info.get("truck_distance_m", 0.0))
            max_setup_steps = float(max(getattr(env.cfg, "erc_tc_support_max_setup_steps", lock_steps), lock_steps))
            if truck_distance_m >= 12000.0 or setup_steps >= 45.0:
                dynamic_lock_steps = int(
                    max(
                        float(lock_steps),
                        min(float(lock_steps) + max_setup_steps, eta_steps + setup_steps + 4.0),
                    )
                )
            else:
                dynamic_lock_steps = int(lock_steps)
            self._support_bound_chain_until_step[str(truck_id)] = int(step_now + dynamic_lock_steps)
            self._support_bound_chain_task_id[str(truck_id)] = str(task_id)
            self._support_bound_chain_uav_id[str(truck_id)] = str(uav_id)
            self._support_bound_chain_truck_by_uav[str(uav_id)] = str(truck_id)
            if "anchor_node" in info:
                self._support_bound_chain_anchor_node_by_truck[str(truck_id)] = int(info.get("anchor_node", int(task.demand_node)))
            self._support_bound_chain_latest_start_by_truck[str(truck_id)] = int(
                max(float(step_now), float(info.get("latest_start_step", step_now + dynamic_lock_steps)))
            )
            self._tc_support_chain_class[str(truck_id)] = "support_required"
            goals[str(truck_id)] = str(task_id)
            if bool(self._docked_uav_sortie_chain_ready(env, str(uav_id), task)):
                goals[str(uav_id)] = str(task_id)
                self.tc_support_lock_to_dispatch_count_total = int(self.tc_support_lock_to_dispatch_count_total) + 1
            else:
                alt_tid = self._uav_direct_feasible_alternative_tc(env, str(uav_id), exclude_task_id=str(task_id))
                if alt_tid is not None:
                    goals[str(uav_id)] = str(alt_tid)
                    used_tasks.add(str(alt_tid))
                else:
                    goals[str(uav_id)] = str(truck_id)
            used_tasks.add(str(task_id))
            active_trucks.add(str(truck_id))
            active_uavs.add(str(uav_id))
            active_tasks.add(str(task_id))
            self.tc_support_lock_created_count_total = int(self.tc_support_lock_created_count_total) + 1

    def _uav_assignment_shortlist_task_ids(self, env, aid: str) -> Optional[set]:
        """
        Build a lightweight per-UAV emergency shortlist to reduce infeasible/far
        cross-assignments in large maps. This is a candidate-space prefilter, not
        an execution-level hard mask.
        """
        topk = int(max(getattr(env.cfg, "hrl_uav_assignment_shortlist_topk", 4), 0))
        if bool(self.use_event_trigger):
            topk = int(min(topk, 2))
        if topk <= 0:
            return None
        radius_m = float(max(getattr(env.cfg, "hrl_uav_assignment_shortlist_radius_m", 2400.0), 0.0))
        dist_norm = float(max(self._distance_norm_m(env), 1.0))
        st = env.state.agents.get(str(aid), None)
        if st is None or st.kind != AgentKind.UAV or bool(getattr(st, "crashed", False)):
            return None

        entries: List[Tuple[float, str]] = []
        for task in env.state.tasks.values():
            if task.kind != TaskKind.EMERGENCY or task.status != TaskStatus.PENDING:
                continue
            # ERC path: shortlist should emphasize currently feasible sorties to
            # reduce high-frequency re-evaluation of obviously blocked targets.
            immediate_feasible = bool(self._uav_task_feasible(env, str(aid), task))
            prelaunch_assignable = bool(self._uav_task_prelaunch_assignable(env, str(aid), task))
            if bool(self.use_event_trigger) and (not immediate_feasible) and (not prelaunch_assignable):
                continue
            d = float(env._agent_distance_to_task(str(aid), task))
            if not np.isfinite(d):
                continue
            urg = float(self._norm_deadline_urgency(task, int(env.state.step_index)))
            island = 1.0 if self._is_island_task(env, task) else 0.0
            near = float(np.clip(1.0 - d / max(radius_m, 1.0), 0.0, 1.0))
            score = float(0.70 * near + 0.45 * urg + 0.22 * island - 0.18 * (d / dist_norm))
            if (not immediate_feasible) and prelaunch_assignable:
                score += float(0.18 * max(self._truck_route_progress_to_task(env, str(st.follow_target), task), 0.0))
            if st.follow_target is not None and d <= float(max(getattr(env.cfg, "uav_docked_heading_dispatch_radius_m", 1500.0), 1.0)):
                if self._uav_docked_heading_aligned_with_task(env, str(aid), task):
                    score += 0.12
            entries.append((score, str(task.task_id)))

        if not entries:
            return set()
        entries.sort(key=lambda x: float(x[0]), reverse=True)
        keep = set(str(tid) for _, tid in entries[: int(topk)])

        # Large-map/C safety valve: if a time-critical task has gone many steps
        # without any goal exposure, force it into at least one UAV shortlist so
        # it can enter the assignment chain instead of timing out unseen.
        extra_force = int(max(getattr(env.cfg, "hrl_timecritical_force_entry_shortlist_extra", 2), 0))
        if extra_force > 0:
            forced_entries: List[Tuple[float, str]] = []
            for task in env.state.tasks.values():
                if task.kind != TaskKind.EMERGENCY or task.status != TaskStatus.PENDING:
                    continue
                force_active = bool(self._timecritical_force_entry_active(env, task))
                far_exposure_active = bool(self._timecritical_far_exposure_active(env, task))
                if not (force_active or far_exposure_active):
                    continue
                d = float(env._agent_distance_to_task(str(aid), task))
                if not np.isfinite(d):
                    continue
                ratio = float(self._task_lifeline_ratio(task))
                gap_steps = float(self._task_goal_gap_steps(env, task))
                urg = float(np.clip(float(getattr(task, "urgency_score", self._norm_deadline_urgency(task, int(env.state.step_index)))), 0.0, 1.0))
                cover = float(np.clip(self._uav_emergency_cover_fraction(env, task), 0.0, 1.0))
                far_pressure = float(np.clip(d / max(dist_norm, 1.0), 0.0, 1.25))
                low_cover = float(1.0 - cover)
                force_score = float(
                    0.55 * (1.0 - ratio)
                    + 0.28 * urg
                    + 0.12 * np.clip(gap_steps / max(float(getattr(env.cfg, "hrl_timecritical_force_entry_min_gap_steps", 12)), 1.0), 0.0, 2.0)
                    + 0.18 * low_cover
                    + (0.16 * far_pressure if far_exposure_active else -0.10 * (d / dist_norm))
                )
                forced_entries.append((force_score, str(task.task_id)))
            if forced_entries:
                forced_entries.sort(key=lambda x: float(x[0]), reverse=True)
                for _, tid in forced_entries[:extra_force]:
                    keep.add(str(tid))

        far_extra = int(max(getattr(env.cfg, "hrl_timecritical_far_exposure_extra", 0), 0))
        if far_extra > 0:
            far_entries: List[Tuple[float, str]] = []
            for task in env.state.tasks.values():
                if task.kind != TaskKind.EMERGENCY or task.status != TaskStatus.PENDING:
                    continue
                if not self._timecritical_far_exposure_active(env, task):
                    continue
                d = float(env._agent_distance_to_task(str(aid), task))
                if not np.isfinite(d):
                    continue
                ratio = float(self._task_lifeline_ratio(task))
                gap_steps = float(self._task_goal_gap_steps(env, task))
                urg = float(np.clip(float(getattr(task, "urgency_score", self._norm_deadline_urgency(task, int(env.state.step_index)))), 0.0, 1.0))
                cover = float(np.clip(self._uav_emergency_cover_fraction(env, task), 0.0, 1.0))
                far_entries.append(
                    (
                        float(
                            0.42 * (1.0 - ratio)
                            + 0.30 * urg
                            + 0.22 * (1.0 - cover)
                            + 0.18 * np.clip(d / max(dist_norm, 1.0), 0.0, 1.35)
                            + 0.08 * np.clip(gap_steps / max(float(getattr(env.cfg, "hrl_timecritical_far_exposure_min_gap_steps", 2)), 1.0), 0.0, 2.0)
                        ),
                        str(task.task_id),
                    )
                )
            if far_entries:
                far_entries.sort(key=lambda x: float(x[0]), reverse=True)
                for _, tid in far_entries[:far_extra]:
                    keep.add(str(tid))

        # Always keep current emergency goal in shortlist to avoid avoidable churn.
        cur = self.state.goals.get(str(aid), None)
        if cur is not None:
            t = env.state.tasks.get(str(cur), None)
            if t is not None and t.status == TaskStatus.PENDING and t.kind == TaskKind.EMERGENCY:
                keep.add(str(cur))
        return keep

    def _score_uav_task(self, env, aid: str, task) -> float:
        step_idx = int(env.state.step_index)
        dist_m = float(env._agent_distance_to_task(aid, task))
        if not np.isfinite(dist_m):
            return -1e9
        urgency = self._norm_deadline_urgency(task, step_idx)
        eta_score = self._norm_eta_score(dist_m, float(getattr(env.cfg, "uav_max_speed_mps", 17.0)))
        risk = self._task_risk(env, task) if self.use_risk_term else 0.0
        margin = self._uav_task_margin(env, aid, task)
        margin_term = float(np.clip(margin, -1.0, 1.0))
        emer_bonus = 1.0 if task.kind == TaskKind.EMERGENCY else 0.0
        is_timecritical = bool(self._is_timecritical_lightweight_task(task))
        lifeline_ratio = float(self._task_lifeline_ratio(task)) if is_timecritical else 1.0
        lifeline_criticality = float(1.0 - lifeline_ratio) if is_timecritical else 0.0
        urgency_direct = float(np.clip(float(getattr(task, "urgency_score", urgency)), 0.0, 1.0)) if is_timecritical else 0.0
        tc_w_urgency = float(max(getattr(env.cfg, "hrl_uav_timecritical_urgency_weight", 0.35), 0.0))
        tc_w_lifeline = float(max(getattr(env.cfg, "hrl_uav_timecritical_lifeline_weight", 0.55), 0.0))
        tc_critical_bonus_cfg = float(max(getattr(env.cfg, "hrl_uav_timecritical_critical_bonus", 0.55), 0.0))
        critical_thr = float(np.clip(getattr(env.cfg, "hrl_timecritical_lifeline_critical_ratio", 0.35), 0.0, 1.0))
        tc_critical_bonus = float(tc_critical_bonus_cfg if (is_timecritical and lifeline_ratio <= critical_thr) else 0.0)
        tc_force_entry_bonus = float(
            max(getattr(env.cfg, "hrl_timecritical_force_entry_uav_bonus", 0.32), 0.0)
            if (is_timecritical and self._timecritical_force_entry_active(env, task))
            else 0.0
        )
        normal_pressure, emergency_pressure = self._pending_task_pressure(env)
        dynamic_emergency_pressure_bonus = float((0.10 if self.use_event_trigger else 0.06) * emergency_pressure * emer_bonus)
        keep = self._keep_goal_bonus(aid, str(task.task_id))
        island_bonus = 1.0 if self._is_island_task(env, task) else 0.0
        island_cfg_bonus = float(max(getattr(env.cfg, "hrl_uav_island_delivery_bonus", 0.0), 0.0))
        map_bonus = 1.0 if self._map_update_active() else 0.0
        supported_sortie = self._supported_sortie_score(env, task)
        recovery_bonus = self._uav_recovery_feasibility_score(env, task)
        locality_term = self._uav_task_locality_term(env, aid, task)
        block_pressure = self._task_shared_map_block_pressure(env, task)
        map_retarget_bonus = float(block_pressure) if bool(map_bonus) and bool(island_bonus) else 0.0
        island_ids_now = set(getattr(env, "_current_island_emergency_task_ids", lambda: set())())
        non_island_penalty = 0.0
        if island_ids_now and (not bool(island_bonus)):
            if self._uav_has_feasible_island_task(env, str(aid), island_ids_now):
                pressure = float(np.clip(len(island_ids_now) / 3.0, 0.0, 1.0))
                non_island_penalty = float((0.10 + 0.08 * pressure) * (0.70 + 0.30 * normal_pressure))

        launch_bonus = 0.0
        continuity_bonus = 0.0
        near_dispatch_bonus = 0.0
        near_dispatch_penalty = 0.0
        ride_stall_bonus = 0.0
        reservation_term = self._uav_task_reservation_term(env, str(aid), str(task.task_id))
        st = env.state.agents.get(str(aid), None)
        near_depot_direct_dispatch = self._uav_near_depot_direct_dispatch_term(env, aid, task, float(dist_m))
        initial_directional_task_term = self._uav_initial_task_direction_term(env, aid, task)
        route_lookahead_term = 0.0

        endgame_term = 0.0
        max_steps = int(max(getattr(env.cfg, "max_steps", 0), 0))
        rem_episode = int(max(max_steps - step_idx, 0))
        endgame_window = int(max(getattr(env.cfg, "hrl_endgame_window_steps", 70), 0))
        if endgame_window > 0:
            endgame_scale = float(np.clip((float(endgame_window) - float(rem_episode)) / float(max(endgame_window, 1)), 0.0, 1.0))
            if endgame_scale > 1e-9:
                far_thr = float(max(getattr(env.cfg, "hrl_endgame_far_distance_m", 2200.0), 1.0))
                far_ratio = float(np.clip((float(dist_m) - far_thr) / max(far_thr, 1.0), 0.0, 1.0))
                near_bonus = float(max(getattr(env.cfg, "hrl_endgame_uav_near_emergency_bonus", 0.14), 0.0))
                far_pen = float(max(getattr(env.cfg, "hrl_endgame_uav_far_emergency_penalty", 0.10), 0.0))
                endgame_term = float(endgame_scale * (near_bonus * float(np.clip(eta_score, 0.0, 1.0)) - far_pen * far_ratio))

        if st is not None and st.kind == AgentKind.UAV and st.follow_target is not None:
            if hasattr(env, "_uav_loaded") and bool(env._uav_loaded(str(aid))):
                launch_bonus = float(0.10 * eta_score + 0.12 * margin_term + 0.06 * supported_sortie)

                near_primary = float(max(getattr(env.cfg, "uav_docked_near_dispatch_radius_m", 900.0), 1.0))
                near_secondary = float(max(getattr(env.cfg, "uav_docked_heading_dispatch_radius_m", 1500.0), near_primary))
                near_bonus = float(max(getattr(env.cfg, "uav_docked_near_dispatch_bonus", 0.65), 0.0))
                heading_bonus = float(max(getattr(env.cfg, "uav_docked_heading_dispatch_bonus", 0.35), 0.0))
                far_penalty = float(max(getattr(env.cfg, "uav_docked_far_task_penalty", 0.45), 0.0))

                if float(dist_m) <= near_primary:
                    near_dispatch_bonus += near_bonus
                elif float(dist_m) <= near_secondary and self._uav_docked_heading_aligned_with_task(env, str(aid), task):
                    near_dispatch_bonus += heading_bonus

                route_lookahead_w = float(max(getattr(env.cfg, "hrl_truck_task_lookahead_weight", 0.28), 0.0))
                route_lookahead_term += float(
                    route_lookahead_w * self._truck_route_progress_to_task(env, str(st.follow_target), task)
                )

                nearest_feasible = float(self._uav_nearest_feasible_emergency_distance(env, str(aid)))
                if np.isfinite(nearest_feasible) and nearest_feasible <= near_primary and float(dist_m) > float(near_primary + 50.0):
                    near_dispatch_penalty += far_penalty
                ride_stall_bonus += float(self._uav_ride_stall_bonus_term(env, str(aid), task, float(dist_m)))

        prev_goal = self.state.goals.get(str(aid), None)
        if self.use_event_trigger and st is not None and st.kind == AgentKind.UAV and st.follow_target is None and prev_goal is not None and str(prev_goal) == str(task.task_id):
            d_cur = float(env._agent_distance_to_task(str(aid), task))
            lock_dist = float(max(getattr(env.cfg, "uav_goal_terminal_lock_distance_m", 180.0), 1.0))
            if np.isfinite(d_cur):
                continuity_bonus = float(np.clip(1.0 - d_cur / lock_dist, 0.0, 1.0))

        ablate_event_bonus = bool(getattr(env.cfg, "erc_ablate_event_scoring_bonus", False))
        if ablate_event_bonus:
            self.event_scoring_bonus_blocked_by_ablation_count_total = int(self.event_scoring_bonus_blocked_by_ablation_count_total) + 1
        else:
            self.event_scoring_bonus_applied_count_total = int(self.event_scoring_bonus_applied_count_total) + 1
        if ablate_event_bonus:
            pass
        event_gain = float(self._event_bonus_gain(env))

        return float(
            self.weights.uav_urgency * urgency
            + self.weights.uav_eta * eta_score
            - self.weights.uav_risk * risk
            + self.weights.uav_margin * margin_term
            + self.weights.uav_emergency_bonus * emer_bonus
            + (0.68 if self.use_event_trigger else 0.62) * island_bonus
            + island_cfg_bonus * island_bonus
            + (0.10 if self.use_event_trigger else 0.12) * map_bonus
            + (0.38 if self.use_event_trigger else 0.36) * supported_sortie
            + (0.25 if self.use_event_trigger else 0.22) * recovery_bonus
            + dynamic_emergency_pressure_bonus
            + tc_w_urgency * urgency_direct
            + tc_w_lifeline * lifeline_criticality
            + tc_critical_bonus
            + tc_force_entry_bonus
            + 0.18 * map_retarget_bonus
            + locality_term
            - non_island_penalty
            + launch_bonus
            + near_dispatch_bonus
            - near_dispatch_penalty
            + route_lookahead_term
            + ride_stall_bonus
            + reservation_term
            + near_depot_direct_dispatch
            + initial_directional_task_term
            + (0.10 if self.use_event_trigger else 0.00) * continuity_bonus
            + endgame_term
            + event_gain * (0.68 * island_bonus + 0.22 * map_bonus + 0.30 * supported_sortie + 0.18 * map_retarget_bonus + 0.50 * island_cfg_bonus * island_bonus)
            + keep
        )

    def _score_uav_truck(self, env, aid: str, truck_id: str, dist_m: float) -> float:
        st = env.state.agents[aid]
        req = self._required_rth_battery(env, aid, float(max(dist_m, 0.0)))
        req = float((req + self.service_battery_buffer) * self._rth_safety_factor(env))
        batt = float(max(st.battery, 1e-6))
        recovery_need = float(max((req - batt) / batt, 0.0))
        dist_term = float(1.0 / (1.0 + max(dist_m, 0.0) / self._distance_norm_m(env)))
        keep = self._keep_goal_bonus(aid, str(truck_id))

        truck = env.state.agents.get(str(truck_id), None)
        truck_supply_bonus = 0.0
        island_support_bonus = 0.0
        emergency_pull_bonus = self._truck_emergency_pull_score(env, str(truck_id))
        follower_balance = self._truck_follower_balance_score(env, aid, str(truck_id))
        supported_sortie_readiness = self._uav_supported_sortie_readiness_from_truck(env, str(aid), str(truck_id))
        if truck is not None and truck.kind == AgentKind.TRUCK:
            emer_units = float(max(getattr(truck, "emergency_supply_units", 0), 0))
            truck_supply_bonus = float(np.clip(emer_units / max(getattr(env.cfg, "truck_initial_emergency_supply_units", 1), 1), 0.0, 1.0))
            island_ids = set(getattr(env, "_current_island_emergency_task_ids", lambda: set())())
            if island_ids:
                txy = self._agent_xy(env, str(truck_id))
                island_nodes = [
                    env.state.tasks[str(tid)].demand_node
                    for tid in sorted(island_ids)
                    if str(tid) in env.state.tasks and env.state.tasks[str(tid)].status == TaskStatus.PENDING
                ]
                if island_nodes:
                    d_island = min(
                        float(np.hypot(float(env._node_xy(int(n))[0]) - float(txy[0]), float(env._node_xy(int(n))[1]) - float(txy[1])))
                        for n in island_nodes
                    )
                    island_support_bonus = float(1.0 / (1.0 + d_island / self._distance_norm_m(env)))

        bind_r = float(max(getattr(env.cfg, "uav_bind_radius_m", 50.0), 1.0))
        bind_term = float(np.clip(bind_r / max(max(dist_m, 1.0), bind_r), 0.0, 1.0))
        low_batt_thr = float(np.clip(getattr(env.cfg, "uav_low_battery_goal_lock_threshold", 0.35), 0.0, 1.0))
        low_batt_mode = bool(float(getattr(st, "battery", 0.0)) <= low_batt_thr)
        emergency_pull_w = float(max(getattr(env.cfg, "hrl_uav_truck_emergency_pull_weight", 0.20), 0.0))
        follower_balance_w = float(max(getattr(env.cfg, "hrl_uav_truck_follower_balance_weight", 0.18), 0.0))
        initial_truck_route_term = self._uav_initial_truck_route_term(env, str(aid), str(truck_id))
        anchor_task = self._uav_anchor_task(env, str(aid))
        lookahead_term = 0.0
        if anchor_task is not None:
            lookahead_w = float(max(getattr(env.cfg, "hrl_uav_truck_lookahead_weight", 0.32), 0.0))
            lookahead_term = float(lookahead_w * self._truck_route_progress_to_task(env, str(truck_id), anchor_task))

        ablate_event_bonus = bool(getattr(env.cfg, "erc_ablate_event_scoring_bonus", False))
        if ablate_event_bonus:
            self.event_scoring_bonus_blocked_by_ablation_count_total = int(self.event_scoring_bonus_blocked_by_ablation_count_total) + 1
        else:
            self.event_scoring_bonus_applied_count_total = int(self.event_scoring_bonus_applied_count_total) + 1
        event_gain = float(self._event_bonus_gain(env))
        if ablate_event_bonus:
            pass

        return float(
            self.weights.uav_recovery_need * recovery_need
            + self.weights.uav_truck_distance * dist_term
            + (0.55 if low_batt_mode else 0.20) * bind_term
            + 0.25 * truck_supply_bonus
            + (0.18 if low_batt_mode else 0.06) * island_support_bonus
            + emergency_pull_w * emergency_pull_bonus
            + follower_balance_w * follower_balance
            + ((0.10 if low_batt_mode else 0.18) * supported_sortie_readiness)
            + event_gain
            * (
                (0.25 if low_batt_mode else 0.05) * bind_term
                + 0.15 * truck_supply_bonus
                + (0.14 if low_batt_mode else 0.04) * island_support_bonus
                + 0.08 * emergency_pull_bonus
                + 0.08 * supported_sortie_readiness
            )
            + initial_truck_route_term
            + lookahead_term
            + keep
        )

    def _truck_proxy_support_goal_valid(self, env, aid: str, task) -> bool:
        if task is None or task.kind != TaskKind.EMERGENCY or task.status != TaskStatus.PENDING:
            return False
        chain = self._support_bound_chain_info_for_truck(env, str(aid))
        if (
            bool(getattr(env.cfg, "erc_tc_support_required_enabled", False))
            and chain is not None
            and str(chain.get("task_id", "")) == str(getattr(task, "task_id", ""))
        ):
            return True
        if not bool(self._is_island_task(env, task)):
            return False
        _pending_norm, norm_reach_by_truck, _any_reachable_normal = self._normal_reachability_snapshot(env)
        if bool(norm_reach_by_truck.get(str(aid), True)):
            return False
        if not bool(self._truck_emergency_relief_allowed(env, str(aid), task)):
            return False
        if not bool(self._truck_emergency_support_candidate(env, str(aid), task)):
            return False
        gain = float(np.clip(self._truck_support_serviceability_gain(env, str(aid), task), 0.0, 1.0))
        return bool(gain >= 0.12)

    def _truck_task_valid(self, env, aid: str, task_id: str) -> bool:
        task = env.state.tasks.get(str(task_id), None)
        if task is None or task.status != TaskStatus.PENDING:
            return False
        if self._truck_stage_blocks_task_goal(env, str(aid)):
            return False
        if not self._truck_task_serviceable_or_support_proxy(env, str(aid), task):
            return False
        if not bool(self._truck_emergency_relief_allowed(env, str(aid), task)):
            return False
        if bool(self._truck_task_reachable(env, str(aid), task)):
            return True
        return bool(self._truck_proxy_support_goal_valid(env, str(aid), task))

    def _nearest_truck_reachable_task(
        self,
        env,
        aid: str,
        *,
        used_tasks: Optional[set] = None,
        allow_used: bool = False,
    ) -> Optional[str]:
        st = env.state.agents.get(str(aid), None)
        if st is None or st.kind != AgentKind.TRUCK or bool(getattr(st, "crashed", False)):
            return None
        if self._truck_stage_blocks_task_goal(env, str(aid)):
            return None

        blocked = set(used_tasks or set())
        best_tid: Optional[str] = None
        best_dist = float("inf")
        best_priority = 10
        has_pending_normal = any(
            t.status == TaskStatus.PENDING and t.kind == TaskKind.NORMAL
            for t in env.state.tasks.values()
        )
        _, norm_reach_by_truck, any_reachable_normal = self._normal_reachability_snapshot(env)
        truck_has_normal_reachable = bool(norm_reach_by_truck.get(str(aid), True))
        hard_normal_first = bool(getattr(env.cfg, "hrl_truck_hard_normal_first_enabled", True))
        emergency_support_mode = bool(
            has_pending_normal
            and (not truck_has_normal_reachable)
            and bool(getattr(env.cfg, "hrl_truck_emergency_support_when_no_normal_enabled", True))
        )

        for task in env.state.tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
            tid = str(task.task_id)
            if (not allow_used) and (tid in blocked):
                continue
            if not self._truck_task_serviceable_or_support_proxy(env, str(aid), task):
                continue
            if not bool(self._truck_emergency_relief_allowed(env, str(aid), task)):
                continue
            reachable = bool(self._truck_task_reachable(env, str(aid), task))
            if task.kind == TaskKind.NORMAL and (not reachable):
                continue
            if task.kind == TaskKind.EMERGENCY and (not reachable) and (not bool(self._truck_proxy_support_goal_valid(env, str(aid), task))):
                continue
            d = float(self._truck_task_distance(env, str(aid), task))
            if not np.isfinite(d):
                continue
            eff_dist = float(d)
            if task.kind == TaskKind.NORMAL:
                priority = 0
            elif hard_normal_first and has_pending_normal and truck_has_normal_reachable and any_reachable_normal:
                continue
            elif emergency_support_mode and task.kind == TaskKind.EMERGENCY:
                if not bool(self._truck_supportworthy_emergency_task(env, str(aid), task)):
                    continue
                priority = 1
                gain = float(np.clip(self._truck_support_serviceability_gain(env, str(aid), task), 0.0, 1.0))
                urg = float(np.clip(self._norm_deadline_urgency(task, int(env.state.step_index)), 0.0, 1.0))
                eff_dist = float(d) * float(max(0.12, 1.0 - 0.60 * gain - 0.28 * urg))
            elif self._is_island_task(env, task) and self._truck_forward_support_score(env, str(aid), task) > 0.05:
                priority = 1
            elif task.kind == TaskKind.EMERGENCY and (not reachable) and bool(self._truck_proxy_support_goal_valid(env, str(aid), task)):
                priority = 1
                gain = float(np.clip(self._truck_support_serviceability_gain(env, str(aid), task), 0.0, 1.0))
                eff_dist = float(d) * float(max(0.15, 1.0 - 0.55 * gain))
            else:
                priority = 2 if has_pending_normal else 1
            better = False
            if int(priority) < int(best_priority):
                better = True
            elif int(priority) == int(best_priority) and eff_dist + 1e-9 < best_dist:
                better = True
            if better:
                best_priority = int(priority)
                best_dist = float(eff_dist)
                best_tid = str(tid)
        return best_tid
    def _truck_has_assignable_task(self, env, aid: str) -> bool:
        return bool(self._nearest_truck_reachable_task(env, str(aid), used_tasks=set(), allow_used=True) is not None)

    def _goal_hold_elapsed(self, env, aid: str) -> bool:
        assigned = self.state.goal_assigned_step.get(str(aid), None)
        if assigned is None:
            return True
        held = int(env.state.step_index) - int(assigned)
        hold_steps = int(max(getattr(env.cfg, "hrl_goal_min_hold_steps", self.min_goal_hold_steps), 0))
        # Scenario B has normal communication, so reproduce the useful part of
        # the C blackout behavior explicitly: keep a valid route commitment for
        # a short window. This is disabled in C (where the physical blackout
        # hold remains authoritative) and never overrides hard invalidity,
        # dead-end, or safety branches handled before this check.
        if (
            bool(getattr(env.cfg, "hrl_b_route_stability_enabled", False))
            and not bool(getattr(env.cfg, "enable_comm_blackout", False))
        ):
            hold_steps = max(
                hold_steps,
                int(max(getattr(env.cfg, "hrl_b_route_stability_hold_steps", 0), 0)),
            )
        return bool(held >= hold_steps)

    def _goal_stable_for_takeoff(self, env, aid: str) -> bool:
        assigned = self.state.goal_assigned_step.get(str(aid), None)
        if assigned is None:
            return False
        held = int(env.state.step_index) - int(assigned)
        req = int(max(self.stable_goal_before_takeoff_steps, 0))
        chain = self._support_bound_chain_info_for_uav(env, str(aid))
        current_goal = self.state.goals.get(str(aid), None)
        if chain is not None and str(current_goal) == str(chain.get("task_id", "")):
            # Once a truck-created support chain has a concrete bound task/UAV pair,
            # let takeoff happen after a short positive dwell instead of waiting for
            # the full generic hold. This keeps support arrival from decaying away.
            req = int(min(req, 2))
        return bool(held >= req)

    def _support_bound_chain_info_for_truck(self, env, aid: str) -> Optional[Dict[str, object]]:
        aid_s = str(aid)
        until = int(self._support_bound_chain_until_step.get(aid_s, -1))
        if int(env.state.step_index) > until:
            return None
        task_id = str(self._support_bound_chain_task_id.get(aid_s, "")).strip()
        uav_id = str(self._support_bound_chain_uav_id.get(aid_s, "")).strip()
        anchor_node = self._support_bound_chain_anchor_node_by_truck.get(aid_s, None)
        if (not task_id) or (not uav_id):
            return None
        task = env.state.tasks.get(task_id, None)
        uav = env.state.agents.get(uav_id, None)
        truck = env.state.agents.get(aid_s, None)
        if (
            task is None
            or task.status != TaskStatus.PENDING
            or uav is None
            or uav.kind != AgentKind.UAV
            or bool(getattr(uav, "crashed", False))
            or truck is None
            or truck.kind != AgentKind.TRUCK
            or bool(getattr(truck, "crashed", False))
        ):
            return None
        return {
            "truck_id": aid_s,
            "task_id": task_id,
            "uav_id": uav_id,
            "task": task,
            "uav": uav,
            "truck": truck,
            "until_step": int(until),
            "anchor_node": anchor_node,
        }

    def _support_bound_chain_info_for_uav(self, env, aid: str) -> Optional[Dict[str, object]]:
        truck_id = str(self._support_bound_chain_truck_by_uav.get(str(aid), "")).strip()
        if not truck_id:
            return None
        info = self._support_bound_chain_info_for_truck(env, truck_id)
        if info is None:
            return None
        if str(info.get("uav_id", "")) != str(aid):
            return None
        return info

    def _goal_valid_and_safe(self, env, aid: str, goal_id: Optional[str]) -> bool:
        if goal_id is None:
            return False
        st = env.state.agents.get(str(aid), None)
        if st is None:
            return False
        t = env.state.tasks.get(str(goal_id), None)
        if t is not None:
            if t.status != TaskStatus.PENDING:
                return False
            if st.kind == AgentKind.TRUCK:
                return self._truck_task_valid(env, str(aid), str(goal_id))
            if st.kind == AgentKind.UAV:
                # Airborne emergency execution is not invalidated by road-path events.
                if t.kind == TaskKind.EMERGENCY and getattr(st, "follow_target", None) is None:
                    if self._uav_needs_recovery(env, str(aid)):
                        return False
                    if self._comm_degraded(env, str(aid)):
                        return False
                    if self._uav_task_hard_risk_blocked(env, t):
                        return False
                    return True
                return self._uav_task_feasible(env, str(aid), t)
            return False
        ag = env.state.agents.get(str(goal_id), None)
        if st.kind == AgentKind.UAV and ag is not None and ag.kind == AgentKind.TRUCK:
            return bool(not bool(ag.crashed))
        return False

    def _docked_uav_soft_invalid_hold(self, env, aid: str, goal_id: Optional[str]) -> bool:
        if not bool(getattr(env.cfg, "hrl_docked_uav_soft_invalid_hold_enabled", False)):
            return False
        st = env.state.agents.get(str(aid), None)
        task = env.state.tasks.get(str(goal_id), None) if goal_id is not None else None
        if (
            st is None
            or st.kind != AgentKind.UAV
            or getattr(st, "follow_target", None) is None
            or task is None
            or task.kind != TaskKind.EMERGENCY
            or task.status != TaskStatus.PENDING
        ):
            return False
        # Keep waiting only while docked to a live truck and loaded. The launch
        # gate still decides whether the sortie is safe enough to leave.
        truck = env.state.agents.get(str(st.follow_target), None)
        if truck is None or truck.kind != AgentKind.TRUCK or bool(getattr(truck, "crashed", False)):
            return False
        loaded_fn = getattr(env, "_uav_loaded", None)
        if callable(loaded_fn):
            try:
                if not bool(loaded_fn(str(aid))):
                    return False
            except Exception:
                return False
        return bool(self._uav_task_prelaunch_assignable(env, str(aid), task))

    def _score_goal_for_agent(self, env, aid: str, goal_id: Optional[str]) -> float:
        if goal_id is None:
            return float("-inf")
        st = env.state.agents.get(str(aid), None)
        if st is None:
            return float("-inf")
        t = env.state.tasks.get(str(goal_id), None)
        if t is not None:
            if t.status != TaskStatus.PENDING:
                return float("-inf")
            if st.kind == AgentKind.TRUCK:
                if self._truck_stage_blocks_task_goal(env, str(aid)):
                    return float("-inf")
                if not self._truck_task_serviceable_or_support_proxy(env, str(aid), t):
                    return float("-inf")
                if not self._truck_task_reachable(env, str(aid), t):
                    return float("-inf")
                if self._region_commitment_active(env) and not self._region_task_allowed(env, str(aid), t):
                    return float("-inf")
                return float(self._score_truck_task(env, str(aid), t) + self._region_score_adjustment(env, str(aid), t))
            if st.kind == AgentKind.UAV:
                if (not self._uav_task_feasible(env, str(aid), t)) and (not self._uav_task_prelaunch_assignable(env, str(aid), t)):
                    return float("-inf")
                if self._region_commitment_active(env) and not self._region_task_allowed(env, str(aid), t):
                    return float("-inf")
                return float(self._score_uav_task(env, str(aid), t) + self._region_score_adjustment(env, str(aid), t))
            return float("-inf")
        ag = env.state.agents.get(str(goal_id), None)
        if st.kind == AgentKind.UAV and ag is not None and ag.kind == AgentKind.TRUCK:
            ax, ay = self._agent_xy(env, str(aid))
            tx, ty = self._agent_xy(env, str(goal_id))
            d = float(((ax - tx) ** 2 + (ay - ty) ** 2) ** 0.5)
            return float(self._score_uav_truck(env, str(aid), str(goal_id), d))
        return float("-inf")

    def _airborne_tc_completion_grace_safe(self, env, aid: str, task) -> bool:
        if not bool(getattr(env.cfg, "hrl_airborne_tc_completion_grace_enabled", False)):
            return False
        if task is None or not self._is_timecritical_lightweight_task(task) or task.status != TaskStatus.PENDING:
            return False
        st = env.state.agents.get(str(aid), None)
        if st is None or st.kind != AgentKind.UAV or bool(getattr(st, "crashed", False)):
            return False
        if getattr(st, "follow_target", None) is not None:
            return False
        if self._comm_degraded(env, str(aid)) or self._uav_task_hard_risk_blocked(env, task):
            return False
        loaded_fn = getattr(env, "_uav_loaded", None)
        if callable(loaded_fn):
            try:
                if not bool(loaded_fn(str(aid))):
                    return False
            except Exception:
                return False
        if bool(getattr(st, "uav_needs_reload_flag", False)) or float(getattr(st, "cargo", 0.0)) <= 0.0:
            return False

        dist_m = float(env._agent_distance_to_task(str(aid), task))
        radius_m = float(max(getattr(env.cfg, "hrl_airborne_tc_completion_grace_radius_m", 950.0), 0.0))
        if (not np.isfinite(dist_m)) or dist_m > radius_m:
            return False

        v_ref = float(max(getattr(env.cfg, "uav_max_speed_mps", 1.0), 1e-6))
        dt = float(max(getattr(env.cfg, "dt_seconds", 1.0), 1e-6))
        service_steps = int(max(getattr(env.cfg, "uav_service_time_steps", getattr(env.cfg, "service_time_steps", 1)), 1))
        needed_steps = int(np.ceil(max(dist_m, 0.0) / max(v_ref * dt, 1e-6))) + service_steps
        min_life_steps = int(max(getattr(env.cfg, "hrl_airborne_tc_completion_grace_min_lifeline_steps", 4), 0))
        if float(self._tc_steps_remaining(env, task)) < float(needed_steps + min_life_steps):
            return False

        batt = float(getattr(st, "battery", 0.0))
        min_batt = float(np.clip(getattr(env.cfg, "hrl_airborne_tc_completion_grace_min_battery", 0.16), 0.0, 1.0))
        if batt < min_batt:
            return False

        recovery_buf = float(max(getattr(env.cfg, "uav_recovery_distance_buffer_m", 600.0), 0.0))
        if hasattr(env, "_effective_recovery_buffer_for_sortie"):
            try:
                recovery_buf = float(env._effective_recovery_buffer_for_sortie(str(aid), task, launch_reason="rendezvous_safe"))
            except Exception:
                recovery_buf = float(max(getattr(env.cfg, "uav_recovery_distance_buffer_m", 600.0), 0.0))
        buffer_scale = float(np.clip(getattr(env.cfg, "hrl_airborne_tc_completion_grace_recovery_buffer_scale", 0.35), 0.0, 1.0))
        bind_r = float(max(getattr(env.cfg, "uav_bind_radius_m", 170.0), 1.0))
        supported_recovery_m = float(max(2.0 * bind_r, buffer_scale * max(recovery_buf, 0.0)))
        req = float(self._required_rth_battery(env, str(aid), max(dist_m, 0.0) + supported_recovery_m))
        req = float((req + self.service_battery_buffer) * self._rth_safety_factor(env))
        return bool(np.isfinite(req) and batt >= req)

    def _hard_safety_override(self, env, aid: str, current_goal: Optional[str]) -> bool:
        st = env.state.agents.get(str(aid), None)
        if st is None:
            return False
        if current_goal is None:
            return False

        # Resolved/invalid goal should bypass hold/hysteresis immediately.
        if (
            not self._goal_valid_and_safe(env, str(aid), current_goal)
            and not self._docked_uav_soft_invalid_hold(env, str(aid), current_goal)
        ):
            return True

        # Severe risk condition can bypass hold when current UAV task is hard-blocked.
        if bool(env.state.hazard.risk_spike):
            t = env.state.tasks.get(str(current_goal), None)
            if st.kind == AgentKind.UAV and t is not None and self._uav_task_hard_risk_blocked(env, t):
                return True

        # Emergency battery recovery overrides goal hold.
        if st.kind == AgentKind.UAV and self._uav_needs_recovery(env, str(aid)):
            ag = env.state.agents.get(str(current_goal), None)
            if ag is None or ag.kind != AgentKind.TRUCK:
                t = env.state.tasks.get(str(current_goal), None)
                if self._airborne_tc_completion_grace_safe(env, str(aid), t):
                    return False
                return True

        return False

    def _should_preempt_for_support_bound_dispatch(
        self,
        env,
        aid: str,
        current_goal: Optional[str],
        proposed_goal: Optional[str],
    ) -> bool:
        if proposed_goal is None or current_goal is None:
            return False
        st = env.state.agents.get(str(aid), None)
        if st is None or st.kind != AgentKind.UAV or bool(getattr(st, "crashed", False)):
            return False
        if st.follow_target is None:
            return False
        chain = self._support_bound_chain_info_for_uav(env, str(aid))
        if chain is None:
            return False
        chain_task_id = str(chain.get("task_id", "")).strip()
        chain_truck_id = str(chain.get("truck_id", "")).strip()
        if (not chain_task_id) or (not chain_truck_id):
            return False
        if str(proposed_goal) != chain_task_id:
            return False
        if str(current_goal) != chain_truck_id:
            return False
        if str(getattr(st, "follow_target", "")) != chain_truck_id:
            return False
        task = env.state.tasks.get(chain_task_id, None)
        if not self._task_planner_active(task):
            return False
        docked_actionable = False
        docked_actionable_fn = getattr(env, "_uav_docked_task_actionable_now", None)
        if callable(docked_actionable_fn):
            try:
                docked_actionable = bool(docked_actionable_fn(str(aid), task))
            except Exception:
                docked_actionable = False
        if docked_actionable:
            return True
        if bool(getattr(st, "uav_needs_reload_flag", False)):
            return False
        loaded_fn = getattr(env, "_uav_loaded", None)
        if callable(loaded_fn):
            try:
                if not bool(loaded_fn(str(aid))):
                    return False
            except Exception:
                return False
        return True

    def _should_preempt_for_critical_timecritical(
        self,
        env,
        aid: str,
        current_goal: Optional[str],
        proposed_goal: Optional[str],
    ) -> bool:
        if proposed_goal is None or current_goal is None:
            return False
        if str(proposed_goal) == str(current_goal):
            return False
        st = env.state.agents.get(str(aid), None)
        if st is None:
            return False
        # Never bypass hard safety recovery.
        if st.kind == AgentKind.UAV and self._uav_needs_recovery(env, str(aid)):
            return False
        proposed_task = env.state.tasks.get(str(proposed_goal), None)
        current_task = env.state.tasks.get(str(current_goal), None)
        if proposed_task is None or proposed_task.status != TaskStatus.PENDING:
            return False
        if not self._is_timecritical_lightweight_task(proposed_task):
            return False
        critical = float(np.clip(getattr(env.cfg, "hrl_timecritical_lifeline_critical_ratio", 0.35), 0.0, 1.0))
        if float(self._task_lifeline_ratio(proposed_task)) > critical:
            return False
        proposed_tier = int(self._task_priority_tier(env, str(aid), proposed_task))
        current_tier = int(self._task_priority_tier(env, str(aid), current_task)) if current_task is not None else -1
        if proposed_tier <= current_tier:
            return False
        return True

    def _apply_switch_hysteresis(
        self,
        env,
        aid: str,
        proposed_goal: Optional[str],
        used_tasks: set,
    ) -> Optional[str]:
        current_goal = self.state.goals.get(str(aid), None)
        st_gate = env.state.agents.get(str(aid), None)
        cur_task_gate = env.state.tasks.get(str(current_goal), None) if current_goal is not None else None
        self._record_switch_decision(str(aid), "none", "")
        if current_goal is None:
            return proposed_goal

        # Hard safety path: bypass hold and margin.
        if self._hard_safety_override(env, str(aid), current_goal):
            if (
                st_gate is not None
                and st_gate.kind == AgentKind.UAV
                and getattr(st_gate, "follow_target", None) is None
                and cur_task_gate is not None
                and cur_task_gate.kind == TaskKind.EMERGENCY
            ):
                self.uav_airborne_safety_abort_count_total = int(self.uav_airborne_safety_abort_count_total) + 1
            self._record_switch_decision(str(aid), "forced", "infeasible")
            return proposed_goal

        # Airborne emergency lock: once UAV is airborne on an emergency task,
        # block routine/path/ranking-driven retarget and keep current mission.
        if (
            st_gate is not None
            and st_gate.kind == AgentKind.UAV
            and getattr(st_gate, "follow_target", None) is None
            and cur_task_gate is not None
            and cur_task_gate.kind == TaskKind.EMERGENCY
            and cur_task_gate.status == TaskStatus.PENDING
        ):
            hard_abort = bool(
                self._uav_needs_recovery(env, str(aid))
                or self._comm_degraded(env, str(aid))
                or self._uav_task_hard_risk_blocked(env, cur_task_gate)
            )
            if (not hard_abort) and proposed_goal is not None and str(proposed_goal) != str(current_goal):
                self.uav_airborne_goal_switch_blocked_count_total = int(self.uav_airborne_goal_switch_blocked_count_total) + 1
                self._record_switch_decision(str(aid), "rejected", "airborne_lock")
                return str(current_goal)

        # Support relay forced switch (same-step): if planner reserved this truck
        # for emergency support relay, bypass truck hysteresis once.
        if proposed_goal is not None:
            st_force = env.state.agents.get(str(aid), None)
            p_task = env.state.tasks.get(str(proposed_goal), None)
            now_step = int(env.state.step_index)
            force_step = int(self._support_relay_force_step.get(str(aid), -1))
            if (
                st_force is not None
                and st_force.kind == AgentKind.TRUCK
                and p_task is not None
                and p_task.kind == TaskKind.EMERGENCY
                and now_step <= force_step
            ):
                self._record_switch_decision(str(aid), "forced", "stall")
                return proposed_goal

        if proposed_goal is not None:
            st_force = env.state.agents.get(str(aid), None)
            p_task = env.state.tasks.get(str(proposed_goal), None)
            now_step = int(env.state.step_index)
            force_step = int(self._far_routine_bootstrap_force_step.get(str(aid), -1))
            force_window = int(max(getattr(env.cfg, "hrl_far_routine_bootstrap_window_steps", 20), 0))
            if (
                st_force is not None
                and st_force.kind == AgentKind.TRUCK
                and p_task is not None
                and p_task.kind == TaskKind.NORMAL
                and force_step >= 0
                and now_step <= force_step + force_window
            ):
                self._record_switch_decision(str(aid), "forced", "stall")
                return proposed_goal

        # Support-bound dispatch preemption: once a truck-created chain is
        # actually actionable now, let the UAV switch from truck-follow goal to
        # the bound task immediately instead of waiting behind generic hold logic.
        if self._should_preempt_for_support_bound_dispatch(env, str(aid), current_goal, proposed_goal):
            self._record_switch_decision(str(aid), "forced", "stall")
            return proposed_goal

        # Anchor-to-TC readiness guard: when a docked UAV is following a truck,
        # do not retarget it to a distant emergency unless the current truck
        # anchor can actually launch a complete sortie now. This keeps the
        # truck-UAV pair moving toward a useful recovery/launch anchor instead
        # of repeatedly accepting paper-feasible TC switches that stall.
        if (
            bool(getattr(env.cfg, "hrl_uav_anchor_to_tc_requires_actionable_enabled", False))
            and st_gate is not None
            and st_gate.kind == AgentKind.UAV
            and proposed_goal is not None
            and str(proposed_goal) != str(current_goal)
        ):
            cur_anchor_agent = env.state.agents.get(str(current_goal), None)
            prop_task_gate = env.state.tasks.get(str(proposed_goal), None)
            if (
                cur_anchor_agent is not None
                and cur_anchor_agent.kind == AgentKind.TRUCK
                and prop_task_gate is not None
                and prop_task_gate.kind == TaskKind.EMERGENCY
                and prop_task_gate.status == TaskStatus.PENDING
            ):
                docked_actionable = False
                docked_fn = getattr(env, "_uav_docked_task_actionable_now", None)
                if callable(docked_fn):
                    try:
                        docked_actionable = bool(docked_fn(str(aid), prop_task_gate))
                    except Exception:
                        docked_actionable = False
                if not docked_actionable:
                    self._record_switch_decision(str(aid), "rejected", "anchor_not_actionable")
                    return str(current_goal)

        # Critical time-critical preemption: bypass hold/hysteresis when the
        # proposed goal is a lower-lifeline tier and safety-recovery is not active.
        if self._should_preempt_for_critical_timecritical(env, str(aid), current_goal, proposed_goal):
            self._record_switch_decision(str(aid), "forced", "infeasible")
            return proposed_goal

        # Respect minimum hold duration while current goal is still valid/safe.
        keep_current = self._can_keep_prev_goal(env, str(aid), used_tasks=used_tasks)
        if keep_current is None:
            if st_gate is not None and st_gate.kind == AgentKind.UAV and cur_task_gate is not None and cur_task_gate.kind == TaskKind.EMERGENCY:
                self.uav_emergency_commit_break_hard_invalid_count_total = int(
                    self.uav_emergency_commit_break_hard_invalid_count_total
                ) + 1
            return proposed_goal

        now_step = int(env.state.step_index)
        st_anchor = env.state.agents.get(str(aid), None)
        cur_task_anchor = env.state.tasks.get(str(keep_current), None)
        cur_score_anchor = float(self._score_goal_for_agent(env, str(aid), keep_current))

        # A blackout freezes the physical effective goal in the environment.
        # Keep the planner's commitment aligned with that execution reality,
        # rather than repeatedly proposing alternatives that cannot be sent.
        if (
            bool(getattr(env.cfg, "hrl_comm_blackout_commit_hold_enabled", True))
            and bool(getattr(env, "comm_blocked", {}).get(str(aid), False))
        ):
            self.comm_blackout_commit_hold_count_total = int(self.comm_blackout_commit_hold_count_total) + 1
            self._record_switch_decision(str(aid), "rejected", "comm_blackout_hold")
            return str(keep_current)

        # Generic task A->B->A guard.  Existing guards cover truck NORMAL and
        # UAV truck-anchor loops; L diagnostics also exposed emergency-task
        # bounce-back, so protect any same-kind pending task pair here.
        if proposed_goal is not None and str(proposed_goal) != str(keep_current):
            proposed_task_aba = env.state.tasks.get(str(proposed_goal), None)
            if (
                cur_task_anchor is not None
                and proposed_task_aba is not None
                and cur_task_anchor.kind == proposed_task_aba.kind
                and cur_task_anchor.status == TaskStatus.PENDING
                and proposed_task_aba.status == TaskStatus.PENDING
            ):
                previous_goal = str(self._task_recent_prev_goal.get(str(aid), ""))
                previous_switch = int(self._task_recent_switch_step.get(str(aid), -10**9))
                window = int(max(getattr(env.cfg, "hrl_task_aba_block_steps", 18), 0))
                if window > 0 and previous_goal and str(proposed_goal) == previous_goal and now_step - previous_switch <= window:
                    self.task_aba_switch_blocked_count_total = int(self.task_aba_switch_blocked_count_total) + 1
                    self._record_switch_decision(str(aid), "rejected", "task_aba")
                    return str(keep_current)

        # UAV truck-anchor A<->B de-oscillation guard:
        # when a UAV is bouncing between two truck follow-goals in a short window,
        # keep current anchor unless a hard override already fired above.
        if (
            st_anchor is not None
            and st_anchor.kind == AgentKind.UAV
            and proposed_goal is not None
            and str(proposed_goal) != str(keep_current)
        ):
            cur_anchor = env.state.agents.get(str(keep_current), None)
            new_anchor = env.state.agents.get(str(proposed_goal), None)
            if (
                cur_anchor is not None
                and new_anchor is not None
                and cur_anchor.kind == AgentKind.TRUCK
                and new_anchor.kind == AgentKind.TRUCK
            ):
                aba_block_steps = int(max(getattr(env.cfg, "hrl_uav_truck_anchor_aba_block_steps", 14), 0))
                prev_anchor = str(self._uav_recent_truck_anchor_prev_goal.get(str(aid), "")).strip()
                last_switch_step = int(self._uav_recent_truck_anchor_switch_step.get(str(aid), -10**9))
                if (
                    aba_block_steps > 0
                    and prev_anchor
                    and str(proposed_goal) == prev_anchor
                    and (int(now_step) - int(last_switch_step)) <= int(aba_block_steps)
                ):
                    self._record_switch_decision(str(aid), "rejected", "anchor_aba")
                    return str(keep_current)

        # Progress-aware UAV emergency execution commitment:
        # keep current emergency execution chain if it is already actionable/progressing.
        if (
            bool(getattr(env.cfg, "hrl_uav_emergency_commit_hold_enabled", True))
            and
            st_anchor is not None
            and st_anchor.kind == AgentKind.UAV
            and cur_task_anchor is not None
            and cur_task_anchor.kind == TaskKind.EMERGENCY
            and cur_task_anchor.status == TaskStatus.PENDING
        ):
            cur_dist = float(self._switch_goal_distance(env, str(aid), str(keep_current)))
            prog5 = float(self._switch_goal_progress_recent(env, str(aid), str(keep_current), 5))
            near_radius = bool(np.isfinite(cur_dist) and cur_dist <= float(max(getattr(env.cfg, "uav_delivery_radius_m", 40.0), 1.0)))
            service_started = bool(cur_task_anchor.first_service_step is not None)
            docked_actionable = False
            docked_fn = getattr(env, "_uav_docked_task_actionable_now", None)
            if callable(docked_fn):
                try:
                    docked_actionable = bool(docked_fn(str(aid), cur_task_anchor))
                except Exception:
                    docked_actionable = False
            airborne = bool(getattr(st_anchor, "follow_target", None) is None)
            recovery_feasible = bool(self._uav_task_feasible(env, str(aid), cur_task_anchor))
            progress_clear = bool(np.isfinite(prog5) and prog5 > 15.0)
            hold_active = bool(
                docked_actionable
                or airborne
                or progress_clear
                or near_radius
                or service_started
                or recovery_feasible
            )
            if hold_active:
                self.uav_emergency_commit_hold_count_total = int(self.uav_emergency_commit_hold_count_total) + 1
                if proposed_goal is not None and str(proposed_goal) != str(keep_current):
                    self.uav_emergency_commit_prevented_switch_count_total = int(
                        self.uav_emergency_commit_prevented_switch_count_total
                    ) + 1
                    self._uav_commit_hold_pending.append(
                        {
                            "step": int(now_step),
                            "launch_base": int(getattr(env, "uav_launch_count_total", 0)),
                            "delivery_base": int(getattr(env, "uav_delivery_count_total", 0)),
                        }
                    )
                return str(keep_current)

        # Support-closure commitment: once a truck support candidate has formed a
        # concrete truck+UAV+task chain, keep that chain stable long enough to turn
        # support arrival into an actual launch, not just a paper binding.
        if st_anchor is not None and st_anchor.kind == AgentKind.TRUCK:
            bound_chain = self._support_bound_chain_info_for_truck(env, str(aid))
            if bound_chain is not None:
                chain_task = bound_chain.get("task", None)
                chain_uav = bound_chain.get("uav", None)
                chain_tid = str(bound_chain.get("task_id", ""))
                current_matches_chain = bool(str(current_goal) == chain_tid or str(keep_current) == chain_tid)
                chain_bind_ok = bool(chain_task is not None and self._support_binding_is_strong_enough(env, chain_task, {"bound_any": 1.0, "bound_timecritical": 1.0, "bound_eta_steps": 0.0}, gain_info=self._support_anchor_service_gain(env, str(aid), chain_task)))
                if (
                    current_matches_chain
                    and chain_bind_ok
                    and chain_task is not None
                    and chain_uav is not None
                    and bool(getattr(chain_uav, "follow_target", None))
                    and str(getattr(chain_uav, "follow_target", "")) == str(aid)
                    and (not self._uav_needs_recovery(env, str(bound_chain.get("uav_id", ""))))
                    and self._truck_task_valid(env, str(aid), chain_tid)
                ):
                    if proposed_goal is None or str(proposed_goal) != chain_tid:
                        return chain_tid

        if st_anchor is not None and st_anchor.kind == AgentKind.UAV:
            bound_chain = self._support_bound_chain_info_for_uav(env, str(aid))
            if bound_chain is not None:
                chain_task = bound_chain.get("task", None)
                chain_tid = str(bound_chain.get("task_id", ""))
                chain_truck = str(bound_chain.get("truck_id", ""))
                current_matches_chain = bool(str(current_goal) in {chain_tid, chain_truck, str(keep_current)})
                chain_bind_ok = bool(chain_task is not None and self._support_binding_is_strong_enough(env, chain_task, {"bound_any": 1.0, "bound_timecritical": 1.0, "bound_eta_steps": 0.0}, gain_info=self._support_anchor_service_gain(env, chain_truck, chain_task)))
                if (
                    current_matches_chain
                    and chain_bind_ok
                    and chain_task is not None
                    and str(getattr(st_anchor, "follow_target", "")) == chain_truck
                    and (not self._uav_needs_recovery(env, str(aid)))
                ):
                    if bool(self._docked_uav_sortie_chain_ready(env, str(aid), chain_task)):
                        if proposed_goal is None or str(proposed_goal) != chain_tid:
                            return chain_tid
                    else:
                        if proposed_goal is None or str(proposed_goal) != chain_truck:
                            return chain_truck

        # Service-anchor short-term commitment for truck support conversion.
        if st_anchor is not None and st_anchor.kind == AgentKind.TRUCK:
            hold_until = int(self._support_anchor_until_step.get(str(aid), -1))
            anchor_tid = str(self._support_anchor_task_id.get(str(aid), ""))
            anchor_gain = float(self._support_anchor_gain.get(str(aid), 0.0))
            max_drift = float(max(getattr(env.cfg, "hrl_support_anchor_max_drift_m", 250.0), 0.0))
            drift_ok = True
            if cur_task_anchor is not None and cur_task_anchor.kind == TaskKind.EMERGENCY:
                d_anchor = float(self._truck_task_distance(env, str(aid), cur_task_anchor))
                drift_ok = bool((not np.isfinite(d_anchor)) or d_anchor <= max(3.0 * max_drift, 1.0e-6) or max_drift <= 1e-9)
            if (
                now_step <= hold_until
                and anchor_gain > 1e-9
                and str(keep_current) == anchor_tid
                and cur_task_anchor is not None
                and cur_task_anchor.kind == TaskKind.EMERGENCY
                and drift_ok
            ):
                # During hold window keep service-anchor emergency unless hard safety already overrode.
                if proposed_goal is None or str(proposed_goal) != str(keep_current):
                    return str(keep_current)

        # Relaxed-chain commitment: once a relaxed-feasible chain is established,
        # avoid immediate churn back to non-emergency targets.
        if st_anchor is not None and st_anchor.kind == AgentKind.UAV:
            chain_until = int(self._relaxed_chain_until_step.get(str(aid), -1))
            if now_step <= chain_until:
                if cur_task_anchor is not None and cur_task_anchor.kind == TaskKind.EMERGENCY and cur_task_anchor.status == TaskStatus.PENDING:
                    if proposed_goal is None:
                        return str(keep_current)
                    proposed_task = env.state.tasks.get(str(proposed_goal), None)
                    proposed_agent = env.state.agents.get(str(proposed_goal), None)
                    if (proposed_task is None or proposed_task.kind != TaskKind.EMERGENCY) and (
                        proposed_agent is None or proposed_agent.kind != AgentKind.TRUCK
                    ):
                        return str(keep_current)
        # Terminal-delivery lock: avoid switching away from almost-completed
        # service trajectories unless hard safety override already fired.
        st_keep = env.state.agents.get(str(aid), None)
        if self.use_event_trigger and st_keep is not None and st_keep.kind == AgentKind.UAV and st_keep.follow_target is None:
            cur_task_keep = env.state.tasks.get(str(keep_current), None)
            if cur_task_keep is not None and cur_task_keep.kind == TaskKind.EMERGENCY and cur_task_keep.status == TaskStatus.PENDING:
                d_cur = float(env._agent_distance_to_task(str(aid), cur_task_keep))
                terminal_lock_dist = float(
                    max(
                        float(getattr(env.cfg, "uav_goal_terminal_lock_distance_m", 180.0)),
                        4.0 * float(getattr(env.cfg, "uav_delivery_radius_m", 30.0)),
                    )
                )
                if np.isfinite(d_cur) and d_cur <= terminal_lock_dist and self._goal_valid_and_safe(env, str(aid), keep_current):
                    return keep_current

        if not self._goal_hold_elapsed(env, str(aid)):
            map_update_bypass = False
            if self.use_event_trigger and bool(self._last_refresh_flags.get("map_update", False)):
                impacted, _critical = self._goal_map_update_impact(env, str(aid), keep_current)
                map_update_bypass = bool(impacted)
            can_bypass_hold = bool(
                self.use_event_trigger
                and (
                    bool(map_update_bypass)
                    or bool(self._last_refresh_flags.get("uav_emergency", False))
                    or bool(self._last_refresh_flags.get("truck_dead_end", False))
                )
            )
            if not can_bypass_hold:
                # Truck routine stuck-escape can break hold locally without global refresh.
                if (
                    st_anchor is not None
                    and st_anchor.kind == AgentKind.TRUCK
                    and cur_task_anchor is not None
                    and cur_task_anchor.kind == TaskKind.NORMAL
                    and cur_task_anchor.status == TaskStatus.PENDING
                ):
                    esc_tid = self._truck_routine_stuck_escape_goal(env, str(aid), str(keep_current), float(cur_score_anchor))
                    if esc_tid is not None and str(esc_tid) != str(keep_current):
                        self._record_switch_decision(str(aid), "accepted", "score")
                        return str(esc_tid)
                return keep_current
        # Hysteresis: only switch if new score is materially better.
        if proposed_goal is None or str(proposed_goal) == str(keep_current):
            return keep_current if proposed_goal is None else proposed_goal
        cur_score = float(self._score_goal_for_agent(env, str(aid), keep_current))
        new_score = float(self._score_goal_for_agent(env, str(aid), proposed_goal))
        st = env.state.agents.get(str(aid), None)
        cur_task = env.state.tasks.get(str(keep_current), None)
        new_task = env.state.tasks.get(str(proposed_goal), None)
        cur_agent = env.state.agents.get(str(keep_current), None)

        # Truck NORMAL<->NORMAL de-oscillation guard:
        # when both current and proposed NORMAL goals are still feasible, require
        # a clear score/distance improvement before switching.
        if (
            st is not None
            and st.kind == AgentKind.TRUCK
            and cur_task is not None
            and new_task is not None
            and cur_task.kind == TaskKind.NORMAL
            and new_task.kind == TaskKind.NORMAL
            and str(cur_task.task_id) != str(new_task.task_id)
        ):
            aba_block_steps = int(max(getattr(env.cfg, "hrl_truck_normal_aba_block_steps", 12), 0))
            prev_normal_goal = str(self._truck_recent_normal_prev_goal.get(str(aid), ""))
            last_switch_step = int(self._truck_recent_normal_switch_step.get(str(aid), -10**9))
            if (
                aba_block_steps > 0
                and prev_normal_goal
                and str(new_task.task_id) == prev_normal_goal
                and (int(now_step) - int(last_switch_step)) <= int(aba_block_steps)
            ):
                return keep_current

            cur_ok = bool(self._truck_task_reachable(env, str(aid), cur_task))
            new_ok = bool(self._truck_task_reachable(env, str(aid), new_task))
            if cur_ok and new_ok:
                map_update_active = bool(self._last_refresh_flags.get("map_update", False))
                cur_impacted = bool(float(self._task_shared_map_block_pressure(env, cur_task)) >= 0.50)
                new_impacted = bool(float(self._task_shared_map_block_pressure(env, new_task)) >= 0.50)
                # Only tighten switching when map-update does not clearly force reroute.
                if (not map_update_active) or ((not cur_impacted) and (not new_impacted)):
                    min_ratio = float(np.clip(getattr(env.cfg, "hrl_truck_normal_to_normal_switch_min_improve_ratio", 0.15), 0.0, 1.0))
                    min_gain = float(max(getattr(env.cfg, "hrl_truck_normal_to_normal_switch_min_score_gain", 0.10), 0.0))
                    d_cur = float(self._truck_task_distance(env, str(aid), cur_task))
                    d_new = float(self._truck_task_distance(env, str(aid), new_task))
                    dist_improved = bool((not np.isfinite(d_cur)) or (not np.isfinite(d_new)) or (d_new <= d_cur * max(1.0 - min_ratio, 0.0)))
                    score_improved = bool(new_score > float(cur_score + min_gain))
                    if (not dist_improved) and (not score_improved):
                        return keep_current
        if not np.isfinite(new_score):
            return keep_current
        if st is not None and st.kind == AgentKind.UAV:
            margin = float(max(getattr(env.cfg, "hrl_uav_goal_switch_margin", self.switch_margin), 0.0))
        else:
            margin = float(max(getattr(env.cfg, "hrl_truck_goal_switch_margin", self.switch_margin), 0.0))
        if self.use_event_trigger and (not bool(self._last_refresh_flags.get("map_update", False))):
            margin = float(max(margin * 0.85, 0.0))
        if (
            bool(getattr(env.cfg, "hrl_b_route_stability_enabled", False))
            and not bool(getattr(env.cfg, "enable_comm_blackout", False))
            and not bool(self._last_refresh_flags.get("map_update", False))
        ):
            margin = float(
                max(
                    margin,
                    margin
                    * float(
                        max(
                            getattr(
                                env.cfg,
                                "hrl_b_route_stability_margin_scale",
                                1.0,
                            ),
                            1.0,
                        )
                    ),
                )
            )

        cur_task = env.state.tasks.get(str(keep_current), None)
        new_task = env.state.tasks.get(str(proposed_goal), None)
        cur_agent = env.state.agents.get(str(keep_current), None)

        # Truck normal-commit guard (ERC-RHC path): when a truck already holds a
        # feasible NORMAL goal and normal backlog is still non-trivial, avoid
        # opportunistic diversion to EMERGENCY unless there is structural need.
        if (
            self.use_event_trigger
            and st is not None
            and st.kind == AgentKind.TRUCK
            and cur_task is not None
            and new_task is not None
            and cur_task.kind == TaskKind.NORMAL
            and new_task.kind == TaskKind.EMERGENCY
        ):
            pending_norm = int(self._pending_normal_task_count(env))
            base_slots = int(max(getattr(env.cfg, "hrl_truck_min_normal_slots", 1), 0))
            high_slots = int(max(getattr(env.cfg, "hrl_truck_min_normal_slots_high_pressure", base_slots), 0))
            pressure_thr = float(np.clip(getattr(env.cfg, "hrl_truck_min_normal_slots_pressure_threshold", 0.55), 0.0, 1.0))
            n_press, _e_press = self._pending_task_pressure(env)
            min_normal_slots = int(high_slots if float(n_press) >= pressure_thr else base_slots)
            normal_slots_now = int(self._count_trucks_assigned_to_normal(env))
            hard_recovery_active = False
            hard_recovery_fn = getattr(env, "_has_hard_recovery_uav", None)
            if callable(hard_recovery_fn):
                hard_recovery_active = bool(hard_recovery_fn())

            cur_reachable = bool(self._truck_task_reachable(env, str(aid), cur_task))
            cur_block_pressure = float(self._task_shared_map_block_pressure(env, cur_task))
            cur_structurally_bad = bool((not cur_reachable) or cur_block_pressure >= 0.55)

            uav_cover = float(self._uav_emergency_cover_fraction(env, new_task))
            cover_thr = float(np.clip(getattr(env.cfg, "hrl_truck_emergency_relief_uav_cover_threshold", 0.5), 0.0, 1.0))
            emergency_needs_truck = bool(uav_cover < cover_thr)
            emergency_urgency = float(self._norm_deadline_urgency(new_task, int(env.state.step_index)))
            force_cover_thr = float(np.clip(getattr(env.cfg, "hrl_truck_emergency_force_relief_uav_cover_threshold", 0.35), 0.0, 1.0))
            force_urg_thr = float(np.clip(getattr(env.cfg, "hrl_truck_emergency_force_relief_urgency_threshold", 0.72), 0.0, 1.0))
            emergency_relief_override = bool((uav_cover < force_cover_thr) or (emergency_urgency >= force_urg_thr))
            emergency_value = bool(
                emergency_urgency >= 0.72
                or self._is_island_task(env, new_task)
                or float(self._truck_forward_support_score(env, str(aid), new_task)) > 0.10
            )

            allow_diversion = bool(
                cur_structurally_bad
                or hard_recovery_active
                or emergency_relief_override
                or (
                    emergency_needs_truck
                    and emergency_value
                    and pending_norm <= 1
                )
            )
            if self._truck_normal_commit_guard2_enabled(env):
                n_press, _e_press = self._pending_task_pressure(env)
                pressure_thr = float(np.clip(getattr(env.cfg, "hrl_truck_normal_commit_pressure_threshold", 0.55), 0.0, 1.0))
                min_commit = int(max(getattr(env.cfg, "hrl_truck_normal_commit_min_steps", 8), 0))
                assigned = self.state.goal_assigned_step.get(str(aid), int(env.state.step_index))
                held_steps = int(env.state.step_index) - int(assigned)
                if (
                    pending_norm > 0
                    and n_press >= pressure_thr
                    and held_steps < min_commit
                    and (not hard_recovery_active)
                    and (not cur_structurally_bad)
                    and (not emergency_relief_override)
                ):
                    return keep_current
            if pending_norm > 0 and (not allow_diversion):
                return keep_current
            # Keep at least a minimum NORMAL truck slot when backlog exists.
            if (
                pending_norm > 0
                and min_normal_slots > 0
                and normal_slots_now <= min_normal_slots
                and (not emergency_relief_override)
                and (not cur_structurally_bad)
            ):
                return keep_current

        # Guard NORMAL -> None clear: keep committing unless truly unreachable / blocked.
        if (
            st is not None
            and st.kind == AgentKind.TRUCK
            and cur_task is not None
            and cur_task.kind == TaskKind.NORMAL
            and proposed_goal is None
            and bool(getattr(env.cfg, "hrl_truck_normal_no_none_clear_enabled", True))
        ):
            cur_reachable = bool(self._truck_task_reachable(env, str(aid), cur_task))
            cur_block_pressure = float(self._task_shared_map_block_pressure(env, cur_task))
            if cur_reachable and cur_block_pressure < 0.70:
                return keep_current

        # Deadline-protect NORMAL tasks already being executed by trucks.
        if (
            st is not None
            and st.kind == AgentKind.TRUCK
            and cur_task is not None
            and cur_task.kind == TaskKind.NORMAL
            and new_task is not None
            and str(new_task.task_id) != str(cur_task.task_id)
        ):
            protect_steps = int(max(getattr(env.cfg, "hrl_truck_normal_deadline_protect_steps", 40), 0))
            if protect_steps > 0:
                cur_reachable = bool(self._truck_task_reachable(env, str(aid), cur_task))
                cur_block_pressure = float(self._task_shared_map_block_pressure(env, cur_task))
                cur_hard_bad = bool((not cur_reachable) or cur_block_pressure >= 0.80)
                remaining = int(getattr(cur_task, "deadline_step", int(env.state.step_index)) - int(env.state.step_index))
                in_protect_window = bool(remaining <= protect_steps)
                if in_protect_window and (not cur_hard_bad):
                    # Keep NORMAL commitment unless emergency is strict force-relief.
                    if new_task.kind == TaskKind.EMERGENCY:
                        uav_cover = float(self._uav_emergency_cover_fraction(env, new_task))
                        emergency_urgency = float(self._norm_deadline_urgency(new_task, int(env.state.step_index)))
                        force_cover_thr = float(np.clip(getattr(env.cfg, "hrl_truck_emergency_force_relief_uav_cover_threshold", 0.35), 0.0, 1.0))
                        force_urg_thr = float(np.clip(getattr(env.cfg, "hrl_truck_emergency_force_relief_urgency_threshold", 0.72), 0.0, 1.0))
                        strict_force_relief = bool((uav_cover < force_cover_thr) or (emergency_urgency >= force_urg_thr))
                        if (not strict_force_relief):
                            return keep_current
                    else:
                        # NORMAL->NORMAL churn inside deadline window is not allowed.
                        return keep_current

        # Truck NORMAL cooldown: reduce "selected -> cancel -> reselection" churn.
        if (
            st is not None
            and st.kind == AgentKind.TRUCK
            and cur_task is not None
            and new_task is not None
            and cur_task.kind == TaskKind.NORMAL
            and str(cur_task.task_id) != str(new_task.task_id)
        ):
            cooldown_steps = int(max(getattr(env.cfg, "hrl_truck_normal_switch_cooldown_steps", 18), 0))
            if cooldown_steps > 0:
                assigned = int(self.state.goal_assigned_step.get(str(aid), int(env.state.step_index)))
                held_steps = int(env.state.step_index) - int(assigned)
                cur_reachable = bool(self._truck_task_reachable(env, str(aid), cur_task))
                map_update_active = bool(self._last_refresh_flags.get("map_update", False))
                if held_steps < cooldown_steps and cur_reachable and (not map_update_active):
                    return keep_current

        # Persistently-unreachable NORMAL tracking (diagnostic + anti-thrash hint).
        if st is not None and st.kind == AgentKind.TRUCK and cur_task is not None and cur_task.kind == TaskKind.NORMAL:
            tid = str(cur_task.task_id)
            reachable = bool(self._truck_task_reachable(env, str(aid), cur_task))
            if reachable:
                self._normal_task_unreachable_streak[tid] = 0
            else:
                prev = int(self._normal_task_unreachable_streak.get(tid, 0))
                now = prev + 1
                self._normal_task_unreachable_streak[tid] = int(now)
                patience = int(max(getattr(env.cfg, "hrl_normal_unreachable_patience_steps", 10), 1))
                if now == patience:
                    self.normal_unreachable_task_count_total += 1

        # Truck de-diversion guard (ERC-RHC path): if truck currently holds an
        # EMERGENCY goal that is sufficiently UAV-coverable while NORMAL backlog
        # is pending, allow immediate switch back to NORMAL proposal.
        if (
            self.use_event_trigger
            and st is not None
            and st.kind == AgentKind.TRUCK
            and cur_task is not None
            and new_task is not None
            and cur_task.kind == TaskKind.EMERGENCY
            and new_task.kind == TaskKind.NORMAL
        ):
            pending_norm = int(self._pending_normal_task_count(env))
            hard_recovery_active = False
            hard_recovery_fn = getattr(env, "_has_hard_recovery_uav", None)
            if callable(hard_recovery_fn):
                hard_recovery_active = bool(hard_recovery_fn())
            if pending_norm > 0 and (not hard_recovery_active) and (not self._is_island_task(env, cur_task)):
                if not bool(self._truck_emergency_relief_allowed(env, str(aid), cur_task)):
                    return proposed_goal

        # Airborne island-commit guard:
        # once UAV commits to an island emergency sortie, avoid mid-air retarget to
        # non-island goals unless recovery safety requires a truck rendezvous.
        island_commit_lock = False
        if st is not None and st.kind == AgentKind.UAV and st.follow_target is None:
            cur_is_island = bool(cur_task is not None and self._is_island_task(env, cur_task))
            new_is_island = bool(new_task is not None and self._is_island_task(env, new_task))
            if cur_is_island and (not new_is_island):
                new_agent = env.state.agents.get(str(proposed_goal), None)
                switching_to_recovery_truck = bool(
                    new_agent is not None
                    and new_agent.kind == AgentKind.TRUCK
                    and self._uav_needs_recovery(env, str(aid))
                )
                if not switching_to_recovery_truck:
                    island_commit_lock = True
                    margin = float(max(margin, 0.35))

        # Safety- and event-aware relief: allow quicker adaptation on map-update and island emergencies.
        if st is not None and st.kind == AgentKind.UAV:
            if cur_agent is not None and cur_agent.kind == AgentKind.TRUCK and new_task is not None:
                # Docked pre-launch stage is sensitive to truck<->task ping-pong.
                # Require stable hold and stronger gain before switching away from truck.
                if st.follow_target is not None:
                    if not self._goal_stable_for_takeoff(env, str(aid)):
                        return keep_current
                    margin = float(max(margin, 0.18))
                elif self._goal_stable_for_takeoff(env, str(aid)):
                    margin = float(min(margin, 0.10))
            if bool(self._last_refresh_flags.get("map_update", False)):
                if st.follow_target is not None:
                    margin = float(max(margin, 0.14))
                else:
                    margin = float(min(margin, 0.08))
            if new_task is not None and str(new_task.task_id) in set(getattr(env, "_current_island_emergency_task_ids", lambda: set())()):
                margin = float(min(margin, 0.06))
        elif st is not None and st.kind == AgentKind.TRUCK and bool(self._last_refresh_flags.get("map_update", False)):
            cur_impacted = False
            new_impacted = False
            if cur_task is not None:
                cur_impacted = bool(
                    (not self._truck_task_reachable(env, str(aid), cur_task))
                    or float(self._task_shared_map_block_pressure(env, cur_task)) >= 0.50
                )
            if new_task is not None:
                new_impacted = bool(
                    (not self._truck_task_reachable(env, str(aid), new_task))
                    or float(self._task_shared_map_block_pressure(env, new_task)) >= 0.50
                )
            if cur_impacted and (not new_impacted):
                margin = float(min(margin, 0.04))
            if new_task is not None and self._is_island_task(env, new_task):
                margin = float(min(margin, 0.05))

        if island_commit_lock:
            margin = float(max(margin, 0.35))

        if new_score <= float(cur_score + margin):
            if (
                st_anchor is not None
                and st_anchor.kind == AgentKind.TRUCK
                and cur_task_anchor is not None
                and cur_task_anchor.kind == TaskKind.NORMAL
                and cur_task_anchor.status == TaskStatus.PENDING
            ):
                esc_tid = self._truck_routine_stuck_escape_goal(env, str(aid), str(keep_current), float(cur_score_anchor))
                if esc_tid is not None and str(esc_tid) != str(keep_current):
                    self._record_switch_decision(str(aid), "accepted", "score")
                    return str(esc_tid)
            self._record_switch_decision(str(aid), "rejected", "threshold")
            return keep_current
        self._record_switch_decision(str(aid), "accepted", "score")
        return proposed_goal

    def _can_keep_prev_goal(self, env, aid: str, used_tasks: set) -> Optional[str]:
        prev = self.state.goals.get(str(aid), None)
        if prev is None:
            return None
        st = env.state.agents.get(str(aid), None)
        if st is None:
            return None

        # Keep task goal only if still pending, unique and feasible.
        t = env.state.tasks.get(str(prev), None)
        if t is not None:
            if not self._task_planner_active(t):
                return None
            if str(t.task_id) in used_tasks:
                return None
            if st.kind == AgentKind.UAV:
                if t.kind == TaskKind.EMERGENCY and getattr(st, "follow_target", None) is None:
                    if self._uav_needs_recovery(env, str(aid)):
                        return None
                    if self._comm_degraded(env, str(aid)):
                        return None
                    if self._uav_task_hard_risk_blocked(env, t):
                        return None
                    return str(prev)
                if self._uav_task_feasible(env, str(aid), t):
                    return str(prev)
                if self._docked_uav_soft_invalid_hold(env, str(aid), str(prev)):
                    return str(prev)
                return None
            if st.kind == AgentKind.TRUCK:
                return str(prev) if self._truck_task_valid(env, str(aid), str(prev)) else None
            return None

        # Keep truck rendezvous goal if that truck still exists.
        ag = env.state.agents.get(str(prev), None)
        if st.kind == AgentKind.UAV and ag is not None and ag.kind == AgentKind.TRUCK:
            if self._uav_needs_recovery(env, str(aid)) or self._comm_degraded(env, str(aid)):
                return str(prev)
        return None

    # --------------------------
    # Deterministic greedy planning + repair
    # --------------------------
    def _select_goal_for_agent(self, env, aid: str, used_tasks: set) -> Optional[str]:
        st = env.state.agents[aid]
        if st.kind == AgentKind.UAV and bool(st.crashed):
            return None

        tids, _, _ = self.build_candidates_for_agent(
            env,
            aid,
            used_tasks=used_tasks,
            enable_rth_mask=bool(getattr(env.cfg, "enable_rth_mask", True)),
        )

        proposed_goal: Optional[str] = None
        if st.kind == AgentKind.TRUCK:
            if self._truck_stage_blocks_task_goal(env, aid):
                return self._apply_switch_hysteresis(env, aid, proposed_goal=None, used_tasks=used_tasks)
            scored: List[Tuple[int, float, str]] = []
            for tid in tids:
                task = env.state.tasks.get(str(tid), None)
                if not self._task_planner_active(task):
                    continue
                if str(task.task_id) in used_tasks:
                    continue
                tier = int(self._task_priority_tier(env, str(aid), task))
                s = float(self._score_truck_task(env, aid, task))
                scored.append((int(tier), float(s), str(task.task_id)))

            if not scored:
                proposed_goal = None
            else:
                scored.sort(key=lambda x: (-int(x[0]), -float(x[1]), str(x[2])))
                proposed_goal = str(scored[0][2])
            return self._apply_switch_hysteresis(env, aid, proposed_goal=proposed_goal, used_tasks=used_tasks)

        # UAV path: choose hard-feasible emergency task greedily; if none feasible,
        # fallback to truck rendezvous only when recovery is needed.
        if st.follow_target is not None:
            cur_goal = self.state.goals.get(str(aid), None)
            cur_task = env.state.tasks.get(str(cur_goal), None) if cur_goal is not None else None
            docked_actionable_fn = getattr(env, "_uav_docked_task_actionable_now", None)
            if (
                cur_task is not None
                and cur_task.kind == TaskKind.EMERGENCY
                and cur_task.status == TaskStatus.PENDING
                and self._goal_stable_for_takeoff(env, aid)
                and callable(docked_actionable_fn)
            ):
                try:
                    if bool(docked_actionable_fn(str(aid), cur_task)):
                        return str(cur_task.task_id)
                except Exception:
                    pass
        scored_task: List[Tuple[int, float, str]] = []
        scored_truck: List[Tuple[float, str]] = []
        for tid in tids:
            goal_agent = env.state.agents.get(str(tid), None)
            if goal_agent is not None and goal_agent.kind == AgentKind.TRUCK:
                ax, ay = self._agent_xy(env, aid)
                tx, ty = self._agent_xy(env, str(tid))
                d = float(((ax - tx) ** 2 + (ay - ty) ** 2) ** 0.5)
                scored_truck.append((self._score_uav_truck(env, aid, str(tid), d), str(tid)))
                continue

            task = env.state.tasks.get(str(tid), None)
            if task is None or task.status != TaskStatus.PENDING:
                continue
            if str(task.task_id) in used_tasks:
                continue
            if not self._uav_task_feasible(env, aid, task):
                continue
            # Conservative short-sortie mode: when UAV is currently docked/following,
            # only allow leaving recovery for stable-goal, short safe sorties.
            if st.follow_target is not None:
                docked_actionable = False
                docked_actionable_fn = getattr(env, "_uav_docked_task_actionable_now", None)
                if callable(docked_actionable_fn):
                    try:
                        docked_actionable = bool(docked_actionable_fn(str(aid), task))
                    except Exception:
                        docked_actionable = False
                if not self._goal_stable_for_takeoff(env, aid):
                    continue
                if not bool(self._docked_uav_sortie_chain_ready(env, aid, task)):
                    continue
                if (not docked_actionable) and not (
                    self._uav_task_short_sortie_safe(env, aid, task)
                    or self._uav_task_clearly_safe_long_range(env, aid, task)
                ):
                    continue
            tier = int(self._task_priority_tier(env, str(aid), task))
            s = float(self._score_uav_task(env, aid, task))
            scored_task.append((int(tier), float(s), str(task.task_id)))

        if scored_task:
            scored_task.sort(key=lambda x: (-int(x[0]), -float(x[1]), str(x[2])))
            proposed_goal = str(scored_task[0][2])
            # Docked UAV sortie launch path: when currently attached to truck,
            # prioritize a stable, hard-safe dispatch over truck-goal inertia.
            if st.follow_target is not None and self._goal_stable_for_takeoff(env, aid):
                cur_goal = self.state.goals.get(str(aid), None)
                cur_agent = env.state.agents.get(str(cur_goal), None)
                if cur_agent is not None and cur_agent.kind == AgentKind.TRUCK:
                    return proposed_goal
            return self._apply_switch_hysteresis(env, aid, proposed_goal=proposed_goal, used_tasks=used_tasks)

        if self._uav_needs_recovery(env, aid):
            if scored_truck:
                scored_truck.sort(key=lambda x: (-x[0], x[1]))
                proposed_goal = str(scored_truck[0][1])
            else:
                near_tid, _ = self._nearest_truck(env, aid)
                proposed_goal = near_tid
            return self._apply_switch_hysteresis(env, aid, proposed_goal=proposed_goal, used_tasks=used_tasks)

        return self._apply_switch_hysteresis(env, aid, proposed_goal=None, used_tasks=used_tasks)

    def _repair_goals(self, env, goals: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
        repaired = dict(goals)

        # Repair 1 (strict): unsafe UAV task target -> nearest truck recovery or None.
        for aid in self._ordered_agents(env):
            st = env.state.agents[aid]
            if st.kind != AgentKind.UAV or bool(st.crashed):
                continue
            gid = repaired.get(aid, None)
            if gid is None:
                continue
            task = env.state.tasks.get(str(gid), None)
            if task is not None and task.kind == TaskKind.EMERGENCY and task.status == TaskStatus.PENDING:
                docked_actionable = False
                docked_actionable_fn = getattr(env, "_uav_docked_task_actionable_now", None)
                if st.follow_target is not None and callable(docked_actionable_fn):
                    try:
                        docked_actionable = bool(docked_actionable_fn(str(aid), task))
                    except Exception:
                        docked_actionable = False
                support_chain_ready = False
                chain = self._support_bound_chain_info_for_uav(env, str(aid))
                if chain is not None and str(chain.get("task_id", "")) == str(task.task_id):
                    support_chain_ready = bool(self._docked_uav_sortie_chain_ready(env, str(aid), task))
                if (not docked_actionable) and (not support_chain_ready) and (not self._uav_task_feasible(env, aid, task)):
                    near_tid, _ = self._nearest_truck(env, aid)
                    repaired[aid] = near_tid if (self.use_rth_repair and near_tid is not None) else None

        # Repair 2 (strict): invalid/unreachable truck target -> clear to None.
        for aid in self._ordered_agents(env):
            st = env.state.agents[aid]
            if st.kind != AgentKind.TRUCK:
                continue
            gid = repaired.get(aid, None)
            if gid is None:
                continue
            if not self._truck_task_valid(env, aid, str(gid)):
                fallback_tid = self._rc_best_truck_fallback_goal(env, str(aid), excluded_task_id=str(gid))
                repaired[aid] = fallback_tid

        if bool(getattr(env.cfg, "truck_force_nonnull_goal_enabled", False)):
            for aid in self._ordered_agents(env):
                st = env.state.agents[aid]
                if st.kind != AgentKind.TRUCK:
                    continue
                gid = repaired.get(aid, None)
                if gid is None:
                    repaired[aid] = self._rc_best_truck_fallback_goal(env, str(aid), excluded_task_id=None)

        # Final duplicate guard for pending tasks (first-come by agent order).
        claimed: set = set()
        for aid in self._ordered_agents(env):
            gid = repaired.get(aid, None)
            if gid is None:
                continue
            task = env.state.tasks.get(str(gid), None)
            if task is None or task.status != TaskStatus.PENDING:
                continue
            tid = str(task.task_id)
            if tid in claimed:
                st = env.state.agents.get(str(aid), None)
                chain = None
                if st is not None:
                    if st.kind == AgentKind.TRUCK:
                        chain = self._support_bound_chain_info_for_truck(env, str(aid))
                    elif st.kind == AgentKind.UAV:
                        chain = self._support_bound_chain_info_for_uav(env, str(aid))
                if (
                    chain is not None
                    and str(chain.get("task_id", "")) == tid
                    and self._tc_support_chain_class.get(str(chain.get("truck_id", "")), "") == "support_required"
                ):
                    continue
                repaired[aid] = None
            else:
                claimed.add(tid)

        return repaired

    def _normalize_assignment_scores(self, score_map: Dict[Tuple[str, str], float]) -> Dict[Tuple[str, str], float]:
        vals = [float(v) for v in score_map.values() if np.isfinite(float(v))]
        if not vals:
            return {k: -1e9 for k in score_map}
        lo = float(min(vals))
        hi = float(max(vals))
        den = float(max(hi - lo, 1e-9))
        # Unified normalization across truck/UAV scoring spaces.
        return {
            k: (float(v) - lo) / den if np.isfinite(float(v)) else -1e9
            for k, v in score_map.items()
        }

    def _solve_assignment(self, costs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Prefer scipy linear_sum_assignment for paper reproducibility.
        if linear_sum_assignment is not None:
            r, c = linear_sum_assignment(costs)
            return np.asarray(r, dtype=np.int64), np.asarray(c, dtype=np.int64)

        # Dependency-free fallback: greedy on sorted costs.
        r_sel: List[int] = []
        c_sel: List[int] = []
        used_r = set()
        used_c = set()
        flat = []
        for i in range(costs.shape[0]):
            for j in range(costs.shape[1]):
                flat.append((float(costs[i, j]), i, j))
        flat.sort(key=lambda x: (x[0], x[1], x[2]))
        for _, i, j in flat:
            if i in used_r or j in used_c:
                continue
            used_r.add(i)
            used_c.add(j)
            r_sel.append(int(i))
            c_sel.append(int(j))
        return np.asarray(r_sel, dtype=np.int64), np.asarray(c_sel, dtype=np.int64)

    def _assign_by_hard_tier(
        self,
        env,
        candidate_pairs: List[Tuple[str, str]],
        norm_scores: Dict[Tuple[str, str], float],
        raw_tiers: Dict[Tuple[str, str], int],
        goals: Dict[str, Optional[str]],
        used_tasks: set,
        tc_sel_stats: Dict[str, int],
    ) -> None:
        if not candidate_pairs:
            return
        remaining_agents = set(sorted({a for a, _ in candidate_pairs if goals.get(str(a), None) is None}))
        remaining_tasks = set(sorted({t for _, t in candidate_pairs if str(t) not in used_tasks}))
        if not remaining_agents or not remaining_tasks:
            return
        big = 1e6

        for min_tier in (3, 2, 1, 0):
            if not remaining_agents or not remaining_tasks:
                break
            stage_pairs = [
                (str(a), str(t))
                for (a, t) in candidate_pairs
                if str(a) in remaining_agents
                and str(t) in remaining_tasks
                and int(raw_tiers.get((str(a), str(t)), -1)) >= int(min_tier)
            ]
            if not stage_pairs:
                continue

            agents = sorted({a for a, _ in stage_pairs})
            tasks = sorted({t for _, t in stage_pairs})
            a_idx = {a: i for i, a in enumerate(agents)}
            t_idx = {t: j for j, t in enumerate(tasks)}
            cost = np.full((len(agents), len(tasks)), fill_value=big, dtype=np.float64)

            for (aid, tid) in stage_pairs:
                sc = float(norm_scores.get((str(aid), str(tid)), -1e9))
                if not np.isfinite(sc):
                    continue
                # Within each tier stage we only rank by tie-break score.
                cost[a_idx[str(aid)], t_idx[str(tid)]] = float(-sc)

            r_sel, c_sel = self._solve_assignment(cost)
            for r, c in zip(r_sel.tolist(), c_sel.tolist()):
                if r < 0 or c < 0 or r >= len(agents) or c >= len(tasks):
                    continue
                aid = agents[int(r)]
                tid = tasks[int(c)]
                if float(cost[int(r), int(c)]) >= big:
                    continue
                goals[str(aid)] = str(tid)
                used_tasks.add(str(tid))
                remaining_agents.discard(str(aid))
                remaining_tasks.discard(str(tid))

                task_sel = env.state.tasks.get(str(tid), None)
                if task_sel is not None and self._is_timecritical_lightweight_task(task_sel):
                    tc_sel_stats["total"] = int(tc_sel_stats.get("total", 0)) + 1
                    tier_sel = int(raw_tiers.get((str(aid), str(tid)), self._task_priority_tier(env, str(aid), task_sel)))
                    if tier_sel >= 3:
                        tc_sel_stats["tier3"] = int(tc_sel_stats.get("tier3", 0)) + 1
                    elif tier_sel >= 2:
                        tc_sel_stats["tier2"] = int(tc_sel_stats.get("tier2", 0)) + 1
    def _plan_attraction_dispatch(self, env) -> Dict[str, Optional[str]]:
        """Greedy exclusive attraction dispatch for truck--UAV clusters.

        Trucks claim routine tasks by ``w_normal / road_distance``. Docked,
        loaded UAVs then claim distinct emergency tasks by
        ``w_emergency / euclidean_distance``.  A claim is inserted into
        ``used`` immediately, so no two agents can receive the same task.
        Truck emergency delivery remains disabled by vehicle compatibility.
        """
        goals: Dict[str, Optional[str]] = {
            str(aid): None
            for aid in env.state.agents
        }
        self._prune_task_contracts(env)
        used: set = set()
        self._truck_assist_waypoint_by_truck.clear()
        normal_w = float(max(getattr(env.cfg, "hrl_attraction_normal_weight", 4.0), 0.0))
        emerg_w = float(max(getattr(env.cfg, "hrl_attraction_emergency_weight", 1.0), 0.0))
        truck_normal_score: Dict[str, float] = {}
        trucks = sorted(str(aid) for aid, st in env.state.agents.items() if st.kind == AgentKind.TRUCK and not bool(st.crashed))
        normals = [t for t in env.state.tasks.values() if t.status == TaskStatus.PENDING and t.kind == TaskKind.NORMAL]
        for aid in trucks:
            st = env.state.agents[aid]
            if st.node is None:
                continue
            best = None
            best_score = -float("inf")
            for task in normals:
                owner = self._task_contract_owner(env, str(task.task_id))
                if owner is not None and str(owner) != str(aid):
                    continue
                if str(task.task_id) in used:
                    continue
                if hasattr(env, "is_task_serviceable_by_agent") and not bool(env.is_task_serviceable_by_agent(aid, task)):
                    continue
                d = float(env._decision_shortest_path_distance(int(st.node), int(task.demand_node)))
                if not np.isfinite(d):
                    continue
                # Compare task value per expected truck execution time, not
                # inverse distance alone.
                truck_v = float(max(getattr(env.cfg, "truck_speed_mps", 1.0), 1e-6))
                service_t = float(max(getattr(env.cfg, "truck_replenish_service_steps", 0), 0) * getattr(env.cfg, "dt_seconds", 1.0))
                score = float(normal_w / max(d / truck_v + service_t, 1.0))
                if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and best is not None and str(task.task_id) < str(best.task_id)):
                    best, best_score = task, score
            if best is not None:
                goals[aid] = str(best.task_id)
                used.add(str(best.task_id))
                truck_normal_score[str(aid)] = float(best_score)

        emergencies = [t for t in env.state.tasks.values() if t.status == TaskStatus.PENDING and t.kind == TaskKind.EMERGENCY]
        # Cluster attraction stage.  A truck retains its normal-task goal (so
        # it cannot unload an emergency task), but may first move toward a
        # distinct emergency task for one of its docked UAVs.  The waypoint is
        # cleared automatically as soon as that UAV takes off, after which the
        # truck resumes the same normal-task route.
        staged_uavs: set = set()
        for tid in trucks:
            truck = env.state.agents.get(str(tid), None)
            if truck is None or truck.node is None:
                continue
            attached = [
                str(uid) for uid, ust in env.state.agents.items()
                if ust.kind == AgentKind.UAV
                and not bool(ust.crashed)
                and str(getattr(ust, "follow_target", "") or "") == str(tid)
                and bool(getattr(env, "_uav_loaded", lambda _aid: False)(str(uid)))
            ]
            if not attached:
                continue
            best_pair = None
            best_score = -float("inf")
            for uid in sorted(attached):
                for task in emergencies:
                    owner = self._task_contract_owner(env, str(task.task_id))
                    if owner is not None and str(owner) != str(uid):
                        continue
                    if str(task.task_id) in used:
                        continue
                    # Find the earliest road-reachable node from which the
                    # UAV can safely launch.  Select the *closest-to-task*
                    # feasible road node, not merely the first node the truck
                    # can reach: the whole point of staging is to move the
                    # truck to the nearest safe launch boundary.
                    max_sortie = float(max(getattr(env.cfg, "uav_max_sortie_m", 6000.0), 1.0))
                    recovery_buf = float(max(getattr(env.cfg, "uav_recovery_distance_buffer_m", 600.0), 0.0))
                    launch_radius = float(max((0.92 * max_sortie - recovery_buf) / 2.0, 1.0))
                    task_xy = env._node_xy(int(task.demand_node))
                    stage_node, stage_d, stage_key = None, float("inf"), (float("inf"), float("inf"))
                    for nid in env.topology.nodes:
                        nxy = env._node_xy(int(nid))
                        air_d = float(np.hypot(float(nxy[0]) - float(task_xy[0]), float(nxy[1]) - float(task_xy[1])))
                        if air_d > launch_radius:
                            continue
                        cand_d = float(env._decision_shortest_path_distance(int(truck.node), int(nid)))
                        cand_key = (air_d, cand_d)
                        if np.isfinite(cand_d) and cand_key < stage_key:
                            stage_node, stage_d, stage_key = int(nid), cand_d, cand_key
                    if stage_node is None:
                        continue
                    stage_xy = env._node_xy(int(stage_node))
                    uav_d = float(np.hypot(float(stage_xy[0]) - float(task_xy[0]), float(stage_xy[1]) - float(task_xy[1])))
                    truck_v = float(max(getattr(env.cfg, "truck_speed_mps", 1.0), 1e-6))
                    uav_v = float(max(getattr(env.cfg, "uav_max_speed_mps", 1.0), 1e-6))
                    # Explicit completion probability.  A sortie that can
                    # return to the staging truck is certain to complete;
                    # otherwise use the nearest non-crashed truck as a
                    # rendezvous fallback and discount by its time slack.
                    truck_xy = env._node_xy(int(stage_node))
                    return_d = float(np.hypot(float(task_xy[0]) - float(truck_xy[0]), float(task_xy[1]) - float(truck_xy[1])))
                    sortie_limit = float(max_sortie * 0.92)
                    p_complete = 0.0
                    recovery_mode = "none"
                    if (uav_d + return_d + recovery_buf) <= sortie_limit:
                        p_complete = 1.0
                        recover_d = return_d
                        recovery_mode = "home_truck"
                    else:
                        # A different truck may collect the UAV after
                        # delivery.  Only trucks with spare follower capacity
                        # are eligible; probability is the remaining time
                        # margin within the emergency lifeline window.
                        nearest_eta = float("inf")
                        for other_tid, other_ts in env.state.agents.items():
                            if other_ts.kind != AgentKind.TRUCK or bool(other_ts.crashed):
                                continue
                            if str(other_tid) == str(tid):
                                continue
                            cap = int(max(getattr(env.cfg, "uav_max_followers_per_truck", 1), 1))
                            followers = sum(1 for uu in env.state.agents.values()
                                            if uu.kind == AgentKind.UAV and str(getattr(uu, "follow_target", "") or "") == str(other_tid))
                            if followers >= cap:
                                continue
                            ox, oy = self._agent_xy(env, str(other_tid))
                            eta = float(np.hypot(float(task_xy[0]) - float(ox), float(task_xy[1]) - float(oy))) / truck_v
                            nearest_eta = min(nearest_eta, eta)
                        recover_d = max_sortie
                        if np.isfinite(nearest_eta):
                            remaining = float(max(getattr(task, "lifeline_current", 0.0), 0.0))
                            dt = float(max(getattr(env.cfg, "dt_seconds", 1.0), 1e-6))
                            window_s = max(remaining * dt, 1.0)
                            slack = (window_s - (uav_d / uav_v + nearest_eta)) / window_s
                            p_complete = float(np.clip(slack, 0.0, 1.0))
                            recovery_mode = "rendezvous" if p_complete > 0.0 else "none"
                    life = float(np.clip(self._task_lifeline_ratio(task), 0.0, 1.0))
                    urgency = float(1.0 + 3.0 * (1.0 - life))
                    mission_t = float(stage_d / truck_v + (uav_d + recover_d) / uav_v)
                    normal_goal = env.state.tasks.get(str(goals.get(str(tid))), None)
                    delay_t = 0.0
                    if normal_goal is not None:
                        direct = float(env._decision_shortest_path_distance(int(truck.node), int(normal_goal.demand_node)))
                        via = float(stage_d + env._decision_shortest_path_distance(int(stage_node), int(normal_goal.demand_node)))
                        if np.isfinite(direct) and np.isfinite(via):
                            delay_t = float(max(via - direct, 0.0) / truck_v)
                    score = float(emerg_w * urgency * p_complete / max(mission_t, 1.0) - 0.35 * delay_t / max(mission_t, 1.0))
                    if score > best_score + 1e-12:
                        best_pair, best_score = (str(uid), task, int(stage_node)), score
            if best_pair is None or best_score <= float(truck_normal_score.get(str(tid), -float("inf"))):
                continue
            uid, task, stage_node = best_pair
            goals[str(uid)] = str(task.task_id)
            used.add(str(task.task_id))
            staged_uavs.add(str(uid))
            normal_goal_id = goals.get(str(tid), None)
            self._truck_assist_waypoint_by_truck[str(tid)] = {
                "assist_waypoint_insert": True,
                "idle_support": bool(normal_goal_id is None),
                "uav_id": str(uid),
                "task_id": str(task.task_id),
                "launch_node": int(stage_node),
                "normal_goal_task_id": str(normal_goal_id) if normal_goal_id is not None else "",
                "step": int(env.state.step_index),
                "extra_distance_m": 0.0,
                "attraction_cluster_stage": True,
            }
        uavs = sorted(str(aid) for aid, st in env.state.agents.items() if st.kind == AgentKind.UAV and not bool(st.crashed))
        for aid in uavs:
            st = env.state.agents[aid]
            if str(aid) in staged_uavs:
                continue
            if st.follow_target is None:
                current = self.state.goals.get(aid, None)
                task = env.state.tasks.get(str(current), None) if current is not None else None
                goals[aid] = str(current) if task is not None and task.status == TaskStatus.PENDING and task.kind == TaskKind.EMERGENCY else None
                if goals[aid] is not None:
                    used.add(str(goals[aid]))
                continue
            if not bool(getattr(env, "_uav_loaded", lambda _aid: False)(aid)):
                goals[aid] = str(st.follow_target)
                continue
            best = None
            best_score = -float("inf")
            for task in emergencies:
                owner = self._task_contract_owner(env, str(task.task_id))
                if owner is not None and str(owner) != str(aid):
                    continue
                if str(task.task_id) in used:
                    continue
                d = float(env._agent_distance_to_task(aid, task))
                if not np.isfinite(d):
                    continue
                score = float(emerg_w / max(d, 1.0))
                if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and best is not None and str(task.task_id) < str(best.task_id)):
                    best, best_score = task, score
            if best is not None:
                goals[aid] = str(best.task_id)
                used.add(str(best.task_id))
            else:
                goals[aid] = str(st.follow_target)
        return self._apply_task_exclusive_contracts(env, goals)

    def _plan_once(self, env) -> Dict[str, Optional[str]]:
        """
        Two-stage assignment:
        1) Global linear assignment over pending tasks for all agents.
        2) UAV recovery fallback (nearest-truck rendezvous) for UAVs left without
           a feasible task target.

        Stage-1 is the paper's global assignment core; stage-2 is a safety repair
        layer for energy/risk recovery and does not replace the global optimization.
        """
        if er_hlns_route_plan_active(env):
            goals_v2 = self._route_plan_v2.plan_or_repair(env)
            # Mirror the persistent v2 ownership into the existing diagnostic
            # fields without invoking the legacy one-task-per-agent contract
            # logic (a full route naturally contains multiple future tasks).
            self._task_contract_by_task = dict(
                self._route_plan_v2.contract_owner_by_task
            )
            self._task_contract_by_agent = {
                str(agent_id): str(task_id)
                for task_id, agent_id in self._task_contract_by_task.items()
                if goals_v2.get(str(agent_id), None) == str(task_id)
            }
            self._truck_assist_waypoint_by_truck = dict(
                self._route_plan_v2.assist_by_truck
            )
            # V2 publishes a persistent cross-truck recovery contract.  Do
            # not erase it with the legacy planner's per-step cleanup.
            self._uav_transfer_target_truck = dict(
                getattr(env, "_uav_transfer_target_truck", {})
            )
            self._uav_transfer_target_task = dict(
                getattr(env, "_uav_transfer_target_task", {})
            )
            return goals_v2

        self._prune_task_contracts(env)
        coordination_active = self._er_hlns_coordination_active(env)
        if coordination_active:
            self._promote_globally_unreachable_normals(env)
        if bool(getattr(env.cfg, "hrl_attraction_dispatch_enabled", False)):
            return self._plan_attraction_dispatch(env)
        ordered_agents = self._ordered_agents(env)
        step_now = int(env.state.step_index)
        self._build_initial_directional_plan(env)
        if coordination_active:
            self._prune_uav_task_reservations(env, step_now)
        if coordination_active:
            self._prune_uav_anchor_tasks(env)
        self._step_unique_agent_task_keys.clear()
        pending_task_ids = [
            str(t.task_id)
            for t in env.state.tasks.values()
            if self._task_planner_active(t)
        ]

        candidate_pairs: List[Tuple[str, str]] = []
        raw_scores: Dict[Tuple[str, str], float] = {}
        raw_tiers: Dict[Tuple[str, str], int] = {}
        pending_norm_count = int(self._pending_normal_task_count(env))
        _, norm_reach_by_truck, _any_reachable_normal = self._normal_reachability_snapshot(env)
        uav_shortlists: Dict[str, Optional[set]] = {}
        tc_tier3_candidate_step = 0
        tc_tier2_candidate_step = 0
        tc_candidate_total_step = 0
        support_filtered_no_bind_step = 0
        ablate_support_chain = bool(getattr(env.cfg, "erc_ablate_support_chain", False))
        support_budget_agents: set = set()
        support_budget_limit = int(max(getattr(env.cfg, "hrl_support_max_trucks_when_normal_pending", 0), 0))
        support_budget_require_warning = bool(getattr(env.cfg, "hrl_support_budget_require_warning_when_normal", False))
        if bool(self.use_event_trigger):
            self.tc_global_assignment_called_count_total = int(self.tc_global_assignment_called_count_total) + 1
            if bool(getattr(env.cfg, "erc_ablate_tc_global_assignment", False)) and (not bool(self._tc_global_assignment_runtime_enabled(env))):
                self.tc_global_assignment_skipped_by_ablation_count_total = int(self.tc_global_assignment_skipped_by_ablation_count_total) + 1
            else:
                self.tc_assignment_epoch_applied_count_total = int(self.tc_assignment_epoch_applied_count_total) + 1
        for aid in ordered_agents:
            st0 = env.state.agents.get(str(aid), None)
            if st0 is None or st0.kind != AgentKind.UAV or bool(getattr(st0, "crashed", False)):
                continue
            uav_shortlists[str(aid)] = self._uav_assignment_shortlist_task_ids(env, str(aid))

        for aid in ordered_agents:
            st = env.state.agents[aid]
            if st.kind == AgentKind.UAV and bool(st.crashed):
                continue
            shortlist = uav_shortlists.get(str(aid), None) if st.kind == AgentKind.UAV else None
            for tid in pending_task_ids:
                task = env.state.tasks.get(str(tid), None)
                if task is None or task.status != TaskStatus.PENDING:
                    continue
                owner = self._task_contract_owner(env, str(tid))
                if owner is not None and str(owner) != str(aid):
                    continue
                if st.kind == AgentKind.UAV and task.kind != TaskKind.EMERGENCY:
                    continue
                if st.kind == AgentKind.UAV and shortlist is not None and str(tid) not in shortlist:
                    continue
                if st.kind == AgentKind.UAV and self._uav_task_reserved_by_other(str(aid), str(tid)):
                    continue
                if self._region_commitment_active(env) and not self._region_task_allowed(env, str(aid), task):
                    continue
                is_support_candidate = False
                if coordination_active and st.kind == AgentKind.TRUCK and task.kind == TaskKind.EMERGENCY:
                    is_support_candidate = bool(self._truck_emergency_support_candidate(env, str(aid), task))
                    if is_support_candidate:
                        self.support_chain_candidate_count_total = int(self.support_chain_candidate_count_total) + 1
                        if int(pending_norm_count) > 0:
                            if support_budget_require_warning:
                                warning = float(np.clip(getattr(env.cfg, "hrl_timecritical_lifeline_warning_ratio", 0.55), 0.0, 1.0))
                                is_warning_tc = bool(
                                    self._is_timecritical_lightweight_task(task)
                                    and float(self._task_lifeline_ratio(task)) <= warning
                                )
                                if not is_warning_tc:
                                    self.support_chain_blocked_by_ablation_count_total = int(self.support_chain_blocked_by_ablation_count_total) + 1
                                    is_support_candidate = False
                            if is_support_candidate and support_budget_limit > 0 and str(aid) not in support_budget_agents:
                                if len(support_budget_agents) >= support_budget_limit:
                                    self.support_chain_blocked_by_ablation_count_total = int(self.support_chain_blocked_by_ablation_count_total) + 1
                                    is_support_candidate = False
                                else:
                                    support_budget_agents.add(str(aid))
                        if (not is_support_candidate):
                            pass
                        elif ablate_support_chain:
                            self.support_chain_blocked_by_ablation_count_total = int(self.support_chain_blocked_by_ablation_count_total) + 1
                            is_support_candidate = False
                        else:
                            support_gain_for_gate = float(np.clip(self._truck_support_serviceability_gain(env, str(aid), task), 0.0, 1.0))
                            if bool(self._support_chain_condition_enabled(env, str(aid), task, support_gain_for_gate)):
                                self.support_chain_applied_count_total = int(self.support_chain_applied_count_total) + 1
                            else:
                                self.support_chain_blocked_by_ablation_count_total = int(self.support_chain_blocked_by_ablation_count_total) + 1
                                is_support_candidate = False

                if (
                    coordination_active
                    and
                    st.kind == AgentKind.TRUCK
                    and task.kind == TaskKind.EMERGENCY
                    and is_support_candidate
                    and pending_norm_count > 0
                    and bool(norm_reach_by_truck.get(str(aid), True))
                    and float(getattr(env.cfg, "map_size_m", 5000.0))
                    >= float(max(getattr(env.cfg, "hrl_support_proxy_warning_gate_min_map_size_m", 9000.0), 0.0))
                ):
                    bind_info_guard = self._support_bound_delivery_info(env, str(aid), task)
                    require_warning = bool(
                        getattr(env.cfg, "hrl_support_proxy_require_warning_bind_when_normal_reachable", True)
                    )
                    if require_warning:
                        bind_ok_guard = bool(
                            float(bind_info_guard.get("bound_timecritical_critical", 0.0)) > 0.0
                            or float(bind_info_guard.get("bound_timecritical_warning", 0.0)) > 0.0
                        )
                    else:
                        bind_ok_guard = bool(float(bind_info_guard.get("bound_timecritical", 0.0)) > 0.0)
                    if not bind_ok_guard:
                        if not self._support_escape_hatch_allows(env, str(aid), task):
                            support_filtered_no_bind_step += 1
                            continue

                if st.kind == AgentKind.TRUCK and hasattr(env, "is_task_serviceable_by_agent"):
                    serviceable = bool(env.is_task_serviceable_by_agent(str(aid), task))
                    if not serviceable:
                        # Support proxy path: emergency support candidate can still be used
                        # as a forward-support navigation goal even when direct service is not allowed.
                        if not (task.kind == TaskKind.EMERGENCY and is_support_candidate):
                            continue
                        self.support_proxy_candidate_count_total = int(self.support_proxy_candidate_count_total) + 1

                if (
                    self.use_event_trigger
                    and st.kind == AgentKind.TRUCK
                    and pending_norm_count > 0
                    and task.kind == TaskKind.EMERGENCY
                    and (not self._is_island_task(env, task))
                    and (not bool(self._truck_emergency_relief_allowed(env, str(aid), task)))
                    and (not bool(is_support_candidate))
                ):
                    # Normal-throughput protection for ERC-RHC: if emergency is
                    # UAV-coverable while normal backlog exists, keep truck
                    # assignment capacity focused on NORMAL tasks.
                    # Exception: keep explicit support candidates in the pool so
                    # support->delivery conversion can still be formed.
                    continue

                if coordination_active and st.kind == AgentKind.TRUCK and task.kind == TaskKind.EMERGENCY:
                    if is_support_candidate and self._support_backoff_active(env, str(aid), task):
                        self.support_no_gain_backoff_block_count_total = int(self.support_no_gain_backoff_block_count_total) + 1
                        continue
                    if is_support_candidate and bool(getattr(env.cfg, "hrl_support_requires_timecritical_binding", True)):
                        bind_info = self._support_bound_delivery_info(env, str(aid), task)
                        allow_bulk = bool(getattr(env.cfg, "hrl_support_fallback_allow_bulk_binding", False))
                        bind_ok = float(bind_info.get("bound_any", 0.0)) > 0.0 if allow_bulk else float(bind_info.get("bound_timecritical", 0.0)) > 0.0

                        # Medium-scale C fallback: do not hard-block support when
                        # task is low-lifeline time-critical and UAV cover is low.
                        enforce_bind_gate = True
                        min_map_bind = float(max(getattr(env.cfg, "hrl_support_bind_enforce_min_map_size_m", 9000.0), 0.0))
                        if float(getattr(env.cfg, "map_size_m", 5000.0)) < min_map_bind:
                            if self._is_timecritical_lightweight_task(task):
                                ratio = float(np.clip(self._task_lifeline_ratio(task), 0.0, 1.0))
                                warning = float(np.clip(getattr(env.cfg, "hrl_timecritical_lifeline_warning_ratio", 0.55), 0.0, 1.0))
                                cover_thr = float(np.clip(getattr(env.cfg, "hrl_support_escape_hatch_low_cover_threshold", 0.32), 0.0, 1.0))
                                cover = float(np.clip(self._uav_emergency_cover_fraction(env, task), 0.0, 1.0))
                                urg = float(np.clip(float(getattr(task, "urgency_score", self._norm_deadline_urgency(task, int(env.state.step_index)))), 0.0, 1.0))
                                if (ratio <= warning and cover < cover_thr and urg >= 0.55):
                                    enforce_bind_gate = False

                        if enforce_bind_gate and (not bool(bind_ok)):
                            if not self._support_escape_hatch_allows(env, str(aid), task):
                                support_filtered_no_bind_step += 1
                                continue
                score = float(self._score_goal_for_agent(env, aid, str(tid)))
                if not np.isfinite(score):
                    continue
                tier = int(self._task_priority_tier(env, str(aid), task))
                if self._is_timecritical_lightweight_task(task):
                    tc_candidate_total_step += 1
                    if tier >= 3:
                        tc_tier3_candidate_step += 1
                    elif tier >= 2:
                        tc_tier2_candidate_step += 1
                self._step_unique_agent_task_keys.add((str(aid), str(tid)))
                candidate_pairs.append((str(aid), str(tid)))
                raw_scores[(str(aid), str(tid))] = score
                raw_tiers[(str(aid), str(tid))] = tier

        goals: Dict[str, Optional[str]] = {aid: None for aid in ordered_agents}
        used_tasks: set = set()

        tc_tier3_selected_step = 0
        tc_tier2_selected_step = 0
        tc_selected_total_step = 0

        # Simplified ERC path: disable heavy TC global assignment epoch and
        # fall back to deterministic per-agent greedy assignment.
        tc_global_runtime = bool(self._tc_global_assignment_runtime_enabled(env))
        if candidate_pairs and (not tc_global_runtime):
            norm_scores = self._normalize_assignment_scores(raw_scores)
            for aid in ordered_agents:
                st = env.state.agents.get(str(aid), None)
                if st is None or bool(getattr(st, "crashed", False)):
                    continue
                if goals.get(str(aid), None) is not None:
                    continue
                best_tid = None
                best_sc = -1e18
                for (aa, tt) in candidate_pairs:
                    if str(aa) != str(aid):
                        continue
                    if str(tt) in used_tasks:
                        continue
                    sc = float(norm_scores.get((str(aa), str(tt)), -1e18))
                    if not np.isfinite(sc):
                        continue
                    if sc > best_sc:
                        best_sc = sc
                        best_tid = str(tt)
                if best_tid is None:
                    continue
                goals[str(aid)] = str(best_tid)
                used_tasks.add(str(best_tid))
                task_sel = env.state.tasks.get(str(best_tid), None)
                if task_sel is not None and self._is_timecritical_lightweight_task(task_sel):
                    tc_selected_total_step += 1
                    tier_sel = int(raw_tiers.get((str(aid), str(best_tid)), self._task_priority_tier(env, str(aid), task_sel)))
                    if tier_sel >= 3:
                        tc_tier3_selected_step += 1
                    elif tier_sel >= 2:
                        tc_tier2_selected_step += 1
            if bool(getattr(env.cfg, "hrl_far_routine_bootstrap_enabled", False)):
                step_boot = int(getattr(env.state, "step_index", 0))
                boot_window = int(max(getattr(env.cfg, "hrl_far_routine_bootstrap_window_steps", 20), 0))
                map_size = float(getattr(env.cfg, "map_size_m", 5000.0))
                min_map = float(max(getattr(env.cfg, "hrl_far_routine_bootstrap_min_map_size_m", 9000.0), 0.0))
                min_far = float(max(getattr(env.cfg, "hrl_far_routine_bootstrap_min_distance_m", 7000.0), 0.0))
                if step_boot <= boot_window and map_size >= min_map:
                    truck_ids_boot = [
                        str(a)
                        for a in ordered_agents
                        if env.state.agents.get(str(a), None) is not None
                        and env.state.agents[str(a)].kind == AgentKind.TRUCK
                        and (not bool(getattr(env.state.agents[str(a)], "crashed", False)))
                    ]
                    best_task_id = None
                    best_task_dist = -1.0
                    best_truck_id = None
                    for _task in env.state.tasks.values():
                        if _task.status != TaskStatus.PENDING or _task.kind != TaskKind.NORMAL:
                            continue
                        nearest_d = float("inf")
                        nearest_truck = None
                        for _taid in truck_ids_boot:
                            _d = float(self._truck_task_distance(env, str(_taid), _task))
                            if np.isfinite(_d) and _d < nearest_d:
                                nearest_d = float(_d)
                                nearest_truck = str(_taid)
                        if nearest_truck is None or not np.isfinite(nearest_d):
                            continue
                        if nearest_d >= min_far and nearest_d > best_task_dist:
                            best_task_dist = float(nearest_d)
                            best_task_id = str(_task.task_id)
                            best_truck_id = str(nearest_truck)
                    if best_task_id is not None and best_truck_id is not None:
                        for _aid_old, _tid_old in list(goals.items()):
                            if str(_aid_old) != str(best_truck_id) and str(_tid_old) == str(best_task_id):
                                goals[str(_aid_old)] = None
                        old_tid = goals.get(str(best_truck_id), None)
                        if old_tid is not None and str(old_tid) in used_tasks:
                            used_tasks.discard(str(old_tid))
                        goals[str(best_truck_id)] = str(best_task_id)
                        used_tasks.add(str(best_task_id))
                        self._far_routine_bootstrap_force_step[str(best_truck_id)] = int(step_boot)
            candidate_pairs = []

        if candidate_pairs:
            # Hard tiered assignment (not soft tier bonus):
            # Tier3 > Tier2 > Tier1 > Tier0.
            norm_scores = self._normalize_assignment_scores(raw_scores)
            split_objective = bool(getattr(env.cfg, "hrl_separate_agent_objectives_enabled", True))
            tc_stats = {"total": 0, "tier3": 0, "tier2": 0}

            if split_objective:
                # Split objectives explicitly:
                # 1) UAV handles time-critical/emergency first;
                # 2) Truck handles routine bulk first;
                # 3) Truck emergency fallback only after above stages.
                uav_pairs = [
                    (str(a), str(t))
                    for (a, t) in candidate_pairs
                    if env.state.agents.get(str(a), None) is not None
                    and env.state.agents[str(a)].kind == AgentKind.UAV
                    and env.state.tasks.get(str(t), None) is not None
                    and env.state.tasks[str(t)].kind == TaskKind.EMERGENCY
                ]
                truck_pairs_normal = [
                    (str(a), str(t))
                    for (a, t) in candidate_pairs
                    if env.state.agents.get(str(a), None) is not None
                    and env.state.agents[str(a)].kind == AgentKind.TRUCK
                    and env.state.tasks.get(str(t), None) is not None
                    and env.state.tasks[str(t)].kind == TaskKind.NORMAL
                ]
                truck_pairs_emergency = [
                    (str(a), str(t))
                    for (a, t) in candidate_pairs
                    if env.state.agents.get(str(a), None) is not None
                    and env.state.agents[str(a)].kind == AgentKind.TRUCK
                    and env.state.tasks.get(str(t), None) is not None
                    and env.state.tasks[str(t)].kind == TaskKind.EMERGENCY
                ]
                # Stage-1: UAV objective first for time-critical responsiveness.
                self._assign_by_hard_tier(env, uav_pairs, norm_scores, raw_tiers, goals, used_tasks, tc_stats)
                # Stage-2: Truck objective on bulk throughput.
                self._assign_by_hard_tier(env, truck_pairs_normal, norm_scores, raw_tiers, goals, used_tasks, tc_stats)

                if bool(getattr(env.cfg, "hrl_far_routine_bootstrap_enabled", False)):
                    step_boot = int(getattr(env.state, "step_index", 0))
                    boot_window = int(max(getattr(env.cfg, "hrl_far_routine_bootstrap_window_steps", 20), 0))
                    map_size = float(getattr(env.cfg, "map_size_m", 5000.0))
                    min_map = float(max(getattr(env.cfg, "hrl_far_routine_bootstrap_min_map_size_m", 9000.0), 0.0))
                    min_far = float(max(getattr(env.cfg, "hrl_far_routine_bootstrap_min_distance_m", 7000.0), 0.0))
                    if step_boot <= boot_window and map_size >= min_map:
                        truck_ids_boot = [
                            str(a)
                            for a in ordered_agents
                            if env.state.agents.get(str(a), None) is not None
                            and env.state.agents[str(a)].kind == AgentKind.TRUCK
                            and (not bool(getattr(env.state.agents[str(a)], "crashed", False)))
                        ]
                        best_task_id = None
                        best_task_dist = -1.0
                        best_truck_id = None
                        for _task in env.state.tasks.values():
                            if _task.status != TaskStatus.PENDING or _task.kind != TaskKind.NORMAL:
                                continue
                            nearest_d = float("inf")
                            nearest_truck = None
                            for _taid in truck_ids_boot:
                                _d = float(self._truck_task_distance(env, str(_taid), _task))
                                if np.isfinite(_d) and _d < nearest_d:
                                    nearest_d = float(_d)
                                    nearest_truck = str(_taid)
                            if nearest_truck is None or not np.isfinite(nearest_d):
                                continue
                            if nearest_d >= min_far and nearest_d > best_task_dist:
                                best_task_dist = float(nearest_d)
                                best_task_id = str(_task.task_id)
                                best_truck_id = str(nearest_truck)
                        if best_task_id is not None and best_truck_id is not None:
                            for _aid_old, _tid_old in list(goals.items()):
                                if str(_aid_old) != str(best_truck_id) and str(_tid_old) == str(best_task_id):
                                    goals[str(_aid_old)] = None
                            old_tid = goals.get(str(best_truck_id), None)
                            if old_tid is not None and str(old_tid) in used_tasks:
                                used_tasks.discard(str(old_tid))
                            goals[str(best_truck_id)] = str(best_task_id)
                            used_tasks.add(str(best_task_id))
                            self._far_routine_bootstrap_force_step[str(best_truck_id)] = int(step_boot)

                # Stage-2.5: reserve one truck for support relay when critical
                # time-critical backlog exists and all trucks are already tied to normal goals.
                if coordination_active and bool(getattr(env.cfg, "hrl_support_relay_reserve_enabled", True)):
                    warning_thr = float(np.clip(getattr(env.cfg, "hrl_timecritical_lifeline_warning_ratio", 0.55), 0.0, 1.0))
                    cover_thr = float(
                        np.clip(
                            getattr(
                                env.cfg,
                                "hrl_support_relay_cover_threshold",
                                getattr(env.cfg, "hrl_truck_emergency_force_relief_uav_cover_threshold", 0.35),
                            ),
                            0.0,
                            1.0,
                        )
                    )
                    pressure_pending = 0
                    for _t in env.state.tasks.values():
                        if _t.status != TaskStatus.PENDING:
                            continue
                        if not self._is_timecritical_lightweight_task(_t):
                            continue
                        _ratio = float(self._task_lifeline_ratio(_t))
                        _urg = float(np.clip(float(getattr(_t, "urgency_score", self._norm_deadline_urgency(_t, int(env.state.step_index)))), 0.0, 1.0))
                        _cover = float(np.clip(self._uav_emergency_cover_fraction(env, _t), 0.0, 1.0))
                        if ((_ratio <= warning_thr) or (_urg >= 0.75)) and (_cover < cover_thr):
                            pressure_pending += 1
                    min_critical = int(max(getattr(env.cfg, "hrl_support_relay_min_critical_timecritical", 1), 0))
                    if pressure_pending >= min_critical:
                        truck_ids = [
                            str(a) for a in ordered_agents
                            if env.state.agents.get(str(a), None) is not None
                            and env.state.agents[str(a)].kind == AgentKind.TRUCK
                            and (not bool(getattr(env.state.agents[str(a)], "crashed", False)))
                        ]
                        assigned_trucks = [str(a) for a in truck_ids if goals.get(str(a), None) is not None]
                        if truck_ids and len(assigned_trucks) >= len(truck_ids):
                            release_aid = None
                            release_tid = None
                            best_key = None
                            for _aid in assigned_trucks:
                                _tid = goals.get(str(_aid), None)
                                _task = env.state.tasks.get(str(_tid), None) if _tid is not None else None
                                if not self._task_planner_active(_task):
                                    continue
                                if _task.kind != TaskKind.NORMAL:
                                    continue
                                _urg = float(self._norm_deadline_urgency(_task, int(env.state.step_index)))
                                _d = float(self._truck_task_distance(env, str(_aid), _task))
                                _key = (float(_urg), -float(_d if np.isfinite(_d) else 0.0))
                                if best_key is None or _key < best_key:
                                    best_key = _key
                                    release_aid = str(_aid)
                                    release_tid = str(_tid)
                            if release_aid is not None:
                                goals[str(release_aid)] = None
                                if release_tid is not None and str(release_tid) in used_tasks:
                                    used_tasks.discard(str(release_tid))
                                self.support_relay_reserved_count_total = int(self.support_relay_reserved_count_total) + 1
                                self._support_relay_force_step[str(release_aid)] = int(env.state.step_index)

                # Stage-2.75: medium-scale critical diversion.
                # If all trucks are occupied by normal goals, allow a tiny number of
                # diversions to critical/warning time-critical emergency support.
                if coordination_active and bool(getattr(env.cfg, "hrl_support_critical_diversion_enabled", True)):
                    map_size = float(getattr(env.cfg, "map_size_m", 5000.0))
                    max_map = float(max(getattr(env.cfg, "hrl_support_critical_diversion_max_map_size_m", 9000.0), 0.0))
                    if map_size <= max_map:
                        max_div = int(max(getattr(env.cfg, "hrl_support_critical_diversion_max_trucks", 1), 0))
                        cover_thr = float(np.clip(getattr(env.cfg, "hrl_support_critical_diversion_cover_threshold", 0.38), 0.0, 1.0))
                        warning = float(np.clip(getattr(env.cfg, "hrl_timecritical_lifeline_warning_ratio", 0.55), 0.0, 1.0))
                        critical = float(np.clip(getattr(env.cfg, "hrl_timecritical_lifeline_critical_ratio", 0.35), 0.0, 1.0))
                        diverted = 0
                        truck_ids_local = [
                            str(a) for a in ordered_agents
                            if env.state.agents.get(str(a), None) is not None
                            and env.state.agents[str(a)].kind == AgentKind.TRUCK
                            and (not bool(getattr(env.state.agents[str(a)], "crashed", False)))
                        ]
                        for taid in truck_ids_local:
                            if diverted >= max_div:
                                break
                            gid_cur = goals.get(str(taid), None)
                            t_cur = env.state.tasks.get(str(gid_cur), None) if gid_cur is not None else None
                            if t_cur is None or t_cur.status != TaskStatus.PENDING or t_cur.kind != TaskKind.NORMAL:
                                continue

                            best_tid = None
                            best_sc = -1e18
                            for (_a,_t) in truck_pairs_emergency:
                                if str(_a) != str(taid):
                                    continue
                                tsk = env.state.tasks.get(str(_t), None)
                                if (not self._task_planner_active(tsk)) or tsk.kind != TaskKind.EMERGENCY:
                                    continue
                                if not self._is_timecritical_lightweight_task(tsk):
                                    continue
                                ratio = float(np.clip(self._task_lifeline_ratio(tsk), 0.0, 1.0))
                                if ratio > warning:
                                    continue
                                cover = float(np.clip(self._uav_emergency_cover_fraction(env, tsk), 0.0, 1.0))
                                if cover >= cover_thr:
                                    continue
                                bind_info = self._support_bound_delivery_info(env, str(taid), tsk)
                                if float(bind_info.get("bound_timecritical", 0.0)) <= 0.0 and (not self._support_escape_hatch_allows(env, str(taid), tsk)):
                                    continue
                                sc = float(norm_scores.get((str(taid), str(_t)), -1e9))
                                if ratio <= critical:
                                    sc += 0.25
                                if np.isfinite(sc) and sc > best_sc + 1e-12:
                                    best_sc = float(sc)
                                    best_tid = str(_t)
                            if best_tid is None:
                                continue
                            # divert this truck from normal to critical time-critical support
                            if gid_cur is not None and str(gid_cur) in used_tasks:
                                used_tasks.discard(str(gid_cur))
                            goals[str(taid)] = str(best_tid)
                            used_tasks.add(str(best_tid))
                            diverted += 1

                # Stage-3: Truck emergency fallback on remaining tasks only.
                self._assign_by_hard_tier(env, truck_pairs_emergency, norm_scores, raw_tiers, goals, used_tasks, tc_stats)
            else:
                self._assign_by_hard_tier(env, candidate_pairs, norm_scores, raw_tiers, goals, used_tasks, tc_stats)

            tc_selected_total_step += int(tc_stats.get("total", 0))
            tc_tier3_selected_step += int(tc_stats.get("tier3", 0))
            tc_tier2_selected_step += int(tc_stats.get("tier2", 0))

        self.timecritical_tier3_candidate_count_total += int(tc_tier3_candidate_step)
        self.timecritical_tier2_candidate_count_total += int(tc_tier2_candidate_step)
        self.timecritical_tier3_selected_count_total += int(tc_tier3_selected_step)
        self.timecritical_tier2_selected_count_total += int(tc_tier2_selected_step)
        self.timecritical_candidate_ignored_count_total += int(max(tc_candidate_total_step - tc_selected_total_step, 0))
        self.support_filtered_no_bound_timecritical_delivery_count_total += int(support_filtered_no_bind_step)

        # TC feasibility triage: direct-feasible tasks stay on the UAV path;
        # support-required tasks reserve a concrete truck-UAV-task chain; truly
        # infeasible tasks are left out of repeated direct launch-gate churn.
        if coordination_active:
            self._apply_tc_support_required_locks(env, ordered_agents, goals, used_tasks)

        # Island coverage repair: ensure at least one feasible UAV island assignment
        # when island emergencies exist, otherwise map-update churn can starve island conversion.
        island_ids = set(getattr(env, "_current_island_emergency_task_ids", lambda: set())())
        island_pending = [
            str(tid)
            for tid in sorted(island_ids)
            if str(tid) in env.state.tasks and env.state.tasks[str(tid)].status == TaskStatus.PENDING
        ]
        if island_pending:
            has_uav_island = False
            for aid, gid in goals.items():
                st = env.state.agents.get(str(aid), None)
                if st is None or st.kind != AgentKind.UAV:
                    continue
                if gid is not None and str(gid) in island_pending:
                    has_uav_island = True
                    break
            if not has_uav_island:
                best_pair: Optional[Tuple[str, str]] = None
                best_score = -1e18
                for aid in ordered_agents:
                    st = env.state.agents.get(str(aid), None)
                    if st is None or st.kind != AgentKind.UAV or bool(st.crashed):
                        continue
                    for tid in island_pending:
                        task = env.state.tasks.get(str(tid), None)
                        if task is None or task.status != TaskStatus.PENDING:
                            continue
                        if not self._uav_task_feasible(env, str(aid), task):
                            continue
                        sc = float(self._score_uav_task(env, str(aid), task))
                        if np.isfinite(sc) and sc > best_score + 1e-12:
                            best_score = float(sc)
                            best_pair = (str(aid), str(tid))
                if best_pair is not None:
                    aid_sel, tid_sel = best_pair
                    prev_tid = goals.get(str(aid_sel), None)
                    if prev_tid is not None and str(prev_tid) in used_tasks:
                        used_tasks.discard(str(prev_tid))
                    goals[str(aid_sel)] = str(tid_sel)
                    used_tasks.add(str(tid_sel))

        # Support-to-delivery dispatch: when truck support has a concrete
        # time-critical bind, proactively dispatch a UAV to that delivery.
        if coordination_active and bool(getattr(env.cfg, "hrl_support_bound_dispatch_enabled", True)):
            dispatched_uavs: set = set()
            for taid in ordered_agents:
                tst = env.state.agents.get(str(taid), None)
                if tst is None or tst.kind != AgentKind.TRUCK or bool(getattr(tst, "crashed", False)):
                    continue
                tgoal = goals.get(str(taid), None)
                if tgoal is None:
                    continue
                ttask = env.state.tasks.get(str(tgoal), None)
                if ttask is None or ttask.status != TaskStatus.PENDING or ttask.kind != TaskKind.EMERGENCY:
                    continue
                if not bool(self._truck_emergency_support_candidate(env, str(taid), ttask)):
                    continue
                chain_info = self._support_bound_chain_info_for_truck(env, str(taid))
                chain_is_support_required = False
                if chain_info is not None and str(chain_info.get("task_id", "")) == str(ttask.task_id):
                    chain_is_support_required = bool(self._tc_support_chain_class.get(str(taid), "") == "support_required")
                    uid = str(chain_info.get("uav_id", "")).strip()
                    tid_bind = str(chain_info.get("task_id", str(ttask.task_id))).strip()
                else:
                    bind_info = self._support_bound_delivery_info(env, str(taid), ttask)
                    if float(bind_info.get("bound_timecritical", 0.0)) <= 0.0:
                        continue
                    uid = str(bind_info.get("bound_timecritical_uav_id", "")).strip()
                    tid_bind = str(bind_info.get("bound_timecritical_task_id", str(ttask.task_id))).strip()
                if (not uid) or (not tid_bind) or uid in dispatched_uavs:
                    continue
                uav_st = env.state.agents.get(uid, None)
                bind_task = env.state.tasks.get(tid_bind, None)
                if uav_st is None or uav_st.kind != AgentKind.UAV or bool(getattr(uav_st, "crashed", False)):
                    continue
                if bind_task is None or bind_task.status != TaskStatus.PENDING:
                    continue
                cur_gid = goals.get(uid, None)
                cur_task = env.state.tasks.get(str(cur_gid), None) if cur_gid is not None else None
                cur_tier = int(self._task_priority_tier(env, str(uid), cur_task)) if cur_task is not None else -1
                new_tier = int(self._task_priority_tier(env, str(uid), bind_task))
                should_take = bool(cur_gid is None or cur_task is None or new_tier > cur_tier)
                rc_force_dispatch = bool(getattr(env.cfg, "support_force_dispatch_enabled", False))
                if rc_force_dispatch:
                    if cur_task is None or (not self._is_timecritical_lightweight_task(cur_task)):
                        should_take = True
                    else:
                        cur_ratio = float(np.clip(self._task_lifeline_ratio(cur_task), 0.0, 1.0))
                        bind_ratio = float(np.clip(self._task_lifeline_ratio(bind_task), 0.0, 1.0))
                        if new_tier >= cur_tier:
                            should_take = True
                        elif bool(getattr(env.cfg, "support_force_uav_preempt_enabled", False)) and bind_ratio + 0.05 < cur_ratio:
                            should_take = True
                if not should_take:
                    continue
                docked_actionable = False
                docked_actionable_fn = getattr(env, "_uav_docked_task_actionable_now", None)
                if callable(docked_actionable_fn):
                    try:
                        docked_actionable = bool(docked_actionable_fn(str(uid), bind_task))
                    except Exception:
                        docked_actionable = False
                dispatch_ready = bool(docked_actionable or self._uav_task_feasible(env, str(uid), bind_task))
                chain_commit_ready = bool(
                    uav_st.follow_target is not None
                    and str(getattr(uav_st, "follow_target", "")) == str(taid)
                    and (not bool(getattr(uav_st, "uav_needs_reload_flag", False)))
                    and (not self._uav_needs_recovery(env, str(uid)))
                )
                loaded_fn = getattr(env, "_uav_loaded", None)
                if chain_commit_ready and callable(loaded_fn):
                    try:
                        chain_commit_ready = bool(loaded_fn(str(uid)))
                    except Exception:
                        chain_commit_ready = False
                support_chain_ready = True
                if chain_is_support_required:
                    support_chain_ready = bool(self._docked_uav_sortie_chain_ready(env, str(uid), bind_task))
                    alt_tid = self._uav_direct_feasible_alternative_tc(env, str(uid), exclude_task_id=str(tid_bind))
                    if (not support_chain_ready) and alt_tid is not None:
                        goals[str(uid)] = str(alt_tid)
                        used_tasks.add(str(alt_tid))
                        dispatched_uavs.add(str(uid))
                        continue
                    dispatch_ready = bool(dispatch_ready and support_chain_ready)
                    chain_commit_ready = bool(chain_commit_ready and support_chain_ready)
                if dispatch_ready or chain_commit_ready:
                    goals[str(uid)] = str(tid_bind)
                    # If truck was temporarily occupying the same emergency as support,
                    # move it to the best available follow-up goal instead of leaving null.
                    if (not chain_is_support_required) and str(goals.get(str(taid), "")) == str(tid_bind):
                        fallback_tid = self._rc_best_truck_fallback_goal(env, str(taid), excluded_task_id=str(tid_bind))
                        goals[str(taid)] = fallback_tid
                    dispatched_uavs.add(str(uid))
                    chain_steps = int(max(getattr(env.cfg, "hrl_relaxed_chain_commitment_steps", 6), 0))
                    if rc_force_dispatch:
                        chain_steps = int(max(chain_steps, int(max(getattr(env.cfg, "support_force_commit_steps", 10), 0))))
                    self._relaxed_chain_until_step[str(uid)] = int(max(self._relaxed_chain_until_step.get(str(uid), -1), int(env.state.step_index) + chain_steps))
                    if dispatch_ready:
                        self.support_bound_dispatch_count_total = int(self.support_bound_dispatch_count_total) + 1
                else:
                    goals[str(uid)] = str(taid)
                    dispatched_uavs.add(str(uid))
                    self.support_bound_recovery_redirect_count_total = int(self.support_bound_recovery_redirect_count_total) + 1
        # Stage-2 safety fallback: unassigned UAV gets recovery truck if needed.
        for aid in ordered_agents:
            st = env.state.agents[aid]
            if st.kind != AgentKind.UAV or bool(st.crashed):
                continue
            if goals.get(aid, None) is not None:
                continue
            if self._uav_needs_recovery(env, aid):
                near_tid, _ = self._nearest_truck(env, aid)
                goals[aid] = near_tid

        # Stage-2.5 proactive staging:
        # If UAV currently has no feasible emergency task, let it proactively pick
        # the best support truck (direction + emergency pull + follower balance)
        # instead of idling at depot.
        if bool(getattr(env.cfg, "hrl_uav_idle_truck_staging_enabled", True)):
            prefer_initial = bool(getattr(env.cfg, "hrl_uav_idle_staging_prefer_initial_plan", True))
            respect_cap = bool(getattr(env.cfg, "hrl_uav_idle_staging_respect_truck_cap", True))
            cap = int(max(getattr(env.cfg, "uav_max_followers_per_truck", 0), 0))
            slot_fn = getattr(env, "_truck_has_follow_slot", None)
            staged_by_truck: Dict[str, int] = {}
            for uaid, ugid in goals.items():
                ust = env.state.agents.get(str(uaid), None)
                if ust is None or ust.kind != AgentKind.UAV:
                    continue
                tag = env.state.agents.get(str(ugid), None) if ugid is not None else None
                if tag is not None and tag.kind == AgentKind.TRUCK:
                    staged_by_truck[str(ugid)] = int(staged_by_truck.get(str(ugid), 0) + 1)
            for aid in ordered_agents:
                st = env.state.agents[aid]
                if st.kind != AgentKind.UAV or bool(st.crashed):
                    continue
                if goals.get(aid, None) is not None:
                    continue
                if self._uav_needs_recovery(env, aid):
                    continue
                has_feasible_task = False
                for task in env.state.tasks.values():
                    if task.kind != TaskKind.EMERGENCY or task.status != TaskStatus.PENDING:
                        continue
                    if self._uav_task_feasible(env, str(aid), task):
                        has_feasible_task = True
                        break
                if has_feasible_task:
                    continue

                def _truck_slot_ok(tid: str) -> bool:
                    if (not respect_cap) or cap <= 0:
                        return True
                    if int(staged_by_truck.get(str(tid), 0)) >= int(cap):
                        return False
                    if callable(slot_fn):
                        try:
                            return bool(slot_fn(str(tid), exclude_aid=str(aid)))
                        except Exception:
                            return True
                    return True

                if prefer_initial and self._in_initial_directional_phase(env):
                    self._build_initial_directional_plan(env)
                    pref_tid = str(self._initial_directional_uav_truck.get(str(aid), ""))
                    pref_ag = env.state.agents.get(str(pref_tid), None)
                    if pref_ag is not None and pref_ag.kind == AgentKind.TRUCK and (not bool(getattr(pref_ag, "crashed", False))):
                        if _truck_slot_ok(str(pref_tid)):
                            goals[aid] = str(pref_tid)
                            staged_by_truck[str(pref_tid)] = int(staged_by_truck.get(str(pref_tid), 0) + 1)
                            continue

                best_tid: Optional[str] = None
                best_sc = -1e18
                for tid, ag in env.state.agents.items():
                    if ag.kind != AgentKind.TRUCK or bool(getattr(ag, "crashed", False)):
                        continue
                    if not _truck_slot_ok(str(tid)):
                        continue
                    ax, ay = self._agent_xy(env, str(aid))
                    tx, ty = self._agent_xy(env, str(tid))
                    d = float(np.hypot(float(tx) - float(ax), float(ty) - float(ay)))
                    sc = float(self._score_uav_truck(env, str(aid), str(tid), d))
                    if np.isfinite(sc) and sc > best_sc + 1e-12:
                        best_sc = float(sc)
                        best_tid = str(tid)
                min_sc = float(getattr(env.cfg, "hrl_uav_idle_truck_staging_min_score", 0.10))
                has_pending_emergency = bool(
                    any(t.kind == TaskKind.EMERGENCY and t.status == TaskStatus.PENDING for t in env.state.tasks.values())
                )
                if best_tid is not None and (best_sc >= min_sc or has_pending_emergency):
                    goals[aid] = str(best_tid)
                    staged_by_truck[str(best_tid)] = int(staged_by_truck.get(str(best_tid), 0) + 1)

        # Stage-3 anti-idle truck fallback:
        # If a truck remains unassigned while pending reachable tasks exist,
        # force nearest-task assignment (prefer emergency, then nearest distance).
        for aid in ordered_agents:
            st = env.state.agents[aid]
            if st.kind != AgentKind.TRUCK or bool(st.crashed):
                continue
            if goals.get(aid, None) is not None:
                continue
            if self._truck_stage_blocks_task_goal(env, str(aid)):
                continue

            tid = self._nearest_truck_reachable_task(
                env,
                str(aid),
                used_tasks=used_tasks,
                allow_used=False,
            )
            if tid is None:
                tid = self._nearest_truck_reachable_task(
                    env,
                    str(aid),
                    used_tasks=used_tasks,
                    allow_used=True,
                )

            if tid is None:
                # Hard anti-idle fallback: if any reachable/serviceable NORMAL task exists,
                # force nearest NORMAL assignment to avoid truck wandering with empty goal.
                best_norm_tid: Optional[str] = None
                best_norm_d = float("inf")
                for task in env.state.tasks.values():
                    if task.status != TaskStatus.PENDING or task.kind != TaskKind.NORMAL:
                        continue
                    if not self._truck_task_serviceable_or_support_proxy(env, str(aid), task):
                        continue
                    if not self._truck_task_reachable(env, str(aid), task):
                        continue
                    d = float(self._truck_task_distance(env, str(aid), task))
                    if np.isfinite(d) and d + 1e-9 < best_norm_d:
                        best_norm_d = float(d)
                        best_norm_tid = str(task.task_id)
                if best_norm_tid is not None:
                    tid = str(best_norm_tid)

            if tid is not None:
                goals[aid] = str(tid)
                used_tasks.add(str(tid))

        # UAV emergency reservation pass:
        # ensure idle/prelaunch UAV gets a stable emergency reservation before launch.
        if coordination_active:
            self._ensure_uav_emergency_reservations(env, goals, used_tasks)
            self._apply_large_map_tc_coverage_intents(env, goals, used_tasks)

        # Truck assist waypoint evaluation (keeps normal goal; records assist intent).
        if coordination_active:
            self._update_truck_uav_assist_waypoints(env, goals)

        # Apply hysteresis/hold logic and finalize used-task exclusivity.
        final_goals: Dict[str, Optional[str]] = {}
        final_used: set = set()
        for aid in ordered_agents:
            proposed = goals.get(aid, None)
            cur_goal = self.state.goals.get(str(aid), None)
            chosen = self._apply_switch_hysteresis(env, aid, proposed_goal=proposed, used_tasks=final_used)
            final_goals[aid] = chosen
            # Switch-decision ledger (diagnostics only).
            ag_any = env.state.agents.get(str(aid), None)
            is_candidate_all = bool(cur_goal is not None and proposed is not None and str(proposed) != str(cur_goal))
            meta_all = self._last_switch_decision_by_aid.get(str(aid), {})
            decision_all = str(meta_all.get("decision", "none"))
            reason_all = str(meta_all.get("reason", ""))
            forced_all = int(is_candidate_all and str(chosen) != str(cur_goal) and decision_all == "forced")
            accepted_all = int(is_candidate_all and str(chosen) != str(cur_goal))
            rejected_all = int(is_candidate_all and str(chosen) == str(cur_goal))

            cur_task_all = env.state.tasks.get(str(cur_goal), None) if cur_goal is not None else None
            prop_task_all = env.state.tasks.get(str(proposed), None) if proposed is not None else None
            cur_score_all = float(self._score_goal_for_agent(env, str(aid), cur_goal)) if cur_goal is not None else float("nan")
            prop_score_all = float(self._score_goal_for_agent(env, str(aid), proposed)) if proposed is not None else float("nan")
            score_delta_all = float(prop_score_all - cur_score_all) if np.isfinite(cur_score_all) and np.isfinite(prop_score_all) else float("nan")
            if ag_any is not None and ag_any.kind == AgentKind.UAV:
                score_thr_all = float(max(getattr(env.cfg, "hrl_uav_goal_switch_margin", self.switch_margin), 0.0))
            else:
                score_thr_all = float(max(getattr(env.cfg, "hrl_truck_goal_switch_margin", self.switch_margin), 0.0))
            cur_eta_all = float(self._switch_goal_eta(env, str(aid), cur_goal))
            prop_eta_all = float(self._switch_goal_eta(env, str(aid), proposed))
            eta_gain_all = float(cur_eta_all - prop_eta_all) if np.isfinite(cur_eta_all) and np.isfinite(prop_eta_all) else float("nan")
            eta_thr_all = float(max(getattr(env.cfg, "goal_switch_eta_gain_min", 0.0), 0.0))
            cur_dist_all = float(self._switch_goal_distance(env, str(aid), cur_goal))
            cur_prog3_all = float(self._switch_goal_progress_recent(env, str(aid), cur_goal, 3))
            cur_prog5_all = float(self._switch_goal_progress_recent(env, str(aid), cur_goal, 5))
            assigned_step_all = int(self.state.goal_assigned_step.get(str(aid), int(env.state.step_index)))
            recency_all = int(int(env.state.step_index) - int(assigned_step_all))
            if cur_task_all is not None and ag_any is not None and ag_any.kind == AgentKind.UAV:
                cur_feas_all = int(self._uav_task_feasible(env, str(aid), cur_task_all))
            elif cur_task_all is not None and ag_any is not None and ag_any.kind == AgentKind.TRUCK:
                cur_feas_all = int(self._truck_task_valid(env, str(aid), str(cur_goal)))
            else:
                cur_feas_all = 1 if cur_goal is not None else 0
            cur_path_blocked_all = 0
            cur_unreachable_all = 0
            if cur_task_all is not None and ag_any is not None and ag_any.kind == AgentKind.TRUCK:
                tr_reach = bool(self._truck_task_reachable(env, str(aid), cur_task_all))
                cur_unreachable_all = int(not tr_reach)
                cur_path_blocked_all = int((not tr_reach) or float(self._task_shared_map_block_pressure(env, cur_task_all)) >= 0.50)
            near_radius_th = float(max(getattr(env.cfg, "uav_delivery_radius_m", 40.0), 1.0))
            near_service_all = int(np.isfinite(cur_dist_all) and cur_dist_all <= near_radius_th)
            service_started_all = int(cur_task_all is not None and cur_task_all.first_service_step is not None)
            event_reason_all = self._switch_event_reason_compact()
            hard_reason_all = self._switch_hard_event_reason_compact()
            uav_current_launchable_all = 0
            uav_current_airborne_all = 0
            uav_current_loaded_all = 0
            uav_current_recovery_feasible_all = 0
            uav_current_reject_reason_all = ""
            proposed_uav_launchable_all = 0
            proposed_uav_recovery_feasible_all = 0
            proposed_uav_reject_reason_all = ""
            if ag_any is not None and ag_any.kind == AgentKind.UAV:
                uav_current_airborne_all = int(getattr(ag_any, "follow_target", None) is None)
                uav_current_loaded_all = int(bool(getattr(ag_any, "loaded", False)))
                if cur_task_all is not None:
                    uav_current_recovery_feasible_all = int(self._uav_task_feasible(env, str(aid), cur_task_all))
                    docked_fn = getattr(env, "_uav_docked_task_actionable_now", None)
                    if callable(docked_fn):
                        try:
                            uav_current_launchable_all = int(bool(docked_fn(str(aid), cur_task_all)))
                        except Exception:
                            uav_current_launchable_all = int(uav_current_recovery_feasible_all)
                    else:
                        uav_current_launchable_all = int(uav_current_recovery_feasible_all)
                if prop_task_all is not None:
                    proposed_uav_recovery_feasible_all = int(self._uav_task_feasible(env, str(aid), prop_task_all))
                    docked_fn = getattr(env, "_uav_docked_task_actionable_now", None)
                    if callable(docked_fn):
                        try:
                            proposed_uav_launchable_all = int(bool(docked_fn(str(aid), prop_task_all)))
                        except Exception:
                            proposed_uav_launchable_all = int(proposed_uav_recovery_feasible_all)
                    else:
                        proposed_uav_launchable_all = int(proposed_uav_recovery_feasible_all)
                uav_current_reject_reason_all = str(getattr(env, "_uav_last_launch_reason", {}).get(str(aid), ""))
                proposed_uav_reject_reason_all = uav_current_reject_reason_all
            truck_path_exists_all = 0
            truck_path_rem_all = float("nan")
            truck_path_blocked_all = 0
            truck_local_repair_success_all = 0
            best_alt_eta_all = float("nan")
            best_alt_tid_all = ""
            if ag_any is not None and ag_any.kind == AgentKind.TRUCK:
                truck_path_exists_all = int(cur_feas_all > 0)
                truck_path_rem_all = float(cur_eta_all) if np.isfinite(cur_eta_all) else float("nan")
                truck_path_blocked_all = int(cur_path_blocked_all)
                if cur_task_all is not None:
                    best_d = float("inf")
                    best_tid = ""
                    for _t in env.state.tasks.values():
                        if _t.status != TaskStatus.PENDING or _t.kind != TaskKind.NORMAL:
                            continue
                        if str(_t.task_id) == str(cur_task_all.task_id):
                            continue
                        if not self._truck_task_reachable(env, str(aid), _t):
                            continue
                        dd = float(self._truck_task_distance(env, str(aid), _t))
                        if np.isfinite(dd) and dd < best_d:
                            best_d = dd
                            best_tid = str(_t.task_id)
                    if np.isfinite(best_d):
                        sp = float(max(getattr(ag_any, "speed", 0.0), 1e-6))
                        dt = float(max(getattr(env.cfg, "dt_seconds", 1.0), 1e-6))
                        best_alt_eta_all = float(best_d / max(sp * dt, 1e-6))
                        best_alt_tid_all = best_tid

            if cur_goal is not None or proposed is not None:
                self._switch_decision_seq = int(self._switch_decision_seq) + 1
                row_audit: Dict[str, Any] = {
                    "decision_id": int(self._switch_decision_seq),
                    "step": int(env.state.step_index),
                    "agent_id": str(aid),
                    "agent_type": "uav" if (ag_any is not None and ag_any.kind == AgentKind.UAV) else ("truck" if (ag_any is not None and ag_any.kind == AgentKind.TRUCK) else "unknown"),
                    "current_goal_type": self._switch_goal_type_name(env, cur_goal),
                    "current_goal_id": "" if cur_goal is None else str(cur_goal),
                    "proposed_goal_type": self._switch_goal_type_name(env, proposed),
                    "proposed_goal_id": "" if proposed is None else str(proposed),
                    "current_task_kind": self._switch_task_kind_name(cur_task_all),
                    "proposed_task_kind": self._switch_task_kind_name(prop_task_all),
                    "current_goal_status": self._switch_goal_status_name(env, cur_goal),
                    "proposed_goal_status": self._switch_goal_status_name(env, proposed),
                    "switch_candidate_count": int(1 if is_candidate_all else 0),
                    "switch_accepted": int(accepted_all),
                    "switch_rejected": int(rejected_all),
                    "switch_forced": int(forced_all),
                    "switch_reason": str(reason_all),
                    "forced_reason": str(reason_all if forced_all else ""),
                    "event_reason": str(event_reason_all),
                    "hard_event_reason": str(hard_reason_all),
                    "current_score": float(cur_score_all),
                    "proposed_score": float(prop_score_all),
                    "score_delta": float(score_delta_all),
                    "switch_score_threshold": float(score_thr_all),
                    "current_eta": float(cur_eta_all),
                    "proposed_eta": float(prop_eta_all),
                    "eta_gain": float(eta_gain_all),
                    "switch_eta_threshold": float(eta_thr_all),
                    "current_goal_distance": float(cur_dist_all),
                    "current_goal_eta": float(cur_eta_all),
                    "current_goal_feasible": int(cur_feas_all),
                    "proposed_goal_feasible": int(
                        1
                        if (
                            (prop_task_all is not None and ((ag_any is not None and ag_any.kind == AgentKind.UAV and self._uav_task_feasible(env, str(aid), prop_task_all)) or (ag_any is not None and ag_any.kind == AgentKind.TRUCK and self._truck_task_valid(env, str(aid), str(proposed)))))
                            or (proposed is None)
                        )
                        else 0
                    ),
                    "current_goal_path_blocked": int(cur_path_blocked_all),
                    "current_goal_unreachable": int(cur_unreachable_all),
                    "current_goal_progress_last_3_steps": float(cur_prog3_all),
                    "current_goal_progress_last_5_steps": float(cur_prog5_all),
                    "current_goal_service_started": int(service_started_all),
                    "current_goal_near_service_radius": int(near_service_all),
                    "current_goal_recently_assigned_steps": int(recency_all),
                    "uav_current_launchable": int(uav_current_launchable_all),
                    "uav_current_airborne": int(uav_current_airborne_all),
                    "uav_current_loaded": int(uav_current_loaded_all),
                    "uav_current_recovery_feasible": int(uav_current_recovery_feasible_all),
                    "uav_current_reject_reason": str(uav_current_reject_reason_all),
                    "proposed_uav_launchable": int(proposed_uav_launchable_all),
                    "proposed_uav_recovery_feasible": int(proposed_uav_recovery_feasible_all),
                    "proposed_uav_reject_reason": str(proposed_uav_reject_reason_all),
                    "truck_current_path_exists": int(truck_path_exists_all),
                    "truck_current_path_remaining_len": float(truck_path_rem_all),
                    "truck_current_path_blocked": int(truck_path_blocked_all),
                    "truck_local_repair_success": int(truck_local_repair_success_all),
                    "truck_local_repair_eta": float("nan"),
                    "best_alternative_routine_eta": float(best_alt_eta_all),
                    "best_alternative_routine_task_id": str(best_alt_tid_all),
                    "outcome_window_steps": 10,
                    "after_switch_distance_progress": float("nan"),
                    "after_switch_service_started": 0,
                    "after_switch_task_completed": 0,
                    "after_switch_uav_launch": 0,
                    "after_switch_uav_delivery": 0,
                    "after_switch_reject_count": 0,
                    "after_switch_stall_count": 0,
                    "after_switch_goal_changed_again": 0,
                    "after_switch_same_goal_retained_steps": 0,
                    "harmful_switch": 0,
                    "missed_switch": 0,
                }
                self._switch_decision_pending_windows.append(
                    {
                        **row_audit,
                        "settled_goal_id": str(chosen) if chosen is not None else (str(cur_goal) if cur_goal is not None else None),
                        "_launch_base": int(getattr(env, "uav_launch_count_total", 0)),
                        "_delivery_base": int(getattr(env, "uav_delivery_count_total", 0)),
                        "_reject_base": int(self._switch_total_uav_reject_count(env)),
                        "_start_dist": float(self._switch_goal_distance(env, str(aid), str(chosen) if chosen is not None else cur_goal)),
                        "_prev_dist": float(self._switch_goal_distance(env, str(aid), str(chosen) if chosen is not None else cur_goal)),
                    }
                )
            ag = env.state.agents.get(str(aid), None)
            if ag is not None and ag.kind == AgentKind.UAV:
                is_candidate = bool(cur_goal is not None and proposed is not None and str(proposed) != str(cur_goal))
                if is_candidate:
                    self.goal_switch_candidate_count_total += 1
                    if str(chosen) != str(cur_goal):
                        self.goal_switch_accepted_count_total += 1
                        meta = self._last_switch_decision_by_aid.get(str(aid), {})
                        decision = str(meta.get("decision", ""))
                        reason = str(meta.get("reason", ""))
                        if decision == "forced":
                            self.goal_switch_forced_count_total += 1
                            if reason == "completed":
                                self.goal_switch_forced_reason_completed_total += 1
                            elif reason == "failed":
                                self.goal_switch_forced_reason_failed_total += 1
                            elif reason == "dead_end":
                                self.goal_switch_forced_reason_dead_end_total += 1
                            elif reason == "uav_recovery":
                                self.goal_switch_forced_reason_uav_recovery_total += 1
                            elif reason == "stall":
                                self.goal_switch_forced_reason_stall_total += 1
                            else:
                                self.goal_switch_forced_reason_infeasible_total += 1
                        else:
                            if reason == "eta":
                                self.goal_switch_accepted_by_eta_count_total += 1
                            else:
                                self.goal_switch_accepted_by_score_count_total += 1
                    else:
                        self.goal_switch_rejected_by_threshold_count_total += 1
            t = env.state.tasks.get(str(chosen), None)
            if t is not None and t.status == TaskStatus.PENDING:
                final_used.add(str(t.task_id))

        if coordination_active:
            self._repair_stalled_routine_goal_ownership(env, final_goals, final_used)
            self._repair_unassigned_reachable_routine(env, final_goals, final_used)
            self._apply_large_map_greedy_tc_fallback(env, final_goals, final_used)
        repaired = self._repair_goals(env, final_goals)
        repaired = self._apply_task_exclusive_contracts(env, repaired)
        if coordination_active:
            self._refresh_uav_task_reservations(env, repaired)
            self._refresh_uav_anchor_tasks(env, repaired)
            self._refresh_uav_transfer_hints(env, repaired)
            self._update_uav_reservation_states(env, repaired)

        # Round4 diagnostics: unique candidate denominator per decision step.
        self.unique_agent_task_candidate_count_total = int(self.unique_agent_task_candidate_count_total) + int(len(self._step_unique_agent_task_keys))

        # Service-anchor conversion accounting + short commitment windows.
        hold_steps = int(max(getattr(env.cfg, "hrl_support_anchor_hold_steps", 8), 0))
        chain_steps = int(max(getattr(env.cfg, "hrl_relaxed_chain_commitment_steps", 6), 0))
        min_bound = int(max(getattr(env.cfg, "hrl_relaxed_chain_min_bound_tasks", 1), 0))
        support_selected_step = 0
        support_gain_step = 0
        support_no_gain_step = 0
        support_selected_with_bound_timecritical_step = 0
        support_selected_without_bound_timecritical_step = 0
        support_selected_with_bound_bulk_step = 0

        for aid, gid in repaired.items():
            st = env.state.agents.get(str(aid), None)
            if st is not None and st.kind == AgentKind.UAV and gid is not None:
                _sel_task = env.state.tasks.get(str(gid), None)
                if _sel_task is not None and _sel_task.status == TaskStatus.PENDING and _sel_task.kind == TaskKind.EMERGENCY:
                    self.uav_task_selected_count_total = int(self.uav_task_selected_count_total) + 1
            if st is None or gid is None:
                continue
            task = env.state.tasks.get(str(gid), None)
            if task is None or task.status != TaskStatus.PENDING:
                continue
            if coordination_active and st.kind == AgentKind.TRUCK and task.kind == TaskKind.EMERGENCY:
                gain_info = self._support_anchor_service_gain(env, str(aid), task)
                gain = float(gain_info.get("gain_score", 0.0))
                is_support_candidate = bool(self._truck_emergency_support_candidate(env, str(aid), task))
                if is_support_candidate:
                    support_selected_step += 1
                    bind_info = self._support_bound_delivery_info(env, str(aid), task, gain_info=gain_info)
                    bound_uid = str(bind_info.get("bound_timecritical_uav_id", "")).strip()
                    if float(bind_info.get("bound_timecritical", 0.0)) > 0.0:
                        support_selected_with_bound_timecritical_step += 1
                    else:
                        support_selected_without_bound_timecritical_step += 1
                    if float(bind_info.get("bound_bulk", 0.0)) > 0.0:
                        support_selected_with_bound_bulk_step += 1
                    if gain > 1e-9:
                        support_gain_step += 1
                        self._support_anchor_until_step[str(aid)] = int(step_now + hold_steps)
                        self._support_anchor_gain[str(aid)] = float(gain)
                        self._support_anchor_task_id[str(aid)] = str(task.task_id)
                        if bound_uid and bool(self._support_binding_is_strong_enough(env, task, bind_info, gain_info=gain_info)):
                            existing_chain = self._rc_locked_support_chain_for_truck(env, str(aid))
                            preserve_existing = False
                            if existing_chain is not None:
                                existing_task = existing_chain.get("task", None)
                                existing_tid = str(existing_chain.get("task_id", "")).strip()
                                if existing_tid and existing_tid != str(task.task_id):
                                    preserve_existing = not self._rc_should_override_locked_support_chain(env, existing_task, task, aid=str(aid))
                            if not preserve_existing:
                                chain_until = int(step_now + max(hold_steps, chain_steps))
                                self._support_bound_chain_until_step[str(aid)] = int(chain_until)
                                self._support_bound_chain_task_id[str(aid)] = str(task.task_id)
                                self._support_bound_chain_uav_id[str(aid)] = str(bound_uid)
                                self._support_bound_chain_truck_by_uav[str(bound_uid)] = str(aid)
                    else:
                        support_no_gain_step += 1
                    self._update_support_backoff_after_selection(env, str(aid), float(gain))
            if coordination_active and st.kind == AgentKind.UAV and task.kind == TaskKind.EMERGENCY:
                launch_reason = str(getattr(env, "_uav_last_launch_reason", {}).get(str(aid), ""))
                if launch_reason.startswith("rendezvous_safe_relaxed"):
                    bound = int(max(min_bound, 1))
                    self._relaxed_chain_until_step[str(aid)] = int(step_now + chain_steps)
                    if bound <= 1:
                        pass

        self.truck_support_selected_count_total = int(self.truck_support_selected_count_total) + int(support_selected_step)
        self.truck_support_improves_serviceability_count_total = int(self.truck_support_improves_serviceability_count_total) + int(support_gain_step)
        self.truck_support_no_gain_count_total = int(self.truck_support_no_gain_count_total) + int(support_no_gain_step)
        self.support_selected_count_total = int(self.support_selected_count_total) + int(support_selected_step)
        self.support_improves_serviceability_count_total = int(self.support_improves_serviceability_count_total) + int(support_gain_step)
        self.support_no_gain_count_total = int(self.support_no_gain_count_total) + int(support_no_gain_step)
        self.support_selected_with_bound_timecritical_delivery_count_total = int(self.support_selected_with_bound_timecritical_delivery_count_total) + int(support_selected_with_bound_timecritical_step)
        self.support_selected_without_bound_timecritical_delivery_count_total = int(self.support_selected_without_bound_timecritical_delivery_count_total) + int(support_selected_without_bound_timecritical_step)
        self.support_selected_with_bound_bulk_delivery_count_total = int(self.support_selected_with_bound_bulk_delivery_count_total) + int(support_selected_with_bound_bulk_step)

        truck_emergency_goal_step = 0
        for _aid, _gid in repaired.items():
            _st = env.state.agents.get(str(_aid), None)
            if _st is None or _st.kind != AgentKind.TRUCK:
                continue
            if _gid is None:
                continue
            _task = env.state.tasks.get(str(_gid), None)
            if _task is not None and _task.status == TaskStatus.PENDING and _task.kind == TaskKind.EMERGENCY:
                truck_emergency_goal_step += 1
        self.truck_emergency_goal_assigned_count_total = int(self.truck_emergency_goal_assigned_count_total) + int(truck_emergency_goal_step)

        return repaired

    def plan(self, env) -> Dict[str, Optional[str]]:
        self._episode_reset_if_needed(env)
        # Update audit windows/histories using latest post-step environment state.
        self._update_progress_aware_pending(env)
        self._switch_update_outcome_windows(env)
        self._switch_update_goal_distance_history(env)
        self._update_uav_docked_steps(env)
        self._update_truck_assist_outcomes(env, force_finalize=False)
        did_refresh = bool(self._should_refresh(env))
        if did_refresh:
            # Close previous event-refresh window using the progress observed up to this step.
            self._finalize_active_event_refresh_window(env, int(env.state.step_index))
            old_goals = dict(self.state.goals)
            new_goals = self._plan_once(env)
            if self._commitment_local_correction_mode(env):
                self.erc_global_replan_count_total = int(self.erc_global_replan_count_total) + 1
            since_last = int(max(int(env.state.step_index) - int(self.state.step_last_refresh), 0))
            by_interval = bool(self._last_refresh_flags.get("interval", False))
            by_fallback = bool(self._last_refresh_flags.get("no_event_fallback_refresh", False))
            by_hard_event = bool(self._last_refresh_flags.get("hard_event_refresh", False))
            by_low_event = bool(
                bool(self._last_refresh_flags.get("low_value_event", False))
                and (not bool(self._last_refresh_flags.get("event_budget_blocked", False)))
            )
            empty_goals = bool(self._last_refresh_flags.get("empty_goals", False))
            self.refresh_total_count = int(self.refresh_total_count) + 1
            self.steps_since_last_refresh_sum = int(self.steps_since_last_refresh_sum) + int(since_last)
            self.steps_since_last_refresh_max = int(max(int(self.steps_since_last_refresh_max), int(since_last)))
            if by_interval:
                self.fixed_interval_refresh_count = int(self.fixed_interval_refresh_count) + 1
            if by_fallback:
                self.no_event_fallback_refresh_count = int(self.no_event_fallback_refresh_count) + 1
            if by_hard_event or by_low_event:
                self.event_refresh_count = int(self.event_refresh_count) + 1
            if empty_goals:
                self.empty_goal_refresh_count = int(self.empty_goal_refresh_count) + 1
            if not bool(self._episode_first_refresh_done):
                self.initial_refresh_count = int(self.initial_refresh_count) + 1
                self._episode_first_refresh_done = True

            changed = 0
            any_goal_changed = False
            if self._commitment_local_correction_mode(env):
                hard_break_reason = bool(
                    bool(self._last_refresh_flags.get("hard_reason_goal_invalid", False))
                    or bool(self._last_refresh_flags.get("hard_reason_goal_unreachable", False))
                    or bool(self._last_refresh_flags.get("hard_reason_path_blocked", False))
                    or bool(self._last_refresh_flags.get("hard_reason_uav_safety", False))
                )
                stall_break_reason = bool(
                    bool(self._last_refresh_flags.get("hard_reason_normal_stall", False))
                    or bool(self._last_refresh_flags.get("hard_reason_assigned_but_not_progressing", False))
                )
            for aid in set(old_goals.keys()) | set(new_goals.keys()):
                if old_goals.get(aid, None) != new_goals.get(aid, None):
                    any_goal_changed = True
                    if self._commitment_local_correction_mode(env) and old_goals.get(aid, None) is not None:
                        self.committed_goal_broken_count_total = int(self.committed_goal_broken_count_total) + 1
                        if hard_break_reason:
                            self.committed_goal_broken_reason_hard_invalid_count_total = int(
                                self.committed_goal_broken_reason_hard_invalid_count_total
                            ) + 1
                        elif stall_break_reason:
                            self.committed_goal_broken_reason_stall_count_total = int(
                                self.committed_goal_broken_reason_stall_count_total
                            ) + 1
                        else:
                            self.committed_goal_broken_reason_tc_gain_count_total = int(
                                self.committed_goal_broken_reason_tc_gain_count_total
                            ) + 1
                else:
                    if self._commitment_local_correction_mode(env) and old_goals.get(aid, None) is not None:
                        self.committed_goal_hold_count_total = int(self.committed_goal_hold_count_total) + 1
                ag = env.state.agents.get(str(aid), None)
                if ag is None or ag.kind != AgentKind.UAV:
                    continue
                if old_goals.get(aid, None) != new_goals.get(aid, None):
                    changed += 1
            # Keep first assignment out of switch statistics.
            if old_goals:
                self.goal_switch_count_total += int(changed)

            # Event refresh value diagnostics + weak-event no-op cooldown window.
            if by_hard_event or by_low_event:
                self._record_event_refresh_reason_counts(self._last_refresh_flags)
                hard_reasons: List[str] = []
                hard_offenders: List[Dict[str, object]] = []
                if by_hard_event:
                    self.hard_event_refresh_count_total = int(self.hard_event_refresh_count_total) + 1
                    if bool(self._last_refresh_flags.get("hard_reason_goal_invalid", False)):
                        self.hard_event_reason_goal_invalid_count_total = int(self.hard_event_reason_goal_invalid_count_total) + 1
                        hard_reasons.append("goal_invalid")
                        rec = self._last_goal_invalid_record if isinstance(self._last_goal_invalid_record, dict) else {}
                        rr = str(rec.get("reason", "")).strip().lower()
                        if rr == "task_completed":
                            self.goal_invalid_reason_task_completed_total = int(self.goal_invalid_reason_task_completed_total) + 1
                        elif rr == "task_failed":
                            self.goal_invalid_reason_task_failed_total = int(self.goal_invalid_reason_task_failed_total) + 1
                        elif rr == "task_missing":
                            self.goal_invalid_reason_task_missing_total = int(self.goal_invalid_reason_task_missing_total) + 1
                        elif rr == "truck_unreachable":
                            self.goal_invalid_reason_truck_unreachable_total = int(self.goal_invalid_reason_truck_unreachable_total) + 1
                        elif rr == "uav_energy_infeasible":
                            self.goal_invalid_reason_uav_energy_infeasible_total = int(self.goal_invalid_reason_uav_energy_infeasible_total) + 1
                        elif rr == "uav_recovery_margin":
                            self.goal_invalid_reason_uav_recovery_margin_total = int(self.goal_invalid_reason_uav_recovery_margin_total) + 1
                        elif rr == "uav_corridor":
                            self.goal_invalid_reason_uav_corridor_total = int(self.goal_invalid_reason_uav_corridor_total) + 1
                        elif rr == "uav_comm_block":
                            self.goal_invalid_reason_uav_comm_block_total = int(self.goal_invalid_reason_uav_comm_block_total) + 1
                        elif rr == "uav_not_loaded":
                            self.goal_invalid_reason_uav_not_loaded_total = int(self.goal_invalid_reason_uav_not_loaded_total) + 1
                        elif rr == "uav_not_docked":
                            self.goal_invalid_reason_uav_not_docked_total = int(self.goal_invalid_reason_uav_not_docked_total) + 1
                        elif rr == "uav_soft_reject_cache":
                            self.goal_invalid_reason_soft_reject_cache_total = int(self.goal_invalid_reason_soft_reject_cache_total) + 1
                        if bool(rec.get("suspect_soft_as_hard", False)):
                            self.suspect_soft_as_hard_count_total = int(self.suspect_soft_as_hard_count_total) + 1
                        for off in list(rec.get("offenders", [])) if isinstance(rec, dict) else []:
                            if isinstance(off, dict):
                                hard_offenders.append(dict(off))
                    if bool(self._last_refresh_flags.get("hard_reason_goal_unreachable", False)):
                        self.hard_event_reason_current_goal_unreachable_count_total = int(
                            self.hard_event_reason_current_goal_unreachable_count_total
                        ) + 1
                        hard_reasons.append("current_goal_unreachable")
                        hard_offenders.append({
                            "reason": "current_goal_unreachable",
                            "agent_id": "",
                            "task_id": "",
                            "step": int(env.state.step_index),
                            "current_goal_type": "unknown",
                            "proposed_goal_type": "unknown",
                            "task_status": "unknown",
                            "battery": float("nan"),
                            "distance_to_goal": float("nan"),
                        })
                    if bool(self._last_refresh_flags.get("hard_reason_path_blocked", False)):
                        self.hard_event_reason_path_blocked_count_total = int(self.hard_event_reason_path_blocked_count_total) + 1
                        hard_reasons.append("path_blocked")
                        hard_offenders.append({
                            "reason": "path_blocked",
                            "agent_id": "",
                            "task_id": "",
                            "step": int(env.state.step_index),
                            "current_goal_type": "unknown",
                            "proposed_goal_type": "unknown",
                            "task_status": "unknown",
                            "battery": float("nan"),
                            "distance_to_goal": float("nan"),
                        })
                    if bool(self._last_refresh_flags.get("hard_reason_uav_safety", False)):
                        rec_u = self._last_uav_emergency_record if isinstance(self._last_uav_emergency_record, dict) else {}
                        if str(rec_u.get("reason", "")) == "uav_recovery":
                            self.hard_event_reason_uav_recovery_count_total = int(self.hard_event_reason_uav_recovery_count_total) + 1
                            hard_reasons.append("uav_recovery")
                        else:
                            self.hard_event_reason_uav_safety_count_total = int(self.hard_event_reason_uav_safety_count_total) + 1
                            hard_reasons.append("uav_safety")
                        for off in list(rec_u.get("offenders", [])) if isinstance(rec_u, dict) else []:
                            if isinstance(off, dict):
                                hard_offenders.append(dict(off))
                    if bool(self._last_refresh_flags.get("hard_reason_truck_dead_end", False)):
                        self.hard_event_reason_truck_dead_end_count_total = int(self.hard_event_reason_truck_dead_end_count_total) + 1
                        hard_reasons.append("truck_dead_end")
                        rec_t = self._last_truck_dead_end_record if isinstance(self._last_truck_dead_end_record, dict) else {}
                        for off in list(rec_t.get("offenders", [])) if isinstance(rec_t, dict) else []:
                            if isinstance(off, dict):
                                hard_offenders.append(dict(off))
                    if bool(self._last_refresh_flags.get("hard_reason_high_priority_uncovered", False)):
                        self.hard_event_reason_high_priority_uncovered_count_total = int(
                            self.hard_event_reason_high_priority_uncovered_count_total
                        ) + 1
                        hard_reasons.append("high_priority_uncovered")
                        rec_h = self._last_high_priority_uncovered_record if isinstance(self._last_high_priority_uncovered_record, dict) else {}
                        for off in list(rec_h.get("offenders", [])) if isinstance(rec_h, dict) else []:
                            if isinstance(off, dict):
                                hard_offenders.append(dict(off))
                    if bool(self._last_refresh_flags.get("hard_reason_normal_stall", False)):
                        self.hard_event_reason_normal_stall_count_total = int(self.hard_event_reason_normal_stall_count_total) + 1
                        hard_reasons.append("normal_stall")
                        rec_n = self._last_normal_stall_record if isinstance(self._last_normal_stall_record, dict) else {}
                        for off in list(rec_n.get("offenders", [])) if isinstance(rec_n, dict) else []:
                            if isinstance(off, dict):
                                hard_offenders.append(dict(off))
                    if bool(self._last_refresh_flags.get("hard_reason_assigned_but_not_progressing", False)):
                        self.hard_event_reason_assigned_but_not_progressing_count_total = int(
                            self.hard_event_reason_assigned_but_not_progressing_count_total
                        ) + 1
                        hard_reasons.append("assigned_but_not_progressing")
                    if str(getattr(self, "_last_goal_terminal_status", "")) == "completed":
                        self.hard_event_reason_goal_completed_count_total = int(self.hard_event_reason_goal_completed_count_total) + 1
                        hard_reasons.append("goal_completed")
                    elif str(getattr(self, "_last_goal_terminal_status", "")) == "failed":
                        self.hard_event_reason_goal_failed_count_total = int(self.hard_event_reason_goal_failed_count_total) + 1
                        hard_reasons.append("goal_failed")
                    self._last_hard_event_offenders = list(hard_offenders)
                else:
                    self._last_hard_event_offenders = []
                if any_goal_changed:
                    self.event_refresh_goal_change_count_total = int(self.event_refresh_goal_change_count_total) + 1
                    if by_hard_event:
                        self.hard_event_refresh_goal_change_count_total = int(self.hard_event_refresh_goal_change_count_total) + 1
                else:
                    self.event_refresh_no_goal_change_count_total = int(self.event_refresh_no_goal_change_count_total) + 1
                    if by_hard_event:
                        self.hard_event_refresh_no_goal_change_count_total = int(self.hard_event_refresh_no_goal_change_count_total) + 1
                    if bool(self._last_refresh_flags.get("hard_reason_path_blocked", False)):
                        self.path_blocked_global_refresh_no_goal_change_count_total = int(
                            self.path_blocked_global_refresh_no_goal_change_count_total
                        ) + 1
                    if bool(self._last_refresh_flags.get("hard_reason_truck_dead_end", False)):
                        self.truck_dead_end_global_refresh_no_goal_change_count_total = int(
                            self.truck_dead_end_global_refresh_no_goal_change_count_total
                        ) + 1
                weak_reasons: List[str] = []
                weak_reason_flag_map = {
                    "arrival": "weak_reason_arrival",
                    "resolution": "weak_reason_resolution",
                    "uav_idle": "weak_reason_uav_idle",
                    "truck_idle": "weak_reason_truck_idle",
                    "map_update_light": "weak_reason_map_update_light",
                    "ranking_changed": "weak_reason_ranking_changed",
                    "noncritical_map_update": "weak_reason_noncritical_map_update",
                }
                for rname, fname in weak_reason_flag_map.items():
                    if bool(self._last_refresh_flags.get(str(fname), False)):
                        weak_reasons.append(str(rname))
                self._start_event_refresh_window(
                    env,
                    int(env.state.step_index),
                    weak_reasons=weak_reasons,
                    no_goal_change=bool(not any_goal_changed),
                    is_hard=bool(by_hard_event),
                    hard_reasons=hard_reasons,
                    hard_offenders=hard_offenders,
                )
                if isinstance(self._active_event_refresh_window, dict):
                    self._active_event_refresh_window["goal_switch_after_event"] = float(changed)

            assigned_step_next: Dict[str, int] = {}
            now_step = int(env.state.step_index)

            # Record NORMAL->NORMAL switches so short-horizon bounce-back can be blocked.
            for aid in set(old_goals.keys()) | set(new_goals.keys()):
                ag = env.state.agents.get(str(aid), None)
                if ag is None or ag.kind != AgentKind.TRUCK:
                    continue
                old_gid = old_goals.get(aid, None)
                new_gid = new_goals.get(aid, None)
                if old_gid is None or new_gid is None or str(old_gid) == str(new_gid):
                    continue
                old_task = env.state.tasks.get(str(old_gid), None)
                new_task = env.state.tasks.get(str(new_gid), None)
                if (
                    old_task is not None
                    and new_task is not None
                    and old_task.kind == TaskKind.NORMAL
                    and new_task.kind == TaskKind.NORMAL
                ):
                    self._truck_recent_normal_prev_goal[str(aid)] = str(old_task.task_id)
                    self._truck_recent_normal_switch_step[str(aid)] = int(now_step)

            # Record all task-to-task switches for the generic same-kind
            # A->B->A guard (including UAV emergency-task bounce-back).
            for aid in set(old_goals.keys()) | set(new_goals.keys()):
                old_gid = old_goals.get(aid, None)
                new_gid = new_goals.get(aid, None)
                if old_gid is None or new_gid is None or str(old_gid) == str(new_gid):
                    continue
                old_task = env.state.tasks.get(str(old_gid), None)
                new_task = env.state.tasks.get(str(new_gid), None)
                if old_task is not None and new_task is not None:
                    self._task_recent_prev_goal[str(aid)] = str(old_task.task_id)
                    self._task_recent_switch_step[str(aid)] = int(now_step)

            # Record UAV truck-anchor switches to block short-horizon A<->B ping-pong.
            for aid in set(old_goals.keys()) | set(new_goals.keys()):
                ag = env.state.agents.get(str(aid), None)
                if ag is None or ag.kind != AgentKind.UAV:
                    continue
                old_gid = old_goals.get(aid, None)
                new_gid = new_goals.get(aid, None)
                if old_gid is None or new_gid is None or str(old_gid) == str(new_gid):
                    continue
                old_anchor = env.state.agents.get(str(old_gid), None)
                new_anchor = env.state.agents.get(str(new_gid), None)
                if (
                    old_anchor is not None
                    and new_anchor is not None
                    and old_anchor.kind == AgentKind.TRUCK
                    and new_anchor.kind == AgentKind.TRUCK
                ):
                    self._uav_recent_truck_anchor_prev_goal[str(aid)] = str(old_gid)
                    self._uav_recent_truck_anchor_switch_step[str(aid)] = int(now_step)

            for aid, gid in new_goals.items():
                if old_goals.get(aid, None) == gid:
                    assigned_step_next[aid] = int(self.state.goal_assigned_step.get(aid, now_step))
                else:
                    assigned_step_next[aid] = now_step

            self.state.goals = new_goals
            self.state.step_last_refresh = int(env.state.step_index)
            self.state.resolved_tasks_last = self._resolved_count(env)
            self.state.goal_assigned_step = assigned_step_next
            # Snapshot committed goals and their lightweight execution context.
            step_now_commit = int(env.state.step_index)
            self.committed_goal = dict(new_goals)
            self.committed_goal_step = {str(a): int(assigned_step_next.get(a, step_now_commit)) for a in new_goals.keys()}
            for _aid, _gid in new_goals.items():
                if _gid is None:
                    self.committed_goal_status[str(_aid)] = "none"
                    self.committed_goal_progress[str(_aid)] = float("nan")
                    self.committed_goal_feasibility[str(_aid)] = 0
                    continue
                _task = env.state.tasks.get(str(_gid), None)
                if _task is None:
                    self.committed_goal_status[str(_aid)] = "agent"
                    self.committed_goal_progress[str(_aid)] = float("nan")
                    self.committed_goal_feasibility[str(_aid)] = 1
                else:
                    self.committed_goal_status[str(_aid)] = str(getattr(_task.status, "name", str(_task.status))).lower()
                    self.committed_goal_progress[str(_aid)] = float(self._switch_goal_progress_recent(env, str(_aid), str(_gid), 3))
                    _st = env.state.agents.get(str(_aid), None)
                    if _st is not None and _st.kind == AgentKind.UAV:
                        self.committed_goal_feasibility[str(_aid)] = int(self._uav_task_feasible(env, str(_aid), _task))
                        if bool(getattr(_st, "follow_target", None) is None) and _task.kind == TaskKind.EMERGENCY:
                            self.airborne_uav_goal_lock_count_total = int(self.airborne_uav_goal_lock_count_total) + 1
                    elif _st is not None and _st.kind == AgentKind.TRUCK:
                        self.committed_goal_feasibility[str(_aid)] = int(self._truck_task_valid(env, str(_aid), str(_gid)))
                    else:
                        self.committed_goal_feasibility[str(_aid)] = 1
            for _aid, _gid in new_goals.items():
                if _gid is None:
                    continue
                _task = env.state.tasks.get(str(_gid), None)
                if _task is None or _task.status != TaskStatus.PENDING:
                    continue
                _tid = str(_task.task_id)
                self._task_last_goal_step[_tid] = int(now_step)
                self._task_goal_exposure_count[_tid] = int(self._task_goal_exposure_count.get(_tid, 0) + 1)

            reasons = [k for k, v in self._last_refresh_flags.items() if k not in {"refresh", "cooldown_blocked"} and bool(v)]
            self.last_replan_reason = "+".join(reasons) if reasons else "none"
            assigned_total = int(sum(1 for _, gid in new_goals.items() if gid is not None))
            assigned_truck = int(
                sum(
                    1
                    for aid, gid in new_goals.items()
                    if gid is not None and env.state.agents.get(str(aid), None) is not None and env.state.agents[str(aid)].kind == AgentKind.TRUCK
                )
            )
            assigned_uav = int(
                sum(
                    1
                    for aid, gid in new_goals.items()
                    if gid is not None and env.state.agents.get(str(aid), None) is not None and env.state.agents[str(aid)].kind == AgentKind.UAV
                )
            )
            self.last_assignment_summary = {
                "assigned_total": int(assigned_total),
                "assigned_truck": int(assigned_truck),
                "assigned_uav": int(assigned_uav),
                "uav_goal_switch_total": int(self.goal_switch_count_total),
                "goal_switch_candidate_count": int(self.goal_switch_candidate_count_total),
                "goal_switch_accepted_count": int(self.goal_switch_accepted_count_total),
                "goal_switch_rejected_by_threshold_count": int(self.goal_switch_rejected_by_threshold_count_total),
                "goal_switch_forced_count": int(self.goal_switch_forced_count_total),
                "goal_switch_accepted_by_score_count": int(self.goal_switch_accepted_by_score_count_total),
                "goal_switch_accepted_by_eta_count": int(self.goal_switch_accepted_by_eta_count_total),
                "goal_switch_forced_reason_completed": int(self.goal_switch_forced_reason_completed_total),
                "goal_switch_forced_reason_failed": int(self.goal_switch_forced_reason_failed_total),
                "goal_switch_forced_reason_infeasible": int(self.goal_switch_forced_reason_infeasible_total),
                "goal_switch_forced_reason_dead_end": int(self.goal_switch_forced_reason_dead_end_total),
                "goal_switch_forced_reason_uav_recovery": int(self.goal_switch_forced_reason_uav_recovery_total),
                "goal_switch_forced_reason_stall": int(self.goal_switch_forced_reason_stall_total),
                "cluster_primary_reject_count": int(self.cluster_primary_reject_count_total),
                "cluster_primary_switch_count": int(self.cluster_primary_switch_count_total),
                "same_task_cooldown_reject_count": int(self.same_task_cooldown_reject_count_total),
                "uav_reject_cache_hit_count_total": int(self.uav_reject_cache_hit_count_total),
                "uav_reject_cache_insert_count_total": int(self.uav_reject_cache_insert_count_total),
                "uav_reject_cache_clear_count_total": int(self.uav_reject_cache_clear_count_total),
                "uav_reject_cache_reason_insufficient_recovery_margin": int(self.uav_reject_cache_reason_insufficient_recovery_margin_total),
                "uav_reject_cache_reason_corridor": int(self.uav_reject_cache_reason_corridor_total),
                "uav_reject_cache_reason_comm_block": int(self.uav_reject_cache_reason_comm_block_total),
                "uav_reject_cache_reason_energy_infeasible": int(self.uav_reject_cache_reason_energy_infeasible_total),
                "uav_reject_cache_reason_no_recovery": int(self.uav_reject_cache_reason_no_recovery_total),
                "uav_task_selected_count_total": int(self.uav_task_selected_count_total),
                "uav_transfer_hint_issue_count_total": int(self.uav_transfer_hint_issue_count_total),
                "uav_transfer_hint_keep_count_total": int(self.uav_transfer_hint_keep_count_total),
                "uav_transfer_hint_active_count": int(len(self._uav_transfer_target_truck)),
                "initial_directional_plan_active": int(bool(self._initial_directional_truck_sector) and self._in_initial_directional_phase(env)),
                "initial_directional_truck_assigned_count": int(len(self._initial_directional_truck_sector)),
                "initial_directional_uav_assigned_count": int(len(self._initial_directional_uav_truck)),
                "region_commitment_setup_count": int(self.region_commitment_setup_count_total),
                "region_commitment_region_count": int(len(self._region_centers_xy)),
                "region_commitment_effective_k": int(self._region_commitment_effective_k),
                "region_commitment_effective_enabled": int(bool(self._region_commitment_enabled_effective)),
                "region_commitment_auto_score": float(self._region_commitment_auto_score),
                "region_commitment_separation_score": float(self._region_commitment_separation_score),
                "region_commitment_load_balance_score": float(self._region_commitment_load_balance_score),
                "region_commitment_coverage_score": float(self._region_commitment_coverage_score),
                "region_commitment_strength": float(self._region_commitment_strength),
                "region_commitment_auto_enabled_count": int(self.region_commitment_auto_enabled_count_total),
                "region_commitment_auto_disabled_count": int(self.region_commitment_auto_disabled_count_total),
                "region_commitment_local_candidate_count": int(self.region_commitment_local_candidate_count_total),
                "region_commitment_cross_filtered_count": int(self.region_commitment_cross_filtered_count_total),
                "region_commitment_cross_override_count": int(self.region_commitment_cross_override_count_total),
                "region_commitment_outlier_task_count": int(self.region_commitment_outlier_task_count_total),
                "region_commitment_outlier_filtered_count": int(self.region_commitment_outlier_filtered_count_total),
                "region_commitment_outlier_override_count": int(self.region_commitment_outlier_override_count_total),
            }
        else:
            self.last_replan_reason = "no_refresh"
            self.uav_transfer_hint_keep_count_total += int(len(self._uav_transfer_target_truck))
            if self._commitment_local_correction_mode(env) and bool(self.state.goals):
                self.committed_goal_hold_count_total = int(self.committed_goal_hold_count_total) + int(
                    sum(1 for _g in self.state.goals.values() if _g is not None)
                )

        self._publish_runtime_sidechannels(env)
        if hasattr(env, "note_planner_replan"):
            try:
                env.note_planner_replan(dict(self._last_refresh_flags), str(self.last_replan_reason))
            except Exception:
                pass
        return dict(self.state.goals)










































































