from __future__ import annotations

"""Two-layer cooperative route planning used by the paper mainline.

Layer 1 owns complete, exclusive truck--UAV task lines.  Layer 2 receives only
the current stop: a normal task node or an emergency launch anchor.  Road
events invalidate and repair route suffixes instead of reassigning every task
at every decision step.

The legacy rolling/attraction planners remain untouched and can be selected by
disabling ``hrl_route_plan_v2_enabled``.
"""

from dataclasses import dataclass, field
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from hetgat_hrl.core.mdp_spec import AgentKind, TaskKind, TaskStatus
from hetgat_hrl.core.algorithm_profile import (
    er_hlns_balanced_all_tasks_active,
    er_hlns_balanced_all_tasks_v2_active,
    er_hlns_balanced_all_tasks_v3_active,
    er_hlns_balanced_all_tasks_v5_active,
    er_hlns_idle_routine_dispatch_active,
    er_hlns_low_seed_rescue_active,
    er_hlns_parallel_routine_emergency_active,
    er_hlns_r4_routine_takeover_active,
)


DIRECT = "DIRECT"
BULK_RELAY = "BULK_RELAY"
NORMAL_SERVICE = "NORMAL_SERVICE"
EMERGENCY_LAUNCH = "EMERGENCY_LAUNCH"
BULK_RELAY_LAUNCH = "BULK_RELAY_LAUNCH"


@dataclass
class TaskContract:
    task_id: str
    owner_agent_id: str
    truck_id: str
    uav_id: Optional[str]
    uav_ids: Tuple[str, ...]
    service_mode: str
    created_step: int
    locked: bool = True
    # Monotone execution-install version. It normally follows PlanVersion,
    # and is incremented for an in-place owner transfer.
    version: int = 0
    recovery_truck_id: Optional[str] = None
    recovery_anchor_node: Optional[int] = None


@dataclass
class RouteStop:
    task_id: str
    stop_type: str
    truck_id: str
    uav_id: Optional[str]
    uav_ids: Tuple[str, ...]
    target_node: int
    anchor_nodes: Tuple[int, ...] = ()
    selected_anchor: Optional[int] = None
    planned_road_distance_m: float = float("inf")
    planned_air_distance_m: float = 0.0
    eta_step: int = 0
    deadline_step: int = 0
    service_mode: str = DIRECT
    recovery_truck_id: Optional[str] = None
    recovery_anchor_node: Optional[int] = None


@dataclass
class ClusterRoute:
    truck_id: str
    uav_ids: Tuple[str, ...]
    stops: List[RouteStop] = field(default_factory=list)
    cursor: int = 0

    def current(self, env) -> Optional[RouteStop]:
        while self.cursor < len(self.stops):
            stop = self.stops[self.cursor]
            task = env.state.tasks.get(str(stop.task_id), None)
            if task is not None and task.status in (TaskStatus.PENDING, TaskStatus.CLAIMED):
                return stop
            self.cursor += 1
        return None


@dataclass
class PlanVersion:
    version: int
    created_step: int
    road_signature: Tuple[Tuple[int, int], ...]
    routes: Dict[str, ClusterRoute]
    contracts: Dict[str, TaskContract]
    objective: float
    reason: str


@dataclass
class PlannerFeedback:
    step: int
    reason: str
    truck_id: str = ""
    task_id: str = ""
    detail: str = ""
    suffix_repair_required: bool = False


class HierarchicalRoutePlanManager:
    """Persistent layer-1 route owner with bounded ALNS suffix repair."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(int(seed))
        self.plan: Optional[PlanVersion] = None
        self.last_seen_step: int = -1
        self.last_replan_step: int = -10**9
        self.plan_version_count: int = 0
        self.full_plan_count: int = 0
        self.suffix_repair_count: int = 0
        self.suffix_repair_success_count: int = 0
        self.anchor_backup_switch_count: int = 0
        self.contract_release_count: int = 0
        self.stalled_contract_transfer_candidate_count: int = 0
        self.stalled_contract_transfer_replan_count: int = 0
        self.contract_transfer_count: int = 0
        self.onsite_takeover_count: int = 0
        self.routine_opportunity_candidate_count: int = 0
        self.routine_opportunity_transfer_count: int = 0
        self.routine_opportunity_blocked_assist_count: int = 0
        self.routine_opportunity_blocked_eta_count: int = 0
        self.emergency_starvation_promotion_count: int = 0
        self.emergency_launch_watchdog_ready_count: int = 0
        self.emergency_launch_watchdog_force_count: int = 0
        self.direct_safe_secondary_emergency_candidate_count: int = 0
        self.direct_safe_secondary_emergency_assignment_count: int = 0
        self.lifecycle_turnaround_cost_evaluation_count: int = 0
        self.lifecycle_turnaround_cost_total: float = 0.0
        self.lexicographic_comparison_count: int = 0
        self.lexicographic_primary_rejection_count: int = 0
        self.deadline_rescue_promotion_count: int = 0
        # Candidate-only road-impact emergency promotion diagnostics.  These
        # remain zero unless the explicit conditional-promotion overlay is
        # enabled; the formal ER-HLNS route manager is unchanged.
        self.road_impact_emergency_promotion_candidate_count: int = 0
        self.road_impact_emergency_promotion_trigger_count: int = 0
        self.road_impact_emergency_promotion_count: int = 0
        self.road_impact_emergency_promotion_reject_count: int = 0
        self.road_impact_emergency_promotion_reject_no_delta_count: int = 0
        self.road_impact_emergency_promotion_reject_no_risk_count: int = 0
        self.road_impact_emergency_promotion_reject_protected_count: int = 0
        self.road_impact_emergency_promotion_reject_cooldown_count: int = 0
        self.residual_emergency_handoff_count: int = 0
        self.routine_inventory_rebalance_count: int = 0
        # Candidate-only risk-slack routine suffix diagnostics.  These remain
        # zero for the default/formal planner because the corresponding
        # configuration gate is false.
        self.risk_slack_routine_candidate_count: int = 0
        self.risk_slack_routine_trigger_count: int = 0
        self.risk_slack_routine_release_count: int = 0
        self.risk_slack_routine_cross_truck_repair_count: int = 0
        self.risk_slack_routine_reserved_inventory_block_count: int = 0
        self.risk_slack_routine_protected_count: int = 0
        self.risk_slack_routine_unreachable_count: int = 0
        self.risk_slack_routine_stalled_count: int = 0
        self.risk_slack_routine_eta_guard_block_count: int = 0
        # Candidate-only parallel corridor diagnostics.  These counters are
        # intentionally planner-local and do not alter the physical contract.
        self.parallel_routine_emergency_candidate_count: int = 0
        self.parallel_routine_emergency_accept_count: int = 0
        self.parallel_routine_emergency_fallback_wait_count: int = 0
        self.parallel_routine_emergency_reject_energy_count: int = 0
        self.parallel_routine_emergency_reject_deadline_count: int = 0
        self.risk_slack_routine_same_truck_block_count: int = 0
        self.risk_slack_routine_tc_guard_block_count: int = 0
        # Candidate-only R4 stalled NORMAL takeover diagnostics.
        self.r4_routine_takeover_candidate_count: int = 0
        self.r4_routine_takeover_trigger_count: int = 0
        self.r4_routine_takeover_success_count: int = 0
        self.r4_routine_takeover_reject_started_count: int = 0
        self.r4_routine_takeover_reject_contract_count: int = 0
        self.r4_routine_takeover_reject_owner_active_count: int = 0
        self.r4_routine_takeover_reject_no_idle_truck_count: int = 0
        self.r4_routine_takeover_reject_unreachable_count: int = 0
        self.r4_routine_takeover_reject_inventory_count: int = 0
        self.r4_routine_takeover_reject_deadline_count: int = 0
        self.r4_routine_takeover_reject_safety_count: int = 0
        # Candidate-only idle NORMAL dispatch diagnostics.
        self.idle_routine_dispatch_candidate_count: int = 0
        self.idle_routine_dispatch_success_count: int = 0
        self.idle_routine_dispatch_emergency_block_count: int = 0
        self.idle_routine_dispatch_reject_no_idle_truck_count: int = 0
        self.idle_routine_dispatch_reject_unreachable_count: int = 0
        self.idle_routine_dispatch_reject_inventory_count: int = 0
        self.idle_routine_dispatch_reject_deadline_count: int = 0
        self.idle_routine_dispatch_reject_contract_count: int = 0
        # Candidate-only service-start rescue diagnostics.
        self.routine_service_start_rescue_candidate_count: int = 0
        self.routine_service_start_rescue_success_count: int = 0
        self.routine_service_start_rescue_alternate_count: int = 0
        self.routine_service_start_rescue_emergency_block_count: int = 0
        self.routine_service_start_rescue_reject_active_count: int = 0
        self.routine_service_start_rescue_reject_no_owner_count: int = 0
        self.routine_service_start_rescue_reject_no_alternate_count: int = 0
        # Candidate-only balanced-all-tasks diagnostics.
        self.balanced_all_tasks_normal_candidate_count: int = 0
        self.balanced_all_tasks_normal_assignment_count: int = 0
        self.balanced_all_tasks_quota_block_count: int = 0
        self.balanced_all_tasks_emergency_tradeoff_count: int = 0
        self.balanced_all_tasks_watchdog_candidate_count: int = 0
        self.balanced_all_tasks_watchdog_transfer_count: int = 0
        self.balanced_all_tasks_watchdog_block_count: int = 0
        self.balanced_all_tasks_v2_reauction_candidate_count: int = 0
        self.balanced_all_tasks_v2_reauction_transfer_count: int = 0
        self.balanced_all_tasks_v2_reauction_deadline_block_count: int = 0
        self.balanced_all_tasks_v2_parallel_candidate_count: int = 0
        self.balanced_all_tasks_v2_parallel_accept_count: int = 0
        self.balanced_all_tasks_v2_parallel_reject_count: int = 0
        self.balanced_all_tasks_v2_parallel_payload_bypass_count: int = 0
        self.balanced_all_tasks_v2_aggressive_auction_candidate_count: int = 0
        self.balanced_all_tasks_v2_aggressive_auction_transfer_count: int = 0
        self.balanced_all_tasks_v2_after_launch_normal_candidate_count: int = 0
        self.balanced_all_tasks_v2_after_launch_normal_accept_count: int = 0
        self.balanced_all_tasks_v2_after_launch_normal_reject_count: int = 0
        self.balanced_all_tasks_v3_normal_candidate_count: int = 0
        self.balanced_all_tasks_v3_emergency_candidate_count: int = 0
        self.balanced_all_tasks_v3_selected_normal_count: int = 0
        self.balanced_all_tasks_v3_selected_emergency_count: int = 0
        self.balanced_all_tasks_v3_fallback_count: int = 0
        self.balanced_all_tasks_v3_fallback_reason_counts: Dict[str, int] = {}
        self.balanced_all_tasks_v3_last_diagnostics: Dict[str, Any] = {}
        self.balanced_all_tasks_v5_promoted_count: int = 0
        self.balanced_all_tasks_v5_rejected_safety_count: int = 0
        self.shadow_total_coverage_candidate_count: int = 0
        self.shadow_total_coverage_accept_count: int = 0
        self.shadow_total_coverage_reject_count: int = 0
        self.shadow_total_coverage_last_diagnostics: Dict[str, Any] = {}
        self.shadow_total_coverage_first_accept_diagnostics: Dict[str, Any] = {}
        self.normal_cleanup_replan_count: int = 0
        self.hard_normal_rescue_candidate_count: int = 0
        self.hard_normal_rescue_transfer_count: int = 0
        self.hard_normal_rescue_no_goal_count: int = 0
        self.hard_normal_rescue_stalled_owner_count: int = 0
        self.hard_normal_rescue_rejected_safety_count: int = 0
        self.hard_normal_rescue_no_truck_count: int = 0
        self.hard_normal_rescue_no_truck_skip_count: int = 0
        self.b_orphaned_routine_rescue_count: int = 0
        self.contract_consistency_block_count: int = 0
        self.unexplained_stay_count: int = 0
        self.alns_iteration_count: int = 0
        self.alns_destroyed_assignment_count: int = 0
        self.alns_repair_attempt_count: int = 0
        self.alns_repair_feasible_count: int = 0
        self.alns_accepted_count: int = 0
        self.alns_improvement_count: int = 0
        self.alns_replan_count: int = 0
        self.alns_objective_evaluation_count: int = 0
        self.alns_feasibility_evaluation_count: int = 0
        self.alns_wall_clock_time_s: float = 0.0
        self._episode_token: Optional[Tuple[int, Tuple[str, ...]]] = None
        self._anchor_cache: Dict[
            Tuple[Tuple[Tuple[int, int], ...], int, str], Tuple[int, ...]
        ] = {}
        self._feedback: List[PlannerFeedback] = []
        self._last_goals: Dict[str, Optional[str]] = {}
        self._assist_by_truck: Dict[str, Dict[str, Any]] = {}
        self._stay_reason_by_agent: Dict[str, str] = {}
        self._transfer_by_uav: Dict[str, Dict[str, str]] = {}
        self._contract_progress: Dict[str, Dict[str, Any]] = {}
        self._contract_last_transfer_step: Dict[str, int] = {}
        self._last_actionable_route_feedback: Dict[
            Tuple[str, str, str], Tuple[int, Any]
        ] = {}
        self._last_missing_contract_fingerprint: Optional[Tuple[Any, ...]] = None
        self._normal_cleanup_last_active_count: Optional[int] = None
        self._normal_cleanup_last_progress_step: int = 0
        self._normal_cleanup_last_replan_step: int = -10**9
        self._normal_cleanup_owner_by_task: Dict[str, str] = {}
        self._hard_normal_rescue_count_by_task: Dict[str, int] = {}
        self._hard_normal_rescue_progress: Dict[str, Dict[str, Any]] = {}
        self._hard_normal_rescue_no_truck_last_step_by_task: Dict[str, int] = {}
        self._aggressive_planning_active: bool = False
        # Repair features are deliberately task-scoped.  A task that has
        # already been transferred once must not bounce between otherwise
        # healthy routes after every road observation.
        self._deadline_rescue_transferred_tasks: set[str] = set()
        self._road_impact_emergency_promotion_last_step_by_route: Dict[str, int] = {}
        self._road_impact_emergency_promotion_last_step_by_task: Dict[str, int] = {}
        self._routine_inventory_rebalanced_tasks: set[str] = set()
        self._risk_slack_routine_transfer_count_by_task: Dict[str, int] = {}
        self._risk_slack_routine_progress: Dict[str, Dict[str, Any]] = {}
        self._residual_handoff_last_step_by_task: Dict[str, int] = {}
        self._last_completed_task_count: int = 0
        self._last_completion_progress_step: int = 0
        self._queue_starvation_repair_done: bool = False
        self._queue_starvation_repair_pending: bool = False
        self.queue_starvation_repair_count: int = 0
        self.initial_lifeline_ordering_enabled: bool = False
        self._global_lifeline_ordering_allowed: bool = False
        self._post_emergency_cleanup_done: bool = False
        self._temporary_forbidden_truck_by_task: Dict[str, str] = {}
        self._routine_opportunity_transfer_count_by_task: Dict[str, int] = {}
        self._r4_routine_takeover_count_by_task: Dict[str, int] = {}
        self._r4_routine_takeover_progress: Dict[str, Dict[str, Any]] = {}
        self._routine_service_start_rescue_count_by_task: Dict[str, int] = {}
        self._routine_service_start_rescue_progress: Dict[str, Dict[str, Any]] = {}
        self._emergency_launch_ready_since_by_task: Dict[str, int] = {}
        self._force_takeoff_task_by_uav: Dict[str, str] = {}
        self._queue_rescue_task_by_uav: Dict[str, str] = {}
        self._queue_rescue_anchor_by_uav: Dict[str, int] = {}
        self._queue_rescue_started_step_by_uav: Dict[str, int] = {}
        self._queue_rescue_launch_ready_step_by_uav: Dict[str, int] = {}
        self._queue_rescue_cooldown_until_by_task: Dict[str, int] = {}
        self._queue_rescue_best_road_distance_by_uav: Dict[str, float] = {}
        self._queue_rescue_last_progress_step_by_uav: Dict[str, int] = {}
        self._queue_rescue_reanchor_count_by_uav: Dict[str, int] = {}
        # A failed temporary rescue must be rebuilt by layer 1.  Keeping the
        # rewritten contract after releasing the temporary anchor leaves a
        # task that looks owned but has no executable route stop.
        self._queue_rescue_failed_task_ids: Dict[str, str] = {}
        self._disconnect_profile_cache_key: Optional[Tuple[Any, ...]] = None
        self._disconnect_profile_cache: Dict[str, Dict[str, float]] = {}
        self.disconnect_profile_evaluation_count: int = 0
        self.disconnect_protected_task_count: int = 0
        self.disconnect_predicted_miss_count: int = 0
        self._emergency_balance_active: Optional[bool] = None
        self._emergency_balance_episode_latched: Optional[bool] = None
        self.emergency_balance_trigger_count: int = 0
        self.emergency_balance_baseline_max_count: int = 0
        self._emergency_capacity_target_by_truck: Dict[str, int] = {}
        self._enforce_emergency_inventory_budget_active: bool = False
        self._emergency_inventory_initial_plan_done: bool = False
        self.emergency_capacity_repair_count: int = 0
        self.emergency_capacity_contract_move_count: int = 0
        self.queue_rescue_assignment_count: int = 0
        self.queue_rescue_delivery_count: int = 0

    @staticmethod
    def _repair_profile() -> str:
        """Return the reproducible mainline repair profile.

        The historical ``PAPER_ALNS_ROUTE_REPAIR_PROFILE`` escape hatch is
        intentionally ignored: an unrecorded process environment must not
        change the paper algorithm.
        """
        return "targeted"

    def _targeted_repairs_enabled(self) -> bool:
        return self._repair_profile() != "baseline"

    def _stamp_contract_on_task(
        self,
        env,
        task_id: str,
        contract: TaskContract,
        *,
        bump: bool = False,
    ) -> None:
        """Publish one contract atomically to task-visible execution metadata."""
        task = env.state.tasks.get(str(task_id), None)
        current_version = int(max(getattr(contract, "version", 0), 0))
        if current_version <= 0:
            current_version = int(
                max(
                    getattr(task, "route_contract_version", 0)
                    if task is not None
                    else 0,
                    0,
                )
                + 1
            )
        if bump:
            current_version += 1
        contract.version = int(current_version)
        if task is None:
            return
        task.route_contract_owner = str(contract.owner_agent_id)
        task.route_contract_truck = str(contract.truck_id)
        task.route_contract_uav_ids = tuple(str(uid) for uid in contract.uav_ids)
        task.route_contract_version = int(current_version)

    def reset(self) -> None:
        self.plan = None
        self.last_seen_step = -1
        self.last_replan_step = -10**9
        self._episode_token = None
        self._anchor_cache.clear()
        self._feedback.clear()
        self._last_goals.clear()
        self._assist_by_truck.clear()
        self._stay_reason_by_agent.clear()
        self._transfer_by_uav.clear()
        self._contract_progress.clear()
        self._contract_last_transfer_step.clear()
        self._last_actionable_route_feedback.clear()
        self._last_missing_contract_fingerprint = None
        self._normal_cleanup_last_active_count = None
        self._normal_cleanup_last_progress_step = 0
        self._normal_cleanup_last_replan_step = -10**9
        self._normal_cleanup_owner_by_task.clear()
        self._hard_normal_rescue_count_by_task.clear()
        self._hard_normal_rescue_progress.clear()
        self._hard_normal_rescue_no_truck_last_step_by_task.clear()
        self._routine_opportunity_transfer_count_by_task.clear()
        self._r4_routine_takeover_count_by_task.clear()
        self._r4_routine_takeover_progress.clear()
        self._routine_service_start_rescue_count_by_task.clear()
        self._routine_service_start_rescue_progress.clear()
        self._emergency_launch_ready_since_by_task.clear()
        self._force_takeoff_task_by_uav.clear()
        self._queue_rescue_task_by_uav.clear()
        self._queue_rescue_anchor_by_uav.clear()
        self._queue_rescue_started_step_by_uav.clear()
        self._queue_rescue_launch_ready_step_by_uav.clear()
        self._queue_rescue_cooldown_until_by_task.clear()
        self._queue_rescue_best_road_distance_by_uav.clear()
        self._queue_rescue_last_progress_step_by_uav.clear()
        self._queue_rescue_reanchor_count_by_uav.clear()
        self._queue_rescue_failed_task_ids.clear()
        self._disconnect_profile_cache_key = None
        self._disconnect_profile_cache.clear()
        self.disconnect_profile_evaluation_count = 0
        self.disconnect_protected_task_count = 0
        self.disconnect_predicted_miss_count = 0
        self._emergency_balance_active = None
        self._emergency_balance_episode_latched = None
        self.emergency_balance_trigger_count = 0
        self.emergency_balance_baseline_max_count = 0
        self._emergency_capacity_target_by_truck.clear()
        self._enforce_emergency_inventory_budget_active = False
        self._emergency_inventory_initial_plan_done = False
        self.emergency_capacity_repair_count = 0
        self.emergency_capacity_contract_move_count = 0
        self.queue_rescue_assignment_count = 0
        self.queue_rescue_delivery_count = 0
        self._aggressive_planning_active = False
        self._deadline_rescue_transferred_tasks.clear()
        self._road_impact_emergency_promotion_last_step_by_route.clear()
        self._road_impact_emergency_promotion_last_step_by_task.clear()
        self._routine_inventory_rebalanced_tasks.clear()
        self._risk_slack_routine_transfer_count_by_task.clear()
        self._risk_slack_routine_progress.clear()
        self._residual_handoff_last_step_by_task.clear()
        self._last_completed_task_count = 0
        self._last_completion_progress_step = 0
        self._queue_starvation_repair_done = False
        self._queue_starvation_repair_pending = False
        self.queue_starvation_repair_count = 0
        self.initial_lifeline_ordering_enabled = False
        self._global_lifeline_ordering_allowed = False
        self._post_emergency_cleanup_done = False
        self.plan_version_count = 0
        self.full_plan_count = 0
        self.suffix_repair_count = 0
        self.suffix_repair_success_count = 0
        self.anchor_backup_switch_count = 0
        self.contract_release_count = 0
        self.stalled_contract_transfer_candidate_count = 0
        self.stalled_contract_transfer_replan_count = 0
        self.contract_transfer_count = 0
        self.onsite_takeover_count = 0
        self.routine_opportunity_candidate_count = 0
        self.routine_opportunity_transfer_count = 0
        self.routine_opportunity_blocked_assist_count = 0
        self.routine_opportunity_blocked_eta_count = 0
        self.emergency_starvation_promotion_count = 0
        self.emergency_launch_watchdog_ready_count = 0
        self.emergency_launch_watchdog_force_count = 0
        self.direct_safe_secondary_emergency_candidate_count = 0
        self.direct_safe_secondary_emergency_assignment_count = 0
        self.lifecycle_turnaround_cost_evaluation_count = 0
        self.lifecycle_turnaround_cost_total = 0.0
        self.lexicographic_comparison_count = 0
        self.lexicographic_primary_rejection_count = 0
        self.deadline_rescue_promotion_count = 0
        self.road_impact_emergency_promotion_candidate_count = 0
        self.road_impact_emergency_promotion_trigger_count = 0
        self.road_impact_emergency_promotion_count = 0
        self.road_impact_emergency_promotion_reject_count = 0
        self.road_impact_emergency_promotion_reject_no_delta_count = 0
        self.road_impact_emergency_promotion_reject_no_risk_count = 0
        self.road_impact_emergency_promotion_reject_protected_count = 0
        self.road_impact_emergency_promotion_reject_cooldown_count = 0
        self.residual_emergency_handoff_count = 0
        self.routine_inventory_rebalance_count = 0
        self.risk_slack_routine_candidate_count = 0
        self.risk_slack_routine_trigger_count = 0
        self.risk_slack_routine_release_count = 0
        self.risk_slack_routine_cross_truck_repair_count = 0
        self.risk_slack_routine_reserved_inventory_block_count = 0
        self.risk_slack_routine_protected_count = 0
        self.risk_slack_routine_unreachable_count = 0
        self.risk_slack_routine_stalled_count = 0
        self.risk_slack_routine_eta_guard_block_count = 0
        self.parallel_routine_emergency_candidate_count = 0
        self.parallel_routine_emergency_accept_count = 0
        self.parallel_routine_emergency_fallback_wait_count = 0
        self.parallel_routine_emergency_reject_energy_count = 0
        self.parallel_routine_emergency_reject_deadline_count = 0
        self.risk_slack_routine_same_truck_block_count = 0
        self.risk_slack_routine_tc_guard_block_count = 0
        self.r4_routine_takeover_candidate_count = 0
        self.r4_routine_takeover_trigger_count = 0
        self.r4_routine_takeover_success_count = 0
        self.r4_routine_takeover_reject_started_count = 0
        self.r4_routine_takeover_reject_contract_count = 0
        self.r4_routine_takeover_reject_owner_active_count = 0
        self.r4_routine_takeover_reject_no_idle_truck_count = 0
        self.r4_routine_takeover_reject_unreachable_count = 0
        self.r4_routine_takeover_reject_inventory_count = 0
        self.r4_routine_takeover_reject_deadline_count = 0
        self.r4_routine_takeover_reject_safety_count = 0
        self.idle_routine_dispatch_candidate_count = 0
        self.idle_routine_dispatch_success_count = 0
        self.idle_routine_dispatch_emergency_block_count = 0
        self.idle_routine_dispatch_reject_no_idle_truck_count = 0
        self.idle_routine_dispatch_reject_unreachable_count = 0
        self.idle_routine_dispatch_reject_inventory_count = 0
        self.idle_routine_dispatch_reject_deadline_count = 0
        self.idle_routine_dispatch_reject_contract_count = 0
        self.routine_service_start_rescue_candidate_count = 0
        self.routine_service_start_rescue_success_count = 0
        self.routine_service_start_rescue_alternate_count = 0
        self.routine_service_start_rescue_emergency_block_count = 0
        self.routine_service_start_rescue_reject_active_count = 0
        self.routine_service_start_rescue_reject_no_owner_count = 0
        self.routine_service_start_rescue_reject_no_alternate_count = 0
        self.balanced_all_tasks_normal_candidate_count = 0
        self.balanced_all_tasks_normal_assignment_count = 0
        self.balanced_all_tasks_quota_block_count = 0
        self.balanced_all_tasks_emergency_tradeoff_count = 0
        self.balanced_all_tasks_watchdog_candidate_count = 0
        self.balanced_all_tasks_watchdog_transfer_count = 0
        self.balanced_all_tasks_watchdog_block_count = 0
        self.balanced_all_tasks_v2_reauction_candidate_count = 0
        self.balanced_all_tasks_v2_reauction_transfer_count = 0
        self.balanced_all_tasks_v2_reauction_deadline_block_count = 0
        self.balanced_all_tasks_v2_parallel_candidate_count = 0
        self.balanced_all_tasks_v2_parallel_accept_count = 0
        self.balanced_all_tasks_v2_parallel_reject_count = 0
        self.balanced_all_tasks_v2_parallel_payload_bypass_count = 0
        self.balanced_all_tasks_v2_aggressive_auction_candidate_count = 0
        self.balanced_all_tasks_v2_aggressive_auction_transfer_count = 0
        self.balanced_all_tasks_v2_after_launch_normal_candidate_count = 0
        self.balanced_all_tasks_v2_after_launch_normal_accept_count = 0
        self.balanced_all_tasks_v2_after_launch_normal_reject_count = 0
        self.balanced_all_tasks_v3_normal_candidate_count = 0
        self.balanced_all_tasks_v3_emergency_candidate_count = 0
        self.balanced_all_tasks_v3_selected_normal_count = 0
        self.balanced_all_tasks_v3_selected_emergency_count = 0
        self.balanced_all_tasks_v3_fallback_count = 0
        self.balanced_all_tasks_v3_fallback_reason_counts = {}
        self.balanced_all_tasks_v3_last_diagnostics = {}
        self.balanced_all_tasks_v5_promoted_count = 0
        self.balanced_all_tasks_v5_rejected_safety_count = 0
        self.shadow_total_coverage_candidate_count = 0
        self.shadow_total_coverage_accept_count = 0
        self.shadow_total_coverage_reject_count = 0
        self.shadow_total_coverage_last_diagnostics = {}
        self.shadow_total_coverage_first_accept_diagnostics = {}
        self.normal_cleanup_replan_count = 0
        self.hard_normal_rescue_candidate_count = 0
        self.hard_normal_rescue_transfer_count = 0
        self.hard_normal_rescue_no_goal_count = 0
        self.hard_normal_rescue_stalled_owner_count = 0
        self.hard_normal_rescue_rejected_safety_count = 0
        self.hard_normal_rescue_no_truck_count = 0
        self.hard_normal_rescue_no_truck_skip_count = 0
        self.b_orphaned_routine_rescue_count = 0
        self.contract_consistency_block_count = 0
        self.unexplained_stay_count = 0
        self.alns_iteration_count = 0
        self.alns_destroyed_assignment_count = 0
        self.alns_repair_attempt_count = 0
        self.alns_repair_feasible_count = 0
        self.alns_accepted_count = 0
        self.alns_improvement_count = 0
        self.alns_replan_count = 0
        self.alns_objective_evaluation_count = 0
        self.alns_feasibility_evaluation_count = 0
        self.alns_wall_clock_time_s = 0.0

    @staticmethod
    def _active(task) -> bool:
        return bool(task.status in (TaskStatus.PENDING, TaskStatus.CLAIMED))

    @staticmethod
    def _is_relay(task) -> bool:
        return str(getattr(task, "service_mode", DIRECT)).upper() == BULK_RELAY

    @staticmethod
    def _xy(env, node_id: int) -> Tuple[float, float]:
        node = env.topology.nodes[int(node_id)]
        return float(node.x), float(node.y)

    def _road_signature(self, env) -> Tuple[Tuple[int, int], ...]:
        try:
            edges = env._decision_blocked_edges()
        except Exception:
            edges = getattr(env.topology, "blocked_edges", set())
        return tuple(
            sorted(
                (min(int(a), int(b)), max(int(a), int(b)))
                for a, b in edges
            )
        )

    def _live_trucks(self, env) -> List[str]:
        return sorted(
            str(aid)
            for aid, state in env.state.agents.items()
            if state.kind == AgentKind.TRUCK
            and not bool(getattr(state, "crashed", False))
            and state.node is not None
        )

    def _cluster_uavs(self, env, truck_ids: Sequence[str]) -> Dict[str, Tuple[str, ...]]:
        out: Dict[str, List[str]] = {str(tid): [] for tid in truck_ids}
        unbound: List[str] = []
        for uid, state in sorted(env.state.agents.items()):
            if state.kind != AgentKind.UAV or bool(getattr(state, "crashed", False)):
                continue
            follow = None if state.follow_target is None else str(state.follow_target)
            if follow in out:
                out[follow].append(str(uid))
                continue
            prior: Optional[TaskContract] = None
            if self.plan is not None:
                for contract in self.plan.contracts.values():
                    if contract.uav_id == str(uid):
                        prior = contract
                        break
            if prior is not None and prior.truck_id in out:
                out[prior.truck_id].append(str(uid))
            else:
                unbound.append(str(uid))
        for index, uid in enumerate(unbound):
            if truck_ids:
                out[str(truck_ids[index % len(truck_ids)])].append(str(uid))
        return {tid: tuple(sorted(uids)) for tid, uids in out.items()}

    @staticmethod
    def _timecritical_unit_kg(env) -> float:
        try:
            return float(max(env._timecritical_supply_unit_kg(), 1e-6))
        except Exception:
            return float(
                max(
                    getattr(
                        env.cfg,
                        "timecritical_supply_unit_kg",
                        getattr(env.cfg, "emergency_task_demand_kg", 150.0),
                    ),
                    1e-6,
                )
            )

    def _cluster_emergency_capacity_units(
        self,
        env,
        truck_id: str,
        uav_ids: Sequence[str],
    ) -> int:
        """Count physical TC packages currently owned by one cooperative unit.

        Packages already mounted on UAVs and packages in the truck body are
        one material pool.  Layer 1 must not issue more emergency contracts to
        a truck--UAV unit than this pool can execute.
        """
        truck = env.state.agents.get(str(truck_id), None)
        unit_kg = self._timecritical_unit_kg(env)
        truck_units = int(
            np.floor(
                float(max(getattr(truck, "timecritical_inventory_kg_current", 0.0), 0.0))
                / unit_kg
            )
        )
        mounted_units = 0
        for uav_id in uav_ids:
            state = env.state.agents.get(str(uav_id), None)
            if state is None or bool(getattr(state, "crashed", False)):
                continue
            mounted_units += int(max(getattr(state, "carried_emergency_units", 0), 0))
        return int(max(truck_units + mounted_units, 0))

    def _capacity_aware_emergency_targets(
        self,
        env,
        truck_ids: Sequence[str],
        cluster_uavs: Dict[str, Tuple[str, ...]],
    ) -> Dict[str, int]:
        """Allocate emergency workload by UAV throughput under package caps.

        With four trucks and six mounted UAVs, the usual 2/2/1/1 mounting
        pattern yields a 4/4/2/2 target for twelve emergency tasks.  Targets
        are soft routing preferences; remaining physical package counts are
        enforced separately as hard feasibility bounds.
        """
        if not bool(
            getattr(
                env.cfg,
                "hrl_route_plan_capacity_aware_emergency_allocation_enabled",
                False,
            )
        ):
            return {}
        active_count = int(
            sum(
                1
                for task in env.state.tasks.values()
                if task.kind == TaskKind.EMERGENCY and self._active(task)
            )
        )
        targets = {str(truck_id): 0 for truck_id in truck_ids}
        if active_count <= 0 or not truck_ids:
            return targets
        capacities = {
            str(truck_id): int(
                self._cluster_emergency_capacity_units(
                    env,
                    str(truck_id),
                    cluster_uavs.get(str(truck_id), ()),
                )
            )
            for truck_id in truck_ids
        }
        weights = {
            str(truck_id): float(
                max(len(cluster_uavs.get(str(truck_id), ())), 1)
            )
            for truck_id in truck_ids
        }
        assignable = int(min(active_count, sum(capacities.values())))
        remaining = int(assignable)
        while remaining > 0:
            eligible = [
                truck_id
                for truck_id in targets
                if targets[truck_id] < capacities[truck_id]
            ]
            if not eligible:
                break
            # Weighted round-robin with hard package caps. A two-UAV unit
            # receives work twice as quickly as a one-UAV unit, while every
            # assignment remains backed by one physical package.
            selected = min(
                eligible,
                key=lambda truck_id: (
                    targets[truck_id] / max(weights[truck_id], 1e-9),
                    -weights[truck_id],
                    truck_id,
                ),
            )
            targets[selected] += 1
            remaining -= 1
        return targets

    def _contract_has_emergency_supply(self, env, contract: TaskContract, task) -> bool:
        if task.kind != TaskKind.EMERGENCY:
            return True
        owner = env.state.agents.get(str(contract.uav_id or ""), None)
        if owner is not None and int(max(getattr(owner, "carried_emergency_units", 0), 0)) > 0:
            return True
        clusters = self._cluster_uavs(env, self._live_trucks(env))
        return bool(
            self._cluster_emergency_capacity_units(
                env,
                str(contract.truck_id),
                clusters.get(str(contract.truck_id), ()),
            )
            > 0
        )

    def _mark_service_modes(self, env, truck_ids: Sequence[str]) -> None:
        relay_enabled = bool(
            getattr(env.cfg, "hrl_route_plan_bulk_relay_enabled", True)
        )
        emergency_work_active = any(
            task.kind == TaskKind.EMERGENCY and self._active(task)
            for task in env.state.tasks.values()
        )
        for task in env.state.tasks.values():
            if not self._active(task):
                continue
            if not str(getattr(task, "original_task_kind", "")).strip():
                task.original_task_kind = str(
                    getattr(task.kind, "name", str(task.kind))
                ).lower()
            if task.kind != TaskKind.NORMAL:
                task.service_mode = DIRECT
                continue
            if not self._targeted_repairs_enabled() and self._is_relay(task):
                continue
            was_relay = self._is_relay(task)
            reachable = False
            truck_at_task = False
            for truck_id in truck_ids:
                state = env.state.agents.get(str(truck_id), None)
                if state is None or state.node is None:
                    continue
                if int(state.node) == int(task.demand_node):
                    truck_at_task = True
                distance = float(
                    env._decision_shortest_path_distance(
                        int(state.node), int(task.demand_node)
                    )
                )
                if np.isfinite(distance):
                    reachable = True
                    break
            # Do not tear down a viable UAV relay merely because a truck in a
            # different part of its route can again reach the task.  Restore
            # DIRECT only when a truck has actually arrived at the task node;
            # this fixes the zero-distance no-service case without creating
            # mode oscillation and contract churn.
            use_dynamic_current_reachability = bool(
                self._aggressive_planning_active
            )
            if (
                not use_dynamic_current_reachability
                and was_relay
                and relay_enabled
                and not truck_at_task
                and emergency_work_active
            ):
                task.service_mode = BULK_RELAY
            else:
                task.service_mode = (
                    DIRECT if reachable or not relay_enabled else BULK_RELAY
                )

    @staticmethod
    def _remaining_demand_kg(task) -> float:
        return float(
            max(
                getattr(
                    task,
                    "remaining_demand_kg",
                    getattr(task, "demand_kg", 0.0),
                ),
                0.0,
            )
        )

    def _uav_contract_rank(self, env, uav_id: str, task, truck_id: str) -> Tuple[int, int, int, float, str]:
        """Prefer a docked, loaded, charged UAV for a newly built contract."""
        state = env.state.agents.get(str(uav_id), None)
        if state is None:
            return (1, 1, 1, 0.0, str(uav_id))
        docked_here = bool(
            state.follow_target is not None
            and str(state.follow_target) == str(truck_id)
        )
        try:
            loaded = bool(env._uav_loaded_for_task(str(uav_id), task))
        except Exception:
            loaded = bool(getattr(state, "payload_kg_current", 0.0) > 1e-9)
        reload_blocked = bool(
            getattr(state, "uav_needs_reload_flag", False)
            or int(getattr(state, "uav_reload_timer", 0)) > 0
        )
        return (
            0 if docked_here else 1,
            0 if loaded else 1,
            1 if reload_blocked else 0,
            -float(getattr(state, "battery", 0.0)),
            str(uav_id),
        )

    def _safe_sortie_radius(self, env) -> float:
        max_sortie = float(max(getattr(env.cfg, "uav_max_sortie_m", 6000.0), 1.0))
        # Layer 1 may consider the full nominal round-trip radius.  The old
        # 2.46 km heuristic pre-filter rejected isolated emergency tasks
        # before a truck could even receive an approach/launch contract.
        # Exact battery, weather and rendezvous validation remains mandatory
        # in the execution layer before take-off.
        return float(max(0.50 * max_sortie, 1.0))

    def _anchor_nodes(
        self,
        env,
        source_node: int,
        task,
        road_signature: Tuple[Tuple[int, int], ...],
    ) -> Tuple[int, ...]:
        key = (road_signature, int(source_node), str(task.task_id))
        cached = self._anchor_cache.get(key, None)
        if cached is not None:
            return cached
        tx, ty = self._xy(env, int(task.demand_node))
        sortie_radius = self._safe_sortie_radius(env)
        cap = int(
            max(getattr(env.cfg, "hrl_route_plan_anchor_candidate_cap", 320), 1)
        )
        backup_count = int(
            max(getattr(env.cfg, "hrl_route_plan_backup_anchor_count", 3), 1)
        )
        air_ranked: List[Tuple[float, int]] = []
        for node_id in env.topology.nodes:
            nx, ny = self._xy(env, int(node_id))
            air = float(np.hypot(nx - tx, ny - ty))
            if air <= sortie_radius + 1e-9:
                air_ranked.append((air, int(node_id)))
        air_ranked.sort(key=lambda item: (item[0], item[1]))
        candidates: List[Tuple[float, float, int]] = []
        for air, node_id in air_ranked[:cap]:
            road = float(
                env._decision_shortest_path_distance(int(source_node), int(node_id))
            )
            if not np.isfinite(road):
                continue
            # The anchor is the closest road-reachable point to the task.
            # Road distance is the tie-breaker; layer 1 already accounts for
            # the truck travel cost when comparing complete route insertions.
            candidates.append((air, road, int(node_id)))
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        result = tuple(int(item[2]) for item in candidates[:backup_count])
        self._anchor_cache[key] = result
        return result

    def _nearest_reachable_boundary(
        self, env, source_node: int, task
    ) -> Optional[Tuple[int, float, float]]:
        """Closest road-reachable boundary node, even when solo return fails."""
        tx, ty = self._xy(env, int(task.demand_node))
        best: Optional[Tuple[float, float, int]] = None
        for node_id in env.topology.nodes:
            road = float(env._decision_shortest_path_distance(int(source_node), int(node_id)))
            if not np.isfinite(road):
                continue
            nx, ny = self._xy(env, int(node_id))
            air = float(np.hypot(nx - tx, ny - ty))
            candidate = (air, road, int(node_id))
            if best is None or candidate < best:
                best = candidate
        if best is None:
            return None
        return int(best[2]), float(best[0]), float(best[1])

    def _cross_truck_recovery_pair(
        self,
        env,
        launch_truck_id: str,
        task,
        cluster_uavs: Dict[str, Tuple[str, ...]],
    ) -> Optional[Tuple[int, float, float, str, int, float, float]]:
        """Build A-launch/B-recovery geometry for an otherwise solo-infeasible TC.

        The UAV flies A-boundary -> task -> B-boundary.  Both truck legs must
        be road reachable and the combined air path must fit the nominal
        sortie budget; execution still performs the authoritative energy and
        weather safety check.
        """
        launch = env.state.agents.get(str(launch_truck_id), None)
        if launch is None or launch.node is None or not cluster_uavs.get(str(launch_truck_id), ()):
            return None
        a = self._nearest_reachable_boundary(env, int(launch.node), task)
        if a is None:
            return None
        max_air = float(max(getattr(env.cfg, "uav_max_sortie_m", 6000.0), 1.0))
        truck_speed = float(max(getattr(env.cfg, "truck_speed_mps", 10.0), 1e-6))
        uav_speed = float(max(getattr(env.cfg, "uav_max_speed_mps", 22.0), 1e-6))
        dt = float(max(getattr(env.cfg, "dt_seconds", 1.0), 1e-6))
        best = None
        for recovery_truck_id, recovery in env.state.agents.items():
            if str(recovery_truck_id) == str(launch_truck_id):
                continue
            if recovery.kind != AgentKind.TRUCK or recovery.node is None or bool(getattr(recovery, "crashed", False)):
                continue
            if int(env._truck_follower_count(str(recovery_truck_id))) >= int(max(getattr(env.cfg, "uav_max_followers_per_truck", 1), 1)):
                continue
            b = self._nearest_reachable_boundary(env, int(recovery.node), task)
            if b is None:
                continue
            air_total = float(a[1] + b[1])
            if air_total > 0.92 * max_air + 1e-9:
                continue
            busy_steps = max(
                (
                    int(max(getattr(active_task, "service_remaining", 0), 0))
                    for active_task in env.state.tasks.values()
                    if str(getattr(active_task, "in_service_by", "") or "")
                    == str(recovery_truck_id)
                ),
                default=0,
            )
            launch_chain_s = float(a[2] / truck_speed + air_total / uav_speed)
            recovery_ready_s = float(busy_steps * dt + b[2] / truck_speed)
            synchronized_s = float(max(launch_chain_s, recovery_ready_s))
            candidate = (
                synchronized_s,
                float(a[2] + b[2]),
                air_total,
                str(recovery_truck_id),
                a,
                b,
            )
            if best is None or candidate[:3] < best[:3]:
                best = candidate
        if best is None:
            return None
        _, _, _, recovery_truck_id, a, b = best
        return int(a[0]), float(a[1]), float(a[2]), str(recovery_truck_id), int(b[0]), float(b[1]), float(b[2])

    def _task_order_key(self, task) -> Tuple[int, float, int, str]:
        emergency_or_relay = bool(task.kind == TaskKind.EMERGENCY or self._is_relay(task))
        life_init = float(max(getattr(task, "lifeline_init", 1.0), 1e-9))
        life_current = float(max(getattr(task, "lifeline_current", life_init), 0.0))
        decay = float(max(getattr(task, "lifeline_decay_rate", 0.0), 0.0))
        remaining_lifeline_steps = (
            float(life_current / decay)
            if decay > 1e-9
            else float(getattr(task, "deadline_step", 10**9))
        )
        # Original emergency work is ahead of the multi-sortie relay of a
        # road-isolated normal task, as required by the planning design.
        priority = 0 if task.kind == TaskKind.EMERGENCY else (1 if emergency_or_relay else 2)
        if self._aggressive_planning_active:
            return (
                priority,
                remaining_lifeline_steps,
                int(task.deadline_step),
                str(task.task_id),
            )
        life_ratio = float(np.clip(life_current / life_init, 0.0, 1.0))
        return (priority, life_ratio, int(task.deadline_step), str(task.task_id))

    def _routine_disconnect_risk(self, env, task) -> float:
        """Estimate whether a direct routine node should be protected early.

        This is deliberately a planning signal rather than a post-blockage
        escape rule: once the only corridor is closed, a truck-only 800 kg
        task cannot be recovered under the frozen physical definition.
        """
        if (
            task.kind != TaskKind.NORMAL
            or self._is_relay(task)
            or not bool(
                getattr(
                    env.cfg,
                    "hrl_route_plan_routine_disconnect_protection_enabled",
                    True,
                )
            )
            or str(getattr(env.cfg, "scenario", "")).upper() not in {"B", "C"}
            or not bool(getattr(env.cfg, "blockage_curve_enabled", False))
        ):
            return 0.0
        node_id = int(task.demand_node)
        neighbors = tuple(int(nb) for nb in env.topology.adjacency.get(node_id, set()))
        if not neighbors:
            return 1.0
        edge_risks: List[float] = []
        for nb in neighbors:
            edge = env.topology.edge_attr(node_id, int(nb))
            base = float(np.clip(getattr(edge, "base_vulnerability", 0.0), 0.0, 1.0))
            learned = float(
                np.clip(
                    getattr(env.hazards, "last_edge_pstep", {}).get(
                        (min(node_id, int(nb)), max(node_id, int(nb))),
                        getattr(env.hazards, "last_pstep_mean", 0.0),
                    ),
                    0.0,
                    1.0,
                )
            )
            edge_risks.append(float(np.clip(0.65 * base + 0.35 * learned, 0.0, 1.0)))
        incident = float(0.60 * max(edge_risks) + 0.40 * np.mean(edge_risks))
        degree_fragility = float(1.0 / max(len(neighbors), 1))
        depot_road = float(env._decision_shortest_path_distance(0, node_id))
        map_size = float(max(getattr(env.cfg, "map_size_m", 1.0), 1.0))
        remoteness = float(np.clip(depot_road / map_size, 0.0, 1.0)) if np.isfinite(depot_road) else 1.0
        return float(np.clip(0.50 * incident + 0.30 * degree_fragility + 0.20 * remoteness, 0.0, 1.0))

    def _global_disconnect_profiles(
        self,
        env,
    ) -> Dict[str, Dict[str, float]]:
        """Predict routine isolation from the complete road cut structure.

        Edge capacity is a survival proxy: a vulnerable, infrastructure-
        bottlenecked or long edge contributes less redundancy.  The minimum
        fleet-to-task cut therefore captures remote bridge chains as well as
        weak two-corridor access, which a demand-node degree cannot see.
        """
        enabled = bool(
            getattr(
                env.cfg,
                "hrl_route_plan_global_disconnect_constraint_enabled",
                False,
            )
        )
        scenario = str(getattr(env.cfg, "scenario", "")).upper()
        if (
            not enabled
            or scenario not in {"B", "C"}
            or not bool(getattr(env.cfg, "blockage_curve_enabled", False))
        ):
            return {}
        routine_tasks = [
            task
            for task in env.state.tasks.values()
            if task.kind == TaskKind.NORMAL
            and self._active(task)
            and not self._is_relay(task)
        ]
        source_nodes = tuple(
            sorted(
                {
                    int(state.node)
                    for truck_id in self._live_trucks(env)
                    for state in (env.state.agents.get(str(truck_id), None),)
                    if state is not None and state.node is not None
                }
            )
        )
        cache_key: Tuple[Any, ...] = (
            id(env.state.tasks),
            self._road_signature(env),
            source_nodes,
            tuple(sorted(str(task.task_id) for task in routine_tasks)),
        )
        if self._disconnect_profile_cache_key == cache_key:
            return self._disconnect_profile_cache
        if not routine_tasks or not source_nodes:
            self._disconnect_profile_cache_key = cache_key
            self._disconnect_profile_cache = {}
            return {}

        blocked = set(self._road_signature(env))
        length_ref = float(
            max(
                getattr(
                    env.cfg,
                    "hrl_route_plan_global_disconnect_edge_length_ref_m",
                    1500.0,
                ),
                1.0,
            )
        )
        graph = nx.Graph()
        graph.add_nodes_from(int(node_id) for node_id in env.topology.nodes)
        for src, neighbors in env.topology.adjacency.items():
            for dst in neighbors:
                if int(src) >= int(dst):
                    continue
                edge_key = (min(int(src), int(dst)), max(int(src), int(dst)))
                if edge_key in blocked:
                    continue
                attr = env.topology.edge_attr(int(src), int(dst))
                vulnerability = float(
                    np.clip(getattr(attr, "base_vulnerability", 0.0), 0.0, 1.0)
                )
                infrastructure = float(
                    np.clip(getattr(attr, "infra_bottleneck_norm", 0.0), 0.0, 1.0)
                )
                building = float(
                    np.clip(getattr(attr, "building_density_norm", 0.0), 0.0, 1.0)
                )
                length = float(env.topology.edge_distance(int(src), int(dst)))
                length_norm = float(np.clip(length / length_ref, 0.0, 1.0))
                edge_failure_pressure = float(
                    np.clip(
                        0.45 * vulnerability
                        + 0.25 * infrastructure
                        + 0.15 * building
                        + 0.15 * length_norm,
                        0.0,
                        1.0,
                    )
                )
                graph.add_edge(
                    int(src),
                    int(dst),
                    capacity=float(max(1.0 - edge_failure_pressure, 0.05)),
                )

        super_source = -1
        while super_source in graph:
            super_source -= 1
        graph.add_node(super_source)
        for node_id in source_nodes:
            graph.add_edge(super_source, int(node_id), capacity=1.0e6)

        cut_by_task: Dict[str, float] = {}
        for task in routine_tasks:
            task_id = str(task.task_id)
            node_id = int(task.demand_node)
            try:
                if not nx.has_path(graph, super_source, node_id):
                    cut_value = 0.0
                else:
                    cut_value, _ = nx.minimum_cut(
                        graph,
                        super_source,
                        node_id,
                        capacity="capacity",
                    )
            except (nx.NetworkXError, nx.NetworkXUnbounded):
                cut_value = 0.0
            cut_by_task[task_id] = float(max(cut_value, 0.0))

        protected_fraction = float(
            np.clip(
                getattr(
                    env.cfg,
                    "hrl_route_plan_global_disconnect_protected_fraction",
                    0.50,
                ),
                0.0,
                1.0,
            )
        )
        protected_count = int(
            min(
                len(routine_tasks),
                max(np.ceil(protected_fraction * len(routine_tasks)), 0),
            )
        )
        protected_ids = {
            task_id
            for task_id, _ in sorted(
                cut_by_task.items(), key=lambda item: (item[1], item[0])
            )[:protected_count]
        }
        head_count = int(
            min(
                protected_count,
                max(
                    getattr(
                        env.cfg,
                        "hrl_route_plan_global_disconnect_head_commitment_count",
                        2,
                    ),
                    0,
                ),
            )
        )
        head_ids = {
            task_id
            for task_id, _ in sorted(
                cut_by_task.items(), key=lambda item: (item[1], item[0])
            )[:head_count]
        }
        complexity = str(getattr(env.cfg, "map_complexity", "M")).upper()
        tau = float(
            max(
                getattr(
                    env.cfg,
                    "blockage_tau_steps_L"
                    if complexity == "L"
                    else (
                        "blockage_tau_steps_R"
                        if complexity == "R"
                        else "blockage_tau_steps_M"
                    ),
                    130.0,
                ),
                1.0,
            )
        )
        safe_base = float(
            max(
                getattr(
                    env.cfg,
                    "hrl_route_plan_global_disconnect_safe_tau_base",
                    0.70,
                ),
                0.0,
            )
        )
        safe_cut_scale = float(
            max(
                getattr(
                    env.cfg,
                    "hrl_route_plan_global_disconnect_safe_tau_cut_scale",
                    0.45,
                ),
                0.0,
            )
        )
        profiles: Dict[str, Dict[str, float]] = {}
        for task in routine_tasks:
            task_id = str(task.task_id)
            cut_value = float(cut_by_task.get(task_id, 0.0))
            protected = bool(task_id in protected_ids)
            structural_deadline = int(
                np.ceil(tau * (safe_base + safe_cut_scale * min(cut_value, 2.0)))
            )
            profiles[task_id] = {
                "cut_capacity": cut_value,
                "protected": 1.0 if protected else 0.0,
                "head_protected": 1.0 if task_id in head_ids else 0.0,
                "safe_visit_step": float(
                    min(int(task.deadline_step), structural_deadline)
                    if protected
                    else int(task.deadline_step)
                ),
            }
        self._disconnect_profile_cache_key = cache_key
        self._disconnect_profile_cache = profiles
        self.disconnect_profile_evaluation_count += 1
        self.disconnect_protected_task_count = int(len(protected_ids))
        return profiles

    def _route_stop_specs(
        self,
        env,
        truck_id: str,
        task_ids: Sequence[str],
        cluster_uavs: Dict[str, Tuple[str, ...]],
        contracts: Dict[str, TaskContract],
        road_signature: Tuple[Tuple[int, int], ...],
    ) -> Tuple[List[RouteStop], float]:
        state = env.state.agents[str(truck_id)]
        if state.node is None:
            return [], float("inf")
        source = int(state.node)
        step_cursor = int(env.state.step_index)
        truck_speed = float(max(getattr(env.cfg, "truck_speed_mps", 10.0), 1e-6))
        uav_speed = float(max(getattr(env.cfg, "uav_max_speed_mps", 22.0), 1e-6))
        dt = float(max(getattr(env.cfg, "dt_seconds", 1.0), 1e-6))
        normal_late_w = float(
            max(getattr(env.cfg, "hrl_route_plan_normal_lateness_weight", 4.0), 0.0)
        )
        emergency_late_w = float(
            max(getattr(env.cfg, "hrl_route_plan_emergency_lateness_weight", 18.0), 0.0)
        )
        route_cost = 0.0
        stops: List[RouteStop] = []
        uav_ids = cluster_uavs.get(str(truck_id), ())
        uav_index = 0
        routine_budget_kg = float(
            max(getattr(state, "bulk_inventory_kg_current", 0.0), 0.0)
        )
        emergency_budget_units = self._cluster_emergency_capacity_units(
            env,
            str(truck_id),
            uav_ids,
        )
        # Keep the package budget observable, but do not freeze initial routes
        # to a truck-local quota. UAV recovery/rebinding can move a package
        # contract between cooperative units; authoritative material
        # conservation remains in the environment at reload/service time.
        enforce_emergency_route_inventory_budget = bool(
            getattr(
                env.cfg,
                "hrl_route_plan_enforce_emergency_inventory_budget_enabled",
                False,
            )
            and self._enforce_emergency_inventory_budget_active
        )
        # The v1 optimization trial enforced the whole suffix against the
        # current inventory.  That changed otherwise successful initial
        # routes too aggressively.  Inventory is now repaired locally only
        # after the owner actually becomes unable to continue a task.
        enforce_route_inventory_budget = bool(self._aggressive_planning_active)
        # Predict the UAV identity used by each aerial stop before evaluating
        # the route.  This lets the objective charge turnaround only when the
        # same aircraft is needed again; the already-modelled recovery flight
        # itself is never counted twice.
        predicted_uav_by_task: Dict[str, str] = {}
        preview_uav_index = 0
        for preview_task_id in task_ids:
            preview_task = env.state.tasks.get(str(preview_task_id), None)
            if preview_task is None or not self._active(preview_task):
                continue
            if not (
                preview_task.kind == TaskKind.EMERGENCY
                or self._is_relay(preview_task)
            ):
                continue
            preview_contract = contracts.get(str(preview_task_id), None)
            preview_uav = (
                preview_contract.uav_id
                if preview_contract is not None
                else (
                    uav_ids[preview_uav_index % len(uav_ids)]
                    if uav_ids
                    else None
                )
            )
            preview_uav_index += 1
            if preview_uav is not None:
                predicted_uav_by_task[str(preview_task_id)] = str(preview_uav)

        for task_position, task_id in enumerate(task_ids):
            task = env.state.tasks.get(str(task_id), None)
            if task is None or not self._active(task):
                continue
            relay = self._is_relay(task)
            is_air = bool(task.kind == TaskKind.EMERGENCY or relay)
            if is_air:
                if (
                    task.kind == TaskKind.EMERGENCY
                    and enforce_emergency_route_inventory_budget
                ):
                    if emergency_budget_units <= 0:
                        return [], float("inf")
                    emergency_budget_units -= 1
                contract = contracts.get(str(task_id), None)
                uav_id = (
                    contract.uav_id
                    if contract is not None
                    else (uav_ids[uav_index % len(uav_ids)] if uav_ids else None)
                )
                assigned_uavs = (
                    tuple(contract.uav_ids)
                    if contract is not None and contract.uav_ids
                    else ((str(uav_id),) if uav_id is not None else ())
                )
                uav_index += 1
                if uav_id is None:
                    return [], float("inf")
                anchors = self._anchor_nodes(
                    env, source, task, road_signature=road_signature
                )
                cross_pair = None
                # Cross-truck recovery is selected on *solo sortie
                # infeasibility*, not merely on absence of a geometric anchor.
                # A road-reachable launch point can still be too far for
                # A -> task -> A, while A -> task -> B is feasible.
                solo_nominal_feasible = False
                if anchors:
                    ax0, ay0 = self._xy(env, int(anchors[0]))
                    tx0, ty0 = self._xy(env, int(task.demand_node))
                    solo_air = float(np.hypot(ax0 - tx0, ay0 - ty0))
                    solo_nominal_feasible = bool(
                        2.0 * solo_air
                        <= 0.92 * float(max(getattr(env.cfg, "uav_max_sortie_m", 6000.0), 1.0))
                    )
                if task.kind == TaskKind.EMERGENCY and (not solo_nominal_feasible):
                    cross_pair = self._cross_truck_recovery_pair(
                        env, str(truck_id), task, cluster_uavs
                    )
                    if cross_pair is not None:
                        anchors = (int(cross_pair[0]),)
                        if contract is not None:
                            contract.recovery_truck_id = str(cross_pair[3])
                            contract.recovery_anchor_node = int(cross_pair[4])
                if not anchors:
                    return [], float("inf")
                anchor = int(anchors[0])
                road = float(
                    env._decision_shortest_path_distance(source, anchor)
                )
                ax, ay = self._xy(env, anchor)
                tx, ty = self._xy(env, int(task.demand_node))
                air = float(np.hypot(ax - tx, ay - ty))
                recovery_air = float(air)
                recovery_truck_id = None
                recovery_anchor = None
                recovery_road = 0.0
                if cross_pair is not None:
                    recovery_air = float(cross_pair[5])
                    recovery_truck_id = str(cross_pair[3])
                    recovery_anchor = int(cross_pair[4])
                    recovery_road = float(cross_pair[6])
                launch_chain_seconds = float(
                    road / truck_speed + (air + recovery_air) / uav_speed
                )
                recovery_ready_seconds = float(recovery_road / truck_speed)
                if recovery_truck_id is not None:
                    recovery_ready_seconds += float(
                        max(
                            (
                                int(max(getattr(active_task, "service_remaining", 0), 0))
                                for active_task in env.state.tasks.values()
                                if str(
                                    getattr(active_task, "in_service_by", "") or ""
                                )
                                == str(recovery_truck_id)
                            ),
                            default=0,
                        )
                        * dt
                    )
                travel_steps = int(
                    np.ceil(
                        max(launch_chain_seconds, recovery_ready_seconds) / dt
                    )
                )
                if recovery_truck_id is not None and self.plan is not None:
                    recovery_route = self.plan.routes.get(
                        str(recovery_truck_id), None
                    )
                    recovery_stop = (
                        recovery_route.current(env)
                        if recovery_route is not None
                        else None
                    )
                    recovery_state = env.state.agents.get(
                        str(recovery_truck_id), None
                    )
                    if (
                        recovery_stop is not None
                        and recovery_state is not None
                        and recovery_state.node is not None
                        and recovery_anchor is not None
                    ):
                        direct = float(
                            env._decision_shortest_path_distance(
                                int(recovery_state.node),
                                int(recovery_stop.target_node),
                            )
                        )
                        after_recovery = float(
                            env._decision_shortest_path_distance(
                                int(recovery_anchor),
                                int(recovery_stop.target_node),
                            )
                        )
                        if np.isfinite(direct) and np.isfinite(after_recovery):
                            recovery_detour_steps = float(
                                max(recovery_road + after_recovery - direct, 0.0)
                                / max(truck_speed * dt, 1e-6)
                            )
                            route_cost += recovery_detour_steps
                if relay:
                    relay_payload = float(max(getattr(env.cfg, "hrl_route_plan_bulk_relay_payload_kg", 100.0), 1e-6))
                    waves = int(max(1, np.ceil(float(getattr(task, "remaining_demand_kg", getattr(task, "demand_kg", relay_payload))) / max(relay_payload * len(assigned_uavs), 1e-6))))
                    # Truck reaches the anchor once; only the short UAV cycle repeats.
                    travel_steps = int(np.ceil((road / truck_speed + waves * 2.0 * air / uav_speed) / dt))
                unload = int(max(getattr(env.cfg, "unload_rounds_uav", 1), 1))
                step_cursor += int(travel_steps + unload)
                effective_deadline = (
                    self._effective_deadline_step(env, task)
                    if self._aggressive_planning_active
                    else int(task.deadline_step)
                )
                late = float(max(step_cursor - int(effective_deadline), 0))
                late_weight = emergency_late_w if task.kind == TaskKind.EMERGENCY else normal_late_w
                urgency = float(np.clip(getattr(task, "urgency_score", 0.5), 0.0, 1.0))
                route_cost += float(travel_steps + late_weight * (1.0 + urgency) * late)
                if (
                    task.kind == TaskKind.EMERGENCY
                    and bool(
                        getattr(
                            env.cfg,
                            "hrl_route_plan_uav_lifecycle_cost_enabled",
                            True,
                        )
                    )
                ):
                    next_same_task = None
                    for future_task_id in task_ids[int(task_position) + 1 :]:
                        future_task = env.state.tasks.get(str(future_task_id), None)
                        if (
                            future_task is not None
                            and future_task.kind == TaskKind.EMERGENCY
                            and predicted_uav_by_task.get(str(future_task_id), None)
                            == str(uav_id)
                        ):
                            next_same_task = future_task
                            break
                    if next_same_task is not None:
                        try:
                            energy_fraction = float(
                                max(
                                    env._uav_energy_cost_fraction(
                                        str(uav_id),
                                        float(air + recovery_air),
                                        (float(ax), float(ay)),
                                    ),
                                    0.0,
                                )
                            )
                        except Exception:
                            max_sortie = float(
                                max(
                                    getattr(env.cfg, "uav_max_sortie_m", 6000.0),
                                    1.0,
                                )
                            )
                            energy_fraction = float(
                                np.clip(
                                    (air + recovery_air) / max_sortie,
                                    0.0,
                                    1.0,
                                )
                            )
                        charge_rate = float(
                            max(
                                getattr(env.cfg, "uav_charge_rate_per_step", 0.085),
                                1e-6,
                            )
                        )
                        charge_steps = int(
                            max(np.ceil(energy_fraction / charge_rate), 0)
                        )
                        reload_steps = int(
                            max(
                                getattr(env.cfg, "uav_reload_service_steps", 1),
                                0,
                            )
                        )
                        turnaround_steps = int(max(charge_steps, reload_steps))
                        next_urgency = float(
                            np.clip(
                                getattr(next_same_task, "urgency_score", 0.5),
                                0.0,
                                1.0,
                            )
                        )
                        lifecycle_weight = float(
                            max(
                                getattr(
                                    env.cfg,
                                    "hrl_route_plan_uav_lifecycle_cost_weight",
                                    0.75,
                                ),
                                0.0,
                            )
                        )
                        lifecycle_cost = float(
                            lifecycle_weight
                            * turnaround_steps
                            * (1.0 + next_urgency)
                        )
                        route_cost += lifecycle_cost
                        self.lifecycle_turnaround_cost_evaluation_count += 1
                        self.lifecycle_turnaround_cost_total += lifecycle_cost
                stop_type = BULK_RELAY_LAUNCH if relay else EMERGENCY_LAUNCH
                stops.append(
                    RouteStop(
                        task_id=str(task_id),
                        stop_type=stop_type,
                        truck_id=str(truck_id),
                        uav_id=str(uav_id),
                        uav_ids=tuple(assigned_uavs),
                        target_node=int(anchor),
                        anchor_nodes=tuple(anchors),
                        selected_anchor=int(anchor),
                        planned_road_distance_m=float(road),
                        planned_air_distance_m=float(air),
                        eta_step=int(step_cursor),
                        deadline_step=int(effective_deadline),
                        service_mode=BULK_RELAY if relay else DIRECT,
                        recovery_truck_id=recovery_truck_id,
                        recovery_anchor_node=recovery_anchor,
                    )
                )
                source = int(anchor)
                continue

            remaining_kg = self._remaining_demand_kg(task)
            # Layer 1 currently has no explicit depot stop in a route.  A
            # direct-normal suffix that exceeds the truck's current stock is
            # therefore not executable and must be assigned to another truck
            # instead of being left locked behind an empty vehicle.
            if (
                enforce_route_inventory_budget
                and remaining_kg > routine_budget_kg + 1e-9
            ):
                # A head stop may consume the truck's last positive stock and
                # hand the residual to another truck on the next replan.  A
                # later stop may not silently rely on an unplanned depot trip.
                if stops or routine_budget_kg <= 1e-9:
                    return [], float("inf")
            if enforce_route_inventory_budget:
                routine_budget_kg = float(
                    max(routine_budget_kg - remaining_kg, 0.0)
                )
            road = float(
                env._decision_shortest_path_distance(source, int(task.demand_node))
            )
            if not np.isfinite(road):
                return [], float("inf")
            travel_steps = int(np.ceil((road / truck_speed) / dt))
            unload = int(max(getattr(env.cfg, "unload_rounds_normal", 1), 1))
            step_cursor += int(travel_steps + unload)
            effective_deadline = (
                self._effective_deadline_step(env, task)
                if self._aggressive_planning_active
                else int(task.deadline_step)
            )
            late = float(max(step_cursor - int(effective_deadline), 0))
            urgency = float(np.clip(getattr(task, "urgency_score", 0.5), 0.0, 1.0))
            # Proactive isolation term: it affects only B/C route order and
            # therefore cannot manufacture an A-scenario advantage.
            disconnect_risk = self._routine_disconnect_risk(env, task)
            risk_weight = float(
                max(
                    getattr(
                        env.cfg,
                        "hrl_route_plan_routine_disconnect_risk_weight",
                        1.20,
                    ),
                    0.0,
                )
            )
            protection_cost = float(
                risk_weight
                * disconnect_risk
                * max(step_cursor - int(env.state.step_index), 0)
            )
            route_cost += float(
                travel_steps
                + normal_late_w * (1.0 + urgency) * late
                + protection_cost
            )
            stops.append(
                RouteStop(
                    task_id=str(task_id),
                    stop_type=NORMAL_SERVICE,
                    truck_id=str(truck_id),
                    uav_id=None,
                    uav_ids=(),
                    target_node=int(task.demand_node),
                    planned_road_distance_m=float(road),
                    eta_step=int(step_cursor),
                    deadline_step=int(effective_deadline),
                    service_mode=DIRECT,
                )
            )
            source = int(task.demand_node)
        return stops, float(route_cost)

    def _single_emergency_eta(
        self,
        env,
        truck_id: str,
        task,
        cluster_uavs: Dict[str, Tuple[str, ...]],
        road_signature: Tuple[Tuple[int, int], ...],
        preferred: Optional[TaskContract] = None,
    ) -> Optional[Tuple[float, TaskContract, RouteStop]]:
        """Estimate a complete truck--UAV service chain from current state."""
        uavs = tuple(cluster_uavs.get(str(truck_id), ()))
        if not uavs:
            return None
        preferred_uav = None if preferred is None else preferred.uav_id
        uav_id = (
            str(preferred_uav)
            if preferred_uav is not None and str(preferred_uav) in uavs
            else str(uavs[0])
        )
        contract = TaskContract(
            task_id=str(task.task_id),
            owner_agent_id=str(uav_id),
            truck_id=str(truck_id),
            uav_id=str(uav_id),
            uav_ids=(str(uav_id),),
            service_mode=DIRECT,
            created_step=int(env.state.step_index),
        )
        stops, cost = self._route_stop_specs(
            env,
            str(truck_id),
            [str(task.task_id)],
            cluster_uavs,
            {str(task.task_id): contract},
            road_signature,
        )
        if not stops or not np.isfinite(cost):
            return None
        stop = stops[0]
        remaining = float(max(int(stop.eta_step) - int(env.state.step_index), 0))
        return remaining, contract, stop

    @staticmethod
    def _effective_deadline_step(env, task) -> int:
        """Use the tighter of the nominal deadline and remaining lifeline.

        Lifeline expiry is an independent failure condition in the environment,
        so using only ``deadline_step`` can make a route look feasible long
        after it has become physically impossible to finish.
        """
        step_now = int(env.state.step_index)
        nominal = int(getattr(task, "deadline_step", getattr(env.cfg, "max_steps", step_now)))
        decay = float(max(getattr(task, "lifeline_decay_rate", 0.0), 0.0))
        if decay <= 1e-9:
            return nominal
        remaining = float(max(getattr(task, "lifeline_current", 0.0), 0.0))
        lifeline_steps = int(max(np.floor(remaining / decay), 0))
        return int(min(nominal, step_now + lifeline_steps))

    def _deadline_risk_emergency_contracts_for_transfer(
        self,
        env,
        road_signature: Tuple[Tuple[int, int], ...],
    ) -> Dict[str, str]:
        """Release only future emergency stops that have become infeasible.

        This is intentionally not a general priority reshuffle.  A suffix is
        left untouched while it can still meet both deadline and lifeline.  It
        is released only when a different idle/non-emergency unit can restore a
        small positive reserve.
        """
        if self.plan is None or not bool(
            getattr(env.cfg, "hrl_route_plan_deadline_rescue_enabled", True)
        ):
            return {}
        step_now = int(env.state.step_index)
        reserve = int(
            max(getattr(env.cfg, "hrl_route_plan_deadline_rescue_reserve_steps", 6), 0)
        )
        cooldown = int(
            max(getattr(env.cfg, "hrl_route_plan_contract_transfer_cooldown_steps", 15), 0)
        )
        clusters = self._cluster_uavs(env, self._live_trucks(env))
        current_by_truck = {
            str(truck_id): route.current(env)
            for truck_id, route in self.plan.routes.items()
        }
        release: Dict[str, str] = {}
        for truck_id, route in sorted(self.plan.routes.items()):
            active_suffix_ids = [
                str(stop.task_id)
                for stop in route.stops[int(route.cursor) :]
                if (
                    env.state.tasks.get(str(stop.task_id), None) is not None
                    and self._active(env.state.tasks[str(stop.task_id)])
                )
            ]
            if len(active_suffix_ids) <= 1:
                continue
            suffix_stops, _ = self._route_stop_specs(
                env,
                str(truck_id),
                active_suffix_ids,
                clusters,
                self.plan.contracts,
                road_signature,
            )
            for suffix_index, stop in enumerate(suffix_stops):
                if suffix_index == 0:
                    continue
                task_id = str(stop.task_id)
                task = env.state.tasks.get(task_id, None)
                contract = self.plan.contracts.get(task_id, None)
                if (
                    task is None
                    or task.kind != TaskKind.EMERGENCY
                    or contract is None
                    or task.status == TaskStatus.CLAIMED
                ):
                    continue
                # An actual airborne sortie is authoritative and is never
                # interrupted by a future-queue repair.
                if any(
                    str(tid) == task_id
                    and env.state.agents.get(str(uid), None) is not None
                    and env.state.agents[str(uid)].follow_target is None
                    for uid, tid in dict(getattr(env, "_uav_sortie_contract_task", {})).items()
                ):
                    continue
                last_transfer = int(self._contract_last_transfer_step.get(task_id, -10**9))
                if step_now - last_transfer < cooldown:
                    continue
                if (
                    self._targeted_repairs_enabled()
                    and task_id in self._deadline_rescue_transferred_tasks
                ):
                    continue
                effective_deadline = self._effective_deadline_step(env, task)
                if int(stop.eta_step) <= effective_deadline - reserve:
                    continue

                best_alternative: Optional[Tuple[float, str]] = None
                for other_truck_id in sorted(clusters):
                    if str(other_truck_id) == str(truck_id):
                        continue
                    other_stop = current_by_truck.get(str(other_truck_id), None)
                    if other_stop is not None and str(other_stop.task_id) != task_id:
                        other_task = env.state.tasks.get(str(other_stop.task_id), None)
                        if other_task is not None and other_task.kind == TaskKind.EMERGENCY:
                            continue
                    alternative = self._single_emergency_eta(
                        env,
                        str(other_truck_id),
                        task,
                        clusters,
                        road_signature,
                    )
                    if alternative is None:
                        continue
                    alternative_remaining, _, alternative_stop = alternative
                    if int(alternative_stop.eta_step) > effective_deadline - reserve:
                        continue
                    candidate = (float(alternative_remaining), str(other_truck_id))
                    if best_alternative is None or candidate < best_alternative:
                        best_alternative = candidate
                if best_alternative is None:
                    continue
                release[task_id] = str(truck_id)
                if self._targeted_repairs_enabled():
                    self._deadline_rescue_transferred_tasks.add(task_id)
                self.deadline_rescue_promotion_count += 1
                self._contract_last_transfer_step[task_id] = int(step_now)
                self._feedback.append(
                    PlannerFeedback(
                        step=step_now,
                        reason="deadline_infeasible_suffix_promotion",
                        truck_id=str(truck_id),
                        task_id=task_id,
                        detail=(
                            f"planned_eta={int(stop.eta_step)},"
                            f"effective_deadline={effective_deadline},"
                            f"target_truck={best_alternative[1]}"
                        ),
                        suffix_repair_required=True,
                    )
                )
        return release

    def _residual_emergency_contracts_for_transfer(
        self,
        env,
        road_signature: Tuple[Tuple[int, int], ...],
    ) -> Dict[str, str]:
        """Immediately reassign a partially served TC task to a ready UAV.

        A UAV becomes empty after one delivery.  Keeping the residual demand
        locked to that UAV forces a recover/reload cycle even when another
        docked UAV can finish the task before its lifeline expires.
        """
        if not self._targeted_repairs_enabled() or self.plan is None:
            return {}
        # Disabled in the evaluated configuration: changing the owner here can
        # race the environment's authoritative sortie/reload state machine.
        # Residual delivery remains handled by the existing locked UAV cycle
        # until a state-machine-native handoff is implemented.
        # Residual repair is local and symptom-driven, so it remains available
        # even when global aggressive ordering is off.  The guards below make
        # it a one-shot rescue instead of a standing source of replans.
        enable_route_level_residual_handoff = True
        if not enable_route_level_residual_handoff:
            return {}
        clusters = self._cluster_uavs(env, self._live_trucks(env))
        current_by_truck = {
            str(truck_id): route.current(env)
            for truck_id, route in self.plan.routes.items()
        }
        release: Dict[str, str] = {}
        for task_id, contract in sorted(self.plan.contracts.items()):
            task = env.state.tasks.get(str(task_id), None)
            if task is None or task.kind != TaskKind.EMERGENCY or not self._active(task):
                continue
            fulfilled = float(max(getattr(task, "fulfilled_mass_kg", 0.0), 0.0))
            if fulfilled <= 1e-9 or self._remaining_demand_kg(task) <= 1e-9:
                continue
            owner = env.state.agents.get(str(contract.uav_id), None)
            owner_loaded = False
            if owner is not None:
                try:
                    owner_loaded = bool(env._uav_loaded_for_task(str(contract.uav_id), task))
                except Exception:
                    owner_loaded = bool(getattr(owner, "payload_kg_current", 0.0) > 1e-9)
            owner_airborne_on_task = bool(
                owner is not None
                and owner.follow_target is None
                and str(
                    dict(getattr(env, "_uav_sortie_contract_task", {})).get(
                        str(contract.uav_id), ""
                    )
                )
                == str(task_id)
            )
            if owner_loaded or owner_airborne_on_task:
                continue

            # A docked UAV can be reloaded by the authoritative environment
            # state machine and should retain its exclusive contract.  Moving
            # the contract here used to race that reload and was the main
            # source of the seed-112 contract oscillation.
            if (
                owner is not None
                and owner.follow_target is not None
                and str(owner.follow_target) == str(contract.truck_id)
            ):
                continue

            step_now = int(env.state.step_index)
            if str(task_id) in self._residual_handoff_last_step_by_task:
                continue
            residual_cooldown = int(
                max(
                    getattr(
                        env.cfg,
                        "hrl_route_plan_contract_transfer_cooldown_steps",
                        15,
                    ),
                    0,
                )
            )
            if (
                step_now
                - int(self._residual_handoff_last_step_by_task.get(str(task_id), -10**9))
                < residual_cooldown
            ):
                continue

            effective_deadline = self._effective_deadline_step(env, task)
            same_truck_best: Optional[Tuple[float, str]] = None
            enable_same_truck_residual_handoff = False
            same_truck_uavs = (
                tuple(clusters.get(str(contract.truck_id), ()))
                if enable_same_truck_residual_handoff
                else ()
            )
            for uav_id in sorted(
                (str(uid) for uid in same_truck_uavs),
                key=lambda uid: self._uav_contract_rank(
                    env, uid, task, str(contract.truck_id)
                ),
            ):
                if str(uav_id) == str(contract.uav_id):
                    continue
                candidate_state = env.state.agents.get(str(uav_id), None)
                if (
                    candidate_state is None
                    or candidate_state.follow_target is None
                    or str(candidate_state.follow_target) != str(contract.truck_id)
                ):
                    continue
                try:
                    loaded = bool(env._uav_loaded_for_task(str(uav_id), task))
                except Exception:
                    loaded = bool(
                        getattr(candidate_state, "payload_kg_current", 0.0) > 1e-9
                    )
                if not loaded:
                    continue
                preferred = TaskContract(
                    task_id=str(task_id),
                    owner_agent_id=str(uav_id),
                    truck_id=str(contract.truck_id),
                    uav_id=str(uav_id),
                    uav_ids=(str(uav_id),),
                    service_mode=DIRECT,
                    created_step=int(env.state.step_index),
                )
                alternative = self._single_emergency_eta(
                    env,
                    str(contract.truck_id),
                    task,
                    clusters,
                    road_signature,
                    preferred=preferred,
                )
                if alternative is None:
                    continue
                remaining_steps, _, candidate_stop = alternative
                if int(candidate_stop.eta_step) > int(effective_deadline):
                    continue
                candidate = (float(remaining_steps), str(uav_id))
                if same_truck_best is None or candidate < same_truck_best:
                    same_truck_best = candidate
            if same_truck_best is not None:
                new_uav_id = str(same_truck_best[1])
                old_uav_id = str(contract.uav_id)
                contract.owner_agent_id = new_uav_id
                contract.uav_id = new_uav_id
                contract.uav_ids = (new_uav_id,)
                contract.created_step = int(env.state.step_index)
                route = self.plan.routes.get(str(contract.truck_id), None)
                if route is not None:
                    for route_stop in route.stops[int(route.cursor) :]:
                        if str(route_stop.task_id) == str(task_id):
                            route_stop.uav_id = new_uav_id
                            route_stop.uav_ids = (new_uav_id,)
                            break
                task.route_contract_owner = new_uav_id
                task.route_contract_uav_ids = (new_uav_id,)
                self._stamp_contract_on_task(
                    env, str(task_id), contract, bump=True
                )
                self._contract_progress.pop(str(task_id), None)
                self.residual_emergency_handoff_count += 1
                self._feedback.append(
                    PlannerFeedback(
                        step=int(env.state.step_index),
                        reason="residual_emergency_same_truck_uav_handoff",
                        truck_id=str(contract.truck_id),
                        task_id=str(task_id),
                        detail=f"old_uav={old_uav_id},new_uav={new_uav_id}",
                    )
                )
                continue

            # The first trial used a global replan for residual handoff.  It
            # improved difficult seeds but regressed previously perfect ones
            # by disturbing unrelated routes.  Keep that implementation for
            # auditability, but let the existing deadline rescue own cross-
            # truck transfers.
            allow_cross_truck_residual_global_replan = True
            if not allow_cross_truck_residual_global_replan:
                continue
            best: Optional[Tuple[float, str, str]] = None
            for truck_id, uavs in sorted(clusters.items()):
                current = current_by_truck.get(str(truck_id), None)
                if current is not None and str(current.task_id) != str(task_id):
                    current_task = env.state.tasks.get(str(current.task_id), None)
                    if current_task is not None and current_task.kind == TaskKind.EMERGENCY:
                        continue
                for uav_id in sorted(
                    (str(uid) for uid in uavs),
                    key=lambda uid: self._uav_contract_rank(env, uid, task, str(truck_id)),
                ):
                    state = env.state.agents.get(str(uav_id), None)
                    if (
                        state is None
                        or state.follow_target is None
                        or str(state.follow_target) != str(truck_id)
                    ):
                        continue
                    try:
                        loaded = bool(env._uav_loaded_for_task(str(uav_id), task))
                    except Exception:
                        loaded = bool(getattr(state, "payload_kg_current", 0.0) > 1e-9)
                    if not loaded:
                        continue
                    preferred = TaskContract(
                        task_id=str(task_id),
                        owner_agent_id=str(uav_id),
                        truck_id=str(truck_id),
                        uav_id=str(uav_id),
                        uav_ids=(str(uav_id),),
                        service_mode=DIRECT,
                        created_step=int(env.state.step_index),
                    )
                    alternative = self._single_emergency_eta(
                        env,
                        str(truck_id),
                        task,
                        clusters,
                        road_signature,
                        preferred=preferred,
                    )
                    if alternative is None:
                        continue
                    remaining_steps, _, stop = alternative
                    if int(stop.eta_step) > int(effective_deadline):
                        continue
                    candidate = (float(remaining_steps), str(truck_id), str(uav_id))
                    if best is None or candidate < best:
                        best = candidate
                    break
            if best is None or best[2] == str(contract.uav_id):
                continue
            release[str(task_id)] = str(contract.truck_id)
            self._residual_handoff_last_step_by_task[str(task_id)] = int(step_now)
            self.residual_emergency_handoff_count += 1
            self._feedback.append(
                PlannerFeedback(
                    step=int(env.state.step_index),
                    reason="residual_emergency_ready_uav_handoff",
                    truck_id=str(contract.truck_id),
                    task_id=str(task_id),
                    detail=f"old_uav={contract.uav_id},new_uav={best[2]}",
                    suffix_repair_required=True,
                )
            )
        return release

    def _inventory_infeasible_routine_contracts(self, env) -> Dict[str, str]:
        """Release only a current routine task its owner cannot continue.

        Future suffixes remain untouched.  This preserves a successful
        initial route and activates rebalancing only after a real partial
        delivery or full stock depletion is observed.
        """
        if not self._targeted_repairs_enabled() or self.plan is None:
            return {}
        release: Dict[str, str] = {}
        live_trucks = self._live_trucks(env)
        for truck_id, route in sorted(self.plan.routes.items()):
            state = env.state.agents.get(str(truck_id), None)
            if state is None or state.node is None:
                continue
            stop = route.current(env)
            if stop is None:
                continue
            task = env.state.tasks.get(str(stop.task_id), None)
            if (
                task is None
                or not self._active(task)
                or task.kind != TaskKind.NORMAL
                or self._is_relay(task)
            ):
                continue
            if str(task.task_id) in self._routine_inventory_rebalanced_tasks:
                continue
            remaining = self._remaining_demand_kg(task)
            available = float(
                max(getattr(state, "bulk_inventory_kg_current", 0.0), 0.0)
            )
            fulfilled = float(max(getattr(task, "fulfilled_mass_kg", 0.0), 0.0))
            # Let the assigned truck finish using its positive stock.  A
            # contract is released only after the owner is actually empty;
            # otherwise repeated partial-stock comparisons churn the entire
            # route while the task is still serviceable.
            if available > 1e-9:
                continue
            if available >= remaining - 1e-9:
                continue
            alternative_exists = False
            for other_id in live_trucks:
                if str(other_id) == str(truck_id):
                    continue
                other = env.state.agents.get(str(other_id), None)
                if other is None or other.node is None:
                    continue
                other_stock = float(
                    max(getattr(other, "bulk_inventory_kg_current", 0.0), 0.0)
                )
                if other_stock <= 1e-9:
                    continue
                road = float(
                    env._decision_shortest_path_distance(
                        int(other.node), int(task.demand_node)
                    )
                )
                if np.isfinite(road):
                    alternative_exists = True
                    break
            if not alternative_exists:
                continue
            release[str(task.task_id)] = str(truck_id)
            self._routine_inventory_rebalanced_tasks.add(str(task.task_id))
            self.routine_inventory_rebalance_count += 1
            self._feedback.append(
                PlannerFeedback(
                    step=int(env.state.step_index),
                    reason="routine_inventory_rebalance",
                    truck_id=str(truck_id),
                    task_id=str(task.task_id),
                    detail=f"remaining_kg={remaining:.1f},available_kg={available:.1f}",
                    suffix_repair_required=True,
                )
            )
        return release

    def _risk_slack_routine_route_eta(
        self,
        env,
        truck_id: str,
        task_id: str,
        *,
        append: bool = False,
    ) -> float:
        """Estimate one routine ETA from current state and known roads.

        The route-stop evaluator is preferred because it includes the current
        route suffix, truck profile and any already-installed emergency
        anchors.  A direct shortest-road fallback keeps the candidate unit
        testable with a minimal environment and still uses the current truck
        state/transit endpoint rather than stale task metadata.
        """
        step_now = int(env.state.step_index)
        route = self.plan.routes.get(str(truck_id), None) if self.plan is not None else None
        if route is not None:
            suffix_ids = [
                str(stop.task_id)
                for stop in route.stops[int(route.cursor) :]
                if (
                    env.state.tasks.get(str(stop.task_id), None) is not None
                    and self._active(env.state.tasks[str(stop.task_id)])
                )
            ]
            if append and str(task_id) not in suffix_ids:
                suffix_ids.append(str(task_id))
            if str(task_id) in suffix_ids and suffix_ids:
                try:
                    clusters = self._cluster_uavs(env, self._live_trucks(env))
                    contracts = dict(self.plan.contracts) if self.plan is not None else {}
                    if str(task_id) not in contracts:
                        contracts[str(task_id)] = TaskContract(
                            task_id=str(task_id),
                            owner_agent_id=str(truck_id),
                            truck_id=str(truck_id),
                            uav_id=None,
                            uav_ids=(),
                            service_mode=DIRECT,
                            created_step=step_now,
                        )
                    specs, cost = self._route_stop_specs(
                        env,
                        str(truck_id),
                        suffix_ids,
                        clusters,
                        contracts,
                        self._road_signature(env),
                    )
                    if np.isfinite(cost):
                        for stop in specs:
                            if str(stop.task_id) == str(task_id):
                                return float(max(int(stop.eta_step), step_now))
                except Exception:
                    # The direct fallback below is deliberately conservative
                    # and keeps candidate diagnostics independent of optional
                    # topology/profile helpers in small test environments.
                    pass

        state = env.state.agents.get(str(truck_id), None)
        if state is None:
            return float("inf")
        source = getattr(state, "node", None)
        transit = getattr(state, "transit", None)
        if transit is not None and len(transit) >= 2:
            source = transit[1]
        if source is None:
            return float("inf")
        task = env.state.tasks.get(str(task_id), None)
        if task is None:
            return float("inf")
        distance = float(
            env._decision_shortest_path_distance(int(source), int(task.demand_node))
        )
        if not np.isfinite(distance):
            return float("inf")
        speed = float(max(getattr(env.cfg, "truck_speed_mps", 10.0), 1e-6))
        dt = float(max(getattr(env.cfg, "dt_seconds", 1.0), 1e-6))
        unload = int(max(getattr(env.cfg, "unload_rounds_normal", 1), 1))
        return float(step_now + np.ceil(distance / max(speed * dt, 1e-6)) + unload)

    def _risk_slack_routine_reserved_inventory_ok(
        self,
        env,
        truck_id: str,
        task_id: str,
    ) -> bool:
        """Check current bulk stock against the candidate route suffix."""
        state = env.state.agents.get(str(truck_id), None)
        if state is None:
            return False
        available = float(max(getattr(state, "bulk_inventory_kg_current", 0.0), 0.0))
        required = float(
            max(
                self._remaining_demand_kg(env.state.tasks[str(task_id)]),
                0.0,
            )
        )
        if self.plan is not None:
            route = self.plan.routes.get(str(truck_id), None)
            if route is not None:
                for stop in route.stops[int(route.cursor) :]:
                    other = env.state.tasks.get(str(stop.task_id), None)
                    if (
                        other is not None
                        and str(other.task_id) != str(task_id)
                        and other.kind == TaskKind.NORMAL
                        and self._active(other)
                        and not self._is_relay(other)
                    ):
                        required += self._remaining_demand_kg(other)
        return bool(available + 1e-9 >= required)

    def _risk_slack_routine_tc_safe_candidate(
        self,
        env,
        candidate_id: str,
        task_id: str,
        candidate_route,
    ) -> bool:
        """Allow a routine suffix on a truck with TC work only if safe.

        The first candidate implementation rejected every truck with an
        active emergency suffix.  That made cross-truck repair impossible in
        the middle of an episode. The published emergency stop ETAs are
        checked cheaply first; the downstream joint-corridor insertion guard
        then evaluates every trial position and rejects any actual TC
        worsening.
        """
        if candidate_route is None:
            return True
        try:
            # The candidate repair is append-only with respect to the
            # receiving truck's current suffix.  Therefore existing emergency
            # ETAs are unchanged; check their already-published route ETAs and
            # deadlines directly instead of rebuilding the full route twice
            # for every trigger/candidate pair.
            for stop in candidate_route.stops[int(candidate_route.cursor) :]:
                emergency = env.state.tasks.get(str(stop.task_id), None)
                if (
                    emergency is None
                    or emergency.kind != TaskKind.EMERGENCY
                    or not self._active(emergency)
                ):
                    continue
                base_eta = int(getattr(stop, "eta_step", env.state.step_index))
                deadline = int(self._effective_deadline_step(env, emergency))
                if base_eta > deadline:
                    return False
            return True
        except Exception:
            # Candidate-only logic must fail closed: a broken optional
            # prediction cannot weaken emergency protection.
            return False

    def _risk_slack_routine_cross_truck_repairs(self, env) -> Dict[str, str]:
        """Release only risky, future NORMAL suffixes for cross-truck repair.

        This candidate mechanism is opt-in.  It never touches emergency
        contracts, claimed/in-service/airborne work or the executed route
        prefix.  The returned mapping is consumed by ``plan_or_repair`` and
        the old owner is temporarily forbidden during insertion, ensuring a
        release cannot silently become a same-truck replan.
        """
        if self.plan is None or not bool(
            getattr(env.cfg, "hrl_route_plan_risk_slack_routine_repair_enabled", False)
        ):
            return {}

        step_now = int(env.state.step_index)
        slack_steps = int(
            max(getattr(env.cfg, "hrl_route_plan_risk_slack_routine_slack_steps", 20), 0)
        )
        stall_steps = int(
            max(getattr(env.cfg, "hrl_route_plan_risk_slack_routine_stall_steps", 12), 1)
        )
        max_transfers = int(
            max(getattr(env.cfg, "hrl_route_plan_risk_slack_routine_max_transfers", 1), 0)
        )
        min_gain_steps = float(
            max(getattr(env.cfg, "hrl_route_plan_risk_slack_routine_eta_gain_steps", 3.0), 0.0)
        )
        min_gain_ratio = float(
            np.clip(
                getattr(env.cfg, "hrl_route_plan_risk_slack_routine_eta_gain_ratio", 0.20),
                0.0,
                1.0,
            )
        )
        radius_m = float(
            max(getattr(env.cfg, "hrl_route_plan_risk_slack_routine_radius_m", 800.0), 0.0)
        )
        reserved_guard = bool(
            getattr(
                env.cfg,
                "hrl_route_plan_risk_slack_routine_reserved_inventory_guard_enabled",
                False,
            )
        )
        release: Dict[str, str] = {}
        live_trucks = self._live_trucks(env)
        # Reuse the known-road disconnect profile when available.  Protected
        # routine tasks receive the tighter safe-visit deadline; ordinary
        # tasks retain their nominal/effective deadline.
        try:
            disconnect_profiles = self._global_disconnect_profiles(env)
        except Exception:
            disconnect_profiles = {}
        sortie_tasks = {
            str(task_id)
            for task_id in dict(getattr(env, "_uav_sortie_contract_task", {})).values()
            if task_id is not None
        }

        for truck_id, route in sorted(self.plan.routes.items()):
            suffix_stops = route.stops[int(route.cursor) :]
            for stop in suffix_stops:
                task_id = str(stop.task_id)
                task = env.state.tasks.get(task_id, None)
                self.risk_slack_routine_candidate_count += 1
                # ``route.cursor`` has already removed the executed prefix;
                # pending current/future stops are considered below and are
                # protected individually when claimed, in service, airborne,
                # or partially executed.
                if (
                    task is None
                    or task.kind != TaskKind.NORMAL
                    or not self._active(task)
                    or self._is_relay(task)
                ):
                    continue
                contract = self.plan.contracts.get(task_id, None)
                owner_id = str(
                    getattr(task, "route_contract_truck", "")
                    or (getattr(contract, "truck_id", "") if contract is not None else "")
                    or truck_id
                )
                if owner_id != str(truck_id):
                    owner_id = str(truck_id)
                if task.status == TaskStatus.CLAIMED or bool(
                    getattr(task, "in_service_by", None)
                ) or int(max(getattr(task, "service_remaining", 0), 0)) > 0:
                    self.risk_slack_routine_protected_count += 1
                    continue
                if getattr(task, "first_service_step", None) is not None or float(
                    max(getattr(task, "fulfilled_mass_kg", 0.0), 0.0)
                ) > 1e-9:
                    self.risk_slack_routine_protected_count += 1
                    continue
                if task_id in sortie_tasks:
                    self.risk_slack_routine_protected_count += 1
                    continue
                if any(
                    (
                        bool(getattr(env.state.agents.get(str(uid), None), "airborne", False))
                        or (
                            env.state.agents.get(str(uid), None) is not None
                            and getattr(
                                env.state.agents.get(str(uid), None), "kind", None
                            ) == AgentKind.UAV
                            and getattr(
                                env.state.agents.get(str(uid), None), "follow_target", "__missing__"
                            ) is None
                        )
                    )
                    for uid in tuple(getattr(contract, "uav_ids", ()) if contract else ())
                ):
                    self.risk_slack_routine_protected_count += 1
                    continue
                if int(self._risk_slack_routine_transfer_count_by_task.get(task_id, 0)) >= max_transfers:
                    continue

                current_eta = self._risk_slack_routine_route_eta(
                    env, owner_id, task_id, append=False
                )
                record = self._risk_slack_routine_progress.get(task_id, None)
                if record is None or str(record.get("owner_id", "")) != owner_id:
                    record = {
                        "owner_id": owner_id,
                        "best_eta": float(current_eta),
                        # The first observation establishes the current-state
                        # baseline; only subsequent unchanged observations can
                        # satisfy the stall window.
                        "last_improve_step": int(step_now),
                    }
                    self._risk_slack_routine_progress[task_id] = record
                elif current_eta < float(record.get("best_eta", float("inf"))) - 0.5:
                    record["best_eta"] = float(current_eta)
                    record["last_improve_step"] = int(step_now)
                stalled = bool(
                    step_now - int(record.get("last_improve_step", step_now)) >= stall_steps
                )
                unreachable = not np.isfinite(current_eta)
                deadline = int(self._effective_deadline_step(env, task))
                profile = disconnect_profiles.get(task_id, {})
                safe_visit = profile.get("safe_visit_step", deadline)
                try:
                    deadline = int(min(deadline, int(np.floor(float(safe_visit)))))
                except (TypeError, ValueError, OverflowError):
                    pass
                low_slack = bool(
                    np.isfinite(current_eta)
                    and deadline - int(np.ceil(current_eta)) <= slack_steps
                )
                if not (unreachable or low_slack or stalled):
                    continue
                self.risk_slack_routine_trigger_count += 1
                if unreachable:
                    self.risk_slack_routine_unreachable_count += 1
                if stalled:
                    self.risk_slack_routine_stalled_count += 1

                best: Optional[Tuple[float, float, str]] = None
                for candidate_id in live_trucks:
                    candidate_id = str(candidate_id)
                    if candidate_id == owner_id:
                        self.risk_slack_routine_same_truck_block_count += 1
                        continue
                    candidate_state = env.state.agents.get(candidate_id, None)
                    candidate_route = self.plan.routes.get(candidate_id, None)
                    if candidate_state is None or candidate_state.node is None:
                        continue
                    if candidate_route is not None:
                        candidate_suffix_tasks = [
                            env.state.tasks.get(str(item.task_id), None)
                            for item in candidate_route.stops[int(candidate_route.cursor) :]
                        ]
                        if any(
                            item is not None
                            and item.kind == TaskKind.EMERGENCY
                            and self._active(item)
                            for item in candidate_suffix_tasks
                        ) and not self._risk_slack_routine_tc_safe_candidate(
                            env,
                            candidate_id,
                            task_id,
                            candidate_route,
                        ):
                            self.risk_slack_routine_tc_guard_block_count += 1
                            continue
                    candidate_source = getattr(candidate_state, "node", None)
                    candidate_transit = getattr(candidate_state, "transit", None)
                    if candidate_transit is not None and len(candidate_transit) >= 2:
                        candidate_source = candidate_transit[1]
                    if candidate_source is None:
                        continue
                    distance = float(
                        env._decision_shortest_path_distance(
                            int(candidate_source), int(task.demand_node)
                        )
                    )
                    if not np.isfinite(distance):
                        continue
                    if radius_m > 0.0 and distance > radius_m + 1e-9:
                        continue
                    if reserved_guard and not self._risk_slack_routine_reserved_inventory_ok(
                        env, candidate_id, task_id
                    ):
                        self.risk_slack_routine_reserved_inventory_block_count += 1
                        continue
                    if not reserved_guard and float(
                        max(getattr(candidate_state, "bulk_inventory_kg_current", 0.0), 0.0)
                    ) + 1e-9 < self._remaining_demand_kg(task):
                        self.risk_slack_routine_reserved_inventory_block_count += 1
                        continue
                    candidate_eta = self._risk_slack_routine_route_eta(
                        env, candidate_id, task_id, append=True
                    )
                    if not np.isfinite(candidate_eta) or candidate_eta > float(deadline) + 1e-9:
                        self.risk_slack_routine_eta_guard_block_count += 1
                        continue
                    gain = float(current_eta - candidate_eta)
                    if np.isfinite(current_eta):
                        ratio = gain / max(float(current_eta - step_now), 1.0)
                        if gain < min_gain_steps - 1e-9 or ratio < min_gain_ratio - 1e-9:
                            self.risk_slack_routine_eta_guard_block_count += 1
                            continue
                    option = (float(candidate_eta), float(distance), candidate_id)
                    if best is None or option < best:
                        best = option
                if best is None:
                    continue
                release[task_id] = owner_id
                self._risk_slack_routine_transfer_count_by_task[task_id] = int(
                    self._risk_slack_routine_transfer_count_by_task.get(task_id, 0) + 1
                )
                self.risk_slack_routine_release_count += 1
                self.risk_slack_routine_cross_truck_repair_count += 1
                self._feedback.append(
                    PlannerFeedback(
                        step=step_now,
                        reason="risk_slack_routine_cross_truck_repair",
                        truck_id=owner_id,
                        task_id=task_id,
                        detail=(
                            f"current_eta={current_eta:.1f},candidate_eta={best[0]:.1f},"
                            f"target_truck={best[2]},low_slack={int(low_slack)},"
                            f"unreachable={int(unreachable)},stalled={int(stalled)}"
                        ),
                        suffix_repair_required=True,
                    )
                )
        return release

    def _stalled_normal_cleanup_contracts(self, env) -> Dict[str, str]:
        """Rebuild routine ownership only after emergency work is terminal."""
        # The broad pilot cleanup could steal trucks that still had executable
        # route suffixes and regressed a protected seed.  The active repair is
        # deliberately narrower: it runs only after emergency work is
        # terminal and only recruits stocked trucks whose current route is
        # already empty.
        enable_post_emergency_cleanup_experiment = bool(
            getattr(
                env.cfg,
                "hrl_route_plan_stalled_normal_cleanup_enabled",
                False,
            )
        )
        if not enable_post_emergency_cleanup_experiment:
            return {}
        # The low-seed candidate is deliberately narrower than a general
        # preemption policy: it may run while emergency tasks are pending only
        # when no emergency has been claimed/started and no UAV is airborne.
        # This keeps every physical execution prefix and safety gate intact.
        preemptive_low_seed = bool(er_hlns_low_seed_rescue_active(env))
        if self._post_emergency_cleanup_done:
            return {}
        step_now = int(env.state.step_index)
        active_emergency = any(
            task.kind == TaskKind.EMERGENCY
            and self._active(task)
            and float(getattr(task, "lifeline_current", 1.0)) > 1e-9
            for task in env.state.tasks.values()
        )
        active_normal_ids = {
            str(task.task_id)
            for task in env.state.tasks.values()
            if task.kind == TaskKind.NORMAL and self._active(task)
        }
        active_count = int(len(active_normal_ids))
        previous_count = self._normal_cleanup_last_active_count
        if previous_count is None or active_count < int(previous_count):
            self._normal_cleanup_last_progress_step = int(step_now)
        self._normal_cleanup_last_active_count = int(active_count)
        if active_emergency and preemptive_low_seed:
            protected_emergency = any(
                task.kind == TaskKind.EMERGENCY
                and self._active(task)
                and (
                    task.status == TaskStatus.CLAIMED
                    or getattr(task, "first_service_step", None) is not None
                )
                for task in env.state.tasks.values()
            )
            airborne_uav = any(
                getattr(state, "follow_target", None) is None
                and not bool(getattr(state, "crashed", False))
                for aid, state in env.state.agents.items()
                if str(aid).startswith("uav_")
            )
            if protected_emergency or airborne_uav:
                return {}
        if ((active_emergency and not preemptive_low_seed)
                or active_count <= 0
                or self.plan is None):
            if active_emergency or active_count <= 0:
                self._normal_cleanup_owner_by_task.clear()
            return {}
        stall_steps = int(
            max(
                2
                * int(
                    max(
                        getattr(env.cfg, "hrl_route_plan_contract_stall_steps", 12),
                        1,
                    )
                ),
                18 if preemptive_low_seed else 24,
            )
        )
        normal_stalled = bool(
            step_now - int(self._normal_cleanup_last_progress_step) >= stall_steps
        )
        if active_emergency and not normal_stalled:
            return {}
        if step_now - int(self._normal_cleanup_last_replan_step) < stall_steps:
            return {}
        candidate_tasks = {
            str(task.task_id): task
            for task in env.state.tasks.values()
            if (
                task.kind == TaskKind.NORMAL
                and self._active(task)
                and not self._is_relay(task)
                and getattr(task, "first_service_step", None) is None
                and (not preemptive_low_seed or task.status == TaskStatus.PENDING)
            )
        }
        available_trucks = {}
        previous_goals = dict(
            getattr(env, "_planner_route_plan_goals", {}) or {}
        )
        previous_assists = dict(
            getattr(env, "_planner_truck_assist_waypoint_by_truck", {}) or {}
        )
        for truck_id in self._live_trucks(env):
            route = self.plan.routes.get(str(truck_id), None)
            route_is_empty = bool(
                route is None or route.current(env) is None
            )
            execution_is_idle = bool(
                previous_goals.get(str(truck_id), None) is None
                and str(truck_id) not in previous_assists
            )
            if not (route_is_empty or execution_is_idle):
                continue
            if preemptive_low_seed:
                # Do not borrow a truck that still owns an active emergency
                # suffix, even if its current cursor happens to be empty.
                owns_active_emergency = any(
                    str(contract.truck_id) == str(truck_id)
                    and (
                        env.state.tasks.get(str(task_id)) is not None
                        and env.state.tasks[str(task_id)].kind == TaskKind.EMERGENCY
                        and self._active(env.state.tasks[str(task_id)])
                    )
                    for task_id, contract in self.plan.contracts.items()
                )
                if owns_active_emergency:
                    continue
            truck_state = env.state.agents[str(truck_id)]
            if float(
                max(
                    getattr(
                        truck_state,
                        "bulk_inventory_kg_current",
                        0.0,
                    ),
                    0.0,
                )
            ) <= 1e-9:
                continue
            available_trucks[str(truck_id)] = truck_state
        cleanup_owner: Dict[str, str] = {}
        max_preemptive_transfers = 2 if preemptive_low_seed else 10**9
        while candidate_tasks and available_trucks and len(cleanup_owner) < max_preemptive_transfers:
            best: Optional[Tuple[int, float, str, str]] = None
            for truck_id, truck_state in sorted(available_trucks.items()):
                if truck_state.node is None:
                    continue
                stock = float(
                    max(getattr(truck_state, "bulk_inventory_kg_current", 0.0), 0.0)
                )
                for task_id, task in sorted(candidate_tasks.items()):
                    if preemptive_low_seed:
                        old_contract = self.plan.contracts.get(str(task_id))
                        if old_contract is not None and str(old_contract.truck_id) == str(truck_id):
                            continue
                    road = float(
                        env._decision_shortest_path_distance(
                            int(truck_state.node), int(task.demand_node)
                        )
                    )
                    if not np.isfinite(road):
                        continue
                    full_finish = int(stock + 1e-9 < self._remaining_demand_kg(task))
                    candidate = (full_finish, road, str(truck_id), str(task_id))
                    if best is None or candidate < best:
                        best = candidate
            if best is None:
                break
            _, _, truck_id, task_id = best
            cleanup_owner[str(task_id)] = str(truck_id)
            candidate_tasks.pop(str(task_id), None)
            available_trucks.pop(str(truck_id), None)
        if not cleanup_owner:
            return {}
        self._normal_cleanup_owner_by_task = dict(cleanup_owner)
        release: Dict[str, str] = {}
        for task_id, new_truck_id in cleanup_owner.items():
            contract = self.plan.contracts.get(str(task_id), None)
            if contract is not None:
                release[str(task_id)] = str(contract.truck_id)
            task = env.state.tasks.get(str(task_id), None)
            if task is not None:
                task.route_contract_owner = str(new_truck_id)
                task.route_contract_truck = str(new_truck_id)
                task.route_contract_uav_ids = ()
        self._normal_cleanup_last_replan_step = int(step_now)
        self._normal_cleanup_last_progress_step = int(step_now)
        self._post_emergency_cleanup_done = True
        self.normal_cleanup_replan_count += 1
        self._feedback.append(
            PlannerFeedback(
                step=step_now,
                reason="post_emergency_stalled_normal_cleanup",
                detail=f"matched={len(cleanup_owner)},released={len(release)}",
                suffix_repair_required=True,
            )
        )
        return release

    def _hard_normal_coverage_rescue(
        self,
        env,
        goals: Dict[str, Optional[str]],
    ) -> None:
        """Rebind a bounded set of stalled/orphaned NORMAL contracts.

        This candidate-only rescue never removes a claimed, in-service,
        airborne, emergency, or recovery prefix.  It only moves a pending
        direct NORMAL stop to the cursor of the nearest stocked truck whose
        road path is finite; each task may be transferred once per episode.
        """
        if self.plan is None or not bool(
            getattr(env.cfg, "hrl_route_plan_hard_normal_rescue_enabled", False)
        ):
            return
        adaptive_gate = bool(
            getattr(
                env.cfg,
                "hrl_route_plan_hard_normal_rescue_adaptive_coverage_enabled",
                False,
            )
        )
        min_orphan_pending = int(
            max(
                getattr(
                    env.cfg,
                    "hrl_route_plan_hard_normal_rescue_min_orphan_pending",
                    0,
                ),
                0,
            )
        )
        if adaptive_gate and min_orphan_pending > 0:
            orphan_pending_count = 0
            for pending_task in env.state.tasks.values():
                if (
                    pending_task.kind == TaskKind.NORMAL
                    and pending_task.status == TaskStatus.PENDING
                    and not self._is_relay(pending_task)
                    # Only unassigned NORMAL work opens this adaptive gate.
                    # A stale route_contract_truck marker by itself must not
                    # turn a protected task into a rescue trigger.
                    and getattr(pending_task, "assigned_to", None) is None
                    and getattr(pending_task, "first_service_step", None) is None
                    and getattr(pending_task, "in_service_by", None) is None
                    and int(max(getattr(pending_task, "service_remaining", 0), 0)) <= 0
                    and float(max(getattr(pending_task, "fulfilled_mass_kg", 0.0), 0.0))
                    <= 1e-9
                    and not tuple(getattr(pending_task, "route_contract_uav_ids", ()) or ())
                ):
                    # A stale route/assignment marker does not make a NORMAL
                    # task executable.  Count it as orphaned whenever its
                    # advertised owner is absent, has no live route/current
                    # stop, or no longer advertises this task as its goal.
                    # This deliberately treats stale assignments as rescue
                    # candidates while leaving valid executable ownership
                    # outside the adaptive threshold.
                    owner_id = str(
                        getattr(pending_task, "route_contract_truck", "") or ""
                    )
                    owner_route = self.plan.routes.get(owner_id, None) if owner_id else None
                    owner_current = (
                        owner_route.current(env) if owner_route is not None else None
                    )
                    owner_goal = str(goals.get(owner_id, "") or "") if owner_id else ""
                    if (
                        not owner_id
                        or owner_route is None
                        or owner_current is None
                        or owner_goal != str(pending_task.task_id)
                        or str(owner_current.task_id) != str(pending_task.task_id)
                    ):
                        orphan_pending_count += 1
            if orphan_pending_count < min_orphan_pending:
                return
        max_per_call = int(
            max(
                getattr(env.cfg, "hrl_route_plan_hard_normal_rescue_max_per_call", 2),
                0,
            )
        )
        if max_per_call <= 0:
            return
        step_now = int(getattr(env.state, "step_index", 0))
        stall_steps = int(
            max(
                getattr(env.cfg, "hrl_route_plan_hard_normal_rescue_stall_steps", 12),
                1,
            )
        )
        orphan_only = bool(
            getattr(
                env.cfg,
                "hrl_route_plan_hard_normal_rescue_orphan_only_enabled",
                False,
            )
        )
        pending_head_guard = bool(
            getattr(
                env.cfg,
                "hrl_route_plan_hard_normal_rescue_pending_head_guard_enabled",
                False,
            )
        )
        candidate_head_guard = bool(
            getattr(
                env.cfg,
                "hrl_route_plan_hard_normal_rescue_candidate_head_guard_enabled",
                False,
            )
        )
        no_truck_once = bool(
            getattr(
                env.cfg,
                "hrl_route_plan_hard_normal_rescue_no_truck_once_enabled",
                False,
            )
        )
        no_truck_cooldown = bool(
            getattr(
                env.cfg,
                "hrl_route_plan_hard_normal_rescue_no_truck_cooldown_enabled",
                False,
            )
        )
        no_truck_cooldown_steps = int(
            max(
                getattr(
                    env.cfg,
                    "hrl_route_plan_hard_normal_rescue_no_truck_cooldown_steps",
                    24,
                ),
                1,
            )
        )
        active_sorties = {
            str(task_id)
            for task_id in dict(
                getattr(env, "_uav_sortie_contract_task", {})
            ).values()
            if task_id is not None
        }
        recovery_trucks = {
            str(truck_id)
            for truck_id, assist in self._assist_by_truck.items()
            if str(getattr(assist, "get", lambda *_: "")("service_mode", "")).upper()
            in {"SAFETY_RECOVERY", "CROSS_TRUCK_RECOVERY"}
        }
        emergency_trucks = set()
        airborne_emergency_trucks = set()
        sortie_contracts = dict(getattr(env, "_uav_sortie_contract_task", {}))
        for task_id, contract in self.plan.contracts.items():
            emergency = env.state.tasks.get(str(task_id), None)
            if (
                emergency is None
                or emergency.kind != TaskKind.EMERGENCY
                or not self._active(emergency)
            ):
                continue
            truck_id = str(getattr(contract, "truck_id", "") or "")
            if not truck_id:
                continue
            airborne = False
            for uav_id in tuple(getattr(contract, "uav_ids", ()) or ()) + (
                (str(contract.uav_id),) if getattr(contract, "uav_id", None) else ()
            ):
                uav = env.state.agents.get(str(uav_id), None)
                if (
                    uav is not None
                    and not bool(getattr(uav, "crashed", False))
                    and getattr(uav, "follow_target", None) is None
                    and str(sortie_contracts.get(str(uav_id), "") or "")
                    == str(task_id)
                ):
                    airborne = True
                    break
            if airborne:
                airborne_emergency_trucks.add(truck_id)
            else:
                emergency_trucks.add(truck_id)
        transferred = 0
        for task in sorted(env.state.tasks.values(), key=lambda item: str(item.task_id)):
            if transferred >= max_per_call:
                break
            if (
                task.kind != TaskKind.NORMAL
                or task.status != TaskStatus.PENDING
                or self._is_relay(task)
                or getattr(task, "first_service_step", None) is not None
                # ``assigned_to`` is only a stale ownership marker here.  A
                # PENDING task that has never started service may be safely
                # rebound; claimed/in-service/airborne prefixes are protected
                # by the status/service checks below.
                or getattr(task, "in_service_by", None) is not None
                or int(max(getattr(task, "service_remaining", 0), 0)) > 0
                or float(max(getattr(task, "fulfilled_mass_kg", 0.0), 0.0)) > 1e-9
                or tuple(getattr(task, "route_contract_uav_ids", ()) or ())
                or str(task.task_id) in active_sorties
                or int(self._hard_normal_rescue_count_by_task.get(str(task.task_id), 0))
                >= 1
            ):
                continue
            task_id = str(task.task_id)
            last_no_truck_step = self._hard_normal_rescue_no_truck_last_step_by_task.get(
                task_id
            )
            if no_truck_once and last_no_truck_step is not None:
                self.hard_normal_rescue_no_truck_skip_count += 1
                continue
            if (
                no_truck_cooldown
                and last_no_truck_step is not None
                and step_now - int(last_no_truck_step) < no_truck_cooldown_steps
            ):
                self.hard_normal_rescue_no_truck_skip_count += 1
                continue
            self.hard_normal_rescue_candidate_count += 1
            contract = self.plan.contracts.get(task_id, None)
            owner_id = str(
                getattr(task, "route_contract_truck", "")
                or getattr(contract, "truck_id", "")
                or ""
            )
            owner_route = self.plan.routes.get(owner_id, None)
            owner_state = env.state.agents.get(owner_id, None)
            owner_current = (
                owner_route.current(env) if owner_route is not None else None
            )
            owner_goal = str(goals.get(owner_id, "") or "")
            executable = bool(
                owner_state is not None
                and owner_route is not None
                and owner_current is not None
                and str(owner_current.task_id) == task_id
                and owner_goal == task_id
            )
            if owner_id in recovery_trucks or owner_id in emergency_trucks:
                executable = False
            no_goal = not executable
            owner_current_task = (
                env.state.tasks.get(str(owner_current.task_id), None)
                if owner_current is not None
                else None
            )
            owner_current_pending_normal = bool(
                owner_current_task is not None
                and owner_current_task.kind == TaskKind.NORMAL
                and owner_current_task.status == TaskStatus.PENDING
                and getattr(owner_current_task, "in_service_by", None) is None
                and str(owner_current_task.task_id) not in active_sorties
            )
            if pending_head_guard and owner_current_pending_normal:
                self.hard_normal_rescue_no_truck_skip_count += 1
                continue
            if orphan_only and not no_goal:
                continue
            stalled = False
            if executable:
                transit = getattr(owner_state, "transit", None)
                try:
                    transit_key = tuple(round(float(value), 3) for value in transit) if transit is not None else ()
                except Exception:
                    transit_key = (str(transit),)
                position = (
                    getattr(owner_state, "node", None),
                    transit_key,
                    int(getattr(owner_route, "cursor", 0)),
                    owner_goal,
                )
                record = self._hard_normal_rescue_progress.get(task_id, None)
                if record is None or str(record.get("owner_id", "")) != owner_id:
                    self._hard_normal_rescue_progress[task_id] = {
                        "owner_id": owner_id,
                        "position": position,
                        "last_progress_step": step_now,
                    }
                    continue
                if record.get("position") != position:
                    record["position"] = position
                    record["last_progress_step"] = step_now
                    continue
                stalled = bool(
                    step_now - int(record.get("last_progress_step", step_now))
                    >= stall_steps
                )
            if not no_goal and not stalled:
                continue
            if no_goal:
                self.hard_normal_rescue_no_goal_count += 1
            else:
                self.hard_normal_rescue_stalled_owner_count += 1

            alternatives: List[Tuple[float, str]] = []
            demand = self._remaining_demand_kg(task)
            for candidate_id in self._live_trucks(env):
                candidate_id = str(candidate_id)
                candidate_state = env.state.agents.get(candidate_id, None)
                candidate_route = self.plan.routes.get(candidate_id, None)
                candidate_current = (
                    candidate_route.current(env)
                    if candidate_route is not None
                    else None
                )
                candidate_current_task = (
                    env.state.tasks.get(str(candidate_current.task_id), None)
                    if candidate_current is not None
                    else None
                )
                candidate_orphan_emergency = bool(
                    candidate_id in emergency_trucks
                    and not any(
                        str(task_id) in active_sorties
                        and str(
                            getattr(self.plan.contracts.get(str(task_id), None), "truck_id", "")
                            or ""
                        )
                        == candidate_id
                        for task_id in self.plan.contracts
                    )
                    and (
                        candidate_current_task is None
                        or (
                            candidate_current_task.kind == TaskKind.EMERGENCY
                            and candidate_current_task.status
                            in (TaskStatus.PENDING, TaskStatus.CLAIMED)
                            and float(
                                getattr(candidate_current_task, "lifeline_current", 0.0)
                            )
                            <= 0.90
                            * max(
                                float(
                                    getattr(
                                        candidate_current_task,
                                        "lifeline_init",
                                        100.0,
                                    )
                                ),
                                1e-9,
                            )
                        )
                    )
                )
                if (
                    candidate_state is None
                    or candidate_state.node is None
                    or candidate_route is None
                    or candidate_id in recovery_trucks
                    # Candidate-only hard coverage may reuse a non-recovery
                    # truck even when a stale support/anchor advertisement is
                    # present; the transfer helper clears that advertisement
                    # atomically. Claimed/in-service/airborne task prefixes
                    # remain protected by the current-stop guard below.
                    or bool(getattr(candidate_state, "crashed", False))
                    or float(max(getattr(candidate_state, "bulk_inventory_kg_current", 0.0), 0.0))
                    + 1e-9
                    < demand
                    or (
                        er_hlns_balanced_all_tasks_v3_active(env)
                        and candidate_id in emergency_trucks
                        and candidate_id not in airborne_emergency_trucks
                    )
                ):
                    continue
                current = candidate_route.current(env)
                if current is not None:
                    current_task = env.state.tasks.get(str(current.task_id), None)
                    if (
                        candidate_head_guard
                        and candidate_id != owner_id
                        and current_task is not None
                        and current_task.kind == TaskKind.NORMAL
                        and current_task.status in (TaskStatus.PENDING, TaskStatus.CLAIMED)
                        and str(current_task.task_id) != task_id
                        and getattr(current_task, "first_service_step", None) is None
                        and getattr(current_task, "in_service_by", None) is None
                        and int(max(getattr(current_task, "service_remaining", 0), 0)) <= 0
                        and float(max(getattr(current_task, "fulfilled_mass_kg", 0.0), 0.0))
                        <= 1e-9
                        and str(current_task.task_id) not in active_sorties
                    ):
                        continue
                    current_is_airborne_emergency = bool(
                        current_task is not None
                        and current_task.kind == TaskKind.EMERGENCY
                        and candidate_id in airborne_emergency_trucks
                        and str(current_task.task_id) in active_sorties
                    )
                    current_is_orphan_emergency = bool(
                        candidate_orphan_emergency
                        and current_task is not None
                        and str(current_task.task_id)
                        == str(candidate_current_task.task_id)
                    )
                    if current_is_airborne_emergency or current_is_orphan_emergency:
                        pass
                    elif (
                        current_task is None
                        or getattr(current_task, "in_service_by", None) is not None
                        or str(current_task.task_id) in active_sorties
                        or (
                            current_task.status == TaskStatus.CLAIMED
                            and current_task.kind == TaskKind.EMERGENCY
                        )
                        or (
                            er_hlns_balanced_all_tasks_v3_active(env)
                            and current_task.kind == TaskKind.EMERGENCY
                        )
                    ):
                        continue
                road = float(
                    env._decision_shortest_path_distance(
                        int(candidate_state.node), int(task.demand_node)
                    )
                )
                if np.isfinite(road):
                    alternatives.append((road, candidate_id))
            if not alternatives:
                self.hard_normal_rescue_no_truck_count += 1
                if no_truck_once or no_truck_cooldown:
                    self._hard_normal_rescue_no_truck_last_step_by_task[task_id] = step_now
                continue
            road, new_owner = min(alternatives)
            if not self._transfer_routine_contract_to_truck(
                env,
                task_id,
                str(new_owner),
                goals,
                planned_road_distance_m=float(road),
            ):
                self.hard_normal_rescue_rejected_safety_count += 1
                continue
            self._hard_normal_rescue_count_by_task[task_id] = 1
            self._hard_normal_rescue_progress.pop(task_id, None)
            self.hard_normal_rescue_transfer_count += 1
            transferred += 1
            self._stay_reason_by_agent[str(new_owner)] = "hard_normal_coverage_rescue"
            self._feedback.append(
                PlannerFeedback(
                    step=step_now,
                    reason="hard_normal_coverage_rescue",
                    truck_id=str(new_owner),
                    task_id=task_id,
                    detail=f"old_owner={owner_id},road_m={road:.1f}",
                    suffix_repair_required=False,
                )
            )

    def _ensure_residual_normal_coverage(self, env) -> None:
        """Give every post-emergency routine residual an executable owner.

        This is deliberately not another global ALNS replan.  It activates
        only after all emergency tasks are terminal, keeps an existing valid
        cleanup owner, and gives at most one current task to each stocked
        truck.  Completed tasks free that truck for the next residual.
        """
        if not bool(
            getattr(
                env.cfg,
                "hrl_route_plan_residual_normal_coverage_enabled",
                True,
            )
        ) or self.plan is None:
            return
        if any(
            task.kind == TaskKind.EMERGENCY
            and self._active(task)
            and float(getattr(task, "lifeline_current", 0.0)) > 1e-9
            for task in env.state.tasks.values()
        ):
            self._normal_cleanup_owner_by_task.clear()
            return

        active = {
            str(task.task_id): task
            for task in env.state.tasks.values()
            if (
                task.kind == TaskKind.NORMAL
                and self._active(task)
                and not self._is_relay(task)
            )
        }
        live_trucks = self._live_trucks(env)
        published_assists = dict(
            getattr(env, "_planner_truck_assist_waypoint_by_truck", {}) or {}
        )
        active_safety_assist_trucks = {
            str(truck_id)
            for truck_id, assist in published_assists.items()
            if isinstance(assist, dict)
            and str(assist.get("service_mode", "")).strip().upper()
            == "SAFETY_RECOVERY"
        }
        if bool(
            getattr(
                env.cfg,
                "hrl_route_plan_residual_normal_bipartite_matching_enabled",
                False,
            )
        ):
            # Maximum-cardinality residual matching.  The legacy greedy code
            # below remains the fallback/ablation.  Greedy deadline order can
            # consume a flexible truck first and strand a task reachable by
            # only that truck; the augmenting-path matching prioritizes the
            # most constrained tasks while retaining prior owners as a soft
            # preference.
            feasible: Dict[str, List[Tuple[int, float, str]]] = {}
            for task_id, task in active.items():
                prior_owner = str(
                    self._normal_cleanup_owner_by_task.get(str(task_id), "")
                )
                options: List[Tuple[int, float, str]] = []
                for truck_id in live_trucks:
                    truck = env.state.agents.get(str(truck_id), None)
                    if truck is None or truck.node is None:
                        continue
                    if str(truck_id) in active_safety_assist_trucks:
                        continue
                    if float(
                        max(
                            getattr(truck, "bulk_inventory_kg_current", 0.0),
                            0.0,
                        )
                    ) + 1e-9 < self._remaining_demand_kg(task):
                        continue
                    road = float(
                        env._decision_shortest_path_distance(
                            int(truck.node), int(task.demand_node)
                        )
                    )
                    if not np.isfinite(road):
                        continue
                    options.append(
                        (
                            0 if str(truck_id) == prior_owner else 1,
                            float(road),
                            str(truck_id),
                        )
                    )
                feasible[str(task_id)] = sorted(options)

            task_order = sorted(
                active,
                key=lambda task_id: (
                    len(feasible.get(str(task_id), ()))
                    if feasible.get(str(task_id), ())
                    else 10**9,
                    int(active[str(task_id)].deadline_step),
                    str(task_id),
                ),
            )
            truck_to_task: Dict[str, str] = {}

            def augment(task_id: str, seen_trucks: set[str]) -> bool:
                for _, _, truck_id in feasible.get(str(task_id), ()):
                    if str(truck_id) in seen_trucks:
                        continue
                    seen_trucks.add(str(truck_id))
                    displaced = truck_to_task.get(str(truck_id), None)
                    if displaced is None or augment(str(displaced), seen_trucks):
                        truck_to_task[str(truck_id)] = str(task_id)
                        return True
                return False

            for task_id in task_order:
                if feasible.get(str(task_id), ()):
                    augment(str(task_id), set())
            matched = {
                str(task_id): str(truck_id)
                for truck_id, task_id in truck_to_task.items()
            }
            for task_id, truck_id in matched.items():
                contract = self.plan.contracts.get(str(task_id), None)
                if contract is None:
                    contract = TaskContract(
                        task_id=str(task_id),
                        owner_agent_id=str(truck_id),
                        truck_id=str(truck_id),
                        uav_id=None,
                        uav_ids=(),
                        service_mode=DIRECT,
                        created_step=int(env.state.step_index),
                    )
                    self.plan.contracts[str(task_id)] = contract
                    self._stamp_contract_on_task(
                        env, str(task_id), contract, bump=False
                    )
                else:
                    changed = bool(
                        str(contract.owner_agent_id) != str(truck_id)
                        or str(contract.truck_id) != str(truck_id)
                        or contract.uav_id is not None
                    )
                    contract.owner_agent_id = str(truck_id)
                    contract.truck_id = str(truck_id)
                    contract.uav_id = None
                    contract.uav_ids = ()
                    contract.service_mode = DIRECT
                    contract.created_step = int(env.state.step_index)
                    self._stamp_contract_on_task(
                        env, str(task_id), contract, bump=changed
                    )
            self._normal_cleanup_owner_by_task = dict(matched)
            return

        # Legacy stable greedy residual assignment retained for ablation.
        valid: Dict[str, str] = {}
        used_trucks: set[str] = set()
        for task_id, truck_id in sorted(
            self._normal_cleanup_owner_by_task.items()
        ):
            task = active.get(str(task_id), None)
            truck = env.state.agents.get(str(truck_id), None)
            if (
                task is None
                or truck is None
                or truck.node is None
                or str(truck_id) in used_trucks
                or float(
                    max(
                        getattr(truck, "bulk_inventory_kg_current", 0.0),
                        0.0,
                    )
                )
                + 1e-9
                < self._remaining_demand_kg(task)
            ):
                continue
            road = float(
                env._decision_shortest_path_distance(
                    int(truck.node), int(task.demand_node)
                )
            )
            if not np.isfinite(road):
                continue
            valid[str(task_id)] = str(truck_id)
            used_trucks.add(str(truck_id))

        for task_id, task in sorted(
            active.items(),
            key=lambda item: (
                int(item[1].deadline_step),
                str(item[0]),
            ),
        ):
            if str(task_id) in valid:
                continue
            best: Optional[Tuple[float, str]] = None
            for truck_id in live_trucks:
                if str(truck_id) in used_trucks:
                    continue
                truck = env.state.agents.get(str(truck_id), None)
                if truck is None or truck.node is None:
                    continue
                if str(truck_id) in active_safety_assist_trucks:
                    continue
                if float(
                    max(
                        getattr(truck, "bulk_inventory_kg_current", 0.0),
                        0.0,
                    )
                ) + 1e-9 < self._remaining_demand_kg(task):
                    continue
                road = float(
                    env._decision_shortest_path_distance(
                        int(truck.node), int(task.demand_node)
                    )
                )
                if not np.isfinite(road):
                    continue
                candidate = (float(road), str(truck_id))
                if best is None or candidate < best:
                    best = candidate
            if best is None:
                continue
            _, truck_id = best
            valid[str(task_id)] = str(truck_id)
            used_trucks.add(str(truck_id))

            contract = self.plan.contracts.get(str(task_id), None)
            if contract is None:
                contract = TaskContract(
                    task_id=str(task_id),
                    owner_agent_id=str(truck_id),
                    truck_id=str(truck_id),
                    uav_id=None,
                    uav_ids=(),
                    service_mode=DIRECT,
                    created_step=int(env.state.step_index),
                )
                self.plan.contracts[str(task_id)] = contract
                self._stamp_contract_on_task(
                    env, str(task_id), contract, bump=False
                )
            elif (
                str(contract.owner_agent_id) != str(truck_id)
                or str(contract.truck_id) != str(truck_id)
                or contract.uav_id is not None
            ):
                contract.owner_agent_id = str(truck_id)
                contract.truck_id = str(truck_id)
                contract.uav_id = None
                contract.uav_ids = ()
                contract.service_mode = DIRECT
                contract.created_step = int(env.state.step_index)
                self._stamp_contract_on_task(
                    env, str(task_id), contract, bump=True
                )
            else:
                self._stamp_contract_on_task(
                    env, str(task_id), contract, bump=False
                )

        self._normal_cleanup_owner_by_task = valid

    def _advertise_idle_post_emergency_normal_fallback(
        self,
        env,
        goals: Dict[str, Optional[str]],
    ) -> None:
        """Give an actually idle stocked truck its nearest residual routine.

        This is an execution fallback, not a global replan.  Emergency work
        retains priority through its published delivery/recovery assists, but
        an otherwise idle stocked truck may serve a routine in parallel.
        Ownership is transferred atomically so the selected routine remains
        invisible to the other agents.
        """
        if (
            self.plan is None
            or not bool(
                getattr(
                    env.cfg,
                    "hrl_route_plan_idle_normal_fallback_enabled",
                    False,
                )
            )
        ):
            return
        active_emergency_count = int(
            sum(
                1
                for task in env.state.tasks.values()
                if task.kind == TaskKind.EMERGENCY
                and self._active(task)
                and float(getattr(task, "lifeline_current", 0.0)) > 1e-9
            )
        )
        max_pending_emergency = int(
            max(
                getattr(
                    env.cfg,
                    "hrl_route_plan_idle_normal_fallback_max_pending_emergency",
                    2,
                ),
                0,
            )
        )
        fallback_stall_steps = int(
            max(
                getattr(
                    env.cfg,
                    "hrl_route_plan_idle_normal_fallback_stall_steps",
                    30,
                ),
                1,
            )
        )
        completion_stalled = bool(
            int(env.state.step_index) - int(self._last_completion_progress_step)
            >= fallback_stall_steps
        )
        if (
            active_emergency_count > max_pending_emergency
            and not completion_stalled
        ):
            return
        previous_assists = dict(
            getattr(env, "_planner_truck_assist_waypoint_by_truck", {}) or {}
        )
        active_safety_trucks = {
            str(truck_id)
            for truck_id, assist in previous_assists.items()
            if isinstance(assist, dict)
            and str(assist.get("service_mode", "")).strip().upper()
            == "SAFETY_RECOVERY"
        }
        reserved = {
            str(goal_id)
            for goal_id in goals.values()
            if goal_id is not None
        }
        for truck_id in self._live_trucks(env):
            truck_id = str(truck_id)
            if goals.get(truck_id, None) is not None:
                continue
            route = self.plan.routes.get(truck_id, None)
            restrict_idle_fallback_to_empty_route_experiment = False
            if (
                restrict_idle_fallback_to_empty_route_experiment
                and route is not None
                and route.current(env) is not None
            ):
                continue
            if truck_id in self._assist_by_truck or truck_id in active_safety_trucks:
                continue
            truck = env.state.agents.get(truck_id, None)
            if truck is None or truck.node is None:
                continue
            stock = float(
                max(getattr(truck, "bulk_inventory_kg_current", 0.0), 0.0)
            )
            best: Optional[Tuple[float, int, str]] = None
            for task in env.state.tasks.values():
                task_id = str(task.task_id)
                if (
                    task_id in reserved
                    or task.kind != TaskKind.NORMAL
                    or not self._active(task)
                    or self._is_relay(task)
                    or stock + 1e-9 < self._remaining_demand_kg(task)
                ):
                    continue
                road = float(
                    env._decision_shortest_path_distance(
                        int(truck.node), int(task.demand_node)
                    )
                )
                if not np.isfinite(road):
                    continue
                candidate = (float(road), int(task.deadline_step), task_id)
                if best is None or candidate < best:
                    best = candidate
            if best is None:
                continue
            road, _, task_id = best
            task = env.state.tasks[task_id]
            if self._transfer_routine_contract_to_truck(
                env,
                str(task_id),
                str(truck_id),
                goals,
                planned_road_distance_m=float(road),
            ):
                self._stay_reason_by_agent[truck_id] = (
                    "post_emergency_idle_truck_normal_fallback"
                )
                reserved.add(str(task_id))
                continue

            # Legacy goal-only fallback retained for auditability.  The active
            # path above moves the contract and route stop together.
            contract = self.plan.contracts.get(task_id, None)
            if contract is None:
                contract = TaskContract(
                    task_id=task_id,
                    owner_agent_id=truck_id,
                    truck_id=truck_id,
                    uav_id=None,
                    uav_ids=(),
                    service_mode=DIRECT,
                    created_step=int(env.state.step_index),
                )
                self.plan.contracts[task_id] = contract
            else:
                contract.owner_agent_id = truck_id
                contract.truck_id = truck_id
                contract.uav_id = None
                contract.uav_ids = ()
                contract.service_mode = DIRECT
                contract.created_step = int(env.state.step_index)
            self._stamp_contract_on_task(env, task_id, contract, bump=True)
            enable_persistent_idle_route_experiment = False
            if (
                enable_persistent_idle_route_experiment
                and route is not None
                and route.current(env) is None
            ):
                road = float(
                    env._decision_shortest_path_distance(
                        int(truck.node), int(task.demand_node)
                    )
                )
                truck_speed = float(
                    max(getattr(env.cfg, "truck_speed_mps", 1.0), 1e-6)
                )
                dt = float(max(getattr(env.cfg, "dt_seconds", 1.0), 1e-6))
                eta = int(
                    env.state.step_index
                    + np.ceil(max(road, 0.0) / max(truck_speed * dt, 1e-6))
                )
                route.stops.insert(
                    int(route.cursor),
                    RouteStop(
                        task_id=task_id,
                        stop_type=NORMAL_SERVICE,
                        truck_id=truck_id,
                        uav_id=None,
                        uav_ids=(),
                        target_node=int(task.demand_node),
                        planned_road_distance_m=float(road),
                        eta_step=int(eta),
                        deadline_step=int(task.deadline_step),
                        service_mode=DIRECT,
                    ),
                )
            self._normal_cleanup_owner_by_task[task_id] = truck_id
            goals[truck_id] = task_id
            reserved.add(task_id)
            self._stay_reason_by_agent[truck_id] = (
                "post_emergency_idle_truck_normal_fallback"
            )

    def _promote_v5_launch_first_emergencies(self, env) -> None:
        """Promote one safe pending emergency ahead of each initial suffix.

        V5 is a candidate-only initial-plan transform.  It never crosses a
        claimed, in-service, or airborne stop, and it asks the existing
        parallel corridor gate whether the pending emergency can be followed
        by the current direct NORMAL stop before mutating the route.
        """
        if self.plan is None or not er_hlns_balanced_all_tasks_v5_active(env):
            return
        # This method is intentionally called only from the first full-plan
        # publication; the guard protects against accidental reuse by future
        # repair paths.
        if int(self.full_plan_count) != 1:
            return
        airborne_tasks = {
            str(task_id)
            for uav_id, task_id in dict(
                getattr(env, "_uav_sortie_contract_task", {})
            ).items()
            if task_id is not None
            and env.state.agents.get(str(uav_id), None) is not None
            and getattr(env.state.agents[str(uav_id)], "follow_target", None)
            is None
        }

        def protected(task: Any) -> bool:
            return bool(
                task.status != TaskStatus.PENDING
                or getattr(task, "assigned_to", None) is not None
                or getattr(task, "in_service_by", None) is not None
                or int(max(getattr(task, "service_remaining", 0), 0)) > 0
                or str(task.task_id) in airborne_tasks
            )

        for truck_id, route in sorted(self.plan.routes.items()):
            cursor = int(max(getattr(route, "cursor", 0), 0))
            if cursor >= len(route.stops):
                continue
            candidate_index: Optional[int] = None
            for index in range(cursor, len(route.stops)):
                task = env.state.tasks.get(str(route.stops[index].task_id), None)
                if task is None or task.kind != TaskKind.EMERGENCY or not self._active(task):
                    continue
                if protected(task):
                    # Do not move an emergency behind an already committed
                    # emergency/stop across that commitment.
                    if index > cursor:
                        self.balanced_all_tasks_v5_rejected_safety_count += 1
                    candidate_index = None
                    break
                candidate_index = index
                break
            if candidate_index is None or candidate_index == cursor:
                continue
            if any(
                protected(
                    env.state.tasks.get(str(route.stops[index].task_id), None)
                )
                for index in range(cursor, candidate_index)
                if env.state.tasks.get(str(route.stops[index].task_id), None) is not None
            ):
                self.balanced_all_tasks_v5_rejected_safety_count += 1
                continue

            emergency_stop = route.stops[candidate_index]
            emergency_task = env.state.tasks.get(str(emergency_stop.task_id), None)
            normal_stop = route.stops[cursor]
            normal_task = env.state.tasks.get(str(normal_stop.task_id), None)
            contract = self.plan.contracts.get(str(emergency_stop.task_id), None)
            if (
                emergency_task is None
                or normal_task is None
                or normal_task.kind != TaskKind.NORMAL
                or self._is_relay(normal_task)
                or normal_task.status != TaskStatus.PENDING
                or getattr(normal_task, "assigned_to", None) is not None
                or getattr(normal_task, "in_service_by", None) is not None
                or getattr(normal_task, "first_service_step", None) is not None
                or contract is None
                or contract.uav_id is None
            ):
                self.balanced_all_tasks_v5_rejected_safety_count += 1
                continue
            safe, _reason, _diag = self._parallel_routine_emergency_corridor(
                env,
                str(truck_id),
                emergency_stop,
                emergency_task,
                contract,
                normal_stop,
            )
            if not safe:
                self.balanced_all_tasks_v5_rejected_safety_count += 1
                continue
            moved = route.stops.pop(int(candidate_index))
            route.stops.insert(cursor, moved)
            self.balanced_all_tasks_v5_promoted_count += 1
            self._feedback.append(
                PlannerFeedback(
                    step=int(env.state.step_index),
                    reason="v5_launch_first_emergency_promotion",
                    truck_id=str(truck_id),
                    task_id=str(emergency_task.task_id),
                    detail=f"normal_successor={normal_task.task_id}",
                )
            )

    def _promote_safe_at_risk_emergency_suffixes(
        self,
        env,
        road_signature: Tuple[Tuple[int, int], ...],
    ) -> None:
        """Move a provably late emergency suffix to the head of its own route."""
        enable_safe_suffix_promotion_experiment = False
        if not enable_safe_suffix_promotion_experiment:
            return
        if self.plan is None:
            return
        reserve = int(
            max(getattr(env.cfg, "hrl_route_plan_deadline_rescue_reserve_steps", 6), 0)
        )
        cooldown = int(
            max(getattr(env.cfg, "hrl_route_plan_contract_transfer_cooldown_steps", 15), 0)
        )
        clusters = self._cluster_uavs(env, self._live_trucks(env))
        step_now = int(env.state.step_index)
        for truck_id, route in sorted(self.plan.routes.items()):
            suffix_ids = [
                str(stop.task_id)
                for stop in route.stops[int(route.cursor) :]
                if (
                    env.state.tasks.get(str(stop.task_id), None) is not None
                    and self._active(env.state.tasks[str(stop.task_id)])
                )
            ]
            if len(suffix_ids) <= 1:
                continue
            baseline_specs, _ = self._route_stop_specs(
                env,
                str(truck_id),
                suffix_ids,
                clusters,
                self.plan.contracts,
                road_signature,
            )
            if len(baseline_specs) <= 1:
                continue
            baseline_by_task = {
                str(stop.task_id): stop for stop in baseline_specs
            }
            for index, stop in enumerate(baseline_specs[1:], start=1):
                task_id = str(stop.task_id)
                task = env.state.tasks.get(task_id, None)
                if task is None or task.kind != TaskKind.EMERGENCY:
                    continue
                last_move = int(
                    self._contract_last_transfer_step.get(task_id, -10**9)
                )
                if step_now - last_move < cooldown:
                    continue
                effective_deadline = self._effective_deadline_step(env, task)
                if int(stop.eta_step) <= effective_deadline - reserve:
                    continue
                trial_ids = [task_id] + [
                    existing_id
                    for existing_id in suffix_ids
                    if str(existing_id) != task_id
                ]
                trial_specs, trial_cost = self._route_stop_specs(
                    env,
                    str(truck_id),
                    trial_ids,
                    clusters,
                    self.plan.contracts,
                    road_signature,
                )
                if not trial_specs or not np.isfinite(trial_cost):
                    continue
                trial_by_task = {str(item.task_id): item for item in trial_specs}
                promoted = trial_by_task.get(task_id, None)
                if (
                    promoted is None
                    or int(promoted.eta_step) > effective_deadline - reserve
                ):
                    continue
                safe = True
                for other_id, baseline_stop in baseline_by_task.items():
                    if other_id == task_id:
                        continue
                    other_task = env.state.tasks.get(str(other_id), None)
                    trial_stop = trial_by_task.get(str(other_id), None)
                    if (
                        other_task is None
                        or trial_stop is None
                    ):
                        continue
                    other_deadline = (
                        self._effective_deadline_step(env, other_task)
                        if other_task.kind == TaskKind.EMERGENCY
                        else int(other_task.deadline_step)
                    )
                    other_reserve = reserve if other_task.kind == TaskKind.EMERGENCY else 0
                    if (
                        other_task.kind == TaskKind.NORMAL
                        and int(trial_stop.eta_step) - int(baseline_stop.eta_step)
                        > int(
                            max(
                                getattr(
                                    env.cfg,
                                    "hrl_route_plan_normal_max_emergency_delay_steps",
                                    12,
                                ),
                                0,
                            )
                        )
                    ):
                        safe = False
                        break
                    if (
                        int(baseline_stop.eta_step) <= other_deadline - other_reserve
                        and int(trial_stop.eta_step) > other_deadline - other_reserve
                    ):
                        safe = False
                        break
                if not safe:
                    continue
                route.stops = list(route.stops[: int(route.cursor)]) + list(trial_specs)
                self.deadline_rescue_promotion_count += 1
                self._contract_last_transfer_step[task_id] = int(step_now)
                self._feedback.append(
                    PlannerFeedback(
                        step=step_now,
                        reason="safe_same_route_emergency_suffix_promotion",
                        truck_id=str(truck_id),
                        task_id=task_id,
                        detail=(
                            f"old_eta={int(stop.eta_step)},"
                            f"new_eta={int(promoted.eta_step)},"
                            f"effective_deadline={effective_deadline}"
                        ),
                    )
                )
                break

    def _apply_onsite_emergency_takeovers(self, env) -> None:
        """Hand a blocked current TC task to a ready UAV already at the site."""
        if self.plan is None or not bool(
            getattr(env.cfg, "hrl_route_plan_onsite_takeover_enabled", True)
        ):
            return
        capture_radius = float(
            max(
                getattr(env.cfg, "uav_delivery_radius_m", 40.0),
                float(getattr(env.cfg, "uav_delivery_capture_motion_factor", 0.80))
                * float(getattr(env.cfg, "uav_max_speed_mps", 0.0))
                * float(getattr(env.cfg, "dt_seconds", 1.0)),
                1.0,
            )
        )
        min_battery = float(
            np.clip(
                getattr(env.cfg, "hrl_route_plan_onsite_takeover_min_battery", 0.98),
                0.0,
                1.0,
            )
        )
        sortie_contracts = dict(getattr(env, "_uav_sortie_contract_task", {}))
        for route in self.plan.routes.values():
            stop = route.current(env)
            if stop is None:
                continue
            task_id = str(stop.task_id)
            task = env.state.tasks.get(task_id, None)
            contract = self.plan.contracts.get(task_id, None)
            if (
                task is None
                or task.kind != TaskKind.EMERGENCY
                or task.status != TaskStatus.PENDING
                or contract is None
                or contract.uav_id is None
            ):
                continue
            owner_id = str(contract.uav_id)
            owner = env.state.agents.get(owner_id, None)
            owner_distance = (
                float("inf")
                if owner is None
                else float(env._agent_distance_to_task(owner_id, task))
            )
            owner_sortie = sortie_contracts.get(owner_id, None)
            owner_conflict = bool(
                owner is not None
                and owner.follow_target is None
                and owner_sortie is not None
                and str(owner_sortie) != task_id
            )
            progress = self._contract_progress.get(task_id, {})
            stalled = bool(
                int(env.state.step_index) - int(progress.get("last_improve_step", env.state.step_index))
                >= int(max(getattr(env.cfg, "hrl_route_plan_contract_stall_steps", 12), 1))
            )
            if owner_distance <= capture_radius or not (owner_conflict or stalled):
                continue

            candidates: List[Tuple[float, str, str]] = []
            for uid, state in sorted(env.state.agents.items()):
                uid_s = str(uid)
                if (
                    state.kind != AgentKind.UAV
                    or uid_s == owner_id
                    or bool(getattr(state, "crashed", False))
                    or state.follow_target is None
                    or float(getattr(state, "battery", 0.0)) + 1e-9 < min_battery
                    or not bool(env._uav_loaded_for_task(uid_s, task))
                ):
                    continue
                candidate_sortie = sortie_contracts.get(uid_s, None)
                candidate_task = (
                    None
                    if candidate_sortie is None
                    else env.state.tasks.get(str(candidate_sortie), None)
                )
                if (
                    candidate_task is not None
                    and self._active(candidate_task)
                    and str(candidate_sortie) != task_id
                ):
                    continue
                distance = float(env._agent_distance_to_task(uid_s, task))
                if distance <= capture_radius + 1e-9:
                    candidates.append((distance, uid_s, str(state.follow_target)))
            if not candidates:
                continue
            _, new_uav_id, new_truck_id = min(candidates)
            contract.owner_agent_id = str(new_uav_id)
            contract.uav_id = str(new_uav_id)
            contract.uav_ids = (str(new_uav_id),)
            contract.truck_id = str(new_truck_id)
            contract.created_step = int(env.state.step_index)
            stop.uav_id = str(new_uav_id)
            stop.uav_ids = (str(new_uav_id),)
            stop.truck_id = str(new_truck_id)
            task.route_contract_owner = str(new_uav_id)
            task.route_contract_truck = str(new_truck_id)
            task.route_contract_uav_ids = (str(new_uav_id),)
            self._stamp_contract_on_task(
                env, task_id, contract, bump=True
            )
            env._uav_sortie_contract_task[str(new_uav_id)] = task_id
            if hasattr(env, "_uav_sortie_contract_version"):
                env._uav_sortie_contract_version[str(new_uav_id)] = int(
                    getattr(contract, "version", 0)
                )
            self._contract_progress.pop(task_id, None)
            self.onsite_takeover_count += 1
            self._feedback.append(
                PlannerFeedback(
                    step=int(env.state.step_index),
                    reason="onsite_emergency_takeover",
                    truck_id=str(new_truck_id),
                    task_id=task_id,
                    detail=f"old_uav={owner_id},new_uav={new_uav_id}",
                )
            )

    def _stalled_emergency_contracts_for_transfer(
        self,
        env,
        road_signature: Tuple[Tuple[int, int], ...],
    ) -> Dict[str, str]:
        """Find current emergency contracts that a different unit can rescue.

        Only the head stop of each truck route is considered.  Normal work is
        never released here, and a candidate truck that is already executing
        another emergency is not interrupted.
        """
        if self.plan is None or not bool(
            getattr(env.cfg, "hrl_route_plan_contract_transfer_enabled", True)
        ):
            return {}
        low_seed_rescue = bool(er_hlns_low_seed_rescue_active(env))
        step_now = int(env.state.step_index)
        stall_steps = int(
            max(getattr(env.cfg, "hrl_route_plan_contract_stall_steps", 12), 1)
        )
        min_gain = float(
            max(
                getattr(
                    env.cfg,
                    "hrl_route_plan_contract_transfer_min_eta_gain_steps",
                    20.0,
                ),
                0.0,
            )
        )
        shadow_enabled = bool(
            getattr(
                env.cfg,
                "hrl_route_plan_shadow_promotion_enabled",
                False,
            )
        )
        shadow_min_gain = float(
            max(
                getattr(
                    env.cfg,
                    "hrl_route_plan_shadow_promotion_min_gain_steps",
                    4,
                ),
                0.0,
            )
        )
        shadow_normal_tolerance = float(
            max(
                getattr(
                    env.cfg,
                    "hrl_route_plan_shadow_promotion_normal_tolerance_steps",
                    0,
                ),
                0.0,
            )
        )
        cooldown = int(
            max(
                getattr(
                    env.cfg,
                    "hrl_route_plan_contract_transfer_cooldown_steps",
                    15,
                ),
                0,
            )
        )
        clusters = self._cluster_uavs(env, self._live_trucks(env))
        current_by_truck = {
            str(truck_id): route.current(env)
            for truck_id, route in self.plan.routes.items()
        }
        active_current_ids = {
            str(stop.task_id) for stop in current_by_truck.values() if stop is not None
        }
        for task_id in list(self._contract_progress):
            if task_id not in active_current_ids:
                self._contract_progress.pop(task_id, None)

        release: Dict[str, str] = {}
        for truck_id, stop in sorted(current_by_truck.items()):
            if stop is None:
                continue
            task_id = str(stop.task_id)
            task = env.state.tasks.get(task_id, None)
            contract = self.plan.contracts.get(task_id, None)
            if (
                task is None
                or task.kind != TaskKind.EMERGENCY
                or contract is None
                or str(contract.truck_id) != str(truck_id)
            ):
                continue
            owner = env.state.agents.get(str(contract.uav_id), None)
            owner_carrier = (
                None if owner is None else getattr(owner, "follow_target", None)
            )
            active_sortie_task = str(
                dict(getattr(env, "_uav_sortie_contract_task", {})).get(
                    str(contract.uav_id), ""
                )
                or ""
            )
            if low_seed_rescue and (
                task.status not in (TaskStatus.PENDING, TaskStatus.CLAIMED)
                or getattr(task, "first_service_step", None) is not None
                or getattr(task, "in_service_by", None) is not None
                or active_sortie_task == str(task_id)
                or (owner is not None and getattr(owner, "follow_target", None) is None)
            ):
                # Candidate-only re-auction is restricted to an unstarted,
                # docked emergency contract. Claimed/in-service/airborne
                # execution prefixes are never released.
                continue
            enable_owner_carrier_mismatch_release_experiment = bool(
                getattr(
                    env.cfg,
                    "hrl_route_plan_owner_carrier_mismatch_repair_enabled",
                    False,
                )
            )
            # A recovered UAV can finish on another truck (or at the depot).
            # The old task contract must then be rebuilt immediately: keeping
            # the original truck/uav pair creates a valid-looking route whose
            # two physical members can never execute the launch together.
            # Airborne sorties are deliberately excluded from this repair.
            if (
                enable_owner_carrier_mismatch_release_experiment
                and
                owner is not None
                and owner_carrier is not None
                and str(owner_carrier) != str(contract.truck_id)
                and active_sortie_task != str(task_id)
            ):
                release[task_id] = str(truck_id)
                self._contract_progress.pop(task_id, None)
                self._feedback.append(
                    PlannerFeedback(
                        step=step_now,
                        reason="contract_owner_carrier_mismatch",
                        truck_id=str(truck_id),
                        task_id=task_id,
                        detail=(
                            f"uav={contract.uav_id},"
                            f"contract_truck={truck_id},"
                            f"actual_carrier={owner_carrier}"
                        ),
                        suffix_repair_required=True,
                    )
                )
                continue
            current_eval = self._single_emergency_eta(
                env,
                str(truck_id),
                task,
                clusters,
                road_signature,
                preferred=contract,
            )
            current_remaining = (
                float("inf") if current_eval is None else float(current_eval[0])
            )
            # A contract whose truck/UAV unit has no remaining emergency
            # package is not executable. Keep the historical branch for
            # auditability, but enable this narrow repair only in the
            # configured large-map B profile.
            enable_immediate_stockout_transfer_experiment = bool(
                getattr(
                    env.cfg,
                    "hrl_route_plan_stockout_transfer_enabled",
                    False,
                )
            )
            current_supply_available = bool(
                not enable_immediate_stockout_transfer_experiment
                or self._contract_has_emergency_supply(env, contract, task)
            )
            if enable_immediate_stockout_transfer_experiment and not current_supply_available:
                current_remaining = float("inf")
            record = self._contract_progress.get(task_id, None)
            if record is None or str(record.get("truck_id", "")) != str(truck_id):
                record = {
                    "truck_id": str(truck_id),
                    "best_remaining": float(current_remaining),
                    "last_improve_step": int(step_now),
                    "current_remaining": float(current_remaining),
                }
                self._contract_progress[task_id] = record
                continue
            if current_remaining < float(record["best_remaining"]) - 0.5:
                record["best_remaining"] = float(current_remaining)
                record["last_improve_step"] = int(step_now)
            record["current_remaining"] = float(current_remaining)
            candidate_stale = bool(
                low_seed_rescue
                and step_now >= 60
                and step_now - int(record["last_improve_step"]) >= max(stall_steps, 18)
            )
            if (
                current_supply_available
                and not candidate_stale
                and step_now - int(record["last_improve_step"]) < stall_steps
            ):
                continue
            last_transfer = int(self._contract_last_transfer_step.get(task_id, -10**9))
            if step_now - last_transfer < cooldown:
                continue

            best_alternative: Optional[Tuple[float, str]] = None
            for other_truck_id in sorted(clusters):
                if str(other_truck_id) == str(truck_id):
                    continue
                if (
                    enable_immediate_stockout_transfer_experiment
                    and
                    self._cluster_emergency_capacity_units(
                        env,
                        str(other_truck_id),
                        clusters.get(str(other_truck_id), ()),
                    )
                    <= 0
                ):
                    continue
                other_stop = current_by_truck.get(str(other_truck_id), None)
                if other_stop is not None and str(other_stop.task_id) != task_id:
                    other_task = env.state.tasks.get(str(other_stop.task_id), None)
                    if other_task is not None and other_task.kind == TaskKind.EMERGENCY:
                        continue
                alternative = self._single_emergency_eta(
                    env,
                    str(other_truck_id),
                    task,
                    clusters,
                    road_signature,
                )
                if alternative is None:
                    continue
                alternative_remaining, _, alternative_stop = alternative
                alternative_deadline = (
                    self._effective_deadline_step(env, task)
                    if self._targeted_repairs_enabled()
                    else int(task.deadline_step)
                )
                if int(alternative_stop.eta_step) > int(alternative_deadline):
                    continue
                candidate = (float(alternative_remaining), str(other_truck_id))
                if best_alternative is None or candidate < best_alternative:
                    best_alternative = candidate
            if best_alternative is None:
                continue
            alternative_remaining, alternative_truck = best_alternative
            if (
                current_remaining - alternative_remaining < min_gain - 1e-9
                and not (low_seed_rescue and np.isfinite(alternative_remaining))
            ):
                continue
            release[task_id] = str(truck_id)
            self._feedback.append(
                PlannerFeedback(
                    step=step_now,
                    reason="stalled_emergency_contract_transfer",
                    truck_id=str(truck_id),
                    task_id=task_id,
                    detail=(
                        f"remaining={current_remaining:.1f},"
                        f"alternative={alternative_remaining:.1f},"
                        f"target_truck={alternative_truck}"
                    ),
                    suffix_repair_required=True,
                )
            )
            if low_seed_rescue:
                break
        if low_seed_rescue and not release and self.plan is not None:
            # Some low-seed failures are not the current route head: the
            # emergency is advertised in a future suffix and therefore never
            # reaches the head-level loop above. Re-auction at most one such
            # still-pending, never-started contract after a long exposure.
            current_head_ids = {
                str(stop.task_id)
                for stop in current_by_truck.values()
                if stop is not None
            }
            sortie_tasks = {
                str(tid)
                for tid in dict(getattr(env, "_uav_sortie_contract_task", {})).values()
                if tid is not None
            }
            for task_id, contract in sorted(self.plan.contracts.items()):
                if str(task_id) in current_head_ids or str(task_id) in sortie_tasks:
                    continue
                task = env.state.tasks.get(str(task_id), None)
                if (
                    task is None
                    or task.kind != TaskKind.EMERGENCY
                    or task.status not in (TaskStatus.PENDING, TaskStatus.CLAIMED)
                    or getattr(task, "first_service_step", None) is not None
                    or getattr(task, "in_service_by", None) is not None
                    or contract.uav_id is None
                ):
                    continue
                owner = env.state.agents.get(str(contract.uav_id), None)
                if owner is None or getattr(owner, "follow_target", None) is None:
                    continue
                progress = self._contract_progress.get(str(task_id), {})
                last_step = int(progress.get("last_improve_step", step_now))
                if step_now < 60 or step_now - last_step < max(stall_steps, 18):
                    continue
                best_alt = None
                for other_truck_id in sorted(clusters):
                    if str(other_truck_id) == str(contract.truck_id):
                        continue
                    other_stop = current_by_truck.get(str(other_truck_id), None)
                    if other_stop is not None:
                        other_task = env.state.tasks.get(str(other_stop.task_id), None)
                        if other_task is not None and other_task.kind == TaskKind.EMERGENCY:
                            continue
                    alt = self._single_emergency_eta(
                        env, str(other_truck_id), task, clusters, road_signature
                    )
                    if alt is None:
                        continue
                    remaining, _, stop = alt
                    if int(stop.eta_step) > self._effective_deadline_step(env, task):
                        continue
                    option = (float(remaining), str(other_truck_id))
                    if best_alt is None or option < best_alt:
                        best_alt = option
                if best_alt is None:
                    continue
                release[str(task_id)] = str(contract.truck_id)
                self._feedback.append(
                    PlannerFeedback(
                        step=step_now,
                        reason="low_seed_stale_future_emergency_reauction",
                        truck_id=str(contract.truck_id),
                        task_id=str(task_id),
                        detail=f"target_truck={best_alt[1]},eta={best_alt[0]:.1f}",
                        suffix_repair_required=True,
                    )
                )
                break
        return release

    def _stockout_emergency_contracts_for_transfer(
        self,
        env,
        road_signature: Tuple[Tuple[int, int], ...],
    ) -> Dict[str, str]:
        """Release a pending emergency contract whose unit has no package.

        This is a hard execution gap, not a general reauction: airborne
        sorties and contracts with mounted cargo remain untouched.  The
        replacement must be a different truck with a physical package and a
        deadline-feasible single-task chain.
        """
        if self.plan is None or not bool(
            getattr(env.cfg, "hrl_route_plan_stockout_transfer_enabled", False)
        ):
            return {}
        clusters = self._cluster_uavs(env, self._live_trucks(env))
        current_by_truck = {
            str(truck_id): route.current(env)
            for truck_id, route in self.plan.routes.items()
        }
        release: Dict[str, str] = {}
        sortie_tasks = {
            str(task_id)
            for task_id in dict(getattr(env, "_uav_sortie_contract_task", {})).values()
            if task_id is not None
        }
        for task_id, contract in sorted(self.plan.contracts.items()):
            task = env.state.tasks.get(str(task_id), None)
            if (
                task is None
                or task.kind != TaskKind.EMERGENCY
                or not self._active(task)
                or str(task_id) in sortie_tasks
                or self._contract_has_emergency_supply(env, contract, task)
            ):
                continue
            owner = env.state.agents.get(str(contract.uav_id or ""), None)
            if owner is not None and getattr(owner, "follow_target", None) is None:
                # A loaded airborne unit is authoritative even if the truck
                # inventory snapshot has already been decremented.
                try:
                    if bool(env._uav_loaded_for_task(str(contract.uav_id), task)):
                        continue
                except Exception:
                    if float(getattr(owner, "payload_kg_current", 0.0)) > 1e-9:
                        continue
            best: Optional[Tuple[float, str]] = None
            for truck_id, uavs in sorted(clusters.items()):
                if str(truck_id) == str(contract.truck_id):
                    continue
                if self._cluster_emergency_capacity_units(env, str(truck_id), uavs) <= 0:
                    continue
                current = current_by_truck.get(str(truck_id), None)
                if current is not None and str(current.task_id) != str(task_id):
                    current_task = env.state.tasks.get(str(current.task_id), None)
                    if current_task is not None and current_task.kind == TaskKind.EMERGENCY:
                        continue
                alternative = self._single_emergency_eta(
                    env, str(truck_id), task, clusters, road_signature
                )
                if alternative is None:
                    continue
                remaining, _, stop = alternative
                if int(stop.eta_step) > self._effective_deadline_step(env, task):
                    continue
                candidate = (float(remaining), str(truck_id))
                if best is None or candidate < best:
                    best = candidate
            if best is None:
                continue
            release[str(task_id)] = str(contract.truck_id)
            self._feedback.append(
                PlannerFeedback(
                    step=int(env.state.step_index),
                    reason="stockout_emergency_contract_rebuild",
                    truck_id=str(contract.truck_id),
                    task_id=str(task_id),
                    detail=f"target_truck={best[1]},eta={best[0]:.1f}",
                    suffix_repair_required=True,
                )
            )
        return release

    def _solution_cost(
        self,
        env,
        routes: Dict[str, List[str]],
        cluster_uavs: Dict[str, Tuple[str, ...]],
        contracts: Dict[str, TaskContract],
        road_signature: Tuple[Tuple[int, int], ...],
    ) -> Tuple[float, Dict[str, List[RouteStop]]]:
        # One call evaluates one complete cooperative route solution.  Count
        # it explicitly so ER-HLNS can be compared with K2 search baselines by
        # actual evaluated solutions, not merely by outer-loop iterations.
        self.alns_objective_evaluation_count += 1
        self.alns_feasibility_evaluation_count += 1
        total = 0.0
        specs: Dict[str, List[RouteStop]] = {}
        for truck_id in sorted(routes):
            stops, cost = self._route_stop_specs(
                env,
                truck_id,
                routes[truck_id],
                cluster_uavs,
                contracts,
                road_signature,
            )
            if not np.isfinite(cost):
                return float("inf"), {}
            specs[str(truck_id)] = stops
            total += float(cost)
        if routes:
            lengths = [len(items) for items in routes.values()]
            total += 0.75 * float(max(lengths) - min(lengths))
        return float(total), specs

    def _solution_rank(
        self,
        env,
        scalar_cost: float,
        specs: Dict[str, List[RouteStop]],
    ) -> Tuple[float, ...]:
        """Lexicographic paper objective for one complete candidate.

        Emergency predicted failures and lateness remain immutable priorities.
        Among plans that preserve those two terms, protect structurally
        exposed truck-only routine tasks before their predicted road cut;
        emergency completion time and scalar travel cost then break ties.
        """
        if not bool(
            getattr(
                env.cfg,
                "hrl_route_plan_lexicographic_objective_enabled",
                True,
            )
        ):
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(scalar_cost))
        missed = 0.0
        total_lateness = 0.0
        total_completion = 0.0
        disconnect_missed = 0.0
        disconnect_lateness = 0.0
        mixed_coverage_enabled = bool(
            getattr(env.cfg, "hrl_route_plan_mixed_coverage_enabled", False)
        )
        routine_scheduled_ids: set[str] = set()
        routine_late_count = 0.0
        routine_lateness = 0.0
        routine_covered_value = 0.0
        routine_fast_value = 0.0
        routine_prefix_penalty = 0.0
        emergency_low_slack_count = 0.0
        emergency_slack_shortfall = 0.0
        emergency_slack_total = 0.0
        emergency_reserve_steps = int(
            max(
                getattr(
                    env.cfg,
                    "hrl_route_plan_mixed_coverage_emergency_reserve_steps",
                    30,
                ),
                0,
            )
        )
        active_routine_ids = {
            str(task.task_id)
            for task in env.state.tasks.values()
            if task.kind == TaskKind.NORMAL
            and self._active(task)
            and not self._is_relay(task)
        }
        emergency_counts: List[int] = []
        step_now = int(env.state.step_index)
        disconnect_profiles = self._global_disconnect_profiles(env)
        emergency_count_by_truck: Dict[str, int] = {}
        for route_truck_id, stops in specs.items():
            preceding_emergency_count = 0
            for stop in stops:
                task = env.state.tasks.get(str(stop.task_id), None)
                if task is None:
                    continue
                if task.kind == TaskKind.NORMAL:
                    if not self._is_relay(task):
                        routine_scheduled_ids.add(str(task.task_id))
                        if er_hlns_balanced_all_tasks_active(env) and bool(
                            getattr(
                                env.cfg,
                                "hrl_route_plan_balanced_all_tasks_enabled",
                                False,
                            )
                        ):
                            # The aggressive pilot explicitly values a
                            # deliverable normal prefix.  Without this term,
                            # emergency insertion can repeatedly move every
                            # normal stop behind UAV work even when all normal
                            # contracts were assigned successfully.
                            routine_prefix_penalty += float(preceding_emergency_count)
                        effective_deadline = int(
                            self._effective_deadline_step(env, task)
                        )
                        eta = int(stop.eta_step)
                        late = float(max(eta - effective_deadline, 0))
                        if late > 1e-9:
                            routine_late_count += 1.0
                            routine_lateness += late
                        else:
                            urgency = float(
                                np.clip(
                                    getattr(task, "urgency_score", 0.5),
                                    0.0,
                                    1.0,
                                )
                            )
                            value = 1.0 + urgency
                            routine_covered_value += value
                            routine_fast_value += value / max(
                                float(eta - step_now), 1.0
                            )
                    profile = disconnect_profiles.get(str(task.task_id), None)
                    if profile is not None and bool(profile.get("protected", 0.0)):
                        safe_visit = int(profile.get("safe_visit_step", task.deadline_step))
                        queue_buffer = int(
                            max(
                                getattr(
                                    env.cfg,
                                    "hrl_route_plan_global_disconnect_emergency_queue_buffer_steps",
                                    30,
                                ),
                                0,
                            )
                        )
                        robust_eta = int(stop.eta_step) + int(
                            preceding_emergency_count * queue_buffer
                        )
                        delay = float(max(robust_eta - safe_visit, 0))
                        if delay > 1e-9:
                            disconnect_missed += 1.0
                            disconnect_lateness += delay
                        elif bool(profile.get("head_protected", 0.0)):
                            # Exposure tie-breaker: when both candidates meet
                            # the structural deadline, prefer the one that
                            # does not queue the weakest-cut task behind UAV
                            # contracts whose recovery time is uncertain.
                            disconnect_lateness += float(
                                preceding_emergency_count
                            )
                    continue
                if task.kind != TaskKind.EMERGENCY:
                    continue
                preceding_emergency_count += 1
                effective_deadline = int(self._effective_deadline_step(env, task))
                eta = int(stop.eta_step)
                lateness = float(max(eta - effective_deadline, 0))
                slack = float(effective_deadline - eta)
                emergency_slack_total += max(slack, 0.0)
                if mixed_coverage_enabled and slack < emergency_reserve_steps:
                    emergency_low_slack_count += 1.0
                    emergency_slack_shortfall += float(
                        emergency_reserve_steps - max(slack, 0.0)
                    )
                if lateness > 1e-9:
                    missed += 1.0
                total_lateness += lateness
                urgency = float(
                    np.clip(getattr(task, "urgency_score", 0.5), 0.0, 1.0)
                )
                total_completion += float(max(eta - step_now, 0)) * (
                    1.0 + urgency
                )
            emergency_counts.append(int(preceding_emergency_count))
            emergency_count_by_truck[str(route_truck_id)] = int(
                preceding_emergency_count
            )
        emergency_imbalance = 0.0
        capacity_aware_enabled = bool(
            getattr(
                env.cfg,
                "hrl_route_plan_capacity_aware_emergency_allocation_enabled",
                False,
            )
        )
        if capacity_aware_enabled and self._emergency_capacity_target_by_truck:
            emergency_imbalance = float(
                sum(
                    max(
                        emergency_count_by_truck.get(str(truck_id), 0)
                        - int(target),
                        0,
                    )
                    ** 2
                    for truck_id, target in self._emergency_capacity_target_by_truck.items()
                )
            )
        balance_enabled = (
            bool(self._emergency_balance_active)
            if self._emergency_balance_active is not None
            else bool(
                getattr(
                    env.cfg,
                    "hrl_route_plan_global_emergency_balance_enabled",
                    False,
                )
            )
        )
        if (
            not capacity_aware_enabled
            and balance_enabled
            and emergency_counts
        ):
            total_scheduled = int(sum(emergency_counts))
            target = int(
                np.ceil(total_scheduled / max(len(emergency_counts), 1))
            )
            emergency_imbalance = float(
                sum(max(count - target, 0) ** 2 for count in emergency_counts)
            )
        self.disconnect_predicted_miss_count = int(disconnect_missed)
        if mixed_coverage_enabled:
            routine_uncovered = float(
                len(active_routine_ids - routine_scheduled_ids)
            ) + routine_late_count
            # Lower is better.  Negative values reward more on-time, faster
            # routine coverage only after emergency safety terms tie.
            return (
                float(missed),
                float(total_lateness),
                float(emergency_imbalance),
                float(disconnect_missed),
                float(disconnect_lateness),
                float(emergency_low_slack_count),
                float(emergency_slack_shortfall),
                float(-emergency_slack_total),
                float(routine_prefix_penalty),
                float(-routine_fast_value),
                float(-routine_covered_value),
                float(routine_uncovered),
                float(total_completion),
                float(scalar_cost),
            )
        return (
            float(missed),
            float(total_lateness),
            float(emergency_imbalance),
            float(disconnect_missed),
            float(disconnect_lateness),
            float(total_completion),
            float(scalar_cost),
        )

    def _best_insertion(
        self,
        env,
        task_id: str,
        routes: Dict[str, List[str]],
        cluster_uavs: Dict[str, Tuple[str, ...]],
        contracts: Dict[str, TaskContract],
        road_signature: Tuple[Tuple[int, int], ...],
        fixed_truck: Optional[str] = None,
    ) -> Optional[Tuple[str, int, float]]:
        best: Optional[Tuple[str, int, float]] = None
        best_rank: Optional[Tuple[float, ...]] = None
        task = env.state.tasks[str(task_id)]
        base_cost, base_specs = self._solution_cost(
            env, routes, cluster_uavs, contracts, road_signature
        )
        base_rank = self._solution_rank(env, base_cost, base_specs)
        base_emergency_eta = {
            str(stop.task_id): int(stop.eta_step)
            for stops in base_specs.values()
            for stop in stops
            if (
                env.state.tasks.get(str(stop.task_id), None) is not None
                and env.state.tasks[str(stop.task_id)].kind == TaskKind.EMERGENCY
            )
        }
        joint_corridor = bool(
            getattr(env.cfg, "hrl_route_plan_joint_corridor_enabled", True)
        )
        for truck_id in sorted(routes):
            if (
                er_hlns_balanced_all_tasks_active(env)
                and bool(
                    getattr(
                        env.cfg,
                        "hrl_route_plan_balanced_all_tasks_enabled",
                        False,
                    )
                )
                and task.kind == TaskKind.NORMAL
                and not self._is_relay(task)
            ):
                configured_quota = int(
                    max(
                        getattr(
                            env.cfg,
                            "hrl_route_plan_balanced_all_tasks_max_normal_per_truck",
                            0,
                        ),
                        0,
                    )
                )
                if configured_quota <= 0:
                    configured_quota = int(
                        max(
                            np.ceil(
                                sum(
                                    1
                                    for candidate_task in env.state.tasks.values()
                                    if candidate_task.kind == TaskKind.NORMAL
                                    and self._active(candidate_task)
                                    and not self._is_relay(candidate_task)
                                )
                                / max(len(routes), 1)
                            ),
                            1,
                        )
                    )
                routine_count = sum(
                    1
                    for existing_id in routes[str(truck_id)]
                    if (
                        env.state.tasks.get(str(existing_id), None) is not None
                        and env.state.tasks[str(existing_id)].kind == TaskKind.NORMAL
                        and not self._is_relay(env.state.tasks[str(existing_id)])
                    )
                )
                if routine_count >= configured_quota:
                    self.balanced_all_tasks_quota_block_count += 1
                    continue
            forbidden_truck = self._temporary_forbidden_truck_by_task.get(
                str(task_id), None
            )
            if forbidden_truck is not None and str(truck_id) == str(forbidden_truck):
                continue
            if fixed_truck is not None and str(truck_id) != str(fixed_truck):
                continue
            if (task.kind == TaskKind.EMERGENCY or self._is_relay(task)) and not cluster_uavs.get(str(truck_id), ()):
                continue
            for position in range(len(routes[truck_id]) + 1):
                # Legacy strict ordering is retained as a fallback.  The new
                # joint-corridor mode permits a DIRECT routine stop before or
                # between emergency launch points only after the bounded-delay
                # checks below prove it is genuinely on the way.
                if task.kind == TaskKind.NORMAL and not joint_corridor:
                    last_original_emergency = max(
                        (
                            index
                            for index, existing_id in enumerate(routes[truck_id])
                            if (
                                env.state.tasks.get(str(existing_id), None)
                                is not None
                                and env.state.tasks[str(existing_id)].kind
                                == TaskKind.EMERGENCY
                            )
                        ),
                        default=-1,
                    )
                    if position <= last_original_emergency:
                        continue
                if self._is_relay(task):
                    last_original_emergency = max(
                        (
                            index
                            for index, existing_id in enumerate(
                                routes[truck_id]
                            )
                            if env.state.tasks.get(str(existing_id), None)
                            is not None
                            and env.state.tasks[
                                str(existing_id)
                            ].kind
                            == TaskKind.EMERGENCY
                        ),
                        default=-1,
                    )
                    if position <= last_original_emergency:
                        continue
                enforce_fallback_emergency_inventory_budget = False
                if (
                    task.kind == TaskKind.EMERGENCY
                    and enforce_fallback_emergency_inventory_budget
                ):
                    first_relay = min(
                        (
                            index
                            for index, existing_id in enumerate(
                                routes[truck_id]
                            )
                            if self._is_relay(
                                env.state.tasks.get(
                                    str(existing_id), None
                                )
                            )
                        ),
                        default=len(routes[truck_id]),
                    )
                    if position > first_relay:
                        continue
                trial = {key: list(value) for key, value in routes.items()}
                trial[str(truck_id)].insert(position, str(task_id))
                cost, trial_specs = self._solution_cost(
                    env, trial, cluster_uavs, contracts, road_signature
                )
                if not np.isfinite(cost):
                    continue
                candidate_rank = self._solution_rank(env, cost, trial_specs)
                # Routine work is allowed to exploit spare corridor capacity,
                # but never by worsening the two emergency-priority terms.
                # Without this hard comparison to the pre-insertion plan, a
                # "least bad" routine insertion could still add an emergency
                # miss when every candidate position was harmful.
                if (
                    task.kind == TaskKind.NORMAL
                    and not self._is_relay(task)
                    and candidate_rank[:2] > base_rank[:2]
                ):
                    balanced_tradeoff = bool(
                        er_hlns_balanced_all_tasks_active(env)
                        and getattr(
                            env.cfg,
                            "hrl_route_plan_balanced_all_tasks_enabled",
                            False,
                        )
                        and getattr(
                            env.cfg,
                            "hrl_route_plan_balanced_all_tasks_allow_emergency_tradeoff",
                            False,
                        )
                        and int(candidate_rank[0]) <= int(base_rank[0])
                        and float(candidate_rank[1])
                        <= float(base_rank[1])
                        + float(
                            max(
                                getattr(
                                    env.cfg,
                                    "hrl_route_plan_balanced_all_tasks_emergency_lateness_tolerance_steps",
                                    0,
                                ),
                                0,
                            )
                        )
                    )
                    if balanced_tradeoff:
                        self.balanced_all_tasks_emergency_tradeoff_count += 1
                    else:
                        self.lexicographic_primary_rejection_count += 1
                        continue
                if (
                    joint_corridor
                    and task.kind == TaskKind.NORMAL
                    and not self._is_relay(task)
                ):
                    marginal_cap = float(
                        max(
                            getattr(
                                env.cfg,
                                "hrl_route_plan_normal_max_marginal_cost_steps",
                                90.0,
                            ),
                            0.0,
                        )
                    )
                    marginal = float(cost - base_cost) if np.isfinite(base_cost) else float("inf")
                    if marginal > marginal_cap + 1e-9:
                        continue
                    trial_emergency_eta = {
                        str(stop.task_id): int(stop.eta_step)
                        for stops in trial_specs.values()
                        for stop in stops
                        if (
                            env.state.tasks.get(str(stop.task_id), None) is not None
                            and env.state.tasks[str(stop.task_id)].kind
                            == TaskKind.EMERGENCY
                        )
                    }
                    max_delay = int(
                        max(
                            getattr(
                                env.cfg,
                                "hrl_route_plan_normal_max_emergency_delay_steps",
                                12,
                            ),
                            0,
                        )
                    )
                    disconnect_profile = self._global_disconnect_profiles(env).get(
                        str(task.task_id), {}
                    )
                    globally_protected = bool(
                        disconnect_profile.get("protected", 0.0)
                    )
                    # Retained local-risk ablation used zero delay. The active
                    # global min-cut constraint instead uses the configured
                    # bounded corridor delay, while the rank gate above still
                    # forbids any added emergency miss or lateness.
                    if (
                        not globally_protected
                        and self._routine_disconnect_risk(env, task) >= 0.55
                    ):
                        max_delay = 0
                    reserve = int(
                        max(
                            getattr(
                                env.cfg,
                                "hrl_route_plan_emergency_deadline_reserve_steps",
                                20,
                            ),
                            0,
                        )
                    )
                    corridor_feasible = True
                    for emergency_id, eta_before in base_emergency_eta.items():
                        eta_after = trial_emergency_eta.get(emergency_id, None)
                        emergency = env.state.tasks.get(emergency_id, None)
                        if eta_after is None or emergency is None:
                            corridor_feasible = False
                            break
                        if int(eta_after) - int(eta_before) > max_delay:
                            corridor_feasible = False
                            break
                        safe_latest = (
                            self._effective_deadline_step(env, emergency) - reserve
                            if self._aggressive_planning_active
                            else int(emergency.deadline_step) - reserve
                        )
                        if int(eta_before) <= safe_latest and int(eta_after) > safe_latest:
                            corridor_feasible = False
                            break
                    if not corridor_feasible:
                        continue
                candidate = (str(truck_id), int(position), float(cost))
                self.lexicographic_comparison_count += 1
                if (
                    best is not None
                    and best_rank is not None
                    and candidate[2] < best[2] - 1e-9
                    and candidate_rank[:-1] > best_rank[:-1]
                ):
                    self.lexicographic_primary_rejection_count += 1
                if (
                    best is None
                    or best_rank is None
                    or candidate_rank < best_rank
                    or (
                        candidate_rank == best_rank
                        and (candidate[0], candidate[1]) < (best[0], best[1])
                    )
                ):
                    best = candidate
                    best_rank = candidate_rank
        return best

    def _build_contracts(
        self,
        env,
        routes: Dict[str, List[str]],
        cluster_uavs: Dict[str, Tuple[str, ...]],
        prior: Dict[str, TaskContract],
    ) -> Dict[str, TaskContract]:
        contracts: Dict[str, TaskContract] = {}
        step_now = int(env.state.step_index)
        # Pilot-only: strict docked-owner rebinding regressed L-C seed114/119
        # (75/70% -> 60/65%) by breaking viable cross-truck recovery chains.
        # Keep the audited implementation below, but retain the stable
        # persistent contract policy for the formal mainline.
        enable_docked_contract_rebinding_experiment = False
        # In a stable road graph, a stock-out contract can safely move to a
        # different cooperative unit. During blockage evolution, automatic
        # stock handoff would race map-driven suffix repair; the existing
        # stalled/deadline transfer mechanisms remain authoritative there.
        stable_road_supply_handoff = len(self._road_signature(env)) == 0
        for truck_id in sorted(routes):
            uavs = cluster_uavs.get(str(truck_id), ())
            next_uav = 0
            for task_id in routes[truck_id]:
                task = env.state.tasks[str(task_id)]
                old = prior.get(str(task_id), None)
                if old is not None and str(old.truck_id) == str(truck_id):
                    owner_state = env.state.agents.get(str(old.owner_agent_id), None)
                    owner_docked_with_contract_truck = bool(
                        owner_state is not None
                        and getattr(owner_state, "follow_target", None) is not None
                        and str(getattr(owner_state, "follow_target", ""))
                        == str(truck_id)
                    )
                    owner_airborne_on_same_contract = bool(
                        owner_state is not None
                        and getattr(owner_state, "follow_target", None) is None
                        and str(
                            dict(
                                getattr(env, "_uav_sortie_contract_task", {})
                            ).get(str(old.owner_agent_id), "")
                        )
                        == str(task_id)
                    )
                    if (
                        owner_state is not None
                        and not bool(getattr(owner_state, "crashed", False))
                        and (
                            not enable_docked_contract_rebinding_experiment
                            or task.kind != TaskKind.EMERGENCY
                            or owner_docked_with_contract_truck
                            or owner_airborne_on_same_contract
                        )
                        and (
                            not stable_road_supply_handoff
                            or self._contract_has_emergency_supply(env, old, task)
                        )
                    ):
                        contracts[str(task_id)] = old
                        continue
                    self.contract_release_count += 1
                relay = self._is_relay(task)
                if task.kind == TaskKind.EMERGENCY or relay:
                    if not uavs:
                        continue
                    if (
                        self._targeted_repairs_enabled()
                        and float(max(getattr(task, "fulfilled_mass_kg", 0.0), 0.0))
                        > 1e-9
                    ):
                        ranked_uavs = tuple(
                            sorted(
                                (str(uid) for uid in uavs),
                                key=lambda uid: self._uav_contract_rank(
                                    env, uid, task, str(truck_id)
                                ),
                            )
                        )
                    elif enable_docked_contract_rebinding_experiment:
                        # A contract must begin with a UAV physically docked
                        # on its assigned cooperative truck.  Retaining a
                        # round-robin ID after recovery/rebinding created
                        # depot-bound owners that could hold a task while the
                        # associated truck had already moved elsewhere.
                        ranked_uavs = tuple(
                            sorted(
                                (str(uid) for uid in uavs),
                                key=lambda uid: self._uav_contract_rank(
                                    env, uid, task, str(truck_id)
                                ),
                            )
                        )
                    else:
                        # Stable persistent assignment: a recovery/rebinding
                        # chain may intentionally leave the owner on another
                        # truck, so do not overwrite it merely for docking.
                        ranked_uavs = tuple(str(uid) for uid in uavs)
                    uav_id = str(ranked_uavs[next_uav % len(ranked_uavs)])
                    next_uav += 1
                    assigned_uavs = (uav_id,)
                    if relay:
                        count = int(max(getattr(env.cfg, "hrl_route_plan_bulk_relay_uav_count", 2), 1))
                        assigned_uavs = tuple(
                            str(uid)
                            for uid in ranked_uavs[: min(count, len(ranked_uavs))]
                        )
                        uav_id = str(assigned_uavs[0])
                    owner = uav_id
                else:
                    uav_id = None
                    assigned_uavs = ()
                    owner = str(truck_id)
                contracts[str(task_id)] = TaskContract(
                    task_id=str(task_id),
                    owner_agent_id=str(owner),
                    truck_id=str(truck_id),
                    uav_id=uav_id,
                    uav_ids=tuple(assigned_uavs),
                    service_mode=BULK_RELAY if relay else DIRECT,
                    created_step=step_now,
                )
        return contracts

    def _v3_emergency_metrics(
        self,
        env,
        specs: Dict[str, List[RouteStop]],
    ) -> Dict[str, float]:
        """Extract same-state predicted emergency metrics for the V3 selector.

        ``RouteStop.deadline_step`` is an execution annotation and may be left
        at its sentinel value during construction.  The task's effective
        deadline is the authoritative value used by the rest of the planner,
        so the selector must use the same helper as ``_solution_rank``.
        """
        completed = 0
        lateness = 0.0
        scheduled = 0
        for stops in specs.values():
            for stop in stops:
                task = env.state.tasks.get(str(stop.task_id), None)
                if task is None or task.kind != TaskKind.EMERGENCY:
                    continue
                scheduled += 1
                deadline = int(self._effective_deadline_step(env, task))
                late = float(max(int(stop.eta_step) - deadline, 0))
                lateness += late
                if late <= 1e-9:
                    completed += 1
        return {
            "tc_predicted_scheduled": float(scheduled),
            "tc_predicted_completed": float(completed),
            "tc_predicted_lateness": float(lateness),
        }

    def _v3_select_candidate(
        self,
        normal_metrics: Dict[str, float],
        emergency_metrics: Dict[str, float],
    ) -> Tuple[bool, str]:
        """Select normal-first only when emergency predictions do not worsen."""
        checks = (
            (
                "tc_completion_degradation",
                float(normal_metrics.get("tc_predicted_completed", 0.0))
                + 1e-9
                < float(emergency_metrics.get("tc_predicted_completed", 0.0)),
            ),
            (
                "tc_lateness_degradation",
                float(normal_metrics.get("tc_predicted_lateness", 0.0))
                > float(emergency_metrics.get("tc_predicted_lateness", 0.0))
                + 1e-9,
            ),
            (
                "emergency_disconnect_miss_degradation",
                float(normal_metrics.get("emergency_disconnect_missed", 0.0))
                > float(emergency_metrics.get("emergency_disconnect_missed", 0.0))
                + 1e-9,
            ),
            (
                "emergency_disconnect_lateness_degradation",
                float(normal_metrics.get("emergency_disconnect_lateness", 0.0))
                > float(emergency_metrics.get("emergency_disconnect_lateness", 0.0))
                + 1e-9,
            ),
        )
        for reason, degraded in checks:
            if degraded:
                self.balanced_all_tasks_v3_fallback_count += 1
                self.balanced_all_tasks_v3_fallback_reason_counts[reason] = int(
                    self.balanced_all_tasks_v3_fallback_reason_counts.get(reason, 0)
                ) + 1
                return False, reason
        return True, "normal_candidate_not_worse"

    def _shadow_total_coverage_metrics(
        self,
        env,
        specs: Dict[str, List[RouteStop]],
    ) -> Dict[str, float]:
        """Count deadline-feasible routine/emergency stops in a shadow plan.

        This is deliberately a planning-time proxy: it uses only the current
        observed road state and the same ETA annotations consumed by the
        route manager.  It is not reported as an achieved completion count.
        The selector uses the emergency metrics as hard guards and consults
        routine coverage only after those guards tie.
        """
        routine_on_time = 0.0
        routine_robust_on_time = 0.0
        routine_late_count = 0.0
        routine_lateness = 0.0
        routine_min_slack = float("inf")
        emergency_on_time = 0.0
        emergency_late_count = 0.0
        emergency_lateness = 0.0
        for stops in specs.values():
            for stop in stops:
                task = env.state.tasks.get(str(stop.task_id), None)
                if task is None or not self._active(task):
                    continue
                deadline = int(self._effective_deadline_step(env, task))
                late = float(max(int(stop.eta_step) - deadline, 0))
                if task.kind == TaskKind.NORMAL and not self._is_relay(task):
                    routine_min_slack = min(routine_min_slack, float(deadline - int(stop.eta_step)))
                    if late <= 1e-9:
                        routine_on_time += 1.0
                        if float(deadline - int(stop.eta_step)) >= float(
                            max(
                                getattr(
                                    env.cfg,
                                    "hrl_route_plan_shadow_total_coverage_min_routine_slack_steps",
                                    0,
                                ),
                                0,
                            )
                        ):
                            routine_robust_on_time += 1.0
                    else:
                        routine_late_count += 1.0
                        routine_lateness += late
                elif task.kind == TaskKind.EMERGENCY:
                    if late <= 1e-9:
                        emergency_on_time += 1.0
                    else:
                        emergency_late_count += 1.0
                        emergency_lateness += late
        return {
            "routine_on_time": float(routine_on_time),
            "routine_robust_on_time": float(routine_robust_on_time),
            "routine_late_count": float(routine_late_count),
            "routine_lateness": float(routine_lateness),
            "routine_min_slack": 0.0 if routine_min_slack == float("inf") else float(routine_min_slack),
            "emergency_on_time": float(emergency_on_time),
            "emergency_late_count": float(emergency_late_count),
            "emergency_lateness": float(emergency_lateness),
            "total_on_time": float(routine_on_time + emergency_on_time),
        }

    def _shadow_total_coverage_select(
        self,
        normal_metrics: Dict[str, float],
        emergency_metrics: Dict[str, float],
        min_gain_tasks: int,
        max_routine_distance_ratio: float,
    ) -> Tuple[bool, str]:
        """Select a normal-first shadow only after emergency non-degradation.

        The emergency-first plan remains the fallback.  A candidate must add
        at least ``min_gain_tasks`` on-time routine tasks and cannot reduce
        predicted emergency coverage or increase emergency lateness.  This
        makes the modification a bounded alternative-plan test rather than a
        permanent normal-task priority rule.
        """
        safe, reason = self._v3_select_candidate(
            normal_metrics,
            emergency_metrics,
        )
        if not safe:
            return False, str(reason)
        if float(normal_metrics.get("tc_predicted_completion_time", 0.0)) > float(
            emergency_metrics.get("tc_predicted_completion_time", 0.0)
        ) + 1e-9:
            return False, "emergency_completion_delay"
        routine_gain = float(normal_metrics.get("routine_robust_on_time", 0.0)) - float(
            emergency_metrics.get("routine_robust_on_time", 0.0)
        )
        if routine_gain + 1e-9 < float(max(int(min_gain_tasks), 1)):
            return False, "insufficient_routine_coverage_gain"
        if float(normal_metrics.get("total_on_time", 0.0)) + 1e-9 < float(
            emergency_metrics.get("total_on_time", 0.0)
        ):
            return False, "total_coverage_degradation"
        baseline_distance = float(
            max(emergency_metrics.get("routine_road_distance_m", 0.0), 0.0)
        )
        candidate_distance = float(
            max(normal_metrics.get("routine_road_distance_m", 0.0), 0.0)
        )
        if baseline_distance <= 1e-9:
            if candidate_distance > 1e-9:
                return False, "routine_distance_increase"
        elif candidate_distance > baseline_distance * float(
            max(float(max_routine_distance_ratio), 1.0)
        ) + 1e-9:
            return False, "routine_distance_increase"
        return True, "routine_coverage_gain"

    def _optimize(
        self,
        env,
        truck_ids: Sequence[str],
        cluster_uavs: Dict[str, Tuple[str, ...]],
        road_signature: Tuple[Tuple[int, int], ...],
        prior_contracts: Dict[str, TaskContract],
    ) -> Tuple[Dict[str, ClusterRoute], Dict[str, TaskContract], float]:
        optimize_started = time.perf_counter()
        self.alns_replan_count += 1
        tasks = sorted(
            (task for task in env.state.tasks.values() if self._active(task)),
            key=self._task_order_key,
        )
        balanced_all_tasks = bool(
            er_hlns_balanced_all_tasks_active(env)
            and getattr(
                env.cfg,
                "hrl_route_plan_balanced_all_tasks_enabled",
                False,
            )
        )
        # The formal constructor is emergency-first.  The candidate pilot
        # deliberately starts with direct NORMAL work, then inserts UAV
        # contracts around that route skeleton.  This prevents the normal
        # queue from becoming an infeasible tail before it ever receives a
        # truck assignment; safety checks still govern every insertion.
        if balanced_all_tasks and bool(
            getattr(
                env.cfg,
                "hrl_route_plan_balanced_all_tasks_normal_first_enabled",
                False,
            )
        ):
            routine_tasks = [
                task
                for task in tasks
                if task.kind == TaskKind.NORMAL and not self._is_relay(task)
            ]
            aerial_tasks = [
                task
                for task in tasks
                if task not in routine_tasks
            ]
            tasks = routine_tasks + aerial_tasks
        provisional = dict(prior_contracts)
        stable_road_supply_handoff = len(road_signature) == 0
        initial_inventory_plan = bool(
            not self._emergency_inventory_initial_plan_done
        )
        self._enforce_emergency_inventory_budget_active = bool(
            getattr(
                env.cfg,
                "hrl_route_plan_enforce_emergency_inventory_budget_enabled",
                False,
            )
            and not self._emergency_inventory_initial_plan_done
        )
        self._emergency_inventory_initial_plan_done = True
        self._emergency_capacity_target_by_truck = (
            self._capacity_aware_emergency_targets(
                env, truck_ids, cluster_uavs
            )
        )
        balance_configured = bool(
            getattr(
                env.cfg,
                "hrl_route_plan_global_emergency_balance_enabled",
                False,
            )
        )
        # The first full plan is diagnosed from the unmodified assignment.
        # Once that diagnosis is made, retain it across event-driven suffix
        # repairs so a shrinking active queue cannot silently disable the
        # balancing policy halfway through an episode.
        if self._emergency_balance_episode_latched is None:
            self._emergency_balance_active = False
        else:
            self._emergency_balance_active = bool(
                self._emergency_balance_episode_latched
            )

        def construct_routes(
            ordered_tasks: Sequence[Any],
            initial_routes: Optional[Dict[str, List[str]]] = None,
        ) -> Tuple[Dict[str, List[str]], List[Any]]:
            candidate_routes: Dict[str, List[str]] = {
                str(tid): list((initial_routes or {}).get(str(tid), []))
                for tid in truck_ids
            }
            rejected: List[Any] = []
            for task in ordered_tasks:
                if (
                    balanced_all_tasks
                    and task.kind == TaskKind.NORMAL
                    and not self._is_relay(task)
                ):
                    self.balanced_all_tasks_normal_candidate_count += 1
                old = prior_contracts.get(str(task.task_id), None)
                fixed_truck = (
                    old.truck_id
                    if (
                        old is not None
                        and old.truck_id in candidate_routes
                        and (
                            not stable_road_supply_handoff
                            or self._contract_has_emergency_supply(env, old, task)
                        )
                    )
                    else None
                )
                insertion = self._best_insertion(
                    env,
                    str(task.task_id),
                    candidate_routes,
                    cluster_uavs,
                    provisional,
                    road_signature,
                    fixed_truck=fixed_truck,
                )
                if insertion is None:
                    rejected.append(task)
                    continue
                truck_id, position, _ = insertion
                candidate_routes[truck_id].insert(position, str(task.task_id))
                if (
                    balanced_all_tasks
                    and task.kind == TaskKind.NORMAL
                    and not self._is_relay(task)
                ):
                    self.balanced_all_tasks_normal_assignment_count += 1
            return candidate_routes, rejected

        # Preserve the established emergency-first constructor as the
        # baseline.  On the first full plan only, also seed one candidate with
        # globally exposed routine tasks. Subsequent emergency insertion may
        # still move ahead of them, and the complete six-term rank selects the
        # candidate, so this adds search coverage rather than a priority
        # exception.
        routes, rejected_tasks = construct_routes(tasks)
        shadow_total_coverage = bool(
            er_hlns_balanced_all_tasks_v3_active(env)
            and getattr(
                env.cfg,
                "hrl_route_plan_shadow_total_coverage_enabled",
                False,
            )
        )
        if shadow_total_coverage:
            # Build both alternatives from the same observed state.  V6's
            # baseline is emergency-first; the shadow deliberately exposes a
            # normal-first ordering without changing the public environment.
            emergency_first_tasks = sorted(
                (task for task in env.state.tasks.values() if self._active(task)),
                key=self._task_order_key,
            )
            routine_first_tasks = [
                task
                for task in emergency_first_tasks
                if task.kind == TaskKind.NORMAL and not self._is_relay(task)
            ]
            routine_first_tasks.extend(
                task
                for task in emergency_first_tasks
                if not (
                    task.kind == TaskKind.NORMAL and not self._is_relay(task)
                )
            )
            emergency_routes, emergency_rejected = construct_routes(
                emergency_first_tasks
            )
            routine_routes, routine_rejected = construct_routes(
                routine_first_tasks
            )
            emergency_contracts = self._build_contracts(
                env,
                emergency_routes,
                cluster_uavs,
                prior_contracts,
            )
            routine_contracts = self._build_contracts(
                env,
                routine_routes,
                cluster_uavs,
                prior_contracts,
            )
            emergency_cost, emergency_specs = self._solution_cost(
                env,
                emergency_routes,
                cluster_uavs,
                emergency_contracts,
                road_signature,
            )
            routine_cost, routine_specs = self._solution_cost(
                env,
                routine_routes,
                cluster_uavs,
                routine_contracts,
                road_signature,
            )
            emergency_rank = self._solution_rank(
                env, emergency_cost, emergency_specs
            )
            routine_rank = self._solution_rank(env, routine_cost, routine_specs)

            def _enrich_shadow_metrics(
                specs: Dict[str, List[RouteStop]],
                rank: Tuple[float, ...],
            ) -> Dict[str, float]:
                metrics = self._v3_emergency_metrics(env, specs)
                metrics["tc_predicted_missed"] = float(
                    rank[0] if len(rank) > 0 else 0.0
                )
                metrics["tc_predicted_completed"] = float(
                    max(
                        sum(
                            1
                            for task in env.state.tasks.values()
                            if task.kind == TaskKind.EMERGENCY
                            and self._active(task)
                        )
                        - metrics["tc_predicted_missed"],
                        0.0,
                    )
                )
                metrics["tc_predicted_lateness"] = float(
                    rank[1] if len(rank) > 1 else 0.0
                )
                metrics["tc_predicted_completion_time"] = float(
                    rank[5] if len(rank) > 5 else 0.0
                )
                metrics["emergency_disconnect_missed"] = float(
                    rank[3] if len(rank) > 3 else 0.0
                )
                metrics["emergency_disconnect_lateness"] = float(
                    rank[4] if len(rank) > 4 else 0.0
                )
                metrics.update(self._shadow_total_coverage_metrics(env, specs))
                metrics["routine_road_distance_m"] = float(
                    sum(
                        max(float(stop.planned_road_distance_m), 0.0)
                        for stops in specs.values()
                        for stop in stops
                        if (
                            env.state.tasks.get(str(stop.task_id), None) is not None
                            and env.state.tasks[str(stop.task_id)].kind == TaskKind.NORMAL
                            and not self._is_relay(env.state.tasks[str(stop.task_id)])
                        )
                    )
                )
                metrics["emergency_road_distance_m"] = float(
                    sum(
                        max(float(stop.planned_road_distance_m), 0.0)
                        for stops in specs.values()
                        for stop in stops
                        if (
                            env.state.tasks.get(str(stop.task_id), None) is not None
                            and env.state.tasks[str(stop.task_id)].kind == TaskKind.EMERGENCY
                        )
                    )
                )
                return metrics

            emergency_metrics = _enrich_shadow_metrics(
                emergency_specs, emergency_rank
            )
            routine_metrics = _enrich_shadow_metrics(routine_specs, routine_rank)
            self.shadow_total_coverage_candidate_count += 1
            selected_routine, selector_reason = self._shadow_total_coverage_select(
                routine_metrics,
                emergency_metrics,
                int(
                    max(
                        getattr(
                            env.cfg,
                            "hrl_route_plan_shadow_total_coverage_min_gain_tasks",
                            1,
                        ),
                        1,
                    )
                ),
                float(
                    max(
                        getattr(
                            env.cfg,
                            "hrl_route_plan_shadow_total_coverage_max_routine_distance_ratio",
                            1.0,
                        ),
                        1.0,
                    )
                ),
            )
            self.shadow_total_coverage_last_diagnostics = {
                "selected": "routine_first" if selected_routine else "emergency_first",
                "reason": str(selector_reason),
                "routine_on_time": float(routine_metrics.get("routine_on_time", 0.0)),
                "routine_robust_on_time": float(routine_metrics.get("routine_robust_on_time", 0.0)),
                "emergency_on_time": float(emergency_metrics.get("emergency_on_time", 0.0)),
                "routine_total_on_time": float(routine_metrics.get("total_on_time", 0.0)),
                "emergency_total_on_time": float(emergency_metrics.get("total_on_time", 0.0)),
                "routine_min_slack": float(routine_metrics.get("routine_min_slack", 0.0)),
                "emergency_min_slack": float(emergency_metrics.get("routine_min_slack", 0.0)),
                "routine_cost": float(routine_cost if np.isfinite(routine_cost) else 1e30),
                "emergency_cost": float(emergency_cost if np.isfinite(emergency_cost) else 1e30),
                "routine_road_distance_m": float(routine_metrics.get("routine_road_distance_m", 0.0)),
                "emergency_routine_road_distance_m": float(emergency_metrics.get("routine_road_distance_m", 0.0)),
                "routine_tc_completed": float(routine_metrics.get("tc_predicted_completed", 0.0)),
                "emergency_tc_completed": float(emergency_metrics.get("tc_predicted_completed", 0.0)),
            }
            if selected_routine:
                routes = routine_routes
                rejected_tasks = routine_rejected
                self.shadow_total_coverage_accept_count += 1
                if not self.shadow_total_coverage_first_accept_diagnostics:
                    self.shadow_total_coverage_first_accept_diagnostics = dict(
                        self.shadow_total_coverage_last_diagnostics
                    )
            else:
                routes = emergency_routes
                rejected_tasks = emergency_rejected
                self.shadow_total_coverage_reject_count += 1
        if er_hlns_balanced_all_tasks_v3_active(env) and not shadow_total_coverage:
            # Build an emergency-first alternative from this same state and
            # compare only predicted emergency terms before publishing either.
            emergency_first_tasks = sorted(
                (task for task in env.state.tasks.values() if self._active(task)),
                key=self._task_order_key,
            )
            emergency_routes, emergency_rejected = construct_routes(
                emergency_first_tasks
            )
            normal_contracts = self._build_contracts(
                env, routes, cluster_uavs, prior_contracts
            )
            emergency_contracts = self._build_contracts(
                env, emergency_routes, cluster_uavs, prior_contracts
            )
            normal_cost, normal_specs = self._solution_cost(
                env, routes, cluster_uavs, normal_contracts, road_signature
            )
            emergency_cost, emergency_specs = self._solution_cost(
                env,
                emergency_routes,
                cluster_uavs,
                emergency_contracts,
                road_signature,
            )
            self.balanced_all_tasks_v3_normal_candidate_count += 1
            self.balanced_all_tasks_v3_emergency_candidate_count += 1
            normal_metrics = self._v3_emergency_metrics(env, normal_specs)
            emergency_metrics = self._v3_emergency_metrics(env, emergency_specs)
            normal_rank = self._solution_rank(env, normal_cost, normal_specs)
            emergency_rank = self._solution_rank(
                env, emergency_cost, emergency_specs
            )
            # Use the same lexicographic emergency terms that govern the
            # planner's own acceptance.  RouteStop annotations can be absent
            # during a suffix rebuild, whereas the rank is already computed
            # from effective task deadlines and disconnect profiles.
            total_active_emergency = float(
                sum(
                    1
                    for task in env.state.tasks.values()
                    if task.kind == TaskKind.EMERGENCY and self._active(task)
                )
            )
            normal_metrics["tc_predicted_missed"] = float(
                normal_rank[0] if len(normal_rank) > 0 else 0.0
            )
            emergency_metrics["tc_predicted_missed"] = float(
                emergency_rank[0] if len(emergency_rank) > 0 else 0.0
            )
            normal_metrics["tc_predicted_completed"] = float(
                max(total_active_emergency - normal_metrics["tc_predicted_missed"], 0.0)
            )
            emergency_metrics["tc_predicted_completed"] = float(
                max(total_active_emergency - emergency_metrics["tc_predicted_missed"], 0.0)
            )
            normal_metrics["tc_predicted_lateness"] = float(
                normal_rank[1] if len(normal_rank) > 1 else normal_metrics.get("tc_predicted_lateness", 0.0)
            )
            emergency_metrics["tc_predicted_lateness"] = float(
                emergency_rank[1] if len(emergency_rank) > 1 else emergency_metrics.get("tc_predicted_lateness", 0.0)
            )
            for metrics, rank in (
                (normal_metrics, normal_rank),
                (emergency_metrics, emergency_rank),
            ):
                metrics["emergency_disconnect_missed"] = float(
                    rank[3] if len(rank) > 3 else 0.0
                )
                metrics["emergency_disconnect_lateness"] = float(
                    rank[4] if len(rank) > 4 else 0.0
                )
            normal_feasible = bool(np.isfinite(normal_cost) and normal_specs)
            emergency_feasible = bool(
                np.isfinite(emergency_cost) and emergency_specs
            )
            if not normal_feasible:
                selected_normal = False
                selector_reason = "normal_candidate_infeasible"
                self.balanced_all_tasks_v3_fallback_count += 1
                self.balanced_all_tasks_v3_fallback_reason_counts[
                    selector_reason
                ] = int(
                    self.balanced_all_tasks_v3_fallback_reason_counts.get(
                        selector_reason, 0
                    )
                    + 1
                )
            elif not emergency_feasible:
                selected_normal = True
                selector_reason = "emergency_candidate_infeasible"
            else:
                selected_normal, selector_reason = self._v3_select_candidate(
                    normal_metrics, emergency_metrics
                )
            self.balanced_all_tasks_v3_last_diagnostics = {
                "normal_candidate": True,
                "emergency_candidate": True,
                "selected": "normal" if selected_normal else "emergency",
                "fallback_reason": str(selector_reason),
                "normal_predicted_tc_completed": float(
                    normal_metrics.get("tc_predicted_completed", 0.0)
                ),
                "emergency_predicted_tc_completed": float(
                    emergency_metrics.get("tc_predicted_completed", 0.0)
                ),
                "normal_predicted_tc_lateness": float(
                    normal_metrics.get("tc_predicted_lateness", 0.0)
                ),
                "emergency_predicted_tc_lateness": float(
                    emergency_metrics.get("tc_predicted_lateness", 0.0)
                ),
                "normal_emergency_disconnect_missed": float(
                    normal_metrics.get("emergency_disconnect_missed", 0.0)
                ),
                "emergency_emergency_disconnect_missed": float(
                    emergency_metrics.get("emergency_disconnect_missed", 0.0)
                ),
                "normal_emergency_disconnect_lateness": float(
                    normal_metrics.get("emergency_disconnect_lateness", 0.0)
                ),
                "emergency_emergency_disconnect_lateness": float(
                    emergency_metrics.get("emergency_disconnect_lateness", 0.0)
                ),
            }
            if selected_normal:
                self.balanced_all_tasks_v3_selected_normal_count += 1
            else:
                routes = emergency_routes
                rejected_tasks = emergency_rejected
                self.balanced_all_tasks_v3_selected_emergency_count += 1
        profiles = self._global_disconnect_profiles(env)
        if profiles and not prior_contracts:
            protected_first = sorted(
                tasks,
                key=lambda task: (
                    0
                    if (
                        task.kind == TaskKind.NORMAL
                        and bool(
                            profiles.get(str(task.task_id), {}).get(
                                "protected", 0.0
                            )
                        )
                    )
                    else 1,
                    self._task_order_key(task),
                ),
            )
            head_tasks = [
                task
                for task in protected_first
                if bool(
                    profiles.get(str(task.task_id), {}).get(
                        "head_protected", 0.0
                    )
                )
            ]
            seeded_heads: Dict[str, List[str]] = {
                str(tid): [] for tid in truck_ids
            }
            unused_trucks = {str(tid) for tid in truck_ids}
            for task in sorted(
                head_tasks,
                key=lambda item: (
                    profiles[str(item.task_id)]["cut_capacity"],
                    str(item.task_id),
                ),
            ):
                choices = []
                for truck_id in sorted(unused_trucks):
                    truck = env.state.agents.get(str(truck_id), None)
                    if truck is None or truck.node is None:
                        continue
                    road = float(
                        env._decision_shortest_path_distance(
                            int(truck.node), int(task.demand_node)
                        )
                    )
                    if np.isfinite(road):
                        choices.append((road, str(truck_id)))
                if not choices:
                    continue
                _, selected_truck = min(choices)
                seeded_heads[selected_truck].append(str(task.task_id))
                unused_trucks.discard(selected_truck)
            seeded_ids = {
                task_id for values in seeded_heads.values() for task_id in values
            }
            seeded_order = [
                task
                for task in protected_first
                if str(task.task_id) not in seeded_ids
            ]
            seeded_routes, seeded_rejected = construct_routes(
                seeded_order, initial_routes=seeded_heads
            )
            # Insertion remains free to explore every position; publish the
            # weak-cut commitments as heads only for this candidate before
            # evaluating its complete emergency feasibility.
            for truck_id, head_ids_for_truck in seeded_heads.items():
                if not head_ids_for_truck:
                    continue
                remainder = [
                    task_id
                    for task_id in seeded_routes[truck_id]
                    if task_id not in set(head_ids_for_truck)
                ]
                seeded_routes[truck_id] = list(head_ids_for_truck) + remainder
            baseline_cost, baseline_specs = self._solution_cost(
                env, routes, cluster_uavs, provisional, road_signature
            )
            seeded_cost, seeded_specs = self._solution_cost(
                env, seeded_routes, cluster_uavs, provisional, road_signature
            )
            if self._solution_rank(env, seeded_cost, seeded_specs) < self._solution_rank(
                env, baseline_cost, baseline_specs
            ):
                routes = seeded_routes
                rejected_tasks = seeded_rejected

        if (
            initial_inventory_plan
            and bool(
                getattr(
                    env.cfg,
                    "hrl_route_plan_initial_emergency_capacity_repair_enabled",
                    False,
                )
            )
        ):
            capacities = {
                str(truck_id): int(
                    self._cluster_emergency_capacity_units(
                        env,
                        str(truck_id),
                        cluster_uavs.get(str(truck_id), ()),
                    )
                )
                for truck_id in routes
            }

            def emergency_counts_for(
                candidate_routes: Dict[str, List[str]],
            ) -> Dict[str, int]:
                return {
                    str(truck_id): int(
                        sum(
                            1
                            for task_id in task_ids
                            if (
                                env.state.tasks.get(str(task_id), None)
                                is not None
                                and env.state.tasks[str(task_id)].kind
                                == TaskKind.EMERGENCY
                            )
                        )
                    )
                    for truck_id, task_ids in candidate_routes.items()
                }

            moved_contracts = 0
            while True:
                counts = emergency_counts_for(routes)
                overloaded = [
                    truck_id
                    for truck_id in sorted(routes)
                    if counts[truck_id] > capacities[truck_id]
                ]
                receivers = [
                    truck_id
                    for truck_id in sorted(routes)
                    if counts[truck_id] < capacities[truck_id]
                    and cluster_uavs.get(str(truck_id), ())
                ]
                if not overloaded or not receivers:
                    break
                best_move = None
                best_move_key = None
                for source_truck in overloaded:
                    for source_position, task_id in enumerate(
                        routes[source_truck]
                    ):
                        task = env.state.tasks.get(str(task_id), None)
                        if task is None or task.kind != TaskKind.EMERGENCY:
                            continue
                        emergency_ordinal = int(
                            sum(
                                1
                                for preceding_id in routes[source_truck][
                                    : source_position + 1
                                ]
                                if (
                                    env.state.tasks.get(
                                        str(preceding_id), None
                                    )
                                    is not None
                                    and env.state.tasks[
                                        str(preceding_id)
                                    ].kind
                                    == TaskKind.EMERGENCY
                                )
                            )
                        )
                        # Earlier contracts are already backed by the unit's
                        # physical packages. Move only the first genuinely
                        # unbacked tail contract (5th when capacity is four),
                        # preserving the established execution prefix.
                        if (
                            bool(
                                getattr(
                                    env.cfg,
                                    "hrl_route_plan_capacity_repair_tail_only_enabled",
                                    False,
                                )
                            )
                            and emergency_ordinal <= capacities[source_truck]
                        ):
                            continue
                        for target_truck in receivers:
                            for target_position in range(
                                len(routes[target_truck]) + 1
                            ):
                                trial = {
                                    truck_id: list(task_ids)
                                    for truck_id, task_ids in routes.items()
                                }
                                trial[source_truck].pop(source_position)
                                trial[target_truck].insert(
                                    target_position, str(task_id)
                                )
                                trial_cost, trial_specs = self._solution_cost(
                                    env,
                                    trial,
                                    cluster_uavs,
                                    provisional,
                                    road_signature,
                                )
                                if not np.isfinite(trial_cost):
                                    continue
                                trial_rank = self._solution_rank(
                                    env, trial_cost, trial_specs
                                )
                                key = (
                                    trial_rank,
                                    str(task_id),
                                    str(source_truck),
                                    str(target_truck),
                                    int(target_position),
                                )
                                if best_move_key is None or key < best_move_key:
                                    best_move_key = key
                                    best_move = trial
                if best_move is None:
                    break
                routes = best_move
                moved_contracts += 1
            if moved_contracts > 0:
                self.emergency_capacity_repair_count += 1
                self.emergency_capacity_contract_move_count += int(
                    moved_contracts
                )

        baseline_emergency_counts = [
            sum(
                1
                for task_id in task_ids
                if (
                    env.state.tasks.get(str(task_id), None) is not None
                    and env.state.tasks[str(task_id)].kind == TaskKind.EMERGENCY
                )
            )
            for task_ids in routes.values()
        ]
        balance_trigger_count = int(
            max(
                getattr(
                    env.cfg,
                    "hrl_route_plan_global_emergency_balance_trigger_count",
                    5,
                ),
                1,
            )
        )
        initial_balance_decision = bool(
            self._emergency_balance_episode_latched is None
        )
        balance_triggered = bool(
            initial_balance_decision
            and balance_configured
            and baseline_emergency_counts
            and max(baseline_emergency_counts) >= balance_trigger_count
        )
        self.emergency_balance_baseline_max_count = int(
            max(
                self.emergency_balance_baseline_max_count,
                max(baseline_emergency_counts, default=0),
            )
        )
        if balance_triggered:
            self.emergency_balance_trigger_count += 1
            self._emergency_balance_active = True
            balanced_routes, balanced_rejected = construct_routes(tasks)
            baseline_cost, baseline_specs = self._solution_cost(
                env, routes, cluster_uavs, provisional, road_signature
            )
            balanced_cost, balanced_specs = self._solution_cost(
                env,
                balanced_routes,
                cluster_uavs,
                provisional,
                road_signature,
            )
            baseline_rank = self._solution_rank(
                env, baseline_cost, baseline_specs
            )
            balanced_rank = self._solution_rank(
                env, balanced_cost, balanced_specs
            )
            if balanced_rank[:2] <= baseline_rank[:2] and balanced_rank < baseline_rank:
                routes = balanced_routes
                rejected_tasks = balanced_rejected
                self._emergency_balance_episode_latched = True
            else:
                self._emergency_balance_active = False
                self._emergency_balance_episode_latched = False
        elif initial_balance_decision:
            self._emergency_balance_active = False
            self._emergency_balance_episode_latched = False

        for task in rejected_tasks:
            self._feedback.append(
                PlannerFeedback(
                    step=int(env.state.step_index),
                    reason="no_feasible_route_insertion",
                    task_id=str(task.task_id),
                    detail=str(getattr(task, "service_mode", DIRECT)),
                )
            )

        contracts = self._build_contracts(env, routes, cluster_uavs, prior_contracts)
        best_cost, best_specs = self._solution_cost(
            env, routes, cluster_uavs, contracts, road_signature
        )
        best_rank = self._solution_rank(env, best_cost, best_specs)
        best_routes = {key: list(value) for key, value in routes.items()}

        iterations = int(
            max(getattr(env.cfg, "hrl_route_plan_alns_iterations", 4), 0)
        )
        evaluation_budget = int(
            max(
                getattr(
                    env.cfg,
                    "hrl_route_plan_alns_objective_evaluation_budget",
                    0,
                ),
                0,
            )
        )
        protected = {
            str(task.task_id)
            for task in tasks
            if task.status == TaskStatus.CLAIMED
            or (
                prior_contracts.get(str(task.task_id), None) is not None
                and prior_contracts[str(task.task_id)].uav_id is not None
                and env.state.agents.get(
                    str(prior_contracts[str(task.task_id)].uav_id), None
                ) is not None
                and env.state.agents[
                    str(prior_contracts[str(task.task_id)].uav_id)
                ].follow_target
                is None
            )
        }
        for iteration in range(iterations):
            if evaluation_budget > 0 and self.alns_objective_evaluation_count >= evaluation_budget:
                break
            removable = [
                task_id
                for task_id in sorted(
                    task_id for values in best_routes.values() for task_id in values
                )
                if task_id not in protected
            ]
            if not removable:
                break
            self.alns_iteration_count += 1
            remove_count = min(2, len(removable))
            offset = int(iteration % len(removable))
            removed = [
                removable[(offset + index) % len(removable)]
                for index in range(remove_count)
            ]
            self.alns_destroyed_assignment_count += int(len(removed))
            trial = {
                truck_id: [
                    task_id for task_id in task_ids if task_id not in set(removed)
                ]
                for truck_id, task_ids in best_routes.items()
            }
            feasible = True
            for task_id in sorted(
                removed, key=lambda tid: self._task_order_key(env.state.tasks[tid])
            ):
                self.alns_repair_attempt_count += 1
                old = prior_contracts.get(str(task_id), None)
                task = env.state.tasks[str(task_id)]
                fixed_truck = (
                    old.truck_id
                    if (
                        old is not None
                        and old.truck_id in trial
                        and (
                            not stable_road_supply_handoff
                            or self._contract_has_emergency_supply(env, old, task)
                        )
                    )
                    else None
                )
                insertion = self._best_insertion(
                    env,
                    task_id,
                    trial,
                    cluster_uavs,
                    contracts,
                    road_signature,
                    fixed_truck=fixed_truck,
                )
                if insertion is None:
                    feasible = False
                    break
                self.alns_repair_feasible_count += 1
                truck_id, position, _ = insertion
                trial[truck_id].insert(position, task_id)
            if not feasible:
                continue
            trial_contracts = self._build_contracts(
                env, trial, cluster_uavs, prior_contracts
            )
            trial_cost, trial_specs = self._solution_cost(
                env, trial, cluster_uavs, trial_contracts, road_signature
            )
            trial_rank = self._solution_rank(env, trial_cost, trial_specs)
            self.lexicographic_comparison_count += 1
            if (
                trial_cost + 1e-9 < best_cost
                and trial_rank[:-1] > best_rank[:-1]
            ):
                self.lexicographic_primary_rejection_count += 1
            if trial_rank < best_rank:
                best_routes = trial
                contracts = trial_contracts
                best_cost = float(trial_cost)
                best_rank = trial_rank
                self.alns_accepted_count += 1
                self.alns_improvement_count += 1

        contracts = self._build_contracts(
            env, best_routes, cluster_uavs, prior_contracts
        )
        if evaluation_budget > 0 and self.alns_objective_evaluation_count >= evaluation_budget:
            final_cost, stop_specs = float(best_cost), best_specs
        else:
            final_cost, stop_specs = self._solution_cost(
                env, best_routes, cluster_uavs, contracts, road_signature
            )
        cluster_routes = {
            str(truck_id): ClusterRoute(
                truck_id=str(truck_id),
                uav_ids=cluster_uavs.get(str(truck_id), ()),
                stops=list(stop_specs.get(str(truck_id), [])),
            )
            for truck_id in truck_ids
        }
        self.alns_wall_clock_time_s += float(
            time.perf_counter() - optimize_started
        )
        return cluster_routes, contracts, float(final_cost)

    def _episode_changed(self, env) -> bool:
        token = (
            id(env.state.tasks),
            tuple(sorted(str(task_id) for task_id in env.state.tasks)),
        )
        changed = bool(
            self._episode_token is None
            or int(env.state.step_index) < self.last_seen_step
            or token != self._episode_token
        )
        self._episode_token = token
        return changed

    def _route_feedback_is_new(
        self,
        truck_id: str,
        task_id: str,
        reason: str,
        source_node: int,
        fingerprint: Any,
    ) -> bool:
        # A verified current-head failure remains actionable: later contract
        # and route changes can make the next repair succeed even when the
        # truck has not moved.  Only bare road-version changes are suppressed.
        # The historical duplicate-state guard remains below for auditability.
        suppress_repeated_impacted_head_experiment = False
        if not suppress_repeated_impacted_head_experiment:
            return True
        key = (str(truck_id), str(task_id), str(reason))
        state = (int(source_node), fingerprint)
        if self._last_actionable_route_feedback.get(key, None) == state:
            return False
        self._last_actionable_route_feedback[key] = state
        return True

    def _current_route_feedback(
        self,
        env,
        road_signature: Tuple[Tuple[int, int], ...],
    ) -> List[PlannerFeedback]:
        feedback: List[PlannerFeedback] = []
        if self.plan is None:
            return feedback
        detour_ratio = float(
            max(getattr(env.cfg, "hrl_route_plan_detour_trigger_ratio", 1.35), 1.0)
        )
        for truck_id, route in self.plan.routes.items():
            stop = route.current(env)
            if stop is None:
                continue
            state = env.state.agents.get(str(truck_id), None)
            if state is None or state.node is None:
                feedback.append(
                    PlannerFeedback(
                        step=int(env.state.step_index),
                        reason="truck_unavailable",
                        truck_id=str(truck_id),
                        task_id=str(stop.task_id),
                        suffix_repair_required=True,
                    )
                )
                continue
            target = (
                int(stop.selected_anchor)
                if stop.selected_anchor is not None
                else int(stop.target_node)
            )
            current = float(
                env._decision_shortest_path_distance(int(state.node), target)
            )
            if not np.isfinite(current):
                switched = False
                for backup in stop.anchor_nodes[1:]:
                    distance = float(
                        env._decision_shortest_path_distance(
                            int(state.node), int(backup)
                        )
                    )
                    if np.isfinite(distance):
                        stop.selected_anchor = int(backup)
                        stop.target_node = int(backup)
                        stop.planned_road_distance_m = float(distance)
                        self.anchor_backup_switch_count += 1
                        switched = True
                        break
                if not switched:
                    if self._route_feedback_is_new(
                        str(truck_id),
                        str(stop.task_id),
                        "current_stop_unreachable",
                        int(state.node),
                        "unreachable",
                    ):
                        feedback.append(
                            PlannerFeedback(
                                step=int(env.state.step_index),
                                reason="current_stop_unreachable",
                                truck_id=str(truck_id),
                                task_id=str(stop.task_id),
                                suffix_repair_required=True,
                            )
                        )
                continue
            self._last_actionable_route_feedback.pop(
                (str(truck_id), str(stop.task_id), "current_stop_unreachable"),
                None,
            )
            planned = float(max(stop.planned_road_distance_m, 1.0))
            if current / planned > detour_ratio + 1e-9:
                if self._route_feedback_is_new(
                    str(truck_id),
                    str(stop.task_id),
                    "detour_ratio_exceeded",
                    int(state.node),
                    round(float(current / planned), 2),
                ):
                    feedback.append(
                        PlannerFeedback(
                            step=int(env.state.step_index),
                            reason="detour_ratio_exceeded",
                            truck_id=str(truck_id),
                            task_id=str(stop.task_id),
                            detail=f"{current / planned:.3f}>{detour_ratio:.3f}",
                            suffix_repair_required=True,
                        )
                    )
            else:
                self._last_actionable_route_feedback.pop(
                    (str(truck_id), str(stop.task_id), "detour_ratio_exceeded"),
                    None,
                )
        if self.plan.road_signature != road_signature:
            # A road-version change invalidates shortest-path caches, but does
            # not by itself invalidate a route.  The checks above already
            # request repair when the executable head becomes unreachable or
            # its detour ratio is materially worse.  Accepting unrelated road
            # observations here prevents global suffix churn.
            self.plan.road_signature = tuple(road_signature)
        return feedback

    def _aggressive_repair_needed(
        self,
        env,
        truck_ids: Sequence[str],
        clusters: Dict[str, Tuple[str, ...]],
        road_signature: Tuple[Tuple[int, int], ...],
    ) -> bool:
        """Enable stronger planning only when the baseline structure is insufficient."""
        if any(
            task.kind == TaskKind.NORMAL
            and self._active(task)
            and self._is_relay(task)
            for task in env.state.tasks.values()
        ):
            return True
        previous = bool(self._aggressive_planning_active)
        self._aggressive_planning_active = False
        try:
            for task in env.state.tasks.values():
                if task.kind != TaskKind.EMERGENCY or not self._active(task):
                    continue
                feasible = False
                effective_deadline = self._effective_deadline_step(env, task)
                for truck_id in truck_ids:
                    alternative = self._single_emergency_eta(
                        env,
                        str(truck_id),
                        task,
                        clusters,
                        road_signature,
                    )
                    if alternative is None:
                        continue
                    _, _, stop = alternative
                    if int(stop.eta_step) <= int(effective_deadline):
                        feasible = True
                        break
                if not feasible:
                    return True
            return False
        finally:
            self._aggressive_planning_active = previous

    def _initial_lifeline_ordering_is_spatially_safe(
        self,
        env,
        truck_ids: Sequence[str],
    ) -> bool:
        """Use lifeline-first initialization only for a compact TC workload.

        When many emergency tasks are several UAV ranges away from every
        initial truck, a pure lifeline ordering destroys the spatial corridor
        structure built by ALNS.  Compact workloads instead benefit from the
        tighter lifeline order because it prevents feasible suffixes from
        starving behind nearby routine work.
        """
        emergencies = [
            task
            for task in env.state.tasks.values()
            if task.kind == TaskKind.EMERGENCY and self._active(task)
        ]
        truck_nodes = [
            int(env.state.agents[str(truck_id)].node)
            for truck_id in truck_ids
            if env.state.agents.get(str(truck_id), None) is not None
            and env.state.agents[str(truck_id)].node is not None
        ]
        if not emergencies or not truck_nodes:
            return False
        max_sortie = float(
            max(getattr(env.cfg, "uav_max_sortie_m", 6000.0), 1.0)
        )
        extended_corridor_m = 1.60 * max_sortie
        far_count = 0
        for task in emergencies:
            nearest = min(
                float(
                    env._decision_shortest_path_distance(
                        int(node), int(task.demand_node)
                    )
                )
                for node in truck_nodes
            )
            if nearest > extended_corridor_m + 1e-9:
                far_count += 1
        far_fraction = float(far_count / max(len(emergencies), 1))
        uav_payload = float(
            max(
                getattr(
                    env.cfg,
                    "uav_emergency_payload_kg",
                    getattr(env.cfg, "uav_payload_capacity_kg", 40.0),
                ),
                1e-6,
            )
        )
        multi_sortie_count = int(
            sum(
                1
                for task in emergencies
                if self._remaining_demand_kg(task) > uav_payload + 1e-9
            )
        )
        multi_sortie_fraction = float(
            multi_sortie_count / max(len(emergencies), 1)
        )
        spatial_payload_overload = bool(
            far_fraction > (1.0 / 3.0) + 1e-9
            and multi_sortie_fraction >= 0.50 - 1e-9
        )
        return not spatial_payload_overload

    def _install_plan(
        self,
        env,
        reason: str,
        road_signature: Tuple[Tuple[int, int], ...],
        prior_contracts: Dict[str, TaskContract],
    ) -> None:
        truck_ids = self._live_trucks(env)
        clusters = self._cluster_uavs(env, truck_ids)
        # The frozen planner is the stable default.  Stronger ordering and
        # inventory constraints are enabled only for a concrete execution
        # repair, never merely because a changing road snapshot contains a
        # relay task.  This prevents a normal road observation from rebuilding
        # every emergency contract in an already feasible plan.
        aggressive_repair_reasons = {
            "no_executable_route_stop",
            "route_queue_starvation",
        }
        initial_lifeline_ordering = bool(
            str(reason) == "initial_plan"
            and self._targeted_repairs_enabled()
            and (
                bool(
                    getattr(
                        env.cfg,
                        "hrl_route_plan_force_initial_lifeline_ordering_enabled",
                        False,
                    )
                )
                or self._initial_lifeline_ordering_is_spatially_safe(
                    env, truck_ids
                )
            )
        )
        if str(reason) == "initial_plan":
            self._global_lifeline_ordering_allowed = bool(
                initial_lifeline_ordering
            )
            self.initial_lifeline_ordering_enabled = bool(
                initial_lifeline_ordering
            )
        self._aggressive_planning_active = bool(
            self._targeted_repairs_enabled()
            and (
                self._global_lifeline_ordering_allowed
                or str(reason) in aggressive_repair_reasons
            )
        )
        routes, contracts, objective = self._optimize(
            env,
            truck_ids,
            clusters,
            road_signature,
            prior_contracts=prior_contracts,
        )
        # Contract versions change only when executable ownership changes,
        # not on every harmless road/ETA plan publication. This lets the
        # execution layer distinguish an explicit release from route churn.
        for task_id, contract in contracts.items():
            prior = prior_contracts.get(str(task_id), None)
            same_execution = bool(
                prior is not None
                and str(prior.owner_agent_id) == str(contract.owner_agent_id)
                and str(prior.truck_id) == str(contract.truck_id)
                and str(prior.uav_id or "") == str(contract.uav_id or "")
                and tuple(prior.uav_ids) == tuple(contract.uav_ids)
                and str(prior.service_mode) == str(contract.service_mode)
                and str(prior.recovery_truck_id or "")
                == str(contract.recovery_truck_id or "")
            )
            if same_execution:
                contract.version = int(max(getattr(prior, "version", 0), 1))
            else:
                task = env.state.tasks.get(str(task_id), None)
                contract.version = int(
                    max(
                        getattr(prior, "version", 0)
                        if prior is not None
                        else 0,
                        getattr(task, "route_contract_version", 0)
                        if task is not None
                        else 0,
                    )
                    + 1
                )
        self.plan_version_count += 1
        self.plan = PlanVersion(
            version=int(self.plan_version_count),
            created_step=int(env.state.step_index),
            road_signature=road_signature,
            routes=routes,
            contracts=contracts,
            objective=float(objective),
            reason=str(reason),
        )
        # A failed global insertion must never leave a live fleet with no
        # executable task.  Keep the ALNS route as the preferred solution,
        # then greedily add only the uncovered active tasks as independent
        # suffixes.  This is the layer-1 recovery path, not a return to the
        # legacy stepwise dispatcher.
        self._append_uncovered_fallback_contracts(env, road_signature)
        self.last_replan_step = int(env.state.step_index)
        if reason == "initial_plan":
            self.full_plan_count += 1
        else:
            self.suffix_repair_count += 1
            if np.isfinite(objective):
                self.suffix_repair_success_count += 1
        for task_id, contract in contracts.items():
            task = env.state.tasks.get(str(task_id), None)
            if task is None:
                continue
            task.route_contract_owner = str(contract.owner_agent_id)
            task.route_contract_truck = str(contract.truck_id)
            task.route_contract_uav_ids = tuple(str(uid) for uid in contract.uav_ids)
        # Atomic publication pass also covers contracts appended by the
        # feasibility fallback above. The older field assignments are kept
        # for backward compatibility; this pass adds a single version shared
        # by layer 1 and the execution layer.
        for task_id, contract in self.plan.contracts.items():
            self._stamp_contract_on_task(
                env, str(task_id), contract, bump=False
            )

    def _append_uncovered_fallback_contracts(
        self,
        env,
        road_signature: Tuple[Tuple[int, int], ...],
    ) -> None:
        """Guarantee one executable layer-1 contract for every feasible task.

        ALNS is allowed to reject an infeasible insertion, but one rejected
        task must not collapse the rest of the plan into all-truck stay.
        """
        if self.plan is None:
            return
        routes = self.plan.routes
        covered = set(str(tid) for tid in self.plan.contracts)
        pending = sorted(
            (task for task in env.state.tasks.values() if self._active(task)),
            key=self._task_order_key,
        )
        for task in pending:
            task_id = str(task.task_id)
            if task_id in covered:
                continue
            best = None
            for truck_id, route in sorted(routes.items()):
                forbidden_truck = self._temporary_forbidden_truck_by_task.get(
                    str(task_id), None
                )
                if forbidden_truck is not None and str(truck_id) == str(forbidden_truck):
                    continue
                uavs = tuple(route.uav_ids)
                relay = self._is_relay(task)
                air_service = bool(task.kind == TaskKind.EMERGENCY or relay)
                if air_service and not uavs:
                    continue
                if task.kind == TaskKind.EMERGENCY:
                    emergency_capacity = self._cluster_emergency_capacity_units(
                        env, str(truck_id), uavs
                    )
                    emergency_reserved = sum(
                        1
                        for existing_stop in route.stops[int(route.cursor) :]
                        if (
                            env.state.tasks.get(str(existing_stop.task_id), None)
                            is not None
                            and self._active(
                                env.state.tasks[str(existing_stop.task_id)]
                            )
                            and env.state.tasks[str(existing_stop.task_id)].kind
                            == TaskKind.EMERGENCY
                        )
                    )
                    if emergency_reserved >= emergency_capacity:
                        continue
                enforce_fallback_inventory_budget = bool(
                    self._aggressive_planning_active
                )
                if not air_service and enforce_fallback_inventory_budget:
                    truck_state = env.state.agents.get(str(truck_id), None)
                    available_kg = float(
                        max(
                            getattr(
                                truck_state,
                                "bulk_inventory_kg_current",
                                0.0,
                            ),
                            0.0,
                        )
                    )
                    reserved_kg = 0.0
                    for existing_stop in route.stops[int(route.cursor) :]:
                        existing_task = env.state.tasks.get(
                            str(existing_stop.task_id), None
                        )
                        if (
                            existing_task is not None
                            and self._active(existing_task)
                            and existing_task.kind == TaskKind.NORMAL
                            and not self._is_relay(existing_task)
                        ):
                            reserved_kg += self._remaining_demand_kg(existing_task)
                    if (
                        self._remaining_demand_kg(task)
                        > max(available_kg - reserved_kg, 0.0) + 1e-9
                    ):
                        continue
                if air_service:
                    chosen = (str(uavs[0]),)
                    if relay:
                        count = int(max(getattr(env.cfg, "hrl_route_plan_bulk_relay_uav_count", 2), 1))
                        chosen = tuple(str(uid) for uid in uavs[: min(count, len(uavs))])
                    contract = TaskContract(
                        task_id=task_id,
                        owner_agent_id=str(chosen[0]),
                        truck_id=str(truck_id),
                        uav_id=str(chosen[0]),
                        uav_ids=tuple(chosen),
                        service_mode=BULK_RELAY if relay else DIRECT,
                        created_step=int(env.state.step_index),
                    )
                else:
                    contract = TaskContract(
                        task_id=task_id,
                        owner_agent_id=str(truck_id),
                        truck_id=str(truck_id),
                        uav_id=None,
                        uav_ids=(),
                        service_mode=DIRECT,
                        created_step=int(env.state.step_index),
                    )
                stops, cost = self._route_stop_specs(
                    env,
                    str(truck_id),
                    [task_id],
                    {str(truck_id): uavs},
                    {task_id: contract},
                    road_signature,
                )
                if not stops or not np.isfinite(cost):
                    continue
                candidate = (float(cost), str(truck_id), contract, stops[0])
                if best is None or candidate[:2] < best[:2]:
                    best = candidate
            if best is None:
                self._feedback.append(
                    PlannerFeedback(
                        step=int(env.state.step_index),
                        reason="fallback_contract_infeasible",
                        task_id=task_id,
                        detail=str(getattr(task, "service_mode", DIRECT)),
                    )
                )
                continue
            _, truck_id, contract, stop = best
            route = routes[truck_id]
            # Emergency work is placed before newly appended routine work;
            # no active stop is displaced because this task was uncovered.
            if task.kind == TaskKind.EMERGENCY:
                route.stops.insert(int(route.cursor), stop)
            else:
                route.stops.append(stop)
            self.plan.contracts[task_id] = contract
            self._stamp_contract_on_task(
                env, task_id, contract, bump=False
            )
            covered.add(task_id)

    def _publish(self, env, goals: Dict[str, Optional[str]]) -> None:
        env._planner_route_plan_v2 = self.audit_snapshot(env)
        env._planner_route_plan_feedback = [
            {
                "step": int(item.step),
                "reason": str(item.reason),
                "truck_id": str(item.truck_id),
                "task_id": str(item.task_id),
                "detail": str(item.detail),
                "suffix_repair_required": bool(item.suffix_repair_required),
            }
            for item in self._feedback[-100:]
        ]
        env._planner_route_plan_stay_reason_by_agent = dict(
            self._stay_reason_by_agent
        )
        env._planner_truck_assist_waypoint_by_truck = dict(self._assist_by_truck)
        env._planner_route_plan_goals = dict(goals)
        env._planner_force_takeoff_task_by_uav = dict(
            self._force_takeoff_task_by_uav
        )
        env._uav_transfer_target_truck = {
            str(uid): str(info["recovery_truck_id"])
            for uid, info in self._transfer_by_uav.items()
        }
        env._uav_transfer_target_task = {
            str(uid): str(info["task_id"])
            for uid, info in self._transfer_by_uav.items()
        }

    def _install_safety_recovery_assists(
        self,
        env,
        goals: Dict[str, Optional[str]],
    ) -> None:
        """Give an immobilized low-battery UAV an executable truck pickup leg.

        The UAV recovery controller already points the aircraft at a truck.
        Under the persistent route-plan backend, however, the selected truck
        previously kept its completed route and stayed forever.  This method
        installs the missing layer-1 truck leg to the closest road-reachable
        boundary node.  One pickup per truck is active at a time.
        """
        forced = dict(getattr(env, "_uav_forced_rth_latch", {}))
        pending_routine = any(
            task.kind == TaskKind.NORMAL
            and self._active(task)
            and not self._is_relay(task)
            for task in env.state.tasks.values()
        )
        sortie_contracts = dict(
            getattr(env, "_uav_sortie_contract_task", {})
        )
        candidates = []
        for uav_id, active in forced.items():
            state = env.state.agents.get(str(uav_id), None)
            if (
                not bool(active)
                or state is None
                or state.kind != AgentKind.UAV
                or state.follow_target is not None
                or bool(getattr(state, "crashed", False))
                or state.pos_xy is None
            ):
                continue
            sortie_task_id = sortie_contracts.get(str(uav_id), None)
            sortie_task = (
                env.state.tasks.get(str(sortie_task_id), None)
                if sortie_task_id is not None
                else None
            )
            carries_emergency_package = bool(
                int(max(getattr(state, "carried_emergency_units", 0), 0)) > 0
            )
            active_delivery_contract = bool(
                sortie_task is not None
                and sortie_task.kind == TaskKind.EMERGENCY
                and self._active(sortie_task)
            )
            # Do not pre-empt a truck-only routine route merely to pick up an
            # empty UAV that can safely wait under the terminal guard. Loaded
            # or still-contracted UAVs retain hard recovery priority.
            if (
                pending_routine
                and not carries_emergency_package
                and not active_delivery_contract
            ):
                continue
            candidates.append(
                (float(getattr(state, "battery", 0.0)), str(uav_id), state)
            )
        used_trucks: set[str] = set()
        follower_cap = int(max(getattr(env.cfg, "uav_max_followers_per_truck", 1), 1))
        for _, uav_id, uav_state in sorted(candidates):
            ux, uy = float(uav_state.pos_xy[0]), float(uav_state.pos_xy[1])
            best = None
            for truck_id in self._live_trucks(env):
                if str(truck_id) in used_trucks:
                    continue
                if int(env._truck_follower_count(str(truck_id))) >= follower_cap:
                    continue
                truck = env.state.agents.get(str(truck_id), None)
                if truck is None or truck.node is None:
                    continue
                # Do not pull a truck away at the exact instant it can unload
                # a contracted DIRECT routine task.  The unload commitment is
                # irreversible and short; recovery can use another truck or
                # be reconsidered on the next planning step.  Without this
                # guard, a late safety-recovery assist can overwrite the
                # onsite goal after the atomic takeover and strand an
                # otherwise completed zero-distance delivery indefinitely.
                onsite_routine_commitment = any(
                    task.kind == TaskKind.NORMAL
                    and self._active(task)
                    and not self._is_relay(task)
                    and int(task.demand_node) == int(truck.node)
                    and str(getattr(task, "route_contract_truck", "") or "")
                    == str(truck_id)
                    and bool(env.is_task_serviceable_by_agent(str(truck_id), task))
                    for task in env.state.tasks.values()
                )
                if onsite_routine_commitment:
                    continue
                boundary = None
                for node_id in env.topology.nodes:
                    road = float(
                        env._decision_shortest_path_distance(
                            int(truck.node), int(node_id)
                        )
                    )
                    if not np.isfinite(road):
                        continue
                    nx, ny = self._xy(env, int(node_id))
                    air = float(np.hypot(nx - ux, ny - uy))
                    option = (air, road, int(node_id))
                    if boundary is None or option < boundary:
                        boundary = option
                if boundary is None:
                    continue
                option = (
                    float(boundary[0]),
                    float(boundary[1]),
                    str(truck_id),
                    int(boundary[2]),
                )
                if best is None or option < best:
                    best = option
            if best is None:
                continue
            air, road, truck_id, node_id = best
            used_trucks.add(str(truck_id))
            goals[str(truck_id)] = None
            goals[str(uav_id)] = str(truck_id)
            self._assist_by_truck[str(truck_id)] = {
                "assist_waypoint_insert": True,
                "route_plan_v2": True,
                "hold_at_anchor": True,
                "idle_support": True,
                "uav_id": str(uav_id),
                "uav_ids": [str(uav_id)],
                "task_id": "",
                "launch_node": int(node_id),
                "normal_goal_task_id": "",
                "step": int(env.state.step_index),
                "service_mode": "SAFETY_RECOVERY",
                "plan_version": int(self.plan.version) if self.plan is not None else 0,
                "planned_road_distance_m": float(road),
                "planned_air_distance_m": float(air),
            }
            self._stay_reason_by_agent[str(truck_id)] = (
                "move_to_low_battery_uav_recovery_boundary"
            )

    def _promote_at_risk_emergency_over_routine(self, env, route: ClusterRoute) -> None:
        """Conditionally promote a risky emergency after a road-impact shock.

        The V7 candidate starts from the balanced/no-UAV-priority route.  This
        execution repair is deliberately narrower than a generic emergency
        watchdog: it requires (i) a current NORMAL head, (ii) a material
        increase in the head's recomputed road ETA over its planned remaining
        ETA, and (iii) a pending emergency suffix whose lifeline/deadline
        reserve is already insufficient for its planned wait.  Claimed,
        started, airborne, and nearby NORMAL work is never pre-empted.
        """
        self.road_impact_emergency_promotion_candidate_count += 1
        if not bool(
            getattr(
                env.cfg,
                "hrl_route_plan_conditional_road_emergency_promotion_enabled",
                False,
            )
        ):
            return
        if self.plan is None or not self._road_signature(env):
            return
        current = route.current(env)
        if current is None:
            return
        current_task = env.state.tasks.get(str(current.task_id), None)
        if current_task is None or current_task.kind != TaskKind.NORMAL:
            return

        # The route head is an execution commitment once service has started;
        # do not move it merely because a different suffix is urgent.
        if (
            current_task.status != TaskStatus.PENDING
            or getattr(current_task, "assigned_to", None) is not None
            or getattr(current_task, "in_service_by", None) is not None
            or getattr(current_task, "first_service_step", None) is not None
            or int(max(getattr(current_task, "service_remaining", 0), 0)) > 0
            or float(max(getattr(current_task, "fulfilled_mass_kg", 0.0), 0.0)) > 1e-9
            or tuple(getattr(current_task, "route_contract_uav_ids", ()) or ())
        ):
            self.road_impact_emergency_promotion_reject_count += 1
            self.road_impact_emergency_promotion_reject_protected_count += 1
            return

        truck = env.state.agents.get(str(route.truck_id), None)
        if truck is None or truck.node is None:
            return
        step_now = int(env.state.step_index)
        route_key = f"{route.truck_id}:{current.task_id}"
        cooldown = int(
            max(
                getattr(
                    env.cfg,
                    "hrl_route_plan_conditional_road_emergency_promotion_cooldown_steps",
                    24,
                ),
                0,
            )
        )
        last_route = int(
            self._road_impact_emergency_promotion_last_step_by_route.get(
                route_key, -10**9
            )
        )
        if step_now - last_route < cooldown:
            self.road_impact_emergency_promotion_reject_count += 1
            self.road_impact_emergency_promotion_reject_cooldown_count += 1
            return

        current_distance = float(
            env._decision_shortest_path_distance(
                int(truck.node), int(current_task.demand_node)
            )
        )
        near_distance = float(
            max(
                getattr(
                    env.cfg,
                    "hrl_route_plan_conditional_road_emergency_promotion_near_normal_distance_m",
                    1000.0,
                ),
                0.0,
            )
        )
        if np.isfinite(current_distance) and current_distance <= near_distance + 1e-9:
            self.road_impact_emergency_promotion_reject_count += 1
            self.road_impact_emergency_promotion_reject_protected_count += 1
            return

        speed = float(max(getattr(env.cfg, "truck_speed_mps", 10.0), 1e-6))
        dt = float(max(getattr(env.cfg, "dt_seconds", 1.0), 1e-6))
        unload = int(max(getattr(env.cfg, "unload_rounds_normal", 1), 1))
        planned_distance = float(getattr(current, "planned_road_distance_m", float("inf")))
        planned_remaining = float(max(int(getattr(current, "eta_step", step_now)) - step_now, 0))
        if not np.isfinite(planned_remaining) or planned_remaining <= 0.0:
            if np.isfinite(planned_distance):
                planned_remaining = float(
                    np.ceil(planned_distance / max(speed * dt, 1e-6)) + unload
                )
            else:
                planned_remaining = float("inf")
        current_eta = float("inf")
        if np.isfinite(current_distance):
            current_eta = float(
                np.ceil(current_distance / max(speed * dt, 1e-6)) + unload
            )
        eta_increase = float(current_eta - planned_remaining)
        threshold = float(
            max(
                getattr(
                    env.cfg,
                    "hrl_route_plan_conditional_road_emergency_promotion_eta_increase_steps",
                    12,
                ),
                0.0,
            )
        )
        if not np.isfinite(eta_increase) or eta_increase <= threshold + 1e-9:
            self.road_impact_emergency_promotion_reject_count += 1
            self.road_impact_emergency_promotion_reject_no_delta_count += 1
            return

        reserve = int(
            max(
                getattr(
                    env.cfg,
                    "hrl_route_plan_conditional_road_emergency_promotion_reserve_steps",
                    8,
                ),
                0,
            )
        )
        min_gain = float(
            max(
                getattr(
                    env.cfg,
                    "hrl_route_plan_conditional_road_emergency_promotion_min_gain_steps",
                    6,
                ),
                0.0,
            )
        )
        airborne_tasks = {
            str(task_id)
            for uid, task_id in dict(
                getattr(env, "_uav_sortie_contract_task", {})
            ).items()
            if task_id is not None
            and env.state.agents.get(str(uid), None) is not None
            and getattr(env.state.agents[str(uid)], "follow_target", None) is None
        }
        best = None
        for index in range(int(route.cursor) + 1, len(route.stops)):
            stop = route.stops[index]
            task = env.state.tasks.get(str(stop.task_id), None)
            if (
                task is None
                or task.kind != TaskKind.EMERGENCY
                or not self._active(task)
                or task.status != TaskStatus.PENDING
                or getattr(task, "assigned_to", None) is not None
                or getattr(task, "in_service_by", None) is not None
                or getattr(task, "first_service_step", None) is not None
                or int(max(getattr(task, "service_remaining", 0), 0)) > 0
                or str(task.task_id) in airborne_tasks
            ):
                continue
            decay = float(max(getattr(task, "lifeline_decay_rate", 0.0), 0.0))
            life_steps = (
                float("inf")
                if decay <= 1e-9
                else float(max(getattr(task, "lifeline_current", 0.0), 0.0) / decay)
            )
            deadline_steps = float(
                max(self._effective_deadline_step(env, task) - step_now, 0)
            )
            budget = float(min(life_steps, deadline_steps))
            planned_wait = float(max(int(getattr(stop, "eta_step", step_now)) - step_now, 0))
            if budget > planned_wait + reserve + 1e-9:
                continue

            # A risk signal alone is not enough to justify changing the
            # truck's current commitment.  Recompute the emergency's actual
            # anchor ETA on the *current* blocked graph and require both:
            # (a) completion remains inside the residual reserve, and
            # (b) moving the stop forward saves a material number of steps.
            # This is deliberately conservative: an emergency that is only
            # theoretically urgent but cannot be reached after the shock is
            # left to the ordinary UAV/recovery logic.
            promoted_distance = float(
                env._decision_shortest_path_distance(
                    int(truck.node), int(getattr(stop, "target_node", task.demand_node))
                )
            )
            if not np.isfinite(promoted_distance):
                self.road_impact_emergency_promotion_reject_count += 1
                self.road_impact_emergency_promotion_reject_no_risk_count += 1
                continue
            promoted_eta = float(
                np.ceil(promoted_distance / max(speed * dt, 1e-6)) + 1
            )
            emergency_gain = float(planned_wait - promoted_eta)
            if promoted_eta + reserve > budget + 1e-9 or emergency_gain < min_gain - 1e-9:
                self.road_impact_emergency_promotion_reject_count += 1
                self.road_impact_emergency_promotion_reject_no_delta_count += 1
                continue
            if shadow_enabled:
                # Shadow-evaluate the two local route choices.  The normal
                # stop is the current route head; after promotion it must be
                # reached from the emergency anchor, including service time.
                # A normal stop that was already infeasible may remain so, but
                # an otherwise feasible normal stop may never be pushed past
                # its deadline by this repair.
                emergency_service = int(
                    max(
                        getattr(task, "service_remaining", 0),
                        getattr(
                            env.cfg,
                            "uav_service_time_steps",
                            getattr(env.cfg, "service_time_steps", 1),
                        ),
                        1,
                    )
                )
                normal_service = int(
                    max(
                        getattr(current_task, "service_remaining", 0),
                        getattr(env.cfg, "service_time_steps", 1),
                        1,
                    )
                )
                anchor_to_normal = float(
                    env._decision_shortest_path_distance(
                        int(getattr(stop, "target_node", task.demand_node)),
                        int(current_task.demand_node),
                    )
                )
                if np.isfinite(anchor_to_normal):
                    normal_after_eta = float(
                        promoted_eta
                        + emergency_service
                        + np.ceil(anchor_to_normal / max(speed * dt, 1e-6))
                        + normal_service
                    )
                else:
                    normal_after_eta = float("inf")
                normal_deadline = float(
                    max(
                        self._effective_deadline_step(env, current_task)
                        - step_now,
                        0,
                    )
                )
                keep_normal_feasible = bool(
                    planned_remaining + shadow_normal_tolerance
                    <= normal_deadline + 1e-9
                )
                promote_normal_feasible = bool(
                    normal_after_eta + shadow_normal_tolerance
                    <= normal_deadline + 1e-9
                )
                emergency_promote_feasible = bool(
                    promoted_eta + reserve <= budget + 1e-9
                )
                if (
                    emergency_gain < shadow_min_gain - 1e-9
                    or not emergency_promote_feasible
                    or (keep_normal_feasible and not promote_normal_feasible)
                ):
                    self.road_impact_emergency_promotion_reject_count += 1
                    self.road_impact_emergency_promotion_reject_no_delta_count += 1
                    continue
            last_task = int(
                self._road_impact_emergency_promotion_last_step_by_task.get(
                    str(task.task_id), -10**9
                )
            )
            if step_now - last_task < cooldown:
                continue
            option = (budget - planned_wait, -emergency_gain, budget, str(task.task_id), index)
            if best is None or option < best:
                best = option
        if best is None:
            self.road_impact_emergency_promotion_reject_count += 1
            self.road_impact_emergency_promotion_reject_no_risk_count += 1
            return

        self.road_impact_emergency_promotion_trigger_count += 1
        _, _, budget, task_id, index = best
        promoted = route.stops.pop(int(index))
        route.stops.insert(int(route.cursor), promoted)
        self._road_impact_emergency_promotion_last_step_by_route[route_key] = step_now
        self._road_impact_emergency_promotion_last_step_by_task[str(task_id)] = step_now
        self.road_impact_emergency_promotion_count += 1
        self.deadline_rescue_promotion_count += 1
        self._feedback.append(
            PlannerFeedback(
                step=step_now,
                reason="conditional_road_impact_emergency_promotion",
                truck_id=str(route.truck_id),
                task_id=str(task_id),
                detail=(
                    f"displaced_routine={current.task_id},road_m={current_distance:.1f},"
                    f"planned_m={planned_distance:.1f},eta_increase={eta_increase:.1f},"
                    f"emergency_budget={budget:.1f}"
                ),
            )
        )

    def _transfer_routine_contract_to_truck(
        self,
        env,
        task_id: str,
        new_owner: str,
        goals: Dict[str, Optional[str]],
        *,
        eta_step: Optional[int] = None,
        planned_road_distance_m: Optional[float] = None,
    ) -> bool:
        """Move one pending routine contract and its route stop atomically."""
        if self.plan is None:
            return False
        task = env.state.tasks.get(str(task_id), None)
        new_route = self.plan.routes.get(str(new_owner), None)
        if (
            task is None
            or task.kind != TaskKind.NORMAL
            or not self._active(task)
            or self._is_relay(task)
            or new_route is None
        ):
            return False

        state = env.state.agents.get(str(new_owner), None)
        if state is None or state.node is None:
            return False
        road = (
            float(planned_road_distance_m)
            if planned_road_distance_m is not None
            else float(
                env._decision_shortest_path_distance(
                    int(state.node), int(task.demand_node)
                )
            )
        )
        if not np.isfinite(road):
            return False
        moved_stop: Optional[RouteStop] = None
        for route in self.plan.routes.values():
            for index in range(len(route.stops) - 1, int(route.cursor) - 1, -1):
                if str(route.stops[index].task_id) == str(task_id):
                    candidate = route.stops.pop(index)
                    if moved_stop is None:
                        moved_stop = candidate
        if eta_step is None:
            speed = float(max(getattr(env.cfg, "truck_speed_mps", 10.0), 1e-6))
            dt = float(max(getattr(env.cfg, "dt_seconds", 1.0), 1e-6))
            unload = int(max(getattr(env.cfg, "unload_rounds_normal", 1), 1))
            eta_step = int(
                env.state.step_index
                + np.ceil(road / max(speed * dt, 1e-6))
                + unload
            )
        if moved_stop is None:
            moved_stop = RouteStop(
                task_id=str(task_id),
                stop_type=NORMAL_SERVICE,
                truck_id=str(new_owner),
                uav_id=None,
                uav_ids=(),
                target_node=int(task.demand_node),
                deadline_step=int(task.deadline_step),
                service_mode=DIRECT,
            )
        moved_stop.truck_id = str(new_owner)
        moved_stop.uav_id = None
        moved_stop.uav_ids = ()
        moved_stop.target_node = int(task.demand_node)
        moved_stop.planned_road_distance_m = float(road)
        moved_stop.eta_step = int(eta_step)
        moved_stop.service_mode = DIRECT
        if (
            er_hlns_balanced_all_tasks_v3_active(env)
            or bool(
                getattr(
                    env.cfg,
                    "hrl_route_plan_hard_normal_rescue_enabled",
                    False,
                )
            )
        ) and bool(
            getattr(
                env.cfg,
                "hrl_route_plan_balanced_all_tasks_v3_tail_insert_after_launch",
                False,
            )
            or bool(
                getattr(
                    env.cfg,
                    "hrl_route_plan_hard_normal_rescue_enabled",
                    False,
                )
            )
        ):
            # The airborne UAV keeps the emergency contract active while the
            # support truck executes this freshly auctioned NORMAL target;
            # append the stop so the existing emergency route prefix remains
            # immutable for the next replan.
            new_route.stops.append(moved_stop)
        else:
            new_route.stops.insert(int(new_route.cursor), moved_stop)

        contract = self.plan.contracts.get(str(task_id), None)
        if contract is None:
            contract = TaskContract(
                task_id=str(task_id),
                owner_agent_id=str(new_owner),
                truck_id=str(new_owner),
                uav_id=None,
                uav_ids=(),
                service_mode=DIRECT,
                created_step=int(env.state.step_index),
            )
            self.plan.contracts[str(task_id)] = contract
            if bool(
                getattr(
                    env.cfg,
                    "hrl_route_plan_hard_normal_rescue_enabled",
                    False,
                )
            ):
                # Keep a candidate rescue binding stable across the next
                # event refresh; otherwise the generic constructor can erase
                # the newly executable suffix before the truck reaches it.
                contract.locked = True
        else:
            contract.owner_agent_id = str(new_owner)
            contract.truck_id = str(new_owner)
            contract.uav_id = None
            contract.uav_ids = ()
            contract.service_mode = DIRECT
            contract.created_step = int(env.state.step_index)
            contract.locked = True
        self._stamp_contract_on_task(env, str(task_id), contract, bump=True)
        task.route_contract_owner = str(new_owner)
        task.route_contract_truck = str(new_owner)
        task.route_contract_uav_ids = ()
        for agent_id, goal_id in list(goals.items()):
            if str(agent_id) != str(new_owner) and str(goal_id) == str(task_id):
                goals[str(agent_id)] = None
        goals[str(new_owner)] = str(task_id)
        self._assist_by_truck.pop(str(new_owner), None)
        self._normal_cleanup_owner_by_task[str(task_id)] = str(new_owner)
        return True

    def _apply_controlled_routine_opportunity_transfers(
        self,
        env,
        goals: Dict[str, Optional[str]],
    ) -> None:
        """Locally reauction a nearby routine task without destabilizing routes.

        Layer-1 contracts are plans, not ownership barriers.  A different truck
        may take a routine task when its *road ETA* is materially better, but only
        if it is stocked, has no UAV assist/recovery commitment, and the task has
        not already bounced between trucks.  Exact-node opportunities remain in
        the existing onsite takeover below; this routine handles 0--R metres.
        """
        if self.plan is None or not bool(
            getattr(env.cfg, "hrl_route_plan_routine_dynamic_reassignment_enabled", True)
        ):
            return
        balanced_v2 = bool(er_hlns_balanced_all_tasks_v2_active(env))
        hard_coverage_airborne_parallel = bool(
            getattr(
                env.cfg,
                "hrl_route_plan_hard_normal_rescue_airborne_parallel_enabled",
                False,
            )
            and er_hlns_parallel_routine_emergency_active(env)
        )
        # The hard-coverage candidate reuses V2's airborne-only safety gates
        # without enabling the rest of the V2 auction policy.
        balanced_v2 = bool(balanced_v2 or hard_coverage_airborne_parallel)
        aggressive_pending_auction = bool(
            balanced_v2
            and getattr(
                env.cfg,
                "hrl_route_plan_balanced_all_tasks_v2_aggressive_pending_auction_enabled",
                False,
            )
        )
        radius_m = float(
            max(
                getattr(
                    env.cfg,
                    "hrl_route_plan_routine_dynamic_reassignment_radius_m",
                    800.0,
                ),
                0.0,
            )
        )
        min_gain_steps = float(
            max(
                getattr(
                    env.cfg,
                    "hrl_route_plan_routine_dynamic_reassignment_min_eta_gain_steps",
                    3.0,
                ),
                0.0,
            )
        )
        min_gain_ratio = float(
            np.clip(
                getattr(
                    env.cfg,
                    "hrl_route_plan_routine_dynamic_reassignment_min_eta_gain_ratio",
                    0.20,
                ),
                0.0,
                1.0,
            )
        )
        if aggressive_pending_auction:
            radius_m = float("inf")
            min_gain_steps = 0.0
            min_gain_ratio = 0.0
        max_transfers = int(
            max(
                getattr(
                    env.cfg,
                    "hrl_route_plan_routine_dynamic_reassignment_max_transfers",
                    1,
                ),
                0,
            )
        )
        cooldown_steps = int(
            max(
                getattr(
                    env.cfg,
                    "hrl_route_plan_routine_dynamic_reassignment_lock_steps",
                    5,
                ),
                0,
            )
        )
        speed = float(max(getattr(env.cfg, "truck_speed_mps", 10.0), 1e-6))
        dt = float(max(getattr(env.cfg, "dt_seconds", 1.0), 1e-6))
        unload = int(max(getattr(env.cfg, "unload_rounds_normal", 1), 1))
        step_now = int(env.state.step_index)
        busy = set(env._servicing_agents())
        emergency_truck_ids = {
            str(contract.truck_id)
            for task_id, contract in self.plan.contracts.items()
            if (
                env.state.tasks.get(str(task_id), None) is not None
                and env.state.tasks[str(task_id)].kind == TaskKind.EMERGENCY
                and self._active(env.state.tasks[str(task_id)])
            )
        }
        active_sortie_tasks = {
            str(task_id)
            for task_id in dict(
                getattr(env, "_uav_sortie_contract_task", {})
            ).values()
            if task_id is not None
        }

        def road_steps(source: Optional[int], target: int) -> Tuple[float, float]:
            if source is None:
                return float("inf"), float("inf")
            distance = float(
                env._decision_shortest_path_distance(int(source), int(target))
            )
            if not np.isfinite(distance):
                return float("inf"), float("inf")
            return distance, float(np.ceil((distance / speed) / dt))

        for task in sorted(env.state.tasks.values(), key=lambda item: str(item.task_id)):
            task_id = str(task.task_id)
            if (
                task.kind != TaskKind.NORMAL
                or not self._active(task)
                or self._is_relay(task)
                or int(self._routine_opportunity_transfer_count_by_task.get(task_id, 0))
                >= max_transfers
            ):
                continue
            if balanced_v2:
                # V2 re-auctions only genuinely pending, never-started
                # routine work. Claimed/service-progress tasks are atomic and
                # cannot be moved by this candidate repair.
                if (
                    task.status != TaskStatus.PENDING
                    or getattr(task, "first_service_step", None) is not None
                    or getattr(task, "assigned_to", None) is not None
                    or getattr(task, "in_service_by", None) is not None
                    or float(max(getattr(task, "fulfilled_mass_kg", 0.0), 0.0)) > 1e-9
                    or tuple(getattr(task, "route_contract_uav_ids", ()) or ())
                    or task_id in active_sortie_tasks
                ):
                    continue
                self.balanced_all_tasks_v2_reauction_candidate_count += 1
                if aggressive_pending_auction:
                    self.balanced_all_tasks_v2_aggressive_auction_candidate_count += 1
            contract = self.plan.contracts.get(task_id, None)
            owner_id = str(
                getattr(task, "route_contract_truck", "")
                or (getattr(contract, "truck_id", "") if contract is not None else "")
                or ""
            )
            owner = env.state.agents.get(owner_id, None)
            owner_source = None if owner is None else owner.node
            if owner is not None and owner.transit is not None:
                owner_source = int(owner.transit[1])
            _, owner_travel = road_steps(owner_source, int(task.demand_node))
            owner_eta = float(owner_travel + unload)

            owner_route = self.plan.routes.get(owner_id, None)
            owner_stop = owner_route.current(env) if owner_route is not None else None
            if owner_stop is not None and str(owner_stop.task_id) != task_id:
                first_dist, first_steps = road_steps(
                    owner_source, int(owner_stop.target_node)
                )
                second_dist, second_steps = road_steps(
                    int(owner_stop.target_node), int(task.demand_node)
                )
                if np.isfinite(first_dist) and np.isfinite(second_dist):
                    first_task = env.state.tasks.get(str(owner_stop.task_id), None)
                    first_unload = int(
                        max(
                            getattr(
                                env.cfg,
                                "unload_rounds_uav"
                                if first_task is not None
                                and first_task.kind == TaskKind.EMERGENCY
                                else "unload_rounds_normal",
                                1,
                            ),
                            1,
                        )
                    )
                    owner_eta = float(first_steps + first_unload + second_steps + unload)

            best: Optional[Tuple[float, float, str]] = None
            for truck_id in self._live_trucks(env):
                truck_id = str(truck_id)
                if truck_id == owner_id or truck_id in busy:
                    continue
                if aggressive_pending_auction and truck_id in emergency_truck_ids:
                    continue
                truck = env.state.agents.get(truck_id, None)
                if truck is None or truck.node is None or truck.transit is not None:
                    continue
                if float(max(getattr(truck, "bulk_inventory_kg_current", 0.0), 0.0)) + 1e-9 < self._remaining_demand_kg(task):
                    continue
                distance, travel = road_steps(int(truck.node), int(task.demand_node))
                if not np.isfinite(distance) or distance <= 1e-9 or distance > radius_m + 1e-9:
                    continue
                self.routine_opportunity_candidate_count += 1
                if truck_id in self._assist_by_truck or bool(
                    env._truck_has_assigned_airborne_hard_recovery_request(truck_id)
                ):
                    self.routine_opportunity_blocked_assist_count += 1
                    continue

                # A routine-to-routine switch is allowed only when the nearby
                # opportunity is also materially closer than this truck's own
                # current routine head.  This preserves the original route line
                # unless the local move is a clear dominance improvement.
                candidate_route = self.plan.routes.get(truck_id, None)
                candidate_stop = (
                    candidate_route.current(env) if candidate_route is not None else None
                )
                if candidate_stop is not None and str(candidate_stop.task_id) != task_id:
                    current_task = env.state.tasks.get(str(candidate_stop.task_id), None)
                    if current_task is not None and current_task.kind == TaskKind.EMERGENCY:
                        self.routine_opportunity_blocked_assist_count += 1
                        continue
                    if balanced_v2 and (
                        current_task is None
                        or current_task.kind != TaskKind.NORMAL
                        or current_task.status != TaskStatus.PENDING
                        or getattr(current_task, "first_service_step", None) is not None
                        or getattr(current_task, "assigned_to", None) is not None
                        or getattr(current_task, "in_service_by", None) is not None
                    ):
                        continue
                    current_distance, current_steps = road_steps(
                        int(truck.node), int(candidate_stop.target_node)
                    )
                    if (
                        not np.isfinite(current_distance)
                        or float(current_steps - travel) < min_gain_steps - 1e-9
                    ):
                        self.routine_opportunity_blocked_eta_count += 1
                        continue

                candidate_eta = float(travel + unload)
                if balanced_v2 and bool(
                    getattr(
                        env.cfg,
                        "hrl_route_plan_balanced_all_tasks_v2_reauction_deadline_guard_enabled",
                        True,
                    )
                ):
                    candidate_deadline = float(
                        self._effective_deadline_step(env, task)
                    )
                    if float(step_now) + candidate_eta > candidate_deadline + 1e-9:
                        self.balanced_all_tasks_v2_reauction_deadline_block_count += 1
                        continue
                required_gain = float(max(min_gain_steps, min_gain_ratio * owner_eta))
                if not np.isfinite(owner_eta) or owner_eta - candidate_eta >= required_gain - 1e-9:
                    option = (candidate_eta, distance, truck_id)
                    if best is None or option < best:
                        best = option
                else:
                    self.routine_opportunity_blocked_eta_count += 1

            if best is None:
                continue
            candidate_eta, distance, new_owner = best

            if self._transfer_routine_contract_to_truck(
                env,
                task_id,
                str(new_owner),
                goals,
                eta_step=int(step_now + candidate_eta),
                planned_road_distance_m=float(distance),
            ):
                self._stay_reason_by_agent[str(new_owner)] = (
                    "controlled_nearby_routine_contract_transfer"
                )
                self._routine_opportunity_transfer_count_by_task[task_id] = int(
                    self._routine_opportunity_transfer_count_by_task.get(task_id, 0)
                ) + 1
                self._contract_last_transfer_step[task_id] = int(step_now)
                self.routine_opportunity_transfer_count += 1
                self.contract_transfer_count += 1
                if balanced_v2:
                    self.balanced_all_tasks_v2_reauction_transfer_count += 1
                self._feedback.append(
                    PlannerFeedback(
                        step=step_now,
                        reason="controlled_nearby_routine_contract_transfer",
                        truck_id=str(new_owner),
                        task_id=task_id,
                        detail=(
                            f"old_owner={owner_id},road_m={distance:.1f},"
                            f"candidate_eta={candidate_eta:.1f},owner_eta={owner_eta:.1f}"
                        ),
                    )
                )
                continue

            # Legacy inline transfer retained for auditability; the shared
            # atomic helper above is the active path.
            moved_stop: Optional[RouteStop] = None
            for route in self.plan.routes.values():
                for index in range(len(route.stops) - 1, int(route.cursor) - 1, -1):
                    if str(route.stops[index].task_id) == task_id:
                        moved_stop = route.stops.pop(index)
            new_route = self.plan.routes.get(str(new_owner), None)
            if new_route is None:
                continue
            if moved_stop is None:
                moved_stop = RouteStop(
                    task_id=task_id,
                    stop_type=NORMAL_SERVICE,
                    truck_id=str(new_owner),
                    uav_id=None,
                    uav_ids=(),
                    target_node=int(task.demand_node),
                    eta_step=int(step_now + candidate_eta),
                    deadline_step=int(task.deadline_step),
                    service_mode=DIRECT,
                )
            moved_stop.truck_id = str(new_owner)
            moved_stop.target_node = int(task.demand_node)
            moved_stop.eta_step = int(step_now + candidate_eta)
            new_route.stops.insert(int(new_route.cursor), moved_stop)

            if contract is not None:
                contract.owner_agent_id = str(new_owner)
                contract.truck_id = str(new_owner)
                contract.uav_id = None
                contract.uav_ids = ()
                contract.created_step = int(step_now)
                contract.locked = True
                self._stamp_contract_on_task(
                    env, task_id, contract, bump=True
                )
            task.route_contract_owner = str(new_owner)
            task.route_contract_truck = str(new_owner)
            task.route_contract_uav_ids = ()
            for agent_id, goal_id in list(goals.items()):
                if str(agent_id) != str(new_owner) and str(goal_id) == task_id:
                    goals[str(agent_id)] = None
            goals[str(new_owner)] = task_id
            self._assist_by_truck.pop(str(new_owner), None)
            self._stay_reason_by_agent[str(new_owner)] = (
                "controlled_nearby_routine_contract_transfer"
            )
            self._routine_opportunity_transfer_count_by_task[task_id] = int(
                self._routine_opportunity_transfer_count_by_task.get(task_id, 0)
            ) + 1
            self._contract_last_transfer_step[task_id] = int(step_now)
            self.routine_opportunity_transfer_count += 1
            self.contract_transfer_count += 1
            self._feedback.append(
                PlannerFeedback(
                    step=step_now,
                    reason="controlled_nearby_routine_contract_transfer",
                    truck_id=str(new_owner),
                    task_id=task_id,
                    detail=(
                        f"old_owner={owner_id},road_m={distance:.1f},"
                        f"candidate_eta={candidate_eta:.1f},owner_eta={owner_eta:.1f}"
                    ),
                )
            )

    def _promote_starving_emergency_queue(self, env, route: ClusterRoute) -> None:
        """Promote a queued emergency only when its predicted wait consumes life."""
        if not bool(
            getattr(env.cfg, "hrl_route_plan_emergency_starvation_promotion_enabled", True)
        ):
            return
        current = route.current(env)
        if current is None:
            return
        current_task = env.state.tasks.get(str(current.task_id), None)
        truck = env.state.agents.get(str(route.truck_id), None)
        # Finish an already-started/onsite routine unload before changing the head.
        if (
            current_task is not None
            and current_task.kind == TaskKind.NORMAL
            and truck is not None
            and truck.node is not None
            and int(truck.node) == int(current_task.demand_node)
        ):
            return
        # An airborne sortie is an immutable delivery commitment.
        airborne_contracts = {
            str(tid)
            for uid, tid in dict(
                getattr(env, "_uav_sortie_contract_task", {})
            ).items()
            if (
                env.state.agents.get(str(uid), None) is not None
                and env.state.agents[str(uid)].kind == AgentKind.UAV
                and env.state.agents[str(uid)].follow_target is None
            )
        }
        if str(current.task_id) in airborne_contracts:
            return

        step_now = int(env.state.step_index)
        reserve = int(
            max(
                getattr(
                    env.cfg,
                    "hrl_route_plan_emergency_starvation_reserve_steps",
                    8,
                ),
                0,
            )
        )
        best = None
        for index in range(int(route.cursor) + 1, len(route.stops)):
            stop = route.stops[index]
            task = env.state.tasks.get(str(stop.task_id), None)
            if task is None or task.kind != TaskKind.EMERGENCY or not self._active(task):
                continue
            decay = float(max(getattr(task, "lifeline_decay_rate", 0.0), 1e-9))
            life_steps = float(max(getattr(task, "lifeline_current", 0.0), 0.0) / decay)
            planned_wait = float(max(int(stop.eta_step) - step_now, 0))
            deadline_wait = float(max(int(task.deadline_step) - step_now, 0))
            predicted_budget = float(min(life_steps, deadline_wait))
            if predicted_budget > planned_wait + reserve + 1e-9:
                continue
            option = (predicted_budget - planned_wait, predicted_budget, str(task.task_id), index)
            if best is None or option < best:
                best = option
        if best is None:
            return
        _, _, task_id, index = best
        promoted = route.stops.pop(int(index))
        route.stops.insert(int(route.cursor), promoted)
        self.emergency_starvation_promotion_count += 1
        self._feedback.append(
            PlannerFeedback(
                step=step_now,
                reason="queued_emergency_starvation_promotion",
                truck_id=str(route.truck_id),
                task_id=str(task_id),
                detail=f"displaced_head={current.task_id}",
            )
        )

    def _update_emergency_launch_watchdog(self, env) -> None:
        """Force only the command transition after the full launch gate is safe."""
        self._force_takeoff_task_by_uav = {}
        if not bool(
            getattr(env.cfg, "hrl_route_plan_emergency_launch_watchdog_enabled", True)
        ):
            return
        step_now = int(env.state.step_index)
        wait_steps = int(
            max(
                getattr(
                    env.cfg,
                    "hrl_route_plan_emergency_launch_watchdog_wait_steps",
                    2,
                ),
                1,
            )
        )
        active_ready_tasks: set[str] = set()
        for truck_id, assist in self._assist_by_truck.items():
            task_id = str(assist.get("task_id", "") or "")
            uav_id = str(assist.get("uav_id", "") or "")
            launch_node = assist.get("launch_node", None)
            task = env.state.tasks.get(task_id, None)
            truck = env.state.agents.get(str(truck_id), None)
            uav = env.state.agents.get(uav_id, None)
            if (
                task is None
                or task.kind != TaskKind.EMERGENCY
                or task.status != TaskStatus.PENDING
                or truck is None
                or truck.node is None
                or launch_node is None
                or int(truck.node) != int(launch_node)
                or uav is None
                or uav.kind != AgentKind.UAV
                or str(getattr(uav, "follow_target", "")) != str(truck_id)
                or not bool(env._uav_loaded_for_task(uav_id, task))
            ):
                continue
            try:
                safe, _, _ = env._uav_launch_gate_check(uav_id, task=task)
            except Exception:
                safe = False
            if not bool(safe):
                continue
            active_ready_tasks.add(task_id)
            self.emergency_launch_watchdog_ready_count += 1
            first = int(
                self._emergency_launch_ready_since_by_task.setdefault(task_id, step_now)
            )
            if step_now - first >= wait_steps:
                self._force_takeoff_task_by_uav[uav_id] = task_id
                self.emergency_launch_watchdog_force_count += 1
        for task_id in list(self._emergency_launch_ready_since_by_task):
            if task_id not in active_ready_tasks:
                self._emergency_launch_ready_since_by_task.pop(task_id, None)

    def _advertise_direct_safe_secondary_emergencies(
        self,
        env,
        goals: Dict[str, Optional[str]],
    ) -> None:
        """Use an idle second UAV without changing the truck's route.

        Only the UAV already named by a future contract may preview that stop,
        and only when the environment confirms a direct-return-safe launch
        from its currently docked truck.  Rendezvous and cross-truck sorties
        remain exclusive to the current route stop.
        """
        if self.plan is None or not bool(
            getattr(
                env.cfg,
                "hrl_route_plan_direct_safe_secondary_emergency_enabled",
                True,
            )
        ):
            return
        for truck_id, route in sorted(self.plan.routes.items()):
            for stop in route.stops[int(route.cursor) + 1 :]:
                task_id = str(stop.task_id)
                task = env.state.tasks.get(task_id, None)
                contract = self.plan.contracts.get(task_id, None)
                if (
                    task is None
                    or task.kind != TaskKind.EMERGENCY
                    or not self._active(task)
                    or contract is None
                    or contract.uav_id is None
                ):
                    continue
                uav_id = str(contract.uav_id)
                uav = env.state.agents.get(uav_id, None)
                if (
                    goals.get(uav_id, None) is not None
                    or uav is None
                    or uav.kind != AgentKind.UAV
                    or uav.follow_target is None
                    or str(uav.follow_target) != str(truck_id)
                ):
                    continue
                try:
                    loaded = bool(env._uav_loaded_for_task(uav_id, task))
                except Exception:
                    loaded = bool(getattr(uav, "payload_kg_current", 0.0) > 1e-9)
                if not loaded:
                    continue
                self.direct_safe_secondary_emergency_candidate_count += 1
                try:
                    launch_ok, launch_reason, _ = env._uav_launch_gate_check(
                        uav_id,
                        task=task,
                        count_reject=False,
                    )
                except Exception:
                    launch_ok, launch_reason = False, ""
                if not (
                    bool(launch_ok)
                    and str(launch_reason) == "direct_safe"
                ):
                    continue
                goals[uav_id] = task_id
                self._stay_reason_by_agent[uav_id] = (
                    "direct_safe_secondary_emergency_preview"
                )
                self.direct_safe_secondary_emergency_assignment_count += 1
                # At most one future contract may be advertised per UAV.

    def _advertise_stalled_queue_rescue(
        self,
        env,
        goals: Dict[str, Optional[str]],
    ) -> None:
        """Expose one hidden emergency suffix after fleet-wide stagnation.

        A direct-return-safe task may launch from the truck's current node.
        Otherwise, the optional anchor stage moves that truck to the closest
        road-reachable candidate and rechecks the authoritative launch gate
        there.  The rescue uses only a docked loaded idle UAV, exposes one
        suffix at a time, and never bypasses the battery/safety gate.
        """
        if self.plan is None or not bool(
            getattr(env.cfg, "hrl_route_plan_stalled_queue_rescue_enabled", True)
        ):
            self._queue_rescue_task_by_uav.clear()
            self._queue_rescue_anchor_by_uav.clear()
            self._queue_rescue_started_step_by_uav.clear()
            return

        def release_rescue(uav_id: str, *, failed: bool = False) -> None:
            task_id = self._queue_rescue_task_by_uav.get(str(uav_id), None)
            if failed and task_id is not None and self.plan is not None:
                contract = self.plan.contracts.get(str(task_id), None)
                if contract is not None:
                    self._queue_rescue_failed_task_ids[str(task_id)] = str(
                        contract.truck_id
                    )
            self._queue_rescue_task_by_uav.pop(str(uav_id), None)
            self._queue_rescue_anchor_by_uav.pop(str(uav_id), None)
            self._queue_rescue_started_step_by_uav.pop(str(uav_id), None)
            self._queue_rescue_launch_ready_step_by_uav.pop(str(uav_id), None)
            self._queue_rescue_best_road_distance_by_uav.pop(str(uav_id), None)
            self._queue_rescue_last_progress_step_by_uav.pop(str(uav_id), None)
            self._queue_rescue_reanchor_count_by_uav.pop(str(uav_id), None)

        # Retain an already advertised rescue through its docked launch
        # handshake. Airborne execution is protected by the environment-side
        # sortie contract and no longer needs a truck hold here.
        active_rescue = False
        for uav_id, task_id in list(self._queue_rescue_task_by_uav.items()):
            task = env.state.tasks.get(str(task_id), None)
            uav = env.state.agents.get(str(uav_id), None)
            if (
                task is None
                or not self._active(task)
                or float(getattr(task, "lifeline_current", 0.0)) <= 1e-9
            ):
                if task is not None and task.status == TaskStatus.DELIVERED:
                    self.queue_rescue_delivery_count += 1
                release_rescue(str(uav_id))
                continue
            if uav is None or bool(getattr(uav, "crashed", False)):
                release_rescue(str(uav_id))
                continue
            active_rescue = True
            if uav.follow_target is None:
                continue
            truck_id = str(uav.follow_target)
            truck = env.state.agents.get(truck_id, None)
            if truck is None or truck.node is None or truck.transit is not None:
                continue
            anchor = int(
                self._queue_rescue_anchor_by_uav.get(str(uav_id), int(truck.node))
            )
            at_anchor = int(truck.node) == int(anchor)
            if not at_anchor:
                road_to_anchor = float(
                    env._decision_shortest_path_distance(
                        int(truck.node), int(anchor)
                    )
                )
                best_road = float(
                    self._queue_rescue_best_road_distance_by_uav.get(
                        str(uav_id), float("inf")
                    )
                )
                if np.isfinite(road_to_anchor) and road_to_anchor < best_road - 40.0:
                    self._queue_rescue_best_road_distance_by_uav[str(uav_id)] = float(
                        road_to_anchor
                    )
                    self._queue_rescue_last_progress_step_by_uav[str(uav_id)] = int(
                        env.state.step_index
                    )
                no_progress_since = int(
                    self._queue_rescue_last_progress_step_by_uav.setdefault(
                        str(uav_id), int(env.state.step_index)
                    )
                )
                # A newly discovered cut edge can leave the truck cycling
                # around an anchor that looked reachable when selected. After
                # sustained non-progress, choose a fresh backup using the new
                # road version. This changes only the rescue waypoint, never
                # the task/UAV contract.
                if (
                    not np.isfinite(road_to_anchor)
                    or int(env.state.step_index) - no_progress_since >= 12
                ):
                    island_ids = set(
                        getattr(env, "_forced_island_task_ids", set())
                    )
                    try:
                        island_ids.update(env._current_island_emergency_task_ids())
                    except Exception:
                        pass
                    replacement: Optional[Tuple[float, float, int]] = None
                    for candidate_anchor in self._anchor_nodes(
                        env,
                        int(truck.node),
                        task,
                        road_signature=self._road_signature(env),
                    ):
                        if int(candidate_anchor) == int(anchor):
                            continue
                        if (
                            str(task_id) in island_ids
                            and int(candidate_anchor) == int(task.demand_node)
                        ):
                            continue
                        road = float(
                            env._decision_shortest_path_distance(
                                int(truck.node), int(candidate_anchor)
                            )
                        )
                        if not np.isfinite(road):
                            continue
                        ax, ay = self._xy(env, int(candidate_anchor))
                        tx, ty = self._xy(env, int(task.demand_node))
                        option = (
                            float(np.hypot(ax - tx, ay - ty)),
                            float(road),
                            int(candidate_anchor),
                        )
                        if replacement is None or option < replacement:
                            replacement = option
                    reanchor_count = int(
                        self._queue_rescue_reanchor_count_by_uav.get(
                            str(uav_id), 0
                        )
                    )
                    if replacement is None or reanchor_count >= 3:
                        release_rescue(str(uav_id), failed=True)
                        self._queue_rescue_cooldown_until_by_task[str(task_id)] = int(
                            env.state.step_index
                        ) + int(
                            max(
                                getattr(
                                    env.cfg,
                                    "hrl_route_plan_stalled_queue_rescue_steps",
                                    30,
                                ),
                                1,
                            )
                        )
                        active_rescue = False
                        continue
                    anchor = int(replacement[2])
                    self._queue_rescue_anchor_by_uav[str(uav_id)] = int(anchor)
                    self._queue_rescue_best_road_distance_by_uav[str(uav_id)] = float(
                        replacement[1]
                    )
                    self._queue_rescue_last_progress_step_by_uav[str(uav_id)] = int(
                        env.state.step_index
                    )
                    self._queue_rescue_reanchor_count_by_uav[str(uav_id)] = (
                        reanchor_count + 1
                    )
            if at_anchor:
                try:
                    launch_ok, _, _ = env._uav_launch_gate_check(
                        str(uav_id), task=task, count_reject=False
                    )
                except Exception:
                    launch_ok = False
                if bool(launch_ok):
                    # A safe gate is not success until the UAV actually
                    # leaves the truck.  Bound the docked handshake so a
                    # stale goal/contract cannot hold this truck forever.
                    self._queue_rescue_started_step_by_uav.pop(str(uav_id), None)
                    ready_since = int(
                        self._queue_rescue_launch_ready_step_by_uav.setdefault(
                            str(uav_id), int(env.state.step_index)
                        )
                    )
                    launch_timeout = int(
                        max(
                            getattr(
                                env.cfg,
                                "hrl_route_plan_queue_rescue_launch_timeout_steps",
                                5,
                            ),
                            1,
                        )
                    )
                    if int(env.state.step_index) - ready_since >= launch_timeout:
                        release_rescue(str(uav_id), failed=True)
                        self._queue_rescue_cooldown_until_by_task[str(task_id)] = int(
                            env.state.step_index
                        ) + int(
                            max(
                                getattr(
                                    env.cfg,
                                    "hrl_route_plan_stalled_queue_rescue_steps",
                                    30,
                                ),
                                1,
                            )
                        )
                        self._feedback.append(
                            PlannerFeedback(
                                step=int(env.state.step_index),
                                reason="stalled_queue_launch_handshake_timeout",
                                truck_id=str(truck_id),
                                task_id=str(task_id),
                                detail=(
                                    f"uav={uav_id},anchor={anchor},"
                                    f"ready_steps={launch_timeout}"
                                ),
                                suffix_repair_required=True,
                            )
                        )
                        active_rescue = False
                        continue
                else:
                    self._queue_rescue_launch_ready_step_by_uav.pop(
                        str(uav_id), None
                    )
                    unsafe_since = int(
                        self._queue_rescue_started_step_by_uav.setdefault(
                            str(uav_id), int(env.state.step_index)
                        )
                    )
                    timeout_steps = int(
                        max(
                            getattr(
                                env.cfg,
                                "hrl_route_plan_stalled_queue_anchor_timeout_steps",
                                12,
                            ),
                            1,
                        )
                    )
                    if int(env.state.step_index) - unsafe_since >= timeout_steps:
                        release_rescue(str(uav_id), failed=True)
                        self._queue_rescue_cooldown_until_by_task[str(task_id)] = int(
                            env.state.step_index
                        ) + int(
                            max(
                                getattr(
                                    env.cfg,
                                    "hrl_route_plan_stalled_queue_rescue_steps",
                                    30,
                                ),
                                1,
                            )
                        )
                        self._feedback.append(
                            PlannerFeedback(
                                step=int(env.state.step_index),
                                reason="stalled_queue_anchor_rescue_released",
                                truck_id=str(truck_id),
                                task_id=str(task_id),
                                detail=f"uav={uav_id},anchor={anchor},unsafe_steps={timeout_steps}",
                            )
                        )
                        active_rescue = False
                        continue
            # While the truck approaches an anchor, keep the UAV explicitly
            # bound to that truck. Advertising the task early lets the lower
            # layer launch from an arbitrary intermediate node before the
            # authoritative anchor recheck.
            goals[str(uav_id)] = str(task_id) if at_anchor else str(truck_id)
            goals[truck_id] = None
            self._assist_by_truck[truck_id] = {
                "assist_waypoint_insert": True,
                "route_plan_v2": True,
                "hold_at_anchor": True,
                "idle_support": True,
                "uav_id": str(uav_id),
                "uav_ids": [str(uav_id)],
                "task_id": str(task_id),
                "launch_node": int(anchor),
                "normal_goal_task_id": "",
                "step": int(env.state.step_index),
                "service_mode": DIRECT,
                "plan_version": int(self.plan.version),
                "queue_rescue": True,
            }
            self._stay_reason_by_agent[str(uav_id)] = (
                "stalled_queue_rescue_wait_for_anchor"
                if not at_anchor
                else "stalled_queue_anchor_launch_handshake"
            )
            self._stay_reason_by_agent[truck_id] = (
                "move_to_stalled_queue_rescue_anchor"
                if not at_anchor
                else "hold_for_stalled_queue_rescue_launch"
            )
        active_rescue_uavs = set(self._queue_rescue_task_by_uav)
        max_active_rescues = int(
            max(
                getattr(
                    env.cfg,
                    "hrl_route_plan_stalled_queue_max_active_rescues",
                    1,
                ),
                1,
            )
        )
        if len(active_rescue_uavs) >= max_active_rescues:
            return

        active_rescue_trucks = set()
        for active_uav_id in active_rescue_uavs:
            active_uav = env.state.agents.get(str(active_uav_id), None)
            if active_uav is not None and active_uav.follow_target is not None:
                active_rescue_trucks.add(str(active_uav.follow_target))

        active_emergency = [
            task
            for task in env.state.tasks.values()
            if task.kind == TaskKind.EMERGENCY
            and self._active(task)
            and float(getattr(task, "lifeline_current", 0.0)) > 1e-9
        ]
        min_pending = int(
            max(
                getattr(
                    env.cfg,
                    "hrl_route_plan_stalled_queue_rescue_min_pending",
                    1,
                ),
                1,
            )
        )
        stall_steps = int(
            max(
                getattr(
                    env.cfg,
                    "hrl_route_plan_stalled_queue_rescue_steps",
                    30,
                ),
                1,
            )
        )
        urgent_horizon = int(
            max(
                getattr(
                    env.cfg,
                    "hrl_route_plan_queue_urgent_rescue_horizon_steps",
                    0,
                ),
                0,
            )
        )
        step_now = int(env.state.step_index)
        urgent_queue_exists = False
        for task in active_emergency if urgent_horizon > 0 else ():
            life_now = float(max(getattr(task, "lifeline_current", 0.0), 0.0))
            decay = float(max(getattr(task, "lifeline_decay_rate", 0.0), 0.0))
            life_steps = (
                float(life_now / decay) if decay > 1e-9 else float("inf")
            )
            deadline_steps = float(
                max(self._effective_deadline_step(env, task) - step_now, 0)
            )
            if min(life_steps, deadline_steps) <= urgent_horizon + 1e-9:
                urgent_queue_exists = True
                break
        # Candidate-only fairness gate.  Retain any already advertised rescue
        # above, but do not create the first one while every NORMAL task is
        # still untouched.  A deadline/lifeline urgent queue is an explicit
        # emergency-starvation guard and is allowed to bypass this gate.
        if bool(
            getattr(
                env.cfg,
                "hrl_route_plan_stalled_queue_rescue_normal_service_gate_enabled",
                False,
            )
        ) and not urgent_queue_exists:
            normal_service_started = any(
                task.kind == TaskKind.NORMAL
                and getattr(task, "first_service_step", None) is not None
                for task in env.state.tasks.values()
            )
            if not normal_service_started:
                return
        if (
            len(active_emergency) < min_pending
            or (
                step_now - int(self._last_completion_progress_step) < stall_steps
                and not urgent_queue_exists
            )
        ):
            return

        occupied_tasks = {
            str(goal_id)
            for goal_id in goals.values()
            if goal_id is not None and str(goal_id) in env.state.tasks
        }
        occupied_tasks.update(
            str(task_id)
            for task_id in dict(
                getattr(env, "_uav_sortie_contract_task", {})
            ).values()
            if task_id is not None
        )
        occupied_tasks.update(
            str(task_id)
            for task_id in self._queue_rescue_task_by_uav.values()
        )
        occupied_tasks.update(
            str(task_id)
            for task_id, until_step in list(
                self._queue_rescue_cooldown_until_by_task.items()
            )
            if int(until_step) > int(env.state.step_index)
        )
        for task_id, until_step in list(
            self._queue_rescue_cooldown_until_by_task.items()
        ):
            if int(until_step) <= int(env.state.step_index):
                self._queue_rescue_cooldown_until_by_task.pop(str(task_id), None)
        candidates: List[Tuple[int, float, str, str, str]] = []
        docked_loaded_pairs: List[Tuple[str, str]] = []
        for uav_id, uav in sorted(env.state.agents.items()):
            if (
                uav.kind != AgentKind.UAV
                or bool(getattr(uav, "crashed", False))
                or str(uav_id) in active_rescue_uavs
                or uav.follow_target is None
                or goals.get(str(uav_id), None)
                not in (None, str(uav.follow_target))
                or not bool(env._uav_loaded(str(uav_id)))
            ):
                continue
            truck_id = str(uav.follow_target)
            if truck_id in active_rescue_trucks:
                continue
            truck = env.state.agents.get(truck_id, None)
            if truck is None or truck.node is None or truck.transit is not None:
                continue
            docked_loaded_pairs.append((str(uav_id), str(truck_id)))
            for task in active_emergency:
                task_id = str(task.task_id)
                if task_id in occupied_tasks:
                    continue
                try:
                    launch_ok, launch_reason, _ = env._uav_launch_gate_check(
                        str(uav_id), task=task, count_reject=False
                    )
                except Exception:
                    launch_ok, launch_reason = False, ""
                if not bool(launch_ok) or str(launch_reason) != "direct_safe":
                    continue
                distance = float(env._agent_distance_to_task(str(uav_id), task))
                candidates.append(
                    (
                        int(self._effective_deadline_step(env, task)),
                        float(distance),
                        task_id,
                        str(uav_id),
                        truck_id,
                    )
                )
        selected_anchor: Optional[int] = None
        rescue_reason = "stalled_queue_direct_safe_rescue"
        if candidates:
            _, _, task_id, uav_id, truck_id = min(candidates)
            selected_anchor = int(env.state.agents[str(truck_id)].node)
        else:
            if not bool(
                getattr(
                    env.cfg,
                    "hrl_route_plan_stalled_queue_anchor_rescue_enabled",
                    False,
                )
            ):
                return
            road_signature = self._road_signature(env)
            island_ids = set(getattr(env, "_forced_island_task_ids", set()))
            try:
                island_ids.update(env._current_island_emergency_task_ids())
            except Exception:
                pass
            anchor_candidates: List[
                Tuple[int, float, float, str, str, str, int]
            ] = []
            for uav_id, truck_id in docked_loaded_pairs:
                truck = env.state.agents[str(truck_id)]
                for task in active_emergency:
                    task_id = str(task.task_id)
                    if task_id in occupied_tasks:
                        continue
                    anchors = self._anchor_nodes(
                        env,
                        int(truck.node),
                        task,
                        road_signature=road_signature,
                    )
                    for anchor in anchors:
                        if (
                            task_id in island_ids
                            and int(anchor) == int(task.demand_node)
                        ):
                            continue
                        road = float(
                            env._decision_shortest_path_distance(
                                int(truck.node), int(anchor)
                            )
                        )
                        if not np.isfinite(road):
                            continue
                        ax, ay = self._xy(env, int(anchor))
                        tx, ty = self._xy(env, int(task.demand_node))
                        air = float(np.hypot(ax - tx, ay - ty))
                        anchor_candidates.append(
                            (
                                int(self._effective_deadline_step(env, task)),
                                air,
                                road,
                                task_id,
                                str(uav_id),
                                str(truck_id),
                                int(anchor),
                            )
                        )
                        # _anchor_nodes is already ordered by air distance,
                        # then road distance. Only its best feasible anchor is
                        # needed for this UAV-task pair.
                        break
            if not anchor_candidates:
                return
            (
                _,
                _,
                _,
                task_id,
                uav_id,
                truck_id,
                selected_anchor,
            ) = min(anchor_candidates)
            rescue_reason = "stalled_queue_anchor_rescue"
        task = env.state.tasks[str(task_id)]
        contract = self.plan.contracts.get(str(task_id), None)
        if contract is None:
            contract = TaskContract(
                task_id=str(task_id),
                owner_agent_id=str(uav_id),
                truck_id=str(truck_id),
                uav_id=str(uav_id),
                uav_ids=(str(uav_id),),
                service_mode=DIRECT,
                created_step=int(env.state.step_index),
            )
            self.plan.contracts[str(task_id)] = contract
            self._stamp_contract_on_task(
                env, str(task_id), contract, bump=False
            )
        else:
            changed = bool(
                str(contract.owner_agent_id) != str(uav_id)
                or str(contract.truck_id) != str(truck_id)
                or str(contract.uav_id or "") != str(uav_id)
            )
            contract.owner_agent_id = str(uav_id)
            contract.truck_id = str(truck_id)
            contract.uav_id = str(uav_id)
            contract.uav_ids = (str(uav_id),)
            contract.service_mode = DIRECT
            contract.created_step = int(env.state.step_index)
            self._stamp_contract_on_task(
                env, str(task_id), contract, bump=changed
            )
        self._queue_rescue_task_by_uav[str(uav_id)] = str(task_id)
        self._queue_rescue_anchor_by_uav[str(uav_id)] = int(selected_anchor)
        self._queue_rescue_started_step_by_uav.pop(str(uav_id), None)
        self._queue_rescue_launch_ready_step_by_uav.pop(str(uav_id), None)
        initial_rescue_road = float(
            env._decision_shortest_path_distance(
                int(env.state.agents[str(truck_id)].node), int(selected_anchor)
            )
        )
        self._queue_rescue_best_road_distance_by_uav[str(uav_id)] = float(
            initial_rescue_road
        )
        self._queue_rescue_last_progress_step_by_uav[str(uav_id)] = int(
            env.state.step_index
        )
        self._queue_rescue_reanchor_count_by_uav[str(uav_id)] = 0
        self.queue_rescue_assignment_count += 1
        truck_at_anchor = bool(
            env.state.agents[str(truck_id)].node is not None
            and int(env.state.agents[str(truck_id)].node) == int(selected_anchor)
        )
        goals[str(uav_id)] = (
            str(task_id) if truck_at_anchor else str(truck_id)
        )
        goals[str(truck_id)] = None
        self._assist_by_truck[str(truck_id)] = {
            "assist_waypoint_insert": True,
            "route_plan_v2": True,
            "hold_at_anchor": True,
            "idle_support": True,
            "uav_id": str(uav_id),
            "uav_ids": [str(uav_id)],
            "task_id": str(task_id),
            "launch_node": int(selected_anchor),
            "normal_goal_task_id": "",
            "step": int(env.state.step_index),
            "service_mode": DIRECT,
            "plan_version": int(self.plan.version),
            "queue_rescue": True,
        }
        self._feedback.append(
            PlannerFeedback(
                step=int(env.state.step_index),
                reason=str(rescue_reason),
                truck_id=str(truck_id),
                task_id=str(task_id),
                detail=(
                    f"uav={uav_id},pending={len(active_emergency)},"
                    f"anchor={selected_anchor}"
                ),
            )
        )

    def _next_parallel_normal_stop(self, env, route: ClusterRoute) -> Optional[RouteStop]:
        """Return the immediate normal stop after the current emergency stop.

        The pilot deliberately does not skip over another emergency stop.  A
        truck may advance through exactly one direct-routine target while its
        docked UAV executes the current emergency sortie; the next replan then
        re-evaluates the corridor from the realized state.
        """
        idx = int(getattr(route, "cursor", 0))
        if idx < 0 or idx >= len(route.stops) - 1:
            return None
        candidate = route.stops[idx + 1]
        task = env.state.tasks.get(str(candidate.task_id), None)
        if task is None or not self._active(task):
            return None
        if candidate.stop_type != NORMAL_SERVICE:
            return None
        if task.kind != TaskKind.NORMAL or self._is_relay(task):
            return None
        if task.status != TaskStatus.PENDING:
            return None
        return candidate

    def _parallel_routine_emergency_corridor(
        self,
        env,
        support_truck: str,
        emergency_stop: RouteStop,
        emergency_task,
        contract: TaskContract,
        normal_stop: Optional[RouteStop],
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Check whether a truck can continue to one routine target.

        This is a candidate-only upper-layer check.  It does not launch or
        retarget a UAV and it does not mutate the environment.  A true result
        means that the existing actor/critic route command may move the truck
        to the normal target while the UAV keeps the emergency contract.  The
        recovery leg is explicitly evaluated from the emergency node to the
        routine target with the post-service payload; otherwise the caller
        keeps the original wait-at-anchor command.
        """
        diag: Dict[str, Any] = {
            "support_truck": str(support_truck),
            "uav_id": str(getattr(contract, "uav_id", "") or ""),
            "emergency_task_id": str(getattr(emergency_task, "task_id", "")),
            "normal_task_id": "" if normal_stop is None else str(normal_stop.task_id),
            "reason": "disabled",
        }
        if not er_hlns_parallel_routine_emergency_active(env):
            return False, "disabled", diag
        balanced_v2 = bool(er_hlns_balanced_all_tasks_v2_active(env))
        if normal_stop is None or contract is None or contract.uav_id is None:
            return False, "no_normal_successor", diag
        if str(support_truck) != str(emergency_stop.truck_id):
            return False, "cross_truck_route", diag
        truck = env.state.agents.get(str(support_truck), None)
        uav = env.state.agents.get(str(contract.uav_id), None)
        normal_task = env.state.tasks.get(str(normal_stop.task_id), None)
        airborne_contract = str(
            dict(getattr(env, "_uav_sortie_contract_task", {})).get(
                str(contract.uav_id), ""
            )
            or ""
        )
        docked_on_truck = bool(
            uav is not None
            and uav.follow_target is not None
            and str(uav.follow_target) == str(support_truck)
        )
        if balanced_v2 and bool(
            getattr(
                env.cfg,
                "hrl_route_plan_balanced_all_tasks_v2_after_launch_only",
                True,
            )
        ):
            # V2 is intentionally stricter than the earlier parallel pilot:
            # the support truck is released only after the environment has
            # published an airborne sortie contract for this exact UAV/task.
            docked_on_truck = False
        airborne_on_contract = bool(
            uav is not None
            and uav.follow_target is None
            and bool(
                getattr(
                    env.cfg,
                    "hrl_route_plan_parallel_routine_emergency_after_launch_enabled",
                    False,
                )
            )
            and airborne_contract == str(emergency_task.task_id)
        )
        emergency_state_executable = bool(
            emergency_task.status == TaskStatus.PENDING
            or (
                airborne_on_contract
                and (
                    emergency_task.status == TaskStatus.CLAIMED
                    or bool(getattr(emergency_task, "in_service_by", None))
                )
            )
        )
        loaded_for_task = bool(
            env._uav_loaded_for_task(str(contract.uav_id), emergency_task)
        )
        airborne_payload_bypass = bool(
            balanced_v2 and airborne_on_contract and not loaded_for_task
        )
        if airborne_payload_bypass:
            self.balanced_all_tasks_v2_parallel_payload_bypass_count += 1
        if (
            truck is None
            or truck.kind != AgentKind.TRUCK
            or truck.node is None
            or uav is None
            or uav.kind != AgentKind.UAV
            or bool(getattr(uav, "crashed", False))
            or not (docked_on_truck or airborne_on_contract)
            or not (loaded_for_task or airborne_payload_bypass)
            or not emergency_state_executable
            or normal_task is None
            or normal_task.status != TaskStatus.PENDING
        ):
            return False, "contract_or_state", diag
        normal_owner = str(getattr(normal_task, "route_contract_truck", "") or "")
        if (
            normal_owner
            and normal_owner != str(support_truck)
            and not (
                balanced_v2
                and normal_task.status == TaskStatus.PENDING
                and getattr(normal_task, "first_service_step", None) is None
                and getattr(normal_task, "assigned_to", None) is None
                and getattr(normal_task, "in_service_by", None) is None
                and not tuple(getattr(normal_task, "route_contract_uav_ids", ()) or ())
            )
        ):
            return False, "normal_contract_owner", diag

        anchor = int(
            emergency_stop.selected_anchor
            if emergency_stop.selected_anchor is not None
            else emergency_stop.target_node
        )
        try:
            anchor_xy = self._xy(env, anchor)
            emergency_xy = self._xy(env, int(emergency_task.demand_node))
            normal_xy = self._xy(env, int(normal_task.demand_node))
        except Exception:
            return False, "geometry", diag
        road_to_anchor = float(
            env._decision_shortest_path_distance(int(truck.node), int(anchor))
        )
        road_to_normal = float(
            env._decision_shortest_path_distance(int(anchor), int(normal_task.demand_node))
        )
        if not np.isfinite(road_to_anchor) or not np.isfinite(road_to_normal):
            return False, "road_disconnected", diag

        # The truck must reach the emergency anchor on the original route
        # before the UAV can launch; only the suffix after that anchor is
        # parallelized.  This keeps the emergency ETA unchanged.
        dt = float(max(getattr(env.cfg, "dt_seconds", 20.0), 1e-6))
        truck_speed = float(max(getattr(env.cfg, "truck_speed_mps", 10.0), 1e-6))
        uav_speed = float(max(getattr(env.cfg, "uav_max_speed_mps", 22.0), 1e-6))
        d_go = float(np.hypot(emergency_xy[0] - anchor_xy[0], emergency_xy[1] - anchor_xy[1]))
        d_recovery = float(np.hypot(normal_xy[0] - emergency_xy[0], normal_xy[1] - emergency_xy[1]))
        diag.update(
            {
                "road_to_anchor_m": road_to_anchor,
                "road_anchor_to_normal_m": road_to_normal,
                "air_outbound_m": d_go,
                "air_recovery_m": d_recovery,
            }
        )

        # Use the same weather gates and direction-aware V1 energy estimator as
        # execution.  A recovery target with unsafe weather is not accepted;
        # the truck will wait at the anchor instead.
        for xy, recovery in ((anchor_xy, False), (emergency_xy, False), (normal_xy, True)):
            reason = str(env._uav_weather_safety_reason(xy, recovery=recovery))
            if reason:
                return False, reason, diag
        payload_now = float(max(getattr(uav, "payload_kg_current", 0.0), 0.0))
        payload_after = float(
            max(env._uav_expected_payload_after_task(str(contract.uav_id), emergency_task), 0.0)
        )
        try:
            recovery_buffer = float(
                max(
                    env._effective_recovery_buffer_for_sortie(
                        str(contract.uav_id), emergency_task, launch_reason="parallel_routine"
                    ),
                    0.0,
                )
            )
        except Exception:
            recovery_buffer = float(max(getattr(env.cfg, "uav_recovery_distance_buffer_m", 600.0), 0.0))
        req_go = float(
            env._uav_energy_cost_fraction(
                str(contract.uav_id), d_go, anchor_xy,
                destination=emergency_xy, payload_override=payload_now,
            )
        )
        req_back = float(
            env._uav_energy_cost_fraction(
                str(contract.uav_id), d_recovery + recovery_buffer, emergency_xy,
                destination=env._uav_extended_destination(
                    emergency_xy, normal_xy, d_recovery + recovery_buffer
                ),
                payload_override=payload_after,
            )
        )
        reserve = float(np.clip(getattr(env.cfg, "uav_emergency_reserve_fraction", 0.20), 0.0, 1.0))
        return_margin = float(np.clip(getattr(env.cfg, "uav_return_margin_fraction", 0.15), 0.0, 1.0))
        rendez_margin = float(np.clip(getattr(env.cfg, "uav_rendezvous_margin_fraction", 0.10), 0.0, 1.0))
        required = float(req_go + req_back + reserve + return_margin + 0.5 * rendez_margin)
        battery = float(max(getattr(uav, "battery", 0.0), 0.0))
        diag.update({"required_battery": required, "battery": battery, "battery_margin": battery - required})
        if battery + 1e-9 < required:
            self.parallel_routine_emergency_reject_energy_count += 1
            return False, "recovery_energy", diag
        if bool(getattr(env, "_legacy_sortie_cap_enabled", lambda: False)()):
            cap = float(max(getattr(env.cfg, "uav_max_sortie_m", 6000.0), 1.0))
            if d_go + d_recovery + recovery_buffer > 0.92 * cap + 1e-9:
                self.parallel_routine_emergency_reject_energy_count += 1
                return False, "sortie_cap", diag

        service_steps = int(
            max(
                getattr(emergency_task, "service_remaining", 0),
                getattr(env.cfg, "uav_service_time_steps", getattr(env.cfg, "service_time_steps", 1)),
                1,
            )
        )
        eta_to_emergency = int(np.ceil(road_to_anchor / max(truck_speed * dt, 1e-6)))
        eta_to_emergency += int(np.ceil(d_go / max(uav_speed * dt, 1e-6))) + service_steps
        slack = int(max(int(getattr(emergency_task, "deadline_step", env.cfg.max_steps)) - int(env.state.step_index), 0))
        deadline_reserve = int(max(getattr(env.cfg, "hrl_route_plan_emergency_deadline_reserve_steps", 20), 0))
        diag.update({"emergency_eta_steps": eta_to_emergency, "emergency_slack_steps": slack})
        if eta_to_emergency + deadline_reserve > slack:
            self.parallel_routine_emergency_reject_deadline_count += 1
            return False, "emergency_deadline", diag
        # The ordinary target itself must be a useful route continuation, not a
        # large detour that merely hides a risky recovery.  This bound is on the
        # truck's post-anchor path and is algorithm-owned, not physical.
        if road_to_normal > float(max(1.6 * getattr(env.cfg, "uav_max_sortie_m", 6000.0), 2500.0)):
            return False, "normal_target_too_far", diag
        diag["reason"] = "accepted"
        return True, "accepted", diag

    def _advertise_after_launch_normal_opportunity(
        self,
        env,
        goals: Dict[str, Optional[str]],
    ) -> None:
        """Auction one fresh NORMAL target after an exact UAV launch.

        V2 is allowed to leave the precomputed route line only after the UAV
        sortie map proves that the matching aircraft is airborne.  The
        candidate normal stop is checked through the same corridor/energy/
        weather/deadline gate as an in-route successor, then moved atomically
        to the support truck.  No claimed, started, partial, UAV-owned or
        airborne task is eligible.
        """
        if self.plan is None or not er_hlns_balanced_all_tasks_v2_active(env):
            return
        sorties = {
            str(uid): str(task_id)
            for uid, task_id in dict(
                getattr(env, "_uav_sortie_contract_task", {})
            ).items()
            if task_id is not None
        }
        if not sorties:
            return
        active_sortie_tasks = set(sorties.values())
        step_now = int(getattr(env.state, "step_index", 0))
        speed = float(max(getattr(env.cfg, "truck_speed_mps", 10.0), 1e-6))
        dt = float(max(getattr(env.cfg, "dt_seconds", 1.0), 1e-6))
        best_global: Optional[Tuple[float, float, str, str, RouteStop, Any, TaskContract]] = None
        for route in self.plan.routes.values():
            stop = route.current(env)
            if stop is None or stop.stop_type not in {
                EMERGENCY_LAUNCH,
                BULK_RELAY_LAUNCH,
            }:
                continue
            emergency = env.state.tasks.get(str(stop.task_id), None)
            contract = self.plan.contracts.get(str(stop.task_id), None)
            if emergency is None or contract is None or contract.uav_id is None:
                continue
            uav_id = str(contract.uav_id)
            uav = env.state.agents.get(uav_id, None)
            if (
                uav is None
                or bool(getattr(uav, "crashed", False))
                or uav.follow_target is not None
                or sorties.get(uav_id, "") != str(emergency.task_id)
            ):
                continue
            support_id = str(contract.truck_id or route.truck_id)
            support = env.state.agents.get(support_id, None)
            if support is None or support.node is None or bool(getattr(support, "crashed", False)):
                continue
            if goals.get(support_id) not in (None, ""):
                continue
            if str(support_id) in self._assist_by_truck and str(
                self._assist_by_truck[support_id].get("service_mode", "")
            ).upper() in {"SAFETY_RECOVERY", "CROSS_TRUCK_RECOVERY"}:
                continue
            if hasattr(env, "_truck_has_assigned_airborne_hard_recovery_request") and bool(
                env._truck_has_assigned_airborne_hard_recovery_request(support_id)
            ):
                continue

            for normal in sorted(env.state.tasks.values(), key=lambda item: str(item.task_id)):
                if (
                    normal.kind != TaskKind.NORMAL
                    or not self._active(normal)
                    or self._is_relay(normal)
                    or normal.status != TaskStatus.PENDING
                    or getattr(normal, "first_service_step", None) is not None
                    or getattr(normal, "assigned_to", None) is not None
                    or getattr(normal, "in_service_by", None) is not None
                    or float(max(getattr(normal, "fulfilled_mass_kg", 0.0), 0.0)) > 1e-9
                    or tuple(getattr(normal, "route_contract_uav_ids", ()) or ())
                    or str(normal.task_id) in active_sortie_tasks
                ):
                    continue
                self.balanced_all_tasks_v2_after_launch_normal_candidate_count += 1
                if float(max(getattr(support, "bulk_inventory_kg_current", 0.0), 0.0)) + 1e-9 < self._remaining_demand_kg(normal):
                    self.balanced_all_tasks_v2_after_launch_normal_reject_count += 1
                    continue
                road = float(
                    env._decision_shortest_path_distance(
                        int(support.node), int(normal.demand_node)
                    )
                )
                if not np.isfinite(road):
                    self.balanced_all_tasks_v2_after_launch_normal_reject_count += 1
                    continue
                eta = float(
                    step_now + np.ceil(road / max(speed * dt, 1e-6))
                    + max(getattr(env.cfg, "unload_rounds_normal", 1), 1)
                )
                if eta > float(self._effective_deadline_step(env, normal)) + 1e-9:
                    self.balanced_all_tasks_v2_after_launch_normal_reject_count += 1
                    continue
                temporary_stop = RouteStop(
                    task_id=str(normal.task_id),
                    stop_type=NORMAL_SERVICE,
                    truck_id=support_id,
                    uav_id=None,
                    uav_ids=(),
                    target_node=int(normal.demand_node),
                    deadline_step=int(self._effective_deadline_step(env, normal)),
                    service_mode=DIRECT,
                )
                ok, _reason, diag = self._parallel_routine_emergency_corridor(
                    env,
                    support_id,
                    stop,
                    emergency,
                    contract,
                    temporary_stop,
                )
                if not ok:
                    self.balanced_all_tasks_v2_after_launch_normal_reject_count += 1
                    continue
                option = (
                    float(eta),
                    float(diag.get("road_anchor_to_normal_m", road)),
                    str(normal.task_id),
                    support_id,
                    temporary_stop,
                    normal,
                    contract,
                )
                if best_global is None or option[:4] < best_global[:4]:
                    best_global = option

        if best_global is None:
            return
        eta, road, task_id, support_id, _stop, _task, _contract = best_global
        if self._transfer_routine_contract_to_truck(
            env,
            task_id,
            support_id,
            goals,
            eta_step=int(eta),
            planned_road_distance_m=float(road),
        ):
            self._assist_by_truck.pop(support_id, None)
            goals[support_id] = str(task_id)
            self._stay_reason_by_agent[support_id] = "v2_after_launch_normal_opportunity"
            self.balanced_all_tasks_v2_after_launch_normal_accept_count += 1
            self._feedback.append(
                PlannerFeedback(
                    step=step_now,
                    reason="v2_after_launch_normal_opportunity",
                    truck_id=support_id,
                    task_id=task_id,
                    detail=f"eta={eta:.1f},road_m={road:.1f}",
                )
            )

    def _current_commands(self, env) -> Dict[str, Optional[str]]:
        goals: Dict[str, Optional[str]] = {
            str(agent_id): None for agent_id in env.state.agents
        }
        self._assist_by_truck = {}
        self._stay_reason_by_agent = {}
        self._transfer_by_uav = {}
        if self.plan is None:
            self._install_safety_recovery_assists(env, goals)
            return goals
        for truck_id, route in self.plan.routes.items():
            self._promote_at_risk_emergency_over_routine(env, route)
            self._promote_starving_emergency_queue(env, route)
            stop = route.current(env)
            if stop is None:
                self._stay_reason_by_agent[str(truck_id)] = "route_complete"
                continue
            task = env.state.tasks.get(str(stop.task_id), None)
            if task is None:
                continue
            contract = self.plan.contracts.get(str(stop.task_id), None)
            if stop.stop_type == NORMAL_SERVICE:
                goals[str(truck_id)] = str(stop.task_id)
                self._stay_reason_by_agent[str(truck_id)] = "execute_normal_route_stop"
                continue
            if contract is None or contract.uav_id is None:
                self._stay_reason_by_agent[str(truck_id)] = "no_uav_contract"
                continue
            uav_state = env.state.agents.get(str(contract.uav_id), None)
            support_truck = str(truck_id)
            # Keep the task owner fixed, but after an emergency recovery allow
            # the truck currently carrying that same UAV to execute the anchor.
            if (
                uav_state is not None
                and uav_state.follow_target is not None
                and str(uav_state.follow_target) in self.plan.routes
            ):
                support_truck = str(uav_state.follow_target)
            assigned_uavs = tuple(contract.uav_ids) or (str(contract.uav_id),)
            for uav_id in assigned_uavs:
                if bool(
                    getattr(
                        env.cfg,
                        "hrl_route_plan_contract_consistency_guard_enabled",
                        True,
                    )
                ):
                    sortie_task_id = dict(
                        getattr(env, "_uav_sortie_contract_task", {})
                    ).get(str(uav_id), None)
                    uav_exec = env.state.agents.get(str(uav_id), None)
                    if (
                        uav_exec is not None
                        and uav_exec.follow_target is None
                        and sortie_task_id is not None
                        and str(sortie_task_id) != str(stop.task_id)
                    ):
                        # Do not advertise a second task as the current command
                        # while the environment is correctly executing the
                        # authoritative airborne sortie contract.
                        self.contract_consistency_block_count += 1
                        self._stay_reason_by_agent[str(uav_id)] = (
                            "blocked_conflicting_airborne_sortie_contract"
                        )
                        continue
                goals[str(uav_id)] = str(stop.task_id)
            # Candidate-only parallel corridor: keep the truck moving toward
            # the immediate direct-routine successor when the same docked UAV
            # can safely recover at that target after the emergency service.
            # Otherwise preserve the established wait-at-anchor command.
            next_normal_stop = self._next_parallel_normal_stop(env, route)
            self.parallel_routine_emergency_candidate_count += int(
                1 if next_normal_stop is not None else 0
            )
            if next_normal_stop is not None and er_hlns_balanced_all_tasks_v2_active(env):
                self.balanced_all_tasks_v2_parallel_candidate_count += 1
            parallel_ok, parallel_reason, parallel_diag = (
                self._parallel_routine_emergency_corridor(
                    env,
                    str(support_truck),
                    stop,
                    task,
                    contract,
                    next_normal_stop,
                )
                if next_normal_stop is not None
                else (False, "no_normal_successor", {})
            )
            if parallel_ok and next_normal_stop is not None:
                normal_goal_id = str(next_normal_stop.task_id)
                goals[str(support_truck)] = normal_goal_id
                self.parallel_routine_emergency_accept_count += 1
                if er_hlns_balanced_all_tasks_v2_active(env):
                    self.balanced_all_tasks_v2_parallel_accept_count += 1
            else:
                normal_goal_id = ""
                goals[str(support_truck)] = None
                if next_normal_stop is not None and er_hlns_balanced_all_tasks_v2_active(env):
                    self.balanced_all_tasks_v2_parallel_reject_count += 1
                if next_normal_stop is not None and er_hlns_parallel_routine_emergency_active(env):
                    self.parallel_routine_emergency_fallback_wait_count += 1
            anchor = int(
                stop.selected_anchor
                if stop.selected_anchor is not None
                else stop.target_node
            )
            self._assist_by_truck[str(support_truck)] = {
                "assist_waypoint_insert": True,
                "route_plan_v2": True,
                "hold_at_anchor": not bool(parallel_ok),
                "idle_support": not bool(parallel_ok),
                "uav_id": str(contract.uav_id),
                "uav_ids": list(assigned_uavs),
                "task_id": str(stop.task_id),
                "launch_node": int(anchor),
                "normal_goal_task_id": str(normal_goal_id),
                "step": int(env.state.step_index),
                "service_mode": str(stop.service_mode),
                "plan_version": int(self.plan.version),
                "planned_road_distance_m": float(stop.planned_road_distance_m),
                "planned_air_distance_m": float(stop.planned_air_distance_m),
                "parallel_routine_emergency": bool(parallel_ok),
                "parallel_corridor_reason": str(parallel_reason),
                "parallel_corridor": dict(parallel_diag),
            }
            recovery_truck = stop.recovery_truck_id or getattr(contract, "recovery_truck_id", None)
            recovery_anchor = stop.recovery_anchor_node or getattr(contract, "recovery_anchor_node", None)
            if recovery_truck is not None and recovery_anchor is not None:
                self._assist_by_truck[str(recovery_truck)] = {
                    "assist_waypoint_insert": True,
                    "route_plan_v2": True,
                    "hold_at_anchor": True,
                    "idle_support": True,
                    "uav_id": str(contract.uav_id),
                    "task_id": str(stop.task_id),
                    "launch_node": int(recovery_anchor),
                    "normal_goal_task_id": "",
                    "step": int(env.state.step_index),
                    "service_mode": "CROSS_TRUCK_RECOVERY",
                    "plan_version": int(self.plan.version),
                }
                self._transfer_by_uav[str(contract.uav_id)] = {
                    "recovery_truck_id": str(recovery_truck),
                    "task_id": str(stop.task_id),
                }
                self._stay_reason_by_agent[str(recovery_truck)] = "move_to_or_hold_cross_truck_recovery_anchor"
            self._stay_reason_by_agent[str(support_truck)] = (
                "move_to_or_hold_emergency_anchor"
                if stop.service_mode != BULK_RELAY
                else "move_to_or_hold_bulk_relay_anchor"
            )
        emergency_work_active = any(
            task.kind == TaskKind.EMERGENCY
            and self._active(task)
            and float(getattr(task, "lifeline_current", 0.0)) > 1e-9
            for task in env.state.tasks.values()
        )
        self._advertise_direct_safe_secondary_emergencies(env, goals)
        self._advertise_stalled_queue_rescue(env, goals)
        self._ensure_residual_normal_coverage(env)
        self._advertise_idle_post_emergency_normal_fallback(env, goals)
        if not emergency_work_active:
            for task_id, cleanup_truck in list(
                self._normal_cleanup_owner_by_task.items()
            ):
                task = env.state.tasks.get(str(task_id), None)
                truck_state = env.state.agents.get(str(cleanup_truck), None)
                if (
                    task is None
                    or not self._active(task)
                    or task.kind != TaskKind.NORMAL
                    or self._is_relay(task)
                    or truck_state is None
                    or truck_state.node is None
                ):
                    self._normal_cleanup_owner_by_task.pop(str(task_id), None)
                    continue
                for agent_id, goal_id in list(goals.items()):
                    if str(agent_id) != str(cleanup_truck) and str(goal_id) == str(task_id):
                        goals[str(agent_id)] = None
                goals[str(cleanup_truck)] = str(task_id)
                self._stay_reason_by_agent[str(cleanup_truck)] = (
                    "post_emergency_nearest_normal_cleanup"
                )
        self._apply_controlled_routine_opportunity_transfers(env, goals)
        self._r4_stalled_routine_takeovers(env, goals)
        self._routine_service_start_rescue(env, goals)
        self._idle_routine_dispatch(env, goals)
        self._advertise_after_launch_normal_opportunity(env, goals)
        self._hard_normal_coverage_rescue(env, goals)
        # Exclusive contracts must not block a zero-distance physical service.
        # Atomically transfer a DIRECT routine task to a stocked truck already
        # at the demand node, and clear the stale owner's advertised goal.
        # The common environment service transition now owns this operation.
        # Keep the planner implementation available only as a compatibility
        # switch for historical experiments.
        planner_onsite_tasks = (
            env.state.tasks.values()
            if bool(
                getattr(
                    env.cfg,
                    "hrl_route_plan_onsite_takeover_enabled",
                    False,
                )
            )
            else ()
        )
        for task in planner_onsite_tasks:
            if (
                task.kind != TaskKind.NORMAL
                or not self._active(task)
                or self._is_relay(task)
            ):
                continue
            onsite_trucks = []
            remaining_kg = self._remaining_demand_kg(task)
            for truck_id in self._live_trucks(env):
                state = env.state.agents.get(str(truck_id), None)
                if (
                    state is None
                    or state.node is None
                    or int(state.node) != int(task.demand_node)
                    or float(
                        max(
                            getattr(state, "bulk_inventory_kg_current", 0.0),
                            0.0,
                        )
                    )
                    + 1e-9
                    < remaining_kg
                ):
                    continue
                onsite_trucks.append(str(truck_id))
            if not onsite_trucks:
                continue
            takeover_truck = min(onsite_trucks)
            for agent_id, goal_id in list(goals.items()):
                if str(agent_id) != takeover_truck and str(goal_id) == str(task.task_id):
                    goals[str(agent_id)] = None
            goals[takeover_truck] = str(task.task_id)
            # A stale emergency-support anchor has higher low-level precedence
            # than a task goal. Once the onsite contract is transferred, that
            # old command must be released atomically as well.
            self._assist_by_truck.pop(takeover_truck, None)
            contract = self.plan.contracts.get(str(task.task_id), None)
            if contract is not None:
                contract.owner_agent_id = takeover_truck
                contract.truck_id = takeover_truck
                contract.uav_id = None
                contract.uav_ids = ()
                contract.created_step = int(env.state.step_index)
                self._stamp_contract_on_task(
                    env, str(task.task_id), contract, bump=True
                )
            task.route_contract_owner = takeover_truck
            task.route_contract_truck = takeover_truck
            task.route_contract_uav_ids = ()
            self._stay_reason_by_agent[takeover_truck] = (
                "onsite_exclusive_routine_takeover"
            )
            self.onsite_takeover_count += 1
        for uav_id, state in env.state.agents.items():
            if state.kind != AgentKind.UAV or goals.get(str(uav_id)) is not None:
                continue
            if state.follow_target is not None:
                goals[str(uav_id)] = str(state.follow_target)
        self._install_safety_recovery_assists(env, goals)
        # Safety-recovery assists are installed last and may otherwise
        # overwrite an onsite routine takeover created above.  Reconcile the
        # executable command once more at the final publication boundary:
        # a stocked truck already at its contracted DIRECT demand node unloads
        # first, while the recovery request remains eligible for reassignment
        # on the following step.
        for task in env.state.tasks.values():
            if (
                task.kind != TaskKind.NORMAL
                or not self._active(task)
                or self._is_relay(task)
            ):
                continue
            contract_truck = str(
                getattr(task, "route_contract_truck", "") or ""
            )
            truck = env.state.agents.get(contract_truck, None)
            if (
                truck is None
                or truck.node is None
                or int(truck.node) != int(task.demand_node)
                or not bool(
                    env.is_task_serviceable_by_agent(contract_truck, task)
                )
            ):
                continue
            self._assist_by_truck.pop(contract_truck, None)
            goals[contract_truck] = str(task.task_id)
            self._stay_reason_by_agent[contract_truck] = (
                "final_onsite_direct_routine_commitment"
            )
        # The assist map is now complete.  A loaded, docked UAV that has
        # remained fully launch-safe for the configured wait may bypass only
        # the low-level goal-stability delay on this published command.
        self._update_emergency_launch_watchdog(env)
        return goals

    def _missing_contract_fingerprint(
        self,
        env,
        missing_task_ids: Sequence[str],
        truck_ids: Sequence[str],
        road_signature: Tuple[Tuple[int, int], ...],
    ) -> Tuple[Any, ...]:
        """Describe only state changes that can make a missing task insertable."""
        clusters = self._cluster_uavs(env, truck_ids)
        rows: List[Tuple[str, Tuple[str, ...]]] = []
        for task_id in sorted(str(item) for item in missing_task_ids):
            task = env.state.tasks.get(str(task_id), None)
            feasible_trucks: List[str] = []
            if task is not None and self._active(task):
                for truck_id in truck_ids:
                    truck = env.state.agents.get(str(truck_id), None)
                    if truck is None or truck.node is None:
                        continue
                    if task.kind == TaskKind.NORMAL and not self._is_relay(task):
                        if float(
                            max(
                                getattr(
                                    truck, "bulk_inventory_kg_current", 0.0
                                ),
                                0.0,
                            )
                        ) + 1e-9 < self._remaining_demand_kg(task):
                            continue
                        distance = float(
                            env._decision_shortest_path_distance(
                                int(truck.node), int(task.demand_node)
                            )
                        )
                        if np.isfinite(distance):
                            feasible_trucks.append(str(truck_id))
                    elif self._single_emergency_eta(
                        env,
                        str(truck_id),
                        task,
                        clusters,
                        road_signature,
                    ) is not None:
                        feasible_trucks.append(str(truck_id))
            rows.append((str(task_id), tuple(feasible_trucks)))
        return (
            tuple(rows),
            int(self._last_completed_task_count),
        )

    def _rescue_orphaned_b_routine_goals(
        self,
        env,
        goals: Dict[str, Optional[str]],
    ) -> None:
        """Recover one stale B routine contract without preempting live work.

        Route-plan v2 can legitimately stamp a routine owner while that truck
        is temporarily serving an emergency anchor.  In a few road-only B
        seeds this leaves a pending routine with no executable goal at all.
        C's communication commitment lets a later idle-truck cleanup reclaim
        that task.  This bounded B analogue only acts after the task's
        lifeline falls below the configured threshold, requires an actually
        idle route (no current stop and no assist), and transfers at most one
        task per planning step.
        """
        if self.plan is None:
            return
        if not bool(getattr(env.cfg, "erc_b_orphaned_routine_rescue_enabled", False)):
            return
        if str(getattr(env.cfg, "scenario", "")).upper().strip() != "B":
            return
        map_key = str(getattr(env.cfg, "map_complexity", "")).upper().strip()
        if map_key not in {"L", "R"}:
            return
        step_now = int(getattr(env.state, "step_index", 0))
        min_step = int(
            max(getattr(env.cfg, "erc_b_orphaned_routine_rescue_min_step", 120), 0)
        )
        if step_now < min_step:
            return
        max_ratio = float(
            np.clip(
                getattr(env.cfg, "erc_b_orphaned_routine_rescue_max_lifeline_ratio", 0.80),
                0.0,
                1.0,
            )
        )
        candidates = []
        for task in env.state.tasks.values():
            if (
                task.kind != TaskKind.NORMAL
                or task.status != TaskStatus.PENDING
                or self._is_relay(task)
                or getattr(task, "assigned_to", None) is not None
                or getattr(task, "first_service_step", None) is not None
            ):
                continue
            ratio = float(
                np.clip(
                    float(getattr(task, "lifeline_current", 0.0))
                    / max(float(getattr(task, "lifeline_init", 100.0)), 1e-9),
                    0.0,
                    1.0,
                )
            )
            if ratio > max_ratio + 1e-9:
                continue
            task_id = str(task.task_id)
            # An executable command already protects this task.  A route
            # contract alone is not enough: that is precisely the orphan case.
            if any(str(goal_id) == task_id for goal_id in goals.values() if goal_id is not None):
                continue
            best = None
            for truck_id in self._live_trucks(env):
                truck_id = str(truck_id)
                truck = env.state.agents.get(truck_id, None)
                route = self.plan.routes.get(truck_id, None)
                if (
                    truck is None
                    or truck.node is None
                    or bool(getattr(truck, "crashed", False))
                    or truck_id in self._assist_by_truck
                    or goals.get(truck_id, None) is not None
                    or route is None
                    or route.current(env) is not None
                    or float(max(getattr(truck, "bulk_inventory_kg_current", 0.0), 0.0))
                    + 1e-9
                    < self._remaining_demand_kg(task)
                ):
                    continue
                road = float(
                    env._decision_shortest_path_distance(
                        int(truck.node), int(task.demand_node)
                    )
                )
                if not np.isfinite(road):
                    continue
                option = (float(ratio), float(road), truck_id, task_id)
                if best is None or option < best:
                    best = option
            if best is not None:
                candidates.append(best)
        if not candidates:
            return
        _ratio, road, truck_id, task_id = min(candidates)
        if self._transfer_routine_contract_to_truck(
            env,
            str(task_id),
            str(truck_id),
            goals,
            planned_road_distance_m=float(road),
        ):
            self.b_orphaned_routine_rescue_count += 1
            self._stay_reason_by_agent[str(truck_id)] = "b_orphaned_routine_rescue"
            self._feedback.append(
                PlannerFeedback(
                    step=step_now,
                    reason="b_orphaned_routine_rescue",
                    truck_id=str(truck_id),
                    task_id=str(task_id),
                    detail=f"lifeline_ratio={float(_ratio):.3f},road_m={float(road):.1f}",
                    suffix_repair_required=False,
                )
            )

    def _routine_dispatch_emergency_hard_risk(self, env) -> bool:
        """Return whether a routine dispatch would endanger emergency work.

        A valid, already-airborne/delivering emergency contract is not treated
        as a missing route merely because its truck stop has advanced past the
        route cursor.  Only a missing contract, a non-executable unlaunched
        contract, or a planned completion inside the reserve blocks dispatch.
        """
        if self.plan is None:
            return True
        reserve_steps = int(
            max(
                getattr(
                    env.cfg,
                    "hrl_route_plan_idle_routine_dispatch_emergency_reserve_steps",
                    12,
                ),
                0,
            )
        )
        step_now = int(getattr(env.state, "step_index", 0))
        active_sorties = {
            str(value)
            for value in dict(
                getattr(env, "_uav_sortie_contract_task", {})
            ).values()
            if value is not None
        }

        balanced_watchdog = bool(
            er_hlns_balanced_all_tasks_active(env)
            and getattr(
                env.cfg,
                "hrl_route_plan_balanced_all_tasks_enabled",
                False,
            )
        )

        def _safe_pending_normal_route(candidate_route: Optional[ClusterRoute], candidate_task_id: str) -> bool:
            """Allow the aggressive pilot to re-use a stalled normal suffix.

            A route whose executable head is another unstarted NORMAL task is
            still safe to re-auction: no service/claim/UAV contract is
            pre-empted, and the displaced stop remains in the route suffix.
            Formal and prior candidate modes retain the old idle-only gate.
            """
            current = candidate_route.current(env) if candidate_route is not None else None
            if current is None:
                return True
            if not balanced_watchdog or str(current.task_id) == str(candidate_task_id):
                return False
            current_task = env.state.tasks.get(str(current.task_id), None)
            return bool(
                current_task is not None
                and current_task.kind == TaskKind.NORMAL
                and not self._is_relay(current_task)
                and current_task.status == TaskStatus.PENDING
                and getattr(current_task, "first_service_step", None) is None
                and float(max(getattr(current_task, "fulfilled_mass_kg", 0.0), 0.0)) <= 1e-9
                and int(max(getattr(current_task, "service_remaining", 0), 0)) <= 0
                and not getattr(current_task, "assigned_to", None)
                and not getattr(current_task, "in_service_by", None)
                and not tuple(getattr(current_task, "route_contract_uav_ids", ()) or ())
            )
        for emergency in sorted(
            env.state.tasks.values(), key=lambda item: str(item.task_id)
        ):
            if emergency.kind != TaskKind.EMERGENCY or not self._active(emergency):
                continue
            task_id = str(emergency.task_id)
            contract = self.plan.contracts.get(task_id, None)
            if contract is None:
                return True
            if (
                getattr(emergency, "in_service_by", None) is not None
                and int(max(getattr(emergency, "service_remaining", 0), 0)) > 0
            ):
                # Physical service has already started; no upper-layer route
                # takeover is allowed or needed.
                continue
            stop = None
            for route in self.plan.routes.values():
                for candidate_stop in route.stops[int(route.cursor) :]:
                    if str(candidate_stop.task_id) == task_id:
                        stop = candidate_stop
                        break
                if stop is not None:
                    break
            # A loaded/airborne UAV contract remains executable after its
            # truck route cursor moves past the launch stop.
            sortie_active = task_id in active_sorties and bool(
                tuple(getattr(contract, "uav_ids", ()) or ())
            )
            if stop is None and not sortie_active:
                return True
            planned_eta = int(
                max(
                    getattr(stop, "eta_step", step_now) if stop is not None else step_now,
                    step_now,
                )
            )
            if int(self._effective_deadline_step(env, emergency) - planned_eta) <= reserve_steps:
                return True
        return False

    def _routine_service_start_rescue(
        self,
        env,
        goals: Dict[str, Optional[str]],
    ) -> None:
        """Repair one pending NORMAL task that never entered service.

        The repair is candidate-only and bounded per task.  A stalled owner
        keeps the existing direct route stop but receives a freshly published
        direct goal after stale routine assist metadata is removed.  An
        alternate truck is considered only when the owner is unavailable and
        the alternate has no emergency/support/recovery commitment.
        """
        if self.plan is None or not bool(
            getattr(env.cfg, "hrl_route_plan_routine_service_start_rescue_enabled", False)
        ) or not er_hlns_idle_routine_dispatch_active(env):
            return
        if self._routine_dispatch_emergency_hard_risk(env):
            self.routine_service_start_rescue_emergency_block_count += 1
            if er_hlns_balanced_all_tasks_active(env):
                self.balanced_all_tasks_watchdog_block_count += 1
            return
        step_now = int(getattr(env.state, "step_index", 0))
        stall_steps = int(
            max(
                getattr(
                    env.cfg,
                    "hrl_route_plan_routine_service_start_rescue_stall_steps",
                    10,
                ),
                1,
            )
        )
        near_distance = float(
            max(
                getattr(
                    env.cfg,
                    "hrl_route_plan_routine_service_start_rescue_near_distance_m",
                    300.0,
                ),
                0.0,
            )
        )
        max_transfers = int(
            max(
                getattr(
                    env.cfg,
                    "hrl_route_plan_routine_service_start_rescue_max_transfers",
                    1,
                ),
                0,
            )
        )
        if max_transfers <= 0:
            return
        servicing = {
            str(agent_id)
            for agent_id in (
                env._servicing_agents() if hasattr(env, "_servicing_agents") else ()
            )
        }
        emergency_contract_trucks = {
            str(contract.truck_id)
            for task_id, contract in self.plan.contracts.items()
            if (
                env.state.tasks.get(str(task_id), None) is not None
                and env.state.tasks[str(task_id)].kind == TaskKind.EMERGENCY
                and self._active(env.state.tasks[str(task_id)])
            )
        }
        active_sorties = {
            str(value)
            for value in dict(
                getattr(env, "_uav_sortie_contract_task", {})
            ).values()
            if value is not None
        }

        for task in sorted(
            env.state.tasks.values(), key=lambda item: str(item.task_id)
        ):
            if (
                task.kind != TaskKind.NORMAL
                or task.status != TaskStatus.PENDING
                or self._is_relay(task)
                or getattr(task, "first_service_step", None) is not None
                or float(max(getattr(task, "fulfilled_mass_kg", 0.0), 0.0)) > 1e-9
                or int(max(getattr(task, "service_remaining", 0), 0)) > 0
                or getattr(task, "assigned_to", None) is not None
                or getattr(task, "in_service_by", None) is not None
            ):
                continue
            task_id = str(task.task_id)
            self.routine_service_start_rescue_candidate_count += 1
            if balanced_watchdog:
                self.balanced_all_tasks_watchdog_candidate_count += 1
            if int(self._routine_service_start_rescue_count_by_task.get(task_id, 0)) >= max_transfers:
                continue
            if task_id in active_sorties or tuple(getattr(task, "route_contract_uav_ids", ()) or ()):
                self.routine_service_start_rescue_reject_active_count += 1
                continue
            contract = self.plan.contracts.get(task_id, None)
            if contract is None or tuple(getattr(contract, "uav_ids", ()) or ()):
                self.routine_service_start_rescue_reject_active_count += 1
                continue
            owner_id = str(
                getattr(task, "route_contract_truck", "")
                or getattr(contract, "truck_id", "")
                or ""
            )
            owner = env.state.agents.get(owner_id, None)
            owner_route = self.plan.routes.get(owner_id, None)
            owner_assist = self._assist_by_truck.get(owner_id, None)
            owner_busy_emergency = bool(
                owner_id in emergency_contract_trucks
                or owner_id in servicing
                or (
                    hasattr(env, "_truck_has_assigned_airborne_hard_recovery_request")
                    and bool(
                        env._truck_has_assigned_airborne_hard_recovery_request(owner_id)
                    )
                )
            )
            owner_available = bool(
                owner is not None
                and owner_route is not None
                and not bool(getattr(owner, "crashed", False))
                and owner.node is not None
                and not owner_busy_emergency
            )
            owner_stop = None
            if owner_route is not None:
                for candidate_stop in owner_route.stops[int(owner_route.cursor) :]:
                    if str(candidate_stop.task_id) == task_id:
                        owner_stop = candidate_stop
                        break
            if owner_available and owner_stop is not None:
                road = float(
                    env._decision_shortest_path_distance(
                        int(owner.node), int(task.demand_node)
                    )
                )
                if np.isfinite(road):
                    transit = getattr(owner, "transit", None)
                    if transit is None:
                        position_key = (
                            "node",
                            getattr(owner, "node", None),
                            int(getattr(owner_route, "cursor", 0)),
                            str(goals.get(owner_id, "") or ""),
                        )
                    else:
                        try:
                            transit_key = tuple(round(float(x), 3) for x in transit)
                        except Exception:
                            transit_key = (str(transit),)
                        position_key = (
                            "transit",
                            transit_key,
                            int(getattr(owner_route, "cursor", 0)),
                            str(goals.get(owner_id, "") or ""),
                        )
                    record = self._routine_service_start_rescue_progress.get(task_id)
                    if record is None or str(record.get("owner_id", "")) != owner_id:
                        self._routine_service_start_rescue_progress[task_id] = {
                            "owner_id": owner_id,
                            "last_progress_step": int(step_now),
                            "position_key": position_key,
                        }
                        continue
                    if record.get("position_key") != position_key:
                        record["position_key"] = position_key
                        record["last_progress_step"] = int(step_now)
                        continue
                    stalled = step_now - int(record.get("last_progress_step", step_now)) >= stall_steps
                    near = road <= near_distance + 1e-9
                    if stalled or near:
                        # Candidate-only owner handoff: if the assigned truck
                        # has genuinely stalled, let one idle, stocked truck
                        # take over only when it is materially faster.  This
                        # targets the dominant assigned-but-never-serviced
                        # failure mode without pre-empting emergency work.
                        allow_owner_transfer = bool(
                            getattr(
                                env.cfg,
                                "hrl_route_plan_routine_service_start_rescue_allow_stalled_owner_transfer",
                                False,
                            )
                        )
                        if stalled and allow_owner_transfer:
                            min_gain_steps = float(
                                max(
                                    getattr(
                                        env.cfg,
                                        "hrl_route_plan_routine_service_start_rescue_transfer_min_gain_steps",
                                        3.0,
                                    ),
                                    0.0,
                                )
                            )
                            min_gain_ratio = float(
                                np.clip(
                                    getattr(
                                        env.cfg,
                                        "hrl_route_plan_routine_service_start_rescue_transfer_min_gain_ratio",
                                        0.20,
                                    ),
                                    0.0,
                                    1.0,
                                )
                            )
                            speed = float(
                                max(getattr(env.cfg, "truck_speed_mps", 10.0), 1e-6)
                            )
                            dt = float(max(getattr(env.cfg, "dt_seconds", 20.0), 1e-6))
                            alternate = None
                            for candidate_id in self._live_trucks(env):
                                candidate_id = str(candidate_id)
                                if candidate_id == owner_id:
                                    continue
                                candidate = env.state.agents.get(candidate_id, None)
                                candidate_route = self.plan.routes.get(candidate_id, None)
                                if (
                                    candidate is None
                                    or candidate_route is None
                                    or candidate.node is None
                                    or bool(getattr(candidate, "crashed", False))
                                    or getattr(candidate, "transit", None) is not None
                                    or goals.get(candidate_id) is not None
                                    or not _safe_pending_normal_route(candidate_route, task_id)
                                    or candidate_id in self._assist_by_truck
                                    or candidate_id in emergency_contract_trucks
                                    or candidate_id in servicing
                                    or float(max(getattr(candidate, "bulk_inventory_kg_current", 0.0), 0.0)) + 1e-9 < self._remaining_demand_kg(task)
                                ):
                                    continue
                                if hasattr(env, "_truck_has_assigned_airborne_hard_recovery_request") and bool(
                                    env._truck_has_assigned_airborne_hard_recovery_request(candidate_id)
                                ):
                                    continue
                                candidate_road = float(
                                    env._decision_shortest_path_distance(
                                        int(candidate.node), int(task.demand_node)
                                    )
                                )
                                if not np.isfinite(candidate_road):
                                    continue
                                owner_steps = float(np.ceil((road / speed) / dt))
                                candidate_steps = float(np.ceil((candidate_road / speed) / dt))
                                required_gain = max(min_gain_steps, min_gain_ratio * owner_steps)
                                if owner_steps - candidate_steps < required_gain - 1e-9:
                                    continue
                                option = (candidate_steps, candidate_road, candidate_id)
                                if alternate is None or option < alternate:
                                    alternate = option
                            if alternate is not None:
                                _, candidate_road, new_owner = alternate
                                if self._transfer_routine_contract_to_truck(
                                    env,
                                    task_id,
                                    str(new_owner),
                                    goals,
                                    planned_road_distance_m=float(candidate_road),
                                ):
                                    self._routine_service_start_rescue_count_by_task[task_id] = int(
                                        self._routine_service_start_rescue_count_by_task.get(task_id, 0) + 1
                                    )
                                    self._routine_service_start_rescue_progress.pop(task_id, None)
                                    self.routine_service_start_rescue_success_count += 1
                                    self.routine_service_start_rescue_alternate_count += 1
                                    self.balanced_all_tasks_watchdog_transfer_count += (
                                        1 if balanced_watchdog else 0
                                    )
                                    self._feedback.append(
                                        PlannerFeedback(
                                            step=step_now,
                                            reason="routine_service_start_rescue",
                                            truck_id=str(new_owner),
                                            task_id=task_id,
                                            detail=f"owner=stalled_alternate,old_owner={owner_id},road_m={candidate_road:.1f}",
                                            suffix_repair_required=False,
                                        )
                                    )
                                    return
                        if owner_assist is not None:
                            assist_task = str(owner_assist.get("task_id", "") or "")
                            service_mode = str(owner_assist.get("service_mode", "") or "").upper()
                            if assist_task not in {"", task_id} and service_mode not in {
                                "DIRECT", "NORMAL_SERVICE", "ROUTINE_SERVICE"
                            }:
                                self.routine_service_start_rescue_reject_active_count += 1
                                continue
                        for agent_id, goal_id in list(goals.items()):
                            if str(agent_id) != owner_id and str(goal_id) == task_id:
                                goals[str(agent_id)] = None
                        self._assist_by_truck.pop(owner_id, None)
                        goals[owner_id] = task_id
                        self._stay_reason_by_agent[owner_id] = "routine_service_start_rescue"
                        self._routine_service_start_rescue_count_by_task[task_id] = int(
                            self._routine_service_start_rescue_count_by_task.get(task_id, 0) + 1
                        )
                        self._routine_service_start_rescue_progress.pop(task_id, None)
                        self.routine_service_start_rescue_success_count += 1
                        self._feedback.append(
                            PlannerFeedback(
                                step=step_now,
                                reason="routine_service_start_rescue",
                                truck_id=owner_id,
                                task_id=task_id,
                                detail=f"owner=direct,stalled={int(stalled)},near={int(near)},road_m={road:.1f}",
                                suffix_repair_required=False,
                            )
                        )
                        return
            # Owner is missing/unavailable.  Rehome only to a safe alternate;
            # never steal a truck carrying emergency/support/recovery work.
            if owner_available:
                self.routine_service_start_rescue_reject_no_owner_count += 1
                continue
            alternate = None
            for candidate_id in self._live_trucks(env):
                candidate = env.state.agents.get(str(candidate_id), None)
                route = self.plan.routes.get(str(candidate_id), None)
                if (
                    candidate is None
                    or route is None
                    or candidate.node is None
                    or bool(getattr(candidate, "crashed", False))
                    or getattr(candidate, "transit", None) is not None
                    or goals.get(str(candidate_id)) is not None
                    or not _safe_pending_normal_route(route, task_id)
                    or str(candidate_id) in self._assist_by_truck
                    or str(candidate_id) in emergency_contract_trucks
                    or str(candidate_id) in servicing
                    or float(max(getattr(candidate, "bulk_inventory_kg_current", 0.0), 0.0))
                    + 1e-9
                    < self._remaining_demand_kg(task)
                ):
                    continue
                if hasattr(env, "_truck_has_assigned_airborne_hard_recovery_request") and bool(
                    env._truck_has_assigned_airborne_hard_recovery_request(str(candidate_id))
                ):
                    continue
                road = float(
                    env._decision_shortest_path_distance(
                        int(candidate.node), int(task.demand_node)
                    )
                )
                if np.isfinite(road):
                    option = (road, str(candidate_id))
                    if alternate is None or option < alternate:
                        alternate = option
            if alternate is None:
                self.routine_service_start_rescue_reject_no_alternate_count += 1
                continue
            road, new_owner = alternate
            if not self._transfer_routine_contract_to_truck(
                env,
                task_id,
                str(new_owner),
                goals,
                planned_road_distance_m=float(road),
            ):
                self.routine_service_start_rescue_reject_no_alternate_count += 1
                continue
            self._routine_service_start_rescue_count_by_task[task_id] = int(
                self._routine_service_start_rescue_count_by_task.get(task_id, 0) + 1
            )
            self.routine_service_start_rescue_success_count += 1
            self.routine_service_start_rescue_alternate_count += 1
            self.balanced_all_tasks_watchdog_transfer_count += (
                1 if balanced_watchdog else 0
            )
            self._feedback.append(
                PlannerFeedback(
                    step=step_now,
                    reason="routine_service_start_rescue",
                    truck_id=str(new_owner),
                    task_id=task_id,
                    detail=f"owner=alternate,old_owner={owner_id},road_m={road:.1f}",
                    suffix_repair_required=False,
                )
            )
            return

    def _idle_routine_dispatch(
        self,
        env,
        goals: Dict[str, Optional[str]],
    ) -> None:
        """Bind one safe pending NORMAL task to an otherwise idle truck.

        This is a candidate-only upper-planning capability.  It does not
        release a contract or displace an active route: the task must remain
        ``PENDING`` and unclaimed, while the receiving truck has no active
        route stop, assist/recovery request, or emergency contract.  Dispatch
        is suppressed whenever an active emergency has no executable contract
        or its planned completion is within the configured deadline reserve.
        """
        if self.plan is None or not bool(
            getattr(env.cfg, "hrl_route_plan_idle_routine_dispatch_enabled", False)
        ) or not er_hlns_idle_routine_dispatch_active(env):
            return

        step_now = int(getattr(env.state, "step_index", 0))
        max_dispatch = int(
            max(
                getattr(
                    env.cfg,
                    "hrl_route_plan_idle_routine_dispatch_max_per_step",
                    1,
                ),
                0,
            )
        )
        if max_dispatch <= 0:
            return

        if self._routine_dispatch_emergency_hard_risk(env):
            self.idle_routine_dispatch_emergency_block_count += 1
            return

        servicing = {
            str(agent_id)
            for agent_id in (
                env._servicing_agents() if hasattr(env, "_servicing_agents") else ()
            )
        }
        emergency_contract_trucks = {
            str(contract.truck_id)
            for task_id, contract in self.plan.contracts.items()
            if (
                env.state.tasks.get(str(task_id), None) is not None
                and env.state.tasks[str(task_id)].kind == TaskKind.EMERGENCY
                and self._active(env.state.tasks[str(task_id)])
            )
        }
        best: Optional[Tuple[int, int, str, str, float]] = None
        speed = float(max(getattr(env.cfg, "truck_speed_mps", 10.0), 1e-6))
        dt = float(max(getattr(env.cfg, "dt_seconds", 1.0), 1e-6))
        unload = int(max(getattr(env.cfg, "unload_rounds_normal", 1), 1))

        for task in sorted(
            env.state.tasks.values(), key=lambda item: str(item.task_id)
        ):
            if (
                task.kind != TaskKind.NORMAL
                or not self._active(task)
                or task.status != TaskStatus.PENDING
                or self._is_relay(task)
            ):
                continue
            task_id = str(task.task_id)
            self.idle_routine_dispatch_candidate_count += 1

            # Never pre-empt an owned/claimed task or an airborne contract.
            if (
                getattr(task, "assigned_to", None) is not None
                or str(getattr(task, "in_service_by", "") or "")
                or str(getattr(task, "route_contract_owner", "") or "")
                or task_id in {
                    str(value)
                    for value in dict(
                        getattr(env, "_uav_sortie_contract_task", {})
                    ).values()
                    if value is not None
                }
                or self.plan.contracts.get(task_id, None) is not None
            ):
                self.idle_routine_dispatch_reject_contract_count += 1
                continue

            for truck_id in self._live_trucks(env):
                truck = env.state.agents.get(str(truck_id), None)
                route = self.plan.routes.get(str(truck_id), None)
                if truck is None or route is None or truck.node is None:
                    self.idle_routine_dispatch_reject_no_idle_truck_count += 1
                    continue
                if (
                    goals.get(str(truck_id)) is not None
                    or route.current(env) is not None
                    or any(
                        env.state.tasks.get(str(stop.task_id), None) is not None
                        and self._active(env.state.tasks[str(stop.task_id)])
                        for stop in route.stops[int(route.cursor) :]
                    )
                    or getattr(truck, "transit", None) is not None
                    or str(truck_id) in self._assist_by_truck
                    or str(truck_id) in servicing
                    or str(truck_id) in emergency_contract_trucks
                ):
                    self.idle_routine_dispatch_reject_no_idle_truck_count += 1
                    continue
                if hasattr(
                    env, "_truck_has_assigned_airborne_hard_recovery_request"
                ) and bool(
                    env._truck_has_assigned_airborne_hard_recovery_request(
                        str(truck_id)
                    )
                ):
                    self.idle_routine_dispatch_reject_no_idle_truck_count += 1
                    continue
                if float(
                    max(getattr(truck, "bulk_inventory_kg_current", 0.0), 0.0)
                ) + 1e-9 < self._remaining_demand_kg(task):
                    self.idle_routine_dispatch_reject_inventory_count += 1
                    continue
                road = float(
                    env._decision_shortest_path_distance(
                        int(truck.node), int(task.demand_node)
                    )
                )
                if not np.isfinite(road):
                    self.idle_routine_dispatch_reject_unreachable_count += 1
                    continue
                eta = int(
                    step_now
                    + np.ceil(road / max(speed * dt, 1e-6))
                    + unload
                )
                slack = int(self._effective_deadline_step(env, task) - eta)
                if slack < 0:
                    self.idle_routine_dispatch_reject_deadline_count += 1
                    continue
                option = (slack, eta, task_id, str(truck_id), road)
                if best is None or option < best:
                    best = option

        if best is None:
            return
        slack, eta, task_id, truck_id, road = best
        if not self._transfer_routine_contract_to_truck(
            env,
            task_id,
            truck_id,
            goals,
            eta_step=int(eta),
            planned_road_distance_m=float(road),
        ):
            self.idle_routine_dispatch_reject_contract_count += 1
            return
        self.idle_routine_dispatch_success_count += 1
        self._stay_reason_by_agent[str(truck_id)] = "idle_routine_dispatch"
        self._feedback.append(
            PlannerFeedback(
                step=step_now,
                reason="idle_routine_dispatch",
                truck_id=str(truck_id),
                task_id=str(task_id),
                detail=f"deadline_slack={int(slack)},eta={int(eta)},road_m={float(road):.1f}",
                suffix_repair_required=False,
            )
        )

    def _r4_stalled_routine_takeovers(
        self,
        env,
        goals: Dict[str, Optional[str]],
    ) -> None:
        """Perform one bounded takeover of a stalled, unstarted NORMAL task.

        R4 is candidate-only and deliberately narrower than the existing
        dynamic/risk-slack repairs.  It only touches a task that is still
        ``PENDING`` with no service progress and whose existing route owner
        has made no executable progress for the configured window.  The
        receiving truck must be idle, stocked and road-reachable; active
        emergency/support/recovery work is never pre-empted.  No UAV or
        weather gate is relaxed because this operation is truck-only.
        """
        if self.plan is None or not bool(
            getattr(env.cfg, "hrl_route_plan_r4_routine_takeover_enabled", False)
        ) or not er_hlns_r4_routine_takeover_active(env):
            return

        step_now = int(getattr(env.state, "step_index", 0))
        stall_steps = int(
            max(
                getattr(env.cfg, "hrl_route_plan_r4_routine_takeover_stall_steps", 12),
                1,
            )
        )
        max_transfers = int(
            max(
                getattr(env.cfg, "hrl_route_plan_r4_routine_takeover_max_transfers", 1),
                0,
            )
        )
        radius_m = float(
            max(
                getattr(env.cfg, "hrl_route_plan_r4_routine_takeover_radius_m", 0.0),
                0.0,
            )
        )
        speed = float(max(getattr(env.cfg, "truck_speed_mps", 10.0), 1e-6))
        dt = float(max(getattr(env.cfg, "dt_seconds", 1.0), 1e-6))
        unload = int(max(getattr(env.cfg, "unload_rounds_normal", 1), 1))
        sortie_tasks = {
            str(task_id)
            for task_id in dict(getattr(env, "_uav_sortie_contract_task", {})).values()
            if task_id is not None
        }
        servicing = {
            str(agent_id)
            for agent_id in (
                env._servicing_agents() if hasattr(env, "_servicing_agents") else ()
            )
        }

        for task in sorted(env.state.tasks.values(), key=lambda item: str(item.task_id)):
            if task.kind != TaskKind.NORMAL or self._is_relay(task) or not self._active(task):
                continue
            task_id = str(task.task_id)
            self.r4_routine_takeover_candidate_count += 1
            if task.status != TaskStatus.PENDING or getattr(task, "first_service_step", None) is not None:
                self.r4_routine_takeover_reject_started_count += 1
                continue
            if float(max(getattr(task, "fulfilled_mass_kg", 0.0), 0.0)) > 1e-9 or int(
                max(getattr(task, "service_remaining", 0), 0)
            ) > 0:
                self.r4_routine_takeover_reject_started_count += 1
                continue
            if int(self._r4_routine_takeover_count_by_task.get(task_id, 0)) >= max_transfers:
                continue

            contract = self.plan.contracts.get(task_id, None)
            owner_id = str(
                getattr(task, "route_contract_truck", "")
                or (getattr(contract, "truck_id", "") if contract is not None else "")
                or ""
            )
            owner_route = self.plan.routes.get(owner_id, None)
            if contract is None or not owner_id or owner_route is None:
                self.r4_routine_takeover_reject_contract_count += 1
                continue
            assigned_to = getattr(task, "assigned_to", None)
            if assigned_to is not None and str(assigned_to) != owner_id:
                self.r4_routine_takeover_reject_contract_count += 1
                continue
            # A NORMAL contract must never carry an emergency UAV sortie.
            if tuple(getattr(contract, "uav_ids", ()) or ()) or task_id in sortie_tasks:
                self.r4_routine_takeover_reject_safety_count += 1
                continue
            task_stop_present = any(
                str(stop.task_id) == task_id
                for route in self.plan.routes.values()
                for stop in route.stops[int(route.cursor) :]
            )
            if not task_stop_present:
                self.r4_routine_takeover_reject_contract_count += 1
                continue

            # A task at the owner's route head is exactly the failure mode R4
            # is meant to diagnose: it can remain PENDING while the truck is
            # held by an emergency/support contract.  Do not reject it merely
            # because it is the current goal.  Instead measure executable
            # progress from the owner's physical cursor and transit state.
            owner_state = env.state.agents.get(owner_id, None)
            if owner_state is None or bool(getattr(owner_state, "crashed", False)):
                self.r4_routine_takeover_reject_owner_active_count += 1
                continue
            transit = getattr(owner_state, "transit", None)
            if transit is None:
                position_key = (
                    "node",
                    getattr(owner_state, "node", None),
                    int(getattr(owner_route, "cursor", 0)),
                    str(goals.get(owner_id, "") or ""),
                )
            else:
                try:
                    transit_key = tuple(round(float(x), 3) for x in transit)
                except Exception:
                    transit_key = (str(transit),)
                position_key = (
                    "transit",
                    transit_key,
                    int(getattr(owner_route, "cursor", 0)),
                    str(goals.get(owner_id, "") or ""),
                )
            record = self._r4_routine_takeover_progress.get(task_id, None)
            if record is None or str(record.get("owner_id", "")) != owner_id:
                self._r4_routine_takeover_progress[task_id] = {
                    "owner_id": owner_id,
                    "last_progress_step": int(step_now),
                    "position_key": position_key,
                }
                continue
            if record.get("position_key") != position_key:
                record["position_key"] = position_key
                record["last_progress_step"] = int(step_now)
                continue
            if step_now - int(record.get("last_progress_step", step_now)) < stall_steps:
                continue
            self.r4_routine_takeover_trigger_count += 1

            best: Optional[Tuple[float, float, str]] = None
            saw_idle = False
            for candidate_id in self._live_trucks(env):
                candidate_id = str(candidate_id)
                if candidate_id == owner_id:
                    continue
                candidate = env.state.agents.get(candidate_id, None)
                candidate_route = self.plan.routes.get(candidate_id, None)
                if candidate is None or bool(getattr(candidate, "crashed", False)):
                    self.r4_routine_takeover_reject_safety_count += 1
                    continue
                if candidate.node is None or candidate.transit is not None:
                    self.r4_routine_takeover_reject_no_idle_truck_count += 1
                    continue
                candidate_goal = goals.get(candidate_id, None)
                current_stop = candidate_route.current(env) if candidate_route is not None else None
                if candidate_goal is not None or current_stop is not None:
                    self.r4_routine_takeover_reject_no_idle_truck_count += 1
                    continue
                if candidate_id in self._assist_by_truck or candidate_id in servicing:
                    self.r4_routine_takeover_reject_safety_count += 1
                    continue
                if hasattr(env, "_truck_has_assigned_airborne_hard_recovery_request") and bool(
                    env._truck_has_assigned_airborne_hard_recovery_request(candidate_id)
                ):
                    self.r4_routine_takeover_reject_safety_count += 1
                    continue
                if any(
                    item.kind == TaskKind.EMERGENCY
                    and self._active(item)
                    and str(getattr(item, "route_contract_truck", "") or "") == candidate_id
                    for item in env.state.tasks.values()
                ):
                    self.r4_routine_takeover_reject_safety_count += 1
                    continue
                saw_idle = True
                inventory = float(max(getattr(candidate, "bulk_inventory_kg_current", 0.0), 0.0))
                if inventory + 1e-9 < self._remaining_demand_kg(task):
                    self.r4_routine_takeover_reject_inventory_count += 1
                    continue
                road = float(
                    env._decision_shortest_path_distance(
                        int(candidate.node), int(task.demand_node)
                    )
                )
                if not np.isfinite(road):
                    self.r4_routine_takeover_reject_unreachable_count += 1
                    continue
                if radius_m > 0.0 and road > radius_m + 1e-9:
                    self.r4_routine_takeover_reject_unreachable_count += 1
                    continue
                eta = float(step_now + np.ceil((road / speed) / dt) + unload)
                if eta > float(self._effective_deadline_step(env, task)) + 1e-9:
                    self.r4_routine_takeover_reject_deadline_count += 1
                    continue
                option = (eta, road, candidate_id)
                if best is None or option < best:
                    best = option

            if best is None:
                if not saw_idle:
                    self.r4_routine_takeover_reject_no_idle_truck_count += 1
                continue
            eta, road, new_owner = best
            if not self._transfer_routine_contract_to_truck(
                env,
                task_id,
                str(new_owner),
                goals,
                eta_step=int(eta),
                planned_road_distance_m=float(road),
            ):
                self.r4_routine_takeover_reject_contract_count += 1
                continue
            self._r4_routine_takeover_count_by_task[task_id] = int(
                self._r4_routine_takeover_count_by_task.get(task_id, 0) + 1
            )
            self._r4_routine_takeover_progress.pop(task_id, None)
            self.r4_routine_takeover_success_count += 1
            self._stay_reason_by_agent[str(new_owner)] = "r4_stalled_routine_takeover"
            self._feedback.append(
                PlannerFeedback(
                    step=step_now,
                    reason="r4_stalled_routine_takeover",
                    truck_id=str(new_owner),
                    task_id=task_id,
                    detail=f"old_owner={owner_id},eta={eta:.1f},road_m={road:.1f}",
                    suffix_repair_required=False,
                )
            )
            # At most one atomic takeover per planning call keeps the
            # candidate bounded and prevents a full auction from replacing
            # the emergency skeleton.
            break

    def _apply_global_pending_reauction_control(
        self,
        env,
        prior_contracts: Dict[str, TaskContract],
    ) -> Dict[str, TaskContract]:
        """Drop only unresolved contract history for the global-repair control."""
        if not bool(
            getattr(
                env.cfg,
                "hrl_route_plan_global_pending_reauction_on_repair_enabled",
                False,
            )
        ):
            return dict(prior_contracts)
        sortie_task_ids = {
            str(task_id)
            for task_id in dict(
                getattr(env, "_uav_sortie_contract_task", {}) or {}
            ).values()
            if task_id is not None
        }
        protected_task_ids = set(sortie_task_ids)
        for task_id, contract in prior_contracts.items():
            task = env.state.tasks.get(str(task_id), None)
            owner = env.state.agents.get(
                str(getattr(contract, "owner_agent_id", "")), None
            )
            uav = env.state.agents.get(
                str(getattr(contract, "uav_id", "")), None
            )
            if (
                task is not None
                and (
                    task.status == TaskStatus.CLAIMED
                    or getattr(task, "in_service_by", None) is not None
                )
            ) or (
                owner is not None
                and getattr(owner, "follow_target", "__truck__") is None
            ) or (
                uav is not None
                and getattr(uav, "follow_target", "__truck__") is None
            ):
                protected_task_ids.add(str(task_id))
        return {
            task_id: contract
            for task_id, contract in prior_contracts.items()
            if str(task_id) in protected_task_ids
        }

    def plan_or_repair(self, env) -> Dict[str, Optional[str]]:
        step_now = int(env.state.step_index)
        if self._episode_changed(env):
            self.reset()
            self._episode_token = (
                id(env.state.tasks),
                tuple(sorted(str(task_id) for task_id in env.state.tasks)),
            )
        road_signature = self._road_signature(env)
        truck_ids = self._live_trucks(env)
        self._mark_service_modes(env, truck_ids)

        completed_task_count = int(
            sum(
                1
                for task in env.state.tasks.values()
                if task.status == TaskStatus.DELIVERED
            )
        )
        if completed_task_count > int(self._last_completed_task_count):
            self._last_completed_task_count = int(completed_task_count)
            self._last_completion_progress_step = int(step_now)

        if self.plan is None:
            self._install_plan(
                env,
                reason="initial_plan",
                road_signature=road_signature,
                prior_contracts={},
            )
            self._promote_v5_launch_first_emergencies(env)
            self._apply_onsite_emergency_takeovers(env)
            self._promote_safe_at_risk_emergency_suffixes(env, road_signature)
        else:
            self._apply_onsite_emergency_takeovers(env)
            self._promote_safe_at_risk_emergency_suffixes(env, road_signature)
            feedback = self._current_route_feedback(env, road_signature)
            active_emergency_count = int(
                sum(
                    1
                    for task in env.state.tasks.values()
                    if task.kind == TaskKind.EMERGENCY and self._active(task)
                )
            )
            starvation_steps = int(
                max(
                    getattr(
                        env.cfg,
                        "hrl_route_plan_queue_starvation_steps",
                        60,
                    ),
                    1,
                )
            )
            starvation_min_pending = int(
                max(
                    getattr(
                        env.cfg,
                        "hrl_route_plan_queue_starvation_min_pending",
                        1,
                    ),
                    1,
                )
            )
            if (
                self._targeted_repairs_enabled()
                and not self._queue_starvation_repair_done
                and not self._queue_starvation_repair_pending
                and active_emergency_count >= starvation_min_pending
                and step_now - int(self._last_completion_progress_step)
                >= starvation_steps
            ):
                self._queue_starvation_repair_pending = True
            if self._queue_starvation_repair_pending:
                feedback.append(
                    PlannerFeedback(
                        step=int(step_now),
                        reason="route_queue_starvation",
                        detail=(
                            f"no_completion_steps="
                            f"{step_now - int(self._last_completion_progress_step)},"
                            f"active_emergency={active_emergency_count}"
                        ),
                        # Starvation alone is an observation, not permission
                        # to rebuild every route.  A global repair is allowed
                        # only when the checks below find a missing route or a
                        # materially better feasible execution unit.  This
                        # preserves routine work when no useful TC repair
                        # exists (the previous unconditional refresh could
                        # lose a task without completing another emergency).
                        suffix_repair_required=False,
                    )
                )
            self._feedback.extend(feedback)
            stockout_release_task_ids = (
                self._stockout_emergency_contracts_for_transfer(
                    env, road_signature
                )
            )
            stalled_release_task_ids = self._stalled_emergency_contracts_for_transfer(
                env, road_signature
            )
            self.stalled_contract_transfer_candidate_count += int(
                len(stalled_release_task_ids)
            )
            release_task_ids = dict(stalled_release_task_ids)
            for task_id, old_truck_id in stockout_release_task_ids.items():
                release_task_ids.setdefault(str(task_id), str(old_truck_id))
            failed_queue_rescue_task_ids = dict(
                self._queue_rescue_failed_task_ids
            )
            for task_id, old_truck_id in failed_queue_rescue_task_ids.items():
                task = env.state.tasks.get(str(task_id), None)
                if task is not None and self._active(task):
                    release_task_ids.setdefault(str(task_id), str(old_truck_id))
            starvation_release_task_ids: Dict[str, str] = {}
            # The cross-truck release experiment worsened protected seeds and
            # did not improve the target stalled seed.  Preserve it for audit,
            # but keep the stable fixed-truck contracts in the active method.
            enable_starvation_contract_release_experiment = False
            if (
                enable_starvation_contract_release_experiment
                and self._queue_starvation_repair_pending
                and self.plan is not None
            ):
                sortie_tasks = {
                    str(task_id)
                    for task_id in dict(
                        getattr(env, "_uav_sortie_contract_task", {})
                    ).values()
                    if task_id is not None
                }
                for task_id, contract in self.plan.contracts.items():
                    task = env.state.tasks.get(str(task_id), None)
                    if (
                        task is None
                        or task.kind != TaskKind.EMERGENCY
                        or not self._active(task)
                        or task.status == TaskStatus.CLAIMED
                        or str(task_id) in sortie_tasks
                    ):
                        continue
                    starvation_release_task_ids[str(task_id)] = str(
                        contract.truck_id
                    )
                for task_id, old_truck_id in starvation_release_task_ids.items():
                    release_task_ids.setdefault(str(task_id), str(old_truck_id))
            residual_release_task_ids = (
                self._residual_emergency_contracts_for_transfer(
                    env, road_signature
                )
            )
            routine_release_task_ids = self._inventory_infeasible_routine_contracts(
                env
            )
            risk_slack_release_task_ids = (
                self._risk_slack_routine_cross_truck_repairs(env)
            )
            normal_cleanup_release_task_ids = (
                self._stalled_normal_cleanup_contracts(env)
            )
            for task_id, old_truck_id in residual_release_task_ids.items():
                release_task_ids.setdefault(str(task_id), str(old_truck_id))
            for task_id, old_truck_id in routine_release_task_ids.items():
                release_task_ids.setdefault(str(task_id), str(old_truck_id))
            for task_id, old_truck_id in risk_slack_release_task_ids.items():
                release_task_ids.setdefault(str(task_id), str(old_truck_id))
            for task_id, old_truck_id in normal_cleanup_release_task_ids.items():
                release_task_ids.setdefault(str(task_id), str(old_truck_id))
            deadline_release_task_ids = (
                self._deadline_risk_emergency_contracts_for_transfer(
                    env, road_signature
                )
            )
            for task_id, old_truck_id in deadline_release_task_ids.items():
                release_task_ids.setdefault(str(task_id), str(old_truck_id))
            active_ids = {
                str(task.task_id)
                for task in env.state.tasks.values()
                if self._active(task)
            }
            contracted_active = {
                task_id
                for task_id, contract in self.plan.contracts.items()
                if task_id in active_ids
                and str(
                    getattr(
                        env.state.tasks.get(str(task_id), None),
                        "service_mode",
                        DIRECT,
                    )
                ).upper()
                == str(contract.service_mode).upper()
                and env.state.agents.get(str(contract.owner_agent_id), None)
                is not None
                and not bool(
                    getattr(
                        env.state.agents[str(contract.owner_agent_id)],
                        "crashed",
                        False,
                    )
                )
            }
            missing_contract_ids = active_ids.difference(contracted_active)
            missing_contract_fingerprint: Optional[Tuple[Any, ...]] = None
            if missing_contract_ids:
                missing_contract_fingerprint = self._missing_contract_fingerprint(
                    env,
                    tuple(missing_contract_ids),
                    truck_ids,
                    road_signature,
                )
                missing_contract = bool(
                    missing_contract_fingerprint
                    != self._last_missing_contract_fingerprint
                )
            else:
                missing_contract = False
                self._last_missing_contract_fingerprint = None
            no_executable_stop = bool(
                active_ids
                and all(route.current(env) is None for route in self.plan.routes.values())
            )
            repair_requested = bool(
                missing_contract
                or no_executable_stop
                or release_task_ids
                or any(item.suffix_repair_required for item in feedback)
            )
            min_interval = int(
                max(
                    getattr(
                        env.cfg, "hrl_route_plan_min_replan_interval", 3
                    ),
                    0,
                )
            )
            event_replan_enabled = bool(
                getattr(env.cfg, "hrl_route_plan_event_replan_enabled", True)
            )
            if (
                event_replan_enabled
                and repair_requested
                and step_now - self.last_replan_step >= min_interval
            ):
                if (
                    self._queue_starvation_repair_pending
                    and bool(stalled_release_task_ids)
                ):
                    reason = "route_queue_starvation"
                elif residual_release_task_ids:
                    reason = "residual_emergency_ready_uav_handoff"
                elif normal_cleanup_release_task_ids:
                    reason = "post_emergency_stalled_normal_cleanup"
                elif routine_release_task_ids:
                    reason = "routine_inventory_rebalance"
                elif risk_slack_release_task_ids:
                    reason = "risk_slack_routine_cross_truck_repair"
                elif deadline_release_task_ids:
                    reason = "deadline_infeasible_suffix_promotion"
                elif stockout_release_task_ids:
                    reason = "stockout_emergency_contract_rebuild"
                elif failed_queue_rescue_task_ids:
                    reason = "failed_queue_rescue_contract_rebuild"
                elif release_task_ids:
                    reason = "stalled_emergency_contract_transfer"
                elif missing_contract:
                    reason = "new_or_released_task"
                elif no_executable_stop:
                    reason = "no_executable_route_stop"
                else:
                    reason = "+".join(sorted({item.reason for item in feedback}))
                prior = {
                    task_id: contract
                    for task_id, contract in self.plan.contracts.items()
                    if task_id in active_ids
                    and task_id not in release_task_ids
                    and str(
                        getattr(
                            env.state.tasks.get(str(task_id), None),
                            "service_mode",
                            DIRECT,
                        )
                    ).upper()
                    == str(contract.service_mode).upper()
                }
                # Global pending-task re-auction control: remove history from unresolved work so
                # each admitted repair may globally re-auction the pending
                # set. Claimed, in-service, and airborne assignments remain
                # protected by the shared execution contract.
                prior = self._apply_global_pending_reauction_control(env, prior)
                old_transfer_trucks = {
                    task_id: str(self.plan.contracts[task_id].truck_id)
                    for task_id in release_task_ids
                    if task_id in self.plan.contracts
                }
                enable_failed_truck_exclusion_experiment = False
                self._temporary_forbidden_truck_by_task = (
                    {
                        str(task_id): str(old_truck_id)
                        for task_id, old_truck_id in old_transfer_trucks.items()
                        if (
                            str(task_id) in stalled_release_task_ids
                            or str(task_id) in risk_slack_release_task_ids
                        )
                    }
                    if (
                        enable_failed_truck_exclusion_experiment
                        or bool(risk_slack_release_task_ids)
                    )
                    else {}
                )
                try:
                    self._install_plan(
                        env,
                        reason=reason,
                        road_signature=road_signature,
                        prior_contracts=prior,
                    )
                finally:
                    self._temporary_forbidden_truck_by_task = {}
                if missing_contract_fingerprint is not None:
                    self._last_missing_contract_fingerprint = (
                        missing_contract_fingerprint
                    )
                if stalled_release_task_ids:
                    self.stalled_contract_transfer_replan_count += 1
                self._apply_onsite_emergency_takeovers(env)
                if reason == "route_queue_starvation":
                    self._queue_starvation_repair_pending = False
                    self._queue_starvation_repair_done = True
                    self.queue_starvation_repair_count += 1
                for task_id, old_truck_id in old_transfer_trucks.items():
                    new_contract = (
                        None if self.plan is None else self.plan.contracts.get(task_id, None)
                    )
                    if (
                        new_contract is not None
                        and str(new_contract.truck_id) != str(old_truck_id)
                    ):
                        # This diagnostic belongs only to the stalled-contract
                        # transfer mechanism. Residual handoff, deadline rescue,
                        # inventory rebalance and cleanup have separate counters.
                        if task_id in stalled_release_task_ids:
                            self.contract_transfer_count += 1
                        self._contract_last_transfer_step[task_id] = int(step_now)
                    # Restart the progress window even if the optimizer found
                    # that the original unit remains globally preferable.
                    self._contract_progress.pop(task_id, None)
                    self._queue_rescue_failed_task_ids.pop(str(task_id), None)

        goals = self._current_commands(env)
        self._rescue_orphaned_b_routine_goals(env, goals)
        self._last_goals = dict(goals)
        self.last_seen_step = step_now
        self._publish(env, goals)
        return goals

    @property
    def assist_by_truck(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._assist_by_truck)

    @property
    def contract_owner_by_task(self) -> Dict[str, str]:
        if self.plan is None:
            return {}
        return {
            str(task_id): str(contract.owner_agent_id)
            for task_id, contract in self.plan.contracts.items()
        }

    def audit_snapshot(self, env) -> Dict[str, Any]:
        if self.plan is None:
            return {
                "enabled": True,
                "version": 0,
                "routes": {},
                "contracts": {},
            }
        routes: Dict[str, Any] = {}
        for truck_id, route in self.plan.routes.items():
            current = route.current(env)
            routes[str(truck_id)] = {
                "cursor": int(route.cursor),
                "current_task_id": "" if current is None else str(current.task_id),
                "uav_ids": list(route.uav_ids),
                "stops": [
                    {
                        "task_id": str(stop.task_id),
                        "stop_type": str(stop.stop_type),
                        "uav_id": "" if stop.uav_id is None else str(stop.uav_id),
                        "target_node": int(stop.target_node),
                        "anchor_nodes": list(stop.anchor_nodes),
                        "selected_anchor": stop.selected_anchor,
                        "eta_step": int(stop.eta_step),
                        "deadline_step": int(stop.deadline_step),
                        "planned_road_distance_m": float(
                            stop.planned_road_distance_m
                        ),
                        "planned_air_distance_m": float(
                            stop.planned_air_distance_m
                        ),
                        "service_mode": str(stop.service_mode),
                    }
                    for stop in route.stops
                ],
            }
        return {
            "enabled": True,
            "version": int(self.plan.version),
            "created_step": int(self.plan.created_step),
            "reason": str(self.plan.reason),
            "objective": float(self.plan.objective),
            "road_version_edge_count": int(len(self.plan.road_signature)),
            "full_plan_count": int(self.full_plan_count),
            "suffix_repair_count": int(self.suffix_repair_count),
            "suffix_repair_success_count": int(
                self.suffix_repair_success_count
            ),
            "anchor_backup_switch_count": int(self.anchor_backup_switch_count),
            "contract_release_count": int(self.contract_release_count),
            "stalled_contract_transfer_candidate_count": int(
                self.stalled_contract_transfer_candidate_count
            ),
            "stalled_contract_transfer_replan_count": int(
                self.stalled_contract_transfer_replan_count
            ),
            "contract_transfer_count": int(self.contract_transfer_count),
            "onsite_takeover_count": int(self.onsite_takeover_count),
            "routine_opportunity_candidate_count": int(
                self.routine_opportunity_candidate_count
            ),
            "routine_opportunity_transfer_count": int(
                self.routine_opportunity_transfer_count
            ),
            "routine_opportunity_blocked_assist_count": int(
                self.routine_opportunity_blocked_assist_count
            ),
            "routine_opportunity_blocked_eta_count": int(
                self.routine_opportunity_blocked_eta_count
            ),
            "deadline_rescue_promotion_count": int(
                self.deadline_rescue_promotion_count
            ),
            "road_impact_emergency_promotion_candidate_count": int(
                self.road_impact_emergency_promotion_candidate_count
            ),
            "road_impact_emergency_promotion_trigger_count": int(
                self.road_impact_emergency_promotion_trigger_count
            ),
            "road_impact_emergency_promotion_count": int(
                self.road_impact_emergency_promotion_count
            ),
            "road_impact_emergency_promotion_reject_count": int(
                self.road_impact_emergency_promotion_reject_count
            ),
            "road_impact_emergency_promotion_reject_no_delta_count": int(
                self.road_impact_emergency_promotion_reject_no_delta_count
            ),
            "road_impact_emergency_promotion_reject_no_risk_count": int(
                self.road_impact_emergency_promotion_reject_no_risk_count
            ),
            "road_impact_emergency_promotion_reject_protected_count": int(
                self.road_impact_emergency_promotion_reject_protected_count
            ),
            "road_impact_emergency_promotion_reject_cooldown_count": int(
                self.road_impact_emergency_promotion_reject_cooldown_count
            ),
            "emergency_starvation_promotion_count": int(
                self.emergency_starvation_promotion_count
            ),
            "emergency_launch_watchdog_ready_count": int(
                self.emergency_launch_watchdog_ready_count
            ),
            "emergency_launch_watchdog_force_count": int(
                self.emergency_launch_watchdog_force_count
            ),
            "direct_safe_secondary_emergency_candidate_count": int(
                self.direct_safe_secondary_emergency_candidate_count
            ),
            "direct_safe_secondary_emergency_assignment_count": int(
                self.direct_safe_secondary_emergency_assignment_count
            ),
            "lifecycle_turnaround_cost_evaluation_count": int(
                self.lifecycle_turnaround_cost_evaluation_count
            ),
            "lifecycle_turnaround_cost_total": float(
                self.lifecycle_turnaround_cost_total
            ),
            "lexicographic_comparison_count": int(
                self.lexicographic_comparison_count
            ),
            "lexicographic_primary_rejection_count": int(
                self.lexicographic_primary_rejection_count
            ),
            "disconnect_profile_evaluation_count": int(
                self.disconnect_profile_evaluation_count
            ),
            "disconnect_protected_task_count": int(
                self.disconnect_protected_task_count
            ),
            "disconnect_predicted_miss_count": int(
                self.disconnect_predicted_miss_count
            ),
            "emergency_balance_trigger_count": int(
                self.emergency_balance_trigger_count
            ),
            "emergency_balance_baseline_max_count": int(
                self.emergency_balance_baseline_max_count
            ),
            "emergency_capacity_repair_count": int(
                self.emergency_capacity_repair_count
            ),
            "emergency_capacity_contract_move_count": int(
                self.emergency_capacity_contract_move_count
            ),
            "residual_emergency_handoff_count": int(
                self.residual_emergency_handoff_count
            ),
            "routine_inventory_rebalance_count": int(
                self.routine_inventory_rebalance_count
            ),
            "risk_slack_routine_candidate_count": int(
                self.risk_slack_routine_candidate_count
            ),
            "risk_slack_routine_trigger_count": int(
                self.risk_slack_routine_trigger_count
            ),
            "risk_slack_routine_release_count": int(
                self.risk_slack_routine_release_count
            ),
            "risk_slack_routine_cross_truck_repair_count": int(
                self.risk_slack_routine_cross_truck_repair_count
            ),
            "risk_slack_routine_reserved_inventory_block_count": int(
                self.risk_slack_routine_reserved_inventory_block_count
            ),
            "risk_slack_routine_protected_count": int(
                self.risk_slack_routine_protected_count
            ),
            "risk_slack_routine_unreachable_count": int(
                self.risk_slack_routine_unreachable_count
            ),
            "risk_slack_routine_stalled_count": int(
                self.risk_slack_routine_stalled_count
            ),
            "risk_slack_routine_eta_guard_block_count": int(
                self.risk_slack_routine_eta_guard_block_count
            ),
            "risk_slack_routine_same_truck_block_count": int(
                self.risk_slack_routine_same_truck_block_count
            ),
            "risk_slack_routine_tc_guard_block_count": int(
                self.risk_slack_routine_tc_guard_block_count
            ),
            "r4_routine_takeover_candidate_count": int(
                self.r4_routine_takeover_candidate_count
            ),
            "r4_routine_takeover_trigger_count": int(
                self.r4_routine_takeover_trigger_count
            ),
            "r4_routine_takeover_success_count": int(
                self.r4_routine_takeover_success_count
            ),
            "r4_routine_takeover_reject_started_count": int(
                self.r4_routine_takeover_reject_started_count
            ),
            "r4_routine_takeover_reject_contract_count": int(
                self.r4_routine_takeover_reject_contract_count
            ),
            "r4_routine_takeover_reject_owner_active_count": int(
                self.r4_routine_takeover_reject_owner_active_count
            ),
            "r4_routine_takeover_reject_no_idle_truck_count": int(
                self.r4_routine_takeover_reject_no_idle_truck_count
            ),
            "r4_routine_takeover_reject_unreachable_count": int(
                self.r4_routine_takeover_reject_unreachable_count
            ),
            "r4_routine_takeover_reject_inventory_count": int(
                self.r4_routine_takeover_reject_inventory_count
            ),
            "r4_routine_takeover_reject_deadline_count": int(
                self.r4_routine_takeover_reject_deadline_count
            ),
            "r4_routine_takeover_reject_safety_count": int(
                self.r4_routine_takeover_reject_safety_count
            ),
            "idle_routine_dispatch_candidate_count": int(
                self.idle_routine_dispatch_candidate_count
            ),
            "idle_routine_dispatch_success_count": int(
                self.idle_routine_dispatch_success_count
            ),
            "idle_routine_dispatch_emergency_block_count": int(
                self.idle_routine_dispatch_emergency_block_count
            ),
            "idle_routine_dispatch_reject_no_idle_truck_count": int(
                self.idle_routine_dispatch_reject_no_idle_truck_count
            ),
            "idle_routine_dispatch_reject_unreachable_count": int(
                self.idle_routine_dispatch_reject_unreachable_count
            ),
            "idle_routine_dispatch_reject_inventory_count": int(
                self.idle_routine_dispatch_reject_inventory_count
            ),
            "idle_routine_dispatch_reject_deadline_count": int(
                self.idle_routine_dispatch_reject_deadline_count
            ),
            "idle_routine_dispatch_reject_contract_count": int(
                self.idle_routine_dispatch_reject_contract_count
            ),
            "routine_service_start_rescue_candidate_count": int(
                self.routine_service_start_rescue_candidate_count
            ),
            "routine_service_start_rescue_success_count": int(
                self.routine_service_start_rescue_success_count
            ),
            "routine_service_start_rescue_alternate_count": int(
                self.routine_service_start_rescue_alternate_count
            ),
            "routine_service_start_rescue_emergency_block_count": int(
                self.routine_service_start_rescue_emergency_block_count
            ),
            "routine_service_start_rescue_reject_active_count": int(
                self.routine_service_start_rescue_reject_active_count
            ),
            "routine_service_start_rescue_reject_no_owner_count": int(
                self.routine_service_start_rescue_reject_no_owner_count
            ),
            "routine_service_start_rescue_reject_no_alternate_count": int(
                self.routine_service_start_rescue_reject_no_alternate_count
            ),
            "balanced_all_tasks_normal_candidate_count": int(
                self.balanced_all_tasks_normal_candidate_count
            ),
            "balanced_all_tasks_normal_assignment_count": int(
                self.balanced_all_tasks_normal_assignment_count
            ),
            "balanced_all_tasks_quota_block_count": int(
                self.balanced_all_tasks_quota_block_count
            ),
            "balanced_all_tasks_emergency_tradeoff_count": int(
                self.balanced_all_tasks_emergency_tradeoff_count
            ),
            "balanced_all_tasks_watchdog_candidate_count": int(
                self.balanced_all_tasks_watchdog_candidate_count
            ),
            "balanced_all_tasks_watchdog_transfer_count": int(
                self.balanced_all_tasks_watchdog_transfer_count
            ),
            "balanced_all_tasks_watchdog_block_count": int(
                self.balanced_all_tasks_watchdog_block_count
            ),
            "balanced_all_tasks_v2_reauction_candidate_count": int(
                self.balanced_all_tasks_v2_reauction_candidate_count
            ),
            "balanced_all_tasks_v2_reauction_transfer_count": int(
                self.balanced_all_tasks_v2_reauction_transfer_count
            ),
            "balanced_all_tasks_v2_reauction_deadline_block_count": int(
                self.balanced_all_tasks_v2_reauction_deadline_block_count
            ),
            "balanced_all_tasks_v2_parallel_candidate_count": int(
                self.balanced_all_tasks_v2_parallel_candidate_count
            ),
            "balanced_all_tasks_v2_parallel_accept_count": int(
                self.balanced_all_tasks_v2_parallel_accept_count
            ),
            "balanced_all_tasks_v2_parallel_reject_count": int(
                self.balanced_all_tasks_v2_parallel_reject_count
            ),
            "balanced_all_tasks_v2_parallel_payload_bypass_count": int(
                self.balanced_all_tasks_v2_parallel_payload_bypass_count
            ),
            "balanced_all_tasks_v2_aggressive_auction_candidate_count": int(
                self.balanced_all_tasks_v2_aggressive_auction_candidate_count
            ),
            "balanced_all_tasks_v2_aggressive_auction_transfer_count": int(
                self.balanced_all_tasks_v2_aggressive_auction_transfer_count
            ),
            "balanced_all_tasks_v2_after_launch_normal_candidate_count": int(
                self.balanced_all_tasks_v2_after_launch_normal_candidate_count
            ),
            "balanced_all_tasks_v2_after_launch_normal_accept_count": int(
                self.balanced_all_tasks_v2_after_launch_normal_accept_count
            ),
            "balanced_all_tasks_v2_after_launch_normal_reject_count": int(
                self.balanced_all_tasks_v2_after_launch_normal_reject_count
            ),
            "balanced_all_tasks_v3_normal_candidate_count": int(
                self.balanced_all_tasks_v3_normal_candidate_count
            ),
            "balanced_all_tasks_v3_emergency_candidate_count": int(
                self.balanced_all_tasks_v3_emergency_candidate_count
            ),
            "balanced_all_tasks_v3_selected_normal_count": int(
                self.balanced_all_tasks_v3_selected_normal_count
            ),
            "balanced_all_tasks_v3_selected_emergency_count": int(
                self.balanced_all_tasks_v3_selected_emergency_count
            ),
            "balanced_all_tasks_v3_fallback_count": int(
                self.balanced_all_tasks_v3_fallback_count
            ),
            "balanced_all_tasks_v3_fallback_reason_counts": dict(
                self.balanced_all_tasks_v3_fallback_reason_counts
            ),
            "balanced_all_tasks_v3_last_diagnostics": dict(
                self.balanced_all_tasks_v3_last_diagnostics
            ),
            "balanced_all_tasks_v5_promoted_count": int(
                self.balanced_all_tasks_v5_promoted_count
            ),
            "balanced_all_tasks_v5_rejected_safety_count": int(
                self.balanced_all_tasks_v5_rejected_safety_count
            ),
            "shadow_total_coverage_candidate_count": int(
                self.shadow_total_coverage_candidate_count
            ),
            "shadow_total_coverage_accept_count": int(
                self.shadow_total_coverage_accept_count
            ),
            "shadow_total_coverage_reject_count": int(
                self.shadow_total_coverage_reject_count
            ),
            "shadow_total_coverage_last_diagnostics": dict(
                self.shadow_total_coverage_last_diagnostics
            ),
            "shadow_total_coverage_first_accept_diagnostics": dict(
                self.shadow_total_coverage_first_accept_diagnostics
            ),
            "parallel_routine_emergency_candidate_count": int(
                self.parallel_routine_emergency_candidate_count
            ),
            "parallel_routine_emergency_accept_count": int(
                self.parallel_routine_emergency_accept_count
            ),
            "parallel_routine_emergency_fallback_wait_count": int(
                self.parallel_routine_emergency_fallback_wait_count
            ),
            "parallel_routine_emergency_reject_energy_count": int(
                self.parallel_routine_emergency_reject_energy_count
            ),
            "parallel_routine_emergency_reject_deadline_count": int(
                self.parallel_routine_emergency_reject_deadline_count
            ),
            "normal_cleanup_replan_count": int(
                self.normal_cleanup_replan_count
            ),
            "hard_normal_rescue_candidate_count": int(
                self.hard_normal_rescue_candidate_count
            ),
            "hard_normal_rescue_transfer_count": int(
                self.hard_normal_rescue_transfer_count
            ),
            "hard_normal_rescue_no_goal_count": int(
                self.hard_normal_rescue_no_goal_count
            ),
            "hard_normal_rescue_stalled_owner_count": int(
                self.hard_normal_rescue_stalled_owner_count
            ),
            "hard_normal_rescue_rejected_safety_count": int(
                self.hard_normal_rescue_rejected_safety_count
            ),
            "hard_normal_rescue_no_truck_count": int(
                self.hard_normal_rescue_no_truck_count
            ),
            "hard_normal_rescue_no_truck_skip_count": int(
                self.hard_normal_rescue_no_truck_skip_count
            ),
            "b_orphaned_routine_rescue_count": int(
                self.b_orphaned_routine_rescue_count
            ),
            "queue_starvation_repair_count": int(
                self.queue_starvation_repair_count
            ),
            "queue_rescue_assignment_count": int(
                self.queue_rescue_assignment_count
            ),
            "queue_rescue_delivery_count": int(
                self.queue_rescue_delivery_count
            ),
            "initial_lifeline_ordering_enabled": bool(
                self.initial_lifeline_ordering_enabled
            ),
            "contract_consistency_block_count": int(
                self.contract_consistency_block_count
            ),
            "alns_iteration_count": int(self.alns_iteration_count),
            "alns_destroyed_assignment_count": int(
                self.alns_destroyed_assignment_count
            ),
            "alns_repair_attempt_count": int(
                self.alns_repair_attempt_count
            ),
            "alns_repair_feasible_count": int(
                self.alns_repair_feasible_count
            ),
            "alns_accepted_count": int(self.alns_accepted_count),
            "alns_improvement_count": int(self.alns_improvement_count),
            "alns_replan_count": int(self.alns_replan_count),
            "alns_objective_evaluation_count": int(
                self.alns_objective_evaluation_count
            ),
            "alns_feasibility_evaluation_count": int(
                self.alns_feasibility_evaluation_count
            ),
            "alns_wall_clock_time_s": float(self.alns_wall_clock_time_s),
            "routes": routes,
            "contracts": {
                str(task_id): {
                    "owner_agent_id": str(contract.owner_agent_id),
                    "truck_id": str(contract.truck_id),
                    "uav_id": ""
                    if contract.uav_id is None
                    else str(contract.uav_id),
                    "service_mode": str(contract.service_mode),
                    "locked": bool(contract.locked),
                    "version": int(getattr(contract, "version", 0)),
                }
                for task_id, contract in self.plan.contracts.items()
            },
            "stay_reason_by_agent": dict(self._stay_reason_by_agent),
        }
