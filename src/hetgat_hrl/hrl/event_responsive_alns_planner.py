from __future__ import annotations

import hashlib
import json
import math
import pickle
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

import numpy as np

from hetgat_hrl.alns.adapters import env_adapter_context, legacy_goals_to_solution, solution_to_legacy_goals
from hetgat_hrl.alns.adaptive_horizon_v2 import AdaptiveHorizonControllerV2, AdaptiveHorizonFeatures
from hetgat_hrl.alns.canonical_operators import (
    CANONICAL_DESTROY_NAMES,
    CANONICAL_REPAIR_NAMES,
    RepairResult,
    greedy_insertion,
    random_removal,
    regret_2_insertion,
    regret_3_insertion,
    related_removal,
    sequence_segment_removal,
    worst_cost_removal,
)
from hetgat_hrl.alns.learning_v2 import FeasibilityCandidateRanker, ranker_active_select, ranker_shadow_scores
from hetgat_hrl.alns.local_search_v2 import LocalSearchBudget, LocalSearchRefinerV2
from hetgat_hrl.alns.objective import evaluate_solution, minimization_acceptance_probability, minimization_delta
from hetgat_hrl.core.algorithm_profile import er_hlns_route_plan_active
from hetgat_hrl.alns.problem_specific_operators import (
    critical_recovery_repair_insertion,
    ER_DESTROY_NAMES,
    ER_REPAIR_NAMES,
    critical_first_insertion,
    critical_task_reassignment_removal,
    feasibility_restoration_insertion,
    risk_aware_insertion,
    road_disruption_removal,
    support_conflict_removal,
    synchronization_risk_removal,
    synchronized_insertion,
)
from hetgat_hrl.alns.repair_candidate_pool import (
    RepairCandidate,
    RepairCandidatePool,
    enumerate_repair_candidate_pool,
    repair_result_from_candidate,
    select_ranker_candidates,
)
from hetgat_hrl.alns.runtime_sequence import (
    AgentSequenceRuntime,
    SequenceRuntimeState,
    TailValidationResult,
    runtime_from_solution,
)
from hetgat_hrl.alns.sequence import (
    construct_k2_solution,
    evaluate_k2_solution,
    evaluate_sequence_cost,
    evaluate_sequence_feasibility,
)
from hetgat_hrl.alns.solution import SortiePlan, SupportBinding
from hetgat_hrl.core.mdp_spec import AgentKind, TaskKind, TaskStatus
from hetgat_hrl.hrl.rolling_planner import EventTriggeredRollingPlanner, RollingPlannerWeights

TAIL_TASK_COMPLETED = "TAIL_TASK_COMPLETED"
TAIL_TASK_NOT_ACTIVE = "TAIL_TASK_NOT_ACTIVE"
TAIL_TASK_ASSIGNED_ELSEWHERE = "TAIL_TASK_ASSIGNED_ELSEWHERE"
TAIL_TASK_DUPLICATE_EXCLUSIVE = "TAIL_TASK_DUPLICATE_EXCLUSIVE"
TAIL_ROAD_UNREACHABLE = "TAIL_ROAD_UNREACHABLE"
TAIL_INVENTORY_INSUFFICIENT = "TAIL_INVENTORY_INSUFFICIENT"
TAIL_UAV_ENERGY_INSUFFICIENT = "TAIL_UAV_ENERGY_INSUFFICIENT"
TAIL_SUPPORT_BINDING_STALE = "TAIL_SUPPORT_BINDING_STALE"
TAIL_RECOVERY_NOT_FEASIBLE = "TAIL_RECOVERY_NOT_FEASIBLE"
TAIL_AGENT_STATE_CONFLICT = "TAIL_AGENT_STATE_CONFLICT"
TAIL_REPLAN_REPLACED = "TAIL_REPLAN_REPLACED"


@dataclass
class ALNSOperatorStats:
    weight: float = 1.0
    attempts: int = 0
    feasible: int = 0
    accepted: int = 0
    improved: int = 0
    global_best: int = 0
    failure: int = 0
    reward_ema: float = 0.0
    reward_total: float = 0.0
    segment_usage: int = 0
    segment_score: float = 0.0


@dataclass
class ALNSDiagnostics:
    iteration_count: int = 0
    destroyed_assignment_count: int = 0
    repair_attempt_count: int = 0
    repair_feasible_count: int = 0
    accepted_count: int = 0
    accepted_improving_count: int = 0
    accepted_worsening_count: int = 0
    improvement_count: int = 0
    noop_iteration_count: int = 0
    destroy_fallback_count: int = 0
    best_objective_gain: float = 0.0
    hard_constraint_violation_count: int = 0
    planner_candidate_hard_reject_count: int = 0
    initial_destroy_weights: Dict[str, float] = field(default_factory=dict)
    final_destroy_weights: Dict[str, float] = field(default_factory=dict)
    initial_repair_weights: Dict[str, float] = field(default_factory=dict)
    final_repair_weights: Dict[str, float] = field(default_factory=dict)
    destroy_operator_usage: Dict[str, int] = field(default_factory=dict)
    repair_operator_usage: Dict[str, int] = field(default_factory=dict)
    destroy_operator_reward: Dict[str, float] = field(default_factory=dict)
    repair_operator_reward: Dict[str, float] = field(default_factory=dict)
    objective_shadow_comparison_count: int = 0
    objective_shadow_agreement_count: int = 0
    objective_shadow_disagreement_count: int = 0
    legacy_k1_roundtrip_mismatch_count: int = 0
    k2_solution_count: int = 0
    k2_nonempty_tail_count: int = 0
    k2_sequence_length_sum: int = 0
    k2_sequence_agent_count: int = 0
    k2_tail_promoted_count: int = 0
    k2_tail_dropped_count: int = 0
    k2_tail_invalidated_count: int = 0
    k2_full_replan_count: int = 0
    sequence_installed_count: int = 0
    sequence_installed_with_tail_count: int = 0
    sequence_retained_step_count: int = 0
    tail_promoted_after_head_completion_count: int = 0
    tail_promoted_after_head_invalidation_count: int = 0
    tail_replaced_by_replan_count: int = 0
    tail_reused_after_replan_count: int = 0
    head_changed_by_replan_count: int = 0
    tail_changed_by_replan_count: int = 0
    full_sequence_changed_by_replan_count: int = 0
    sequence_runtime_missing_count: int = 0
    sequence_runtime_reset_count: int = 0
    average_tail_lifetime_steps: float = 0.0
    median_tail_lifetime_steps: float = 0.0
    maximum_tail_lifetime_steps: float = 0.0
    k2_first_task_difference_vs_k1: int = 0
    k2_candidate_ranking_difference_vs_k1: int = 0
    k2_second_task_travel_cost: float = 0.0
    k2_second_task_energy_cost: float = 0.0
    k2_second_task_lifeline_cost: float = 0.0
    k2_feasible_sequence_count: int = 0
    k2_infeasible_sequence_count: int = 0
    replan_count: int = 0
    operator_attempt_count: int = 0
    objective_evaluation_count: int = 0
    feasibility_evaluation_count: int = 0
    wall_clock_time_s: float = 0.0
    support_plan_created_count: int = 0
    support_plan_installed_count: int = 0
    support_plan_launched_count: int = 0
    support_plan_task_served_count: int = 0
    support_plan_recovered_count: int = 0
    support_plan_completed_count: int = 0
    support_plan_invalidated_count: int = 0
    unified_event_trigger_count: int = 0
    feasible_nonidentical_candidate_count: int = 0
    adaptive_horizon_decision_count: int = 0
    adaptive_horizon_k1_count: int = 0
    adaptive_horizon_k2_count: int = 0
    local_search_attempt_count: int = 0
    local_search_feasible_count: int = 0
    local_search_accepted_move_count: int = 0
    local_search_exact_check_count: int = 0
    local_search_runtime_ms: float = 0.0
    critical_recovery_candidates: int = 0
    critical_recovery_attempts: int = 0
    critical_recovery_direct_insertions: int = 0
    critical_recovery_safe_reorders: int = 0
    critical_recovery_rejected_infeasible: int = 0
    critical_recovery_rejected_no_slot: int = 0
    critical_recovery_rejected_duplicate_claim: int = 0
    critical_recovery_avoided_failed_agent: int = 0
    critical_support_rebind_candidates: int = 0
    critical_support_rebind_attempts: int = 0
    critical_support_rebind_historical_reuse: int = 0
    critical_support_rebind_reconstructed: int = 0
    critical_support_rebind_rejected_no_truck: int = 0
    critical_support_rebind_rejected_no_anchor: int = 0
    critical_support_rebind_rejected_energy: int = 0
    critical_support_rebind_rejected_reserve: int = 0
    critical_support_rebind_rejected_road: int = 0
    critical_support_rebind_rejected_infeasible: int = 0
    critical_support_rebind_accept_count: int = 0
    critical_support_rebind_failed_binding_penalized: int = 0
    critical_support_rebind_failed_binding_skipped: int = 0
    critical_support_rebind_best_accepted_margin_m: float = float("-inf")
    critical_support_rebind_best_rejected_margin_m: float = float("-inf")
    critical_support_rebind_best_accepted_battery_margin: float = float("-inf")
    critical_support_rebind_best_rejected_battery_margin: float = float("-inf")
    lc_critical_recovery_path_candidates: int = 0
    lc_critical_recovery_path_attempts: int = 0
    lc_critical_recovery_path_successes: int = 0
    lc_critical_recovery_path_rejected_insufficient_margin: int = 0
    lc_critical_recovery_path_rejected_no_bindable_truck: int = 0
    lc_critical_recovery_path_rejected_uav_not_docked: int = 0
    lc_critical_recovery_path_rejected_no_sequence_capacity: int = 0
    lc_critical_recovery_path_rejected_augmented_infeasible: int = 0
    lc_critical_recovery_path_trucks_considered: int = 0
    lc_critical_recovery_path_best_margin: float = float("-inf")
    lc_critical_recovery_path_success_margin: float = float("-inf")
    lc_critical_recovery_path_failed_tuple_avoided: int = 0
    assigned_critical_reconstruct_candidates: int = 0
    assigned_critical_reconstruct_path_candidates: int = 0
    assigned_critical_reconstruct_trucks_considered: int = 0
    assigned_critical_reconstruct_margin_probed: int = 0
    assigned_critical_reconstruct_positive_margin_count: int = 0
    assigned_critical_reconstruct_selected_path_count: int = 0
    assigned_critical_reconstruct_success_count: int = 0
    assigned_critical_reconstruct_rejected_no_bindable_truck: int = 0
    assigned_critical_reconstruct_rejected_insufficient_margin: int = 0
    assigned_critical_reconstruct_rejected_no_sequence_capacity: int = 0
    assigned_critical_reconstruct_rejected_target_unreachable: int = 0
    assigned_critical_reconstruct_rejected_uav_not_docked: int = 0
    assigned_critical_reconstruct_rejected_infeasible: int = 0
    assigned_critical_reconstruct_no_progress_tasks_targeted: int = 0
    support_reposition_shadow_candidates: int = 0
    support_reposition_shadow_feasible_suggestions: int = 0
    support_reposition_shadow_low_battery_rescue_possible: int = 0
    support_reposition_shadow_unreachable_rescue_possible: int = 0
    support_reposition_shadow_no_progress_tasks_covered: int = 0
    support_reposition_shadow_estimated_battery_gain: float = 0.0
    support_reposition_shadow_estimated_truck_cost: float = 0.0

    def to_flat_dict(self) -> Dict[str, float]:
        return {
            "alns_iteration_count": float(self.iteration_count),
            "alns_destroyed_assignment_count": float(self.destroyed_assignment_count),
            "alns_repair_attempt_count": float(self.repair_attempt_count),
            "alns_repair_feasible_count": float(self.repair_feasible_count),
            "alns_accepted_count": float(self.accepted_count),
            "alns_accepted_improving_count": float(self.accepted_improving_count),
            "alns_accepted_worsening_count": float(self.accepted_worsening_count),
            "alns_improvement_count": float(self.improvement_count),
            "alns_noop_iteration_count": float(self.noop_iteration_count),
            "alns_destroy_fallback_count": float(self.destroy_fallback_count),
            "alns_best_objective_gain": float(self.best_objective_gain),
            "planner_candidate_hard_reject_count": float(self.planner_candidate_hard_reject_count),
            "hard_constraint_violation_count": float(self.hard_constraint_violation_count),
            "objective_shadow_comparison_count": float(self.objective_shadow_comparison_count),
            "objective_shadow_agreement_count": float(self.objective_shadow_agreement_count),
            "objective_shadow_disagreement_count": float(self.objective_shadow_disagreement_count),
            "legacy_k1_roundtrip_mismatch_count": float(self.legacy_k1_roundtrip_mismatch_count),
            "k2_solution_count": float(self.k2_solution_count),
            "k2_nonempty_tail_count": float(self.k2_nonempty_tail_count),
            "k2_average_sequence_length": float(self.k2_sequence_length_sum / max(self.k2_sequence_agent_count, 1)),
            "k2_tail_promoted_count": float(self.k2_tail_promoted_count),
            "k2_tail_dropped_count": float(self.k2_tail_dropped_count),
            "k2_tail_invalidated_count": float(self.k2_tail_invalidated_count),
            "k2_full_replan_count": float(self.k2_full_replan_count),
            "sequence_installed_count": float(self.sequence_installed_count),
            "sequence_installed_with_tail_count": float(self.sequence_installed_with_tail_count),
            "sequence_retained_step_count": float(self.sequence_retained_step_count),
            "tail_promoted_after_head_completion_count": float(self.tail_promoted_after_head_completion_count),
            "tail_promoted_after_head_invalidation_count": float(self.tail_promoted_after_head_invalidation_count),
            "tail_replaced_by_replan_count": float(self.tail_replaced_by_replan_count),
            "tail_reused_after_replan_count": float(self.tail_reused_after_replan_count),
            "head_changed_by_replan_count": float(self.head_changed_by_replan_count),
            "tail_changed_by_replan_count": float(self.tail_changed_by_replan_count),
            "full_sequence_changed_by_replan_count": float(self.full_sequence_changed_by_replan_count),
            "sequence_runtime_missing_count": float(self.sequence_runtime_missing_count),
            "sequence_runtime_reset_count": float(self.sequence_runtime_reset_count),
            "average_tail_lifetime_steps": float(self.average_tail_lifetime_steps),
            "median_tail_lifetime_steps": float(self.median_tail_lifetime_steps),
            "maximum_tail_lifetime_steps": float(self.maximum_tail_lifetime_steps),
            "k2_first_task_difference_vs_k1": float(self.k2_first_task_difference_vs_k1),
            "k2_candidate_ranking_difference_vs_k1": float(self.k2_candidate_ranking_difference_vs_k1),
            "k2_second_task_travel_cost": float(self.k2_second_task_travel_cost),
            "k2_second_task_energy_cost": float(self.k2_second_task_energy_cost),
            "k2_second_task_lifeline_cost": float(self.k2_second_task_lifeline_cost),
            "k2_feasible_sequence_count": float(self.k2_feasible_sequence_count),
            "k2_infeasible_sequence_count": float(self.k2_infeasible_sequence_count),
            "alns_replan_count": float(self.replan_count),
            "alns_iterations_per_replan": float(self.iteration_count / max(self.replan_count, 1)),
            "alns_operator_attempt_count": float(self.operator_attempt_count),
            "alns_objective_evaluation_count": float(self.objective_evaluation_count),
            "alns_feasibility_evaluation_count": float(self.feasibility_evaluation_count),
            "alns_wall_clock_time_s": float(self.wall_clock_time_s),
            "support_plan_created_count": float(self.support_plan_created_count),
            "support_plan_installed_count": float(self.support_plan_installed_count),
            "support_plan_launched_count": float(self.support_plan_launched_count),
            "support_plan_task_served_count": float(self.support_plan_task_served_count),
            "support_plan_recovered_count": float(self.support_plan_recovered_count),
            "support_plan_completed_count": float(self.support_plan_completed_count),
            "support_plan_invalidated_count": float(self.support_plan_invalidated_count),
            "unified_event_trigger_count": float(self.unified_event_trigger_count),
            "feasible_nonidentical_candidates": float(self.feasible_nonidentical_candidate_count),
            "adaptive_horizon_decision_count": float(self.adaptive_horizon_decision_count),
            "adaptive_horizon_k1_count": float(self.adaptive_horizon_k1_count),
            "adaptive_horizon_k2_count": float(self.adaptive_horizon_k2_count),
            "adaptive_horizon_k1_ratio": float(self.adaptive_horizon_k1_count / max(self.adaptive_horizon_decision_count, 1)),
            "adaptive_horizon_k2_ratio": float(self.adaptive_horizon_k2_count / max(self.adaptive_horizon_decision_count, 1)),
            "local_search_attempt_count": float(self.local_search_attempt_count),
            "local_search_accepted_move_count": float(self.local_search_accepted_move_count),
            "local_search_exact_check_count": float(self.local_search_exact_check_count),
            "local_search_runtime_ms": float(self.local_search_runtime_ms),
            "local_search_attempts": float(self.local_search_attempt_count),
            "local_search_feasible": float(self.local_search_feasible_count),
            "local_search_accepted": float(self.local_search_accepted_move_count),
            "local_search_exact_checks": float(self.local_search_exact_check_count),
            "alns_critical_recovery_candidates": float(self.critical_recovery_candidates),
            "alns_critical_recovery_attempts": float(self.critical_recovery_attempts),
            "alns_critical_recovery_direct_insertions": float(self.critical_recovery_direct_insertions),
            "alns_critical_recovery_safe_reorders": float(self.critical_recovery_safe_reorders),
            "alns_critical_recovery_rejected_infeasible": float(self.critical_recovery_rejected_infeasible),
            "alns_critical_recovery_rejected_no_slot": float(self.critical_recovery_rejected_no_slot),
            "alns_critical_recovery_rejected_duplicate_claim": float(self.critical_recovery_rejected_duplicate_claim),
            "alns_critical_recovery_avoided_failed_agent": float(self.critical_recovery_avoided_failed_agent),
            "critical_support_rebind_candidates": float(self.critical_support_rebind_candidates),
            "critical_support_rebind_attempts": float(self.critical_support_rebind_attempts),
            "critical_support_rebind_historical_reuse": float(self.critical_support_rebind_historical_reuse),
            "critical_support_rebind_reconstructed": float(self.critical_support_rebind_reconstructed),
            "critical_support_rebind_rejected_no_truck": float(self.critical_support_rebind_rejected_no_truck),
            "critical_support_rebind_rejected_no_anchor": float(self.critical_support_rebind_rejected_no_anchor),
            "critical_support_rebind_rejected_energy": float(self.critical_support_rebind_rejected_energy),
            "critical_support_rebind_rejected_reserve": float(self.critical_support_rebind_rejected_reserve),
            "critical_support_rebind_rejected_road": float(self.critical_support_rebind_rejected_road),
            "critical_support_rebind_rejected_infeasible": float(self.critical_support_rebind_rejected_infeasible),
            "critical_support_rebind_accept_count": float(self.critical_support_rebind_accept_count),
            "critical_support_rebind_failed_binding_penalized": float(self.critical_support_rebind_failed_binding_penalized),
            "critical_support_rebind_failed_binding_skipped": float(self.critical_support_rebind_failed_binding_skipped),
            "critical_support_rebind_best_accepted_margin_m": float(self.critical_support_rebind_best_accepted_margin_m),
            "critical_support_rebind_best_rejected_margin_m": float(self.critical_support_rebind_best_rejected_margin_m),
            "critical_support_rebind_best_accepted_battery_margin": float(self.critical_support_rebind_best_accepted_battery_margin),
            "critical_support_rebind_best_rejected_battery_margin": float(self.critical_support_rebind_best_rejected_battery_margin),
            "lc_critical_recovery_path_candidates": float(self.lc_critical_recovery_path_candidates),
            "lc_critical_recovery_path_attempts": float(self.lc_critical_recovery_path_attempts),
            "lc_critical_recovery_path_successes": float(self.lc_critical_recovery_path_successes),
            "lc_critical_recovery_path_rejected_insufficient_margin": float(self.lc_critical_recovery_path_rejected_insufficient_margin),
            "lc_critical_recovery_path_rejected_no_bindable_truck": float(self.lc_critical_recovery_path_rejected_no_bindable_truck),
            "lc_critical_recovery_path_rejected_uav_not_docked": float(self.lc_critical_recovery_path_rejected_uav_not_docked),
            "lc_critical_recovery_path_rejected_no_sequence_capacity": float(self.lc_critical_recovery_path_rejected_no_sequence_capacity),
            "lc_critical_recovery_path_rejected_augmented_infeasible": float(self.lc_critical_recovery_path_rejected_augmented_infeasible),
            "lc_critical_recovery_path_trucks_considered": float(self.lc_critical_recovery_path_trucks_considered),
            "lc_critical_recovery_path_best_margin": float(self.lc_critical_recovery_path_best_margin),
            "lc_critical_recovery_path_success_margin": float(self.lc_critical_recovery_path_success_margin),
            "lc_critical_recovery_path_failed_tuple_avoided": float(self.lc_critical_recovery_path_failed_tuple_avoided),
            "assigned_critical_reconstruct_candidates": float(self.assigned_critical_reconstruct_candidates),
            "assigned_critical_reconstruct_path_candidates": float(self.assigned_critical_reconstruct_path_candidates),
            "assigned_critical_reconstruct_trucks_considered": float(self.assigned_critical_reconstruct_trucks_considered),
            "assigned_critical_reconstruct_margin_probed": float(self.assigned_critical_reconstruct_margin_probed),
            "assigned_critical_reconstruct_positive_margin_count": float(self.assigned_critical_reconstruct_positive_margin_count),
            "assigned_critical_reconstruct_selected_path_count": float(self.assigned_critical_reconstruct_selected_path_count),
            "assigned_critical_reconstruct_success_count": float(self.assigned_critical_reconstruct_success_count),
            "assigned_critical_reconstruct_rejected_no_bindable_truck": float(self.assigned_critical_reconstruct_rejected_no_bindable_truck),
            "assigned_critical_reconstruct_rejected_insufficient_margin": float(self.assigned_critical_reconstruct_rejected_insufficient_margin),
            "assigned_critical_reconstruct_rejected_no_sequence_capacity": float(self.assigned_critical_reconstruct_rejected_no_sequence_capacity),
            "assigned_critical_reconstruct_rejected_target_unreachable": float(self.assigned_critical_reconstruct_rejected_target_unreachable),
            "assigned_critical_reconstruct_rejected_uav_not_docked": float(self.assigned_critical_reconstruct_rejected_uav_not_docked),
            "assigned_critical_reconstruct_rejected_infeasible": float(self.assigned_critical_reconstruct_rejected_infeasible),
            "assigned_critical_reconstruct_no_progress_tasks_targeted": float(self.assigned_critical_reconstruct_no_progress_tasks_targeted),
            "support_reposition_shadow_candidates": float(self.support_reposition_shadow_candidates),
            "support_reposition_shadow_feasible_suggestions": float(self.support_reposition_shadow_feasible_suggestions),
            "support_reposition_shadow_low_battery_rescue_possible": float(self.support_reposition_shadow_low_battery_rescue_possible),
            "support_reposition_shadow_unreachable_rescue_possible": float(self.support_reposition_shadow_unreachable_rescue_possible),
            "support_reposition_shadow_no_progress_tasks_covered": float(self.support_reposition_shadow_no_progress_tasks_covered),
            "support_reposition_shadow_estimated_battery_gain": float(self.support_reposition_shadow_estimated_battery_gain),
            "support_reposition_shadow_estimated_truck_cost": float(self.support_reposition_shadow_estimated_truck_cost),
            "total_exact_feasibility_checks": float(self.feasibility_evaluation_count + self.local_search_exact_check_count),
            "alns_exact_feasibility_checks": float(self.feasibility_evaluation_count),
            "runtime_total": float(self.wall_clock_time_s),
            "runtime_planner": float(max(self.wall_clock_time_s - self.local_search_runtime_ms / 1000.0, 0.0)),
            "runtime_local_search": float(self.local_search_runtime_ms / 1000.0),
        }


class EventResponsiveALNSPlanner(EventTriggeredRollingPlanner):
    """Event-responsive adaptive rolling ALNS built on top of ERC.

    The planner keeps ERC's safety, support and repair layers as the execution
    guardrail. ALNS operates on the rolling assignment dictionary only: it
    destroys a small subset of fragile assignments, greedily repairs them with
    road-risk/risk-pressure-aware scores, and accepts improving local changes.
    """

    def __init__(
        self,
        decision_interval: int = 5,
        seed: int = 0,
        weights: Optional[RollingPlannerWeights] = None,
        use_risk_term: bool = True,
        use_rth_repair: bool = True,
        use_event_trigger: bool = True,
        replan_cooldown_steps: int = 4,
        min_goal_hold_steps: int = 8,
        switch_margin: float = 0.12,
        iterations: int = 24,
    ) -> None:
        super().__init__(
            decision_interval=decision_interval,
            seed=seed,
            weights=weights,
            use_risk_term=use_risk_term,
            use_rth_repair=use_rth_repair,
            use_event_trigger=use_event_trigger,
            replan_cooldown_steps=replan_cooldown_steps,
            min_goal_hold_steps=min_goal_hold_steps,
            switch_margin=switch_margin,
        )
        self.base_decision_interval = int(max(decision_interval, 1))
        self.alns_iterations = int(max(iterations, 0))
        if self.alns_iterations <= 0:
            raise ValueError("EventResponsiveALNSPlanner requires alns_iterations > 0")
        self.rng = np.random.default_rng(int(seed) + 7919)
        self.destroy_ops = {
            "road_disruption": ALNSOperatorStats(1.25),
            "stale_or_low_value": ALNSOperatorStats(1.00),
            "tc_uncovered": ALNSOperatorStats(1.20),
            "support_gap": ALNSOperatorStats(1.15),
            "random_light": ALNSOperatorStats(0.75),
        }
        self.repair_ops = {
            "direct_service_insert": ALNSOperatorStats(1.30),
            "risk_greedy_insert": ALNSOperatorStats(1.20),
            "tc_first_insert": ALNSOperatorStats(1.15),
            "risk_balanced_insert": ALNSOperatorStats(1.00),
        }
        self.alns_diagnostics = ALNSDiagnostics(
            initial_destroy_weights={k: float(v.weight) for k, v in self.destroy_ops.items()},
            initial_repair_weights={k: float(v.weight) for k, v in self.repair_ops.items()},
        )
        self.alns_iteration_count_total: int = 0
        self.alns_improvement_count_total: int = 0
        self.alns_accepted_count_total: int = 0
        self.alns_destroyed_assignment_count_total: int = 0
        self.alns_risk_pressure_task_count_total: int = 0
        self.alns_ghost_task_count_total: int = 0
        self.alns_disturbance_rate_last: float = 0.0
        self.alns_horizon_steps_last: int = int(max(decision_interval * 8, 1))
        self._operator_pool_name: str = "legacy"
        self._last_blocked_edge_count: int = 0
        self._last_pending_task_count: int = 0
        self._risk_pressure_by_node: Dict[int, float] = {}
        self._last_blocked_edges_seen: Set[Tuple[int, int]] = set()
        self._road_event_active_for_plan: bool = False
        self.objective_shadow_records: List[Dict[str, Any]] = []
        self.k2_sequence_records: List[Dict[str, Any]] = []
        self.k2_runtime_sequence_records: List[Dict[str, Any]] = []
        self.k2_sa_delta_records: List[Dict[str, Any]] = []
        self.canonical_operator_records: List[Dict[str, Any]] = []
        self.support_execution_records: List[Dict[str, Any]] = []
        self.operator_weight_trajectory_records: List[Dict[str, Any]] = []
        self.event_trigger_records: List[Dict[str, Any]] = []
        self.sa_calibration_records: List[Dict[str, Any]] = []
        self.live_candidate_records: List[Dict[str, Any]] = []
        self.ranker_runtime_records: List[Dict[str, Any]] = []
        self.repair_candidate_pool_records: List[Dict[str, Any]] = []
        self.adaptive_horizon_records: List[Dict[str, Any]] = []
        self.local_search_records: List[Dict[str, Any]] = []
        self.adaptive_horizon_controller = AdaptiveHorizonControllerV2()
        self._live_candidate_ranker: Optional[FeasibilityCandidateRanker] = None
        self._live_candidate_ranker_source: str = "not_loaded"
        self._deduped_event_keys: Set[Tuple[int, str]] = set()
        self._support_status_seen: Set[Tuple[str, str]] = set()
        self._support_rebind_failure_counts: Dict[Tuple[str, str, str, int, int], int] = {}
        self._support_rebind_success_counts: Dict[Tuple[str, str, str, int, int], int] = {}
        self._lc_critical_recovery_failure_counts: Dict[Tuple[str, str, str, int, int], int] = {}
        self._lc_critical_recovery_success_counts: Dict[Tuple[str, str, str, int, int], int] = {}
        self._task_assignment_counts: Dict[str, int] = {}
        self._task_first_assigned_step: Dict[str, int] = {}
        self._task_last_assigned_step: Dict[str, int] = {}
        self._last_temperature: float = float("nan")
        self.sequence_runtime_state = SequenceRuntimeState()
        self._k2_runtime_last_seen_step: int = -1
        self._last_effective_k: int = 2

    def _environment_disturbance_rate(self, env) -> float:
        blocked = int(getattr(env, "blocked_edge_count", getattr(env, "blocked_edge_count_total", 0)))
        if blocked <= 0:
            blocked = int(getattr(env, "shared_blocked_edge_count", 0))
        pending = int(
            sum(
                1
                for t in env.state.tasks.values()
                if getattr(t, "status", None) == TaskStatus.PENDING
            )
        )
        db = max(blocked - int(self._last_blocked_edge_count), 0)
        dt = abs(pending - int(self._last_pending_task_count))
        denom = max(len(getattr(env.topology, "edges", [])), 1)
        rate = float(np.clip(3.0 * db / denom + 0.08 * dt, 0.0, 1.0))
        self._last_blocked_edge_count = int(blocked)
        self._last_pending_task_count = int(pending)
        self.alns_disturbance_rate_last = float(rate)
        return float(rate)

    def _update_adaptive_rolling_params(self, env) -> None:
        if not bool(getattr(env.cfg, "alns_adaptive_horizon_enabled", True)):
            return
        rate = float(self._environment_disturbance_rate(env))
        base = int(max(self.base_decision_interval, 1))
        min_dt = int(max(getattr(env.cfg, "alns_min_replan_interval_steps", max(2, base // 2)), 1))
        max_dt = int(max(getattr(env.cfg, "alns_max_replan_interval_steps", max(base * 2, min_dt)), min_dt))
        new_dt = int(round(max_dt - rate * (max_dt - min_dt)))
        self.decision_interval = int(max(min_dt, min(max_dt, new_dt)))

        min_h = int(max(getattr(env.cfg, "alns_min_horizon_steps", self.decision_interval * 4), 1))
        max_h = int(max(getattr(env.cfg, "alns_max_horizon_steps", self.decision_interval * 14), min_h))
        self.alns_horizon_steps_last = int(round(max_h - rate * (max_h - min_h)))

    def _adaptive_horizon_features(self, env) -> AdaptiveHorizonFeatures:
        physical = getattr(env, "physical_v2", None)
        edge_count = float(max(len(getattr(getattr(env, "topology", None), "edges", [])), 1))
        blocked = float(getattr(env, "blocked_edge_count", getattr(env, "shared_blocked_edge_count", 0)))
        degraded = float(getattr(env, "degraded_edge_count", 0))
        weather_ledger = list(getattr(physical, "weather_ledger", []) if physical is not None else [])
        last_weather = weather_ledger[-1] if weather_ledger else {}
        if not isinstance(last_weather, dict):
            last_weather = {}
        pending_tasks = [
            t for t in getattr(getattr(env, "state", None), "tasks", {}).values()
            if getattr(t, "status", None) == TaskStatus.PENDING
        ]
        critical = [
            t for t in pending_tasks
            if getattr(t, "kind", None) == TaskKind.EMERGENCY
        ]
        agents = list(getattr(getattr(env, "state", None), "agents", {}).values())
        batteries = [
            float(getattr(st, "battery", 1.0))
            for st in agents
            if getattr(st, "kind", None) == AgentKind.UAV and hasattr(st, "battery")
        ]
        repair_rate = float(self.alns_diagnostics.repair_feasible_count / max(self.alns_diagnostics.repair_attempt_count, 1))
        shield_rate = float(getattr(physical, "shield_intervention_count", 0) / max(self.alns_diagnostics.iteration_count, 1)) if physical is not None else 0.0
        return AdaptiveHorizonFeatures(
            road_damage_ratio=float(np.clip(blocked / edge_count, 0.0, 1.0)),
            blocked_edge_ratio=float(np.clip(blocked / edge_count, 0.0, 1.0)),
            degraded_edge_ratio=float(np.clip(degraded / edge_count, 0.0, 1.0)),
            weather_severity=float(np.clip(last_weather.get("severity", 0.0), 0.0, 1.0)),
            no_fly_ratio=float(np.clip(last_weather.get("no_fly_ratio", 0.0), 0.0, 1.0)),
            wind_speed=float(last_weather.get("wind_speed", last_weather.get("wind_speed_mps", 0.0)) or 0.0),
            rain_intensity=float(np.clip(last_weather.get("rain_intensity", 0.0), 0.0, 1.0)),
            visibility=float(np.clip(last_weather.get("visibility", 1.0), 0.0, 1.0)),
            battery_reserve=float(np.clip(min(batteries) if batteries else 1.0, 0.0, 1.0)),
            recovery_reserve=float(np.clip(getattr(env, "physical_v2_minimum_energy_reserve_seen", 1.0), 0.0, 1.0)),
            recent_feasible_repair_rate=float(np.clip(repair_rate, 0.0, 1.0)),
            recent_shield_intervention_rate=float(np.clip(shield_rate, 0.0, 1.0)),
            recent_stagnation=float(np.clip(self.alns_diagnostics.noop_iteration_count / max(self.alns_diagnostics.iteration_count, 1), 0.0, 1.0)),
            support_conflict_rate=float(np.clip(self.alns_diagnostics.support_plan_invalidated_count / max(self.alns_diagnostics.support_plan_created_count, 1), 0.0, 1.0)),
            critical_task_ratio=float(np.clip(len(critical) / max(len(pending_tasks), 1), 0.0, 1.0)),
            scenario_scale=float(max(len(agents), 1) / 6.0),
        )

    def _decide_effective_k(self, env, *, objective_before: float = 0.0, objective_after: float = 0.0, accepted: bool = False, improved: bool = False) -> int:
        mode = str(getattr(env.cfg, "adaptive_horizon_mode", "disabled")).strip().lower() or "disabled"
        configured_k = 2 if self._solution_mode_configured(env) in {"k2_shadow", "k2_active"} else 1
        if mode == "disabled":
            self._last_effective_k = int(configured_k)
            return int(configured_k)
        decision = self.adaptive_horizon_controller.decide(self._adaptive_horizon_features(env))
        chosen = int(decision.chosen_k)
        allowed = set(int(x) for x in getattr(env.cfg, "adaptive_horizon_allowed_values", (1, 2)))
        if chosen not in allowed:
            chosen = 2 if 2 in allowed else 1
        effective = int(chosen if mode == "active" else configured_k)
        self._last_effective_k = int(effective)
        rec = {
            "episode": int(getattr(env, "current_episode_index", 0)),
            "step": int(getattr(getattr(env, "state", None), "step_index", 0)),
            "scenario": str(getattr(env.cfg, "scenario", "")),
            "method": str(getattr(env, "current_method", "")),
            "seed": int(getattr(env.cfg, "seed", 0)),
            "protocol": str(getattr(env.cfg, "physical_environment_safety_protocol", "")),
            "mode": mode,
            "chosen_K": int(effective),
            "shadow_K": int(decision.chosen_k),
            "reason_codes": "|".join(decision.reason_codes),
            "risk_features": decision.features.to_dict(),
            "recent_metrics": {
                "repair_feasible_rate": float(decision.features.recent_feasible_repair_rate),
                "shield_intervention_rate": float(decision.features.recent_shield_intervention_rate),
                "stagnation": float(decision.features.recent_stagnation),
            },
            "objective_before": float(objective_before),
            "objective_after": float(objective_after),
            "repair_feasible": bool(self.alns_diagnostics.repair_feasible_count > 0),
            "accepted": bool(accepted),
            "improved": bool(improved),
            "confidence": float(decision.confidence),
            "risk_score": float(decision.risk_score),
        }
        self.adaptive_horizon_records.append(rec)
        self.alns_diagnostics.adaptive_horizon_decision_count += 1
        if effective == 1:
            self.alns_diagnostics.adaptive_horizon_k1_count += 1
        else:
            self.alns_diagnostics.adaptive_horizon_k2_count += 1
        return int(effective)

    def _solution_mode_configured(self, env) -> str:
        return str(getattr(env.cfg, "alns_solution_mode", "legacy_k1")).strip().lower()

    def _agent_sequence_pairs(self, env) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        for aid, st in sorted(getattr(env.state, "agents", {}).items(), key=lambda kv: str(kv[0])):
            kind = getattr(st, "kind", None)
            if kind == AgentKind.TRUCK:
                pairs.append((str(aid), "truck"))
            elif kind == AgentKind.UAV:
                pairs.append((str(aid), "uav"))
        return pairs

    def _refresh_tail_lifetime_metrics(self) -> None:
        lifetimes = list(self.sequence_runtime_state.tail_lifetime_steps)
        if lifetimes:
            self.alns_diagnostics.average_tail_lifetime_steps = float(sum(lifetimes) / len(lifetimes))
            self.alns_diagnostics.median_tail_lifetime_steps = float(statistics.median(lifetimes))
            self.alns_diagnostics.maximum_tail_lifetime_steps = float(max(lifetimes))
        else:
            self.alns_diagnostics.average_tail_lifetime_steps = 0.0
            self.alns_diagnostics.median_tail_lifetime_steps = 0.0
            self.alns_diagnostics.maximum_tail_lifetime_steps = 0.0

    def _reset_k2_runtime_state(self, reason: str) -> None:
        self.sequence_runtime_state.clear()
        self.alns_diagnostics.sequence_runtime_reset_count += 1
        self._refresh_tail_lifetime_metrics()
        self.k2_runtime_sequence_records.append(
            {
                "step": int(self._k2_runtime_last_seen_step if self._k2_runtime_last_seen_step >= 0 else 0),
                "agent_id": "",
                "agent_type": "",
                "event_type": "SEQUENCE_RUNTIME_RESET",
                "event_reason": str(reason),
                "sequence_before": [],
                "sequence_after": [],
                "head_before": None,
                "head_after": None,
                "tail_before": [],
                "tail_after": [],
                "solution_digest_before": None,
                "solution_digest_after": None,
                "trigger": str(reason),
                "validation_result": "reset",
                "reason_codes": "",
            }
        )

    def _reset_k2_runtime_if_needed(self, env) -> None:
        step_now = int(getattr(getattr(env, "state", None), "step_index", 0))
        if self._solution_mode(env) == "legacy_k1":
            if self.sequence_runtime_state.by_agent:
                self._reset_k2_runtime_state("legacy_mode")
            self._k2_runtime_last_seen_step = int(step_now)
            return
        if self._k2_runtime_last_seen_step >= 0 and step_now == 0 and self._k2_runtime_last_seen_step > 0:
            self._reset_k2_runtime_state("episode_reset")
        elif self._k2_runtime_last_seen_step >= 0 and step_now < self._k2_runtime_last_seen_step:
            self._reset_k2_runtime_state("step_rewind")
        self._k2_runtime_last_seen_step = int(step_now)

    def _record_runtime_event(
        self,
        env,
        *,
        agent_id: str,
        agent_type: str,
        event_type: str,
        event_reason: str,
        sequence_before: Tuple[str, ...] | tuple = (),
        sequence_after: Tuple[str, ...] | tuple = (),
        solution_digest_before: str | None = None,
        solution_digest_after: str | None = None,
        trigger: str = "",
        validation_result: str = "",
        reason_codes: Tuple[str, ...] | tuple = (),
    ) -> None:
        before = tuple(str(x) for x in sequence_before)
        after = tuple(str(x) for x in sequence_after)
        self.k2_runtime_sequence_records.append(
            {
                "step": int(getattr(getattr(env, "state", None), "step_index", 0)),
                "agent_id": str(agent_id),
                "agent_type": str(agent_type),
                "event_type": str(event_type),
                "event_reason": str(event_reason),
                "sequence_before": list(before),
                "sequence_after": list(after),
                "head_before": before[0] if before else None,
                "head_after": after[0] if after else None,
                "tail_before": list(before[1:]) if len(before) >= 2 else [],
                "tail_after": list(after[1:]) if len(after) >= 2 else [],
                "solution_digest_before": solution_digest_before,
                "solution_digest_after": solution_digest_after,
                "trigger": str(trigger),
                "validation_result": str(validation_result),
                "reason_codes": "|".join(str(x) for x in reason_codes),
            }
        )

    def export_k2_runtime_sequence_records(self) -> List[Dict[str, Any]]:
        return list(self.k2_runtime_sequence_records)

    def export_k2_sa_delta_records(self) -> List[Dict[str, Any]]:
        return list(self.k2_sa_delta_records)

    def export_canonical_operator_records(self) -> List[Dict[str, Any]]:
        return [{k: v for k, v in rec.items() if k != "_finalized"} for rec in self.canonical_operator_records]

    def export_support_execution_records(self) -> List[Dict[str, Any]]:
        return list(self.support_execution_records)

    def export_operator_weight_trajectory_records(self) -> List[Dict[str, Any]]:
        return list(self.operator_weight_trajectory_records)

    def export_event_trigger_records(self) -> List[Dict[str, Any]]:
        return list(self.event_trigger_records)

    def export_sa_calibration_records(self) -> List[Dict[str, Any]]:
        return list(self.sa_calibration_records)

    def export_live_candidate_records(self) -> List[Dict[str, Any]]:
        return list(self.live_candidate_records)

    def export_ranker_runtime_records(self) -> List[Dict[str, Any]]:
        return list(self.ranker_runtime_records)

    def export_repair_candidate_pool_records(self) -> List[Dict[str, Any]]:
        return list(self.repair_candidate_pool_records)

    def _record_critical_recovery_diagnostics(self, repair_operator: str, repair_result: RepairResult) -> None:
        if str(repair_operator) != "critical_recovery_repair_insertion":
            return
        diag = dict(getattr(repair_result, "diagnostics", {}))
        self.alns_diagnostics.critical_recovery_candidates += int(diag.get("critical_recovery_candidates", 0))
        self.alns_diagnostics.critical_recovery_attempts += int(diag.get("critical_recovery_attempts", 0))
        self.alns_diagnostics.critical_recovery_direct_insertions += int(diag.get("critical_recovery_direct_insertions", 0))
        self.alns_diagnostics.critical_recovery_safe_reorders += int(diag.get("critical_recovery_safe_reorders", 0))
        self.alns_diagnostics.critical_recovery_rejected_infeasible += int(diag.get("critical_recovery_rejected_infeasible", 0))
        self.alns_diagnostics.critical_recovery_rejected_no_slot += int(diag.get("critical_recovery_rejected_no_slot", 0))
        self.alns_diagnostics.critical_recovery_rejected_duplicate_claim += int(diag.get("critical_recovery_rejected_duplicate_claim", 0))
        self.alns_diagnostics.critical_recovery_avoided_failed_agent += int(diag.get("critical_recovery_avoided_failed_agent", 0))
        self.alns_diagnostics.critical_support_rebind_candidates += int(diag.get("critical_support_rebind_candidates", 0))
        self.alns_diagnostics.critical_support_rebind_attempts += int(diag.get("critical_support_rebind_attempts", 0))
        self.alns_diagnostics.critical_support_rebind_historical_reuse += int(diag.get("critical_support_rebind_historical_reuse", 0))
        self.alns_diagnostics.critical_support_rebind_reconstructed += int(diag.get("critical_support_rebind_reconstructed", 0))
        self.alns_diagnostics.critical_support_rebind_rejected_no_truck += int(diag.get("critical_support_rebind_rejected_no_truck", 0))
        self.alns_diagnostics.critical_support_rebind_rejected_no_anchor += int(diag.get("critical_support_rebind_rejected_no_anchor", 0))
        self.alns_diagnostics.critical_support_rebind_rejected_energy += int(diag.get("critical_support_rebind_rejected_energy", 0))
        self.alns_diagnostics.critical_support_rebind_rejected_reserve += int(diag.get("critical_support_rebind_rejected_reserve", 0))
        self.alns_diagnostics.critical_support_rebind_rejected_road += int(diag.get("critical_support_rebind_rejected_road", 0))
        self.alns_diagnostics.critical_support_rebind_rejected_infeasible += int(diag.get("critical_support_rebind_rejected_infeasible", 0))
        self.alns_diagnostics.critical_support_rebind_accept_count += int(diag.get("critical_support_rebind_accept_count", 0))
        self.alns_diagnostics.critical_support_rebind_failed_binding_penalized += int(diag.get("critical_support_rebind_failed_binding_penalized", 0))
        self.alns_diagnostics.critical_support_rebind_failed_binding_skipped += int(diag.get("critical_support_rebind_failed_binding_skipped", 0))
        self.alns_diagnostics.critical_support_rebind_best_accepted_margin_m = max(
            float(self.alns_diagnostics.critical_support_rebind_best_accepted_margin_m),
            float(diag.get("critical_support_rebind_best_accepted_margin_m", float("-inf"))),
        )
        self.alns_diagnostics.critical_support_rebind_best_rejected_margin_m = max(
            float(self.alns_diagnostics.critical_support_rebind_best_rejected_margin_m),
            float(diag.get("critical_support_rebind_best_rejected_margin_m", float("-inf"))),
        )
        self.alns_diagnostics.critical_support_rebind_best_accepted_battery_margin = max(
            float(self.alns_diagnostics.critical_support_rebind_best_accepted_battery_margin),
            float(diag.get("critical_support_rebind_best_accepted_battery_margin", float("-inf"))),
        )
        self.alns_diagnostics.critical_support_rebind_best_rejected_battery_margin = max(
            float(self.alns_diagnostics.critical_support_rebind_best_rejected_battery_margin),
            float(diag.get("critical_support_rebind_best_rejected_battery_margin", float("-inf"))),
        )
        self.alns_diagnostics.lc_critical_recovery_path_candidates += int(diag.get("lc_critical_recovery_path_candidates", 0))
        self.alns_diagnostics.lc_critical_recovery_path_attempts += int(diag.get("lc_critical_recovery_path_attempts", 0))
        self.alns_diagnostics.lc_critical_recovery_path_successes += int(diag.get("lc_critical_recovery_path_successes", 0))
        self.alns_diagnostics.lc_critical_recovery_path_rejected_insufficient_margin += int(
            diag.get("lc_critical_recovery_path_rejected_insufficient_margin", 0)
        )
        self.alns_diagnostics.lc_critical_recovery_path_rejected_no_bindable_truck += int(
            diag.get("lc_critical_recovery_path_rejected_no_bindable_truck", 0)
        )
        self.alns_diagnostics.lc_critical_recovery_path_rejected_uav_not_docked += int(
            diag.get("lc_critical_recovery_path_rejected_uav_not_docked", 0)
        )
        self.alns_diagnostics.lc_critical_recovery_path_rejected_no_sequence_capacity += int(
            diag.get("lc_critical_recovery_path_rejected_no_sequence_capacity", 0)
        )
        self.alns_diagnostics.lc_critical_recovery_path_rejected_augmented_infeasible += int(
            diag.get("lc_critical_recovery_path_rejected_augmented_infeasible", 0)
        )
        self.alns_diagnostics.lc_critical_recovery_path_trucks_considered += int(
            diag.get("lc_critical_recovery_path_trucks_considered", 0)
        )
        self.alns_diagnostics.lc_critical_recovery_path_best_margin = max(
            float(self.alns_diagnostics.lc_critical_recovery_path_best_margin),
            float(diag.get("lc_critical_recovery_path_best_margin", float("-inf"))),
        )
        self.alns_diagnostics.lc_critical_recovery_path_success_margin = max(
            float(self.alns_diagnostics.lc_critical_recovery_path_success_margin),
            float(diag.get("lc_critical_recovery_path_success_margin", float("-inf"))),
        )
        self.alns_diagnostics.lc_critical_recovery_path_failed_tuple_avoided += int(
            diag.get("lc_critical_recovery_path_failed_tuple_avoided", 0)
        )
        self.alns_diagnostics.assigned_critical_reconstruct_candidates += int(
            diag.get("assigned_critical_reconstruct_candidates", 0)
        )
        self.alns_diagnostics.assigned_critical_reconstruct_path_candidates += int(
            diag.get("assigned_critical_reconstruct_path_candidates", 0)
        )
        self.alns_diagnostics.assigned_critical_reconstruct_trucks_considered += int(
            diag.get("assigned_critical_reconstruct_trucks_considered", 0)
        )
        self.alns_diagnostics.assigned_critical_reconstruct_margin_probed += int(
            diag.get("assigned_critical_reconstruct_margin_probed", 0)
        )
        self.alns_diagnostics.assigned_critical_reconstruct_positive_margin_count += int(
            diag.get("assigned_critical_reconstruct_positive_margin_count", 0)
        )
        self.alns_diagnostics.assigned_critical_reconstruct_selected_path_count += int(
            diag.get("assigned_critical_reconstruct_selected_path_count", 0)
        )
        self.alns_diagnostics.assigned_critical_reconstruct_success_count += int(
            diag.get("assigned_critical_reconstruct_success_count", 0)
        )
        self.alns_diagnostics.assigned_critical_reconstruct_rejected_no_bindable_truck += int(
            diag.get("assigned_critical_reconstruct_rejected_no_bindable_truck", 0)
        )
        self.alns_diagnostics.assigned_critical_reconstruct_rejected_insufficient_margin += int(
            diag.get("assigned_critical_reconstruct_rejected_insufficient_margin", 0)
        )
        self.alns_diagnostics.assigned_critical_reconstruct_rejected_no_sequence_capacity += int(
            diag.get("assigned_critical_reconstruct_rejected_no_sequence_capacity", 0)
        )
        self.alns_diagnostics.assigned_critical_reconstruct_rejected_target_unreachable += int(
            diag.get("assigned_critical_reconstruct_rejected_target_unreachable", 0)
        )
        self.alns_diagnostics.assigned_critical_reconstruct_rejected_uav_not_docked += int(
            diag.get("assigned_critical_reconstruct_rejected_uav_not_docked", 0)
        )
        self.alns_diagnostics.assigned_critical_reconstruct_rejected_infeasible += int(
            diag.get("assigned_critical_reconstruct_rejected_infeasible", 0)
        )
        self.alns_diagnostics.assigned_critical_reconstruct_no_progress_tasks_targeted += int(
            diag.get("assigned_critical_reconstruct_no_progress_tasks_targeted", 0)
        )
    def _critical_support_rebind_enabled(self, env) -> bool:
        return bool(getattr(env.cfg, "alns_critical_support_rebind_enabled", False))

    def _task_support_binding(self, solution, task_id: str) -> SupportBinding | None:
        for binding in tuple(getattr(solution, "support_bindings", ())):
            if str(getattr(binding, "task_id", "")) == str(task_id):
                return binding
        return None

    def _latest_historical_support_binding(self, task_id: str) -> SupportBinding | None:
        for binding in reversed(tuple(getattr(self, "_installed_support_bindings", ()))):
            if str(getattr(binding, "task_id", "")) == str(task_id):
                return binding
        for row in reversed(list(getattr(self, "support_execution_records", []))):
            if str(row.get("task_id", "")) != str(task_id):
                continue
            truck_id = str(row.get("truck_id", ""))
            uav_id = str(row.get("uav_id", ""))
            launch_anchor = row.get("launch_anchor", None)
            recovery_anchor = row.get("recovery_anchor", None)
            if truck_id and uav_id and launch_anchor is not None and recovery_anchor is not None:
                return SupportBinding(
                    uav_id=uav_id,
                    truck_id=truck_id,
                    task_id=str(task_id),
                    launch_anchor=int(launch_anchor),
                    recovery_anchor=int(recovery_anchor),
                )
        return None

    def _support_rebind_rejection_bucket(self, reason: str) -> str:
        text = str(reason).upper()
        if any(token in text for token in ("NOT_DOCKED", "SUPPORT_BINDING_STALE", "NO_TRUCK", "NOT_ON_TRUCK")):
            return "critical_support_rebind_rejected_no_truck"
        if "ANCHOR" in text:
            return "critical_support_rebind_rejected_no_anchor"
        if "RESERVE" in text or "RECOVERY" in text:
            return "critical_support_rebind_rejected_reserve"
        if "ENERGY" in text or "BATTERY" in text:
            return "critical_support_rebind_rejected_energy"
        if any(token in text for token in ("ROAD", "PATH", "UNREACH")):
            return "critical_support_rebind_rejected_road"
        return "critical_support_rebind_rejected_infeasible"

    def _support_rebind_failure_key(self, *, task_id: str, truck_id: str, uav_id: str, launch_anchor: int, recovery_anchor: int) -> Tuple[str, str, str, int, int]:
        return (str(task_id), str(truck_id), str(uav_id), int(launch_anchor), int(recovery_anchor))

    def _support_rebind_failure_count(self, *, task_id: str, truck_id: str, uav_id: str, launch_anchor: int, recovery_anchor: int) -> int:
        key = self._support_rebind_failure_key(
            task_id=str(task_id),
            truck_id=str(truck_id),
            uav_id=str(uav_id),
            launch_anchor=int(launch_anchor),
            recovery_anchor=int(recovery_anchor),
        )
        return int(self._support_rebind_failure_counts.get(key, 0))

    def _support_rebind_success_count(self, *, task_id: str, truck_id: str, uav_id: str, launch_anchor: int, recovery_anchor: int) -> int:
        key = self._support_rebind_failure_key(
            task_id=str(task_id),
            truck_id=str(truck_id),
            uav_id=str(uav_id),
            launch_anchor=int(launch_anchor),
            recovery_anchor=int(recovery_anchor),
        )
        return int(self._support_rebind_success_counts.get(key, 0))

    def _record_task_assignment_observations(self, goals: Dict[str, Optional[str]], step_now: int) -> None:
        for aid, task_id in dict(goals).items():
            del aid
            if task_id is None:
                continue
            tid = str(task_id)
            self._task_assignment_counts[tid] = int(self._task_assignment_counts.get(tid, 0) + 1)
            if tid not in self._task_first_assigned_step:
                self._task_first_assigned_step[tid] = int(step_now)
            self._task_last_assigned_step[tid] = int(step_now)

    def _task_assignment_count(self, task_id: str) -> int:
        return int(self._task_assignment_counts.get(str(task_id), 0))

    def _task_assignment_span(self, task_id: str) -> int:
        tid = str(task_id)
        if tid not in self._task_first_assigned_step or tid not in self._task_last_assigned_step:
            return 0
        return int(max(self._task_last_assigned_step[tid] - self._task_first_assigned_step[tid], 0))

    def _lc_critical_recovery_path_enabled(self, env) -> bool:
        return bool(getattr(env.cfg, "alns_lc_critical_recovery_path_enabled", False))

    def _assigned_critical_reconstruct_enabled(self, env) -> bool:
        return bool(getattr(env.cfg, "alns_assigned_critical_reconstruct_enabled", False))

    def _support_reposition_shadow_enabled(self, env) -> bool:
        return bool(getattr(env.cfg, "alns_support_reposition_shadow_enabled", False))

    def _task_service_started(self, env, task_id: str) -> bool:
        task = getattr(env.state, "tasks", {}).get(str(task_id), None)
        if task is None:
            return False
        return bool(
            getattr(task, "first_service_step", None) is not None
            or getattr(task, "service_start_step", None) is not None
        )


    def _lc_critical_recovery_count(self, *, task_id: str, truck_id: str, uav_id: str, launch_anchor: int, recovery_anchor: int) -> int:
        key = self._support_rebind_failure_key(
            task_id=str(task_id),
            truck_id=str(truck_id),
            uav_id=str(uav_id),
            launch_anchor=int(launch_anchor),
            recovery_anchor=int(recovery_anchor),
        )
        return int(self._lc_critical_recovery_failure_counts.get(key, 0))

    def _lc_critical_recovery_success_count(self, *, task_id: str, truck_id: str, uav_id: str, launch_anchor: int, recovery_anchor: int) -> int:
        key = self._support_rebind_failure_key(
            task_id=str(task_id),
            truck_id=str(truck_id),
            uav_id=str(uav_id),
            launch_anchor=int(launch_anchor),
            recovery_anchor=int(recovery_anchor),
        )
        return int(self._lc_critical_recovery_success_counts.get(key, 0))

    def _task_support_failure_pressure(self, task_id: str) -> int:
        tid = str(task_id)
        total = 0
        for key, count in self._support_rebind_failure_counts.items():
            if str(key[0]) == tid:
                total += int(count)
        for key, count in self._lc_critical_recovery_failure_counts.items():
            if str(key[0]) == tid:
                total += int(count)
        return int(total)

    def _lc_critical_recovery_rejection_bucket(self, reason: str) -> str:
        text = str(reason).upper()
        if any(token in text for token in ("NO_BINDABLE_TRUCK", "NO_TRUCK", "SUPPORT_BINDING_MISSING")):
            return "lc_critical_recovery_path_rejected_no_bindable_truck"
        if "NOT_DOCKED" in text:
            return "lc_critical_recovery_path_rejected_uav_not_docked"
        if "NO_SEQUENCE_CAPACITY" in text:
            return "lc_critical_recovery_path_rejected_no_sequence_capacity"
        if any(token in text for token in ("RECOVERY_MARGIN", "RESERVE", "RECOVERY_NOT_FEASIBLE")):
            return "lc_critical_recovery_path_rejected_insufficient_margin"
        if "AUGMENTED_INFEASIBLE" in text or "SUPPORT_BINDING_CONFLICT" in text:
            return "lc_critical_recovery_path_rejected_augmented_infeasible"
        return "lc_critical_recovery_path_rejected_augmented_infeasible"

    def _lc_critical_recovery_priority_key(self, env, solution, task_id: str) -> tuple[float, float, float, float, str]:
        del solution
        task = getattr(env.state, "tasks", {}).get(str(task_id), None)
        if task is None:
            return (1.0, 0.0, 0.0, 0.0, str(task_id))
        critical_rank = 0.0 if bool(getattr(task, "kind", None) == TaskKind.EMERGENCY or self._is_timecritical_lightweight_task(task)) else 1.0
        assignment_count = -float(self._task_assignment_count(str(task_id)))
        failure_pressure = -float(self._task_support_failure_pressure(str(task_id)))
        urgency = -(1.0 - float(np.clip(self._task_lifeline_ratio(task), 0.0, 1.0)))
        return (critical_rank, assignment_count, failure_pressure, urgency, str(task_id))

    def _lc_critical_recovery_candidate_task_ids(self, env, solution, repair_result: RepairResult) -> list[str]:
        max_tasks = int(max(getattr(env.cfg, "alns_lc_critical_recovery_path_max_tasks", 3), 0))
        min_assigned = int(max(getattr(env.cfg, "alns_lc_critical_recovery_path_min_assigned_count", 20), 0))
        critical_only = bool(getattr(env.cfg, "alns_lc_critical_recovery_path_target_critical_only", True))
        seeds: list[str] = [str(tid) for tid in getattr(repair_result, "diagnostics", {}).get("critical_recovery_task_ids", ()) if str(tid)]
        for aid, agent_type in self._agent_sequence_pairs_from_solution(solution):
            if str(agent_type) != "truck":
                continue
            seq = tuple(str(x) for x in solution.sequence_for(aid, agent_type))
            for task_id in seq:
                if task_id:
                    seeds.append(str(task_id))
        seen: set[str] = set()
        candidates: list[str] = []
        for task_id in seeds:
            tid = str(task_id)
            if not tid or tid in seen:
                continue
            seen.add(tid)
            task = getattr(env.state, "tasks", {}).get(tid, None)
            if task is None or getattr(task, "status", None) not in {TaskStatus.PENDING, TaskStatus.FAILED}:
                continue
            is_critical = bool(getattr(task, "kind", None) == TaskKind.EMERGENCY or self._is_timecritical_lightweight_task(task))
            if critical_only and not is_critical:
                continue
            if self._task_support_binding(solution, tid) is not None:
                continue
            owner_truck = ""
            seq_position = -1
            for aid, agent_type in self._agent_sequence_pairs_from_solution(solution):
                if str(agent_type) != "truck":
                    continue
                seq = tuple(str(x) for x in solution.sequence_for(aid, agent_type))
                if tid in seq:
                    owner_truck = str(aid)
                    seq_position = seq.index(tid)
                    break
            if not owner_truck:
                continue
            if bool(self._truck_task_reachable(env, owner_truck, task)) and bool(self._truck_task_direct_serviceable(env, owner_truck, task)):
                continue
            assignment_count = self._task_assignment_count(tid)
            support_pressure = self._task_support_failure_pressure(tid)
            if assignment_count < min_assigned and support_pressure <= 0:
                continue
            if seq_position == 0 and bool(self._goal_is_protected(env, owner_truck, task)):
                continue
            candidates.append(tid)
        candidates.sort(key=lambda tid: self._lc_critical_recovery_priority_key(env, solution, tid))
        return candidates[:max_tasks] if max_tasks > 0 else []

    def _assigned_critical_reconstruct_priority_key(self, env, solution, task_id: str) -> tuple[float, float, float, float, str]:
        return self._lc_critical_recovery_priority_key(env, solution, task_id)

    def _assigned_critical_reconstruct_candidate_task_ids(self, env, solution, repair_result: RepairResult) -> list[str]:
        max_tasks = int(max(getattr(env.cfg, "alns_assigned_critical_reconstruct_max_tasks", 3), 0))
        min_assigned = int(max(getattr(env.cfg, "alns_assigned_critical_reconstruct_min_assigned_count", 20), 0))
        critical_only = bool(getattr(env.cfg, "alns_assigned_critical_reconstruct_target_critical_only", True))
        seeds: list[str] = [str(tid) for tid in getattr(repair_result, "diagnostics", {}).get("critical_recovery_task_ids", ()) if str(tid)]
        for aid, agent_type in self._agent_sequence_pairs_from_solution(solution):
            if str(agent_type) != "truck":
                continue
            for task_id in tuple(str(x) for x in solution.sequence_for(aid, agent_type)):
                if task_id:
                    seeds.append(str(task_id))
        for task_id, task in getattr(env.state, "tasks", {}).items():
            if task is None:
                continue
            tid = str(task_id)
            if self._task_assignment_count(tid) >= min_assigned:
                seeds.append(tid)
        seen: set[str] = set()
        candidates: list[str] = []
        for task_id in seeds:
            tid = str(task_id)
            if not tid or tid in seen:
                continue
            seen.add(tid)
            task = getattr(env.state, "tasks", {}).get(tid, None)
            if task is None or bool(getattr(task, "completed", False)):
                continue
            if getattr(task, "status", None) not in {TaskStatus.PENDING, TaskStatus.FAILED}:
                continue
            is_critical = bool(getattr(task, "kind", None) == TaskKind.EMERGENCY or self._is_timecritical_lightweight_task(task))
            if critical_only and not is_critical:
                continue
            if self._task_assignment_count(tid) < min_assigned:
                continue
            if self._task_service_started(env, tid):
                continue
            candidates.append(tid)
        candidates.sort(key=lambda tid: self._assigned_critical_reconstruct_priority_key(env, solution, tid))
        return candidates[:max_tasks] if max_tasks > 0 else []

    def _support_rebind_penalty_mode(self, env) -> str:
        return str(getattr(env.cfg, "alns_support_rebind_failed_binding_penalty", "mild")).strip().lower() or "mild"

    def _support_rebind_priority_key(self, env, solution, task_id: str) -> tuple[float, float, float, float, str]:
        task = getattr(env.state, "tasks", {}).get(str(task_id), None)
        if task is None:
            return (1.0, 1.0, 1.0, 1.0, str(task_id))
        is_critical = 0.0 if bool(getattr(task, "kind", None) == TaskKind.EMERGENCY or self._is_timecritical_lightweight_task(task)) else 1.0
        lifeline_ratio = float(np.clip(self._task_lifeline_ratio(task), 0.0, 1.0))
        urgency = float(getattr(task, "urgency_score", getattr(task, "priority", 0.0)) or 0.0)
        owner_penalty = 1.0
        for aid, agent_type in self._agent_sequence_pairs_from_solution(solution):
            if str(task_id) in tuple(str(x) for x in solution.sequence_for(aid, agent_type)):
                owner_penalty = 0.0
                break
        return (is_critical, lifeline_ratio, -urgency, owner_penalty, str(task_id))

    def _support_rebind_override_diagnostics(
        self,
        env,
        *,
        task,
        truck_id: str,
        uav_id: str,
        launch_anchor: int,
        launchable_now: bool,
    ) -> dict[str, Any]:
        diag: dict[str, Any] = {
            "override_reason": "",
            "predicted_recovery_margin_m": float("-inf"),
            "predicted_battery_margin_ratio": float("-inf"),
            "predicted_launchable": 0.0,
            "predicted_full_sortie_feasible": 0.0,
            "predicted_lifeline_remaining": 0.0,
        }
        override_fn = getattr(env, "_is_tc_override_delivery_feasible", None)
        truck_st = getattr(env.state, "agents", {}).get(str(truck_id), None)
        if not callable(override_fn) or task is None or truck_st is None:
            return diag
        try:
            ok, reason, extra = override_fn(
                str(truck_id),
                str(uav_id),
                task,
                int(launch_anchor),
                0.0,
                0.0,
                bool(launchable_now),
            )
        except Exception:
            return diag
        extra = dict(extra or {})
        diag.update(extra)
        diag["override_reason"] = str(reason or "")
        diag["override_feasible"] = bool(ok)
        diag["predicted_recovery_margin_m"] = float(extra.get("predicted_recovery_margin_m", float("-inf")))
        diag["predicted_battery_margin_ratio"] = float(extra.get("predicted_battery_margin_ratio", float("-inf")))
        diag["predicted_launchable"] = float(extra.get("predicted_launchable", 0.0))
        diag["predicted_full_sortie_feasible"] = 1.0 if bool(ok) else float(extra.get("predicted_full_sortie_feasible", 0.0))
        diag["predicted_lifeline_remaining"] = float(extra.get("predicted_lifeline_remaining", 0.0))
        return diag

    @staticmethod
    def _override_margin_status(*, predicted_margin: float, override_reason: str) -> str:
        if np.isfinite(predicted_margin):
            return "finite_positive" if predicted_margin > 0.0 else "finite_nonpositive"
        if np.isposinf(predicted_margin):
            return "pos_inf"
        if override_reason:
            return "not_computed"
        return "neg_inf"

    def _assigned_reconstruct_margin_reject_reason(
        self,
        *,
        override_feasible: bool,
        override_reason: str,
        predicted_margin: float,
        predicted_battery_margin: float,
    ) -> str:
        margin_status = self._override_margin_status(
            predicted_margin=predicted_margin,
            override_reason=override_reason,
        )
        if margin_status == "finite_nonpositive":
            return "INSUFFICIENT_RECOVERY_MARGIN"
        if predicted_battery_margin < 0.0:
            return "LOW_BATTERY_MARGIN"
        if not override_feasible:
            if margin_status in {"neg_inf", "not_computed"}:
                return "RECOVERY_MARGIN_NOT_COMPUTED" if override_reason else "MARGIN_DIAGNOSTIC_MISSING"
            if override_reason:
                return str(override_reason).strip().upper()
            return "MARGIN_DIAGNOSTIC_MISSING"
        if margin_status in {"neg_inf", "not_computed"}:
            return "MARGIN_DIAGNOSTIC_MISSING"
        return ""

    def _support_rebind_anchor_candidates(
        self,
        env,
        *,
        task,
        truck_id: str,
        launch_anchor: int,
        historical_recovery: int | None,
    ) -> tuple[int, ...]:
        candidates: list[int] = []
        if historical_recovery is not None:
            candidates.append(int(historical_recovery))
        task_node = int(getattr(task, "demand_node", launch_anchor) or launch_anchor)
        candidates.append(task_node)
        candidates.append(int(launch_anchor))
        for row in reversed(list(getattr(self, "support_execution_records", []))):
            if str(row.get("truck_id", "")) != str(truck_id):
                continue
            recovery_anchor = row.get("recovery_anchor", None)
            if recovery_anchor is None:
                continue
            try:
                candidates.append(int(recovery_anchor))
            except Exception:
                continue
            if len(candidates) >= 5:
                break
        seen: set[int] = set()
        ordered: list[int] = []
        for anchor in candidates:
            if int(anchor) in seen:
                continue
            seen.add(int(anchor))
            ordered.append(int(anchor))
        return tuple(ordered)

    def _support_rebind_anchor_metrics(self, env, *, task, truck_id: str, launch_anchor: int, recovery_anchor: int) -> dict[str, Any]:
        distance_fn = getattr(env, "_decision_shortest_path_distance", None)
        task_node = int(getattr(task, "demand_node", launch_anchor) or launch_anchor)
        road_distance = float("inf")
        try:
            if callable(distance_fn):
                road_distance = float(distance_fn(int(launch_anchor), int(recovery_anchor)))
            else:
                road_distance = float(abs(int(launch_anchor) - int(recovery_anchor)))
        except Exception:
            road_distance = float("inf")
        task_to_recovery = float("inf")
        try:
            if callable(distance_fn):
                task_to_recovery = float(distance_fn(int(task_node), int(recovery_anchor)))
            else:
                task_to_recovery = float(abs(int(task_node) - int(recovery_anchor)))
        except Exception:
            task_to_recovery = float("inf")
        historical_success = 0
        for row in list(getattr(self, "support_execution_records", [])):
            if str(row.get("truck_id", "")) != str(truck_id):
                continue
            if int(row.get("recovery_anchor", -1)) != int(recovery_anchor):
                continue
            if str(row.get("status", "")) == "COMPLETED":
                historical_success += 1
        return {
            "anchor_reachable": bool(np.isfinite(task_to_recovery)),
            "anchor_road_distance_m": float(task_to_recovery if np.isfinite(task_to_recovery) else 1e12),
            "launch_to_recovery_distance_m": float(road_distance if np.isfinite(road_distance) else 1e12),
            "anchor_historical_success_count": int(historical_success),
        }

    def _support_rebind_candidate_rows(self, env, solution, task_id: str, *, limit_override: Optional[int] = None) -> list[dict[str, Any]]:
        task = getattr(env.state, "tasks", {}).get(str(task_id), None)
        if task is None:
            return []
        rows: list[dict[str, Any]] = []
        prefer_historical = bool(getattr(env.cfg, "alns_critical_support_rebind_prefer_historical_binding", True))
        allow_nearest = bool(getattr(env.cfg, "alns_critical_support_rebind_allow_nearest_feasible_truck", True))
        margin_aware = bool(getattr(env.cfg, "alns_support_rebind_margin_aware_enabled", False))
        anchor_ranked = bool(getattr(env.cfg, "alns_support_rebind_anchor_ranking_enabled", False))
        failed_binding_avoidance = bool(getattr(env.cfg, "alns_support_rebind_failed_binding_avoidance_enabled", False))
        safe_guard = bool(getattr(env.cfg, "alns_support_rebind_safe_uav_guard_enabled", False))
        penalty_mode = self._support_rebind_penalty_mode(env)
        margin_top_k = int(max(limit_override if limit_override is not None else getattr(env.cfg, "alns_support_rebind_margin_top_k", 3), 1))
        historical = self._latest_historical_support_binding(task_id) if prefer_historical else None
        historical_truck = str(getattr(historical, "truck_id", "")) if historical is not None else ""
        historical_uav = str(getattr(historical, "uav_id", "")) if historical is not None else ""
        historical_recovery = getattr(historical, "recovery_anchor", None) if historical is not None else None

        truck_ids: list[str] = []
        if historical_truck:
            truck_ids.append(historical_truck)
        owner_truck = ""
        for aid, agent_type in self._agent_sequence_pairs_from_solution(solution):
            seq = tuple(str(x) for x in solution.sequence_for(aid, agent_type))
            if str(task_id) in seq and str(agent_type) == "truck":
                owner_truck = str(aid)
                break
        if owner_truck and owner_truck not in truck_ids:
            truck_ids.append(owner_truck)
        if allow_nearest:
            other_trucks = sorted(
                [
                    str(aid)
                    for aid, st in getattr(env.state, "agents", {}).items()
                    if getattr(st, "kind", None) == AgentKind.TRUCK and not bool(getattr(st, "crashed", False))
                ],
                key=lambda aid: (
                    float(self._truck_task_distance(env, str(aid), task)),
                    str(aid),
                ),
            )
            for aid in other_trucks:
                if aid not in truck_ids:
                    truck_ids.append(aid)
        for truck_id in truck_ids:
            truck_st = getattr(env.state, "agents", {}).get(str(truck_id), None)
            if truck_st is None or getattr(truck_st, "kind", None) != AgentKind.TRUCK or bool(getattr(truck_st, "crashed", False)):
                continue
            docked_uavs = sorted(
                [
                    str(aid)
                    for aid, st in getattr(env.state, "agents", {}).items()
                    if getattr(st, "kind", None) == AgentKind.UAV
                    and not bool(getattr(st, "crashed", False))
                    and str(getattr(st, "follow_target", "")) == str(truck_id)
                ],
                key=str,
            )
            gain_info = self._support_anchor_service_gain(env, str(truck_id), task)
            bind_info = self._support_bound_delivery_info(env, str(truck_id), task, gain_info=gain_info)
            launch_anchor = int(getattr(truck_st, "node", 0) or 0)
            recovery_anchors = self._support_rebind_anchor_candidates(
                env,
                task=task,
                truck_id=str(truck_id),
                launch_anchor=int(launch_anchor),
                historical_recovery=(int(historical_recovery) if historical is not None and str(truck_id) == historical_truck and historical_recovery is not None else None),
            )
            strong_binding = bool(bind_info.get("bound_any", 0.0)) and bool(
                self._support_binding_is_strong_enough(env, task, bind_info, gain_info=gain_info)
            )
            candidate_uavs: list[tuple[str, str]] = []
            if strong_binding:
                uav_id = str(bind_info.get("bound_timecritical_uav_id", "") or historical_uav)
                if uav_id:
                    candidate_uavs.append(
                        (
                            str(uav_id),
                            "historical" if historical is not None and str(truck_id) == historical_truck and str(uav_id) == historical_uav else "reconstructed",
                        )
                    )
            if allow_nearest:
                for docked_uav in docked_uavs:
                    if all(str(docked_uav) != str(existing_uav) for existing_uav, _src in candidate_uavs):
                        candidate_uavs.append((str(docked_uav), "fallback_docked"))
            for uav_id, source in candidate_uavs:
                selected_anchors = recovery_anchors if anchor_ranked else recovery_anchors[:1]
                radius_factor = float(max(getattr(env.cfg, "alns_support_rebind_anchor_search_radius_factor", 1.0), 0.1))
                for recovery_anchor in selected_anchors:
                    anchor_metrics = self._support_rebind_anchor_metrics(
                        env,
                        task=task,
                        truck_id=str(truck_id),
                        launch_anchor=int(launch_anchor),
                        recovery_anchor=int(recovery_anchor),
                    )
                    if anchor_ranked and float(anchor_metrics.get("anchor_road_distance_m", float("inf"))) > float(self._truck_task_distance(env, str(truck_id), task)) * radius_factor + 1e-9:
                        continue
                    failure_count = self._support_rebind_failure_count(
                        task_id=str(task_id),
                        truck_id=str(truck_id),
                        uav_id=str(uav_id),
                        launch_anchor=int(launch_anchor),
                        recovery_anchor=int(recovery_anchor),
                    )
                    success_count = self._support_rebind_success_count(
                        task_id=str(task_id),
                        truck_id=str(truck_id),
                        uav_id=str(uav_id),
                        launch_anchor=int(launch_anchor),
                        recovery_anchor=int(recovery_anchor),
                    )
                    launch_gate_reason = ""
                    launchable_now = False
                    launch_gate = getattr(env, "_uav_launch_gate_check", None)
                    if callable(launch_gate):
                        try:
                            launchable_now, launch_gate_reason, _force = launch_gate(str(uav_id), task=task, count_reject=False)
                        except TypeError:
                            launchable_now, launch_gate_reason, _force = launch_gate(str(uav_id), task=task)
                    override_diag = self._support_rebind_override_diagnostics(
                        env,
                        task=task,
                        truck_id=str(truck_id),
                        uav_id=str(uav_id),
                        launch_anchor=int(launch_anchor),
                        launchable_now=bool(launchable_now),
                    )
                    row = {
                        "task_id": str(task_id),
                        "truck_id": str(truck_id),
                        "uav_id": str(uav_id),
                        "launch_anchor": int(launch_anchor),
                        "recovery_anchor": int(recovery_anchor),
                        "source": str(source),
                        "bound_eta_steps": float(bind_info.get("bound_eta_steps", float("inf"))),
                        "distance_to_task_m": float(self._truck_task_distance(env, str(truck_id), task)),
                        "launch_gate_reason": str(launch_gate_reason or ""),
                        "launchable_now": bool(launchable_now),
                        "failed_binding_count": int(failure_count),
                        "failure_skip_recommended": bool(failed_binding_avoidance and penalty_mode == "medium" and failure_count >= 3),
                        "historical_success_count": int(success_count + int(anchor_metrics.get("anchor_historical_success_count", 0))),
                        "failed_binding_penalty_applied": int(failure_count > 0 and failed_binding_avoidance),
                        **anchor_metrics,
                        **override_diag,
                    }
                    predicted_margin = float(row.get("predicted_recovery_margin_m", float("-inf")))
                    predicted_battery_margin = float(row.get("predicted_battery_margin_ratio", float("-inf")))
                    if (
                        safe_guard
                        and margin_aware
                        and (
                            not bool(row.get("override_feasible", False))
                            or predicted_margin < 0.0
                            or predicted_battery_margin < 0.0
                        )
                    ):
                        row["guard_recommended_reject"] = True
                    else:
                        row["guard_recommended_reject"] = False
                    rows.append(row)
        rows = sorted(
            rows,
            key=lambda row: (
                1 if bool(row.get("failure_skip_recommended", False)) else 0,
                0 if (not margin_aware or bool(row.get("override_feasible", False))) else 1,
                0 if (not margin_aware or float(row.get("predicted_recovery_margin_m", float("-inf"))) >= 0.0) else 1,
                -float(row.get("predicted_recovery_margin_m", float("-inf")) if margin_aware else 0.0),
                -float(row.get("predicted_battery_margin_ratio", float("-inf")) if margin_aware else 0.0),
                0 if bool(row.get("anchor_reachable", True)) else 1,
                int(row.get("failed_binding_count", 0)) if failed_binding_avoidance else 0,
                0 if str(row.get("source", "")) == "historical" else 1 if str(row.get("source", "")) == "reconstructed" else 2,
                -int(row.get("historical_success_count", 0)),
                float(row.get("anchor_road_distance_m", float("inf")) if anchor_ranked else row.get("distance_to_task_m", float("inf"))),
                float(row.get("launch_to_recovery_distance_m", float("inf")) if anchor_ranked else row.get("bound_eta_steps", float("inf"))),
                float(row.get("bound_eta_steps", float("inf"))),
                float(row.get("distance_to_task_m", float("inf"))),
                str(row.get("truck_id", "")),
                str(row.get("uav_id", "")),
                int(row.get("launch_anchor", -1)),
                int(row.get("recovery_anchor", -1)),
            ),
        )
        if failed_binding_avoidance and penalty_mode == "medium":
            preferred = [row for row in rows if not bool(row.get("failure_skip_recommended", False))]
            fallback = [row for row in rows if bool(row.get("failure_skip_recommended", False))]
            rows = preferred[:margin_top_k]
            if len(rows) < margin_top_k:
                rows.extend(fallback[: margin_top_k - len(rows)])
            return rows
        return rows[:margin_top_k]

    def _lc_critical_recovery_candidate_rows(self, env, solution, task_id: str) -> tuple[list[dict[str, Any]], int]:
        top_k = int(max(getattr(env.cfg, "alns_lc_critical_recovery_path_top_k_bindings", 8), 1))
        require_positive_margin = bool(getattr(env.cfg, "alns_lc_critical_recovery_path_require_positive_margin", True))
        avoid_failed_tuple = bool(getattr(env.cfg, "alns_lc_critical_recovery_path_avoid_repeated_failed_tuple", True))
        prioritize_no_bindable = bool(getattr(env.cfg, "alns_lc_critical_recovery_path_prioritize_no_bindable_truck", True))
        rows = [dict(row) for row in self._support_rebind_candidate_rows(env, solution, task_id, limit_override=top_k)]
        unique_trucks = {str(row.get("truck_id", "")) for row in rows if str(row.get("truck_id", ""))}
        for row in rows:
            predicted_margin = float(row.get("predicted_recovery_margin_m", float("-inf")))
            predicted_battery_margin = float(row.get("predicted_battery_margin_ratio", float("-inf")))
            tuple_failures = self._lc_critical_recovery_count(
                task_id=str(task_id),
                truck_id=str(row.get("truck_id", "")),
                uav_id=str(row.get("uav_id", "")),
                launch_anchor=int(row.get("launch_anchor", -1)),
                recovery_anchor=int(row.get("recovery_anchor", -1)),
            )
            row["lc_failed_tuple_count"] = int(tuple_failures)
            row["lc_positive_margin_ok"] = bool(
                (not require_positive_margin)
                or (np.isfinite(predicted_margin) and predicted_margin > 0.0 and predicted_battery_margin >= 0.0)
            )
            row["lc_failed_tuple_skip_recommended"] = bool(
                avoid_failed_tuple and int(tuple_failures) >= 2
            )
            row["lc_priority_no_bindable_hint"] = 0 if (prioritize_no_bindable and str(row.get("source", "")) != "historical") else 1
        rows.sort(
            key=lambda row: (
                1 if bool(row.get("lc_failed_tuple_skip_recommended", False)) else 0,
                0 if bool(row.get("lc_positive_margin_ok", False)) else 1,
                int(row.get("lc_priority_no_bindable_hint", 1)),
                -float(row.get("predicted_recovery_margin_m", float("-inf"))),
                -float(row.get("predicted_battery_margin_ratio", float("-inf"))),
                int(row.get("lc_failed_tuple_count", 0)),
                float(row.get("anchor_road_distance_m", float("inf"))),
                str(row.get("truck_id", "")),
                str(row.get("uav_id", "")),
                int(row.get("recovery_anchor", -1)),
            )
        )
        return rows[:top_k], int(len(unique_trucks))

    def _assigned_critical_reconstruct_candidate_rows(self, env, solution, task_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        task = getattr(env.state, "tasks", {}).get(str(task_id), None)
        if task is None:
            return [], {"trucks_considered": 0, "path_candidates": 0, "margin_probed": 0, "positive_margin_count": 0, "rejected_no_bindable_truck": 0, "best_margin": float("-inf")}
        top_k = int(max(getattr(env.cfg, "alns_assigned_critical_reconstruct_top_k_paths", 12), 1))
        historical = self._latest_historical_support_binding(task_id)
        historical_truck = str(getattr(historical, "truck_id", "")) if historical is not None else ""
        historical_uav = str(getattr(historical, "uav_id", "")) if historical is not None else ""
        historical_recovery = getattr(historical, "recovery_anchor", None) if historical is not None else None
        owner_truck = ""
        for aid, agent_type in self._agent_sequence_pairs_from_solution(solution):
            if str(agent_type) != "truck":
                continue
            if str(task_id) in tuple(str(x) for x in solution.sequence_for(aid, agent_type)):
                owner_truck = str(aid)
                break
        truck_ids: list[str] = []
        if owner_truck:
            truck_ids.append(owner_truck)
        if historical_truck and historical_truck not in truck_ids:
            truck_ids.append(historical_truck)
        other_trucks = sorted(
            [
                str(aid)
                for aid, st in getattr(env.state, "agents", {}).items()
                if getattr(st, "kind", None) == AgentKind.TRUCK and not bool(getattr(st, "crashed", False))
            ],
            key=lambda aid: (float(self._truck_task_distance(env, str(aid), task)), str(aid)),
        )
        for aid in other_trucks:
            if aid not in truck_ids:
                truck_ids.append(aid)
        rows: list[dict[str, Any]] = []
        diag = {
            "trucks_considered": 0,
            "path_candidates": 0,
            "margin_probed": 0,
            "positive_margin_count": 0,
            "rejected_no_bindable_truck": 0,
            "best_margin": float("-inf"),
        }
        for truck_id in truck_ids:
            truck_st = getattr(env.state, "agents", {}).get(str(truck_id), None)
            if truck_st is None or getattr(truck_st, "kind", None) != AgentKind.TRUCK or bool(getattr(truck_st, "crashed", False)):
                continue
            diag["trucks_considered"] += 1
            docked_uavs = sorted(
                [
                    str(aid)
                    for aid, st in getattr(env.state, "agents", {}).items()
                    if getattr(st, "kind", None) == AgentKind.UAV
                    and not bool(getattr(st, "crashed", False))
                    and str(getattr(st, "follow_target", "")) == str(truck_id)
                ],
                key=str,
            )
            gain_info = self._support_anchor_service_gain(env, str(truck_id), task)
            bind_info = self._support_bound_delivery_info(env, str(truck_id), task, gain_info=gain_info)
            launch_anchor = int(getattr(truck_st, "node", 0) or 0)
            recovery_anchors = self._support_rebind_anchor_candidates(
                env,
                task=task,
                truck_id=str(truck_id),
                launch_anchor=int(launch_anchor),
                historical_recovery=(int(historical_recovery) if historical is not None and str(truck_id) == historical_truck and historical_recovery is not None else None),
            )
            strong_binding = bool(bind_info.get("bound_any", 0.0)) and bool(
                self._support_binding_is_strong_enough(env, task, bind_info, gain_info=gain_info)
            )
            candidate_uavs: list[tuple[str, str]] = []
            bound_uav = str(bind_info.get("bound_timecritical_uav_id", "") or historical_uav)
            if strong_binding and bound_uav:
                candidate_uavs.append((bound_uav, "bound"))
            for docked_uav in docked_uavs:
                if all(str(docked_uav) != existing_uav for existing_uav, _src in candidate_uavs):
                    candidate_uavs.append((str(docked_uav), "docked"))
            if not candidate_uavs:
                diag["rejected_no_bindable_truck"] += 1
                continue
            for uav_id, source in candidate_uavs:
                for recovery_anchor in recovery_anchors:
                    anchor_metrics = self._support_rebind_anchor_metrics(
                        env,
                        task=task,
                        truck_id=str(truck_id),
                        launch_anchor=int(launch_anchor),
                        recovery_anchor=int(recovery_anchor),
                    )
                    launch_gate_reason = ""
                    launchable_now = False
                    launch_gate = getattr(env, "_uav_launch_gate_check", None)
                    if callable(launch_gate):
                        try:
                            launchable_now, launch_gate_reason, _force = launch_gate(str(uav_id), task=task, count_reject=False)
                        except TypeError:
                            launchable_now, launch_gate_reason, _force = launch_gate(str(uav_id), task=task)
                    override_diag = self._support_rebind_override_diagnostics(
                        env,
                        task=task,
                        truck_id=str(truck_id),
                        uav_id=str(uav_id),
                        launch_anchor=int(launch_anchor),
                        launchable_now=bool(launchable_now),
                    )
                    predicted_margin = float(override_diag.get("predicted_recovery_margin_m", float("-inf")))
                    predicted_battery_margin = float(override_diag.get("predicted_battery_margin_ratio", float("-inf")))
                    if not np.isneginf(predicted_margin):
                        diag["margin_probed"] += 1
                        diag["best_margin"] = max(float(diag["best_margin"]), predicted_margin)
                        if predicted_margin > 0.0 and predicted_battery_margin >= 0.0:
                            diag["positive_margin_count"] += 1
                    row = {
                        "task_id": str(task_id),
                        "truck_id": str(truck_id),
                        "uav_id": str(uav_id),
                        "launch_anchor": int(launch_anchor),
                        "recovery_anchor": int(recovery_anchor),
                        "source": str(source),
                        "launchable_now": bool(launchable_now),
                        "launch_gate_reason": str(launch_gate_reason or ""),
                        "owner_truck_match": int(str(truck_id) == owner_truck) if owner_truck else 0,
                        "historical_truck_match": int(str(truck_id) == historical_truck) if historical_truck else 0,
                        "historical_uav_match": int(str(uav_id) == historical_uav) if historical_uav else 0,
                        "override_reason": str(override_diag.get("override_reason", "") or ""),
                        "override_feasible": bool(override_diag.get("override_feasible", False)),
                        "predicted_recovery_margin_m": predicted_margin,
                        "predicted_battery_margin_ratio": predicted_battery_margin,
                        **anchor_metrics,
                        **override_diag,
                    }
                    rows.append(row)
        for row in rows:
            row["path_rank"] = (
                0 if bool(row.get("override_feasible", False)) else 1,
                0 if (np.isfinite(float(row.get("predicted_recovery_margin_m", float("-inf")))) and float(row.get("predicted_recovery_margin_m", float("-inf"))) > 0.0) else 1,
                -float(row.get("predicted_recovery_margin_m", float("-inf"))),
                -float(row.get("predicted_battery_margin_ratio", float("-inf"))),
                0 if bool(row.get("owner_truck_match", 0)) else 1,
                0 if bool(row.get("historical_truck_match", 0)) else 1,
                0 if bool(row.get("anchor_reachable", True)) else 1,
                float(row.get("anchor_road_distance_m", float("inf"))),
                str(row.get("truck_id", "")),
                str(row.get("uav_id", "")),
                int(row.get("launch_anchor", -1)),
                int(row.get("recovery_anchor", -1)),
            )
        rows.sort(key=lambda row: row.get("path_rank"))
        diag["path_candidates"] = int(len(rows))
        return rows[:top_k], diag


    def _support_rebind_trial_solution(self, solution, *, task_id: str, target_truck_id: str):
        trial = solution.without_task(task_id)
        seq = tuple(str(x) for x in trial.sequence_for(target_truck_id, "truck"))
        if len(seq) >= 2 and str(task_id) not in seq:
            return None
        if str(task_id) in seq:
            new_seq = seq
        elif len(seq) == 0:
            new_seq = (str(task_id),)
        else:
            new_seq = (seq[0], str(task_id))
        return trial.with_sequence(target_truck_id, "truck", new_seq)

    def _assigned_critical_reconstruct_trial_solution(self, env, solution, *, task_id: str, target_truck_id: str):
        trial = solution.without_task(task_id)
        seq = tuple(str(x) for x in trial.sequence_for(target_truck_id, "truck"))
        task = getattr(env.state, "tasks", {}).get(str(task_id), None)
        is_critical = bool(task is not None and (getattr(task, "kind", None) == TaskKind.EMERGENCY or self._is_timecritical_lightweight_task(task)))
        if str(task_id) in seq:
            return trial, "ALREADY_PRESENT"
        if len(seq) == 0:
            return trial.with_sequence(target_truck_id, "truck", (str(task_id),)), "APPEND_EMPTY"
        if len(seq) == 1:
            return trial.with_sequence(target_truck_id, "truck", (seq[0], str(task_id))), "APPEND_TAIL"
        head, tail = seq[0], seq[1]
        tail_task = getattr(env.state, "tasks", {}).get(str(tail), None)
        tail_is_critical = bool(
            tail_task is not None and (getattr(tail_task, "kind", None) == TaskKind.EMERGENCY or self._is_timecritical_lightweight_task(tail_task))
        )
        tail_started = bool(tail_task is not None and getattr(tail_task, "first_service_step", None) is not None)
        tail_completed = bool(tail_task is not None and getattr(tail_task, "status", None) == TaskStatus.DELIVERED)
        if is_critical and tail_task is not None and (not tail_is_critical) and (not tail_started) and (not tail_completed):
            trial = trial.without_task(str(tail))
            return trial.with_sequence(target_truck_id, "truck", (head, str(task_id))), "REPLACE_ROUTINE_TAIL"
        return None, "NO_SEQUENCE_CAPACITY"

    def _support_rebind_add_binding(self, solution, *, binding: SupportBinding, sortie: SortiePlan):
        task_id = str(binding.task_id)
        truck_id = str(binding.truck_id)
        uav_id = str(binding.uav_id)
        for existing in tuple(getattr(solution, "support_bindings", ())):
            existing_task = str(getattr(existing, "task_id", ""))
            existing_truck = str(getattr(existing, "truck_id", ""))
            existing_uav = str(getattr(existing, "uav_id", ""))
            if not existing_task or existing_task == task_id:
                continue
            if existing_truck == truck_id or existing_uav == uav_id:
                return None
        return solution.__class__(
            truck_sequences=dict(getattr(solution, "truck_sequences", ())),
            uav_sequences=dict(getattr(solution, "uav_sequences", ())),
            support_bindings=tuple(
                b
                for b in tuple(getattr(solution, "support_bindings", ()))
                if str(getattr(b, "task_id", "")) != task_id
                and str(getattr(b, "uav_id", "")) != uav_id
                and str(getattr(b, "truck_id", "")) != truck_id
            )
            + (binding,),
            sortie_plans=tuple(
                p
                for p in tuple(getattr(solution, "sortie_plans", ()))
                if str(getattr(p, "task_id", "")) != task_id
                and str(getattr(p, "uav_id", "")) != uav_id
            )
            + (sortie,),
        )

    def _apply_lc_critical_recovery_path(self, env, repair_result: RepairResult) -> RepairResult:
        if not self._lc_critical_recovery_path_enabled(env):
            return repair_result
        diag = dict(getattr(repair_result, "diagnostics", {}))
        diag.setdefault("lc_critical_recovery_path_enabled", True)
        diag.setdefault("lc_critical_recovery_path_candidates", 0)
        diag.setdefault("lc_critical_recovery_path_attempts", 0)
        diag.setdefault("lc_critical_recovery_path_successes", 0)
        diag.setdefault("lc_critical_recovery_path_rejected_insufficient_margin", 0)
        diag.setdefault("lc_critical_recovery_path_rejected_no_bindable_truck", 0)
        diag.setdefault("lc_critical_recovery_path_rejected_uav_not_docked", 0)
        diag.setdefault("lc_critical_recovery_path_rejected_no_sequence_capacity", 0)
        diag.setdefault("lc_critical_recovery_path_rejected_augmented_infeasible", 0)
        diag.setdefault("lc_critical_recovery_path_trucks_considered", 0)
        diag.setdefault("lc_critical_recovery_path_best_margin", float("-inf"))
        diag.setdefault("lc_critical_recovery_path_success_margin", float("-inf"))
        diag.setdefault("lc_critical_recovery_path_failed_tuple_avoided", 0)
        diag.setdefault("lc_critical_recovery_path_task_ids", [])
        diag.setdefault("lc_critical_recovery_path_selected_uav_ids", [])
        diag.setdefault("lc_critical_recovery_path_selected_truck_ids", [])
        diag.setdefault("lc_critical_recovery_path_selected_recovery_anchors", [])
        diag.setdefault("lc_critical_recovery_path_attempt_rows", [])
        diag.setdefault("support_coordination", list(diag.get("support_coordination", [])))

        solution = repair_result.candidate_solution
        candidate_task_ids = self._lc_critical_recovery_candidate_task_ids(env, solution, repair_result)
        require_positive_margin = bool(getattr(env.cfg, "alns_lc_critical_recovery_path_require_positive_margin", True))

        for task_id in candidate_task_ids:
            task = getattr(env.state, "tasks", {}).get(str(task_id), None)
            if task is None:
                diag["lc_critical_recovery_path_attempt_rows"].append(
                    {"task_id": str(task_id), "result": "skipped", "reason": "TASK_MISSING"}
                )
                continue
            diag["lc_critical_recovery_path_candidates"] += 1
            rows, trucks_considered = self._lc_critical_recovery_candidate_rows(env, solution, str(task_id))
            diag["lc_critical_recovery_path_trucks_considered"] += int(trucks_considered)
            if not rows:
                diag["lc_critical_recovery_path_rejected_no_bindable_truck"] += 1
                diag["lc_critical_recovery_path_attempt_rows"].append(
                    {"task_id": str(task_id), "result": "rejected", "reason": "NO_BINDABLE_TRUCK"}
                )
                continue
            rebound = False
            best_margin = float("-inf")
            for row in rows:
                diag["lc_critical_recovery_path_attempts"] += 1
                truck_id = str(row.get("truck_id", ""))
                uav_id = str(row.get("uav_id", ""))
                launch_anchor = int(row.get("launch_anchor", -1))
                recovery_anchor = int(row.get("recovery_anchor", -1))
                predicted_margin = float(row.get("predicted_recovery_margin_m", float("-inf")))
                predicted_battery_margin = float(row.get("predicted_battery_margin_ratio", float("-inf")))
                best_margin = max(best_margin, predicted_margin)
                diag["lc_critical_recovery_path_best_margin"] = max(
                    float(diag.get("lc_critical_recovery_path_best_margin", float("-inf"))),
                    predicted_margin,
                )
                failure_key = self._support_rebind_failure_key(
                    task_id=str(task_id),
                    truck_id=truck_id,
                    uav_id=uav_id,
                    launch_anchor=launch_anchor,
                    recovery_anchor=recovery_anchor,
                )
                if bool(row.get("lc_failed_tuple_skip_recommended", False)):
                    diag["lc_critical_recovery_path_failed_tuple_avoided"] += 1
                    diag["lc_critical_recovery_path_attempt_rows"].append(
                        {
                            "task_id": str(task_id),
                            "truck_id": truck_id,
                            "uav_id": uav_id,
                            "launch_anchor": launch_anchor,
                            "recovery_anchor": recovery_anchor,
                            "result": "skipped",
                            "reason": "FAILED_TUPLE_AVOIDED",
                            "predicted_recovery_margin_m": predicted_margin,
                            "predicted_battery_margin_ratio": predicted_battery_margin,
                            "failed_tuple_count": int(row.get("lc_failed_tuple_count", 0)),
                        }
                    )
                    continue
                reject_reason = ""
                if (
                    require_positive_margin
                    and not bool(row.get("lc_positive_margin_ok", False))
                ):
                    reject_reason = "INSUFFICIENT_RECOVERY_MARGIN"
                truck_st = getattr(env.state, "agents", {}).get(truck_id, None)
                uav_st = getattr(env.state, "agents", {}).get(uav_id, None)
                if not reject_reason:
                    if truck_st is None or getattr(truck_st, "kind", None) != AgentKind.TRUCK or bool(getattr(truck_st, "crashed", False)):
                        reject_reason = "NO_BINDABLE_TRUCK"
                    elif uav_st is None or getattr(uav_st, "kind", None) != AgentKind.UAV or bool(getattr(uav_st, "crashed", False)):
                        reject_reason = "NO_BINDABLE_TRUCK"
                    elif str(getattr(uav_st, "follow_target", "")) != truck_id:
                        reject_reason = "UAV_NOT_DOCKED_TO_TRUCK"
                    elif int(getattr(truck_st, "node", -2)) != launch_anchor:
                        reject_reason = "INVALID_LAUNCH_ANCHOR"
                trial = None if reject_reason else self._support_rebind_trial_solution(solution, task_id=str(task_id), target_truck_id=truck_id)
                if not reject_reason and trial is None:
                    reject_reason = "NO_SEQUENCE_CAPACITY"
                if not reject_reason:
                    truck_seq = tuple(str(x) for x in trial.sequence_for(truck_id, "truck"))
                    seq_feas = evaluate_sequence_feasibility(env, truck_id, truck_seq)
                    if not bool(seq_feas.feasible):
                        reject_reason = "|".join(str(x) for x in seq_feas.reason_codes) or "TRIAL_SEQUENCE_INFEASIBLE"
                if not reject_reason:
                    binding = SupportBinding(
                        uav_id=uav_id,
                        truck_id=truck_id,
                        task_id=str(task_id),
                        launch_anchor=launch_anchor,
                        recovery_anchor=recovery_anchor,
                    )
                    sortie = SortiePlan(
                        uav_id=uav_id,
                        task_id=str(task_id),
                        launch_anchor=int(launch_anchor),
                        recovery_anchor=int(recovery_anchor),
                        estimated_launch_step=int(getattr(env.state, "step_index", 0)),
                        estimated_service_step=None,
                        estimated_recovery_step=None,
                    )
                    augmented = self._support_rebind_add_binding(trial, binding=binding, sortie=sortie)
                    if augmented is None:
                        reject_reason = "SUPPORT_BINDING_CONFLICT"
                    else:
                        hard_ok = bool(self._goals_hard_feasible(env, solution_to_legacy_goals(augmented)))
                        augmented_eval = evaluate_k2_solution(env, augmented, hard_feasible=hard_ok)
                        if not bool(augmented_eval.feasible):
                            reject_reason = "|".join(str(x) for x in augmented_eval.infeasibility_reasons) or "AUGMENTED_INFEASIBLE"
                if reject_reason:
                    self._lc_critical_recovery_failure_counts[failure_key] = self._lc_critical_recovery_failure_counts.get(failure_key, 0) + 1
                    bucket = self._lc_critical_recovery_rejection_bucket(reject_reason)
                    diag[bucket] = int(diag.get(bucket, 0)) + 1
                    diag["lc_critical_recovery_path_attempt_rows"].append(
                        {
                            "task_id": str(task_id),
                            "truck_id": truck_id,
                            "uav_id": uav_id,
                            "launch_anchor": launch_anchor,
                            "recovery_anchor": recovery_anchor,
                            "result": "rejected",
                            "reason": str(reject_reason),
                            "predicted_recovery_margin_m": predicted_margin,
                            "predicted_battery_margin_ratio": predicted_battery_margin,
                            "failed_tuple_count": int(row.get("lc_failed_tuple_count", 0)),
                        }
                    )
                    continue
                solution = augmented
                self._lc_critical_recovery_success_counts[failure_key] = self._lc_critical_recovery_success_counts.get(failure_key, 0) + 1
                diag["lc_critical_recovery_path_successes"] += 1
                diag["lc_critical_recovery_path_success_margin"] = max(
                    float(diag.get("lc_critical_recovery_path_success_margin", float("-inf"))),
                    predicted_margin,
                )
                diag["lc_critical_recovery_path_task_ids"].append(str(task_id))
                diag["lc_critical_recovery_path_selected_uav_ids"].append(str(uav_id))
                diag["lc_critical_recovery_path_selected_truck_ids"].append(str(truck_id))
                diag["lc_critical_recovery_path_selected_recovery_anchors"].append(int(recovery_anchor))
                diag["lc_critical_recovery_path_attempt_rows"].append(
                    {
                        "task_id": str(task_id),
                        "truck_id": truck_id,
                        "uav_id": uav_id,
                        "launch_anchor": launch_anchor,
                        "recovery_anchor": recovery_anchor,
                        "result": "accepted",
                        "reason": "",
                        "predicted_recovery_margin_m": predicted_margin,
                        "predicted_battery_margin_ratio": predicted_battery_margin,
                        "failed_tuple_count": int(row.get("lc_failed_tuple_count", 0)),
                    }
                )
                diag["support_coordination"].append(
                    {
                        "truck_id": truck_id,
                        "uav_id": uav_id,
                        "task_id": str(task_id),
                        "launch_anchor": int(launch_anchor),
                        "recovery_anchor": int(recovery_anchor),
                    }
                )
                rebound = True
                break
            if not rebound and np.isfinite(best_margin):
                diag["lc_critical_recovery_path_best_margin"] = max(
                    float(diag.get("lc_critical_recovery_path_best_margin", float("-inf"))),
                    best_margin,
                )
        diag["lc_critical_recovery_path_task_ids"] = tuple(diag["lc_critical_recovery_path_task_ids"])
        diag["lc_critical_recovery_path_selected_uav_ids"] = tuple(diag["lc_critical_recovery_path_selected_uav_ids"])
        diag["lc_critical_recovery_path_selected_truck_ids"] = tuple(diag["lc_critical_recovery_path_selected_truck_ids"])
        diag["lc_critical_recovery_path_selected_recovery_anchors"] = tuple(diag["lc_critical_recovery_path_selected_recovery_anchors"])
        return RepairResult(
            candidate_solution=solution,
            inserted_items=getattr(repair_result, "inserted_items", ()),
            feasible=bool(getattr(repair_result, "feasible", True)),
            reason_codes=tuple(getattr(repair_result, "reason_codes", ())),
            diagnostics=diag,
        )

    def _apply_assigned_critical_reconstruct(self, env, repair_result: RepairResult) -> RepairResult:
        if not self._assigned_critical_reconstruct_enabled(env):
            return repair_result
        diag = dict(getattr(repair_result, "diagnostics", {}))
        diag.setdefault("assigned_critical_reconstruct_enabled", True)
        diag.setdefault("assigned_critical_reconstruct_candidates", 0)
        diag.setdefault("assigned_critical_reconstruct_path_candidates", 0)
        diag.setdefault("assigned_critical_reconstruct_trucks_considered", 0)
        diag.setdefault("assigned_critical_reconstruct_margin_probed", 0)
        diag.setdefault("assigned_critical_reconstruct_positive_margin_count", 0)
        diag.setdefault("assigned_critical_reconstruct_selected_path_count", 0)
        diag.setdefault("assigned_critical_reconstruct_success_count", 0)
        diag.setdefault("assigned_critical_reconstruct_rejected_no_bindable_truck", 0)
        diag.setdefault("assigned_critical_reconstruct_rejected_insufficient_margin", 0)
        diag.setdefault("assigned_critical_reconstruct_rejected_no_sequence_capacity", 0)
        diag.setdefault("assigned_critical_reconstruct_rejected_target_unreachable", 0)
        diag.setdefault("assigned_critical_reconstruct_rejected_uav_not_docked", 0)
        diag.setdefault("assigned_critical_reconstruct_rejected_infeasible", 0)
        diag.setdefault("assigned_critical_reconstruct_best_margin_by_task", {})
        diag.setdefault("assigned_critical_reconstruct_selected_uav_ids", [])
        diag.setdefault("assigned_critical_reconstruct_selected_truck_ids", [])
        diag.setdefault("assigned_critical_reconstruct_selected_launch_anchors", [])
        diag.setdefault("assigned_critical_reconstruct_selected_recovery_anchors", [])
        diag.setdefault("assigned_critical_reconstruct_no_progress_tasks_targeted", 0)
        diag.setdefault("assigned_critical_reconstruct_task_ids", [])
        diag.setdefault("assigned_critical_reconstruct_attempt_rows", [])

        solution = repair_result.candidate_solution
        candidate_task_ids = self._assigned_critical_reconstruct_candidate_task_ids(env, solution, repair_result)
        for task_id in candidate_task_ids:
            task = getattr(env.state, "tasks", {}).get(str(task_id), None)
            if task is None:
                continue
            diag["assigned_critical_reconstruct_candidates"] += 1
            diag["assigned_critical_reconstruct_no_progress_tasks_targeted"] += 1
            diag["assigned_critical_reconstruct_task_ids"].append(str(task_id))
            rows, row_diag = self._assigned_critical_reconstruct_candidate_rows(env, solution, str(task_id))
            diag["assigned_critical_reconstruct_path_candidates"] += int(row_diag.get("path_candidates", 0))
            diag["assigned_critical_reconstruct_trucks_considered"] += int(row_diag.get("trucks_considered", 0))
            diag["assigned_critical_reconstruct_margin_probed"] += int(row_diag.get("margin_probed", 0))
            diag["assigned_critical_reconstruct_positive_margin_count"] += int(row_diag.get("positive_margin_count", 0))
            if np.isfinite(float(row_diag.get("best_margin", float("-inf")))):
                diag["assigned_critical_reconstruct_best_margin_by_task"][str(task_id)] = float(row_diag.get("best_margin", float("-inf")))
            if not rows:
                diag["assigned_critical_reconstruct_rejected_no_bindable_truck"] += max(int(row_diag.get("rejected_no_bindable_truck", 0)), 1)
                diag["assigned_critical_reconstruct_attempt_rows"].append({"task_id": str(task_id), "result": "rejected", "reason": "NO_BINDABLE_TRUCK"})
                continue
            installed = False
            for row in rows:
                truck_id = str(row.get("truck_id", ""))
                uav_id = str(row.get("uav_id", ""))
                launch_anchor = int(row.get("launch_anchor", -1))
                recovery_anchor = int(row.get("recovery_anchor", -1))
                predicted_margin = float(row.get("predicted_recovery_margin_m", float("-inf")))
                predicted_battery_margin = float(row.get("predicted_battery_margin_ratio", float("-inf")))
                reason = ""
                if not bool(row.get("anchor_reachable", True)):
                    reason = "TARGET_UNREACHABLE"
                    diag["assigned_critical_reconstruct_rejected_target_unreachable"] += 1
                else:
                    reason = self._assigned_reconstruct_margin_reject_reason(
                        override_feasible=bool(row.get("override_feasible", False)),
                        override_reason=str(row.get("override_reason", "") or ""),
                        predicted_margin=predicted_margin,
                        predicted_battery_margin=predicted_battery_margin,
                    )
                    if reason == "INSUFFICIENT_RECOVERY_MARGIN":
                        diag["assigned_critical_reconstruct_rejected_insufficient_margin"] += 1
                    elif reason:
                        diag["assigned_critical_reconstruct_rejected_infeasible"] += 1
                if not reason:
                    truck_st = getattr(env.state, "agents", {}).get(truck_id, None)
                    uav_st = getattr(env.state, "agents", {}).get(uav_id, None)
                    if truck_st is None or getattr(truck_st, "kind", None) != AgentKind.TRUCK or bool(getattr(truck_st, "crashed", False)):
                        reason = "NO_BINDABLE_TRUCK"
                        diag["assigned_critical_reconstruct_rejected_no_bindable_truck"] += 1
                    elif uav_st is None or getattr(uav_st, "kind", None) != AgentKind.UAV or bool(getattr(uav_st, "crashed", False)):
                        reason = "NO_BINDABLE_TRUCK"
                        diag["assigned_critical_reconstruct_rejected_no_bindable_truck"] += 1
                    elif str(getattr(uav_st, "follow_target", "")) != truck_id:
                        reason = "UAV_NOT_DOCKED"
                        diag["assigned_critical_reconstruct_rejected_uav_not_docked"] += 1
                    elif int(getattr(truck_st, "node", -2)) != launch_anchor:
                        reason = "INVALID_LAUNCH_ANCHOR"
                        diag["assigned_critical_reconstruct_rejected_infeasible"] += 1
                trial = None
                placement_reason = ""
                if not reason:
                    trial, placement_reason = self._assigned_critical_reconstruct_trial_solution(
                        env,
                        solution,
                        task_id=str(task_id),
                        target_truck_id=truck_id,
                    )
                    if trial is None:
                        reason = str(placement_reason or "NO_SEQUENCE_CAPACITY")
                        diag["assigned_critical_reconstruct_rejected_no_sequence_capacity"] += 1
                if not reason and trial is not None:
                    truck_seq = tuple(str(x) for x in trial.sequence_for(truck_id, "truck"))
                    seq_feas = evaluate_sequence_feasibility(env, truck_id, truck_seq)
                    if not bool(seq_feas.feasible):
                        reason = "|".join(str(x) for x in seq_feas.reason_codes) or "TRIAL_SEQUENCE_INFEASIBLE"
                        diag["assigned_critical_reconstruct_rejected_infeasible"] += 1
                if not reason and trial is not None:
                    binding = SupportBinding(
                        uav_id=uav_id,
                        truck_id=truck_id,
                        task_id=str(task_id),
                        launch_anchor=launch_anchor,
                        recovery_anchor=recovery_anchor,
                    )
                    sortie = SortiePlan(
                        uav_id=uav_id,
                        task_id=str(task_id),
                        launch_anchor=launch_anchor,
                        recovery_anchor=recovery_anchor,
                        estimated_launch_step=int(getattr(env.state, "step_index", 0)),
                        estimated_service_step=None,
                        estimated_recovery_step=None,
                    )
                    augmented = self._support_rebind_add_binding(trial, binding=binding, sortie=sortie)
                    if augmented is None:
                        reason = "SUPPORT_BINDING_CONFLICT"
                        diag["assigned_critical_reconstruct_rejected_infeasible"] += 1
                    else:
                        hard_ok = bool(self._goals_hard_feasible(env, solution_to_legacy_goals(augmented)))
                        augmented_eval = evaluate_k2_solution(env, augmented, hard_feasible=hard_ok)
                        if not bool(augmented_eval.feasible):
                            reason = "|".join(str(x) for x in augmented_eval.infeasibility_reasons) or "AUGMENTED_INFEASIBLE"
                            diag["assigned_critical_reconstruct_rejected_infeasible"] += 1
                if reason:
                    diag["assigned_critical_reconstruct_attempt_rows"].append(
                        {
                            "task_id": str(task_id),
                            "task_kind": str(getattr(task, "kind", "")),
                            "task_class": str(getattr(task, "task_class", "")),
                            "truck_id": truck_id,
                            "uav_id": uav_id,
                            "launch_anchor": launch_anchor,
                            "recovery_anchor": recovery_anchor,
                            "predicted_recovery_margin_m": predicted_margin,
                            "predicted_battery_margin_ratio": predicted_battery_margin,
                            "predicted_total_required_energy_ratio": float(row.get("predicted_total_required_energy_ratio", float("nan"))),
                            "predicted_required_reserve_fraction": float(row.get("predicted_required_reserve_fraction", float("nan"))),
                            "predicted_battery_available_ratio": float(row.get("predicted_battery_available_ratio", float("nan"))),
                            "predicted_go_energy_ratio": float(row.get("predicted_go_energy_ratio", float("nan"))),
                            "predicted_recovery_energy_ratio": float(row.get("predicted_recovery_energy_ratio", float("nan"))),
                            "target_reachable": bool(row.get("anchor_reachable", False)),
                            "truck_bindable": bool(reason not in {"NO_BINDABLE_TRUCK"}),
                            "uav_docked": bool(reason not in {"UAV_NOT_DOCKED"}),
                            "sequence_capacity_ok": bool(reason not in {"NO_SEQUENCE_CAPACITY"}),
                            "feasibility_function_called": True,
                            "returned_feasible": bool(row.get("override_feasible", False)),
                            "returned_reason": str(row.get("override_reason", "")),
                            "launch_gate_reason": str(row.get("launch_gate_reason", "")),
                            "evidence_level": "direct_override_diag",
                            "result": "rejected",
                            "reason": str(reason),
                        }
                    )
                    continue
                solution = augmented
                diag["assigned_critical_reconstruct_selected_path_count"] += 1
                diag["assigned_critical_reconstruct_success_count"] += 1
                diag["assigned_critical_reconstruct_selected_uav_ids"].append(str(uav_id))
                diag["assigned_critical_reconstruct_selected_truck_ids"].append(str(truck_id))
                diag["assigned_critical_reconstruct_selected_launch_anchors"].append(int(launch_anchor))
                diag["assigned_critical_reconstruct_selected_recovery_anchors"].append(int(recovery_anchor))
                diag["assigned_critical_reconstruct_attempt_rows"].append(
                    {
                        "task_id": str(task_id),
                        "task_kind": str(getattr(task, "kind", "")),
                        "task_class": str(getattr(task, "task_class", "")),
                        "truck_id": truck_id,
                        "uav_id": uav_id,
                        "launch_anchor": launch_anchor,
                        "recovery_anchor": recovery_anchor,
                        "predicted_recovery_margin_m": predicted_margin,
                        "predicted_battery_margin_ratio": predicted_battery_margin,
                        "predicted_total_required_energy_ratio": float(row.get("predicted_total_required_energy_ratio", float("nan"))),
                        "predicted_required_reserve_fraction": float(row.get("predicted_required_reserve_fraction", float("nan"))),
                        "predicted_battery_available_ratio": float(row.get("predicted_battery_available_ratio", float("nan"))),
                        "predicted_go_energy_ratio": float(row.get("predicted_go_energy_ratio", float("nan"))),
                        "predicted_recovery_energy_ratio": float(row.get("predicted_recovery_energy_ratio", float("nan"))),
                        "target_reachable": bool(row.get("anchor_reachable", False)),
                        "truck_bindable": True,
                        "uav_docked": True,
                        "sequence_capacity_ok": True,
                        "feasibility_function_called": True,
                        "returned_feasible": bool(row.get("override_feasible", False)),
                        "returned_reason": str(row.get("override_reason", "")),
                        "launch_gate_reason": str(row.get("launch_gate_reason", "")),
                        "evidence_level": "direct_override_diag",
                        "result": "accepted",
                        "reason": str(placement_reason or ""),
                    }
                )
                diag.setdefault("support_coordination", list(diag.get("support_coordination", [])))
                diag["support_coordination"].append(
                    {
                        "truck_id": truck_id,
                        "uav_id": uav_id,
                        "task_id": str(task_id),
                        "launch_anchor": int(launch_anchor),
                        "recovery_anchor": int(recovery_anchor),
                    }
                )
                installed = True
                break
            if not installed and rows:
                diag["assigned_critical_reconstruct_selected_path_count"] += 0

        diag["assigned_critical_reconstruct_task_ids"] = tuple(diag["assigned_critical_reconstruct_task_ids"])
        diag["assigned_critical_reconstruct_selected_uav_ids"] = tuple(diag["assigned_critical_reconstruct_selected_uav_ids"])
        diag["assigned_critical_reconstruct_selected_truck_ids"] = tuple(diag["assigned_critical_reconstruct_selected_truck_ids"])
        diag["assigned_critical_reconstruct_selected_launch_anchors"] = tuple(diag["assigned_critical_reconstruct_selected_launch_anchors"])
        diag["assigned_critical_reconstruct_selected_recovery_anchors"] = tuple(diag["assigned_critical_reconstruct_selected_recovery_anchors"])
        return RepairResult(
            candidate_solution=solution,
            inserted_items=getattr(repair_result, "inserted_items", ()),
            feasible=bool(getattr(repair_result, "feasible", True)),
            reason_codes=tuple(getattr(repair_result, "reason_codes", ())),
            diagnostics=diag,
        )

    def _support_reposition_shadow_candidate_task_ids(self, env, solution) -> list[str]:
        max_tasks = int(max(getattr(env.cfg, "alns_support_reposition_shadow_max_tasks", 8), 0))
        min_assigned = int(max(getattr(env.cfg, "alns_support_reposition_shadow_min_assigned_count", 20), 0))
        candidates: list[str] = []
        for task_id, task in getattr(env.state, "tasks", {}).items():
            if task is None or bool(getattr(task, "completed", False)):
                continue
            if getattr(task, "status", None) not in {TaskStatus.PENDING, TaskStatus.FAILED}:
                continue
            is_critical = bool(getattr(task, "kind", None) == TaskKind.EMERGENCY or self._is_timecritical_lightweight_task(task))
            if not is_critical:
                continue
            tid = str(task_id)
            if self._task_assignment_count(tid) < min_assigned:
                continue
            if self._task_service_started(env, tid):
                continue
            candidates.append(tid)
        candidates.sort(key=lambda tid: self._assigned_critical_reconstruct_priority_key(env, solution, tid))
        return candidates[:max_tasks] if max_tasks > 0 else []

    @staticmethod
    def _support_reposition_shadow_rescue_type(row: Mapping[str, Any]) -> str:
        target_reachable = bool(row.get("anchor_reachable", True))
        batt_margin = float(row.get("predicted_battery_margin_ratio", float("-inf")))
        override_reason = str(row.get("override_reason", "") or "")
        if not target_reachable:
            return "unreachable"
        if batt_margin >= 0.0 and override_reason not in {"low_battery_margin", "task_has_airborne_uav"}:
            return "feasible"
        if batt_margin >= 0.0:
            return "battery_threshold_only"
        return "low_battery"

    def _apply_support_reposition_shadow(self, env, repair_result: RepairResult) -> RepairResult:
        if not self._support_reposition_shadow_enabled(env):
            return repair_result
        diag = dict(getattr(repair_result, "diagnostics", {}))
        diag.setdefault("support_reposition_shadow_enabled", True)
        diag.setdefault("support_reposition_shadow_candidates", 0)
        diag.setdefault("support_reposition_shadow_feasible_suggestions", 0)
        diag.setdefault("support_reposition_shadow_low_battery_rescue_possible", 0)
        diag.setdefault("support_reposition_shadow_unreachable_rescue_possible", 0)
        diag.setdefault("support_reposition_shadow_no_progress_tasks_covered", 0)
        diag.setdefault("support_reposition_shadow_truck_ids", [])
        diag.setdefault("support_reposition_shadow_anchor_ids", [])
        diag.setdefault("support_reposition_shadow_estimated_battery_gain", 0.0)
        diag.setdefault("support_reposition_shadow_estimated_truck_cost", 0.0)
        diag.setdefault("support_reposition_shadow_rows", [])
        local_candidates = 0
        local_feasible = 0
        local_low_battery_rescue = 0
        local_unreachable_rescue = 0
        local_covered = 0
        local_battery_gain = 0.0
        local_truck_cost = 0.0

        solution = repair_result.candidate_solution
        task_ids = self._support_reposition_shadow_candidate_task_ids(env, solution)
        for task_id in task_ids:
            rows, _row_diag = self._assigned_critical_reconstruct_candidate_rows(env, solution, str(task_id))
            if not rows:
                continue
            local_covered += 1
            baseline = rows[0]
            baseline_energy = float(baseline.get("predicted_total_required_energy_ratio", float("inf")))
            suggestion = None
            for row in rows:
                batt_margin = float(row.get("predicted_battery_margin_ratio", float("-inf")))
                target_reachable = bool(row.get("anchor_reachable", True))
                override_reason = str(row.get("override_reason", "") or "")
                if target_reachable and batt_margin >= 0.0 and override_reason not in {"low_battery_margin", "task_has_airborne_uav"}:
                    suggestion = row
                    break
            if suggestion is None:
                for row in rows:
                    target_reachable = bool(row.get("anchor_reachable", True))
                    batt_margin = float(row.get("predicted_battery_margin_ratio", float("-inf")))
                    if target_reachable and batt_margin > float(baseline.get("predicted_battery_margin_ratio", float("-inf"))):
                        suggestion = row
                        break
            if suggestion is None:
                suggestion = baseline
            suggested_energy = float(suggestion.get("predicted_total_required_energy_ratio", float("inf")))
            batt_gain = baseline_energy - suggested_energy if np.isfinite(baseline_energy) and np.isfinite(suggested_energy) else 0.0
            truck_cost = float(suggestion.get("launch_to_recovery_distance_m", float("inf")))
            rescue_type = self._support_reposition_shadow_rescue_type(suggestion)
            feasible_suggestion = bool(
                bool(suggestion.get("anchor_reachable", True))
                and float(suggestion.get("predicted_battery_margin_ratio", float("-inf"))) >= 0.0
                and str(suggestion.get("override_reason", "") or "") not in {"low_battery_margin", "task_has_airborne_uav", "low_delivery_score_gain"}
            )
            local_candidates += 1
            if feasible_suggestion:
                local_feasible += 1
            if rescue_type in {"feasible", "battery_threshold_only"}:
                local_low_battery_rescue += 1
            if bool(suggestion.get("anchor_reachable", True)):
                if not bool(baseline.get("anchor_reachable", True)):
                    local_unreachable_rescue += 1
            if np.isfinite(batt_gain):
                local_battery_gain += float(batt_gain)
            if np.isfinite(truck_cost) and truck_cost < 1e9:
                local_truck_cost += float(truck_cost)
            diag["support_reposition_shadow_truck_ids"].append(str(suggestion.get("truck_id", "")))
            diag["support_reposition_shadow_anchor_ids"].append(int(suggestion.get("recovery_anchor", -1)))
            diag["support_reposition_shadow_rows"].append(
                {
                    "task_id": str(task_id),
                    "suggested_truck_id": str(suggestion.get("truck_id", "")),
                    "suggested_reposition_anchor": int(suggestion.get("recovery_anchor", -1)),
                    "baseline_truck_id": str(baseline.get("truck_id", "")),
                    "baseline_recovery_anchor": int(baseline.get("recovery_anchor", -1)),
                    "estimated_truck_cost": float(truck_cost if np.isfinite(truck_cost) else 0.0),
                    "estimated_uav_battery_gain": float(batt_gain if np.isfinite(batt_gain) else 0.0),
                    "estimated_reachability_gain": int(bool(suggestion.get("anchor_reachable", True)) and not bool(baseline.get("anchor_reachable", True))),
                    "would_make_failed_path_feasible": bool(feasible_suggestion),
                    "baseline_reason": str(baseline.get("override_reason", "") or ""),
                    "suggested_reason": str(suggestion.get("override_reason", "") or ""),
                    "baseline_battery_margin_ratio": float(baseline.get("predicted_battery_margin_ratio", float("nan"))),
                    "suggested_battery_margin_ratio": float(suggestion.get("predicted_battery_margin_ratio", float("nan"))),
                    "baseline_target_reachable": bool(baseline.get("anchor_reachable", True)),
                    "suggested_target_reachable": bool(suggestion.get("anchor_reachable", True)),
                    "rescue_type": str(rescue_type),
                }
            )

        diag["support_reposition_shadow_candidates"] = int(diag.get("support_reposition_shadow_candidates", 0)) + local_candidates
        diag["support_reposition_shadow_feasible_suggestions"] = int(diag.get("support_reposition_shadow_feasible_suggestions", 0)) + local_feasible
        diag["support_reposition_shadow_low_battery_rescue_possible"] = int(diag.get("support_reposition_shadow_low_battery_rescue_possible", 0)) + local_low_battery_rescue
        diag["support_reposition_shadow_unreachable_rescue_possible"] = int(diag.get("support_reposition_shadow_unreachable_rescue_possible", 0)) + local_unreachable_rescue
        diag["support_reposition_shadow_no_progress_tasks_covered"] = int(diag.get("support_reposition_shadow_no_progress_tasks_covered", 0)) + local_covered
        diag["support_reposition_shadow_estimated_battery_gain"] = float(diag.get("support_reposition_shadow_estimated_battery_gain", 0.0)) + local_battery_gain
        diag["support_reposition_shadow_estimated_truck_cost"] = float(diag.get("support_reposition_shadow_estimated_truck_cost", 0.0)) + local_truck_cost
        diag["support_reposition_shadow_truck_ids"] = tuple(diag["support_reposition_shadow_truck_ids"])
        diag["support_reposition_shadow_anchor_ids"] = tuple(diag["support_reposition_shadow_anchor_ids"])
        self.alns_diagnostics.support_reposition_shadow_candidates += int(local_candidates)
        self.alns_diagnostics.support_reposition_shadow_feasible_suggestions += int(local_feasible)
        self.alns_diagnostics.support_reposition_shadow_low_battery_rescue_possible += int(local_low_battery_rescue)
        self.alns_diagnostics.support_reposition_shadow_unreachable_rescue_possible += int(local_unreachable_rescue)
        self.alns_diagnostics.support_reposition_shadow_no_progress_tasks_covered += int(local_covered)
        self.alns_diagnostics.support_reposition_shadow_estimated_battery_gain += float(local_battery_gain)
        self.alns_diagnostics.support_reposition_shadow_estimated_truck_cost += float(local_truck_cost)
        return RepairResult(
            candidate_solution=repair_result.candidate_solution,
            inserted_items=getattr(repair_result, "inserted_items", ()),
            feasible=bool(getattr(repair_result, "feasible", True)),
            reason_codes=tuple(getattr(repair_result, "reason_codes", ())),
            diagnostics=diag,
        )

    def _apply_critical_support_rebind(self, env, repair_result: RepairResult) -> RepairResult:
        if not self._critical_support_rebind_enabled(env):
            return repair_result
        diag = dict(getattr(repair_result, "diagnostics", {}))
        diag.setdefault("critical_support_rebind_enabled", True)
        diag.setdefault("critical_support_rebind_candidates", 0)
        diag.setdefault("critical_support_rebind_attempts", 0)
        diag.setdefault("critical_support_rebind_historical_reuse", 0)
        diag.setdefault("critical_support_rebind_reconstructed", 0)
        diag.setdefault("critical_support_rebind_rejected_no_truck", 0)
        diag.setdefault("critical_support_rebind_rejected_no_anchor", 0)
        diag.setdefault("critical_support_rebind_rejected_energy", 0)
        diag.setdefault("critical_support_rebind_rejected_reserve", 0)
        diag.setdefault("critical_support_rebind_rejected_road", 0)
        diag.setdefault("critical_support_rebind_rejected_infeasible", 0)
        diag.setdefault("critical_support_rebind_accept_count", 0)
        diag.setdefault("critical_support_rebind_failed_binding_penalized", 0)
        diag.setdefault("critical_support_rebind_failed_binding_skipped", 0)
        diag.setdefault("critical_support_rebind_best_accepted_margin_m", float("-inf"))
        diag.setdefault("critical_support_rebind_best_rejected_margin_m", float("-inf"))
        diag.setdefault("critical_support_rebind_best_accepted_battery_margin", float("-inf"))
        diag.setdefault("critical_support_rebind_best_rejected_battery_margin", float("-inf"))
        diag.setdefault("critical_support_rebind_task_ids", [])
        diag.setdefault("critical_support_rebind_support_truck_ids", [])
        diag.setdefault("critical_support_rebind_recovery_anchor_ids", [])
        diag.setdefault("critical_support_rebind_attempt_rows", [])
        diag.setdefault("support_coordination", list(diag.get("support_coordination", [])))

        solution = repair_result.candidate_solution
        task_ids = list(diag.get("critical_recovery_task_ids", ()))
        if not task_ids:
            task_ids = [str(getattr(item, "task_id", "")) for item in getattr(repair_result, "inserted_items", ())]
        assigned_unbound_critical: list[str] = []
        seen_task_ids = {str(tid) for tid in task_ids if str(tid)}
        for _aid, _agent_type in self._agent_sequence_pairs_from_solution(solution):
            seq = tuple(str(x) for x in solution.sequence_for(_aid, _agent_type))
            for tid in seq:
                if not tid or tid in seen_task_ids:
                    continue
                task = getattr(env.state, "tasks", {}).get(str(tid), None)
                if task is None or getattr(task, "status", None) not in {TaskStatus.PENDING, TaskStatus.FAILED}:
                    continue
                if not bool(getattr(task, "kind", None) == TaskKind.EMERGENCY or self._is_timecritical_lightweight_task(task)):
                    continue
                if self._task_support_binding(solution, tid) is not None:
                    continue
                assigned_unbound_critical.append(tid)
                seen_task_ids.add(tid)
        task_ids.extend(assigned_unbound_critical)
        max_tasks = int(max(getattr(env.cfg, "alns_critical_support_rebind_max_tasks", 2), 0))
        if max_tasks <= 0:
            return repair_result
        if bool(getattr(env.cfg, "alns_support_rebind_critical_first_ordering_enabled", False)):
            candidate_cache = {
                str(task_id): self._support_rebind_candidate_rows(env, solution, str(task_id))
                for task_id in task_ids
            }
            task_ids = sorted(
                task_ids,
                key=lambda tid: self._support_rebind_priority_key(env, solution, str(tid))
                + (
                    -float(candidate_cache.get(str(tid), [{}])[0].get("predicted_recovery_margin_m", float("-inf")))
                    if candidate_cache.get(str(tid))
                    else float("inf"),
                ),
            )
        handled = 0
        for task_id in task_ids:
            if handled >= max_tasks:
                break
            task = getattr(env.state, "tasks", {}).get(str(task_id), None)
            if task is None:
                diag["critical_support_rebind_attempt_rows"].append(
                    {"task_id": str(task_id), "result": "skipped", "reason": "TASK_MISSING"}
                )
                continue
            if bool(getattr(env.cfg, "alns_critical_support_rebind_target_only_failed_or_pending", True)):
                if getattr(task, "status", None) not in {TaskStatus.PENDING, TaskStatus.FAILED}:
                    diag["critical_support_rebind_attempt_rows"].append(
                        {
                            "task_id": str(task_id),
                            "result": "skipped",
                            "reason": f"TASK_STATUS_{getattr(task, 'status', None)}",
                        }
                    )
                    continue
            if not bool(getattr(task, "kind", None) == TaskKind.EMERGENCY or self._is_timecritical_lightweight_task(task)):
                diag["critical_support_rebind_attempt_rows"].append(
                    {"task_id": str(task_id), "result": "skipped", "reason": "NOT_CRITICAL_SUPPORT_TARGET"}
                )
                continue
            if self._task_support_binding(solution, str(task_id)) is not None:
                diag["critical_support_rebind_attempt_rows"].append(
                    {"task_id": str(task_id), "result": "skipped", "reason": "ALREADY_HAS_BINDING"}
                )
                continue
            owner_agent = ""
            owner_type = ""
            for aid, agent_type in self._agent_sequence_pairs_from_solution(solution):
                if str(task_id) in tuple(str(x) for x in solution.sequence_for(aid, agent_type)):
                    owner_agent = str(aid)
                    owner_type = str(agent_type)
                    break
            if owner_type not in {"truck", "uav"}:
                diag["critical_support_rebind_attempt_rows"].append(
                    {"task_id": str(task_id), "result": "skipped", "reason": "TASK_NOT_IN_ACTIVE_SEQUENCE"}
                )
                continue
            if owner_type == "truck" and bool(self._truck_task_reachable(env, owner_agent, task)) and bool(
                self._truck_task_direct_serviceable(env, owner_agent, task)
            ):
                diag["critical_support_rebind_attempt_rows"].append(
                    {
                        "task_id": str(task_id),
                        "result": "skipped",
                        "reason": "OWNER_TRUCK_DIRECT_SERVICEABLE",
                        "truck_id": str(owner_agent),
                    }
                )
                continue
            diag["critical_support_rebind_candidates"] += 1
            candidate_rows = self._support_rebind_candidate_rows(env, solution, str(task_id))
            if not candidate_rows:
                diag["critical_support_rebind_rejected_no_truck"] += 1
                diag["critical_support_rebind_attempt_rows"].append(
                    {"task_id": str(task_id), "result": "rejected", "reason": "NO_BINDABLE_TRUCK"}
                )
                continue
            handled += 1
            rebound = False
            for candidate_row in candidate_rows:
                diag["critical_support_rebind_attempts"] += 1
                truck_id = str(candidate_row.get("truck_id", ""))
                uav_id = str(candidate_row.get("uav_id", ""))
                launch_anchor = int(candidate_row.get("launch_anchor", -1))
                recovery_anchor = int(candidate_row.get("recovery_anchor", -1))
                failure_key = self._support_rebind_failure_key(
                    task_id=str(task_id),
                    truck_id=truck_id,
                    uav_id=uav_id,
                    launch_anchor=launch_anchor,
                    recovery_anchor=recovery_anchor,
                )
                predicted_margin = float(candidate_row.get("predicted_recovery_margin_m", float("-inf")))
                predicted_battery_margin = float(candidate_row.get("predicted_battery_margin_ratio", float("-inf")))
                if int(candidate_row.get("failed_binding_penalty_applied", 0)) > 0:
                    diag["critical_support_rebind_failed_binding_penalized"] += 1
                if bool(candidate_row.get("failure_skip_recommended", False)):
                    diag["critical_support_rebind_failed_binding_skipped"] += 1
                    diag["critical_support_rebind_best_rejected_margin_m"] = max(
                        float(diag.get("critical_support_rebind_best_rejected_margin_m", float("-inf"))),
                        predicted_margin,
                    )
                    diag["critical_support_rebind_best_rejected_battery_margin"] = max(
                        float(diag.get("critical_support_rebind_best_rejected_battery_margin", float("-inf"))),
                        predicted_battery_margin,
                    )
                    diag["critical_support_rebind_attempt_rows"].append(
                        {
                            "task_id": str(task_id),
                            "truck_id": truck_id,
                            "uav_id": uav_id,
                            "launch_anchor": launch_anchor,
                            "recovery_anchor": recovery_anchor,
                            "result": "skipped",
                            "reason": "FAILED_BINDING_SKIP",
                            "source": str(candidate_row.get("source", "")),
                            "predicted_recovery_margin_m": predicted_margin,
                            "predicted_battery_margin_ratio": predicted_battery_margin,
                        }
                    )
                    continue
                truck_st = getattr(env.state, "agents", {}).get(truck_id, None)
                uav_st = getattr(env.state, "agents", {}).get(uav_id, None)
                reject_reason = ""
                if truck_st is None or getattr(truck_st, "kind", None) != AgentKind.TRUCK or bool(getattr(truck_st, "crashed", False)):
                    reject_reason = "NO_TRUCK_STATE"
                elif uav_st is None or getattr(uav_st, "kind", None) != AgentKind.UAV or bool(getattr(uav_st, "crashed", False)):
                    reject_reason = "NO_UAV_STATE"
                elif str(getattr(uav_st, "follow_target", "")) != truck_id:
                    reject_reason = "UAV_NOT_DOCKED_TO_TRUCK"
                elif int(candidate_row.get("launch_anchor", -1)) != int(getattr(truck_st, "node", -2)):
                    reject_reason = "INVALID_LAUNCH_ANCHOR"
                else:
                    launch_gate = getattr(env, "_uav_launch_gate_check", None)
                    if callable(launch_gate):
                        try:
                            allowed, reason, _counted = launch_gate(str(uav_id), task=task, count_reject=False)
                        except TypeError:
                            allowed, reason, _counted = launch_gate(str(uav_id), task=task)
                        if not bool(allowed):
                            reject_reason = str(reason or "LAUNCH_GATE_REJECT")
                if not reject_reason and bool(getattr(env.cfg, "alns_support_rebind_safe_uav_guard_enabled", False)):
                    if bool(candidate_row.get("guard_recommended_reject", False)):
                        reject_reason = str(candidate_row.get("override_reason", "") or "SUPPORT_SAFETY_GUARD_REJECT")
                trial = None if reject_reason else self._support_rebind_trial_solution(solution, task_id=str(task_id), target_truck_id=truck_id)
                if not reject_reason and trial is None:
                    reject_reason = "NO_SEQUENCE_CAPACITY"
                if not reject_reason:
                    seq = tuple(str(x) for x in trial.sequence_for(truck_id, "truck"))
                    seq_feas = evaluate_sequence_feasibility(env, truck_id, seq)
                    if not bool(seq_feas.feasible):
                        reject_reason = "|".join(str(x) for x in seq_feas.reason_codes) or "TRIAL_SEQUENCE_INFEASIBLE"
                if not reject_reason:
                    binding = SupportBinding(
                        uav_id=uav_id,
                        truck_id=truck_id,
                        task_id=str(task_id),
                        launch_anchor=int(candidate_row.get("launch_anchor", getattr(truck_st, "node", 0) or 0)),
                        recovery_anchor=int(candidate_row.get("recovery_anchor", getattr(task, "demand_node", 0) or 0)),
                    )
                    sortie = SortiePlan(
                        uav_id=uav_id,
                        task_id=str(task_id),
                        launch_anchor=int(binding.launch_anchor),
                        recovery_anchor=int(binding.recovery_anchor),
                        estimated_launch_step=int(getattr(env.state, "step_index", 0)),
                        estimated_service_step=None,
                        estimated_recovery_step=None,
                    )
                    augmented = self._support_rebind_add_binding(trial, binding=binding, sortie=sortie)
                    if augmented is None:
                        reject_reason = "SUPPORT_BINDING_CONFLICT"
                    else:
                        hard_ok = bool(self._goals_hard_feasible(env, solution_to_legacy_goals(augmented)))
                        augmented_eval = evaluate_k2_solution(env, augmented, hard_feasible=hard_ok)
                        if not bool(augmented_eval.feasible):
                            reject_reason = "|".join(str(x) for x in augmented_eval.infeasibility_reasons) or "AUGMENTED_INFEASIBLE"
                if reject_reason:
                    self._support_rebind_failure_counts[failure_key] = self._support_rebind_failure_counts.get(failure_key, 0) + 1
                    bucket = self._support_rebind_rejection_bucket(reject_reason)
                    diag[bucket] = int(diag.get(bucket, 0)) + 1
                    diag["critical_support_rebind_best_rejected_margin_m"] = max(
                        float(diag.get("critical_support_rebind_best_rejected_margin_m", float("-inf"))),
                        predicted_margin,
                    )
                    diag["critical_support_rebind_best_rejected_battery_margin"] = max(
                        float(diag.get("critical_support_rebind_best_rejected_battery_margin", float("-inf"))),
                        predicted_battery_margin,
                    )
                    diag["critical_support_rebind_attempt_rows"].append(
                        {
                            "task_id": str(task_id),
                            "truck_id": truck_id,
                            "uav_id": uav_id,
                            "launch_anchor": launch_anchor,
                            "recovery_anchor": recovery_anchor,
                            "result": "rejected",
                            "reason": str(reject_reason),
                            "source": str(candidate_row.get("source", "")),
                            "predicted_recovery_margin_m": predicted_margin,
                            "predicted_battery_margin_ratio": predicted_battery_margin,
                        }
                    )
                    continue
                solution = augmented
                self._support_rebind_success_counts[failure_key] = self._support_rebind_success_counts.get(failure_key, 0) + 1
                diag["critical_support_rebind_task_ids"].append(str(task_id))
                diag["critical_support_rebind_support_truck_ids"].append(truck_id)
                diag["critical_support_rebind_recovery_anchor_ids"].append(int(binding.recovery_anchor))
                diag["critical_support_rebind_accept_count"] += 1
                diag["critical_support_rebind_best_accepted_margin_m"] = max(
                    float(diag.get("critical_support_rebind_best_accepted_margin_m", float("-inf"))),
                    predicted_margin,
                )
                diag["critical_support_rebind_best_accepted_battery_margin"] = max(
                    float(diag.get("critical_support_rebind_best_accepted_battery_margin", float("-inf"))),
                    predicted_battery_margin,
                )
                diag["critical_support_rebind_attempt_rows"].append(
                    {
                        "task_id": str(task_id),
                        "truck_id": truck_id,
                        "uav_id": uav_id,
                        "launch_anchor": launch_anchor,
                        "recovery_anchor": recovery_anchor,
                        "result": "accepted",
                        "reason": "",
                        "source": str(candidate_row.get("source", "")),
                        "predicted_recovery_margin_m": predicted_margin,
                        "predicted_battery_margin_ratio": predicted_battery_margin,
                    }
                )
                diag["support_coordination"].append(
                    {
                        "truck_id": truck_id,
                        "uav_id": uav_id,
                        "task_id": str(task_id),
                        "launch_anchor": int(binding.launch_anchor),
                        "recovery_anchor": int(binding.recovery_anchor),
                    }
                )
                if str(candidate_row.get("source", "")) == "historical":
                    diag["critical_support_rebind_historical_reuse"] += 1
                else:
                    diag["critical_support_rebind_reconstructed"] += 1
                rebound = True
                break
            if not rebound:
                continue
        diag["critical_support_rebind_task_ids"] = tuple(diag["critical_support_rebind_task_ids"])
        diag["critical_support_rebind_support_truck_ids"] = tuple(diag["critical_support_rebind_support_truck_ids"])
        diag["critical_support_rebind_recovery_anchor_ids"] = tuple(diag["critical_support_rebind_recovery_anchor_ids"])
        return RepairResult(
            candidate_solution=solution,
            inserted_items=getattr(repair_result, "inserted_items", ()),
            feasible=bool(getattr(repair_result, "feasible", True)),
            reason_codes=tuple(getattr(repair_result, "reason_codes", ())),
            diagnostics=diag,
        )

    def export_adaptive_horizon_records(self) -> List[Dict[str, Any]]:
        return list(self.adaptive_horizon_records)

    def export_local_search_records(self) -> List[Dict[str, Any]]:
        return list(self.local_search_records)

    def _load_live_candidate_ranker(self) -> FeasibilityCandidateRanker:
        if self._live_candidate_ranker is not None:
            return self._live_candidate_ranker
        model_dir = Path(__file__).resolve().parents[3] / "artifacts" / "v2_ranker" / "models"
        schema_path = model_dir / "ranker_feature_schema.json"
        model_path = model_dir / "gradient_boosting.pkl"
        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            feature_columns = tuple(str(x) for x in schema.get("features", ()))
            with model_path.open("rb") as f:
                model = pickle.load(f)
            self._live_candidate_ranker = FeasibilityCandidateRanker(model, feature_columns)
            self._live_candidate_ranker_source = str(model_path)
        except Exception as exc:
            self._live_candidate_ranker = FeasibilityCandidateRanker()
            self._live_candidate_ranker_source = f"fallback:{exc.__class__.__name__}"
        return self._live_candidate_ranker

    def _goals_digest(self, goals: Dict[str, Optional[str]]) -> str:
        payload = {str(k): None if v is None else str(v) for k, v in sorted(goals.items())}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def _task_feature_snapshot(self, env, candidate: Dict[str, Optional[str]]) -> Dict[str, float]:
        state = getattr(env, "state", None)
        tasks_by_id = getattr(state, "tasks", {}) if state is not None else {}
        task_ids = [str(tid) for tid in candidate.values() if tid is not None and str(tid) in tasks_by_id]
        tasks = [tasks_by_id[str(tid)] for tid in task_ids]
        step = int(getattr(state, "step_index", 0))
        if not tasks:
            return {"deadline_slack": 0.0, "lifeline_value": 0.0, "payload": 0.0, "sequence_position": 0.0}
        return {
            "deadline_slack": float(min(float(getattr(t, "deadline_step", step) - step) for t in tasks)),
            "lifeline_value": float(max(float(getattr(t, "lifeline_value", getattr(t, "priority", 0.0)) or 0.0) for t in tasks)),
            "payload": float(max(float(getattr(t, "demand_kg", getattr(t, "payload_kg", 0.0)) or 0.0) for t in tasks)),
            "sequence_position": 0.0,
        }

    def _live_candidate_feature_row(
        self,
        env,
        *,
        current: Dict[str, Optional[str]],
        candidate: Dict[str, Optional[str]],
        destroy_operator: str,
        repair_operator: str,
        legacy_current_score: float,
        legacy_candidate_score: float,
        exact_feasible: bool,
        failure_reason: str,
    ) -> Dict[str, Any]:
        task_features = self._task_feature_snapshot(env, candidate)
        uav_batteries = [
            float(getattr(st, "battery", 0.0))
            for st in getattr(getattr(env, "state", None), "agents", {}).values()
            if getattr(st, "kind", None) == AgentKind.UAV
        ]
        try:
            physical_v2 = getattr(env, "physical_v2", None)
            weather_ledger = getattr(physical_v2, "weather_ledger", [])
            last_weather = weather_ledger[-1] if weather_ledger else None
            wind_speed = float(getattr(last_weather, "wind_speed", 0.0))
            rain_intensity = float(getattr(last_weather, "rain_intensity", 0.0))
            visibility = float(getattr(last_weather, "visibility", 10.0))
            temperature = float(getattr(last_weather, "temperature", 20.0))
            weather_severity = float(last_weather.no_fly_status) if last_weather is not None else 0.0
        except Exception:
            wind_speed = 0.0
            rain_intensity = 0.0
            visibility = 10.0
            temperature = 20.0
            weather_severity = 0.0
        road_blocked = float(getattr(getattr(getattr(env, "state", None), "hazard", None), "blocked_ratio", 0.0))
        row: Dict[str, Any] = {
            "scenario": str(getattr(env.cfg, "scenario", getattr(env.cfg, "phase", ""))),
            "seed": int(getattr(env.cfg, "seed", 0)),
            "step": int(getattr(getattr(env, "state", None), "step_index", 0)),
            "method": "event_responsive_alns",
            "destroy_operator": str(destroy_operator),
            "repair_operator": str(repair_operator),
            "operator": f"{destroy_operator}+{repair_operator}",
            "agent_type": "mixed",
            "task_type": "mixed",
            "sequence_position": float(task_features["sequence_position"]),
            "deadline_slack": float(task_features["deadline_slack"]),
            "lifeline_value": float(task_features["lifeline_value"]),
            "payload": float(task_features["payload"]),
            "battery_reserve": float(min(uav_batteries) if uav_batteries else 0.0),
            "recovery_reserve": float(getattr(env, "physical_v2_minimum_energy_reserve_seen", 0.0)),
            "support_conflict": 0.0,
            "road_damage_probability": float(road_blocked),
            "road_blocked": float(road_blocked > 0.0),
            "weather_severity": float(weather_severity),
            "wind_speed": float(wind_speed),
            "rain_intensity": float(rain_intensity),
            "visibility": float(visibility),
            "temperature": float(temperature),
            "travel_estimate": float(abs(legacy_candidate_score - legacy_current_score)),
            "objective_before": float(legacy_current_score),
            "objective_after": float(legacy_candidate_score),
            "delta_objective": float(legacy_candidate_score - legacy_current_score),
            "feasible": bool(exact_feasible),
            "failure_reason": str(failure_reason),
            "accepted": False,
            "improved": False,
            "runtime": 0.0,
            "ranker_mode": str(getattr(env.cfg, "candidate_ranker_mode", "disabled")).lower(),
            "ranker_source": str(self._live_candidate_ranker_source),
            "exact_feasibility_authoritative": True,
            "current_goals_digest": self._goals_digest(current),
            "candidate_goals_digest": self._goals_digest(candidate),
            "predicted_feasibility": 1.0 if bool(exact_feasible) else 0.0,
        }
        digest_payload = {k: v for k, v in row.items() if k not in {"accepted", "improved", "runtime"}}
        row["candidate_digest"] = hashlib.sha256(
            json.dumps(digest_payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return row

    def _apply_live_ranker(self, env, candidate_row: Dict[str, Any], *, exact_feasibility) -> Tuple[bool, Dict[str, Any]]:
        mode = str(getattr(env.cfg, "candidate_ranker_mode", "disabled")).strip().lower() or "disabled"
        if mode == "disabled":
            return bool(exact_feasibility(candidate_row)), {"ranker_mode": "disabled"}
        ranker = self._load_live_candidate_ranker()
        if mode == "shadow":
            scores, latency = ranker_shadow_scores([candidate_row], ranker)
            diag = {
                "ranker_mode": "shadow",
                "ranker_score": float(scores[0]) if scores else 0.0,
                "ranker_latency": float(latency),
                "ranker_source": str(self._live_candidate_ranker_source),
                "exact_feasibility_authoritative": True,
            }
            self.ranker_runtime_records.append({**candidate_row, **diag})
            return bool(exact_feasibility(candidate_row)), diag
        feasible_candidates, diag_raw = ranker_active_select(
            [candidate_row],
            ranker,
            exact_feasibility=exact_feasibility,
            top_m=1,
            exploration_count=0,
        )
        scores, latency = ranker_shadow_scores([candidate_row], ranker)
        diag = {
            **diag_raw,
            "ranker_mode": "active",
            "ranker_score": float(scores[0]) if scores else 0.0,
            "ranker_latency": float(latency),
            "ranker_source": str(self._live_candidate_ranker_source),
            "exact_feasibility_authoritative": True,
        }
        self.ranker_runtime_records.append({**candidate_row, **diag})
        return bool(feasible_candidates), diag

    def _repair_candidate_to_row(
        self,
        env,
        pool: RepairCandidatePool,
        cand: RepairCandidate,
        *,
        candidate_rank_model: int | None = None,
        selected_for_exact_check: bool = False,
        exact_feasible: bool | None = None,
        failure_reason: str = "",
        objective_delta: float = 0.0,
        accepted: bool = False,
        improved: bool = False,
    ) -> Dict[str, Any]:
        feature_dict = cand.to_feature_dict()
        row: Dict[str, Any] = {
            **feature_dict,
            "pool_id": str(pool.pool_id),
            "candidate_id": str(cand.candidate_id),
            "candidate_digest": str(cand.candidate_id),
            "candidate_rank_before": int(cand.rank_before),
            "candidate_rank_model": -1 if candidate_rank_model is None else int(candidate_rank_model),
            "selected_for_exact_check": bool(selected_for_exact_check),
            "exact_feasible": None if exact_feasible is None else bool(exact_feasible),
            "feasible": None if exact_feasible is None else bool(exact_feasible),
            "failure_reason": str(failure_reason),
            "objective_delta": float(objective_delta),
            "accepted": bool(accepted),
            "improved": bool(improved),
            "task_id": str(cand.task_id),
            "agent_id": str(cand.agent_id),
            "position": int(cand.position),
            "operator": str(cand.operator_name),
            "operator_name": str(cand.operator_name),
            "scenario": str(getattr(env.cfg, "scenario", getattr(env.cfg, "phase", ""))),
            "seed": int(getattr(env.cfg, "seed", 0)),
            "step": int(getattr(getattr(env, "state", None), "step_index", 0)),
            "method": "event_responsive_alns",
            "ranker_mode": str(getattr(env.cfg, "candidate_ranker_mode", "disabled")).lower(),
            "ranker_source": str(self._live_candidate_ranker_source),
            "exact_feasibility_authoritative": True,
        }
        return row

    def _select_from_repair_candidate_pool(
        self,
        env,
        pool: RepairCandidatePool,
        *,
        current_solution,
        current_objective: float,
        repair_operator: str,
    ) -> Tuple[Optional[RepairResult], Dict[str, Any]]:
        mode = str(getattr(env.cfg, "candidate_ranker_mode", "disabled")).strip().lower() or "disabled"
        pool_size_cfg = int(max(getattr(env.cfg, "candidate_ranker_pool_size", 16), 1))
        budget = int(max(getattr(env.cfg, "candidate_ranker_exact_check_budget", 4), 1))
        exploration = int(max(getattr(env.cfg, "candidate_ranker_exploration_count", 1), 0))
        candidates = tuple(pool.candidates[:pool_size_cfg])
        pool_diag: Dict[str, Any] = {
            "step": int(getattr(getattr(env, "state", None), "step_index", 0)),
            "pool_id": str(pool.pool_id),
            "ranker_mode": mode,
            "operator": str(repair_operator),
            "pool_size_configured": int(pool_size_cfg),
            "pool_size_actual": int(len(candidates)),
            "raw_candidate_count": int(pool.raw_candidate_count),
            "duplicate_digest_count": int(pool.duplicate_digest_count),
            "exact_check_budget": int(budget),
            "exploration_count": int(exploration),
            "selected_exact_count": 0,
            "feasible_exact_count": 0,
            "active_exact_checks_lt_pool_size": bool(budget < max(len(candidates), 1)),
        }
        if not candidates:
            self.repair_candidate_pool_records.append(dict(pool_diag))
            return None, pool_diag

        ranker = self._load_live_candidate_ranker() if mode in {"shadow", "active"} else FeasibilityCandidateRanker()
        feature_rows = [cand.to_feature_dict() for cand in candidates]
        if mode in {"shadow", "active"}:
            scores = ranker.score_candidates(feature_rows)
            ranked_idx = sorted(range(len(candidates)), key=lambda i: (-float(scores[i]), i))
        else:
            scores = [float(len(candidates) - i) for i in range(len(candidates))]
            ranked_idx = list(range(len(candidates)))

        if mode == "active":
            selected = select_ranker_candidates(
                candidates,
                scores,
                exact_check_budget=budget,
                exploration_count=exploration,
                rng=self.rng,
            )
        else:
            # Disabled and shadow deliberately share the same raw-order exact budget.
            selected = tuple(candidates[:budget])
        selected_ids = {str(c.candidate_id) for c in selected}
        model_rank = {str(candidates[i].candidate_id): rank for rank, i in enumerate(ranked_idx)}
        selected_rows: list[Dict[str, Any]] = []
        best: tuple[float, RepairCandidate, Any] | None = None
        for cand in candidates:
            selected_for_exact = str(cand.candidate_id) in selected_ids
            exact_feasible: bool | None = None
            failure_reason = ""
            objective_delta = 0.0
            if selected_for_exact:
                ev = evaluate_k2_solution(env, cand.solution, hard_feasible=self._goals_hard_feasible(env, solution_to_legacy_goals(cand.solution)))
                exact_feasible = bool(ev.feasible)
                objective_delta = float(ev.breakdown.total_cost - current_objective)
                if not exact_feasible:
                    failure_reason = "|".join(str(x) for x in getattr(ev, "infeasibility_reasons", ())) or "EXACT_INFEASIBLE"
                elif best is None or float(ev.breakdown.total_cost) < float(best[0]):
                    best = (float(ev.breakdown.total_cost), cand, ev)
            row = self._repair_candidate_to_row(
                env,
                pool,
                cand,
                candidate_rank_model=model_rank.get(str(cand.candidate_id), -1),
                selected_for_exact_check=selected_for_exact,
                exact_feasible=exact_feasible,
                failure_reason=failure_reason,
                objective_delta=objective_delta,
            )
            selected_rows.append(row)
            self.live_candidate_records.append(dict(row))
            if mode in {"shadow", "active"}:
                self.ranker_runtime_records.append(
                    {
                        **row,
                        "ranker_score": float(scores[int(cand.rank_before)]),
                        "ranker_source": str(self._live_candidate_ranker_source),
                    }
                )
        pool_diag["selected_exact_count"] = int(sum(1 for r in selected_rows if bool(r.get("selected_for_exact_check", False))))
        pool_diag["feasible_exact_count"] = int(sum(1 for r in selected_rows if bool(r.get("exact_feasible", False))))
        pool_diag["ranker_top_selected_count"] = int(len(selected_ids))
        self.repair_candidate_pool_records.append(dict(pool_diag))
        if best is None:
            return None, pool_diag
        result = repair_result_from_candidate(env, best[1], pool)
        return result, pool_diag

    def _record_support_execution_event(
        self,
        env,
        *,
        status: str,
        binding=None,
        sortie=None,
        reason: str = "",
        source: str = "",
    ) -> None:
        task_id = getattr(binding, "task_id", getattr(sortie, "task_id", ""))
        uav_id = getattr(binding, "uav_id", getattr(sortie, "uav_id", ""))
        truck_id = getattr(binding, "truck_id", "")
        row = {
            "step": int(getattr(getattr(env, "state", None), "step_index", 0)),
            "status": str(status),
            "reason": str(reason),
            "source": str(source),
            "uav_id": str(uav_id),
            "truck_id": str(truck_id),
            "task_id": str(task_id),
            "launch_anchor": getattr(binding, "launch_anchor", getattr(sortie, "launch_anchor", None)),
            "recovery_anchor": getattr(binding, "recovery_anchor", getattr(sortie, "recovery_anchor", None)),
        }
        seen_key = (str(row["status"]), f"{row['uav_id']}|{row['truck_id']}|{row['task_id']}")
        if str(status) in {"LAUNCHED", "TASK_SERVED", "RECOVERED", "COMPLETED"} and seen_key in self._support_status_seen:
            return
        self._support_status_seen.add(seen_key)
        self.support_execution_records.append(row)
        if status == "CREATED":
            self.alns_diagnostics.support_plan_created_count += 1
        elif status == "INSTALLED":
            self.alns_diagnostics.support_plan_installed_count += 1
        elif status == "LAUNCHED":
            self.alns_diagnostics.support_plan_launched_count += 1
        elif status == "TASK_SERVED":
            self.alns_diagnostics.support_plan_task_served_count += 1
        elif status == "RECOVERED":
            self.alns_diagnostics.support_plan_recovered_count += 1
        elif status == "COMPLETED":
            self.alns_diagnostics.support_plan_completed_count += 1
        elif status == "INVALIDATED":
            self.alns_diagnostics.support_plan_invalidated_count += 1

    def _unified_event_type(self, env) -> str:
        reason = str(getattr(self, "last_replan_reason", "") or "").upper()
        flags = getattr(self, "_last_refresh_flags", {})
        if "BLOCK" in reason or bool(getattr(self, "_road_event_active_for_plan", False)):
            return "ROAD_BLOCKED"
        if "REOPEN" in reason:
            return "ROAD_REOPENED"
        if isinstance(flags, dict) and bool(flags.get("high_priority_uncovered", False)):
            return "NEW_TASK"
        if isinstance(flags, dict) and bool(flags.get("normal_stall", False)):
            return "EXECUTION_STALL"
        if any("TAIL_SUPPORT_BINDING_STALE" in str(r.get("reason_codes", "")) for r in self.k2_runtime_sequence_records[-8:]):
            return "SUPPORT_BINDING_INVALIDATED"
        return "CRITICALITY_ESCALATION"

    def _record_unified_event_trigger(self, env, *, trigger: str) -> str:
        event_type = self._unified_event_type(env)
        step = int(getattr(getattr(env, "state", None), "step_index", 0))
        key = (step, event_type)
        if key in self._deduped_event_keys:
            return event_type
        self._deduped_event_keys.add(key)
        affected_agents = sorted(str(aid) for aid, gid in getattr(self.state, "goals", {}).items() if gid is not None)
        affected_tasks = sorted(str(gid) for gid in getattr(self.state, "goals", {}).values() if gid is not None)
        self.event_trigger_records.append(
            {
                "step": step,
                "event_type": event_type,
                "trigger": str(trigger),
                "hard_or_soft": "hard" if event_type in {"ROAD_BLOCKED", "SUPPORT_BINDING_INVALIDATED", "EXECUTION_STALL"} else "soft",
                "response": "full_alns" if self._solution_mode(env) == "k2_active" else "local_or_shadow",
                "affected_agents": "|".join(affected_agents),
                "affected_tasks": "|".join(affected_tasks),
                "cooldown_steps": int(max(getattr(env.cfg, "hrl_replan_cooldown_steps", 0), 0)),
                "deduplicated": False,
            }
        )
        self.alns_diagnostics.unified_event_trigger_count += 1
        return event_type

    def _tail_reason_codes_for_assignment(self, env, aid: str, tail_task: str) -> Tuple[str, ...]:
        reasons: List[str] = []
        task = getattr(env.state, "tasks", {}).get(str(tail_task), None)
        st = getattr(env.state, "agents", {}).get(str(aid), None)
        if st is None or bool(getattr(st, "crashed", False)):
            reasons.append(TAIL_AGENT_STATE_CONFLICT)
            return tuple(sorted(set(reasons)))
        if task is None:
            reasons.append(TAIL_TASK_COMPLETED)
            return tuple(sorted(set(reasons)))
        if getattr(task, "status", None) != TaskStatus.PENDING:
            if getattr(task, "status", None) == TaskStatus.DELIVERED:
                reasons.append(TAIL_TASK_COMPLETED)
            else:
                reasons.append(TAIL_TASK_NOT_ACTIVE)
            return tuple(sorted(set(reasons)))
        for other_aid, other_runtime in self.sequence_runtime_state.by_agent.items():
            if str(other_aid) == str(aid):
                continue
            other_head = other_runtime.current_head_task
            if other_head is not None and str(other_head) == str(tail_task):
                reasons.append(TAIL_TASK_ASSIGNED_ELSEWHERE)
                reasons.append(TAIL_TASK_DUPLICATE_EXCLUSIVE)
                break
        if getattr(st, "kind", None) == AgentKind.TRUCK:
            if not bool(self._truck_task_valid(env, str(aid), str(tail_task))):
                if not bool(self._truck_task_reachable(env, str(aid), task)):
                    reasons.append(TAIL_ROAD_UNREACHABLE)
                elif not bool(self._truck_task_direct_serviceable(env, str(aid), task)):
                    reasons.append(TAIL_INVENTORY_INSUFFICIENT)
                else:
                    reasons.append(TAIL_AGENT_STATE_CONFLICT)
        elif getattr(st, "kind", None) == AgentKind.UAV:
            if not bool(self._uav_task_feasible(env, str(aid), task)):
                try:
                    margin = float(self._uav_task_margin(env, str(aid), task))
                except Exception:
                    margin = -1.0
                if margin < 0.0:
                    reasons.append(TAIL_UAV_ENERGY_INSUFFICIENT)
                elif bool(getattr(st, "airborne", False)) and getattr(st, "follow_target", None) is None:
                    reasons.append(TAIL_RECOVERY_NOT_FEASIBLE)
                elif getattr(st, "follow_target", None) is None:
                    reasons.append(TAIL_SUPPORT_BINDING_STALE)
                else:
                    reasons.append(TAIL_AGENT_STATE_CONFLICT)
        return tuple(sorted(set(reasons)))

    def validate_runtime_tail(self, env, runtime_sequence: AgentSequenceRuntime) -> TailValidationResult:
        if not runtime_sequence.tail_tasks:
            return TailValidationResult(valid=False, reason_codes=(TAIL_TASK_NOT_ACTIVE,), promotable=False)
        tail_task = str(runtime_sequence.tail_tasks[0])
        reasons = list(self._tail_reason_codes_for_assignment(env, str(runtime_sequence.agent_id), tail_task))
        candidate_goals = dict(self.state.goals)
        candidate_goals[str(runtime_sequence.agent_id)] = str(tail_task)
        if reasons:
            return TailValidationResult(valid=False, reason_codes=tuple(sorted(set(reasons))), promotable=False)
        if not bool(self._goals_hard_feasible(env, candidate_goals)):
            reasons.extend(self._tail_reason_codes_for_assignment(env, str(runtime_sequence.agent_id), tail_task))
        unique_reasons = tuple(sorted(set(str(x) for x in reasons)))
        return TailValidationResult(valid=not unique_reasons, reason_codes=unique_reasons, promotable=not unique_reasons)

    def _advance_runtime_sequences(self, env) -> None:
        if self._solution_mode(env) != "k2_active":
            return
        if not self.sequence_runtime_state.by_agent:
            if any(g is not None for g in self.state.goals.values()):
                self.alns_diagnostics.sequence_runtime_missing_count += 1
            return
        updated_goals = dict(self.state.goals)
        step_now = int(getattr(env.state, "step_index", 0))
        for aid, runtime in list(sorted(self.sequence_runtime_state.by_agent.items(), key=lambda kv: kv[0])):
            seq_before = tuple(str(x) for x in runtime.planned_sequence)
            if not seq_before:
                continue
            head_task = runtime.current_head_task
            head_state = getattr(env.state, "tasks", {}).get(str(head_task), None) if head_task is not None else None
            head_status = getattr(head_state, "status", None)
            head_active = head_state is not None and head_status in {TaskStatus.PENDING, TaskStatus.CLAIMED}
            if not head_active:
                head_reason = "HEAD_COMPLETED" if head_status == TaskStatus.DELIVERED else "HEAD_INVALIDATED"
                for binding in tuple(getattr(self, "_installed_support_bindings", ())):
                    if str(getattr(binding, "task_id", "")) == str(head_task):
                        if head_reason == "HEAD_COMPLETED":
                            self._record_support_execution_event(env, status="TASK_SERVED", binding=binding, reason=head_reason, source="runtime")
                            self._record_support_execution_event(env, status="RECOVERED", binding=binding, reason=head_reason, source="runtime")
                            self._record_support_execution_event(env, status="COMPLETED", binding=binding, reason=head_reason, source="runtime")
                        else:
                            self._record_support_execution_event(env, status="INVALIDATED", binding=binding, reason=head_reason, source="runtime")
                self._record_runtime_event(
                    env,
                    agent_id=str(aid),
                    agent_type=runtime.agent_type,
                    event_type=head_reason,
                    event_reason=head_reason,
                    sequence_before=seq_before,
                    sequence_after=seq_before,
                    solution_digest_before=runtime.solution_digest,
                    solution_digest_after=runtime.solution_digest,
                    trigger="step_update",
                    validation_result="head_terminal",
                    reason_codes=(),
                )
                if runtime.tail_tasks:
                    validation = self.validate_runtime_tail(env, runtime)
                    if validation.promotable:
                        seq_after = (str(runtime.tail_tasks[0]),)
                        updated_goals[str(aid)] = str(runtime.tail_tasks[0])
                        self.state.goal_assigned_step[str(aid)] = int(step_now)
                        self.sequence_runtime_state.register_tail_lifetime(runtime.tail_created_step, step_now)
                        promoted = runtime.with_sequence(
                            seq_after,
                            created_step=step_now,
                            last_validated_step=step_now,
                            status="TAIL_READY",
                            tail_created_step=None,
                        )
                        self.sequence_runtime_state.set_runtime(promoted)
                        self.alns_diagnostics.k2_tail_promoted_count += 1
                        if head_reason == "HEAD_COMPLETED":
                            self.alns_diagnostics.tail_promoted_after_head_completion_count += 1
                        else:
                            self.alns_diagnostics.tail_promoted_after_head_invalidation_count += 1
                        self._record_runtime_event(
                            env,
                            agent_id=str(aid),
                            agent_type=runtime.agent_type,
                            event_type="TAIL_PROMOTED",
                            event_reason=head_reason,
                            sequence_before=seq_before,
                            sequence_after=seq_after,
                            solution_digest_before=runtime.solution_digest,
                            solution_digest_after=runtime.solution_digest,
                            trigger="step_update",
                            validation_result="promoted",
                            reason_codes=validation.reason_codes,
                        )
                    else:
                        updated_goals[str(aid)] = None
                        self.state.goal_assigned_step.pop(str(aid), None)
                        self.sequence_runtime_state.register_tail_lifetime(runtime.tail_created_step, step_now)
                        self.sequence_runtime_state.increment_reason_count("invalidation", validation.reason_codes)
                        self.sequence_runtime_state.remove_runtime(str(aid))
                        self.alns_diagnostics.k2_tail_invalidated_count += 1
                        self.alns_diagnostics.k2_tail_dropped_count += 1
                        self._record_runtime_event(
                            env,
                            agent_id=str(aid),
                            agent_type=runtime.agent_type,
                            event_type="TAIL_INVALIDATED",
                            event_reason=head_reason,
                            sequence_before=seq_before,
                            sequence_after=(),
                            solution_digest_before=runtime.solution_digest,
                            solution_digest_after=None,
                            trigger="step_update",
                            validation_result="rejected",
                            reason_codes=validation.reason_codes,
                        )
                        self._record_runtime_event(
                            env,
                            agent_id=str(aid),
                            agent_type=runtime.agent_type,
                            event_type="SEQUENCE_CLEARED",
                            event_reason=head_reason,
                            sequence_before=seq_before,
                            sequence_after=(),
                            solution_digest_before=runtime.solution_digest,
                            solution_digest_after=None,
                            trigger="step_update",
                            validation_result="cleared",
                            reason_codes=validation.reason_codes,
                        )
                else:
                    updated_goals[str(aid)] = None
                    self.state.goal_assigned_step.pop(str(aid), None)
                    self.sequence_runtime_state.remove_runtime(str(aid))
                    self.alns_diagnostics.k2_tail_dropped_count += 1
                    self._record_runtime_event(
                        env,
                        agent_id=str(aid),
                        agent_type=runtime.agent_type,
                        event_type="SEQUENCE_CLEARED",
                        event_reason=head_reason,
                        sequence_before=seq_before,
                        sequence_after=(),
                        solution_digest_before=runtime.solution_digest,
                        solution_digest_after=None,
                        trigger="step_update",
                        validation_result="cleared",
                        reason_codes=(),
                    )
                continue
            if runtime.tail_tasks:
                validation = self.validate_runtime_tail(env, runtime)
                if validation.valid:
                    for binding in tuple(getattr(self, "_installed_support_bindings", ())):
                        if str(getattr(binding, "task_id", "")) == str(head_task):
                            self._record_support_execution_event(env, status="LAUNCHED", binding=binding, reason="HEAD_ACTIVE", source="runtime")
                    self.alns_diagnostics.sequence_retained_step_count += 1
                    retained = runtime.with_sequence(
                        seq_before,
                        last_validated_step=step_now,
                        status="HEAD_WITH_TAIL",
                    )
                    self.sequence_runtime_state.set_runtime(retained)
                    self._record_runtime_event(
                        env,
                        agent_id=str(aid),
                        agent_type=runtime.agent_type,
                        event_type="SEQUENCE_RETAINED",
                        event_reason="TAIL_RETAINED",
                        sequence_before=seq_before,
                        sequence_after=seq_before,
                        solution_digest_before=runtime.solution_digest,
                        solution_digest_after=runtime.solution_digest,
                        trigger="step_update",
                        validation_result="retained",
                        reason_codes=validation.reason_codes,
                    )
                else:
                    seq_after = (str(head_task),)
                    updated_goals[str(aid)] = str(head_task)
                    self.sequence_runtime_state.register_tail_lifetime(runtime.tail_created_step, step_now)
                    self.sequence_runtime_state.increment_reason_count("invalidation", validation.reason_codes)
                    updated_runtime = runtime.with_sequence(
                        seq_after,
                        last_validated_step=step_now,
                        status="HEAD_ONLY",
                        tail_created_step=None,
                    )
                    self.sequence_runtime_state.set_runtime(updated_runtime)
                    self.alns_diagnostics.k2_tail_invalidated_count += 1
                    self._record_runtime_event(
                        env,
                        agent_id=str(aid),
                        agent_type=runtime.agent_type,
                        event_type="TAIL_INVALIDATED",
                        event_reason="TAIL_INVALIDATED",
                        sequence_before=seq_before,
                        sequence_after=seq_after,
                        solution_digest_before=runtime.solution_digest,
                        solution_digest_after=runtime.solution_digest,
                        trigger="step_update",
                        validation_result="rejected",
                        reason_codes=validation.reason_codes,
                    )
        self.state.goals = updated_goals
        self._refresh_tail_lifetime_metrics()

    def _maybe_reuse_tail_after_replan(self, env, aid: str, agent_type: str, old_runtime: AgentSequenceRuntime, new_seq: Tuple[str, ...]):
        if len(new_seq) >= 2 or len(old_runtime.tail_tasks) == 0 or not new_seq:
            return tuple(new_seq), False
        if str(new_seq[0]) != str(old_runtime.current_head_task):
            return tuple(new_seq), False
        validation = self.validate_runtime_tail(env, old_runtime)
        if not validation.promotable:
            return tuple(new_seq), False
        reused_seq = (str(new_seq[0]), str(old_runtime.tail_tasks[0]))
        trial_goals = dict(self.state.goals)
        trial_goals[str(aid)] = reused_seq[0]
        base_solution = construct_k2_solution(env, trial_goals)
        reused_solution = base_solution.with_sequence(str(aid), agent_type, reused_seq)
        base_eval = evaluate_k2_solution(env, base_solution, hard_feasible=self._goals_hard_feasible(env, trial_goals))
        reused_eval = evaluate_k2_solution(env, reused_solution, hard_feasible=self._goals_hard_feasible(env, trial_goals))
        if reused_eval.feasible and reused_eval.breakdown.total_cost <= base_eval.breakdown.total_cost + 1e-12:
            return reused_seq, True
        return tuple(new_seq), False

    def _install_runtime_solution(self, env, solution, *, trigger: str):
        if self._solution_mode(env) not in {"k2_shadow", "k2_active"}:
            return
        unified_trigger = self._record_unified_event_trigger(env, trigger=trigger)
        for binding in tuple(getattr(solution, "support_bindings", ())):
            self._record_support_execution_event(env, status="CREATED", binding=binding, reason=unified_trigger, source="solution")
            self._record_support_execution_event(env, status="INSTALLED", binding=binding, reason=unified_trigger, source="runtime")
            self._record_support_execution_event(env, status="LAUNCH_PENDING", binding=binding, reason=unified_trigger, source="runtime")
        for sortie in tuple(getattr(solution, "sortie_plans", ())):
            self._record_support_execution_event(env, status="CREATED", sortie=sortie, reason=unified_trigger, source="sortie_plan")
        old_by_agent = dict(self.sequence_runtime_state.by_agent)
        new_state = SequenceRuntimeState(
            by_agent={},
            tail_lifetime_steps=list(self.sequence_runtime_state.tail_lifetime_steps),
            tail_invalidation_reason_counts=dict(self.sequence_runtime_state.tail_invalidation_reason_counts),
            tail_replacement_reason_counts=dict(self.sequence_runtime_state.tail_replacement_reason_counts),
        )
        step_now = int(getattr(env.state, "step_index", 0))
        for aid, agent_type in self._agent_sequence_pairs(env):
            new_seq_raw = tuple(str(x) for x in solution.sequence_for(aid, agent_type))
            old_runtime = old_by_agent.get(str(aid), None)
            if old_runtime is not None and new_seq_raw:
                reused_seq, reused = self._maybe_reuse_tail_after_replan(env, aid, agent_type, old_runtime, new_seq_raw)
                if reused:
                    new_seq_raw = tuple(reused_seq)
                    solution = solution.with_sequence(aid, agent_type, new_seq_raw)
                    self.alns_diagnostics.tail_reused_after_replan_count += 1
                    self._record_runtime_event(
                        env,
                        agent_id=aid,
                        agent_type=agent_type,
                        event_type="TAIL_REUSED_AFTER_REPLAN",
                        event_reason=trigger,
                        sequence_before=tuple(str(x) for x in old_runtime.planned_sequence),
                        sequence_after=new_seq_raw,
                        solution_digest_before=old_runtime.solution_digest,
                        solution_digest_after=solution.digest(),
                        trigger=trigger,
                        validation_result="reused",
                        reason_codes=(),
                    )
            old_seq = tuple(str(x) for x in old_runtime.planned_sequence) if old_runtime is not None else ()
            new_seq = tuple(str(x) for x in new_seq_raw)
            if not new_seq:
                if old_runtime is not None:
                    if old_runtime.tail_tasks:
                        new_state.register_tail_lifetime(old_runtime.tail_created_step, step_now)
                    self._record_runtime_event(
                        env,
                        agent_id=aid,
                        agent_type=agent_type,
                        event_type="SEQUENCE_CLEARED",
                        event_reason=trigger,
                        sequence_before=old_seq,
                        sequence_after=(),
                        solution_digest_before=old_runtime.solution_digest,
                        solution_digest_after=None,
                        trigger=trigger,
                        validation_result="cleared",
                        reason_codes=(),
                    )
                continue
            runtime = runtime_from_solution(
                agent_id=aid,
                agent_type=agent_type,
                sequence=new_seq,
                solution_digest=solution.digest(),
                step=step_now,
                source_replan_reason=trigger,
                status="HEAD_WITH_TAIL" if len(new_seq) >= 2 else "HEAD_ONLY",
                existing=old_runtime,
            )
            new_state.set_runtime(runtime)
            self.alns_diagnostics.sequence_installed_count += 1
            if len(new_seq) >= 2:
                self.alns_diagnostics.sequence_installed_with_tail_count += 1
            event_type = "SEQUENCE_INSTALLED"
            if old_runtime is not None:
                if old_seq == new_seq:
                    event_type = "SEQUENCE_RETAINED"
                elif old_seq[:1] == new_seq[:1]:
                    if old_seq[1:] != new_seq[1:]:
                        event_type = "TAIL_REPLACED_BY_REPLAN"
                        self.alns_diagnostics.tail_replaced_by_replan_count += 1
                        self.alns_diagnostics.tail_changed_by_replan_count += 1
                        new_state.increment_reason_count("replacement", (TAIL_REPLAN_REPLACED,))
                        new_state.register_tail_lifetime(old_runtime.tail_created_step, step_now)
                else:
                    self.alns_diagnostics.head_changed_by_replan_count += 1
                    if old_seq[1:] != new_seq[1:]:
                        self.alns_diagnostics.full_sequence_changed_by_replan_count += 1
                        if old_runtime.tail_tasks:
                            new_state.register_tail_lifetime(old_runtime.tail_created_step, step_now)
            if old_runtime is None:
                self._record_runtime_event(
                    env,
                    agent_id=aid,
                    agent_type=agent_type,
                    event_type=event_type,
                    event_reason=trigger,
                    sequence_before=(),
                    sequence_after=new_seq,
                    solution_digest_before=None,
                    solution_digest_after=runtime.solution_digest,
                    trigger=trigger,
                    validation_result="installed",
                    reason_codes=(),
                )
            else:
                self._record_runtime_event(
                    env,
                    agent_id=aid,
                    agent_type=agent_type,
                    event_type=event_type,
                    event_reason=trigger,
                    sequence_before=old_seq,
                    sequence_after=new_seq,
                    solution_digest_before=old_runtime.solution_digest,
                    solution_digest_after=runtime.solution_digest,
                    trigger=trigger,
                    validation_result="replanned",
                    reason_codes=((TAIL_REPLAN_REPLACED,) if event_type == "TAIL_REPLACED_BY_REPLAN" else ()),
                )
        self.sequence_runtime_state = new_state
        self._installed_support_bindings = tuple(getattr(solution, "support_bindings", ()))
        self._installed_sortie_plans = tuple(getattr(solution, "sortie_plans", ()))
        self._refresh_tail_lifetime_metrics()
        return solution

    def _record_k2_sa_delta(
        self,
        env,
        *,
        current_eval,
        candidate_eval,
        current_solution=None,
        candidate_solution=None,
        delta: float,
        temperature: float,
        acceptance_probability: float | None = None,
        random_draw: float | None = None,
        accepted: bool,
        destroy_operator: str = "",
        repair_operator: str = "",
        fallback_used: bool = False,
    ) -> None:
        current_cost = float(current_eval.breakdown.total_cost)
        candidate_cost = float(candidate_eval.breakdown.total_cost)
        improved = bool(candidate_cost < current_cost - 1e-12)
        equal = bool(abs(candidate_cost - current_cost) <= 1e-12)
        probability = (
            float(acceptance_probability)
            if acceptance_probability is not None
            else (1.0 if delta <= 0.0 else float(math.exp(-float(delta) / max(float(temperature), 1e-12))))
        )
        current_digest = None if current_solution is None else str(current_solution.digest())
        candidate_digest = None if candidate_solution is None else str(candidate_solution.digest())
        solutions_identical = bool(current_digest is not None and candidate_digest is not None and current_digest == candidate_digest)
        if not np.isfinite(current_cost) or not np.isfinite(candidate_cost):
            rejection_reason = "NONFINITE_OBJECTIVE"
        elif solutions_identical:
            rejection_reason = "IDENTICAL_SOLUTION"
        elif bool(candidate_eval.feasible) and float(delta) < -1e-12:
            rejection_reason = "IMPROVING_DELTA"
        elif bool(candidate_eval.feasible) and abs(float(delta)) <= 1e-12:
            rejection_reason = "ZERO_DELTA"
        elif bool(candidate_eval.feasible) and float(delta) > 0.0 and probability <= 1e-6:
            rejection_reason = "TEMPERATURE_SCALE_MISMATCH"
        elif bool(fallback_used) and solutions_identical:
            rejection_reason = "FALLBACK_RESTORES_CURRENT"
        elif not bool(candidate_eval.feasible):
            rejection_reason = "CANDIDATE_INFEASIBLE"
        else:
            rejection_reason = "WORSENING_DELTA"
        self.k2_sa_delta_records.append(
            {
                "step": int(getattr(getattr(env, "state", None), "step_index", 0)),
                "current_solution_digest": current_digest,
                "candidate_solution_digest": candidate_digest,
                "solutions_identical": bool(solutions_identical),
                "current_total_cost": current_cost,
                "candidate_total_cost": candidate_cost,
                "delta_cost": float(delta),
                "temperature": float(temperature),
                "random_draw": None if random_draw is None else float(random_draw),
                "improved": bool(improved),
                "equal_candidate": bool(equal),
                "acceptance_probability": float(np.clip(probability, 0.0, 1.0)),
                "accepted": bool(accepted),
                "rejection_reason": "ACCEPTED" if bool(accepted) else rejection_reason,
                "destroy_operator": str(destroy_operator),
                "repair_operator": str(repair_operator),
                "fallback_used": bool(fallback_used),
                "current_feasible": bool(current_eval.feasible),
                "candidate_feasible": bool(candidate_eval.feasible),
                "current_infeasibility_reasons": list(getattr(current_eval, "infeasibility_reasons", ())),
                "candidate_infeasibility_reasons": list(getattr(candidate_eval, "infeasibility_reasons", ())),
                "current_hard_violation_cost": float(current_eval.breakdown.hard_violation_cost),
                "current_unserved_timecritical_cost": float(current_eval.breakdown.unserved_timecritical_cost),
                "current_lifeline_loss_cost": float(current_eval.breakdown.lifeline_loss_cost),
                "current_unserved_routine_cost": float(current_eval.breakdown.unserved_routine_cost),
                "current_travel_cost": float(current_eval.breakdown.travel_cost),
                "current_energy_cost": float(current_eval.breakdown.energy_cost),
                "current_switching_cost": float(current_eval.breakdown.switching_cost),
                "current_support_cost": float(current_eval.breakdown.support_cost),
                "candidate_hard_violation_cost": float(candidate_eval.breakdown.hard_violation_cost),
                "candidate_unserved_timecritical_cost": float(candidate_eval.breakdown.unserved_timecritical_cost),
                "candidate_lifeline_loss_cost": float(candidate_eval.breakdown.lifeline_loss_cost),
                "candidate_unserved_routine_cost": float(candidate_eval.breakdown.unserved_routine_cost),
                "candidate_travel_cost": float(candidate_eval.breakdown.travel_cost),
                "candidate_energy_cost": float(candidate_eval.breakdown.energy_cost),
                "candidate_switching_cost": float(candidate_eval.breakdown.switching_cost),
                "candidate_support_cost": float(candidate_eval.breakdown.support_cost),
            }
        )

    def plan(self, env) -> Dict[str, Optional[str]]:
        self._reset_k2_runtime_if_needed(env)
        self._advance_runtime_sequences(env)
        self._update_adaptive_rolling_params(env)
        return super().plan(env)

    def _plan_once(self, env) -> Dict[str, Optional[str]]:
        base_goals = super()._plan_once(env)
        # Route-plan v2 already performs ALNS over complete cooperative task
        # lines.  The legacy K1/K2 goal-map optimizer is intentionally kept
        # below for compatibility, but must not overwrite the current stop of
        # a persistent v2 route.
        if er_hlns_route_plan_active(env):
            route_alns = self._route_plan_v2
            self.alns_iteration_count_total = int(
                route_alns.alns_iteration_count
            )
            self.alns_destroyed_assignment_count_total = int(
                route_alns.alns_destroyed_assignment_count
            )
            self.alns_repair_attempt_count_total = int(
                route_alns.alns_repair_attempt_count
            )
            self.alns_repair_feasible_count_total = int(
                route_alns.alns_repair_feasible_count
            )
            self.alns_accepted_count_total = int(
                route_alns.alns_accepted_count
            )
            self.alns_improvement_count_total = int(
                route_alns.alns_improvement_count
            )
            self.alns_diagnostics.iteration_count = int(
                route_alns.alns_iteration_count
            )
            self.alns_diagnostics.destroyed_assignment_count = int(
                route_alns.alns_destroyed_assignment_count
            )
            self.alns_diagnostics.repair_attempt_count = int(
                route_alns.alns_repair_attempt_count
            )
            self.alns_diagnostics.repair_feasible_count = int(
                route_alns.alns_repair_feasible_count
            )
            self.alns_diagnostics.accepted_count = int(
                route_alns.alns_accepted_count
            )
            self.alns_diagnostics.accepted_improving_count = int(
                route_alns.alns_accepted_count
            )
            self.alns_diagnostics.improvement_count = int(
                route_alns.alns_improvement_count
            )
            self.alns_diagnostics.replan_count = int(
                route_alns.alns_replan_count
            )
            self.alns_diagnostics.objective_evaluation_count = int(
                route_alns.alns_objective_evaluation_count
            )
            self.alns_diagnostics.feasibility_evaluation_count = int(
                route_alns.alns_feasibility_evaluation_count
            )
            self.alns_diagnostics.wall_clock_time_s = float(
                route_alns.alns_wall_clock_time_s
            )
            return base_goals
        if not bool(getattr(env.cfg, "alns_enabled", True)):
            return base_goals
        self._update_risk_pressure(env)
        self._road_event_active_for_plan = bool(self._recent_road_event_active(env))
        goals = self._alns_optimize_goals(env, base_goals)
        mode = self._solution_mode(env)
        if mode in {"k2_shadow", "k2_active"}:
            solution, _ev, _rec = self._record_k2_solution(env, "selected", goals)
            solution = self._install_runtime_solution(
                env,
                solution,
                trigger=str(getattr(self, "last_replan_reason", "refresh")),
            )
            if mode == "k2_active":
                selected_goals = solution_to_legacy_goals(solution)
                self._record_task_assignment_observations(selected_goals, int(getattr(env.state, "step_index", 0)))
                return selected_goals
        elif self.sequence_runtime_state.by_agent:
            self._reset_k2_runtime_state("legacy_plan")
        self._record_task_assignment_observations(goals, int(getattr(env.state, "step_index", 0)))
        return goals

    def _update_risk_pressure(self, env) -> None:
        self._risk_pressure_by_node = {}
        risk_pressure_enabled = bool(
            getattr(
                env.cfg,
                "alns_risk_pressure_enabled",
                getattr(env.cfg, "alns_ghost_tasks_enabled", True),
            )
        )
        if not risk_pressure_enabled:
            return
        blocked_ratio = float(getattr(env, "blocked_ratio_total", getattr(env, "blocked_ratio", 0.0)))
        island_ids = set()
        fn = getattr(env, "_current_island_emergency_task_ids", None)
        if callable(fn):
            try:
                island_ids = set(fn())
            except Exception:
                island_ids = set()
        for task in env.state.tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
            node = int(getattr(task, "demand_node", -1))
            if node < 0:
                continue
            urgency = 1.0 - float(np.clip(self._task_lifeline_ratio(task), 0.0, 1.0))
            island_bonus = 0.35 if str(getattr(task, "task_id", "")) in island_ids else 0.0
            kind_bonus = 0.25 if task.kind == TaskKind.EMERGENCY else 0.05
            pressure = float(np.clip(0.20 + 0.60 * urgency + 0.80 * blocked_ratio + island_bonus + kind_bonus, 0.0, 2.0))
            self._risk_pressure_by_node[node] = max(float(self._risk_pressure_by_node.get(node, 0.0)), pressure)
        count = int(len(self._risk_pressure_by_node))
        self.alns_risk_pressure_task_count_total = int(self.alns_risk_pressure_task_count_total) + count
        self.alns_ghost_task_count_total = int(self.alns_risk_pressure_task_count_total)

    def _blocked_edges_snapshot(self, env) -> Set[Tuple[int, int]]:
        edges = getattr(getattr(env, "topology", None), "blocked_edges", set())
        out: Set[Tuple[int, int]] = set()
        try:
            for e in edges:
                if e is None or len(e) < 2:
                    continue
                a, b = int(e[0]), int(e[1])
                out.add((min(a, b), max(a, b)))
        except Exception:
            return set()
        return out

    def _recent_road_event_active(self, env) -> bool:
        current = self._blocked_edges_snapshot(env)
        new_edges = current.difference(self._last_blocked_edges_seen)
        self._last_blocked_edges_seen = set(current)
        if new_edges:
            return True
        flags = getattr(self, "_last_refresh_flags", {})
        if isinstance(flags, dict) and (
            bool(flags.get("hard_reason_path_blocked", False))
            or bool(flags.get("route_blocked", False))
            or int(flags.get("map_update_hard_reason_path_blocked_step", 0) or 0) > 0
        ):
            return True
        return bool(int(getattr(env, "_shared_map_new_blocked_step", 0) or 0) > 0)

    def _risk_pressure_for_task(self, task) -> float:
        node = int(getattr(task, "demand_node", -1))
        return float(self._risk_pressure_by_node.get(node, 0.0))

    def _any_uav_direct_feasible(self, env, task) -> bool:
        if task is None or getattr(task, "kind", None) != TaskKind.EMERGENCY:
            return False
        for uid, st in env.state.agents.items():
            if st.kind != AgentKind.UAV or bool(getattr(st, "crashed", False)):
                continue
            try:
                if bool(self._uav_task_feasible(env, str(uid), task)):
                    return True
            except Exception:
                continue
        return False

    def _sample_operator(self, ops: Dict[str, ALNSOperatorStats]) -> str:
        names = sorted(ops.keys())
        if str(getattr(self, "_selection_mode", "adaptive")) == "uniform":
            return str(names[int(self.rng.integers(0, len(names)))])
        weights = np.asarray([max(float(ops[n].weight), 1e-6) for n in names], dtype=np.float64)
        weights = weights / max(float(weights.sum()), 1e-9)
        idx = int(self.rng.choice(len(names), p=weights))
        return str(names[idx])

    def _requested_operator_pool(self, env) -> str:
        requested = str(getattr(env.cfg, "alns_operator_pool", "legacy")).strip().lower() or "legacy"
        aliases = {
            "canonical_only": "canonical_k2",
            "er_only": "er_k2",
            "combined": "combined_k2",
            "no_road_group": "combined_k2",
            "no_critical_group": "combined_k2",
            "no_support_group": "combined_k2",
            "no_synchronization_group": "combined_k2",
        }
        return str(aliases.get(requested, requested))

    def _raw_operator_group(self, env) -> str:
        return str(getattr(env.cfg, "alns_operator_pool", "legacy")).strip().lower() or "legacy"

    def _use_k2_operator_pool(self, env) -> bool:
        return bool(
            self._solution_mode(env) == "k2_active"
            and self._requested_operator_pool(env) in {"canonical_k2", "er_k2", "combined_k2"}
        )

    def _ensure_operator_pool(self, env) -> None:
        requested = self._requested_operator_pool(env) if self._use_k2_operator_pool(env) else "legacy"
        if requested == self._operator_pool_name:
            return
        raw_group = self._raw_operator_group(env)
        weight_profile = str(getattr(env.cfg, "alns_operator_weight_profile", "uniform")).strip().lower() or "uniform"

        def _make_stats(names, phase: str):
            weights = {}
            for name in names:
                weight = 1.0
                if weight_profile == "critical_repair_bias":
                    if phase == "destroy":
                        if name == "critical_task_reassignment_removal":
                            weight = 1.35
                        elif name in {"road_disruption_removal", "support_conflict_removal"}:
                            weight = 1.15
                    else:
                        if name == "critical_first_insertion":
                            weight = 1.40
                        elif name in {"risk_aware_insertion", "feasibility_restoration_insertion"}:
                            weight = 1.20
                elif weight_profile == "feasibility_restore_bias":
                    if phase == "destroy":
                        if name == "synchronization_risk_removal":
                            weight = 1.30
                        elif name in {"support_conflict_removal", "road_disruption_removal"}:
                            weight = 1.15
                    else:
                        if name == "feasibility_restoration_insertion":
                            weight = 1.45
                        elif name in {"risk_aware_insertion", "synchronized_insertion"}:
                            weight = 1.20
                weights[str(name)] = ALNSOperatorStats(float(weight))
            return weights

        if requested == "canonical_k2":
            self.destroy_ops = _make_stats(CANONICAL_DESTROY_NAMES, "destroy")
            self.repair_ops = _make_stats(CANONICAL_REPAIR_NAMES, "repair")
        elif requested == "er_k2":
            self.destroy_ops = _make_stats(ER_DESTROY_NAMES, "destroy")
            repair_names = list(ER_REPAIR_NAMES)
            if not bool(getattr(env.cfg, "alns_critical_recovery_repair_enabled", False)):
                repair_names = [n for n in repair_names if n != "critical_recovery_repair_insertion"]
            self.repair_ops = _make_stats(repair_names, "repair")
        elif requested == "combined_k2":
            destroy_names = list((*CANONICAL_DESTROY_NAMES, *ER_DESTROY_NAMES))
            repair_names = list((*CANONICAL_REPAIR_NAMES, *ER_REPAIR_NAMES))
            if not bool(getattr(env.cfg, "alns_critical_recovery_repair_enabled", False)):
                repair_names = [n for n in repair_names if n != "critical_recovery_repair_insertion"]
            if raw_group == "no_road_group":
                destroy_names = [n for n in destroy_names if n != "road_disruption_removal"]
                repair_names = [n for n in repair_names if n != "risk_aware_insertion"]
            elif raw_group == "no_critical_group":
                destroy_names = [n for n in destroy_names if n != "critical_task_reassignment_removal"]
                repair_names = [n for n in repair_names if n != "critical_first_insertion"]
            elif raw_group == "no_support_group":
                destroy_names = [n for n in destroy_names if n != "support_conflict_removal"]
                repair_names = [n for n in repair_names if n != "synchronized_insertion"]
            elif raw_group == "no_synchronization_group":
                destroy_names = [n for n in destroy_names if n != "synchronization_risk_removal"]
                repair_names = [n for n in repair_names if n != "feasibility_restoration_insertion"]
            self.destroy_ops = _make_stats(destroy_names, "destroy")
            self.repair_ops = _make_stats(repair_names, "repair")
        else:
            self.destroy_ops = {
                "road_disruption": ALNSOperatorStats(1.25),
                "stale_or_low_value": ALNSOperatorStats(1.00),
                "tc_uncovered": ALNSOperatorStats(1.20),
                "support_gap": ALNSOperatorStats(1.15),
                "random_light": ALNSOperatorStats(0.75),
            }
            self.repair_ops = {
                "direct_service_insert": ALNSOperatorStats(1.30),
                "risk_greedy_insert": ALNSOperatorStats(1.20),
                "tc_first_insert": ALNSOperatorStats(1.15),
                "risk_balanced_insert": ALNSOperatorStats(1.00),
            }
        self._operator_pool_name = requested
        self.alns_diagnostics.initial_destroy_weights = {k: float(v.weight) for k, v in self.destroy_ops.items()}
        self.alns_diagnostics.initial_repair_weights = {k: float(v.weight) for k, v in self.repair_ops.items()}

    def _k2_destroy_solution(self, env, goals: Dict[str, Optional[str]], op: str):
        solution = construct_k2_solution(env, goals)
        if op == "random_removal":
            return random_removal(env, solution, self.rng)
        if op == "worst_cost_removal":
            return worst_cost_removal(env, solution, self.rng)
        if op == "related_removal":
            return related_removal(env, solution, self.rng)
        if op == "sequence_segment_removal":
            return sequence_segment_removal(env, solution, self.rng)
        if op == "road_disruption_removal":
            return road_disruption_removal(env, solution, self.rng)
        if op == "critical_task_reassignment_removal":
            return critical_task_reassignment_removal(env, solution, self.rng)
        if op == "support_conflict_removal":
            return support_conflict_removal(env, solution, self.rng)
        if op == "synchronization_risk_removal":
            return synchronization_risk_removal(env, solution, self.rng)
        raise ValueError(f"unknown K2 destroy operator: {op}")

    def _k2_repair_solution(self, env, partial_solution, removed_items, op: str):
        if op == "greedy_insertion":
            return greedy_insertion(env, partial_solution, removed_items, self.rng)
        if op == "regret_2_insertion":
            return regret_2_insertion(env, partial_solution, removed_items, self.rng)
        if op == "regret_3_insertion":
            return regret_3_insertion(env, partial_solution, removed_items, self.rng)
        if op == "critical_first_insertion":
            return critical_first_insertion(env, partial_solution, removed_items, self.rng)
        if op == "risk_aware_insertion":
            return risk_aware_insertion(env, partial_solution, removed_items, self.rng)
        if op == "synchronized_insertion":
            return synchronized_insertion(env, partial_solution, removed_items, self.rng)
        if op == "feasibility_restoration_insertion":
            return feasibility_restoration_insertion(env, partial_solution, removed_items, self.rng)
        if op == "critical_recovery_repair_insertion":
            result = critical_recovery_repair_insertion(env, partial_solution, removed_items, self.rng)
            result = self._apply_critical_support_rebind(env, result)
            result = self._apply_lc_critical_recovery_path(env, result)
            result = self._apply_assigned_critical_reconstruct(env, result)
            return self._apply_support_reposition_shadow(env, result)
        raise ValueError(f"unknown K2 repair operator: {op}")

    def _truncate_k2_tails_for_safe_candidate(self, solution):
        repaired = solution
        for aid, agent_type in self._agent_sequence_pairs_from_solution(solution):
            seq = tuple(str(x) for x in repaired.sequence_for(aid, agent_type))
            if len(seq) > 1:
                repaired = repaired.with_sequence(aid, agent_type, seq[:1])
        return repaired

    def _objective_safe_k2_solution(self, env, solution, *, hard_feasible: Optional[bool] = None):
        del hard_feasible
        objective_evaluations = 0
        feasibility_evaluations = 0

        def finish(candidate, evaluation, changed):
            self._record_objective_safe_evaluation_counts(
                objective_evaluations=objective_evaluations,
                feasibility_evaluations=feasibility_evaluations,
            )
            return candidate, evaluation, changed

        pruned = solution
        claims: Dict[str, str] = {}
        for aid, agent_type in self._agent_sequence_pairs_from_solution(solution):
            original = tuple(str(x) for x in solution.sequence_for(aid, agent_type))[:2]
            chosen: Tuple[str, ...] = ()
            for prefix_len in range(len(original), 0, -1):
                trial = original[:prefix_len]
                feas = evaluate_sequence_feasibility(env, aid, trial, exclusive_claims=claims)
                feasibility_evaluations += 1
                if bool(feas.feasible):
                    chosen = trial
                    break
            pruned = pruned.with_sequence(aid, agent_type, chosen)
            for tid in chosen:
                claims[str(tid)] = str(aid)
        active_tasks = {str(tid) for tid in claims}
        if active_tasks:
            pruned = pruned.__class__(
                truck_sequences=dict(getattr(pruned, "truck_sequences", ())),
                uav_sequences=dict(getattr(pruned, "uav_sequences", ())),
                support_bindings=tuple(
                    b for b in getattr(pruned, "support_bindings", ()) if str(getattr(b, "task_id", "")) in active_tasks
                ),
                sortie_plans=tuple(
                    p for p in getattr(pruned, "sortie_plans", ()) if str(getattr(p, "task_id", "")) in active_tasks
                ),
            )
        else:
            pruned = pruned.__class__(truck_sequences={}, uav_sequences={})
        pruned_head_feasible = self._goals_hard_feasible(
            env, solution_to_legacy_goals(pruned)
        )
        feasibility_evaluations += 1
        ev = evaluate_k2_solution(env, pruned, hard_feasible=pruned_head_feasible)
        objective_evaluations += 1
        if bool(ev.feasible):
            return finish(pruned, ev, str(pruned.digest()) != str(solution.digest()))
        goals = solution_to_legacy_goals(solution)
        head_feasible = bool(self._goals_hard_feasible(env, goals))
        feasibility_evaluations += 1
        ev = evaluate_k2_solution(env, solution, hard_feasible=head_feasible)
        objective_evaluations += 1
        if bool(ev.feasible):
            return finish(solution, ev, False)
        truncated = self._truncate_k2_tails_for_safe_candidate(solution)
        if str(truncated.digest()) == str(solution.digest()):
            return finish(solution, ev, False)
        truncated_goals = solution_to_legacy_goals(truncated)
        truncated_head_feasible = bool(self._goals_hard_feasible(env, truncated_goals))
        feasibility_evaluations += 1
        truncated_ev = evaluate_k2_solution(env, truncated, hard_feasible=truncated_head_feasible)
        objective_evaluations += 1
        if bool(truncated_ev.feasible):
            return finish(truncated, truncated_ev, True)
        return finish(truncated, truncated_ev, True)

    def _record_objective_safe_evaluation_counts(
        self,
        *,
        objective_evaluations: int,
        feasibility_evaluations: int,
    ) -> None:
        """Hook for search baselines whose evaluations flow through this helper.

        The main ALNS path already records its candidate budgets at the caller.
        Dedicated Tabu/HGA/VNS implementations override this hook so their
        shared-oracle work is visible without double-counting the mainline.
        """

        del objective_evaluations, feasibility_evaluations

    def _agent_sequence_pairs_from_solution(self, solution) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        for aid, _seq in tuple(getattr(solution, "truck_sequences", ())):
            pairs.append((str(aid), "truck"))
        for aid, _seq in tuple(getattr(solution, "uav_sequences", ())):
            pairs.append((str(aid), "uav"))
        return pairs

    def _assigned_task_ids(self, env, goals: Dict[str, Optional[str]], *, for_agent_id: Optional[str] = None) -> set:
        out = set()
        for_st = env.state.agents.get(str(for_agent_id), None) if for_agent_id is not None else None
        for gid in goals.values():
            task = env.state.tasks.get(str(gid), None) if gid is not None else None
            if task is not None and task.status == TaskStatus.PENDING:
                if (
                    for_st is not None
                    and for_st.kind == AgentKind.TRUCK
                    and task.kind == TaskKind.EMERGENCY
                    and self._support_bound_delivery_info(env, str(for_agent_id), task).get("bound_timecritical", 0.0)
                ):
                    continue
                out.add(str(task.task_id))
        return out

    def _goal_is_protected(self, env, aid: str, task) -> bool:
        if not bool(getattr(env.cfg, "alns_safe_overlay_enabled", True)):
            return False
        st = env.state.agents.get(str(aid), None)
        if st is None or task is None or getattr(task, "status", None) != TaskStatus.PENDING:
            return False
        step = int(getattr(env.state, "step_index", 0))
        assigned_step = int(self.state.goal_assigned_step.get(str(aid), step))
        age = int(max(step - assigned_step, 0))
        recent_steps = int(max(getattr(env.cfg, "alns_protect_recent_goal_steps", 10), 0))
        stale_steps = int(max(getattr(env.cfg, "alns_stale_goal_steps", 28), 1))
        progress_eps = float(max(getattr(env.cfg, "alns_protect_progress_epsilon_m", 20.0), 0.0))
        try:
            progress = float(self._switch_goal_progress_recent(env, str(aid), str(task.task_id), 5))
        except Exception:
            progress = 0.0

        if st.kind == AgentKind.TRUCK:
            valid = bool(self._truck_task_valid(env, str(aid), str(task.task_id)))
            if not valid:
                return False
            if task.kind == TaskKind.NORMAL:
                return bool(age <= recent_steps or progress >= progress_eps or age < stale_steps)
            if task.kind == TaskKind.EMERGENCY:
                urgency = 1.0 - float(np.clip(self._task_lifeline_ratio(task), 0.0, 1.0))
                return bool(progress >= progress_eps or (urgency >= 0.65 and age <= stale_steps))
        if st.kind == AgentKind.UAV:
            if task.kind != TaskKind.EMERGENCY:
                return False
            feasible = bool(self._uav_task_feasible(env, str(aid), task))
            if not feasible:
                return False
            return bool(age <= recent_steps or progress >= 1e-6 or bool(getattr(st, "airborne", False)))
        return False

    def _goal_map_impact_score(self, env, aid: str, gid: Optional[str]) -> float:
        if gid is None:
            return 0.0
        try:
            impacted, critical = self._goal_map_update_impact(env, str(aid), str(gid))
        except Exception:
            impacted, critical = False, False
        if critical:
            return 2.0
        if impacted:
            return 1.0
        task = env.state.tasks.get(str(gid), None)
        st = env.state.agents.get(str(aid), None)
        if task is not None and st is not None:
            if st.kind == AgentKind.TRUCK and (not bool(self._truck_task_reachable(env, str(aid), task))):
                return 1.8
            if st.kind == AgentKind.UAV and task.kind == TaskKind.EMERGENCY and (not bool(self._uav_task_feasible(env, str(aid), task))):
                return 1.6
        return 0.0

    def _support_gap_score(self, env, aid: str, task) -> float:
        if task is None or getattr(task, "kind", None) != TaskKind.EMERGENCY:
            return 0.0
        st = env.state.agents.get(str(aid), None)
        if st is None:
            return 0.0
        urgency = 1.0 - float(np.clip(self._task_lifeline_ratio(task), 0.0, 1.0))
        cover = 1.0
        try:
            cover = float(np.clip(self._uav_emergency_cover_fraction(env, task), 0.0, 1.0))
        except Exception:
            cover = 1.0
        if st.kind == AgentKind.TRUCK:
            bind_info = self._support_bound_delivery_info(env, str(aid), task)
            if float(bind_info.get("bound_timecritical", 0.0)) > 0.0:
                return float(0.45 + 0.45 * urgency + 0.40 * max(0.0, 1.0 - cover))
            if not self._any_uav_direct_feasible(env, task):
                return float(0.30 + 0.55 * urgency + 0.45 * max(0.0, 1.0 - cover))
        if st.kind == AgentKind.UAV and not bool(self._uav_task_feasible(env, str(aid), task)):
            return float(0.30 + 0.50 * urgency + 0.40 * max(0.0, 1.0 - cover))
        return 0.0

    def _destroy_candidate_score(self, env, aid: str, gid: Optional[str], task, op: str, road_event: bool) -> float:
        st = env.state.agents.get(str(aid), None)
        if st is None or task is None or getattr(task, "status", None) != TaskStatus.PENDING:
            return 0.0
        urgency = 1.0 - float(np.clip(self._task_lifeline_ratio(task), 0.0, 1.0))
        risk_pressure = float(self._risk_pressure_for_task(task))
        if op == "tc_uncovered" and task.kind == TaskKind.EMERGENCY:
            if st.kind == AgentKind.TRUCK:
                bind_info = self._support_bound_delivery_info(env, str(aid), task)
                if float(bind_info.get("bound_timecritical", 0.0)) <= 0.0 and (not bool(self._truck_task_direct_serviceable(env, str(aid), task))):
                    return float(0.35 + 0.55 * urgency + 0.35 * risk_pressure)
            if st.kind == AgentKind.UAV and (not bool(self._uav_task_feasible(env, str(aid), task))):
                return float(0.30 + 0.50 * urgency + 0.30 * risk_pressure)
            return 0.0
        if op == "support_gap":
            gap = float(self._support_gap_score(env, str(aid), task))
            return float(max(gap, 0.0))
        if op == "random_light":
            if task.kind == TaskKind.NORMAL and st.kind == AgentKind.TRUCK:
                return 0.15 + 0.05 * float(self.rng.random())
            if task.kind == TaskKind.EMERGENCY and st.kind == AgentKind.UAV and (not bool(getattr(st, "airborne", False))):
                return 0.15 + 0.05 * float(self.rng.random())
            return 0.0
        return 0.0

    def _assignment_destroy_score(self, env, aid: str, gid: Optional[str], task, op: str, road_event: bool) -> float:
        if gid is not None and (task is None or getattr(task, "status", None) != TaskStatus.PENDING):
            return 3.0
        if task is None:
            return 0.0
        st = env.state.agents.get(str(aid), None)
        if st is None:
            return 0.0

        map_score = float(self._goal_map_impact_score(env, str(aid), str(gid)))
        if op == "road_disruption":
            if not road_event and map_score <= 0.0:
                return 0.0
            return float(map_score + 0.08 * self._risk_pressure_for_task(task))

        if op == "stale_or_low_value":
            step = int(getattr(env.state, "step_index", 0))
            age = int(max(step - int(self.state.goal_assigned_step.get(str(aid), step)), 0))
            stale_steps = int(max(getattr(env.cfg, "alns_stale_goal_steps", 28), 1))
            try:
                progress = float(self._switch_goal_progress_recent(env, str(aid), str(task.task_id), 5))
            except Exception:
                progress = 0.0
            if age < int(2 * stale_steps) or progress > 1e-6:
                return 0.0
            if st.kind == AgentKind.TRUCK and task.kind == TaskKind.NORMAL and bool(self._truck_task_valid(env, str(aid), str(task.task_id))):
                return 0.0
            if st.kind == AgentKind.UAV and bool(getattr(st, "airborne", False)):
                return 0.0
            return float(0.35 + float(age / max(stale_steps * 4, 1)) - 0.05 * progress)

        extra = float(self._destroy_candidate_score(env, aid, gid, task, op, road_event))
        if extra > 0.0:
            return extra
        return 0.0

    def _eligible_destroy_agents(self, env, goals: Dict[str, Optional[str]], op: str, road_event: bool) -> List[Tuple[float, str]]:
        scored: List[Tuple[float, str]] = []
        for aid in self._ordered_agents(env):
            gid = goals.get(str(aid), None)
            task = env.state.tasks.get(str(gid), None) if gid is not None else None
            st = env.state.agents.get(str(aid), None)
            if st is None or gid is None or task is None:
                continue
            if task.status != TaskStatus.PENDING:
                continue
            if self._goal_is_protected(env, str(aid), task):
                continue
            score = float(self._assignment_destroy_score(env, str(aid), gid, task, op, road_event))
            if np.isfinite(score) and score > 0.10:
                scored.append((float(score), str(aid)))
        return scored

    def _destroy_fallback(self, env, goals: Dict[str, Optional[str]], max_remove: int) -> Tuple[Dict[str, Optional[str]], List[str]]:
        cand = dict(goals)
        road_event = bool(self._road_event_active_for_plan)
        eligible = self._eligible_destroy_agents(env, goals, "random_light", road_event)
        if not eligible:
            return cand, []
        self.alns_diagnostics.destroy_fallback_count += 1
        eligible.sort(key=lambda x: (-float(x[0]), str(x[1])))
        names = [aid for _, aid in eligible]
        self.rng.shuffle(names)
        removed: List[str] = []
        for aid in names[:max_remove]:
            if cand.get(str(aid), None) is not None:
                cand[str(aid)] = None
                removed.append(str(aid))
        return cand, removed

    def _destroy_goals(self, env, goals: Dict[str, Optional[str]], op: str) -> Tuple[Dict[str, Optional[str]], List[str]]:
        cand = dict(goals)
        removed: List[str] = []
        max_remove = int(max(getattr(env.cfg, "alns_destroy_max_assignments", 3), 1))
        road_event = bool(self._road_event_active_for_plan)
        scored = self._eligible_destroy_agents(env, cand, op, road_event)
        scored.sort(key=lambda x: (-float(x[0]), str(x[1])))
        for _, aid in scored[:max_remove]:
            if cand.get(aid, None) is not None:
                cand[aid] = None
                removed.append(str(aid))
        if not removed:
            cand, removed = self._destroy_fallback(env, goals, max_remove)
        self.alns_destroyed_assignment_count_total += int(len(removed))
        return cand, removed

    def _candidate_score(self, env, aid: str, task, repair_op: str, used_tasks: set) -> float:
        if str(task.task_id) in used_tasks or not self._task_planner_active(task):
            return -1e9
        st = env.state.agents.get(str(aid), None)
        if st is None:
            return -1e9
        if st.kind == AgentKind.TRUCK:
            if not self._truck_task_valid(env, str(aid), str(task.task_id)):
                return -1e9
            base = float(self._score_truck_task(env, str(aid), task))
            if task.kind == TaskKind.EMERGENCY:
                direct_reachable = bool(self._truck_task_reachable(env, str(aid), task))
                direct_serviceable = bool(self._truck_task_direct_serviceable(env, str(aid), task))
                bind_info = self._support_bound_delivery_info(env, str(aid), task)
                if direct_reachable and direct_serviceable:
                    base += 0.85
                elif float(bind_info.get("bound_timecritical", 0.0)) > 0.0:
                    base += 0.95
                elif not self._any_uav_direct_feasible(env, task):
                    base += 0.80
                else:
                    base -= 0.45
        elif st.kind == AgentKind.UAV:
            if task.kind != TaskKind.EMERGENCY:
                return -1e9
            if not self._uav_task_feasible(env, str(aid), task):
                return -1e9
            base = float(self._score_uav_task(env, str(aid), task))
            margin = float(self._uav_task_margin(env, str(aid), task))
            base += 0.35 * float(np.clip(margin, -1.0, 1.0))
            if margin < 0.10:
                base -= 0.45
        else:
            return -1e9
        risk_pressure = float(self._risk_pressure_for_task(task))
        urgency = 1.0 - float(np.clip(self._task_lifeline_ratio(task), 0.0, 1.0))
        tc_bonus = 0.35 if self._is_timecritical_lightweight_task(task) else 0.0
        pending_normal = int(
            sum(
                1
                for t in env.state.tasks.values()
                if getattr(t, "status", None) == TaskStatus.PENDING and getattr(t, "kind", None) == TaskKind.NORMAL
            )
        )
        if st.kind == AgentKind.TRUCK and task.kind == TaskKind.EMERGENCY and pending_normal >= 3 and urgency < 0.72:
            base -= 0.55
        if repair_op == "direct_service_insert":
            if st.kind == AgentKind.TRUCK and bool(self._truck_task_reachable(env, str(aid), task)):
                base += 0.80
                if task.kind == TaskKind.NORMAL:
                    base += 0.25
            elif st.kind == AgentKind.UAV:
                base += 0.35 * tc_bonus + 0.20 * urgency
        elif repair_op == "tc_first_insert":
            base += 0.55 * tc_bonus + 0.35 * urgency
            if st.kind == AgentKind.TRUCK and task.kind == TaskKind.EMERGENCY:
                if bool(self._truck_task_reachable(env, str(aid), task)):
                    base += 0.45
                else:
                    base += 0.20
        elif repair_op == "risk_balanced_insert":
            base += 0.40 * risk_pressure + 0.10 * urgency
        else:
            base += 0.20 * risk_pressure + 0.20 * tc_bonus
        return float(base)

    def _repair_destroyed(self, env, goals: Dict[str, Optional[str]], removed_agents: List[str], repair_op: str) -> Dict[str, Optional[str]]:
        cand = dict(goals)
        agents = list(removed_agents)
        if not agents:
            agents = [a for a in self._ordered_agents(env) if cand.get(str(a), None) is None]
        agents = [str(a) for a in agents if str(a) in env.state.agents]
        used_global = self._assigned_task_ids(env, cand)
        pair_scores: List[Tuple[float, str, str]] = []
        for aid in agents:
            st = env.state.agents.get(str(aid), None)
            if st is None or bool(getattr(st, "crashed", False)):
                continue
            for task in env.state.tasks.values():
                sc = float(self._candidate_score(env, str(aid), task, repair_op, used_global))
                if np.isfinite(sc) and sc > -1e8:
                    tier = float(self._task_priority_tier(env, str(aid), task))
                    pair_scores.append((float(sc + 4.0 * tier), str(aid), str(task.task_id)))

        assigned_agents: Set[str] = set()
        assigned_tasks: Set[str] = set()
        pair_scores.sort(key=lambda x: (-float(x[0]), str(x[1]), str(x[2])))
        for _, aid, tid in pair_scores:
            if aid in assigned_agents or tid in assigned_tasks or tid in used_global:
                continue
            cand[str(aid)] = str(tid)
            assigned_agents.add(str(aid))
            assigned_tasks.add(str(tid))
            used_global.add(str(tid))

        for aid in agents:
            if aid in assigned_agents:
                continue
            st = env.state.agents.get(str(aid), None)
            if st is not None and st.kind == AgentKind.UAV and self._uav_needs_recovery(env, str(aid)):
                near_tid, _ = self._nearest_truck(env, str(aid))
                cand[str(aid)] = near_tid
        return self._repair_goals(env, cand)

    def _goals_hard_feasible(self, env, goals: Dict[str, Optional[str]]) -> bool:
        claimed: Dict[str, List[str]] = {}
        for aid in self._ordered_agents(env):
            gid = goals.get(str(aid), None)
            if gid is None:
                continue
            st = env.state.agents.get(str(aid), None)
            if st is None:
                return False
            if str(gid) in env.state.agents:
                if st.kind != AgentKind.UAV:
                    return False
                continue
            task = env.state.tasks.get(str(gid), None)
            if task is None or task.status != TaskStatus.PENDING:
                return False
            if st.kind == AgentKind.TRUCK:
                if not bool(self._truck_task_valid(env, str(aid), str(task.task_id))):
                    return False
            elif st.kind == AgentKind.UAV:
                if task.kind != TaskKind.EMERGENCY or not bool(self._uav_task_feasible(env, str(aid), task)):
                    return False
            claimed.setdefault(str(task.task_id), []).append(str(aid))
        for tid, aids in claimed.items():
            if len(aids) <= 1:
                continue
            task = env.state.tasks.get(str(tid), None)
            if task is None or task.kind != TaskKind.EMERGENCY:
                return False
            kinds = {env.state.agents[str(aid)].kind for aid in aids if str(aid) in env.state.agents}
            if kinds != {AgentKind.TRUCK, AgentKind.UAV}:
                return False
        return True

    def _should_accept_candidate(self, delta: float, temperature: float) -> bool:
        if float(delta) >= 0.0:
            return True
        temp = float(max(temperature, 1e-9))
        return bool(float(self.rng.random()) < math.exp(float(delta) / temp))

    def _legacy_solution_context(self, env):
        seq_len = int(getattr(env.cfg, "alns_sequence_length", 1))
        mode = str(getattr(env.cfg, "alns_solution_mode", "legacy_k1")).strip().lower()
        if mode == "legacy_k1" and seq_len != 1:
            raise ValueError("alns_solution_mode='legacy_k1' requires alns_sequence_length == 1")
        if mode in {"k2_shadow", "k2_active"} and seq_len != 2:
            raise ValueError("k2 modes require alns_sequence_length == 2")
        return env_adapter_context(env, sequence_length=1)

    def _solution_mode(self, env) -> str:
        configured = self._solution_mode_configured(env)
        if configured in {"k2_shadow", "k2_active"}:
            k = self._decide_effective_k(env)
            if str(getattr(env.cfg, "adaptive_horizon_mode", "disabled")).strip().lower() == "active" and int(k) == 1:
                return "legacy_k1"
        return configured

    def _record_k2_solution(self, env, label: str, goals: Dict[str, Optional[str]]):
        solution = construct_k2_solution(env, goals)
        return self._record_k2_solution_object(
            env,
            label,
            solution,
            hard_feasible=self._goals_hard_feasible(env, goals),
        )

    def _record_k2_solution_object(self, env, label: str, solution, *, hard_feasible: bool):
        ev = evaluate_k2_solution(env, solution, hard_feasible=bool(hard_feasible))
        self.alns_diagnostics.k2_solution_count += 1
        seqs = list(solution.truck_sequences) + list(solution.uav_sequences)
        self.alns_diagnostics.k2_sequence_length_sum += int(sum(len(seq) for _aid, seq in seqs))
        self.alns_diagnostics.k2_sequence_agent_count += int(len(seqs))
        nonempty_tail = int(sum(1 for _aid, seq in seqs if len(seq) >= 2))
        self.alns_diagnostics.k2_nonempty_tail_count += nonempty_tail
        if ev.feasible:
            self.alns_diagnostics.k2_feasible_sequence_count += 1
        else:
            self.alns_diagnostics.k2_infeasible_sequence_count += 1
        second_travel = 0.0
        second_energy = 0.0
        second_life = 0.0
        for aid, seq in seqs:
            if len(seq) >= 2:
                cost = evaluate_sequence_cost(env, aid, seq)
                second_travel += float(cost.travel_cost_second)
                second_energy += float(cost.energy_cost_second)
                second_life += float(cost.lifeline_loss_second)
        self.alns_diagnostics.k2_second_task_travel_cost += float(second_travel)
        self.alns_diagnostics.k2_second_task_energy_cost += float(second_energy)
        self.alns_diagnostics.k2_second_task_lifeline_cost += float(second_life)
        rec = {
            "step": int(getattr(getattr(env, "state", None), "step_index", 0)),
            "label": str(label),
            "solution_digest": solution.digest(),
            "total_cost": float(ev.breakdown.total_cost),
            "feasible": bool(ev.feasible),
            "nonempty_tail_count": int(nonempty_tail),
            "average_sequence_length": float(sum(len(seq) for _aid, seq in seqs) / max(len(seqs), 1)),
            "second_task_travel_cost": float(second_travel),
            "second_task_energy_cost": float(second_energy),
            "second_task_lifeline_cost": float(second_life),
            "reason_codes": "|".join(ev.infeasibility_reasons),
        }
        self.k2_sequence_records.append(rec)
        return solution, ev, rec

    def _apply_local_search_refiner(self, env, solution, evaluation):
        mode = str(getattr(env.cfg, "local_search_mode", "disabled")).strip().lower() or "disabled"
        if mode != "active":
            return solution, evaluation
        refiner = LocalSearchRefinerV2(
            LocalSearchBudget(
                max_moves=int(getattr(env.cfg, "local_search_max_moves_per_iteration", 5)),
                max_exact_checks=int(getattr(env.cfg, "local_search_max_exact_checks_per_iteration", 5)),
                max_time_ms=int(getattr(env.cfg, "local_search_max_time_ms_per_iteration", 20)),
            ),
            disabled_moves=tuple(getattr(env.cfg, "local_search_disabled_moves", ())),
        )

        def _evaluate(candidate_solution):
            goals = solution_to_legacy_goals(candidate_solution)
            return evaluate_k2_solution(
                env,
                candidate_solution,
                hard_feasible=self._goals_hard_feasible(env, goals),
            )

        def _exact(candidate_solution) -> bool:
            goals = solution_to_legacy_goals(candidate_solution)
            return bool(self._goals_hard_feasible(env, goals))

        result = refiner.refine(solution, evaluation, evaluate=_evaluate, exact_feasible=_exact)
        self.alns_diagnostics.local_search_attempt_count += int(result.attempted_moves)
        self.alns_diagnostics.local_search_feasible_count += sum(1 for row in result.ledger_rows if bool(row.get("feasible", False)))
        self.alns_diagnostics.local_search_accepted_move_count += int(result.accepted_moves)
        self.alns_diagnostics.local_search_exact_check_count += int(result.exact_checks)
        self.alns_diagnostics.local_search_runtime_ms += float(result.runtime_ms)
        for row in result.ledger_rows:
            self.local_search_records.append(
                {
                    "episode": int(getattr(env, "current_episode_index", 0)),
                    "step": int(getattr(getattr(env, "state", None), "step_index", 0)),
                    "scenario": str(getattr(env.cfg, "scenario", "")),
                    "method": str(getattr(env, "current_method", "")),
                    "seed": int(getattr(env.cfg, "seed", 0)),
                    "protocol": str(getattr(env.cfg, "physical_environment_safety_protocol", "")),
                    **dict(row),
                }
            )
        return result.solution, result.evaluation

    def export_k2_sequence_records(self) -> List[Dict[str, Any]]:
        return list(self.k2_sequence_records)

    def _record_objective_shadow_comparison(
        self,
        env,
        current_goals: Dict[str, Optional[str]],
        candidate_goals: Dict[str, Optional[str]],
        legacy_current_score: float,
        legacy_candidate_score: float,
    ) -> None:
        try:
            context = self._legacy_solution_context(env)
            cur_solution = legacy_goals_to_solution(current_goals, context)
            cand_solution = legacy_goals_to_solution(candidate_goals, context)
            roundtrip_current = solution_to_legacy_goals(cur_solution)
            roundtrip_candidate = solution_to_legacy_goals(cand_solution)
            roundtrip_mismatch = bool(roundtrip_current != current_goals or roundtrip_candidate != candidate_goals)
            if roundtrip_mismatch:
                self.alns_diagnostics.legacy_k1_roundtrip_mismatch_count += 1
            current_eval = evaluate_solution(
                env,
                cur_solution,
                previous_goals=current_goals,
                hard_feasibility_checker=self._goals_hard_feasible,
            )
            candidate_eval = evaluate_solution(
                env,
                cand_solution,
                previous_goals=current_goals,
                hard_feasibility_checker=self._goals_hard_feasible,
            )
            legacy_prefers = float(legacy_candidate_score) > float(legacy_current_score)
            k2_current_cost = float("nan")
            k2_candidate_cost = float("nan")
            if self._solution_mode(env) in {"k2_shadow", "k2_active"}:
                _cur_k2, cur_k2_eval, _ = self._record_k2_solution(env, "current", current_goals)
                _cand_k2, cand_k2_eval, _ = self._record_k2_solution(env, "candidate", candidate_goals)
                k2_current_cost = float(cur_k2_eval.breakdown.total_cost)
                k2_candidate_cost = float(cand_k2_eval.breakdown.total_cost)
                k2_prefers = bool(cand_k2_eval.feasible and (not cur_k2_eval.feasible or k2_candidate_cost < k2_current_cost))
                if k2_prefers != legacy_prefers:
                    self.alns_diagnostics.k2_candidate_ranking_difference_vs_k1 += 1
            new_prefers = bool(
                candidate_eval.feasible
                and (
                    (not current_eval.feasible)
                    or float(candidate_eval.breakdown.total_cost) < float(current_eval.breakdown.total_cost)
                )
            )
            agreement = bool(legacy_prefers == new_prefers)
            reason = "AGREE" if agreement else "SEMANTIC_OBJECTIVE_DIFFERENCE"
            if candidate_eval.feasible != current_eval.feasible:
                reason = "FEASIBILITY_PRIORITY_DIFFERENCE"
            elif abs(float(legacy_candidate_score) - float(legacy_current_score)) <= 1e-12:
                reason = "LEGACY_TIE"
            record = {
                "step": int(getattr(getattr(env, "state", None), "step_index", 0)),
                "legacy_current_score": float(legacy_current_score),
                "legacy_candidate_score": float(legacy_candidate_score),
                "legacy_prefers_candidate": bool(legacy_prefers),
                "new_current_cost": float(current_eval.breakdown.total_cost),
                "new_candidate_cost": float(candidate_eval.breakdown.total_cost),
                "new_prefers_candidate": bool(new_prefers),
                "ranking_agreement": bool(agreement),
                "disagreement_reason": str(reason),
                "current_solution_digest": cur_solution.digest(),
                "candidate_solution_digest": cand_solution.digest(),
                "current_feasible": bool(current_eval.feasible),
                "candidate_feasible": bool(candidate_eval.feasible),
                "legacy_k1_roundtrip_mismatch": bool(roundtrip_mismatch),
                "k2_current_cost": float(k2_current_cost),
                "k2_candidate_cost": float(k2_candidate_cost),
            }
        except Exception as exc:
            record = {
                "step": int(getattr(getattr(env, "state", None), "step_index", 0)),
                "legacy_current_score": float(legacy_current_score),
                "legacy_candidate_score": float(legacy_candidate_score),
                "legacy_prefers_candidate": bool(float(legacy_candidate_score) > float(legacy_current_score)),
                "new_current_cost": float("nan"),
                "new_candidate_cost": float("nan"),
                "new_prefers_candidate": False,
                "ranking_agreement": False,
                "disagreement_reason": f"SHADOW_EVALUATION_ERROR:{type(exc).__name__}",
                "current_solution_digest": "",
                "candidate_solution_digest": "",
                "current_feasible": False,
                "candidate_feasible": False,
                "legacy_k1_roundtrip_mismatch": True,
            }
        self.objective_shadow_records.append(record)
        self.alns_diagnostics.objective_shadow_comparison_count += 1
        if bool(record.get("ranking_agreement", False)):
            self.alns_diagnostics.objective_shadow_agreement_count += 1
        else:
            self.alns_diagnostics.objective_shadow_disagreement_count += 1

    def export_objective_shadow_records(self) -> List[Dict[str, Any]]:
        return list(self.objective_shadow_records)

    def _restore_protected_goals(
        self,
        env,
        reference: Dict[str, Optional[str]],
        candidate: Dict[str, Optional[str]],
    ) -> Dict[str, Optional[str]]:
        guarded = dict(candidate)
        for aid in self._ordered_agents(env):
            ref_gid = reference.get(str(aid), None)
            task = env.state.tasks.get(str(ref_gid), None) if ref_gid is not None else None
            if task is not None and self._goal_is_protected(env, str(aid), task):
                guarded[str(aid)] = ref_gid
        return self._repair_goals(env, guarded)

    def _solution_objective(self, env, goals: Dict[str, Optional[str]]) -> float:
        score = 0.0
        claimed = {}
        for aid in self._ordered_agents(env):
            gid = goals.get(str(aid), None)
            st = env.state.agents.get(str(aid), None)
            if st is None or gid is None:
                continue
            task = env.state.tasks.get(str(gid), None)
            if task is None or task.status != TaskStatus.PENDING:
                if st.kind == AgentKind.UAV and str(gid) in env.state.agents:
                    score += 0.15
                continue
            existing = claimed.get(str(task.task_id), [])
            support_pair = False
            if existing and task.kind == TaskKind.EMERGENCY:
                kinds = {
                    env.state.agents.get(str(x), None).kind
                    for x in existing
                    if env.state.agents.get(str(x), None) is not None
                }
                support_pair = bool(
                    (st.kind == AgentKind.TRUCK and AgentKind.UAV in kinds)
                    or (st.kind == AgentKind.UAV and AgentKind.TRUCK in kinds)
                )
            if existing and not support_pair:
                score -= 2.0
                continue
            claimed.setdefault(str(task.task_id), []).append(str(aid))
            urgency = 1.0 - float(np.clip(self._task_lifeline_ratio(task), 0.0, 1.0))
            risk_pressure = float(self._risk_pressure_for_task(task))
            if st.kind == AgentKind.TRUCK:
                if self._truck_task_valid(env, str(aid), str(task.task_id)):
                    score += 1.0 + 0.35 * urgency + 0.20 * risk_pressure
                    if task.kind == TaskKind.EMERGENCY:
                        direct_reachable = bool(self._truck_task_reachable(env, str(aid), task))
                        direct_serviceable = bool(self._truck_task_direct_serviceable(env, str(aid), task))
                        bind_info = self._support_bound_delivery_info(env, str(aid), task)
                        if direct_reachable and direct_serviceable:
                            score += 0.95 + 0.30 * urgency
                        elif float(bind_info.get("bound_timecritical", 0.0)) > 0.0:
                            score += 0.85 + 0.30 * urgency
                            pending_normal = int(
                                sum(
                                    1
                                    for t in env.state.tasks.values()
                                    if getattr(t, "status", None) == TaskStatus.PENDING
                                    and getattr(t, "kind", None) == TaskKind.NORMAL
                                )
                            )
                            if pending_normal >= 3 and urgency < 0.72:
                                score -= 0.75
                        elif not self._any_uav_direct_feasible(env, task):
                            score += 0.70 + 0.35 * urgency + 0.15 * risk_pressure
                        else:
                            score -= 0.60
            elif st.kind == AgentKind.UAV and task.kind == TaskKind.EMERGENCY:
                if self._uav_task_feasible(env, str(aid), task):
                    margin = float(self._uav_task_margin(env, str(aid), task))
                    score += 1.20 + 0.55 * urgency + 0.25 * risk_pressure + 0.25 * float(np.clip(margin, -1.0, 1.0))
                    if self._is_timecritical_lightweight_task(task):
                        score += 0.55
                    if margin < 0.10:
                        score -= 0.50
                else:
                    score -= 1.0
        pending_tc = [
            t for t in env.state.tasks.values()
            if self._is_timecritical_lightweight_task(t) and t.status == TaskStatus.PENDING
        ]
        assigned_tc = sum(1 for t in pending_tc if str(t.task_id) in claimed)
        score += 0.25 * float(assigned_tc)
        return float(score)

    def _update_operator_stats(
        self,
        env,
        ops: Dict[str, ALNSOperatorStats] | str,
        name: str | float,
        reward: float | bool,
        accepted: bool | None = None,
        *,
        feasible: bool = False,
        improved: bool = False,
        global_best: bool = False,
    ) -> None:
        if accepted is None:
            legacy_ops = env
            legacy_name = ops
            legacy_reward = name
            legacy_accepted = reward
            stat = legacy_ops[str(legacy_name)]
            stat.attempts += 1
            if bool(legacy_accepted):
                stat.accepted += 1
            alpha = 0.25
            stat.reward_total += float(legacy_reward)
            stat.reward_ema = float((1.0 - alpha) * stat.reward_ema + alpha * float(legacy_reward))
            target = float(stat.weight + stat.reward_ema)
            stat.weight = float(np.clip(0.90 * stat.weight + 0.10 * target, 0.10, 5.0))
            return
        stat = ops[str(name)]
        stat.attempts += 1
        stat.segment_usage += 1
        stat.segment_score += float(reward)
        if feasible:
            stat.feasible += 1
        if accepted:
            stat.accepted += 1
        if improved:
            stat.improved += 1
        if global_best:
            stat.global_best += 1
        if not feasible:
            stat.failure += 1
        stat.reward_total += float(reward)
        alpha = 0.25
        stat.reward_ema = float((1.0 - alpha) * stat.reward_ema + alpha * float(reward))
        if str(getattr(self, "_selection_mode", "adaptive")) == "uniform":
            return
        seg_len = int(max(getattr(env.cfg, "alns_weight_segment_length", 12), 1))
        if stat.segment_usage < seg_len:
            return
        rho = float(np.clip(getattr(env.cfg, "alns_weight_learning_rate", 0.25), 0.0, 1.0))
        min_w = float(max(getattr(env.cfg, "alns_weight_min", 0.10), 1e-9))
        avg_score = float(stat.segment_score / max(stat.segment_usage, 1))
        old_weight = float(stat.weight)
        target = float(max(avg_score, min_w))
        stat.weight = float(max((1.0 - rho) * stat.weight + rho * target, min_w))
        self.operator_weight_trajectory_records.append(
            {
                "step": int(getattr(getattr(env, "state", None), "step_index", 0)),
                "operator": str(name),
                "old_weight": old_weight,
                "new_weight": float(stat.weight),
                "usage": int(stat.attempts),
                "feasible": int(stat.feasible),
                "accepted": int(stat.accepted),
                "improved": int(stat.improved),
                "global_best": int(stat.global_best),
                "failure": int(stat.failure),
                "segment_usage": int(stat.segment_usage),
                "segment_score": float(stat.segment_score),
                "selection_mode": str(getattr(self, "_selection_mode", "adaptive")),
            }
        )
        stat.segment_usage = 0
        stat.segment_score = 0.0

    def _initial_temperature(self, env, current_goals: Dict[str, Optional[str]], current_cost: float | None) -> float:
        fallback = float(max(getattr(env.cfg, "alns_accept_temperature", 0.05), 1e-12))
        if self._solution_mode(env) != "k2_active" or not bool(getattr(env.cfg, "alns_sa_auto_calibration_enabled", False)):
            self.sa_calibration_records.append(
                {
                    "step": int(getattr(getattr(env, "state", None), "step_index", 0)),
                    "sample_count": 0,
                    "positive_delta_count": 0,
                    "delta_quantile": float(getattr(env.cfg, "alns_sa_delta_quantile", 0.75)),
                    "target_worse_accept_probability": float(getattr(env.cfg, "alns_sa_initial_worse_accept_probability", 0.20)),
                    "temperature": fallback,
                    "reason": "fallback_fixed_temperature",
                }
            )
            return fallback
        base_cost = float(current_cost if current_cost is not None else 0.0)
        positive: List[float] = []
        sample_count = int(max(getattr(env.cfg, "alns_sa_sample_count", 24), 1))
        op_names = sorted(self.destroy_ops)
        repair_names = sorted(self.repair_ops)
        for idx in range(sample_count):
            try:
                budget = int(getattr(self, "_active_evaluation_budget", 0))
                if budget > 0 and self.alns_diagnostics.objective_evaluation_count >= budget:
                    break
                destroy_op = op_names[int(idx % max(len(op_names), 1))]
                repair_op = repair_names[int(idx % max(len(repair_names), 1))]
                if self._use_k2_operator_pool(env):
                    destroy_result = self._k2_destroy_solution(env, current_goals, destroy_op)
                    if not destroy_result.removed_items:
                        continue
                    repair_result = self._k2_repair_solution(env, destroy_result.partial_solution, destroy_result.removed_items, repair_op)
                    if not bool(repair_result.feasible):
                        continue
                    candidate = repair_result.candidate_solution
                    cand_eval = evaluate_k2_solution(env, candidate, hard_feasible=self._goals_hard_feasible(env, solution_to_legacy_goals(candidate)))
                else:
                    continue
                if budget > 0 and self.alns_diagnostics.objective_evaluation_count >= budget:
                    break
                self.alns_diagnostics.objective_evaluation_count += 1
                if not bool(cand_eval.feasible):
                    continue
                delta = float(cand_eval.breakdown.total_cost - base_cost)
                if delta > 1e-12 and math.isfinite(delta):
                    positive.append(delta)
            except Exception:
                continue
        if positive:
            q = float(np.quantile(positive, float(getattr(env.cfg, "alns_sa_delta_quantile", 0.75))))
            p0 = float(np.clip(getattr(env.cfg, "alns_sa_initial_worse_accept_probability", 0.20), 1e-6, 1.0 - 1e-6))
            temp = float(max(-q / math.log(p0), float(getattr(env.cfg, "alns_sa_minimum_temperature", 1e-4))))
            reason = "positive_delta_quantile"
        else:
            temp = fallback
            q = 0.0
            reason = "no_positive_delta_fallback"
        self.sa_calibration_records.append(
            {
                "step": int(getattr(getattr(env, "state", None), "step_index", 0)),
                "sample_count": int(sample_count),
                "positive_delta_count": int(len(positive)),
                "delta_quantile": float(q),
                "target_worse_accept_probability": float(getattr(env.cfg, "alns_sa_initial_worse_accept_probability", 0.20)),
                "temperature": float(temp),
                "reason": reason,
            }
        )
        return float(temp)

    def _alns_optimize_goals(self, env, base_goals: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
        start_time = time.perf_counter()
        self.alns_diagnostics.replan_count += 1
        self._selection_mode = str(getattr(env.cfg, "alns_selection_mode", "adaptive")).strip().lower() or "adaptive"
        self._active_evaluation_budget = int(
            max(
                getattr(env.cfg, "hrl_route_plan_alns_objective_evaluation_budget", 0),
                0,
            )
        )
        evaluation_budget = int(self._active_evaluation_budget)
        self._ensure_operator_pool(env)
        current = self._repair_goals(env, dict(base_goals))
        best = dict(current)
        mode = self._solution_mode(env)
        use_k2_operator_pool = bool(self._use_k2_operator_pool(env))
        if mode == "k2_active":
            raw_cur_k2 = construct_k2_solution(env, current)
            safe_cur_k2, _safe_cur_eval, current_tail_truncated = self._objective_safe_k2_solution(env, raw_cur_k2)
            if current_tail_truncated:
                current = solution_to_legacy_goals(safe_cur_k2)
            _cur_k2, cur_eval, _ = self._record_k2_solution_object(
                env,
                "initial",
                safe_cur_k2,
                hard_feasible=self._goals_hard_feasible(env, solution_to_legacy_goals(safe_cur_k2)),
            )
            current_solution_for_eval = _cur_k2
            current_eval = cur_eval
            cur_obj = float(cur_eval.breakdown.total_cost)
            initial_obj = float(cur_obj)
            best_obj = float(cur_obj)
        else:
            current_solution_for_eval = None
            cur_obj = float(self._solution_objective(env, current))
            initial_obj = float(cur_obj)
            best_obj = float(cur_obj)
        temp = float(self._initial_temperature(env, current, cur_obj if mode == "k2_active" else None))
        self._last_temperature = float(temp)
        for _ in range(int(self.alns_iterations)):
            if evaluation_budget > 0 and self.alns_diagnostics.objective_evaluation_count >= evaluation_budget:
                break
            self.alns_iteration_count_total += 1
            self.alns_diagnostics.iteration_count += 1
            destroy_op = self._sample_operator(self.destroy_ops)
            repair_op = self._sample_operator(self.repair_ops)
            self.alns_diagnostics.operator_attempt_count += 2
            if use_k2_operator_pool:
                destroy_result = self._k2_destroy_solution(env, current, destroy_op)
                destroyed = solution_to_legacy_goals(destroy_result.partial_solution)
                removed = list(destroy_result.removed_items)
                self.canonical_operator_records.append(
                    {
                        "step": int(getattr(env.state, "step_index", 0)),
                        "phase": "destroy",
                        "destroy_operator": str(destroy_op),
                        "repair_operator": str(repair_op),
                        "attempts": 1,
                        "feasible_candidates": int(len(removed) > 0),
                        "infeasible_candidates": int(len(removed) <= 0),
                        "removed_count": int(len(removed)),
                        "inserted_count": 0,
                        "accepted": False,
                        "improved": False,
                        "failure_reason_counts": {},
                        "reason_codes": list(getattr(destroy_result, "reason_codes", ())),
                    }
                )
            else:
                destroyed, removed = self._destroy_goals(env, current, destroy_op)
            self.alns_diagnostics.destroy_operator_usage[destroy_op] = int(
                self.alns_diagnostics.destroy_operator_usage.get(destroy_op, 0) + 1
            )
            if not removed:
                self.alns_diagnostics.noop_iteration_count += 1
                self._update_operator_stats(env, self.destroy_ops, destroy_op, -0.10, False, feasible=False)
                self._update_operator_stats(env, self.repair_ops, repair_op, -0.10, False, feasible=False)
                continue
            self.alns_diagnostics.destroyed_assignment_count += int(len(removed))
            self.alns_diagnostics.repair_attempt_count += 1
            if use_k2_operator_pool:
                use_pool = str(repair_op) not in {"critical_recovery_repair_insertion"}
                pool_diag: Dict[str, Any] = {}
                if use_pool:
                    repair_pool = enumerate_repair_candidate_pool(
                        env,
                        destroy_result.partial_solution,
                        removed,
                        repair_op,
                        pool_size=int(getattr(env.cfg, "candidate_ranker_pool_size", 16)),
                    )
                    pool_repair_result, pool_diag = self._select_from_repair_candidate_pool(
                        env,
                        repair_pool,
                        current_solution=current_solution_for_eval,
                        current_objective=float(cur_obj),
                        repair_operator=repair_op,
                    )
                    if str(getattr(env.cfg, "candidate_ranker_mode", "disabled")).strip().lower() == "active":
                        if pool_repair_result is None:
                            self.alns_diagnostics.planner_candidate_hard_reject_count += 1
                            self._update_operator_stats(env, self.destroy_ops, destroy_op, -0.20, False, feasible=False)
                            self._update_operator_stats(env, self.repair_ops, repair_op, -0.20, False, feasible=False)
                            continue
                        repair_result = pool_repair_result
                    else:
                        repair_result = self._k2_repair_solution(env, destroy_result.partial_solution, removed, repair_op)
                else:
                    repair_result = self._k2_repair_solution(env, destroy_result.partial_solution, removed, repair_op)
                candidate = solution_to_legacy_goals(repair_result.candidate_solution)
                self.canonical_operator_records.append(
                    {
                        "step": int(getattr(env.state, "step_index", 0)),
                        "phase": "repair",
                        "destroy_operator": str(destroy_op),
                        "repair_operator": str(repair_op),
                        "attempts": int(getattr(repair_result, "diagnostics", {}).get("attempts", 1)),
                        "feasible_candidates": int(getattr(repair_result, "diagnostics", {}).get("feasible_candidates", 0)),
                        "infeasible_candidates": int(getattr(repair_result, "diagnostics", {}).get("infeasible_candidates", 0)),
                        "removed_count": int(len(removed)),
                        "inserted_count": int(len(getattr(repair_result, "inserted_items", ()))),
                        "accepted": False,
                        "improved": False,
                        "critical_recovery_enabled": bool(getattr(repair_result, "diagnostics", {}).get("critical_recovery_enabled", False)),
                        "critical_recovery_candidates": int(getattr(repair_result, "diagnostics", {}).get("critical_recovery_candidates", 0)),
                        "critical_recovery_attempts": int(getattr(repair_result, "diagnostics", {}).get("critical_recovery_attempts", 0)),
                        "critical_recovery_direct_insertions": int(getattr(repair_result, "diagnostics", {}).get("critical_recovery_direct_insertions", 0)),
                        "critical_recovery_safe_reorders": int(getattr(repair_result, "diagnostics", {}).get("critical_recovery_safe_reorders", 0)),
                        "critical_recovery_rejected_infeasible": int(getattr(repair_result, "diagnostics", {}).get("critical_recovery_rejected_infeasible", 0)),
                        "critical_recovery_rejected_no_slot": int(getattr(repair_result, "diagnostics", {}).get("critical_recovery_rejected_no_slot", 0)),
                        "critical_recovery_rejected_duplicate_claim": int(getattr(repair_result, "diagnostics", {}).get("critical_recovery_rejected_duplicate_claim", 0)),
                        "critical_recovery_avoided_failed_agent": int(getattr(repair_result, "diagnostics", {}).get("critical_recovery_avoided_failed_agent", 0)),
                        "critical_recovery_task_ids": list(getattr(repair_result, "diagnostics", {}).get("critical_recovery_task_ids", ())),
                        "critical_support_rebind_enabled": bool(getattr(repair_result, "diagnostics", {}).get("critical_support_rebind_enabled", False)),
                        "critical_support_rebind_candidates": int(getattr(repair_result, "diagnostics", {}).get("critical_support_rebind_candidates", 0)),
                        "critical_support_rebind_attempts": int(getattr(repair_result, "diagnostics", {}).get("critical_support_rebind_attempts", 0)),
                        "critical_support_rebind_historical_reuse": int(getattr(repair_result, "diagnostics", {}).get("critical_support_rebind_historical_reuse", 0)),
                        "critical_support_rebind_reconstructed": int(getattr(repair_result, "diagnostics", {}).get("critical_support_rebind_reconstructed", 0)),
                        "critical_support_rebind_rejected_no_truck": int(getattr(repair_result, "diagnostics", {}).get("critical_support_rebind_rejected_no_truck", 0)),
                        "critical_support_rebind_rejected_no_anchor": int(getattr(repair_result, "diagnostics", {}).get("critical_support_rebind_rejected_no_anchor", 0)),
                        "critical_support_rebind_rejected_energy": int(getattr(repair_result, "diagnostics", {}).get("critical_support_rebind_rejected_energy", 0)),
                        "critical_support_rebind_rejected_reserve": int(getattr(repair_result, "diagnostics", {}).get("critical_support_rebind_rejected_reserve", 0)),
                        "critical_support_rebind_rejected_road": int(getattr(repair_result, "diagnostics", {}).get("critical_support_rebind_rejected_road", 0)),
                        "critical_support_rebind_rejected_infeasible": int(getattr(repair_result, "diagnostics", {}).get("critical_support_rebind_rejected_infeasible", 0)),
                        "critical_support_rebind_task_ids": list(getattr(repair_result, "diagnostics", {}).get("critical_support_rebind_task_ids", ())),
                        "critical_support_rebind_support_truck_ids": list(getattr(repair_result, "diagnostics", {}).get("critical_support_rebind_support_truck_ids", ())),
                        "critical_support_rebind_recovery_anchor_ids": list(getattr(repair_result, "diagnostics", {}).get("critical_support_rebind_recovery_anchor_ids", ())),
                        "critical_support_rebind_attempt_rows": list(getattr(repair_result, "diagnostics", {}).get("critical_support_rebind_attempt_rows", [])),
                        "lc_critical_recovery_path_enabled": bool(getattr(repair_result, "diagnostics", {}).get("lc_critical_recovery_path_enabled", False)),
                        "lc_critical_recovery_path_candidates": int(getattr(repair_result, "diagnostics", {}).get("lc_critical_recovery_path_candidates", 0)),
                        "lc_critical_recovery_path_attempts": int(getattr(repair_result, "diagnostics", {}).get("lc_critical_recovery_path_attempts", 0)),
                        "lc_critical_recovery_path_successes": int(getattr(repair_result, "diagnostics", {}).get("lc_critical_recovery_path_successes", 0)),
                        "lc_critical_recovery_path_rejected_insufficient_margin": int(
                            getattr(repair_result, "diagnostics", {}).get("lc_critical_recovery_path_rejected_insufficient_margin", 0)
                        ),
                        "lc_critical_recovery_path_rejected_no_bindable_truck": int(
                            getattr(repair_result, "diagnostics", {}).get("lc_critical_recovery_path_rejected_no_bindable_truck", 0)
                        ),
                        "lc_critical_recovery_path_rejected_uav_not_docked": int(
                            getattr(repair_result, "diagnostics", {}).get("lc_critical_recovery_path_rejected_uav_not_docked", 0)
                        ),
                        "lc_critical_recovery_path_rejected_no_sequence_capacity": int(
                            getattr(repair_result, "diagnostics", {}).get("lc_critical_recovery_path_rejected_no_sequence_capacity", 0)
                        ),
                        "lc_critical_recovery_path_rejected_augmented_infeasible": int(
                            getattr(repair_result, "diagnostics", {}).get("lc_critical_recovery_path_rejected_augmented_infeasible", 0)
                        ),
                        "lc_critical_recovery_path_trucks_considered": int(
                            getattr(repair_result, "diagnostics", {}).get("lc_critical_recovery_path_trucks_considered", 0)
                        ),
                        "lc_critical_recovery_path_best_margin": float(
                            getattr(repair_result, "diagnostics", {}).get("lc_critical_recovery_path_best_margin", float("-inf"))
                        ),
                        "lc_critical_recovery_path_success_margin": float(
                            getattr(repair_result, "diagnostics", {}).get("lc_critical_recovery_path_success_margin", float("-inf"))
                        ),
                        "lc_critical_recovery_path_failed_tuple_avoided": int(
                            getattr(repair_result, "diagnostics", {}).get("lc_critical_recovery_path_failed_tuple_avoided", 0)
                        ),
                        "lc_critical_recovery_path_task_ids": list(getattr(repair_result, "diagnostics", {}).get("lc_critical_recovery_path_task_ids", ())),
                        "lc_critical_recovery_path_selected_uav_ids": list(
                            getattr(repair_result, "diagnostics", {}).get("lc_critical_recovery_path_selected_uav_ids", ())
                        ),
                        "lc_critical_recovery_path_selected_truck_ids": list(
                            getattr(repair_result, "diagnostics", {}).get("lc_critical_recovery_path_selected_truck_ids", ())
                        ),
                        "lc_critical_recovery_path_selected_recovery_anchors": list(
                            getattr(repair_result, "diagnostics", {}).get("lc_critical_recovery_path_selected_recovery_anchors", ())
                        ),
                        "lc_critical_recovery_path_attempt_rows": list(
                            getattr(repair_result, "diagnostics", {}).get("lc_critical_recovery_path_attempt_rows", [])
                        ),
                        "assigned_critical_reconstruct_enabled": bool(
                            getattr(repair_result, "diagnostics", {}).get("assigned_critical_reconstruct_enabled", False)
                        ),
                        "assigned_critical_reconstruct_candidates": int(
                            getattr(repair_result, "diagnostics", {}).get("assigned_critical_reconstruct_candidates", 0)
                        ),
                        "assigned_critical_reconstruct_path_candidates": int(
                            getattr(repair_result, "diagnostics", {}).get("assigned_critical_reconstruct_path_candidates", 0)
                        ),
                        "assigned_critical_reconstruct_trucks_considered": int(
                            getattr(repair_result, "diagnostics", {}).get("assigned_critical_reconstruct_trucks_considered", 0)
                        ),
                        "assigned_critical_reconstruct_margin_probed": int(
                            getattr(repair_result, "diagnostics", {}).get("assigned_critical_reconstruct_margin_probed", 0)
                        ),
                        "assigned_critical_reconstruct_positive_margin_count": int(
                            getattr(repair_result, "diagnostics", {}).get("assigned_critical_reconstruct_positive_margin_count", 0)
                        ),
                        "assigned_critical_reconstruct_selected_path_count": int(
                            getattr(repair_result, "diagnostics", {}).get("assigned_critical_reconstruct_selected_path_count", 0)
                        ),
                        "assigned_critical_reconstruct_success_count": int(
                            getattr(repair_result, "diagnostics", {}).get("assigned_critical_reconstruct_success_count", 0)
                        ),
                        "assigned_critical_reconstruct_rejected_no_bindable_truck": int(
                            getattr(repair_result, "diagnostics", {}).get("assigned_critical_reconstruct_rejected_no_bindable_truck", 0)
                        ),
                        "assigned_critical_reconstruct_rejected_insufficient_margin": int(
                            getattr(repair_result, "diagnostics", {}).get("assigned_critical_reconstruct_rejected_insufficient_margin", 0)
                        ),
                        "assigned_critical_reconstruct_rejected_no_sequence_capacity": int(
                            getattr(repair_result, "diagnostics", {}).get("assigned_critical_reconstruct_rejected_no_sequence_capacity", 0)
                        ),
                        "assigned_critical_reconstruct_rejected_target_unreachable": int(
                            getattr(repair_result, "diagnostics", {}).get("assigned_critical_reconstruct_rejected_target_unreachable", 0)
                        ),
                        "assigned_critical_reconstruct_rejected_uav_not_docked": int(
                            getattr(repair_result, "diagnostics", {}).get("assigned_critical_reconstruct_rejected_uav_not_docked", 0)
                        ),
                        "assigned_critical_reconstruct_rejected_infeasible": int(
                            getattr(repair_result, "diagnostics", {}).get("assigned_critical_reconstruct_rejected_infeasible", 0)
                        ),
                        "assigned_critical_reconstruct_best_margin_by_task": dict(
                            getattr(repair_result, "diagnostics", {}).get("assigned_critical_reconstruct_best_margin_by_task", {})
                        ),
                        "assigned_critical_reconstruct_selected_uav_ids": list(
                            getattr(repair_result, "diagnostics", {}).get("assigned_critical_reconstruct_selected_uav_ids", ())
                        ),
                        "assigned_critical_reconstruct_selected_truck_ids": list(
                            getattr(repair_result, "diagnostics", {}).get("assigned_critical_reconstruct_selected_truck_ids", ())
                        ),
                        "assigned_critical_reconstruct_selected_launch_anchors": list(
                            getattr(repair_result, "diagnostics", {}).get("assigned_critical_reconstruct_selected_launch_anchors", ())
                        ),
                        "assigned_critical_reconstruct_selected_recovery_anchors": list(
                            getattr(repair_result, "diagnostics", {}).get("assigned_critical_reconstruct_selected_recovery_anchors", ())
                        ),
                        "assigned_critical_reconstruct_no_progress_tasks_targeted": int(
                            getattr(repair_result, "diagnostics", {}).get("assigned_critical_reconstruct_no_progress_tasks_targeted", 0)
                        ),
                        "assigned_critical_reconstruct_task_ids": list(
                            getattr(repair_result, "diagnostics", {}).get("assigned_critical_reconstruct_task_ids", ())
                        ),
                        "assigned_critical_reconstruct_attempt_rows": list(
                            getattr(repair_result, "diagnostics", {}).get("assigned_critical_reconstruct_attempt_rows", [])
                        ),
                        "support_reposition_shadow_enabled": bool(
                            getattr(repair_result, "diagnostics", {}).get("support_reposition_shadow_enabled", False)
                        ),
                        "support_reposition_shadow_candidates": int(
                            getattr(repair_result, "diagnostics", {}).get("support_reposition_shadow_candidates", 0)
                        ),
                        "support_reposition_shadow_feasible_suggestions": int(
                            getattr(repair_result, "diagnostics", {}).get("support_reposition_shadow_feasible_suggestions", 0)
                        ),
                        "support_reposition_shadow_low_battery_rescue_possible": int(
                            getattr(repair_result, "diagnostics", {}).get("support_reposition_shadow_low_battery_rescue_possible", 0)
                        ),
                        "support_reposition_shadow_unreachable_rescue_possible": int(
                            getattr(repair_result, "diagnostics", {}).get("support_reposition_shadow_unreachable_rescue_possible", 0)
                        ),
                        "support_reposition_shadow_no_progress_tasks_covered": int(
                            getattr(repair_result, "diagnostics", {}).get("support_reposition_shadow_no_progress_tasks_covered", 0)
                        ),
                        "support_reposition_shadow_truck_ids": list(
                            getattr(repair_result, "diagnostics", {}).get("support_reposition_shadow_truck_ids", ())
                        ),
                        "support_reposition_shadow_anchor_ids": list(
                            getattr(repair_result, "diagnostics", {}).get("support_reposition_shadow_anchor_ids", ())
                        ),
                        "support_reposition_shadow_estimated_battery_gain": float(
                            getattr(repair_result, "diagnostics", {}).get("support_reposition_shadow_estimated_battery_gain", 0.0)
                        ),
                        "support_reposition_shadow_estimated_truck_cost": float(
                            getattr(repair_result, "diagnostics", {}).get("support_reposition_shadow_estimated_truck_cost", 0.0)
                        ),
                        "support_reposition_shadow_rows": list(
                            getattr(repair_result, "diagnostics", {}).get("support_reposition_shadow_rows", [])
                        ),
                        "failure_reason_counts": dict(getattr(repair_result, "diagnostics", {}).get("failure_reason_counts", {})),
                        "reason_codes": list(getattr(repair_result, "reason_codes", ())),
                        "support_coordination": list(getattr(repair_result, "diagnostics", {}).get("support_coordination", [])),
                        "pool_id": str(pool_diag.get("pool_id", "")),
                        "pool_size_actual": int(pool_diag.get("pool_size_actual", 0)),
                        "exact_check_budget": int(pool_diag.get("exact_check_budget", 0)),
                        "selected_exact_count": int(pool_diag.get("selected_exact_count", 0)),
                    }
                )
                self._record_critical_recovery_diagnostics(repair_op, repair_result)
                if not bool(repair_result.feasible):
                    truncated_solution = self._truncate_k2_tails_for_safe_candidate(repair_result.candidate_solution)
                    truncated_goals = solution_to_legacy_goals(truncated_solution)
                    truncated_ok = bool(self._goals_hard_feasible(env, truncated_goals))
                    truncated_eval = evaluate_k2_solution(env, truncated_solution, hard_feasible=truncated_ok)
                    self.alns_diagnostics.objective_evaluation_count += 1
                    if bool(truncated_eval.feasible):
                        candidate = truncated_goals
                        repair_result = repair_result.__class__(
                            candidate_solution=truncated_solution,
                            inserted_items=getattr(repair_result, "inserted_items", ()),
                            feasible=True,
                            reason_codes=tuple(getattr(repair_result, "reason_codes", ())) + ("TAIL_TRUNCATED_FOR_SAFE_CANDIDATE",),
                            diagnostics={
                                **dict(getattr(repair_result, "diagnostics", {})),
                                "tail_truncated_for_safe_candidate": True,
                            },
                        )
                    else:
                        legacy_removed_agents = [str(getattr(item, "agent_id", item)) for item in removed]
                        legacy_candidate = self._repair_destroyed(env, destroyed, legacy_removed_agents, "risk_greedy_insert")
                        if bool(self._goals_hard_feasible(env, legacy_candidate)):
                            candidate = legacy_candidate
                            legacy_solution = legacy_goals_to_solution(
                                legacy_candidate,
                                env_adapter_context(
                                    env,
                                    sequence_length=1,
                                ),
                            )
                            repair_result = repair_result.__class__(
                                candidate_solution=legacy_solution,
                                inserted_items=getattr(repair_result, "inserted_items", ()),
                                feasible=True,
                                reason_codes=tuple(getattr(repair_result, "reason_codes", ())) + ("LEGACY_FIRST_TASK_REPAIR_FALLBACK",),
                                diagnostics={
                                    **dict(getattr(repair_result, "diagnostics", {})),
                                    "legacy_first_task_repair_fallback": True,
                                },
                            )
                        else:
                            self.alns_diagnostics.planner_candidate_hard_reject_count += 1
                            self._update_operator_stats(env, self.destroy_ops, destroy_op, -0.20, False, feasible=False)
                            self._update_operator_stats(env, self.repair_ops, repair_op, -0.20, False, feasible=False)
                            continue
            else:
                candidate = self._repair_destroyed(env, destroyed, removed, repair_op)
            if evaluation_budget > 0 and self.alns_diagnostics.objective_evaluation_count + 2 > evaluation_budget:
                break
            self.alns_diagnostics.repair_operator_usage[repair_op] = int(
                self.alns_diagnostics.repair_operator_usage.get(repair_op, 0) + 1
            )
            legacy_cur_obj = float(self._solution_objective(env, current))
            cand_obj = float(self._solution_objective(env, candidate))
            self.alns_diagnostics.objective_evaluation_count += 2
            candidate_hard_feasible = bool(self._goals_hard_feasible(env, candidate))
            live_candidate_row = self._live_candidate_feature_row(
                env,
                current=current,
                candidate=candidate,
                destroy_operator=destroy_op,
                repair_operator=repair_op,
                legacy_current_score=legacy_cur_obj,
                legacy_candidate_score=cand_obj,
                exact_feasible=candidate_hard_feasible,
                failure_reason="" if candidate_hard_feasible else "GOALS_HARD_FEASIBILITY_REJECT",
            )
            ranker_ok, ranker_diag = self._apply_live_ranker(
                env,
                live_candidate_row,
                exact_feasibility=lambda _row: candidate_hard_feasible,
            )
            live_candidate_row.update(ranker_diag)
            if not bool(ranker_ok):
                self.live_candidate_records.append(dict(live_candidate_row))
                self.alns_diagnostics.planner_candidate_hard_reject_count += 1
                self.alns_diagnostics.feasibility_evaluation_count += 1
                self._update_operator_stats(env, self.destroy_ops, destroy_op, -0.20, False, feasible=False)
                self._update_operator_stats(env, self.repair_ops, repair_op, -0.20, False, feasible=False)
                continue
            self.alns_diagnostics.feasibility_evaluation_count += 1
            self._record_objective_shadow_comparison(env, current, candidate, legacy_cur_obj, cand_obj)
            if mode == "k2_active":
                candidate_solution_for_eval = (
                    getattr(repair_result, "candidate_solution", None)
                    if use_k2_operator_pool
                    else None
                )
                if candidate_solution_for_eval is not None:
                    normalized_solution, normalized_eval, normalized_tail_truncated = self._objective_safe_k2_solution(
                        env,
                        candidate_solution_for_eval,
                        hard_feasible=self._goals_hard_feasible(env, candidate),
                    )
                    if normalized_tail_truncated:
                        candidate_solution_for_eval = normalized_solution
                        candidate = solution_to_legacy_goals(normalized_solution)
                        repair_result = repair_result.__class__(
                            candidate_solution=normalized_solution,
                            inserted_items=getattr(repair_result, "inserted_items", ()),
                            feasible=bool(normalized_eval.feasible),
                            reason_codes=tuple(getattr(repair_result, "reason_codes", ())) + ("TAIL_TRUNCATED_FOR_OBJECTIVE_FEASIBILITY",),
                            diagnostics={
                                **dict(getattr(repair_result, "diagnostics", {})),
                                "tail_truncated_for_objective_feasibility": True,
                            },
                        )
                    _cand_k2, cand_eval, _ = self._record_k2_solution_object(
                        env,
                        "active_candidate",
                        candidate_solution_for_eval,
                        hard_feasible=self._goals_hard_feasible(env, candidate),
                    )
                else:
                    _cand_k2, cand_eval, _ = self._record_k2_solution(env, "active_candidate", candidate)
                self.alns_diagnostics.objective_evaluation_count += 1
                if not bool(cand_eval.feasible):
                    self.alns_diagnostics.planner_candidate_hard_reject_count += 1
                    self._record_k2_sa_delta(
                        env,
                        current_eval=current_eval,
                        candidate_eval=cand_eval,
                        current_solution=current_solution_for_eval,
                        candidate_solution=_cand_k2,
                        delta=float(minimization_delta(cur_obj, float(cand_eval.breakdown.total_cost))),
                        temperature=temp,
                        acceptance_probability=0.0,
                        random_draw=None,
                        accepted=False,
                        destroy_operator=destroy_op,
                        repair_operator=repair_op,
                        fallback_used=bool(
                            use_k2_operator_pool
                            and isinstance(getattr(repair_result, "diagnostics", None), dict)
                            and (
                                bool(repair_result.diagnostics.get("tail_truncated_for_safe_candidate", False))
                                or bool(repair_result.diagnostics.get("legacy_first_task_repair_fallback", False))
                                or bool(repair_result.diagnostics.get("tail_truncated_for_objective_feasibility", False))
                            )
                        ),
                    )
                    self._update_operator_stats(env, self.destroy_ops, destroy_op, -0.20, False, feasible=False)
                    self._update_operator_stats(env, self.repair_ops, repair_op, -0.20, False, feasible=False)
                    continue
                _cand_k2, cand_eval = self._apply_local_search_refiner(env, _cand_k2, cand_eval)
                candidate = solution_to_legacy_goals(_cand_k2)
                self.alns_diagnostics.repair_feasible_count += 1
                cand_cost = float(cand_eval.breakdown.total_cost)
                delta = float(minimization_delta(cur_obj, cand_cost))
                acceptance_probability = float(minimization_acceptance_probability(delta, temp))
                random_draw = 0.0 if acceptance_probability >= 1.0 else float(self.rng.random())
                accepted = bool(cand_eval.feasible and random_draw < acceptance_probability)
                if bool(cand_eval.feasible) and current_solution_for_eval is not None:
                    if str(current_solution_for_eval.digest()) != str(_cand_k2.digest()):
                        self.alns_diagnostics.feasible_nonidentical_candidate_count += 1
                self._record_k2_sa_delta(
                    env,
                    current_eval=current_eval,
                    candidate_eval=cand_eval,
                    current_solution=current_solution_for_eval,
                    candidate_solution=_cand_k2,
                    delta=delta,
                    temperature=temp,
                    acceptance_probability=acceptance_probability,
                    random_draw=random_draw,
                    accepted=accepted,
                    destroy_operator=destroy_op,
                    repair_operator=repair_op,
                    fallback_used=bool(
                        use_k2_operator_pool
                        and isinstance(getattr(repair_result, "diagnostics", None), dict)
                        and (
                            bool(repair_result.diagnostics.get("tail_truncated_for_safe_candidate", False))
                            or bool(repair_result.diagnostics.get("legacy_first_task_repair_fallback", False))
                            or bool(repair_result.diagnostics.get("tail_truncated_for_objective_feasibility", False))
                        )
                    ),
                )
                reward = float(-delta)
                compare_better = bool(cand_eval.feasible and cand_cost < best_obj - 1e-12)
                effective_obj = cand_cost
            else:
                delta = float(cand_obj - cur_obj)
                accepted = bool(self._should_accept_candidate(delta, temp))
                reward = float(delta)
                compare_better = bool(cand_obj > best_obj + 1e-12)
                effective_obj = cand_obj
                self.alns_diagnostics.repair_feasible_count += 1
            live_candidate_row["accepted"] = bool(accepted)
            live_candidate_row["improved"] = bool(compare_better)
            live_candidate_row["runtime"] = float(time.perf_counter() - start_time)
            self.live_candidate_records.append(dict(live_candidate_row))
            temp = float(max(temp * float(getattr(env.cfg, "alns_sa_cooling_rate", 0.985)), float(getattr(env.cfg, "alns_sa_minimum_temperature", 1e-4))))
            if use_k2_operator_pool:
                for rec in reversed(self.canonical_operator_records):
                    if rec.get("phase") == "repair" and not bool(rec.get("_finalized", False)):
                        rec["accepted"] = bool(accepted)
                        rec["improved"] = bool(compare_better)
                        rec["_finalized"] = True
                        break
            self._update_operator_stats(
                env,
                self.destroy_ops,
                destroy_op,
                reward,
                accepted,
                feasible=True,
                improved=bool(compare_better),
                global_best=bool(compare_better),
            )
            self._update_operator_stats(
                env,
                self.repair_ops,
                repair_op,
                reward,
                accepted,
                feasible=True,
                improved=bool(compare_better),
                global_best=bool(compare_better),
            )
            self.alns_diagnostics.destroy_operator_reward[destroy_op] = float(
                self.alns_diagnostics.destroy_operator_reward.get(destroy_op, 0.0) + reward
            )
            self.alns_diagnostics.repair_operator_reward[repair_op] = float(
                self.alns_diagnostics.repair_operator_reward.get(repair_op, 0.0) + reward
            )
            if accepted:
                self.alns_accepted_count_total += 1
                self.alns_diagnostics.accepted_count += 1
                if (mode == "k2_active" and delta <= 0.0) or (mode != "k2_active" and delta >= 0.0):
                    self.alns_diagnostics.accepted_improving_count += 1
                else:
                    self.alns_diagnostics.accepted_worsening_count += 1
                current = dict(candidate)
                cur_obj = float(effective_obj)
                if mode == "k2_active":
                    current_eval = cand_eval
                    current_solution_for_eval = _cand_k2
            if compare_better:
                self.alns_improvement_count_total += 1
                self.alns_diagnostics.improvement_count += 1
                best = dict(candidate)
                best_obj = float(effective_obj)
        if mode == "k2_active":
            self.alns_diagnostics.best_objective_gain = float(max(initial_obj - best_obj, 0.0))
        else:
            self.alns_diagnostics.best_objective_gain = float(max(best_obj - initial_obj, 0.0))
        self.alns_diagnostics.final_destroy_weights = {k: float(v.weight) for k, v in self.destroy_ops.items()}
        self.alns_diagnostics.final_repair_weights = {k: float(v.weight) for k, v in self.repair_ops.items()}
        self.alns_diagnostics.wall_clock_time_s += float(time.perf_counter() - start_time)
        if mode == "k2_active":
            if best_obj >= cur_obj - max(0.20, temp):
                return current
            return self._restore_protected_goals(env, current, best)
        if best_obj <= cur_obj + max(0.20, temp):
            return current
        return self._restore_protected_goals(env, current, best)

    def get_alns_diagnostics(self) -> ALNSDiagnostics:
        self.alns_diagnostics.iteration_count = int(self.alns_iteration_count_total)
        self.alns_diagnostics.destroyed_assignment_count = int(self.alns_destroyed_assignment_count_total)
        self.alns_diagnostics.accepted_count = int(self.alns_accepted_count_total)
        self.alns_diagnostics.improvement_count = int(self.alns_improvement_count_total)
        self._refresh_tail_lifetime_metrics()
        return self.alns_diagnostics


PredictiveAdaptiveRollingALNSPlanner = EventResponsiveALNSPlanner
