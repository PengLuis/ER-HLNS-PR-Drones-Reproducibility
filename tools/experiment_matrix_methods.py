from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import json
from typing import Any, Dict, Optional

import numpy as np

from hetgat_hrl.core.mdp_spec import AgentKind, EnvConfig, TaskKind, TaskStatus
from hetgat_hrl.alns.tabu_search import TabuSearchK2Planner
from hetgat_hrl.alns.hybrid_genetic import HybridGeneticK2Planner
from hetgat_hrl.alns.vns_search import VariableNeighborhoodSearchK2Planner
from hetgat_hrl.alns.rolling_horizon_alns import RollingHorizonALNSPlanner
from hetgat_hrl.alns.dynamic_replanning import DynamicReplanningALNSPlanner
from hetgat_hrl.core.algorithm_profile import (
    AlgorithmProfile,
    ER_HLNS_B_DOCKED_LATCH_REARM_CAPABILITY,
    ER_HLNS_B_PRELAUNCH_CONTRACT_LOCK_CAPABILITY,
    ER_HLNS_COORDINATION_CAPABILITY,
    ER_HLNS_BALANCED_ALL_TASKS_CAPABILITY,
    ER_HLNS_BALANCED_ALL_TASKS_V2_CAPABILITY,
    ER_HLNS_BALANCED_ALL_TASKS_V3_CAPABILITY,
    ER_HLNS_IDLE_ROUTINE_DISPATCH_CAPABILITY,
    ER_HLNS_PARALLEL_ROUTINE_EMERGENCY_CAPABILITY,
    ER_HLNS_R4_ROUTINE_TAKEOVER_CAPABILITY,
    ER_HLNS_ROUTE_PLAN_CAPABILITY,
    UAV_SCOUT_INFORMATION_CAPABILITY,
)
from hetgat_hrl.hrl.event_responsive_alns_planner import EventResponsiveALNSPlanner
from hetgat_hrl.hrl.er_hlns_planner import ERHLNSPlanner
from hetgat_hrl.hrl.algorithm_package import AlgorithmPackage
from hetgat_hrl.hrl.planner import RiskTriggeredHRLPlanner
from hetgat_hrl.hrl.rolling_planner import EventTriggeredRollingPlanner
from hetgat_hrl.hrl_v2 import ErcRhcV2Planner

ALNS_MAINLINE_METHOD = "event_responsive_alns"
ER_ALNS_MAINLINE_METHOD = "er_alns_mainline"
ER_ALNS_CURRENT_METHOD = "er_alns_current"
ER_ALNS_INIT_PLUS_METHOD = "er_alns_init_plus"
ER_ALNS_REPAIR_PLUS_METHOD = "er_alns_repair_plus"
ER_ALNS_FEASIBILITY_RESTORE_METHOD = "er_alns_feasibility_restore"
ER_ALNS_BUDGET_1_25_METHOD = "er_alns_budget_1_25"
ER_ALNS_BUDGET_1_50_METHOD = "er_alns_budget_1_50"
ER_ALNS_COMBINED_CANDIDATE_METHOD = "er_alns_combined_candidate"
UNIFORM_LNS_METHOD = "uniform_lns"
CANONICAL_ALNS_METHOD = "canonical_alns"
TABU_SEARCH_METHOD = "tabu_search"
ER_HLNS_METHOD = "er_hlns"
# Candidate-only ER-HLNS route repair. It is intentionally excluded from the
# formal method matrix and must be requested explicitly by a candidate runner.
ER_HLNS_RISK_SLACK_ROUTINE_METHOD = "er_hlns_risk_slack_routine_candidate"
ER_HLNS_PARALLEL_ROUTINE_EMERGENCY_METHOD = "er_hlns_parallel_routine_emergency_candidate"
ER_HLNS_FORCE_INITIAL_LIFELINE_METHOD = (
    "er_hlns_force_initial_lifeline_ordering_candidate"
)
ER_HLNS_R4_ROUTINE_TAKEOVER_METHOD = "er_hlns_r4_routine_takeover_candidate"
ER_HLNS_IDLE_ROUTINE_DISPATCH_METHOD = "er_hlns_idle_routine_dispatch_candidate"
ER_HLNS_IDLE_BALANCED_ROUTINE_METHOD = "er_hlns_idle_balanced_routine_candidate"
ER_HLNS_BALANCED_ALL_TASKS_METHOD = "er_hlns_balanced_all_tasks_candidate"
ER_HLNS_BALANCED_ALL_TASKS_V2_METHOD = "er_hlns_balanced_all_tasks_v2_candidate"
ER_HLNS_BALANCED_ALL_TASKS_V3_METHOD = "er_hlns_balanced_all_tasks_v3_candidate"
ER_HLNS_BALANCED_ALL_TASKS_V4_METHOD = (
    "er_hlns_balanced_all_tasks_v4_prelaunch_parallel_candidate"
)
ER_HLNS_BALANCED_ALL_TASKS_V5_METHOD = (
    "er_hlns_balanced_all_tasks_v5_launch_first_candidate"
)
ER_HLNS_LB_HARD_COVERAGE_METHOD = "er_hlns_lb_hard_coverage_candidate"
ER_HLNS_LB_HARD_COVERAGE_SINGLE_RESCUE_METHOD = (
    "er_hlns_lb_hard_coverage_single_rescue_candidate"
)
ER_HLNS_LB_HARD_COVERAGE_PROTECTED_METHOD = (
    "er_hlns_lb_hard_coverage_protected_candidate"
)
ER_HLNS_LB_HARD_COVERAGE_COMMITMENT_METHOD = (
    "er_hlns_lb_hard_coverage_commitment_candidate"
)
ER_HLNS_LB_HARD_COVERAGE_ORPHAN_GUARD_METHOD = (
    "er_hlns_lb_hard_coverage_orphan_guard_candidate"
)
ER_HLNS_LB_ROUTINE_PROTECTED_METHOD = "er_hlns_lb_routine_protected_candidate"
ER_HLNS_LB_ROUTINE_PROTECTED_OWNER_REPAIR_METHOD = (
    "er_hlns_lb_routine_protected_owner_repair_candidate"
)
ER_HLNS_LB_HARD_COVERAGE_SAFETY_GATED_METHOD = (
    "er_hlns_lb_hard_coverage_safety_gated_candidate"
)
ER_HLNS_LB_ROUTINE_PROTECTED_EMERGENCY_RESCUE_METHOD = (
    "er_hlns_lb_routine_protected_emergency_rescue_candidate"
)
ER_HLNS_LB_ROUTINE_PROTECTED_EMERGENCY_RESCUE_REPAIR_METHOD = (
    "er_hlns_lb_routine_protected_emergency_rescue_repair_candidate"
)
ER_HLNS_LB_ROUTINE_PROTECTED_V3_SELECTOR_METHOD = (
    "er_hlns_lb_routine_protected_v3_selector_candidate"
)
ER_HLNS_LB_ADAPTIVE_COVERAGE_METHOD = "er_hlns_lb_adaptive_coverage_candidate"
ER_HLNS_LB_ADAPTIVE_SINGLE_RESCUE_METHOD = (
    "er_hlns_lb_adaptive_single_rescue_candidate"
)
ER_HLNS_LB_BALANCED_SELECTOR_METHOD = "er_hlns_lb_balanced_selector_candidate"
ER_HLNS_BALANCED_ALL_TASKS_V6_METHOD = (
    "er_hlns_balanced_all_tasks_v6_emergency_first_parallel_candidate"
)
ER_HLNS_BALANCED_ALL_TASKS_V7_METHOD = (
    "er_hlns_balanced_all_tasks_v7_conditional_road_promotion_candidate"
)
ER_HLNS_BALANCED_ALL_TASKS_V8_METHOD = (
    "er_hlns_balanced_all_tasks_v8_shadow_total_coverage_candidate"
)
PRIORITY_GREEDY_METHOD = "priority_greedy"
# Pure nearest eligible-task greedy comparator.  It is deliberately separate
# from the priority/deadline rule so the two baselines have distinct runtime
# identities and algorithm hashes.
NEAREST_GREEDY_METHOD = "greedy_rule"
ROLLING_HORIZON_METHOD = "rolling_horizon"
HYBRID_GENETIC_METHOD = "hybrid_genetic"
VNS_METHOD = "variable_neighborhood_search"
ROLLING_HORIZON_ALNS_METHOD = "rolling_horizon_alns"
DYNAMIC_REPLANNING_ALNS_METHOD = "dynamic_replanning_alns"
# Candidate-only clean comparator.  The formal C-ALNS identifier remains
# ``rolling_horizon_alns``; this alias applies the capability isolation overlay
# only when explicitly requested by a candidate runner.
C_ALNS_CLEAN_METHOD = "c_alns_clean"
FORMAL_ER_HLNS_ABLATION_OVERRIDES = {
    "er_hlns_no_lns": {"hrl_route_plan_alns_iterations": 0},
    "er_hlns_no_joint_corridor": {"hrl_route_plan_joint_corridor_enabled": False},
    "er_hlns_no_queue_rescue": {"hrl_route_plan_stalled_queue_rescue_enabled": False},
    "er_hlns_no_stalled_contract_transfer": {"hrl_route_plan_contract_transfer_enabled": False},
    "er_hlns_no_deadline_rescue": {"hrl_route_plan_deadline_rescue_enabled": False},
    "er_hlns_no_uav_scout_information": {"hrl_route_plan_uav_scout_enabled": False},
    "er_hlns_no_route_suffix_event_repair": {"hrl_route_plan_event_replan_enabled": False},
}
# Historical result identifiers remain readable, but new experiments use names
# that match the exact mechanism being removed.
LEGACY_ER_HLNS_ABLATION_ALIASES = {
    "er_hlns_no_contract_transfer": "er_hlns_no_stalled_contract_transfer",
    "er_hlns_no_uav_scout": "er_hlns_no_uav_scout_information",
    "er_hlns_no_event_replan": "er_hlns_no_route_suffix_event_repair",
}
ER_HLNS_ABLATION_OVERRIDES = {
    **FORMAL_ER_HLNS_ABLATION_OVERRIDES,
    # Onsite service is now a common physical rule and is no longer a valid
    # algorithm ablation.  Keep the historical identifier readable only.
    "er_hlns_no_onsite_takeover": {"hrl_route_plan_onsite_takeover_enabled": False},
    # Historical 800/200 migration experiments used this identifier.  Keep it
    # readable, but do not include it in FORMAL_ER_HLNS_ABLATIONS: basic-task
    # UAV relay is now a public physical invariant (already False for every
    # method), so disabling it is not a valid algorithm ablation.
    "er_hlns_no_bulk_relay": {"hrl_route_plan_bulk_relay_enabled": False},
    **{
        legacy: FORMAL_ER_HLNS_ABLATION_OVERRIDES[current]
        for legacy, current in LEGACY_ER_HLNS_ABLATION_ALIASES.items()
    },
}
FORMAL_ER_HLNS_ABLATIONS = tuple(FORMAL_ER_HLNS_ABLATION_OVERRIDES)
V2_BASE_METHOD = "v2_base"
V2_DYNAMIC_K_METHOD = "v2_dynamic_k"
V2_LOCAL_SEARCH_METHOD = "v2_local_search"
V2_COMBINED_METHOD = "v2_combined"
V2_BASE_EXTRA_BUDGET_METHOD = "v2_base_extra_budget"
LS_ABLATION_METHODS = {
    "ls_full": (),
    "ls_no_relocate": ("relocate",),
    "ls_no_swap": ("swap",),
    "ls_no_tail_exchange": ("tail_exchange",),
    "ls_no_support_binding_refinement": ("support_binding_refinement",),
    "ls_no_recovery_anchor_refinement": ("recovery_anchor_refinement",),
}
ALNS_METHOD_ALIASES = {
    "predictive_alns": ALNS_MAINLINE_METHOD,
    "er_alns_mainline": V2_BASE_METHOD,
    "er_alns_current": V2_BASE_METHOD,
    "v2_base": V2_BASE_METHOD,
    "v2_dynamic_k": V2_DYNAMIC_K_METHOD,
    "v2_local_search": V2_LOCAL_SEARCH_METHOD,
    "v2_combined": V2_COMBINED_METHOD,
    "v2_base_extra_budget": V2_BASE_EXTRA_BUDGET_METHOD,
    "greedy": PRIORITY_GREEDY_METHOD,
    "nn_greedy": NEAREST_GREEDY_METHOD,
    "nearest_greedy": NEAREST_GREEDY_METHOD,
    "hga": HYBRID_GENETIC_METHOD,
    "vns": VNS_METHOD,
    "rh_alns": ROLLING_HORIZON_ALNS_METHOD,
    "dynamic_alns": DYNAMIC_REPLANNING_ALNS_METHOD,
    "c-alns-clean": C_ALNS_CLEAN_METHOD,
    "c_alns": C_ALNS_CLEAN_METHOD,
    "c-alns": C_ALNS_CLEAN_METHOD,
    "er-hlns-balanced-all-v4-prelaunch-parallel": (
        ER_HLNS_BALANCED_ALL_TASKS_V4_METHOD
    ),
    "er_hlns-balanced-all-v4-prelaunch-parallel": (
        ER_HLNS_BALANCED_ALL_TASKS_V4_METHOD
    ),
    "er_hlns_balanced_all_v4_prelaunch_parallel": (
        ER_HLNS_BALANCED_ALL_TASKS_V4_METHOD
    ),
    "er-hlns-balanced-all-v5-launch-first": (
        ER_HLNS_BALANCED_ALL_TASKS_V5_METHOD
    ),
    "er_hlns_balanced_all_tasks_v5_launch_first": (
        ER_HLNS_BALANCED_ALL_TASKS_V5_METHOD
    ),
    "er-hlns-lb-hard-coverage": ER_HLNS_LB_HARD_COVERAGE_METHOD,
    "er_hlns_lb_hard_coverage": ER_HLNS_LB_HARD_COVERAGE_METHOD,
    "er-hlns-lb-hard-coverage-single-rescue": (
        ER_HLNS_LB_HARD_COVERAGE_SINGLE_RESCUE_METHOD
    ),
    "er_hlns_lb_hard_coverage_single_rescue": (
        ER_HLNS_LB_HARD_COVERAGE_SINGLE_RESCUE_METHOD
    ),
    "er-hlns-lb-hard-coverage-protected": ER_HLNS_LB_HARD_COVERAGE_PROTECTED_METHOD,
    "er_hlns_lb_hard_coverage_protected": ER_HLNS_LB_HARD_COVERAGE_PROTECTED_METHOD,
    "er-hlns-lb-hard-coverage-commitment": ER_HLNS_LB_HARD_COVERAGE_COMMITMENT_METHOD,
    "er_hlns_lb_hard_coverage_commitment": ER_HLNS_LB_HARD_COVERAGE_COMMITMENT_METHOD,
    "er-hlns-lb-hard-coverage-orphan-guard": ER_HLNS_LB_HARD_COVERAGE_ORPHAN_GUARD_METHOD,
    "er_hlns_lb_hard_coverage_orphan_guard": ER_HLNS_LB_HARD_COVERAGE_ORPHAN_GUARD_METHOD,
    "er-hlns-lb-routine-protected": ER_HLNS_LB_ROUTINE_PROTECTED_METHOD,
    "er_hlns_lb_routine_protected": ER_HLNS_LB_ROUTINE_PROTECTED_METHOD,
    "er-hlns-lb-routine-protected-owner-repair": (
        ER_HLNS_LB_ROUTINE_PROTECTED_OWNER_REPAIR_METHOD
    ),
    "er_hlns_lb_routine_protected_owner_repair": (
        ER_HLNS_LB_ROUTINE_PROTECTED_OWNER_REPAIR_METHOD
    ),
    "er-hlns-lb-hard-coverage-safety-gated": ER_HLNS_LB_HARD_COVERAGE_SAFETY_GATED_METHOD,
    "er_hlns_lb_hard_coverage_safety_gated": ER_HLNS_LB_HARD_COVERAGE_SAFETY_GATED_METHOD,
    "er-hlns-lb-routine-protected-emergency-rescue": ER_HLNS_LB_ROUTINE_PROTECTED_EMERGENCY_RESCUE_METHOD,
    "er_hlns_lb_routine_protected_emergency_rescue": ER_HLNS_LB_ROUTINE_PROTECTED_EMERGENCY_RESCUE_METHOD,
    "er-hlns-lb-routine-protected-emergency-rescue-repair": ER_HLNS_LB_ROUTINE_PROTECTED_EMERGENCY_RESCUE_REPAIR_METHOD,
    "er_hlns_lb_routine_protected_emergency_rescue_repair": ER_HLNS_LB_ROUTINE_PROTECTED_EMERGENCY_RESCUE_REPAIR_METHOD,
    "er-hlns-lb-routine-protected-v3-selector": ER_HLNS_LB_ROUTINE_PROTECTED_V3_SELECTOR_METHOD,
    "er_hlns_lb_routine_protected_v3_selector": ER_HLNS_LB_ROUTINE_PROTECTED_V3_SELECTOR_METHOD,
    "er-hlns-lb-adaptive-coverage": ER_HLNS_LB_ADAPTIVE_COVERAGE_METHOD,
    "er_hlns_lb_adaptive_coverage": ER_HLNS_LB_ADAPTIVE_COVERAGE_METHOD,
    "er-hlns-lb-adaptive-single-rescue": (
        ER_HLNS_LB_ADAPTIVE_SINGLE_RESCUE_METHOD
    ),
    "er_hlns_lb_adaptive_single_rescue": (
        ER_HLNS_LB_ADAPTIVE_SINGLE_RESCUE_METHOD
    ),
    "er-hlns-lb-balanced-selector": ER_HLNS_LB_BALANCED_SELECTOR_METHOD,
    "er_hlns_lb_balanced_selector": ER_HLNS_LB_BALANCED_SELECTOR_METHOD,
    "er-hlns-balanced-all-v6-emergency-first-parallel": (
        ER_HLNS_BALANCED_ALL_TASKS_V6_METHOD
    ),
    "er_hlns-balanced-all-v6-emergency-first-parallel": (
        ER_HLNS_BALANCED_ALL_TASKS_V6_METHOD
    ),
    "er-hlns-balanced-all-v7-conditional-road-promotion": (
        ER_HLNS_BALANCED_ALL_TASKS_V7_METHOD
    ),
    "er_hlns_balanced_all_tasks_v7_conditional_road_promotion": (
        ER_HLNS_BALANCED_ALL_TASKS_V7_METHOD
    ),
    "er-hlns-balanced-all-v8-shadow-total-coverage": (
        ER_HLNS_BALANCED_ALL_TASKS_V8_METHOD
    ),
    "er_hlns_balanced_all_tasks_v8_shadow_total_coverage": (
        ER_HLNS_BALANCED_ALL_TASKS_V8_METHOD
    ),
    "er-hlns-force-initial-lifeline-ordering": (
        ER_HLNS_FORCE_INITIAL_LIFELINE_METHOD
    ),
    "er_hlns_force_initial_lifeline_ordering": (
        ER_HLNS_FORCE_INITIAL_LIFELINE_METHOD
    ),
}


def _canonical_method_name(method: str) -> str:
    m = str(method).strip().lower()
    return str(ALNS_METHOD_ALIASES.get(m, m))

SUPPORTED_METHODS = [
    PRIORITY_GREEDY_METHOD,
    ROLLING_HORIZON_METHOD,
    HYBRID_GENETIC_METHOD,
    VNS_METHOD,
    ROLLING_HORIZON_ALNS_METHOD,
    DYNAMIC_REPLANNING_ALNS_METHOD,
    ER_HLNS_METHOD,
    ER_HLNS_PARALLEL_ROUTINE_EMERGENCY_METHOD,
    ER_HLNS_FORCE_INITIAL_LIFELINE_METHOD,
    ER_HLNS_R4_ROUTINE_TAKEOVER_METHOD,
    ER_HLNS_IDLE_ROUTINE_DISPATCH_METHOD,
    ER_HLNS_IDLE_BALANCED_ROUTINE_METHOD,
    ER_HLNS_BALANCED_ALL_TASKS_METHOD,
    ER_HLNS_BALANCED_ALL_TASKS_V2_METHOD,
    ER_HLNS_BALANCED_ALL_TASKS_V3_METHOD,
    ER_HLNS_BALANCED_ALL_TASKS_V4_METHOD,
    ER_HLNS_BALANCED_ALL_TASKS_V5_METHOD,
    ER_HLNS_LB_HARD_COVERAGE_METHOD,
    ER_HLNS_LB_HARD_COVERAGE_SINGLE_RESCUE_METHOD,
    ER_HLNS_LB_HARD_COVERAGE_PROTECTED_METHOD,
    ER_HLNS_LB_HARD_COVERAGE_COMMITMENT_METHOD,
    ER_HLNS_LB_HARD_COVERAGE_ORPHAN_GUARD_METHOD,
    ER_HLNS_LB_ROUTINE_PROTECTED_METHOD,
    ER_HLNS_LB_ROUTINE_PROTECTED_OWNER_REPAIR_METHOD,
    ER_HLNS_LB_HARD_COVERAGE_SAFETY_GATED_METHOD,
    ER_HLNS_LB_ROUTINE_PROTECTED_EMERGENCY_RESCUE_METHOD,
    ER_HLNS_LB_ROUTINE_PROTECTED_EMERGENCY_RESCUE_REPAIR_METHOD,
    ER_HLNS_LB_ROUTINE_PROTECTED_V3_SELECTOR_METHOD,
    ER_HLNS_LB_ADAPTIVE_COVERAGE_METHOD,
    ER_HLNS_LB_ADAPTIVE_SINGLE_RESCUE_METHOD,
    ER_HLNS_LB_BALANCED_SELECTOR_METHOD,
    ER_HLNS_BALANCED_ALL_TASKS_V6_METHOD,
    ER_HLNS_BALANCED_ALL_TASKS_V7_METHOD,
    ER_HLNS_BALANCED_ALL_TASKS_V8_METHOD,
    *ER_HLNS_ABLATION_OVERRIDES.keys(),
    "random_rule",
    "greedy_rule",
    "rolling_fixed",
    "erc_rhc_old",
    "erc_rhc",
    "erc_rhc_v2",
    "erc_rhc_v2_support_recovery_repair",
    "erc_rhc_v2_gate16",
    "erc_rhc_v2_gate17",
    "erc_rhc_v2_direct_tc_first",
    "erc_rhc_v2_support_anchor_strict",
    "erc_rhc_v2_anchor_arrival_force_launch",
    "erc_rhc_v2_safety_narrow_routine_guard",
    "erc_rhc_v2_gate18_core",
    "erc_rhc_v2_gate19_launch_binding",
    "erc_rhc_v2_gate19_support_lock",
    "erc_rhc_v2_gate19_core",
    "erc_rhc_v2_gate20_passenger_invariant",
    "erc_rhc_v2_gate20_support_reserve_launch",
    "erc_rhc_v2_gate20_rebind_at_anchor",
    "erc_rhc_v2_gate20_core",
    "erc_rhc_v2_workflow_repair",
    "erc_rhc_v2_support_quality_audit",
    "erc_rhc_v2_support_quality_gate",
    "erc_rhc_v2_support_quality_relaxed",
    "erc_rhc_current",
    "erc_base_no_event",
    "erc_support_authorized",
    "erc_routine_commit",
    "erc_launch_quality_gate",
    "erc_tc_completion_chain",
    "erc_event_minimal_local",
    "erc_scoring_shrink",
    "erc_combined_safe_core",
    "erc_full",
    "erc_hard_events_only",
    "erc_no_map_ranking_refresh",
    "erc_no_tc_global_assignment",
    "erc_no_support_chain",
    "erc_no_cluster_primary_reservation",
    "erc_no_event_scoring_bonus",
    "erc_no_normal_protection",
    "erc_only_uav_commit",
    "erc_only_truck_escape",
    "erc_only_eta_exit",
    "erc_routine_rescue_combo",
    "erc_km_hard_routine",
    "erc_km_hard_routine_v2",
    "erc_same_config",
    ALNS_MAINLINE_METHOD,
    ER_ALNS_MAINLINE_METHOD,
    ER_ALNS_CURRENT_METHOD,
    ER_ALNS_INIT_PLUS_METHOD,
    ER_ALNS_REPAIR_PLUS_METHOD,
    ER_ALNS_FEASIBILITY_RESTORE_METHOD,
    ER_ALNS_BUDGET_1_25_METHOD,
    ER_ALNS_BUDGET_1_50_METHOD,
    ER_ALNS_COMBINED_CANDIDATE_METHOD,
    V2_BASE_METHOD,
    V2_DYNAMIC_K_METHOD,
    V2_LOCAL_SEARCH_METHOD,
    V2_COMBINED_METHOD,
    V2_BASE_EXTRA_BUDGET_METHOD,
    *LS_ABLATION_METHODS.keys(),
    UNIFORM_LNS_METHOD,
    CANONICAL_ALNS_METHOD,
    TABU_SEARCH_METHOD,
    "erc_ya_balanced",
    "erc_mc_lc_boost",
    "erc_ya_km_bridge",
    "erc_mc_lc_safe",
    "erc_ya_tune_a",
    "erc_ya_tune_b",
    "erc_ya_tune_c",
    "erc_support_recovery_launch",
    "erc_support_recovery_launch_relaxed",
    "erc_support_recovery_bound",
    "erc_support_recovery_budget",
    "erc_support_recovery_budget_strict",
    "erc_support_recovery_budget_urgent",
    "erc_support_priority",
    "erc_support_priority_strict",
    "erc_support_priority_reserve",
    "erc_full_new",
    "erc_reservation_only",
    "erc_airborne_lock_only",
    "erc_truck_assist_only",
    "erc_reservation_lock_no_assist",
    "ppo_pooled",
    "ppo_mlp",
    "ppo_hetgat_mask",
]


def _method_backend_family(method: str) -> str:
    m = _canonical_method_name(method)
    if m in {"random_rule", "greedy_rule", PRIORITY_GREEDY_METHOD}:
        return "rule_planner"
    if m == ROLLING_HORIZON_METHOD:
        return "rolling_horizon_planner"
    if m == HYBRID_GENETIC_METHOD:
        return "hybrid_genetic_k2_planner"
    if m == VNS_METHOD:
        return "variable_neighborhood_k2_planner"
    if m == ROLLING_HORIZON_ALNS_METHOD:
        return "rolling_horizon_alns_planner"
    if m == DYNAMIC_REPLANNING_ALNS_METHOD:
        return "dynamic_replanning_alns_planner"
    if m == C_ALNS_CLEAN_METHOD:
        return "rolling_horizon_alns_planner"
    if (
        m in {
            ER_HLNS_METHOD,
            ER_HLNS_RISK_SLACK_ROUTINE_METHOD,
            ER_HLNS_PARALLEL_ROUTINE_EMERGENCY_METHOD,
            ER_HLNS_FORCE_INITIAL_LIFELINE_METHOD,
            ER_HLNS_R4_ROUTINE_TAKEOVER_METHOD,
            ER_HLNS_IDLE_ROUTINE_DISPATCH_METHOD,
            ER_HLNS_IDLE_BALANCED_ROUTINE_METHOD,
            ER_HLNS_BALANCED_ALL_TASKS_METHOD,
            ER_HLNS_BALANCED_ALL_TASKS_V2_METHOD,
            ER_HLNS_BALANCED_ALL_TASKS_V3_METHOD,
            ER_HLNS_BALANCED_ALL_TASKS_V4_METHOD,
            ER_HLNS_BALANCED_ALL_TASKS_V5_METHOD,
            ER_HLNS_BALANCED_ALL_TASKS_V6_METHOD,
            ER_HLNS_BALANCED_ALL_TASKS_V7_METHOD,
            ER_HLNS_BALANCED_ALL_TASKS_V8_METHOD,
            ER_HLNS_LB_HARD_COVERAGE_METHOD,
            ER_HLNS_LB_HARD_COVERAGE_SINGLE_RESCUE_METHOD,
            ER_HLNS_LB_HARD_COVERAGE_PROTECTED_METHOD,
            ER_HLNS_LB_HARD_COVERAGE_COMMITMENT_METHOD,
            ER_HLNS_LB_HARD_COVERAGE_ORPHAN_GUARD_METHOD,
            ER_HLNS_LB_ROUTINE_PROTECTED_METHOD,
            ER_HLNS_LB_ROUTINE_PROTECTED_OWNER_REPAIR_METHOD,
            ER_HLNS_LB_HARD_COVERAGE_SAFETY_GATED_METHOD,
            ER_HLNS_LB_ROUTINE_PROTECTED_EMERGENCY_RESCUE_METHOD,
            ER_HLNS_LB_ROUTINE_PROTECTED_V3_SELECTOR_METHOD,
            ER_HLNS_LB_ADAPTIVE_COVERAGE_METHOD,
            ER_HLNS_LB_ADAPTIVE_SINGLE_RESCUE_METHOD,
        }
        or m in ER_HLNS_ABLATION_OVERRIDES
    ):
        return "er_hlns_planner"
    if m in {"erc_rhc_v2", "erc_rhc_v2_support_recovery_repair", "erc_rhc_v2_gate16", "erc_rhc_v2_gate17", "erc_rhc_v2_direct_tc_first", "erc_rhc_v2_support_anchor_strict", "erc_rhc_v2_anchor_arrival_force_launch", "erc_rhc_v2_safety_narrow_routine_guard", "erc_rhc_v2_gate18_core", "erc_rhc_v2_gate19_launch_binding", "erc_rhc_v2_gate19_support_lock", "erc_rhc_v2_gate19_core", "erc_rhc_v2_gate20_passenger_invariant", "erc_rhc_v2_gate20_support_reserve_launch", "erc_rhc_v2_gate20_rebind_at_anchor", "erc_rhc_v2_gate20_core", "erc_rhc_v2_workflow_repair", "erc_rhc_v2_support_quality_audit", "erc_rhc_v2_support_quality_gate", "erc_rhc_v2_support_quality_relaxed"}:
        return "erc_rhc_v2_planner"
    if m in {
        "rolling_fixed",
        "erc_rhc_old",
        "erc_rhc",
        "erc_rhc_current",
        "erc_base_no_event",
        "erc_support_authorized",
        "erc_routine_commit",
        "erc_launch_quality_gate",
        "erc_tc_completion_chain",
        "erc_event_minimal_local",
        "erc_scoring_shrink",
        "erc_combined_safe_core",
        "erc_full",
        "erc_hard_events_only",
        "erc_no_map_ranking_refresh",
        "erc_no_tc_global_assignment",
        "erc_no_support_chain",
        "erc_no_cluster_primary_reservation",
        "erc_no_event_scoring_bonus",
        "erc_no_normal_protection",
        "erc_only_uav_commit",
        "erc_only_truck_escape",
        "erc_only_eta_exit",
        "erc_routine_rescue_combo",
        "erc_km_hard_routine",
        "erc_km_hard_routine_v2",
        "erc_same_config",
        ALNS_MAINLINE_METHOD,
        ER_ALNS_MAINLINE_METHOD,
        ER_ALNS_CURRENT_METHOD,
        ER_ALNS_INIT_PLUS_METHOD,
        ER_ALNS_REPAIR_PLUS_METHOD,
        ER_ALNS_FEASIBILITY_RESTORE_METHOD,
        ER_ALNS_BUDGET_1_25_METHOD,
        ER_ALNS_BUDGET_1_50_METHOD,
        ER_ALNS_COMBINED_CANDIDATE_METHOD,
        V2_BASE_METHOD,
        V2_DYNAMIC_K_METHOD,
        V2_LOCAL_SEARCH_METHOD,
        V2_COMBINED_METHOD,
        V2_BASE_EXTRA_BUDGET_METHOD,
        *LS_ABLATION_METHODS.keys(),
        UNIFORM_LNS_METHOD,
        CANONICAL_ALNS_METHOD,
        TABU_SEARCH_METHOD,
        UNIFORM_LNS_METHOD,
        CANONICAL_ALNS_METHOD,
        "erc_ya_balanced",
        "erc_mc_lc_boost",
        "erc_ya_km_bridge",
        "erc_mc_lc_safe",
        "erc_ya_tune_a",
        "erc_ya_tune_b",
        "erc_ya_tune_c",
        "erc_support_recovery_launch",
        "erc_support_recovery_launch_relaxed",
        "erc_support_recovery_bound",
        "erc_support_recovery_budget",
        "erc_support_recovery_budget_strict",
        "erc_support_recovery_budget_urgent",
        "erc_support_priority",
        "erc_support_priority_strict",
        "erc_support_priority_reserve",
        "erc_full_new",
        "erc_reservation_only",
        "erc_airborne_lock_only",
        "erc_truck_assist_only",
        "erc_reservation_lock_no_assist",
    }:
        return "event_triggered_rolling_planner"
    if m in {"ppo_pooled", "ppo_mlp", "ppo_hetgat_mask"}:
        return "risk_triggered_hrl_planner"
    return "unknown"

def _scenario_cfg(scenario: str, cfg: EnvConfig) -> EnvConfig:
    sc = str(scenario).upper().strip()
    # Scenario C differs from B through EnvConfig's spatially correlated,
    # persistent communication-blackout behavior.  Road and communication
    # parameters remain config-owned so that
    # controlled B/C comparisons cannot be silently overridden here.
    return replace(cfg, scenario=sc)


class RandomRulePlanner:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(int(seed))

    def plan(self, env) -> Dict[str, Optional[str]]:
        pending = [t for t in env.state.tasks.values() if t.status == TaskStatus.PENDING]
        truck_ids = [aid for aid, st in env.state.agents.items() if st.kind == AgentKind.TRUCK]
        goals: Dict[str, Optional[str]] = {}
        used = set()
        for aid, st in env.state.agents.items():
            if st.kind == AgentKind.UAV and bool(st.crashed):
                goals[aid] = None
                continue
            cands = pending
            if st.kind == AgentKind.UAV:
                cands = [t for t in pending if t.kind == TaskKind.EMERGENCY]
            cands = [t for t in cands if str(t.task_id) not in used]
            if cands:
                idx = int(self.rng.integers(0, len(cands)))
                tid = str(cands[idx].task_id)
                goals[aid] = tid
                used.add(tid)
            elif st.kind == AgentKind.UAV and truck_ids:
                goals[aid] = str(truck_ids[int(self.rng.integers(0, len(truck_ids)))])
            else:
                goals[aid] = None
        return goals


class GreedyRulePlanner:
    def plan(self, env) -> Dict[str, Optional[str]]:
        used = set()
        goals: Dict[str, Optional[str]] = {}
        for aid, st in env.state.agents.items():
            if st.kind == AgentKind.UAV and bool(st.crashed):
                goals[aid] = None
                continue
            tasks = [t for t in env.state.tasks.values() if t.status == TaskStatus.PENDING]
            if st.kind == AgentKind.UAV:
                tasks = [t for t in tasks if t.kind == TaskKind.EMERGENCY]
            tasks = [t for t in tasks if str(t.task_id) not in used]
            if not tasks:
                if st.kind == AgentKind.UAV:
                    truck_ids = [x for x, xs in env.state.agents.items() if xs.kind == AgentKind.TRUCK]
                    goals[aid] = truck_ids[0] if truck_ids else None
                else:
                    goals[aid] = None
                continue
            tasks.sort(key=lambda t: float(env._agent_distance_to_task(aid, t)))
            goals[aid] = str(tasks[0].task_id)
            used.add(str(tasks[0].task_id))
        return goals


class PriorityGreedyPlanner:
    """Priority/deadline/distance rule used by the PG baseline.

    This keeps the shared eligibility semantics of ``GreedyRulePlanner`` but
    ranks time-critical tasks ahead of routine tasks, then uses urgency,
    deadline slack, and distance as deterministic tie-breakers.  It is an
    algorithm-only distinction; no physical or task-generation field is
    changed.
    """

    def plan(self, env) -> Dict[str, Optional[str]]:
        used = set()
        goals: Dict[str, Optional[str]] = {}
        step = int(getattr(env.state, "step_index", 0))
        for aid, st in env.state.agents.items():
            if st.kind == AgentKind.UAV and bool(st.crashed):
                goals[aid] = None
                continue
            tasks = [
                t for t in env.state.tasks.values()
                if t.status == TaskStatus.PENDING and str(t.task_id) not in used
            ]
            if st.kind == AgentKind.UAV:
                tasks = [t for t in tasks if t.kind == TaskKind.EMERGENCY]
            if not tasks:
                if st.kind == AgentKind.UAV:
                    truck_ids = [x for x, xs in env.state.agents.items() if xs.kind == AgentKind.TRUCK]
                    goals[aid] = truck_ids[0] if truck_ids else None
                else:
                    goals[aid] = None
                continue

            def key(task):
                is_emergency = 0 if task.kind == TaskKind.EMERGENCY else 1
                urgency = -float(getattr(task, "urgency_score", 0.0) or 0.0)
                deadline = float(getattr(task, "deadline_step", float("inf")))
                slack = deadline - float(step)
                distance = float(env._agent_distance_to_task(aid, task))
                return (is_emergency, urgency, slack, distance, str(task.task_id))

            task = min(tasks, key=key)
            goals[aid] = str(task.task_id)
            used.add(str(task.task_id))
        return goals


def _build_planner(
    method: str,
    cfg: EnvConfig,
    seed: int,
    use_event_trigger_override: Optional[bool] = None,
    use_risk_term_override: Optional[bool] = None,
    use_rth_repair_override: Optional[bool] = None,
    enable_rth_mask_override: Optional[bool] = None,
    encoder_type_override: str = "",
):
    def _erc_ablation_cfg(base_cfg: EnvConfig, mm: str) -> EnvConfig:
        mm = _canonical_method_name(mm)
        # Keep a legacy full profile for rollback/ablation.
        if mm == "erc_full":
            return base_cfg
        if mm == "erc_rhc_current":
            return _erc_ablation_cfg(base_cfg, "erc_rhc")
        # erc_rhc_old: keep previous ERC behavior (event + interval).
        if mm == "erc_rhc_old":
            return replace(
                base_cfg,
                # Keep hard-event execution but disable low-value churn.
                erc_ablate_low_value_refresh=True,
                erc_ablate_map_ranking_refresh=True,
                # Disable heavy TC global assignment/epoch by default.
                timecritical_global_assignment_enabled=False,
                erc_ablate_tc_global_assignment=True,
                # Keep light reservation, disable heavier cluster/cooldown behavior.
                cluster_primary_task_enabled=False,
                recent_release_cooldown_enabled=False,
                task_reservation_enabled=True,
                hrl_uav_task_reservation_enabled=True,
                # Disable hard normal-protection guard by default.
                erc_ablate_normal_protection=True,
                # Support chain/event bonus are condition-enabled in planner.
                erc_ablate_support_chain=False,
                erc_ablate_event_scoring_bonus=False,
                # OLD refresh behavior: interval + event.
                hrl_event_first_refresh_enabled=False,
                hrl_max_no_refresh_steps=5,
                hrl_event_admission_gate_enabled=False,
                hrl_noop_event_cooldown_enabled=False,
                # OLD hard-event handling (before de-noising): keep stall/soft invalid as hard-refresh eligible.
                hrl_normal_stall_hard_refresh_enabled=True,
                hrl_normal_stall_local_only=False,
                hrl_soft_invalid_hard_refresh_enabled=True,
                hrl_truck_dead_end_local_first=False,
                hrl_truck_dead_end_persist_steps=1,
                hrl_truck_dead_end_cooldown_steps=0,
                hrl_truck_dead_end_global_refresh_enabled=True,
                hrl_path_blocked_impact_gate_enabled=False,
                hrl_path_blocked_local_repair_first=False,
                hrl_path_blocked_global_refresh_enabled=True,
                hrl_uav_emergency_commit_hold_enabled=False,
                hrl_truck_routine_stuck_escape_enabled=False,
                hrl_routine_localize_eta_exit_enabled=False,
                hrl_uav_task_reservation_exec_enabled=False,
                hrl_uav_assist_enabled=False,
            )
        # erc_rhc: event-first + max-no-refresh fallback.
        if mm == "erc_rhc":
            # Default ERC-RHC strategy:
            # - keep validated old backbone;
            # - execution commitment baseline + event-evidence gate + local correction;
            # - retain airborne lock;
            # - default disable reservation/assist family.
            cfg_old = _erc_ablation_cfg(base_cfg, "erc_rhc_old")
            return replace(
                cfg_old,
                # Event updates stay enabled, but refresh follows event-first policy
                # with default (not expanded) fallback window.
                hrl_event_first_refresh_enabled=True,
                hrl_max_no_refresh_steps=5,
                hrl_event_admission_gate_enabled=True,
                # Keep hard-event de-noising and local-first correction.
                hrl_normal_stall_hard_refresh_enabled=False,
                hrl_normal_stall_local_only=True,
                hrl_soft_invalid_hard_refresh_enabled=False,
                hrl_truck_dead_end_local_first=True,
                hrl_truck_dead_end_global_refresh_enabled=False,
                hrl_path_blocked_impact_gate_enabled=True,
                hrl_path_blocked_local_repair_first=True,
                hrl_path_blocked_global_refresh_enabled=False,
                hrl_uav_emergency_commit_hold_enabled=True,
                hrl_uav_task_reservation_exec_enabled=False,
                hrl_uav_assist_enabled=False,
                hrl_uav_task_reservation_enabled=False,
                task_reservation_enabled=False,
                recent_release_cooldown_enabled=False,
            )
        if mm == "erc_base_no_event":
            return replace(
                base_cfg,
                use_event_trigger=False,
                erc_ablate_low_value_refresh=True,
                erc_ablate_map_ranking_refresh=True,
                erc_ablate_tc_global_assignment=True,
                erc_ablate_support_chain=True,
                erc_ablate_cluster_primary_reservation=True,
                erc_ablate_event_scoring_bonus=True,
                erc_ablate_normal_protection=True,
                hrl_event_first_refresh_enabled=False,
                hrl_event_admission_gate_enabled=False,
                hrl_noop_event_cooldown_enabled=False,
                hrl_normal_stall_hard_refresh_enabled=False,
                hrl_soft_invalid_hard_refresh_enabled=False,
                hrl_truck_dead_end_local_first=False,
                hrl_path_blocked_impact_gate_enabled=False,
                hrl_path_blocked_local_repair_first=False,
                hrl_uav_emergency_commit_hold_enabled=False,
                hrl_truck_routine_stuck_escape_enabled=False,
                hrl_routine_localize_eta_exit_enabled=False,
                hrl_uav_task_reservation_exec_enabled=False,
                hrl_uav_assist_enabled=False,
                hrl_uav_task_reservation_enabled=False,
                task_reservation_enabled=False,
                recent_release_cooldown_enabled=False,
                timecritical_global_assignment_enabled=False,
                cluster_primary_task_enabled=False,
                support_force_dispatch_enabled=False,
                support_force_uav_preempt_enabled=False,
                truck_force_nonnull_goal_enabled=False,
                truck_loop_break_enabled=False,
            )
        if mm == "erc_support_authorized":
            cfg_new = _erc_ablation_cfg(base_cfg, "erc_rhc")
            return replace(
                cfg_new,
                hrl_support_bound_dispatch_enabled=True,
                hrl_support_requires_timecritical_binding=True,
                hrl_support_require_actionable_gain=True,
                hrl_support_no_gain_backoff_enabled=True,
                hrl_support_no_gain_streak_threshold=2,
                hrl_support_no_gain_cooldown_steps=12,
                hrl_support_relay_reserve_enabled=False,
                hrl_uav_assist_enabled=False,
                hrl_uav_task_reservation_exec_enabled=False,
            )
        if mm == "erc_routine_commit":
            cfg_new = _erc_ablation_cfg(base_cfg, "erc_rhc")
            return replace(
                cfg_new,
                truck_routine_near_goal_support_protect_enabled=True,
                truck_routine_near_goal_support_protect_steps=5,
                hrl_routine_near_completion_eta_steps=5,
                hrl_routine_near_completion_route_dist_m=1000.0,
                hrl_routine_protection_tc_override_enabled=False,
                hrl_routine_protection_delivery_feasible_tc_override_enabled=False,
                hrl_uav_assist_enabled=False,
                hrl_uav_task_reservation_exec_enabled=False,
            )
        if mm == "erc_launch_quality_gate":
            cfg_new = _erc_ablation_cfg(base_cfg, "erc_rhc")
            return replace(
                cfg_new,
                hrl_uav_docked_require_launch_gate_strict=True,
                uav_reject_cache_window_steps=max(int(getattr(cfg_new, "uav_reject_cache_window_steps", 20)), 20),
                uav_reject_cache_min_repeat=max(int(getattr(cfg_new, "uav_reject_cache_min_repeat", 2)), 2),
                uav_reject_cache_ttl_steps=max(int(getattr(cfg_new, "uav_reject_cache_ttl_steps", 30)), 30),
                hrl_routine_protection_delivery_feasible_tc_override_enabled=True,
                hrl_tc_override_require_full_sortie_feasible=True,
                hrl_tc_override_block_if_recent_reject=True,
                hrl_tc_override_min_recovery_margin_m=max(float(getattr(cfg_new, "hrl_tc_override_min_recovery_margin_m", 300.0)), 300.0),
                hrl_tc_override_min_battery_margin_ratio=max(float(getattr(cfg_new, "hrl_tc_override_min_battery_margin_ratio", 0.12)), 0.12),
                hrl_uav_assist_enabled=False,
                hrl_uav_task_reservation_exec_enabled=False,
            )
        if mm == "erc_tc_completion_chain":
            cfg_new = _erc_ablation_cfg(base_cfg, "erc_rhc")
            return replace(
                cfg_new,
                hrl_timecritical_force_entry_enabled=True,
                hrl_timecritical_force_entry_min_gap_steps=8,
                hrl_timecritical_force_entry_shortlist_extra=3,
                hrl_timecritical_force_entry_uav_bonus=0.42,
                hrl_uav_timecritical_lifeline_weight=0.70,
                hrl_uav_timecritical_critical_bonus=0.70,
                hrl_routine_protection_delivery_feasible_tc_override_enabled=True,
                hrl_tc_override_require_full_sortie_feasible=True,
                hrl_uav_task_reservation_exec_enabled=True,
                hrl_uav_assist_enabled=True,
            )
        if mm in {"erc_support_priority", "erc_support_priority_strict", "erc_support_priority_reserve"}:
            cfg_new = _erc_ablation_cfg(base_cfg, "erc_rhc")
            common = dict(
                erc_ablate_support_chain=False,
                hrl_support_bound_dispatch_enabled=True,
                hrl_support_requires_timecritical_binding=True,
                hrl_support_require_actionable_gain=True,
                hrl_support_actionable_min_new_serviceable=0.25,
                hrl_support_actionable_post_distance_m=3200.0,
                hrl_support_relay_reserve_enabled=True,
                hrl_support_relay_min_critical_timecritical=1,
                hrl_support_critical_diversion_enabled=True,
                hrl_support_critical_diversion_max_trucks=1,
                hrl_support_critical_diversion_cover_threshold=0.45,
                hrl_support_max_trucks_when_normal_pending=1,
                hrl_support_budget_require_warning_when_normal=True,
                hrl_support_escape_hatch_enabled=True,
                hrl_support_escape_hatch_min_pending_emergency=4,
                hrl_support_escape_hatch_min_gain=0.22,
                hrl_support_escape_hatch_min_urgency=0.50,
                hrl_support_no_gain_backoff_enabled=True,
                hrl_support_no_gain_streak_threshold=3,
                hrl_support_no_gain_cooldown_steps=8,
                hrl_truck_emergency_force_relief_uav_cover_threshold=0.45,
                hrl_truck_emergency_relief_uav_cover_threshold=0.58,
                hrl_truck_no_normal_support_min_gain=0.12,
                hrl_truck_no_normal_support_urgency_floor=0.45,
                hrl_truck_support_gain_min_when_normal_reachable=0.24,
                hrl_uav_task_reservation_enabled=True,
                hrl_uav_task_reservation_exec_enabled=True,
                task_reservation_enabled=True,
                hrl_uav_assist_enabled=True,
                hrl_uav_assist_max_extra_distance_m=900.0,
                hrl_uav_assist_max_extra_ratio=0.32,
                hrl_uav_assist_min_launch_distance_reduction_m=250.0,
            )
            if mm == "erc_support_priority":
                return replace(
                    cfg_new,
                    **common,
                    hrl_truck_support_when_normal_reachable_scale=0.45,
                    hrl_support_actionable_min_gain_score=0.16,
                    hrl_support_soft_clamp_enabled=True,
                    hrl_support_soft_clamp_min_gain=0.22,
                    hrl_support_soft_clamp_long_distance_m=3400.0,
                )
            if mm == "erc_support_priority_strict":
                return replace(
                    cfg_new,
                    **common,
                    hrl_truck_support_when_normal_reachable_scale=0.30,
                    hrl_support_actionable_min_gain_score=0.20,
                    hrl_support_soft_clamp_enabled=True,
                    hrl_support_soft_clamp_min_gain=0.28,
                    hrl_support_soft_clamp_long_distance_m=2800.0,
                    truck_routine_near_goal_support_protect_enabled=True,
                    hrl_routine_protection_delivery_feasible_tc_override_enabled=True,
                    hrl_tc_override_require_full_sortie_feasible=True,
                )
            return replace(
                cfg_new,
                **common,
                hrl_truck_support_when_normal_reachable_scale=0.55,
                hrl_support_actionable_min_gain_score=0.14,
                support_force_dispatch_enabled=True,
                support_force_uav_preempt_enabled=True,
                support_force_commit_steps=12,
                hrl_relaxed_chain_commitment_steps=10,
                hrl_support_soft_clamp_enabled=False,
            )
        if mm == "erc_event_minimal_local":
            cfg_new = _erc_ablation_cfg(base_cfg, "erc_rhc")
            return replace(
                cfg_new,
                erc_ablate_low_value_refresh=True,
                erc_ablate_map_ranking_refresh=True,
                hrl_event_first_refresh_enabled=True,
                hrl_event_admission_gate_enabled=True,
                hrl_noop_event_cooldown_enabled=True,
                hrl_normal_stall_hard_refresh_enabled=False,
                hrl_normal_stall_local_only=True,
                hrl_soft_invalid_hard_refresh_enabled=False,
                hrl_truck_dead_end_local_first=True,
                hrl_truck_dead_end_global_refresh_enabled=False,
                hrl_path_blocked_impact_gate_enabled=True,
                hrl_path_blocked_local_repair_first=True,
                hrl_path_blocked_global_refresh_enabled=False,
                hrl_event_bonus_base_gain=0.30,
                hrl_event_bonus_hard_gain=0.50,
            )
        if mm == "erc_scoring_shrink":
            cfg_new = _erc_ablation_cfg(base_cfg, "erc_rhc")
            return replace(
                cfg_new,
                erc_ablate_event_scoring_bonus=True,
                hrl_event_bonus_conditional_enabled=True,
                hrl_event_bonus_base_gain=0.20,
                hrl_event_bonus_hard_gain=0.35,
                hrl_serviceable_island_bonus=0.08,
                hrl_serviceable_high_pressure_emergency_bonus=0.08,
                hrl_uav_island_delivery_bonus=0.08,
                hrl_support_bind_bonus_critical=0.20,
                hrl_support_bind_bonus_warning=0.10,
                hrl_support_bind_bonus_bulk=0.04,
                hrl_uav_assist_enabled=False,
                hrl_uav_task_reservation_exec_enabled=False,
            )
        if mm == "erc_combined_safe_core":
            cfg_new = _erc_ablation_cfg(base_cfg, "erc_rhc")
            return replace(
                cfg_new,
                hrl_support_bound_dispatch_enabled=True,
                hrl_support_requires_timecritical_binding=True,
                hrl_support_require_actionable_gain=True,
                hrl_support_no_gain_backoff_enabled=True,
                hrl_support_no_gain_streak_threshold=2,
                hrl_support_no_gain_cooldown_steps=12,
                hrl_uav_docked_require_launch_gate_strict=True,
                hrl_routine_protection_delivery_feasible_tc_override_enabled=True,
                hrl_tc_override_require_full_sortie_feasible=True,
                hrl_tc_override_block_if_recent_reject=True,
                hrl_event_first_refresh_enabled=True,
                hrl_event_admission_gate_enabled=True,
                erc_ablate_low_value_refresh=True,
                erc_ablate_map_ranking_refresh=True,
                hrl_normal_stall_hard_refresh_enabled=False,
                hrl_soft_invalid_hard_refresh_enabled=False,
                hrl_truck_dead_end_local_first=True,
                hrl_path_blocked_impact_gate_enabled=True,
                hrl_uav_emergency_commit_hold_enabled=True,
                hrl_uav_task_reservation_exec_enabled=False,
                hrl_uav_assist_enabled=False,
                timecritical_global_assignment_enabled=False,
                cluster_primary_task_enabled=False,
                task_reservation_enabled=False,
                recent_release_cooldown_enabled=False,
            )
        if mm == "erc_only_uav_commit":
            cfg_new = _erc_ablation_cfg(base_cfg, "erc_rhc")
            return replace(
                cfg_new,
                hrl_uav_emergency_commit_hold_enabled=True,
                hrl_truck_routine_stuck_escape_enabled=False,
                hrl_routine_localize_eta_exit_enabled=False,
            )
        if mm == "erc_only_truck_escape":
            cfg_new = _erc_ablation_cfg(base_cfg, "erc_rhc")
            return replace(
                cfg_new,
                hrl_uav_emergency_commit_hold_enabled=False,
                hrl_truck_routine_stuck_escape_enabled=True,
                hrl_routine_localize_eta_exit_enabled=False,
            )
        if mm == "erc_only_eta_exit":
            cfg_new = _erc_ablation_cfg(base_cfg, "erc_rhc")
            return replace(
                cfg_new,
                hrl_uav_emergency_commit_hold_enabled=False,
                hrl_truck_routine_stuck_escape_enabled=False,
                hrl_routine_localize_eta_exit_enabled=True,
            )
        if mm == "erc_routine_rescue_combo":
            cfg_new = _erc_ablation_cfg(base_cfg, "erc_rhc")
            return replace(
                cfg_new,
                hrl_uav_emergency_commit_hold_enabled=False,
                hrl_truck_routine_stuck_escape_enabled=True,
                hrl_routine_localize_eta_exit_enabled=True,
                hrl_event_admission_gate_enabled=False,
                hrl_event_first_refresh_enabled=False,
                hrl_max_no_refresh_steps=8,
            )
        if mm in {"erc_km_hard_routine", "erc_same_config", ALNS_MAINLINE_METHOD, UNIFORM_LNS_METHOD, CANONICAL_ALNS_METHOD}:
            cfg_new = _erc_ablation_cfg(base_cfg, "erc_rhc")
            large_scale = int(getattr(cfg_new, "num_nodes", 0)) >= 180
            scenario_key = str(getattr(cfg_new, "scenario", "")).upper()
            is_alns_family = bool(mm in {ALNS_MAINLINE_METHOD, UNIFORM_LNS_METHOD, CANONICAL_ALNS_METHOD})
            tc_global_enabled = bool((not large_scale) or scenario_key == "C" or is_alns_family)
            return replace(
                cfg_new,
                # TC triage + support-required chain:
                # keep direct-safe launches strict, but reserve truck-UAV-task
                # chains when a time-critical task needs a moving recovery anchor.
                timecritical_global_assignment_enabled=tc_global_enabled,
                erc_ablate_tc_global_assignment=not tc_global_enabled,
                hrl_tc_global_assignment_adaptive_escape_enabled=False,
                hrl_tc_global_assignment_escape_min_map_size_m=999999.0,
                hrl_tc_global_assignment_escape_low_cover_threshold=0.35,
                hrl_tc_global_assignment_escape_max_lifeline_ratio=0.55,
                erc_tc_support_required_enabled=bool(large_scale or scenario_key == "C"),
                erc_tc_support_lock_steps=22,
                erc_tc_support_max_active_chains=3,
                erc_tc_support_latest_start_margin_steps=3,
                erc_tc_support_max_setup_steps=120,
                erc_tc_support_min_gain_score=0.08,
                erc_tc_support_post_distance_m=(
                    3000.0 if bool(large_scale) and scenario_key == "C"
                    else 3800.0
                ),
                erc_tc_support_high_urgency_post_distance_m=0.0,
                erc_tc_support_high_urgency_threshold=0.88,
                erc_tc_support_max_lifeline_ratio=1.0,
                erc_tc_support_require_follower_uav=not (scenario_key == "C" and not bool(large_scale)),
                erc_tc_support_allow_normal_preemption=True,
                erc_tc_support_anchor_waypoint_enabled=True,
                erc_tc_support_anchor_search_node_cap=320,
                erc_tc_support_anchor_search_min_balance=1.01,
                erc_tc_support_release_uav_for_direct_tc_enabled=True,
                erc_tc_uncovered_support_repair_enabled=bool(
                    scenario_key == "C" or (is_alns_family and bool(large_scale) and scenario_key == "B")
                ),
                erc_tc_uncovered_support_repair_min_step=(
                    70 if bool(large_scale) and scenario_key == "B" else (18 if bool(large_scale) else 10)
                ),
                erc_tc_uncovered_support_repair_min_gap_steps=(
                    26 if bool(large_scale) and scenario_key == "B" else (10 if bool(large_scale) else 5)
                ),
                erc_tc_uncovered_support_repair_max_lifeline_ratio=(
                    0.78 if bool(large_scale) and scenario_key == "B" else (0.84 if bool(large_scale) else 0.92)
                ),
                erc_tc_uncovered_support_repair_cover_threshold=(
                    0.35 if bool(large_scale) and scenario_key == "B" else (0.42 if bool(large_scale) else 0.55)
                ),
                erc_tc_uncovered_support_repair_min_nearest_truck_m=(
                    7200.0 if bool(large_scale) and scenario_key == "B" else (5200.0 if bool(large_scale) else 1800.0)
                ),
                erc_tc_uncovered_support_repair_min_urgency=(
                    0.0
                ),
                erc_tc_stale_assigned_support_repair_enabled=False,
                erc_tc_stale_assigned_support_repair_min_exposure=28,
                erc_tc_stale_assigned_support_repair_min_step=80,
                erc_tc_stale_assigned_support_repair_max_lifeline_ratio=0.75,
                erc_tc_stale_assigned_support_repair_min_nearest_truck_m=5200.0,
                erc_tc_coverage_intent_enabled=False,
                erc_tc_coverage_intent_min_step=70,
                erc_tc_coverage_intent_max_step=170,
                erc_tc_coverage_intent_max_support_deliveries=4,
                erc_tc_coverage_intent_min_pending=4,
                erc_tc_coverage_intent_cover_threshold=0.55,
                erc_tc_coverage_intent_max_lifeline_ratio=0.94,
                erc_tc_coverage_intent_min_gap_steps=4,
                erc_tc_coverage_intent_max_per_step=1,
                erc_large_map_greedy_tc_fallback_enabled=True,
                alns_enabled=is_alns_family,
                alns_adaptive_horizon_enabled=not bool(large_scale),
                alns_risk_pressure_enabled=True,
                alns_ghost_tasks_enabled=True,
                alns_iterations=(
                    int(getattr(cfg, "diagnostic_alns_iterations", 0))
                    if int(getattr(cfg, "diagnostic_alns_iterations", 0)) > 0
                    else 4
                ) if is_alns_family else 0,
                alns_min_replan_interval_steps=(
                    int(getattr(cfg, "diagnostic_alns_min_replan_interval_steps", 0))
                    if int(getattr(cfg, "diagnostic_alns_min_replan_interval_steps", 0)) > 0
                    else 3
                ),
                alns_max_replan_interval_steps=(12 if bool(large_scale) else 9),
                alns_min_horizon_steps=(24 if bool(large_scale) else 18),
                alns_max_horizon_steps=(96 if bool(large_scale) else 64),
                alns_destroy_max_assignments=(2 if bool(large_scale) else 2),
                alns_accept_temperature=0.04,
                alns_safe_overlay_enabled=True,
                alns_destroy_existing_enabled=bool(large_scale),
                alns_protect_recent_goal_steps=(12 if bool(large_scale) else 8),
                alns_protect_progress_epsilon_m=20.0,
                alns_stale_goal_steps=(32 if bool(large_scale) else 20),
                erc_tc_support_b_dynamic_second_chain_enabled=False,
                erc_tc_support_b_second_chain_min_step=70,
                erc_tc_support_b_second_chain_max_deliveries=4,
                # In large-map B, keep the same bounded ownership repair that
                # C uses.  It only touches a pending routine task after a
                # long exposure window, and therefore cannot preempt a task
                # already in service.  This is the smallest transferable part
                # of C's anti-starvation behavior; normal route commitments
                # and emergency contracts remain unchanged.
                erc_stalled_routine_ownership_repair_enabled=bool(
                    bool(large_scale) and scenario_key in {"B", "C"}
                ),
                # The large-map B profile may exhaust a truck's emergency
                # packages before its remaining contract reaches the route
                # head. Release only that stockout contract for a stocked
                # unit; keep other scenarios on fixed-owner behavior.
                hrl_route_plan_stockout_transfer_enabled=bool(
                    bool(large_scale) and scenario_key == "B"
                ),
                erc_stalled_routine_ownership_min_step=36,
                erc_stalled_routine_ownership_exposure_steps=42,
                erc_stalled_routine_ownership_max_repairs_per_step=1,
                # If a reachable routine task is still completely unassigned
                # on a large B map, give it one nearest idle truck after the
                # same delayed gate used by the C repair.  No active contract
                # is stolen by this branch and max_repairs remains one.
                erc_unassigned_routine_repair_enabled=bool(
                    ((not bool(large_scale)) and scenario_key == "C")
                    or (bool(large_scale) and scenario_key == "B")
                ),
                erc_unassigned_routine_repair_min_step=60,
                erc_unassigned_routine_repair_max_per_step=1,
                erc_last_routine_rescue_pending_threshold=1,
                erc_last_routine_rescue_min_completed_tc=7,
                uav_terminal_failure_battery_floor=0.04,
                uav_low_soc_failure_risk_scale=0.0,
                routine_bulk_lifeline_decay_base=0.055,
                time_critical_lightweight_lifeline_decay_base=0.16,
                task_lifeline_hazard_weight=0.25,
                unload_rounds_normal=int(getattr(cfg_new, "unload_rounds_normal", 5)),
                routine_bulk_demand_kg_min=float(getattr(cfg_new, "routine_bulk_demand_kg_min", 200.0)),
                routine_bulk_demand_kg_max=float(getattr(cfg_new, "routine_bulk_demand_kg_max", 300.0)),
                # Large-map spatial decomposition: assign truck-UAV teams to
                # task regions and suppress avoidable cross-region interference.
                region_commitment_enabled=bool(scenario_key != "B"),
                region_commitment_min_map_size_m=9000.0,
                region_commitment_count=0,
                region_commitment_auto_select_enabled=True,
                region_commitment_auto_max_k=3,
                region_commitment_enable_score_threshold=0.18,
                region_commitment_min_separation_score=0.18,
                region_commitment_min_load_balance_score=0.20,
                region_commitment_unbalanced_min_separation_score=0.75,
                region_commitment_unbalanced_min_outlier_tasks=3,
                region_commitment_overpartition_penalty=0.20,
                region_commitment_strength_min=0.30,
                region_commitment_cross_region_filter_enabled=False,
                region_commitment_local_bonus=0.24,
                region_commitment_cross_region_penalty=0.45,
                region_commitment_override_lifeline_ratio=0.45,
                region_commitment_keep_current_goal_enabled=True,
                region_commitment_include_gateways=False,
                region_commitment_outlier_gate_enabled=True,
                region_commitment_outlier_distance_ratio=0.20,
                region_commitment_outlier_min_distance_m=3600.0,
                region_commitment_outlier_penalty=0.70,
                region_commitment_outlier_override_lifeline_ratio=0.34,
                region_commitment_outlier_require_support_gain=True,
                region_commitment_outlier_min_support_gain=0.10,
                region_commitment_routine_guard_enabled=False,
                region_commitment_routine_guard_lifeline_ratio=0.42,
                region_commitment_routine_guard_penalty=0.28,
                region_commitment_routine_guard_support_gain_relief=0.60,
                region_commitment_routine_guard_max_normal_dist_m=1200.0,
                erc_routine_progress_watchdog_enabled=True,
                erc_routine_watchdog_full_time_support_enabled=True,
                # Physical inventory is an environment property, never a
                # method-specific advantage. The historical 1800 kg M-B boost
                # is deliberately retired from active construction.
                truck_initial_bulk_inventory_kg=float(
                    getattr(cfg_new, "truck_initial_bulk_inventory_kg", 1200.0)
                ),
                # Routine-first backbone with bounded TC rescue support.
                erc_ablate_normal_protection=False,
                hrl_uav_emergency_commit_hold_enabled=True,
                hrl_uav_anchor_to_tc_requires_actionable_enabled=bool(bool(large_scale) or scenario_key != "C"),
                hrl_docked_uav_soft_invalid_hold_enabled=True,
                truck_support_uav_recovery_enabled=True,
                truck_recovery_require_request_when_normal_pending=True,
                truck_recovery_request_min_urgency_when_normal_pending=(0.98 if bool(large_scale) else 0.95),
                uav_forced_recovery_bind_latency_steps=0,
                uav_allow_rendezvous_launch=bool(is_alns_family and bool(large_scale) and scenario_key == "C"),
                uav_rendezvous_launch_requires_docked_truck_goal=False,
                hrl_truck_support_when_normal_reachable_scale=(0.06 if bool(large_scale) else 0.12),
                hrl_truck_emergency_min_pending_normal_to_block=1,
                hrl_truck_emergency_force_relief_urgency_threshold=(0.88 if bool(large_scale) else 0.72),
                hrl_truck_emergency_force_relief_uav_cover_threshold=(0.18 if bool(large_scale) else 0.35),
                hrl_truck_emergency_cover_threshold_when_normal_reachable=(0.16 if bool(large_scale) else 0.30),
                hrl_truck_support_gain_min_when_normal_reachable=(0.65 if bool(large_scale) else 0.45),
                hrl_truck_emergency_support_when_no_normal_enabled=True,
                hrl_truck_no_normal_support_min_gain=0.18,
                hrl_support_bound_dispatch_enabled=True,
                hrl_support_relay_reserve_enabled=True,
                hrl_support_relay_min_critical_timecritical=1,
                support_force_dispatch_enabled=False,
                support_force_uav_preempt_enabled=False,
                hrl_support_critical_diversion_enabled=True,
                hrl_support_critical_diversion_max_trucks=(2 if bool(large_scale) else 1),
                hrl_support_critical_diversion_max_map_size_m=20000.0,
                hrl_support_critical_diversion_cover_threshold=0.35,
                hrl_support_no_gain_backoff_enabled=True,
                hrl_support_no_gain_streak_threshold=3,
                hrl_support_no_gain_cooldown_steps=8,
                hrl_support_require_actionable_gain=True,
                hrl_support_actionable_min_gain_score=0.16,
                hrl_support_actionable_post_distance_m=3600.0,
                hrl_support_escape_hatch_enabled=True,
                hrl_support_escape_hatch_min_pending_emergency=3,
                hrl_support_escape_hatch_min_gain=0.16,
                hrl_support_escape_hatch_min_urgency=0.45,
                hrl_support_chain_max_direct_ready_timecritical=12,
                hrl_support_chain_min_gain_for_enable=0.0,
                hrl_support_chain_critical_escape_enabled=True,
                hrl_support_chain_critical_escape_scenarios=(
                    "ALL" if (is_alns_family and bool(large_scale) and scenario_key == "B") else "C"
                ),
                hrl_support_chain_critical_escape_max_lifeline_ratio=(0.72 if bool(large_scale) else 0.55),
                hrl_support_chain_critical_escape_low_cover_threshold=(0.50 if bool(large_scale) else 0.35),
                hrl_support_chain_critical_escape_min_gain=0.0,
                hrl_timecritical_force_entry_enabled=True,
                hrl_timecritical_force_entry_min_map_size_m=12000.0,
                hrl_timecritical_force_entry_max_lifeline_ratio=0.90,
                hrl_timecritical_force_entry_min_gap_steps=8,
                hrl_timecritical_force_entry_shortlist_extra=3,
                hrl_timecritical_force_entry_uav_bonus=0.42,
                hrl_airborne_tc_completion_grace_enabled=True,
                hrl_airborne_tc_completion_grace_radius_m=1100.0,
                hrl_airborne_tc_completion_grace_recovery_buffer_scale=0.35,
                hrl_airborne_tc_completion_grace_min_battery=0.16,
                hrl_airborne_tc_completion_grace_min_lifeline_steps=4,
                hrl_timecritical_far_exposure_enabled=False,
                hrl_timecritical_far_exposure_min_map_size_m=12000.0,
                hrl_timecritical_far_exposure_extra=1,
                hrl_timecritical_far_exposure_max_lifeline_ratio=0.55,
                hrl_timecritical_far_exposure_min_gap_steps=10,
                hrl_timecritical_far_exposure_low_cover_threshold=0.30,
                hrl_timecritical_far_exposure_urgent_bypass_threshold=0.88,
                hrl_uav_timecritical_lifeline_weight=0.85,
                hrl_uav_timecritical_critical_bonus=0.80,
                uav_reject_cache_window_steps=(5 if scenario_key == "C" else (10 if bool(large_scale) else 8)),
                uav_launch_min_horizon_buffer_steps=(2 if bool(large_scale) else 2),
                uav_high_pressure_recovery_margin_bonus_m=(90.0 if bool(large_scale) else 90.0),
                hrl_tc_override_block_if_recent_reject=False if scenario_key == "C" else (True if bool(large_scale) else False),
                hrl_tc_override_min_recovery_margin_m=(100.0 if bool(large_scale) else 120.0),
                hrl_tc_override_min_battery_margin_ratio=(0.08 if bool(large_scale) else 0.09),
                uav_initial_distinct_emergency_assign=True,
                uav_initial_distinct_window_steps=6,
                # Keep routine watchdogs on.
                hrl_truck_routine_stuck_escape_enabled=True,
                hrl_truck_routine_escape_allow_any_alt_when_stuck=(
                    str(getattr(cfg_new, "scenario", "")).upper() in {"B", "C"}
                ),
                hrl_truck_routine_escape_allow_any_alt_min_step=(
                    0 if str(getattr(cfg_new, "scenario", "")).upper() == "C" else 80
                ),
                hrl_far_routine_bootstrap_enabled=True,
                hrl_far_routine_bootstrap_window_steps=20,
                hrl_far_routine_bootstrap_min_map_size_m=9000.0,
                hrl_far_routine_bootstrap_min_distance_m=7000.0,
                hrl_truck_routine_stuck_persist_steps=3,
                hrl_truck_routine_progress_epsilon_m=20.0,
                hrl_truck_routine_escape_min_eta_gain_steps=0,
                hrl_truck_routine_escape_min_score_gain=-0.03,
                hrl_routine_localize_eta_exit_enabled=True,
                hrl_truck_normal_to_normal_switch_min_improve_ratio=0.00,
                hrl_truck_normal_to_normal_switch_min_score_gain=-0.02,
                # Lower event churn, keep local correction.
                hrl_event_first_refresh_enabled=False,
                hrl_event_admission_gate_enabled=(str(getattr(cfg_new, "scenario", "")).upper() == "C"),
                hrl_noop_event_cooldown_enabled=(str(getattr(cfg_new, "scenario", "")).upper() == "C"),
                hrl_noop_event_cooldown_steps=12,
                hrl_event_replan_budget_per_window=(1 if str(getattr(cfg_new, "scenario", "")).upper() == "C" else 2),
                hrl_max_no_refresh_steps=(10 if bool(large_scale) else 7),
                hrl_normal_stall_hard_refresh_enabled=False,
                hrl_normal_stall_local_only=True,
                hrl_path_blocked_impact_gate_enabled=True,
                hrl_path_blocked_local_repair_first=True,
                hrl_path_blocked_global_refresh_enabled=False,
                hrl_truck_dead_end_local_first=True,
                hrl_truck_dead_end_global_refresh_enabled=False,
                # Keep heavy reservation/assist execution off; routine progress
                # is handled by the truck watchdog and localized escape above.
                hrl_uav_task_reservation_exec_enabled=False,
                hrl_uav_assist_enabled=False,
                hrl_uav_task_reservation_enabled=True,
                task_reservation_enabled=True,
                recent_release_cooldown_enabled=True,
            )
        if mm == "erc_km_hard_routine_v2":
            cfg_new = _erc_ablation_cfg(base_cfg, "erc_rhc")
            return replace(
                cfg_new,
                # Keep routine-first posture.
                truck_support_uav_recovery_enabled=False,
                hrl_truck_support_when_normal_reachable_scale=0.15,
                hrl_truck_emergency_support_when_no_normal_enabled=True,
                hrl_truck_no_normal_support_min_gain=0.35,
                hrl_support_require_actionable_gain=True,
                hrl_support_actionable_min_gain_score=0.35,
                hrl_support_bound_dispatch_enabled=True,
                hrl_support_relay_reserve_enabled=False,
                hrl_support_critical_diversion_enabled=False,
                hrl_support_no_gain_backoff_enabled=True,
                hrl_support_no_gain_streak_threshold=1,
                hrl_support_no_gain_cooldown_steps=24,
                # Restore a bit of UAV TC continuity.
                hrl_uav_emergency_commit_hold_enabled=True,
                hrl_docked_uav_soft_invalid_hold_enabled=True,
                # Keep routine watchdogs on.
                hrl_truck_routine_stuck_escape_enabled=True,
                hrl_routine_localize_eta_exit_enabled=True,
                hrl_truck_normal_to_normal_switch_min_improve_ratio=0.05,
                hrl_truck_normal_to_normal_switch_min_score_gain=0.02,
                # Lower event churn, keep local correction.
                hrl_event_first_refresh_enabled=False,
                hrl_event_admission_gate_enabled=False,
                hrl_max_no_refresh_steps=10,
                hrl_normal_stall_hard_refresh_enabled=False,
                hrl_normal_stall_local_only=True,
                hrl_path_blocked_impact_gate_enabled=True,
                hrl_path_blocked_local_repair_first=True,
                hrl_path_blocked_global_refresh_enabled=False,
                hrl_truck_dead_end_local_first=True,
                hrl_truck_dead_end_global_refresh_enabled=False,
                # Disable reservation/assist family to avoid extra coupling.
                hrl_uav_task_reservation_exec_enabled=False,
                hrl_uav_assist_enabled=False,
                hrl_uav_task_reservation_enabled=False,
                task_reservation_enabled=False,
                recent_release_cooldown_enabled=False,
            )
        if mm == "erc_ya_balanced":
            cfg_new = _erc_ablation_cfg(base_cfg, "erc_rhc")
            return replace(
                cfg_new,
                # Keep routine stabilization from hard-routine line.
                hrl_truck_routine_stuck_escape_enabled=True,
                hrl_routine_localize_eta_exit_enabled=True,
                hrl_truck_normal_to_normal_switch_min_improve_ratio=0.05,
                hrl_truck_normal_to_normal_switch_min_score_gain=0.02,
                # Recover TC ability compared with hard-routine.
                hrl_uav_emergency_commit_hold_enabled=True,
                truck_support_uav_recovery_enabled=True,
                hrl_truck_support_when_normal_reachable_scale=0.20,
                hrl_truck_emergency_support_when_no_normal_enabled=True,
                hrl_support_bound_dispatch_enabled=True,
                hrl_support_relay_reserve_enabled=False,
                hrl_support_critical_diversion_enabled=False,
                hrl_support_no_gain_backoff_enabled=True,
                hrl_support_no_gain_streak_threshold=1,
                hrl_support_no_gain_cooldown_steps=20,
                hrl_support_require_actionable_gain=True,
                hrl_support_actionable_min_gain_score=0.30,
                # Keep event churn moderate.
                hrl_event_first_refresh_enabled=False,
                hrl_event_admission_gate_enabled=False,
                hrl_max_no_refresh_steps=8,
                hrl_normal_stall_hard_refresh_enabled=False,
                hrl_normal_stall_local_only=True,
                hrl_path_blocked_impact_gate_enabled=True,
                hrl_path_blocked_local_repair_first=True,
                hrl_path_blocked_global_refresh_enabled=False,
                hrl_truck_dead_end_local_first=True,
                hrl_truck_dead_end_global_refresh_enabled=False,
                # Avoid reservation/assist side effects.
                hrl_uav_task_reservation_exec_enabled=False,
                hrl_uav_assist_enabled=False,
                hrl_uav_task_reservation_enabled=False,
                task_reservation_enabled=False,
                recent_release_cooldown_enabled=False,
            )
        if mm == "erc_mc_lc_boost":
            cfg_new = _erc_ablation_cfg(base_cfg, "erc_rhc")
            return replace(
                cfg_new,
                # Aggressive TC conversion for MC/LC.
                hrl_uav_emergency_commit_hold_enabled=True,
                hrl_support_bound_dispatch_enabled=True,
                hrl_support_relay_reserve_enabled=True,
                hrl_support_critical_diversion_enabled=True,
                hrl_support_critical_diversion_max_trucks=2,
                hrl_support_require_actionable_gain=True,
                hrl_support_actionable_min_gain_score=0.18,
                hrl_timecritical_force_entry_enabled=True,
                hrl_timecritical_force_entry_min_gap_steps=6,
                hrl_timecritical_force_entry_shortlist_extra=4,
                hrl_timecritical_force_entry_uav_bonus=0.55,
                hrl_uav_timecritical_lifeline_weight=0.80,
                hrl_uav_timecritical_critical_bonus=0.80,
                # Keep launch safety but reduce stale loops.
                hrl_uav_docked_require_launch_gate_strict=True,
                uav_reject_cache_window_steps=max(int(getattr(cfg_new, "uav_reject_cache_window_steps", 20)), 24),
                uav_reject_cache_ttl_steps=max(int(getattr(cfg_new, "uav_reject_cache_ttl_steps", 30)), 36),
                # Faster refresh cadence for compact MC/LC maps.
                hrl_event_first_refresh_enabled=True,
                hrl_event_admission_gate_enabled=True,
                hrl_max_no_refresh_steps=4,
                hrl_noop_event_cooldown_enabled=True,
                # Keep some routine protection to avoid collapse.
                hrl_truck_routine_stuck_escape_enabled=True,
                hrl_routine_localize_eta_exit_enabled=True,
                hrl_truck_normal_to_normal_switch_min_improve_ratio=0.08,
                hrl_truck_normal_to_normal_switch_min_score_gain=0.04,
                # Keep reservation/assist off for cleaner behavior.
                hrl_uav_task_reservation_exec_enabled=False,
                hrl_uav_assist_enabled=False,
                hrl_uav_task_reservation_enabled=False,
                task_reservation_enabled=False,
                recent_release_cooldown_enabled=False,
            )
        if mm == "erc_ya_km_bridge":
            cfg_new = _erc_ablation_cfg(base_cfg, "erc_rhc")
            return replace(
                cfg_new,
                # Keep routine backbone from KM-hard line.
                hrl_truck_routine_stuck_escape_enabled=True,
                hrl_routine_localize_eta_exit_enabled=True,
                hrl_truck_normal_to_normal_switch_min_improve_ratio=0.05,
                hrl_truck_normal_to_normal_switch_min_score_gain=0.02,
                # Restore limited TC continuity and bounded recovery support.
                hrl_uav_emergency_commit_hold_enabled=True,
                truck_support_uav_recovery_enabled=True,
                hrl_truck_support_when_normal_reachable_scale=0.12,
                hrl_truck_emergency_support_when_no_normal_enabled=True,
                hrl_support_require_actionable_gain=True,
                hrl_support_actionable_min_gain_score=0.35,
                hrl_support_bound_dispatch_enabled=True,
                hrl_support_relay_reserve_enabled=False,
                hrl_support_critical_diversion_enabled=False,
                hrl_support_no_gain_backoff_enabled=True,
                hrl_support_no_gain_streak_threshold=1,
                hrl_support_no_gain_cooldown_steps=24,
                # Moderate event refresh to reduce stale deadlocks without over-churn.
                hrl_event_first_refresh_enabled=False,
                hrl_event_admission_gate_enabled=False,
                hrl_max_no_refresh_steps=8,
                hrl_normal_stall_hard_refresh_enabled=False,
                hrl_normal_stall_local_only=True,
                hrl_path_blocked_impact_gate_enabled=True,
                hrl_path_blocked_local_repair_first=True,
                hrl_path_blocked_global_refresh_enabled=False,
                hrl_truck_dead_end_local_first=True,
                hrl_truck_dead_end_global_refresh_enabled=False,
                # Keep reservation/assist family off for stability.
                hrl_uav_task_reservation_exec_enabled=False,
                hrl_uav_assist_enabled=False,
                hrl_uav_task_reservation_enabled=False,
                task_reservation_enabled=False,
                recent_release_cooldown_enabled=False,
            )
        if mm == "erc_mc_lc_safe":
            cfg_new = _erc_ablation_cfg(base_cfg, "erc_rhc")
            return replace(
                cfg_new,
                # Conservative TC lift over KM-hard baseline.
                hrl_uav_emergency_commit_hold_enabled=True,
                truck_support_uav_recovery_enabled=True,
                hrl_truck_support_when_normal_reachable_scale=0.10,
                hrl_truck_emergency_support_when_no_normal_enabled=True,
                hrl_support_require_actionable_gain=True,
                hrl_support_actionable_min_gain_score=0.32,
                hrl_support_bound_dispatch_enabled=True,
                hrl_support_relay_reserve_enabled=False,
                hrl_support_critical_diversion_enabled=False,
                hrl_support_no_gain_backoff_enabled=True,
                hrl_support_no_gain_streak_threshold=1,
                hrl_support_no_gain_cooldown_steps=20,
                # Slightly faster refresh than KM-hard for compact maps.
                hrl_event_first_refresh_enabled=True,
                hrl_event_admission_gate_enabled=False,
                hrl_max_no_refresh_steps=7,
                hrl_noop_event_cooldown_enabled=True,
                # Preserve routine protection to avoid MC collapse.
                hrl_truck_routine_stuck_escape_enabled=True,
                hrl_routine_localize_eta_exit_enabled=True,
                hrl_truck_normal_to_normal_switch_min_improve_ratio=0.05,
                hrl_truck_normal_to_normal_switch_min_score_gain=0.02,
                hrl_normal_stall_hard_refresh_enabled=False,
                hrl_normal_stall_local_only=True,
                hrl_path_blocked_impact_gate_enabled=True,
                hrl_path_blocked_local_repair_first=True,
                hrl_path_blocked_global_refresh_enabled=False,
                hrl_truck_dead_end_local_first=True,
                hrl_truck_dead_end_global_refresh_enabled=False,
                # Keep reservation/assist off.
                hrl_uav_task_reservation_exec_enabled=False,
                hrl_uav_assist_enabled=False,
                hrl_uav_task_reservation_enabled=False,
                task_reservation_enabled=False,
                recent_release_cooldown_enabled=False,
            )
        if mm == "erc_ya_tune_a":
            cfg_new = _erc_ablation_cfg(base_cfg, "erc_rhc")
            return replace(
                cfg_new,
                # YA-oriented: add bounded support + tighter actionable gate.
                hrl_uav_emergency_commit_hold_enabled=False,
                truck_support_uav_recovery_enabled=True,
                hrl_truck_support_when_normal_reachable_scale=0.08,
                hrl_truck_emergency_support_when_no_normal_enabled=True,
                hrl_support_require_actionable_gain=True,
                hrl_support_actionable_min_gain_score=0.38,
                hrl_support_bound_dispatch_enabled=True,
                hrl_support_relay_reserve_enabled=False,
                hrl_support_critical_diversion_enabled=False,
                hrl_support_no_gain_backoff_enabled=True,
                hrl_support_no_gain_streak_threshold=1,
                hrl_support_no_gain_cooldown_steps=24,
                hrl_truck_routine_stuck_escape_enabled=True,
                hrl_routine_localize_eta_exit_enabled=True,
                hrl_truck_normal_to_normal_switch_min_improve_ratio=0.05,
                hrl_truck_normal_to_normal_switch_min_score_gain=0.02,
                hrl_event_first_refresh_enabled=False,
                hrl_event_admission_gate_enabled=False,
                hrl_max_no_refresh_steps=9,
                hrl_normal_stall_hard_refresh_enabled=False,
                hrl_normal_stall_local_only=True,
                hrl_path_blocked_impact_gate_enabled=True,
                hrl_path_blocked_local_repair_first=True,
                hrl_path_blocked_global_refresh_enabled=False,
                hrl_truck_dead_end_local_first=True,
                hrl_truck_dead_end_global_refresh_enabled=False,
                hrl_uav_task_reservation_exec_enabled=False,
                hrl_uav_assist_enabled=False,
                hrl_uav_task_reservation_enabled=False,
                task_reservation_enabled=False,
                recent_release_cooldown_enabled=False,
            )
        if mm == "erc_ya_tune_b":
            cfg_new = _erc_ablation_cfg(base_cfg, "erc_rhc")
            return replace(
                cfg_new,
                # YA-oriented: mild commit hold + moderate support.
                hrl_uav_emergency_commit_hold_enabled=True,
                truck_support_uav_recovery_enabled=True,
                hrl_truck_support_when_normal_reachable_scale=0.10,
                hrl_truck_emergency_support_when_no_normal_enabled=True,
                hrl_support_require_actionable_gain=True,
                hrl_support_actionable_min_gain_score=0.40,
                hrl_support_bound_dispatch_enabled=True,
                hrl_support_relay_reserve_enabled=False,
                hrl_support_critical_diversion_enabled=False,
                hrl_support_no_gain_backoff_enabled=True,
                hrl_support_no_gain_streak_threshold=1,
                hrl_support_no_gain_cooldown_steps=24,
                hrl_truck_routine_stuck_escape_enabled=True,
                hrl_routine_localize_eta_exit_enabled=True,
                hrl_truck_normal_to_normal_switch_min_improve_ratio=0.05,
                hrl_truck_normal_to_normal_switch_min_score_gain=0.02,
                hrl_event_first_refresh_enabled=False,
                hrl_event_admission_gate_enabled=False,
                hrl_max_no_refresh_steps=8,
                hrl_normal_stall_hard_refresh_enabled=False,
                hrl_normal_stall_local_only=True,
                hrl_path_blocked_impact_gate_enabled=True,
                hrl_path_blocked_local_repair_first=True,
                hrl_path_blocked_global_refresh_enabled=False,
                hrl_truck_dead_end_local_first=True,
                hrl_truck_dead_end_global_refresh_enabled=False,
                hrl_uav_task_reservation_exec_enabled=False,
                hrl_uav_assist_enabled=False,
                hrl_uav_task_reservation_enabled=False,
                task_reservation_enabled=False,
                recent_release_cooldown_enabled=False,
            )
        if mm == "erc_ya_tune_c":
            cfg_new = _erc_ablation_cfg(base_cfg, "erc_rhc")
            return replace(
                cfg_new,
                # YA-oriented: no commit hold, slightly more refresh.
                hrl_uav_emergency_commit_hold_enabled=False,
                truck_support_uav_recovery_enabled=True,
                hrl_truck_support_when_normal_reachable_scale=0.12,
                hrl_truck_emergency_support_when_no_normal_enabled=True,
                hrl_support_require_actionable_gain=True,
                hrl_support_actionable_min_gain_score=0.36,
                hrl_support_bound_dispatch_enabled=True,
                hrl_support_relay_reserve_enabled=False,
                hrl_support_critical_diversion_enabled=False,
                hrl_support_no_gain_backoff_enabled=True,
                hrl_support_no_gain_streak_threshold=1,
                hrl_support_no_gain_cooldown_steps=22,
                hrl_truck_routine_stuck_escape_enabled=True,
                hrl_routine_localize_eta_exit_enabled=True,
                hrl_truck_normal_to_normal_switch_min_improve_ratio=0.05,
                hrl_truck_normal_to_normal_switch_min_score_gain=0.02,
                hrl_event_first_refresh_enabled=True,
                hrl_event_admission_gate_enabled=False,
                hrl_max_no_refresh_steps=7,
                hrl_noop_event_cooldown_enabled=True,
                hrl_normal_stall_hard_refresh_enabled=False,
                hrl_normal_stall_local_only=True,
                hrl_path_blocked_impact_gate_enabled=True,
                hrl_path_blocked_local_repair_first=True,
                hrl_path_blocked_global_refresh_enabled=False,
                hrl_truck_dead_end_local_first=True,
                hrl_truck_dead_end_global_refresh_enabled=False,
                hrl_uav_task_reservation_exec_enabled=False,
                hrl_uav_assist_enabled=False,
                hrl_uav_task_reservation_enabled=False,
                task_reservation_enabled=False,
                recent_release_cooldown_enabled=False,
            )
        if mm in {"erc_support_recovery_launch", "erc_support_recovery_launch_relaxed"}:
            cfg_new = _erc_ablation_cfg(base_cfg, "erc_rhc")
            relaxed = bool(mm == "erc_support_recovery_launch_relaxed")
            return replace(
                cfg_new,
                # Preserve the routine-first backbone while restoring a bounded
                # truck-to-UAV recovery chain.
                hrl_uav_emergency_commit_hold_enabled=True,
                hrl_docked_uav_soft_invalid_hold_enabled=True,
                truck_support_uav_recovery_enabled=True,
                uav_allow_rendezvous_launch=True,
                uav_recovery_distance_buffer_m=(420.0 if relaxed else 500.0),
                uav_reject_cache_window_steps=(10 if relaxed else 14),
                uav_reject_cache_min_repeat=3,
                uav_reject_cache_ttl_steps=(10 if relaxed else 16),
                hrl_truck_support_when_normal_reachable_scale=(0.08 if relaxed else 0.04),
                hrl_truck_emergency_support_when_no_normal_enabled=True,
                hrl_support_require_actionable_gain=True,
                hrl_support_actionable_min_gain_score=(0.32 if relaxed else 0.38),
                hrl_support_bound_dispatch_enabled=True,
                hrl_support_relay_reserve_enabled=False,
                hrl_support_critical_diversion_enabled=False,
                hrl_support_no_gain_backoff_enabled=True,
                hrl_support_no_gain_streak_threshold=1,
                hrl_support_no_gain_cooldown_steps=20,
                support_force_dispatch_enabled=True,
                support_force_commit_steps=8,
                support_force_uav_preempt_enabled=False,
                # Keep routine protection and localized event correction.
                hrl_truck_routine_stuck_escape_enabled=True,
                hrl_routine_localize_eta_exit_enabled=True,
                hrl_truck_normal_to_normal_switch_min_improve_ratio=0.05,
                hrl_truck_normal_to_normal_switch_min_score_gain=0.02,
                hrl_event_first_refresh_enabled=False,
                hrl_event_admission_gate_enabled=False,
                hrl_max_no_refresh_steps=8,
                hrl_normal_stall_hard_refresh_enabled=False,
                hrl_normal_stall_local_only=True,
                hrl_path_blocked_impact_gate_enabled=True,
                hrl_path_blocked_local_repair_first=True,
                hrl_path_blocked_global_refresh_enabled=False,
                hrl_truck_dead_end_local_first=True,
                hrl_truck_dead_end_global_refresh_enabled=False,
                # Keep reservation/assist coupling off; the support chain is the
                # only mechanism allowed to relax launch conservatism.
                hrl_uav_task_reservation_exec_enabled=False,
                hrl_uav_assist_enabled=False,
                hrl_uav_task_reservation_enabled=False,
                task_reservation_enabled=False,
                recent_release_cooldown_enabled=False,
            )
        if mm == "erc_support_recovery_bound":
            cfg_new = _erc_ablation_cfg(base_cfg, "erc_rhc")
            return replace(
                cfg_new,
                # Bound-only rendezvous: UAV can launch on recovery margin only
                # when its docked truck is already committed to that task.
                hrl_uav_emergency_commit_hold_enabled=True,
                hrl_docked_uav_soft_invalid_hold_enabled=True,
                truck_support_uav_recovery_enabled=True,
                uav_allow_rendezvous_launch=True,
                uav_rendezvous_launch_requires_docked_truck_goal=True,
                uav_recovery_distance_buffer_m=500.0,
                uav_reject_cache_window_steps=12,
                uav_reject_cache_min_repeat=3,
                uav_reject_cache_ttl_steps=12,
                hrl_truck_support_when_normal_reachable_scale=0.04,
                hrl_truck_emergency_support_when_no_normal_enabled=True,
                hrl_support_require_actionable_gain=True,
                hrl_support_actionable_min_gain_score=0.36,
                hrl_support_bound_dispatch_enabled=True,
                hrl_support_relay_reserve_enabled=False,
                hrl_support_critical_diversion_enabled=False,
                hrl_support_no_gain_backoff_enabled=True,
                hrl_support_no_gain_streak_threshold=1,
                hrl_support_no_gain_cooldown_steps=20,
                support_force_dispatch_enabled=False,
                support_force_uav_preempt_enabled=False,
                hrl_truck_routine_stuck_escape_enabled=True,
                hrl_routine_localize_eta_exit_enabled=True,
                hrl_truck_normal_to_normal_switch_min_improve_ratio=0.05,
                hrl_truck_normal_to_normal_switch_min_score_gain=0.02,
                hrl_event_first_refresh_enabled=False,
                hrl_event_admission_gate_enabled=False,
                hrl_max_no_refresh_steps=8,
                hrl_normal_stall_hard_refresh_enabled=False,
                hrl_normal_stall_local_only=True,
                hrl_path_blocked_impact_gate_enabled=True,
                hrl_path_blocked_local_repair_first=True,
                hrl_path_blocked_global_refresh_enabled=False,
                hrl_truck_dead_end_local_first=True,
                hrl_truck_dead_end_global_refresh_enabled=False,
                hrl_uav_task_reservation_exec_enabled=False,
                hrl_uav_assist_enabled=False,
                hrl_uav_task_reservation_enabled=False,
                task_reservation_enabled=False,
                recent_release_cooldown_enabled=False,
            )
        if mm == "erc_support_recovery_budget":
            cfg_new = _erc_ablation_cfg(base_cfg, "erc_rhc")
            return replace(
                cfg_new,
                # Bound-only launch plus an explicit support budget to preserve
                # routine throughput when normal backlog is still reachable.
                hrl_uav_emergency_commit_hold_enabled=True,
                hrl_docked_uav_soft_invalid_hold_enabled=True,
                truck_support_uav_recovery_enabled=True,
                uav_allow_rendezvous_launch=True,
                uav_rendezvous_launch_requires_docked_truck_goal=True,
                uav_recovery_distance_buffer_m=500.0,
                uav_reject_cache_window_steps=12,
                uav_reject_cache_min_repeat=3,
                uav_reject_cache_ttl_steps=12,
                hrl_support_max_trucks_when_normal_pending=1,
                hrl_support_budget_require_warning_when_normal=True,
                hrl_truck_support_when_normal_reachable_scale=0.02,
                hrl_truck_emergency_support_when_no_normal_enabled=True,
                hrl_support_require_actionable_gain=True,
                hrl_support_actionable_min_gain_score=0.40,
                hrl_support_bound_dispatch_enabled=True,
                hrl_support_relay_reserve_enabled=False,
                hrl_support_critical_diversion_enabled=False,
                hrl_support_no_gain_backoff_enabled=True,
                hrl_support_no_gain_streak_threshold=1,
                hrl_support_no_gain_cooldown_steps=24,
                support_force_dispatch_enabled=False,
                support_force_uav_preempt_enabled=False,
                truck_routine_near_goal_support_protect_enabled=True,
                truck_routine_near_goal_support_protect_steps=8,
                hrl_truck_routine_stuck_escape_enabled=True,
                hrl_routine_localize_eta_exit_enabled=True,
                hrl_truck_normal_to_normal_switch_min_improve_ratio=0.05,
                hrl_truck_normal_to_normal_switch_min_score_gain=0.02,
                hrl_event_first_refresh_enabled=False,
                hrl_event_admission_gate_enabled=False,
                hrl_max_no_refresh_steps=8,
                hrl_normal_stall_hard_refresh_enabled=False,
                hrl_normal_stall_local_only=True,
                hrl_path_blocked_impact_gate_enabled=True,
                hrl_path_blocked_local_repair_first=True,
                hrl_path_blocked_global_refresh_enabled=False,
                hrl_truck_dead_end_local_first=True,
                hrl_truck_dead_end_global_refresh_enabled=False,
                hrl_uav_task_reservation_exec_enabled=False,
                hrl_uav_assist_enabled=False,
                hrl_uav_task_reservation_enabled=False,
                task_reservation_enabled=False,
                recent_release_cooldown_enabled=False,
            )
        if mm == "erc_support_recovery_budget_strict":
            cfg_new = _erc_ablation_cfg(base_cfg, "erc_rhc")
            return replace(
                cfg_new,
                hrl_uav_emergency_commit_hold_enabled=True,
                hrl_docked_uav_soft_invalid_hold_enabled=True,
                truck_support_uav_recovery_enabled=True,
                truck_recovery_require_request_when_normal_pending=True,
                uav_allow_rendezvous_launch=True,
                uav_rendezvous_launch_requires_docked_truck_goal=True,
                uav_recovery_distance_buffer_m=500.0,
                uav_reject_cache_window_steps=12,
                uav_reject_cache_min_repeat=3,
                uav_reject_cache_ttl_steps=12,
                hrl_support_max_trucks_when_normal_pending=1,
                hrl_support_budget_require_warning_when_normal=True,
                hrl_truck_support_when_normal_reachable_scale=0.01,
                hrl_truck_emergency_support_when_no_normal_enabled=True,
                hrl_support_require_actionable_gain=True,
                hrl_support_actionable_min_gain_score=0.42,
                hrl_support_bound_dispatch_enabled=True,
                hrl_support_relay_reserve_enabled=False,
                hrl_support_critical_diversion_enabled=False,
                hrl_support_no_gain_backoff_enabled=True,
                hrl_support_no_gain_streak_threshold=1,
                hrl_support_no_gain_cooldown_steps=24,
                support_force_dispatch_enabled=False,
                support_force_uav_preempt_enabled=False,
                truck_routine_near_goal_support_protect_enabled=True,
                truck_routine_near_goal_support_protect_steps=8,
                hrl_truck_routine_stuck_escape_enabled=True,
                hrl_routine_localize_eta_exit_enabled=True,
                hrl_truck_normal_to_normal_switch_min_improve_ratio=0.05,
                hrl_truck_normal_to_normal_switch_min_score_gain=0.02,
                hrl_event_first_refresh_enabled=False,
                hrl_event_admission_gate_enabled=False,
                hrl_max_no_refresh_steps=8,
                hrl_normal_stall_hard_refresh_enabled=False,
                hrl_normal_stall_local_only=True,
                hrl_path_blocked_impact_gate_enabled=True,
                hrl_path_blocked_local_repair_first=True,
                hrl_path_blocked_global_refresh_enabled=False,
                hrl_truck_dead_end_local_first=True,
                hrl_truck_dead_end_global_refresh_enabled=False,
                hrl_uav_task_reservation_exec_enabled=False,
                hrl_uav_assist_enabled=False,
                hrl_uav_task_reservation_enabled=False,
                task_reservation_enabled=False,
                recent_release_cooldown_enabled=False,
            )
        if mm == "erc_support_recovery_budget_urgent":
            cfg_new = _erc_ablation_cfg(base_cfg, "erc_support_recovery_budget_strict")
            return replace(
                cfg_new,
                truck_recovery_request_min_urgency_when_normal_pending=0.95,
                hrl_support_actionable_min_gain_score=0.45,
                hrl_truck_support_when_normal_reachable_scale=0.0,
            )
        if mm in {
            "erc_full_new",
            "erc_reservation_only",
            "erc_airborne_lock_only",
            "erc_truck_assist_only",
            "erc_reservation_lock_no_assist",
        }:
            # Component ablation baseline:
            # keep all non-related logic identical to old ERC, then toggle only
            # reservation / airborne-lock / truck-assist trio.
            cfg_new = _erc_ablation_cfg(base_cfg, "erc_rhc_old")
            if mm == "erc_full_new":
                return replace(
                    cfg_new,
                    hrl_uav_task_reservation_exec_enabled=True,
                    hrl_uav_emergency_commit_hold_enabled=True,
                    hrl_uav_assist_enabled=True,
                )
            if mm == "erc_reservation_only":
                return replace(
                    cfg_new,
                    hrl_uav_task_reservation_exec_enabled=True,
                    hrl_uav_emergency_commit_hold_enabled=False,
                    hrl_uav_assist_enabled=False,
                )
            if mm == "erc_airborne_lock_only":
                return replace(
                    cfg_new,
                    hrl_uav_task_reservation_exec_enabled=False,
                    hrl_uav_emergency_commit_hold_enabled=True,
                    hrl_uav_assist_enabled=False,
                )
            if mm == "erc_truck_assist_only":
                return replace(
                    cfg_new,
                    hrl_uav_task_reservation_exec_enabled=False,
                    hrl_uav_emergency_commit_hold_enabled=False,
                    hrl_uav_assist_enabled=True,
                )
            if mm == "erc_reservation_lock_no_assist":
                return replace(
                    cfg_new,
                    hrl_uav_task_reservation_exec_enabled=True,
                    hrl_uav_emergency_commit_hold_enabled=True,
                    hrl_uav_assist_enabled=False,
                )
        # Build overrides without touching reward/task/map generation.
        overrides: Dict[str, Any] = {}
        if mm == "erc_hard_events_only":
            overrides.update(
                {
                    "erc_ablate_low_value_refresh": True,
                    "erc_ablate_map_ranking_refresh": True,
                }
            )
        elif mm == "erc_no_map_ranking_refresh":
            overrides.update({"erc_ablate_map_ranking_refresh": True})
        elif mm == "erc_no_tc_global_assignment":
            overrides.update(
                {
                    "timecritical_global_assignment_enabled": False,
                    "erc_ablate_tc_global_assignment": True,
                }
            )
        elif mm == "erc_no_support_chain":
            overrides.update(
                {
                    "erc_ablate_support_chain": True,
                    "support_force_dispatch_enabled": False,
                    "support_force_uav_preempt_enabled": False,
                    "hrl_support_bound_dispatch_enabled": False,
                    "hrl_support_requires_timecritical_binding": False,
                    "hrl_support_relay_reserve_enabled": False,
                    "hrl_support_critical_diversion_enabled": False,
                }
            )
        elif mm == "erc_no_cluster_primary_reservation":
            overrides.update(
                {
                    "erc_ablate_cluster_primary_reservation": True,
                    "cluster_primary_task_enabled": False,
                    "task_reservation_enabled": False,
                    "recent_release_cooldown_enabled": False,
                    "hrl_uav_task_reservation_enabled": False,
                }
            )
        elif mm == "erc_no_event_scoring_bonus":
            overrides.update({"erc_ablate_event_scoring_bonus": True})
        elif mm == "erc_no_normal_protection":
            overrides.update(
                {
                    "erc_ablate_normal_protection": True,
                    "hrl_truck_emergency_min_pending_normal_to_block": 0,
                    "hrl_truck_emergency_cover_threshold_when_normal_reachable": 0.0,
                    "hrl_truck_emergency_relief_uav_cover_threshold": 0.0,
                }
            )
        if not overrides:
            return base_cfg
        return replace(base_cfg, **overrides)

    requested_method = str(method).strip().lower()
    m = _canonical_method_name(method)
    if m == ER_HLNS_RISK_SLACK_ROUTINE_METHOD:
        # Lower-level callers may request the candidate directly.  Keep the
        # formal ER-HLNS builder and apply the candidate overlay locally.
        cfg = replace(cfg, **ER_HLNS_RISK_SLACK_ROUTINE_OVERLAY)
        m = ER_HLNS_METHOD
    if m == ER_HLNS_FORCE_INITIAL_LIFELINE_METHOD:
        # Candidate-only initial emergency ordering override.  The route
        # manager still owns all feasibility/safety checks; this flag only
        # bypasses the spatial-overload veto for the initial lifeline order.
        cfg = replace(cfg, **ER_HLNS_FORCE_INITIAL_LIFELINE_OVERLAY)
        m = ER_HLNS_METHOD
    if m == ER_HLNS_R4_ROUTINE_TAKEOVER_METHOD:
        # Candidate-only bounded routine takeover.  Keep the formal ER-HLNS
        # planner/route construction and opt in only through the capability
        # and private algorithm overlay below.
        cfg = replace(cfg, **ER_HLNS_R4_ROUTINE_TAKEOVER_OVERLAY)
        m = ER_HLNS_METHOD
    if m == ER_HLNS_IDLE_ROUTINE_DISPATCH_METHOD:
        # Candidate-only idle routine dispatch.  The public/frozen
        # environment remains inherited; only this private planner overlay is
        # enabled for the explicit candidate method.
        cfg = replace(cfg, **ER_HLNS_IDLE_ROUTINE_DISPATCH_OVERLAY)
        m = ER_HLNS_METHOD
    if m == ER_HLNS_IDLE_BALANCED_ROUTINE_METHOD:
        # Candidate-only composition of the idle routine rescue and the LB
        # balanced route overlay.  Keep both controls private to this explicit
        # method identifier; the formal ER-HLNS builder remains untouched.
        cfg = replace(
            cfg,
            **ER_HLNS_LB_BALANCED_OVERLAY,
            **ER_HLNS_IDLE_ROUTINE_DISPATCH_OVERLAY,
        )
        m = ER_HLNS_METHOD
    if m == ER_HLNS_BALANCED_ALL_TASKS_METHOD:
        # Candidate-only aggressive LB pilot.  It composes the balanced
        # constructor and execution watchdog behind an explicit method id.
        cfg = replace(cfg, **ER_HLNS_BALANCED_ALL_TASKS_OVERLAY)
        m = ER_HLNS_METHOD
    if m == ER_HLNS_BALANCED_ALL_TASKS_V2_METHOD:
        # Candidate-only V2: post-launch parallel corridor plus ETA-guarded
        # routine re-auction.  Formal ER-HLNS remains untouched.
        cfg = replace(cfg, **ER_HLNS_BALANCED_ALL_TASKS_V2_OVERLAY)
        m = ER_HLNS_METHOD
    if m == ER_HLNS_BALANCED_ALL_TASKS_V3_METHOD:
        # Candidate-only dual-plan selector; formal ER-HLNS remains unchanged.
        cfg = replace(cfg, **ER_HLNS_BALANCED_ALL_TASKS_V3_OVERLAY)
        m = ER_HLNS_METHOD
    if m == ER_HLNS_BALANCED_ALL_TASKS_V4_METHOD:
        # Candidate-only pre-launch parallel extension of V3.  The selector,
        # corridor safety gates, and all physical fields remain shared with
        # V3; only the candidate-owned launch timing policy is relaxed.
        cfg = replace(cfg, **ER_HLNS_BALANCED_ALL_TASKS_V4_OVERLAY)
        m = ER_HLNS_METHOD
    if m == ER_HLNS_BALANCED_ALL_TASKS_V5_METHOD:
        # Candidate-only launch-first extension of V4.  It reorders only the
        # initial route suffix after the existing corridor gate approves it.
        cfg = replace(cfg, **ER_HLNS_BALANCED_ALL_TASKS_V5_OVERLAY)
        m = ER_HLNS_METHOD
    if m == ER_HLNS_LB_HARD_COVERAGE_METHOD:
        # Candidate-only hard NORMAL coverage rescue; formal ER-HLNS stays on
        # the original refresh and route ownership policy.
        cfg = replace(cfg, **ER_HLNS_LB_HARD_COVERAGE_OVERLAY)
        m = ER_HLNS_METHOD
    if m == ER_HLNS_LB_HARD_COVERAGE_SINGLE_RESCUE_METHOD:
        # Candidate-only hard coverage with one bounded rescue transfer per
        # planning call; formal ER-HLNS remains unchanged.
        cfg = replace(cfg, **ER_HLNS_LB_HARD_COVERAGE_SINGLE_RESCUE_OVERLAY)
        m = ER_HLNS_METHOD
    if m == ER_HLNS_LB_HARD_COVERAGE_PROTECTED_METHOD:
        # Candidate-only protected hard NORMAL coverage: use exactly the hard
        # coverage pilot, but restore ordinary-task protection for this branch.
        cfg = replace(cfg, **ER_HLNS_LB_HARD_COVERAGE_PROTECTED_OVERLAY)
        m = ER_HLNS_METHOD
    if m == ER_HLNS_LB_HARD_COVERAGE_COMMITMENT_METHOD:
        # Candidate-only commitment extension: preserve ordinary-task
        # protection and add the two bounded commitment guards.
        cfg = replace(cfg, **ER_HLNS_LB_HARD_COVERAGE_COMMITMENT_OVERLAY)
        m = ER_HLNS_METHOD
    if m == ER_HLNS_LB_HARD_COVERAGE_ORPHAN_GUARD_METHOD:
        # Candidate-only orphan guard: retain the hard-coverage rescue but
        # restrict transfers to true no-goal tasks and remember no-truck tries.
        cfg = replace(cfg, **ER_HLNS_LB_HARD_COVERAGE_ORPHAN_GUARD_OVERLAY)
        m = ER_HLNS_METHOD
    if m == ER_HLNS_LB_ROUTINE_PROTECTED_METHOD:
        # Candidate-only light LB routine-protection branch.  It restores
        # normal-task refresh/protection without enabling hard rescue.
        cfg = replace(cfg, **ER_HLNS_LB_ROUTINE_PROTECTED_OVERLAY)
        m = ER_HLNS_METHOD
    if m == ER_HLNS_LB_ROUTINE_PROTECTED_OWNER_REPAIR_METHOD:
        # Candidate-only routine protection plus bounded ownership repair.
        cfg = replace(cfg, **ER_HLNS_LB_ROUTINE_PROTECTED_OWNER_REPAIR_OVERLAY)
        m = ER_HLNS_METHOD
    if m == ER_HLNS_LB_HARD_COVERAGE_SAFETY_GATED_METHOD:
        # Candidate-only safety-gated hard coverage: retain bounded hard
        # rescue while disabling the three high-risk parallel switches.
        cfg = replace(cfg, **ER_HLNS_LB_HARD_COVERAGE_SAFETY_GATED_OVERLAY)
        m = ER_HLNS_METHOD
    if m == ER_HLNS_LB_ROUTINE_PROTECTED_EMERGENCY_RESCUE_METHOD:
        # Candidate-only emergency rescue on the light routine-protected
        # branch; physical/safety fields remain inherited.
        cfg = replace(cfg, **ER_HLNS_LB_ROUTINE_PROTECTED_EMERGENCY_RESCUE_OVERLAY)
        m = ER_HLNS_METHOD
    if m == ER_HLNS_LB_ROUTINE_PROTECTED_V3_SELECTOR_METHOD:
        # Candidate-only V3 selector on the light routine-protected branch;
        # no hard NORMAL rescue is included.
        cfg = replace(cfg, **ER_HLNS_LB_ROUTINE_PROTECTED_V3_SELECTOR_OVERLAY)
        m = ER_HLNS_METHOD
    if m == ER_HLNS_LB_ADAPTIVE_COVERAGE_METHOD:
        # Candidate-only adaptive hard coverage: the route manager activates
        # hard rescue only when a configured orphan-pending threshold is met.
        cfg = replace(cfg, **ER_HLNS_LB_ADAPTIVE_COVERAGE_OVERLAY)
        m = ER_HLNS_METHOD
    if m == ER_HLNS_LB_ADAPTIVE_SINGLE_RESCUE_METHOD:
        # Candidate-only adaptive hard coverage with one bounded rescue per
        # planning call; formal ER-HLNS remains unchanged.
        cfg = replace(cfg, **ER_HLNS_LB_ADAPTIVE_SINGLE_RESCUE_OVERLAY)
        m = ER_HLNS_METHOD
    if m == ER_HLNS_LB_BALANCED_SELECTOR_METHOD:
        # Candidate-only emergency-safe V3 selector plus hard NORMAL rescue.
        cfg = replace(cfg, **ER_HLNS_LB_BALANCED_SELECTOR_OVERLAY)
        m = ER_HLNS_METHOD
    if m == ER_HLNS_BALANCED_ALL_TASKS_V6_METHOD:
        # Candidate-only emergency-first parallel variant.
        cfg = replace(cfg, **ER_HLNS_BALANCED_ALL_TASKS_V6_OVERLAY)
        m = ER_HLNS_METHOD
    if m == ER_HLNS_BALANCED_ALL_TASKS_V7_METHOD:
        # Candidate-only conditional road-impact promotion variant.  Its
        # upper plan remains V6 balanced/no-UAV-priority; only execution-time
        # road-impact promotion is enabled.
        cfg = replace(cfg, **ER_HLNS_BALANCED_ALL_TASKS_V7_OVERLAY)
        m = ER_HLNS_METHOD
    if m == ER_HLNS_BALANCED_ALL_TASKS_V8_METHOD:
        # Candidate-only structural shadow selector.  Formal ER-HLNS and V6
        # remain unchanged; the overlay is applied only by explicit runners.
        cfg = replace(cfg, **ER_HLNS_BALANCED_ALL_TASKS_V8_OVERLAY)
        m = ER_HLNS_METHOD
    candidate_parallel_routine_emergency = bool(
        m == ER_HLNS_PARALLEL_ROUTINE_EMERGENCY_METHOD
    )
    if candidate_parallel_routine_emergency:
        # No public physical field is changed.  The planner capability below
        # is the sole opt-in for this candidate pilot.
        m = ER_HLNS_METHOD
    if m == C_ALNS_CLEAN_METHOD:
        # Keep the candidate alias usable through the lower-level factory too;
        # the public/formal ``rolling_horizon_alns`` path is unchanged.
        cfg = replace(cfg, **C_ALNS_CLEAN_OVERLAY)
        m = ROLLING_HORIZON_ALNS_METHOD
    if m == "random_rule":
        cfg_rule = replace(
            cfg,
            # Rule baseline: it uses the shared UAV physics/return gate, but
            # does not implement ERC's coordinated truck-UAV support chain.
            truck_support_uav_recovery_enabled=False,
            uav_hard_recovery_battery_guard=False,
        )
        return RandomRulePlanner(seed=seed), "rule", bool(cfg_rule.enable_rth_mask), cfg_rule
    if m == PRIORITY_GREEDY_METHOD:
        cfg_rule = replace(
            cfg,
            truck_support_uav_recovery_enabled=False,
            uav_hard_recovery_battery_guard=False,
        )
        return PriorityGreedyPlanner(), "priority_rule", bool(cfg_rule.enable_rth_mask), cfg_rule
    if m == NEAREST_GREEDY_METHOD:
        cfg_rule = replace(
            cfg,
            truck_support_uav_recovery_enabled=False,
            uav_hard_recovery_battery_guard=False,
        )
        return GreedyRulePlanner(), "nearest_rule", bool(cfg_rule.enable_rth_mask), cfg_rule
    if m in {"rolling_fixed", ROLLING_HORIZON_METHOD}:
        cfg = replace(
            cfg,
            # Fixed-interval rolling keeps the shared return/recovery support
            # mechanism, but stale replanning can still miss the recovery
            # window before the truck moves away.
            truck_support_uav_recovery_enabled=True,
            uav_hard_recovery_battery_guard=False,
            uav_allow_rendezvous_launch=True,
            uav_rendezvous_launch_requires_docked_truck_goal=True,
            hrl_interval=max(int(getattr(cfg, "hrl_interval", 5)), 8),
        )
        return EventTriggeredRollingPlanner(
            decision_interval=int(cfg.hrl_interval),
            seed=seed,
            use_event_trigger=(False if use_event_trigger_override is None else bool(use_event_trigger_override)),
            use_risk_term=(bool(getattr(cfg, "use_risk_term", True)) if use_risk_term_override is None else bool(use_risk_term_override)),
            use_rth_repair=(bool(getattr(cfg, "use_rth_repair", True)) if use_rth_repair_override is None else bool(use_rth_repair_override)),
            replan_cooldown_steps=int(max(getattr(cfg, "hrl_replan_cooldown_steps", 4), 0)),
            min_goal_hold_steps=int(max(getattr(cfg, "hrl_goal_min_hold_steps", 8), 0)),
            switch_margin=float(max(getattr(cfg, "hrl_goal_switch_margin", 0.12), 0.0)),
        ), "rolling", bool(cfg.enable_rth_mask), cfg
    if m == ER_HLNS_METHOD:
        # ER-HLNS has a dedicated construction path (separate from the
        # historical ERC/ALNS family below).  Keep the B-side anti-starvation
        # repair explicitly wired here as well, otherwise the public config
        # knob is silently dropped for the paper mainline on L-B.
        scenario_key = str(getattr(cfg, "scenario", "")).upper().strip()
        map_key = str(getattr(cfg, "map_complexity", "")).upper().strip()
        large_route = bool(
            int(getattr(cfg, "num_nodes", 0)) >= 180 or map_key in {"L", "R"}
        )
        if large_route and scenario_key == "B":
            cfg = replace(
                cfg,
                # The C-side anti-starvation repairs are intentionally not
                # enabled for B: on B they can rewrite a still-viable route
                # after a transient road observation and hurt seed stability.
                # Keep the implementation/config knobs available for C and
                # targeted diagnostics, but leave the B mainline conservative.
                erc_stalled_routine_ownership_repair_enabled=False,
                erc_unassigned_routine_repair_enabled=False,
                erc_stalled_routine_ownership_min_step=36,
                erc_stalled_routine_ownership_exposure_steps=42,
                erc_stalled_routine_ownership_max_repairs_per_step=1,
                erc_unassigned_routine_repair_min_step=70,
                erc_unassigned_routine_repair_max_per_step=1,
                # Validated B transfer: rescue only an unassigned/orphaned
                # routine contract early enough to remain road-reachable.
                # It never preempts an active route and is bounded to one
                # transfer per planning step.
                erc_b_orphaned_routine_rescue_enabled=True,
                erc_b_orphaned_routine_rescue_min_step=36,
                erc_b_orphaned_routine_rescue_max_lifeline_ratio=0.90,
                # Keep the C-only critical-support escape.  The B improvement
                # above addresses orphaned routine ownership only; enabling a
                # second emergency preemption rule in B was not needed and
                # would make the cross-seed effect harder to attribute.
                hrl_support_chain_critical_escape_scenarios="C",
                # The global B anti-churn hold remains disabled: isolation
                # showed that it can regress individual seeds even when the
                # route is nominally valid.  The orphan rescue is the safer
                # C-derived commitment component for the LB mainline.
                hrl_b_route_stability_enabled=False,
                hrl_b_prelaunch_contract_lock_enabled=False,
                hrl_b_docked_latch_rearm_enabled=False,
                hrl_b_anchor_unreachable_uav_launch_enabled=False,
                hrl_b_anchor_unreachable_uav_launch_max_lifeline_ratio=0.10,
                # The watchdog pilot did not change the two target seeds;
                # keep it disabled in the B mainline.
                hrl_route_plan_emergency_launch_watchdog_enabled=False,
                hrl_route_plan_emergency_starvation_promotion_enabled=False,
                # C-derived pre-launch contract ownership: an already loaded
                # docked UAV keeps its atomic emergency task until launch or
                # a hard safety release. This does not lock truck routes or
                # airborne sorties.
                uav_docked_contract_owner_lock_enabled=False,
                # If the assigned truck has exhausted its emergency package,
                # release that pending contract for a stocked unit. This is
                # limited to the large-map B mainline.
                hrl_route_plan_stockout_transfer_enabled=True,
                # A/B validation regressed LB seed117 by breaking an ongoing
                # recovery chain; keep the branch available but disabled.
                hrl_route_plan_owner_carrier_mismatch_repair_enabled=False,
            )
        cfg = replace(cfg, hrl_route_plan_v2_enabled=True)
        return ERHLNSPlanner(
            decision_interval=int(cfg.hrl_interval),
            seed=seed,
            use_event_trigger=(
                bool(getattr(cfg, "use_event_trigger", True))
                if use_event_trigger_override is None
                else bool(use_event_trigger_override)
            ),
            use_risk_term=(
                bool(getattr(cfg, "use_risk_term", True))
                if use_risk_term_override is None
                else bool(use_risk_term_override)
            ),
            use_rth_repair=(
                bool(getattr(cfg, "use_rth_repair", True))
                if use_rth_repair_override is None
                else bool(use_rth_repair_override)
            ),
            replan_cooldown_steps=int(
                max(getattr(cfg, "hrl_replan_cooldown_steps", 4), 0)
            ),
            min_goal_hold_steps=int(
                max(getattr(cfg, "hrl_goal_min_hold_steps", 8), 0)
            ),
            switch_margin=float(
                max(getattr(cfg, "hrl_goal_switch_margin", 0.12), 0.0)
            ),
            iterations=int(max(getattr(cfg, "alns_iterations", 24), 1)),
        ), "er_hlns", bool(cfg.enable_rth_mask), cfg
    if m == DYNAMIC_REPLANNING_ALNS_METHOD:
        # Benchmark-only mode: canonical destroy/repair pool and a generic
        # dynamic trigger.  ER-HLNS route planning and ER-specific operators
        # remain disabled through the isolated algorithm package.
        cfg = replace(
            cfg,
            hrl_route_plan_v2_enabled=False,
            alns_enabled=True,
            alns_solution_mode="k2_active",
            alns_sequence_length=2,
            alns_operator_pool="canonical_k2",
            alns_selection_mode="adaptive",
            adaptive_horizon_mode="disabled",
            local_search_mode="disabled",
            # Keep this baseline on the common physical/safety layer while
            # disabling ER-specific coordination and search enhancements.
            erc_ablate_support_chain=True,
            erc_ablate_map_ranking_refresh=True,
            erc_ablate_event_scoring_bonus=True,
            erc_ablate_normal_protection=True,
            erc_ablate_tc_global_assignment=True,
            timecritical_global_assignment_enabled=False,
            hrl_uav_task_reservation_enabled=False,
            hrl_uav_task_reservation_exec_enabled=False,
            task_reservation_enabled=False,
            recent_release_cooldown_enabled=False,
            cluster_primary_task_enabled=False,
            hrl_uav_assist_enabled=False,
            support_force_dispatch_enabled=False,
            support_force_uav_preempt_enabled=False,
            hrl_support_bound_dispatch_enabled=False,
            hrl_support_requires_timecritical_binding=False,
            hrl_support_relay_reserve_enabled=False,
            hrl_support_critical_diversion_enabled=False,
            alns_critical_recovery_repair_enabled=False,
            alns_critical_support_rebind_enabled=False,
        )
        return DynamicReplanningALNSPlanner(
            decision_interval=int(cfg.hrl_interval),
            seed=seed,
            use_risk_term=(
                bool(getattr(cfg, "use_risk_term", True))
                if use_risk_term_override is None
                else bool(use_risk_term_override)
            ),
            use_rth_repair=(
                bool(getattr(cfg, "use_rth_repair", True))
                if use_rth_repair_override is None
                else bool(use_rth_repair_override)
            ),
            iterations=int(max(getattr(cfg, "hrl_route_plan_alns_iterations", getattr(cfg, "alns_iterations", 4)), 1)),
        ), "dynamic_replanning_alns", bool(cfg.enable_rth_mask), cfg
    if m in {HYBRID_GENETIC_METHOD, VNS_METHOD, ROLLING_HORIZON_ALNS_METHOD}:
        fixed_horizon = bool(m == ROLLING_HORIZON_ALNS_METHOD)
        population_horizon = bool(m == HYBRID_GENETIC_METHOD)
        neighborhood_horizon = bool(m == VNS_METHOD)
        cfg = replace(
            cfg,
            hrl_route_plan_v2_enabled=False,
            alns_enabled=True,
            alns_solution_mode="k2_active",
            alns_sequence_length=2,
            alns_operator_pool=(
                "canonical_k2" if fixed_horizon else "combined_k2"
            ),
            # The dedicated planner class is the method identity. Keep the
            # legacy diagnostic enum within EnvConfig's validated values.
            alns_selection_mode=("adaptive" if fixed_horizon else "uniform"),
            candidate_ranker_mode="disabled",
            adaptive_horizon_mode="disabled",
            local_search_mode="disabled",
            hrl_interval=(
                max(
                    int(getattr(cfg, "hrl_interval", 5)),
                    20 if population_horizon else 8,
                )
                if fixed_horizon or population_horizon or neighborhood_horizon
                else int(getattr(cfg, "hrl_interval", 5))
            ),
        )
        planner_cls = {
            HYBRID_GENETIC_METHOD: HybridGeneticK2Planner,
            VNS_METHOD: VariableNeighborhoodSearchK2Planner,
            ROLLING_HORIZON_ALNS_METHOD: RollingHorizonALNSPlanner,
        }[m]
        extra_search_kwargs: Dict[str, Any] = {}
        if m == VNS_METHOD:
            extra_search_kwargs["candidate_limit"] = 24
        planner = planner_cls(
            decision_interval=int(cfg.hrl_interval),
            seed=seed,
            use_event_trigger=(
                False
                if fixed_horizon or population_horizon or neighborhood_horizon
                else (
                    bool(getattr(cfg, "use_event_trigger", True))
                    if use_event_trigger_override is None
                    else bool(use_event_trigger_override)
                )
            ),
            use_risk_term=(
                bool(getattr(cfg, "use_risk_term", True))
                if use_risk_term_override is None
                else bool(use_risk_term_override)
            ),
            use_rth_repair=(
                bool(getattr(cfg, "use_rth_repair", True))
                if use_rth_repair_override is None
                else bool(use_rth_repair_override)
            ),
            replan_cooldown_steps=int(
                max(getattr(cfg, "hrl_replan_cooldown_steps", 4), 0)
            ),
            min_goal_hold_steps=int(
                max(getattr(cfg, "hrl_goal_min_hold_steps", 8), 0)
            ),
            switch_margin=float(
                max(getattr(cfg, "hrl_goal_switch_margin", 0.12), 0.0)
            ),
            iterations=int(max(getattr(cfg, "alns_iterations", 24), 1)),
            **extra_search_kwargs,
        )
        return planner, "search", bool(cfg.enable_rth_mask), cfg
    gate18_map = {
        "erc_rhc_v2_direct_tc_first": "direct_tc_first",
        "erc_rhc_v2_support_anchor_strict": "support_anchor_strict",
        "erc_rhc_v2_anchor_arrival_force_launch": "anchor_arrival_force_launch",
        "erc_rhc_v2_safety_narrow_routine_guard": "safety_narrow_routine_guard",
        "erc_rhc_v2_gate18_core": "gate18_core",
    }
    gate19_map = {
        "erc_rhc_v2_gate19_launch_binding": "launch_binding",
        "erc_rhc_v2_gate19_support_lock": "support_lock",
        "erc_rhc_v2_gate19_core": "gate19_core",
    }
    gate20_map = {
        "erc_rhc_v2_gate20_passenger_invariant": "passenger_invariant",
        "erc_rhc_v2_gate20_support_reserve_launch": "support_reserve_launch",
        "erc_rhc_v2_gate20_rebind_at_anchor": "rebind_at_anchor",
        "erc_rhc_v2_gate20_core": "gate20_core",
        "erc_rhc_v2_workflow_repair": "workflow_repair",
        "erc_rhc_v2_support_quality_audit": "support_quality_audit",
        "erc_rhc_v2_support_quality_gate": "support_quality_gate",
        "erc_rhc_v2_support_quality_relaxed": "support_quality_relaxed",
    }
    if m in {"erc_rhc_v2", "erc_rhc_v2_support_recovery_repair", "erc_rhc_v2_gate16", "erc_rhc_v2_gate17", *gate18_map.keys(), *gate19_map.keys(), *gate20_map.keys()}:
        return ErcRhcV2Planner(
            decision_interval=int(cfg.hrl_interval),
            seed=seed,
            support_recovery_repair=bool(m in {"erc_rhc_v2_support_recovery_repair", "erc_rhc_v2_gate16", "erc_rhc_v2_gate17", *gate18_map.keys(), *gate19_map.keys(), *gate20_map.keys()}),
            gate16_command_quality=bool(m in {"erc_rhc_v2_gate16", "erc_rhc_v2_gate17", *gate18_map.keys(), *gate19_map.keys(), *gate20_map.keys()}),
            gate17_lifecycle_closure=bool(m == "erc_rhc_v2_gate17" or m in gate18_map or m in gate19_map or m in gate20_map),
            gate18_variant=str(gate18_map.get(m, "")),
            gate19_variant=str(gate19_map.get(m, "")),
            gate20_variant=str(gate20_map.get(m, "")),
        ), "erc_v2", bool(cfg.enable_rth_mask), cfg
    if m in {
        "erc_rhc_old",
        "erc_rhc",
        "erc_rhc_current",
        "erc_base_no_event",
        "erc_support_authorized",
        "erc_routine_commit",
        "erc_launch_quality_gate",
        "erc_tc_completion_chain",
        "erc_event_minimal_local",
        "erc_scoring_shrink",
        "erc_combined_safe_core",
        "erc_full",
        "erc_hard_events_only",
        "erc_no_map_ranking_refresh",
        "erc_no_tc_global_assignment",
        "erc_no_support_chain",
        "erc_no_cluster_primary_reservation",
        "erc_no_event_scoring_bonus",
        "erc_no_normal_protection",
        "erc_only_uav_commit",
        "erc_only_truck_escape",
        "erc_only_eta_exit",
        "erc_routine_rescue_combo",
        "erc_km_hard_routine",
        "erc_km_hard_routine_v2",
        "erc_same_config",
        ALNS_MAINLINE_METHOD,
        ER_ALNS_MAINLINE_METHOD,
        ER_ALNS_CURRENT_METHOD,
        ER_ALNS_INIT_PLUS_METHOD,
        ER_ALNS_REPAIR_PLUS_METHOD,
        ER_ALNS_FEASIBILITY_RESTORE_METHOD,
        ER_ALNS_BUDGET_1_25_METHOD,
        ER_ALNS_BUDGET_1_50_METHOD,
        ER_ALNS_COMBINED_CANDIDATE_METHOD,
        V2_BASE_METHOD,
        V2_DYNAMIC_K_METHOD,
        V2_LOCAL_SEARCH_METHOD,
        V2_COMBINED_METHOD,
        V2_BASE_EXTRA_BUDGET_METHOD,
        *LS_ABLATION_METHODS.keys(),
        UNIFORM_LNS_METHOD,
        CANONICAL_ALNS_METHOD,
        TABU_SEARCH_METHOD,
        "erc_ya_balanced",
        "erc_mc_lc_boost",
        "erc_ya_km_bridge",
        "erc_mc_lc_safe",
        "erc_ya_tune_a",
        "erc_ya_tune_b",
        "erc_ya_tune_c",
        "erc_support_recovery_launch",
        "erc_support_recovery_launch_relaxed",
        "erc_support_recovery_bound",
        "erc_support_recovery_budget",
        "erc_support_recovery_budget_strict",
        "erc_support_recovery_budget_urgent",
        "erc_support_priority",
        "erc_support_priority_strict",
        "erc_support_priority_reserve",
        "erc_full_new",
        "erc_reservation_only",
        "erc_airborne_lock_only",
        "erc_truck_assist_only",
        "erc_reservation_lock_no_assist",
    }:
        cfg = _erc_ablation_cfg(cfg, m)
        event_enabled = False if m == "erc_base_no_event" else True
        if m in {
            ALNS_MAINLINE_METHOD,
            ER_ALNS_MAINLINE_METHOD,
            ER_ALNS_CURRENT_METHOD,
            ER_ALNS_INIT_PLUS_METHOD,
            ER_ALNS_REPAIR_PLUS_METHOD,
            ER_ALNS_FEASIBILITY_RESTORE_METHOD,
            ER_ALNS_BUDGET_1_25_METHOD,
            ER_ALNS_BUDGET_1_50_METHOD,
            ER_ALNS_COMBINED_CANDIDATE_METHOD,
            V2_BASE_METHOD,
            V2_DYNAMIC_K_METHOD,
            V2_LOCAL_SEARCH_METHOD,
            V2_COMBINED_METHOD,
            V2_BASE_EXTRA_BUDGET_METHOD,
            *LS_ABLATION_METHODS.keys(),
            UNIFORM_LNS_METHOD,
            CANONICAL_ALNS_METHOD,
            TABU_SEARCH_METHOD,
        }:
            pool = "combined_k2"
            selection = "adaptive"
            event_enabled_for_mode = event_enabled
            if m == UNIFORM_LNS_METHOD:
                pool = "combined_k2"
                selection = "uniform"
            elif m == CANONICAL_ALNS_METHOD:
                pool = "canonical_k2"
                selection = "adaptive"
            elif m == TABU_SEARCH_METHOD:
                pool = "tabu_k2"
                selection = "tabu"
            cfg = replace(
                cfg,
                alns_enabled=True,
                alns_solution_mode="k2_active",
                alns_sequence_length=2,
                alns_operator_pool=pool,
                alns_selection_mode=selection,
                alns_sa_auto_calibration_enabled=True,
                candidate_ranker_mode="disabled",
                adaptive_horizon_mode=("active" if m in {V2_DYNAMIC_K_METHOD, V2_COMBINED_METHOD} else "disabled"),
                local_search_mode=("active" if m in {V2_LOCAL_SEARCH_METHOD, V2_COMBINED_METHOD} else "disabled"),
            )
            if requested_method in {ER_ALNS_MAINLINE_METHOD, ER_ALNS_CURRENT_METHOD}:
                cfg = replace(
                    cfg,
                    alns_initialization_mode="objective_greedy",
                    alns_operator_weight_profile="uniform",
                )
            elif requested_method == ER_ALNS_INIT_PLUS_METHOD:
                cfg = replace(
                    cfg,
                    alns_initialization_mode="critical_first",
                    alns_operator_weight_profile="uniform",
                    hrl_far_routine_bootstrap_enabled=True,
                )
            elif requested_method == ER_ALNS_REPAIR_PLUS_METHOD:
                cfg = replace(
                    cfg,
                    alns_operator_pool="er_k2",
                    alns_initialization_mode="objective_greedy",
                    alns_operator_weight_profile="critical_repair_bias",
                )
            elif requested_method == ER_ALNS_FEASIBILITY_RESTORE_METHOD:
                cfg = replace(
                    cfg,
                    alns_operator_pool="er_k2",
                    alns_initialization_mode="critical_first",
                    alns_operator_weight_profile="feasibility_restore_bias",
                )
            elif requested_method == ER_ALNS_BUDGET_1_25_METHOD:
                cfg = replace(
                    cfg,
                    alns_initialization_mode="objective_greedy",
                    alns_operator_weight_profile="uniform",
                    alns_iterations=int(round(int(getattr(cfg, "alns_iterations", 24)) * 1.25)),
                )
            elif requested_method == ER_ALNS_BUDGET_1_50_METHOD:
                cfg = replace(
                    cfg,
                    alns_initialization_mode="objective_greedy",
                    alns_operator_weight_profile="uniform",
                    alns_iterations=int(round(int(getattr(cfg, "alns_iterations", 24)) * 1.50)),
                )
            elif requested_method == ER_ALNS_COMBINED_CANDIDATE_METHOD:
                cfg = replace(
                    cfg,
                    alns_operator_pool="er_k2",
                    alns_initialization_mode="critical_first",
                    alns_operator_weight_profile="critical_repair_bias",
                    alns_iterations=int(round(int(getattr(cfg, "alns_iterations", 24)) * 1.25)),
                    hrl_far_routine_bootstrap_enabled=True,
                    hrl_conditional_h2_refresh_enabled=True,
                    hrl_uav_ride_stall_release_enabled=True,
                    hrl_truck_idle_hard_refresh_enabled=True,
                )
            if m == V2_BASE_EXTRA_BUDGET_METHOD:
                cfg = replace(cfg, alns_iterations=int(getattr(cfg, "alns_iterations", 24)) + int(getattr(cfg, "local_search_max_exact_checks_per_iteration", 5)))
            if m in LS_ABLATION_METHODS:
                cfg = replace(cfg, local_search_mode="active", local_search_disabled_moves=tuple(LS_ABLATION_METHODS[m]))
            iterations = int(max(getattr(cfg, "alns_iterations", 24), 0))
            if iterations <= 0:
                raise ValueError(f"{m} requires cfg.alns_iterations > 0")
            planner_cls = TabuSearchK2Planner if m == TABU_SEARCH_METHOD else EventResponsiveALNSPlanner
            return planner_cls(
                decision_interval=int(cfg.hrl_interval),
                seed=seed,
                use_event_trigger=(event_enabled_for_mode if use_event_trigger_override is None else bool(use_event_trigger_override)),
                use_risk_term=(bool(getattr(cfg, "use_risk_term", True)) if use_risk_term_override is None else bool(use_risk_term_override)),
                use_rth_repair=(bool(getattr(cfg, "use_rth_repair", True)) if use_rth_repair_override is None else bool(use_rth_repair_override)),
                replan_cooldown_steps=int(max(getattr(cfg, "hrl_replan_cooldown_steps", 4), 0)),
                min_goal_hold_steps=int(max(getattr(cfg, "hrl_goal_min_hold_steps", 8), 0)),
                switch_margin=float(max(getattr(cfg, "hrl_goal_switch_margin", 0.12), 0.0)),
                iterations=iterations,
            ), "alns", bool(cfg.enable_rth_mask), cfg
        return EventTriggeredRollingPlanner(
            decision_interval=int(cfg.hrl_interval),
            seed=seed,
            use_event_trigger=(event_enabled if use_event_trigger_override is None else bool(use_event_trigger_override)),
            use_risk_term=(bool(getattr(cfg, "use_risk_term", True)) if use_risk_term_override is None else bool(use_risk_term_override)),
            use_rth_repair=(bool(getattr(cfg, "use_rth_repair", True)) if use_rth_repair_override is None else bool(use_rth_repair_override)),
            replan_cooldown_steps=int(max(getattr(cfg, "hrl_replan_cooldown_steps", 4), 0)),
            min_goal_hold_steps=int(max(getattr(cfg, "hrl_goal_min_hold_steps", 8), 0)),
            switch_margin=float(max(getattr(cfg, "hrl_goal_switch_margin", 0.12), 0.0)),
        ), "rolling", bool(cfg.enable_rth_mask), cfg
    if m in {"ppo_pooled", "ppo_mlp", "ppo_hetgat_mask"}:
        default_enc = {
            "ppo_pooled": "pooled",
            "ppo_mlp": "mlp",
            "ppo_hetgat_mask": "hetgat",
        }[m]
        enc = str(encoder_type_override).strip().lower() or default_enc
        if enc not in {"pooled", "mlp", "hetgat"}:
            raise ValueError(f"Unsupported encoder_type override: {encoder_type_override!r}")
        default_mask = True if m == "ppo_hetgat_mask" else False
        mask = default_mask if enable_rth_mask_override is None else bool(enable_rth_mask_override)
        cfg2 = replace(cfg, enable_rth_mask=bool(mask), use_hetgat=bool(enc == "hetgat"))
        return RiskTriggeredHRLPlanner(decision_interval=int(cfg2.hrl_interval), seed=seed, encoder_type=enc), enc, bool(mask), cfg2
    raise ValueError(f"Unsupported method: {method}")


_ALGORITHM_FIELD_PREFIXES = (
    "alns_",
    "hrl_",
    "erc_",
    "candidate_ranker_",
    "adaptive_horizon_",
    "local_search_",
    "execution_",
    "goal_switch_",
)
_ALGORITHM_FIELD_NAMES = {
    "enable_rth_mask",
    "use_event_trigger",
    "use_risk_term",
    "use_rth_repair",
    "task_reservation_enabled",
    "recent_release_cooldown_enabled",
    "cluster_primary_task_enabled",
    "timecritical_global_assignment_enabled",
    "road_uav_scout_enabled",
    "road_uav_scout_radius_m",
}
_COMMON_SAFETY_FIELDS = {
    "physical_environment_safety_protocol",
    "uav_hard_recovery_battery_guard",
    "uav_allow_rendezvous_launch",
    "uav_rendezvous_launch_requires_docked_truck_goal",
    "uav_launch_min_horizon_buffer_steps",
    "hrl_tc_override_min_recovery_margin_m",
    "hrl_tc_override_min_battery_margin_ratio",
    "uav_reject_cache_window_steps",
    "truck_support_uav_recovery_enabled",
    # Physical sensor availability is common. The no-scout information
    # ablation is expressed through AlgorithmProfile, not by mutating hardware.
    "road_uav_scout_enabled",
    "road_uav_scout_radius_m",
}

# Candidate-only C-ALNS overlay.  All fields are algorithm-owned ``hrl_`` or
# ``erc_`` controls; physical payload, energy, communication and recovery
# safety fields remain inherited from the paired scenario configuration.
C_ALNS_CLEAN_OVERLAY = {
    "erc_ablate_support_chain": True,
    "erc_tc_support_required_enabled": False,
    "erc_tc_support_anchor_waypoint_enabled": False,
    "hrl_support_bound_dispatch_enabled": False,
    "hrl_support_relay_reserve_enabled": False,
    "hrl_support_critical_diversion_enabled": False,
    "hrl_uav_assist_enabled": False,
    "hrl_uav_task_reservation_enabled": False,
    "hrl_uav_task_reservation_exec_enabled": False,
    "hrl_uav_task_transfer_enabled": False,
    "hrl_uav_idle_truck_staging_enabled": False,
    "hrl_initial_directional_cover_enabled": False,
    "hrl_truck_directional_split_enabled": False,
    "hrl_unreachable_normal_uav_takeover_enabled": False,
    "hrl_route_plan_stalled_queue_rescue_enabled": False,
    "hrl_route_plan_stalled_queue_anchor_rescue_enabled": False,
    "hrl_route_plan_deadline_rescue_enabled": False,
    "hrl_route_plan_onsite_takeover_enabled": False,
    "hrl_route_plan_routine_dynamic_reassignment_enabled": False,
}

# Candidate-only LB rescue overlay.  This composes the already existing safe
# parallel-corridor capability with the two algorithm-side ablations that were
# suppressing routine refresh/protection in the frozen LB scenario.  It does
# not touch task/map/weather/road/safety fields and is never applied to the
# formal ``er_hlns`` identifier.
ER_HLNS_LB_BALANCED_OVERLAY = {
    "erc_ablate_low_value_refresh": False,
    "erc_ablate_normal_protection": False,
    "hrl_route_plan_mixed_coverage_enabled": True,
    "hrl_route_plan_mixed_coverage_emergency_reserve_steps": 30,
    "hrl_route_plan_residual_normal_bipartite_matching_enabled": True,
    "hrl_route_plan_routine_dynamic_reassignment_enabled": True,
    "hrl_route_plan_routine_dynamic_reassignment_radius_m": 800.0,
    "hrl_route_plan_routine_dynamic_reassignment_min_eta_gain_steps": 3.0,
    "hrl_route_plan_routine_dynamic_reassignment_min_eta_gain_ratio": 0.20,
    "hrl_route_plan_routine_dynamic_reassignment_max_transfers": 1,
}

# Candidate-only initial emergency ordering override.  It keeps the formal
# spatial-overload guard intact, but lets a controlled pilot force the
# remaining-lifeline order at the first route construction.
ER_HLNS_FORCE_INITIAL_LIFELINE_OVERLAY = {
    "hrl_route_plan_force_initial_lifeline_ordering_enabled": True,
}

# Candidate-only risk-slack routine suffix repair. All fields are algorithm
# controls; public physical/task/road/safety fields remain inherited.
ER_HLNS_RISK_SLACK_ROUTINE_OVERLAY = {
    "hrl_route_plan_mixed_coverage_enabled": True,
    "hrl_route_plan_mixed_coverage_emergency_reserve_steps": 30,
    "hrl_route_plan_risk_slack_routine_repair_enabled": True,
    "hrl_route_plan_risk_slack_routine_reserved_inventory_guard_enabled": True,
    "hrl_route_plan_risk_slack_routine_slack_steps": 20,
    "hrl_route_plan_risk_slack_routine_stall_steps": 12,
    "hrl_route_plan_risk_slack_routine_max_transfers": 1,
    "hrl_route_plan_risk_slack_routine_eta_gain_steps": 3.0,
    "hrl_route_plan_risk_slack_routine_eta_gain_ratio": 0.20,
    "hrl_route_plan_risk_slack_routine_radius_m": 800.0,
}

# Candidate-only R4 stalled-routine takeover.  This deliberately keeps a
# narrow radius disabled (0 means any reachable node), but requires one
# stalled window, one transfer per task, and the existing stock/deadline/route
# safety checks in HierarchicalRoutePlanManager.
ER_HLNS_R4_ROUTINE_TAKEOVER_OVERLAY = {
    "hrl_route_plan_r4_routine_takeover_enabled": True,
    "hrl_route_plan_r4_routine_takeover_stall_steps": 12,
    "hrl_route_plan_r4_routine_takeover_max_transfers": 1,
    "hrl_route_plan_r4_routine_takeover_radius_m": 0.0,
}

# Candidate-only dispatch for a genuinely idle, stocked truck.  The reserve
# gate is intentionally conservative: an active emergency inside this window
# blocks the routine binding and no active contract is pre-empted.
ER_HLNS_IDLE_ROUTINE_DISPATCH_OVERLAY = {
    "hrl_route_plan_idle_routine_dispatch_enabled": True,
    "hrl_route_plan_idle_routine_dispatch_emergency_reserve_steps": 12,
    "hrl_route_plan_idle_routine_dispatch_max_per_step": 1,
    "hrl_route_plan_routine_service_start_rescue_enabled": True,
    "hrl_route_plan_routine_service_start_rescue_stall_steps": 10,
    "hrl_route_plan_routine_service_start_rescue_near_distance_m": 300.0,
    "hrl_route_plan_routine_service_start_rescue_max_transfers": 1,
}

# Candidate-only composition used by the LB smoke runner.  Keep the two
# source overlays named and independently testable while exposing one method
# id for the combined pilot.
ER_HLNS_IDLE_BALANCED_ROUTINE_OVERLAY = {
    **ER_HLNS_LB_BALANCED_OVERLAY,
    **ER_HLNS_IDLE_ROUTINE_DISPATCH_OVERLAY,
}

# Candidate-only aggressive LB pilot.  The route constructor reserves a
# direct-normal quota on each truck and starts with NORMAL work, then inserts
# emergency UAV contracts.  Execution uses the existing atomic rescue helper
# with a wider, one-transfer watchdog.  No physical/map/task field is
# modified; the formal ``er_hlns`` profile never receives this overlay.
ER_HLNS_BALANCED_ALL_TASKS_OVERLAY = {
    **ER_HLNS_IDLE_BALANCED_ROUTINE_OVERLAY,
    "hrl_route_plan_balanced_all_tasks_enabled": True,
    "hrl_route_plan_balanced_all_tasks_normal_first_enabled": True,
    "hrl_route_plan_balanced_all_tasks_max_normal_per_truck": 3,
    "hrl_route_plan_balanced_all_tasks_allow_emergency_tradeoff": True,
    "hrl_route_plan_balanced_all_tasks_emergency_lateness_tolerance_steps": 40,
    "hrl_route_plan_balanced_all_tasks_watchdog_stall_steps": 6,
    "hrl_route_plan_balanced_all_tasks_watchdog_near_distance_m": 1200.0,
    "hrl_route_plan_balanced_all_tasks_watchdog_max_transfers": 1,
    "hrl_route_plan_balanced_all_tasks_watchdog_transfer_min_gain_steps": 0.0,
    "hrl_route_plan_balanced_all_tasks_watchdog_transfer_min_gain_ratio": 0.0,
    # Existing rescue implementation consumes these knobs.  They are kept in
    # the overlay so the candidate remains a single reproducible package.
    "hrl_route_plan_routine_service_start_rescue_enabled": True,
    "hrl_route_plan_routine_service_start_rescue_stall_steps": 6,
    "hrl_route_plan_routine_service_start_rescue_near_distance_m": 1200.0,
    "hrl_route_plan_routine_service_start_rescue_max_transfers": 1,
    "hrl_route_plan_routine_service_start_rescue_allow_stalled_owner_transfer": True,
    "hrl_route_plan_routine_service_start_rescue_transfer_min_gain_steps": 0.0,
    "hrl_route_plan_routine_service_start_rescue_transfer_min_gain_ratio": 0.0,
    "hrl_route_plan_normal_max_emergency_delay_steps": 40,
}

# V2 keeps the balanced constructor but changes execution/reauction policy:
# the support truck is released only after a matching UAV is airborne, while
# every dynamic routine move must remain deadline-feasible.
ER_HLNS_BALANCED_ALL_TASKS_V2_OVERLAY = {
    **ER_HLNS_BALANCED_ALL_TASKS_OVERLAY,
    "hrl_route_plan_balanced_all_tasks_v2_enabled": True,
    "hrl_route_plan_balanced_all_tasks_v2_after_launch_only": True,
    "hrl_route_plan_balanced_all_tasks_v2_reauction_deadline_guard_enabled": True,
    "hrl_route_plan_balanced_all_tasks_v2_aggressive_pending_auction_enabled": True,
    "hrl_route_plan_parallel_routine_emergency_after_launch_enabled": True,
    "hrl_route_plan_routine_dynamic_reassignment_enabled": True,
    "hrl_route_plan_routine_dynamic_reassignment_radius_m": 1000000000.0,
    "hrl_route_plan_routine_dynamic_reassignment_min_eta_gain_steps": 0.0,
    "hrl_route_plan_routine_dynamic_reassignment_min_eta_gain_ratio": 0.0,
    "hrl_route_plan_routine_dynamic_reassignment_max_transfers": 1,
    "hrl_route_plan_routine_dynamic_reassignment_lock_steps": 3,
}

# V3 dual-plan selector.  It retains V2's post-launch physical gate but
# chooses the normal-first plan only when predicted emergency terms do not
# worsen against the same-state emergency-first construction.
ER_HLNS_BALANCED_ALL_TASKS_V3_OVERLAY = {
    **ER_HLNS_BALANCED_ALL_TASKS_V2_OVERLAY,
    "hrl_route_plan_balanced_all_tasks_v3_enabled": True,
    "hrl_route_plan_balanced_all_tasks_v3_tail_insert_after_launch": True,
}

# V4 keeps the V3 selector and corridor checks but permits a support truck to
# continue toward a verified NORMAL successor while its UAV is still docked.
# The corridor's existing payload, energy, recovery, and deadline gates remain
# authoritative; this only removes the V2 after-launch timing gate.
ER_HLNS_BALANCED_ALL_TASKS_V4_OVERLAY = {
    **ER_HLNS_BALANCED_ALL_TASKS_V3_OVERLAY,
    "hrl_route_plan_balanced_all_tasks_v2_after_launch_only": False,
}

# V5 reuses V4's pre-launch parallel corridor and adds one bounded initial
# route transform: a safe, still-pending emergency can be promoted to each
# truck's route cursor so the UAV launch is exposed before the NORMAL suffix.
ER_HLNS_BALANCED_ALL_TASKS_V5_OVERLAY = {
    **ER_HLNS_BALANCED_ALL_TASKS_V4_OVERLAY,
    "hrl_route_plan_balanced_all_tasks_v5_launch_first_enabled": True,
}

# Candidate-only LB hard-coverage pilot.  It opens the existing rolling
# planner's global normal-stall refresh and adds a bounded route-manager
# rescue for pending NORMAL contracts with no executable progress.
ER_HLNS_LB_HARD_COVERAGE_OVERLAY = {
    "hrl_normal_stall_hard_refresh_enabled": True,
    "hrl_normal_stall_local_only": False,
    "hrl_route_plan_stalled_normal_cleanup_enabled": True,
    "hrl_route_plan_hard_normal_rescue_enabled": True,
    "hrl_route_plan_hard_normal_rescue_stall_steps": 12,
    "hrl_route_plan_hard_normal_rescue_max_per_call": 2,
    "hrl_route_plan_hard_normal_rescue_airborne_parallel_enabled": True,
    "hrl_route_plan_hard_normal_rescue_tail_after_airborne": True,
    "hrl_route_plan_parallel_routine_emergency_after_launch_enabled": True,
}

# Candidate-only bounded variant: preserve every hard-coverage control but
# limit each planner call to a single NORMAL rescue transfer.
ER_HLNS_LB_HARD_COVERAGE_SINGLE_RESCUE_OVERLAY = {
    **ER_HLNS_LB_HARD_COVERAGE_OVERLAY,
    "hrl_route_plan_hard_normal_rescue_max_per_call": 1,
}

# Candidate-only protected variant.  It is intentionally a one-field
# perturbation of the hard-coverage pilot so its experiment isolates the
# ordinary-task protection policy without changing the physical environment.
ER_HLNS_LB_HARD_COVERAGE_PROTECTED_OVERLAY = {
    **ER_HLNS_LB_HARD_COVERAGE_OVERLAY,
    "erc_ablate_normal_protection": False,
}

# Candidate-only commitment variant.  It starts from the protected
# hard-coverage branch and enables the existing bounded truck-goal commitment
# and routine-disconnect guards; no formal/default setting is changed.
ER_HLNS_LB_HARD_COVERAGE_COMMITMENT_OVERLAY = {
    **ER_HLNS_LB_HARD_COVERAGE_PROTECTED_OVERLAY,
    "hrl_truck_normal_commit_guard2_enabled": True,
    "hrl_route_plan_routine_disconnect_protection_enabled": True,
}

# Candidate-only orphan-guard variant.  It deliberately starts from the
# hard-coverage overlay (rather than the protected/commitment branches) and
# changes only rescue eligibility/retry behavior.
ER_HLNS_LB_HARD_COVERAGE_ORPHAN_GUARD_OVERLAY = {
    **ER_HLNS_LB_HARD_COVERAGE_OVERLAY,
    "hrl_route_plan_hard_normal_rescue_orphan_only_enabled": True,
    "hrl_route_plan_hard_normal_rescue_pending_head_guard_enabled": True,
    "hrl_route_plan_hard_normal_rescue_candidate_head_guard_enabled": True,
    "hrl_route_plan_hard_normal_rescue_no_truck_once_enabled": True,
    "hrl_route_plan_hard_normal_rescue_no_truck_cooldown_enabled": True,
    "hrl_route_plan_hard_normal_rescue_no_truck_cooldown_steps": 24,
}

# Candidate-only light routine-protection branch.  This intentionally keeps
# hard NORMAL rescue disabled and changes only the requested ordinary-task
# protection/refresh controls.
ER_HLNS_LB_ROUTINE_PROTECTED_OVERLAY = {
    "erc_ablate_normal_protection": False,
    "erc_ablate_low_value_refresh": False,
    "hrl_normal_stall_hard_refresh_enabled": True,
    "hrl_normal_stall_local_only": False,
    "hrl_route_plan_stalled_normal_cleanup_enabled": True,
}

# Candidate-only bounded ownership repair.  It keeps the light routine
# protection branch and enables only the existing delayed unassigned/stalled
# ownership repairs; hard NORMAL rescue and emergency watchdogs stay off.
ER_HLNS_LB_ROUTINE_PROTECTED_OWNER_REPAIR_OVERLAY = {
    **ER_HLNS_LB_ROUTINE_PROTECTED_OVERLAY,
    "erc_stalled_routine_ownership_repair_enabled": True,
    "erc_stalled_routine_ownership_min_step": 36,
    "erc_stalled_routine_ownership_exposure_steps": 42,
    "erc_stalled_routine_ownership_max_repairs_per_step": 1,
    "erc_unassigned_routine_repair_enabled": True,
    "erc_unassigned_routine_repair_min_step": 70,
    "erc_unassigned_routine_repair_max_per_step": 1,
}

# Candidate-only hard-coverage safety gate.  All hard rescue controls remain
# inherited; only the parallel airborne/after-launch paths are disabled.
ER_HLNS_LB_HARD_COVERAGE_SAFETY_GATED_OVERLAY = {
    **ER_HLNS_LB_HARD_COVERAGE_OVERLAY,
    "hrl_route_plan_hard_normal_rescue_airborne_parallel_enabled": False,
    "hrl_route_plan_parallel_routine_emergency_after_launch_enabled": False,
    "hrl_route_plan_hard_normal_rescue_tail_after_airborne": False,
}

# Candidate-only emergency-rescue extension of the light routine-protected
# branch.  These are algorithm-side watchdog/queue controls only; all public
# physical and safety fields remain inherited from the scenario.
ER_HLNS_LB_ROUTINE_PROTECTED_EMERGENCY_RESCUE_OVERLAY = {
    **ER_HLNS_LB_ROUTINE_PROTECTED_OVERLAY,
    "hrl_route_plan_emergency_launch_watchdog_enabled": True,
    "hrl_route_plan_emergency_starvation_promotion_enabled": True,
    "hrl_route_plan_direct_safe_secondary_emergency_enabled": True,
    "hrl_route_plan_stalled_queue_rescue_enabled": True,
    "hrl_route_plan_stalled_queue_anchor_rescue_enabled": True,
}

# Candidate-only composition of the light routine-protected branch and the
# emergency-safe V3 dual-plan selector.  Hard NORMAL rescue remains off.
ER_HLNS_LB_ROUTINE_PROTECTED_V3_SELECTOR_OVERLAY = {
    **ER_HLNS_LB_ROUTINE_PROTECTED_OVERLAY,
    **ER_HLNS_BALANCED_ALL_TASKS_V3_OVERLAY,
}

# Candidate-only adaptive hard coverage.  It combines routine protection with
# hard rescue controls, but runtime-gates rescue on a minimum orphan-pending
# population so ordinary routes remain untouched in lightly orphaned states.
ER_HLNS_LB_ADAPTIVE_COVERAGE_OVERLAY = {
    **ER_HLNS_LB_ROUTINE_PROTECTED_OVERLAY,
    **ER_HLNS_LB_HARD_COVERAGE_OVERLAY,
    "hrl_route_plan_hard_normal_rescue_adaptive_coverage_enabled": True,
    "hrl_route_plan_hard_normal_rescue_min_orphan_pending": 3,
}

# Candidate-only adaptive single-rescue branch.  It keeps the same orphan
# threshold while limiting each planning call to one bounded transfer.
ER_HLNS_LB_ADAPTIVE_SINGLE_RESCUE_OVERLAY = {
    **ER_HLNS_LB_ROUTINE_PROTECTED_OVERLAY,
    **ER_HLNS_LB_HARD_COVERAGE_SINGLE_RESCUE_OVERLAY,
    "hrl_route_plan_hard_normal_rescue_adaptive_coverage_enabled": True,
    "hrl_route_plan_hard_normal_rescue_min_orphan_pending": 3,
}

# Candidate-only composition: retain the V3 emergency-safe dual-plan selector
# while enabling hard NORMAL coverage repair.  Unlike the aggressive hard
# coverage pilot, this variant never preempts an unlaunched emergency suffix.
ER_HLNS_LB_BALANCED_SELECTOR_OVERLAY = {
    **ER_HLNS_BALANCED_ALL_TASKS_V3_OVERLAY,
    **ER_HLNS_LB_HARD_COVERAGE_OVERLAY,
}

# V6 starts from the emergency-first route skeleton, then uses the same
# verified parallel corridor to let the truck continue to a NORMAL successor
# while the UAV executes the head emergency sortie.  This is intentionally a
# separate candidate: it tests whether the execution bottleneck, rather than
# the normal-first constructor, is causing routine starvation.
ER_HLNS_BALANCED_ALL_TASKS_V6_OVERLAY = {
    **ER_HLNS_BALANCED_ALL_TASKS_V4_OVERLAY,
    "hrl_route_plan_balanced_all_tasks_normal_first_enabled": False,
    "hrl_route_plan_balanced_all_tasks_v5_launch_first_enabled": False,
}

# V7 keeps V6's balanced/no-UAV-priority upper plan.  During execution it
# permits only a road-impact emergency promotion: the current NORMAL head
# must suffer a material ETA increase and a pending emergency suffix must
# already be inside its lifeline/deadline reserve.  Static emergency priority
# is disabled for this candidate so the trigger remains attributable.
ER_HLNS_BALANCED_ALL_TASKS_V7_OVERLAY = {
    **ER_HLNS_BALANCED_ALL_TASKS_V6_OVERLAY,
    "hrl_route_plan_conditional_road_emergency_promotion_enabled": True,
    "hrl_route_plan_conditional_road_emergency_promotion_eta_increase_steps": 12,
    "hrl_route_plan_conditional_road_emergency_promotion_near_normal_distance_m": 1000.0,
    "hrl_route_plan_conditional_road_emergency_promotion_reserve_steps": 8,
    "hrl_route_plan_conditional_road_emergency_promotion_cooldown_steps": 24,
    "hrl_route_plan_conditional_road_emergency_promotion_min_gain_steps": 6,
    "hrl_route_plan_emergency_starvation_promotion_enabled": False,
}

# Candidate-only structural shadow selector.  It starts from V6's
# emergency-first/no-UAV-priority policy, builds a normal-first alternative
# from the same observed state, and publishes it only when emergency metrics
# are non-worse and at least one additional routine task is predicted on time.
ER_HLNS_BALANCED_ALL_TASKS_V8_OVERLAY = {
    **ER_HLNS_BALANCED_ALL_TASKS_V6_OVERLAY,
    "hrl_route_plan_shadow_total_coverage_enabled": True,
    "hrl_route_plan_shadow_total_coverage_min_gain_tasks": 1,
    "hrl_route_plan_shadow_total_coverage_min_routine_slack_steps": 24,
    "hrl_route_plan_shadow_total_coverage_max_routine_distance_ratio": 1.10,
}


def _is_algorithm_owned_field(name: str) -> bool:
    field_name = str(name)
    if field_name in _COMMON_SAFETY_FIELDS:
        return False
    return bool(
        field_name in _ALGORITHM_FIELD_NAMES
        or any(field_name.startswith(prefix) for prefix in _ALGORITHM_FIELD_PREFIXES)
    )


def _stable_cfg_hash(payload: Dict[str, Any]) -> str:
    text = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cfg_payload(cfg: EnvConfig, *, algorithm_owned: bool) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for field in fields(cfg):
        name = str(field.name)
        if _is_algorithm_owned_field(name) != bool(algorithm_owned):
            continue
        payload[name] = getattr(cfg, name)
    return payload


def _build_algorithm_package(
    method: str,
    cfg: EnvConfig,
    seed: int,
    use_event_trigger_override: Optional[bool] = None,
    use_risk_term_override: Optional[bool] = None,
    use_rth_repair_override: Optional[bool] = None,
    enable_rth_mask_override: Optional[bool] = None,
    encoder_type_override: str = "",
) -> AlgorithmPackage:
    """Build one isolated algorithm package against an immutable public world.

    The caller supplies a scenario configuration. Algorithm-specific overlays
    are private to the returned package. Any accidental change to a public
    map/task/hazard/vehicle/safety field is restored and verified before the
    environment is constructed.
    """

    algorithm_id = _canonical_method_name(method)
    candidate_c_clean = bool(algorithm_id == C_ALNS_CLEAN_METHOD)
    candidate_risk_slack = bool(
        algorithm_id == ER_HLNS_RISK_SLACK_ROUTINE_METHOD
    )
    candidate_force_initial_lifeline = bool(
        algorithm_id == ER_HLNS_FORCE_INITIAL_LIFELINE_METHOD
    )
    candidate_r4 = bool(
        algorithm_id == ER_HLNS_R4_ROUTINE_TAKEOVER_METHOD
    )
    candidate_idle_routine_dispatch = bool(
        algorithm_id == ER_HLNS_IDLE_ROUTINE_DISPATCH_METHOD
    )
    candidate_idle_balanced_routine = bool(
        algorithm_id == ER_HLNS_IDLE_BALANCED_ROUTINE_METHOD
    )
    candidate_balanced_all_tasks = bool(
        algorithm_id == ER_HLNS_BALANCED_ALL_TASKS_METHOD
    )
    candidate_balanced_all_tasks_v2 = bool(
        algorithm_id == ER_HLNS_BALANCED_ALL_TASKS_V2_METHOD
    )
    candidate_balanced_all_tasks_v3 = bool(
        algorithm_id == ER_HLNS_BALANCED_ALL_TASKS_V3_METHOD
    )
    candidate_balanced_all_tasks_v4 = bool(
        algorithm_id == ER_HLNS_BALANCED_ALL_TASKS_V4_METHOD
    )
    candidate_balanced_all_tasks_v5 = bool(
        algorithm_id == ER_HLNS_BALANCED_ALL_TASKS_V5_METHOD
    )
    candidate_lb_hard_coverage = bool(
        algorithm_id == ER_HLNS_LB_HARD_COVERAGE_METHOD
    )
    candidate_lb_hard_coverage_single_rescue = bool(
        algorithm_id == ER_HLNS_LB_HARD_COVERAGE_SINGLE_RESCUE_METHOD
    )
    candidate_lb_hard_coverage_protected = bool(
        algorithm_id == ER_HLNS_LB_HARD_COVERAGE_PROTECTED_METHOD
    )
    candidate_lb_hard_coverage_commitment = bool(
        algorithm_id == ER_HLNS_LB_HARD_COVERAGE_COMMITMENT_METHOD
    )
    candidate_lb_hard_coverage_orphan_guard = bool(
        algorithm_id == ER_HLNS_LB_HARD_COVERAGE_ORPHAN_GUARD_METHOD
    )
    candidate_lb_routine_protected = bool(
        algorithm_id == ER_HLNS_LB_ROUTINE_PROTECTED_METHOD
    )
    candidate_lb_routine_protected_owner_repair = bool(
        algorithm_id == ER_HLNS_LB_ROUTINE_PROTECTED_OWNER_REPAIR_METHOD
    )
    candidate_lb_hard_coverage_safety_gated = bool(
        algorithm_id == ER_HLNS_LB_HARD_COVERAGE_SAFETY_GATED_METHOD
    )
    candidate_lb_routine_protected_emergency_rescue = bool(
        algorithm_id == ER_HLNS_LB_ROUTINE_PROTECTED_EMERGENCY_RESCUE_METHOD
    )
    candidate_lb_routine_protected_v3_selector = bool(
        algorithm_id == ER_HLNS_LB_ROUTINE_PROTECTED_V3_SELECTOR_METHOD
    )
    candidate_lb_adaptive_coverage = bool(
        algorithm_id == ER_HLNS_LB_ADAPTIVE_COVERAGE_METHOD
    )
    candidate_lb_adaptive_single_rescue = bool(
        algorithm_id == ER_HLNS_LB_ADAPTIVE_SINGLE_RESCUE_METHOD
    )
    candidate_lb_balanced_selector = bool(
        algorithm_id == ER_HLNS_LB_BALANCED_SELECTOR_METHOD
    )
    candidate_balanced_all_tasks_v6 = bool(
        algorithm_id == ER_HLNS_BALANCED_ALL_TASKS_V6_METHOD
    )
    candidate_balanced_all_tasks_v7 = bool(
        algorithm_id == ER_HLNS_BALANCED_ALL_TASKS_V7_METHOD
    )
    candidate_balanced_all_tasks_v8 = bool(
        algorithm_id == ER_HLNS_BALANCED_ALL_TASKS_V8_METHOD
    )
    candidate_parallel_routine_emergency = bool(
        algorithm_id == ER_HLNS_PARALLEL_ROUTINE_EMERGENCY_METHOD
        or candidate_idle_balanced_routine
        or candidate_balanced_all_tasks_v2
        or candidate_balanced_all_tasks_v3
        or candidate_balanced_all_tasks_v4
        or candidate_balanced_all_tasks_v5
        or candidate_lb_routine_protected_v3_selector
        or candidate_lb_adaptive_coverage
        or candidate_lb_adaptive_single_rescue
        or candidate_lb_balanced_selector
        or candidate_balanced_all_tasks_v6
        or candidate_balanced_all_tasks_v7
        or candidate_balanced_all_tasks_v8
    )
    candidate_parallel_lb = bool(
        candidate_parallel_routine_emergency
        and not candidate_balanced_all_tasks_v2
        and not candidate_balanced_all_tasks_v3
        and not candidate_balanced_all_tasks_v4
        and not candidate_balanced_all_tasks_v5
        and not candidate_lb_routine_protected_v3_selector
        and not candidate_lb_adaptive_coverage
        and not candidate_lb_adaptive_single_rescue
        and not candidate_lb_balanced_selector
        and not candidate_balanced_all_tasks_v6
        and not candidate_balanced_all_tasks_v7
        and not candidate_balanced_all_tasks_v8
        and str(getattr(cfg, "scenario", "")).upper().strip() == "B"
        and str(getattr(cfg, "map_complexity", "")).upper().strip() == "L"
    )
    owns_er_hlns = bool(
        algorithm_id == ER_HLNS_METHOD
        or candidate_risk_slack
        or candidate_force_initial_lifeline
        or candidate_r4
        or candidate_idle_routine_dispatch
        or candidate_idle_balanced_routine
        or candidate_balanced_all_tasks
        or candidate_balanced_all_tasks_v2
        or candidate_balanced_all_tasks_v3
        or candidate_balanced_all_tasks_v4
        or candidate_balanced_all_tasks_v5
        or candidate_lb_hard_coverage
        or candidate_lb_hard_coverage_single_rescue
        or candidate_lb_hard_coverage_protected
        or candidate_lb_hard_coverage_commitment
        or candidate_lb_hard_coverage_orphan_guard
        or candidate_lb_routine_protected
        or candidate_lb_routine_protected_owner_repair
        or candidate_lb_hard_coverage_safety_gated
        or candidate_lb_routine_protected_emergency_rescue
        or candidate_lb_routine_protected_v3_selector
        or candidate_lb_adaptive_coverage
        or candidate_lb_balanced_selector
        or candidate_balanced_all_tasks_v6
        or candidate_balanced_all_tasks_v7
        or candidate_balanced_all_tasks_v8
        or candidate_parallel_routine_emergency
        or algorithm_id in ER_HLNS_ABLATION_OVERRIDES
    )
    public_payload = _cfg_payload(cfg, algorithm_owned=False)
    public_hash = _stable_cfg_hash(public_payload)

    private_cfg = replace(
        cfg,
        hrl_route_plan_v2_enabled=bool(owns_er_hlns),
        **(C_ALNS_CLEAN_OVERLAY if candidate_c_clean else {}),
        **(ER_HLNS_RISK_SLACK_ROUTINE_OVERLAY if candidate_risk_slack else {}),
        **(
            ER_HLNS_FORCE_INITIAL_LIFELINE_OVERLAY
            if candidate_force_initial_lifeline
            else {}
        ),
        **(ER_HLNS_R4_ROUTINE_TAKEOVER_OVERLAY if candidate_r4 else {}),
        **(
            ER_HLNS_IDLE_ROUTINE_DISPATCH_OVERLAY
            if (
                candidate_idle_routine_dispatch
                or candidate_idle_balanced_routine
            )
            else {}
        ),
        **(
            ER_HLNS_LB_BALANCED_OVERLAY if candidate_parallel_lb else {}
        ),
        **(
            ER_HLNS_BALANCED_ALL_TASKS_OVERLAY
            if candidate_balanced_all_tasks
            else {}
        ),
        **(
            ER_HLNS_BALANCED_ALL_TASKS_V2_OVERLAY
            if candidate_balanced_all_tasks_v2
            else {}
        ),
        **(
            ER_HLNS_BALANCED_ALL_TASKS_V3_OVERLAY
            if candidate_balanced_all_tasks_v3
            else {}
        ),
        **(
            ER_HLNS_BALANCED_ALL_TASKS_V4_OVERLAY
            if candidate_balanced_all_tasks_v4
            else {}
        ),
        **(
            ER_HLNS_BALANCED_ALL_TASKS_V5_OVERLAY
            if candidate_balanced_all_tasks_v5
            else {}
        ),
        **(
            ER_HLNS_LB_HARD_COVERAGE_OVERLAY
            if candidate_lb_hard_coverage
            else {}
        ),
        **(
            ER_HLNS_LB_HARD_COVERAGE_SINGLE_RESCUE_OVERLAY
            if candidate_lb_hard_coverage_single_rescue
            else {}
        ),
        **(
            ER_HLNS_LB_HARD_COVERAGE_PROTECTED_OVERLAY
            if candidate_lb_hard_coverage_protected
            else {}
        ),
        **(
            ER_HLNS_LB_HARD_COVERAGE_COMMITMENT_OVERLAY
            if candidate_lb_hard_coverage_commitment
            else {}
        ),
        **(
            ER_HLNS_LB_HARD_COVERAGE_ORPHAN_GUARD_OVERLAY
            if candidate_lb_hard_coverage_orphan_guard
            else {}
        ),
        **(
            ER_HLNS_LB_ROUTINE_PROTECTED_OVERLAY
            if candidate_lb_routine_protected
            else {}
        ),
        **(
            ER_HLNS_LB_ROUTINE_PROTECTED_OWNER_REPAIR_OVERLAY
            if candidate_lb_routine_protected_owner_repair
            else {}
        ),
        **(
            ER_HLNS_LB_HARD_COVERAGE_SAFETY_GATED_OVERLAY
            if candidate_lb_hard_coverage_safety_gated
            else {}
        ),
        **(
            ER_HLNS_LB_ROUTINE_PROTECTED_EMERGENCY_RESCUE_OVERLAY
            if candidate_lb_routine_protected_emergency_rescue
            else {}
        ),
        **(
            ER_HLNS_LB_ROUTINE_PROTECTED_V3_SELECTOR_OVERLAY
            if candidate_lb_routine_protected_v3_selector
            else {}
        ),
        **(
            ER_HLNS_LB_ADAPTIVE_COVERAGE_OVERLAY
            if candidate_lb_adaptive_coverage
            else {}
        ),
        **(
            ER_HLNS_LB_ADAPTIVE_SINGLE_RESCUE_OVERLAY
            if candidate_lb_adaptive_single_rescue
            else {}
        ),
        **(
            ER_HLNS_LB_BALANCED_SELECTOR_OVERLAY
            if candidate_lb_balanced_selector
            else {}
        ),
        **(
            ER_HLNS_BALANCED_ALL_TASKS_V6_OVERLAY
            if candidate_balanced_all_tasks_v6
            else {}
        ),
        **(
            ER_HLNS_BALANCED_ALL_TASKS_V7_OVERLAY
            if candidate_balanced_all_tasks_v7
            else {}
        ),
        **(
            ER_HLNS_BALANCED_ALL_TASKS_V8_OVERLAY
            if candidate_balanced_all_tasks_v8
            else {}
        ),
    )
    # E0 search-budget parity: all search-based baselines receive the same
    # configured outer iteration count as the frozen ER-HLNS route LNS. Exact
    # objective-evaluation counts remain exported for the later calibration
    # gate because one iteration has method-specific internal work.
    if algorithm_id in {
        HYBRID_GENETIC_METHOD,
        VNS_METHOD,
        CANONICAL_ALNS_METHOD,
        ROLLING_HORIZON_ALNS_METHOD,
        DYNAMIC_REPLANNING_ALNS_METHOD,
        TABU_SEARCH_METHOD,
    }:
        private_cfg = replace(
            private_cfg,
            alns_iterations=int(
                max(getattr(cfg, "hrl_route_plan_alns_iterations", 4), 1)
            ),
        )
    builder_method = (
        ROLLING_HORIZON_ALNS_METHOD
        if candidate_c_clean
        else (
            ER_HLNS_METHOD
            if (
                candidate_risk_slack
                or candidate_force_initial_lifeline
                or candidate_r4
                or candidate_idle_routine_dispatch
                or candidate_idle_balanced_routine
                or candidate_balanced_all_tasks
                or candidate_balanced_all_tasks_v2
                or candidate_balanced_all_tasks_v3
                or candidate_balanced_all_tasks_v4
                 or candidate_balanced_all_tasks_v5
                 or candidate_lb_hard_coverage
                 or candidate_lb_hard_coverage_single_rescue
                 or candidate_lb_hard_coverage_protected
                 or candidate_lb_hard_coverage_commitment
                 or candidate_lb_hard_coverage_orphan_guard
                 or candidate_lb_routine_protected
                 or candidate_lb_routine_protected_owner_repair
                  or candidate_lb_hard_coverage_safety_gated
                  or candidate_lb_routine_protected_emergency_rescue
                  or candidate_lb_routine_protected_v3_selector
                  or candidate_lb_adaptive_coverage
                  or candidate_lb_adaptive_single_rescue
                  or candidate_lb_balanced_selector
                 or candidate_balanced_all_tasks_v6
                 or candidate_balanced_all_tasks_v7
                 or candidate_balanced_all_tasks_v8
                or candidate_parallel_routine_emergency
            )
            else algorithm_id
        )
    )
    if owns_er_hlns:
        builder_method = ER_HLNS_METHOD
        overrides = dict(ER_HLNS_ABLATION_OVERRIDES.get(algorithm_id, {}))
        if overrides:
            private_cfg = replace(private_cfg, **overrides)

    planner, encoder_type, enable_rth_mask, runtime_cfg = _build_planner(
        builder_method,
        private_cfg,
        int(seed),
        use_event_trigger_override=use_event_trigger_override,
        use_risk_term_override=use_risk_term_override,
        use_rth_repair_override=use_rth_repair_override,
        enable_rth_mask_override=enable_rth_mask_override,
        encoder_type_override=encoder_type_override,
    )

    # A legacy builder may still carry method-specific safety defaults. Formal
    # packages restore every public field from the paired scenario config.
    runtime_cfg = replace(runtime_cfg, **public_payload)
    # The B/L ER-HLNS builder intentionally disables several watchdogs for the
    # formal mainline. Reapply this explicit candidate overlay after that
    # conservative builder normalization so the candidate actually exercises
    # its requested algorithm-side rescue controls.
    if candidate_lb_routine_protected_emergency_rescue:
        runtime_cfg = replace(
            runtime_cfg,
            **ER_HLNS_LB_ROUTINE_PROTECTED_EMERGENCY_RESCUE_OVERLAY,
        )
    if candidate_lb_routine_protected_owner_repair:
        runtime_cfg = replace(
            runtime_cfg,
            **ER_HLNS_LB_ROUTINE_PROTECTED_OWNER_REPAIR_OVERLAY,
        )
    if _stable_cfg_hash(_cfg_payload(runtime_cfg, algorithm_owned=False)) != public_hash:
        raise AssertionError(
            f"algorithm {algorithm_id} mutated the public environment configuration"
        )

    planner_class = (
        f"{planner.__class__.__module__}.{planner.__class__.__qualname__}"
    )
    profile = AlgorithmProfile(
        algorithm_id=str(algorithm_id),
        planner_class=str(planner_class),
        capabilities={
            ER_HLNS_ROUTE_PLAN_CAPABILITY: bool(owns_er_hlns),
            ER_HLNS_COORDINATION_CAPABILITY: bool(owns_er_hlns),
            ER_HLNS_PARALLEL_ROUTINE_EMERGENCY_CAPABILITY: bool(
                algorithm_id == ER_HLNS_PARALLEL_ROUTINE_EMERGENCY_METHOD
                or candidate_idle_balanced_routine
                or candidate_balanced_all_tasks_v2
                or candidate_balanced_all_tasks_v3
                or candidate_balanced_all_tasks_v4
                or candidate_balanced_all_tasks_v5
                or candidate_lb_hard_coverage
                or candidate_lb_hard_coverage_single_rescue
                or candidate_lb_hard_coverage_protected
                or candidate_lb_hard_coverage_commitment
                or candidate_lb_hard_coverage_orphan_guard
                or candidate_lb_routine_protected
                or candidate_lb_hard_coverage_safety_gated
                or candidate_lb_routine_protected_v3_selector
                or candidate_lb_adaptive_coverage
                or candidate_lb_adaptive_single_rescue
                or candidate_balanced_all_tasks_v6
                or candidate_balanced_all_tasks_v7
                or candidate_balanced_all_tasks_v8
            ),
            ER_HLNS_R4_ROUTINE_TAKEOVER_CAPABILITY: bool(candidate_r4),
            ER_HLNS_IDLE_ROUTINE_DISPATCH_CAPABILITY: bool(
                candidate_idle_routine_dispatch
                or candidate_idle_balanced_routine
            ),
            ER_HLNS_BALANCED_ALL_TASKS_CAPABILITY: bool(
                candidate_balanced_all_tasks
                or candidate_balanced_all_tasks_v2
                or candidate_balanced_all_tasks_v3
                or candidate_balanced_all_tasks_v4
                or candidate_balanced_all_tasks_v5
                or candidate_lb_routine_protected_v3_selector
                or candidate_balanced_all_tasks_v6
                or candidate_balanced_all_tasks_v7
                or candidate_balanced_all_tasks_v8
            ),
            ER_HLNS_BALANCED_ALL_TASKS_V2_CAPABILITY: bool(
                candidate_balanced_all_tasks_v2
                or candidate_balanced_all_tasks_v3
                or candidate_balanced_all_tasks_v4
                or candidate_balanced_all_tasks_v5
                or candidate_lb_routine_protected_v3_selector
                or candidate_balanced_all_tasks_v6
                or candidate_balanced_all_tasks_v7
                or candidate_balanced_all_tasks_v8
            ),
            ER_HLNS_BALANCED_ALL_TASKS_V3_CAPABILITY: bool(
                candidate_balanced_all_tasks_v3
                or candidate_balanced_all_tasks_v4
                or candidate_balanced_all_tasks_v5
                or candidate_lb_balanced_selector
                or candidate_lb_routine_protected_v3_selector
                or candidate_balanced_all_tasks_v6
                or candidate_balanced_all_tasks_v7
                or candidate_balanced_all_tasks_v8
            ),
            ER_HLNS_B_PRELAUNCH_CONTRACT_LOCK_CAPABILITY: bool(
                algorithm_id == ER_HLNS_METHOD
                and str(getattr(cfg, "scenario", "")).upper().strip() == "B"
                and str(getattr(cfg, "map_complexity", "")).upper().strip()
                in {"L", "R"}
                and bool(
                    getattr(
                        cfg, "hrl_b_prelaunch_contract_lock_enabled", False
                    )
                )
            ),
            ER_HLNS_B_DOCKED_LATCH_REARM_CAPABILITY: bool(
                algorithm_id == ER_HLNS_METHOD
                and str(getattr(cfg, "scenario", "")).upper().strip() == "B"
                and str(getattr(cfg, "map_complexity", "")).upper().strip()
                in {"L", "R"}
                and bool(
                    getattr(cfg, "hrl_b_docked_latch_rearm_enabled", False)
                )
            ),
            UAV_SCOUT_INFORMATION_CAPABILITY: algorithm_id not in {
                "er_hlns_no_uav_scout",
                "er_hlns_no_uav_scout_information",
            },
        },
    )
    return AlgorithmPackage(
        algorithm_id=str(algorithm_id),
        planner=planner,
        planner_class=str(planner_class),
        backend_family=str(_method_backend_family(algorithm_id)),
        encoder_type=str(encoder_type),
        enable_rth_mask=bool(enable_rth_mask),
        runtime_cfg=runtime_cfg,
        profile=profile,
        public_environment_hash=str(public_hash),
        # Include the resolved planner identity as well as algorithm-owned
        # configuration.  PG and nearest-task Greedy may share physical
        # fields, but they must never collapse to the same algorithm hash.
        algorithm_config_hash=_stable_cfg_hash(
            {
                "algorithm_id": str(algorithm_id),
                "planner_class": str(planner_class),
                "backend_family": str(_method_backend_family(algorithm_id)),
                "algorithm_cfg": _cfg_payload(runtime_cfg, algorithm_owned=True),
            }
        ),
    )


def _method_ablation_flags(method: str) -> Dict[str, bool]:
    m = _canonical_method_name(method)
    flags = {
        "ablation_low_value_refresh_enabled": False,
        "ablation_map_ranking_refresh_enabled": False,
        "ablation_tc_global_assignment_enabled": False,
        "ablation_support_chain_enabled": False,
        "ablation_cluster_primary_reservation_enabled": False,
        "ablation_event_scoring_bonus_enabled": False,
        "ablation_normal_protection_enabled": False,
    }
    if m in {
        "erc_rhc",
        "erc_rhc_old",
        "erc_rhc_current",
        "erc_base_no_event",
        "erc_support_authorized",
        "erc_routine_commit",
        "erc_launch_quality_gate",
        "erc_tc_completion_chain",
        "erc_event_minimal_local",
        "erc_scoring_shrink",
        "erc_combined_safe_core",
        "erc_only_uav_commit",
        "erc_only_truck_escape",
        "erc_only_eta_exit",
        "erc_routine_rescue_combo",
        "erc_km_hard_routine",
        "erc_km_hard_routine_v2",
        "erc_same_config",
        ALNS_MAINLINE_METHOD,
        "erc_ya_balanced",
        "erc_mc_lc_boost",
        "erc_ya_km_bridge",
        "erc_mc_lc_safe",
        "erc_ya_tune_a",
        "erc_ya_tune_b",
        "erc_ya_tune_c",
        "erc_support_recovery_launch",
        "erc_support_recovery_launch_relaxed",
        "erc_support_recovery_bound",
        "erc_support_recovery_budget",
        "erc_support_recovery_budget_strict",
        "erc_support_recovery_budget_urgent",
        "erc_full_new",
        "erc_reservation_only",
        "erc_airborne_lock_only",
        "erc_truck_assist_only",
        "erc_reservation_lock_no_assist",
    }:
        # Simplified default ERC-RHC profile.
        flags["ablation_low_value_refresh_enabled"] = True
        flags["ablation_map_ranking_refresh_enabled"] = True
        flags["ablation_tc_global_assignment_enabled"] = True
        flags["ablation_normal_protection_enabled"] = True
    elif m == "erc_hard_events_only":
        flags["ablation_low_value_refresh_enabled"] = True
        flags["ablation_map_ranking_refresh_enabled"] = True
    elif m == "erc_no_map_ranking_refresh":
        flags["ablation_map_ranking_refresh_enabled"] = True
    elif m == "erc_no_tc_global_assignment":
        flags["ablation_tc_global_assignment_enabled"] = True
    elif m == "erc_no_support_chain":
        flags["ablation_support_chain_enabled"] = True
    elif m == "erc_no_cluster_primary_reservation":
        flags["ablation_cluster_primary_reservation_enabled"] = True
    elif m == "erc_no_event_scoring_bonus":
        flags["ablation_event_scoring_bonus_enabled"] = True
    elif m == "erc_no_normal_protection":
        flags["ablation_normal_protection_enabled"] = True
    return flags
