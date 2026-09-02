from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Mapping, Optional, Tuple, Union

import numpy as np


class AgentKind(str, Enum):
    TRUCK = "truck"
    UAV = "uav"


class TaskKind(str, Enum):
    NORMAL = "normal"
    EMERGENCY = "emergency"


class TaskClass(str, Enum):
    ROUTINE_BULK = "routine_bulk"
    TIME_CRITICAL_LIGHTWEIGHT = "time_critical_lightweight"


class TaskStatus(str, Enum):
    PENDING = "pending"
    CLAIMED = "claimed"
    DELIVERED = "delivered"
    FAILED = "failed"


@dataclass(frozen=True)
class EnvConfig:
    # Legacy-compatible names (from previous framework)
    n_nodes: int = 40
    n_trucks: int = 2
    n_uavs: int = 3
    n_normal_tasks: int = 6
    n_emergency_tasks: int = 4
    # Task deadline templates (step index based).
    normal_task_deadline_start_step: int = 120
    normal_task_deadline_interval_step: int = 5
    emergency_task_deadline_start_step: int = 80
    emergency_task_deadline_interval_step: int = 4
    dt: float = 20.0

    phase: str = "S"
    scenario: str = "B"  # A:楠炲磭菙閺冪姷浼ㄧ€? B:妞嬪酣娲﹂崷浼存缚, C:妞嬪酣娲﹂崷浼存缚+闁矮淇婄仦蹇氭杸
    map_complexity: str = "M"
    seed: int = 0
    num_nodes: int = 40
    num_edges: int = 64
    avg_degree_min: float = 3.2
    avg_degree_max: float = 3.8
    map_source: str = "disaster_map"  # disaster_map | synthetic | osm_dem
    # Real-city benchmark mode: keep synthetic L intact and expose a separate
    # real-road case chain (e.g., Dujiangyan) backed by OSM GraphML + optional DEM.
    real_case_enabled: bool = False
    real_city_case: str = ""
    real_case_name: str = ""
    real_case_bbox_mode: str = "center_size"
    real_case_center_lon: float = 103.61941
    real_case_center_lat: float = 30.99825
    real_case_size_m: float = 15000.0
    real_case_keep_intersection_level: bool = True
    real_case_cleaning_profile: str = "dujiangyan_v1"
    real_case_task_sampling_profile: str = "dujiangyan_relief_v1"
    real_case_hazard_profile: str = "wenchuan_frontline_v1"
    # Real-case road cleaning / abstraction profile.
    real_case_use_prepared_clean_graph: bool = True
    real_case_prepared_graphml_path: str = ""
    real_case_poi_json_path: str = ""
    # Paper-grade RB inputs are opt-in so legacy R and all synthetic M/L
    # configurations retain their original generation paths.
    real_case_fixed_tasks_json_path: str = ""
    earthquake_field_mode: str = "legacy_proxy"  # legacy_proxy | usgs_shakemap
    rb_road_damage_mode: str = "legacy_mixed"  # legacy_mixed | earthquake_only
    real_case_truck_only_road_cleaning: bool = True
    real_case_min_leaf_edge_m: float = 260.0
    real_case_postmerge_leaf_edge_m: float = 360.0
    real_case_chain_collapse_angle_deg: float = 145.0
    real_case_junction_merge_cell_m: float = 260.0
    map_size_m: float = 5000.0
    min_node_spacing_m: float = 300.0
    redundant_edge_radius_m: float = 850.0
    redundant_edge_prob: float = 0.55
    # L map is a mesoscopic decision graph (not intersection-level GIS graph).
    l_map_generation_max_attempts: int = 8
    l_map_variant: str = "L_v1b_orientation_tighter"
    # L acceptance priority: realism_first prefers structural realism constraints
    # over strict historical degree-shape targets.
    l_map_acceptance_mode: str = "realism_first"
    # L benchmark source selector for difficulty attribution:
    # - old: legacy/easier L distribution used by earlier benchmark rounds
    # - new: frozen realism-first L benchmark (paper default)
    l_benchmark_mode: str = "new"
    # Optional L map cache: when enabled, topology may load pre-generated maps
    # from cache directory (seed-mod indexing by default).
    l_map_cache_enabled: bool = False
    l_map_cache_dir: str = "data/maps_core/L_map"
    l_map_cache_size: int = 30
    l_map_cache_index: int = -1
    # Use L-map realism metadata for synthetic task placement. This affects only
    # benchmark generation, not any planner/algorithm logic.
    synthetic_realism_task_sampling_enabled: bool = False
    synthetic_realism_task_sampling_min_map_size_m: float = 10000.0
    # L realism spacing hard constraints.
    l_min_node_spacing_m: float = 220.0
    l_min_gateway_spacing_m: float = 300.0
    l_min_arterial_junction_spacing_m: float = 350.0
    l_collinear_triangle_angle_deg: float = 165.0
    l_task_min_spacing_m: float = 240.0
    l_target_num_nodes_min: int = 320
    l_target_num_nodes_max: int = 380
    l_target_num_edges_min: int = 460
    l_target_num_edges_max: int = 540
    l_target_avg_degree_min: float = 2.8
    l_target_avg_degree_max: float = 3.3
    l_target_median_edge_length_m_min: float = 450.0
    l_target_median_edge_length_m_max: float = 800.0
    l_target_p90_edge_length_m_min: float = 1800.0
    l_target_p90_edge_length_m_max: float = 3200.0
    l_target_leaf_fraction_max: float = 0.08
    l_target_deg3_fraction_min: float = 0.55
    l_target_deg3_fraction_max: float = 0.70
    l_target_deg4_fraction_min: float = 0.18
    l_target_deg4_fraction_max: float = 0.30
    l_target_deg_gt4_fraction_max: float = 0.08
    l_target_arterial_length_share_min: float = 0.10
    l_target_arterial_length_share_max: float = 0.15
    l_target_collector_length_share_min: float = 0.25
    l_target_collector_length_share_max: float = 0.35
    l_target_local_length_share_min: float = 0.50
    l_target_local_length_share_max: float = 0.60
    l_target_max_crossing_fraction: float = 0.05
    l_target_main_orientation_modes: int = 2
    l_target_off_axis_edge_fraction_min: float = 0.20
    l_target_off_axis_edge_fraction_max: float = 0.35
    l_target_builtup_area_fraction_min: float = 0.40
    l_target_builtup_area_fraction_max: float = 0.55
    l_target_barrier_area_fraction_min: float = 0.08
    l_target_barrier_area_fraction_max: float = 0.18
    osm_graphml_path: str = ""
    dem_npy_path: str = ""
    num_trucks: int = 2
    num_uavs: int = 3
    num_normal_tasks: int = 6
    num_emergency_tasks: int = 4
    # Paper semantic names (legacy aliases are kept for compatibility).
    # If >0, these take priority over num_normal_tasks/num_emergency_tasks.
    num_routine_bulk_tasks: int = 0
    num_time_critical_lightweight_tasks: int = 0
    # Force-create island emergency tasks by physically isolating selected
    # emergency demand nodes from road graph (UAV reachable, truck unreachable).
    forced_island_emergency_tasks: int = 0
    # Keep forced island cut-edges blocked throughout episode even if repair is enabled.
    forced_island_lock_edges: bool = True
    # Extra deadline slack for forced island emergency tasks (UAV-only access).
    forced_island_deadline_extension_steps: int = 24
    max_steps: int = 300
    dt_seconds: float = 20.0
    hrl_interval: int = 5
    risk_spike_threshold: float = 0.30
    # Truck physics
    truck_speed_mps: float = 8.0
    # Legacy experiments may opt out explicitly; paper-facing configurations
    # set this to False and are guarded by the kg invariants in __post_init__.
    ignore_payload_constraints: bool = True
    truck_payload_kg: float = 3000.0
    # Cargo unit conversion and capacity-demand scale:
    # 1.0 abstract cargo unit == cargo_unit_kg kilograms.
    cargo_unit_kg: float = 200.0
    truck_cargo_capacity_units: float = 15.0  # 3000 kg
    uav_cargo_capacity_units: float = 0.25    # 50 kg
    task_demand_normal_units: float = 4.0     # 800 kg
    task_demand_emergency_units: float = 0.25 # 50 kg
    replenish_freeze_steps: int = 2
    # Minimal paper-grade material-constraint model (single-unit task demand, no partial delivery).
    normal_task_demand_kg: float = 800.0
    emergency_task_demand_kg: float = 50.0
    # Canonical paper package is one 50 kg FC100 dual-battery operating unit.
    # Larger values remain readable only when an archived experiment overrides it.
    time_critical_package_kg: float = 50.0
    # Semantic demand defaults.
    routine_bulk_demand_kg_min: float = 800.0
    routine_bulk_demand_kg_max: float = 800.0
    time_critical_lightweight_demand_kg_min: float = 50.0
    time_critical_lightweight_demand_kg_max: float = 50.0
    # Task urgency profile.
    routine_bulk_urgency_min: float = 0.20
    routine_bulk_urgency_max: float = 0.60
    time_critical_lightweight_urgency_min: float = 0.70
    time_critical_lightweight_urgency_max: float = 1.00
    # Lifeline contract.
    task_lifeline_enabled: bool = True
    task_lifeline_init_default: float = 100.0
    routine_bulk_lifeline_decay_base: float = 0.08
    time_critical_lightweight_lifeline_decay_base: float = 0.22
    task_lifeline_hazard_weight: float = 0.35
    # Partial fulfillment (routine bulk only).
    routine_bulk_partial_fulfillment_enabled: bool = False
    routine_bulk_partial_chunk_kg: float = 200.0
    # Truck inventory is split into two material pools (kg-based):
    # - bulk inventory for routine bulk tasks
    # - time-critical inventory for time-critical lightweight tasks / UAV reload
    truck_initial_bulk_inventory_kg: float = 2400.0
    truck_initial_timecritical_inventory_kg: float = 200.0
    # Supply-unit mass baselines used for compatibility counters and reload package.
    bulk_supply_unit_kg: float = 800.0
    timecritical_supply_unit_kg: float = 50.0
    truck_initial_normal_supply_units: int = 3
    truck_initial_emergency_supply_units: int = 4
    uav_max_emergency_units: int = 1
    # If False, trucks are not allowed to directly execute emergency tasks.
    # Emergency tasks are then handled by UAVs (with truck support/reload).
    truck_can_serve_emergency_tasks: bool = False
    # Direct emergency delivery is UAV-only. Trucks retain their bulk-task
    # routes and act only as mobile launch/reload/recovery platforms.
    truck_conditional_emergency_service_enabled: bool = False
    truck_emergency_service_max_distance_m: float = 900.0
    truck_emergency_service_max_deadline_slack_steps: int = 18
    # High-pressure fallback (L-scale / island / urgent overload):
    # allow wider truck emergency fallback only under pressure conditions.
    truck_high_pressure_emergency_service_max_distance_m: float = 1400.0
    truck_high_pressure_emergency_service_max_deadline_slack_steps: int = 28
    # If False, truck depot-refill demand only considers normal-task stockout;
    # emergency stockout alone will not force truck depot return.
    truck_replenish_for_emergency_stock: bool = False
    truck_replenish_only_at_depot: bool = True
    uav_replenish_only_from_truck: bool = True
    # Allow UAV to recharge and reload emergency package while docked at depot.
    uav_reload_at_depot_enabled: bool = True
    uav_must_replenish_after_each_service: bool = True
    truck_replenish_service_steps: int = 2
    uav_reload_service_steps: int = 1
    # UAV physics
    uav_max_speed_mps: float = 10.0
    # Optional far-field speed floor for low-level control (0.0 disables).
    # Applied only outside terminal approach zones to reduce dithering.
    uav_far_field_min_cruise_mps: float = 0.0
    # UAV high-goal lock window (steps) to suppress rapid A<->B flipping.
    # Direct sortie envelope used by low-level takeoff gating.
    uav_short_sortie_max_distance_m: float = 1200.0
    uav_short_sortie_min_battery: float = 0.40
    # Low-battery recovery truck selection.
    # If nearest truck is farther than this radius, enable directional scoring
    # to pick a more suitable truck (not strictly nearest).
    uav_recovery_near_truck_radius_m: float = 700.0
    uav_recovery_directional_select_enabled: bool = True
    uav_recovery_direction_weight: float = 0.70
    uav_recovery_distance_weight: float = 0.55
    uav_recovery_stock_weight: float = 0.20
    uav_recovery_truck_request_enabled: bool = True
    # Truck-side bonus when serving the UAV-requested recovery target.
    truck_recovery_request_match_bonus: float = 0.35
    uav_goal_lock_steps: int = 3
    # Planner-side docked UAV shortlist: when UAV is already on truck and
    # not in urgent recovery, high-level assignment only considers nearest
    # emergency tasks to avoid rigid far-task sticking.
    uav_docked_task_shortlist_enabled: bool = True
    uav_docked_task_shortlist_radius_m: float = 1800.0
    uav_docked_task_shortlist_topk: int = 3
    # Hard guard margin for preventing docked UAV switches to farther emergency tasks.
    uav_docked_hard_far_switch_margin_m: float = 60.0
    # Opportunistic docked retarget:
    # while UAV is docked and sortie-safe, allow switching to a much closer
    # emergency task instead of rigidly following the original far task goal.
    uav_docked_opportunistic_retarget_enabled: bool = True
    uav_docked_opportunistic_radius_m: float = 500.0
    uav_docked_opportunistic_gain_ratio: float = 0.55
    uav_docked_opportunistic_min_margin_m: float = 180.0
    uav_docked_opportunistic_cooldown_steps: int = 12
    # Docked UAV periodic retarget: every N steps, re-evaluate nearest emergency
    # for riding UAVs using distinct (non-duplicate) assignment.
    uav_docked_retarget_enabled: bool = False
    uav_docked_retarget_interval_steps: int = 6
    # Environment-side island override: when enabled, env may rewrite planner UAV goals
    # toward island emergencies. Keep disabled for planner-consistent paper runs.
    uav_env_island_goal_override_enabled: bool = False
    # Docked dispatch preference window:
    # - within near radius: strongly prefer immediate dispatch
    # - within heading radius: allow dispatch when truck heading is aligned
    uav_docked_near_dispatch_radius_m: float = 900.0
    uav_docked_heading_dispatch_radius_m: float = 1500.0
    uav_docked_heading_min_cosine: float = 0.05
    uav_docked_near_dispatch_bonus: float = 0.65
    uav_docked_heading_dispatch_bonus: float = 0.35
    uav_docked_far_task_penalty: float = 0.45
    # Urgent emergency watchdog for docked UAV assignment:
    # when an emergency task is close to deadline and still uncovered by current
    # high-level goals, allow controlled retarget to reduce "unclaimed timeout".
    uav_urgent_watchdog_enabled: bool = False
    uav_urgent_watchdog_slack_steps: int = 20
    uav_urgent_watchdog_retarget_cooldown_steps: int = 4
    uav_urgent_watchdog_distance_bonus_m: float = 900.0
    uav_urgent_watchdog_max_assign_per_step: int = 1
    # Early-window diversification: force distinct emergency assignments for UAVs
    # at episode start to avoid all UAVs converging to one target.
    uav_initial_distinct_emergency_assign: bool = False
    uav_initial_distinct_window_steps: int = 6
    # Hard recovery threshold: below this battery fraction, UAV prioritizes truck recovery.
    uav_hard_recovery_battery_threshold: float = 0.38
    # If enabled, battery<=0 in forced-recovery corridor is converted to emergency rescue
    # (non-crash) to satisfy safety-contract hard constraints.
    uav_hard_recovery_battery_guard: bool = True
    # Hard launch safety gate: UAV cannot leave truck when battery is below this threshold.
    uav_launch_min_battery_fraction: float = 0.50
    # Canonical V2 outbound SOC gate. The legacy launch fraction is retained
    # for archived V1 configurations only.
    uav_min_takeoff_soc: float = 0.10
    # Emergency reserve is not double-counted; return feasibility carries the safety margin.
    uav_emergency_reserve_fraction: float = 0.25
    # Additional safety margin used for return-to-truck feasibility checks.
    uav_return_margin_fraction: float = 0.0
    # Planner/environment belief scaling for recovery safety margins.
    # Values <1 mean the controller underestimates moving-recovery uncertainty.
    uav_launch_recovery_margin_scale: float = 1.0
    # Conservative distance inflation for moving-truck rendezvous uncertainty.
    # Applied in launch/task feasibility so UAV does not depart with razor-thin return margins.
    uav_recovery_distance_buffer_m: float = 100.0
    # Conservative truck drift term used in UAV launch gate recovery margin:
    # drift = min(raw_drift * scale, max_m), raw_drift = truck_speed * dt * decision_interval.
    uav_recovery_truck_drift_margin_scale: float = 0.5
    uav_recovery_truck_drift_margin_max_m: float = 600.0
    # When forced recovery battery is critically low and still far from truck,
    # UAV may hold position to conserve battery while truck performs active pickup.
    uav_recovery_idle_hold_threshold: float = 0.18
    uav_recovery_idle_hold_min_dist_m: float = 600.0
    # Additional safety margin used for rendezvous-only recovery feasibility checks.
    uav_rendezvous_margin_fraction: float = 0.0
    # Allow launching missions that rely on in-flight rendezvous recovery only.
    # For paper safety baseline keep False: launch requires direct-safe return margin.
    uav_allow_rendezvous_launch: bool = False
    # Optional guard for controlled rendezvous launch: only release if the UAV's
    # current truck is already committed to the same emergency/support target.
    uav_rendezvous_launch_requires_docked_truck_goal: bool = False
    # Planned recovery request for rendezvous sorties: the UAV keeps flying to
    # the emergency task, while its launch truck is marked as the preferred
    # recovery anchor for truck-side support.
    uav_rendezvous_planned_recovery_request_enabled: bool = True
    uav_rendezvous_planned_recovery_urgency: float = 0.55
    # Conditional rendezvous launch patch: keeps global rendezvous launch disabled
    # by default but allows urgent/high-demand controlled release.
    uav_conditional_rendezvous_launch_enabled: bool = True
    uav_conditional_rendezvous_min_pending_emergency: int = 4
    uav_conditional_rendezvous_max_deadline_slack_steps: int = 18
    uav_conditional_rendezvous_max_nearest_truck_m: float = 1400.0
    # Adaptive launch gate patch under emergency backlog pressure.
    uav_adaptive_launch_gate_enabled: bool = True
    uav_adaptive_launch_min_floor: float = 0.48
    uav_adaptive_launch_relax_delta: float = 0.06
    uav_adaptive_force_takeoff_gap: float = 0.16
    # High-pressure second-stage launch relaxation for emergency/island only.
    uav_high_pressure_launch_min_floor: float = 0.44
    uav_high_pressure_force_takeoff_gap: float = 0.12
    # Planner-side repeated reject cache to suppress immediate reselection loops.
    uav_reject_cache_window_steps: int = 20
    uav_reject_cache_min_repeat: int = 2
    uav_reject_cache_ttl_steps: int = 30
    # High-pressure conditional rendezvous extension (still corridor-constrained).
    uav_high_pressure_rendezvous_enabled: bool = True
    uav_high_pressure_rendezvous_max_deadline_slack_steps: int = 28
    uav_high_pressure_rendezvous_max_nearest_truck_m: float = 2200.0
    # Reduce forced-recovery over-intervention for high-pressure tasks when
    # rendezvous corridor remains feasible.
    uav_high_pressure_recovery_margin_bonus_m: float = 250.0
    # Additional recovery margin bonus only for relaxed-rendezvous launches.
    uav_relaxed_rendezvous_recovery_margin_bonus_m: float = 300.0
    # Launch horizon gate: avoid late/short-window takeoff that cannot complete
    # an emergency sortie lifecycle before episode horizon.
    uav_launch_speed_utilization: float = 0.70
    uav_launch_min_horizon_buffer_steps: int = 4
    uav_launch_min_remaining_steps: int = 8
    # Allow env-side automatic departure when docked, full battery and loaded.
    # For stability keep False and let policy issue explicit takeoff.
    uav_auto_depart_when_ready: bool = False
    # Keep task goal visible while docked so planner/low-level can track the
    # anchored emergency target instead of collapsing to truck-only idle state.
    uav_docked_keep_task_goal_enabled: bool = True
    # Initial UAV state: when True, UAVs start attached to trucks at depot.
    uav_start_docked_on_truck: bool = False
    # Below this threshold, UAV goal switching to farther targets is suppressed.
    uav_low_battery_goal_lock_threshold: float = 0.30
    # Below this threshold, UAV is forced into recovery mode.
    uav_low_battery_force_recover_threshold: float = 0.20
    # Recovery bind commitment and post-bind dwell constraints.
    uav_bind_commit_steps: int = 4
    uav_post_bind_min_dwell_steps: int = 3
    # No fixed state-of-charge departure threshold: the sortie gate evaluates
    # the actual delivery-and-recovery energy requirement for the assigned task.
    uav_post_bind_force_recharge: bool = False
    uav_post_bind_force_reload: bool = True
    # Support-chain response latency after a forced-recovery request is raised.
    # ERC-style event-driven coordination uses 0; delayed planners can use >0.
    uav_forced_recovery_bind_latency_steps: int = 0
    # Truck-side cooperative recovery support for rendezvous.
    truck_support_uav_recovery_enabled: bool = True
    truck_recovery_max_detour_cost_weight: float = 1.0
    truck_recovery_priority_weight: float = 1.0
    truck_recovery_require_request_when_normal_pending: bool = False
    truck_recovery_request_min_urgency_when_normal_pending: float = 0.0
    hrl_support_chain_max_direct_ready_timecritical: int = 1
    hrl_support_chain_min_gain_for_enable: float = 0.25
    hrl_support_chain_critical_escape_enabled: bool = True
    hrl_support_chain_critical_escape_scenarios: str = "C"
    hrl_support_chain_critical_escape_max_lifeline_ratio: float = 0.55
    hrl_support_chain_critical_escape_low_cover_threshold: float = 0.35
    hrl_support_chain_critical_escape_min_gain: float = 0.04
    # Protect the final approach to a routine/bulk task from soft UAV support
    # detours. The guard is intentionally short-horizon: if the current node or
    # one legal next truck move can bring the routine goal within this many
    # travel steps, the truck should finish that delivery unless an airborne UAV
    # has a hard safety/recovery emergency.
    truck_routine_near_goal_support_protect_enabled: bool = True
    truck_routine_near_goal_support_protect_steps: int = 5
    hrl_routine_near_completion_eta_steps: int = 5
    hrl_routine_near_completion_route_dist_m: float = 1000.0
    hrl_routine_protection_tc_override_enabled: bool = True
    hrl_routine_protection_tc_override_max_routine_delay_steps: int = 3
    hrl_routine_protection_tc_override_max_support_steps: int = 5
    hrl_routine_protection_tc_override_min_launch_gain_m: float = 300.0
    hrl_routine_protection_tc_override_require_recovery_feasible: bool = True
    hrl_routine_protection_tc_override_require_loaded_uav: bool = True
    hrl_routine_protection_delivery_feasible_tc_override_enabled: bool = True
    hrl_tc_override_require_full_sortie_feasible: bool = True
    # Minimum recoverable-range reserve after a complete UAV sortie.
    hrl_tc_override_min_recovery_margin_m: float = 100.0
    hrl_tc_override_min_battery_margin_ratio: float = 0.12
    hrl_tc_override_max_expected_lifeline_decay_ratio: float = 0.85
    hrl_tc_override_block_if_recent_reject: bool = True
    hrl_tc_override_reject_cache_ttl_steps: int = 20
    hrl_tc_override_max_routine_delay_steps: int = 3
    hrl_tc_override_min_delivery_score_gain: float = 0.10
    hrl_tc_override_max_support_steps: int = 5
    # Operational transport mass is the official 55.2 kg lifting aircraft plus
    # two 14.7 kg batteries. Accessories are not claimed as manufacturer-
    # calibrated mass; the resulting 84.6 kg is a documented lower-bound proxy.
    uav_self_weight_kg: float = 84.6
    uav_payload_kg: float = 50.0
    uav_payload_capacity_kg: float = 50.0
    truck_payload_capacity_kg: float = 3000.0
    max_time_critical_packages_per_uav_sortie: int = 1
    max_standard_packages_per_truck: int = 3
    uav_max_sortie_m: float = 6000.0
    # Canonical operational envelope; unlike the official to-zero ideal range,
    # this cumulative sortie cap is enforced in paper runs.
    uav_enforce_max_sortie_limit: bool = True
    uav_battery_init: float = 1.0
    uav_bind_radius_m: float = 170.0
    # Dynamic bind window for moving-truck rendezvous:
    # bind_window = base_bind_radius + k_motion * truck_speed * dt + truck_speed * latency_margin_s
    uav_bind_motion_window_gain: float = 0.30
    uav_bind_latency_margin_s: float = 5.0
    uav_max_followers_per_truck: int = 2
    # Deprecated compatibility field. The sortie gate is authoritative.
    uav_force_takeoff_battery_threshold: float = 0.0
    # Cooldown after unsafe takeoff block to avoid repeated launch attempts
    # against the same temporary gate condition.
    uav_unsafe_launch_block_cooldown_steps: int = 2
    uav_idle_discharge_per_step: float = 0.027777777777777776
    # Low-power idle consumption scale during forced-recovery hold mode.
    uav_recovery_idle_discharge_scale: float = 0.25
    # Physical terminal floor: below this airborne battery fraction, the UAV is
    # treated as no longer safely controllable unless recovery support/guard has
    # already taken over.
    uav_terminal_failure_battery_floor: float = 0.04
    uav_flight_discharge_per_step: float = 0.04
    uav_flight_discharge_per_m: float = 0.00005555555555555556
    uav_headwind_energy_coeff: float = 0.03
    uav_rain_energy_coeff: float = 0.01
    uav_payload_energy_coeff_per_kg: float = 0.024
    # V2 uses a normalized payload ratio rather than an energy coefficient per
    # kilogram, which was tied to the obsolete 40 kg UAV capacity.
    uav_full_load_energy_penalty: float = 0.50
    uav_energy_norm_wind_mps: float = 10.0
    uav_energy_norm_rain_mmh: float = 30.0
    # Optional wind-failure model for UAV safety analysis.
    # Scenario-level safety gates. These are conservative engineering
    # assumptions, not manufacturer-calibrated continuous weather curves.
    uav_no_launch_wind_mps: float = 6.0
    uav_recover_wind_mps: float = 10.0
    uav_no_launch_rain_mmh: float = 24.0
    uav_recover_rain_mmh: float = 30.0
    wind_failure_threshold_mps: float = 12.0
    wind_failure_risk_scale: float = 0.0
    # Critically-low-SOC airborne instability risk. This is only activated when
    # the UAV is still airborne below a danger threshold.
    uav_low_soc_failure_threshold: float = 0.10
    uav_low_soc_failure_risk_scale: float = 0.0
    uav_charge_rate_per_step: float = 0.085
    uav_crash_penalty: float = -5.0
    # UAV local sensing / auto-approach
    uav_monitor_radius_m: float = 340.0
    # Ablation / paper-control toggles
    use_hetgat: bool = True
    enable_rth_mask: bool = True
    # Global risk-encoder mask gain. Set to 0.0 to force no-mask behavior.
    risk_gat_beta: float = 1.0
    # Rolling/ERC-RHC ablation switches (consumed by planner entrypoints).
    use_event_trigger: bool = True
    use_risk_term: bool = True
    use_rth_repair: bool = True
    rth_safety_factor: float = 1.2
    # Rule-based takeover switches (disable for pure RL exploration).
    uav_auto_approach_enabled: bool = False
    uav_monitor_snap_enabled: bool = False
    # Backward-compatible aliases used by some training loops.
    enable_auto_approach: bool = False
    enable_monitor_snap: bool = False
    # Service (unloading) rounds
    unload_rounds_normal: int = 5
    unload_rounds_emergency: int = 5
    unload_rounds_uav: int = 1
    # Task generation (seed-controlled stochastic demand).
    # emergency mode:
    # - task_emergency_epicenter_bias <= 0.0 : uniform sampling
    # - task_emergency_epicenter_bias > 0.0  : epicenter-biased sampling
    task_emergency_epicenter_bias: float = 0.0
    task_emergency_sigma_m: float = 1200.0
    # Reward
    reward_step_penalty: float = -0.01
    reward_invalid_action: float = -0.05
    reward_idle_with_task: float = -0.08
    reward_delivery_normal: float = 5.0
    reward_delivery_emergency: float = 10.0
    # Extra shaping only when UAV delivers emergency tasks.
    reward_uav_emergency_delivery_bonus: float = 0.0
    # Docking reward only for meaningful low-battery recovery.
    reward_docking_low_battery: float = 1.0
    docking_reward_battery_threshold: float = 0.5
    reward_uav_discover_blocked_edge: float = 0.5
    reward_pickup: float = 1.0
    reward_delivery_shared: float = 5.0
    penalty_timeout_normal: float = -0.5
    penalty_timeout_emergency: float = -1.3
    # Optional PBRS hook
    use_pbrs: bool = False
    pbrs_scale: float = 1.0
    # HRL anti-oscillation controls for high-level goal assignments.
    hrl_replan_cooldown_steps: int = 8
    hrl_goal_min_hold_steps: int = 12
    hrl_goal_switch_margin: float = 0.20
    hrl_uav_goal_switch_margin: float = 0.35
    hrl_truck_goal_switch_margin: float = 0.20
    # Keep at least this many trucks stably serving NORMAL tasks when backlog exists.
    hrl_truck_min_normal_slots: int = 1
    hrl_truck_min_normal_slots_high_pressure: int = 2
    hrl_truck_min_normal_slots_pressure_threshold: float = 0.55
    # Cooldown window for switching away from a NORMAL goal (unless hard override).
    hrl_truck_normal_switch_cooldown_steps: int = 18
    # Prevent NORMAL goal from being cleared to None unless truly unreachable.
    hrl_truck_normal_no_none_clear_enabled: bool = True
    # Deadline-protection window for NORMAL tasks already under truck execution:
    # within this many steps to deadline, avoid diversion unless hard-infeasible.
    hrl_truck_normal_deadline_protect_steps: int = 40
    # Consecutive unreachable steps before treating a NORMAL task as persistently unreachable.
    hrl_normal_unreachable_patience_steps: int = 10
    hrl_risk_spike_edge_trigger_only: bool = False
    # Event-trigger budget per decision window (0 = interval-only refresh).
    hrl_event_replan_budget_per_window: int = 2
    # Event-first refresh mode:
    # - refresh on events first
    # - if no refresh for max_no_refresh_steps, do one fallback refresh.
    hrl_event_first_refresh_enabled: bool = False
    hrl_max_no_refresh_steps: int = 5
    # Event admission gate:
    # weak events (arrival/idle/light updates/ranking) must pass actionable checks.
    hrl_event_admission_gate_enabled: bool = False
    hrl_event_admission_lifeline_threshold_ratio: float = 0.40
    # Weak-event no-op cooldown:
    # if a weak-event refresh yields no goal-change/launch/completion, block that
    # reason for a short cooldown window.
    hrl_noop_event_cooldown_enabled: bool = False
    hrl_noop_event_cooldown_steps: int = 10
    # UAV truck-follow anchor anti-ABA guard (reduce A<->B truck ping-pong).
    hrl_uav_truck_anchor_aba_block_steps: int = 14
    # Generic task A->B->A guard and communication-blackout commitment hold.
    # These operate after safety checks, so they cannot suppress a required
    # recovery or an actually infeasible-goal reroute.
    hrl_task_aba_block_steps: int = 18
    hrl_comm_blackout_commit_hold_enabled: bool = True
    # In road-only operation, preserve a valid route commitment for a short
    # window instead of relying on a communication blackout to suppress
    # low-value goal churn. Hard invalidity, dead ends, and safety events still
    # bypass this stability window in the planner.
    hrl_b_route_stability_enabled: bool = False
    hrl_b_route_stability_hold_steps: int = 12
    hrl_b_route_stability_margin_scale: float = 1.25
    # Optional algorithm-owned pre-launch contract lock. Kept off by default
    # until its cross-seed effect is separately validated.
    hrl_b_prelaunch_contract_lock_enabled: bool = False
    hrl_b_docked_latch_rearm_enabled: bool = False
    # Optional B-only escape for a loaded UAV whose planned road anchor was
    # invalidated; the normal launch gate must still certify the sortie.
    hrl_b_anchor_unreachable_uav_launch_enabled: bool = False
    hrl_b_anchor_unreachable_uav_launch_max_lifeline_ratio: float = 0.10
    # Hard-event de-noising knobs (normal stall / soft invalid / recovery).
    hrl_normal_stall_hard_refresh_enabled: bool = False
    hrl_normal_stall_min_persist_steps: int = 8
    hrl_normal_stall_progress_epsilon_m: float = 30.0
    hrl_normal_stall_cooldown_steps: int = 12
    hrl_normal_stall_local_only: bool = True
    hrl_soft_invalid_hard_refresh_enabled: bool = False
    hrl_soft_invalid_retry_cooldown_steps: int = 10
    hrl_soft_invalid_escalate_after_count: int = 3
    # Truck dead-end / path-blocked localization (R-C no-op hard refresh reduction).
    hrl_truck_dead_end_local_first: bool = True
    hrl_truck_dead_end_persist_steps: int = 3
    hrl_truck_dead_end_cooldown_steps: int = 10
    hrl_truck_dead_end_global_refresh_enabled: bool = False
    hrl_path_blocked_impact_gate_enabled: bool = True
    hrl_path_blocked_local_repair_first: bool = True
    hrl_path_blocked_global_refresh_enabled: bool = False
    # Truck emergency relief gating under normal backlog pressure.
    hrl_truck_emergency_min_pending_normal_to_block: int = 1
    hrl_truck_emergency_relief_uav_cover_threshold: float = 0.5
    hrl_truck_emergency_force_relief_urgency_threshold: float = 0.72
    hrl_truck_emergency_force_relief_uav_cover_threshold: float = 0.35
    # Service-anchor hold and relaxed-chain commitment controls.
    hrl_support_anchor_hold_steps: int = 8
    hrl_support_anchor_max_drift_m: float = 250.0
    hrl_relaxed_chain_commitment_steps: int = 6
    hrl_relaxed_chain_min_bound_tasks: int = 1
    hrl_serviceable_island_bonus: float = 0.25
    hrl_serviceable_high_pressure_emergency_bonus: float = 0.20
    # Extra island-delivery preference for UAV emergency assignment.
    hrl_uav_island_delivery_bonus: float = 0.20
    # Early truck direction diversification to reduce same-direction convoying.
    hrl_truck_directional_split_enabled: bool = True
    hrl_truck_directional_split_steps: int = 24
    hrl_truck_directional_split_bonus: float = 0.12
    # Step-0/early-step directional coverage planning:
    # evaluate sector-level pending demand (normal/emergency) and assign trucks/UAVs
    # to different directions before fine-grained rolling updates dominate.
    hrl_initial_directional_cover_enabled: bool = True
    hrl_initial_directional_window_steps: int = 10
    hrl_initial_directional_sector_count: int = 4
    hrl_initial_directional_normal_weight: float = 1.0
    hrl_initial_directional_emergency_weight: float = 2.2
    hrl_initial_directional_urgency_weight: float = 0.35
    hrl_initial_directional_duplicate_penalty: float = 0.60
    hrl_initial_directional_task_bonus: float = 0.24
    hrl_initial_directional_task_mismatch_penalty: float = 0.08
    hrl_initial_directional_uav_truck_bonus: float = 0.26
    hrl_initial_directional_uav_truck_mismatch_penalty: float = 0.10
    hrl_initial_directional_uav_task_bonus: float = 0.16
    hrl_initial_directional_uav_per_truck_cap: int = 2
    hrl_initial_route_dispatch_enabled: bool = True
    # Route-aware startup docking: UAVs may start attached to different trucks
    # according to depot outbound roads and early task mix instead of round-robin.
    hrl_initial_route_docked_assignment_enabled: bool = True
    # UAV emergency-task locality preference (nearer UAV should claim the task).
    hrl_uav_task_locality_weight: float = 0.22
    # Early near-depot direct dispatch: allow UAV to serve close emergency first.
    hrl_uav_near_depot_direct_dispatch_radius_m: float = 700.0
    hrl_uav_near_depot_direct_dispatch_bonus: float = 0.40
    hrl_uav_near_depot_direct_dispatch_steps: int = 18
    # Idle UAV truck-staging and truck selection shaping.
    hrl_uav_idle_truck_staging_enabled: bool = True
    hrl_uav_idle_truck_staging_min_score: float = 0.06
    # Keep early-stage UAV truck assignment spread; avoid all-idle UAVs clustering
    # onto the same truck when no immediate emergency sortie is feasible.
    hrl_uav_idle_staging_prefer_initial_plan: bool = True
    hrl_uav_idle_staging_respect_truck_cap: bool = True
    hrl_uav_truck_emergency_pull_weight: float = 0.28
    hrl_uav_truck_follower_balance_weight: float = 0.18
    hrl_truck_task_lookahead_steps: int = 3
    hrl_truck_task_lookahead_max_steps: int = 5
    hrl_truck_task_lookahead_weight: float = 0.28
    hrl_uav_truck_lookahead_weight: float = 0.32
    region_commitment_enabled: bool = False
    region_commitment_min_map_size_m: float = 9000.0
    region_commitment_count: int = 0
    region_commitment_auto_select_enabled: bool = False
    region_commitment_auto_max_k: int = 3
    region_commitment_enable_score_threshold: float = 0.18
    region_commitment_min_separation_score: float = 0.22
    region_commitment_min_load_balance_score: float = 0.20
    region_commitment_unbalanced_min_separation_score: float = 0.75
    region_commitment_unbalanced_min_outlier_tasks: int = 3
    region_commitment_overpartition_penalty: float = 0.16
    region_commitment_strength_min: float = 0.35
    region_commitment_cross_region_filter_enabled: bool = True
    region_commitment_local_bonus: float = 0.28
    region_commitment_cross_region_penalty: float = 0.85
    region_commitment_override_lifeline_ratio: float = 0.32
    region_commitment_keep_current_goal_enabled: bool = True
    region_commitment_include_gateways: bool = True
    region_commitment_outlier_gate_enabled: bool = False
    region_commitment_outlier_distance_ratio: float = 0.22
    region_commitment_outlier_min_distance_m: float = 3200.0
    region_commitment_outlier_penalty: float = 0.65
    region_commitment_outlier_override_lifeline_ratio: float = 0.40
    region_commitment_outlier_require_support_gain: bool = True
    region_commitment_outlier_min_support_gain: float = 0.12
    region_commitment_routine_guard_enabled: bool = False
    region_commitment_routine_guard_lifeline_ratio: float = 0.42
    region_commitment_routine_guard_penalty: float = 0.36
    region_commitment_routine_guard_support_gain_relief: float = 0.55
    region_commitment_routine_guard_max_normal_dist_m: float = 1200.0
    hrl_uav_docked_prelaunch_assign_min_score: float = 0.16
    # Conservative truck-to-truck UAV transfer: keep task anchor, but allow a
    # docked UAV to detach and chase a better truck when near-future route
    # progress improves enough to justify a swap.
    hrl_uav_task_transfer_enabled: bool = True
    hrl_uav_task_transfer_score_gain_min: float = 0.22
    hrl_uav_task_transfer_progress_gain_min: float = 0.18
    hrl_uav_task_transfer_commit_steps: int = 12
    hrl_uav_task_transfer_hint_hold_steps: int = 6
    hrl_uav_task_transfer_max_target_dist_m: float = 2600.0
    uav_transfer_min_battery_fraction: float = 0.50
    uav_transfer_reserve_fraction: float = 0.08
    # Ablation toggles for stepwise planner upgrades (mod1/mod2/mod3 replay).
    hrl_uav_task_reservation_enabled: bool = True
    hrl_supported_sortie_joint_enabled: bool = True
    hrl_dynamic_task_pressure_enabled: bool = True
    # Post-mod2 targeted improvements (default off; enable one-by-one in experiments).
    hrl_support_conversion_gate_enabled: bool = False
    hrl_support_conversion_target_ratio: float = 0.45
    hrl_support_conversion_min_support_count: int = 8
    hrl_support_conversion_penalty_strength: float = 0.55
    hrl_truck_normal_commit_guard2_enabled: bool = False
    hrl_truck_normal_commit_min_steps: int = 8
    hrl_truck_normal_commit_pressure_threshold: float = 0.55
    # Truck support mode switch:
    # - when trucks still have reachable NORMAL tasks, reduce emergency-support bias;
    # - when a truck has no reachable NORMAL while normal backlog remains, allow
    #   dedicated emergency-support behavior for UAV service-chain conversion.
    hrl_truck_support_when_normal_reachable_scale: float = 0.45
    hrl_truck_support_when_no_normal_bonus: float = 0.30
    hrl_truck_emergency_support_when_no_normal_enabled: bool = True
    # Hard split gate:
    # - while reachable NORMAL exists for this truck, emergency candidates are
    #   strictly de-prioritized at candidate stage;
    # - once NORMAL is unreachable for this truck (while normal backlog exists),
    #   switch to emergency-support-only candidate mode.
    hrl_truck_hard_normal_first_enabled: bool = True
    # Planner objective split: assign UAV and truck with separate objective stages.
    hrl_separate_agent_objectives_enabled: bool = True
    # Lifeline-aware priority tiers for time-critical lightweight tasks.
    hrl_timecritical_lifeline_warning_ratio: float = 0.55
    hrl_timecritical_lifeline_critical_ratio: float = 0.35
    # Large-map/C safeguard: if a time-critical task has gone too long without
    # ever entering the goal chain, force it into shortlist/assignment priority.
    hrl_timecritical_force_entry_enabled: bool = True
    hrl_timecritical_force_entry_min_map_size_m: float = 12000.0
    hrl_timecritical_force_entry_min_gap_steps: int = 12
    hrl_timecritical_force_entry_max_lifeline_ratio: float = 0.85
    hrl_timecritical_force_entry_shortlist_extra: int = 2
    hrl_timecritical_force_entry_uav_bonus: float = 0.32
    hrl_timecritical_force_entry_truck_bonus: float = 0.20
    # ERC-only execution repair: a direct-feasible but long-uncovered
    # time-critical task can still receive a truck-UAV support lock on large
    # disrupted maps, preventing far tasks from being repeatedly exposed but
    # never committed to an executable chain.
    erc_tc_uncovered_support_repair_enabled: bool = False
    erc_tc_uncovered_support_repair_min_step: int = 18
    erc_tc_uncovered_support_repair_min_gap_steps: int = 12
    erc_tc_uncovered_support_repair_max_lifeline_ratio: float = 0.82
    erc_tc_uncovered_support_repair_cover_threshold: float = 0.40
    erc_tc_uncovered_support_repair_min_nearest_truck_m: float = 5500.0
    erc_tc_uncovered_support_repair_min_urgency: float = 0.0
    erc_tc_stale_assigned_support_repair_enabled: bool = False
    erc_tc_stale_assigned_support_repair_min_exposure: int = 28
    erc_tc_stale_assigned_support_repair_min_step: int = 80
    erc_tc_stale_assigned_support_repair_max_lifeline_ratio: float = 0.75
    erc_tc_stale_assigned_support_repair_min_nearest_truck_m: float = 5200.0
    erc_tc_coverage_intent_enabled: bool = False
    erc_tc_coverage_intent_min_step: int = 0
    erc_tc_coverage_intent_max_step: int = 160
    erc_tc_coverage_intent_max_support_deliveries: int = -1
    erc_tc_coverage_intent_min_pending: int = 4
    erc_tc_coverage_intent_cover_threshold: float = 0.55
    erc_tc_coverage_intent_max_lifeline_ratio: float = 0.92
    erc_tc_coverage_intent_min_gap_steps: int = 4
    erc_tc_coverage_intent_max_per_step: int = 2
    erc_large_map_greedy_tc_fallback_enabled: bool = True
    erc_large_map_greedy_tc_fallback_b_min_step: int = 80
    erc_large_map_greedy_tc_fallback_b_max_support_deliveries: int = 4
    erc_large_map_greedy_tc_fallback_b_min_support_locks: int = 0
    erc_large_map_greedy_tc_fallback_replace_stale_enabled: bool = False
    erc_large_map_greedy_tc_fallback_stale_steps: int = 24
    alns_enabled: bool = False
    alns_adaptive_horizon_enabled: bool = True
    alns_risk_pressure_enabled: bool = True
    alns_ghost_tasks_enabled: bool = True
    alns_iterations: int = 24
    # Optional diagnostic-only overrides.  Zero retains the method profile.
    diagnostic_alns_iterations: int = 0
    diagnostic_alns_min_replan_interval_steps: int = 0
    alns_min_replan_interval_steps: int = 3
    alns_max_replan_interval_steps: int = 12
    alns_min_horizon_steps: int = 20
    alns_max_horizon_steps: int = 90
    alns_destroy_max_assignments: int = 3
    alns_accept_temperature: float = 0.05
    alns_solution_mode: str = "legacy_k1"
    alns_sequence_length: int = 1
    adaptive_horizon_mode: str = "disabled"
    adaptive_horizon_allowed_values: tuple[int, ...] = (1, 2)
    local_search_mode: str = "disabled"
    local_search_max_moves_per_iteration: int = 5
    local_search_max_exact_checks_per_iteration: int = 5
    local_search_max_time_ms_per_iteration: int = 20
    local_search_disabled_moves: tuple[str, ...] = ()
    alns_operator_pool: str = "legacy"
    alns_initialization_mode: str = "objective_greedy"
    alns_operator_weight_profile: str = "uniform"
    alns_selection_mode: str = "adaptive"
    alns_critical_recovery_repair_enabled: bool = False
    alns_critical_recovery_repair_max_tasks: int = 3
    alns_critical_recovery_repair_min_priority: float = 0.0
    alns_critical_recovery_repair_prefer_truck: bool = True
    alns_critical_recovery_repair_avoid_failed_agent: bool = True
    alns_critical_support_rebind_enabled: bool = False
    alns_critical_support_rebind_max_tasks: int = 2
    alns_critical_support_rebind_min_assigned_count: int = 2
    alns_critical_support_rebind_prefer_historical_binding: bool = True
    alns_critical_support_rebind_allow_nearest_feasible_truck: bool = True
    alns_critical_support_rebind_preserve_recovery_anchor: bool = True
    alns_critical_support_rebind_target_only_failed_or_pending: bool = True
    alns_support_rebind_margin_aware_enabled: bool = False
    alns_support_rebind_anchor_ranking_enabled: bool = False
    alns_support_rebind_failed_binding_avoidance_enabled: bool = False
    alns_support_rebind_failed_binding_penalty: str = "mild"
    alns_support_rebind_critical_first_ordering_enabled: bool = False
    alns_support_rebind_safe_uav_guard_enabled: bool = False
    alns_support_rebind_margin_top_k: int = 3
    alns_support_rebind_anchor_search_radius_factor: float = 1.0
    alns_lc_critical_recovery_path_enabled: bool = False
    alns_lc_critical_recovery_path_max_tasks: int = 3
    alns_lc_critical_recovery_path_min_assigned_count: int = 20
    alns_lc_critical_recovery_path_top_k_bindings: int = 8
    alns_lc_critical_recovery_path_require_positive_margin: bool = True
    alns_lc_critical_recovery_path_prioritize_no_bindable_truck: bool = True
    alns_lc_critical_recovery_path_avoid_repeated_failed_tuple: bool = True
    alns_lc_critical_recovery_path_target_critical_only: bool = True
    alns_assigned_critical_reconstruct_enabled: bool = False
    alns_assigned_critical_reconstruct_max_tasks: int = 3
    alns_assigned_critical_reconstruct_min_assigned_count: int = 20
    alns_assigned_critical_reconstruct_top_k_paths: int = 12
    alns_assigned_critical_reconstruct_target_critical_only: bool = True
    alns_support_reposition_shadow_enabled: bool = False
    alns_support_reposition_shadow_max_tasks: int = 8
    alns_support_reposition_shadow_min_assigned_count: int = 20
    physical_environment_version: str = "v1"
    physical_environment_safety_protocol: str = "shielded_operation"
    candidate_ranker_mode: str = "disabled"
    candidate_ranker_pool_size: int = 16
    candidate_ranker_exact_check_budget: int = 4
    candidate_ranker_exploration_count: int = 1
    alns_weight_segment_length: int = 12
    alns_weight_learning_rate: float = 0.25
    alns_weight_min: float = 0.10
    alns_sa_auto_calibration_enabled: bool = False
    alns_sa_sample_count: int = 24
    alns_sa_delta_quantile: float = 0.75
    alns_sa_initial_worse_accept_probability: float = 0.20
    alns_sa_cooling_rate: float = 0.985
    alns_sa_minimum_temperature: float = 1e-4
    alns_sa_reheat_enabled: bool = False
    alns_safe_overlay_enabled: bool = True
    alns_destroy_existing_enabled: bool = False
    alns_protect_recent_goal_steps: int = 10
    alns_protect_progress_epsilon_m: float = 20.0
    alns_stale_goal_steps: int = 28
    erc_tc_support_b_dynamic_second_chain_enabled: bool = False
    erc_tc_support_b_second_chain_min_step: int = 70
    erc_tc_support_b_second_chain_max_deliveries: int = 4
    erc_stalled_routine_ownership_repair_enabled: bool = False
    erc_stalled_routine_ownership_min_step: int = 40
    erc_stalled_routine_ownership_exposure_steps: int = 48
    erc_stalled_routine_ownership_max_repairs_per_step: int = 1
    erc_unassigned_routine_repair_enabled: bool = False
    erc_unassigned_routine_repair_min_step: int = 70
    erc_unassigned_routine_repair_max_per_step: int = 1
    # B-only rescue for a pending routine whose layer-1 contract is stale but
    # no executable goal/claim is currently published.  The rescue is
    # bounded by lifeline and step gates in the mainline planner.
    erc_b_orphaned_routine_rescue_enabled: bool = False
    erc_b_orphaned_routine_rescue_min_step: int = 120
    erc_b_orphaned_routine_rescue_max_lifeline_ratio: float = 0.80
    erc_last_routine_rescue_pending_threshold: int = 1
    erc_last_routine_rescue_min_completed_tc: int = 7
    # ERC execution guard: when an airborne UAV is already close to a
    # time-critical lightweight task, keep the delivery leg if battery/weather
    # still leave a short supported-recovery margin. This prevents near-finish
    # sorties from being aborted by a full-return estimate that ignores moving
    # truck recovery support.
    hrl_airborne_tc_completion_grace_enabled: bool = False
    hrl_airborne_tc_completion_grace_radius_m: float = 950.0
    hrl_airborne_tc_completion_grace_recovery_buffer_scale: float = 0.35
    hrl_airborne_tc_completion_grace_min_battery: float = 0.16
    hrl_airborne_tc_completion_grace_min_lifeline_steps: int = 4
    hrl_tc_global_assignment_adaptive_escape_enabled: bool = True
    hrl_tc_global_assignment_escape_min_map_size_m: float = 12000.0
    hrl_tc_global_assignment_escape_low_cover_threshold: float = 0.35
    hrl_tc_global_assignment_escape_max_lifeline_ratio: float = 0.55
    # Universal large-map exposure guard: far time-critical tasks may be
    # infeasible from the current UAV pose but still need early route/support
    # planning. Expose a small set of such tasks to the UAV assignment stage.
    hrl_timecritical_far_exposure_enabled: bool = True
    hrl_timecritical_far_exposure_min_map_size_m: float = 9000.0
    hrl_timecritical_far_exposure_extra: int = 1
    hrl_timecritical_far_exposure_max_lifeline_ratio: float = 0.55
    hrl_timecritical_far_exposure_min_gap_steps: int = 10
    hrl_timecritical_far_exposure_low_cover_threshold: float = 0.30
    hrl_timecritical_far_exposure_urgent_bypass_threshold: float = 0.88
    # Direct UAV priority terms for time-critical tasks.
    hrl_uav_timecritical_urgency_weight: float = 0.35
    hrl_uav_timecritical_lifeline_weight: float = 0.55
    hrl_uav_timecritical_critical_bonus: float = 0.55
    # Truck penalty down-scaling and support/recovery amplification for
    # low-lifeline time-critical tasks.
    hrl_truck_timecritical_penalty_scale_warning: float = 0.25
    hrl_truck_timecritical_penalty_scale_critical: float = 0.10
    hrl_truck_timecritical_support_amp_warning: float = 1.25
    hrl_truck_timecritical_support_amp_critical: float = 1.45
    hrl_truck_timecritical_recovery_amp_warning: float = 1.15
    hrl_truck_timecritical_recovery_amp_critical: float = 1.35
    hrl_truck_timecritical_supported_sortie_amp_warning: float = 1.12
    hrl_truck_timecritical_supported_sortie_amp_critical: float = 1.30
    # Support binding priority bonuses:
    # critical time-critical > warning time-critical > bulk > no-bound.
    hrl_support_bind_bonus_critical: float = 0.45
    hrl_support_bind_bonus_warning: float = 0.25
    hrl_support_bind_bonus_bulk: float = 0.10
    hrl_truck_no_normal_support_min_gain: float = 0.20
    hrl_truck_no_normal_support_urgency_floor: float = 0.55
    # Support-to-delivery binding gate:
    # only keep support candidates that can bind short-horizon delivery.
    hrl_support_bind_horizon_steps: int = 4
    hrl_support_bind_horizon_steps_large_map: int = 8
    hrl_support_requires_timecritical_binding: bool = True
    hrl_support_proxy_require_warning_bind_when_normal_reachable: bool = True
    hrl_support_proxy_warning_gate_min_map_size_m: float = 9000.0
    hrl_support_fallback_allow_bulk_binding: bool = False
    hrl_support_candidate_max_distance_m: float = 5000.0
    hrl_support_bind_enforce_min_map_size_m: float = 9000.0
    # Hard support-quality gate: only allow support candidates that create
    # real near-term serviceability gain for high-pressure/island tasks.
    hrl_support_require_actionable_gain: bool = True
    hrl_support_actionable_min_gain_score: float = 0.22
    hrl_support_actionable_min_new_serviceable: float = 0.5
    hrl_support_actionable_post_distance_m: float = 2200.0
    # Support conversion chain hardening:
    # (1) dispatch a bound UAV when support has a concrete time-critical bind,
    # (2) back off repeated no-gain support loops per truck.
    hrl_support_bound_dispatch_enabled: bool = True
    hrl_support_no_gain_backoff_enabled: bool = True
    hrl_support_no_gain_streak_threshold: int = 3
    hrl_support_no_gain_cooldown_steps: int = 8
    hrl_support_max_trucks_when_normal_pending: int = 0
    hrl_support_budget_require_warning_when_normal: bool = False
    hrl_support_relay_reserve_enabled: bool = True
    hrl_support_relay_min_critical_timecritical: int = 2
    hrl_support_relay_cover_threshold: float = 0.35
    # Medium-scale fallback: allow at most a small number of truck diversions
    # from normal to critical time-critical support when all trucks are occupied.
    hrl_support_critical_diversion_enabled: bool = True
    hrl_support_critical_diversion_max_trucks: int = 1
    hrl_support_critical_diversion_max_map_size_m: float = 9000.0
    hrl_support_critical_diversion_cover_threshold: float = 0.38
    hrl_uav_cover_eval_max_distance_m: float = 2800.0
    # Multi-task support-gain proxy: evaluate a small nearby emergency set
    # to avoid underestimating support value in large maps.
    hrl_support_gain_multi_task_k: int = 3
    # Docked UAV feasibility should honor launch-gate safety result to reduce
    # assignment/execution mismatch on recovery margin.
    hrl_uav_docked_require_launch_gate_strict: bool = True
    # A docked UAV may keep an emergency goal while its truck moves toward a
    # recoverable launch window. This is opt-in to keep existing baselines stable.
    hrl_docked_uav_soft_invalid_hold_enabled: bool = False
    # When a UAV is docked/following a truck, only switch it from the truck
    # anchor to a TC task if that anchor can launch a complete sortie now.
    hrl_uav_anchor_to_tc_requires_actionable_enabled: bool = False
    # Soft clamp for long, low-conversion truck support:
    # only suppress support when it is long-distance, low-gain, and a direct
    # executable delivery candidate already exists.
    hrl_support_soft_clamp_enabled: bool = False
    hrl_support_soft_clamp_long_distance_m: float = 2200.0
    hrl_support_soft_clamp_min_gain: float = 0.30
    hrl_support_soft_clamp_bindable_min_new_serviceable: int = 1
    hrl_support_soft_clamp_require_direct_delivery_candidates: bool = True
    # Escape hatch for L-C style pressure pockets: temporarily relax clamp
    # only when support is the only viable conversion path.
    hrl_support_escape_hatch_enabled: bool = False
    hrl_support_escape_hatch_min_pending_emergency: int = 6
    hrl_support_escape_hatch_min_gain: float = 0.32
    hrl_support_escape_hatch_min_urgency: float = 0.60
    # Additional escape hatch: allow support when time-critical task is already
    # in warning/critical lifeline zone and UAV cover is low.
    hrl_support_escape_hatch_allow_low_cover_timecritical: bool = True
    hrl_support_escape_hatch_low_cover_threshold: float = 0.32
    # TC support-required chain:
    # classify time-critical tasks as direct-feasible, support-required, or
    # infeasible; support-required tasks reserve one truck-UAV-task chain instead
    # of repeatedly entering the direct launch gate.
    erc_tc_support_required_enabled: bool = False
    erc_tc_support_lock_steps: int = 18
    erc_tc_support_max_active_chains: int = 2
    erc_tc_support_latest_start_margin_steps: int = 4
    erc_tc_support_max_setup_steps: int = 28
    erc_tc_support_min_gain_score: float = 0.10
    erc_tc_support_post_distance_m: float = 3600.0
    erc_tc_support_high_urgency_post_distance_m: float = 0.0
    erc_tc_support_high_urgency_threshold: float = 0.88
    erc_tc_support_max_lifeline_ratio: float = 0.70
    erc_tc_support_require_follower_uav: bool = True
    erc_tc_support_allow_normal_preemption: bool = True
    erc_tc_support_anchor_waypoint_enabled: bool = False
    erc_tc_support_anchor_search_node_cap: int = 320
    erc_tc_support_anchor_search_min_balance: float = 0.20
    erc_tc_support_release_uav_for_direct_tc_enabled: bool = True
    # Routine progress watchdog:
    # keep reachable routine work progressing, but let trucks with no executable
    # routine task become full-time UAV support carriers.
    erc_routine_progress_watchdog_enabled: bool = False
    erc_routine_watchdog_full_time_support_enabled: bool = False
    # Progress-aware switch control toggles (component ablations).
    hrl_uav_emergency_commit_hold_enabled: bool = True
    hrl_truck_routine_stuck_escape_enabled: bool = True
    hrl_truck_routine_escape_allow_any_alt_when_stuck: bool = False
    hrl_truck_routine_escape_allow_any_alt_min_step: int = 0
    hrl_far_routine_bootstrap_enabled: bool = False
    hrl_far_routine_bootstrap_window_steps: int = 20
    hrl_far_routine_bootstrap_min_map_size_m: float = 9000.0
    hrl_far_routine_bootstrap_min_distance_m: float = 7000.0
    hrl_routine_localize_eta_exit_enabled: bool = True
    # UAV reservation + truck assist-waypoint coordination (ERC-RHC).
    hrl_uav_task_reservation_exec_enabled: bool = True
    hrl_uav_task_reservation_stale_steps: int = 24
    # A selected task is hidden from all non-owner agents until it is resolved
    # or its owner becomes unavailable.
    hrl_task_exclusive_contract_enabled: bool = True
    # When no truck can reach a bulk task after a road update, promote it to
    # a UAV-served emergency task instead of letting its truck wait forever.
    hrl_unreachable_normal_uav_takeover_enabled: bool = True
    # A UAV sortie is an atomic delivery commitment: after launch, preserve
    # its bound emergency task until delivery or a hard safety invalidation.
    uav_strict_sortie_contract_enabled: bool = True
    # Keep a loaded docked UAV's assigned emergency task while it is charging,
    # so replanning cannot replace a viable near-term sortie before launch.
    uav_docked_sortie_commitment_enabled: bool = True
    # Experimental attraction-based cluster dispatch. A truck and its docked
    # UAVs are coordinated through exclusive task claims.
    hrl_attraction_dispatch_enabled: bool = False
    hrl_attraction_normal_weight: float = 4.0
    hrl_attraction_emergency_weight: float = 1.0
    # Hierarchical route-plan v2.  This is the new paper mainline:
    # layer 1 builds one complete truck--UAV cooperative task line, while
    # layer 2 only executes the current stop and reports invalidated suffixes.
    # The legacy attraction/stepwise branches remain available when disabled.
    hrl_route_plan_v2_enabled: bool = False
    hrl_route_plan_detour_trigger_ratio: float = 1.35
    hrl_route_plan_backup_anchor_count: int = 3
    hrl_route_plan_alns_iterations: int = 4
    # Optional hard ceiling for objective evaluations in a matched-budget
    # control. Zero keeps the historical uncapped behaviour.
    hrl_route_plan_alns_objective_evaluation_budget: int = 0
    hrl_route_plan_min_replan_interval: int = 3
    # Candidate-only initial emergency ordering override.  Formal ER-HLNS
    # keeps the spatial-overload guard; an explicit candidate may force the
    # remaining-lifeline ordering for a controlled ablation/pilot.
    hrl_route_plan_force_initial_lifeline_ordering_enabled: bool = False
    # Test-only mechanism switch: when disabled, keep the initial cooperative
    # plan and local execution safeguards but do not reinstall a route plan in
    # response to road/task/queue feedback.
    hrl_route_plan_event_replan_enabled: bool = True
    # Repair-scope control switch: on a repair event, globally re-auction all
    # pending/unstarted contracts while retaining claimed, in-service, and
    # airborne work. False preserves the execution-consistent unresolved-
    # suffix contract used by the paper mainline.
    hrl_route_plan_global_pending_reauction_on_repair_enabled: bool = False
    hrl_route_plan_uav_scout_enabled: bool = True
    hrl_route_plan_anchor_candidate_cap: int = 320
    hrl_route_plan_emergency_lateness_weight: float = 18.0
    hrl_route_plan_normal_lateness_weight: float = 4.0
    # Proactive protection for routine-bulk nodes expected to be isolated by
    # the configured road-blockage curve.  It is active only in B/C; A keeps
    # the ordinary distance/deadline objective.
    hrl_route_plan_routine_disconnect_protection_enabled: bool = True
    hrl_route_plan_routine_disconnect_risk_weight: float = 1.20
    # Layer-1 global disconnection prediction. Unlike the retained local-risk
    # proxy above, this uses the weighted minimum cut from the truck fleet to
    # each routine task and gives the most structurally exposed half a latest
    # safe visit time under B/C blockage curves.
    hrl_route_plan_global_disconnect_constraint_enabled: bool = False
    hrl_route_plan_global_disconnect_protected_fraction: float = 0.50
    hrl_route_plan_global_disconnect_edge_length_ref_m: float = 1500.0
    hrl_route_plan_global_disconnect_safe_tau_base: float = 0.70
    hrl_route_plan_global_disconnect_safe_tau_cut_scale: float = 0.45
    # Execution uncertainty contributed by each preceding UAV emergency
    # contract (launch settling, recovery, recharge/reload and event repair).
    # This affects only the predicted safe-visit rank, never task deadlines.
    hrl_route_plan_global_disconnect_emergency_queue_buffer_steps: int = 0
    # At most this many weakest-cut routine tasks seed distinct truck route
    # heads. The complete candidate is accepted only if emergency miss and
    # lateness remain unchanged.
    hrl_route_plan_global_disconnect_head_commitment_count: int = 2
    # Lexicographic emergency workload balance across cooperative truck--UAV
    # units. It is evaluated after miss/lateness and before routine exposure.
    hrl_route_plan_global_emergency_balance_enabled: bool = False
    hrl_route_plan_global_emergency_balance_trigger_count: int = 5
    # Capacity-aware emergency allocation uses mounted UAV count as parallel
    # throughput and remaining physical packages as a hard upper bound.
    hrl_route_plan_capacity_aware_emergency_allocation_enabled: bool = False
    hrl_route_plan_enforce_emergency_inventory_budget_enabled: bool = False
    hrl_route_plan_initial_emergency_capacity_repair_enabled: bool = False
    hrl_route_plan_capacity_repair_tail_only_enabled: bool = False
    hrl_route_plan_risk_weight: float = 0.20
    hrl_route_plan_switch_penalty_steps: float = 1.5
    hrl_route_plan_bulk_relay_enabled: bool = False
    # Historical road-isolated bulk-relay implementation is retained for
    # archived experiments, but the frozen paper model disables it: an 800 kg
    # basic-supply task is truck-only.
    hrl_route_plan_bulk_relay_uav_count: int = 2
    hrl_route_plan_bulk_relay_payload_kg: float = 50.0
    # Joint corridor insertion: emergency sortie feasibility is the hard
    # skeleton; direct routine tasks may be inserted anywhere along it when
    # their marginal detour and emergency delay remain bounded.
    hrl_route_plan_joint_corridor_enabled: bool = True
    hrl_route_plan_normal_max_marginal_cost_steps: float = 90.0
    hrl_route_plan_normal_max_emergency_delay_steps: int = 12
    hrl_route_plan_emergency_deadline_reserve_steps: int = 20
    # A running emergency contract may move as one unit (truck + UAV) when
    # its estimated remaining completion time has stopped improving and a
    # different cooperative unit can complete it materially sooner.
    hrl_route_plan_contract_transfer_enabled: bool = True
    hrl_route_plan_contract_stall_steps: int = 12
    hrl_route_plan_contract_transfer_min_eta_gain_steps: float = 20.0
    hrl_route_plan_contract_transfer_cooldown_steps: int = 15
    hrl_route_plan_stockout_transfer_enabled: bool = False
    hrl_route_plan_owner_carrier_mismatch_repair_enabled: bool = False
    # Controlled local reauction for routine bulk tasks.  Contracts remain
    # stable outside the road-radius gate; a stocked, assist-free truck may
    # take a clearly faster nearby task at most once per episode.
    hrl_route_plan_routine_dynamic_reassignment_enabled: bool = True
    hrl_route_plan_routine_dynamic_reassignment_radius_m: float = 800.0
    hrl_route_plan_routine_dynamic_reassignment_min_eta_gain_steps: float = 3.0
    hrl_route_plan_routine_dynamic_reassignment_min_eta_gain_ratio: float = 0.20
    hrl_route_plan_routine_dynamic_reassignment_max_transfers: int = 1
    hrl_route_plan_routine_dynamic_reassignment_lock_steps: int = 5
    # Candidate-only routine suffix repair.  The feature is intentionally
    # disabled by default; candidate overlays may opt in without changing the
    # formal route planner or any physical/safety parameter.  When enabled,
    # only a future, unstarted NORMAL suffix may be released to a different
    # truck after a robust road ETA reports low slack, an unreachable route or
    # a stalled progress window.
    hrl_route_plan_risk_slack_routine_repair_enabled: bool = False
    hrl_route_plan_risk_slack_routine_slack_steps: int = 20
    hrl_route_plan_risk_slack_routine_stall_steps: int = 12
    hrl_route_plan_risk_slack_routine_max_transfers: int = 1
    hrl_route_plan_risk_slack_routine_eta_gain_steps: float = 3.0
    hrl_route_plan_risk_slack_routine_eta_gain_ratio: float = 0.20
    hrl_route_plan_risk_slack_routine_radius_m: float = 800.0
    hrl_route_plan_risk_slack_routine_reserved_inventory_guard_enabled: bool = False
    # Candidate-only R4 repair: a stalled, never-started NORMAL task may be
    # atomically taken over once by an idle, stocked truck.  The default is
    # inert so formal ER-HLNS and the frozen environment are unchanged.
    hrl_route_plan_r4_routine_takeover_enabled: bool = False
    hrl_route_plan_r4_routine_takeover_stall_steps: int = 12
    hrl_route_plan_r4_routine_takeover_max_transfers: int = 1
    hrl_route_plan_r4_routine_takeover_radius_m: float = 0.0
    # Candidate-only idle routine dispatch.  A stocked truck with no active
    # route may accept one pending NORMAL task only while every active
    # emergency retains the configured deadline reserve.
    hrl_route_plan_idle_routine_dispatch_enabled: bool = False
    hrl_route_plan_idle_routine_dispatch_emergency_reserve_steps: int = 12
    hrl_route_plan_idle_routine_dispatch_max_per_step: int = 1
    # Candidate-only LB pilot: build a quota-balanced route for all task
    # classes before execution, then apply bounded normal-task watchdogs.
    # Defaults are deliberately off so formal ER-HLNS remains unchanged.
    hrl_route_plan_balanced_all_tasks_enabled: bool = False
    hrl_route_plan_balanced_all_tasks_v2_enabled: bool = False
    hrl_route_plan_balanced_all_tasks_v2_after_launch_only: bool = True
    hrl_route_plan_balanced_all_tasks_v2_reauction_deadline_guard_enabled: bool = True
    hrl_route_plan_balanced_all_tasks_v2_aggressive_pending_auction_enabled: bool = False
    hrl_route_plan_balanced_all_tasks_v3_enabled: bool = False
    hrl_route_plan_balanced_all_tasks_v3_tail_insert_after_launch: bool = False
    hrl_route_plan_balanced_all_tasks_v5_launch_first_enabled: bool = False
    hrl_route_plan_balanced_all_tasks_normal_first_enabled: bool = False
    hrl_route_plan_balanced_all_tasks_max_normal_per_truck: int = 0
    hrl_route_plan_balanced_all_tasks_allow_emergency_tradeoff: bool = False
    hrl_route_plan_balanced_all_tasks_emergency_lateness_tolerance_steps: int = 0
    hrl_route_plan_balanced_all_tasks_watchdog_stall_steps: int = 10
    hrl_route_plan_balanced_all_tasks_watchdog_near_distance_m: float = 300.0
    hrl_route_plan_balanced_all_tasks_watchdog_max_transfers: int = 1
    hrl_route_plan_balanced_all_tasks_watchdog_transfer_min_gain_steps: float = 3.0
    hrl_route_plan_balanced_all_tasks_watchdog_transfer_min_gain_ratio: float = 0.20
    # Candidate-only execution repair: after a balanced (non-UAV-priority)
    # route is installed, allow a pending emergency suffix to move ahead of a
    # NORMAL head only when a newly blocked road materially increases the
    # NORMAL ETA and the emergency is already inside its lifeline/deadline
    # reserve.  Defaults remain inert for the formal planner.
    hrl_route_plan_conditional_road_emergency_promotion_enabled: bool = False
    hrl_route_plan_conditional_road_emergency_promotion_eta_increase_steps: int = 12
    hrl_route_plan_conditional_road_emergency_promotion_near_normal_distance_m: float = 1000.0
    hrl_route_plan_conditional_road_emergency_promotion_reserve_steps: int = 8
    hrl_route_plan_conditional_road_emergency_promotion_cooldown_steps: int = 24
    # The promoted emergency must also gain a material amount of ETA and be
    # directly reachable within its remaining deadline/lifeline budget.  This
    # prevents a road shock from causing a speculative suffix reorder that
    # harms other emergency sorties.
    hrl_route_plan_conditional_road_emergency_promotion_min_gain_steps: int = 6
    # V8 candidate: evaluate keep-versus-promote locally before mutating a
    # route.  The check is algorithm-only and remains disabled by default.
    hrl_route_plan_shadow_promotion_enabled: bool = False
    hrl_route_plan_shadow_promotion_min_gain_steps: int = 4
    hrl_route_plan_shadow_promotion_normal_tolerance_steps: int = 0
    # Candidate-only upper-planning shadow selector.  It compares the
    # emergency-first route skeleton with a normal-first alternative from the
    # same observed state, and may switch only when emergency terms are
    # non-worse and the alternative adds at least one on-time task.
    hrl_route_plan_shadow_total_coverage_enabled: bool = False
    hrl_route_plan_shadow_total_coverage_min_gain_tasks: int = 1
    hrl_route_plan_shadow_total_coverage_min_routine_slack_steps: int = 0
    hrl_route_plan_shadow_total_coverage_max_routine_distance_ratio: float = 1.0
    # Candidate-only repair for a NORMAL contract that reached a truck but
    # never entered service.  It may republish the same direct goal once, or
    # use one safe alternate truck when the owner is unavailable.
    hrl_route_plan_routine_service_start_rescue_enabled: bool = False
    hrl_route_plan_routine_service_start_rescue_stall_steps: int = 10
    hrl_route_plan_routine_service_start_rescue_near_distance_m: float = 300.0
    hrl_route_plan_routine_service_start_rescue_max_transfers: int = 1
    # Candidate-only extension: when the current owner is stalled, allow one
    # materially faster idle truck to take over.  Formal defaults remain off.
    hrl_route_plan_routine_service_start_rescue_allow_stalled_owner_transfer: bool = False
    hrl_route_plan_routine_service_start_rescue_transfer_min_gain_steps: float = 3.0
    hrl_route_plan_routine_service_start_rescue_transfer_min_gain_ratio: float = 0.20
    # Candidate-only LB hard coverage rescue. Defaults remain inert for the
    # formal ER-HLNS method.
    hrl_route_plan_stalled_normal_cleanup_enabled: bool = False
    hrl_route_plan_hard_normal_rescue_enabled: bool = False
    hrl_route_plan_hard_normal_rescue_stall_steps: int = 12
    hrl_route_plan_hard_normal_rescue_max_per_call: int = 2
    hrl_route_plan_hard_normal_rescue_airborne_parallel_enabled: bool = False
    hrl_route_plan_hard_normal_rescue_tail_after_airborne: bool = False
    # Candidate-only orphan guard.  These controls deliberately default off
    # so formal ER-HLNS keeps the existing bounded stalled-owner behavior.
    hrl_route_plan_hard_normal_rescue_orphan_only_enabled: bool = False
    hrl_route_plan_hard_normal_rescue_pending_head_guard_enabled: bool = False
    hrl_route_plan_hard_normal_rescue_candidate_head_guard_enabled: bool = False
    hrl_route_plan_hard_normal_rescue_no_truck_once_enabled: bool = False
    hrl_route_plan_hard_normal_rescue_no_truck_cooldown_enabled: bool = False
    hrl_route_plan_hard_normal_rescue_no_truck_cooldown_steps: int = 24
    # Candidate-only runtime gate for adaptive hard coverage.  Formal default
    # remains disabled with no minimum orphan threshold.
    hrl_route_plan_hard_normal_rescue_adaptive_coverage_enabled: bool = False
    hrl_route_plan_hard_normal_rescue_min_orphan_pending: int = 0
    # Candidate-only truck movement while its UAV is already airborne on the
    # current emergency contract.  Formal default remains conservative/off.
    hrl_route_plan_parallel_routine_emergency_after_launch_enabled: bool = False
    # Candidate-only mixed coverage objective.  When enabled, emergency
    # misses/lateness remain hard lexicographic priorities; among equally safe
    # plans, faster and higher-urgency NORMAL tasks receive preference.
    hrl_route_plan_mixed_coverage_enabled: bool = False
    hrl_route_plan_mixed_coverage_emergency_reserve_steps: int = 30
    # Narrow emergency robustness controls.  Queue promotion changes only a
    # still-pending suffix; the launch watchdog may skip goal-stability delay
    # only after the complete environment launch gate remains safe.
    # Rejected pilot retained for audit: unrestricted suffix promotion caused
    # repeated queue reshuffles and reduced the protected 10-seed mean.
    hrl_route_plan_emergency_starvation_promotion_enabled: bool = False
    hrl_route_plan_emergency_starvation_reserve_steps: int = 8
    # The watchdog was outcome-neutral in the same pilot; keep the mechanism
    # available for ablation/diagnosis but outside the paper mainline.
    hrl_route_plan_emergency_launch_watchdog_enabled: bool = False
    hrl_route_plan_emergency_launch_watchdog_wait_steps: int = 2
    # Preserve an accepted loaded airborne sortie through its delivery leg.
    # A terminal approach may bypass the legacy direct-return predictor only
    # while retaining this battery reserve; recovery remains mandatory after
    # service.
    uav_authoritative_sortie_goal_precedence_enabled: bool = False
    uav_terminal_delivery_commitment_enabled: bool = False
    uav_terminal_delivery_min_reserve_fraction: float = 0.20
    # A spare UAV may preview only a direct-safe emergency suffix belonging
    # to its own truck; this cannot add a truck detour or recovery obligation.
    # Retained for ablation only: the 10-seed pilot showed that early package
    # consumption can disturb later contracts despite direct-return safety.
    hrl_route_plan_direct_safe_secondary_emergency_enabled: bool = False
    # Retained for ablation only.  Scalar lifecycle weighting traded two
    # emergency completions for two routine completions in the 10-seed pilot.
    hrl_route_plan_uav_lifecycle_cost_enabled: bool = True
    hrl_route_plan_uav_lifecycle_cost_weight: float = 0.75
    hrl_route_plan_lexicographic_objective_enabled: bool = True
    # Local robustness guards.  They are deliberately narrower than changing
    # the global ALNS objective: only a deadline-infeasible suffix or a loaded
    # UAV already at the task may break the original exclusive contract.
    hrl_route_plan_onsite_takeover_enabled: bool = False
    hrl_route_plan_onsite_takeover_min_battery: float = 0.98
    hrl_route_plan_deadline_rescue_enabled: bool = True
    hrl_route_plan_deadline_rescue_reserve_steps: int = 6
    hrl_route_plan_contract_consistency_guard_enabled: bool = True
    # Tail-robust execution closure. A loaded docked UAV retains the task
    # owned by its current route-contract version even at full SOC; a new
    # owner/version releases it atomically. Once emergency work is terminal,
    # reachable routine residuals receive stable nearest-stocked-truck owners.
    hrl_route_plan_atomic_contract_enabled: bool = True
    uav_docked_contract_owner_lock_enabled: bool = True
    hrl_route_plan_residual_normal_coverage_enabled: bool = True
    hrl_route_plan_idle_normal_fallback_enabled: bool = False
    hrl_route_plan_idle_normal_fallback_max_pending_emergency: int = 2
    hrl_route_plan_idle_normal_fallback_stall_steps: int = 30
    hrl_route_plan_residual_normal_bipartite_matching_enabled: bool = False
    hrl_route_plan_stalled_queue_rescue_enabled: bool = True
    # Candidate-only fairness gate: do not create the first hidden emergency
    # queue rescue while every NORMAL task is still untouched.  An explicit
    # urgent-queue condition may bypass this gate; the default remains off so
    # the formal/frozen planner behavior is unchanged.
    hrl_route_plan_stalled_queue_rescue_normal_service_gate_enabled: bool = False
    hrl_route_plan_stalled_queue_rescue_steps: int = 30
    hrl_route_plan_stalled_queue_rescue_min_pending: int = 1
    # Fleet-level layer-1 repair and docked rescue-handshake timeout.  These
    # are separate from the suffix-rescue threshold above.
    hrl_route_plan_queue_starvation_steps: int = 30
    hrl_route_plan_queue_starvation_min_pending: int = 1
    hrl_route_plan_queue_rescue_launch_timeout_steps: int = 5
    hrl_route_plan_queue_urgent_rescue_horizon_steps: int = 0
    # Optional second stage for a stalled hidden emergency suffix.  When no
    # sortie is safe from the truck's current node, move the truck to the
    # closest road-reachable launch anchor and validate the full launch gate
    # there.  An unsafe anchor is released; take-off is never forced.
    hrl_route_plan_stalled_queue_anchor_rescue_enabled: bool = False
    hrl_route_plan_stalled_queue_anchor_timeout_steps: int = 12
    hrl_route_plan_stalled_queue_max_active_rescues: int = 1
    # Once a truck has partially served a routine task, finish the available
    # local unloading before soft UAV support can pull it away.  Hard airborne
    # recovery remains allowed.
    hrl_route_plan_routine_multiround_commitment_enabled: bool = True
    hrl_route_plan_audit_enabled: bool = True
    hrl_uav_assist_enabled: bool = True
    hrl_uav_assist_max_extra_distance_m: float = 600.0
    hrl_uav_assist_max_extra_ratio: float = 0.20
    hrl_uav_assist_min_launch_distance_reduction_m: float = 400.0
    # Simplified ERC defaults: truck-idle is treated as low-value refresh unless explicitly promoted.
    hrl_truck_idle_hard_refresh_enabled: bool = False
    # Simplified ERC defaults: ranking_changed alone does not trigger hard map-impact refresh.
    hrl_map_update_allow_ranking_changed_impact: bool = False
    # Event scoring bonus is condition-enabled (hard-impact events can amplify gain).
    hrl_event_bonus_conditional_enabled: bool = True
    hrl_event_bonus_base_gain: float = 0.42
    hrl_event_bonus_hard_gain: float = 1.00
    # Generic ERC-RHC execution/switch controls (map-source agnostic).
    execution_commitment_enabled: bool = True
    execution_commitment_override_lifeline_margin: float = 0.10
    execution_no_switch_if_commit_active: bool = True
    execution_no_switch_if_current_goal_launchable: bool = True
    goal_switch_penalty_enabled: bool = True
    goal_switch_hard_cap_enabled: bool = True
    goal_switch_score_gain_min: float = 0.20
    goal_switch_eta_gain_min: float = 5.0
    timecritical_global_assignment_enabled: bool = True
    cluster_primary_task_enabled: bool = True
    task_reservation_enabled: bool = True
    recent_release_cooldown_enabled: bool = True
    unreachable_bulk_watchlist_enabled: bool = False
    support_force_dispatch_enabled: bool = False
    support_force_commit_steps: int = 10
    support_force_uav_preempt_enabled: bool = False
    truck_force_nonnull_goal_enabled: bool = False
    truck_loop_break_enabled: bool = False
    truck_loop_break_window_steps: int = 12
    # ERC mechanism-ablation flags (no algorithm expansion; switch-off only).
    erc_ablate_low_value_refresh: bool = False
    erc_ablate_map_ranking_refresh: bool = False
    erc_ablate_tc_global_assignment: bool = False
    erc_ablate_support_chain: bool = False
    erc_ablate_cluster_primary_reservation: bool = False
    erc_ablate_event_scoring_bonus: bool = False
    erc_ablate_normal_protection: bool = False
    # Legacy RC-prefixed fields kept for backward compatibility only.
    rc_strong_planner_mode_enabled: bool = False
    rc_unreachable_bulk_watchlist_enabled: bool = False
    rc_support_force_dispatch_enabled: bool = False
    rc_support_force_commit_steps: int = 10
    rc_support_force_uav_preempt_enabled: bool = False
    rc_support_chain_lock_enabled: bool = True
    rc_support_chain_override_lifeline_margin: float = 0.10
    rc_truck_force_nonnull_goal_enabled: bool = False
    rc_truck_loop_break_enabled: bool = False
    rc_truck_loop_break_window_steps: int = 12
    # Symptom-driven map-refresh gate (H2 as conditional patch, not global on).
    hrl_conditional_h2_refresh_enabled: bool = False
    hrl_conditional_h2_new_info_threshold: int = 2
    hrl_conditional_h2_delivery_stall_steps: int = 10
    hrl_conditional_h2_support_quality_max: float = 0.55
    hrl_conditional_h2_support_distance_threshold_m: float = 85000.0
    hrl_conditional_h2_island_serviceability_low: float = 0.45
    # Anti-oscillation guard for truck NORMAL-to-NORMAL switching.
    hrl_truck_normal_to_normal_switch_min_improve_ratio: float = 0.15
    hrl_truck_normal_to_normal_switch_min_score_gain: float = 0.10
    # ABA back-and-forth block window for truck NORMAL task switching.
    hrl_truck_normal_aba_block_steps: int = 12
    # Routine-progress watchdog for large-map deadline pressure.
    hrl_truck_routine_stuck_persist_steps: int = 5
    hrl_truck_routine_progress_epsilon_m: float = 30.0
    hrl_truck_routine_escape_min_eta_gain_steps: int = 6
    hrl_truck_routine_escape_min_score_gain: float = 0.12
    hrl_truck_emergency_cover_threshold_when_normal_reachable: float = 0.30
    hrl_truck_support_gain_min_when_normal_reachable: float = 0.45
    hrl_uav_ride_stall_release_enabled: bool = False
    hrl_uav_ride_stall_trigger_steps: int = 6
    hrl_uav_ride_stall_bonus: float = 0.28
    hrl_uav_ride_stall_max_dist_m: float = 1200.0
    pbrs_gamma: float = 0.99
    pbrs_distance_norm_m: float = 3000.0
    # Weather / comms
    stochastic_weather: bool = True
    weather_num_vortices: int = 3
    weather_num_rain_stations: int = 5
    base_rainfall_mmh: float = 12.0
    base_wind_mps: float = 6.0
    rain_idw_power: float = 2.0
    rain_idw_eps_m: float = 1.0
    weather_osc_amp: float = 0.18
    weather_osc_w1: float = 0.05
    weather_osc_w2: float = 0.03
    wind_vortex_strength_min: float = -1.0
    wind_vortex_strength_max: float = 1.0
    wind_vortex_radius_min_m: float = 300.0
    wind_vortex_radius_max_m: float = 900.0
    wind_vortex_scale_mps: float = 1.8
    # Percolation/logistic collapse model
    logistic_beta0: float = -3.4
    logistic_beta_slope: float = 1.4
    logistic_beta_rain: float = 1.2
    logistic_beta_quake: float = 1.8
    logistic_beta_sr: float = 3.0
    logistic_beta_re: float = 1.1
    logistic_beta_vbase: float = 1.0
    # Additional edge-structure effects (explicit terms).
    logistic_beta_bldg: float = 0.9
    logistic_beta_infra: float = 1.3
    logistic_beta_length: float = 0.8
    logistic_beta_lr: float = 0.9  # length-rain interaction
    lambda_aggressive: float = 0.045
    lambda_residual: float = 0.001
    percolation_lock_threshold: float = 0.35
    lambda_warmup_steps: int = 30
    lambda_warmup_min_factor: float = 0.35
    stochastic_block_max_prob: float = 0.85
    # Blockage model v2: monotone asymptotic blockage curve (paper mainline).
    blockage_curve_enabled: bool = True
    blockage_curve_gain_k: float = 1.0
    blockage_curve_gate_cap: float = 0.12
    # B/C share the same road-disruption trajectory.  C is reserved for the
    # communication-blackout condition, not a harsher road environment.
    blockage_asymptote_B: float = 0.25
    blockage_asymptote_C: float = 0.25
    blockage_asymptote_scale_M: float = 1.00
    blockage_asymptote_scale_L: float = 1.00
    blockage_asymptote_scale_R: float = 0.75
    blockage_tau_steps_M: float = 110.0
    blockage_tau_steps_L: float = 130.0
    blockage_tau_steps_R: float = 120.0
    # Deprecated compatibility fields. Road recovery is not part of the paper
    # mainline; these keys are accepted only so old YAML files still load.
    edge_repair_prob_per_step: float = 0.0
    edge_repair_min_steps_blocked: int = 10
    edge_reopen_prob_per_step: float = 0.0
    edge_reopen_min_steps_blocked: int = 10
    l_edge_reopen_prob_per_step: float = 0.0
    l_edge_reopen_min_steps_blocked: int = 20
    # Risk-aware routing: decision shortest-path uses hazard-weighted edge cost
    # instead of pure geometric distance.
    road_risk_aware_routing_enabled: bool = True
    road_risk_edge_prob_weight: float = 1.25
    road_risk_vulnerability_weight: float = 0.35
    road_risk_cost_multiplier_cap: float = 3.0
    # Diagnostic-only acceleration: retain route-distance entries until the
    # known blocked-edge set changes.  Disabled by default because risk-aware
    # edge costs can otherwise change between road updates.
    decision_sp_cache_road_version_only: bool = False
    # Scenario C communication contract: spatially correlated, persistent
    # blackout zones.  It replaces the legacy independent per-agent/per-step
    # draw for paper experiments, while retaining ``iid_risk_v0`` for legacy
    # reproduction only.
    comm_blackout_model: str = "regional_persistent_v1"
    # Fraction of time-critical task nodes deliberately placed in blackout
    # regions.  The regions themselves also affect any agent physically inside.
    comm_blackout_emergency_coverage: float = 0.30
    comm_blackout_zone_count: int = 2
    comm_blackout_zone_radius_map_fraction: float = 0.12
    comm_blackout_start_step: int = 20
    comm_blackout_duration_steps: int = 6
    comm_blackout_recovery_steps: int = 10
    # Experiment-only hard-off switch used by the preregistered 0% sensitivity
    # condition. This takes precedence over Scenario C's default enable rule.
    comm_blackout_force_disabled: bool = False
    # Legacy IID model parameters; used only when comm_blackout_model is
    # explicitly set to ``iid_risk_v0``.
    comm_block_prob: float = 0.10
    comm_risk_score_weight: float = 0.55
    enable_comm_blackout: bool = False
    # Stage-C road awareness:
    # - perfect: planner sees true blocked graph immediately (legacy behavior)
    # - shared: planner uses shared belief map updated by scout/contact events
    road_awareness_mode: str = "perfect"
    road_shared_awareness_enabled: bool = True
    road_shared_replan_on_update: bool = True
    road_uav_scout_enabled: bool = True
    road_truck_scout_enabled: bool = True
    road_uav_scout_radius_m: float = 340.0
    road_truck_scout_radius_m: float = 220.0
    truck_island_support_nonmonotonic_slack_m: float = 450.0
    # Delivery radius for UAV
    uav_delivery_radius_m: float = 40.0
    # Motion-aware capture factor for UAV delivery trigger:
    # effective radius = max(uav_delivery_radius_m, factor * uav_max_speed_mps * dt_seconds)
    # Stable frozen value.  The 0.9/1.0 swept-segment pilots are retained in
    # artifacts but regressed protected seeds, so they are not mainline.
    uav_delivery_capture_motion_factor: float = 0.80

    def __post_init__(self) -> None:
        sc = str(self.scenario).upper().strip()
        if sc == "A":
            object.__setattr__(self, "stochastic_weather", False)
            object.__setattr__(self, "base_rainfall_mmh", 0.0)
            object.__setattr__(self, "base_wind_mps", 0.0)
            object.__setattr__(self, "lambda_aggressive", 0.0)
            object.__setattr__(self, "lambda_residual", 0.0)
            object.__setattr__(self, "comm_block_prob", 0.0)
            object.__setattr__(self, "enable_comm_blackout", False)
        elif sc == "B":
            object.__setattr__(self, "stochastic_weather", True)
            object.__setattr__(self, "comm_block_prob", 0.0)
            object.__setattr__(self, "enable_comm_blackout", False)
        elif sc == "C":
            object.__setattr__(self, "stochastic_weather", True)
            object.__setattr__(
                self,
                "enable_comm_blackout",
                not bool(self.comm_blackout_force_disabled),
            )
            if float(self.comm_block_prob) <= 0.0:
                object.__setattr__(self, "comm_block_prob", 0.10)

        # Paper baseline contract: both B and C use shared road awareness.
        if sc in {"B", "C"}:
            object.__setattr__(self, "road_awareness_mode", "shared")
            object.__setattr__(self, "road_shared_awareness_enabled", True)
            object.__setattr__(self, "road_shared_replan_on_update", True)
            object.__setattr__(self, "road_uav_scout_enabled", True)
            object.__setattr__(self, "road_truck_scout_enabled", True)

        # 婢跺秵娼呮惔锕傤暕鐠佹拝绱欒ぐ鎾冲閸忓牆娴愰崠?M閿?
        # Map-complexity defaults.
        # L is defined as a mesoscopic decision graph (not intersection-level graph).
        cx = str(self.map_complexity).upper().strip()
        default_nodes = 40
        default_map_size = 5000.0
        explicit_num_nodes = int(self.num_nodes) != default_nodes
        explicit_n_nodes = int(self.n_nodes) != default_nodes
        explicit_map_size = float(self.map_size_m) != default_map_size

        if cx == "M" and not (explicit_num_nodes or explicit_n_nodes or explicit_map_size):
            object.__setattr__(self, "map_size_m", 5000.0)
            object.__setattr__(self, "num_nodes", 40)
            object.__setattr__(self, "n_nodes", 40)
            object.__setattr__(self, "min_node_spacing_m", 300.0)
            object.__setattr__(self, "redundant_edge_radius_m", 850.0)
            object.__setattr__(self, "redundant_edge_prob", 0.55)

        real_city_enabled = bool(self.real_case_enabled) and str(self.map_source).strip().lower() == "osm_dem"

        if cx == "L":
            if not real_city_enabled:
                object.__setattr__(self, "map_source", "disaster_map")
            if (not explicit_map_size) or float(self.map_size_m) < 12000.0:
                object.__setattr__(
                    self,
                    "map_size_m",
                    float(max(15000.0, float(self.real_case_size_m) if real_city_enabled else 15000.0)),
                )
            if (not explicit_num_nodes and not explicit_n_nodes) or max(int(self.num_nodes), int(self.n_nodes)) <= 120:
                if real_city_enabled:
                    object.__setattr__(self, "num_nodes", max(int(self.num_nodes), 320))
                    object.__setattr__(self, "n_nodes", max(int(self.n_nodes), 320))
                else:
                    object.__setattr__(self, "num_nodes", 400)
                    object.__setattr__(self, "n_nodes", 400)
            if int(self.num_edges) <= 220:
                object.__setattr__(self, "num_edges", 580 if not real_city_enabled else max(int(self.num_edges), 460))
            if not real_city_enabled:
                object.__setattr__(self, "avg_degree_min", float(max(float(self.l_target_avg_degree_min), 1.0)))
                object.__setattr__(self, "avg_degree_max", float(max(float(self.l_target_avg_degree_max), float(self.avg_degree_min))))
                object.__setattr__(self, "min_node_spacing_m", float(max(float(self.min_node_spacing_m), 120.0)))
                object.__setattr__(self, "redundant_edge_radius_m", float(max(float(self.redundant_edge_radius_m), 1400.0)))
                object.__setattr__(self, "redundant_edge_prob", float(np.clip(float(self.redundant_edge_prob), 0.0, 1.0)))
        elif cx == "R":
            # R = real-road profile. Topology comes from real map inputs rather than
            # synthetic graph defaults, so only enforce a sane map size fallback.
            if (not explicit_map_size) and real_city_enabled:
                object.__setattr__(self, "map_size_m", float(max(15000.0, float(self.real_case_size_m))))

        # Keep legacy aliases synchronized after optional complexity defaults.
        if explicit_num_nodes and not explicit_n_nodes:
            object.__setattr__(self, "n_nodes", int(self.num_nodes))
        elif explicit_n_nodes and not explicit_num_nodes:
            object.__setattr__(self, "num_nodes", int(self.n_nodes))
        elif explicit_num_nodes and explicit_n_nodes and int(self.num_nodes) != int(self.n_nodes):
            object.__setattr__(self, "n_nodes", int(self.num_nodes))

        # Paper-facing manifests use the ``num_*`` names as the resolved source
        # of truth. Keep historical aliases synchronized so observation sizing,
        # diagnostics and legacy helpers cannot describe a different fleet.
        object.__setattr__(self, "n_trucks", int(self.num_trucks))
        object.__setattr__(self, "n_uavs", int(self.num_uavs))

        # Keep takeover switches synchronized with aliases.
        if bool(self.enable_auto_approach):
            object.__setattr__(self, "uav_auto_approach_enabled", True)
        if bool(self.enable_monitor_snap):
            object.__setattr__(self, "uav_monitor_snap_enabled", True)
        if bool(self.uav_auto_approach_enabled):
            object.__setattr__(self, "enable_auto_approach", True)
        if bool(self.uav_monitor_snap_enabled):
            object.__setattr__(self, "enable_monitor_snap", True)
        # Keep legacy payload parameters functional when capacity constraints are enabled.
        if not bool(self.ignore_payload_constraints):
            unit_kg = float(max(self.cargo_unit_kg, 1e-6))
            truck_cap_units = float(max(self.truck_payload_capacity_kg / unit_kg, 0.0))
            uav_cap_units = float(max(self.uav_payload_capacity_kg / unit_kg, 0.0))
            object.__setattr__(self, "truck_cargo_capacity_units", truck_cap_units)
            object.__setattr__(self, "uav_cargo_capacity_units", uav_cap_units)
            # Keep initial UAV payload within capacity when explicit payload is provided.
            if float(self.uav_payload_kg) > 0.0:
                init_units = float(max(self.uav_payload_kg / unit_kg, 0.0))
                object.__setattr__(
                    self,
                    "uav_cargo_capacity_units",
                    float(max(init_units, float(self.uav_cargo_capacity_units))),
                )

        object.__setattr__(self, "map_source", str(self.map_source).strip().lower())
        object.__setattr__(self, "real_case_enabled", bool(self.real_case_enabled))
        object.__setattr__(self, "real_city_case", str(self.real_city_case).strip().lower())
        if not str(self.real_case_name).strip():
            object.__setattr__(self, "real_case_name", str(self.real_city_case).strip())
        else:
            object.__setattr__(self, "real_case_name", str(self.real_case_name).strip())
        bbox_mode = str(self.real_case_bbox_mode).strip().lower()
        if bbox_mode not in {"center_size"}:
            bbox_mode = "center_size"
        object.__setattr__(self, "real_case_bbox_mode", bbox_mode)
        object.__setattr__(self, "real_case_center_lon", float(self.real_case_center_lon))
        object.__setattr__(self, "real_case_center_lat", float(self.real_case_center_lat))
        object.__setattr__(self, "real_case_size_m", float(max(float(self.real_case_size_m), 1000.0)))
        object.__setattr__(self, "real_case_keep_intersection_level", bool(self.real_case_keep_intersection_level))
        object.__setattr__(self, "real_case_cleaning_profile", str(self.real_case_cleaning_profile).strip().lower())
        object.__setattr__(self, "real_case_task_sampling_profile", str(self.real_case_task_sampling_profile).strip().lower())
        object.__setattr__(self, "real_case_hazard_profile", str(self.real_case_hazard_profile).strip().lower())
        object.__setattr__(self, "real_case_use_prepared_clean_graph", bool(self.real_case_use_prepared_clean_graph))
        object.__setattr__(self, "real_case_prepared_graphml_path", str(self.real_case_prepared_graphml_path).strip())
        object.__setattr__(self, "real_case_poi_json_path", str(self.real_case_poi_json_path).strip())
        object.__setattr__(self, "real_case_fixed_tasks_json_path", str(self.real_case_fixed_tasks_json_path).strip())
        earthquake_field_mode = str(self.earthquake_field_mode).strip().lower()
        if earthquake_field_mode not in {"legacy_proxy", "usgs_shakemap"}:
            raise ValueError("earthquake_field_mode must be legacy_proxy or usgs_shakemap")
        object.__setattr__(self, "earthquake_field_mode", earthquake_field_mode)
        rb_road_damage_mode = str(self.rb_road_damage_mode).strip().lower()
        if rb_road_damage_mode not in {"legacy_mixed", "earthquake_only"}:
            raise ValueError("rb_road_damage_mode must be legacy_mixed or earthquake_only")
        object.__setattr__(self, "rb_road_damage_mode", rb_road_damage_mode)
        object.__setattr__(self, "real_case_truck_only_road_cleaning", bool(self.real_case_truck_only_road_cleaning))
        object.__setattr__(self, "real_case_min_leaf_edge_m", float(max(float(self.real_case_min_leaf_edge_m), 0.0)))
        object.__setattr__(self, "real_case_postmerge_leaf_edge_m", float(max(float(self.real_case_postmerge_leaf_edge_m), 0.0)))
        object.__setattr__(self, "real_case_chain_collapse_angle_deg", float(np.clip(float(self.real_case_chain_collapse_angle_deg), 90.0, 179.0)))
        object.__setattr__(self, "real_case_junction_merge_cell_m", float(max(float(self.real_case_junction_merge_cell_m), 20.0)))
        object.__setattr__(self, "osm_graphml_path", str(self.osm_graphml_path).strip())
        object.__setattr__(self, "dem_npy_path", str(self.dem_npy_path).strip())
        object.__setattr__(
            self,
            "task_emergency_epicenter_bias",
            float(np.clip(float(self.task_emergency_epicenter_bias), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "task_emergency_sigma_m",
            float(max(float(self.task_emergency_sigma_m), 1.0)),
        )
        object.__setattr__(self, "blockage_curve_enabled", bool(self.blockage_curve_enabled))
        object.__setattr__(
            self,
            "blockage_curve_gain_k",
            float(max(float(self.blockage_curve_gain_k), 0.0)),
        )
        object.__setattr__(
            self,
            "blockage_curve_gate_cap",
            float(np.clip(float(self.blockage_curve_gate_cap), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "blockage_asymptote_B",
            float(np.clip(float(self.blockage_asymptote_B), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "blockage_asymptote_C",
            float(np.clip(float(self.blockage_asymptote_C), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "blockage_asymptote_scale_M",
            float(max(float(self.blockage_asymptote_scale_M), 0.0)),
        )
        object.__setattr__(
            self,
            "blockage_asymptote_scale_L",
            float(max(float(self.blockage_asymptote_scale_L), 0.0)),
        )
        object.__setattr__(
            self,
            "blockage_asymptote_scale_R",
            float(max(float(self.blockage_asymptote_scale_R), 0.0)),
        )
        object.__setattr__(
            self,
            "blockage_tau_steps_M",
            float(max(float(self.blockage_tau_steps_M), 1.0)),
        )
        object.__setattr__(
            self,
            "blockage_tau_steps_L",
            float(max(float(self.blockage_tau_steps_L), 1.0)),
        )
        object.__setattr__(
            self,
            "blockage_tau_steps_R",
            float(max(float(self.blockage_tau_steps_R), 1.0)),
        )
        # Paper mainline uses persistent episode-level blockages. Ignore any
        # historical road-repair/reopen knobs supplied by old configs.
        object.__setattr__(self, "edge_reopen_prob_per_step", 0.0)
        object.__setattr__(self, "edge_repair_prob_per_step", 0.0)
        object.__setattr__(self, "l_edge_reopen_prob_per_step", 0.0)

        object.__setattr__(
            self,
            "road_risk_aware_routing_enabled",
            bool(self.road_risk_aware_routing_enabled),
        )
        object.__setattr__(
            self,
            "road_risk_edge_prob_weight",
            float(max(float(self.road_risk_edge_prob_weight), 0.0)),
        )
        object.__setattr__(
            self,
            "road_risk_vulnerability_weight",
            float(max(float(self.road_risk_vulnerability_weight), 0.0)),
        )
        object.__setattr__(
            self,
            "road_risk_cost_multiplier_cap",
            float(max(float(self.road_risk_cost_multiplier_cap), 1.0)),
        )
        object.__setattr__(
            self,
            "wind_failure_threshold_mps",
            float(max(float(self.wind_failure_threshold_mps), 0.0)),
        )
        object.__setattr__(
            self,
            "wind_failure_risk_scale",
            float(np.clip(float(self.wind_failure_risk_scale), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "uav_low_soc_failure_threshold",
            float(np.clip(float(self.uav_low_soc_failure_threshold), 0.0, 0.30)),
        )
        object.__setattr__(
            self,
            "uav_low_soc_failure_risk_scale",
            float(np.clip(float(self.uav_low_soc_failure_risk_scale), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "uav_payload_energy_coeff_per_kg",
            float(max(float(self.uav_payload_energy_coeff_per_kg), 0.0)),
        )
        object.__setattr__(
            self,
            "uav_full_load_energy_penalty",
            float(max(float(self.uav_full_load_energy_penalty), 0.0)),
        )
        object.__setattr__(
            self,
            "uav_forced_recovery_bind_latency_steps",
            int(max(0, int(self.uav_forced_recovery_bind_latency_steps))),
        )
        object.__setattr__(
            self,
            "uav_terminal_failure_battery_floor",
            float(np.clip(float(self.uav_terminal_failure_battery_floor), 0.0, 0.20)),
        )
        object.__setattr__(
            self,
            "uav_launch_recovery_margin_scale",
            float(np.clip(float(self.uav_launch_recovery_margin_scale), 0.2, 1.5)),
        )
        object.__setattr__(
            self,
            "l_map_generation_max_attempts",
            int(max(1, int(self.l_map_generation_max_attempts))),
        )
        object.__setattr__(
            self,
            "l_map_variant",
            str(self.l_map_variant).strip() or "L_v1b_orientation_tighter",
        )
        mode_l = str(self.l_map_acceptance_mode).strip().lower()
        if mode_l not in {"strict", "relaxed", "realism_first"}:
            mode_l = "realism_first"
        object.__setattr__(self, "l_map_acceptance_mode", mode_l)
        mode_b = str(self.l_benchmark_mode).strip().lower()
        if mode_b not in {"old", "new"}:
            mode_b = "new"
        object.__setattr__(self, "l_benchmark_mode", mode_b)
        object.__setattr__(self, "l_map_cache_enabled", bool(self.l_map_cache_enabled))
        object.__setattr__(self, "l_map_cache_dir", str(self.l_map_cache_dir).strip() or "data/maps_core/L_map")
        object.__setattr__(self, "l_map_cache_size", int(max(1, int(self.l_map_cache_size))))
        object.__setattr__(self, "l_map_cache_index", int(self.l_map_cache_index))
        object.__setattr__(self, "synthetic_realism_task_sampling_enabled", bool(self.synthetic_realism_task_sampling_enabled))
        object.__setattr__(
            self,
            "synthetic_realism_task_sampling_min_map_size_m",
            float(max(float(self.synthetic_realism_task_sampling_min_map_size_m), 0.0)),
        )
        object.__setattr__(self, "l_min_node_spacing_m", float(max(float(self.l_min_node_spacing_m), 80.0)))
        object.__setattr__(self, "l_min_gateway_spacing_m", float(max(float(self.l_min_gateway_spacing_m), float(self.l_min_node_spacing_m))))
        object.__setattr__(self, "l_min_arterial_junction_spacing_m", float(max(float(self.l_min_arterial_junction_spacing_m), float(self.l_min_gateway_spacing_m))))
        object.__setattr__(self, "l_collinear_triangle_angle_deg", float(np.clip(float(self.l_collinear_triangle_angle_deg), 150.0, 179.5)))
        object.__setattr__(self, "l_task_min_spacing_m", float(max(float(self.l_task_min_spacing_m), 40.0)))
        object.__setattr__(self, "l_target_num_nodes_min", int(max(1, int(self.l_target_num_nodes_min))))
        object.__setattr__(self, "l_target_num_nodes_max", int(max(int(self.l_target_num_nodes_min), int(self.l_target_num_nodes_max))))
        object.__setattr__(self, "l_target_num_edges_min", int(max(1, int(self.l_target_num_edges_min))))
        object.__setattr__(self, "l_target_num_edges_max", int(max(int(self.l_target_num_edges_min), int(self.l_target_num_edges_max))))
        object.__setattr__(self, "l_target_avg_degree_min", float(max(float(self.l_target_avg_degree_min), 1.0)))
        object.__setattr__(self, "l_target_avg_degree_max", float(max(float(self.l_target_avg_degree_min), float(self.l_target_avg_degree_max))))
        object.__setattr__(self, "l_target_median_edge_length_m_min", float(max(float(self.l_target_median_edge_length_m_min), 1.0)))
        object.__setattr__(self, "l_target_median_edge_length_m_max", float(max(float(self.l_target_median_edge_length_m_min), float(self.l_target_median_edge_length_m_max))))
        object.__setattr__(self, "l_target_p90_edge_length_m_min", float(max(float(self.l_target_p90_edge_length_m_min), 1.0)))
        object.__setattr__(self, "l_target_p90_edge_length_m_max", float(max(float(self.l_target_p90_edge_length_m_min), float(self.l_target_p90_edge_length_m_max))))
        object.__setattr__(self, "l_target_leaf_fraction_max", float(np.clip(float(self.l_target_leaf_fraction_max), 0.0, 1.0)))
        object.__setattr__(self, "l_target_deg3_fraction_min", float(np.clip(float(self.l_target_deg3_fraction_min), 0.0, 1.0)))
        object.__setattr__(self, "l_target_deg3_fraction_max", float(np.clip(float(max(self.l_target_deg3_fraction_min, self.l_target_deg3_fraction_max)), 0.0, 1.0)))
        object.__setattr__(self, "l_target_deg4_fraction_min", float(np.clip(float(self.l_target_deg4_fraction_min), 0.0, 1.0)))
        object.__setattr__(self, "l_target_deg4_fraction_max", float(np.clip(float(max(self.l_target_deg4_fraction_min, self.l_target_deg4_fraction_max)), 0.0, 1.0)))
        object.__setattr__(self, "l_target_deg_gt4_fraction_max", float(np.clip(float(self.l_target_deg_gt4_fraction_max), 0.0, 1.0)))
        object.__setattr__(self, "l_target_arterial_length_share_min", float(np.clip(float(self.l_target_arterial_length_share_min), 0.0, 1.0)))
        object.__setattr__(self, "l_target_arterial_length_share_max", float(np.clip(float(max(self.l_target_arterial_length_share_min, self.l_target_arterial_length_share_max)), 0.0, 1.0)))
        object.__setattr__(self, "l_target_collector_length_share_min", float(np.clip(float(self.l_target_collector_length_share_min), 0.0, 1.0)))
        object.__setattr__(self, "l_target_collector_length_share_max", float(np.clip(float(max(self.l_target_collector_length_share_min, self.l_target_collector_length_share_max)), 0.0, 1.0)))
        object.__setattr__(self, "l_target_local_length_share_min", float(np.clip(float(self.l_target_local_length_share_min), 0.0, 1.0)))
        object.__setattr__(self, "l_target_local_length_share_max", float(np.clip(float(max(self.l_target_local_length_share_min, self.l_target_local_length_share_max)), 0.0, 1.0)))
        object.__setattr__(self, "l_target_max_crossing_fraction", float(np.clip(float(self.l_target_max_crossing_fraction), 0.0, 1.0)))
        object.__setattr__(self, "l_target_main_orientation_modes", int(max(1, int(self.l_target_main_orientation_modes))))
        object.__setattr__(self, "l_target_off_axis_edge_fraction_min", float(np.clip(float(self.l_target_off_axis_edge_fraction_min), 0.0, 1.0)))
        object.__setattr__(self, "l_target_off_axis_edge_fraction_max", float(np.clip(float(max(self.l_target_off_axis_edge_fraction_min, self.l_target_off_axis_edge_fraction_max)), 0.0, 1.0)))
        object.__setattr__(self, "l_target_builtup_area_fraction_min", float(np.clip(float(self.l_target_builtup_area_fraction_min), 0.0, 1.0)))
        object.__setattr__(self, "l_target_builtup_area_fraction_max", float(np.clip(float(max(self.l_target_builtup_area_fraction_min, self.l_target_builtup_area_fraction_max)), 0.0, 1.0)))
        object.__setattr__(self, "l_target_barrier_area_fraction_min", float(np.clip(float(self.l_target_barrier_area_fraction_min), 0.0, 1.0)))
        object.__setattr__(self, "l_target_barrier_area_fraction_max", float(np.clip(float(max(self.l_target_barrier_area_fraction_min, self.l_target_barrier_area_fraction_max)), 0.0, 1.0)))
        mode = str(self.road_awareness_mode).strip().lower()
        if mode not in {"perfect", "shared"}:
            mode = "perfect"
        object.__setattr__(self, "road_awareness_mode", mode)
        object.__setattr__(
            self,
            "road_shared_awareness_enabled",
            bool(self.road_shared_awareness_enabled),
        )
        object.__setattr__(
            self,
            "road_shared_replan_on_update",
            bool(self.road_shared_replan_on_update),
        )
        object.__setattr__(
            self,
            "road_uav_scout_enabled",
            bool(self.road_uav_scout_enabled),
        )
        object.__setattr__(
            self,
            "road_truck_scout_enabled",
            bool(self.road_truck_scout_enabled),
        )
        object.__setattr__(
            self,
            "road_uav_scout_radius_m",
            float(max(float(self.road_uav_scout_radius_m), 0.0)),
        )
        object.__setattr__(
            self,
            "road_truck_scout_radius_m",
            float(max(float(self.road_truck_scout_radius_m), 0.0)),
        )
        object.__setattr__(
            self,
            "risk_gat_beta",
            float(max(float(self.risk_gat_beta), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_event_replan_budget_per_window",
            int(max(0, int(self.hrl_event_replan_budget_per_window))),
        )
        object.__setattr__(
            self,
            "hrl_event_first_refresh_enabled",
            bool(self.hrl_event_first_refresh_enabled),
        )
        object.__setattr__(
            self,
            "hrl_max_no_refresh_steps",
            int(max(1, int(self.hrl_max_no_refresh_steps))),
        )
        object.__setattr__(
            self,
            "hrl_event_admission_gate_enabled",
            bool(self.hrl_event_admission_gate_enabled),
        )
        object.__setattr__(
            self,
            "hrl_event_admission_lifeline_threshold_ratio",
            float(np.clip(float(self.hrl_event_admission_lifeline_threshold_ratio), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "hrl_noop_event_cooldown_enabled",
            bool(self.hrl_noop_event_cooldown_enabled),
        )
        object.__setattr__(
            self,
            "hrl_noop_event_cooldown_steps",
            int(max(1, int(self.hrl_noop_event_cooldown_steps))),
        )
        object.__setattr__(
            self,
            "hrl_uav_truck_anchor_aba_block_steps",
            int(max(0, int(self.hrl_uav_truck_anchor_aba_block_steps))),
        )
        object.__setattr__(self, "hrl_normal_stall_hard_refresh_enabled", bool(self.hrl_normal_stall_hard_refresh_enabled))
        object.__setattr__(self, "hrl_normal_stall_min_persist_steps", int(max(1, int(self.hrl_normal_stall_min_persist_steps))))
        object.__setattr__(self, "hrl_normal_stall_progress_epsilon_m", float(max(float(self.hrl_normal_stall_progress_epsilon_m), 0.0)))
        object.__setattr__(self, "hrl_normal_stall_cooldown_steps", int(max(0, int(self.hrl_normal_stall_cooldown_steps))))
        object.__setattr__(self, "hrl_normal_stall_local_only", bool(self.hrl_normal_stall_local_only))
        object.__setattr__(self, "hrl_soft_invalid_hard_refresh_enabled", bool(self.hrl_soft_invalid_hard_refresh_enabled))
        object.__setattr__(self, "hrl_soft_invalid_retry_cooldown_steps", int(max(0, int(self.hrl_soft_invalid_retry_cooldown_steps))))
        object.__setattr__(self, "hrl_soft_invalid_escalate_after_count", int(max(1, int(self.hrl_soft_invalid_escalate_after_count))))
        object.__setattr__(self, "hrl_truck_dead_end_local_first", bool(self.hrl_truck_dead_end_local_first))
        object.__setattr__(self, "hrl_truck_dead_end_persist_steps", int(max(1, int(self.hrl_truck_dead_end_persist_steps))))
        object.__setattr__(self, "hrl_truck_dead_end_cooldown_steps", int(max(0, int(self.hrl_truck_dead_end_cooldown_steps))))
        object.__setattr__(self, "hrl_truck_dead_end_global_refresh_enabled", bool(self.hrl_truck_dead_end_global_refresh_enabled))
        object.__setattr__(self, "hrl_path_blocked_impact_gate_enabled", bool(self.hrl_path_blocked_impact_gate_enabled))
        object.__setattr__(self, "hrl_path_blocked_local_repair_first", bool(self.hrl_path_blocked_local_repair_first))
        object.__setattr__(self, "hrl_path_blocked_global_refresh_enabled", bool(self.hrl_path_blocked_global_refresh_enabled))
        object.__setattr__(
            self,
            "hrl_truck_emergency_min_pending_normal_to_block",
            int(max(0, int(self.hrl_truck_emergency_min_pending_normal_to_block))),
        )
        object.__setattr__(
            self,
            "hrl_truck_emergency_relief_uav_cover_threshold",
            float(np.clip(float(self.hrl_truck_emergency_relief_uav_cover_threshold), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "hrl_truck_emergency_force_relief_urgency_threshold",
            float(np.clip(float(self.hrl_truck_emergency_force_relief_urgency_threshold), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "hrl_truck_emergency_force_relief_uav_cover_threshold",
            float(np.clip(float(self.hrl_truck_emergency_force_relief_uav_cover_threshold), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "hrl_uav_island_delivery_bonus",
            float(max(float(self.hrl_uav_island_delivery_bonus), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_truck_directional_split_enabled",
            bool(self.hrl_truck_directional_split_enabled),
        )
        object.__setattr__(
            self,
            "hrl_truck_directional_split_steps",
            int(max(0, int(self.hrl_truck_directional_split_steps))),
        )
        object.__setattr__(
            self,
            "hrl_truck_directional_split_bonus",
            float(max(float(self.hrl_truck_directional_split_bonus), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_initial_directional_cover_enabled",
            bool(self.hrl_initial_directional_cover_enabled),
        )
        object.__setattr__(
            self,
            "hrl_initial_directional_window_steps",
            int(max(0, int(self.hrl_initial_directional_window_steps))),
        )
        object.__setattr__(
            self,
            "hrl_initial_directional_sector_count",
            int(max(2, int(self.hrl_initial_directional_sector_count))),
        )
        object.__setattr__(
            self,
            "hrl_initial_directional_normal_weight",
            float(max(float(self.hrl_initial_directional_normal_weight), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_initial_directional_emergency_weight",
            float(max(float(self.hrl_initial_directional_emergency_weight), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_initial_directional_urgency_weight",
            float(max(float(self.hrl_initial_directional_urgency_weight), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_initial_directional_duplicate_penalty",
            float(max(float(self.hrl_initial_directional_duplicate_penalty), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_initial_directional_task_bonus",
            float(max(float(self.hrl_initial_directional_task_bonus), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_initial_directional_task_mismatch_penalty",
            float(max(float(self.hrl_initial_directional_task_mismatch_penalty), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_initial_directional_uav_truck_bonus",
            float(max(float(self.hrl_initial_directional_uav_truck_bonus), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_initial_directional_uav_truck_mismatch_penalty",
            float(max(float(self.hrl_initial_directional_uav_truck_mismatch_penalty), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_initial_directional_uav_task_bonus",
            float(max(float(self.hrl_initial_directional_uav_task_bonus), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_initial_directional_uav_per_truck_cap",
            int(max(0, int(self.hrl_initial_directional_uav_per_truck_cap))),
        )
        object.__setattr__(
            self,
            "hrl_initial_route_dispatch_enabled",
            bool(self.hrl_initial_route_dispatch_enabled),
        )
        object.__setattr__(
            self,
            "hrl_initial_route_docked_assignment_enabled",
            bool(self.hrl_initial_route_docked_assignment_enabled),
        )
        object.__setattr__(
            self,
            "hrl_uav_task_locality_weight",
            float(max(float(self.hrl_uav_task_locality_weight), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_uav_near_depot_direct_dispatch_radius_m",
            float(max(float(self.hrl_uav_near_depot_direct_dispatch_radius_m), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_uav_near_depot_direct_dispatch_bonus",
            float(max(float(self.hrl_uav_near_depot_direct_dispatch_bonus), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_uav_near_depot_direct_dispatch_steps",
            int(max(0, int(self.hrl_uav_near_depot_direct_dispatch_steps))),
        )
        object.__setattr__(
            self,
            "hrl_uav_idle_truck_staging_enabled",
            bool(self.hrl_uav_idle_truck_staging_enabled),
        )
        object.__setattr__(
            self,
            "hrl_uav_idle_truck_staging_min_score",
            float(max(float(self.hrl_uav_idle_truck_staging_min_score), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_uav_idle_staging_prefer_initial_plan",
            bool(self.hrl_uav_idle_staging_prefer_initial_plan),
        )
        object.__setattr__(
            self,
            "hrl_uav_idle_staging_respect_truck_cap",
            bool(self.hrl_uav_idle_staging_respect_truck_cap),
        )
        object.__setattr__(
            self,
            "hrl_uav_truck_emergency_pull_weight",
            float(max(float(self.hrl_uav_truck_emergency_pull_weight), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_uav_truck_follower_balance_weight",
            float(max(float(self.hrl_uav_truck_follower_balance_weight), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_truck_task_lookahead_steps",
            int(max(1, int(self.hrl_truck_task_lookahead_steps))),
        )
        object.__setattr__(
            self,
            "hrl_truck_task_lookahead_max_steps",
            int(max(int(self.hrl_truck_task_lookahead_steps), int(self.hrl_truck_task_lookahead_max_steps))),
        )
        object.__setattr__(
            self,
            "hrl_truck_task_lookahead_weight",
            float(max(float(self.hrl_truck_task_lookahead_weight), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_uav_truck_lookahead_weight",
            float(max(float(self.hrl_uav_truck_lookahead_weight), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_uav_docked_prelaunch_assign_min_score",
            float(max(float(self.hrl_uav_docked_prelaunch_assign_min_score), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_uav_task_transfer_enabled",
            bool(self.hrl_uav_task_transfer_enabled),
        )
        object.__setattr__(
            self,
            "hrl_uav_task_transfer_score_gain_min",
            float(max(float(self.hrl_uav_task_transfer_score_gain_min), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_uav_task_transfer_progress_gain_min",
            float(max(float(self.hrl_uav_task_transfer_progress_gain_min), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_uav_task_transfer_commit_steps",
            int(max(1, int(self.hrl_uav_task_transfer_commit_steps))),
        )
        object.__setattr__(
            self,
            "hrl_uav_task_transfer_hint_hold_steps",
            int(max(1, int(self.hrl_uav_task_transfer_hint_hold_steps))),
        )
        object.__setattr__(
            self,
            "hrl_uav_task_transfer_max_target_dist_m",
            float(max(float(self.hrl_uav_task_transfer_max_target_dist_m), 0.0)),
        )
        object.__setattr__(
            self,
            "uav_transfer_min_battery_fraction",
            float(np.clip(float(self.uav_transfer_min_battery_fraction), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "uav_transfer_reserve_fraction",
            float(np.clip(float(self.uav_transfer_reserve_fraction), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "hrl_uav_task_reservation_enabled",
            bool(self.hrl_uav_task_reservation_enabled),
        )
        object.__setattr__(
            self,
            "hrl_supported_sortie_joint_enabled",
            bool(self.hrl_supported_sortie_joint_enabled),
        )
        object.__setattr__(
            self,
            "hrl_dynamic_task_pressure_enabled",
            bool(self.hrl_dynamic_task_pressure_enabled),
        )
        object.__setattr__(
            self,
            "hrl_support_conversion_gate_enabled",
            bool(self.hrl_support_conversion_gate_enabled),
        )
        object.__setattr__(
            self,
            "hrl_support_conversion_target_ratio",
            float(np.clip(float(self.hrl_support_conversion_target_ratio), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "hrl_support_conversion_min_support_count",
            int(max(0, int(self.hrl_support_conversion_min_support_count))),
        )
        object.__setattr__(
            self,
            "hrl_support_conversion_penalty_strength",
            float(max(float(self.hrl_support_conversion_penalty_strength), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_truck_normal_commit_guard2_enabled",
            bool(self.hrl_truck_normal_commit_guard2_enabled),
        )
        object.__setattr__(
            self,
            "hrl_truck_normal_commit_min_steps",
            int(max(0, int(self.hrl_truck_normal_commit_min_steps))),
        )
        object.__setattr__(
            self,
            "hrl_truck_normal_commit_pressure_threshold",
            float(np.clip(float(self.hrl_truck_normal_commit_pressure_threshold), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "hrl_truck_support_when_normal_reachable_scale",
            float(np.clip(float(self.hrl_truck_support_when_normal_reachable_scale), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "hrl_truck_support_when_no_normal_bonus",
            float(max(float(self.hrl_truck_support_when_no_normal_bonus), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_truck_emergency_support_when_no_normal_enabled",
            bool(self.hrl_truck_emergency_support_when_no_normal_enabled),
        )
        object.__setattr__(
            self,
            "hrl_truck_hard_normal_first_enabled",
            bool(self.hrl_truck_hard_normal_first_enabled),
        )
        object.__setattr__(
            self,
            "hrl_separate_agent_objectives_enabled",
            bool(self.hrl_separate_agent_objectives_enabled),
        )
        object.__setattr__(
            self,
            "hrl_timecritical_lifeline_warning_ratio",
            float(np.clip(float(self.hrl_timecritical_lifeline_warning_ratio), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "hrl_timecritical_lifeline_critical_ratio",
            float(np.clip(float(self.hrl_timecritical_lifeline_critical_ratio), 0.0, 1.0)),
        )
        object.__setattr__(self, "hrl_timecritical_force_entry_enabled", bool(self.hrl_timecritical_force_entry_enabled))
        object.__setattr__(self, "hrl_timecritical_force_entry_min_map_size_m", float(max(float(self.hrl_timecritical_force_entry_min_map_size_m), 0.0)))
        object.__setattr__(self, "hrl_timecritical_force_entry_min_gap_steps", int(max(0, int(self.hrl_timecritical_force_entry_min_gap_steps))))
        object.__setattr__(self, "hrl_timecritical_force_entry_max_lifeline_ratio", float(np.clip(float(self.hrl_timecritical_force_entry_max_lifeline_ratio), 0.0, 1.0)))
        object.__setattr__(self, "hrl_timecritical_force_entry_shortlist_extra", int(max(0, int(self.hrl_timecritical_force_entry_shortlist_extra))))
        object.__setattr__(self, "hrl_timecritical_force_entry_uav_bonus", float(max(float(self.hrl_timecritical_force_entry_uav_bonus), 0.0)))
        object.__setattr__(self, "hrl_timecritical_force_entry_truck_bonus", float(max(float(self.hrl_timecritical_force_entry_truck_bonus), 0.0)))
        object.__setattr__(self, "hrl_airborne_tc_completion_grace_enabled", bool(self.hrl_airborne_tc_completion_grace_enabled))
        object.__setattr__(self, "hrl_airborne_tc_completion_grace_radius_m", float(max(float(self.hrl_airborne_tc_completion_grace_radius_m), 0.0)))
        object.__setattr__(
            self,
            "hrl_airborne_tc_completion_grace_recovery_buffer_scale",
            float(np.clip(float(self.hrl_airborne_tc_completion_grace_recovery_buffer_scale), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "hrl_airborne_tc_completion_grace_min_battery",
            float(np.clip(float(self.hrl_airborne_tc_completion_grace_min_battery), 0.0, 1.0)),
        )
        object.__setattr__(self, "hrl_airborne_tc_completion_grace_min_lifeline_steps", int(max(0, int(self.hrl_airborne_tc_completion_grace_min_lifeline_steps))))
        object.__setattr__(self, "hrl_tc_global_assignment_adaptive_escape_enabled", bool(self.hrl_tc_global_assignment_adaptive_escape_enabled))
        object.__setattr__(self, "hrl_tc_global_assignment_escape_min_map_size_m", float(max(float(self.hrl_tc_global_assignment_escape_min_map_size_m), 0.0)))
        object.__setattr__(self, "hrl_tc_global_assignment_escape_low_cover_threshold", float(np.clip(float(self.hrl_tc_global_assignment_escape_low_cover_threshold), 0.0, 1.0)))
        object.__setattr__(self, "hrl_tc_global_assignment_escape_max_lifeline_ratio", float(np.clip(float(self.hrl_tc_global_assignment_escape_max_lifeline_ratio), 0.0, 1.0)))
        object.__setattr__(self, "hrl_timecritical_far_exposure_enabled", bool(self.hrl_timecritical_far_exposure_enabled))
        object.__setattr__(self, "hrl_timecritical_far_exposure_min_map_size_m", float(max(float(self.hrl_timecritical_far_exposure_min_map_size_m), 0.0)))
        object.__setattr__(self, "hrl_timecritical_far_exposure_extra", int(max(0, int(self.hrl_timecritical_far_exposure_extra))))
        object.__setattr__(self, "hrl_timecritical_far_exposure_max_lifeline_ratio", float(np.clip(float(self.hrl_timecritical_far_exposure_max_lifeline_ratio), 0.0, 1.0)))
        object.__setattr__(self, "hrl_timecritical_far_exposure_min_gap_steps", int(max(0, int(self.hrl_timecritical_far_exposure_min_gap_steps))))
        object.__setattr__(self, "hrl_timecritical_far_exposure_low_cover_threshold", float(np.clip(float(self.hrl_timecritical_far_exposure_low_cover_threshold), 0.0, 1.0)))
        object.__setattr__(self, "hrl_timecritical_far_exposure_urgent_bypass_threshold", float(np.clip(float(self.hrl_timecritical_far_exposure_urgent_bypass_threshold), 0.0, 1.0)))
        if float(self.hrl_timecritical_lifeline_critical_ratio) > float(self.hrl_timecritical_lifeline_warning_ratio):
            object.__setattr__(
                self,
                "hrl_timecritical_lifeline_critical_ratio",
                float(self.hrl_timecritical_lifeline_warning_ratio),
            )
        object.__setattr__(
            self,
            "hrl_uav_timecritical_urgency_weight",
            float(max(float(self.hrl_uav_timecritical_urgency_weight), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_uav_timecritical_lifeline_weight",
            float(max(float(self.hrl_uav_timecritical_lifeline_weight), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_uav_timecritical_critical_bonus",
            float(max(float(self.hrl_uav_timecritical_critical_bonus), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_truck_timecritical_penalty_scale_warning",
            float(np.clip(float(self.hrl_truck_timecritical_penalty_scale_warning), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "hrl_truck_timecritical_penalty_scale_critical",
            float(np.clip(float(self.hrl_truck_timecritical_penalty_scale_critical), 0.0, 1.0)),
        )
        if float(self.hrl_truck_timecritical_penalty_scale_critical) > float(self.hrl_truck_timecritical_penalty_scale_warning):
            object.__setattr__(
                self,
                "hrl_truck_timecritical_penalty_scale_critical",
                float(self.hrl_truck_timecritical_penalty_scale_warning),
            )
        object.__setattr__(
            self,
            "hrl_truck_timecritical_support_amp_warning",
            float(max(float(self.hrl_truck_timecritical_support_amp_warning), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_truck_timecritical_support_amp_critical",
            float(max(float(self.hrl_truck_timecritical_support_amp_critical), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_truck_timecritical_recovery_amp_warning",
            float(max(float(self.hrl_truck_timecritical_recovery_amp_warning), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_truck_timecritical_recovery_amp_critical",
            float(max(float(self.hrl_truck_timecritical_recovery_amp_critical), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_truck_timecritical_supported_sortie_amp_warning",
            float(max(float(self.hrl_truck_timecritical_supported_sortie_amp_warning), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_truck_timecritical_supported_sortie_amp_critical",
            float(max(float(self.hrl_truck_timecritical_supported_sortie_amp_critical), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_support_bind_bonus_critical",
            float(max(float(self.hrl_support_bind_bonus_critical), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_support_bind_bonus_warning",
            float(max(float(self.hrl_support_bind_bonus_warning), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_support_bind_bonus_bulk",
            float(max(float(self.hrl_support_bind_bonus_bulk), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_support_bind_horizon_steps",
            int(max(0, int(self.hrl_support_bind_horizon_steps))),
        )
        object.__setattr__(
            self,
            "hrl_support_bind_horizon_steps_large_map",
            int(max(0, int(self.hrl_support_bind_horizon_steps_large_map))),
        )
        object.__setattr__(
            self,
            "hrl_support_requires_timecritical_binding",
            bool(self.hrl_support_requires_timecritical_binding),
        )
        object.__setattr__(
            self,
            "hrl_support_gain_multi_task_k",
            int(max(1, int(self.hrl_support_gain_multi_task_k))),
        )
        object.__setattr__(
            self,
            "hrl_uav_cover_eval_max_distance_m",
            float(max(float(self.hrl_uav_cover_eval_max_distance_m), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_uav_docked_require_launch_gate_strict",
            bool(self.hrl_uav_docked_require_launch_gate_strict),
        )
        object.__setattr__(
            self,
            "hrl_docked_uav_soft_invalid_hold_enabled",
            bool(self.hrl_docked_uav_soft_invalid_hold_enabled),
        )
        object.__setattr__(
            self,
            "hrl_support_proxy_require_warning_bind_when_normal_reachable",
            bool(self.hrl_support_proxy_require_warning_bind_when_normal_reachable),
        )
        object.__setattr__(
            self,
            "hrl_support_proxy_warning_gate_min_map_size_m",
            float(max(float(self.hrl_support_proxy_warning_gate_min_map_size_m), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_support_fallback_allow_bulk_binding",
            bool(self.hrl_support_fallback_allow_bulk_binding),
        )
        object.__setattr__(
            self,
            "hrl_support_candidate_max_distance_m",
            float(max(float(self.hrl_support_candidate_max_distance_m), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_support_bind_enforce_min_map_size_m",
            float(max(float(self.hrl_support_bind_enforce_min_map_size_m), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_support_require_actionable_gain",
            bool(self.hrl_support_require_actionable_gain),
        )
        object.__setattr__(
            self,
            "hrl_support_actionable_min_gain_score",
            float(np.clip(float(self.hrl_support_actionable_min_gain_score), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "hrl_support_actionable_min_new_serviceable",
            float(max(float(self.hrl_support_actionable_min_new_serviceable), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_support_actionable_post_distance_m",
            float(max(float(self.hrl_support_actionable_post_distance_m), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_support_bound_dispatch_enabled",
            bool(self.hrl_support_bound_dispatch_enabled),
        )
        object.__setattr__(
            self,
            "hrl_support_no_gain_backoff_enabled",
            bool(self.hrl_support_no_gain_backoff_enabled),
        )
        object.__setattr__(
            self,
            "hrl_support_no_gain_streak_threshold",
            int(max(1, int(self.hrl_support_no_gain_streak_threshold))),
        )
        object.__setattr__(
            self,
            "hrl_support_no_gain_cooldown_steps",
            int(max(0, int(self.hrl_support_no_gain_cooldown_steps))),
        )
        object.__setattr__(
            self,
            "hrl_support_max_trucks_when_normal_pending",
            int(max(0, int(self.hrl_support_max_trucks_when_normal_pending))),
        )
        object.__setattr__(
            self,
            "hrl_support_budget_require_warning_when_normal",
            bool(self.hrl_support_budget_require_warning_when_normal),
        )
        object.__setattr__(self, "hrl_support_chain_critical_escape_enabled", bool(self.hrl_support_chain_critical_escape_enabled))
        object.__setattr__(self, "hrl_support_chain_critical_escape_max_lifeline_ratio", float(np.clip(float(self.hrl_support_chain_critical_escape_max_lifeline_ratio), 0.0, 1.0)))
        object.__setattr__(self, "hrl_support_chain_critical_escape_low_cover_threshold", float(np.clip(float(self.hrl_support_chain_critical_escape_low_cover_threshold), 0.0, 1.0)))
        object.__setattr__(self, "hrl_support_chain_critical_escape_min_gain", float(max(float(self.hrl_support_chain_critical_escape_min_gain), 0.0)))
        object.__setattr__(
            self,
            "hrl_support_relay_reserve_enabled",
            bool(self.hrl_support_relay_reserve_enabled),
        )
        object.__setattr__(
            self,
            "hrl_support_relay_min_critical_timecritical",
            int(max(0, int(self.hrl_support_relay_min_critical_timecritical))),
        )
        object.__setattr__(
            self,
            "hrl_support_relay_cover_threshold",
            float(np.clip(float(self.hrl_support_relay_cover_threshold), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "hrl_support_critical_diversion_enabled",
            bool(self.hrl_support_critical_diversion_enabled),
        )
        object.__setattr__(
            self,
            "hrl_support_critical_diversion_max_trucks",
            int(max(0, int(self.hrl_support_critical_diversion_max_trucks))),
        )
        object.__setattr__(
            self,
            "hrl_support_critical_diversion_max_map_size_m",
            float(max(float(self.hrl_support_critical_diversion_max_map_size_m), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_support_critical_diversion_cover_threshold",
            float(np.clip(float(self.hrl_support_critical_diversion_cover_threshold), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "hrl_truck_no_normal_support_min_gain",
            float(np.clip(float(self.hrl_truck_no_normal_support_min_gain), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "hrl_truck_no_normal_support_urgency_floor",
            float(np.clip(float(self.hrl_truck_no_normal_support_urgency_floor), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "hrl_support_soft_clamp_enabled",
            bool(self.hrl_support_soft_clamp_enabled),
        )
        object.__setattr__(
            self,
            "hrl_support_soft_clamp_long_distance_m",
            float(max(float(self.hrl_support_soft_clamp_long_distance_m), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_support_soft_clamp_min_gain",
            float(np.clip(float(self.hrl_support_soft_clamp_min_gain), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "hrl_support_soft_clamp_bindable_min_new_serviceable",
            int(max(0, int(self.hrl_support_soft_clamp_bindable_min_new_serviceable))),
        )
        object.__setattr__(
            self,
            "hrl_support_soft_clamp_require_direct_delivery_candidates",
            bool(self.hrl_support_soft_clamp_require_direct_delivery_candidates),
        )
        object.__setattr__(
            self,
            "hrl_support_escape_hatch_enabled",
            bool(self.hrl_support_escape_hatch_enabled),
        )
        object.__setattr__(
            self,
            "hrl_support_escape_hatch_min_pending_emergency",
            int(max(0, int(self.hrl_support_escape_hatch_min_pending_emergency))),
        )
        object.__setattr__(
            self,
            "hrl_support_escape_hatch_min_gain",
            float(np.clip(float(self.hrl_support_escape_hatch_min_gain), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "hrl_support_escape_hatch_min_urgency",
            float(np.clip(float(self.hrl_support_escape_hatch_min_urgency), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "hrl_support_escape_hatch_allow_low_cover_timecritical",
            bool(self.hrl_support_escape_hatch_allow_low_cover_timecritical),
        )
        object.__setattr__(
            self,
            "hrl_support_escape_hatch_low_cover_threshold",
            float(np.clip(float(self.hrl_support_escape_hatch_low_cover_threshold), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "hrl_truck_idle_hard_refresh_enabled",
            bool(self.hrl_truck_idle_hard_refresh_enabled),
        )
        object.__setattr__(
            self,
            "hrl_map_update_allow_ranking_changed_impact",
            bool(self.hrl_map_update_allow_ranking_changed_impact),
        )
        object.__setattr__(
            self,
            "hrl_event_bonus_conditional_enabled",
            bool(self.hrl_event_bonus_conditional_enabled),
        )
        object.__setattr__(
            self,
            "hrl_event_bonus_base_gain",
            float(np.clip(float(self.hrl_event_bonus_base_gain), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "hrl_event_bonus_hard_gain",
            float(max(float(self.hrl_event_bonus_hard_gain), float(self.hrl_event_bonus_base_gain))),
        )
        object.__setattr__(self, "execution_commitment_enabled", bool(self.execution_commitment_enabled))
        object.__setattr__(
            self,
            "execution_commitment_override_lifeline_margin",
            float(max(float(self.execution_commitment_override_lifeline_margin), 0.0)),
        )
        object.__setattr__(self, "execution_no_switch_if_commit_active", bool(self.execution_no_switch_if_commit_active))
        object.__setattr__(
            self,
            "execution_no_switch_if_current_goal_launchable",
            bool(self.execution_no_switch_if_current_goal_launchable),
        )
        object.__setattr__(self, "goal_switch_penalty_enabled", bool(self.goal_switch_penalty_enabled))
        object.__setattr__(self, "goal_switch_hard_cap_enabled", bool(self.goal_switch_hard_cap_enabled))
        object.__setattr__(self, "goal_switch_score_gain_min", float(max(float(self.goal_switch_score_gain_min), 0.0)))
        object.__setattr__(self, "goal_switch_eta_gain_min", float(max(float(self.goal_switch_eta_gain_min), 0.0)))
        object.__setattr__(self, "timecritical_global_assignment_enabled", bool(self.timecritical_global_assignment_enabled))
        object.__setattr__(self, "cluster_primary_task_enabled", bool(self.cluster_primary_task_enabled))
        object.__setattr__(self, "task_reservation_enabled", bool(self.task_reservation_enabled))
        object.__setattr__(self, "recent_release_cooldown_enabled", bool(self.recent_release_cooldown_enabled))
        object.__setattr__(self, "unreachable_bulk_watchlist_enabled", bool(self.unreachable_bulk_watchlist_enabled))
        object.__setattr__(self, "support_force_dispatch_enabled", bool(self.support_force_dispatch_enabled))
        object.__setattr__(self, "support_force_commit_steps", int(max(0, int(self.support_force_commit_steps))))
        object.__setattr__(self, "support_force_uav_preempt_enabled", bool(self.support_force_uav_preempt_enabled))
        object.__setattr__(self, "truck_force_nonnull_goal_enabled", bool(self.truck_force_nonnull_goal_enabled))
        object.__setattr__(self, "truck_loop_break_enabled", bool(self.truck_loop_break_enabled))
        object.__setattr__(self, "truck_loop_break_window_steps", int(max(0, int(self.truck_loop_break_window_steps))))
        # Legacy aliases mirror generic fields; RC-specific forcing stays disabled by default.
        object.__setattr__(self, "rc_unreachable_bulk_watchlist_enabled", bool(self.unreachable_bulk_watchlist_enabled))
        object.__setattr__(self, "rc_support_force_dispatch_enabled", bool(self.support_force_dispatch_enabled))
        object.__setattr__(self, "rc_support_force_commit_steps", int(max(0, int(self.support_force_commit_steps))))
        object.__setattr__(self, "rc_support_force_uav_preempt_enabled", bool(self.support_force_uav_preempt_enabled))
        object.__setattr__(self, "rc_truck_force_nonnull_goal_enabled", bool(self.truck_force_nonnull_goal_enabled))
        object.__setattr__(self, "rc_truck_loop_break_enabled", bool(self.truck_loop_break_enabled))
        object.__setattr__(self, "rc_truck_loop_break_window_steps", int(max(0, int(self.truck_loop_break_window_steps))))
        object.__setattr__(
            self,
            "rc_strong_planner_mode_enabled",
            bool(self.rc_strong_planner_mode_enabled),
        )
        object.__setattr__(
            self,
            "rc_unreachable_bulk_watchlist_enabled",
            bool(self.unreachable_bulk_watchlist_enabled),
        )
        object.__setattr__(
            self,
            "rc_support_force_dispatch_enabled",
            bool(self.support_force_dispatch_enabled),
        )
        object.__setattr__(
            self,
            "rc_support_force_commit_steps",
            int(max(0, int(self.support_force_commit_steps))),
        )
        object.__setattr__(
            self,
            "rc_support_force_uav_preempt_enabled",
            bool(self.support_force_uav_preempt_enabled),
        )
        object.__setattr__(
            self,
            "rc_truck_force_nonnull_goal_enabled",
            bool(self.truck_force_nonnull_goal_enabled),
        )
        object.__setattr__(
            self,
            "rc_truck_loop_break_enabled",
            bool(self.truck_loop_break_enabled),
        )
        object.__setattr__(
            self,
            "rc_truck_loop_break_window_steps",
            int(max(0, int(self.truck_loop_break_window_steps))),
        )
        object.__setattr__(
            self,
            "hrl_conditional_h2_refresh_enabled",
            bool(self.hrl_conditional_h2_refresh_enabled),
        )
        object.__setattr__(
            self,
            "hrl_conditional_h2_new_info_threshold",
            int(max(0, int(self.hrl_conditional_h2_new_info_threshold))),
        )
        object.__setattr__(
            self,
            "hrl_conditional_h2_delivery_stall_steps",
            int(max(0, int(self.hrl_conditional_h2_delivery_stall_steps))),
        )
        object.__setattr__(
            self,
            "hrl_conditional_h2_support_quality_max",
            float(np.clip(float(self.hrl_conditional_h2_support_quality_max), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "hrl_conditional_h2_support_distance_threshold_m",
            float(max(float(self.hrl_conditional_h2_support_distance_threshold_m), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_conditional_h2_island_serviceability_low",
            float(np.clip(float(self.hrl_conditional_h2_island_serviceability_low), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "hrl_truck_normal_to_normal_switch_min_improve_ratio",
            float(np.clip(float(self.hrl_truck_normal_to_normal_switch_min_improve_ratio), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "hrl_truck_normal_to_normal_switch_min_score_gain",
            float(max(float(self.hrl_truck_normal_to_normal_switch_min_score_gain), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_truck_normal_aba_block_steps",
            int(max(0, int(self.hrl_truck_normal_aba_block_steps))),
        )
        object.__setattr__(
            self,
            "hrl_truck_routine_stuck_persist_steps",
            int(max(1, int(self.hrl_truck_routine_stuck_persist_steps))),
        )
        object.__setattr__(
            self,
            "hrl_truck_routine_progress_epsilon_m",
            float(max(float(self.hrl_truck_routine_progress_epsilon_m), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_truck_routine_escape_min_eta_gain_steps",
            int(max(0, int(self.hrl_truck_routine_escape_min_eta_gain_steps))),
        )
        object.__setattr__(
            self,
            "hrl_truck_routine_escape_min_score_gain",
            float(max(float(self.hrl_truck_routine_escape_min_score_gain), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_truck_emergency_cover_threshold_when_normal_reachable",
            float(np.clip(float(self.hrl_truck_emergency_cover_threshold_when_normal_reachable), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "hrl_truck_support_gain_min_when_normal_reachable",
            float(np.clip(float(self.hrl_truck_support_gain_min_when_normal_reachable), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "hrl_support_anchor_hold_steps",
            int(max(0, int(self.hrl_support_anchor_hold_steps))),
        )
        object.__setattr__(
            self,
            "hrl_support_anchor_max_drift_m",
            float(max(float(self.hrl_support_anchor_max_drift_m), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_relaxed_chain_commitment_steps",
            int(max(0, int(self.hrl_relaxed_chain_commitment_steps))),
        )
        object.__setattr__(
            self,
            "hrl_relaxed_chain_min_bound_tasks",
            int(max(0, int(self.hrl_relaxed_chain_min_bound_tasks))),
        )
        object.__setattr__(
            self,
            "hrl_serviceable_island_bonus",
            float(max(float(self.hrl_serviceable_island_bonus), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_serviceable_high_pressure_emergency_bonus",
            float(max(float(self.hrl_serviceable_high_pressure_emergency_bonus), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_uav_ride_stall_release_enabled",
            bool(self.hrl_uav_ride_stall_release_enabled),
        )
        object.__setattr__(
            self,
            "hrl_uav_ride_stall_trigger_steps",
            int(max(0, int(self.hrl_uav_ride_stall_trigger_steps))),
        )
        object.__setattr__(
            self,
            "hrl_uav_ride_stall_bonus",
            float(max(float(self.hrl_uav_ride_stall_bonus), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_uav_ride_stall_max_dist_m",
            float(max(float(self.hrl_uav_ride_stall_max_dist_m), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_uav_task_reservation_exec_enabled",
            bool(self.hrl_uav_task_reservation_exec_enabled),
        )
        object.__setattr__(
            self,
            "hrl_uav_task_reservation_stale_steps",
            int(max(1, int(self.hrl_uav_task_reservation_stale_steps))),
        )
        object.__setattr__(
            self,
            "hrl_task_exclusive_contract_enabled",
            bool(self.hrl_task_exclusive_contract_enabled),
        )
        object.__setattr__(
            self,
            "hrl_unreachable_normal_uav_takeover_enabled",
            bool(self.hrl_unreachable_normal_uav_takeover_enabled),
        )
        object.__setattr__(
            self,
            "uav_strict_sortie_contract_enabled",
            bool(self.uav_strict_sortie_contract_enabled),
        )
        object.__setattr__(
            self,
            "hrl_uav_assist_enabled",
            bool(self.hrl_uav_assist_enabled),
        )
        object.__setattr__(
            self,
            "hrl_uav_assist_max_extra_distance_m",
            float(max(float(self.hrl_uav_assist_max_extra_distance_m), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_uav_assist_max_extra_ratio",
            float(np.clip(float(self.hrl_uav_assist_max_extra_ratio), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "hrl_uav_assist_min_launch_distance_reduction_m",
            float(max(float(self.hrl_uav_assist_min_launch_distance_reduction_m), 0.0)),
        )
        object.__setattr__(self, "use_event_trigger", bool(self.use_event_trigger))
        object.__setattr__(self, "use_risk_term", bool(self.use_risk_term))
        object.__setattr__(self, "use_rth_repair", bool(self.use_rth_repair))
        object.__setattr__(
            self,
            "normal_task_demand_kg",
            float(max(float(self.normal_task_demand_kg), 1e-6)),
        )
        object.__setattr__(
            self,
            "emergency_task_demand_kg",
            float(max(float(self.emergency_task_demand_kg), 1e-6)),
        )
        object.__setattr__(
            self,
            "time_critical_package_kg",
            float(max(float(self.time_critical_package_kg), 1e-6)),
        )
        object.__setattr__(
            self,
            "truck_payload_capacity_kg",
            float(max(float(self.truck_payload_capacity_kg), 1e-6)),
        )
        object.__setattr__(
            self,
            "uav_payload_capacity_kg",
            float(max(float(self.uav_payload_capacity_kg), 1e-6)),
        )
        object.__setattr__(
            self,
            "uav_self_weight_kg",
            float(max(float(self.uav_self_weight_kg), 1e-6)),
        )
        object.__setattr__(
            self,
            "uav_battery_init",
            float(np.clip(float(self.uav_battery_init), 0.0, 1.0)),
        )
        if not bool(self.ignore_payload_constraints):
            # The kg fields are the paper-facing source of truth.  Legacy
            # abstract-unit fields remain synchronized for old components.
            object.__setattr__(self, "truck_payload_kg", float(self.truck_payload_capacity_kg))
            initial_truck_load_kg = float(
                max(float(self.truck_initial_bulk_inventory_kg), 0.0)
                + max(float(self.truck_initial_timecritical_inventory_kg), 0.0)
            )
            if initial_truck_load_kg > float(self.truck_payload_capacity_kg) + 1e-9:
                raise ValueError(
                    "initial truck inventory exceeds truck_payload_capacity_kg: "
                    f"{initial_truck_load_kg:.3f} > {float(self.truck_payload_capacity_kg):.3f}"
                )
            if float(self.hrl_route_plan_bulk_relay_payload_kg) > float(self.uav_payload_capacity_kg) + 1e-9:
                raise ValueError(
                    "bulk-relay UAV payload exceeds uav_payload_capacity_kg: "
                    f"{float(self.hrl_route_plan_bulk_relay_payload_kg):.3f} > "
                    f"{float(self.uav_payload_capacity_kg):.3f}"
                )
            if float(self.uav_payload_kg) > float(self.uav_payload_capacity_kg) + 1e-9:
                raise ValueError(
                    "initial UAV payload exceeds uav_payload_capacity_kg: "
                    f"{float(self.uav_payload_kg):.3f} > "
                    f"{float(self.uav_payload_capacity_kg):.3f}"
                )
        object.__setattr__(self, "max_time_critical_packages_per_uav_sortie", int(max(1, int(self.max_time_critical_packages_per_uav_sortie))))
        object.__setattr__(self, "max_standard_packages_per_truck", int(max(1, int(self.max_standard_packages_per_truck))))
        object.__setattr__(self, "num_routine_bulk_tasks", int(max(0, int(self.num_routine_bulk_tasks))))
        object.__setattr__(self, "num_time_critical_lightweight_tasks", int(max(0, int(self.num_time_critical_lightweight_tasks))))
        if int(self.num_routine_bulk_tasks) > 0:
            object.__setattr__(self, "num_normal_tasks", int(self.num_routine_bulk_tasks))
            object.__setattr__(self, "n_normal_tasks", int(self.num_routine_bulk_tasks))
        if int(self.num_time_critical_lightweight_tasks) > 0:
            object.__setattr__(self, "num_emergency_tasks", int(self.num_time_critical_lightweight_tasks))
            object.__setattr__(self, "n_emergency_tasks", int(self.num_time_critical_lightweight_tasks))
        object.__setattr__(self, "routine_bulk_demand_kg_min", float(max(float(self.routine_bulk_demand_kg_min), 1e-6)))
        object.__setattr__(
            self,
            "routine_bulk_demand_kg_max",
            float(max(float(self.routine_bulk_demand_kg_min), float(self.routine_bulk_demand_kg_max))),
        )
        object.__setattr__(
            self,
            "time_critical_lightweight_demand_kg_min",
            float(max(float(self.time_critical_lightweight_demand_kg_min), 1e-6)),
        )
        object.__setattr__(
            self,
            "time_critical_lightweight_demand_kg_max",
            float(
                max(
                    float(self.time_critical_lightweight_demand_kg_min),
                    float(self.time_critical_lightweight_demand_kg_max),
                )
            ),
        )
        object.__setattr__(self, "routine_bulk_urgency_min", float(np.clip(float(self.routine_bulk_urgency_min), 0.0, 1.0)))
        object.__setattr__(
            self,
            "routine_bulk_urgency_max",
            float(np.clip(float(max(self.routine_bulk_urgency_min, self.routine_bulk_urgency_max)), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "time_critical_lightweight_urgency_min",
            float(np.clip(float(self.time_critical_lightweight_urgency_min), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "time_critical_lightweight_urgency_max",
            float(
                np.clip(
                    float(
                        max(
                            self.time_critical_lightweight_urgency_min,
                            self.time_critical_lightweight_urgency_max,
                        )
                    ),
                    0.0,
                    1.0,
                )
            ),
        )
        object.__setattr__(self, "task_lifeline_enabled", bool(self.task_lifeline_enabled))
        object.__setattr__(self, "task_lifeline_init_default", float(max(float(self.task_lifeline_init_default), 1e-6)))
        object.__setattr__(
            self,
            "routine_bulk_lifeline_decay_base",
            float(max(float(self.routine_bulk_lifeline_decay_base), 0.0)),
        )
        object.__setattr__(
            self,
            "time_critical_lightweight_lifeline_decay_base",
            float(max(float(self.time_critical_lightweight_lifeline_decay_base), 0.0)),
        )
        object.__setattr__(
            self,
            "task_lifeline_hazard_weight",
            float(max(float(self.task_lifeline_hazard_weight), 0.0)),
        )
        object.__setattr__(
            self,
            "routine_bulk_partial_fulfillment_enabled",
            bool(self.routine_bulk_partial_fulfillment_enabled),
        )
        object.__setattr__(
            self,
            "routine_bulk_partial_chunk_kg",
            float(max(float(self.routine_bulk_partial_chunk_kg), 1e-6)),
        )
        object.__setattr__(
            self,
            "truck_initial_bulk_inventory_kg",
            float(max(float(self.truck_initial_bulk_inventory_kg), 0.0)),
        )
        object.__setattr__(
            self,
            "truck_initial_timecritical_inventory_kg",
            float(max(float(self.truck_initial_timecritical_inventory_kg), 0.0)),
        )
        object.__setattr__(
            self,
            "bulk_supply_unit_kg",
            float(max(float(self.bulk_supply_unit_kg), 1e-6)),
        )
        object.__setattr__(
            self,
            "timecritical_supply_unit_kg",
            float(max(float(self.timecritical_supply_unit_kg), 1e-6)),
        )
        object.__setattr__(
            self,
            "truck_initial_normal_supply_units",
            int(max(0, int(self.truck_initial_normal_supply_units))),
        )
        object.__setattr__(
            self,
            "truck_initial_emergency_supply_units",
            int(max(0, int(self.truck_initial_emergency_supply_units))),
        )
        object.__setattr__(
            self,
            "uav_max_emergency_units",
            int(max(1, int(self.uav_max_emergency_units))),
        )
        object.__setattr__(
            self,
            "truck_can_serve_emergency_tasks",
            bool(self.truck_can_serve_emergency_tasks),
        )
        object.__setattr__(
            self,
            "truck_conditional_emergency_service_enabled",
            bool(self.truck_conditional_emergency_service_enabled),
        )
        object.__setattr__(
            self,
            "truck_emergency_service_max_distance_m",
            float(max(float(self.truck_emergency_service_max_distance_m), 0.0)),
        )
        object.__setattr__(
            self,
            "truck_emergency_service_max_deadline_slack_steps",
            int(max(0, int(self.truck_emergency_service_max_deadline_slack_steps))),
        )
        object.__setattr__(
            self,
            "truck_high_pressure_emergency_service_max_distance_m",
            float(max(float(self.truck_high_pressure_emergency_service_max_distance_m), 0.0)),
        )
        object.__setattr__(
            self,
            "truck_high_pressure_emergency_service_max_deadline_slack_steps",
            int(max(0, int(self.truck_high_pressure_emergency_service_max_deadline_slack_steps))),
        )
        object.__setattr__(
            self,
            "truck_replenish_for_emergency_stock",
            bool(self.truck_replenish_for_emergency_stock),
        )
        object.__setattr__(
            self,
            "truck_replenish_only_at_depot",
            bool(self.truck_replenish_only_at_depot),
        )
        object.__setattr__(
            self,
            "uav_replenish_only_from_truck",
            bool(self.uav_replenish_only_from_truck),
        )
        object.__setattr__(
            self,
            "uav_reload_at_depot_enabled",
            bool(self.uav_reload_at_depot_enabled),
        )
        object.__setattr__(
            self,
            "uav_must_replenish_after_each_service",
            bool(self.uav_must_replenish_after_each_service),
        )
        object.__setattr__(
            self,
            "truck_replenish_service_steps",
            int(max(1, int(self.truck_replenish_service_steps))),
        )
        object.__setattr__(
            self,
            "uav_reload_service_steps",
            int(max(1, int(self.uav_reload_service_steps))),
        )
        object.__setattr__(
            self,
            "forced_island_emergency_tasks",
            int(max(0, int(self.forced_island_emergency_tasks))),
        )
        object.__setattr__(
            self,
            "forced_island_lock_edges",
            bool(self.forced_island_lock_edges),
        )
        object.__setattr__(
            self,
            "forced_island_deadline_extension_steps",
            int(max(0, int(self.forced_island_deadline_extension_steps))),
        )
        object.__setattr__(self, "alns_enabled", bool(self.alns_enabled))
        object.__setattr__(self, "alns_adaptive_horizon_enabled", bool(self.alns_adaptive_horizon_enabled))
        object.__setattr__(self, "alns_risk_pressure_enabled", bool(self.alns_risk_pressure_enabled))
        object.__setattr__(self, "alns_ghost_tasks_enabled", bool(self.alns_ghost_tasks_enabled))
        if bool(self.alns_risk_pressure_enabled) != bool(self.alns_ghost_tasks_enabled):
            object.__setattr__(
                self,
                "alns_ghost_tasks_enabled",
                bool(self.alns_risk_pressure_enabled),
            )
        object.__setattr__(self, "alns_iterations", int(max(0, int(self.alns_iterations))))
        object.__setattr__(self, "alns_min_replan_interval_steps", int(max(1, int(self.alns_min_replan_interval_steps))))
        object.__setattr__(
            self,
            "alns_max_replan_interval_steps",
            int(max(int(self.alns_min_replan_interval_steps), int(self.alns_max_replan_interval_steps))),
        )
        object.__setattr__(self, "alns_min_horizon_steps", int(max(1, int(self.alns_min_horizon_steps))))
        object.__setattr__(
            self,
            "alns_max_horizon_steps",
            int(max(int(self.alns_min_horizon_steps), int(self.alns_max_horizon_steps))),
        )
        object.__setattr__(self, "alns_destroy_max_assignments", int(max(1, int(self.alns_destroy_max_assignments))))
        object.__setattr__(self, "alns_accept_temperature", float(max(float(self.alns_accept_temperature), 1e-6)))
        object.__setattr__(self, "alns_solution_mode", str(self.alns_solution_mode).strip().lower() or "legacy_k1")
        object.__setattr__(self, "alns_sequence_length", int(max(1, int(self.alns_sequence_length))))
        object.__setattr__(self, "adaptive_horizon_mode", str(self.adaptive_horizon_mode).strip().lower() or "disabled")
        allowed_horizon = tuple(int(x) for x in tuple(self.adaptive_horizon_allowed_values))
        object.__setattr__(self, "adaptive_horizon_allowed_values", allowed_horizon)
        object.__setattr__(self, "local_search_mode", str(self.local_search_mode).strip().lower() or "disabled")
        object.__setattr__(self, "local_search_max_moves_per_iteration", int(max(0, int(self.local_search_max_moves_per_iteration))))
        object.__setattr__(
            self,
            "local_search_max_exact_checks_per_iteration",
            int(max(0, int(self.local_search_max_exact_checks_per_iteration))),
        )
        object.__setattr__(self, "local_search_max_time_ms_per_iteration", int(max(0, int(self.local_search_max_time_ms_per_iteration))))
        object.__setattr__(
            self,
            "local_search_disabled_moves",
            tuple(str(x).strip().lower() for x in tuple(self.local_search_disabled_moves) if str(x).strip()),
        )
        object.__setattr__(self, "alns_operator_pool", str(self.alns_operator_pool).strip().lower() or "legacy")
        object.__setattr__(self, "alns_initialization_mode", str(self.alns_initialization_mode).strip().lower() or "objective_greedy")
        object.__setattr__(self, "alns_operator_weight_profile", str(self.alns_operator_weight_profile).strip().lower() or "uniform")
        object.__setattr__(self, "alns_selection_mode", str(self.alns_selection_mode).strip().lower() or "adaptive")
        object.__setattr__(self, "alns_critical_recovery_repair_enabled", bool(self.alns_critical_recovery_repair_enabled))
        object.__setattr__(self, "alns_critical_recovery_repair_max_tasks", int(max(0, int(self.alns_critical_recovery_repair_max_tasks))))
        object.__setattr__(
            self,
            "alns_critical_recovery_repair_min_priority",
            float(max(float(self.alns_critical_recovery_repair_min_priority), 0.0)),
        )
        object.__setattr__(self, "alns_critical_recovery_repair_prefer_truck", bool(self.alns_critical_recovery_repair_prefer_truck))
        object.__setattr__(self, "alns_critical_recovery_repair_avoid_failed_agent", bool(self.alns_critical_recovery_repair_avoid_failed_agent))
        object.__setattr__(self, "alns_critical_support_rebind_enabled", bool(self.alns_critical_support_rebind_enabled))
        object.__setattr__(self, "alns_critical_support_rebind_max_tasks", int(max(0, int(self.alns_critical_support_rebind_max_tasks))))
        object.__setattr__(
            self,
            "alns_critical_support_rebind_min_assigned_count",
            int(max(0, int(self.alns_critical_support_rebind_min_assigned_count))),
        )
        object.__setattr__(
            self,
            "alns_critical_support_rebind_prefer_historical_binding",
            bool(self.alns_critical_support_rebind_prefer_historical_binding),
        )
        object.__setattr__(
            self,
            "alns_critical_support_rebind_allow_nearest_feasible_truck",
            bool(self.alns_critical_support_rebind_allow_nearest_feasible_truck),
        )
        object.__setattr__(
            self,
            "alns_critical_support_rebind_preserve_recovery_anchor",
            bool(self.alns_critical_support_rebind_preserve_recovery_anchor),
        )
        object.__setattr__(
            self,
            "alns_critical_support_rebind_target_only_failed_or_pending",
            bool(self.alns_critical_support_rebind_target_only_failed_or_pending),
        )
        object.__setattr__(self, "alns_support_rebind_margin_aware_enabled", bool(self.alns_support_rebind_margin_aware_enabled))
        object.__setattr__(self, "alns_support_rebind_anchor_ranking_enabled", bool(self.alns_support_rebind_anchor_ranking_enabled))
        object.__setattr__(
            self,
            "alns_support_rebind_failed_binding_avoidance_enabled",
            bool(self.alns_support_rebind_failed_binding_avoidance_enabled),
        )
        penalty_mode = str(self.alns_support_rebind_failed_binding_penalty).strip().lower() or "mild"
        if penalty_mode not in {"mild", "medium"}:
            penalty_mode = "mild"
        object.__setattr__(self, "alns_support_rebind_failed_binding_penalty", penalty_mode)
        object.__setattr__(
            self,
            "alns_support_rebind_critical_first_ordering_enabled",
            bool(self.alns_support_rebind_critical_first_ordering_enabled),
        )
        object.__setattr__(self, "alns_support_rebind_safe_uav_guard_enabled", bool(self.alns_support_rebind_safe_uav_guard_enabled))
        object.__setattr__(self, "alns_support_rebind_margin_top_k", int(max(1, int(self.alns_support_rebind_margin_top_k))))
        object.__setattr__(
            self,
            "alns_support_rebind_anchor_search_radius_factor",
            float(max(float(self.alns_support_rebind_anchor_search_radius_factor), 0.1)),
        )
        object.__setattr__(self, "alns_lc_critical_recovery_path_enabled", bool(self.alns_lc_critical_recovery_path_enabled))
        object.__setattr__(
            self,
            "alns_lc_critical_recovery_path_max_tasks",
            int(max(0, int(self.alns_lc_critical_recovery_path_max_tasks))),
        )
        object.__setattr__(
            self,
            "alns_lc_critical_recovery_path_min_assigned_count",
            int(max(0, int(self.alns_lc_critical_recovery_path_min_assigned_count))),
        )
        object.__setattr__(
            self,
            "alns_lc_critical_recovery_path_top_k_bindings",
            int(max(1, int(self.alns_lc_critical_recovery_path_top_k_bindings))),
        )
        object.__setattr__(
            self,
            "alns_lc_critical_recovery_path_require_positive_margin",
            bool(self.alns_lc_critical_recovery_path_require_positive_margin),
        )
        object.__setattr__(
            self,
            "alns_lc_critical_recovery_path_prioritize_no_bindable_truck",
            bool(self.alns_lc_critical_recovery_path_prioritize_no_bindable_truck),
        )
        object.__setattr__(
            self,
            "alns_lc_critical_recovery_path_avoid_repeated_failed_tuple",
            bool(self.alns_lc_critical_recovery_path_avoid_repeated_failed_tuple),
        )
        object.__setattr__(
            self,
            "alns_lc_critical_recovery_path_target_critical_only",
            bool(self.alns_lc_critical_recovery_path_target_critical_only),
        )
        object.__setattr__(self, "alns_assigned_critical_reconstruct_enabled", bool(self.alns_assigned_critical_reconstruct_enabled))
        object.__setattr__(
            self,
            "alns_assigned_critical_reconstruct_max_tasks",
            int(max(0, int(self.alns_assigned_critical_reconstruct_max_tasks))),
        )
        object.__setattr__(
            self,
            "alns_assigned_critical_reconstruct_min_assigned_count",
            int(max(0, int(self.alns_assigned_critical_reconstruct_min_assigned_count))),
        )
        object.__setattr__(
            self,
            "alns_assigned_critical_reconstruct_top_k_paths",
            int(max(1, int(self.alns_assigned_critical_reconstruct_top_k_paths))),
        )
        object.__setattr__(
            self,
            "alns_assigned_critical_reconstruct_target_critical_only",
            bool(self.alns_assigned_critical_reconstruct_target_critical_only),
        )
        object.__setattr__(self, "alns_support_reposition_shadow_enabled", bool(self.alns_support_reposition_shadow_enabled))
        object.__setattr__(
            self,
            "alns_support_reposition_shadow_max_tasks",
            int(max(0, int(self.alns_support_reposition_shadow_max_tasks))),
        )
        object.__setattr__(
            self,
            "alns_support_reposition_shadow_min_assigned_count",
            int(max(0, int(self.alns_support_reposition_shadow_min_assigned_count))),
        )
        object.__setattr__(self, "physical_environment_version", str(self.physical_environment_version).strip().lower() or "v1")
        object.__setattr__(
            self,
            "physical_environment_safety_protocol",
            str(self.physical_environment_safety_protocol).strip().lower() or "shielded_operation",
        )
        object.__setattr__(self, "candidate_ranker_mode", str(self.candidate_ranker_mode).strip().lower() or "disabled")
        object.__setattr__(self, "candidate_ranker_pool_size", int(max(1, int(self.candidate_ranker_pool_size))))
        object.__setattr__(self, "candidate_ranker_exact_check_budget", int(max(1, int(self.candidate_ranker_exact_check_budget))))
        object.__setattr__(self, "candidate_ranker_exploration_count", int(max(0, int(self.candidate_ranker_exploration_count))))
        object.__setattr__(self, "alns_weight_segment_length", int(max(1, int(self.alns_weight_segment_length))))
        object.__setattr__(self, "alns_weight_learning_rate", float(np.clip(float(self.alns_weight_learning_rate), 0.0, 1.0)))
        object.__setattr__(self, "alns_weight_min", float(max(float(self.alns_weight_min), 1e-9)))
        object.__setattr__(self, "alns_sa_auto_calibration_enabled", bool(self.alns_sa_auto_calibration_enabled))
        object.__setattr__(self, "alns_sa_sample_count", int(max(1, int(self.alns_sa_sample_count))))
        object.__setattr__(self, "alns_sa_delta_quantile", float(np.clip(float(self.alns_sa_delta_quantile), 0.0, 1.0)))
        object.__setattr__(
            self,
            "alns_sa_initial_worse_accept_probability",
            float(np.clip(float(self.alns_sa_initial_worse_accept_probability), 1e-6, 1.0 - 1e-6)),
        )
        object.__setattr__(self, "alns_sa_cooling_rate", float(np.clip(float(self.alns_sa_cooling_rate), 0.0, 1.0)))
        object.__setattr__(self, "alns_sa_minimum_temperature", float(max(float(self.alns_sa_minimum_temperature), 1e-12)))
        object.__setattr__(self, "alns_sa_reheat_enabled", bool(self.alns_sa_reheat_enabled))
        if str(self.alns_solution_mode) == "legacy_k1" and int(self.alns_sequence_length) != 1:
            raise ValueError("alns_solution_mode='legacy_k1' requires alns_sequence_length == 1")
        if str(self.alns_solution_mode) in {"k2_shadow", "k2_active"} and int(self.alns_sequence_length) != 2:
            raise ValueError("alns_solution_mode in {'k2_shadow', 'k2_active'} requires alns_sequence_length == 2")
        if str(self.alns_solution_mode) not in {"legacy_k1", "k2_shadow", "k2_active"}:
            raise ValueError("alns_solution_mode must be one of: legacy_k1, k2_shadow, k2_active")
        if str(self.adaptive_horizon_mode) not in {"disabled", "shadow", "active"}:
            raise ValueError("adaptive_horizon_mode must be one of: disabled, shadow, active")
        if set(self.adaptive_horizon_allowed_values) - {1, 2}:
            raise ValueError("adaptive_horizon_allowed_values may only contain K=1 and K=2")
        if str(self.local_search_mode) not in {"disabled", "active"}:
            raise ValueError("local_search_mode must be one of: disabled, active")
        if str(self.alns_operator_pool) not in {
            "legacy",
            "canonical_k2",
            "tabu_k2",
            "er_k2",
            "combined_k2",
            "canonical_only",
            "er_only",
            "combined",
            "no_road_group",
            "no_critical_group",
            "no_support_group",
            "no_synchronization_group",
        }:
            raise ValueError("unsupported alns_operator_pool")
        if str(self.alns_initialization_mode) not in {"objective_greedy", "critical_first"}:
            raise ValueError("unsupported alns_initialization_mode")
        if str(self.alns_operator_weight_profile) not in {"uniform", "critical_repair_bias", "feasibility_restore_bias"}:
            raise ValueError("unsupported alns_operator_weight_profile")
        if str(self.alns_selection_mode) not in {"adaptive", "uniform", "tabu"}:
            raise ValueError("alns_selection_mode must be one of: adaptive, uniform, tabu")
        if str(self.physical_environment_version) not in {"v1", "v2"}:
            raise ValueError("physical_environment_version must be one of: v1, v2")
        if str(self.physical_environment_safety_protocol) not in {"shielded_operation", "unshielded_stress"}:
            raise ValueError("physical_environment_safety_protocol must be one of: shielded_operation, unshielded_stress")
        if str(self.candidate_ranker_mode) not in {"disabled", "shadow", "active"}:
            raise ValueError("candidate_ranker_mode must be one of: disabled, shadow, active")
        if int(self.candidate_ranker_exact_check_budget) >= int(self.candidate_ranker_pool_size):
            raise ValueError("candidate_ranker_exact_check_budget must be < candidate_ranker_pool_size")
        if int(self.candidate_ranker_exploration_count) > int(self.candidate_ranker_exact_check_budget):
            raise ValueError("candidate_ranker_exploration_count must be <= candidate_ranker_exact_check_budget")
        object.__setattr__(self, "alns_safe_overlay_enabled", bool(self.alns_safe_overlay_enabled))
        object.__setattr__(self, "alns_destroy_existing_enabled", bool(self.alns_destroy_existing_enabled))
        object.__setattr__(self, "alns_protect_recent_goal_steps", int(max(0, int(self.alns_protect_recent_goal_steps))))
        object.__setattr__(self, "alns_protect_progress_epsilon_m", float(max(float(self.alns_protect_progress_epsilon_m), 0.0)))
        object.__setattr__(self, "alns_stale_goal_steps", int(max(1, int(self.alns_stale_goal_steps))))
        object.__setattr__(
            self,
            "uav_hard_recovery_battery_guard",
            bool(self.uav_hard_recovery_battery_guard),
        )
        object.__setattr__(
            self,
            "uav_launch_min_battery_fraction",
            float(np.clip(float(self.uav_launch_min_battery_fraction), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "uav_min_takeoff_soc",
            float(np.clip(float(self.uav_min_takeoff_soc), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "uav_emergency_reserve_fraction",
            float(np.clip(float(self.uav_emergency_reserve_fraction), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "uav_return_margin_fraction",
            float(np.clip(float(self.uav_return_margin_fraction), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "uav_recovery_distance_buffer_m",
            float(max(float(self.uav_recovery_distance_buffer_m), 0.0)),
        )
        object.__setattr__(
            self,
            "uav_recovery_truck_drift_margin_scale",
            float(max(float(self.uav_recovery_truck_drift_margin_scale), 0.0)),
        )
        object.__setattr__(
            self,
            "uav_recovery_truck_drift_margin_max_m",
            float(max(float(self.uav_recovery_truck_drift_margin_max_m), 0.0)),
        )
        object.__setattr__(
            self,
            "uav_recovery_idle_hold_threshold",
            float(np.clip(float(self.uav_recovery_idle_hold_threshold), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "uav_recovery_idle_hold_min_dist_m",
            float(max(float(self.uav_recovery_idle_hold_min_dist_m), 0.0)),
        )
        object.__setattr__(
            self,
            "uav_rendezvous_margin_fraction",
            float(np.clip(float(self.uav_rendezvous_margin_fraction), 0.0, 1.0)),
        )
        object.__setattr__(self, "uav_allow_rendezvous_launch", bool(self.uav_allow_rendezvous_launch))
        object.__setattr__(
            self,
            "uav_rendezvous_launch_requires_docked_truck_goal",
            bool(self.uav_rendezvous_launch_requires_docked_truck_goal),
        )
        object.__setattr__(
            self,
            "uav_conditional_rendezvous_launch_enabled",
            bool(self.uav_conditional_rendezvous_launch_enabled),
        )
        object.__setattr__(
            self,
            "uav_conditional_rendezvous_min_pending_emergency",
            int(max(0, int(self.uav_conditional_rendezvous_min_pending_emergency))),
        )
        object.__setattr__(
            self,
            "uav_conditional_rendezvous_max_deadline_slack_steps",
            int(max(0, int(self.uav_conditional_rendezvous_max_deadline_slack_steps))),
        )
        object.__setattr__(
            self,
            "uav_conditional_rendezvous_max_nearest_truck_m",
            float(max(float(self.uav_conditional_rendezvous_max_nearest_truck_m), 0.0)),
        )
        object.__setattr__(
            self,
            "uav_adaptive_launch_gate_enabled",
            bool(self.uav_adaptive_launch_gate_enabled),
        )
        object.__setattr__(
            self,
            "uav_adaptive_launch_min_floor",
            float(np.clip(float(self.uav_adaptive_launch_min_floor), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "uav_adaptive_launch_relax_delta",
            float(np.clip(float(self.uav_adaptive_launch_relax_delta), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "uav_adaptive_force_takeoff_gap",
            float(np.clip(float(self.uav_adaptive_force_takeoff_gap), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "uav_high_pressure_launch_min_floor",
            float(np.clip(float(self.uav_high_pressure_launch_min_floor), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "uav_high_pressure_force_takeoff_gap",
            float(np.clip(float(self.uav_high_pressure_force_takeoff_gap), 0.0, 1.0)),
        )

        object.__setattr__(
            self,
            "uav_reject_cache_window_steps",
            int(max(1, int(self.uav_reject_cache_window_steps))),
        )
        object.__setattr__(
            self,
            "uav_reject_cache_min_repeat",
            int(max(1, int(self.uav_reject_cache_min_repeat))),
        )
        object.__setattr__(
            self,
            "uav_reject_cache_ttl_steps",
            int(max(1, int(self.uav_reject_cache_ttl_steps))),
        )
        object.__setattr__(
            self,
            "uav_high_pressure_rendezvous_enabled",
            bool(self.uav_high_pressure_rendezvous_enabled),
        )
        object.__setattr__(
            self,
            "uav_high_pressure_rendezvous_max_deadline_slack_steps",
            int(max(0, int(self.uav_high_pressure_rendezvous_max_deadline_slack_steps))),
        )
        object.__setattr__(
            self,
            "uav_high_pressure_rendezvous_max_nearest_truck_m",
            float(max(float(self.uav_high_pressure_rendezvous_max_nearest_truck_m), 0.0)),
        )
        object.__setattr__(
            self,
            "uav_high_pressure_recovery_margin_bonus_m",
            float(max(float(self.uav_high_pressure_recovery_margin_bonus_m), 0.0)),
        )
        object.__setattr__(
            self,
            "uav_relaxed_rendezvous_recovery_margin_bonus_m",
            float(max(float(self.uav_relaxed_rendezvous_recovery_margin_bonus_m), 0.0)),
        )
        object.__setattr__(
            self,
            "uav_low_battery_goal_lock_threshold",
            float(np.clip(float(self.uav_low_battery_goal_lock_threshold), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "uav_low_battery_force_recover_threshold",
            float(np.clip(float(self.uav_low_battery_force_recover_threshold), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "uav_launch_speed_utilization",
            float(np.clip(float(self.uav_launch_speed_utilization), 0.1, 1.0)),
        )
        object.__setattr__(
            self,
            "uav_launch_min_horizon_buffer_steps",
            int(max(0, int(self.uav_launch_min_horizon_buffer_steps))),
        )
        object.__setattr__(
            self,
            "uav_launch_min_remaining_steps",
            int(max(0, int(self.uav_launch_min_remaining_steps))),
        )
        object.__setattr__(
            self,
            "uav_recovery_idle_discharge_scale",
            float(np.clip(float(self.uav_recovery_idle_discharge_scale), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "uav_short_sortie_max_distance_m",
            float(max(float(self.uav_short_sortie_max_distance_m), 1.0)),
        )
        object.__setattr__(
            self,
            "uav_short_sortie_min_battery",
            float(np.clip(float(self.uav_short_sortie_min_battery), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "uav_recovery_near_truck_radius_m",
            float(max(float(self.uav_recovery_near_truck_radius_m), 0.0)),
        )
        object.__setattr__(
            self,
            "uav_recovery_directional_select_enabled",
            bool(self.uav_recovery_directional_select_enabled),
        )
        object.__setattr__(
            self,
            "uav_recovery_direction_weight",
            float(max(float(self.uav_recovery_direction_weight), 0.0)),
        )
        object.__setattr__(
            self,
            "uav_recovery_distance_weight",
            float(max(float(self.uav_recovery_distance_weight), 0.0)),
        )
        object.__setattr__(
            self,
            "uav_recovery_stock_weight",
            float(max(float(self.uav_recovery_stock_weight), 0.0)),
        )
        object.__setattr__(
            self,
            "uav_recovery_truck_request_enabled",
            bool(self.uav_recovery_truck_request_enabled),
        )
        object.__setattr__(
            self,
            "truck_recovery_request_match_bonus",
            float(max(float(self.truck_recovery_request_match_bonus), 0.0)),
        )
        object.__setattr__(
            self,
            "uav_docked_task_shortlist_enabled",
            bool(self.uav_docked_task_shortlist_enabled),
        )
        object.__setattr__(
            self,
            "uav_docked_task_shortlist_radius_m",
            float(max(float(self.uav_docked_task_shortlist_radius_m), 0.0)),
        )
        object.__setattr__(
            self,
            "uav_docked_task_shortlist_topk",
            int(max(1, int(self.uav_docked_task_shortlist_topk))),
        )
        object.__setattr__(
            self,
            "uav_docked_hard_far_switch_margin_m",
            float(max(float(self.uav_docked_hard_far_switch_margin_m), 0.0)),
        )
        object.__setattr__(
            self,
            "uav_docked_opportunistic_retarget_enabled",
            bool(self.uav_docked_opportunistic_retarget_enabled),
        )
        object.__setattr__(
            self,
            "uav_docked_opportunistic_radius_m",
            float(max(float(self.uav_docked_opportunistic_radius_m), 0.0)),
        )
        object.__setattr__(
            self,
            "uav_docked_opportunistic_gain_ratio",
            float(np.clip(float(self.uav_docked_opportunistic_gain_ratio), 0.05, 0.99)),
        )
        object.__setattr__(
            self,
            "uav_docked_opportunistic_min_margin_m",
            float(max(float(self.uav_docked_opportunistic_min_margin_m), 0.0)),
        )
        object.__setattr__(
            self,
            "uav_docked_opportunistic_cooldown_steps",
            int(max(0, int(self.uav_docked_opportunistic_cooldown_steps))),
        )
        object.__setattr__(
            self,
            "uav_docked_retarget_enabled",
            bool(self.uav_docked_retarget_enabled),
        )
        object.__setattr__(
            self,
            "uav_docked_retarget_interval_steps",
            int(max(1, int(self.uav_docked_retarget_interval_steps))),
        )
        object.__setattr__(
            self,
            "uav_env_island_goal_override_enabled",
            bool(self.uav_env_island_goal_override_enabled),
        )
        object.__setattr__(
            self,
            "uav_docked_near_dispatch_radius_m",
            float(max(float(self.uav_docked_near_dispatch_radius_m), 1.0)),
        )
        object.__setattr__(
            self,
            "uav_docked_heading_dispatch_radius_m",
            float(max(float(self.uav_docked_heading_dispatch_radius_m), float(self.uav_docked_near_dispatch_radius_m))),
        )
        object.__setattr__(
            self,
            "uav_docked_heading_min_cosine",
            float(np.clip(float(self.uav_docked_heading_min_cosine), -1.0, 1.0)),
        )
        object.__setattr__(
            self,
            "uav_docked_near_dispatch_bonus",
            float(max(float(self.uav_docked_near_dispatch_bonus), 0.0)),
        )
        object.__setattr__(
            self,
            "uav_docked_heading_dispatch_bonus",
            float(max(float(self.uav_docked_heading_dispatch_bonus), 0.0)),
        )
        object.__setattr__(
            self,
            "uav_docked_far_task_penalty",
            float(max(float(self.uav_docked_far_task_penalty), 0.0)),
        )
        object.__setattr__(
            self,
            "uav_urgent_watchdog_enabled",
            bool(self.uav_urgent_watchdog_enabled),
        )
        object.__setattr__(
            self,
            "uav_urgent_watchdog_slack_steps",
            int(max(0, int(self.uav_urgent_watchdog_slack_steps))),
        )
        object.__setattr__(
            self,
            "uav_urgent_watchdog_retarget_cooldown_steps",
            int(max(1, int(self.uav_urgent_watchdog_retarget_cooldown_steps))),
        )
        object.__setattr__(
            self,
            "uav_urgent_watchdog_distance_bonus_m",
            float(max(float(self.uav_urgent_watchdog_distance_bonus_m), 0.0)),
        )
        object.__setattr__(
            self,
            "uav_urgent_watchdog_max_assign_per_step",
            int(max(0, int(self.uav_urgent_watchdog_max_assign_per_step))),
        )
        object.__setattr__(
            self,
            "uav_initial_distinct_emergency_assign",
            bool(self.uav_initial_distinct_emergency_assign),
        )
        object.__setattr__(
            self,
            "uav_initial_distinct_window_steps",
            int(max(0, int(self.uav_initial_distinct_window_steps))),
        )
        if float(self.uav_low_battery_force_recover_threshold) > float(self.uav_low_battery_goal_lock_threshold):
            object.__setattr__(
                self,
                "uav_low_battery_goal_lock_threshold",
                float(self.uav_low_battery_force_recover_threshold),
            )
        object.__setattr__(
            self,
            "uav_bind_commit_steps",
            int(max(1, int(self.uav_bind_commit_steps))),
        )
        object.__setattr__(
            self,
            "uav_post_bind_min_dwell_steps",
            int(max(0, int(self.uav_post_bind_min_dwell_steps))),
        )
        object.__setattr__(
            self,
            "uav_bind_motion_window_gain",
            float(max(float(self.uav_bind_motion_window_gain), 0.0)),
        )
        object.__setattr__(
            self,
            "uav_bind_latency_margin_s",
            float(max(float(self.uav_bind_latency_margin_s), 0.0)),
        )
        object.__setattr__(
            self,
            "uav_post_bind_force_recharge",
            bool(self.uav_post_bind_force_recharge),
        )
        object.__setattr__(
            self,
            "uav_post_bind_force_reload",
            bool(self.uav_post_bind_force_reload),
        )
        object.__setattr__(
            self,
            "uav_unsafe_launch_block_cooldown_steps",
            int(max(0, int(self.uav_unsafe_launch_block_cooldown_steps))),
        )
        object.__setattr__(
            self,
            "truck_support_uav_recovery_enabled",
            bool(self.truck_support_uav_recovery_enabled),
        )
        object.__setattr__(
            self,
            "truck_recovery_max_detour_cost_weight",
            float(max(float(self.truck_recovery_max_detour_cost_weight), 0.0)),
        )
        object.__setattr__(
            self,
            "truck_recovery_priority_weight",
            float(max(float(self.truck_recovery_priority_weight), 0.0)),
        )
        object.__setattr__(
            self,
            "truck_recovery_require_request_when_normal_pending",
            bool(self.truck_recovery_require_request_when_normal_pending),
        )
        object.__setattr__(
            self,
            "truck_recovery_request_min_urgency_when_normal_pending",
            float(np.clip(float(self.truck_recovery_request_min_urgency_when_normal_pending), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "truck_routine_near_goal_support_protect_enabled",
            bool(self.truck_routine_near_goal_support_protect_enabled),
        )
        object.__setattr__(
            self,
            "truck_routine_near_goal_support_protect_steps",
            int(max(0, int(self.truck_routine_near_goal_support_protect_steps))),
        )
        object.__setattr__(
            self,
            "hrl_routine_near_completion_eta_steps",
            int(max(0, int(self.hrl_routine_near_completion_eta_steps))),
        )
        object.__setattr__(
            self,
            "hrl_routine_near_completion_route_dist_m",
            float(max(float(self.hrl_routine_near_completion_route_dist_m), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_routine_protection_tc_override_enabled",
            bool(self.hrl_routine_protection_tc_override_enabled),
        )
        object.__setattr__(
            self,
            "hrl_routine_protection_tc_override_max_routine_delay_steps",
            int(max(0, int(self.hrl_routine_protection_tc_override_max_routine_delay_steps))),
        )
        object.__setattr__(
            self,
            "hrl_routine_protection_tc_override_max_support_steps",
            int(max(0, int(self.hrl_routine_protection_tc_override_max_support_steps))),
        )
        object.__setattr__(
            self,
            "hrl_routine_protection_tc_override_min_launch_gain_m",
            float(max(float(self.hrl_routine_protection_tc_override_min_launch_gain_m), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_routine_protection_tc_override_require_recovery_feasible",
            bool(self.hrl_routine_protection_tc_override_require_recovery_feasible),
        )
        object.__setattr__(
            self,
            "hrl_routine_protection_tc_override_require_loaded_uav",
            bool(self.hrl_routine_protection_tc_override_require_loaded_uav),
        )
        object.__setattr__(
            self,
            "hrl_routine_protection_delivery_feasible_tc_override_enabled",
            bool(self.hrl_routine_protection_delivery_feasible_tc_override_enabled),
        )
        object.__setattr__(
            self,
            "hrl_tc_override_require_full_sortie_feasible",
            bool(self.hrl_tc_override_require_full_sortie_feasible),
        )
        object.__setattr__(
            self,
            "hrl_tc_override_min_recovery_margin_m",
            float(max(float(self.hrl_tc_override_min_recovery_margin_m), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_tc_override_min_battery_margin_ratio",
            float(max(float(self.hrl_tc_override_min_battery_margin_ratio), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_tc_override_max_expected_lifeline_decay_ratio",
            float(np.clip(float(self.hrl_tc_override_max_expected_lifeline_decay_ratio), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "hrl_tc_override_block_if_recent_reject",
            bool(self.hrl_tc_override_block_if_recent_reject),
        )
        object.__setattr__(
            self,
            "hrl_tc_override_reject_cache_ttl_steps",
            int(max(0, int(self.hrl_tc_override_reject_cache_ttl_steps))),
        )
        object.__setattr__(
            self,
            "hrl_tc_override_max_routine_delay_steps",
            int(max(0, int(self.hrl_tc_override_max_routine_delay_steps))),
        )
        object.__setattr__(
            self,
            "hrl_tc_override_min_delivery_score_gain",
            float(max(float(self.hrl_tc_override_min_delivery_score_gain), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_tc_override_max_support_steps",
            int(max(0, int(self.hrl_tc_override_max_support_steps))),
        )
        # Candidate-only risk-slack routine repair controls.  Keep these
        # algorithm-owned knobs normalized even when supplied by an external
        # candidate config; defaults remain fully disabled.
        object.__setattr__(
            self,
            "hrl_route_plan_risk_slack_routine_repair_enabled",
            bool(self.hrl_route_plan_risk_slack_routine_repair_enabled),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_risk_slack_routine_slack_steps",
            int(max(0, int(self.hrl_route_plan_risk_slack_routine_slack_steps))),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_risk_slack_routine_stall_steps",
            int(max(1, int(self.hrl_route_plan_risk_slack_routine_stall_steps))),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_risk_slack_routine_max_transfers",
            int(max(0, int(self.hrl_route_plan_risk_slack_routine_max_transfers))),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_risk_slack_routine_eta_gain_steps",
            float(max(float(self.hrl_route_plan_risk_slack_routine_eta_gain_steps), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_risk_slack_routine_eta_gain_ratio",
            float(np.clip(float(self.hrl_route_plan_risk_slack_routine_eta_gain_ratio), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_risk_slack_routine_radius_m",
            float(max(float(self.hrl_route_plan_risk_slack_routine_radius_m), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_risk_slack_routine_reserved_inventory_guard_enabled",
            bool(self.hrl_route_plan_risk_slack_routine_reserved_inventory_guard_enabled),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_r4_routine_takeover_enabled",
            bool(self.hrl_route_plan_r4_routine_takeover_enabled),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_r4_routine_takeover_stall_steps",
            int(max(1, int(self.hrl_route_plan_r4_routine_takeover_stall_steps))),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_r4_routine_takeover_max_transfers",
            int(max(0, int(self.hrl_route_plan_r4_routine_takeover_max_transfers))),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_r4_routine_takeover_radius_m",
            float(max(float(self.hrl_route_plan_r4_routine_takeover_radius_m), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_idle_routine_dispatch_enabled",
            bool(self.hrl_route_plan_idle_routine_dispatch_enabled),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_idle_routine_dispatch_emergency_reserve_steps",
            int(max(0, int(self.hrl_route_plan_idle_routine_dispatch_emergency_reserve_steps))),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_idle_routine_dispatch_max_per_step",
            int(max(0, int(self.hrl_route_plan_idle_routine_dispatch_max_per_step))),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_force_initial_lifeline_ordering_enabled",
            bool(self.hrl_route_plan_force_initial_lifeline_ordering_enabled),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_balanced_all_tasks_enabled",
            bool(self.hrl_route_plan_balanced_all_tasks_enabled),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_balanced_all_tasks_v2_enabled",
            bool(self.hrl_route_plan_balanced_all_tasks_v2_enabled),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_balanced_all_tasks_v2_after_launch_only",
            bool(self.hrl_route_plan_balanced_all_tasks_v2_after_launch_only),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_balanced_all_tasks_v2_reauction_deadline_guard_enabled",
            bool(self.hrl_route_plan_balanced_all_tasks_v2_reauction_deadline_guard_enabled),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_balanced_all_tasks_v2_aggressive_pending_auction_enabled",
            bool(self.hrl_route_plan_balanced_all_tasks_v2_aggressive_pending_auction_enabled),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_balanced_all_tasks_v3_enabled",
            bool(self.hrl_route_plan_balanced_all_tasks_v3_enabled),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_balanced_all_tasks_v3_tail_insert_after_launch",
            bool(self.hrl_route_plan_balanced_all_tasks_v3_tail_insert_after_launch),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_balanced_all_tasks_v5_launch_first_enabled",
            bool(self.hrl_route_plan_balanced_all_tasks_v5_launch_first_enabled),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_balanced_all_tasks_normal_first_enabled",
            bool(self.hrl_route_plan_balanced_all_tasks_normal_first_enabled),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_balanced_all_tasks_max_normal_per_truck",
            int(max(0, int(self.hrl_route_plan_balanced_all_tasks_max_normal_per_truck))),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_balanced_all_tasks_allow_emergency_tradeoff",
            bool(self.hrl_route_plan_balanced_all_tasks_allow_emergency_tradeoff),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_balanced_all_tasks_emergency_lateness_tolerance_steps",
            int(max(0, int(self.hrl_route_plan_balanced_all_tasks_emergency_lateness_tolerance_steps))),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_balanced_all_tasks_watchdog_stall_steps",
            int(max(1, int(self.hrl_route_plan_balanced_all_tasks_watchdog_stall_steps))),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_balanced_all_tasks_watchdog_near_distance_m",
            float(max(float(self.hrl_route_plan_balanced_all_tasks_watchdog_near_distance_m), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_balanced_all_tasks_watchdog_max_transfers",
            int(max(0, int(self.hrl_route_plan_balanced_all_tasks_watchdog_max_transfers))),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_balanced_all_tasks_watchdog_transfer_min_gain_steps",
            float(max(float(self.hrl_route_plan_balanced_all_tasks_watchdog_transfer_min_gain_steps), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_balanced_all_tasks_watchdog_transfer_min_gain_ratio",
            float(np.clip(self.hrl_route_plan_balanced_all_tasks_watchdog_transfer_min_gain_ratio, 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_shadow_total_coverage_enabled",
            bool(self.hrl_route_plan_shadow_total_coverage_enabled),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_shadow_total_coverage_min_gain_tasks",
            int(max(1, int(self.hrl_route_plan_shadow_total_coverage_min_gain_tasks))),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_shadow_total_coverage_min_routine_slack_steps",
            int(max(0, int(self.hrl_route_plan_shadow_total_coverage_min_routine_slack_steps))),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_shadow_total_coverage_max_routine_distance_ratio",
            float(max(1.0, float(self.hrl_route_plan_shadow_total_coverage_max_routine_distance_ratio))),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_routine_service_start_rescue_enabled",
            bool(self.hrl_route_plan_routine_service_start_rescue_enabled),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_routine_service_start_rescue_stall_steps",
            int(max(1, int(self.hrl_route_plan_routine_service_start_rescue_stall_steps))),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_routine_service_start_rescue_near_distance_m",
            float(max(float(self.hrl_route_plan_routine_service_start_rescue_near_distance_m), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_routine_service_start_rescue_max_transfers",
            int(max(0, int(self.hrl_route_plan_routine_service_start_rescue_max_transfers))),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_routine_service_start_rescue_allow_stalled_owner_transfer",
            bool(self.hrl_route_plan_routine_service_start_rescue_allow_stalled_owner_transfer),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_routine_service_start_rescue_transfer_min_gain_steps",
            float(max(float(self.hrl_route_plan_routine_service_start_rescue_transfer_min_gain_steps), 0.0)),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_routine_service_start_rescue_transfer_min_gain_ratio",
            float(np.clip(self.hrl_route_plan_routine_service_start_rescue_transfer_min_gain_ratio, 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_stalled_normal_cleanup_enabled",
            bool(self.hrl_route_plan_stalled_normal_cleanup_enabled),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_hard_normal_rescue_enabled",
            bool(self.hrl_route_plan_hard_normal_rescue_enabled),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_hard_normal_rescue_stall_steps",
            int(max(1, int(self.hrl_route_plan_hard_normal_rescue_stall_steps))),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_hard_normal_rescue_max_per_call",
            int(max(0, int(self.hrl_route_plan_hard_normal_rescue_max_per_call))),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_hard_normal_rescue_airborne_parallel_enabled",
            bool(self.hrl_route_plan_hard_normal_rescue_airborne_parallel_enabled),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_hard_normal_rescue_tail_after_airborne",
            bool(self.hrl_route_plan_hard_normal_rescue_tail_after_airborne),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_hard_normal_rescue_orphan_only_enabled",
            bool(self.hrl_route_plan_hard_normal_rescue_orphan_only_enabled),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_hard_normal_rescue_pending_head_guard_enabled",
            bool(self.hrl_route_plan_hard_normal_rescue_pending_head_guard_enabled),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_hard_normal_rescue_candidate_head_guard_enabled",
            bool(self.hrl_route_plan_hard_normal_rescue_candidate_head_guard_enabled),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_hard_normal_rescue_no_truck_once_enabled",
            bool(self.hrl_route_plan_hard_normal_rescue_no_truck_once_enabled),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_hard_normal_rescue_no_truck_cooldown_enabled",
            bool(self.hrl_route_plan_hard_normal_rescue_no_truck_cooldown_enabled),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_hard_normal_rescue_no_truck_cooldown_steps",
            int(max(1, int(self.hrl_route_plan_hard_normal_rescue_no_truck_cooldown_steps))),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_hard_normal_rescue_adaptive_coverage_enabled",
            bool(self.hrl_route_plan_hard_normal_rescue_adaptive_coverage_enabled),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_hard_normal_rescue_min_orphan_pending",
            int(max(0, int(self.hrl_route_plan_hard_normal_rescue_min_orphan_pending))),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_parallel_routine_emergency_after_launch_enabled",
            bool(self.hrl_route_plan_parallel_routine_emergency_after_launch_enabled),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_mixed_coverage_enabled",
            bool(self.hrl_route_plan_mixed_coverage_enabled),
        )
        object.__setattr__(
            self,
            "hrl_route_plan_mixed_coverage_emergency_reserve_steps",
            int(max(0, int(self.hrl_route_plan_mixed_coverage_emergency_reserve_steps))),
        )
        # Runtime-configurable delta t (physics and MDP share the same dt).
        dt_s = float(self.dt_seconds)
        if dt_s <= 0.0:
            dt_s = float(self.dt)
        if dt_s <= 0.0:
            raise ValueError(f"dt_seconds must be > 0, got dt_seconds={self.dt_seconds}, dt={self.dt}")
        object.__setattr__(self, "dt_seconds", dt_s)
        object.__setattr__(self, "dt", dt_s)


@dataclass
class AgentRuntimeState:
    agent_id: str
    kind: AgentKind
    node: Optional[int] = None
    pos_xy: Optional[Tuple[float, float]] = None
    vel_xy: Optional[Tuple[float, float]] = None
    battery: float = 1.0
    crashed: bool = False
    transit: Optional[Tuple[int, int, float]] = None
    follow_target: Optional[str] = None
    sortie_distance_m: float = 0.0
    lifetime_distance_m: float = 0.0
    cargo: float = 0.0
    replenish_timer: int = 0
    # Material/supply states (paper minimal discrete-unit model).
    normal_supply_units: int = 0
    emergency_supply_units: int = 0
    bulk_inventory_kg_current: float = 0.0
    timecritical_inventory_kg_current: float = 0.0
    truck_inventory_kg_current: float = 0.0
    truck_needs_replenish_flag: bool = False
    truck_replenish_timer: int = 0
    carried_emergency_units: int = 0
    payload_kg_current: float = 0.0
    # Payload provenance is needed by the v2 BULK_RELAY mode.  Existing UAV
    # behavior keeps the default emergency package; relay sorties explicitly
    # load a small bulk chunk without changing the task's original category.
    payload_supply_type: str = "emergency"
    uav_needs_reload_flag: bool = False
    uav_reload_timer: int = 0


@dataclass
class DeliveryTask:
    task_id: str
    kind: TaskKind
    demand_node: int
    deadline_step: int
    task_class: str = TaskClass.ROUTINE_BULK.value
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: Optional[str] = None
    in_service_by: Optional[str] = None
    service_remaining: int = 0
    demand_left: float = 0.0
    # Remaining demand in kilograms.
    remaining_demand_kg: float = 0.0
    fulfilled_mass_kg: float = 0.0
    demand_kg: float = 0.0
    demand_units: int = 1
    # Inventory-unit gating (decoupled from service demand units).
    supply_units_required: int = 1
    supply_units_consumed: bool = False
    # Material type used by inventory gate.
    # NORMAL -> "normal", EMERGENCY -> "emergency".
    supply_type: str = "normal"
    urgency_score: float = 0.5
    lifeline_init: float = 100.0
    lifeline_current: float = 100.0
    lifeline_decay_rate: float = 0.0
    remaining_lifeline_at_service: float = 0.0
    failed_due_to_lifeline_zero: bool = False
    created_step: int = 0
    first_service_step: Optional[int] = None
    delivered_by: Optional[str] = None
    delivered_step: Optional[int] = None
    failed_step: Optional[int] = None
    # V2 planning metadata.  These fields intentionally do not replace kind
    # or task_class, so paper statistics always retain the original category.
    service_mode: str = "DIRECT"
    original_task_kind: str = ""
    route_contract_owner: Optional[str] = None
    route_contract_truck: Optional[str] = None
    # Monotone layer-1 publication version. Zero denotes legacy ownership.
    route_contract_version: int = 0
    # A BULK_RELAY contract may intentionally name both UAVs of one truck.
    # This is a cooperative exception to ordinary one-task/one-agent locking.
    route_contract_uav_ids: Tuple[str, ...] = ()
    relay_service_agents: Tuple[str, ...] = ()


@dataclass
class HazardSnapshot:
    rainfall_mean: float = 0.0
    wind_mean: float = 0.0
    blocked_ratio: float = 0.0
    blocked_ratio_stochastic: float = 0.0
    blocked_ratio_forced_island: float = 0.0
    blocked_ratio_total: float = 0.0
    blockage_target_ratio: float = 0.0
    blockage_gap: float = 0.0
    blockage_global_gate: float = 0.0
    risk_spike: bool = False
    epicenter_node: Optional[int] = None


@dataclass
class JointState:
    step_index: int
    agents: Dict[str, AgentRuntimeState] = field(default_factory=dict)
    tasks: Dict[str, DeliveryTask] = field(default_factory=dict)
    hazard: HazardSnapshot = field(default_factory=HazardSnapshot)
    done: bool = False


@dataclass(frozen=True)
class TruckAction:
    # For graph-constrained movement.
    target_node: Optional[int] = None
    stay: bool = False


@dataclass(frozen=True)
class UAVAction:
    # Continuous control + optional mode command.
    vx: float = 0.0
    vy: float = 0.0
    bind_truck_id: Optional[str] = None
    takeoff: bool = False


Action = Union[TruckAction, UAVAction]
JointAction = Mapping[str, Action]


@dataclass
class StepResult:
    state: JointState
    rewards: Dict[str, float]
    terminated: bool
    truncated: bool
    info: Dict[str, object]


class HeteroDisasterMDP(ABC):
    """
    Low-level MDP + high-level SMDP trigger interface.
    """

    cfg: EnvConfig

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> JointState:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: JointAction) -> StepResult:
        raise NotImplementedError

    @abstractmethod
    def observe(self) -> Dict[str, List[float]]:
        """Per-agent observation vectors."""
        raise NotImplementedError

    @abstractmethod
    def observe_task_matrix(self) -> Dict[str, List[List[float]]]:
        """
        Per-agent task feature matrix with fixed slots:
        shape = [task_attention_slots, task_feat_dim]
        """
        raise NotImplementedError

    @abstractmethod
    def legal_actions(self) -> Dict[str, object]:
        """Per-agent legality descriptor or mask."""
        raise NotImplementedError

    @abstractmethod
    def should_trigger_hrl(self) -> bool:
        """SMDP decision trigger: interval or risk spike."""
        raise NotImplementedError





























