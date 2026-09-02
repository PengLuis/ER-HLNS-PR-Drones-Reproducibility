from __future__ import annotations


def init_planner_runtime_state(planner) -> None:
            # Diagnostics for eval scripts.
            planner.goal_switch_count_total: int = 0
            planner.goal_switch_candidate_count_total: int = 0
            planner.goal_switch_accepted_count_total: int = 0
            planner.goal_switch_rejected_by_threshold_count_total: int = 0
            planner.goal_switch_forced_count_total: int = 0
            planner.goal_switch_accepted_by_score_count_total: int = 0
            planner.goal_switch_accepted_by_eta_count_total: int = 0
            planner.goal_switch_forced_reason_completed_total: int = 0
            planner.goal_switch_forced_reason_failed_total: int = 0
            planner.goal_switch_forced_reason_infeasible_total: int = 0
            planner.goal_switch_forced_reason_dead_end_total: int = 0
            planner.goal_switch_forced_reason_uav_recovery_total: int = 0
            planner.goal_switch_forced_reason_stall_total: int = 0
            planner.cluster_primary_reject_count_total: int = 0
            planner.cluster_primary_switch_count_total: int = 0
            planner.same_task_cooldown_reject_count_total: int = 0
            planner._last_switch_decision_by_aid: Dict[str, Dict[str, str]] = {}
            planner._last_seen_step: int = -1
            planner.last_replan_reason: str = "init"
            planner.last_assignment_summary: Dict[str, int] = {"assigned_total": 0, "assigned_truck": 0, "assigned_uav": 0}
            planner._last_refresh_flags: Dict[str, bool] = {}
            # Refresh diagnostics (episode-level).
            planner.refresh_total_count: int = 0
            planner.fixed_interval_refresh_count: int = 0
            planner.event_refresh_count: int = 0
            planner.no_event_fallback_refresh_count: int = 0
            planner.initial_refresh_count: int = 0
            planner.empty_goal_refresh_count: int = 0
            planner.steps_since_last_refresh_sum: int = 0
            planner.steps_since_last_refresh_max: int = 0
            planner._episode_first_refresh_done: bool = False
            # Event-refresh diagnostics: reason distribution.
            planner.event_refresh_reason_arrival_count_total: int = 0
            planner.event_refresh_reason_resolution_count_total: int = 0
            planner.event_refresh_reason_uav_idle_count_total: int = 0
            planner.event_refresh_reason_truck_idle_count_total: int = 0
            planner.event_refresh_reason_map_update_light_count_total: int = 0
            planner.event_refresh_reason_map_update_hard_count_total: int = 0
            planner.event_refresh_reason_goal_invalid_count_total: int = 0
            planner.event_refresh_reason_path_blocked_count_total: int = 0
            planner.event_refresh_reason_goal_unreachable_count_total: int = 0
            planner.event_refresh_reason_uav_safety_count_total: int = 0
            planner.event_refresh_reason_truck_dead_end_count_total: int = 0
            planner.event_refresh_reason_high_priority_uncovered_count_total: int = 0
            planner.event_refresh_reason_normal_stall_count_total: int = 0
            # Event-refresh diagnostics: outcome/value chain.
            planner.event_refresh_no_goal_change_count_total: int = 0
            planner.event_refresh_goal_change_count_total: int = 0
            planner.event_refresh_to_launch_count_total: int = 0
            planner.event_refresh_to_completion_count_total: int = 0
            planner.event_refresh_followed_by_reject_count_total: int = 0
            planner.event_refresh_followed_by_stall_count_total: int = 0
            # Hard-event attribution diagnostics.
            planner.hard_event_refresh_count_total: int = 0
            planner.hard_event_reason_goal_invalid_count_total: int = 0
            planner.hard_event_reason_current_goal_unreachable_count_total: int = 0
            planner.hard_event_reason_path_blocked_count_total: int = 0
            planner.hard_event_reason_uav_safety_count_total: int = 0
            planner.hard_event_reason_uav_recovery_count_total: int = 0
            planner.hard_event_reason_truck_dead_end_count_total: int = 0
            planner.hard_event_reason_high_priority_uncovered_count_total: int = 0
            planner.hard_event_reason_normal_stall_count_total: int = 0
            planner.hard_event_reason_assigned_but_not_progressing_count_total: int = 0
            planner.hard_event_reason_goal_completed_count_total: int = 0
            planner.hard_event_reason_goal_failed_count_total: int = 0
            planner.goal_invalid_reason_task_completed_total: int = 0
            planner.goal_invalid_reason_task_failed_total: int = 0
            planner.goal_invalid_reason_task_missing_total: int = 0
            planner.goal_invalid_reason_truck_unreachable_total: int = 0
            planner.goal_invalid_reason_uav_energy_infeasible_total: int = 0
            planner.goal_invalid_reason_uav_recovery_margin_total: int = 0
            planner.goal_invalid_reason_uav_corridor_total: int = 0
            planner.goal_invalid_reason_uav_comm_block_total: int = 0
            planner.goal_invalid_reason_uav_not_loaded_total: int = 0
            planner.goal_invalid_reason_uav_not_docked_total: int = 0
            planner.goal_invalid_reason_soft_reject_cache_total: int = 0
            planner.suspect_soft_as_hard_count_total: int = 0
            planner.hard_event_refresh_no_goal_change_count_total: int = 0
            planner.hard_event_refresh_goal_change_count_total: int = 0
            planner.hard_event_refresh_to_launch_count_total: int = 0
            planner.hard_event_refresh_to_completion_count_total: int = 0
            planner.hard_event_refresh_followed_by_reject_count_total: int = 0
            planner.hard_event_refresh_followed_by_stall_count_total: int = 0
            planner.normal_stall_candidate_count_total: int = 0
            planner.normal_stall_blocked_by_persist_count_total: int = 0
            planner.normal_stall_blocked_by_cooldown_count_total: int = 0
            planner.normal_stall_local_correction_count_total: int = 0
            planner.normal_stall_global_refresh_count_total: int = 0
            planner.goal_invalid_hard_count_total: int = 0
            planner.goal_invalid_soft_count_total: int = 0
            planner.goal_invalid_soft_suppressed_count_total: int = 0
            planner.goal_invalid_soft_escalated_count_total: int = 0
            planner.uav_recovery_hard_count_total: int = 0
            planner.uav_recovery_soft_count_total: int = 0
            planner.uav_recovery_soft_suppressed_count_total: int = 0
            planner.uav_recovery_local_action_count_total: int = 0
            planner.uav_recovery_global_refresh_count_total: int = 0
            planner.truck_dead_end_candidate_count_total: int = 0
            planner.truck_dead_end_blocked_by_persist_count_total: int = 0
            planner.truck_dead_end_blocked_by_cooldown_count_total: int = 0
            planner.truck_dead_end_local_path_repair_count_total: int = 0
            planner.truck_dead_end_local_goal_reassign_count_total: int = 0
            planner.truck_dead_end_global_refresh_count_total: int = 0
            planner.truck_dead_end_noop_count_total: int = 0
            planner.truck_dead_end_routine_localized_count_total: int = 0
            planner.truck_dead_end_emergency_kept_hard_count_total: int = 0
            planner.truck_dead_end_support_kept_hard_count_total: int = 0
            planner.truck_dead_end_recovery_kept_hard_count_total: int = 0
            planner.truck_dead_end_local_repair_no_goal_change_count_total: int = 0
            planner.truck_dead_end_global_refresh_no_goal_change_count_total: int = 0
            planner.path_blocked_candidate_count_total: int = 0
            planner.path_blocked_nonimpact_suppressed_count_total: int = 0
            planner.path_blocked_impacted_current_path_count_total: int = 0
            planner.path_blocked_impacted_goal_reachability_count_total: int = 0
            planner.path_blocked_impacted_recovery_count_total: int = 0
            planner.path_blocked_local_path_repair_count_total: int = 0
            planner.path_blocked_local_goal_reassign_count_total: int = 0
            planner.path_blocked_global_refresh_count_total: int = 0
            planner.path_blocked_noop_count_total: int = 0
            planner.path_blocked_routine_localized_count_total: int = 0
            planner.path_blocked_emergency_kept_hard_count_total: int = 0
            planner.path_blocked_recovery_kept_hard_count_total: int = 0
            planner.path_blocked_support_kept_hard_count_total: int = 0
            planner.path_blocked_goal_unreachable_kept_hard_count_total: int = 0
            planner.path_blocked_local_repair_no_goal_change_count_total: int = 0
            planner.path_blocked_global_refresh_no_goal_change_count_total: int = 0
            # Execution-commitment + event-evidence gate diagnostics.
            planner.erc_event_detected_count_total: int = 0
            planner.erc_event_gate_pass_count_total: int = 0
            planner.erc_event_gate_reject_count_total: int = 0
            planner.erc_local_correction_count_total: int = 0
            planner.erc_global_replan_count_total: int = 0
            planner.committed_goal_hold_count_total: int = 0
            planner.committed_goal_broken_count_total: int = 0
            planner.committed_goal_broken_reason_hard_invalid_count_total: int = 0
            planner.committed_goal_broken_reason_stall_count_total: int = 0
            planner.committed_goal_broken_reason_tc_gain_count_total: int = 0
            planner.airborne_uav_goal_lock_count_total: int = 0
            planner.path_blocked_local_agent_count_total: int = 0
            planner.high_priority_event_rejected_no_launchable_uav_count_total: int = 0
            # Lightweight commitment state snapshot.
            planner.committed_goal: Dict[str, Optional[str]] = {}
            planner.committed_goal_step: Dict[str, int] = {}
            planner.committed_goal_progress: Dict[str, float] = {}
            planner.committed_goal_status: Dict[str, str] = {}
            planner.committed_goal_feasibility: Dict[str, int] = {}
            planner._last_goal_invalid_record: Optional[Dict[str, object]] = None
            planner._last_uav_emergency_record: Optional[Dict[str, object]] = None
            planner._last_truck_dead_end_record: Optional[Dict[str, object]] = None
            planner._last_high_priority_uncovered_record: Optional[Dict[str, object]] = None
            planner._last_normal_stall_record: Optional[Dict[str, object]] = None
            planner._last_goal_terminal_status: str = ""
            planner._last_hard_event_offenders: List[Dict[str, object]] = []
            planner._hard_event_offender_stats: Dict[Tuple[str, str, str], Dict[str, float]] = {}
            planner._hard_event_reason_outcome_stats: Dict[str, Dict[str, float]] = {}
            planner._normal_stall_cooldown_until_by_truck: Dict[str, int] = {}
            planner._truck_dead_end_persist_by_truck: Dict[str, int] = {}
            planner._truck_dead_end_cooldown_until_by_truck: Dict[str, int] = {}
            planner._debug_truck_dead_end_localize_by_truck: Dict[str, int] = {}
            planner._debug_truck_path_blocked_localize_by_truck: Dict[str, int] = {}
            planner._soft_invalid_repeat: Dict[Tuple[str, str, str], int] = {}
            planner._soft_invalid_cooldown_until: Dict[Tuple[str, str, str], int] = {}
            # No-op weak-event cooldown state.
            planner._noop_event_cooldown_until_step_by_reason: Dict[str, int] = {}
            planner._active_event_refresh_window: Optional[Dict[str, float]] = None
            planner._last_map_update_impacted_count: int = 0
            planner._last_map_update_critical_count: int = 0
            planner._event_window_start_step: int = 0
            planner._event_replans_in_window: int = 0
            planner._low_value_event_streak: int = 0
            planner._normal_pending_unchanged_steps: int = 0
            planner._last_pending_normal_count: int = -1
            planner._delivery_stall_steps: int = 0
            planner._last_delivered_count: int = -1
            planner._last_island_serviceability: float = 1.0
            planner.map_update_hard_seen_count_total: int = 0
            planner.map_update_hard_actionable_count_total: int = 0
            planner.map_update_hard_deferred_count_total: int = 0
            planner.map_update_hard_immediate_refresh_count_total: int = 0
            planner._map_update_hard_actionable_reasons_total: Dict[str, int] = {
                "path_blocked": 0,
                "goal_unreachable": 0,
                "ranking_changed": 0,
                "dead_end": 0,
                "recovery_path_fractured": 0,
            }
            # Soft reservation to reduce UAV task contention and short-horizon
            # ping-pong assignment under frequent replans.
            planner._uav_task_reservations: Dict[str, Tuple[str, int]] = {}
            # Track continuous docked-follow steps to detect ride-stall.
            planner._uav_docked_steps: Dict[str, int] = {}
            # Round4 diagnostics and lightweight commitment state.
            planner.uav_recovery_feasibility_eval_count_total: int = 0
            planner.unique_agent_task_candidate_count_total: int = 0
            planner.truck_support_selected_count_total: int = 0
            planner.truck_support_improves_serviceability_count_total: int = 0
            planner.truck_support_no_gain_count_total: int = 0
            # Unified support event counters (alias to truck_support_* for stable export).
            planner.support_selected_count_total: int = 0
            planner.support_improves_serviceability_count_total: int = 0
            planner.support_no_gain_count_total: int = 0
            planner._uav_task_feasible_cache_step: int = -1
            planner._uav_task_feasible_cache: Dict[Tuple[int, str, str, int, Tuple[Tuple[str, int], ...]], bool] = {}
            planner._step_unique_agent_task_keys: set = set()
            planner._support_anchor_until_step: Dict[str, int] = {}
            planner._support_anchor_gain: Dict[str, float] = {}
            planner._support_anchor_task_id: Dict[str, str] = {}
            planner._support_bound_chain_until_step: Dict[str, int] = {}
            planner._support_bound_chain_task_id: Dict[str, str] = {}
            planner._support_bound_chain_uav_id: Dict[str, str] = {}
            planner._support_bound_chain_truck_by_uav: Dict[str, str] = {}
            planner._support_bound_chain_anchor_node_by_truck: Dict[str, int] = {}
            planner._support_bound_chain_latest_start_by_truck: Dict[str, int] = {}
            planner._tc_support_chain_class: Dict[str, str] = {}
            # Region-aware commitment: fixed episode-level task regions and
            # truck/UAV home regions for large spatially clustered maps.
            planner._region_commitment_signature: Optional[Tuple[int, int, int, float]] = None
            planner._region_centers_xy: Dict[int, Tuple[float, float]] = {}
            planner._task_region_by_task: Dict[str, int] = {}
            planner._region_task_distance_m: Dict[str, float] = {}
            planner._region_outlier_task_ids: set = set()
            planner._region_commitment_effective_k: int = 0
            planner._region_commitment_enabled_effective: bool = False
            planner._region_commitment_auto_score: float = 0.0
            planner._region_commitment_separation_score: float = 0.0
            planner._region_commitment_load_balance_score: float = 0.0
            planner._region_commitment_coverage_score: float = 0.0
            planner._region_commitment_strength: float = 0.0
            planner._agent_home_region: Dict[str, int] = {}
            planner.region_commitment_setup_count_total: int = 0
            planner.region_commitment_local_candidate_count_total: int = 0
            planner.region_commitment_cross_filtered_count_total: int = 0
            planner.region_commitment_cross_override_count_total: int = 0
            planner.region_commitment_outlier_task_count_total: int = 0
            planner.region_commitment_outlier_filtered_count_total: int = 0
            planner.region_commitment_outlier_override_count_total: int = 0
            planner.region_commitment_auto_enabled_count_total: int = 0
            planner.region_commitment_auto_disabled_count_total: int = 0
            planner.tc_direct_feasible_count_total: int = 0
            planner.tc_support_required_count_total: int = 0
            planner.tc_truly_infeasible_count_total: int = 0
            planner.tc_support_lock_created_count_total: int = 0
            planner.tc_support_lock_to_dispatch_count_total: int = 0
            planner._relaxed_chain_until_step: Dict[str, int] = {}
            # Step-0 directional coverage plan (truck sectors + UAV ride routes).
            planner._initial_directional_plan_step: int = -1
            planner._initial_directional_truck_sector: Dict[str, int] = {}
            planner._initial_directional_uav_truck: Dict[str, str] = {}
            planner._initial_directional_sector_stats: Dict[int, Dict[str, float]] = {}
            planner._uav_anchor_task_goal: Dict[str, str] = {}
            planner._uav_transfer_target_truck: Dict[str, str] = {}
            planner._uav_transfer_target_task: Dict[str, str] = {}
            planner.uav_transfer_hint_issue_count_total: int = 0
            planner.uav_transfer_hint_keep_count_total: int = 0
            # Per-step snapshot cache: truck NORMAL reachability context.
            planner._normal_reachability_cache_step: int = -1
            planner._normal_reachability_cache: Optional[Tuple[int, Dict[str, bool], bool]] = None
            planner._planner_eval_cache_step: int = -1
            planner._truck_task_distance_cache: Dict[Tuple[str, str], float] = {}
            planner._truck_task_serviceable_cache: Dict[Tuple[str, str], bool] = {}
            planner._truck_nearest_reachable_cache: Dict[str, float] = {}
            planner._support_anchor_gain_cache: Dict[Tuple[str, str], Dict[str, float]] = {}
            planner._truck_support_candidate_cache: Dict[Tuple[str, str], bool] = {}
            planner._task_high_pressure_cache: Dict[str, bool] = {}
            # Truck NORMAL task anti-ABA memory (to reduce short-horizon back-and-forth).
            planner._truck_recent_normal_prev_goal: Dict[str, str] = {}
            planner._truck_recent_normal_switch_step: Dict[str, int] = {}
            # UAV truck-anchor anti-ABA memory (reduce A<->B truck-follow ping-pong).
            planner._uav_recent_truck_anchor_prev_goal: Dict[str, str] = {}
            planner._uav_recent_truck_anchor_switch_step: Dict[str, int] = {}
            planner._task_recent_prev_goal: Dict[str, str] = {}
            planner._task_recent_switch_step: Dict[str, int] = {}
            planner.task_aba_switch_blocked_count_total: int = 0
            planner.comm_blackout_commit_hold_count_total: int = 0
            planner._normal_task_unreachable_streak: Dict[str, int] = {}
            planner.normal_unreachable_task_count_total: int = 0
            # Lifeline-aware time-critical priority diagnostics.
            planner.timecritical_tier3_candidate_count_total: int = 0
            planner.timecritical_tier3_selected_count_total: int = 0
            planner.timecritical_tier2_candidate_count_total: int = 0
            planner.timecritical_tier2_selected_count_total: int = 0
            planner.timecritical_candidate_ignored_count_total: int = 0
            # Support binding diagnostics.
            planner.support_selected_with_bound_timecritical_delivery_count_total: int = 0
            planner.support_selected_without_bound_timecritical_delivery_count_total: int = 0
            planner.support_filtered_no_bound_timecritical_delivery_count_total: int = 0
            planner.support_selected_with_bound_bulk_delivery_count_total: int = 0
            planner.support_bound_dispatch_count_total: int = 0
            planner.support_bound_recovery_redirect_count_total: int = 0
            planner.support_no_gain_backoff_block_count_total: int = 0
            planner.support_proxy_candidate_count_total: int = 0
            planner.support_relay_reserved_count_total: int = 0
            planner.truck_emergency_goal_assigned_count_total: int = 0
            planner._support_relay_force_step: Dict[str, int] = {}
            planner._support_no_gain_streak: Dict[str, int] = {}
            planner._support_backoff_until_step: Dict[str, int] = {}
            # Per-task goal exposure memory prevents large-map time-critical tasks
            # from never entering the goal chain.
            planner._task_last_goal_step: Dict[str, int] = {}
            planner._task_goal_exposure_count: Dict[str, int] = {}
            # Reject-cache diagnostics for repeated infeasible UAV-task reselection.
            planner.uav_reject_cache_hit_count_total: int = 0
            planner.uav_reject_cache_insert_count_total: int = 0
            planner.uav_reject_cache_clear_count_total: int = 0
            planner.uav_reject_cache_reason_insufficient_recovery_margin_total: int = 0
            planner.uav_reject_cache_reason_corridor_total: int = 0
            planner.uav_reject_cache_reason_comm_block_total: int = 0
            planner.uav_reject_cache_reason_energy_infeasible_total: int = 0
            planner.uav_reject_cache_reason_no_recovery_total: int = 0
            planner.uav_task_selected_count_total: int = 0
            planner._uav_reject_cache: Dict[Tuple[str, str, str], Dict[str, object]] = {}
            planner._uav_reject_state_sig: Dict[str, Tuple[float, str, int, int]] = {}
            # Ablation wiring audit counters.
            planner.low_value_refresh_candidate_count_total: int = 0
            planner.low_value_refresh_allowed_count_total: int = 0
            planner.low_value_refresh_blocked_by_ablation_count_total: int = 0
            planner.map_ranking_refresh_candidate_count_total: int = 0
            planner.map_ranking_refresh_allowed_count_total: int = 0
            planner.map_ranking_refresh_blocked_by_ablation_count_total: int = 0
            planner.tc_global_assignment_called_count_total: int = 0
            planner.tc_global_assignment_skipped_by_ablation_count_total: int = 0
            planner.tc_assignment_epoch_applied_count_total: int = 0
            planner.support_chain_candidate_count_total: int = 0
            planner.support_chain_applied_count_total: int = 0
            planner.support_chain_blocked_by_ablation_count_total: int = 0
            planner.cluster_primary_candidate_count_total: int = 0
            planner.cluster_primary_applied_count_total: int = 0
            planner.cluster_primary_blocked_by_ablation_count_total: int = 0
            planner.task_reservation_applied_count_total: int = 0
            planner.task_reservation_blocked_by_ablation_count_total: int = 0
            planner.event_scoring_bonus_applied_count_total: int = 0
            planner.event_scoring_bonus_blocked_by_ablation_count_total: int = 0
            planner.normal_protection_candidate_count_total: int = 0
            planner.normal_protection_applied_count_total: int = 0
            planner.normal_protection_blocked_by_ablation_count_total: int = 0
            # Progress-aware switch control diagnostics.
            planner.uav_emergency_commit_hold_count_total: int = 0
            planner.uav_emergency_commit_break_hard_invalid_count_total: int = 0
            planner.uav_emergency_commit_prevented_switch_count_total: int = 0
            planner.uav_emergency_commit_followed_by_launch_count_total: int = 0
            planner.uav_emergency_commit_followed_by_delivery_count_total: int = 0
            planner.truck_routine_stuck_candidate_count_total: int = 0
            planner.truck_routine_stuck_escape_count_total: int = 0
            planner.truck_routine_stuck_escape_blocked_no_alt_count_total: int = 0
            planner.truck_routine_stuck_escape_blocked_insufficient_gain_count_total: int = 0
            planner.truck_routine_stuck_escape_followed_by_service_count_total: int = 0
            planner.truck_routine_stuck_escape_followed_by_completion_count_total: int = 0
            planner.routine_localize_eta_check_count_total: int = 0
            planner.routine_localize_keep_current_count_total: int = 0
            planner.routine_localize_escape_by_eta_worse_count_total: int = 0
            planner.routine_localize_escape_followed_by_service_count_total: int = 0
            planner.routine_localize_escape_followed_by_completion_count_total: int = 0
            # UAV reservation + airborne commitment diagnostics.
            planner.uav_task_reserved_count_total: int = 0
            planner.uav_task_reservation_release_count_total: int = 0
            planner.uav_task_airborne_committed_count_total: int = 0
            planner.uav_task_reserved_to_launch_count_total: int = 0
            planner.uav_task_reserved_to_completion_count_total: int = 0
            planner.uav_task_reservation_stale_count_total: int = 0
            planner.uav_airborne_goal_switch_blocked_count_total: int = 0
            planner.uav_airborne_safety_abort_count_total: int = 0
            planner.uav_airborne_task_completed_count_total: int = 0
            # Truck assist-waypoint diagnostics.
            planner.truck_uav_assist_candidate_count_total: int = 0
            planner.truck_uav_assist_accepted_count_total: int = 0
            planner.truck_uav_assist_rejected_extra_distance_count_total: int = 0
            planner.truck_uav_assist_rejected_normal_service_count_total: int = 0
            planner.truck_uav_assist_launch_success_count_total: int = 0
            planner.truck_uav_assist_followed_by_emergency_completion_count_total: int = 0
            planner.truck_uav_assist_extra_distance_m_total: float = 0.0
            # Internal reservation/intent/assist state.
            planner._uav_task_reservation_state_by_task: Dict[str, Dict[str, Any]] = {}
            planner._uav_task_reservation_by_uav: Dict[str, str] = {}
            # Global task contracts: every pending delivery task has at most
            # one visible owner across all trucks and UAVs.
            planner._task_contract_by_task: Dict[str, str] = {}
            planner._task_contract_by_agent: Dict[str, str] = {}
            planner._uav_intent_signal_by_uav: Dict[str, Dict[str, Any]] = {}
            planner._truck_assist_waypoint_by_truck: Dict[str, Dict[str, Any]] = {}
            planner._truck_assist_pending_windows: List[Dict[str, Any]] = []
            planner._far_routine_bootstrap_force_step: Dict[str, int] = {}
            planner.harmful_switch_proxy_count_total: int = 0
            planner.missed_switch_proxy_count_total: int = 0
            planner._uav_commit_hold_pending: List[Dict[str, Any]] = []
            planner._truck_routine_escape_pending: List[Dict[str, Any]] = []
            planner._routine_localize_escape_pending: List[Dict[str, Any]] = []
            # Switch-decision audit ledger (diagnostics only; no behavior impact).
            planner._switch_decision_ledger_rows: List[Dict[str, Any]] = []
            planner._switch_decision_pending_windows: List[Dict[str, Any]] = []
            planner._switch_decision_seq: int = 0
            planner._switch_goal_distance_history_by_agent: Dict[str, List[Tuple[int, str, float]]] = {}
