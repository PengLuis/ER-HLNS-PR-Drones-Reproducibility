from __future__ import annotations

from typing import Dict, Optional

from hetgat_hrl.core.mdp_spec import AgentKind, TaskKind, TaskStatus
from hetgat_hrl.hrl_v2.command_layer import (
    CommandBatch,
    TruckCommand,
    TruckCommandKind,
    UAVCommand,
    UAVCommandKind,
)
from hetgat_hrl.hrl_v2.commitment_manager import Commitment, CommitmentManager
from hetgat_hrl.hrl_v2.evidence_monitor import EvidenceMonitor
from hetgat_hrl.hrl_v2.event_gate import EventGate
from hetgat_hrl.hrl_v2.feasibility_oracle import FeasibilityOracle
from hetgat_hrl.hrl_v2.objective_model import ObjectiveModel
from hetgat_hrl.hrl_v2.world_state import WorldState


class ErcRhcV2Planner:
    """Command-first ERC-RHC v2 planner.

    The first implementation deliberately keeps the planning surface compact:
    it produces an explicit CommandBatch for command-gated execution and returns
    the compatible high-level goals dict consumed by the existing low-level
    policy. This lets us test v2 invariants before replacing the full low-level
    action stack.
    """

    def __init__(
        self,
        decision_interval: int = 5,
        seed: int = 0,
        support_recovery_repair: bool = False,
        gate16_command_quality: bool = False,
        gate17_lifecycle_closure: bool = False,
        gate18_variant: str = "",
        gate19_variant: str = "",
        gate20_variant: str = "",
    ):
        self.decision_interval = int(max(decision_interval, 1))
        self.seed = int(seed)
        self.support_recovery_repair = bool(support_recovery_repair)
        self.gate16_command_quality = bool(gate16_command_quality)
        self.gate17_lifecycle_closure = bool(gate17_lifecycle_closure)
        self.gate18_variant = str(gate18_variant or "").strip().lower()
        self.gate19_variant = str(gate19_variant or "").strip().lower()
        self.gate20_variant = str(gate20_variant or "").strip().lower()
        self.gate19_launch_binding = self.gate19_variant in {"launch_binding", "gate19_core"}
        self.gate19_support_lock = self.gate19_variant in {"support_lock", "gate19_core"}
        self.gate20_passenger_invariant = self.gate20_variant in {"passenger_invariant", "gate20_core"}
        self.gate20_support_reserve_launch = self.gate20_variant in {"support_reserve_launch", "gate20_core"}
        self.gate20_rebind_at_anchor = self.gate20_variant == "rebind_at_anchor"
        self.gate20_workflow_repair = self.gate20_variant == "workflow_repair"
        self.gate112_quality_audit = self.gate20_variant == "support_quality_audit"
        self.gate112_quality_gate = self.gate20_variant in {"support_quality_gate", "support_quality_relaxed"}
        self.gate112_quality_relaxed = self.gate20_variant == "support_quality_relaxed"
        if self.gate20_variant:
            self.gate19_launch_binding = True
            self.gate19_support_lock = self.gate19_support_lock or self.gate20_support_reserve_launch or self.gate20_passenger_invariant or self.gate20_workflow_repair
            if self.gate112_quality_audit or self.gate112_quality_gate:
                self.gate19_support_lock = True
        self.gate18_direct_tc_first = (
            self.gate18_variant in {"direct_tc_first", "gate18_core"}
            or self.gate19_variant == "gate19_core"
            or self.gate20_workflow_repair
            or bool(self.gate20_variant)
            or self.gate112_quality_gate
        )
        self.gate18_support_anchor_strict = self.gate18_variant in {"support_anchor_strict", "gate18_core"}
        self.gate18_anchor_arrival_force_launch = self.gate18_variant in {"anchor_arrival_force_launch", "gate18_core"}
        self.gate18_safety_narrow_routine_guard = self.gate18_variant in {"safety_narrow_routine_guard", "gate18_core"}
        self.v2_support_anchor_arrival_radius_m = 120.0
        self.commitments = CommitmentManager()
        self.objective = ObjectiveModel()
        self.monitor = EvidenceMonitor()
        self.event_gate = EventGate()
        self._last_batch = CommandBatch()
        self._active_support_commitments: Dict[str, Dict[str, object]] = {}
        self._active_support_commands: Dict[str, Dict[str, object]] = {}
        self._active_safety_recovery_commands: Dict[str, Dict[str, object]] = {}
        self._support_to_sortie: Dict[str, str] = {}
        self._sortie_to_support: Dict[str, str] = {}
        self._support_cooldown_until: Dict[str, int] = {}
        self._support_launch_marked = set()
        self._support_delivery_marked = set()
        self._delivery_without_sortie_marked = set()
        self._duplicate_delivery_marked = set()
        self._next_support_command_seq = 0
        self._next_safety_recovery_seq = 0
        self._next_sortie_seq = 0
        self._active_sorties: Dict[str, Dict[str, object]] = {}
        self._delivered_sortie_ids = set()
        self._delivered_task_ids_by_sortie = set()
        self._sortie_lifecycle_rows = []
        self._support_lifecycle_rows = []
        self._safety_lifecycle_rows = []
        self._support_passenger_trace_rows = []
        self._support_quality_rows = []
        self._init_public_counters()

    def _init_public_counters(self) -> None:
        names = [
            "unauthorized_support_blocked_count",
            "support_command_count",
            "support_command_to_launch_count",
            "support_command_to_delivery_count",
            "safety_recovery_command_count",
            "sortie_candidate_count",
            "full_sortie_feasible_count",
            "sortie_blocked_recovery_count",
            "sortie_blocked_energy_count",
            "sortie_blocked_lifeline_count",
            "sortie_to_launch_count",
            "sortie_to_delivery_count",
            "sortie_prediction_mismatch_count",
            "tc_residual_followup_candidate_count",
            "tc_residual_followup_commitment_count",
            "tc_residual_followup_to_delivery_count",
            "routine_commitment_count",
            "routine_commitment_to_completion_count",
            "event_detected_count",
            "event_impact_positive_count",
            "event_impact_none_count",
            "local_command_generated_count",
            "global_replan_count",
            "weak_event_suppressed_count",
            "forced_switch_blocked_count",
            "selected_reason_safety",
            "selected_reason_tc_delivery",
            "selected_reason_routine_completion",
            "selected_reason_continue_commitment",
            "selected_reason_low_cost",
            "goal_switch_count_total",
            "goal_switch_candidate_count_total",
            "goal_switch_accepted_count_total",
            "goal_switch_rejected_by_threshold_count_total",
            "goal_switch_forced_count_total",
            "truck_command_count",
            "truck_go_to_routine_count",
            "truck_continue_routine_count",
            "truck_service_routine_count",
            "truck_support_uav_count",
            "truck_safety_recovery_count",
            "truck_hold_count",
            "uav_command_count",
            "uav_bind_to_truck_count",
            "uav_prepare_tc_count",
            "uav_launch_tc_count",
            "uav_continue_sortie_count",
            "uav_return_to_anchor_count",
            "uav_hold_count",
            "v2_support_command_candidate_count",
            "v2_support_command_generated_count",
            "v2_support_command_to_launch_count",
            "v2_support_command_to_delivery_count",
            "v2_support_command_expired_count",
            "v2_support_command_aborted_no_benefit_count",
            "v2_support_command_blocked_not_full_sortie_feasible_count",
            "v2_support_command_blocked_routine_delay_count",
            "v2_support_command_blocked_no_loaded_uav_count",
            "v2_support_command_blocked_no_anchor_count",
            "v2_safety_recovery_command_candidate_count",
            "v2_safety_recovery_command_generated_count",
            "v2_safety_recovery_to_recovered_count",
            "v2_safety_recovery_failed_count",
            "sortie_oracle_candidate_count",
            "sortie_oracle_full_feasible_count",
            "sortie_oracle_to_launch_count",
            "sortie_oracle_to_delivery_count",
            "sortie_oracle_predicted_feasible_but_no_delivery_count",
            "sortie_oracle_predicted_feasible_but_forced_recovery_count",
            "sortie_oracle_prediction_mismatch_count",
            "sortie_oracle_blocked_recovery_count",
            "sortie_oracle_blocked_energy_count",
            "sortie_oracle_blocked_lifeline_count",
            "sortie_id_created_count",
            "sortie_launch_success_count",
            "sortie_delivery_success_count",
            "sortie_duplicate_delivery_blocked_count",
            "delivery_without_sortie_blocked_count",
            "delivery_task_already_delivered_blocked_count",
            "launch_to_completion_ratio_raw",
            "launch_to_completion_ratio_checked",
            "support_command_created_count",
            "support_command_truck_arrived_anchor_count",
            "support_command_launch_triggered_count",
            "support_command_delivery_completed_count",
            "support_command_expired_count",
            "support_command_expired_before_anchor_count",
            "support_command_expired_before_launch_count",
            "support_command_aborted_no_benefit_count",
            "support_allowed_but_no_delivery_count",
            "support_command_blocked_routine_commitment_count",
            "support_command_failed_count",
            "support_command_blocked_no_loaded_uav_count",
            "support_command_blocked_not_full_sortie_feasible_count",
            "support_command_blocked_low_recovery_margin_count",
            "support_command_blocked_lifeline_risk_count",
            "support_command_blocked_routine_delay_count",
            "support_command_blocked_active_limit_count",
            "support_command_blocked_cooldown_count",
            "support_blocked_by_routine_commitment_count",
            "support_allowed_despite_routine_commitment_count",
            "support_allowed_routine_delay_steps_sum",
            "support_allowed_to_delivery_count",
            "safety_recovery_candidate_count",
            "safety_recovery_command_created_count",
            "safety_recovery_anchor_assigned_count",
            "safety_recovery_uav_returning_count",
            "safety_recovery_command_generated_count",
            "safety_recovery_to_recovered_count",
            "safety_recovery_expired_count",
            "safety_recovery_failed_count",
            "safety_recovery_recovered_without_command_count",
            "safety_recovery_blocked_no_anchor_count",
            "safety_recovery_blocked_routine_commitment_count",
            "safety_recovery_blocked_not_hard_risk_count",
            "direct_tc_candidate_count",
            "direct_tc_launch_generated_count",
            "direct_tc_to_delivery_count",
            "support_skipped_due_to_direct_tc_count",
            "direct_tc_launch_no_delivery_count",
            "strict_anchor_candidate_count",
            "strict_anchor_selected_count",
            "strict_anchor_blocked_eta_count",
            "strict_anchor_blocked_lifeline_count",
            "strict_anchor_blocked_margin_count",
            "strict_anchor_blocked_routine_delay_count",
            "strict_anchor_to_arrival_count",
            "strict_anchor_to_launch_count",
            "strict_anchor_to_delivery_count",
            "anchor_arrival_recheck_count",
            "anchor_arrival_launch_forced_count",
            "anchor_arrival_release_infeasible_count",
            "anchor_arrival_release_reason_recovery_count",
            "anchor_arrival_release_reason_energy_count",
            "anchor_arrival_release_reason_lifeline_count",
            "anchor_arrival_to_delivery_count",
            "safety_narrow_candidate_count",
            "safety_narrow_generated_count",
            "safety_narrow_suppressed_soft_risk_count",
            "safety_narrow_interrupted_active_sortie_count",
            "safety_narrow_beneficial_recovery_count",
            "routine_guard_blocked_support_count",
            "routine_guard_allowed_support_count",
            "routine_guard_allowed_to_delivery_count",
            "routine_guard_allowed_no_delivery_count",
            "support_arrived_by_exact_node_count",
            "support_arrived_by_radius_count",
            "support_arrival_missed_by_exact_node_count",
            "support_anchor_distance_at_arrival_sum",
            "support_anchor_distance_at_arrival_count",
            "support_anchor_distance_at_arrival_mean",
            "support_lock_created_count",
            "support_lock_released_count",
            "support_lock_blocked_uav_reassign_count",
            "support_lock_blocked_task_steal_count",
            "support_lock_broken_by_hard_safety_count",
            "support_lock_expired_count",
            "support_anchor_arrived_launch_attempt_count",
            "support_anchor_arrived_launch_success_count",
            "support_anchor_arrived_launch_rejected_count",
            "support_delivery_count",
            "support_binding_integrity_violation_count",
            "support_anchor_launch_reject_uav_not_on_truck_count",
            "support_anchor_launch_reject_uav_not_loaded_count",
            "support_anchor_launch_reject_uav_already_airborne_count",
            "support_anchor_launch_reject_task_expired_count",
            "support_anchor_launch_reject_task_already_delivered_or_failed_count",
            "support_anchor_launch_reject_full_sortie_infeasible_recovery_count",
            "support_anchor_launch_reject_full_sortie_infeasible_energy_count",
            "support_anchor_launch_reject_full_sortie_infeasible_lifeline_count",
            "support_anchor_launch_reject_corridor_or_comm_block_count",
            "support_anchor_launch_reject_launch_gate_env_rejected_count",
            "support_anchor_launch_reject_unknown_count",
            "passenger_invariant_check_count",
            "passenger_invariant_violation_count",
            "passenger_invariant_preserved_to_anchor_count",
            "support_failed_due_to_passenger_violation_count",
            "support_reserve_to_launch_count",
            "support_reserve_to_delivery_count",
            "support_rebind_success_count",
            "support_rebind_to_delivery_count",
            "uav_not_on_truck_at_creation_count",
            "uav_left_truck_due_to_direct_launch_count",
            "uav_left_truck_due_to_other_assignment_count",
            "uav_left_truck_due_to_recovery_count",
            "uav_follow_target_changed_count",
            "uav_docked_truck_changed_count",
            "uav_became_airborne_without_support_launch_count",
            "uav_loaded_became_false_count",
            "uav_state_sync_mismatch_count",
            "unknown_passenger_violation_count",
            "support_candidate_count",
            "support_allowed_count",
            "support_blocked_direct_better_count",
            "support_blocked_routine_better_count",
            "support_blocked_net_gain_low_count",
            "support_blocked_routine_delay_high_count",
            "support_blocked_not_full_sortie_feasible_count",
            "support_blocked_lifeline_risk_count",
            "support_blocked_recovery_margin_low_count",
            "support_blocked_no_loaded_uav_count",
            "support_blocked_no_valid_anchor_count",
            "routine_completed_after_support_block_count",
        ]
        for name in names:
            setattr(self, name, 0.0)

    def _bump(self, name: str, value: float = 1.0) -> None:
        setattr(self, name, float(getattr(self, name, 0.0)) + float(value))

    def _pending_tasks(self, env, kind: Optional[TaskKind] = None):
        tasks = [
            t
            for t in env.state.tasks.values()
            if getattr(t, "status", None) == TaskStatus.PENDING
        ]
        if kind is not None:
            tasks = [t for t in tasks if getattr(t, "kind", None) == kind]
        return tasks

    def _nearest_task(self, env, aid: str, tasks) -> Optional[object]:
        if not tasks:
            return None
        try:
            return min(tasks, key=lambda t: float(env._agent_distance_to_task(str(aid), t)))
        except Exception:
            return tasks[0]

    def _next_step_toward(self, env, src_node: int, dst_node: int) -> Optional[int]:
        neighbors = list(env._decision_neighbors(int(src_node))) if hasattr(env, "_decision_neighbors") else []
        if not neighbors:
            return None
        try:
            return int(
                min(
                    neighbors,
                    key=lambda nb: float(env._decision_shortest_path_distance(int(nb), int(dst_node))),
                )
            )
        except Exception:
            return int(neighbors[0])

    def _routine_delay_ok(self, env, truck_id: str, support_node: int, max_delay_steps: float = 5.0) -> bool:
        goal_id = getattr(env, "_effective_goals", {}).get(str(truck_id), getattr(env, "_recommended_goals", {}).get(str(truck_id)))
        task = env.state.tasks.get(str(goal_id)) if goal_id is not None else None
        truck = env.state.agents.get(str(truck_id))
        if task is None or truck is None or getattr(task, "kind", None) != TaskKind.NORMAL or getattr(truck, "node", None) is None:
            return True
        try:
            direct = float(env._decision_shortest_path_distance(int(truck.node), int(task.demand_node)))
            via = float(env.topology.edge_distance(int(truck.node), int(support_node))) + float(
                env._decision_shortest_path_distance(int(support_node), int(task.demand_node))
            )
            step_m = max(float(getattr(env.cfg, "truck_speed_mps", 8.0)) * float(getattr(env.cfg, "dt_seconds", 60.0)), 1.0)
            return bool((via - direct) / step_m <= float(max_delay_steps))
        except Exception:
            return True

    def _loaded_docked_uavs(self, env, truck_id: Optional[str] = None):
        out = []
        for uid, us in env.state.agents.items():
            if us.kind != AgentKind.UAV or bool(getattr(us, "crashed", False)):
                continue
            if getattr(us, "follow_target", None) is None:
                continue
            if truck_id is not None and str(getattr(us, "follow_target", "")) != str(truck_id):
                continue
            if bool(getattr(us, "uav_needs_reload_flag", False)):
                continue
            if int(getattr(us, "uav_reload_timer", 0)) > 0:
                continue
            out.append((str(uid), us))
        return out

    def _new_support_command_id(self, env, truck_id: str, uav_id: str, task_id: str) -> str:
        self._next_support_command_seq += 1
        return f"support_{int(getattr(env.state, 'step_index', 0))}_{self._next_support_command_seq}_{truck_id}_{uav_id}_{task_id}"

    def _new_sortie_id(self, env, uav_id: str, task_id: str) -> str:
        self._next_sortie_seq += 1
        return f"sortie_{int(getattr(env.state, 'step_index', 0))}_{self._next_sortie_seq}_{uav_id}_{task_id}"

    def _new_safety_recovery_command_id(self, env, truck_id: str, uav_id: str) -> str:
        self._next_safety_recovery_seq += 1
        return f"safety_{int(getattr(env.state, 'step_index', 0))}_{self._next_safety_recovery_seq}_{truck_id}_{uav_id}"

    def _support_row(self, support_command_id: str) -> Optional[Dict[str, object]]:
        for row in self._support_lifecycle_rows:
            if str(row.get("support_command_id", "")) == str(support_command_id):
                return row
        return None

    def _sortie_row(self, sortie_id: str) -> Optional[Dict[str, object]]:
        for row in self._sortie_lifecycle_rows:
            if str(row.get("sortie_id", "")) == str(sortie_id):
                return row
        return None

    def _safety_row(self, safety_recovery_command_id: str) -> Optional[Dict[str, object]]:
        for row in self._safety_lifecycle_rows:
            if str(row.get("safety_recovery_command_id", "")) == str(safety_recovery_command_id):
                return row
        return None

    def _set_support_status(self, support_command_id: str, status: str, step_key: Optional[str] = None, step: Optional[int] = None) -> None:
        row = self._support_row(support_command_id)
        if row is None:
            return
        row["status_final"] = str(status)
        if step_key is not None and step is not None and row.get(step_key, "") in {"", None}:
            row[step_key] = int(step)

    def _set_safety_status(self, command_id: str, status: str, step_key: Optional[str] = None, step: Optional[int] = None) -> None:
        row = self._safety_row(command_id)
        if row is None:
            return
        row["status_final"] = str(status)
        if step_key is not None and step is not None and row.get(step_key, "") in {"", None}:
            row[step_key] = int(step)

    def _task_delivered(self, task) -> bool:
        return bool(task is not None and getattr(task, "status", None) == TaskStatus.DELIVERED)

    def _support_anchor_distance(self, env, truck, support_node: int) -> float:
        if truck is None or getattr(truck, "node", None) is None:
            return float("inf")
        try:
            return float(env._decision_shortest_path_distance(int(truck.node), int(support_node)))
        except Exception:
            pass
        try:
            a = env._node_xy(int(truck.node))
            b = env._node_xy(int(support_node))
            return float(((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2) ** 0.5)
        except Exception:
            return float("inf")

    def _reject_reason_key(self, reason: str) -> str:
        r = str(reason or "").lower()
        if "not_on_truck" in r:
            return "uav_not_on_truck"
        if "not_loaded" in r or "reload" in r:
            return "uav_not_loaded"
        if "airborne" in r:
            return "uav_already_airborne"
        if "expired" in r or "lifeline" in r:
            return "task_expired" if "task" in r else "full_sortie_infeasible_lifeline"
        if "delivered" in r or "failed" in r:
            return "task_already_delivered_or_failed"
        if "recovery" in r or "no_recovery" in r:
            return "full_sortie_infeasible_recovery"
        if "energy" in r or "battery" in r:
            return "full_sortie_infeasible_energy"
        if "corridor" in r or "comm" in r:
            return "corridor_or_comm_block"
        if "env" in r or "gate" in r:
            return "launch_gate_env_rejected"
        return "unknown"

    def _mark_support_lock_released(self, rec: Dict[str, object], expired: bool = False) -> None:
        if not (self.gate19_support_lock or self.gate19_launch_binding):
            return
        if bool(rec.get("support_lock_released", False)):
            return
        rec["support_lock_released"] = True
        if expired:
            self._bump("support_lock_expired_count")
        else:
            self._bump("support_lock_released_count")

    def _is_uav_loaded(self, env, uav_id: str) -> bool:
        try:
            return bool(env._uav_loaded(str(uav_id)))
        except Exception:
            uav = env.state.agents.get(str(uav_id))
            return bool(uav is not None and not bool(getattr(uav, "uav_needs_reload_flag", False)) and int(getattr(uav, "uav_reload_timer", 0)) <= 0)

    def _support_violation_counter(self, violation: str) -> str:
        mapping = {
            "uav_not_on_truck_at_creation": "uav_not_on_truck_at_creation_count",
            "uav_left_truck_due_to_direct_launch": "uav_left_truck_due_to_direct_launch_count",
            "uav_left_truck_due_to_other_assignment": "uav_left_truck_due_to_other_assignment_count",
            "uav_left_truck_due_to_recovery": "uav_left_truck_due_to_recovery_count",
            "uav_follow_target_changed": "uav_follow_target_changed_count",
            "uav_docked_truck_changed": "uav_docked_truck_changed_count",
            "uav_became_airborne_without_support_launch": "uav_became_airborne_without_support_launch_count",
            "uav_loaded_became_false": "uav_loaded_became_false_count",
            "uav_state_sync_mismatch": "uav_state_sync_mismatch_count",
        }
        return mapping.get(str(violation), "unknown_passenger_violation_count")

    def _support_passenger_state(self, env, rec: Dict[str, object]) -> Dict[str, object]:
        uav_id = str(rec.get("uav_id", ""))
        truck_id = str(rec.get("truck_id", ""))
        task_id = str(rec.get("task_id", ""))
        uav = env.state.agents.get(uav_id)
        truck = env.state.agents.get(truck_id)
        loaded = self._is_uav_loaded(env, uav_id) if uav is not None else False
        airborne = bool(getattr(uav, "airborne", False)) if uav is not None else False
        follow = "" if uav is None else str(getattr(uav, "follow_target", "") or "")
        status = str(self._support_row(str(rec.get("support_command_id", ""))).get("status_final", "")) if self._support_row(str(rec.get("support_command_id", ""))) is not None else ""
        violation = ""
        if uav is None:
            violation = "unknown"
        elif airborne and not rec.get("sortie_id"):
            violation = "uav_became_airborne_without_support_launch"
        elif not loaded:
            violation = "uav_loaded_became_false"
        elif follow != truck_id:
            initial = str(rec.get("initial_follow_target", truck_id))
            if follow and follow != initial:
                violation = "uav_follow_target_changed"
            elif follow:
                violation = "uav_docked_truck_changed"
            else:
                violation = "uav_left_truck_due_to_other_assignment"
        invariant_ok = not bool(violation)
        return {
            "scenario": "",
            "method": "",
            "step": int(getattr(env.state, "step_index", 0)),
            "support_command_id": str(rec.get("support_command_id", "")),
            "truck_id": truck_id,
            "uav_id": uav_id,
            "task_id": task_id,
            "truck_node": "" if truck is None else getattr(truck, "node", ""),
            "support_status": status,
            "uav_state": "missing" if uav is None else ("airborne" if airborne else "docked"),
            "uav_docked_on_truck": int(bool(follow == truck_id and not airborne)),
            "uav_follow_target": follow,
            "uav_active_command_id": str(rec.get("support_command_id", "")) if bool(rec.get("support_lock_active", False)) else "",
            "uav_assigned_task": task_id if bool(rec.get("support_lock_active", False)) else "",
            "uav_loaded": int(bool(loaded)),
            "uav_airborne": int(bool(airborne)),
            "invariant_ok": int(bool(invariant_ok)),
            "invariant_violation_type": violation,
            "changed_by_module": "unknown" if violation else "",
            "change_reason": violation,
        }

    def _trace_support_passenger(self, env, rec: Dict[str, object]) -> str:
        if not self.gate20_variant:
            return ""
        state = self._support_passenger_state(env, rec)
        self._support_passenger_trace_rows.append(state)
        self._bump("passenger_invariant_check_count")
        violation = str(state.get("invariant_violation_type", ""))
        if violation:
            self._bump("passenger_invariant_violation_count")
            self._bump(self._support_violation_counter(violation))
        elif bool(rec.get("truck_arrived_anchor", False)):
            self._bump("passenger_invariant_preserved_to_anchor_count")
        return violation

    def _support_quality_eval(
        self,
        env,
        oracle: FeasibilityOracle,
        truck_id: str,
        uav_id: str,
        task,
        support_node: Optional[int],
        support_feasible: bool,
        support_feas,
        delay_steps: float,
        near_routine: bool,
    ) -> Dict[str, object]:
        now = int(getattr(env.state, "step_index", 0))
        direct_feas = oracle.evaluate_uav_sortie(str(uav_id), str(task.task_id))
        direct_feasible = bool(getattr(direct_feas, "full_sortie_feasible", False))
        direct_step = float(getattr(direct_feas, "expected_completion_step", now + 999))
        direct_margin = float(getattr(direct_feas, "recovery_margin", 0.0))
        direct_lifeline = float(getattr(direct_feas, "expected_lifeline_remaining", 0.0))
        direct_score = (1.0 if direct_feasible else 0.0) + max(direct_lifeline, 0.0) / 100.0 + max(direct_margin, 0.0) / 5000.0 - max(direct_step - now, 0.0) / 100.0

        routine_task_id = ""
        routine_reachable = False
        routine_eta = 999.0
        routine_score = 0.0
        goal_id = getattr(env, "_effective_goals", {}).get(str(truck_id), getattr(env, "_recommended_goals", {}).get(str(truck_id)))
        rtask = env.state.tasks.get(str(goal_id)) if goal_id is not None else None
        truck = env.state.agents.get(str(truck_id))
        if rtask is not None and getattr(rtask, "kind", None) == TaskKind.NORMAL and truck is not None and getattr(truck, "node", None) is not None:
            routine_task_id = str(getattr(rtask, "task_id", ""))
            try:
                dist = float(env._decision_shortest_path_distance(int(truck.node), int(rtask.demand_node)))
                step_m = max(float(getattr(env.cfg, "truck_speed_mps", 8.0)) * float(getattr(env.cfg, "dt_seconds", 60.0)), 1.0)
                routine_eta = float(dist / step_m)
                routine_reachable = True
            except Exception:
                routine_reachable = False
            routine_score = (0.75 if routine_reachable else 0.0) + (0.4 if near_routine else 0.0) - min(max(routine_eta, 0.0), 100.0) / 200.0

        support_step = float(getattr(support_feas, "expected_completion_step", now + 999)) if support_feas is not None else now + 999
        support_margin = float(getattr(support_feas, "recovery_margin", 0.0)) if support_feas is not None else 0.0
        support_lifeline = float(getattr(support_feas, "expected_lifeline_remaining", 0.0)) if support_feas is not None else 0.0
        support_score = (1.0 if support_feasible else 0.0) + max(support_lifeline, 0.0) / 100.0 + max(support_margin, 0.0) / 5000.0 - max(support_step - now, 0.0) / 100.0
        routine_delay_penalty = max(float(delay_steps), 0.0) * 0.04
        switch_cost = 0.05
        uncertainty = 0.05
        support_net_gain = float(support_score - max(direct_score, routine_score) - switch_cost - routine_delay_penalty - uncertainty)
        threshold = 0.03 if self.gate112_quality_relaxed else 0.10
        reason = ""
        if support_node is None:
            reason = "no_valid_anchor"
        elif not support_feasible:
            reason = "support_not_full_sortie_feasible"
        elif float(delay_steps) > 2.0:
            reason = "routine_delay_too_high"
        elif direct_score >= support_score and direct_feasible:
            reason = "direct_launch_better"
        elif routine_score >= support_score and routine_reachable:
            reason = "routine_continue_better"
        elif support_net_gain <= threshold:
            reason = "support_net_gain_too_low"
        allowed = bool(not reason)
        return {
            "scenario": "",
            "method": "",
            "step": now,
            "truck_id": str(truck_id),
            "uav_id": str(uav_id),
            "task_id": str(task.task_id),
            "routine_task_id": routine_task_id,
            "direct_feasible": int(direct_feasible),
            "direct_predicted_delivery_step": direct_step,
            "direct_recovery_margin": direct_margin,
            "direct_lifeline_margin": direct_lifeline,
            "direct_score": direct_score,
            "routine_reachable": int(routine_reachable),
            "routine_eta": routine_eta,
            "routine_near_completion": int(bool(near_routine)),
            "routine_score": routine_score,
            "support_feasible": int(bool(support_feasible)),
            "support_anchor": "" if support_node is None else int(support_node),
            "support_eta_to_anchor": 1.0 if support_node is not None else 999.0,
            "support_predicted_delivery_step": support_step,
            "support_recovery_margin": support_margin,
            "support_routine_delay": float(delay_steps),
            "support_score": support_score,
            "support_net_gain": support_net_gain,
            "support_allowed": int(allowed),
            "support_block_reason": reason,
            "actual_support_created": 0,
            "actual_launch": 0,
            "actual_delivery": 0,
            "actual_routine_completed": 0,
        }

    def _record_support_quality(self, row: Dict[str, object]) -> None:
        self._support_quality_rows.append(dict(row))
        self._bump("support_candidate_count")
        reason = str(row.get("support_block_reason", ""))
        if int(row.get("support_allowed", 0)):
            self._bump("support_allowed_count")
        elif reason == "direct_launch_better":
            self._bump("support_blocked_direct_better_count")
        elif reason == "routine_continue_better":
            self._bump("support_blocked_routine_better_count")
        elif reason == "support_net_gain_too_low":
            self._bump("support_blocked_net_gain_low_count")
        elif reason == "routine_delay_too_high":
            self._bump("support_blocked_routine_delay_high_count")
        elif reason == "support_not_full_sortie_feasible":
            self._bump("support_blocked_not_full_sortie_feasible_count")
        elif reason == "support_lifeline_risk":
            self._bump("support_blocked_lifeline_risk_count")
        elif reason == "support_recovery_margin_low":
            self._bump("support_blocked_recovery_margin_low_count")
        elif reason == "no_loaded_uav":
            self._bump("support_blocked_no_loaded_uav_count")
        elif reason == "no_valid_anchor":
            self._bump("support_blocked_no_valid_anchor_count")

    def _fail_support_launch_attempt(self, rec: Dict[str, object], reason: str, now: int) -> None:
        support_id = str(rec.get("support_command_id", ""))
        key = str(rec.get("support_key", support_id))
        reject_key = self._reject_reason_key(reason)
        if not bool(rec.get("launch_attempted", False)):
            rec["launch_attempted"] = True
            rec["launch_attempt_step"] = now
            self._bump("support_anchor_arrived_launch_attempt_count")
            row = self._support_row(support_id)
            if row is not None:
                row["launch_attempt_step"] = now
        if not bool(rec.get("launch_attempt_rejected", False)):
            rec["launch_attempt_rejected"] = True
            rec["launch_reject_reason"] = reject_key
            self._bump("support_anchor_arrived_launch_rejected_count")
            self._bump(f"support_anchor_launch_reject_{reject_key}_count")
        self._set_support_status(support_id, "failed_anchor_launch_rejected", "expired_step", now)
        row = self._support_row(support_id)
        if row is not None:
            row["launch_attempt_step"] = now
            row["launch_reject_reason"] = reject_key
            row["abort_reason"] = reject_key
        self._mark_support_lock_released(rec)
        self._support_cooldown_until[f"{rec.get('truck_id')}:{rec.get('uav_id')}:{rec.get('task_id')}"] = now + 10
        self._active_support_commands.pop(support_id, None)
        self._active_support_commitments.pop(key, None)

    def _support_arrived(self, env, rec: Dict[str, object]) -> bool:
        truck = env.state.agents.get(str(rec.get("truck_id", "")))
        support_node = int(rec.get("support_node", -1))
        if truck is None or getattr(truck, "node", None) is None or support_node < 0:
            return False
        exact = bool(int(truck.node) == support_node)
        dist = self._support_anchor_distance(env, truck, support_node)
        radius = bool(self.gate19_launch_binding and dist <= float(self.v2_support_anchor_arrival_radius_m))
        arrived = bool(exact or radius)
        if arrived and not bool(rec.get("truck_arrived_anchor", False)):
            rec["truck_arrived_anchor"] = True
            rec["arrival_distance_m"] = float(dist if dist != float("inf") else 0.0)
            rec["arrival_reason"] = "exact_node" if exact else "radius_not_exact_node"
            self._bump("support_command_truck_arrived_anchor_count")
            if exact:
                self._bump("support_arrived_by_exact_node_count")
            if radius:
                self._bump("support_arrived_by_radius_count")
            if radius and not exact:
                self._bump("support_arrival_missed_by_exact_node_count")
            if dist != float("inf"):
                self._bump("support_anchor_distance_at_arrival_sum", float(dist))
                self._bump("support_anchor_distance_at_arrival_count")
                cnt = max(float(getattr(self, "support_anchor_distance_at_arrival_count", 0.0)), 1.0)
                self.support_anchor_distance_at_arrival_mean = float(getattr(self, "support_anchor_distance_at_arrival_sum", 0.0)) / cnt
            if self.gate18_support_anchor_strict:
                self._bump("strict_anchor_to_arrival_count")
            self._set_support_status(str(rec.get("support_command_id", "")), "truck_arrived_anchor", "truck_arrived_anchor_step", int(getattr(env.state, "step_index", 0)))
            row = self._support_row(str(rec.get("support_command_id", "")))
            if row is not None:
                row["arrival_reason"] = rec.get("arrival_reason", "")
                row["support_anchor_distance_at_arrival"] = rec.get("arrival_distance_m", "")
        return arrived

    def _routine_delay_steps(self, env, truck_id: str, support_node: int) -> float:
        goal_id = getattr(env, "_effective_goals", {}).get(str(truck_id), getattr(env, "_recommended_goals", {}).get(str(truck_id)))
        task = env.state.tasks.get(str(goal_id)) if goal_id is not None else None
        truck = env.state.agents.get(str(truck_id))
        if task is None or truck is None or getattr(task, "kind", None) != TaskKind.NORMAL or getattr(truck, "node", None) is None:
            return 0.0
        try:
            direct = float(env._decision_shortest_path_distance(int(truck.node), int(task.demand_node)))
            via = float(env.topology.edge_distance(int(truck.node), int(support_node))) + float(
                env._decision_shortest_path_distance(int(support_node), int(task.demand_node))
            )
            step_m = max(float(getattr(env.cfg, "truck_speed_mps", 8.0)) * float(getattr(env.cfg, "dt_seconds", 60.0)), 1.0)
            return float((via - direct) / step_m)
        except Exception:
            return 0.0

    def _has_near_completion_routine_commitment(self, env, truck_id: str) -> bool:
        goal_id = getattr(env, "_effective_goals", {}).get(str(truck_id), getattr(env, "_recommended_goals", {}).get(str(truck_id)))
        task = env.state.tasks.get(str(goal_id)) if goal_id is not None else None
        truck = env.state.agents.get(str(truck_id))
        if task is None or truck is None or getattr(task, "kind", None) != TaskKind.NORMAL:
            return False
        if getattr(task, "status", None) != TaskStatus.PENDING or getattr(truck, "node", None) is None:
            return False
        try:
            dist = float(env._decision_shortest_path_distance(int(truck.node), int(task.demand_node)))
            step_m = max(float(getattr(env.cfg, "truck_speed_mps", 8.0)) * float(getattr(env.cfg, "dt_seconds", 60.0)), 1.0)
            return bool(dist <= 1000.0 or (dist / step_m) <= 5.0)
        except Exception:
            return False

    def _hard_recovery_uav_ids(self, env):
        force_thr = float(getattr(getattr(env, "cfg", object()), "uav_low_battery_force_recover_threshold", 0.25))
        low_thr = float(getattr(getattr(env, "cfg", object()), "uav_low_battery_goal_lock_threshold", 0.35))
        latch = getattr(env, "_uav_forced_rth_latch", {})
        out = []
        for uid, us in env.state.agents.items():
            if us.kind != AgentKind.UAV or bool(getattr(us, "crashed", False)):
                continue
            if getattr(us, "follow_target", None) is not None:
                continue
            batt = float(getattr(us, "battery", 0.0))
            forced = bool(latch.get(str(uid), False)) if isinstance(latch, dict) else False
            needs_reload = bool(getattr(us, "uav_needs_reload_flag", False))
            unloaded = False
            try:
                unloaded = not bool(env._uav_loaded(str(uid)))
            except Exception:
                unloaded = False
            if batt <= force_thr or forced or ((needs_reload or unloaded) and batt <= low_thr):
                out.append(str(uid))
        return out

    def _observe_safety_recovery_commands(self, env) -> None:
        now = int(getattr(env.state, "step_index", 0))
        expired = []
        for command_id, rec in list(self._active_safety_recovery_commands.items()):
            uid = str(rec.get("uav_id", ""))
            uav = env.state.agents.get(uid)
            if uav is None or bool(getattr(uav, "crashed", False)):
                self._bump("safety_recovery_failed_count")
                self._bump("v2_safety_recovery_failed_count")
                row = self._safety_row(command_id)
                if row is not None:
                    row["status_final"] = "uav_lost"
                    row["fail_reason"] = "uav_missing_or_crashed"
                expired.append(command_id)
                continue
            if getattr(uav, "follow_target", None) is not None or not bool(getattr(uav, "airborne", False)):
                self._bump("safety_recovery_to_recovered_count")
                self._bump("v2_safety_recovery_to_recovered_count")
                if self.gate18_safety_narrow_routine_guard:
                    self._bump("safety_narrow_beneficial_recovery_count")
                self._set_safety_status(command_id, "recovered", "recovered_step", now)
                expired.append(command_id)
                continue
            if rec.get("uav_returning_step", "") == "":
                rec["uav_returning_step"] = now
                self._bump("safety_recovery_uav_returning_count")
                self._set_safety_status(command_id, "uav_returning", "uav_returning_step", now)
            if now > int(rec.get("expires_step", now)):
                self._bump("safety_recovery_expired_count")
                self._set_safety_status(command_id, "expired", "expired_step", now)
                expired.append(command_id)
        for command_id in expired:
            self._active_safety_recovery_commands.pop(str(command_id), None)

    def _observe_support_commitments(self, env) -> None:
        now = int(getattr(env.state, "step_index", 0))
        if self.gate17_lifecycle_closure:
            self._observe_safety_recovery_commands(env)
        expired = []
        for key, rec in list(self._active_support_commitments.items()):
            task = env.state.tasks.get(str(rec.get("task_id", "")))
            support_id = str(rec.get("support_command_id", ""))
            self._support_arrived(env, rec)
            violation = self._trace_support_passenger(env, rec)
            if violation and (self.gate20_passenger_invariant or self.gate20_support_reserve_launch):
                self._bump("support_failed_due_to_passenger_violation_count")
                self._set_support_status(support_id, "failed_passenger_violation", "expired_step", now)
                row = self._support_row(support_id)
                if row is not None:
                    row["abort_reason"] = violation
                    row["launch_reject_reason"] = violation
                self._mark_support_lock_released(rec)
                expired.append(key)
                continue
            if self._task_delivered(task):
                sortie_id = str(rec.get("sortie_id", ""))
                if sortie_id and sortie_id in self._delivered_sortie_ids and key not in self._support_delivery_marked:
                    self._bump("v2_support_command_to_delivery_count")
                    self._bump("support_command_delivery_completed_count")
                    self._bump("support_delivery_count")
                    self._bump("support_allowed_to_delivery_count" if bool(rec.get("routine_override", False)) else "support_command_to_delivery_count")
                    self._support_delivery_marked.add(key)
                    self._set_support_status(support_id, "delivery_completed", "delivery_step", now)
                    self._mark_support_lock_released(rec)
                    row = self._support_row(support_id)
                    if row is not None:
                        row["linked_delivery_valid"] = 1
                    if self.gate18_support_anchor_strict:
                        self._bump("strict_anchor_to_delivery_count")
                    if self.gate18_anchor_arrival_force_launch:
                        self._bump("anchor_arrival_to_delivery_count")
                    if self.gate18_safety_narrow_routine_guard and bool(rec.get("routine_override", False)):
                        self._bump("routine_guard_allowed_to_delivery_count")
                    if self.gate20_support_reserve_launch or self.gate20_passenger_invariant:
                        self._bump("support_reserve_to_delivery_count")
                    if self.gate20_rebind_at_anchor and bool(rec.get("rebound_at_anchor", False)):
                        self._bump("support_rebind_to_delivery_count")
                expired.append(key)
                continue
            launch_step = int(rec.get("launch_triggered_step", -1))
            delivery_deadline = int(rec.get("delivery_deadline_step", int(rec.get("expires_step", now))))
            if launch_step >= 0 and now > delivery_deadline:
                self._bump("support_allowed_but_no_delivery_count")
                if self.gate18_safety_narrow_routine_guard and bool(rec.get("routine_override", False)):
                    self._bump("routine_guard_allowed_no_delivery_count")
                self._set_support_status(support_id, "launched_no_delivery", "expired_step", now)
                expired.append(key)
                continue
            if launch_step < 0 and now > int(rec.get("expires_step", now)):
                self._bump("v2_support_command_expired_count")
                self._bump("support_command_expired_count")
                if bool(rec.get("truck_arrived_anchor", False)):
                    self._bump("support_command_expired_before_launch_count")
                    self._set_support_status(support_id, "expired_before_launch", "expired_step", now)
                else:
                    self._bump("support_command_expired_before_anchor_count")
                    self._set_support_status(support_id, "expired_before_anchor", "expired_step", now)
                expired.append(key)
        for key in expired:
            rec = self._active_support_commitments.pop(key, None)
            if isinstance(rec, dict):
                sid = str(rec.get("support_command_id", ""))
                self._mark_support_lock_released(rec, expired=True)
                self._active_support_commands.pop(sid, None)

        for sortie_id, rec in list(self._active_sorties.items()):
            task = env.state.tasks.get(str(rec.get("task_id", "")))
            uav = env.state.agents.get(str(rec.get("uav_id", "")))
            if (not bool(rec.get("launch_success", False))) and uav is not None and bool(getattr(uav, "airborne", False)):
                rec["launch_success"] = True
                self._bump("sortie_launch_success_count")
            if self._task_delivered(task):
                if sortie_id in self._delivered_sortie_ids:
                    if sortie_id not in self._duplicate_delivery_marked:
                        self._bump("sortie_duplicate_delivery_blocked_count")
                        self._duplicate_delivery_marked.add(sortie_id)
                    continue
                task_id = str(rec.get("task_id", ""))
                if task_id in self._delivered_task_ids_by_sortie:
                    if task_id not in self._duplicate_delivery_marked:
                        self._bump("delivery_task_already_delivered_blocked_count")
                        self._duplicate_delivery_marked.add(task_id)
                    continue
                rec["delivered"] = True
                rec["actual_delivery_step"] = now
                self._delivered_sortie_ids.add(sortie_id)
                self._delivered_task_ids_by_sortie.add(task_id)
                self._bump("sortie_delivery_success_count")
                self._bump("sortie_to_delivery_count")
                self._bump("sortie_oracle_to_delivery_count")
                support_id = str(rec.get("support_command_id", ""))
                if support_id:
                    if support_id in self._active_support_commands:
                        self._active_support_commands[support_id]["sortie_id"] = sortie_id
                    if support_id in self._support_to_sortie and support_id not in self._support_delivery_marked:
                        self._bump("v2_support_command_to_delivery_count")
                        self._bump("support_command_delivery_completed_count")
                        self._bump("support_delivery_count")
                        self._support_delivery_marked.add(support_id)
                        self._set_support_status(support_id, "delivery_completed", "delivery_step", now)
                        srow = self._support_row(support_id)
                        if srow is not None:
                            srow["linked_delivery_valid"] = 1
                        if self.gate20_support_reserve_launch or self.gate20_passenger_invariant:
                            self._bump("support_reserve_to_delivery_count")
                        if self.gate20_rebind_at_anchor:
                            srec0 = self._active_support_commands.get(support_id, {})
                            if isinstance(srec0, dict) and bool(srec0.get("rebound_at_anchor", False)):
                                self._bump("support_rebind_to_delivery_count")
                        srec = self._active_support_commands.get(support_id)
                        if isinstance(srec, dict):
                            self._mark_support_lock_released(srec)
                        self._active_support_commands.pop(support_id, None)
                for row in self._sortie_lifecycle_rows:
                    if row.get("sortie_id") == sortie_id:
                        row["actual_delivery_step"] = now
                        row["delivery_step"] = now
                        row["lifecycle_status"] = "delivered"
                        row["delivery_link_valid"] = 1
                if self.gate18_direct_tc_first and not support_id:
                    self._bump("direct_tc_to_delivery_count")
        delivered_emergency_tasks = {
            str(t.task_id)
            for t in env.state.tasks.values()
            if getattr(t, "kind", None) == TaskKind.EMERGENCY and self._task_delivered(t)
        }
        untracked = delivered_emergency_tasks - set(str(r.get("task_id", "")) for r in self._active_sorties.values()) - self._delivered_task_ids_by_sortie
        new_untracked = set(untracked) - self._delivery_without_sortie_marked
        if new_untracked:
            self._bump("delivery_without_sortie_blocked_count", len(new_untracked))
            self._delivery_without_sortie_marked.update(new_untracked)

    def _active_support_for_uav(self, env, uav_id: str) -> Optional[Dict[str, object]]:
        now = int(getattr(env.state, "step_index", 0))
        for key, rec in list(self._active_support_commitments.items()):
            if str(rec.get("uav_id", "")) != str(uav_id):
                continue
            if now > int(rec.get("expires_step", now)):
                self._bump("v2_support_command_expired_count")
                support_id = str(rec.get("support_command_id", ""))
                self._bump("support_command_expired_count")
                if bool(rec.get("truck_arrived_anchor", False)):
                    self._bump("support_command_expired_before_launch_count")
                    self._set_support_status(support_id, "expired_before_launch", "expired_step", now)
                else:
                    self._bump("support_command_expired_before_anchor_count")
                    self._set_support_status(support_id, "expired_before_anchor", "expired_step", now)
                self._mark_support_lock_released(rec, expired=True)
                self._active_support_commitments.pop(key, None)
                self._active_support_commands.pop(support_id, None)
                continue
            task = env.state.tasks.get(str(rec.get("task_id", "")))
            truck = env.state.agents.get(str(rec.get("truck_id", "")))
            uav = env.state.agents.get(str(uav_id))
            if task is None or truck is None or uav is None:
                if self.gate19_launch_binding and bool(rec.get("truck_arrived_anchor", False)):
                    self._fail_support_launch_attempt(rec, "unknown", now)
                continue
            if getattr(task, "status", None) != TaskStatus.PENDING:
                if self.gate19_launch_binding and bool(rec.get("truck_arrived_anchor", False)):
                    self._fail_support_launch_attempt(rec, "task_already_delivered_or_failed", now)
                continue
            if bool(getattr(uav, "airborne", False)) or bool(getattr(uav, "crashed", False)):
                if self.gate19_launch_binding and bool(rec.get("truck_arrived_anchor", False)):
                    self._fail_support_launch_attempt(rec, "uav_already_airborne", now)
                continue
            if str(getattr(uav, "follow_target", "")) != str(rec.get("truck_id", "")):
                if self.gate19_launch_binding and bool(rec.get("truck_arrived_anchor", False)) and not self.gate20_rebind_at_anchor:
                    self._fail_support_launch_attempt(rec, "uav_not_on_truck", now)
                    continue
                if not self.gate20_rebind_at_anchor:
                    continue
            if bool(getattr(uav, "uav_needs_reload_flag", False)) or int(getattr(uav, "uav_reload_timer", 0)) > 0:
                if self.gate19_launch_binding and bool(rec.get("truck_arrived_anchor", False)):
                    self._fail_support_launch_attempt(rec, "uav_not_loaded", now)
                continue
            if getattr(truck, "node", None) is None:
                continue
            if not self._support_arrived(env, rec):
                continue
            if self.gate20_rebind_at_anchor and str(getattr(uav, "follow_target", "")) != str(rec.get("truck_id", "")):
                task = env.state.tasks.get(str(rec.get("task_id", "")))
                truck_id = str(rec.get("truck_id", ""))
                for alt_uid, _alt in self._loaded_docked_uavs(env, truck_id=truck_id):
                    if str(alt_uid) == str(uav_id):
                        continue
                    if any(str(r.get("uav_id", "")) == str(alt_uid) and str(r.get("support_command_id", "")) != str(rec.get("support_command_id", "")) for r in self._active_support_commitments.values()):
                        continue
                    if task is None:
                        continue
                    feas = FeasibilityOracle(env).evaluate_uav_sortie(str(alt_uid), str(task.task_id), launch_anchor=int(rec.get("support_node", -1)))
                    if bool(getattr(feas, "full_sortie_feasible", False)):
                        old_uid = str(rec.get("uav_id", ""))
                        rec["uav_id"] = str(alt_uid)
                        rec["rebound_at_anchor"] = True
                        self._bump("support_rebind_success_count")
                        row = self._support_row(str(rec.get("support_command_id", "")))
                        if row is not None:
                            row["uav_id"] = str(alt_uid)
                            row["abort_reason"] = f"rebound_from_{old_uid}"
                        if str(alt_uid) == str(uav_id):
                            return rec
                        return None
                if self.gate19_launch_binding and bool(rec.get("truck_arrived_anchor", False)):
                    self._fail_support_launch_attempt(rec, "uav_not_on_truck", now)
                continue
            if self.gate19_launch_binding and not bool(rec.get("launch_attempted", False)):
                rec["launch_attempted"] = True
                rec["launch_attempt_step"] = now
                self._bump("support_anchor_arrived_launch_attempt_count")
                row = self._support_row(str(rec.get("support_command_id", "")))
                if row is not None:
                    row["launch_attempt_step"] = now
            return rec
        return None

    def _support_sortie_feasible_after_anchor(self, env, oracle: FeasibilityOracle, uav_id: str, task, support_node: int):
        self._bump("sortie_oracle_candidate_count")
        current = oracle.evaluate_uav_sortie(str(uav_id), str(task.task_id), launch_anchor=int(support_node))
        if current.full_sortie_feasible:
            self._bump("sortie_oracle_full_feasible_count")
            return True, current
        if self.gate16_command_quality:
            reason = str(current.reject_reason)
            if "recovery" in reason or "no_recovery" in reason:
                self._bump("sortie_oracle_blocked_recovery_count")
            elif "battery" in reason or "energy" in reason:
                self._bump("sortie_oracle_blocked_energy_count")
            elif float(getattr(task, "lifeline_current", 0.0)) <= 0:
                self._bump("sortie_oracle_blocked_lifeline_count")
            return False, current
        # Lightweight anchor-projection: use the same launch gate first, then
        # allow a support command only when the projected anchor sharply reduces
        # sortie distance and the UAV is otherwise healthy.
        try:
            task_xy = env._node_xy(int(task.demand_node))
            anchor_xy = env._node_xy(int(support_node))
            dist = float(((task_xy[0] - anchor_xy[0]) ** 2 + (task_xy[1] - anchor_xy[1]) ** 2) ** 0.5)
        except Exception:
            dist = float("inf")
        uav = env.state.agents.get(str(uav_id))
        battery = float(getattr(uav, "battery", 0.0)) if uav is not None else 0.0
        lifeline = float(getattr(task, "lifeline_current", 0.0))
        heuristic_ok = bool(dist <= 2500.0 and battery >= 0.58 and lifeline > 0.0)
        if heuristic_ok:
            self._bump("sortie_oracle_full_feasible_count")
            return True, current
        reason = str(current.reject_reason)
        if "recovery" in reason or "no_recovery" in reason:
            self._bump("sortie_oracle_blocked_recovery_count")
        elif "battery" in reason or "energy" in reason:
            self._bump("sortie_oracle_blocked_energy_count")
        elif lifeline <= 0:
            self._bump("sortie_oracle_blocked_lifeline_count")
        return False, current

    def _try_support_command(self, env, oracle: FeasibilityOracle, batch: CommandBatch, aid: str) -> bool:
        if not self.support_recovery_repair:
            return False
        if self.gate16_command_quality:
            active_truck = any(str(r.get("truck_id", "")) == str(aid) for r in self._active_support_commitments.values())
            if active_truck:
                self._bump("support_command_blocked_active_limit_count")
                return False
        truck = env.state.agents.get(str(aid))
        if truck is None or getattr(truck, "node", None) is None:
            self._bump("v2_support_command_blocked_no_anchor_count")
            return False
        emergencies = sorted(
            self._pending_tasks(env, TaskKind.EMERGENCY),
            key=lambda t: float(getattr(t, "lifeline_current", 0.0)),
        )
        loaded = self._loaded_docked_uavs(env, truck_id=aid)
        if not loaded:
            self._bump("v2_support_command_blocked_no_loaded_uav_count")
            self._bump("support_command_blocked_no_loaded_uav_count")
            return False
        if self.gate20_workflow_repair and not (self.gate112_quality_audit or self.gate112_quality_gate):
            for uid0, _us0 in loaded:
                for task0 in emergencies:
                    direct_feas0 = oracle.evaluate_uav_sortie(str(uid0), str(task0.task_id))
                    if bool(getattr(direct_feas0, "full_sortie_feasible", False)):
                        self._bump("support_skipped_due_to_direct_tc_count")
                        return False
        for task in emergencies:
            if self.gate19_support_lock and any(
                str(r.get("task_id", "")) == str(task.task_id) and bool(r.get("support_lock_active", False))
                for r in self._active_support_commitments.values()
            ):
                self._bump("support_lock_blocked_task_steal_count")
                continue
            for uid, _us in loaded:
                if self.gate18_direct_tc_first:
                    self._bump("direct_tc_candidate_count")
                    direct_feas = oracle.evaluate_uav_sortie(str(uid), str(task.task_id))
                    if bool(getattr(direct_feas, "full_sortie_feasible", False)):
                        self._bump("support_skipped_due_to_direct_tc_count")
                        self._bump("support_blocked_direct_better_count")
                        continue
                if self.gate16_command_quality:
                    cool_key = f"{aid}:{uid}:{task.task_id}"
                    if int(self._support_cooldown_until.get(cool_key, -1)) > int(env.state.step_index):
                        self._bump("support_command_blocked_cooldown_count")
                        continue
                    if any(str(r.get("uav_id", "")) == str(uid) for r in self._active_support_commitments.values()):
                        self._bump("support_command_blocked_active_limit_count")
                        continue
                if self.gate20_variant:
                    ustate = env.state.agents.get(str(uid))
                    if ustate is None or str(getattr(ustate, "follow_target", "")) != str(aid):
                        self._bump("uav_not_on_truck_at_creation_count")
                        continue
                    if bool(getattr(ustate, "airborne", False)):
                        self._bump("uav_became_airborne_without_support_launch_count")
                        continue
                    if not self._is_uav_loaded(env, str(uid)):
                        self._bump("uav_loaded_became_false_count")
                        continue
                self._bump("v2_support_command_candidate_count")
                if self.gate18_support_anchor_strict:
                    self._bump("strict_anchor_candidate_count")
                support_node = self._next_step_toward(env, int(truck.node), int(task.demand_node))
                if support_node is None:
                    if self.gate112_quality_audit or self.gate112_quality_gate:
                        qrow = self._support_quality_eval(env, oracle, aid, uid, task, None, False, None, 999.0, False)
                        qrow["support_block_reason"] = "no_valid_anchor"
                        self._record_support_quality(qrow)
                    self._bump("v2_support_command_blocked_no_anchor_count")
                    if self.gate18_support_anchor_strict:
                        self._bump("strict_anchor_blocked_eta_count")
                    continue
                delay_steps = self._routine_delay_steps(env, aid, int(support_node))
                near_routine = self._has_near_completion_routine_commitment(env, aid)
                max_delay = 2.0 if self.gate16_command_quality and near_routine else 5.0
                if near_routine and self.gate16_command_quality:
                    self._bump("support_blocked_by_routine_commitment_count")
                    if self.gate18_safety_narrow_routine_guard:
                        self._bump("routine_guard_blocked_support_count")
                    if self.gate20_workflow_repair and not (self.gate112_quality_audit or self.gate112_quality_gate):
                        continue
                if not self._routine_delay_ok(env, aid, int(support_node), max_delay_steps=max_delay):
                    self._bump("v2_support_command_blocked_routine_delay_count")
                    self._bump("support_command_blocked_routine_delay_count")
                    if self.gate18_support_anchor_strict:
                        self._bump("strict_anchor_blocked_routine_delay_count")
                    continue
                feasible, feas = self._support_sortie_feasible_after_anchor(env, oracle, uid, task, int(support_node))
                qrow = None
                if self.gate112_quality_audit or self.gate112_quality_gate:
                    qrow = self._support_quality_eval(env, oracle, aid, uid, task, int(support_node), bool(feasible), feas, float(delay_steps), bool(near_routine))
                    self._record_support_quality(qrow)
                if not feasible:
                    self._bump("v2_support_command_blocked_not_full_sortie_feasible_count")
                    self._bump("support_command_blocked_not_full_sortie_feasible_count")
                    if self.gate18_support_anchor_strict:
                        self._bump("strict_anchor_blocked_margin_count")
                    continue
                if self.gate112_quality_gate and qrow is not None and not bool(int(qrow.get("support_allowed", 0))):
                    reason = str(qrow.get("support_block_reason", "support_net_gain_too_low"))
                    if reason == "direct_launch_better":
                        self._bump("support_skipped_due_to_direct_tc_count")
                        self._bump("support_blocked_direct_better_count")
                    continue
                if self.gate16_command_quality:
                    if float(getattr(feas, "recovery_margin", 0.0)) < 0.0:
                        self._bump("support_command_blocked_low_recovery_margin_count")
                        if self.gate18_support_anchor_strict:
                            self._bump("strict_anchor_blocked_margin_count")
                        continue
                    if float(getattr(feas, "expected_lifeline_remaining", 0.0)) <= 0.0:
                        self._bump("support_command_blocked_lifeline_risk_count")
                        if self.gate18_support_anchor_strict:
                            self._bump("strict_anchor_blocked_lifeline_count")
                        continue
                    if near_routine:
                        self._bump("support_allowed_despite_routine_commitment_count")
                        self._bump("support_allowed_routine_delay_steps_sum", max(float(delay_steps), 0.0))
                        if self.gate18_safety_narrow_routine_guard:
                            self._bump("routine_guard_allowed_support_count")
                self._bump("v2_support_command_generated_count")
                self._bump("support_command_created_count")
                if qrow is not None:
                    qrow["actual_support_created"] = 1
                if self.gate18_support_anchor_strict:
                    self._bump("strict_anchor_selected_count")
                command_id = self._new_support_command_id(env, aid, uid, str(task.task_id))
                key = command_id if self.gate17_lifecycle_closure else f"{aid}:{uid}:{task.task_id}"
                support_state = {
                    "support_command_id": command_id,
                    "support_key": key,
                    "truck_id": str(aid),
                    "uav_id": str(uid),
                    "task_id": str(task.task_id),
                    "support_node": int(support_node),
                    "launch_anchor": int(support_node),
                    "recovery_anchor": int(support_node),
                    "created_step": int(env.state.step_index),
                    "expires_step": int(env.state.step_index) + 8,
                    "routine_override": bool(near_routine),
                    "support_lock_active": bool(self.gate19_support_lock or self.gate19_launch_binding),
                    "initial_follow_target": str(aid),
                }
                self._active_support_commitments[key] = support_state
                self._active_support_commands[command_id] = support_state
                if self.gate19_support_lock or self.gate19_launch_binding:
                    self._bump("support_lock_created_count")
                self._support_lifecycle_rows.append(
                    {
                        "support_key": key,
                        "support_command_id": command_id,
                        "truck_id": str(aid),
                        "uav_id": str(uid),
                        "task_id": str(task.task_id),
                        "support_anchor": int(support_node),
                        "launch_anchor": int(support_node),
                        "recovery_anchor": int(support_node),
                        "created_step": int(env.state.step_index),
                        "ttl_steps": 8,
                        "expected_launch_step": int(env.state.step_index) + 1,
                        "expected_delivery_step": int(env.state.step_index) + 4,
                        "status_final": "created",
                        "truck_arrived_anchor_step": "",
                        "launch_triggered_step": "",
                        "sortie_id": "",
                        "delivery_step": "",
                        "expired_step": "",
                        "abort_reason": "",
                        "launch_attempt_step": "",
                        "launch_reject_reason": "",
                        "arrival_reason": "",
                        "support_anchor_distance_at_arrival": "",
                        "routine_commitment_active": int(bool(near_routine)),
                        "routine_delay_estimated": float(delay_steps),
                        "routine_delay_actual": "",
                        "linked_delivery_valid": 0,
                    }
                )
                batch.add_truck(
                    TruckCommand(
                        aid,
                        TruckCommandKind.SUPPORT_UAV,
                        task_id=str(task.task_id),
                        target_node=int(support_node),
                        support_point=int(support_node),
                        launch_anchor=int(support_node),
                        recovery_anchor=int(support_node),
                        support_uav_id=str(uid),
                        ttl_steps=8,
                        expected_launch_step=int(env.state.step_index) + 1,
                        expected_delivery_step=int(env.state.step_index) + 4,
                        expected_recovery_margin=float(getattr(feas, "recovery_margin", 0.0)),
                        reason="v2_support_delivery_feasible",
                    )
                )
                return True
        return False

    def _try_safety_recovery_command(self, env, batch: CommandBatch, aid: str) -> bool:
        if not self.gate16_command_quality:
            return False
        if not self.gate17_lifecycle_closure:
            if not bool(getattr(env, "_has_hard_recovery_uav", lambda: False)()):
                self._bump("safety_recovery_blocked_not_hard_risk_count")
                return False
            truck = env.state.agents.get(str(aid))
            if truck is None or getattr(truck, "node", None) is None:
                self._bump("safety_recovery_blocked_no_anchor_count")
                return False
            neighbors = list(env._decision_neighbors(int(truck.node))) if hasattr(env, "_decision_neighbors") else []
            if not neighbors:
                self._bump("safety_recovery_blocked_no_anchor_count")
                return False
            self._bump("safety_recovery_candidate_count")
            target = None
            if hasattr(env, "_truck_recovery_support_target"):
                try:
                    target = env._truck_recovery_support_target(str(aid), [int(x) for x in neighbors])
                except Exception:
                    target = None
            if target is None:
                self._bump("safety_recovery_blocked_no_anchor_count")
                return False
            self._bump("safety_recovery_command_generated_count")
            self._bump("v2_safety_recovery_command_generated_count")
            batch.add_truck(
                TruckCommand(
                    aid,
                    TruckCommandKind.SAFETY_RECOVERY,
                    target_node=int(target),
                    recovery_anchor=int(target),
                    ttl_steps=8,
                    safety_reason="hard_recovery_risk",
                    reason="v2_safety_recovery",
                )
            )
            self._safety_lifecycle_rows.append(
                {
                    "truck_id": str(aid),
                    "recovery_point": int(target),
                    "created_step": int(env.state.step_index),
                    "ttl_steps": 8,
                    "status": "generated",
                }
            )
            return True
        has_hard = bool(getattr(env, "_has_hard_recovery_uav", lambda: False)())
        if not has_hard:
            if self.gate18_safety_narrow_routine_guard:
                self._bump("safety_narrow_suppressed_soft_risk_count")
            self._bump("safety_recovery_blocked_not_hard_risk_count")
            return False
        hard_uavs = self._hard_recovery_uav_ids(env)
        if self.gate18_safety_narrow_routine_guard:
            self._bump("safety_narrow_candidate_count")
            if not hard_uavs:
                self._bump("safety_narrow_suppressed_soft_risk_count")
                self._bump("safety_recovery_blocked_not_hard_risk_count")
                return False
        truck = env.state.agents.get(str(aid))
        if truck is None or getattr(truck, "node", None) is None:
            self._bump("safety_recovery_blocked_no_anchor_count")
            return False
        neighbors = list(env._decision_neighbors(int(truck.node))) if hasattr(env, "_decision_neighbors") else []
        if not neighbors:
            self._bump("safety_recovery_blocked_no_anchor_count")
            return False
        self._bump("safety_recovery_candidate_count")
        target = None
        if hasattr(env, "_truck_recovery_support_target"):
            try:
                target = env._truck_recovery_support_target(str(aid), [int(x) for x in neighbors])
            except Exception:
                target = None
        if target is None:
            self._bump("safety_recovery_blocked_no_anchor_count")
            return False
        uav_id = str(hard_uavs[0]) if hard_uavs else ""
        command_id = self._new_safety_recovery_command_id(env, str(aid), uav_id or "unknown")
        self._bump("safety_recovery_command_generated_count")
        self._bump("safety_recovery_command_created_count")
        self._bump("safety_recovery_anchor_assigned_count")
        self._bump("v2_safety_recovery_command_generated_count")
        if self.gate18_safety_narrow_routine_guard:
            self._bump("safety_narrow_generated_count")
            if any(self._active_sorties.values()):
                self._bump("safety_narrow_interrupted_active_sortie_count")
        if uav_id:
            self._active_safety_recovery_commands[command_id] = {
                "safety_recovery_command_id": command_id,
                "uav_id": uav_id,
                "truck_id": str(aid),
                "recovery_anchor_id": int(target),
                "recovery_point": int(target),
                "created_step": int(env.state.step_index),
                "expires_step": int(env.state.step_index) + 8,
                "uav_returning_step": "",
            }
        batch.add_truck(
            TruckCommand(
                aid,
                TruckCommandKind.SAFETY_RECOVERY,
                target_node=int(target),
                recovery_anchor=int(target),
                ttl_steps=8,
                safety_reason="hard_recovery_risk",
                reason="v2_safety_recovery",
            )
        )
        self._safety_lifecycle_rows.append(
            {
                "safety_recovery_command_id": command_id,
                "uav_id": uav_id,
                "truck_id": str(aid),
                "recovery_anchor_id": int(target),
                "recovery_point": int(target),
                "created_step": int(env.state.step_index),
                "ttl_steps": 8,
                "recovery_reason": "hard_recovery_risk",
                "status_final": "recovery_anchor_assigned",
                "uav_returning_step": "",
                "recovered_step": "",
                "expired_step": "",
                "fail_reason": "",
                "recovered_without_command": 0,
            }
        )
        return True

    def _plan_truck(self, env, oracle: FeasibilityOracle, batch: CommandBatch, aid: str) -> None:
        if self._try_safety_recovery_command(env, batch, aid):
            return
        if self._try_support_command(env, oracle, batch, aid):
            return
        active = self.commitments.active_for_agent(aid)
        if active is not None and str(active.task_id) in env.state.tasks:
            task = env.state.tasks[str(active.task_id)]
            if task.status == TaskStatus.PENDING:
                self._bump("selected_reason_continue_commitment")
                batch.add_truck(TruckCommand(aid, TruckCommandKind.MOVE_TO_TASK, task_id=str(task.task_id), reason="continue_commitment"))
                return
            self.commitments.release(aid, "task_not_pending")

        normals = self._pending_tasks(env, TaskKind.NORMAL)
        task = self._nearest_task(env, aid, normals)
        if task is None:
            batch.add_truck(TruckCommand(aid, TruckCommandKind.HOLD, reason="no_routine_task"))
            return
        near = oracle.evaluate_routine_near_completion(aid, str(task.task_id))
        if near.near_completion:
            self._bump("routine_commitment_count")
            self.commitments.hold(Commitment(aid, str(task.task_id), "routine_service_commitment", int(env.state.step_index)))
        self._bump("selected_reason_routine_completion")
        batch.add_truck(
            TruckCommand(
                aid,
                TruckCommandKind.MOVE_TO_TASK,
                task_id=str(task.task_id),
                target_node=int(task.demand_node),
                reason="routine_completion",
            )
        )

    def _plan_uav(self, env, oracle: FeasibilityOracle, batch: CommandBatch, aid: str) -> None:
        support_rec = self._active_support_for_uav(env, aid) if self.support_recovery_repair else None
        if support_rec is None and self.gate19_support_lock:
            locked = [
                r for r in self._active_support_commitments.values()
                if str(r.get("uav_id", "")) == str(aid) and bool(r.get("support_lock_active", False))
            ]
            if locked:
                self._bump("support_lock_blocked_uav_reassign_count")
                batch.add_uav(UAVCommand(aid, UAVCommandKind.HOLD, reason="support_lock_waiting_for_anchor"))
                return
        if support_rec is not None:
            task_id = str(support_rec.get("task_id", ""))
            feas = oracle.evaluate_uav_sortie(aid, task_id)
            if feas.full_sortie_feasible:
                support_id = str(support_rec.get("support_command_id", ""))
                key = support_id if self.gate17_lifecycle_closure and support_id else f"{support_rec.get('truck_id')}:{aid}:{task_id}"
                if key not in self._support_launch_marked:
                    self._bump("v2_support_command_to_launch_count")
                    self._bump("support_command_launch_triggered_count")
                    if self.gate18_support_anchor_strict:
                        self._bump("strict_anchor_to_launch_count")
                    if self.gate18_anchor_arrival_force_launch:
                        self._bump("anchor_arrival_recheck_count")
                        self._bump("anchor_arrival_launch_forced_count")
                    if self.gate19_launch_binding:
                        self._bump("support_anchor_arrived_launch_success_count")
                    if self.gate20_support_reserve_launch or self.gate20_passenger_invariant:
                        self._bump("support_reserve_to_launch_count")
                    self._support_launch_marked.add(key)
                self._bump("sortie_oracle_to_launch_count")
                self._bump("selected_reason_tc_delivery")
                self.commitments.hold(Commitment(aid, task_id, "tc_sortie_commitment", int(env.state.step_index)))
                sortie_id = self._new_sortie_id(env, aid, task_id)
                support_rec["sortie_id"] = sortie_id
                support_rec["launch_triggered_step"] = int(env.state.step_index)
                support_rec["delivery_deadline_step"] = int(env.state.step_index) + 30
                if support_id:
                    self._support_to_sortie[support_id] = sortie_id
                    self._sortie_to_support[sortie_id] = support_id
                    row = self._support_row(support_id)
                    if row is not None:
                        row["sortie_id"] = sortie_id
                    self._set_support_status(support_id, "launch_triggered", "launch_triggered_step", int(env.state.step_index))
                self._active_sorties[sortie_id] = {
                    "sortie_id": sortie_id,
                    "uav_id": str(aid),
                    "task_id": task_id,
                    "launch_step": int(env.state.step_index),
                    "launch_anchor": int(support_rec.get("support_node", -1)),
                    "recovery_anchor": int(support_rec.get("support_node", -1)),
                    "predicted_full_sortie_feasible": True,
                    "predicted_energy_margin": float(getattr(feas, "energy_margin", 0.0)),
                    "predicted_recovery_margin": float(getattr(feas, "recovery_margin", 0.0)),
                    "predicted_completion_step": float(getattr(feas, "expected_completion_step", 0.0)),
                    "support_command_id": support_id,
                }
                self._sortie_lifecycle_rows.append(
                    {
                        **self._active_sorties[sortie_id],
                        "service_start_step": "",
                        "delivery_step": "",
                        "recovery_step": "",
                        "forced_recovery": 0,
                        "reject_reason": "",
                        "actual_service_start_step": "",
                        "actual_delivery_step": "",
                        "actual_recovery_step": "",
                        "actual_forced_recovery": 0,
                        "actual_reject_reason": "",
                        "lifecycle_status": "launched_no_service",
                        "delivery_link_valid": 0,
                        "duplicate_delivery_blocked": 0,
                        "delivery_without_sortie_blocked": 0,
                    }
                )
                self._bump("sortie_id_created_count")
                batch.add_uav(
                    UAVCommand(
                        aid,
                        UAVCommandKind.LAUNCH_TC,
                        task_id=task_id,
                        support_command_id=support_id or None,
                        sortie_id=sortie_id,
                        launch_anchor=int(support_rec.get("support_node", -1)),
                        recovery_anchor=int(support_rec.get("support_node", -1)),
                        reason="support_anchor_full_sortie_feasible",
                    )
                )
                return
            self._bump("sortie_oracle_prediction_mismatch_count")
            if self.gate19_launch_binding:
                reason = str(getattr(feas, "reject_reason", "unknown"))
                self._fail_support_launch_attempt(support_rec, reason, int(env.state.step_index))
                return
            if self.gate18_anchor_arrival_force_launch:
                self._bump("anchor_arrival_recheck_count")
                self._bump("anchor_arrival_release_infeasible_count")
                reason = str(getattr(feas, "reject_reason", ""))
                if "recovery" in reason or "no_recovery" in reason:
                    self._bump("anchor_arrival_release_reason_recovery_count")
                elif "energy" in reason or "battery" in reason:
                    self._bump("anchor_arrival_release_reason_energy_count")
                else:
                    self._bump("anchor_arrival_release_reason_lifeline_count")

        emergencies = self._pending_tasks(env, TaskKind.EMERGENCY)
        best_task = None
        best_feas = None
        for task in sorted(emergencies, key=lambda t: float(getattr(t, "lifeline_current", 0.0))):
            self._bump("sortie_candidate_count")
            if self.gate18_direct_tc_first:
                self._bump("direct_tc_candidate_count")
            feas = oracle.evaluate_uav_sortie(aid, str(task.task_id))
            if feas.full_sortie_feasible:
                self._bump("full_sortie_feasible_count")
                best_task = task
                best_feas = feas
                break
            reason = str(feas.reject_reason)
            if "recovery" in reason or "no_recovery" in reason:
                self._bump("sortie_blocked_recovery_count")
            elif "battery" in reason or "energy" in reason:
                self._bump("sortie_blocked_energy_count")
            elif feas.expected_lifeline_remaining <= 0:
                self._bump("sortie_blocked_lifeline_count")
        if best_task is None:
            batch.add_uav(UAVCommand(aid, UAVCommandKind.HOLD, reason="no_full_sortie_feasible_tc"))
            return
        self._bump("selected_reason_tc_delivery")
        self.commitments.hold(Commitment(aid, str(best_task.task_id), "tc_sortie_commitment", int(env.state.step_index)))
        if self.gate16_command_quality:
            sortie_id = self._new_sortie_id(env, aid, str(best_task.task_id))
            self._active_sorties[sortie_id] = {
                "sortie_id": sortie_id,
                "uav_id": str(aid),
                "task_id": str(best_task.task_id),
                "launch_step": int(env.state.step_index),
                "launch_anchor": "",
                "recovery_anchor": "",
                "predicted_full_sortie_feasible": True,
                "predicted_energy_margin": float(getattr(best_feas, "energy_margin", 0.0)),
                "predicted_recovery_margin": float(getattr(best_feas, "recovery_margin", 0.0)),
                "predicted_completion_step": float(getattr(best_feas, "expected_completion_step", 0.0)),
                "support_command_id": "",
            }
            self._sortie_lifecycle_rows.append(
                {
                    **self._active_sorties[sortie_id],
                    "service_start_step": "",
                    "delivery_step": "",
                    "recovery_step": "",
                    "forced_recovery": 0,
                    "reject_reason": "",
                    "actual_service_start_step": "",
                    "actual_delivery_step": "",
                    "actual_recovery_step": "",
                    "actual_forced_recovery": 0,
                    "actual_reject_reason": "",
                    "lifecycle_status": "launched_no_service",
                    "delivery_link_valid": 0,
                    "duplicate_delivery_blocked": 0,
                    "delivery_without_sortie_blocked": 0,
                }
            )
            self._bump("sortie_id_created_count")
        if self.gate18_direct_tc_first:
            self._bump("direct_tc_launch_generated_count")
        batch.add_uav(UAVCommand(aid, UAVCommandKind.LAUNCH_TC, task_id=str(best_task.task_id), reason="full_sortie_feasible"))

    def _attach_batch(self, env, batch: CommandBatch) -> None:
        env._erc_v2_command_gate_enabled = True
        env._erc_v2_command_batch = batch
        support_count = sum(
            1
            for cmd in batch.truck_commands.values()
            if cmd.kind in {TruckCommandKind.SUPPORT_UAV, TruckCommandKind.SAFETY_RECOVERY}
        )
        safety_count = sum(1 for cmd in batch.truck_commands.values() if cmd.kind == TruckCommandKind.SAFETY_RECOVERY)
        self._bump("support_command_count", support_count)
        self._bump("safety_recovery_command_count", safety_count)
        self._bump("truck_command_count", len(batch.truck_commands))
        self._bump("uav_command_count", len(batch.uav_commands))
        for cmd in batch.truck_commands.values():
            if cmd.kind == TruckCommandKind.HOLD:
                self._bump("truck_hold_count")
            elif cmd.kind == TruckCommandKind.SUPPORT_UAV:
                self._bump("truck_support_uav_count")
            elif cmd.kind == TruckCommandKind.SAFETY_RECOVERY:
                self._bump("truck_safety_recovery_count")
            elif cmd.reason == "continue_commitment":
                self._bump("truck_continue_routine_count")
            elif cmd.kind == TruckCommandKind.MOVE_TO_TASK:
                self._bump("truck_go_to_routine_count")
        for cmd in batch.uav_commands.values():
            if cmd.kind == UAVCommandKind.LAUNCH_TC:
                self._bump("uav_launch_tc_count")
                self._bump("sortie_oracle_to_launch_count")
            elif cmd.kind == UAVCommandKind.RETURN_OR_RTH:
                self._bump("uav_return_to_anchor_count")
            elif cmd.kind == UAVCommandKind.HOLD:
                self._bump("uav_hold_count")
        env.support_command_count = int(support_count)
        env.safety_recovery_command_count = int(safety_count)
        env.v2_sortie_lifecycle_rows = list(self._sortie_lifecycle_rows)
        env.v2_support_lifecycle_rows = list(self._support_lifecycle_rows)
        env.v2_safety_lifecycle_rows = list(self._safety_lifecycle_rows)
        env.v2_support_passenger_trace_rows = list(self._support_passenger_trace_rows)
        env.v2_support_quality_rows = list(self._support_quality_rows)

    def plan(self, env) -> Dict[str, Optional[str]]:
        _ = WorldState.from_env(env)
        self._observe_support_commitments(env)
        oracle = FeasibilityOracle(env)
        batch = CommandBatch()
        residual = CommitmentManager.residual_followup_candidates(env.state.tasks.values())
        self.tc_residual_followup_candidate_count = float(len(residual))

        for aid, state in env.state.agents.items():
            if bool(getattr(state, "crashed", False)):
                continue
            if state.kind == AgentKind.TRUCK:
                self._plan_truck(env, oracle, batch, str(aid))
            elif state.kind == AgentKind.UAV:
                self._plan_uav(env, oracle, batch, str(aid))

        self.event_detected_count = float(self.event_gate.event_detected_count)
        self.event_impact_positive_count = float(self.event_gate.event_impact_positive_count)
        self.event_impact_none_count = float(self.event_gate.event_impact_none_count)
        self.weak_event_suppressed_count = float(self.event_gate.weak_event_suppressed_count)
        self.forced_switch_blocked_count = float(self.event_gate.forced_switch_blocked_count)
        self.local_command_generated_count = float(len(list(batch.commands())))
        self.global_replan_count = 0.0
        raw_launch = float(getattr(env, "uav_launch_count_total", 0.0))
        raw_delivery = float(getattr(env, "uav_delivery_count_total", 0.0))
        self.launch_to_completion_ratio_raw = float(raw_delivery / max(raw_launch, 1.0))
        if raw_launch > float(self.sortie_launch_success_count):
            self.sortie_launch_success_count = float(raw_launch)
        self.launch_to_completion_ratio_checked = float(
            min(float(self.sortie_delivery_success_count), raw_delivery, raw_launch) / max(raw_launch, 1.0)
        )
        self._last_batch = batch
        self._attach_batch(env, batch)
        return batch.goals()
