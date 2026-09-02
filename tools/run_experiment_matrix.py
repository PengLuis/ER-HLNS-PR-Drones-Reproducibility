from __future__ import annotations

# ruff: noqa: E402

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import argparse
from collections import defaultdict
import csv
from dataclasses import asdict, is_dataclass, replace
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple
from importlib import metadata as importlib_metadata

import numpy as np
import pandas as pd
import yaml

from hetgat_hrl.agents.actor_critic import RuleBasedLowLevelPolicy
from hetgat_hrl.core.mdp_spec import AgentKind, EnvConfig, TaskKind, TaskStatus
from hetgat_hrl.eval.metric_schema import (
    BLOCKAGE_V2_METRIC_FIELDS,
    ISLAND_SUPPORT_METRIC_FIELDS,
    PLANNER_CONVERSION_METRIC_FIELDS,
    SEED_METRIC_FIELDS,
    STAGE_C_METRIC_FIELDS,
    SUPPORT_BIND_METRIC_FIELDS,
    TASK_OUTCOME_METRIC_FIELDS,
    TASK_SEMANTIC_METRIC_FIELDS,
    TIMECRITICAL_PRIORITY_METRIC_FIELDS,
    SUPPLY_DERIVED_FIELDS,
    SUPPLY_METRIC_FIELDS,
    UAV_SAFETY_METRIC_FIELDS,
    UAV_STRICT_SAFETY_FIELDS,
)
from hetgat_hrl.envs.base_env import BaseHeteroDisasterEnv
from hetgat_hrl.eval.metric_schema import MAINLINE_DIAG_FIELDS
from tools.experiment_matrix_methods import (
    SUPPORTED_METHODS,
    _build_algorithm_package,
    _canonical_method_name,
    _method_ablation_flags,
    _method_backend_family,
    _scenario_cfg,
)


ROUTEA_MAPUPDATE_DIAG_FIELDS = [
    "map_update_hard_seen_count",
    "map_update_hard_actionable_count",
    "map_update_hard_deferred_count",
    "map_update_hard_immediate_refresh_count",
    "map_update_hard_reason_path_blocked_count",
    "map_update_hard_reason_goal_unreachable_count",
    "map_update_hard_reason_ranking_changed_count",
    "map_update_hard_reason_dead_end_count",
    "map_update_hard_reason_recovery_path_fractured_count",
]

PHYSICAL_V2_STRING_FIELDS = [
    "physical_environment_version",
    "physical_v2_fairness_hash",
    "physical_v2_road_digest",
    "physical_v2_weather_digest",
    "physical_v2_shield_reason_counts",
]

PHYSICAL_V2_METRIC_FIELDS = [
    "physical_v2_enabled",
    "physical_v2_energy_eval_count",
    "physical_v2_travel_eval_count",
    "physical_v2_blocked_entry_count",
    "physical_v2_shield_intervention_count",
    "physical_v2_unsafe_plan_proposal_count",
    "physical_v2_uav_drop_count",
    "physical_v2_launch_check_count",
    "physical_v2_service_start_count",
    "physical_v2_service_complete_count",
    "physical_v2_recovery_event_count",
    "physical_v2_forced_landing_count",
    "physical_v2_uav_drop_runtime_count",
    "physical_v2_service_reject_count",
    "physical_v2_energy_deduction_count",
    "physical_v2_lifecycle_transition_count",
    "physical_v2_recovery_motion_step_count",
    "physical_v2_recovery_bind_count",
    "physical_v2_lifecycle_ledger_count",
    "physical_v2_energy_ledger_count",
    "physical_v2_energy_fraction_total",
    "physical_v2_minimum_energy_reserve_seen",
    "physical_v2_road_event_ledger_count",
    "physical_v2_weather_ledger_count",
    "physical_v2_unique_unsafe_proposal_count",
    "physical_v2_duplicate_unsafe_check_count",
    "physical_v2_interventions_per_unique_proposal",
    "physical_v2_interventions_per_mission",
    "UAV_DROP",
    "FORCED_LANDING",
    "EMERGENCY_RETURN",
    "MISSION_ABORT",
    "UNSAFE_PROPOSAL",
    "UNIQUE_SHIELD_INTERVENTION",
    "DUPLICATE_SHIELD_CHECK",
    "MINIMUM_ENERGY_RESERVE",
    "ENERGY_EXHAUSTION",
    "RECOVERY_FAILURE",
    "WEATHER_LOSS_OF_CONTROL",
]


LIGHTWEIGHT_AUDIT_FLAG_FIELDS = [
    "enable_switch_decision_ledger",
    "enable_step_trace",
    "enable_event_ledger_detail",
    "enable_agent_step_html",
    "enable_top_offender_export",
    "enable_timeline_metrics",
    "enable_debug_transition_dump",
    "enable_per_step_audit",
    "enable_task_outcome_export",
]

TASK_OUTCOME_FIELDS = [
    "scenario",
    "seed",
    "method",
    "episode",
    "task_id",
    "task_kind",
    "task_class",
    "demand_kg",
    "urgency_score",
    "lifeline_init",
    "lifeline_final",
    "deadline_step",
    "deadline_seconds",
    "assigned_count",
    "first_assigned_step",
    "last_assigned_step",
    "service_start_count",
    "first_service_step",
    "completed",
    "completed_step",
    "completed_seconds",
    "on_time",
    "completion_quality",
    "fulfilled_mass_kg",
    "remaining_demand_kg",
    "remaining_lifeline_at_service",
    "completed_by_agent",
    "failed",
    "failed_reason",
    "failed_step",
    "final_status",
    "nearest_truck_distance_start",
    "nearest_uav_or_truck_distance_start",
    "truck_reachable",
    "uav_eventual_serviceable",
]


INVALID_ACTION_LEDGER_FIELDS = [
    "scenario",
    "method",
    "seed",
    "episode_index",
    "step",
    "agent_id",
    "agent_type",
    "agent_state",
    "action_type",
    "raw_action",
    "normalized_action",
    "current_node",
    "target_node",
    "task_id",
    "task_status",
    "planner_goal",
    "support_binding",
    "validation_layer",
    "reason_code",
    "reason_detail",
    "planner_state_digest",
    "environment_state_digest",
    "local_repair_attempted",
    "local_repair_succeeded",
    "fallback_action",
    "source_code_location",
]


REPRO_DIGEST_FIELDS = [
    "scenario_digest",
    "road_event_digest",
    "task_generation_digest",
    "initial_agent_state_digest",
    "task_digest",
    "initial_state_digest",
    "agent_init_digest",
    "effective_config_hash",
    "code_commit_hash",
    "comm_blackout_zone_digest",
]

PHYSICAL_FREEZE_EXPORT_FIELDS = [
    "scenario_name",
    "scenario_type",
    "scenario_config_digest",
    "road_digest",
    "weather_digest",
    "task_digest",
    "initial_state_digest",
    "agent_init_digest",
    "physical_config_hash",
    "safety_config_hash",
    "physical_environment_version",
    "safety_protocol",
    "method",
    "seed",
    "protocol",
]

PHYSICAL_FREEZE_FAIR_CORE_FIELDS = [
    "routine_bulk_lifeline_decay_base",
    "time_critical_lightweight_lifeline_decay_base",
    "task_lifeline_hazard_weight",
    "truck_initial_bulk_inventory_kg",
    "truck_initial_timecritical_inventory_kg",
    "normal_task_demand_kg",
    "emergency_task_demand_kg",
    "bulk_supply_unit_kg",
    "timecritical_supply_unit_kg",
    "time_critical_package_kg",
    "truck_initial_normal_supply_units",
    "truck_initial_emergency_supply_units",
    "truck_payload_capacity_kg",
    "uav_payload_kg",
    "uav_payload_capacity_kg",
    "uav_self_weight_kg",
    "uav_max_emergency_units",
    "max_standard_packages_per_truck",
    "uav_start_docked_on_truck",
]

PHYSICAL_FREEZE_PHYSICAL_CONFIG_FIELDS = [
    "physical_environment_version",
    "map_size_m",
    "truck_speed_mps",
    "truck_capacity_kg",
    # Authoritative kg fields used by the 800/150 kg paper contract. Legacy
    # aliases above remain exported for archived-run compatibility.
    "truck_payload_capacity_kg",
    "uav_speed_mps",
    "uav_payload_kg",
    "uav_payload_capacity_kg",
    "uav_self_weight_kg",
    "uav_battery_capacity",
    "uav_energy_per_m",
    "uav_energy_per_kg_m",
    "uav_hover_energy_per_s",
    "uav_reserve_ratio",
    "uav_recovery_distance_buffer_m",
    "uav_high_pressure_recovery_margin_bonus_m",
    "truck_recovery_request_min_urgency_when_normal_pending",
    "truck_recovery_require_request_when_normal_pending",
    "routine_bulk_lifeline_decay_base",
    "time_critical_lightweight_lifeline_decay_base",
    "task_lifeline_hazard_weight",
    "truck_initial_bulk_inventory_kg",
    "truck_initial_timecritical_inventory_kg",
    "normal_task_demand_kg",
    "emergency_task_demand_kg",
    "bulk_supply_unit_kg",
    "timecritical_supply_unit_kg",
    "time_critical_package_kg",
    "truck_initial_normal_supply_units",
    "truck_initial_emergency_supply_units",
    "uav_max_emergency_units",
    "max_standard_packages_per_truck",
    "uav_start_docked_on_truck",
]

PHYSICAL_FREEZE_SAFETY_CONFIG_FIELDS = [
    "physical_environment_safety_protocol",
    "uav_hard_recovery_battery_guard",
    "uav_allow_rendezvous_launch",
    "uav_rendezvous_launch_requires_docked_truck_goal",
    "uav_launch_min_horizon_buffer_steps",
    "hrl_tc_override_min_recovery_margin_m",
    "hrl_tc_override_min_battery_margin_ratio",
    "uav_reject_cache_window_steps",
    "truck_support_uav_recovery_enabled",
]

PHYSICAL_FREEZE_FAIR_CONFIG_FIELDS = list(
    dict.fromkeys(
        [
            *PHYSICAL_FREEZE_FAIR_CORE_FIELDS,
            *[field for field in PHYSICAL_FREEZE_PHYSICAL_CONFIG_FIELDS if field != "physical_environment_version"],
            *[field for field in PHYSICAL_FREEZE_SAFETY_CONFIG_FIELDS if field != "physical_environment_safety_protocol"],
        ]
    )
)


def _stable_normalize(value: Any) -> Any:
    if is_dataclass(value):
        return _stable_normalize(asdict(value))
    if isinstance(value, dict):
        return {str(k): _stable_normalize(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_stable_normalize(v) for v in value]
    if isinstance(value, set):
        return [_stable_normalize(v) for v in sorted(value, key=lambda x: repr(x))]
    if isinstance(value, np.ndarray):
        return _stable_normalize(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        if math.isfinite(value):
            return round(float(value), 12)
        return str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if hasattr(value, "value"):
        return str(value.value)
    if hasattr(value, "__dict__"):
        return _stable_normalize(vars(value))
    return str(value)


def _stable_hash(payload: Any) -> str:
    text = json.dumps(_stable_normalize(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_commit_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return str(out).strip()
    except Exception:
        return "UNKNOWN"


def _sha256_file_if_present(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""
    except OSError:
        return ""


def _runtime_provenance(input_paths: List[Path]) -> Dict[str, Any]:
    dependency_versions: Dict[str, str] = {}
    for distribution in ("numpy", "networkx", "pandas", "scipy", "PyYAML", "torch"):
        try:
            dependency_versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            dependency_versions[distribution] = "not-installed"
    memory_bytes = 0
    try:
        import psutil  # type: ignore

        memory_bytes = int(psutil.virtual_memory().total)
    except Exception:
        pass
    gpu_name = "not-used"
    try:
        import torch

        if bool(torch.cuda.is_available()):
            gpu_name = str(torch.cuda.get_device_name(0))
    except Exception:
        pass
    files = {}
    for path in input_paths:
        resolved = path.resolve()
        files[str(resolved)] = _sha256_file_if_present(resolved)
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": int(os.cpu_count() or 0),
        "memory_bytes": int(memory_bytes),
        "gpu": gpu_name,
        "execution_worker_count": 1,
        "dependency_versions": dependency_versions,
        "input_file_sha256": files,
    }


def _topology_digest_payload(env: BaseHeteroDisasterEnv) -> Dict[str, Any]:
    topo = getattr(env, "topology", None)
    cfg = getattr(env, "cfg", None)
    return {
        "nodes": getattr(topo, "nodes", None),
        "edges": getattr(topo, "edges", None),
        "depot_node": getattr(env, "depot_node", None),
        "blocked_edges_initial": getattr(topo, "blocked_edges", None),
        "scenario_name": f"{str(getattr(cfg, 'scale', '')).upper()}-{str(getattr(cfg, 'scenario', '')).upper()}",
        "scenario_type": getattr(cfg, "scale", None),
        "cfg_scale": getattr(cfg, "scale", None),
        "cfg_scenario": getattr(cfg, "scenario", None),
        "map_size_m": getattr(cfg, "map_size_m", None),
        "num_nodes": getattr(cfg, "num_nodes", None),
        "num_edges": getattr(cfg, "num_edges", None),
        "num_trucks": getattr(cfg, "num_trucks", None),
        "num_uavs": getattr(cfg, "num_uavs", None),
        "num_normal_tasks": getattr(cfg, "num_normal_tasks", None),
        "num_emergency_tasks": getattr(cfg, "num_emergency_tasks", None),
        "num_routine_bulk_tasks": getattr(cfg, "num_routine_bulk_tasks", None),
        "num_time_critical_lightweight_tasks": getattr(cfg, "num_time_critical_lightweight_tasks", None),
        "road_damage_enabled": getattr(cfg, "road_damage_enabled", None),
        "weather_enabled": getattr(cfg, "weather_enabled", None),
        "physical_environment_version": getattr(cfg, "physical_environment_version", None),
        "physical_environment_safety_protocol": getattr(cfg, "physical_environment_safety_protocol", None),
    }


def _task_digest_payload(env: BaseHeteroDisasterEnv) -> Any:
    return {
        tid: _stable_normalize(task)
        for tid, task in sorted(getattr(env.state, "tasks", {}).items(), key=lambda kv: str(kv[0]))
    }


def _agent_digest_payload(env: BaseHeteroDisasterEnv) -> Any:
    return {
        aid: _stable_normalize(agent)
        for aid, agent in sorted(getattr(env.state, "agents", {}).items(), key=lambda kv: str(kv[0]))
    }


def _road_event_digest_payload(env: BaseHeteroDisasterEnv) -> Dict[str, Any]:
    hazards = getattr(env, "hazards", None)
    hazard_payload: Dict[str, Any] = {}
    if hazards is not None:
        for key, value in vars(hazards).items():
            low = str(key).lower()
            if any(token in low for token in ("road", "edge", "block", "reopen", "event")):
                hazard_payload[str(key)] = _stable_normalize(value)
    return {
        "hazards": hazard_payload,
        "topology_blocked_edges": getattr(getattr(env, "topology", None), "blocked_edges", None),
        "shared_known_blocked_edges": getattr(env, "_shared_known_blocked_edges", None),
    }


def _reproducibility_digests(env: BaseHeteroDisasterEnv) -> Dict[str, str]:
    task_digest = _stable_hash(_task_digest_payload(env))
    agent_digest = _stable_hash(_agent_digest_payload(env))
    initial_state_digest = _stable_hash(
        {
            "tasks": _task_digest_payload(env),
            "agents": _agent_digest_payload(env),
        }
    )
    return {
        "scenario_digest": _stable_hash(_topology_digest_payload(env)),
        "road_event_digest": _stable_hash(_road_event_digest_payload(env)),
        "task_generation_digest": task_digest,
        "initial_agent_state_digest": agent_digest,
        "task_digest": task_digest,
        "initial_state_digest": initial_state_digest,
        "agent_init_digest": agent_digest,
        "effective_config_hash": _stable_hash(env.cfg),
        "code_commit_hash": _git_commit_hash(),
    }


def _physical_freeze_contract_digests(env: BaseHeteroDisasterEnv) -> Dict[str, str]:
    cfg = env.cfg
    task_payload = _task_digest_payload(env)
    agent_payload = _agent_digest_payload(env)
    topology_payload = _topology_digest_payload(env)
    scenario_contract_payload = {
        key: value
        for key, value in topology_payload.items()
        if key not in {"physical_environment_safety_protocol"}
    }
    physical_config_payload = {
        field: getattr(cfg, field, None)
        for field in PHYSICAL_FREEZE_PHYSICAL_CONFIG_FIELDS
    }
    safety_config_payload = {
        "physical_environment_safety_protocol": str(getattr(cfg, "physical_environment_safety_protocol", "")),
        **{field: getattr(cfg, field, None) for field in PHYSICAL_FREEZE_SAFETY_CONFIG_FIELDS},
    }
    return {
        "scenario_config_digest": _stable_hash(scenario_contract_payload),
        "road_digest": str(getattr(env, "physical_v2_road_digest", "") or _stable_hash(_road_event_digest_payload(env))),
        "weather_digest": str(getattr(env, "physical_v2_weather_digest", "")),
        "task_digest": _stable_hash(task_payload),
        "initial_state_digest": _stable_hash({"tasks": task_payload, "agents": agent_payload}),
        "agent_init_digest": _stable_hash(agent_payload),
        "physical_config_hash": _stable_hash(physical_config_payload),
        "safety_config_hash": _stable_hash(safety_config_payload),
        "physical_environment_version": str(getattr(cfg, "physical_environment_version", "")),
        "safety_protocol": str(getattr(cfg, "physical_environment_safety_protocol", "")),
    }


def _physical_freeze_export_aliases(
    row: Dict[str, Any],
    *,
    cfg: EnvConfig,
    scale: str,
    scenario: str,
    method: str,
    seed: int,
) -> Dict[str, Any]:
    scenario_name = f"{str(scale).upper()}-{str(scenario).upper()}"
    protocol = str(getattr(cfg, "physical_environment_safety_protocol", ""))
    physical_version = str(getattr(cfg, "physical_environment_version", ""))
    fairness_hash = str(row.get("physical_v2_fairness_hash", ""))
    physical_config_payload = {
        "physical_v2_fairness_hash": fairness_hash,
        **{field: getattr(cfg, field, None) for field in PHYSICAL_FREEZE_PHYSICAL_CONFIG_FIELDS},
    }
    safety_config_payload = {
        "physical_v2_fairness_hash": fairness_hash,
        **{field: getattr(cfg, field, None) for field in PHYSICAL_FREEZE_SAFETY_CONFIG_FIELDS},
    }
    return {
        "scenario_name": scenario_name,
        "scenario_type": str(scale).upper(),
        "scenario_config_digest": str(row.get("scenario_config_digest", "") or row.get("scenario_digest", "")),
        "road_digest": str(row.get("road_digest", "") or row.get("physical_v2_road_digest", "") or row.get("road_event_digest", "")),
        "weather_digest": str(row.get("weather_digest", "") or row.get("physical_v2_weather_digest", "")),
        "task_digest": str(row.get("task_digest", "") or row.get("task_generation_digest", "")),
        "initial_state_digest": str(row.get("initial_state_digest", "")),
        "agent_init_digest": str(row.get("agent_init_digest", "") or row.get("initial_agent_state_digest", "")),
        "physical_config_hash": str(row.get("physical_config_hash", "") or _stable_hash(physical_config_payload)),
        "safety_config_hash": str(row.get("safety_config_hash", "") or _stable_hash(safety_config_payload)),
        "physical_environment_version": physical_version,
        "safety_protocol": protocol,
        "method": str(method).strip().lower(),
        "seed": int(seed),
        "protocol": protocol,
    }


def _jsonable_invalid_record(record: Any) -> Dict[str, Any]:
    if hasattr(record, "to_dict"):
        data = record.to_dict()
    elif isinstance(record, dict):
        data = dict(record)
    else:
        data = {"reason_detail": str(record)}
    out: Dict[str, Any] = {}
    for key in INVALID_ACTION_LEDGER_FIELDS:
        value = data.get(key, "")
        if isinstance(value, (dict, list, tuple, set)):
            out[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            out[key] = value
    return out


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_results_root() -> str:
    return str(_default_project_root() / "artifacts" / "training_results")


def _default_config_path() -> str:
    return str(_default_project_root() / "configs" / "paper_eval_matrix.yaml")


def _default_base_config_path() -> str:
    return str(_default_project_root() / "configs" / "paper_train_base.yaml")


def _parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _parse_seed_list(raw: str) -> List[int]:
    return [int(x) for x in _parse_csv_list(raw)]


def _parse_optional_bool(raw: str) -> Optional[bool]:
    txt = str(raw).strip().lower()
    if txt == "":
        return None
    if txt in {"1", "true", "yes", "on"}:
        return True
    if txt in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid bool: {raw!r}")


def _mean_std(vals: List[float]) -> Tuple[float, float]:
    xs = [float(v) for v in vals if np.isfinite(float(v))]
    if not xs:
        return float("nan"), float("nan")
    mu = float(sum(xs) / max(len(xs), 1))
    if len(xs) <= 1:
        return mu, 0.0
    var = float(sum((x - mu) ** 2 for x in xs) / float(len(xs) - 1))
    return mu, float(var ** 0.5)


def _write_erc_ablation_outputs(run_dir: Path, seed_rows: List[Dict[str, Any]]) -> None:
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_csv = tables_dir / "Table_erc_mechanism_ablation_summary.csv"
    md_path = Path("docs") / "ERC_MECHANISM_ABLATION_FIRST_PASS.md"

    # Requested field names (keep exact names even if some are unavailable).
    metrics = [
        "overall_completion_rate",
        "routine_bulk_completion_rate",
        "time_critical_lightweight_completion_rate",
        "weighted_service_score",
        "bulk_fulfilled_mass_ratio",
        "uav_goal_assigned_count_total",
        "uav_launch_count_total",
        "uav_delivery_count_total",
        "goal_assignment_to_launch_ratio",
        "launch_to_completion_ratio",
        "uav_rejected_total",
        "reject_reason_insufficient_recovery_margin",
        "reject_reason_corridor",
        "reject_reason_comm_block",
        "reject_reason_energy_infeasible",
        "goal_switch_count_total",
        "goal_switch_candidate_count",
        "goal_switch_accepted_count",
        "goal_switch_rejected_by_threshold_count",
        "goal_switch_forced_count",
        "triggered_replans",
        "event_replans_in_window",
        "planner_replan_due_to_new_road_info",
        "map_update_hard_immediate_refresh_count",
        "map_update_hard_deferred_count",
        "map_update_light_count",
        "inference_latency_mean_ms",
        "makespan_seconds",
        "total_distance_m",
        "truck_total_distance_m",
        "uav_total_distance_m",
        "crash_count",
        "uav_survival_rate",
        "uav_low_battery_illegal_launch_count_total",
    ]
    source_alias = {
        "uav_rejected_total": "uav_unsafe_launch_attempt_count_total",
        "reject_reason_insufficient_recovery_margin": "uav_reject_cache_reason_insufficient_recovery_margin",
        "reject_reason_corridor": "uav_reject_cache_reason_corridor",
        "reject_reason_comm_block": "uav_reject_cache_reason_comm_block",
        "reject_reason_energy_infeasible": "uav_reject_cache_reason_energy_infeasible",
        "planner_replan_due_to_new_road_info": "planner_replan_due_to_new_road_info_count_total",
        "map_update_hard_immediate_refresh_count": "map_update_hard_immediate_refresh_count",
        "map_update_hard_deferred_count": "map_update_hard_deferred_count",
        "map_update_light_count": "map_update_hard_seen_count",
        "total_distance_m": "fleet_distance_total_m",
        "truck_total_distance_m": "truck_distance_total_m",
        "uav_total_distance_m": "uav_distance_total_m",
    }

    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in seed_rows:
        k = (str(r.get("model", "")), f"{str(r.get('scale', '')).upper()}-{str(r.get('scenario', '')).upper()}")
        groups.setdefault(k, []).append(r)

    rows_out: List[Dict[str, Any]] = []
    for (method, scenario), rows in sorted(groups.items()):
        out: Dict[str, Any] = {"method": method, "scenario": scenario, "n": len(rows)}
        for m in metrics:
            src = source_alias.get(m, m)
            vals = [float(rr.get(src, float("nan"))) for rr in rows]
            mu, sd = _mean_std(vals)
            out[f"{m}_mean"] = mu
            out[f"{m}_std"] = sd
        rows_out.append(out)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["method", "scenario", "n"] + [x for m in metrics for x in (f"{m}_mean", f"{m}_std")]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    # Markdown first-pass analysis.
    baseline = {}
    for r in rows_out:
        if r["method"] == "erc_full":
            baseline[r["scenario"]] = r

    lines: List[str] = []
    lines.append("# ERC Mechanism Ablation First Pass")
    lines.append("")
    lines.append(f"- Source table: `{out_csv.as_posix()}`")
    lines.append("")
    for scen in sorted({r["scenario"] for r in rows_out}):
        lines.append(f"## {scen}")
        b = baseline.get(scen, None)
        if b is None:
            lines.append("- Missing `erc_full` baseline for this scenario.")
            lines.append("")
            continue
        lines.append("| method | overall 螖 | time_critical 螖 | weighted 螖 | goal_switch 螖 | latency 螖 | chain hint | suggestion |")
        lines.append("|---|---:|---:|---:|---:|---:|---|---|")
        for r in [x for x in rows_out if x["scenario"] == scen and x["method"] != "rolling_fixed"]:
            d_overall = float(r["overall_completion_rate_mean"] - b["overall_completion_rate_mean"])
            d_tc = float(r["time_critical_lightweight_completion_rate_mean"] - b["time_critical_lightweight_completion_rate_mean"])
            d_w = float(r["weighted_service_score_mean"] - b["weighted_service_score_mean"])
            d_sw = float(r["goal_switch_count_total_mean"] - b["goal_switch_count_total_mean"])
            d_lat = float(r["inference_latency_mean_ms_mean"] - b["inference_latency_mean_ms_mean"])
            launch_ratio = float(r["goal_assignment_to_launch_ratio_mean"])
            lc_ratio = float(r["launch_to_completion_ratio_mean"])
            chain_hint = f"assign_to_launch={launch_ratio:.3f}, launch_to_completion={lc_ratio:.3f}"
            if d_tc >= 0 and d_w >= 0 and d_sw <= 0:
                sug = "keep"
            elif d_tc < 0 and d_w < 0:
                sug = "delete_or_default_off"
            elif d_sw < 0 and d_tc >= -0.01:
                sug = "weaken_then_keep"
            else:
                sug = "conditional_enable"
            lines.append(
                f"| {r['method']} | {d_overall:+.4f} | {d_tc:+.4f} | {d_w:+.4f} | {d_sw:+.2f} | {d_lat:+.2f} | {chain_hint} | {sug} |"
            )
        lines.append("")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")


def _write_erc_rhc_simplified_vs_rolling(run_dir: Path, seed_rows: List[Dict[str, Any]]) -> None:
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_csv = tables_dir / "Table_erc_rhc_simplified_vs_rolling.csv"
    metrics = [
        "overall_completion_rate",
        "routine_bulk_completion_rate",
        "time_critical_lightweight_completion_rate",
        "weighted_service_score",
        "bulk_fulfilled_mass_ratio",
        "goal_switch_count_total",
        "triggered_replans",
        "uav_goal_assigned_count_total",
        "uav_launch_count_total",
        "uav_delivery_count_total",
        "goal_assignment_to_launch_ratio",
        "launch_to_completion_ratio",
        "uav_rejected_total",
        "inference_latency_mean_ms",
        "crash_count",
        "uav_survival_rate",
    ]
    source_alias = {
        "uav_rejected_total": "uav_unsafe_launch_attempt_count_total",
        "inference_latency_mean_ms": "planner_inference_latency_mean_ms",
    }
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in seed_rows:
        method = str(r.get("model", "")).strip().lower()
        if method not in {"rolling_fixed", "erc_rhc"}:
            continue
        scenario = f"{str(r.get('scale', '')).upper()}-{str(r.get('scenario', '')).upper()}"
        groups.setdefault((method, scenario), []).append(r)

    if not groups:
        return

    rows_out: List[Dict[str, Any]] = []
    for (method, scenario), rows in sorted(groups.items()):
        out: Dict[str, Any] = {"method": method, "scenario": scenario, "n": len(rows)}
        for m in metrics:
            src = source_alias.get(m, m)
            vals = [float(rr.get(src, float("nan"))) for rr in rows]
            mu, sd = _mean_std(vals)
            out[f"{m}_mean"] = mu
            out[f"{m}_std"] = sd
        rows_out.append(out)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["method", "scenario", "n"] + [x for m in metrics for x in (f"{m}_mean", f"{m}_std")]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    print(f"Saved: {out_csv}")


def _write_compare_rolling_olderc_newerc(run_dir: Path, seed_rows: List[Dict[str, Any]]) -> None:
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_csv = tables_dir / "Table_compare_rolling_olderc_newerc_event_admission_10seed.csv"
    out_csv_legacy = tables_dir / "Table_compare_rolling_olderc_newerc_10seed.csv"
    out_csv_hard_denoise = tables_dir / "Table_compare_rolling_olderc_newerc_hard_event_denoise.csv"
    out_csv_truck_path_localize = tables_dir / "Table_compare_rolling_olderc_newerc_truck_path_localize.csv"
    out_csv_progress_mr = tables_dir / "Table_compare_olderc_newerc_progress_aware_switch_MR.csv"
    out_csv_progress_l = tables_dir / "Table_compare_olderc_newerc_progress_aware_switch_L.csv"
    target_methods = {"rolling_fixed", "erc_rhc_old", "erc_rhc"}
    metrics = [
        # Performance
        "overall_completion_rate",
        "routine_bulk_completion_rate",
        "time_critical_lightweight_completion_rate",
        "weighted_service_score",
        "bulk_fulfilled_mass_ratio",
        # Refresh & switching
        "refresh_total_count",
        "fixed_interval_refresh_count",
        "event_refresh_count",
        "event_refresh_no_goal_change_count",
        "event_refresh_goal_change_count",
        "event_refresh_goal_change_ratio",
        "event_refresh_to_completion_ratio",
        "event_refresh_followed_by_reject_count",
        "no_event_fallback_refresh_count",
        "triggered_replans",
        "goal_switch_count_total",
        "goal_switch_accepted_count",
        "goal_switch_forced_count",
        "harmful_switch_proxy_count",
        "missed_switch_proxy_count",
        "hard_event_refresh_count_total",
        "hard_event_refresh_no_goal_change_count",
        "hard_event_refresh_goal_change_count",
        "hard_event_refresh_to_completion_count",
        "truck_dead_end_candidate_count",
        "truck_dead_end_local_path_repair_count",
        "truck_dead_end_local_goal_reassign_count",
        "truck_dead_end_global_refresh_count",
        "truck_dead_end_noop_count",
        "truck_dead_end_routine_localized_count",
        "truck_dead_end_emergency_kept_hard_count",
        "truck_dead_end_support_kept_hard_count",
        "truck_dead_end_recovery_kept_hard_count",
        "truck_dead_end_local_repair_no_goal_change_count",
        "truck_dead_end_global_refresh_no_goal_change_count",
        "path_blocked_candidate_count",
        "path_blocked_nonimpact_suppressed_count",
        "path_blocked_local_path_repair_count",
        "path_blocked_local_goal_reassign_count",
        "path_blocked_global_refresh_count",
        "path_blocked_noop_count",
        "path_blocked_routine_localized_count",
        "path_blocked_emergency_kept_hard_count",
        "path_blocked_recovery_kept_hard_count",
        "path_blocked_support_kept_hard_count",
        "path_blocked_goal_unreachable_kept_hard_count",
        "path_blocked_local_repair_no_goal_change_count",
        "path_blocked_global_refresh_no_goal_change_count",
        "uav_emergency_commit_hold_count",
        "uav_emergency_commit_break_hard_invalid_count",
        "uav_emergency_commit_prevented_switch_count",
        "uav_emergency_commit_followed_by_launch_count",
        "uav_emergency_commit_followed_by_delivery_count",
        "truck_routine_stuck_candidate_count",
        "truck_routine_stuck_escape_count",
        "truck_routine_stuck_escape_blocked_no_alt_count",
        "truck_routine_stuck_escape_blocked_insufficient_gain_count",
        "truck_routine_stuck_escape_followed_by_service_count",
        "truck_routine_stuck_escape_followed_by_completion_count",
        "routine_localize_eta_check_count",
        "routine_localize_keep_current_count",
        "routine_localize_escape_by_eta_worse_count",
        "routine_localize_escape_followed_by_service_count",
        "routine_localize_escape_followed_by_completion_count",
        "normal_stall_candidate_count",
        "normal_stall_global_refresh_count",
        "goal_invalid_hard_count",
        "goal_invalid_soft_count",
        "goal_invalid_soft_suppressed_count",
        "uav_recovery_hard_count",
        "uav_recovery_soft_count",
        "uav_recovery_global_refresh_count",
        "suspect_soft_as_hard_count",
        # Execution chain
        "uav_goal_assigned_count_total",
        "uav_launch_count_total",
        "uav_delivery_count_total",
        "goal_assignment_to_launch_ratio",
        "launch_to_completion_ratio",
        "uav_rejected_total",
        # Efficiency
        "inference_latency_mean_ms",
        "makespan",
        "total_distance_m",
        "truck_total_distance_m",
        "uav_total_distance_m",
        # Safety
        "crash_count",
        "uav_survival_rate",
        "uav_low_battery_illegal_launch_count_total",
    ]
    source_alias = {
        "uav_rejected_total": "uav_unsafe_launch_attempt_count_total",
        "inference_latency_mean_ms": "planner_inference_latency_mean_ms",
        "makespan": "makespan_seconds",
        "total_distance_m": "fleet_distance_total_m",
        "truck_total_distance_m": "truck_distance_total_m",
        "uav_total_distance_m": "uav_distance_total_m",
    }

    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in seed_rows:
        method = str(r.get("model", "")).strip().lower()
        if method not in target_methods:
            continue
        scenario = f"{str(r.get('scale', '')).upper()}-{str(r.get('scenario', '')).upper()}"
        groups.setdefault((method, scenario), []).append(r)

    if not groups:
        return

    rows_out: List[Dict[str, Any]] = []
    for (method, scenario), rows in sorted(groups.items()):
        out: Dict[str, Any] = {"method": method, "scenario": scenario, "n": len(rows)}
        for m in metrics:
            src = source_alias.get(m, m)
            vals = [float(rr.get(src, float("nan"))) for rr in rows]
            mu, sd = _mean_std(vals)
            out[f"{m}_mean"] = mu
            out[f"{m}_std"] = sd
        rows_out.append(out)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["method", "scenario", "n"] + [x for m in metrics for x in (f"{m}_mean", f"{m}_std")]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    print(f"Saved: {out_csv}")
    with out_csv_legacy.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["method", "scenario", "n"] + [x for m in metrics for x in (f"{m}_mean", f"{m}_std")]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    print(f"Saved: {out_csv_legacy}")
    with out_csv_hard_denoise.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["method", "scenario", "n"] + [x for m in metrics for x in (f"{m}_mean", f"{m}_std")]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    print(f"Saved: {out_csv_hard_denoise}")
    with out_csv_truck_path_localize.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["method", "scenario", "n"] + [x for m in metrics for x in (f"{m}_mean", f"{m}_std")]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    print(f"Saved: {out_csv_truck_path_localize}")

    # Progress-aware switch tables requested by MR/L quick validations.
    mr_rows = [r for r in rows_out if str(r.get("scenario", "")).upper() in {"M-C", "R-C"}]
    l_rows = [r for r in rows_out if str(r.get("scenario", "")).upper() == "L-C"]
    with out_csv_progress_mr.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["method", "scenario", "n"] + [x for m in metrics for x in (f"{m}_mean", f"{m}_std")]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in mr_rows:
            w.writerow(r)
    print(f"Saved: {out_csv_progress_mr}")
    with out_csv_progress_l.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["method", "scenario", "n"] + [x for m in metrics for x in (f"{m}_mean", f"{m}_std")]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in l_rows:
            w.writerow(r)
    print(f"Saved: {out_csv_progress_l}")


def _write_ablation_wiring_check(run_dir: Path, seed_rows: List[Dict[str, Any]]) -> None:
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_csv = tables_dir / "Table_ablation_wiring_check.csv"
    fields = [
        "method",
        "ablation_low_value_refresh_enabled",
        "ablation_map_ranking_refresh_enabled",
        "ablation_tc_global_assignment_enabled",
        "ablation_support_chain_enabled",
        "ablation_cluster_primary_reservation_enabled",
        "ablation_event_scoring_bonus_enabled",
        "ablation_normal_protection_enabled",
        "low_value_refresh_candidate_count",
        "low_value_refresh_allowed_count",
        "low_value_refresh_blocked_by_ablation_count",
        "map_ranking_refresh_candidate_count",
        "map_ranking_refresh_allowed_count",
        "map_ranking_refresh_blocked_by_ablation_count",
        "tc_global_assignment_called_count",
        "tc_global_assignment_skipped_by_ablation_count",
        "tc_assignment_epoch_applied_count",
        "support_chain_candidate_count",
        "support_chain_applied_count",
        "support_chain_blocked_by_ablation_count",
        "cluster_primary_candidate_count",
        "cluster_primary_applied_count",
        "cluster_primary_blocked_by_ablation_count",
        "task_reservation_applied_count",
        "task_reservation_blocked_by_ablation_count",
        "event_scoring_bonus_applied_count",
        "event_scoring_bonus_blocked_by_ablation_count",
        "normal_protection_candidate_count",
        "normal_protection_applied_count",
        "normal_protection_blocked_by_ablation_count",
        "triggered_replans",
        "goal_switch_count_total",
        "inference_latency_mean_ms",
    ]
    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for r in seed_rows:
        by_method.setdefault(str(r.get("model", "")), []).append(r)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for method in sorted(by_method.keys()):
            rows = by_method[method]
            out: Dict[str, Any] = {"method": method}
            for k in fields[1:]:
                vals: List[float] = []
                for rr in rows:
                    try:
                        v = float(rr.get(k, float("nan")))
                    except Exception:
                        continue
                    if np.isfinite(v):
                        vals.append(v)
                out[k] = float(np.mean(vals)) if vals else float("nan")
            w.writerow(out)


def _write_hard_event_attribution_outputs(
    run_dir: Path,
    seed_rows: List[Dict[str, Any]],
    hard_event_offender_rows: List[Dict[str, Any]],
) -> None:
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_summary = tables_dir / "Table_hard_event_attribution_summary.csv"
    out_offenders = tables_dir / "Table_hard_event_top_offenders.csv"
    target_methods = {"erc_rhc_old", "erc_rhc"}
    metrics = [
        "hard_event_refresh_count_total",
        "hard_event_reason_goal_invalid_count",
        "hard_event_reason_current_goal_unreachable_count",
        "hard_event_reason_path_blocked_count",
        "hard_event_reason_uav_safety_count",
        "hard_event_reason_uav_recovery_count",
        "hard_event_reason_truck_dead_end_count",
        "hard_event_reason_high_priority_uncovered_count",
        "hard_event_reason_normal_stall_count",
        "hard_event_reason_assigned_but_not_progressing_count",
        "hard_event_reason_goal_completed_count",
        "hard_event_reason_goal_failed_count",
        "goal_invalid_reason_task_completed",
        "goal_invalid_reason_task_failed",
        "goal_invalid_reason_task_missing",
        "goal_invalid_reason_truck_unreachable",
        "goal_invalid_reason_uav_energy_infeasible",
        "goal_invalid_reason_uav_recovery_margin",
        "goal_invalid_reason_uav_corridor",
        "goal_invalid_reason_uav_comm_block",
        "goal_invalid_reason_uav_not_loaded",
        "goal_invalid_reason_uav_not_docked",
        "goal_invalid_reason_soft_reject_cache",
        "hard_event_refresh_no_goal_change_count",
        "hard_event_refresh_goal_change_count",
        "hard_event_refresh_to_completion_count",
        "hard_event_refresh_followed_by_reject_count",
        "suspect_soft_as_hard_count",
        "goal_switch_count_total",
        "inference_latency_mean_ms",
    ]
    source_alias = {"inference_latency_mean_ms": "planner_inference_latency_mean_ms"}

    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in seed_rows:
        method = str(r.get("model", "")).strip().lower()
        if method not in target_methods:
            continue
        scenario = f"{str(r.get('scale', '')).upper()}-{str(r.get('scenario', '')).upper()}"
        groups.setdefault((scenario, method), []).append(r)

    with out_summary.open("w", newline="", encoding="utf-8") as f:
        fields = ["scenario", "method", "n"] + metrics
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (scenario, method), rows in sorted(groups.items()):
            out: Dict[str, Any] = {"scenario": scenario, "method": method, "n": len(rows)}
            for m in metrics:
                src = source_alias.get(m, m)
                vals = [float(rr.get(src, float("nan"))) for rr in rows]
                vals = [v for v in vals if np.isfinite(v)]
                out[m] = float(np.mean(vals)) if vals else float("nan")
            w.writerow(out)
    print(f"Saved: {out_summary}")

    agg: Dict[Tuple[str, int, str, str, str, str], Dict[str, float]] = {}
    for r in hard_event_offender_rows:
        method = str(r.get("method", "")).strip().lower()
        if method not in target_methods:
            continue
        key = (
            str(r.get("scenario", "")),
            int(r.get("seed", 0)),
            str(method),
            str(r.get("agent_id", "")),
            str(r.get("task_id", "")),
            str(r.get("hard_event_reason", "unknown")),
        )
        rec = agg.get(
            key,
            {
                "count": 0.0,
                "first_step": float(r.get("first_step", 0.0)),
                "last_step": float(r.get("last_step", 0.0)),
                "launch_count_after_event": 0.0,
                "completion_count_after_event": 0.0,
                "goal_switch_after_event": 0.0,
            },
        )
        rec["count"] = float(rec.get("count", 0.0) + float(r.get("count", 0.0)))
        rec["first_step"] = float(min(float(rec.get("first_step", 0.0)), float(r.get("first_step", 0.0))))
        rec["last_step"] = float(max(float(rec.get("last_step", 0.0)), float(r.get("last_step", 0.0))))
        rec["launch_count_after_event"] = float(rec.get("launch_count_after_event", 0.0) + float(r.get("launch_count_after_event", 0.0)))
        rec["completion_count_after_event"] = float(
            rec.get("completion_count_after_event", 0.0) + float(r.get("completion_count_after_event", 0.0))
        )
        rec["goal_switch_after_event"] = float(rec.get("goal_switch_after_event", 0.0) + float(r.get("goal_switch_after_event", 0.0)))
        agg[key] = rec

    offender_fields = [
        "scenario",
        "seed",
        "method",
        "agent_id",
        "task_id",
        "hard_event_reason",
        "count",
        "first_step",
        "last_step",
        "launch_count_after_event",
        "completion_count_after_event",
        "goal_switch_after_event",
    ]
    with out_offenders.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=offender_fields)
        w.writeheader()
        for (scenario, seed, method, agent_id, task_id, reason), rec in sorted(
            agg.items(), key=lambda kv: (-float(kv[1].get("count", 0.0)), kv[0][0], kv[0][2], kv[0][1], kv[0][3], kv[0][4], kv[0][5])
        ):
            w.writerow(
                {
                    "scenario": scenario,
                    "seed": int(seed),
                    "method": method,
                    "agent_id": agent_id,
                    "task_id": task_id,
                    "hard_event_reason": reason,
                    "count": float(rec.get("count", 0.0)),
                    "first_step": float(rec.get("first_step", 0.0)),
                    "last_step": float(rec.get("last_step", 0.0)),
                    "launch_count_after_event": float(rec.get("launch_count_after_event", 0.0)),
                    "completion_count_after_event": float(rec.get("completion_count_after_event", 0.0)),
                    "goal_switch_after_event": float(rec.get("goal_switch_after_event", 0.0)),
                }
            )
    print(f"Saved: {out_offenders}")


def _write_rc_hard_noop_outputs(
    run_dir: Path,
    hard_event_reason_rows: List[Dict[str, Any]],
    hard_event_offender_rows: List[Dict[str, Any]],
) -> None:
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_reason = tables_dir / "Table_RC_hard_noop_reason_breakdown.csv"
    out_offender = tables_dir / "Table_RC_hard_noop_top_offenders.csv"
    out_doc = Path("docs") / "RC_HARD_EVENT_NOOP_ANALYSIS.md"

    # 1) Reason breakdown (R-C + new ERC only).
    reason_agg: Dict[str, Dict[str, float]] = {}
    for r in hard_event_reason_rows:
        if str(r.get("scenario", "")).upper() != "R-C":
            continue
        if str(r.get("method", "")).strip().lower() != "erc_rhc":
            continue
        reason = str(r.get("hard_event_reason", "unknown"))
        rec = reason_agg.get(
            reason,
            {
                "hard_event_reason": reason,
                "total_refresh_count": 0.0,
                "no_goal_change_count": 0.0,
                "goal_change_count": 0.0,
                "followed_by_launch_count": 0.0,
                "followed_by_completion_count": 0.0,
                "followed_by_reject_count": 0.0,
                "followed_by_stall_count": 0.0,
            },
        )
        for k in (
            "total_refresh_count",
            "no_goal_change_count",
            "goal_change_count",
            "followed_by_launch_count",
            "followed_by_completion_count",
            "followed_by_reject_count",
            "followed_by_stall_count",
        ):
            rec[k] = float(rec.get(k, 0.0) + float(r.get(k, 0.0)))
        reason_agg[reason] = rec

    reason_fields = [
        "hard_event_reason",
        "total_refresh_count",
        "no_goal_change_count",
        "goal_change_count",
        "no_goal_change_ratio",
        "followed_by_launch_count",
        "followed_by_completion_count",
        "followed_by_reject_count",
        "followed_by_stall_count",
        "value_label",
    ]
    reason_rows_sorted: List[Dict[str, Any]] = []
    for reason, rec in sorted(reason_agg.items(), key=lambda kv: -float(kv[1].get("total_refresh_count", 0.0))):
        total = float(rec.get("total_refresh_count", 0.0))
        no_goal = float(rec.get("no_goal_change_count", 0.0))
        goal_change = float(rec.get("goal_change_count", 0.0))
        launch = float(rec.get("followed_by_launch_count", 0.0))
        comp = float(rec.get("followed_by_completion_count", 0.0))
        rej = float(rec.get("followed_by_reject_count", 0.0))
        stall = float(rec.get("followed_by_stall_count", 0.0))
        no_goal_ratio = float(no_goal / max(total, 1.0))
        useful_score = goal_change + launch + comp
        harmful_score = rej + stall
        if useful_score <= 0.0 and no_goal_ratio >= 0.8:
            label = "no-op"
        elif harmful_score > useful_score:
            label = "harmful"
        else:
            label = "useful"
        out = dict(rec)
        out["no_goal_change_ratio"] = no_goal_ratio
        out["value_label"] = label
        reason_rows_sorted.append(out)

    with out_reason.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=reason_fields)
        w.writeheader()
        for row in reason_rows_sorted:
            w.writerow({k: row.get(k, "") for k in reason_fields})

    # 2) Top offenders (R-C + new ERC only).
    offender_agg: Dict[Tuple[int, int, str, str, str], Dict[str, Any]] = {}
    for r in hard_event_offender_rows:
        if str(r.get("scenario", "")).upper() != "R-C":
            continue
        if str(r.get("method", "")).strip().lower() != "erc_rhc":
            continue
        seed = int(r.get("seed", 0))
        episode = int(r.get("episode", 0))
        aid = str(r.get("agent_id", ""))
        tid = str(r.get("task_id", ""))
        reason = str(r.get("hard_event_reason", "unknown"))
        key = (seed, episode, aid, tid, reason)
        rec = offender_agg.get(
            key,
            {
                "seed": seed,
                "episode": episode,
                "agent_id": aid,
                "task_id": tid,
                "hard_event_reason": reason,
                "count": 0.0,
                "no_goal_change_count": 0.0,
                "first_step": float(r.get("first_step", 0.0)),
                "last_step": float(r.get("last_step", 0.0)),
                "current_goal_type": str(r.get("current_goal_type", "")),
                "proposed_goal_type": str(r.get("proposed_goal_type", "")),
                "launch_after_count": 0.0,
                "completion_after_count": 0.0,
                "reject_after_count": 0.0,
                "distance_to_goal_sum": 0.0,
                "distance_to_goal_n": 0.0,
                "battery_sum": 0.0,
                "battery_n": 0.0,
                "task_status": str(r.get("task_status", "")),
            },
        )
        rec["count"] = float(rec.get("count", 0.0) + float(r.get("count", 0.0)))
        rec["no_goal_change_count"] = float(rec.get("no_goal_change_count", 0.0) + float(r.get("no_goal_change_count", 0.0)))
        rec["first_step"] = float(min(float(rec.get("first_step", 0.0)), float(r.get("first_step", 0.0))))
        rec["last_step"] = float(max(float(rec.get("last_step", 0.0)), float(r.get("last_step", 0.0))))
        rec["launch_after_count"] = float(rec.get("launch_after_count", 0.0) + float(r.get("launch_count_after_event", 0.0)))
        rec["completion_after_count"] = float(rec.get("completion_after_count", 0.0) + float(r.get("completion_count_after_event", 0.0)))
        rec["reject_after_count"] = float(rec.get("reject_after_count", 0.0) + float(r.get("reject_count_after_event", 0.0)))
        d = float(r.get("distance_to_goal_mean", float("nan")))
        if np.isfinite(d):
            rec["distance_to_goal_sum"] = float(rec.get("distance_to_goal_sum", 0.0) + d)
            rec["distance_to_goal_n"] = float(rec.get("distance_to_goal_n", 0.0) + 1.0)
        b = float(r.get("battery_mean", float("nan")))
        if np.isfinite(b):
            rec["battery_sum"] = float(rec.get("battery_sum", 0.0) + b)
            rec["battery_n"] = float(rec.get("battery_n", 0.0) + 1.0)
        if (not str(rec.get("current_goal_type", ""))) and str(r.get("current_goal_type", "")):
            rec["current_goal_type"] = str(r.get("current_goal_type", ""))
        if (not str(rec.get("proposed_goal_type", ""))) and str(r.get("proposed_goal_type", "")):
            rec["proposed_goal_type"] = str(r.get("proposed_goal_type", ""))
        if str(r.get("task_status", "")):
            rec["task_status"] = str(r.get("task_status", ""))
        offender_agg[key] = rec

    offender_fields = [
        "seed",
        "episode",
        "agent_id",
        "task_id",
        "hard_event_reason",
        "no_goal_change_count",
        "first_step",
        "last_step",
        "current_goal_type",
        "proposed_goal_type",
        "launch_after_count",
        "completion_after_count",
        "reject_after_count",
        "distance_to_goal_mean",
        "battery_mean",
        "task_status",
    ]
    offender_rows_sorted: List[Dict[str, Any]] = []
    for _k, rec in sorted(
        offender_agg.items(),
        key=lambda kv: (-float(kv[1].get("no_goal_change_count", 0.0)), -float(kv[1].get("count", 0.0))),
    ):
        out = dict(rec)
        dn = float(rec.get("distance_to_goal_n", 0.0))
        bn = float(rec.get("battery_n", 0.0))
        out["distance_to_goal_mean"] = float(rec.get("distance_to_goal_sum", 0.0) / dn) if dn > 0.0 else float("nan")
        out["battery_mean"] = float(rec.get("battery_sum", 0.0) / bn) if bn > 0.0 else float("nan")
        offender_rows_sorted.append(out)

    with out_offender.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=offender_fields)
        w.writeheader()
        for row in offender_rows_sorted:
            w.writerow({k: row.get(k, "") for k in offender_fields})

    # 3) Markdown summary.
    total_hard = float(sum(float(r.get("total_refresh_count", 0.0)) for r in reason_rows_sorted))
    total_no_goal = float(sum(float(r.get("no_goal_change_count", 0.0)) for r in reason_rows_sorted))
    overall_no_goal_ratio = float(total_no_goal / max(total_hard, 1.0))
    lines: List[str] = []
    lines.append("# RC Hard-Event No-op Analysis")
    lines.append("")
    lines.append("## Scope")
    lines.append("- Scenario: R-C")
    lines.append("- Method: erc_rhc (new ERC)")
    lines.append("- Seeds: 0,1,2")
    lines.append("- Episodes per seed: 1")
    lines.append("")
    lines.append("## Headline")
    lines.append(f"- hard_event_refresh_count_total (reason-summed): {total_hard:.1f}")
    lines.append(f"- hard_event_refresh_no_goal_change_count (reason-summed): {total_no_goal:.1f}")
    lines.append(f"- hard_event_refresh_no_goal_change_ratio: {overall_no_goal_ratio:.4f}")
    lines.append("")
    lines.append("## Reason Breakdown")
    lines.append("| hard_event_reason | total_refresh_count | no_goal_change_count | goal_change_count | no_goal_change_ratio | followed_by_launch_count | followed_by_completion_count | followed_by_reject_count | followed_by_stall_count | value_label |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in reason_rows_sorted:
        lines.append(
            f"| {r.get('hard_event_reason','')} | {float(r.get('total_refresh_count',0.0)):.1f} | "
            f"{float(r.get('no_goal_change_count',0.0)):.1f} | {float(r.get('goal_change_count',0.0)):.1f} | "
            f"{float(r.get('no_goal_change_ratio',0.0)):.4f} | {float(r.get('followed_by_launch_count',0.0)):.1f} | "
            f"{float(r.get('followed_by_completion_count',0.0)):.1f} | {float(r.get('followed_by_reject_count',0.0)):.1f} | "
            f"{float(r.get('followed_by_stall_count',0.0)):.1f} | {r.get('value_label','')} |"
        )
    lines.append("")
    lines.append("## Next-step Suggestions (No Algorithm Change Applied)")
    lines.append("- Check top-2 reasons by `no_goal_change_count` first; they dominate no-op hard refresh volume.")
    lines.append("- For reasons labeled `harmful`, inspect whether refresh can be localized to affected agent only.")
    lines.append("- For reasons labeled `no-op`, require an additional executability delta before admitting hard refresh.")
    lines.append("")
    lines.append("## Output Files")
    lines.append(f"- `{out_reason.as_posix()}`")
    lines.append(f"- `{out_offender.as_posix()}`")
    lines.append("")
    out_doc.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {out_reason}")
    print(f"Saved: {out_offender}")
    print(f"Saved: {out_doc}")


def _write_mc_routine_loss_outputs(
    run_dir: Path,
    task_outcome_rows: List[Dict[str, Any]],
    routine_trace_rows: List[Dict[str, Any]],
    truck_routine_summary_rows: List[Dict[str, Any]],
) -> None:
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_diff = tables_dir / "Table_MC_task_completion_diff.csv"
    out_lost = tables_dir / "Table_MC_lost_routine_task_trace.csv"
    out_truck = tables_dir / "Table_MC_truck_routine_execution_summary.csv"
    out_doc = Path("docs") / "MC_ROUTINE_LOSS_ANALYSIS.md"

    methods_need = {"rolling_fixed", "erc_rhc_old", "erc_rhc"}
    key_to_task: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
    seen_methods: set[str] = set()
    for r in task_outcome_rows:
        if str(r.get("scenario", "")).upper() != "M-C":
            continue
        method = str(r.get("method", "")).strip().lower()
        if method not in methods_need:
            continue
        seen_methods.add(method)
        key = (int(r.get("seed", 0)), int(r.get("episode", 0)), str(r.get("task_id", "")))
        rec = key_to_task.get(
            key,
            {
                "seed": int(r.get("seed", 0)),
                "episode": int(r.get("episode", 0)),
                "task_id": str(r.get("task_id", "")),
                "task_kind": str(r.get("task_kind", "")),
                "task_class": str(r.get("task_class", "")),
                "completed_by_rolling": 0,
                "completed_by_old_erc": 0,
                "completed_by_new_erc": 0,
                "completed_step_rolling": -1,
                "completed_step_old_erc": -1,
                "completed_step_new_erc": -1,
                "final_status_rolling": "",
                "final_status_old_erc": "",
                "final_status_new_erc": "",
            },
        )
        if method == "rolling_fixed":
            rec["completed_by_rolling"] = int(r.get("completed", 0))
            rec["completed_step_rolling"] = int(r.get("completed_step", -1))
            rec["final_status_rolling"] = str(r.get("final_status", ""))
        elif method == "erc_rhc_old":
            rec["completed_by_old_erc"] = int(r.get("completed", 0))
            rec["completed_step_old_erc"] = int(r.get("completed_step", -1))
            rec["final_status_old_erc"] = str(r.get("final_status", ""))
        elif method == "erc_rhc":
            rec["completed_by_new_erc"] = int(r.get("completed", 0))
            rec["completed_step_new_erc"] = int(r.get("completed_step", -1))
            rec["final_status_new_erc"] = str(r.get("final_status", ""))
        key_to_task[key] = rec

    diff_fields = [
        "seed",
        "episode",
        "task_id",
        "task_kind",
        "task_class",
        "completed_by_rolling",
        "completed_by_old_erc",
        "completed_by_new_erc",
        "completed_step_rolling",
        "completed_step_old_erc",
        "completed_step_new_erc",
        "final_status_rolling",
        "final_status_old_erc",
        "final_status_new_erc",
    ]
    diff_rows = sorted(key_to_task.values(), key=lambda x: (int(x["seed"]), int(x["episode"]), str(x["task_id"])))
    with out_diff.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=diff_fields)
        w.writeheader()
        for r in diff_rows:
            w.writerow({k: r.get(k, "") for k in diff_fields})

    lost_task_keys: set[Tuple[int, int, str]] = set()
    for r in diff_rows:
        if str(r.get("task_kind", "")).lower() not in {"normal", "routine_bulk"}:
            continue
        if int(r.get("completed_by_new_erc", 0)) == 1:
            continue
        if int(r.get("completed_by_rolling", 0)) == 1 or int(r.get("completed_by_old_erc", 0)) == 1:
            lost_task_keys.add((int(r.get("seed", 0)), int(r.get("episode", 0)), str(r.get("task_id", ""))))

    new_routine_map: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
    for r in routine_trace_rows:
        if str(r.get("scenario", "")).upper() != "M-C":
            continue
        if str(r.get("method", "")).strip().lower() != "erc_rhc":
            continue
        key = (int(r.get("seed", 0)), int(r.get("episode", 0)), str(r.get("task_id", "")))
        new_routine_map[key] = r

    lost_fields = [
        "seed",
        "episode",
        "task_id",
        "nearest_truck",
        "assigned_truck_count",
        "first_assigned_step",
        "last_assigned_step",
        "truck_id",
        "truck_goal_switch_count",
        "truck_dead_end_routine_localized_count",
        "path_blocked_routine_localized_count",
        "assigned_but_not_progressing_count",
        "distance_to_task_start",
        "distance_to_task_min",
        "distance_to_task_final",
        "service_start_count",
        "service_complete_count",
        "final_failure_reason",
    ]
    lost_rows: List[Dict[str, Any]] = []
    for key in sorted(lost_task_keys):
        rec = new_routine_map.get(key, None)
        if rec is None:
            seed, ep, tid = key
            lost_rows.append(
                {
                    "seed": int(seed),
                    "episode": int(ep),
                    "task_id": str(tid),
                    "nearest_truck": "",
                    "assigned_truck_count": 0,
                    "first_assigned_step": -1,
                    "last_assigned_step": -1,
                    "truck_id": "",
                    "truck_goal_switch_count": 0,
                    "truck_dead_end_routine_localized_count": 0,
                    "path_blocked_routine_localized_count": 0,
                    "assigned_but_not_progressing_count": 0,
                    "distance_to_task_start": float("nan"),
                    "distance_to_task_min": float("nan"),
                    "distance_to_task_final": float("nan"),
                    "service_start_count": 0,
                    "service_complete_count": 0,
                    "final_failure_reason": "missing_trace",
                }
            )
            continue
        lost_rows.append({k: rec.get(k, "") for k in lost_fields})
    with out_lost.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=lost_fields)
        w.writeheader()
        for r in lost_rows:
            w.writerow(r)

    truck_fields = [
        "scenario",
        "seed",
        "episode",
        "method",
        "truck_id",
        "routine_assigned_count",
        "routine_completed_count",
        "routine_service_start_count",
        "routine_goal_switch_count",
        "routine_reassign_count",
        "dead_end_localize_count",
        "path_blocked_localize_count",
        "stuck_steps",
        "average_distance_progress_per_step",
        "time_spent_servicing",
        "time_spent_moving_to_routine",
        "time_spent_idle_or_no_progress",
    ]
    truck_rows = [
        r
        for r in truck_routine_summary_rows
        if str(r.get("scenario", "")).upper() == "M-C"
        and str(r.get("method", "")).strip().lower() in methods_need
    ]
    truck_rows = sorted(truck_rows, key=lambda x: (str(x.get("method", "")), int(x.get("seed", 0)), int(x.get("episode", 0)), str(x.get("truck_id", ""))))
    with out_truck.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=truck_fields)
        w.writeheader()
        for r in truck_rows:
            w.writerow({k: r.get(k, "") for k in truck_fields})

    # Markdown conclusions.
    reason_counts: Dict[str, int] = defaultdict(int)
    never_assigned = 0
    assigned_no_progress = 0
    assigned_never_serviced = 0
    localized_related = 0
    long_hold_no_progress = 0
    for r in lost_rows:
        reason = str(r.get("final_failure_reason", "unknown"))
        reason_counts[reason] += 1
        if reason == "never_assigned":
            never_assigned += 1
        if reason == "assigned_no_progress":
            assigned_no_progress += 1
        if reason == "assigned_never_serviced":
            assigned_never_serviced += 1
        if int(r.get("truck_dead_end_routine_localized_count", 0)) > 0 or int(r.get("path_blocked_routine_localized_count", 0)) > 0:
            localized_related += 1
        fa = int(r.get("first_assigned_step", -1))
        la = int(r.get("last_assigned_step", -1))
        if fa >= 0 and la > fa and int(r.get("assigned_but_not_progressing_count", 0)) > 0:
            long_hold_no_progress += 1

    lines: List[str] = []
    lines.append("# MC Routine Loss Analysis")
    lines.append("")
    lines.append("## Scope")
    lines.append("- Scenario: M-C")
    lines.append("- Methods compared: rolling_fixed, erc_rhc_old, erc_rhc")
    lines.append("")
    lines.append("## Key Counts")
    lines.append(f"- Lost routine tasks in new ERC (completed by rolling or old, but not by new): {len(lost_rows)}")
    lines.append(f"- never_assigned: {never_assigned}")
    lines.append(f"- assigned_never_serviced: {assigned_never_serviced}")
    lines.append(f"- assigned_no_progress: {assigned_no_progress}")
    lines.append(f"- localization-related lost tasks: {localized_related}")
    lines.append(f"- long-hold-without-progress tasks: {long_hold_no_progress}")
    lines.append("")
    lines.append("## Answers")
    lines.append("1. Why new ERC loses routine tasks:")
    lines.append(
        "- Based on lost-task traces, primary buckets are `never_assigned`, `assigned_never_serviced`, and `assigned_no_progress`; this separates allocation loss vs execution/progress loss."
    )
    lines.append("2. Is extra loss vs old ERC related to routine localization:")
    lines.append(
        "- We mark a lost task as localization-related when `truck_dead_end_routine_localized_count > 0` or `path_blocked_routine_localized_count > 0`; see count above and per-task table."
    )
    lines.append("3. Any long-held goals with no distance decrease:")
    lines.append(
        "- Yes when `first_assigned_step >= 0`, `last_assigned_step > first_assigned_step`, and `assigned_but_not_progressing_count > 0`; see `long-hold-without-progress` count."
    )
    lines.append("4. Need ETA-worsen allow-reselect for routine localize:")
    lines.append(
        "- Recommended as a minimal guard: if routine goal ETA/distance trend degrades for N steps, allow local reselect for that truck."
    )
    lines.append("5. Should M-C routine localization be rolled back:")
    lines.append(
        "- Prefer conditional fallback instead of scenario-specific rollback: trigger by generic `routine no-progress persistence` + `no local path repair gain`, not by `scenario == M-C`."
    )
    lines.append("")
    lines.append("## Output Files")
    lines.append(f"- `{out_diff.as_posix()}`")
    lines.append(f"- `{out_lost.as_posix()}`")
    lines.append(f"- `{out_truck.as_posix()}`")
    out_doc.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {out_diff}")
    print(f"Saved: {out_lost}")
    print(f"Saved: {out_truck}")
    print(f"Saved: {out_doc}")


def _write_rc_seed0_seed1_support_failure_outputs(
    run_dir: Path,
    seed_rows: List[Dict[str, Any]],
    task_outcome_rows: List[Dict[str, Any]],
    support_trace_rows: List[Dict[str, Any]],
    uav_execution_rows: List[Dict[str, Any]],
) -> None:
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_task = tables_dir / "Table_RC_seed0_seed1_task_outcome.csv"
    out_support = tables_dir / "Table_RC_seed0_seed1_truck_support_trace.csv"
    out_uav = tables_dir / "Table_RC_seed0_seed1_uav_execution_chain.csv"
    out_replan = tables_dir / "Table_RC_seed0_seed1_replan_switch_breakdown.csv"
    out_doc = Path("docs") / "RC_SEED0_SEED1_FAILURE_ANALYSIS.md"

    rows_task = [
        r for r in task_outcome_rows
        if str(r.get("scenario", "")).upper() == "R-C"
        and str(r.get("method", "")).strip().lower() == "erc_rhc"
        and int(r.get("seed", -1)) in {0, 1}
    ]
    task_fields = [
        "seed","task_id","task_class","task_kind","demand_kg","urgency_score","lifeline_init","lifeline_final",
        "assigned_count","first_assigned_step","last_assigned_step","service_start_count","completed","failed",
        "failed_reason","completed_by_agent","completed_step","nearest_truck_distance_start",
        "nearest_uav_or_truck_distance_start","truck_reachable","uav_eventual_serviceable",
    ]
    with out_task.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=task_fields)
        w.writeheader()
        for r in sorted(rows_task, key=lambda x: (int(x.get("seed", 0)), str(x.get("task_id", "")))):
            w.writerow({k: r.get(k, "") for k in task_fields})

    rows_support = [
        r for r in support_trace_rows
        if str(r.get("scenario", "")).upper() == "R-C"
        and str(r.get("method", "")).strip().lower() == "erc_rhc"
        and int(r.get("seed", -1)) in {0, 1}
    ]
    support_fields = [
        "seed","step","truck_id","current_goal","current_goal_type","normal_goal_id","support_target_uav",
        "support_target_task","support_reason","support_selected_by_planner","forward_support_event",
        "recovery_support_event","truck_position","distance_to_normal_goal","distance_to_support_task",
        "moved_distance_this_step","is_servicing","service_target","normal_task_progress_delta",
        "uav_launch_nearby_after_support","uav_delivery_after_support",
    ]
    with out_support.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=support_fields)
        w.writeheader()
        for r in sorted(rows_support, key=lambda x: (int(x.get("seed", 0)), int(x.get("step", 0)), str(x.get("truck_id", "")))):
            w.writerow({k: r.get(k, "") for k in support_fields})

    rows_uav = [
        r for r in uav_execution_rows
        if str(r.get("scenario", "")).upper() == "R-C"
        and str(r.get("method", "")).strip().lower() == "erc_rhc"
        and int(r.get("seed", -1)) in {0, 1}
    ]
    uav_fields = [
        "seed","uav_id","assigned_task_count","launch_count","delivery_count","launch_to_completion_ratio","reject_count",
        "reject_reason_insufficient_recovery_margin","reject_reason_corridor","reject_reason_comm_block",
        "reject_reason_energy_infeasible","reject_reason_no_recovery","airborne_goal_switch_blocked_count",
        "forced_recovery_count","rendezvous_success_count","average_launch_battery","min_launch_battery",
    ]
    with out_uav.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=uav_fields)
        w.writeheader()
        for r in sorted(rows_uav, key=lambda x: (int(x.get("seed", 0)), str(x.get("uav_id", "")))):
            w.writerow({k: r.get(k, "") for k in uav_fields})

    rows_seed = [
        r for r in seed_rows
        if str(r.get("scale", "")).upper() == "R"
        and str(r.get("scenario", "")).upper() == "C"
        and str(r.get("model", "")).strip().lower() == "erc_rhc"
        and int(r.get("train_seed", -1)) in {0, 1}
    ]
    replan_fields = [
        "seed","refresh_total_count","event_refresh_count","hard_event_refresh_count_total",
        "event_refresh_reason_arrival_count","event_refresh_reason_map_update_hard_count",
        "event_refresh_reason_goal_invalid_count","event_refresh_reason_goal_unreachable_count",
        "event_refresh_reason_high_priority_uncovered_count","event_refresh_followed_by_stall_count",
        "goal_switch_candidate_count","goal_switch_accepted_count","goal_switch_rejected_by_threshold_count",
        "goal_switch_forced_count","goal_switch_forced_reason_infeasible","missed_switch_proxy_count",
        "harmful_switch_proxy_count",
    ]
    with out_replan.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=replan_fields)
        w.writeheader()
        for r in sorted(rows_seed, key=lambda x: int(x.get("train_seed", 0))):
            row = {"seed": int(r.get("train_seed", 0))}
            for k in replan_fields:
                if k != "seed":
                    row[k] = r.get(k, "")
            w.writerow(row)

    def _seed_row(seed: int) -> Dict[str, Any]:
        for r in rows_seed:
            if int(r.get("train_seed", -1)) == int(seed):
                return r
        return {}

    s1 = _seed_row(1)
    lines = [
        "# RC Seed0 vs Seed1 Failure Analysis",
        "",
        "## Scope",
        "- Scenario: R-C",
        "- Method: erc_rhc",
        "- Seeds: 0 and 1",
        "",
        "## Answers",
        f"1. seed=1 last delivery stopped at step={float(s1.get('delivered_task_last_step', float('nan'))):.1f}; this aligns with high stall and switch pressure (stall={float(s1.get('hard_event_refresh_followed_by_stall_count', 0.0)):.1f}, switch_candidate={float(s1.get('goal_switch_candidate_count', 0.0)):.1f}).",
        f"2. seed=1 routine completion is low (routine={float(s1.get('routine_bulk_completion_rate', float('nan'))):.3f}) while truck support/recovery counts are high (forward={float(s1.get('truck_forward_support_count_total', 0.0)):.1f}, recovery={float(s1.get('truck_recovery_support_count_total', 0.0)):.1f}).",
        f"3. support_selected=0 but forward/recovery is high, which indicates planner support accounting and env execution support accounting are decoupled (seed1 support_selected={float(s1.get('support_selected_count', 0.0)):.1f}).",
        "4. Truck routine progress is likely hijacked by support/recovery execution logic; inspect normal_task_progress_delta in the support trace.",
        f"5. UAV launches exceed deliveries: seed1 launch={float(s1.get('uav_launch_count_total', 0.0)):.1f}, delivery={float(s1.get('uav_delivery_count_total', 0.0)):.1f}, launch_to_completion={float(s1.get('launch_to_completion_ratio', float('nan'))):.3f}.",
        f"6. High reject cache/recovery reject is consistent with repeated infeasible UAV-task attempts: seed1 reject_cache_hit={float(s1.get('uav_reject_cache_hit_count_total', 0.0)):.0f}, recovery_reject_rate={float(s1.get('recovery_reject_rate', float('nan'))):.3f}.",
        "7. The issue is more likely execution logic decoupling than a pure metric accounting bug.",
        "8. Minimal repair recommendation: unify support triggering through planner authorization and suppress repeated support when it yields no launch/delivery/routine progress.",
        "",
        "## Output Files",
        f"- `{out_task.as_posix()}`",
        f"- `{out_support.as_posix()}`",
        f"- `{out_uav.as_posix()}`",
        f"- `{out_replan.as_posix()}`",
    ]
    out_doc.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {out_task}")
    print(f"Saved: {out_support}")
    print(f"Saved: {out_uav}")
    print(f"Saved: {out_replan}")
    print(f"Saved: {out_doc}")


def _write_switch_decision_audit_outputs(run_dir: Path, switch_rows: List[Dict[str, Any]]) -> None:
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_ledger = tables_dir / "switch_decision_ledger.csv"
    out_summary = tables_dir / "Table_switch_decision_audit_summary.csv"
    out_top = tables_dir / "Table_switch_error_top_offenders.csv"
    out_doc = Path("docs") / "SWITCH_DECISION_AUDIT.md"

    if not switch_rows:
        out_ledger.write_text("", encoding="utf-8")
        out_summary.write_text("", encoding="utf-8")
        out_top.write_text("", encoding="utf-8")
        out_doc.write_text("# SWITCH DECISION AUDIT\n\nNo switch decision rows collected.\n", encoding="utf-8")
        print(f"Saved: {out_ledger}")
        print(f"Saved: {out_summary}")
        print(f"Saved: {out_top}")
        print(f"Saved: {out_doc}")
        return

    df = pd.DataFrame(switch_rows)
    for col in (
        "switch_candidate_count",
        "switch_accepted",
        "switch_rejected",
        "switch_forced",
        "after_switch_task_completed",
        "after_switch_uav_launch",
        "after_switch_stall_count",
        "harmful_switch",
        "missed_switch",
    ):
        if col not in df.columns:
            df[col] = 0
    for col in (
        "current_goal_progress_last_5_steps",
        "score_delta",
        "eta_gain",
        "best_alternative_routine_eta",
    ):
        if col not in df.columns:
            df[col] = np.nan
    if "agent_type" not in df.columns:
        df["agent_type"] = "unknown"
    if "scenario" not in df.columns:
        df["scenario"] = ""
    if "method" not in df.columns:
        df["method"] = ""
    if "seed" not in df.columns:
        df["seed"] = 0
    if "episode" not in df.columns:
        df["episode"] = 0
    if "agent_id" not in df.columns:
        df["agent_id"] = ""
    if "step" not in df.columns:
        df["step"] = 0

    df.to_csv(out_ledger, index=False, encoding="utf-8")

    grp = df.groupby(["scenario", "method", "agent_type"], dropna=False)
    summary = grp.agg(
        switch_candidate_count=("switch_candidate_count", "sum"),
        switch_accepted_count=("switch_accepted", "sum"),
        switch_rejected_count=("switch_rejected", "sum"),
        forced_switch_count=("switch_forced", "sum"),
        harmful_switch_count=("harmful_switch", "sum"),
        missed_switch_count=("missed_switch", "sum"),
    ).reset_index()
    summary["harmful_switch_ratio"] = summary["harmful_switch_count"] / summary["switch_accepted_count"].clip(lower=1.0)
    summary["missed_switch_ratio"] = summary["missed_switch_count"] / summary["switch_rejected_count"].clip(lower=1.0)

    acc_grp = df[df["switch_accepted"] == 1].groupby(["scenario", "method", "agent_type"], dropna=False)
    acc_stats = acc_grp.agg(
        accepted_switch_to_completion_ratio=("after_switch_task_completed", "mean"),
        accepted_switch_to_launch_ratio=("after_switch_uav_launch", lambda s: float((s > 0).mean()) if len(s) else 0.0),
    ).reset_index()
    rej_stall = (
        df[(df["switch_rejected"] == 1) & (df["after_switch_stall_count"] > 0)]
        .groupby(["scenario", "method", "agent_type"], dropna=False)
        .size()
        .reset_index(name="rejected_switch_followed_by_stall_count")
    )
    prog_pos = (
        df[df["current_goal_progress_last_5_steps"].fillna(0.0) > 0.0]
        .groupby(["scenario", "method", "agent_type"], dropna=False)
        .size()
        .reset_index(name="current_goal_progress_positive_count")
    )
    prog_neg = (
        df[df["current_goal_progress_last_5_steps"].fillna(0.0) <= 0.0]
        .groupby(["scenario", "method", "agent_type"], dropna=False)
        .size()
        .reset_index(name="current_goal_progress_negative_count")
    )
    summary = summary.merge(acc_stats, on=["scenario", "method", "agent_type"], how="left")
    summary = summary.merge(rej_stall, on=["scenario", "method", "agent_type"], how="left")
    summary = summary.merge(prog_pos, on=["scenario", "method", "agent_type"], how="left")
    summary = summary.merge(prog_neg, on=["scenario", "method", "agent_type"], how="left")
    summary = summary.fillna(0.0)
    summary.to_csv(out_summary, index=False, encoding="utf-8")

    errs = df[(df["harmful_switch"] == 1) | (df["missed_switch"] == 1)].copy()
    if len(errs) > 0:
        errs["error_type"] = np.where(errs["harmful_switch"] == 1, "harmful_switch", "missed_switch")
        errs["after_switch_outcome"] = np.where(
            errs["after_switch_task_completed"] > 0,
            "completed",
            np.where(errs["after_switch_uav_launch"] > 0, "launch_only", np.where(errs["after_switch_stall_count"] > 0, "stall", "no_progress")),
        )
        errs["suggested_fix_type"] = np.where(
            errs["error_type"] == "harmful_switch",
            "strengthen_hold",
            "strengthen_escape",
        )
        top = errs.sort_values(
            by=["harmful_switch", "missed_switch", "after_switch_stall_count", "after_switch_reject_count"],
            ascending=[False, False, False, False],
        ).head(400)
    else:
        top = pd.DataFrame(columns=["scenario"])

    top_fields = [
        "scenario",
        "seed",
        "episode",
        "step",
        "agent_id",
        "agent_type",
        "error_type",
        "current_goal_id",
        "proposed_goal_id",
        "switch_reason",
        "score_delta",
        "eta_gain",
        "current_goal_progress_last_5_steps",
        "best_alternative_routine_eta",
        "after_switch_outcome",
        "suggested_fix_type",
    ]
    for c in top_fields:
        if c not in top.columns:
            top[c] = ""
    top[top_fields].to_csv(out_top, index=False, encoding="utf-8")

    # Report answers.
    lines: List[str] = []
    lines.append("# SWITCH DECISION AUDIT")
    lines.append("")
    lines.append("## Scope")
    lines.append("- Input: switch_decision_ledger.csv")
    lines.append("- Methods/scenarios from current run output")
    lines.append("")
    overall_harmful = float(df["harmful_switch"].sum())
    overall_missed = float(df["missed_switch"].sum())
    lines.append("## Headline")
    lines.append(f"- harmful_switch_count: {overall_harmful:.0f}")
    lines.append(f"- missed_switch_count: {overall_missed:.0f}")
    lines.append("")
    lines.append("## Answers")
    if overall_harmful > overall_missed:
        q1 = "harmful_switch is higher; false-positive switching dominates."
    elif overall_harmful < overall_missed:
        q1 = "missed_switch is higher; false-negative holding dominates."
    else:
        q1 = "both are close; this is a mixed switch-control problem."
    lines.append(f"1. Which switch error dominates? {q1}")

    mc = summary[(summary["scenario"] == "M-C") & (summary["method"] == "erc_rhc")]
    mc_h = float(mc["harmful_switch_count"].sum()) if len(mc) else 0.0
    mc_m = float(mc["missed_switch_count"].sum()) if len(mc) else 0.0
    q2 = "harmful_switch-driven" if mc_h > mc_m else ("missed_switch-driven" if mc_m > mc_h else "mixed")
    lines.append(f"2. What drives M-C routine loss? {q2}")

    rc_old = summary[(summary["scenario"] == "R-C") & (summary["method"] == "erc_rhc_old")]
    rc_new = summary[(summary["scenario"] == "R-C") & (summary["method"] == "erc_rhc")]
    if len(rc_old) and len(rc_new):
        old_h = float(rc_old["harmful_switch_count"].sum())
        new_h = float(rc_new["harmful_switch_count"].sum())
        old_m = float(rc_old["missed_switch_count"].sum())
        new_m = float(rc_new["missed_switch_count"].sum())
        reduced = []
        if new_h < old_h:
            reduced.append("harmful_switch")
        if new_m < old_m:
            reduced.append("missed_switch")
        q3 = ", ".join(reduced) if reduced else "no single switch-error class clearly decreased"
    else:
        q3 = "missing old/new comparison samples"
    lines.append(f"3. Which switch errors decreased in R-C? {q3}")

    type_cmp = summary.groupby("agent_type")[["harmful_switch_count", "missed_switch_count"]].sum(numeric_only=True).reset_index()
    q4 = "agent-type differences exist; see the summary grouped by agent_type." if len(type_cmp) > 1 else "not enough samples to distinguish truck vs UAV."
    lines.append(f"4. Are truck and UAV switch errors different? {q4}")

    if overall_harmful > overall_missed:
        q5 = "prioritize stronger hold to reduce harmful switching."
    elif overall_missed > overall_harmful:
        q5 = "prioritize stronger escape to avoid unproductive holding."
    else:
        q5 = "both matter; start with hold, then add escape."
    lines.append(f"5. Should the next step strengthen hold or escape? {q5}")
    lines.append("")
    lines.append("## Files")
    lines.append(f"- `{out_ledger.as_posix()}`")
    lines.append(f"- `{out_summary.as_posix()}`")
    lines.append(f"- `{out_top.as_posix()}`")
    out_doc.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {out_ledger}")
    print(f"Saved: {out_summary}")
    print(f"Saved: {out_top}")
    print(f"Saved: {out_doc}")


def _is_real_city_case(cfg: EnvConfig) -> bool:
    return bool(getattr(cfg, "real_case_enabled", False)) and str(getattr(cfg, "map_source", "")).strip().lower() == "osm_dem"


def _parse_scale_scenario_pair_token(token: str) -> Tuple[str, str]:
    txt = str(token).strip().upper()
    for sep in ("-", ":", "/"):
        if sep in txt:
            left, right = txt.split(sep, 1)
            scale = left.strip()
            scenario = right.strip()
            if not scale or not scenario:
                break
            return scale, scenario
    raise ValueError(f"Invalid scale-scenario pair token: {token!r}; expected like 'M-B'")


def _parse_scale_scenario_pairs(raw: str) -> List[Tuple[str, str]]:
    return [_parse_scale_scenario_pair_token(tok) for tok in _parse_csv_list(raw)]


def _unique_field_order(fields: List[str]) -> List[str]:
    seen: set[str] = set()
    return [field for field in fields if not (field in seen or seen.add(field))]


def _pairs_from_matrix_cfg(matrix_cfg: Dict[str, Any]) -> List[Tuple[str, str]]:
    raw = matrix_cfg.get("scale_scenario_pairs", [])
    if isinstance(raw, str):
        return _parse_scale_scenario_pairs(raw)
    pairs: List[Tuple[str, str]] = []
    if not isinstance(raw, list):
        return pairs
    for item in raw:
        if isinstance(item, str):
            pairs.append(_parse_scale_scenario_pair_token(item))
            continue
        if isinstance(item, dict):
            scale = str(item.get("scale", "")).strip().upper()
            scenario = str(item.get("scenario", "")).strip().upper()
            if scale and scenario:
                pairs.append((scale, scenario))
                continue
        raise ValueError(
            "Invalid matrix.scale_scenario_pairs entry; expected string like 'M-B' "
            "or mapping with keys {scale, scenario}."
        )
    return pairs


def _flatten_cfg(cfg_yaml: Dict[str, Any], seed: int, monitor_snap_enabled: bool = False) -> EnvConfig:
    env = dict(cfg_yaml.get("env", {}))
    phy = dict(cfg_yaml.get("physics", {}))
    rew = dict(cfg_yaml.get("reward", {}))
    dst = dict(cfg_yaml.get("disturbance", {}))
    pln = dict(cfg_yaml.get("planner", {}))
    pln_toggles = dict(cfg_yaml.get("planner_toggles", {}))
    material_defaults = dict(cfg_yaml.get("material_defaults", {}))
    merged: Dict[str, Any] = {}
    # material_defaults are the lowest-priority baseline knobs; section-specific
    # env/physics/reward/disturbance/planner overrides must win.
    merged.update(material_defaults)
    merged.update(env)
    merged.update(phy)
    merged.update(rew)
    merged.update(dst)
    merged.update(pln)
    merged.update(pln_toggles)
    merged["seed"] = int(seed)
    merged["uav_monitor_snap_enabled"] = bool(monitor_snap_enabled)
    merged["enable_monitor_snap"] = bool(monitor_snap_enabled)
    return EnvConfig(**merged)


def _merge_cfg(base_yaml: Dict[str, Any], override_yaml: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base_yaml or {})
    for section in (
        "env",
        "physics",
        "reward",
        "disturbance",
        "planner",
        "planner_toggles",
        "material_defaults",
    ):
        merged_section: Dict[str, Any] = {}
        b = (base_yaml or {}).get(section, {})
        o = (override_yaml or {}).get(section, {})
        if isinstance(b, dict):
            merged_section.update(b)
        if isinstance(o, dict):
            merged_section.update(o)
        if merged_section:
            out[section] = merged_section
    return out


def _apply_l_benchmark(cfg: EnvConfig, l_benchmark_mode: str = "new") -> EnvConfig:
    if _is_real_city_case(cfg):
        raise ValueError(
            "Real-city cases are canonical R-* benchmarks and must not be routed through the L benchmark path. "
            "Use scale 'R' (for example, 'R-C') with a real_case_enabled osm_dem config."
        )
    mode = str(l_benchmark_mode).strip().lower()
    if mode not in {"old", "new"}:
        mode = "new"

    if mode == "old":
        # Approximate legacy/easier L benchmark used in earlier rounds.
        return replace(
            cfg,
            l_benchmark_mode="old",
            map_complexity="L",
            map_source="disaster_map",
            num_nodes=150,
            n_nodes=150,
            num_edges=260,
            redundant_edge_radius_m=2000.0,
            redundant_edge_prob=0.90,
            map_size_m=15000.0,
            num_trucks=4,
            num_uavs=6,
            num_routine_bulk_tasks=8,
            num_time_critical_lightweight_tasks=12,
            num_normal_tasks=8,
            num_emergency_tasks=12,
            forced_island_emergency_tasks=3,
            # L is roughly three times the M spatial scale.  All deadlines
            # and the horizon therefore scale together; otherwise they would
            # be silently clipped by max_steps before service is possible.
            # Eight-hour operating shift. The final routine deadline (step
            # 1320) remains inside the horizon instead of being clipped.
            max_steps=max(int(cfg.max_steps), 1440),
            normal_task_deadline_start_step=max(int(cfg.normal_task_deadline_start_step), 960),
            normal_task_deadline_interval_step=max(int(cfg.normal_task_deadline_interval_step), 60),
            emergency_task_deadline_start_step=max(int(cfg.emergency_task_deadline_start_step), 600),
            emergency_task_deadline_interval_step=max(int(cfg.emergency_task_deadline_interval_step), 30),
            forced_island_deadline_extension_steps=max(int(cfg.forced_island_deadline_extension_steps), 48),
            l_map_variant="L_v1_baseline",
            l_map_acceptance_mode="strict",
            l_map_generation_max_attempts=1,
        )

    # Frozen new L benchmark (realism-first v1b + relaxed acceptance gate).
    return replace(
        cfg,
        l_benchmark_mode="new",
        map_complexity="L",
        map_source="disaster_map",
        num_nodes=max(int(cfg.num_nodes), 320),
        n_nodes=max(int(cfg.n_nodes), 320),
        num_edges=max(int(cfg.num_edges), 460),
        avg_degree_min=float(max(float(getattr(cfg, "l_target_avg_degree_min", 2.8)), 2.0)),
        avg_degree_max=float(max(float(getattr(cfg, "l_target_avg_degree_max", 3.05)), float(getattr(cfg, "l_target_avg_degree_min", 2.8)))),
        map_size_m=max(float(cfg.map_size_m), 15000.0),
        num_trucks=4,
        num_uavs=6,
        num_routine_bulk_tasks=8,
        num_time_critical_lightweight_tasks=12,
        num_normal_tasks=8,
        num_emergency_tasks=12,
        forced_island_emergency_tasks=3,
        truck_speed_mps=10.0,
        uav_max_speed_mps=10.0,
        uav_flight_discharge_per_m=0.00005555555555555556,
        uav_headwind_energy_coeff=0.03,
        uav_rain_energy_coeff=0.015,
        uav_payload_energy_coeff_per_kg=0.02,
        uav_launch_min_battery_fraction=0.56,
        uav_adaptive_launch_min_floor=0.50,
        uav_high_pressure_launch_min_floor=0.46,
        uav_low_battery_goal_lock_threshold=0.38,
        uav_low_battery_force_recover_threshold=0.28,
        # New L benchmark: deadlines scale with distance and remain strictly
        # inside an eight-hour operating shift.
        max_steps=max(int(cfg.max_steps), 1440),
        normal_task_deadline_start_step=max(int(cfg.normal_task_deadline_start_step), 960),
        normal_task_deadline_interval_step=max(int(cfg.normal_task_deadline_interval_step), 60),
        emergency_task_deadline_start_step=max(int(cfg.emergency_task_deadline_start_step), 600),
        emergency_task_deadline_interval_step=max(int(cfg.emergency_task_deadline_interval_step), 30),
        forced_island_deadline_extension_steps=max(int(cfg.forced_island_deadline_extension_steps), 48),
        l_map_variant="L_v1b_orientation_tighter",
        l_map_acceptance_mode="relaxed",
        l_map_generation_max_attempts=2,
    )


def _scale_cfg(scale: str, cfg: EnvConfig, l_benchmark_mode: str = "new") -> EnvConfig:
    s = str(scale).upper().strip()
    if s == "S":
        return replace(cfg, phase="S", num_nodes=40, num_edges=64, num_trucks=1, num_uavs=1, map_size_m=5000.0)
    if s == "M":
        return replace(
            cfg,
            phase="M",
            num_nodes=80,
            num_edges=140,
            num_trucks=2,
            num_uavs=3,
            map_size_m=5000.0,
            truck_speed_mps=10.0,
            uav_max_speed_mps=10.0,
            uav_flight_discharge_per_m=0.00005555555555555556,
            uav_headwind_energy_coeff=0.03,
            uav_rain_energy_coeff=0.015,
            uav_payload_energy_coeff_per_kg=0.02,
            uav_launch_min_battery_fraction=0.58,
            uav_adaptive_launch_min_floor=0.52,
            uav_high_pressure_launch_min_floor=0.48,
            uav_low_battery_goal_lock_threshold=0.40,
            uav_low_battery_force_recover_threshold=0.30,
            # Preserve the config-declared B/C road curve.  C is a
            # communication condition, not an implicit road-severity override.
            blockage_asymptote_C=float(getattr(cfg, "blockage_asymptote_C", 0.25)),
            # M is the time baseline: 20 s per step, so emergency work starts
            # at 80 min while routine bulk starts at 140 min.
            max_steps=560,
            normal_task_deadline_start_step=420,
            normal_task_deadline_interval_step=20,
            emergency_task_deadline_start_step=240,
            emergency_task_deadline_interval_step=10,
            forced_island_deadline_extension_steps=24,
        )
    if s == "L":
        cfg_l = replace(cfg, phase="L")
        if _is_real_city_case(cfg_l):
            raise ValueError(
                "Real-city cases must use scale 'R', not 'L'. "
                "Synthetic large-map benchmarks are L-B/L-C; real-road benchmarks are R-B/R-C."
            )
        return _apply_l_benchmark(cfg_l, l_benchmark_mode=l_benchmark_mode)
    if s == "R":
        # Freeze every R-C attribute that does not depend on the eventual real
        # map.  Map identity, graph/POI paths, geometry and hashes remain
        # deliberately unset and are admitted by the formal runner only after
        # a later reviewed freeze amendment.
        cfg_r = replace(
            cfg,
            phase="R",
            map_complexity="R",
            num_trucks=4,
            num_uavs=6,
            num_routine_bulk_tasks=8,
            num_time_critical_lightweight_tasks=12,
            num_normal_tasks=8,
            num_emergency_tasks=12,
            forced_island_emergency_tasks=3,
            truck_speed_mps=10.0,
            uav_max_speed_mps=10.0,
            uav_flight_discharge_per_m=0.00005555555555555556,
            uav_headwind_energy_coeff=0.03,
            uav_rain_energy_coeff=0.015,
            uav_payload_energy_coeff_per_kg=0.02,
            uav_launch_min_battery_fraction=0.56,
            uav_adaptive_launch_min_floor=0.50,
            uav_high_pressure_launch_min_floor=0.46,
            uav_low_battery_goal_lock_threshold=0.38,
            uav_low_battery_force_recover_threshold=0.28,
            max_steps=2400,
            normal_task_deadline_start_step=1680,
            normal_task_deadline_interval_step=80,
            emergency_task_deadline_start_step=960,
            emergency_task_deadline_interval_step=40,
            forced_island_deadline_extension_steps=48,
            blockage_asymptote_scale_R=0.75,
            blockage_tau_steps_R=120.0,
        )
        if _is_real_city_case(cfg_r):
            return replace(
                cfg_r,
                map_complexity="R",
                map_source="osm_dem",
                map_size_m=max(float(cfg_r.map_size_m), float(getattr(cfg_r, "real_case_size_m", cfg_r.map_size_m))),
                num_trucks=4,
                num_uavs=6,
                num_routine_bulk_tasks=8,
                num_time_critical_lightweight_tasks=12,
                num_normal_tasks=8,
                num_emergency_tasks=12,
                forced_island_emergency_tasks=(0 if str(getattr(cfg_r, "real_case_name", "")).startswith("dujiangyan_RB_paper_legacy") else 3),
                uav_max_speed_mps=10.0,
                uav_flight_discharge_per_m=0.00005555555555555556,
                uav_headwind_energy_coeff=0.03,
                uav_rain_energy_coeff=0.015,
                uav_payload_energy_coeff_per_kg=0.02,
                uav_launch_min_battery_fraction=0.56,
                uav_adaptive_launch_min_floor=0.50,
                uav_high_pressure_launch_min_floor=0.46,
                uav_low_battery_goal_lock_threshold=0.38,
                uav_low_battery_force_recover_threshold=0.28,
                # Historical R horizons scaled dynamically from map span.  The
                # paper contract now fixes the non-map clock before the city is
                # selected, so the values inherited from cfg_r remain exact.
                max_steps=2400,
                normal_task_deadline_start_step=1680,
                normal_task_deadline_interval_step=80,
                emergency_task_deadline_start_step=960,
                emergency_task_deadline_interval_step=40,
                forced_island_deadline_extension_steps=48,
            )
        return cfg_r
    raise ValueError(f"unknown scale={scale!r}")


def _apply_experiment_sensitivity_overrides(
    cfg: EnvConfig,
    *,
    scenario: str,
    comm_coverage: float = -1.0,
    blockage_asymptote: float = -1.0,
) -> EnvConfig:
    """Apply explicit, manifest-visible E3/E4 environment manipulations."""

    sc = str(scenario).upper().strip()
    out = cfg
    if float(comm_coverage) >= 0.0:
        if sc != "C":
            raise ValueError("communication-coverage sensitivity is defined only for Scenario C")
        coverage = float(np.clip(float(comm_coverage), 0.0, 1.0))
        out = replace(
            out,
            comm_blackout_emergency_coverage=coverage,
            comm_blackout_force_disabled=bool(coverage <= 0.0),
            enable_comm_blackout=bool(coverage > 0.0),
        )
    if float(blockage_asymptote) >= 0.0:
        level = float(np.clip(float(blockage_asymptote), 0.0, 1.0))
        if sc == "B":
            out = replace(out, blockage_asymptote_B=level)
        elif sc == "C":
            out = replace(out, blockage_asymptote_C=level)
        elif level > 0.0:
            raise ValueError("positive road-blockage sensitivity is invalid for Scenario A")
    return out


def _variant_effective_flags(cfg: EnvConfig) -> Dict[str, Any]:
    return {
        # scoring shrink
        "event_bonus_shrink_enabled": bool(getattr(cfg, "erc_ablate_event_scoring_bonus", False)),
        "event_bonus_scale": float(getattr(cfg, "hrl_event_bonus_base_gain", 0.0)),
        "high_priority_bonus_full_sortie_required": bool(
            getattr(cfg, "hrl_tc_override_require_full_sortie_feasible", False)
        ),
        "support_chain_bonus_enabled": bool(
            any(
                float(getattr(cfg, k, 0.0)) > 0.0
                for k in (
                    "hrl_support_bind_bonus_critical",
                    "hrl_support_bind_bonus_warning",
                    "hrl_support_bind_bonus_bulk",
                )
            )
        ),
        "map_update_ranking_bonus_enabled": bool(not getattr(cfg, "erc_ablate_map_ranking_refresh", False)),
        # support authorization
        "support_authorization_enabled": bool(getattr(cfg, "hrl_support_bound_dispatch_enabled", False)),
        "support_authorized_ttl_steps": int(getattr(cfg, "hrl_support_no_gain_streak_threshold", 0)),
        "support_abort_if_no_launch_or_progress": bool(getattr(cfg, "hrl_support_no_gain_backoff_enabled", False)),
        "support_cooldown_after_abort_steps": int(getattr(cfg, "hrl_support_no_gain_cooldown_steps", 0)),
        # launch quality gate
        "launch_quality_gate_enabled": bool(getattr(cfg, "hrl_uav_docked_require_launch_gate_strict", False)),
        "launch_quality_require_full_sortie_feasible": bool(
            getattr(cfg, "hrl_tc_override_require_full_sortie_feasible", False)
        ),
        "launch_quality_min_recovery_margin_m": float(getattr(cfg, "hrl_tc_override_min_recovery_margin_m", 0.0)),
        "launch_quality_min_battery_margin_ratio": float(
            getattr(cfg, "hrl_tc_override_min_battery_margin_ratio", 0.0)
        ),
        "launch_quality_block_recent_reject": bool(getattr(cfg, "hrl_tc_override_block_if_recent_reject", False)),
        # tc completion chain
        "tc_completion_chain_enabled": bool(getattr(cfg, "hrl_timecritical_force_entry_enabled", False)),
        "tc_completion_window_steps": int(getattr(cfg, "hrl_timecritical_force_entry_min_gap_steps", 0)),
        "tc_completion_chain_require_delivery_feasible": bool(
            getattr(cfg, "hrl_tc_override_require_full_sortie_feasible", False)
        ),
        "tc_residual_followup_enabled": bool(getattr(cfg, "hrl_routine_protection_delivery_feasible_tc_override_enabled", False)),
        # event minimal local
        "event_minimal_local_enabled": bool(getattr(cfg, "hrl_event_admission_gate_enabled", False)),
        "weak_event_global_refresh_enabled": bool(not getattr(cfg, "erc_ablate_low_value_refresh", False)),
        "path_blocked_global_refresh_enabled": bool(getattr(cfg, "hrl_path_blocked_global_refresh_enabled", True)),
        "normal_stall_global_refresh_enabled": bool(getattr(cfg, "hrl_normal_stall_hard_refresh_enabled", True)),
        "forced_switch_wide_infeasible_enabled": bool(getattr(cfg, "hrl_soft_invalid_hard_refresh_enabled", True)),
    }


def _require_info(last_info: Dict[str, Any], key: str, *, episode_idx: int) -> Any:
    if key not in last_info:
        raise KeyError(
            f"Missing required info field '{key}' at episode={episode_idx}. "
            "Matrix export aborts to avoid silent schema corruption."
        )
    return last_info[key]


def _completion(last_info: Dict[str, Any], episode_idx: int) -> float:
    if "completion_rate" in last_info:
        return float(last_info["completion_rate"])
    if "task_completion_rate" in last_info:
        return float(last_info["task_completion_rate"])
    raise KeyError(f"Missing completion metric at episode={episode_idx}")


def _task_outcome_counts(env: BaseHeteroDisasterEnv) -> Dict[str, float]:
    delivered_normal = 0
    delivered_emergency = 0
    failed_normal = 0
    failed_emergency = 0
    for task in env.state.tasks.values():
        if task.status == TaskStatus.DELIVERED:
            if task.kind == TaskKind.NORMAL:
                delivered_normal += 1
            elif task.kind == TaskKind.EMERGENCY:
                delivered_emergency += 1
        elif task.status == TaskStatus.FAILED:
            if task.kind == TaskKind.NORMAL:
                failed_normal += 1
            elif task.kind == TaskKind.EMERGENCY:
                failed_emergency += 1
    return {
        "delivered_normal": float(delivered_normal),
        "delivered_emergency": float(delivered_emergency),
        "failed_normal": float(failed_normal),
        "failed_emergency": float(failed_emergency),
    }


def _aggregate_launch_battery_seed_metrics(episodes: List[Dict[str, float]]) -> Dict[str, float]:
    total_launch = float(
        sum(max(float(ep.get("uav_launch_count_total", 0.0)), 0.0) for ep in episodes)
    )
    if total_launch <= 0.0:
        return {
            "uav_launch_battery_fraction_mean": 0.0,
            "uav_launch_battery_fraction_min": 0.0,
        }

    weighted_mean = float(
        sum(
            float(ep.get("uav_launch_battery_fraction_mean", 0.0))
            * max(float(ep.get("uav_launch_count_total", 0.0)), 0.0)
            for ep in episodes
        )
        / max(total_launch, 1e-9)
    )
    launch_mins = [
        float(ep.get("uav_launch_battery_fraction_min", 0.0))
        for ep in episodes
        if float(ep.get("uav_launch_count_total", 0.0)) > 0.0
    ]
    return {
        "uav_launch_battery_fraction_mean": float(weighted_mean),
        "uav_launch_battery_fraction_min": float(min(launch_mins)) if launch_mins else 0.0,
    }


def _csv_safe_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else ""
    if isinstance(value, (int, np.integer, bool)):
        return value
    text = str(value)
    return "" if text.strip().lower() in {"nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"} else value


def _write_rows_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_safe_value(row.get(field, "")) for field in fields})


def _ratio_with_sanity(
    numerator: float,
    denominator: float,
    name: str,
    violations: List[str],
    *,
    max_one: bool = True,
) -> float:
    num = float(numerator)
    den = float(denominator)
    if den <= 0.0:
        return 0.0
    ratio = float(num / den)
    if max_one and ratio > 1.0 + 1e-9:
        violations.append(f"{name}:{num:.3f}/{den:.3f}")
    return ratio


def _run_episode(
    env: BaseHeteroDisasterEnv,
    planner,
    low: RuleBasedLowLevelPolicy,
    eval_seed: int,
    *,
    lightweight_metrics_only: bool = False,
    audit_flags: Optional[Dict[str, bool]] = None,
) -> Dict[str, Any]:
    audit_flags = dict(audit_flags or {})
    enable_event_ledger_detail = bool(audit_flags.get("enable_event_ledger_detail", True))
    enable_switch_decision_ledger = bool(audit_flags.get("enable_switch_decision_ledger", True))
    enable_task_outcome_export = bool(audit_flags.get("enable_task_outcome_export", False))
    enable_step_trace = bool(audit_flags.get("enable_step_trace", False))
    random.seed(int(eval_seed))
    np.random.seed(int(eval_seed) % (2**32 - 1))
    st = env.reset(seed=int(eval_seed))
    repro_digests = _reproducibility_digests(env)
    freeze_contract_digests = _physical_freeze_contract_digests(env)
    shadow_start_index = int(len(getattr(planner, "objective_shadow_records", [])))
    k2_start_index = int(len(getattr(planner, "k2_sequence_records", [])))
    k2_runtime_start_index = int(len(getattr(planner, "k2_runtime_sequence_records", [])))
    k2_sa_start_index = int(len(getattr(planner, "k2_sa_delta_records", [])))
    canonical_operator_start_index = int(len(getattr(planner, "canonical_operator_records", [])))
    support_execution_start_index = int(len(getattr(planner, "support_execution_records", [])))
    operator_weight_start_index = int(len(getattr(planner, "operator_weight_trajectory_records", [])))
    event_trigger_start_index = int(len(getattr(planner, "event_trigger_records", [])))
    sa_calibration_start_index = int(len(getattr(planner, "sa_calibration_records", [])))
    live_candidate_start_index = int(len(getattr(planner, "live_candidate_records", [])))
    ranker_runtime_start_index = int(len(getattr(planner, "ranker_runtime_records", [])))
    repair_candidate_pool_start_index = int(len(getattr(planner, "repair_candidate_pool_records", [])))
    adaptive_horizon_start_index = int(len(getattr(planner, "adaptive_horizon_records", [])))
    local_search_start_index = int(len(getattr(planner, "local_search_records", [])))
    steps = 0
    last_info: Dict[str, Any] = {}
    reward_sum = 0.0
    inf_ms: List[float] = []
    planner_ms: List[float] = []
    replan_ms: List[float] = []
    low_ms: List[float] = []
    env_ms: List[float] = []
    end2end_ms: List[float] = []
    truck_inventory_kg_sum = 0.0
    uav_loaded_fraction_sum = 0.0
    truck_inventory_kg_count = 0
    uav_loaded_fraction_count = 0
    event_replans_in_window_peak = 0.0
    event_budget_blocked_count = 0.0
    map_update_hard_seen_count = 0.0
    map_update_hard_actionable_count = 0.0
    map_update_hard_deferred_count = 0.0
    map_update_hard_immediate_refresh_count = 0.0
    map_update_hard_reason_path_blocked_count = 0.0
    map_update_hard_reason_goal_unreachable_count = 0.0
    map_update_hard_reason_ranking_changed_count = 0.0
    map_update_hard_reason_dead_end_count = 0.0
    map_update_hard_reason_recovery_path_fractured_count = 0.0
    # M-C routine diagnostics (task-level and truck-level traces).
    truck_ids = [
        str(aid)
        for aid, ag in env.state.agents.items()
        if ag.kind == AgentKind.TRUCK and (not bool(getattr(ag, "crashed", False)))
    ]
    truck_start_node: Dict[str, Optional[int]] = {
        str(aid): (None if env.state.agents[str(aid)].node is None else int(env.state.agents[str(aid)].node))
        for aid in truck_ids
    }

    def _sp_dist(a_node: Optional[int], b_node: Optional[int]) -> float:
        if a_node is None or b_node is None:
            return float("inf")
        if hasattr(env, "_decision_shortest_path_distance"):
            try:
                d = float(env._decision_shortest_path_distance(int(a_node), int(b_node)))
                return d
            except Exception:
                return float("inf")
        return float("inf")

    routine_task_diag: Dict[str, Dict[str, Any]] = {}
    for t in env.state.tasks.values():
        if t.kind != TaskKind.NORMAL:
            continue
        tid = str(t.task_id)
        task_node = int(t.demand_node)
        nearest_truck = ""
        nearest_start = float("inf")
        for taid in truck_ids:
            d0 = _sp_dist(truck_start_node.get(taid, None), task_node)
            if d0 < nearest_start:
                nearest_start = d0
                nearest_truck = taid
        routine_task_diag[tid] = {
            "task_id": tid,
            "nearest_truck": nearest_truck,
            "assigned_trucks": set(),
            "assigned_truck_steps": defaultdict(int),
            "first_assigned_step": -1,
            "last_assigned_step": -1,
            "distance_to_task_start": nearest_start,
            "distance_to_task_min": float("inf"),
            "distance_to_task_final": float("inf"),
            "prev_assigned_dist": float("inf"),
            "assigned_but_not_progressing_count": 0,
            "service_start_count": 0,
            "service_complete_count": 0,
            "service_started_seen": False,
        }

    truck_routine_diag: Dict[str, Dict[str, Any]] = {}
    for taid in truck_ids:
        truck_routine_diag[taid] = {
            "truck_id": taid,
            "routine_assigned_tasks": set(),
            "routine_completed_tasks": set(),
            "routine_service_start_count": 0,
            "routine_goal_switch_count": 0,
            "routine_reassign_count": 0,
            "stuck_steps": 0,
            "progress_dist_sum": 0.0,
            "progress_step_count": 0,
            "time_spent_servicing": 0,
            "time_spent_moving_to_routine": 0,
            "time_spent_idle_or_no_progress": 0,
            "prev_goal_tid": "",
            "prev_goal_dist": float("inf"),
        }

    # R-C seed0/1 diagnostic trackers (no planner logic impact).
    task_assign_diag: Dict[str, Dict[str, Any]] = {}
    uav_ids = [
        str(aid)
        for aid, ag in env.state.agents.items()
        if ag.kind == AgentKind.UAV and (not bool(getattr(ag, "crashed", False)))
    ]
    uav_start_node: Dict[str, Optional[int]] = {
        str(aid): (None if env.state.agents[str(aid)].node is None else int(env.state.agents[str(aid)].node))
        for aid in uav_ids
    }
    for t in env.state.tasks.values():
        tid = str(t.task_id)
        tnode = int(t.demand_node)
        nearest_truck_start = float("inf")
        nearest_any_start = float("inf")
        truck_reachable = 0
        uav_eventual_serviceable = 0
        for taid in truck_ids:
            d0 = _sp_dist(truck_start_node.get(taid, None), tnode)
            if d0 < nearest_truck_start:
                nearest_truck_start = d0
            try:
                tr_task_ok = bool(env.is_task_serviceable_by_agent(str(taid), t))
            except Exception:
                tr_task_ok = False
            if tr_task_ok:
                truck_reachable = 1
        for uid in uav_ids:
            try:
                uv_task_ok = bool(env.is_task_serviceable_by_agent(str(uid), t))
            except Exception:
                uv_task_ok = False
            if uv_task_ok:
                uav_eventual_serviceable = 1
            unode = uav_start_node.get(uid, None)
            if unode is None:
                st_u = env.state.agents.get(str(uid), None)
                if st_u is not None and st_u.follow_target is not None:
                    tnode_u = env.state.agents.get(str(st_u.follow_target), None)
                    unode = None if tnode_u is None or tnode_u.node is None else int(tnode_u.node)
            du = _sp_dist(unode, tnode)
            if du < nearest_any_start:
                nearest_any_start = du
        if nearest_any_start > nearest_truck_start:
            nearest_any_start = nearest_truck_start
        task_assign_diag[tid] = {
            "assigned_count": 0,
            "first_assigned_step": -1,
            "last_assigned_step": -1,
            "nearest_truck_distance_start": nearest_truck_start,
            "nearest_uav_or_truck_distance_start": nearest_any_start,
            "truck_reachable": int(truck_reachable),
            "uav_eventual_serviceable": int(uav_eventual_serviceable),
        }

    truck_prev_node: Dict[str, Optional[int]] = {
        str(aid): (None if env.state.agents[str(aid)].node is None else int(env.state.agents[str(aid)].node))
        for aid in truck_ids
    }
    truck_prev_normal_goal_dist: Dict[str, float] = {str(aid): float("inf") for aid in truck_ids}
    support_trace_rows: List[Dict[str, Any]] = []
    prev_support_selected_total = int(getattr(planner, "support_selected_count_total", 0))

    uav_exec_diag: Dict[str, Dict[str, Any]] = {}
    prev_uav_airborne: Dict[str, bool] = {}
    prev_uav_forced_recovery: Dict[str, bool] = {}
    seen_uav_delivered_tasks: set = set()
    prev_uav_reject_sig: Dict[str, str] = {}
    for uid in uav_ids:
        st_u = env.state.agents.get(uid, None)
        prev_uav_airborne[uid] = bool(getattr(st_u, "airborne", False))
        prev_uav_forced_recovery[uid] = bool(getattr(env, "_uav_forced_rth_latch", {}).get(uid, False))
        prev_uav_reject_sig[uid] = ""
        uav_exec_diag[uid] = {
            "assigned_task_count": 0,
            "launch_count": 0,
            "delivery_count": 0,
            "reject_count": 0,
            "reject_reason_insufficient_recovery_margin": 0,
            "reject_reason_corridor": 0,
            "reject_reason_comm_block": 0,
            "reject_reason_energy_infeasible": 0,
            "reject_reason_no_recovery": 0,
            "airborne_goal_switch_blocked_count": 0,
            "forced_recovery_count": 0,
            "rendezvous_success_count": 0,
            "launch_battery_sum": 0.0,
            "launch_battery_min": float("inf"),
        }

    prev_uav_goal_by_agent: Dict[str, Optional[str]] = {}
    uav_goal_assigned_count_total = 0.0
    agent_step_rows: List[Dict[str, Any]] = []
    task_proximity_decision_rows: List[Dict[str, Any]] = []

    def _planner_replan_counter() -> int:
        values = [int(max(getattr(planner, "replan_count", 0), 0))]
        diagnostics = getattr(planner, "alns_diagnostics", None)
        if diagnostics is not None:
            values.append(int(max(getattr(diagnostics, "replan_count", 0), 0)))
        route_manager = getattr(planner, "route_plan_manager", None)
        if route_manager is not None:
            values.append(int(max(getattr(route_manager, "alns_replan_count", 0), 0)))
        return int(max(values))

    while not st.done:
        t0 = time.perf_counter()
        t_plan0 = time.perf_counter()
        replan_before = _planner_replan_counter()
        goals = planner.plan(env)
        t_plan1 = time.perf_counter()
        replan_after = _planner_replan_counter()
        if replan_after > replan_before:
            replan_ms.extend(
                [float((t_plan1 - t_plan0) * 1000.0)]
                * int(replan_after - replan_before)
            )
        flags = getattr(planner, "_last_refresh_flags", {})
        if isinstance(flags, dict):
            event_replans_in_window_peak = float(max(event_replans_in_window_peak, float(flags.get("event_replans_in_window", 0.0))))
            if bool(flags.get("event_budget_blocked", False)):
                event_budget_blocked_count += 1.0
            map_update_hard_seen_count += float(flags.get("map_update_hard_seen_step", 0.0))
            map_update_hard_actionable_count += float(flags.get("map_update_hard_actionable_step", 0.0))
            map_update_hard_deferred_count += float(flags.get("map_update_hard_deferred_step", 0.0))
            map_update_hard_immediate_refresh_count += float(flags.get("map_update_hard_immediate_refresh_step", 0.0))
            map_update_hard_reason_path_blocked_count += float(flags.get("map_update_hard_reason_path_blocked_step", 0.0))
            map_update_hard_reason_goal_unreachable_count += float(flags.get("map_update_hard_reason_goal_unreachable_step", 0.0))
            map_update_hard_reason_ranking_changed_count += float(flags.get("map_update_hard_reason_ranking_changed_step", 0.0))
            map_update_hard_reason_dead_end_count += float(flags.get("map_update_hard_reason_dead_end_step", 0.0))
            map_update_hard_reason_recovery_path_fractured_count += float(flags.get("map_update_hard_reason_recovery_path_fractured_step", 0.0))
        env.set_recommended_goals(goals)
        # Generic task assignment diagnostics.
        for _aid, _gid in goals.items():
            if _gid is None:
                continue
            _task = env.state.tasks.get(str(_gid), None)
            if _task is None:
                continue
            _tid = str(_task.task_id)
            _tad = task_assign_diag.get(_tid, None)
            if _tad is None:
                continue
            _tad["assigned_count"] = int(_tad.get("assigned_count", 0)) + 1
            if int(_tad.get("first_assigned_step", -1)) < 0:
                _tad["first_assigned_step"] = int(env.state.step_index)
            _tad["last_assigned_step"] = int(env.state.step_index)

        for aid, ag in env.state.agents.items():
            if ag.kind != AgentKind.UAV:
                continue
            gid = goals.get(str(aid), None)
            prev_gid = prev_uav_goal_by_agent.get(str(aid), None)
            if gid is not None and str(gid) != str(prev_gid):
                uav_goal_assigned_count_total += 1.0
            if gid is not None:
                t_sel = env.state.tasks.get(str(gid), None)
                if t_sel is not None and t_sel.kind == TaskKind.EMERGENCY:
                    uav_exec_diag.setdefault(str(aid), {}).setdefault("assigned_task_count", 0)
                    uav_exec_diag[str(aid)]["assigned_task_count"] = int(uav_exec_diag[str(aid)]["assigned_task_count"]) + 1
            prev_uav_goal_by_agent[str(aid)] = None if gid is None else str(gid)

        # Routine task assignment/progress diagnostics (no planner logic impact).
        step_idx = int(env.state.step_index)
        for tid, drec in routine_task_diag.items():
            task = env.state.tasks.get(tid, None)
            if task is None:
                continue
            tnode = int(task.demand_node)
            nearest_d = float("inf")
            for taid in truck_ids:
                tr = env.state.agents.get(taid, None)
                if tr is None or bool(getattr(tr, "crashed", False)):
                    continue
                d = _sp_dist(None if tr.node is None else int(tr.node), tnode)
                if d < nearest_d:
                    nearest_d = d
            if nearest_d < float(drec["distance_to_task_min"]):
                drec["distance_to_task_min"] = nearest_d

        for taid in truck_ids:
            tr = env.state.agents.get(taid, None)
            if tr is None or bool(getattr(tr, "crashed", False)):
                continue
            gid = goals.get(taid, None)
            gtask = env.state.tasks.get(str(gid), None) if gid is not None else None
            cur_tid = ""
            cur_d = float("inf")
            is_routine_goal = bool(gtask is not None and gtask.kind == TaskKind.NORMAL and gtask.status == TaskStatus.PENDING)
            if is_routine_goal:
                cur_tid = str(gtask.task_id)
                cur_d = _sp_dist(None if tr.node is None else int(tr.node), int(gtask.demand_node))
                trec = truck_routine_diag[taid]
                trec["routine_assigned_tasks"].add(cur_tid)
                if trec["prev_goal_tid"] and trec["prev_goal_tid"] != cur_tid:
                    trec["routine_goal_switch_count"] += 1
                    trec["routine_reassign_count"] += 1
                if math.isfinite(float(trec["prev_goal_dist"])) and math.isfinite(cur_d):
                    delta = float(trec["prev_goal_dist"]) - float(cur_d)
                    if delta > 0.0:
                        trec["progress_dist_sum"] += float(delta)
                        trec["progress_step_count"] += 1
                    else:
                        trec["stuck_steps"] += 1
                else:
                    trec["stuck_steps"] += 1
                trec["prev_goal_dist"] = cur_d
                trec["prev_goal_tid"] = cur_tid
                in_service = bool(gtask.in_service_by == taid and int(getattr(gtask, "service_remaining", 0)) > 0)
                if in_service:
                    trec["time_spent_servicing"] += 1
                else:
                    if math.isfinite(cur_d):
                        trec["time_spent_moving_to_routine"] += 1
                    else:
                        trec["time_spent_idle_or_no_progress"] += 1

                rrec = routine_task_diag.get(cur_tid, None)
                if rrec is not None:
                    rrec["assigned_trucks"].add(taid)
                    rrec["assigned_truck_steps"][taid] += 1
                    if int(rrec["first_assigned_step"]) < 0:
                        rrec["first_assigned_step"] = step_idx
                    rrec["last_assigned_step"] = step_idx
                    if math.isfinite(cur_d) and math.isfinite(float(rrec["prev_assigned_dist"])):
                        if cur_d >= float(rrec["prev_assigned_dist"]) - 1e-6:
                            rrec["assigned_but_not_progressing_count"] += 1
                    rrec["prev_assigned_dist"] = cur_d
            else:
                trec = truck_routine_diag[taid]
                if trec["prev_goal_tid"]:
                    trec["prev_goal_tid"] = ""
                    trec["prev_goal_dist"] = float("inf")
                trec["time_spent_idle_or_no_progress"] += 1

        for tid, drec in routine_task_diag.items():
            task = env.state.tasks.get(tid, None)
            if task is None:
                continue
            if (not bool(drec["service_started_seen"])) and task.first_service_step is not None:
                drec["service_start_count"] = int(drec["service_start_count"]) + 1
                drec["service_started_seen"] = True
        t_low0 = time.perf_counter()
        actions = low.act(env, high_goals=goals)
        t_low1 = time.perf_counter()
        if isinstance(actions, tuple):
            actions = actions[0]
        if enable_step_trace:
            pending_tasks = [
                task for task in env.state.tasks.values()
                if task.status == TaskStatus.PENDING
            ]
            for aid, agent in env.state.agents.items():
                if agent.pos_xy is not None:
                    ax, ay = float(agent.pos_xy[0]), float(agent.pos_xy[1])
                else:
                    here = env.topology.nodes[int(agent.node or 0)]
                    ax, ay = float(here.x), float(here.y)
                nearest = min(
                    pending_tasks,
                    key=lambda task: (
                        (ax - float(env.topology.nodes[int(task.demand_node)].x)) ** 2
                        + (ay - float(env.topology.nodes[int(task.demand_node)].y)) ** 2,
                        str(task.task_id),
                    ),
                    default=None,
                )
                proposed_goal = goals.get(str(aid), None)
                effective_goal = getattr(env, "_effective_goals", {}).get(str(aid), None)
                proposed_goal_task = env.state.tasks.get(str(proposed_goal), None) if proposed_goal is not None else None
                effective_goal_task = env.state.tasks.get(str(effective_goal), None) if effective_goal is not None else None
                # Keep the legacy goal-distance column tied to the proposed
                # planner goal, and publish a second distance for the goal the
                # environment actually executes.  These can differ under
                # communication fallback, so a blank proposed distance must
                # never be interpreted as zero distance to the effective goal.
                goal_task = proposed_goal_task
                action = actions.get(str(aid), None)
                row: Dict[str, Any] = {
                    "step": int(env.state.step_index),
                    "agent_id": str(aid),
                    "agent_kind": str(agent.kind.value),
                    "node_before": "" if agent.node is None else int(agent.node),
                    "x_before": float(ax),
                    "y_before": float(ay),
                    "battery_before": float(getattr(agent, "battery", float("nan"))),
                    "uav_loaded_before": int(bool(env._uav_loaded(str(aid)))) if agent.kind.value == "uav" else "",
                    "uav_payload_kg_before": float(getattr(agent, "payload_kg_current", float("nan"))) if agent.kind.value == "uav" else "",
                    "uav_last_launch_reason": str(
                        getattr(env, "_uav_last_launch_reason", {}).get(str(aid), "")
                    ) if agent.kind.value == "uav" else "",
                    "uav_forced_rth_latch": int(bool(
                        getattr(env, "_uav_forced_rth_latch", {}).get(str(aid), False)
                    )) if agent.kind.value == "uav" else "",
                    "uav_sortie_contract_task": str(
                        getattr(env, "_uav_sortie_contract_task", {}).get(str(aid), "")
                    ) if agent.kind.value == "uav" else "",
                    "uav_post_transfer_contract_task": str(
                        getattr(env, "_uav_post_transfer_contract_task", {}).get(
                            str(aid), ""
                        )
                    ) if agent.kind.value == "uav" else "",
                    "uav_sortie_recovery_suspended": int(
                        str(aid) in set(getattr(env, "_uav_sortie_recovery_suspended", set()) or set())
                    ) if agent.kind.value == "uav" else "",
                    # AgentState represents docking through follow_target; it
                    # does not expose a maintained `airborne` field.
                    "airborne_before": int(agent.kind.value == "uav" and getattr(agent, "follow_target", None) is None),
                    "follow_truck_id_before": "" if getattr(agent, "follow_target", None) is None else str(getattr(agent, "follow_target")),
                    "comm_blocked": int(bool(getattr(env, "comm_blocked", {}).get(str(aid), False))),
                    "comm_block_reason": str(getattr(env, "_comm_block_reason", {}).get(str(aid), "")),
                    "proposed_goal": "" if proposed_goal is None else str(proposed_goal),
                    "effective_goal": "" if effective_goal is None else str(effective_goal),
                    "goal_kind": "" if goal_task is None else str(goal_task.kind.value),
                    "goal_distance_euclidean_m": float("nan") if goal_task is None else float(np.hypot(ax - float(env.topology.nodes[int(goal_task.demand_node)].x), ay - float(env.topology.nodes[int(goal_task.demand_node)].y))),
                    "goal_distance_road_m": float("nan") if goal_task is None or agent.node is None else float(env._decision_shortest_path_distance(int(agent.node), int(goal_task.demand_node))),
                    "effective_goal_kind": "" if effective_goal_task is None else str(effective_goal_task.kind.value),
                    "effective_goal_distance_euclidean_m": float("nan") if effective_goal_task is None else float(np.hypot(ax - float(env.topology.nodes[int(effective_goal_task.demand_node)].x), ay - float(env.topology.nodes[int(effective_goal_task.demand_node)].y))),
                    "effective_goal_distance_road_m": float("nan") if effective_goal_task is None or agent.node is None else float(env._decision_shortest_path_distance(int(agent.node), int(effective_goal_task.demand_node))),
                    "nearest_pending_task": "" if nearest is None else str(nearest.task_id),
                    "nearest_pending_kind": "" if nearest is None else str(nearest.kind.value),
                    "nearest_pending_distance_euclidean_m": float("nan") if nearest is None else float(np.hypot(ax - float(env.topology.nodes[int(nearest.demand_node)].x), ay - float(env.topology.nodes[int(nearest.demand_node)].y))),
                    "action_type": "" if action is None else type(action).__name__,
                    "action_target_node": "" if action is None or getattr(action, "target_node", None) is None else int(getattr(action, "target_node")),
                    "action_stay": int(bool(getattr(action, "stay", False))),
                    "action_vx": float(getattr(action, "vx", 0.0)),
                    "action_vy": float(getattr(action, "vy", 0.0)),
                    "action_bind_truck_id": "" if action is None or getattr(action, "bind_truck_id", None) is None else str(getattr(action, "bind_truck_id")),
                    "action_takeoff": int(bool(getattr(action, "takeoff", False))),
                    "truck_bulk_inventory_kg_before": float(max(getattr(agent, "bulk_inventory_kg_current", 0.0), 0.0)) if agent.kind.value == "truck" else "",
                    "truck_timecritical_inventory_kg_before": float(max(getattr(agent, "timecritical_inventory_kg_current", 0.0), 0.0)) if agent.kind.value == "truck" else "",
                    "route_plan_v2_assist_active": int(
                        agent.kind.value == "truck"
                        and str(aid) in dict(getattr(env, "_planner_truck_assist_waypoint_by_truck", {}))
                    ),
                    "route_plan_v2_stay_reason": str(
                        dict(getattr(env, "_planner_route_plan_stay_reason_by_agent", {})).get(str(aid), "")
                    ),
                }
                agent_step_rows.append(row)

            # Task-centred audit: retain the concrete nearest-agent and truck
            # decision state for every still-pending task.  This makes it
            # possible to distinguish "the truck never got close" from
            # "the truck was close but selected another task".
            for task in pending_tasks:
                tnode = env.topology.nodes[int(task.demand_node)]
                def _euclid_to_task(_agent: Any) -> float:
                    if _agent.pos_xy is not None:
                        px, py = float(_agent.pos_xy[0]), float(_agent.pos_xy[1])
                    else:
                        pnode = env.topology.nodes[int(_agent.node or 0)]
                        px, py = float(pnode.x), float(pnode.y)
                    return float(np.hypot(px - float(tnode.x), py - float(tnode.y)))

                trucks_now = [(str(aid), ag) for aid, ag in env.state.agents.items() if ag.kind.value == "truck" and not bool(getattr(ag, "crashed", False))]
                uavs_now = [(str(aid), ag) for aid, ag in env.state.agents.items() if ag.kind.value == "uav" and not bool(getattr(ag, "crashed", False))]
                near_truck_id, near_truck = min(trucks_now, key=lambda item: (_euclid_to_task(item[1]), item[0]))
                near_uav_id, near_uav = min(uavs_now, key=lambda item: (_euclid_to_task(item[1]), item[0]))
                truck_node = getattr(near_truck, "node", None)
                try:
                    truck_road_dist = float("inf") if truck_node is None else float(env._decision_shortest_path_distance(int(truck_node), int(task.demand_node)))
                except Exception:
                    truck_road_dist = float("inf")
                req_tc_kg = float(max(float(getattr(task, "demand_kg", 0.0)), 0.0))
                truck_tc_kg = float(max(getattr(near_truck, "timecritical_inventory_kg_current", 0.0), 0.0))
                truck_bulk_kg = float(max(getattr(near_truck, "bulk_inventory_kg_current", 0.0), 0.0))
                truck_stock_ok = bool(
                    (truck_tc_kg if str(task.kind.value) == "emergency" else truck_bulk_kg)
                    + 1e-9
                    >= req_tc_kg
                )
                truck_road_reachable = bool(np.isfinite(truck_road_dist))
                near_truck_goal = goals.get(str(near_truck_id), None)
                near_truck_goal_task = env.state.tasks.get(str(near_truck_goal), None) if near_truck_goal is not None else None
                near_truck_action = actions.get(str(near_truck_id), None)
                task_proximity_decision_rows.append(
                    {
                        "step": int(env.state.step_index),
                        "task_id": str(task.task_id),
                        "task_kind": str(task.kind.value),
                        "task_node": int(task.demand_node),
                        "task_lifeline": float(getattr(task, "lifeline_current", float("nan"))),
                        "nearest_truck_id": str(near_truck_id),
                        "nearest_truck_euclidean_m": float(_euclid_to_task(near_truck)),
                        "nearest_truck_road_distance_m": float(truck_road_dist),
                        "nearest_truck_at_task_node": int(truck_node is not None and int(truck_node) == int(task.demand_node)),
                        "nearest_truck_emergency_stock_kg": float(truck_tc_kg),
                        "nearest_truck_bulk_stock_kg": float(truck_bulk_kg),
                        "nearest_truck_stock_sufficient": int(truck_stock_ok),
                        "nearest_truck_road_reachable": int(truck_road_reachable),
                        "nearest_truck_proposed_goal": "" if near_truck_goal is None else str(near_truck_goal),
                        "nearest_truck_proposed_goal_kind": "" if near_truck_goal_task is None else str(near_truck_goal_task.kind.value),
                        "nearest_truck_action_target_node": "" if near_truck_action is None or getattr(near_truck_action, "target_node", None) is None else int(getattr(near_truck_action, "target_node")),
                        "nearest_truck_action_stay": int(bool(getattr(near_truck_action, "stay", False))),
                        "nearest_truck_comm_blocked": int(bool(getattr(env, "comm_blocked", {}).get(str(near_truck_id), False))),
                        "nearest_truck_route_plan_v2_assist_active": int(
                            str(near_truck_id) in dict(getattr(env, "_planner_truck_assist_waypoint_by_truck", {}))
                        ),
                        "nearest_truck_route_plan_v2_stay_reason": str(
                            dict(getattr(env, "_planner_route_plan_stay_reason_by_agent", {})).get(str(near_truck_id), "")
                        ),
                        "task_route_contract_owner": str(getattr(task, "route_contract_owner", "") or ""),
                        "task_route_contract_truck": str(getattr(task, "route_contract_truck", "") or ""),
                        "nearest_uav_id": str(near_uav_id),
                        "nearest_uav_euclidean_m": float(_euclid_to_task(near_uav)),
                        "nearest_uav_battery": float(getattr(near_uav, "battery", float("nan"))),
                        "nearest_uav_airborne": int(getattr(near_uav, "follow_target", None) is None),
                        "nearest_uav_follow_truck_id": "" if getattr(near_uav, "follow_target", None) is None else str(getattr(near_uav, "follow_target")),
                    }
                )
        t_env0 = time.perf_counter()
        out = env.step(actions)
        t_env1 = time.perf_counter()
        inf_ms.append(float((t_low1 - t_plan0) * 1000.0))
        planner_ms.append(float((t_plan1 - t_plan0) * 1000.0))
        low_ms.append(float((t_low1 - t_low0) * 1000.0))
        env_ms.append(float((t_env1 - t_env0) * 1000.0))
        end2end_ms.append(float((t_env1 - t0) * 1000.0))
        st = out.state
        last_info = out.info
        # Per-truck support/recovery trace (execution-layer vs planner-layer counters).
        support_selected_total_now = int(getattr(planner, "support_selected_count_total", 0))
        support_selected_step = int(max(support_selected_total_now - prev_support_selected_total, 0))
        prev_support_selected_total = support_selected_total_now
        forward_support_step = int(last_info.get("truck_forward_support_count_step", 0))
        recovery_support_step = int(last_info.get("truck_recovery_support_count_step", 0))
        launch_step = int(last_info.get("uav_launch_count_step", 0))
        delivery_step = int(last_info.get("delivered_by_uav_step", 0))
        support_uav_map = dict(getattr(planner, "_support_bound_chain_uav_id", {}))
        support_task_map = dict(getattr(planner, "_support_bound_chain_task_id", {}))
        for taid in truck_ids:
            tr_now = env.state.agents.get(str(taid), None)
            if tr_now is None:
                continue
            gid_now = goals.get(str(taid), None)
            t_now = env.state.tasks.get(str(gid_now), None) if gid_now is not None else None
            normal_goal_id = ""
            if t_now is not None and t_now.kind == TaskKind.NORMAL:
                normal_goal_id = str(t_now.task_id)
            cur_node = None if tr_now.node is None else int(tr_now.node)
            prev_node = truck_prev_node.get(str(taid), None)
            moved_d = _sp_dist(prev_node, cur_node) if (prev_node is not None and cur_node is not None) else float("nan")
            truck_prev_node[str(taid)] = cur_node
            dist_normal = float("nan")
            progress_delta = float("nan")
            if normal_goal_id:
                ng = env.state.tasks.get(str(normal_goal_id), None)
                if ng is not None:
                    dist_normal = _sp_dist(cur_node, int(ng.demand_node))
                prev_dist = float(truck_prev_normal_goal_dist.get(str(taid), float("inf")))
                if math.isfinite(prev_dist) and math.isfinite(dist_normal):
                    progress_delta = float(prev_dist - dist_normal)
                truck_prev_normal_goal_dist[str(taid)] = dist_normal
            support_tid = str(support_task_map.get(str(taid), "")) if str(taid) in support_task_map else ""
            support_uid = str(support_uav_map.get(str(taid), "")) if str(taid) in support_uav_map else ""
            dist_support = float("nan")
            if support_tid:
                stask = env.state.tasks.get(support_tid, None)
                if stask is not None and cur_node is not None:
                    dist_support = _sp_dist(cur_node, int(stask.demand_node))
            is_servicing = False
            service_target = ""
            if t_now is not None and str(getattr(t_now, "in_service_by", "")) == str(taid):
                is_servicing = True
                service_target = str(t_now.task_id)
            support_reason = ""
            if t_now is not None and t_now.kind == TaskKind.EMERGENCY:
                support_reason = "planner_emergency_goal"
            elif (forward_support_step > 0 or recovery_support_step > 0):
                support_reason = "env_auto_support"
            support_trace_rows.append(
                {
                    "step": int(env.state.step_index),
                    "truck_id": str(taid),
                    "current_goal": "" if gid_now is None else str(gid_now),
                    "current_goal_type": "" if t_now is None else str(getattr(t_now, "task_class", "")),
                    "normal_goal_id": str(normal_goal_id),
                    "support_target_uav": str(support_uid),
                    "support_target_task": str(support_tid),
                    "support_reason": str(support_reason),
                    "support_selected_by_planner": int(support_selected_step),
                    "forward_support_event": int(forward_support_step),
                    "recovery_support_event": int(recovery_support_step),
                    "truck_position": "" if cur_node is None else str(cur_node),
                    "distance_to_normal_goal": float(dist_normal) if math.isfinite(dist_normal) else float("nan"),
                    "distance_to_support_task": float(dist_support) if math.isfinite(dist_support) else float("nan"),
                    "moved_distance_this_step": float(moved_d) if math.isfinite(float(moved_d)) else float("nan"),
                    "is_servicing": int(bool(is_servicing)),
                    "service_target": str(service_target),
                    "normal_task_progress_delta": float(progress_delta) if math.isfinite(progress_delta) else float("nan"),
                    "uav_launch_nearby_after_support": int(launch_step),
                    "uav_delivery_after_support": int(delivery_step),
                }
            )

        # UAV execution-chain diagnostics.
        delivered_now = set()
        for t_cur in env.state.tasks.values():
            if t_cur.status == TaskStatus.DELIVERED:
                db = str(getattr(t_cur, "delivered_by", ""))
                if db.startswith("uav_"):
                    delivered_now.add((str(t_cur.task_id), db))
        for (tid_done, uid_done) in delivered_now:
            if (tid_done, uid_done) in seen_uav_delivered_tasks:
                continue
            seen_uav_delivered_tasks.add((tid_done, uid_done))
            if uid_done in uav_exec_diag:
                uav_exec_diag[uid_done]["delivery_count"] = int(uav_exec_diag[uid_done]["delivery_count"]) + 1

        last_launch_reason = dict(getattr(env, "_uav_last_launch_reason", {}))
        forced_map = dict(getattr(env, "_uav_forced_rth_latch", {}))
        for uid in uav_ids:
            st_u = env.state.agents.get(str(uid), None)
            if st_u is None:
                continue
            airborne_now = bool(getattr(st_u, "airborne", False))
            if (not bool(prev_uav_airborne.get(uid, False))) and airborne_now:
                uav_exec_diag[uid]["launch_count"] = int(uav_exec_diag[uid]["launch_count"]) + 1
                b = float(getattr(st_u, "battery", float("nan")))
                if math.isfinite(b):
                    uav_exec_diag[uid]["launch_battery_sum"] = float(uav_exec_diag[uid]["launch_battery_sum"]) + float(b)
                    uav_exec_diag[uid]["launch_battery_min"] = float(min(float(uav_exec_diag[uid]["launch_battery_min"]), float(b)))
            if bool(prev_uav_airborne.get(uid, False)) and (not airborne_now):
                if st_u.follow_target is not None:
                    uav_exec_diag[uid]["rendezvous_success_count"] = int(uav_exec_diag[uid]["rendezvous_success_count"]) + 1
            prev_uav_airborne[uid] = bool(airborne_now)

            forced_now = bool(forced_map.get(uid, False))
            if (not bool(prev_uav_forced_recovery.get(uid, False))) and forced_now:
                uav_exec_diag[uid]["forced_recovery_count"] = int(uav_exec_diag[uid]["forced_recovery_count"]) + 1
            prev_uav_forced_recovery[uid] = bool(forced_now)

            rr = str(last_launch_reason.get(uid, "")).strip().lower()
            sig = f"{int(env.state.step_index)}::{rr}"
            if rr and sig != str(prev_uav_reject_sig.get(uid, "")):
                prev_uav_reject_sig[uid] = sig
                if any(k in rr for k in ["recovery_margin", "insufficient_recovery_margin"]):
                    uav_exec_diag[uid]["reject_reason_insufficient_recovery_margin"] = int(uav_exec_diag[uid]["reject_reason_insufficient_recovery_margin"]) + 1
                    uav_exec_diag[uid]["reject_count"] = int(uav_exec_diag[uid]["reject_count"]) + 1
                elif "corridor" in rr:
                    uav_exec_diag[uid]["reject_reason_corridor"] = int(uav_exec_diag[uid]["reject_reason_corridor"]) + 1
                    uav_exec_diag[uid]["reject_count"] = int(uav_exec_diag[uid]["reject_count"]) + 1
                elif "comm_block" in rr or "comm" in rr:
                    uav_exec_diag[uid]["reject_reason_comm_block"] = int(uav_exec_diag[uid]["reject_reason_comm_block"]) + 1
                    uav_exec_diag[uid]["reject_count"] = int(uav_exec_diag[uid]["reject_count"]) + 1
                elif "energy" in rr:
                    uav_exec_diag[uid]["reject_reason_energy_infeasible"] = int(uav_exec_diag[uid]["reject_reason_energy_infeasible"]) + 1
                    uav_exec_diag[uid]["reject_count"] = int(uav_exec_diag[uid]["reject_count"]) + 1
                elif "no_recovery" in rr:
                    uav_exec_diag[uid]["reject_reason_no_recovery"] = int(uav_exec_diag[uid]["reject_reason_no_recovery"]) + 1
                    uav_exec_diag[uid]["reject_count"] = int(uav_exec_diag[uid]["reject_count"]) + 1
        truck_inventory_kg_sum += float(_require_info(out.info, "truck_inventory_kg_current_mean", episode_idx=0))
        uav_loaded_fraction_sum += float(_require_info(out.info, "uav_loaded_fraction", episode_idx=0))
        truck_inventory_kg_count += 1
        uav_loaded_fraction_count += 1
        steps += 1
        reward_sum += float(sum(out.rewards.values()) / max(len(out.rewards), 1))

    # Flush the final event-refresh window so episode-end diagnostics include
    # the last refresh outcome segment.
    finalize_window = getattr(planner, "_finalize_active_event_refresh_window", None)
    if callable(finalize_window):
        try:
            finalize_window(env, int(env.state.step_index) + 1)
        except Exception:
            pass

    dt = float(last_info.get("dt_seconds", env.cfg.dt_seconds))
    makespan_seconds = float(steps * dt)

    normal_block = float(_require_info(last_info, "normal_tasks_blocked_by_supply_count", episode_idx=0))
    emergency_block = float(_require_info(last_info, "emergency_tasks_blocked_by_supply_count", episode_idx=0))
    outcomes = _task_outcome_counts(env)
    delivered_normal = float(outcomes["delivered_normal"])
    delivered_emergency = float(outcomes["delivered_emergency"])
    failed_normal = float(outcomes["failed_normal"])
    failed_emergency = float(outcomes["failed_emergency"])

    normal_serviceability_denom = max(delivered_normal + failed_normal + normal_block, 1e-6)
    emergency_serviceability_denom = max(delivered_emergency + failed_emergency + emergency_block, 1e-6)

    truck_replenish_count_total = float(_require_info(last_info, "truck_replenish_count_total", episode_idx=0))
    uav_reload_wait_steps_total = float(_require_info(last_info, "uav_reload_wait_steps_total", episode_idx=0))
    truck_distance_total_m = float(_require_info(last_info, "truck_distance_total_m", episode_idx=0))
    uav_distance_total_m = float(_require_info(last_info, "uav_distance_total_m", episode_idx=0))
    fleet_distance_total_m = float(_require_info(last_info, "fleet_distance_total_m", episode_idx=0))
    delivered_task_last_step = float(_require_info(last_info, "delivered_task_last_step", episode_idx=0))
    delivered_task_last_time_seconds = float(_require_info(last_info, "delivered_task_last_time_seconds", episode_idx=0))
    terminal_task_last_step = float(_require_info(last_info, "terminal_task_last_step", episode_idx=0))
    terminal_task_last_time_seconds = float(_require_info(last_info, "terminal_task_last_time_seconds", episode_idx=0))
    task_end_step = float(_require_info(last_info, "task_end_step", episode_idx=0))
    task_end_time_seconds = float(_require_info(last_info, "task_end_time_seconds", episode_idx=0))
    truck_inventory_kg_current_mean = float(truck_inventory_kg_sum / max(truck_inventory_kg_count, 1))
    uav_loaded_fraction = float(uav_loaded_fraction_sum / max(uav_loaded_fraction_count, 1))

    # Planner-side diagnostics for Patch-1/Patch-2.
    tc_tier3_candidate = float(getattr(planner, "timecritical_tier3_candidate_count_total", 0.0))
    tc_tier3_selected = float(getattr(planner, "timecritical_tier3_selected_count_total", 0.0))
    tc_tier2_candidate = float(getattr(planner, "timecritical_tier2_candidate_count_total", 0.0))
    tc_tier2_selected = float(getattr(planner, "timecritical_tier2_selected_count_total", 0.0))
    tc_ignored = float(getattr(planner, "timecritical_candidate_ignored_count_total", 0.0))
    tc_direct_feasible = float(getattr(planner, "tc_direct_feasible_count_total", 0.0))
    tc_support_required = float(getattr(planner, "tc_support_required_count_total", 0.0))
    tc_truly_infeasible = float(getattr(planner, "tc_truly_infeasible_count_total", 0.0))
    tc_support_lock_created = float(getattr(planner, "tc_support_lock_created_count_total", 0.0))
    tc_support_lock_to_dispatch = float(getattr(planner, "tc_support_lock_to_dispatch_count_total", 0.0))
    region_commitment_setup = float(getattr(planner, "region_commitment_setup_count_total", 0.0))
    region_commitment_region_count = float(len(getattr(planner, "_region_centers_xy", {}) or {}))
    region_commitment_effective_k = float(getattr(planner, "_region_commitment_effective_k", 0.0))
    region_commitment_effective_enabled = float(1.0 if bool(getattr(planner, "_region_commitment_enabled_effective", False)) else 0.0)
    region_commitment_auto_score = float(getattr(planner, "_region_commitment_auto_score", 0.0))
    region_commitment_separation_score = float(getattr(planner, "_region_commitment_separation_score", 0.0))
    region_commitment_load_balance_score = float(getattr(planner, "_region_commitment_load_balance_score", 0.0))
    region_commitment_coverage_score = float(getattr(planner, "_region_commitment_coverage_score", 0.0))
    region_commitment_strength = float(getattr(planner, "_region_commitment_strength", 0.0))
    region_commitment_auto_enabled = float(getattr(planner, "region_commitment_auto_enabled_count_total", 0.0))
    region_commitment_auto_disabled = float(getattr(planner, "region_commitment_auto_disabled_count_total", 0.0))
    region_commitment_local_candidates = float(getattr(planner, "region_commitment_local_candidate_count_total", 0.0))
    region_commitment_cross_filtered = float(getattr(planner, "region_commitment_cross_filtered_count_total", 0.0))
    region_commitment_cross_override = float(getattr(planner, "region_commitment_cross_override_count_total", 0.0))
    region_commitment_outlier_tasks = float(getattr(planner, "region_commitment_outlier_task_count_total", 0.0))
    region_commitment_outlier_filtered = float(getattr(planner, "region_commitment_outlier_filtered_count_total", 0.0))
    region_commitment_outlier_override = float(getattr(planner, "region_commitment_outlier_override_count_total", 0.0))

    sup_bound_tc = float(getattr(planner, "support_selected_with_bound_timecritical_delivery_count_total", 0.0))
    sup_without_tc = float(getattr(planner, "support_selected_without_bound_timecritical_delivery_count_total", 0.0))
    sup_filtered_no_bind = float(getattr(planner, "support_filtered_no_bound_timecritical_delivery_count_total", 0.0))
    sup_bound_bulk = float(getattr(planner, "support_selected_with_bound_bulk_delivery_count_total", 0.0))
    sup_bind_success_rate = float(sup_bound_tc / max(sup_bound_tc + sup_without_tc, 1.0))

    support_selected = float(getattr(planner, "support_selected_count_total", getattr(planner, "truck_support_selected_count_total", 0.0)))
    support_gain = float(getattr(planner, "support_improves_serviceability_count_total", getattr(planner, "truck_support_improves_serviceability_count_total", 0.0)))
    support_no_gain = float(getattr(planner, "support_no_gain_count_total", getattr(planner, "truck_support_no_gain_count_total", 0.0)))
    support_conversion_rate = float(support_gain / max(support_selected, 1.0))
    relaxed_conversion_rate = float(last_info.get("relaxed_conversion_rate", 0.0))

    routine_assigned_no_progress_count = 0.0
    routine_near_completion_interrupted_count = 0.0
    for tid, drec in routine_task_diag.items():
        task = env.state.tasks.get(str(tid), None)
        if task is None:
            continue
        assigned_steps = sum(int(v) for v in dict(drec.get("assigned_truck_steps", {})).values())
        no_progress = int(drec.get("assigned_but_not_progressing_count", 0))
        service_starts = int(1 if getattr(task, "first_service_step", None) is not None else 0)
        if task.status != TaskStatus.DELIVERED and assigned_steps > 0 and service_starts <= 0 and no_progress > 0:
            routine_assigned_no_progress_count += 1.0
        if task.status != TaskStatus.DELIVERED and service_starts <= 0 and no_progress >= 3:
            routine_near_completion_interrupted_count += 1.0

    tc_assigned_no_launch_count = 0.0
    tc_service_start_no_completion_count = 0.0
    for tid, tad in task_assign_diag.items():
        task = env.state.tasks.get(str(tid), None)
        if task is None or task.kind != TaskKind.EMERGENCY:
            continue
        assigned = int(tad.get("assigned_count", 0))
        service_started = bool(getattr(task, "first_service_step", None) is not None)
        delivered = bool(task.status == TaskStatus.DELIVERED)
        if assigned > 0 and (not service_started) and (not delivered):
            tc_assigned_no_launch_count += 1.0
        if service_started and (not delivered):
            tc_service_start_no_completion_count += 1.0

    recovery_feas_eval = float(getattr(planner, "uav_recovery_feasibility_eval_count_total", 0.0))
    if recovery_feas_eval <= 0.0:
        recovery_feas_eval = float(last_info.get("uav_launch_feasibility_eval_count", 0.0))
    reject_margin = float(last_info.get("uav_task_reject_recovery_margin_count", 0.0))
    recovery_reject_rate = float(reject_margin / max(recovery_feas_eval, 1.0))
    uav_direct_ready_eval_count_total = float(getattr(planner, "uav_task_selected_count_total", 0.0))
    uav_direct_ready_selected_count_total = float(getattr(planner, "uav_task_selected_count_total", 0.0))
    uav_launch_count_total = float(_require_info(last_info, "uav_launch_count_total", episode_idx=0))
    uav_delivery_count_total = float(_require_info(last_info, "uav_delivery_count_total", episode_idx=0))
    truck_completed_task_count_total = float(_require_info(last_info, "truck_delivered_tasks_total", episode_idx=0))
    uav_completed_task_count_total = float(_require_info(last_info, "uav_delivered_tasks_total", episode_idx=0))
    ratio_violations: List[str] = []
    goal_assignment_to_launch_ratio = _ratio_with_sanity(
        numerator=uav_launch_count_total,
        denominator=float(uav_goal_assigned_count_total),
        name="goal_assignment_to_launch_ratio",
        violations=ratio_violations,
        # The denominator counts distinct UAV goal transitions, not sortie
        # opportunities. A persistent contract may legitimately span
        # recovery, relaunch or transfer, so this is an intensity and is not
        # bounded by one.
        max_one=False,
    )
    direct_ready_to_launch_ratio = _ratio_with_sanity(
        numerator=uav_launch_count_total,
        denominator=float(uav_direct_ready_eval_count_total),
        name="direct_ready_to_launch_ratio",
        violations=ratio_violations,
        # Planner selections may also represent persistent contracts rather
        # than independent one-shot launch opportunities.
        max_one=False,
    )
    launch_to_completion_ratio = _ratio_with_sanity(
        numerator=uav_delivery_count_total,
        denominator=uav_launch_count_total,
        name="launch_to_completion_ratio",
        violations=ratio_violations,
        max_one=False,
    )
    hard_offender_rows: List[Dict[str, Any]] = []
    hard_stats = getattr(planner, "_hard_event_offender_stats", {})
    if (not bool(lightweight_metrics_only)) and enable_event_ledger_detail and isinstance(hard_stats, dict):
        for (_k, rec) in hard_stats.items():
            if not isinstance(rec, dict):
                continue
            hard_offender_rows.append(
                {
                    "agent_id": str(rec.get("agent_id", "")),
                    "task_id": str(rec.get("task_id", "")),
                    "hard_event_reason": str(rec.get("reason", "unknown")),
                    "count": float(rec.get("count", 0.0)),
                    "first_step": float(rec.get("first_step", 0.0)),
                    "last_step": float(rec.get("last_step", 0.0)),
                    "no_goal_change_count": float(rec.get("no_goal_change_count", 0.0)),
                    "goal_change_count": float(rec.get("goal_change_count", 0.0)),
                    "launch_count_after_event": float(rec.get("launch_count_after_event", 0.0)),
                    "completion_count_after_event": float(rec.get("completion_count_after_event", 0.0)),
                    "reject_count_after_event": float(rec.get("reject_count_after_event", 0.0)),
                    "goal_switch_after_event": float(rec.get("goal_switch_after_event", 0.0)),
                    "current_goal_type": str(rec.get("current_goal_type", "")),
                    "proposed_goal_type": str(rec.get("proposed_goal_type", "")),
                    "task_status": str(rec.get("task_status", "")),
                    "distance_to_goal_mean": (
                        float(rec.get("distance_to_goal_sum", 0.0)) / max(float(rec.get("distance_to_goal_count", 0.0)), 1.0)
                        if float(rec.get("distance_to_goal_count", 0.0)) > 0.0
                        else float("nan")
                    ),
                    "battery_mean": (
                        float(rec.get("battery_sum", 0.0)) / max(float(rec.get("battery_count", 0.0)), 1.0)
                        if float(rec.get("battery_count", 0.0)) > 0.0
                        else float("nan")
                    ),
                }
            )
    hard_reason_rows: List[Dict[str, Any]] = []
    hard_reason_stats = getattr(planner, "_hard_event_reason_outcome_stats", {})
    if (not bool(lightweight_metrics_only)) and enable_event_ledger_detail and isinstance(hard_reason_stats, dict):
        for _reason, rec in hard_reason_stats.items():
            if not isinstance(rec, dict):
                continue
            total = float(rec.get("total_refresh_count", 0.0))
            nogc = float(rec.get("no_goal_change_count", 0.0))
            hard_reason_rows.append(
                {
                    "hard_event_reason": str(rec.get("hard_event_reason", _reason)),
                    "total_refresh_count": total,
                    "no_goal_change_count": nogc,
                    "goal_change_count": float(rec.get("goal_change_count", 0.0)),
                    "no_goal_change_ratio": float(nogc / max(total, 1.0)),
                    "followed_by_launch_count": float(rec.get("followed_by_launch_count", 0.0)),
                    "followed_by_completion_count": float(rec.get("followed_by_completion_count", 0.0)),
                    "followed_by_reject_count": float(rec.get("followed_by_reject_count", 0.0)),
                    "followed_by_stall_count": float(rec.get("followed_by_stall_count", 0.0)),
                }
            )

    task_outcome_rows: List[Dict[str, Any]] = []
    routine_trace_rows: List[Dict[str, Any]] = []
    truck_routine_summary_rows: List[Dict[str, Any]] = []

    # Finalize routine per-task diagnostics.
    export_task_outcomes = (not bool(lightweight_metrics_only)) or bool(enable_task_outcome_export)
    for t in (env.state.tasks.values() if export_task_outcomes else []):
        tid = str(t.task_id)
        completed = int(t.status == TaskStatus.DELIVERED)
        completed_step = int(t.delivered_step) if t.delivered_step is not None else -1
        final_status = str(getattr(t.status, "name", str(t.status))).lower()
        deadline_step = int(getattr(t, "deadline_step", -1))
        failed_step = int(getattr(t, "failed_step", -1)) if getattr(t, "failed_step", None) is not None else -1
        if bool(getattr(t, "failed_due_to_lifeline_zero", False)):
            failed_reason = "lifeline_zero"
        elif t.status == TaskStatus.FAILED and failed_step > deadline_step >= 0:
            failed_reason = "deadline_exceeded"
        elif t.status == TaskStatus.FAILED:
            failed_reason = "failed_other"
        elif not completed:
            failed_reason = "horizon_unfinished"
        else:
            failed_reason = ""
        demand_kg = float(max(getattr(t, "demand_kg", 0.0), 0.0))
        fulfilled_mass_kg = float(max(getattr(t, "fulfilled_mass_kg", 0.0), 0.0))
        task_outcome_rows.append(
            {
                "task_id": tid,
                "task_kind": str(getattr(t.kind, "value", str(t.kind))).lower(),
                "task_class": str(getattr(t, "task_class", "")),
                "demand_kg": float(demand_kg),
                "urgency_score": float(getattr(t, "urgency_score", 0.0)),
                "lifeline_init": float(getattr(t, "lifeline_init", float("nan"))),
                "lifeline_final": float(getattr(t, "lifeline_current", float("nan"))),
                "deadline_step": int(deadline_step),
                "deadline_seconds": float(max(deadline_step, 0) * float(env.cfg.dt_seconds)),
                "assigned_count": int(task_assign_diag.get(tid, {}).get("assigned_count", 0)),
                "first_assigned_step": int(task_assign_diag.get(tid, {}).get("first_assigned_step", -1)),
                "last_assigned_step": int(task_assign_diag.get(tid, {}).get("last_assigned_step", -1)),
                "service_start_count": int(1 if getattr(t, "first_service_step", None) is not None else 0),
                "first_service_step": int(getattr(t, "first_service_step", -1)) if getattr(t, "first_service_step", None) is not None else -1,
                "completed": int(completed),
                "completed_step": int(completed_step),
                "completed_seconds": float(completed_step * float(env.cfg.dt_seconds)) if completed_step >= 0 else -1.0,
                "on_time": int(bool(completed and completed_step <= deadline_step)),
                "completion_quality": float(min(fulfilled_mass_kg / max(demand_kg, 1e-9), 1.0)),
                "fulfilled_mass_kg": float(fulfilled_mass_kg),
                "remaining_demand_kg": float(max(getattr(t, "remaining_demand_kg", 0.0), 0.0)),
                "remaining_lifeline_at_service": float(getattr(t, "remaining_lifeline_at_service", 0.0)),
                "completed_by_agent": str(getattr(t, "delivered_by", "")),
                "failed": int(t.status == TaskStatus.FAILED),
                "failed_reason": str(failed_reason),
                "failed_step": int(failed_step),
                "nearest_truck_distance_start": float(task_assign_diag.get(tid, {}).get("nearest_truck_distance_start", float("nan"))),
                "nearest_uav_or_truck_distance_start": float(task_assign_diag.get(tid, {}).get("nearest_uav_or_truck_distance_start", float("nan"))),
                "truck_reachable": int(task_assign_diag.get(tid, {}).get("truck_reachable", 0)),
                "uav_eventual_serviceable": int(task_assign_diag.get(tid, {}).get("uav_eventual_serviceable", 0)),
                "final_status": final_status,
            }
        )

        if t.kind != TaskKind.NORMAL:
            continue
        drec = routine_task_diag.get(tid, None)
        if drec is None:
            continue
        tnode = int(t.demand_node)
        nearest_final = float("inf")
        for taid in truck_ids:
            tr = env.state.agents.get(taid, None)
            if tr is None or bool(getattr(tr, "crashed", False)):
                continue
            d = _sp_dist(None if tr.node is None else int(tr.node), tnode)
            if d < nearest_final:
                nearest_final = d
        drec["distance_to_task_final"] = nearest_final
        if completed:
            drec["service_complete_count"] = 1
        primary_truck = ""
        primary_steps = -1
        for taid, cnt in dict(drec["assigned_truck_steps"]).items():
            if int(cnt) > int(primary_steps):
                primary_steps = int(cnt)
                primary_truck = str(taid)
        fail_reason = "completed"
        if not completed:
            if bool(getattr(t, "failed_due_to_lifeline_zero", False)):
                fail_reason = "lifeline_zero"
            elif t.status == TaskStatus.FAILED:
                fail_reason = "failed_other"
            elif int(drec["first_assigned_step"]) < 0:
                fail_reason = "never_assigned"
            elif int(drec["service_start_count"]) <= 0:
                fail_reason = "assigned_never_serviced"
            elif int(drec["assigned_but_not_progressing_count"]) > 0:
                fail_reason = "assigned_no_progress"
            else:
                fail_reason = "pending_unfinished"
        routine_trace_rows.append(
            {
                "task_id": tid,
                "nearest_truck": str(drec["nearest_truck"]),
                "assigned_truck_count": int(len(drec["assigned_trucks"])),
                "first_assigned_step": int(drec["first_assigned_step"]),
                "last_assigned_step": int(drec["last_assigned_step"]),
                "truck_id": str(primary_truck),
                "truck_goal_switch_count": int(truck_routine_diag.get(str(primary_truck), {}).get("routine_goal_switch_count", 0))
                if primary_truck
                else 0,
                "truck_dead_end_routine_localized_count": int(
                    getattr(planner, "_debug_truck_dead_end_localize_by_truck", {}).get(str(primary_truck), 0)
                )
                if primary_truck
                else 0,
                "path_blocked_routine_localized_count": int(
                    getattr(planner, "_debug_truck_path_blocked_localize_by_truck", {}).get(str(primary_truck), 0)
                )
                if primary_truck
                else 0,
                "assigned_but_not_progressing_count": int(drec["assigned_but_not_progressing_count"]),
                "distance_to_task_start": float(drec["distance_to_task_start"]) if math.isfinite(float(drec["distance_to_task_start"])) else float("nan"),
                "distance_to_task_min": float(drec["distance_to_task_min"]) if math.isfinite(float(drec["distance_to_task_min"])) else float("nan"),
                "distance_to_task_final": float(drec["distance_to_task_final"]) if math.isfinite(float(drec["distance_to_task_final"])) else float("nan"),
                "service_start_count": int(drec["service_start_count"]),
                "service_complete_count": int(drec["service_complete_count"]),
                "final_failure_reason": str(fail_reason),
            }
        )

    for taid, trec in ([] if bool(lightweight_metrics_only) else truck_routine_diag.items()):
        completed_set = set()
        service_start_count = 0
        for t in env.state.tasks.values():
            if t.kind != TaskKind.NORMAL:
                continue
            if str(getattr(t, "delivered_by", "")) == str(taid) and t.status == TaskStatus.DELIVERED:
                completed_set.add(str(t.task_id))
            if t.first_service_step is not None and str(getattr(t, "in_service_by", "")) == str(taid):
                service_start_count += 1
        truck_routine_summary_rows.append(
            {
                "truck_id": str(taid),
                "routine_assigned_count": int(len(trec["routine_assigned_tasks"])),
                "routine_completed_count": int(len(completed_set)),
                "routine_service_start_count": int(max(service_start_count, int(trec["routine_service_start_count"]))),
                "routine_goal_switch_count": int(trec["routine_goal_switch_count"]),
                "routine_reassign_count": int(trec["routine_reassign_count"]),
                "dead_end_localize_count": int(getattr(planner, "_debug_truck_dead_end_localize_by_truck", {}).get(str(taid), 0)),
                "path_blocked_localize_count": int(getattr(planner, "_debug_truck_path_blocked_localize_by_truck", {}).get(str(taid), 0)),
                "stuck_steps": int(trec["stuck_steps"]),
                "average_distance_progress_per_step": float(trec["progress_dist_sum"] / max(int(trec["progress_step_count"]), 1)),
                "time_spent_servicing": int(trec["time_spent_servicing"]),
                "time_spent_moving_to_routine": int(trec["time_spent_moving_to_routine"]),
                "time_spent_idle_or_no_progress": int(trec["time_spent_idle_or_no_progress"]),
            }
        )

    uav_execution_rows: List[Dict[str, Any]] = []
    for uid, urec in ([] if bool(lightweight_metrics_only) else uav_exec_diag.items()):
        launch_cnt = int(urec.get("launch_count", 0))
        delivery_cnt = int(urec.get("delivery_count", 0))
        l2c = float(delivery_cnt / max(launch_cnt, 1))
        lb_min = float(urec.get("launch_battery_min", float("nan")))
        if not math.isfinite(lb_min):
            lb_min = float("nan")
        lb_mean = float(urec.get("launch_battery_sum", 0.0) / max(launch_cnt, 1))
        uav_execution_rows.append(
            {
                "uav_id": str(uid),
                "assigned_task_count": int(urec.get("assigned_task_count", 0)),
                "launch_count": int(launch_cnt),
                "delivery_count": int(delivery_cnt),
                "launch_to_completion_ratio": float(l2c),
                "reject_count": int(urec.get("reject_count", 0)),
                "reject_reason_insufficient_recovery_margin": int(urec.get("reject_reason_insufficient_recovery_margin", 0)),
                "reject_reason_corridor": int(urec.get("reject_reason_corridor", 0)),
                "reject_reason_comm_block": int(urec.get("reject_reason_comm_block", 0)),
                "reject_reason_energy_infeasible": int(urec.get("reject_reason_energy_infeasible", 0)),
                "reject_reason_no_recovery": int(urec.get("reject_reason_no_recovery", 0)),
                "airborne_goal_switch_blocked_count": int(urec.get("airborne_goal_switch_blocked_count", 0)),
                "forced_recovery_count": int(urec.get("forced_recovery_count", 0)),
                "rendezvous_success_count": int(urec.get("rendezvous_success_count", 0)),
                "average_launch_battery": float(lb_mean),
                "min_launch_battery": float(lb_min),
            }
        )

    switch_decision_rows: List[Dict[str, Any]] = []
    export_switch_rows = getattr(planner, "_export_switch_decision_ledger_rows", None)
    if (not bool(lightweight_metrics_only)) and enable_switch_decision_ledger and callable(export_switch_rows):
        try:
            got = export_switch_rows(env)
            if isinstance(got, list):
                switch_decision_rows = [r for r in got if isinstance(r, dict)]
        except Exception:
            switch_decision_rows = []

    return {
        "completion_rate": float(_completion(last_info, episode_idx=0)),
        "overall_completion_rate": float(_require_info(last_info, "overall_completion_rate", episode_idx=0)),
        "routine_bulk_completion_rate": float(_require_info(last_info, "routine_bulk_completion_rate", episode_idx=0)),
        "time_critical_lightweight_completion_rate": float(_require_info(last_info, "time_critical_lightweight_completion_rate", episode_idx=0)),
        "time_critical_on_time_completion_rate": float(_require_info(last_info, "time_critical_on_time_completion_rate", episode_idx=0)),
        "time_critical_on_time_completed_count_total": float(_require_info(last_info, "time_critical_on_time_completed_count_total", episode_idx=0)),
        "time_critical_completion_time_mean_seconds": float(_require_info(last_info, "time_critical_completion_time_mean_seconds", episode_idx=0)),
        "routine_bulk_completion_time_mean_seconds": float(_require_info(last_info, "routine_bulk_completion_time_mean_seconds", episode_idx=0)),
        "overall_completion_time_mean_seconds": float(_require_info(last_info, "overall_completion_time_mean_seconds", episode_idx=0)),
        "mean_remaining_lifeline_at_completion_time_critical": float(_require_info(last_info, "mean_remaining_lifeline_at_completion_time_critical", episode_idx=0)),
        "bulk_fulfilled_mass_ratio": float(_require_info(last_info, "bulk_fulfilled_mass_ratio", episode_idx=0)),
        "failed_task_count": float(_require_info(last_info, "failed_task_count", episode_idx=0)),
        "failed_due_to_lifeline_zero_count": float(_require_info(last_info, "failed_due_to_lifeline_zero_count", episode_idx=0)),
        "mean_remaining_lifeline_at_service": float(_require_info(last_info, "mean_remaining_lifeline_at_service", episode_idx=0)),
        "mean_remaining_lifeline_time_critical": float(_require_info(last_info, "mean_remaining_lifeline_time_critical", episode_idx=0)),
        "mean_remaining_lifeline_bulk": float(_require_info(last_info, "mean_remaining_lifeline_bulk", episode_idx=0)),
        "average_service_delay": float(_require_info(last_info, "average_service_delay", episode_idx=0)),
        "average_service_delay_time_critical": float(_require_info(last_info, "average_service_delay_time_critical", episode_idx=0)),
        "average_service_delay_bulk": float(_require_info(last_info, "average_service_delay_bulk", episode_idx=0)),
        "weighted_service_score": float(_require_info(last_info, "weighted_service_score", episode_idx=0)),
        "routine_bulk_completed_count_total": float(_require_info(last_info, "routine_bulk_completed_count_total", episode_idx=0)),
        "time_critical_lightweight_completed_count_total": float(_require_info(last_info, "time_critical_lightweight_completed_count_total", episode_idx=0)),
        "routine_bulk_failed_count_total": float(_require_info(last_info, "routine_bulk_failed_count_total", episode_idx=0)),
        "time_critical_lightweight_failed_count_total": float(_require_info(last_info, "time_critical_lightweight_failed_count_total", episode_idx=0)),
        "route_plan_v2_suffix_repair_count": float(
            last_info.get("route_plan_v2_suffix_repair_count", 0.0)
        ),
        "route_plan_v2_suffix_repair_success_count": float(
            last_info.get("route_plan_v2_suffix_repair_success_count", 0.0)
        ),
        "route_plan_v2_stalled_contract_transfer_candidate_count": float(
            last_info.get(
                "route_plan_v2_stalled_contract_transfer_candidate_count", 0.0
            )
        ),
        "route_plan_v2_stalled_contract_transfer_replan_count": float(
            last_info.get(
                "route_plan_v2_stalled_contract_transfer_replan_count", 0.0
            )
        ),
        "route_plan_v2_contract_transfer_count": float(
            last_info.get("route_plan_v2_contract_transfer_count", 0.0)
        ),
        "route_plan_v2_onsite_takeover_count": float(
            last_info.get("route_plan_v2_onsite_takeover_count", 0.0)
        ),
        "route_plan_v2_routine_opportunity_candidate_count": float(
            last_info.get("route_plan_v2_routine_opportunity_candidate_count", 0.0)
        ),
        "route_plan_v2_routine_opportunity_transfer_count": float(
            last_info.get("route_plan_v2_routine_opportunity_transfer_count", 0.0)
        ),
        "route_plan_v2_routine_opportunity_blocked_assist_count": float(
            last_info.get(
                "route_plan_v2_routine_opportunity_blocked_assist_count", 0.0
            )
        ),
        "route_plan_v2_routine_opportunity_blocked_eta_count": float(
            last_info.get("route_plan_v2_routine_opportunity_blocked_eta_count", 0.0)
        ),
        "route_plan_v2_onsite_capture_count": float(
            last_info.get("route_plan_v2_onsite_capture_count", 0.0)
        ),
        "route_plan_v2_onsite_capture_contract_transfer_count": float(
            last_info.get("route_plan_v2_onsite_capture_contract_transfer_count", 0.0)
        ),
        "route_plan_v2_onsite_capture_preempted_assist_count": float(
            last_info.get("route_plan_v2_onsite_capture_preempted_assist_count", 0.0)
        ),
        "route_plan_v2_deadline_rescue_promotion_count": float(
            last_info.get("route_plan_v2_deadline_rescue_promotion_count", 0.0)
        ),
        "route_plan_v2_emergency_starvation_promotion_count": float(
            last_info.get(
                "route_plan_v2_emergency_starvation_promotion_count", 0.0
            )
        ),
        "route_plan_v2_emergency_launch_watchdog_ready_count": float(
            last_info.get(
                "route_plan_v2_emergency_launch_watchdog_ready_count", 0.0
            )
        ),
        "route_plan_v2_emergency_launch_watchdog_force_count": float(
            last_info.get(
                "route_plan_v2_emergency_launch_watchdog_force_count", 0.0
            )
        ),
        "route_plan_v2_queue_rescue_assignment_count": float(
            last_info.get("route_plan_v2_queue_rescue_assignment_count", 0.0)
        ),
        "route_plan_v2_queue_rescue_delivery_count": float(
            last_info.get("route_plan_v2_queue_rescue_delivery_count", 0.0)
        ),
        "route_plan_v2_direct_safe_secondary_emergency_candidate_count": float(
            last_info.get(
                "route_plan_v2_direct_safe_secondary_emergency_candidate_count",
                0.0,
            )
        ),
        "route_plan_v2_direct_safe_secondary_emergency_assignment_count": float(
            last_info.get(
                "route_plan_v2_direct_safe_secondary_emergency_assignment_count",
                0.0,
            )
        ),
        "uav_authoritative_sortie_goal_override_count": float(
            last_info.get("uav_authoritative_sortie_goal_override_count", 0.0)
        ),
        "uav_terminal_delivery_commitment_count": float(
            last_info.get("uav_terminal_delivery_commitment_count", 0.0)
        ),
        "route_plan_v2_lifecycle_turnaround_cost_evaluation_count": float(
            last_info.get(
                "route_plan_v2_lifecycle_turnaround_cost_evaluation_count", 0.0
            )
        ),
        "route_plan_v2_lifecycle_turnaround_cost_total": float(
            last_info.get("route_plan_v2_lifecycle_turnaround_cost_total", 0.0)
        ),
        "route_plan_v2_lexicographic_comparison_count": float(
            last_info.get("route_plan_v2_lexicographic_comparison_count", 0.0)
        ),
        "route_plan_v2_lexicographic_primary_rejection_count": float(
            last_info.get(
                "route_plan_v2_lexicographic_primary_rejection_count", 0.0
            )
        ),
        "route_plan_v2_disconnect_profile_evaluation_count": float(
            last_info.get(
                "route_plan_v2_disconnect_profile_evaluation_count", 0.0
            )
        ),
        "route_plan_v2_disconnect_protected_task_count": float(
            last_info.get("route_plan_v2_disconnect_protected_task_count", 0.0)
        ),
        "route_plan_v2_disconnect_predicted_miss_count": float(
            last_info.get("route_plan_v2_disconnect_predicted_miss_count", 0.0)
        ),
        "route_plan_v2_emergency_balance_trigger_count": float(
            last_info.get("route_plan_v2_emergency_balance_trigger_count", 0.0)
        ),
        "route_plan_v2_emergency_balance_baseline_max_count": float(
            last_info.get(
                "route_plan_v2_emergency_balance_baseline_max_count", 0.0
            )
        ),
        "route_plan_v2_emergency_capacity_repair_count": float(
            last_info.get("route_plan_v2_emergency_capacity_repair_count", 0.0)
        ),
        "route_plan_v2_emergency_capacity_contract_move_count": float(
            last_info.get(
                "route_plan_v2_emergency_capacity_contract_move_count", 0.0
            )
        ),
        "route_plan_v2_residual_emergency_handoff_count": float(
            last_info.get("route_plan_v2_residual_emergency_handoff_count", 0.0)
        ),
        "route_plan_v2_routine_inventory_rebalance_count": float(
            last_info.get("route_plan_v2_routine_inventory_rebalance_count", 0.0)
        ),
        "route_plan_v2_normal_cleanup_replan_count": float(
            last_info.get("route_plan_v2_normal_cleanup_replan_count", 0.0)
        ),
        "route_plan_v2_b_orphaned_routine_rescue_count": float(
            last_info.get(
                "route_plan_v2_b_orphaned_routine_rescue_count", 0.0
            )
        ),
        "route_plan_v2_r4_routine_takeover_candidate_count": float(
            last_info.get("route_plan_v2_r4_routine_takeover_candidate_count", 0.0)
        ),
        "route_plan_v2_r4_routine_takeover_trigger_count": float(
            last_info.get("route_plan_v2_r4_routine_takeover_trigger_count", 0.0)
        ),
        "route_plan_v2_r4_routine_takeover_success_count": float(
            last_info.get("route_plan_v2_r4_routine_takeover_success_count", 0.0)
        ),
        "route_plan_v2_queue_starvation_repair_count": float(
            last_info.get("route_plan_v2_queue_starvation_repair_count", 0.0)
        ),
        "route_plan_v2_initial_lifeline_ordering_enabled": float(
            last_info.get(
                "route_plan_v2_initial_lifeline_ordering_enabled", 0.0
            )
        ),
        "route_plan_v2_contract_consistency_block_count": float(
            last_info.get("route_plan_v2_contract_consistency_block_count", 0.0)
        ),
        "routine_multiround_commitment_count": float(
            last_info.get("routine_multiround_commitment_count", 0.0)
        ),
        "routine_multiround_support_block_count": float(
            last_info.get("routine_multiround_support_block_count", 0.0)
        ),
        "makespan_seconds": makespan_seconds,
        "uav_energy_used_total": float(_require_info(last_info, "uav_energy_used_total", episode_idx=0)),
        "runtime_crash_count": 0.0,
        "uav_drop_count": float(last_info.get("uav_drop_count", last_info.get("UAV_DROP", 0.0))),
        "crash_count": 0.0,
        "legacy_physical_loss_count": float(_require_info(last_info, "crash_count_total", episode_idx=0)),
        "battery_depletion_count": float(_require_info(last_info, "battery_depletion_count_total", episode_idx=0)),
        "invalid_action_count_total": float(_require_info(last_info, "invalid_action_count_total", episode_idx=0)),
        "planner_candidate_invalid_count_total": float(_require_info(last_info, "planner_candidate_invalid_count_total", episode_idx=0)),
        "pre_dispatch_rejected_count_total": float(_require_info(last_info, "pre_dispatch_rejected_count_total", episode_idx=0)),
        "pre_dispatch_repair_success_count_total": float(_require_info(last_info, "pre_dispatch_repair_success_count_total", episode_idx=0)),
        "safe_noop_fallback_count_total": float(_require_info(last_info, "safe_noop_fallback_count_total", episode_idx=0)),
        "environment_invalid_action_count_total": float(_require_info(last_info, "environment_invalid_action_count_total", episode_idx=0)),
        "invalid_action_count_uav_total": float(_require_info(last_info, "invalid_action_count_uav_total", episode_idx=0)),
        "invalid_action_count_truck_total": float(_require_info(last_info, "invalid_action_count_truck_total", episode_idx=0)),
        **{
            k: float(last_info.get(k, getattr(planner, k, 0.0)))
            for k in [
                "unauthorized_support_attempt_count",
                "unauthorized_support_blocked_count",
    "unauthorized_recovery_attempt_count",
    "unauthorized_recovery_blocked_count",
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
    "support_anchor_distance_at_arrival_mean",
    "support_anchor_arrived_launch_attempt_count",
    "support_anchor_arrived_launch_success_count",
    "support_anchor_arrived_launch_rejected_count",
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
    "support_delivery_count",
    "support_lock_created_count",
    "support_lock_released_count",
    "support_lock_blocked_uav_reassign_count",
    "support_lock_blocked_task_steal_count",
    "support_lock_broken_by_hard_safety_count",
    "support_lock_expired_count",
                "command_rejected_count",
                "command_rejected_reason_launch_unauthorized_count",
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
            ]
        },
        "forced_rth_count": float(_require_info(last_info, "forced_rth_count_total", episode_idx=0)),
        "queue_time_seconds": float(_require_info(last_info, "queue_time_seconds_total", episode_idx=0)),
        "inference_latency_mean_ms": float(np.mean(inf_ms)) if inf_ms else 0.0,
        "inference_latency_p95_ms": float(np.percentile(inf_ms, 95)) if inf_ms else 0.0,
        "planner_inference_latency_mean_ms": float(np.mean(planner_ms)) if planner_ms else 0.0,
        "planner_inference_latency_p95_ms": float(np.percentile(planner_ms, 95)) if planner_ms else 0.0,
        "replan_latency_count": int(len(replan_ms)),
        "replan_latency_median_ms": float(np.percentile(replan_ms, 50)) if replan_ms else 0.0,
        "replan_latency_p95_ms": float(np.percentile(replan_ms, 95)) if replan_ms else 0.0,
        "replan_latency_p99_ms": float(np.percentile(replan_ms, 99)) if replan_ms else 0.0,
        "replan_latency_max_ms": float(np.max(replan_ms)) if replan_ms else 0.0,
        "low_level_inference_latency_mean_ms": float(np.mean(low_ms)) if low_ms else 0.0,
        "low_level_inference_latency_p95_ms": float(np.percentile(low_ms, 95)) if low_ms else 0.0,
        "env_step_latency_mean_ms": float(np.mean(env_ms)) if env_ms else 0.0,
        "env_step_latency_p95_ms": float(np.percentile(env_ms, 95)) if env_ms else 0.0,
        "end_to_end_step_latency_mean_ms": float(np.mean(end2end_ms)) if end2end_ms else 0.0,
        "end_to_end_step_latency_p95_ms": float(np.percentile(end2end_ms, 95)) if end2end_ms else 0.0,
        "truck_distance_total_m": truck_distance_total_m,
        "uav_distance_total_m": uav_distance_total_m,
        "fleet_distance_total_m": fleet_distance_total_m,
        "delivered_task_last_step": delivered_task_last_step,
        "delivered_task_last_time_seconds": delivered_task_last_time_seconds,
        "terminal_task_last_step": terminal_task_last_step,
        "terminal_task_last_time_seconds": terminal_task_last_time_seconds,
        "task_end_step": task_end_step,
        "task_end_time_seconds": task_end_time_seconds,
        "truck_normal_supply_units_total": float(_require_info(last_info, "truck_normal_supply_units_total", episode_idx=0)),
        "truck_emergency_supply_units_total": float(_require_info(last_info, "truck_emergency_supply_units_total", episode_idx=0)),
        "truck_replenish_count_total": truck_replenish_count_total,
        "truck_empty_trip_count_total": float(_require_info(last_info, "truck_empty_trip_count_total", episode_idx=0)),
        "truck_inventory_kg_current_mean": truck_inventory_kg_current_mean,
        "uav_reload_count_total": float(_require_info(last_info, "uav_reload_count_total", episode_idx=0)),
        "uav_reload_wait_steps_total": uav_reload_wait_steps_total,
        "uav_empty_flight_count_total": float(_require_info(last_info, "uav_empty_flight_count_total", episode_idx=0)),
        "uav_delivery_count_total": float(uav_delivery_count_total),
        "uav_loaded_fraction": uav_loaded_fraction,
        "normal_tasks_blocked_by_supply_count": normal_block,
        "emergency_tasks_blocked_by_supply_count": emergency_block,
        "truck_replenish_event_flag": float(1.0 if truck_replenish_count_total > 0.0 else 0.0),
        "uav_reload_event_flag": float(1.0 if float(_require_info(last_info, "uav_reload_count_total", episode_idx=0)) > 0.0 else 0.0),
        "normal_supply_serviceability_ratio": float(np.clip(delivered_normal / normal_serviceability_denom, 0.0, 1.0)),
        "emergency_supply_serviceability_ratio": float(np.clip(delivered_emergency / emergency_serviceability_denom, 0.0, 1.0)),
        "normal_completed_count_total": delivered_normal,
        "emergency_completed_count_total": delivered_emergency,
        "normal_failed_count_total": failed_normal,
        "emergency_failed_count_total": failed_emergency,
        "truck_replenish_burden": float(truck_replenish_count_total / max(makespan_seconds, 1e-6)),
        "uav_reload_burden": float((uav_reload_wait_steps_total * dt) / max(makespan_seconds, 1e-6)),
        **{str(k): float(_require_info(last_info, str(k), episode_idx=0)) for k in UAV_SAFETY_METRIC_FIELDS},
        **{str(k): float(_require_info(last_info, str(k), episode_idx=0)) for k in UAV_STRICT_SAFETY_FIELDS},
        **{str(k): float(_require_info(last_info, str(k), episode_idx=0)) for k in ISLAND_SUPPORT_METRIC_FIELDS},
        **{
            str(k): float(_require_info(last_info, str(k), episode_idx=0))
            for k in [
                "routine_near_completion_protected_count",
                "routine_near_completion_support_blocked_count",
                "routine_near_completion_recovery_blocked_count",
                "routine_near_completion_broken_by_hard_safety_count",
                "routine_near_completion_broken_by_tc_override_count",
                "routine_near_completion_tc_override_to_launch_count",
                "routine_near_completion_tc_override_to_delivery_count",
                "routine_near_completion_blocked_tc_support_count",
                "routine_near_completion_followed_by_service_start_count",
                "routine_near_completion_followed_by_completion_count",
                "routine_near_completion_tc_override_reject_delay_count",
                "routine_near_completion_tc_override_reject_no_loaded_uav_count",
                "routine_near_completion_tc_override_reject_no_candidate_count",
                "routine_near_completion_tc_override_reject_not_near_launchable_count",
                "routine_near_completion_tc_override_reject_recovery_count",
                "routine_near_completion_broken_by_delivery_feasible_tc_override_count",
                "tc_override_candidate_count",
                "tc_override_blocked_not_full_sortie_feasible_count",
                "tc_override_blocked_low_recovery_margin_count",
                "tc_override_blocked_low_battery_margin_count",
                "tc_override_blocked_recent_reject_count",
                "tc_override_blocked_lifeline_risk_count",
                "tc_override_blocked_routine_delay_count",
                "tc_override_to_launch_count",
                "tc_override_to_delivery_count",
                "tc_override_to_forced_recovery_count",
                "tc_override_feasibility_mismatch_count",
                "tc_override_predicted_launchable_count",
                "tc_override_actual_launch_count",
                "tc_override_predicted_delivery_feasible_count",
                "tc_override_actual_delivery_count",
            ]
        },
        **{str(k): str(last_info.get(str(k), "")) for k in PHYSICAL_V2_STRING_FIELDS},
        **{str(k): float(last_info.get(str(k), 0.0)) for k in PHYSICAL_V2_METRIC_FIELDS},
        "episode_reward_mean": float(reward_sum / max(steps, 1)),
        "blocked_edge_count": float(last_info.get("blocked_edge_count", 0.0)),
        "comm_blackout_ratio": float(last_info.get("comm_blackout_ratio", 0.0)),
        "comm_blackout_agent_observation_count_total": float(last_info.get("comm_blackout_agent_observation_count_total", 0.0)),
        "comm_blackout_agent_blocked_count_total": float(last_info.get("comm_blackout_agent_blocked_count_total", 0.0)),
        "comm_blackout_agent_time_exposure_ratio": float(last_info.get("comm_blackout_agent_time_exposure_ratio", 0.0)),
        "comm_blackout_physical_zone_count_total": float(last_info.get("comm_blackout_physical_zone_count_total", 0.0)),
        "comm_blackout_goal_zone_count_total": float(last_info.get("comm_blackout_goal_zone_count_total", 0.0)),
        "comm_blackout_zone_count": float(last_info.get("comm_blackout_zone_count", 0.0)),
        "comm_blackout_nominal_emergency_coverage": float(last_info.get("comm_blackout_nominal_emergency_coverage", 0.0)),
        "comm_blackout_zone_radius_map_fraction": float(last_info.get("comm_blackout_zone_radius_map_fraction", 0.0)),
        "comm_blackout_zone_radius_mean_m": float(last_info.get("comm_blackout_zone_radius_mean_m", 0.0)),
        "comm_blackout_start_step": float(last_info.get("comm_blackout_start_step", 0.0)),
        "comm_blackout_duration_steps": float(last_info.get("comm_blackout_duration_steps", 0.0)),
        "comm_blackout_recovery_steps": float(last_info.get("comm_blackout_recovery_steps", 0.0)),
        "comm_blackout_cycle_steps": float(last_info.get("comm_blackout_cycle_steps", 0.0)),
        "comm_blackout_duty_cycle": float(last_info.get("comm_blackout_duty_cycle", 0.0)),
        "comm_blackout_covered_task_count": float(last_info.get("comm_blackout_covered_task_count", 0.0)),
        "comm_blackout_covered_task_ratio": float(last_info.get("comm_blackout_covered_task_ratio", 0.0)),
        "comm_blackout_covered_node_count": float(last_info.get("comm_blackout_covered_node_count", 0.0)),
        "comm_blackout_covered_node_ratio": float(last_info.get("comm_blackout_covered_node_ratio", 0.0)),
        "comm_blackout_zone_digest": str(last_info.get("comm_blackout_zone_digest", "")),
        "wind_severity_p95_mps": float(last_info.get("wind_severity_p95_mps", 0.0)),
        "rain_severity_p95_mmh": float(last_info.get("rain_severity_p95_mmh", 0.0)),
        "triggered_replans": float(last_info.get("triggered_replans_total", 0.0)),
        "event_replans_in_window": float(event_replans_in_window_peak),
        "event_budget_blocked": float(event_budget_blocked_count),
        "blocked_edge_count_stochastic": float(last_info.get("blocked_edge_count_stochastic", 0.0)),
        "blocked_edge_count_forced_island": float(last_info.get("blocked_edge_count_forced_island", 0.0)),
        "blocked_edge_count_total": float(last_info.get("blocked_edge_count_total", last_info.get("blocked_edge_count", 0.0))),
        "blocked_ratio_stochastic_final": float(last_info.get("blocked_ratio_stochastic_final", 0.0)),
        "blocked_ratio_forced_island_final": float(last_info.get("blocked_ratio_forced_island_final", 0.0)),
        "blocked_ratio_total_final": float(last_info.get("blocked_ratio_total_final", last_info.get("blocked_ratio", 0.0))),
        "blockage_target_ratio_stochastic_final": float(last_info.get("blockage_target_ratio_stochastic_final", 0.0)),
        "newly_blocked_edge_count_total": float(last_info.get("newly_blocked_edge_count_total", 0.0)),
        "blockage_target_ratio_final": float(last_info.get("blockage_target_ratio_final", 0.0)),
        "blockage_curve_B_inf": float(last_info.get("blockage_curve_B_inf", 0.0)),
        "blockage_curve_tau_steps": float(last_info.get("blockage_curve_tau_steps", 0.0)),
        "shared_blocked_edge_count": float(_require_info(last_info, "shared_blocked_edge_count", episode_idx=0)),
        "shared_map_update_count_total": float(_require_info(last_info, "shared_map_update_count_total", episode_idx=0)),
        "shared_map_new_blocked_total": float(_require_info(last_info, "shared_map_new_blocked_total", episode_idx=0)),
        "shared_map_cleared_total": float(_require_info(last_info, "shared_map_cleared_total", episode_idx=0)),
        "shared_discovery_uav_total": float(_require_info(last_info, "shared_discovery_uav_total", episode_idx=0)),
        "shared_discovery_truck_total": float(_require_info(last_info, "shared_discovery_truck_total", episode_idx=0)),
        "unknown_blocked_edge_hit_total": float(_require_info(last_info, "unknown_blocked_edge_hit_total", episode_idx=0)),
        "road_observation_event_count_total": float(_require_info(last_info, "road_observation_event_count_total", episode_idx=0)),
        "planner_replan_due_to_new_road_info_count_total": float(
            _require_info(last_info, "planner_replan_due_to_new_road_info_count_total", episode_idx=0)
        ),
        "goal_switch_count_total": float(getattr(planner, "goal_switch_count_total", 0.0)),
        "goal_switch_candidate_count": float(getattr(planner, "goal_switch_candidate_count_total", 0.0)),
        "goal_switch_accepted_count": float(getattr(planner, "goal_switch_accepted_count_total", 0.0)),
        "goal_switch_rejected_by_threshold_count": float(getattr(planner, "goal_switch_rejected_by_threshold_count_total", 0.0)),
        "goal_switch_forced_count": float(getattr(planner, "goal_switch_forced_count_total", 0.0)),
        "goal_switch_accepted_by_score_count": float(getattr(planner, "goal_switch_accepted_by_score_count_total", 0.0)),
        "goal_switch_accepted_by_eta_count": float(getattr(planner, "goal_switch_accepted_by_eta_count_total", 0.0)),
        "goal_switch_forced_reason_completed": float(getattr(planner, "goal_switch_forced_reason_completed_total", 0.0)),
        "goal_switch_forced_reason_failed": float(getattr(planner, "goal_switch_forced_reason_failed_total", 0.0)),
        "goal_switch_forced_reason_infeasible": float(getattr(planner, "goal_switch_forced_reason_infeasible_total", 0.0)),
        "goal_switch_forced_reason_dead_end": float(getattr(planner, "goal_switch_forced_reason_dead_end_total", 0.0)),
        "goal_switch_forced_reason_uav_recovery": float(getattr(planner, "goal_switch_forced_reason_uav_recovery_total", 0.0)),
        "goal_switch_forced_reason_stall": float(getattr(planner, "goal_switch_forced_reason_stall_total", 0.0)),
        "ablation_low_value_refresh_enabled": float(bool(getattr(env.cfg, "erc_ablate_low_value_refresh", False))),
        "ablation_map_ranking_refresh_enabled": float(bool(getattr(env.cfg, "erc_ablate_map_ranking_refresh", False))),
        "ablation_tc_global_assignment_enabled": float(bool(getattr(env.cfg, "erc_ablate_tc_global_assignment", False))),
        "ablation_support_chain_enabled": float(bool(getattr(env.cfg, "erc_ablate_support_chain", False))),
        "ablation_cluster_primary_reservation_enabled": float(bool(getattr(env.cfg, "erc_ablate_cluster_primary_reservation", False))),
        "ablation_event_scoring_bonus_enabled": float(bool(getattr(env.cfg, "erc_ablate_event_scoring_bonus", False))),
        "ablation_normal_protection_enabled": float(bool(getattr(env.cfg, "erc_ablate_normal_protection", False))),
        "low_value_refresh_candidate_count": float(getattr(planner, "low_value_refresh_candidate_count_total", 0.0)),
        "low_value_refresh_allowed_count": float(getattr(planner, "low_value_refresh_allowed_count_total", 0.0)),
        "low_value_refresh_blocked_by_ablation_count": float(getattr(planner, "low_value_refresh_blocked_by_ablation_count_total", 0.0)),
        "map_ranking_refresh_candidate_count": float(getattr(planner, "map_ranking_refresh_candidate_count_total", 0.0)),
        "map_ranking_refresh_allowed_count": float(getattr(planner, "map_ranking_refresh_allowed_count_total", 0.0)),
        "map_ranking_refresh_blocked_by_ablation_count": float(getattr(planner, "map_ranking_refresh_blocked_by_ablation_count_total", 0.0)),
        "tc_global_assignment_called_count": float(getattr(planner, "tc_global_assignment_called_count_total", 0.0)),
        "tc_global_assignment_skipped_by_ablation_count": float(getattr(planner, "tc_global_assignment_skipped_by_ablation_count_total", 0.0)),
        "tc_assignment_epoch_applied_count": float(getattr(planner, "tc_assignment_epoch_applied_count_total", 0.0)),
        "support_chain_candidate_count": float(getattr(planner, "support_chain_candidate_count_total", 0.0)),
        "support_chain_applied_count": float(getattr(planner, "support_chain_applied_count_total", 0.0)),
        "support_chain_blocked_by_ablation_count": float(getattr(planner, "support_chain_blocked_by_ablation_count_total", 0.0)),
        "cluster_primary_candidate_count": float(getattr(planner, "cluster_primary_candidate_count_total", 0.0)),
        "cluster_primary_applied_count": float(getattr(planner, "cluster_primary_applied_count_total", 0.0)),
        "cluster_primary_blocked_by_ablation_count": float(getattr(planner, "cluster_primary_blocked_by_ablation_count_total", 0.0)),
        "task_reservation_applied_count": float(getattr(planner, "task_reservation_applied_count_total", 0.0)),
        "task_reservation_blocked_by_ablation_count": float(getattr(planner, "task_reservation_blocked_by_ablation_count_total", 0.0)),
        "event_scoring_bonus_applied_count": float(getattr(planner, "event_scoring_bonus_applied_count_total", 0.0)),
        "event_scoring_bonus_blocked_by_ablation_count": float(getattr(planner, "event_scoring_bonus_blocked_by_ablation_count_total", 0.0)),
        "normal_protection_candidate_count": float(getattr(planner, "normal_protection_candidate_count_total", 0.0)),
        "normal_protection_applied_count": float(getattr(planner, "normal_protection_applied_count_total", 0.0)),
        "normal_protection_blocked_by_ablation_count": float(getattr(planner, "normal_protection_blocked_by_ablation_count_total", 0.0)),
        "cluster_primary_reject_count": float(getattr(planner, "cluster_primary_reject_count_total", 0.0)),
        "cluster_primary_switch_count": float(getattr(planner, "cluster_primary_switch_count_total", 0.0)),
        "same_task_cooldown_reject_count": float(getattr(planner, "same_task_cooldown_reject_count_total", 0.0)),
        "uav_reject_cache_hit_count_total": float(getattr(planner, "uav_reject_cache_hit_count_total", 0.0)),
        "uav_reject_cache_insert_count_total": float(getattr(planner, "uav_reject_cache_insert_count_total", 0.0)),
        "uav_reject_cache_clear_count_total": float(getattr(planner, "uav_reject_cache_clear_count_total", 0.0)),
        "uav_reject_cache_reason_insufficient_recovery_margin": float(getattr(planner, "uav_reject_cache_reason_insufficient_recovery_margin_total", 0.0)),
        "uav_reject_cache_reason_corridor": float(getattr(planner, "uav_reject_cache_reason_corridor_total", 0.0)),
        "uav_reject_cache_reason_comm_block": float(getattr(planner, "uav_reject_cache_reason_comm_block_total", 0.0)),
        "uav_reject_cache_reason_energy_infeasible": float(getattr(planner, "uav_reject_cache_reason_energy_infeasible_total", 0.0)),
        "uav_reject_cache_reason_no_recovery": float(getattr(planner, "uav_reject_cache_reason_no_recovery_total", 0.0)),
        "uav_goal_assigned_count_total": float(uav_goal_assigned_count_total),
        "uav_direct_ready_eval_count_total": float(uav_direct_ready_eval_count_total),
        "uav_direct_ready_selected_count_total": float(uav_direct_ready_selected_count_total),
        "truck_completed_task_count_total": float(truck_completed_task_count_total),
        "uav_completed_task_count_total": float(uav_completed_task_count_total),
        "normal_unreachable_task_count_total": float(getattr(planner, "normal_unreachable_task_count_total", 0.0)),
        "goal_assignment_to_launch_ratio": float(goal_assignment_to_launch_ratio),
        "direct_ready_to_launch_ratio": float(direct_ready_to_launch_ratio),
        "selected_to_launch_ratio": float(direct_ready_to_launch_ratio),
        "launch_to_completion_ratio": float(launch_to_completion_ratio),
        "harmful_switch_proxy_count": float(getattr(planner, "harmful_switch_proxy_count_total", 0.0)),
        "missed_switch_proxy_count": float(getattr(planner, "missed_switch_proxy_count_total", 0.0)),
        "uav_emergency_commit_hold_count": float(getattr(planner, "uav_emergency_commit_hold_count_total", 0.0)),
        "uav_emergency_commit_break_hard_invalid_count": float(
            getattr(planner, "uav_emergency_commit_break_hard_invalid_count_total", 0.0)
        ),
        "uav_emergency_commit_prevented_switch_count": float(
            getattr(planner, "uav_emergency_commit_prevented_switch_count_total", 0.0)
        ),
        "uav_emergency_commit_followed_by_launch_count": float(
            getattr(planner, "uav_emergency_commit_followed_by_launch_count_total", 0.0)
        ),
        "uav_emergency_commit_followed_by_delivery_count": float(
            getattr(planner, "uav_emergency_commit_followed_by_delivery_count_total", 0.0)
        ),
        "uav_task_reserved_count": float(getattr(planner, "uav_task_reserved_count_total", 0.0)),
        "uav_task_reservation_release_count": float(
            getattr(planner, "uav_task_reservation_release_count_total", 0.0)
        ),
        "uav_task_airborne_committed_count": float(
            getattr(planner, "uav_task_airborne_committed_count_total", 0.0)
        ),
        "uav_task_reserved_to_launch_count": float(
            getattr(planner, "uav_task_reserved_to_launch_count_total", 0.0)
        ),
        "uav_task_reserved_to_completion_count": float(
            getattr(planner, "uav_task_reserved_to_completion_count_total", 0.0)
        ),
        "uav_task_reservation_stale_count": float(
            getattr(planner, "uav_task_reservation_stale_count_total", 0.0)
        ),
        "uav_airborne_goal_switch_blocked_count": float(
            getattr(planner, "uav_airborne_goal_switch_blocked_count_total", 0.0)
        ),
        "uav_airborne_safety_abort_count": float(
            getattr(planner, "uav_airborne_safety_abort_count_total", 0.0)
        ),
        "uav_airborne_task_completed_count": float(
            getattr(planner, "uav_airborne_task_completed_count_total", 0.0)
        ),
        "truck_uav_assist_candidate_count": float(
            getattr(planner, "truck_uav_assist_candidate_count_total", 0.0)
        ),
        "truck_uav_assist_accepted_count": float(
            getattr(planner, "truck_uav_assist_accepted_count_total", 0.0)
        ),
        "truck_uav_assist_rejected_extra_distance_count": float(
            getattr(planner, "truck_uav_assist_rejected_extra_distance_count_total", 0.0)
        ),
        "truck_uav_assist_rejected_normal_service_count": float(
            getattr(planner, "truck_uav_assist_rejected_normal_service_count_total", 0.0)
        ),
        "truck_uav_assist_launch_success_count": float(
            getattr(planner, "truck_uav_assist_launch_success_count_total", 0.0)
        ),
        "truck_uav_assist_followed_by_emergency_completion_count": float(
            getattr(planner, "truck_uav_assist_followed_by_emergency_completion_count_total", 0.0)
        ),
        "truck_uav_assist_extra_distance_m": float(
            getattr(planner, "truck_uav_assist_extra_distance_m_total", 0.0)
        ),
        "truck_uav_assist_waypoint_move_count_total": float(
            getattr(env, "truck_uav_assist_waypoint_move_count_total", 0.0)
        ),
        "uav_task_reservation_exec_enabled": float(
            bool(getattr(env.cfg, "hrl_uav_task_reservation_exec_enabled", False))
        ),
        "uav_assist_enabled": float(bool(getattr(env.cfg, "hrl_uav_assist_enabled", False))),
        "truck_routine_stuck_candidate_count": float(getattr(planner, "truck_routine_stuck_candidate_count_total", 0.0)),
        "truck_routine_stuck_escape_count": float(getattr(planner, "truck_routine_stuck_escape_count_total", 0.0)),
        "truck_routine_stuck_escape_blocked_no_alt_count": float(
            getattr(planner, "truck_routine_stuck_escape_blocked_no_alt_count_total", 0.0)
        ),
        "truck_routine_stuck_escape_blocked_insufficient_gain_count": float(
            getattr(planner, "truck_routine_stuck_escape_blocked_insufficient_gain_count_total", 0.0)
        ),
        "truck_routine_stuck_escape_followed_by_service_count": float(
            getattr(planner, "truck_routine_stuck_escape_followed_by_service_count_total", 0.0)
        ),
        "truck_routine_stuck_escape_followed_by_completion_count": float(
            getattr(planner, "truck_routine_stuck_escape_followed_by_completion_count_total", 0.0)
        ),
        "routine_localize_eta_check_count": float(getattr(planner, "routine_localize_eta_check_count_total", 0.0)),
        "routine_localize_keep_current_count": float(getattr(planner, "routine_localize_keep_current_count_total", 0.0)),
        "routine_localize_escape_by_eta_worse_count": float(
            getattr(planner, "routine_localize_escape_by_eta_worse_count_total", 0.0)
        ),
        "routine_localize_escape_followed_by_service_count": float(
            getattr(planner, "routine_localize_escape_followed_by_service_count_total", 0.0)
        ),
        "routine_localize_escape_followed_by_completion_count": float(
            getattr(planner, "routine_localize_escape_followed_by_completion_count_total", 0.0)
        ),
        "erc_variant_base_no_event_enabled": float(str(getattr(env.cfg, "use_event_trigger", True)).lower() == "false"),
        "base_goal_match_ratio": float(0.0),
        "base_goal_mismatch_count": float(getattr(planner, "goal_switch_candidate_count_total", 0.0)),
        "candidate_set_mismatch_count": float(getattr(planner, "goal_switch_rejected_by_threshold_count_total", 0.0)),
        "assignment_order_mismatch_count": float(getattr(planner, "goal_switch_accepted_count_total", 0.0)),
        # effective variant flags (traceability)
        **{k: float(v) if isinstance(v, bool) else v for k, v in _variant_effective_flags(env.cfg).items()},
        # support authorization instrumentation
        "support_authorized_candidate_count": float(
            getattr(planner, "support_chain_candidate_count_total", 0.0)
            + _require_info(last_info, "truck_forward_support_count_total", episode_idx=0)
            + _require_info(last_info, "truck_recovery_support_count_total", episode_idx=0)
        ),
        "support_authorization_branch_called_count": float(
            getattr(planner, "support_chain_candidate_count_total", 0.0)
        ),
        "support_authorized_count": float(support_selected),
        "support_authorized_to_launch_count": float(min(support_selected, uav_launch_count_total)),
        "support_authorized_to_delivery_count": float(min(support_selected, uav_delivery_count_total)),
        "support_aborted_no_benefit_count": float(getattr(planner, "support_no_gain_backoff_block_count_total", support_no_gain)),
        "support_cooldown_block_count": float(getattr(planner, "support_no_gain_backoff_block_count_total", 0.0)),
        "support_unauthorized_forward_blocked_count": float(getattr(planner, "support_no_gain_backoff_block_count_total", 0.0)),
        "support_unauthorized_recovery_blocked_count": float(
            _require_info(last_info, "routine_near_completion_recovery_blocked_count", episode_idx=0)
            if "routine_near_completion_recovery_blocked_count" in last_info
            else getattr(planner, "support_no_gain_backoff_block_count_total", 0.0)
        ),
        "support_unauthorized_goal_hijack_blocked_count": float(
            float(getattr(planner, "support_no_gain_backoff_block_count_total", 0.0))
            + float(_require_info(last_info, "routine_near_completion_support_blocked_count", episode_idx=0))
        ),
        "support_unauthorized_blocked_count": float(
            float(getattr(planner, "support_no_gain_backoff_block_count_total", 0.0))
            + float(_require_info(last_info, "routine_near_completion_support_blocked_count", episode_idx=0))
            + float(_require_info(last_info, "routine_near_completion_recovery_blocked_count", episode_idx=0))
        ),
        "safety_recovery_authorized_count": float(_require_info(last_info, "uav_forced_recovery_count_total", episode_idx=0)),
        # launch quality instrumentation
        "launch_quality_candidate_count": float(_require_info(last_info, "uav_launch_gate_enter_count", episode_idx=0)),
        "launch_quality_branch_called_count": float(_require_info(last_info, "uav_launch_gate_enter_count", episode_idx=0)),
        "launch_quality_allowed_count": float(uav_launch_count_total),
        "launch_quality_blocked_low_recovery_margin_count": float(_require_info(last_info, "uav_launch_gate_block_recovery_margin_count", episode_idx=0)),
        "launch_quality_blocked_low_battery_count": float(_require_info(last_info, "uav_launch_gate_block_below_launch_min_count", episode_idx=0)),
        "launch_quality_blocked_recent_reject_count": float(getattr(planner, "uav_reject_cache_hit_count_total", 0.0)),
        "launch_quality_blocked_lifeline_risk_count": float(_require_info(last_info, "tc_override_blocked_lifeline_risk_count", episode_idx=0)),
        "launch_quality_blocked_count": float(
            float(_require_info(last_info, "uav_launch_gate_block_recovery_margin_count", episode_idx=0))
            + float(_require_info(last_info, "uav_launch_gate_block_below_launch_min_count", episode_idx=0))
            + float(_require_info(last_info, "uav_launch_gate_block_corridor_count", episode_idx=0))
            + float(_require_info(last_info, "uav_launch_gate_block_other_count", episode_idx=0))
        ),
        "good_launch_count": float(uav_delivery_count_total),
        "wasted_launch_count": float(max(uav_launch_count_total - uav_delivery_count_total, 0.0)),
        "risky_launch_count": float(_require_info(last_info, "uav_launch_gate_rendezvous_safe_relaxed_count", episode_idx=0)),
        "stale_launch_count": float(max(_require_info(last_info, "uav_launch_gate_enter_count", episode_idx=0) - uav_launch_count_total, 0.0)),
        # tc completion chain instrumentation
        "tc_completion_chain_candidate_count": float(tc_tier2_candidate + tc_tier3_candidate),
        "tc_completion_chain_branch_called_count": float(tc_tier2_candidate + tc_tier3_candidate),
        "tc_completion_chain_committed_count": float(tc_tier2_selected + tc_tier3_selected),
        "tc_completion_chain_to_launch_count": float(min(tc_tier2_selected + tc_tier3_selected, uav_launch_count_total)),
        "tc_completion_chain_to_delivery_count": float(min(tc_tier2_selected + tc_tier3_selected, uav_delivery_count_total)),
        "tc_completion_chain_blocked_not_delivery_feasible_count": float(_require_info(last_info, "tc_override_blocked_not_full_sortie_feasible_count", episode_idx=0)),
        "tc_completion_chain_blocked_routine_delay_count": float(last_info.get("tc_override_blocked_routine_delay_count", 0.0)),
        "tc_completion_chain_blocked_recent_reject_count": float(last_info.get("tc_override_blocked_recent_reject_count", 0.0)),
        "tc_residual_followup_candidate_count": float(getattr(planner, "tc_residual_followup_candidate_count", tc_service_start_no_completion_count)),
        "tc_residual_followup_committed_count": float(min(tc_service_start_no_completion_count, tc_tier2_selected + tc_tier3_selected)),
        "tc_residual_followup_to_launch_count": float(min(tc_service_start_no_completion_count, uav_launch_count_total)),
        "tc_residual_followup_to_delivery_count": float(getattr(planner, "tc_residual_followup_to_delivery_count", 0.0)),
        "tc_residual_followup_remaining_demand_resolved_count": float(0.0),
        # event minimal local instrumentation
        "event_detected_count": float(
            getattr(
                planner,
                "event_detected_count",
                getattr(planner, "event_refresh_count", 0.0) + getattr(planner, "hard_event_refresh_count_total", 0.0),
            )
        ),
        "event_minimal_local_branch_called_count": float(
            getattr(planner, "erc_event_gate_pass_count_total", 0.0)
            + getattr(planner, "erc_event_gate_reject_count_total", 0.0)
        ),
        "event_local_gate_pass_count": float(getattr(planner, "erc_event_gate_pass_count_total", 0.0)),
        "event_local_gate_reject_count": float(getattr(planner, "erc_event_gate_reject_count_total", 0.0)),
        "global_refresh_blocked_count": float(
            getattr(
                planner,
                "global_refresh_blocked_count",
                getattr(planner, "low_value_refresh_blocked_by_ablation_count_total", 0.0)
                + getattr(planner, "map_ranking_refresh_blocked_by_ablation_count_total", 0.0),
            )
        ),
        "local_correction_count": float(
            getattr(
                planner,
                "local_command_generated_count",
                getattr(planner, "normal_stall_local_correction_count_total", 0.0)
                + getattr(planner, "truck_dead_end_local_path_repair_count_total", 0.0)
                + getattr(planner, "path_blocked_local_path_repair_count_total", 0.0),
            )
        ),
        "hard_event_kept_count": float(getattr(planner, "hard_event_refresh_count_total", 0.0)),
        "weak_event_suppressed_count": float(getattr(planner, "weak_event_suppressed_count", getattr(planner, "low_value_refresh_blocked_by_ablation_count_total", 0.0))),
        "forced_switch_blocked_count": float(getattr(planner, "forced_switch_blocked_count", getattr(planner, "goal_switch_rejected_by_threshold_count_total", 0.0))),
        # scoring shrink instrumentation
        "event_bonus_candidate_count": float(getattr(planner, "goal_switch_candidate_count_total", 0.0)),
        "scoring_shrink_branch_called_count": float(getattr(planner, "goal_switch_candidate_count_total", 0.0)),
        "event_bonus_applied_count": float(getattr(planner, "event_scoring_bonus_applied_count_total", 0.0)),
        "event_bonus_suppressed_count": float(getattr(planner, "event_scoring_bonus_blocked_by_ablation_count_total", 0.0)),
        "event_bonus_scaled_count": float(
            max(
                getattr(planner, "goal_switch_candidate_count_total", 0.0)
                - getattr(planner, "event_scoring_bonus_applied_count_total", 0.0)
                - getattr(planner, "event_scoring_bonus_blocked_by_ablation_count_total", 0.0),
                0.0,
            )
        ),
        "high_priority_bonus_full_sortie_count": float(_require_info(last_info, "tc_override_predicted_delivery_feasible_count", episode_idx=0)),
        "bonus_changed_top1_count": float(0.0),
        "candidate_rank_flip_count": float(0.0),
        "routine_assigned_no_progress_count": float(routine_assigned_no_progress_count),
        "routine_near_completion_interrupted_count": float(routine_near_completion_interrupted_count),
        "tc_assigned_no_launch_count": float(tc_assigned_no_launch_count),
        "tc_launch_no_delivery_count": float(max(uav_launch_count_total - uav_delivery_count_total, 0.0)),
        "tc_service_start_no_completion_count": float(tc_service_start_no_completion_count),
        "support_no_benefit_count": float(support_no_gain),
        "recovery_pressure_count": float(_require_info(last_info, "uav_forced_recovery_count_total", episode_idx=0)),
        "forced_infeasible_switch_count": float(getattr(planner, "goal_switch_forced_reason_infeasible_total", 0.0)),
        **(
            getattr(planner, "get_alns_diagnostics")().to_flat_dict()
            if callable(getattr(planner, "get_alns_diagnostics", None))
            else {}
        ),
        "hard_constraint_violation_count": float(
            getattr(getattr(planner, "alns_diagnostics", None), "hard_constraint_violation_count", 0.0)
        ),
        "alns_risk_pressure_task_count": float(getattr(planner, "alns_risk_pressure_task_count_total", 0.0)),
        "alns_ghost_task_count": float(getattr(planner, "alns_ghost_task_count_total", 0.0)),
        "alns_disturbance_rate_last": float(getattr(planner, "alns_disturbance_rate_last", 0.0)),
        "alns_horizon_steps_last": float(getattr(planner, "alns_horizon_steps_last", 0.0)),
        "alns_solution_mode": str(getattr(env.cfg, "alns_solution_mode", "")),
        "alns_sequence_length": float(getattr(env.cfg, "alns_sequence_length", 0)),
        "alns_operator_pool": str(getattr(env.cfg, "alns_operator_pool", "")),
        "alns_selection_mode": str(getattr(env.cfg, "alns_selection_mode", "")),
        "metric_sanity_violation_count": float(len(ratio_violations)),
        "refresh_total_count": float(getattr(planner, "refresh_total_count", 0.0)),
        "fixed_interval_refresh_count": float(getattr(planner, "fixed_interval_refresh_count", 0.0)),
        "event_refresh_count": float(getattr(planner, "event_refresh_count", 0.0)),
        "erc_event_detected_count": float(getattr(planner, "erc_event_detected_count_total", 0.0)),
        "erc_event_gate_pass_count": float(getattr(planner, "erc_event_gate_pass_count_total", 0.0)),
        "erc_event_gate_reject_count": float(getattr(planner, "erc_event_gate_reject_count_total", 0.0)),
        "erc_local_correction_count": float(getattr(planner, "erc_local_correction_count_total", 0.0)),
        "erc_global_replan_count": float(getattr(planner, "erc_global_replan_count_total", 0.0)),
        "committed_goal_hold_count": float(getattr(planner, "committed_goal_hold_count_total", 0.0)),
        "committed_goal_broken_count": float(getattr(planner, "committed_goal_broken_count_total", 0.0)),
        "committed_goal_broken_reason_hard_invalid_count": float(
            getattr(planner, "committed_goal_broken_reason_hard_invalid_count_total", 0.0)
        ),
        "committed_goal_broken_reason_stall_count": float(
            getattr(planner, "committed_goal_broken_reason_stall_count_total", 0.0)
        ),
        "committed_goal_broken_reason_tc_gain_count": float(
            getattr(planner, "committed_goal_broken_reason_tc_gain_count_total", 0.0)
        ),
        "airborne_uav_goal_lock_count": float(getattr(planner, "airborne_uav_goal_lock_count_total", 0.0)),
        "path_blocked_local_agent_count": float(getattr(planner, "path_blocked_local_agent_count_total", 0.0)),
        "high_priority_event_rejected_no_launchable_uav_count": float(
            getattr(planner, "high_priority_event_rejected_no_launchable_uav_count_total", 0.0)
        ),
        "no_event_fallback_refresh_count": float(getattr(planner, "no_event_fallback_refresh_count", 0.0)),
        "initial_refresh_count": float(getattr(planner, "initial_refresh_count", 0.0)),
        "empty_goal_refresh_count": float(getattr(planner, "empty_goal_refresh_count", 0.0)),
        "steps_since_last_refresh_mean": float(
            float(getattr(planner, "steps_since_last_refresh_sum", 0.0))
            / max(float(getattr(planner, "refresh_total_count", 0.0)), 1.0)
        ),
        "steps_since_last_refresh_max": float(getattr(planner, "steps_since_last_refresh_max", 0.0)),
        "event_refresh_reason_arrival_count": float(getattr(planner, "event_refresh_reason_arrival_count_total", 0.0)),
        "event_refresh_reason_resolution_count": float(getattr(planner, "event_refresh_reason_resolution_count_total", 0.0)),
        "event_refresh_reason_uav_idle_count": float(getattr(planner, "event_refresh_reason_uav_idle_count_total", 0.0)),
        "event_refresh_reason_truck_idle_count": float(getattr(planner, "event_refresh_reason_truck_idle_count_total", 0.0)),
        "event_refresh_reason_map_update_light_count": float(getattr(planner, "event_refresh_reason_map_update_light_count_total", 0.0)),
        "event_refresh_reason_map_update_hard_count": float(getattr(planner, "event_refresh_reason_map_update_hard_count_total", 0.0)),
        "event_refresh_reason_goal_invalid_count": float(getattr(planner, "event_refresh_reason_goal_invalid_count_total", 0.0)),
        "event_refresh_reason_path_blocked_count": float(getattr(planner, "event_refresh_reason_path_blocked_count_total", 0.0)),
        "event_refresh_reason_goal_unreachable_count": float(getattr(planner, "event_refresh_reason_goal_unreachable_count_total", 0.0)),
        "event_refresh_reason_uav_safety_count": float(getattr(planner, "event_refresh_reason_uav_safety_count_total", 0.0)),
        "event_refresh_reason_truck_dead_end_count": float(getattr(planner, "event_refresh_reason_truck_dead_end_count_total", 0.0)),
        "event_refresh_reason_high_priority_uncovered_count": float(getattr(planner, "event_refresh_reason_high_priority_uncovered_count_total", 0.0)),
        "event_refresh_reason_normal_stall_count": float(getattr(planner, "event_refresh_reason_normal_stall_count_total", 0.0)),
        "event_refresh_no_goal_change_count": float(getattr(planner, "event_refresh_no_goal_change_count_total", 0.0)),
        "event_refresh_goal_change_count": float(getattr(planner, "event_refresh_goal_change_count_total", 0.0)),
        "event_refresh_goal_change_ratio": float(
            float(getattr(planner, "event_refresh_goal_change_count_total", 0.0))
            / max(float(getattr(planner, "event_refresh_count", 0.0)), 1.0)
        ),
        "event_refresh_to_launch_count": float(getattr(planner, "event_refresh_to_launch_count_total", 0.0)),
        "event_refresh_to_completion_count": float(getattr(planner, "event_refresh_to_completion_count_total", 0.0)),
        "event_refresh_to_completion_ratio": float(
            float(getattr(planner, "event_refresh_to_completion_count_total", 0.0))
            / max(float(getattr(planner, "event_refresh_count", 0.0)), 1.0)
        ),
        "event_refresh_followed_by_reject_count": float(getattr(planner, "event_refresh_followed_by_reject_count_total", 0.0)),
        "event_refresh_followed_by_stall_count": float(getattr(planner, "event_refresh_followed_by_stall_count_total", 0.0)),
        "hard_event_refresh_count_total": float(getattr(planner, "hard_event_refresh_count_total", 0.0)),
        "hard_event_reason_goal_invalid_count": float(getattr(planner, "hard_event_reason_goal_invalid_count_total", 0.0)),
        "hard_event_reason_current_goal_unreachable_count": float(
            getattr(planner, "hard_event_reason_current_goal_unreachable_count_total", 0.0)
        ),
        "hard_event_reason_path_blocked_count": float(getattr(planner, "hard_event_reason_path_blocked_count_total", 0.0)),
        "hard_event_reason_uav_safety_count": float(getattr(planner, "hard_event_reason_uav_safety_count_total", 0.0)),
        "hard_event_reason_uav_recovery_count": float(getattr(planner, "hard_event_reason_uav_recovery_count_total", 0.0)),
        "hard_event_reason_truck_dead_end_count": float(getattr(planner, "hard_event_reason_truck_dead_end_count_total", 0.0)),
        "hard_event_reason_high_priority_uncovered_count": float(
            getattr(planner, "hard_event_reason_high_priority_uncovered_count_total", 0.0)
        ),
        "hard_event_reason_normal_stall_count": float(getattr(planner, "hard_event_reason_normal_stall_count_total", 0.0)),
        "hard_event_reason_assigned_but_not_progressing_count": float(
            getattr(planner, "hard_event_reason_assigned_but_not_progressing_count_total", 0.0)
        ),
        "hard_event_reason_goal_completed_count": float(getattr(planner, "hard_event_reason_goal_completed_count_total", 0.0)),
        "hard_event_reason_goal_failed_count": float(getattr(planner, "hard_event_reason_goal_failed_count_total", 0.0)),
        "goal_invalid_reason_task_completed": float(getattr(planner, "goal_invalid_reason_task_completed_total", 0.0)),
        "goal_invalid_reason_task_failed": float(getattr(planner, "goal_invalid_reason_task_failed_total", 0.0)),
        "goal_invalid_reason_task_missing": float(getattr(planner, "goal_invalid_reason_task_missing_total", 0.0)),
        "goal_invalid_reason_truck_unreachable": float(getattr(planner, "goal_invalid_reason_truck_unreachable_total", 0.0)),
        "goal_invalid_reason_uav_energy_infeasible": float(
            getattr(planner, "goal_invalid_reason_uav_energy_infeasible_total", 0.0)
        ),
        "goal_invalid_reason_uav_recovery_margin": float(
            getattr(planner, "goal_invalid_reason_uav_recovery_margin_total", 0.0)
        ),
        "goal_invalid_reason_uav_corridor": float(getattr(planner, "goal_invalid_reason_uav_corridor_total", 0.0)),
        "goal_invalid_reason_uav_comm_block": float(getattr(planner, "goal_invalid_reason_uav_comm_block_total", 0.0)),
        "goal_invalid_reason_uav_not_loaded": float(getattr(planner, "goal_invalid_reason_uav_not_loaded_total", 0.0)),
        "goal_invalid_reason_uav_not_docked": float(getattr(planner, "goal_invalid_reason_uav_not_docked_total", 0.0)),
        "goal_invalid_reason_soft_reject_cache": float(
            getattr(planner, "goal_invalid_reason_soft_reject_cache_total", 0.0)
        ),
        "suspect_soft_as_hard_count": float(getattr(planner, "suspect_soft_as_hard_count_total", 0.0)),
        "hard_event_refresh_no_goal_change_count": float(
            getattr(planner, "hard_event_refresh_no_goal_change_count_total", 0.0)
        ),
        "hard_event_refresh_goal_change_count": float(
            getattr(planner, "hard_event_refresh_goal_change_count_total", 0.0)
        ),
        "hard_event_refresh_goal_change_ratio": float(
            float(getattr(planner, "hard_event_refresh_goal_change_count_total", 0.0))
            / max(float(getattr(planner, "hard_event_refresh_count_total", 0.0)), 1.0)
        ),
        "hard_event_refresh_to_launch_count": float(getattr(planner, "hard_event_refresh_to_launch_count_total", 0.0)),
        "hard_event_refresh_to_completion_count": float(
            getattr(planner, "hard_event_refresh_to_completion_count_total", 0.0)
        ),
        "hard_event_refresh_followed_by_reject_count": float(
            getattr(planner, "hard_event_refresh_followed_by_reject_count_total", 0.0)
        ),
        "hard_event_refresh_followed_by_stall_count": float(
            getattr(planner, "hard_event_refresh_followed_by_stall_count_total", 0.0)
        ),
        "normal_stall_candidate_count": float(getattr(planner, "normal_stall_candidate_count_total", 0.0)),
        "normal_stall_blocked_by_persist_count": float(getattr(planner, "normal_stall_blocked_by_persist_count_total", 0.0)),
        "normal_stall_blocked_by_cooldown_count": float(getattr(planner, "normal_stall_blocked_by_cooldown_count_total", 0.0)),
        "normal_stall_local_correction_count": float(getattr(planner, "normal_stall_local_correction_count_total", 0.0)),
        "normal_stall_global_refresh_count": float(getattr(planner, "normal_stall_global_refresh_count_total", 0.0)),
        "truck_dead_end_candidate_count": float(getattr(planner, "truck_dead_end_candidate_count_total", 0.0)),
        "truck_dead_end_blocked_by_persist_count": float(getattr(planner, "truck_dead_end_blocked_by_persist_count_total", 0.0)),
        "truck_dead_end_blocked_by_cooldown_count": float(getattr(planner, "truck_dead_end_blocked_by_cooldown_count_total", 0.0)),
        "truck_dead_end_local_path_repair_count": float(getattr(planner, "truck_dead_end_local_path_repair_count_total", 0.0)),
        "truck_dead_end_local_goal_reassign_count": float(getattr(planner, "truck_dead_end_local_goal_reassign_count_total", 0.0)),
        "truck_dead_end_global_refresh_count": float(getattr(planner, "truck_dead_end_global_refresh_count_total", 0.0)),
        "truck_dead_end_noop_count": float(getattr(planner, "truck_dead_end_noop_count_total", 0.0)),
        "truck_dead_end_routine_localized_count": float(getattr(planner, "truck_dead_end_routine_localized_count_total", 0.0)),
        "truck_dead_end_emergency_kept_hard_count": float(getattr(planner, "truck_dead_end_emergency_kept_hard_count_total", 0.0)),
        "truck_dead_end_support_kept_hard_count": float(getattr(planner, "truck_dead_end_support_kept_hard_count_total", 0.0)),
        "truck_dead_end_recovery_kept_hard_count": float(getattr(planner, "truck_dead_end_recovery_kept_hard_count_total", 0.0)),
        "truck_dead_end_local_repair_no_goal_change_count": float(
            getattr(planner, "truck_dead_end_local_repair_no_goal_change_count_total", 0.0)
        ),
        "truck_dead_end_global_refresh_no_goal_change_count": float(
            getattr(planner, "truck_dead_end_global_refresh_no_goal_change_count_total", 0.0)
        ),
        "goal_invalid_hard_count": float(getattr(planner, "goal_invalid_hard_count_total", 0.0)),
        "goal_invalid_soft_count": float(getattr(planner, "goal_invalid_soft_count_total", 0.0)),
        "goal_invalid_soft_suppressed_count": float(getattr(planner, "goal_invalid_soft_suppressed_count_total", 0.0)),
        "goal_invalid_soft_escalated_count": float(getattr(planner, "goal_invalid_soft_escalated_count_total", 0.0)),
        "uav_recovery_hard_count": float(getattr(planner, "uav_recovery_hard_count_total", 0.0)),
        "uav_recovery_soft_count": float(getattr(planner, "uav_recovery_soft_count_total", 0.0)),
        "uav_recovery_soft_suppressed_count": float(getattr(planner, "uav_recovery_soft_suppressed_count_total", 0.0)),
        "uav_recovery_local_action_count": float(getattr(planner, "uav_recovery_local_action_count_total", 0.0)),
        "uav_recovery_global_refresh_count": float(getattr(planner, "uav_recovery_global_refresh_count_total", 0.0)),
        "path_blocked_candidate_count": float(getattr(planner, "path_blocked_candidate_count_total", 0.0)),
        "path_blocked_nonimpact_suppressed_count": float(getattr(planner, "path_blocked_nonimpact_suppressed_count_total", 0.0)),
        "path_blocked_impacted_current_path_count": float(getattr(planner, "path_blocked_impacted_current_path_count_total", 0.0)),
        "path_blocked_impacted_goal_reachability_count": float(getattr(planner, "path_blocked_impacted_goal_reachability_count_total", 0.0)),
        "path_blocked_impacted_recovery_count": float(getattr(planner, "path_blocked_impacted_recovery_count_total", 0.0)),
        "path_blocked_local_path_repair_count": float(getattr(planner, "path_blocked_local_path_repair_count_total", 0.0)),
        "path_blocked_local_goal_reassign_count": float(getattr(planner, "path_blocked_local_goal_reassign_count_total", 0.0)),
        "path_blocked_global_refresh_count": float(getattr(planner, "path_blocked_global_refresh_count_total", 0.0)),
        "path_blocked_noop_count": float(getattr(planner, "path_blocked_noop_count_total", 0.0)),
        "path_blocked_routine_localized_count": float(getattr(planner, "path_blocked_routine_localized_count_total", 0.0)),
        "path_blocked_emergency_kept_hard_count": float(getattr(planner, "path_blocked_emergency_kept_hard_count_total", 0.0)),
        "path_blocked_recovery_kept_hard_count": float(getattr(planner, "path_blocked_recovery_kept_hard_count_total", 0.0)),
        "path_blocked_support_kept_hard_count": float(getattr(planner, "path_blocked_support_kept_hard_count_total", 0.0)),
        "path_blocked_goal_unreachable_kept_hard_count": float(
            getattr(planner, "path_blocked_goal_unreachable_kept_hard_count_total", 0.0)
        ),
        "path_blocked_local_repair_no_goal_change_count": float(
            getattr(planner, "path_blocked_local_repair_no_goal_change_count_total", 0.0)
        ),
        "path_blocked_global_refresh_no_goal_change_count": float(
            getattr(planner, "path_blocked_global_refresh_no_goal_change_count_total", 0.0)
        ),
        "uav_launch_gate_enter_count": float(_require_info(last_info, "uav_launch_gate_enter_count", episode_idx=0)),
        "uav_launch_gate_direct_safe_count": float(_require_info(last_info, "uav_launch_gate_direct_safe_count", episode_idx=0)),
        "uav_launch_gate_rendezvous_safe_count": float(_require_info(last_info, "uav_launch_gate_rendezvous_safe_count", episode_idx=0)),
        "uav_launch_gate_rendezvous_safe_relaxed_count": float(_require_info(last_info, "uav_launch_gate_rendezvous_safe_relaxed_count", episode_idx=0)),
        "uav_launch_gate_block_below_launch_min_count": float(_require_info(last_info, "uav_launch_gate_block_below_launch_min_count", episode_idx=0)),
        "uav_launch_gate_block_recovery_margin_count": float(_require_info(last_info, "uav_launch_gate_block_recovery_margin_count", episode_idx=0)),
        "uav_launch_gate_block_corridor_count": float(_require_info(last_info, "uav_launch_gate_block_corridor_count", episode_idx=0)),
        "uav_launch_gate_block_other_count": float(_require_info(last_info, "uav_launch_gate_block_other_count", episode_idx=0)),
        "map_update_hard_seen_count": float(map_update_hard_seen_count),
        "map_update_hard_actionable_count": float(map_update_hard_actionable_count),
        "map_update_hard_deferred_count": float(map_update_hard_deferred_count),
        "map_update_hard_immediate_refresh_count": float(map_update_hard_immediate_refresh_count),
        "map_update_hard_reason_path_blocked_count": float(map_update_hard_reason_path_blocked_count),
        "map_update_hard_reason_goal_unreachable_count": float(map_update_hard_reason_goal_unreachable_count),
        "map_update_hard_reason_ranking_changed_count": float(map_update_hard_reason_ranking_changed_count),
        "map_update_hard_reason_dead_end_count": float(map_update_hard_reason_dead_end_count),
        "map_update_hard_reason_recovery_path_fractured_count": float(map_update_hard_reason_recovery_path_fractured_count),
        "timecritical_tier3_candidate_count": float(tc_tier3_candidate),
        "timecritical_tier3_selected_count": float(tc_tier3_selected),
        "timecritical_tier2_candidate_count": float(tc_tier2_candidate),
        "timecritical_tier2_selected_count": float(tc_tier2_selected),
        "timecritical_candidate_ignored_count": float(tc_ignored),
        "tc_direct_feasible_count": float(tc_direct_feasible),
        "tc_support_required_count": float(tc_support_required),
        "tc_truly_infeasible_count": float(tc_truly_infeasible),
        "tc_support_lock_created_count": float(tc_support_lock_created),
        "tc_support_lock_to_dispatch_count": float(tc_support_lock_to_dispatch),
        "region_commitment_setup_count": float(region_commitment_setup),
        "region_commitment_region_count": float(region_commitment_region_count),
        "region_commitment_effective_k": float(region_commitment_effective_k),
        "region_commitment_effective_enabled": float(region_commitment_effective_enabled),
        "region_commitment_auto_score": float(region_commitment_auto_score),
        "region_commitment_separation_score": float(region_commitment_separation_score),
        "region_commitment_load_balance_score": float(region_commitment_load_balance_score),
        "region_commitment_coverage_score": float(region_commitment_coverage_score),
        "region_commitment_strength": float(region_commitment_strength),
        "region_commitment_auto_enabled_count": float(region_commitment_auto_enabled),
        "region_commitment_auto_disabled_count": float(region_commitment_auto_disabled),
        "region_commitment_local_candidate_count": float(region_commitment_local_candidates),
        "region_commitment_cross_filtered_count": float(region_commitment_cross_filtered),
        "region_commitment_cross_override_count": float(region_commitment_cross_override),
        "region_commitment_outlier_task_count": float(region_commitment_outlier_tasks),
        "region_commitment_outlier_filtered_count": float(region_commitment_outlier_filtered),
        "region_commitment_outlier_override_count": float(region_commitment_outlier_override),
        "support_selected_with_bound_timecritical_delivery_count": float(sup_bound_tc),
        "support_selected_without_bound_timecritical_delivery_count": float(sup_without_tc),
        "support_filtered_no_bound_timecritical_delivery_count": float(sup_filtered_no_bind),
        "support_selected_with_bound_bulk_delivery_count": float(sup_bound_bulk),
                "support_bind_success_rate": float(sup_bind_success_rate),
        "support_selected_count": float(support_selected),
        "support_improves_serviceability_count": float(support_gain),
        "support_no_gain_count": float(support_no_gain),
        "recovery_reject_numerator_count": float(reject_margin),
        "recovery_reject_denominator_count": float(recovery_feas_eval),
        "uav_recovery_feasibility_eval_count": float(recovery_feas_eval),
        "recovery_reject_rate": float(recovery_reject_rate),
        "support_conversion_rate": float(support_conversion_rate),
        "relaxed_conversion_rate": float(relaxed_conversion_rate),
        **repro_digests,
        **freeze_contract_digests,
        "_hard_event_offender_rows": hard_offender_rows,
        "_hard_event_reason_rows": hard_reason_rows,
        "_task_outcome_rows": task_outcome_rows,
        "_routine_trace_rows": routine_trace_rows,
        "_truck_routine_summary_rows": truck_routine_summary_rows,
        "_support_trace_rows": support_trace_rows,
        "_uav_execution_rows": uav_execution_rows,
        "_agent_step_rows": agent_step_rows,
        "_task_proximity_decision_rows": task_proximity_decision_rows,
        "_switch_decision_ledger_rows": switch_decision_rows,
        "_tc_override_trace_rows": list(getattr(env, "_tc_override_trace_rows", [])),
        "_v2_sortie_lifecycle_rows": list(getattr(env, "v2_sortie_lifecycle_rows", [])),
        "_v2_support_lifecycle_rows": list(getattr(env, "v2_support_lifecycle_rows", [])),
        "_v2_safety_lifecycle_rows": list(getattr(env, "v2_safety_lifecycle_rows", [])),
        "_v2_support_passenger_trace_rows": list(getattr(env, "v2_support_passenger_trace_rows", [])),
        "_v2_support_quality_rows": list(getattr(env, "v2_support_quality_rows", [])),
        "_invalid_action_records": [
            _jsonable_invalid_record(r)
            for r in getattr(env, "invalid_action_records", [])
        ],
        "_objective_shadow_records": (
            getattr(planner, "export_objective_shadow_records")()[shadow_start_index:]
            if callable(getattr(planner, "export_objective_shadow_records", None))
            else []
        ),
        "_k2_sequence_records": (
            getattr(planner, "export_k2_sequence_records")()[k2_start_index:]
            if callable(getattr(planner, "export_k2_sequence_records", None))
            else []
        ),
        "_k2_runtime_sequence_records": (
            getattr(planner, "export_k2_runtime_sequence_records")()[k2_runtime_start_index:]
            if callable(getattr(planner, "export_k2_runtime_sequence_records", None))
            else []
        ),
        "_k2_sa_delta_records": (
            getattr(planner, "export_k2_sa_delta_records")()[k2_sa_start_index:]
            if callable(getattr(planner, "export_k2_sa_delta_records", None))
            else []
        ),
        "_canonical_operator_records": (
            getattr(planner, "export_canonical_operator_records")()[canonical_operator_start_index:]
            if callable(getattr(planner, "export_canonical_operator_records", None))
            else []
        ),
        "_support_execution_records": (
            getattr(planner, "export_support_execution_records")()[support_execution_start_index:]
            if callable(getattr(planner, "export_support_execution_records", None))
            else []
        ),
        "_operator_weight_trajectory_records": (
            getattr(planner, "export_operator_weight_trajectory_records")()[operator_weight_start_index:]
            if callable(getattr(planner, "export_operator_weight_trajectory_records", None))
            else []
        ),
        "_event_trigger_records": (
            getattr(planner, "export_event_trigger_records")()[event_trigger_start_index:]
            if callable(getattr(planner, "export_event_trigger_records", None))
            else []
        ),
        "_sa_calibration_records": (
            getattr(planner, "export_sa_calibration_records")()[sa_calibration_start_index:]
            if callable(getattr(planner, "export_sa_calibration_records", None))
            else []
        ),
        "_live_candidate_records": (
            getattr(planner, "export_live_candidate_records")()[live_candidate_start_index:]
            if callable(getattr(planner, "export_live_candidate_records", None))
            else []
        ),
        "_ranker_runtime_records": (
            getattr(planner, "export_ranker_runtime_records")()[ranker_runtime_start_index:]
            if callable(getattr(planner, "export_ranker_runtime_records", None))
            else []
        ),
        "_repair_candidate_pool_records": (
            getattr(planner, "export_repair_candidate_pool_records")()[repair_candidate_pool_start_index:]
            if callable(getattr(planner, "export_repair_candidate_pool_records", None))
            else []
        ),
        "_adaptive_horizon_records": (
            getattr(planner, "export_adaptive_horizon_records")()[adaptive_horizon_start_index:]
            if callable(getattr(planner, "export_adaptive_horizon_records", None))
            else []
        ),
        "_local_search_records": (
            getattr(planner, "export_local_search_records")()[local_search_start_index:]
            if callable(getattr(planner, "export_local_search_records", None))
            else []
        ),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=_default_config_path())
    p.add_argument("--base-config", type=str, default=_default_base_config_path())
    p.add_argument("--mainline-v2-config", type=str, default="")
    p.add_argument("--results-root", type=str, default=_default_results_root())
    p.add_argument("--run-name", type=str, default="paper_matrix")
    p.add_argument("--scales", type=str, default="")
    p.add_argument("--scenarios", type=str, default="")
    p.add_argument("--scale-scenario-pairs", type=str, default="")
    p.add_argument("--methods", type=str, default="")
    p.add_argument("--seeds", type=str, default="")
    p.add_argument("--episodes-per-seed", type=int, default=0)
    p.add_argument("--monitor-snap-enabled", action="store_true")
    p.add_argument("--use-event-trigger", type=str, default="")
    p.add_argument("--use-risk-term", type=str, default="")
    p.add_argument("--use-rth-repair", type=str, default="")
    p.add_argument("--enable-rth-mask", type=str, default="")
    p.add_argument("--encoder-type", type=str, default="")
    p.add_argument("--alns-solution-mode", type=str, default="")
    p.add_argument("--alns-sequence-length", type=int, default=0)
    p.add_argument("--alns-operator-pool", type=str, default="")
    p.add_argument("--adaptive-horizon-mode", type=str, default="")
    p.add_argument("--local-search-mode", type=str, default="")
    p.add_argument("--local-search-max-moves-per-iteration", type=int, default=-1)
    p.add_argument("--local-search-max-exact-checks-per-iteration", type=int, default=-1)
    p.add_argument("--local-search-max-time-ms-per-iteration", type=int, default=-1)
    p.add_argument("--local-search-disabled-moves", type=str, default="")
    p.add_argument("--physical-environment-version", type=str, default="")
    p.add_argument("--physical-environment-safety-protocol", type=str, default="")
    p.add_argument("--physical-freeze-fair-config", action="store_true")
    p.add_argument("--candidate-ranker-mode", type=str, default="")
    p.add_argument("--candidate-ranker-pool-size", type=int, default=0)
    p.add_argument("--candidate-ranker-exact-check-budget", type=int, default=0)
    p.add_argument("--candidate-ranker-exploration-count", type=int, default=-1)
    p.add_argument("--enable-critical-recovery-repair", action="store_true")
    p.add_argument("--enable-critical-support-rebind", action="store_true")
    p.add_argument("--enable-support-rebind-margin-aware", action="store_true")
    p.add_argument("--enable-support-rebind-anchor-ranking", action="store_true")
    p.add_argument("--enable-support-rebind-failed-binding-avoidance", action="store_true")
    p.add_argument("--support-rebind-failed-binding-penalty", type=str, default="")
    p.add_argument("--enable-support-rebind-critical-first-ordering", action="store_true")
    p.add_argument("--enable-safe-uav-dispatch-guard", action="store_true")
    p.add_argument("--support-rebind-margin-top-k", type=int, default=0)
    p.add_argument("--support-rebind-anchor-search-radius-factor", type=float, default=0.0)
    p.add_argument("--enable-lc-critical-recovery-path", action="store_true")
    p.add_argument("--enable-assigned-critical-reconstruct", action="store_true")
    p.add_argument("--enable-support-reposition-shadow", action="store_true")
    p.add_argument("--enable-single-task-trace", action="store_true")
    p.add_argument("--trace-task-id", type=str, default="")
    p.add_argument("--l-benchmark-mode", type=str, default="", help="L benchmark source: old|new")
    p.add_argument("--lightweight-metrics-only", action="store_true")
    p.add_argument("--enable-task-outcome-export", action="store_true")
    p.add_argument("--comm-blackout-emergency-coverage", type=float, default=-1.0)
    p.add_argument("--blockage-asymptote", type=float, default=-1.0)
    p.add_argument(
        "--publish-refactor-invalid-ledger",
        action="store_true",
        help="Also publish invalid-action outputs to docs/refactor (disabled for isolated runs).",
    )
    p.add_argument("--enable-event-ledger-detail", action="store_true")
    p.add_argument("--disable-event-ledger-detail", action="store_true")
    p.add_argument("--enable-agent-step-html", action="store_true")
    p.add_argument("--disable-agent-step-html", action="store_true")
    p.add_argument("--enable-switch-decision-ledger", action="store_true")
    p.add_argument("--disable-switch-decision-ledger", action="store_true")
    p.add_argument("--enable-tc-override-trace", action="store_true")
    p.add_argument(
        "--real-rc-config",
        type=str,
        default="",
        help="Override R-C real-case config path (default: configs/real_dujiangyan_RC_current.yaml).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    lightweight_metrics_only = bool(getattr(args, "lightweight_metrics_only", False))
    if lightweight_metrics_only:
        lightweight_audit_flags: Dict[str, bool] = {
            "enable_switch_decision_ledger": False,
            "enable_step_trace": False,
            "enable_event_ledger_detail": False,
            "enable_agent_step_html": False,
            "enable_top_offender_export": False,
            "enable_timeline_metrics": False,
            "enable_debug_transition_dump": False,
            "enable_per_step_audit": False,
            "enable_task_outcome_export": False,
        }
    else:
        lightweight_audit_flags = {k: True for k in LIGHTWEIGHT_AUDIT_FLAG_FIELDS}

    # CLI overrides for lightweight audit flags (explicit enable/disable).
    if bool(getattr(args, "enable_event_ledger_detail", False)):
        lightweight_audit_flags["enable_event_ledger_detail"] = True
    if bool(getattr(args, "disable_event_ledger_detail", False)):
        lightweight_audit_flags["enable_event_ledger_detail"] = False
    if bool(getattr(args, "enable_agent_step_html", False)):
        lightweight_audit_flags["enable_agent_step_html"] = True
    if bool(getattr(args, "disable_agent_step_html", False)):
        lightweight_audit_flags["enable_agent_step_html"] = False
    if bool(getattr(args, "enable_switch_decision_ledger", False)):
        lightweight_audit_flags["enable_switch_decision_ledger"] = True
    if bool(getattr(args, "disable_switch_decision_ledger", False)):
        lightweight_audit_flags["enable_switch_decision_ledger"] = False
    if bool(getattr(args, "enable_task_outcome_export", False)):
        lightweight_audit_flags["enable_task_outcome_export"] = True

    cfg_matrix = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg_base = yaml.safe_load(Path(args.base_config).read_text(encoding="utf-8"))
    if not isinstance(cfg_matrix, dict):
        raise TypeError("matrix config root must be mapping")
    if not isinstance(cfg_base, dict):
        raise TypeError("base config root must be mapping")
    cfg_effective_yaml = _merge_cfg(cfg_base, cfg_matrix)
    mainline_v2_yaml: Dict[str, Any] = {}
    if str(args.mainline_v2_config).strip():
        v2_path = Path(args.mainline_v2_config).resolve()
        loaded_v2 = yaml.safe_load(v2_path.read_text(encoding="utf-8"))
        if not isinstance(loaded_v2, dict):
            raise TypeError("mainline V2 config root must be mapping")
        mainline_v2_yaml = loaded_v2
        cfg_effective_yaml = _merge_cfg(cfg_effective_yaml, mainline_v2_yaml)
    if str(getattr(args, "real_rc_config", "")).strip():
        real_rc_cfg_path = (_default_project_root() / str(args.real_rc_config)).resolve()
    else:
        real_rc_cfg_path = _default_project_root() / "configs" / "real_dujiangyan_RC_current.yaml"
    real_rc_cfg_yaml: Dict[str, Any] = {}
    if real_rc_cfg_path.exists():
        loaded = yaml.safe_load(real_rc_cfg_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            real_rc_cfg_yaml = loaded

    matrix_cfg = dict(cfg_matrix.get("matrix", {}))
    runtime_cfg = dict(cfg_matrix.get("runtime", {}))

    if str(args.scale_scenario_pairs).strip():
        scale_scenario_pairs = _parse_scale_scenario_pairs(args.scale_scenario_pairs)
    else:
        cfg_pairs = _pairs_from_matrix_cfg(matrix_cfg)
        if cfg_pairs:
            scale_scenario_pairs = cfg_pairs
        else:
            scales_cfg = _parse_csv_list(args.scales) if str(args.scales).strip() else [str(x) for x in matrix_cfg.get("scales", ["M", "L"])]
            scenarios_cfg = _parse_csv_list(args.scenarios) if str(args.scenarios).strip() else [str(x) for x in matrix_cfg.get("scenarios", ["B", "C"])]
            scale_scenario_pairs = [(str(s).upper(), str(sc).upper()) for s in scales_cfg for sc in scenarios_cfg]

    methods = _parse_csv_list(args.methods) if str(args.methods).strip() else [str(x) for x in matrix_cfg.get("methods", SUPPORTED_METHODS)]
    methods = [_canonical_method_name(m) for m in methods]
    seeds = _parse_seed_list(args.seeds) if str(args.seeds).strip() else [int(x) for x in matrix_cfg.get("seeds", [0, 1, 2, 3, 4])]
    episodes_per_seed = int(args.episodes_per_seed) if int(args.episodes_per_seed) > 0 else int(matrix_cfg.get("episodes_per_seed", 5))

    # L-only default fast path: avoid heavy full matrix when caller does not specify overrides.
    l_only_pairs = bool(scale_scenario_pairs) and all(str(sc).upper() == "L" for sc, _ in scale_scenario_pairs)
    if l_only_pairs and (not str(args.methods).strip()):
        methods = ["erc_rhc"]
    if l_only_pairs and (not str(args.seeds).strip()):
        seeds = [0, 1]
    if l_only_pairs and int(args.episodes_per_seed) <= 0:
        episodes_per_seed = 1
    monitor_snap = bool(args.monitor_snap_enabled or runtime_cfg.get("monitor_snap_enabled", False))
    use_event_trigger_override = _parse_optional_bool(args.use_event_trigger)
    use_risk_term_override = _parse_optional_bool(args.use_risk_term)
    use_rth_repair_override = _parse_optional_bool(args.use_rth_repair)
    enable_rth_mask_override = _parse_optional_bool(args.enable_rth_mask)

    l_benchmark_mode_cli = str(args.l_benchmark_mode).strip().lower()
    cfg_preview = _flatten_cfg(cfg_effective_yaml, seed=int(seeds[0]), monitor_snap_enabled=monitor_snap)
    l_benchmark_mode_default = str(getattr(cfg_preview, "l_benchmark_mode", "new")).strip().lower()
    if l_benchmark_mode_default not in {"old", "new"}:
        l_benchmark_mode_default = "new"
    l_benchmark_mode = l_benchmark_mode_cli if l_benchmark_mode_cli in {"old", "new"} else l_benchmark_mode_default

    unknown = [m for m in methods if m not in SUPPORTED_METHODS]
    if unknown:
        raise ValueError(f"Unsupported methods: {unknown}; supported={SUPPORTED_METHODS}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.results_root) / f"{args.run_name}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    episode_rows: List[Dict[str, Any]] = []
    seed_rows: List[Dict[str, Any]] = []
    hard_event_offender_rows: List[Dict[str, Any]] = []
    hard_event_reason_rows: List[Dict[str, Any]] = []
    task_outcome_rows: List[Dict[str, Any]] = []
    routine_trace_rows: List[Dict[str, Any]] = []
    truck_routine_summary_rows: List[Dict[str, Any]] = []
    support_trace_rows: List[Dict[str, Any]] = []
    uav_execution_rows: List[Dict[str, Any]] = []
    agent_step_rows: List[Dict[str, Any]] = []
    task_proximity_decision_rows: List[Dict[str, Any]] = []
    tc_override_trace_rows: List[Dict[str, Any]] = []
    switch_decision_ledger_rows: List[Dict[str, Any]] = []
    v2_sortie_lifecycle_rows: List[Dict[str, Any]] = []
    v2_support_lifecycle_rows: List[Dict[str, Any]] = []
    v2_safety_lifecycle_rows: List[Dict[str, Any]] = []
    v2_support_passenger_trace_rows: List[Dict[str, Any]] = []
    v2_support_quality_rows: List[Dict[str, Any]] = []
    invalid_action_rows: List[Dict[str, Any]] = []
    objective_shadow_rows: List[Dict[str, Any]] = []
    k2_sequence_rows: List[Dict[str, Any]] = []
    k2_runtime_sequence_rows: List[Dict[str, Any]] = []
    k2_sa_delta_rows: List[Dict[str, Any]] = []
    canonical_operator_rows: List[Dict[str, Any]] = []
    support_execution_rows: List[Dict[str, Any]] = []
    operator_weight_rows: List[Dict[str, Any]] = []
    event_trigger_rows: List[Dict[str, Any]] = []
    sa_calibration_rows: List[Dict[str, Any]] = []
    live_candidate_rows: List[Dict[str, Any]] = []
    ranker_runtime_rows: List[Dict[str, Any]] = []
    repair_candidate_pool_rows: List[Dict[str, Any]] = []
    adaptive_horizon_rows: List[Dict[str, Any]] = []
    local_search_rows: List[Dict[str, Any]] = []
    method_effective_flags_by_method: Dict[str, Dict[str, Any]] = {}
    method_identity_by_method: Dict[str, Dict[str, Any]] = {}

    for scale, scenario in scale_scenario_pairs:
        for method in methods:
            for seed in seeds:
                    cfg_source_yaml = cfg_effective_yaml
                    if str(scale).upper() == "R" and real_rc_cfg_yaml:
                        cfg_source_yaml = _merge_cfg(cfg_effective_yaml, real_rc_cfg_yaml)
                    cfg = _flatten_cfg(cfg_source_yaml, seed=int(seed), monitor_snap_enabled=monitor_snap)
                    cfg = _scale_cfg(scale, _scenario_cfg(scenario, cfg), l_benchmark_mode=l_benchmark_mode)
                    cfg = _apply_experiment_sensitivity_overrides(
                        cfg,
                        scenario=scenario,
                        comm_coverage=float(args.comm_blackout_emergency_coverage),
                        blockage_asymptote=float(args.blockage_asymptote),
                    )
                    if str(scale).upper() == "R":
                        map_src = str(getattr(cfg, "map_source", "")).strip().lower()
                        if not bool(getattr(cfg, "real_case_enabled", False)):
                            raise ValueError("R-scale requires real_case_enabled=True")
                        if map_src not in {"osm_dem", "real", "real_city"}:
                            raise ValueError(f"R-scale invalid map_source={map_src!r}")
                        print(
                            "[R-SANITY]",
                            f"seed={int(seed)}",
                            f"method={str(method)}",
                            f"num_trucks={int(getattr(cfg, 'num_trucks', 0))}",
                            f"num_uavs={int(getattr(cfg, 'num_uavs', 0))}",
                            f"num_routine_bulk_tasks={int(getattr(cfg, 'num_routine_bulk_tasks', 0))}",
                            f"num_time_critical_lightweight_tasks={int(getattr(cfg, 'num_time_critical_lightweight_tasks', 0))}",
                            f"forced_island_emergency_tasks={int(getattr(cfg, 'forced_island_emergency_tasks', 0))}",
                            f"rc_strong_planner_mode_enabled={bool(getattr(cfg, 'rc_strong_planner_mode_enabled', False))}",
                        )

                    algorithm_package = _build_algorithm_package(
                        method,
                        cfg,
                        int(seed),
                        use_event_trigger_override=use_event_trigger_override,
                        use_risk_term_override=use_risk_term_override,
                        use_rth_repair_override=use_rth_repair_override,
                        enable_rth_mask_override=enable_rth_mask_override,
                        encoder_type_override=str(args.encoder_type),
                    )
                    planner = algorithm_package.planner
                    encoder_type = algorithm_package.encoder_type
                    enable_rth_mask = algorithm_package.enable_rth_mask
                    cfg = algorithm_package.runtime_cfg
                    method_identity_by_method[str(method).strip().lower()] = {
                        "algorithm_id": str(algorithm_package.algorithm_id),
                        "planner_class": str(algorithm_package.planner_class),
                        "backend_family": str(algorithm_package.backend_family),
                        "er_hlns_capability": bool(
                            algorithm_package.profile.has("er_hlns_route_plan")
                        ),
                        "public_environment_hash": str(
                            algorithm_package.public_environment_hash
                        ),
                        "algorithm_config_hash": str(
                            algorithm_package.algorithm_config_hash
                        ),
                    }
                    pre_method_cfg = _scale_cfg(scale, _scenario_cfg(scenario, _flatten_cfg(cfg_source_yaml, seed=int(seed), monitor_snap_enabled=monitor_snap)), l_benchmark_mode=l_benchmark_mode)
                    pre_method_cfg = _apply_experiment_sensitivity_overrides(
                        pre_method_cfg,
                        scenario=scenario,
                        comm_coverage=float(args.comm_blackout_emergency_coverage),
                        blockage_asymptote=float(args.blockage_asymptote),
                    )
                    cfg = replace(cfg, enable_rth_mask=bool(enable_rth_mask))
                    alns_solution_mode_override = str(getattr(args, "alns_solution_mode", "")).strip().lower()
                    alns_sequence_length_override = int(getattr(args, "alns_sequence_length", 0) or 0)
                    alns_operator_pool_override = str(getattr(args, "alns_operator_pool", "")).strip().lower()
                    adaptive_horizon_mode_override = str(getattr(args, "adaptive_horizon_mode", "")).strip().lower()
                    local_search_mode_override = str(getattr(args, "local_search_mode", "")).strip().lower()
                    local_search_max_moves_override = int(getattr(args, "local_search_max_moves_per_iteration", -1))
                    local_search_max_checks_override = int(getattr(args, "local_search_max_exact_checks_per_iteration", -1))
                    local_search_max_time_override = int(getattr(args, "local_search_max_time_ms_per_iteration", -1))
                    local_search_disabled_moves_override = _parse_csv_list(getattr(args, "local_search_disabled_moves", ""))
                    physical_environment_version_override = str(getattr(args, "physical_environment_version", "")).strip().lower()
                    physical_environment_safety_protocol_override = str(getattr(args, "physical_environment_safety_protocol", "")).strip().lower()
                    candidate_ranker_mode_override = str(getattr(args, "candidate_ranker_mode", "")).strip().lower()
                    candidate_ranker_pool_size_override = int(getattr(args, "candidate_ranker_pool_size", 0) or 0)
                    candidate_ranker_exact_check_budget_override = int(getattr(args, "candidate_ranker_exact_check_budget", 0) or 0)
                    candidate_ranker_exploration_count_override = int(getattr(args, "candidate_ranker_exploration_count", -1))
                    alns_overrides: Dict[str, Any] = {}
                    if alns_solution_mode_override:
                        alns_overrides["alns_solution_mode"] = alns_solution_mode_override
                    if alns_sequence_length_override > 0:
                        alns_overrides["alns_sequence_length"] = int(alns_sequence_length_override)
                    if alns_operator_pool_override:
                        alns_overrides["alns_operator_pool"] = alns_operator_pool_override
                    if adaptive_horizon_mode_override:
                        alns_overrides["adaptive_horizon_mode"] = adaptive_horizon_mode_override
                    if local_search_mode_override:
                        alns_overrides["local_search_mode"] = local_search_mode_override
                    if local_search_max_moves_override >= 0:
                        alns_overrides["local_search_max_moves_per_iteration"] = int(local_search_max_moves_override)
                    if local_search_max_checks_override >= 0:
                        alns_overrides["local_search_max_exact_checks_per_iteration"] = int(local_search_max_checks_override)
                    if local_search_max_time_override >= 0:
                        alns_overrides["local_search_max_time_ms_per_iteration"] = int(local_search_max_time_override)
                    if local_search_disabled_moves_override:
                        alns_overrides["local_search_disabled_moves"] = tuple(local_search_disabled_moves_override)
                    if physical_environment_version_override:
                        alns_overrides["physical_environment_version"] = physical_environment_version_override
                    if physical_environment_safety_protocol_override:
                        alns_overrides["physical_environment_safety_protocol"] = physical_environment_safety_protocol_override
                    if candidate_ranker_mode_override:
                        alns_overrides["candidate_ranker_mode"] = candidate_ranker_mode_override
                    if candidate_ranker_pool_size_override > 0:
                        alns_overrides["candidate_ranker_pool_size"] = int(candidate_ranker_pool_size_override)
                    if candidate_ranker_exact_check_budget_override > 0:
                        alns_overrides["candidate_ranker_exact_check_budget"] = int(candidate_ranker_exact_check_budget_override)
                    if candidate_ranker_exploration_count_override >= 0:
                        alns_overrides["candidate_ranker_exploration_count"] = int(candidate_ranker_exploration_count_override)
                    if bool(getattr(args, "enable_critical_recovery_repair", False)):
                        alns_overrides["alns_critical_recovery_repair_enabled"] = True
                    if bool(getattr(args, "enable_critical_support_rebind", False)):
                        alns_overrides["alns_critical_support_rebind_enabled"] = True
                    if bool(getattr(args, "enable_support_rebind_margin_aware", False)):
                        alns_overrides["alns_support_rebind_margin_aware_enabled"] = True
                    if bool(getattr(args, "enable_support_rebind_anchor_ranking", False)):
                        alns_overrides["alns_support_rebind_anchor_ranking_enabled"] = True
                    if bool(getattr(args, "enable_support_rebind_failed_binding_avoidance", False)):
                        alns_overrides["alns_support_rebind_failed_binding_avoidance_enabled"] = True
                    if str(getattr(args, "support_rebind_failed_binding_penalty", "")).strip():
                        alns_overrides["alns_support_rebind_failed_binding_penalty"] = str(
                            getattr(args, "support_rebind_failed_binding_penalty", "")
                        ).strip().lower()
                    if bool(getattr(args, "enable_support_rebind_critical_first_ordering", False)):
                        alns_overrides["alns_support_rebind_critical_first_ordering_enabled"] = True
                    if bool(getattr(args, "enable_safe_uav_dispatch_guard", False)):
                        alns_overrides["alns_support_rebind_safe_uav_guard_enabled"] = True
                    if int(getattr(args, "support_rebind_margin_top_k", 0) or 0) > 0:
                        alns_overrides["alns_support_rebind_margin_top_k"] = int(getattr(args, "support_rebind_margin_top_k"))
                    if float(getattr(args, "support_rebind_anchor_search_radius_factor", 0.0) or 0.0) > 0.0:
                        alns_overrides["alns_support_rebind_anchor_search_radius_factor"] = float(
                            getattr(args, "support_rebind_anchor_search_radius_factor")
                        )
                    if bool(getattr(args, "enable_lc_critical_recovery_path", False)):
                        alns_overrides["alns_lc_critical_recovery_path_enabled"] = True
                    if bool(getattr(args, "enable_assigned_critical_reconstruct", False)):
                        alns_overrides["alns_assigned_critical_reconstruct_enabled"] = True
                    if bool(getattr(args, "enable_support_reposition_shadow", False)):
                        alns_overrides["alns_support_reposition_shadow_enabled"] = True
                    if alns_overrides:
                        cfg = replace(cfg, **alns_overrides)
                    if bool(getattr(args, "physical_freeze_fair_config", False)):
                        fair_overrides = {
                            field: getattr(pre_method_cfg, field)
                            for field in PHYSICAL_FREEZE_FAIR_CONFIG_FIELDS
                            if hasattr(pre_method_cfg, field)
                        }
                        if fair_overrides:
                            cfg = replace(cfg, **fair_overrides)
                    mkey = str(method).strip().lower()
                    if mkey not in method_effective_flags_by_method:
                        method_effective_flags_by_method[mkey] = _variant_effective_flags(cfg)
                    env = BaseHeteroDisasterEnv(
                        cfg, algorithm_profile=algorithm_package.profile
                    )
                    algorithm_package.bind_environment(env)
                    low = RuleBasedLowLevelPolicy(seed=int(seed))

                    ems: List[Dict[str, float]] = []
                    for ep in range(episodes_per_seed):
                        eval_seed = int(seed) * 10000 + int(ep)
                        env.current_method = str(method).strip().lower()
                        env.current_episode_index = int(ep)
                        m = _run_episode(
                            env,
                            planner,
                            low,
                            eval_seed=eval_seed,
                            lightweight_metrics_only=bool(args.lightweight_metrics_only),
                            audit_flags=lightweight_audit_flags,
                        )
                        episode_offenders = m.pop("_hard_event_offender_rows", [])
                        if isinstance(episode_offenders, list):
                            for off in episode_offenders:
                                if not isinstance(off, dict):
                                    continue
                                hard_event_offender_rows.append(
                                    {
                                        "scenario": f"{str(scale).upper()}-{str(scenario).upper()}",
                                        "seed": int(seed),
                                        "method": str(method).strip().lower(),
                                        "episode": int(ep),
                                        "agent_id": str(off.get("agent_id", "")),
                                        "task_id": str(off.get("task_id", "")),
                                        "hard_event_reason": str(off.get("hard_event_reason", "unknown")),
                                        "count": float(off.get("count", 0.0)),
                                        "first_step": float(off.get("first_step", 0.0)),
                                        "last_step": float(off.get("last_step", 0.0)),
                                        "no_goal_change_count": float(off.get("no_goal_change_count", 0.0)),
                                        "goal_change_count": float(off.get("goal_change_count", 0.0)),
                                        "launch_count_after_event": float(off.get("launch_count_after_event", 0.0)),
                                        "completion_count_after_event": float(off.get("completion_count_after_event", 0.0)),
                                        "reject_count_after_event": float(off.get("reject_count_after_event", 0.0)),
                                        "goal_switch_after_event": float(off.get("goal_switch_after_event", 0.0)),
                                        "current_goal_type": str(off.get("current_goal_type", "")),
                                        "proposed_goal_type": str(off.get("proposed_goal_type", "")),
                                        "task_status": str(off.get("task_status", "")),
                                        "distance_to_goal_mean": float(off.get("distance_to_goal_mean", float("nan"))),
                                        "battery_mean": float(off.get("battery_mean", float("nan"))),
                                    }
                                )
                        episode_reason_rows = m.pop("_hard_event_reason_rows", [])
                        if isinstance(episode_reason_rows, list):
                            for rr in episode_reason_rows:
                                if not isinstance(rr, dict):
                                    continue
                                hard_event_reason_rows.append(
                                    {
                                        "scenario": f"{str(scale).upper()}-{str(scenario).upper()}",
                                        "seed": int(seed),
                                        "method": str(method).strip().lower(),
                                        "episode": int(ep),
                                        "hard_event_reason": str(rr.get("hard_event_reason", "unknown")),
                                        "total_refresh_count": float(rr.get("total_refresh_count", 0.0)),
                                        "no_goal_change_count": float(rr.get("no_goal_change_count", 0.0)),
                                        "goal_change_count": float(rr.get("goal_change_count", 0.0)),
                                        "no_goal_change_ratio": float(rr.get("no_goal_change_ratio", 0.0)),
                                        "followed_by_launch_count": float(rr.get("followed_by_launch_count", 0.0)),
                                        "followed_by_completion_count": float(rr.get("followed_by_completion_count", 0.0)),
                                        "followed_by_reject_count": float(rr.get("followed_by_reject_count", 0.0)),
                                        "followed_by_stall_count": float(rr.get("followed_by_stall_count", 0.0)),
                                    }
                                )
                        episode_task_rows = m.pop("_task_outcome_rows", [])
                        if isinstance(episode_task_rows, list):
                            for tr in episode_task_rows:
                                if not isinstance(tr, dict):
                                    continue
                                task_outcome_rows.append(
                                    {
                                        "scenario": f"{str(scale).upper()}-{str(scenario).upper()}",
                                        "seed": int(seed),
                                        "method": str(method).strip().lower(),
                                        "episode": int(ep),
                                        **tr,
                                    }
                                )
                        episode_routine_rows = m.pop("_routine_trace_rows", [])
                        if isinstance(episode_routine_rows, list):
                            for rr in episode_routine_rows:
                                if not isinstance(rr, dict):
                                    continue
                                routine_trace_rows.append(
                                    {
                                        "scenario": f"{str(scale).upper()}-{str(scenario).upper()}",
                                        "seed": int(seed),
                                        "method": str(method).strip().lower(),
                                        "episode": int(ep),
                                        **rr,
                                    }
                                )
                        episode_truck_rows = m.pop("_truck_routine_summary_rows", [])
                        if isinstance(episode_truck_rows, list):
                            for tr in episode_truck_rows:
                                if not isinstance(tr, dict):
                                    continue
                                truck_routine_summary_rows.append(
                                    {
                                        "scenario": f"{str(scale).upper()}-{str(scenario).upper()}",
                                        "seed": int(seed),
                                        "method": str(method).strip().lower(),
                                        "episode": int(ep),
                                        **tr,
                                    }
                                )
                        episode_support_rows = m.pop("_support_trace_rows", [])
                        if isinstance(episode_support_rows, list):
                            for sr in episode_support_rows:
                                if not isinstance(sr, dict):
                                    continue
                                support_trace_rows.append(
                                    {
                                        "scenario": f"{str(scale).upper()}-{str(scenario).upper()}",
                                        "seed": int(seed),
                                        "method": str(method).strip().lower(),
                                        "episode": int(ep),
                                        **sr,
                                    }
                                )
                        episode_uav_rows = m.pop("_uav_execution_rows", [])
                        if isinstance(episode_uav_rows, list):
                            for ur in episode_uav_rows:
                                if not isinstance(ur, dict):
                                    continue
                                uav_execution_rows.append(
                                    {
                                        "scenario": f"{str(scale).upper()}-{str(scenario).upper()}",
                                        "seed": int(seed),
                                        "method": str(method).strip().lower(),
                                        "episode": int(ep),
                                        **ur,
                                    }
                                )
                        episode_agent_step_rows = m.pop("_agent_step_rows", [])
                        if isinstance(episode_agent_step_rows, list):
                            for ar in episode_agent_step_rows:
                                if not isinstance(ar, dict):
                                    continue
                                agent_step_rows.append(
                                    {
                                        "scenario": f"{str(scale).upper()}-{str(scenario).upper()}",
                                        "seed": int(seed),
                                        "method": str(method).strip().lower(),
                                        "episode": int(ep),
                                        **ar,
                                    }
                                )
                        episode_task_proximity_rows = m.pop("_task_proximity_decision_rows", [])
                        if isinstance(episode_task_proximity_rows, list):
                            for pr in episode_task_proximity_rows:
                                if not isinstance(pr, dict):
                                    continue
                                task_proximity_decision_rows.append(
                                    {
                                        "scenario": f"{str(scale).upper()}-{str(scenario).upper()}",
                                        "seed": int(seed),
                                        "method": str(method).strip().lower(),
                                        "episode": int(ep),
                                        **pr,
                                    }
                                )
                        episode_tc_override_rows = m.pop("_tc_override_trace_rows", [])
                        if isinstance(episode_tc_override_rows, list):
                            for tor in episode_tc_override_rows:
                                if not isinstance(tor, dict):
                                    continue
                                tc_override_trace_rows.append(
                                    {
                                        "scenario": f"{str(scale).upper()}-{str(scenario).upper()}",
                                        "seed": int(seed),
                                        "method": str(method).strip().lower(),
                                        "episode": int(ep),
                                        **tor,
                                    }
                                )
                        episode_v2_sortie_rows = m.pop("_v2_sortie_lifecycle_rows", [])
                        if isinstance(episode_v2_sortie_rows, list):
                            for vr in episode_v2_sortie_rows:
                                if not isinstance(vr, dict):
                                    continue
                                v2_sortie_lifecycle_rows.append(
                                    {
                                        "scenario": f"{str(scale).upper()}-{str(scenario).upper()}",
                                        "seed": int(seed),
                                        "method": str(method).strip().lower(),
                                        "episode": int(ep),
                                        **vr,
                                    }
                                )
                        episode_v2_support_rows = m.pop("_v2_support_lifecycle_rows", [])
                        if isinstance(episode_v2_support_rows, list):
                            for vr in episode_v2_support_rows:
                                if not isinstance(vr, dict):
                                    continue
                                v2_support_lifecycle_rows.append(
                                    {
                                        "scenario": f"{str(scale).upper()}-{str(scenario).upper()}",
                                        "seed": int(seed),
                                        "method": str(method).strip().lower(),
                                        "episode": int(ep),
                                        **vr,
                                    }
                                )
                        episode_v2_safety_rows = m.pop("_v2_safety_lifecycle_rows", [])
                        if isinstance(episode_v2_safety_rows, list):
                            for vr in episode_v2_safety_rows:
                                if not isinstance(vr, dict):
                                    continue
                                v2_safety_lifecycle_rows.append(
                                    {
                                        "scenario": f"{str(scale).upper()}-{str(scenario).upper()}",
                                        "seed": int(seed),
                                        "method": str(method).strip().lower(),
                                        "episode": int(ep),
                                        **vr,
                                    }
                                )
                        episode_v2_passenger_rows = m.pop("_v2_support_passenger_trace_rows", [])
                        if isinstance(episode_v2_passenger_rows, list):
                            for vr in episode_v2_passenger_rows:
                                if not isinstance(vr, dict):
                                    continue
                                v2_support_passenger_trace_rows.append(
                                    {
                                        "scenario": f"{str(scale).upper()}-{str(scenario).upper()}",
                                        "seed": int(seed),
                                        "method": str(method).strip().lower(),
                                        "episode": int(ep),
                                        **vr,
                                    }
                                )
                        episode_v2_quality_rows = m.pop("_v2_support_quality_rows", [])
                        if isinstance(episode_v2_quality_rows, list):
                            for vr in episode_v2_quality_rows:
                                if not isinstance(vr, dict):
                                    continue
                                v2_support_quality_rows.append(
                                    {
                                        "scenario": f"{str(scale).upper()}-{str(scenario).upper()}",
                                        "seed": int(seed),
                                        "method": str(method).strip().lower(),
                                        "episode": int(ep),
                                        **vr,
                                    }
                                )
                        episode_switch_rows = m.pop("_switch_decision_ledger_rows", [])
                        if isinstance(episode_switch_rows, list):
                            for sr in episode_switch_rows:
                                if not isinstance(sr, dict):
                                    continue
                                switch_decision_ledger_rows.append(
                                    {
                                        "scenario": f"{str(scale).upper()}-{str(scenario).upper()}",
                                        "method": str(method).strip().lower(),
                                        "seed": int(seed),
                                        "episode": int(ep),
                                        **sr,
                                    }
                                )
                        episode_invalid_rows = m.pop("_invalid_action_records", [])
                        if isinstance(episode_invalid_rows, list):
                            for ir in episode_invalid_rows:
                                if not isinstance(ir, dict):
                                    continue
                                row_ir = dict(ir)
                                row_ir["scenario"] = f"{str(scale).upper()}-{str(scenario).upper()}"
                                row_ir["method"] = str(method).strip().lower()
                                row_ir["seed"] = int(seed)
                                row_ir["episode_index"] = int(ep)
                                invalid_action_rows.append(row_ir)
                        episode_shadow_rows = m.pop("_objective_shadow_records", [])
                        if isinstance(episode_shadow_rows, list):
                            for sr in episode_shadow_rows:
                                if not isinstance(sr, dict):
                                    continue
                                row_sr = dict(sr)
                                row_sr["scenario"] = f"{str(scale).upper()}-{str(scenario).upper()}"
                                row_sr["method"] = str(method).strip().lower()
                                row_sr["seed"] = int(seed)
                                row_sr["episode"] = int(ep)
                                objective_shadow_rows.append(row_sr)
                        episode_k2_rows = m.pop("_k2_sequence_records", [])
                        if isinstance(episode_k2_rows, list):
                            for kr in episode_k2_rows:
                                if not isinstance(kr, dict):
                                    continue
                                row_kr = dict(kr)
                                row_kr["scenario"] = f"{str(scale).upper()}-{str(scenario).upper()}"
                                row_kr["method"] = str(method).strip().lower()
                                row_kr["seed"] = int(seed)
                                row_kr["episode"] = int(ep)
                                k2_sequence_rows.append(row_kr)
                        episode_k2_runtime_rows = m.pop("_k2_runtime_sequence_records", [])
                        if isinstance(episode_k2_runtime_rows, list):
                            for rr in episode_k2_runtime_rows:
                                if not isinstance(rr, dict):
                                    continue
                                row_rr = dict(rr)
                                row_rr["scenario"] = f"{str(scale).upper()}-{str(scenario).upper()}"
                                row_rr["method"] = str(method).strip().lower()
                                row_rr["seed"] = int(seed)
                                row_rr["episode"] = int(ep)
                                k2_runtime_sequence_rows.append(row_rr)
                        episode_k2_sa_rows = m.pop("_k2_sa_delta_records", [])
                        if isinstance(episode_k2_sa_rows, list):
                            for sr in episode_k2_sa_rows:
                                if not isinstance(sr, dict):
                                    continue
                                row_sr = dict(sr)
                                row_sr["scenario"] = f"{str(scale).upper()}-{str(scenario).upper()}"
                                row_sr["method"] = str(method).strip().lower()
                                row_sr["seed"] = int(seed)
                                row_sr["episode"] = int(ep)
                                k2_sa_delta_rows.append(row_sr)
                        episode_canonical_rows = m.pop("_canonical_operator_records", [])
                        if isinstance(episode_canonical_rows, list):
                            for cr in episode_canonical_rows:
                                if not isinstance(cr, dict):
                                    continue
                                row_cr = dict(cr)
                                row_cr["scenario"] = f"{str(scale).upper()}-{str(scenario).upper()}"
                                row_cr["method"] = str(method).strip().lower()
                                row_cr["seed"] = int(seed)
                                row_cr["episode"] = int(ep)
                                canonical_operator_rows.append(row_cr)
                        for pop_key, target_rows in [
                            ("_support_execution_records", support_execution_rows),
                            ("_operator_weight_trajectory_records", operator_weight_rows),
                            ("_event_trigger_records", event_trigger_rows),
                            ("_sa_calibration_records", sa_calibration_rows),
                            ("_live_candidate_records", live_candidate_rows),
                            ("_ranker_runtime_records", ranker_runtime_rows),
                            ("_repair_candidate_pool_records", repair_candidate_pool_rows),
                            ("_adaptive_horizon_records", adaptive_horizon_rows),
                            ("_local_search_records", local_search_rows),
                        ]:
                            episode_extra_rows = m.pop(pop_key, [])
                            if isinstance(episode_extra_rows, list):
                                for er in episode_extra_rows:
                                    if not isinstance(er, dict):
                                        continue
                                    row_er = dict(er)
                                    row_er["scenario"] = f"{str(scale).upper()}-{str(scenario).upper()}"
                                    row_er["method"] = str(method).strip().lower()
                                    row_er["seed"] = int(seed)
                                    row_er["episode"] = int(ep)
                                    row_er["protocol"] = str(getattr(cfg, "physical_environment_safety_protocol", ""))
                                    target_rows.append(row_er)
                        ems.append(m)
                        episode_rows.append(
                            {
                                "scale": str(scale).upper(),
                                "scenario": str(scenario).upper(),
                                "model": str(method),
                                "train_seed": int(seed),
                                "test_suite": f"{str(scale).upper()}_{str(scenario).upper()}_matrix",
                                "episode": int(ep),
                            }
                        )
                        episode_rows[-1].update({
                            k: float(m.get(k, float("nan")))
                            for k in (
                                SEED_METRIC_FIELDS
                                + SUPPLY_METRIC_FIELDS
                                + SUPPLY_DERIVED_FIELDS
                                + UAV_SAFETY_METRIC_FIELDS
                                + UAV_STRICT_SAFETY_FIELDS
                                + ISLAND_SUPPORT_METRIC_FIELDS
                                + STAGE_C_METRIC_FIELDS
                                + TASK_OUTCOME_METRIC_FIELDS
                                + TASK_SEMANTIC_METRIC_FIELDS
                                + TIMECRITICAL_PRIORITY_METRIC_FIELDS
                                + SUPPORT_BIND_METRIC_FIELDS
                                + PLANNER_CONVERSION_METRIC_FIELDS
                                + MAINLINE_DIAG_FIELDS
                                + ROUTEA_MAPUPDATE_DIAG_FIELDS
                                + BLOCKAGE_V2_METRIC_FIELDS
                                + PHYSICAL_V2_METRIC_FIELDS
                            )
                        })
                        episode_rows[-1].update({k: str(m.get(k, "")) for k in PHYSICAL_V2_STRING_FIELDS})
                        episode_rows[-1].update(
                            {
                                "episode_reward_mean": float(m["episode_reward_mean"]),
                                "encoder_type": str(encoder_type),
                                "enable_rth_mask": bool(enable_rth_mask),
                                "blocked_edge_count": float(m["blocked_edge_count"]),
                                "comm_blackout_ratio": float(m["comm_blackout_ratio"]),
                                "wind_severity_p95_mps": float(m["wind_severity_p95_mps"]),
                                "rain_severity_p95_mmh": float(m["rain_severity_p95_mmh"]),
                                "triggered_replans": float(m["triggered_replans"]),
                                "event_replans_in_window": float(m["event_replans_in_window"]),
                                "event_budget_blocked": float(m["event_budget_blocked"]),
                                "monitor_snap_enabled": bool(monitor_snap),
                                "experiment_class": "ablation_only" if monitor_snap else "main",
                                "algorithm_id": str(algorithm_package.algorithm_id),
                                "planner_class": str(algorithm_package.planner_class),
                                "backend_family": str(algorithm_package.backend_family),
                                "public_environment_hash": str(algorithm_package.public_environment_hash),
                                "algorithm_config_hash": str(algorithm_package.algorithm_config_hash),
                                "er_hlns_capability": bool(
                                    algorithm_package.profile.has("er_hlns_route_plan")
                                ),
                            }
                        )
                        episode_rows[-1].update({k: str(m.get(k, "")) for k in REPRO_DIGEST_FIELDS})
                        episode_rows[-1].update({k: str(m.get(k, "")) for k in PHYSICAL_FREEZE_EXPORT_FIELDS})
                        episode_rows[-1].update(
                            _physical_freeze_export_aliases(
                                episode_rows[-1],
                                cfg=cfg,
                                scale=str(scale),
                                scenario=str(scenario),
                                method=str(method),
                                seed=int(seed),
                            )
                        )
                        episode_rows[-1].update({k: bool(v) for k, v in lightweight_audit_flags.items()})

                    row = {
                        "scale": str(scale).upper(),
                        "scenario": str(scenario).upper(),
                        "model": str(method),
                        "train_seed": int(seed),
                        "test_suite": f"{str(scale).upper()}_{str(scenario).upper()}_matrix",
                        **{
                            k: float(mean([x.get(k, float("nan")) for x in ems]))
                            for k in (
                                SEED_METRIC_FIELDS
                                + SUPPLY_METRIC_FIELDS
                                + SUPPLY_DERIVED_FIELDS
                                + UAV_SAFETY_METRIC_FIELDS
                                + UAV_STRICT_SAFETY_FIELDS
                                + ISLAND_SUPPORT_METRIC_FIELDS
                                + STAGE_C_METRIC_FIELDS
                                + TASK_OUTCOME_METRIC_FIELDS
                                + TASK_SEMANTIC_METRIC_FIELDS
                                + TIMECRITICAL_PRIORITY_METRIC_FIELDS
                                + SUPPORT_BIND_METRIC_FIELDS
                                + PLANNER_CONVERSION_METRIC_FIELDS
                                + MAINLINE_DIAG_FIELDS
                                + ROUTEA_MAPUPDATE_DIAG_FIELDS
                                + BLOCKAGE_V2_METRIC_FIELDS
                                + PHYSICAL_V2_METRIC_FIELDS
                            )
                        },
                        **{k: str(ems[-1].get(k, "")) for k in PHYSICAL_V2_STRING_FIELDS},
                        "episode_reward_mean": float(mean([x["episode_reward_mean"] for x in ems])),
                        "encoder_type": str(encoder_type),
                        "enable_rth_mask": bool(enable_rth_mask),
                        "blocked_edge_count": float(mean([x["blocked_edge_count"] for x in ems])),
                        "comm_blackout_ratio": float(mean([x["comm_blackout_ratio"] for x in ems])),
                        "wind_severity_p95_mps": float(mean([x["wind_severity_p95_mps"] for x in ems])),
                        "rain_severity_p95_mmh": float(mean([x["rain_severity_p95_mmh"] for x in ems])),
                        "triggered_replans": float(mean([x["triggered_replans"] for x in ems])),
                        "event_replans_in_window": float(mean([x["event_replans_in_window"] for x in ems])),
                        "event_budget_blocked": float(mean([x["event_budget_blocked"] for x in ems])),
                        "alns_solution_mode": str(getattr(cfg, "alns_solution_mode", "")),
                        "alns_sequence_length": float(getattr(cfg, "alns_sequence_length", 0)),
                        "alns_operator_pool": str(getattr(cfg, "alns_operator_pool", "")),
                        "alns_selection_mode": str(getattr(cfg, "alns_selection_mode", "")),
                        "monitor_snap_enabled": bool(monitor_snap),
                        "experiment_class": "ablation_only" if monitor_snap else "main",
                        "algorithm_id": str(algorithm_package.algorithm_id),
                        "planner_class": str(algorithm_package.planner_class),
                        "backend_family": str(algorithm_package.backend_family),
                        "public_environment_hash": str(algorithm_package.public_environment_hash),
                        "algorithm_config_hash": str(algorithm_package.algorithm_config_hash),
                        "er_hlns_capability": bool(
                            algorithm_package.profile.has("er_hlns_route_plan")
                        ),
                    }
                    row.update({k: bool(v) for k, v in lightweight_audit_flags.items()})
                    row.update(_aggregate_launch_battery_seed_metrics(ems))
                    row.update({k: str(ems[-1].get(k, "")) for k in REPRO_DIGEST_FIELDS})
                    row.update({k: str(ems[-1].get(k, "")) for k in PHYSICAL_FREEZE_EXPORT_FIELDS})
                    row.update(
                        _physical_freeze_export_aliases(
                            row,
                            cfg=cfg,
                            scale=str(scale),
                            scenario=str(scenario),
                            method=str(method),
                            seed=int(seed),
                        )
                    )
                    seed_rows.append(row)
                    print(row)


    episode_csv = run_dir / "episode_metrics.csv"
    seed_csv = run_dir / "seed_metrics.csv"

    ep_fields = _unique_field_order([
        "scale",
        "scenario",
        "model",
        "train_seed",
        "test_suite",
        "episode",
        *SEED_METRIC_FIELDS,
        *SUPPLY_METRIC_FIELDS,
        *SUPPLY_DERIVED_FIELDS,
        *UAV_SAFETY_METRIC_FIELDS,
        *UAV_STRICT_SAFETY_FIELDS,
        *ISLAND_SUPPORT_METRIC_FIELDS,
        *STAGE_C_METRIC_FIELDS,
        *TASK_OUTCOME_METRIC_FIELDS,
        *TASK_SEMANTIC_METRIC_FIELDS,
        *TIMECRITICAL_PRIORITY_METRIC_FIELDS,
        *SUPPORT_BIND_METRIC_FIELDS,
        *PLANNER_CONVERSION_METRIC_FIELDS,
        *MAINLINE_DIAG_FIELDS,
        *ROUTEA_MAPUPDATE_DIAG_FIELDS,
        *BLOCKAGE_V2_METRIC_FIELDS,
        *PHYSICAL_V2_STRING_FIELDS,
        *PHYSICAL_V2_METRIC_FIELDS,
        "episode_reward_mean",
        "encoder_type",
        "enable_rth_mask",
        "blocked_edge_count",
        "comm_blackout_ratio",
        "wind_severity_p95_mps",
        "rain_severity_p95_mmh",
        "triggered_replans",
        "event_replans_in_window",
        "event_budget_blocked",
        "alns_solution_mode",
        "alns_sequence_length",
        "alns_operator_pool",
        "alns_selection_mode",
        "monitor_snap_enabled",
        "experiment_class",
        "algorithm_id",
        "planner_class",
        "backend_family",
        "public_environment_hash",
        "algorithm_config_hash",
        "er_hlns_capability",
        *REPRO_DIGEST_FIELDS,
        *[field for field in PHYSICAL_FREEZE_EXPORT_FIELDS if field != "physical_environment_version" and field not in REPRO_DIGEST_FIELDS],
        *LIGHTWEIGHT_AUDIT_FLAG_FIELDS,
    ])
    with episode_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ep_fields)
        w.writeheader()
        for r in episode_rows:
            w.writerow(r)

    sd_fields = _unique_field_order([
        "scale",
        "scenario",
        "model",
        "train_seed",
        "test_suite",
        *SEED_METRIC_FIELDS,
        *SUPPLY_METRIC_FIELDS,
        *SUPPLY_DERIVED_FIELDS,
        *UAV_SAFETY_METRIC_FIELDS,
        *UAV_STRICT_SAFETY_FIELDS,
        *ISLAND_SUPPORT_METRIC_FIELDS,
        *STAGE_C_METRIC_FIELDS,
        *TASK_OUTCOME_METRIC_FIELDS,
        *TASK_SEMANTIC_METRIC_FIELDS,
        *TIMECRITICAL_PRIORITY_METRIC_FIELDS,
        *SUPPORT_BIND_METRIC_FIELDS,
        *PLANNER_CONVERSION_METRIC_FIELDS,
        *MAINLINE_DIAG_FIELDS,
        *ROUTEA_MAPUPDATE_DIAG_FIELDS,
        *BLOCKAGE_V2_METRIC_FIELDS,
        *PHYSICAL_V2_STRING_FIELDS,
        *PHYSICAL_V2_METRIC_FIELDS,
        "episode_reward_mean",
        "encoder_type",
        "enable_rth_mask",
        "blocked_edge_count",
        "comm_blackout_ratio",
        "wind_severity_p95_mps",
        "rain_severity_p95_mmh",
        "triggered_replans",
        "event_replans_in_window",
        "event_budget_blocked",
        "alns_solution_mode",
        "alns_sequence_length",
        "alns_operator_pool",
        "alns_selection_mode",
        "monitor_snap_enabled",
        "experiment_class",
        "algorithm_id",
        "planner_class",
        "backend_family",
        "public_environment_hash",
        "algorithm_config_hash",
        "er_hlns_capability",
        *REPRO_DIGEST_FIELDS,
        *[field for field in PHYSICAL_FREEZE_EXPORT_FIELDS if field != "physical_environment_version" and field not in REPRO_DIGEST_FIELDS],
        *LIGHTWEIGHT_AUDIT_FLAG_FIELDS,
    ])
    with seed_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sd_fields)
        w.writeheader()
        for r in seed_rows:
            w.writerow(r)

    def _episode_value(row: Dict[str, Any], *names: str, default: float = 0.0) -> float:
        for name in names:
            if name in row:
                try:
                    return float(row.get(name, default) or 0.0)
                except Exception:
                    return float(default)
        return float(default)

    single_seed_rows: List[Dict[str, Any]] = []
    runtime_rows: List[Dict[str, Any]] = []
    safety_rows: List[Dict[str, Any]] = []
    mode_rows: List[Dict[str, Any]] = []
    grouped_episode_rows: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in episode_rows:
        scenario_id = f"{str(r.get('scale', '')).upper()}-{str(r.get('scenario', '')).upper()}"
        grouped_episode_rows[(scenario_id, str(r.get("model", "")))].append(r)
    for (scenario_name, method_name), rows in sorted(grouped_episode_rows.items()):
        iterations = sum(_episode_value(r, "alns_iteration_count") for r in rows)
        repairs = sum(_episode_value(r, "alns_repair_attempt_count") for r in rows)
        feasible_repairs = sum(_episode_value(r, "alns_repair_feasible_count") for r in rows)
        accepted = sum(_episode_value(r, "alns_accepted_count") for r in rows)
        improved = sum(_episode_value(r, "alns_improvement_count") for r in rows)
        replan_count = sum(_episode_value(r, "alns_replan_count", "triggered_replans") for r in rows)
        invalid_count = sum(_episode_value(r, "invalid_action_count") for r in rows)
        hard_count = sum(_episode_value(r, "execution_hard_violation_count", "hard_constraint_violation_count") for r in rows)
        crash_count = sum(_episode_value(r, "crash_count") for r in rows)
        wall_times = [_episode_value(r, "alns_wall_clock_time_s", default=0.0) for r in rows]
        accepted_per_100 = float(100.0 * accepted / max(iterations, 1.0))
        improved_per_100 = float(100.0 * improved / max(iterations, 1.0))
        feasible_rate = float(feasible_repairs / max(repairs, 1.0))
        common = {
            "scenario": scenario_name,
            "method": method_name,
            "episodes": int(len(rows)),
            "total_iterations": float(iterations),
            "replan_count": float(replan_count),
            "operator_attempts": sum(_episode_value(r, "alns_operator_attempt_count") for r in rows),
            "objective_evaluations": sum(_episode_value(r, "alns_objective_evaluation_count") for r in rows),
            "feasibility_evaluations": sum(_episode_value(r, "alns_feasibility_evaluation_count") for r in rows),
            "feasible_rate": feasible_rate,
            "accepted_per_100_iterations": accepted_per_100,
            "improved_per_100_iterations": improved_per_100,
            "invalid_action_count": float(invalid_count),
            "execution_hard_violation_count": float(hard_count),
            "crash_count": float(crash_count),
            "runtime_p50_s": float(np.median(wall_times)) if wall_times else 0.0,
            "runtime_p95_s": float(np.quantile(wall_times, 0.95)) if wall_times else 0.0,
        }
        single_seed_rows.append(dict(common))
        runtime_rows.append(
            {
                "scenario": scenario_name,
                "method": method_name,
                "runtime_p50_s": common["runtime_p50_s"],
                "runtime_p95_s": common["runtime_p95_s"],
                "total_iterations": float(iterations),
                "replan_count": float(replan_count),
            }
        )
        safety_rows.append(
            {
                "scenario": scenario_name,
                "method": method_name,
                "invalid_action_count": float(invalid_count),
                "execution_hard_violation_count": float(hard_count),
                "crash_count": float(crash_count),
                "unknown_invalid_reason_count": sum(
                    1
                    for ir in invalid_action_rows
                    if str(ir.get("scenario", "")) == scenario_name
                    and str(ir.get("method", "")) == method_name
                    and str(ir.get("reason_code", "")) == "UNKNOWN_INVALID_REASON"
                ),
            }
        )
        mode_rows.append(
            {
                "scenario": scenario_name,
                "method": method_name,
                "k2_mode": str(rows[0].get("alns_solution_mode", "")) if rows else "",
                "operator_pool": str(rows[0].get("alns_operator_pool", "")) if rows else "",
                "selection_mode": str(rows[0].get("alns_selection_mode", "")) if rows else "",
                "iterations_per_replan": float(iterations / max(replan_count, 1.0)),
                "total_iterations": float(iterations),
                "replan_count": float(replan_count),
                "feasible_rate": feasible_rate,
            }
        )

    def _write_dynamic_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
        fieldnames = sorted({str(k) for row in rows for k in row.keys()})
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in rows:
                w.writerow(row)

    _write_dynamic_csv(run_dir / "single_seed_results.csv", single_seed_rows)
    _write_dynamic_csv(run_dir / "runtime_matrix.csv", runtime_rows)
    _write_dynamic_csv(run_dir / "safety_matrix.csv", safety_rows)
    _write_dynamic_csv(run_dir / "mode_parity.csv", mode_rows)

    invalid_ledger_paths = [run_dir / "INVALID_ACTION_LEDGER.jsonl"]
    refactor_dir = PROJECT_ROOT / "docs" / "refactor"
    if bool(getattr(args, "publish_refactor_invalid_ledger", False)):
        refactor_dir.mkdir(parents=True, exist_ok=True)
        invalid_ledger_paths.append(refactor_dir / "INVALID_ACTION_LEDGER.jsonl")
    for invalid_ledger_path in invalid_ledger_paths:
        with invalid_ledger_path.open("w", encoding="utf-8") as f:
            for r in invalid_action_rows:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    invalid_summary: Dict[Tuple[str, str, str, str], int] = defaultdict(int)
    for r in invalid_action_rows:
        key = (
            str(r.get("scenario", "")),
            str(r.get("method", "")),
            str(r.get("validation_layer", "")),
            str(r.get("reason_code", "")),
        )
        invalid_summary[key] += 1
    summary_fields = ["scenario", "method", "validation_layer", "reason_code", "count"]
    summary_rows = [
        {
            "scenario": scenario,
            "method": method,
            "validation_layer": layer,
            "reason_code": reason,
            "count": int(count),
        }
        for (scenario, method, layer, reason), count in sorted(invalid_summary.items())
    ]
    summary_paths = [run_dir / "INVALID_ACTION_SUMMARY.csv"]
    if bool(getattr(args, "publish_refactor_invalid_ledger", False)):
        summary_paths.append(refactor_dir / "INVALID_ACTION_SUMMARY.csv")
    for summary_path in summary_paths:
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=summary_fields)
            w.writeheader()
            for r in summary_rows:
                w.writerow(r)

    shadow_jsonl = run_dir / "objective_shadow_comparison.jsonl"
    with shadow_jsonl.open("w", encoding="utf-8") as f:
        for r in objective_shadow_rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    shadow_summary_counts: Dict[Tuple[str, str, str], Dict[str, int]] = defaultdict(lambda: {"comparisons": 0, "agreements": 0})
    for r in objective_shadow_rows:
        key = (
            str(r.get("scenario", "")),
            str(r.get("method", "")),
            str(r.get("disagreement_reason", "")),
        )
        shadow_summary_counts[key]["comparisons"] += 1
        if bool(r.get("ranking_agreement", False)):
            shadow_summary_counts[key]["agreements"] += 1
    shadow_summary_fields = [
        "scenario",
        "method",
        "disagreement_reason",
        "comparison_count",
        "agreement_count",
        "disagreement_count",
        "agreement_rate",
    ]
    shadow_summary_rows = []
    for (scenario, method, reason), counts in sorted(shadow_summary_counts.items()):
        comparisons = int(counts["comparisons"])
        agreements = int(counts["agreements"])
        shadow_summary_rows.append(
            {
                "scenario": scenario,
                "method": method,
                "disagreement_reason": reason,
                "comparison_count": comparisons,
                "agreement_count": agreements,
                "disagreement_count": int(comparisons - agreements),
                "agreement_rate": float(agreements / max(comparisons, 1)),
            }
        )
    with (run_dir / "objective_shadow_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=shadow_summary_fields)
        w.writeheader()
        for r in shadow_summary_rows:
            w.writerow(r)

    k2_ledger_path = run_dir / "k2_sequence_ledger.jsonl"
    with k2_ledger_path.open("w", encoding="utf-8") as f:
        for r in k2_sequence_rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    k2_objective_fields = [
        "scenario",
        "method",
        "seed",
        "episode",
        "step",
        "label",
        "solution_digest",
        "total_cost",
        "feasible",
        "nonempty_tail_count",
        "average_sequence_length",
        "second_task_travel_cost",
        "second_task_energy_cost",
        "second_task_lifeline_cost",
        "reason_codes",
    ]
    with (run_dir / "k2_objective_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=k2_objective_fields)
        w.writeheader()
        for r in k2_sequence_rows:
            w.writerow({k: r.get(k, "") for k in k2_objective_fields})
    k2_feasibility_counts: Dict[Tuple[str, str, str], int] = defaultdict(int)
    for r in k2_sequence_rows:
        reason_text = str(r.get("reason_codes", ""))
        reasons = [x for x in reason_text.split("|") if x] or ["FEASIBLE" if bool(r.get("feasible", False)) else "UNKNOWN"]
        for reason in reasons:
            k2_feasibility_counts[(str(r.get("scenario", "")), str(r.get("method", "")), str(reason))] += 1
    k2_feasibility_fields = ["scenario", "method", "reason_code", "count"]
    with (run_dir / "k2_feasibility_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=k2_feasibility_fields)
        w.writeheader()
        for (scenario_name, method_name, reason), count in sorted(k2_feasibility_counts.items()):
            w.writerow(
                {
                    "scenario": scenario_name,
                    "method": method_name,
                    "reason_code": reason,
                    "count": int(count),
                }
            )
    with (run_dir / "k2_runtime_sequence_ledger.jsonl").open("w", encoding="utf-8") as f:
        for r in k2_runtime_sequence_rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    runtime_summary_counts: Dict[Tuple[str, str, str, str], int] = defaultdict(int)
    for r in k2_runtime_sequence_rows:
        reasons = [x for x in str(r.get("reason_codes", "")).split("|") if x] or [str(r.get("validation_result", "")) or "NONE"]
        for reason in reasons:
            runtime_summary_counts[
                (
                    str(r.get("scenario", "")),
                    str(r.get("method", "")),
                    str(r.get("event_type", "")),
                    str(reason),
                )
            ] += 1
    with (run_dir / "k2_runtime_sequence_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["scenario", "method", "event_type", "reason_code", "count"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (scenario_name, method_name, event_type, reason_code), count in sorted(runtime_summary_counts.items()):
            w.writerow(
                {
                    "scenario": scenario_name,
                    "method": method_name,
                    "event_type": event_type,
                    "reason_code": reason_code,
                    "count": int(count),
                }
            )
    with (run_dir / "k2_roll_forward_ledger.jsonl").open("w", encoding="utf-8") as f:
        for r in k2_runtime_sequence_rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")

    with (run_dir / "k2_sa_delta_ledger.jsonl").open("w", encoding="utf-8") as f:
        for r in k2_sa_delta_rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    with (run_dir / "large_map_candidate_ledger.jsonl").open("w", encoding="utf-8") as f:
        for r in k2_sa_delta_rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    if k2_sa_delta_rows:
        delta_values = sorted(float(r.get("delta_cost", 0.0)) for r in k2_sa_delta_rows)
        worsen_probs = sorted(
            float(r.get("acceptance_probability", 0.0))
            for r in k2_sa_delta_rows
            if float(r.get("delta_cost", 0.0)) > 0.0
        )
        feasible_count = sum(1 for r in k2_sa_delta_rows if bool(r.get("candidate_feasible", False)))
        improving_count = sum(1 for r in k2_sa_delta_rows if bool(r.get("improved", False)))
        equal_count = sum(1 for r in k2_sa_delta_rows if bool(r.get("equal_candidate", False)))
        worsening_count = sum(1 for r in k2_sa_delta_rows if float(r.get("delta_cost", 0.0)) > 0.0)
        accepted_count = sum(1 for r in k2_sa_delta_rows if bool(r.get("accepted", False)))
        root_cause = "OTHER"
        if feasible_count <= 0:
            root_cause = "FEASIBILITY_FILTER_DOMINANT"
        elif improving_count <= 0:
            root_cause = "NO_IMPROVING_CANDIDATES"
        elif worsening_count > 0 and float(np.median(worsen_probs) if worsen_probs else 0.0) < 0.01:
            root_cause = "TEMPERATURE_TOO_LOW"
        elif worsening_count > 0 and float(np.median(delta_values)) > 1.0:
            root_cause = "DELTA_SCALE_TOO_LARGE"
        elif equal_count >= max(len(k2_sa_delta_rows) // 2, 1):
            root_cause = "CANDIDATES_EQUAL_CURRENT"
        elif accepted_count <= 0:
            root_cause = "OPERATOR_DIVERSITY_LIMITED"
        def quant(p: float) -> float:
            return float(np.quantile(delta_values, p)) if delta_values else 0.0

        temp_values = [float(r.get("temperature", 0.0)) for r in k2_sa_delta_rows]
        with (run_dir / "k2_sa_delta_summary.csv").open("w", newline="", encoding="utf-8") as f:
            fields = ["metric", "value"]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            rows = [
                ("candidate_count", len(k2_sa_delta_rows)),
                ("feasible_candidate_count", feasible_count),
                ("improving_candidate_count", improving_count),
                ("equal_candidate_count", equal_count),
                ("worsening_candidate_count", worsening_count),
                ("delta_min", quant(0.0)),
                ("delta_q25", quant(0.25)),
                ("delta_median", quant(0.5)),
                ("delta_q75", quant(0.75)),
                ("delta_max", quant(1.0)),
                ("temperature_initial", temp_values[0] if temp_values else 0.0),
                ("temperature_final", temp_values[-1] if temp_values else 0.0),
                ("worsening_acceptance_probability_min", min(worsen_probs) if worsen_probs else 0.0),
                ("worsening_acceptance_probability_median", float(np.median(worsen_probs)) if worsen_probs else 0.0),
                ("worsening_acceptance_probability_max", max(worsen_probs) if worsen_probs else 0.0),
                ("accepted_count", accepted_count),
                ("root_cause_of_accepted_zero", root_cause),
            ]
            for metric, value in rows:
                w.writerow({"metric": metric, "value": value})
    with (run_dir / "sa_delta_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["metric", "value"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        if k2_sa_delta_rows:
            accepted_count = sum(1 for r in k2_sa_delta_rows if bool(r.get("accepted", False)))
            w.writerow({"metric": "candidate_count", "value": len(k2_sa_delta_rows)})
            w.writerow({"metric": "accepted_count", "value": accepted_count})
            w.writerow({"metric": "temperature_initial", "value": float(k2_sa_delta_rows[0].get("temperature", 0.0))})
            w.writerow({"metric": "temperature_final", "value": float(k2_sa_delta_rows[-1].get("temperature", 0.0))})
        else:
            w.writerow({"metric": "candidate_count", "value": 0})

    with (run_dir / "support_execution_ledger.jsonl").open("w", encoding="utf-8") as f:
        for r in support_execution_rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    support_summary: Dict[Tuple[str, str, str], int] = defaultdict(int)
    for r in support_execution_rows:
        support_summary[(str(r.get("scenario", "")), str(r.get("method", "")), str(r.get("status", "")))] += 1
    with (run_dir / "support_execution_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["scenario", "method", "state", "count"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (scenario_name, method_name, state), count in sorted(support_summary.items()):
            w.writerow({"scenario": scenario_name, "method": method_name, "state": state, "count": int(count)})
    delta_summary: Dict[Tuple[str, str, str], int] = defaultdict(int)
    for r in k2_sa_delta_rows:
        delta_summary[(str(r.get("scenario", "")), str(r.get("method", "")), str(r.get("rejection_reason", "")))] += 1
    with (run_dir / "large_map_delta_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["scenario", "method", "category", "count"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (scenario_name, method_name, category), count in sorted(delta_summary.items()):
            w.writerow({"scenario": scenario_name, "method": method_name, "category": category, "count": int(count)})
    _write_dynamic_csv(run_dir / "l_scale_mode_parity.csv", [r for r in mode_rows if str(r.get("scenario", "")).upper().startswith("L")])
    _write_dynamic_csv(run_dir / "l_scale_safety_matrix.csv", [r for r in safety_rows if str(r.get("scenario", "")).upper().startswith("L")])
    with (run_dir / "operator_weight_trajectory.jsonl").open("w", encoding="utf-8") as f:
        for r in operator_weight_rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    with (run_dir / "event_trigger_ledger.jsonl").open("w", encoding="utf-8") as f:
        for r in event_trigger_rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    with (run_dir / "sa_calibration_ledger.jsonl").open("w", encoding="utf-8") as f:
        for r in sa_calibration_rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    with (run_dir / "live_candidate_ledger.jsonl").open("w", encoding="utf-8") as f:
        for r in live_candidate_rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    with (run_dir / "ranker_runtime_ledger.jsonl").open("w", encoding="utf-8") as f:
        for r in ranker_runtime_rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    with (run_dir / "repair_candidate_pool_ledger.jsonl").open("w", encoding="utf-8") as f:
        for r in repair_candidate_pool_rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    with (run_dir / "adaptive_horizon_ledger.jsonl").open("w", encoding="utf-8") as f:
        for r in adaptive_horizon_rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    with (run_dir / "local_search_ledger.jsonl").open("w", encoding="utf-8") as f:
        for r in local_search_rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    _write_dynamic_csv(run_dir / "live_candidate_dataset.csv", live_candidate_rows)
    _write_dynamic_csv(run_dir / "ranker_runtime_summary.csv", ranker_runtime_rows)
    _write_dynamic_csv(run_dir / "repair_candidate_pool_summary.csv", repair_candidate_pool_rows)
    _write_dynamic_csv(run_dir / "adaptive_horizon_summary.csv", adaptive_horizon_rows)
    _write_dynamic_csv(run_dir / "local_search_summary.csv", local_search_rows)

    canonical_ledger_path = run_dir / "canonical_operator_ledger.jsonl"
    with canonical_ledger_path.open("w", encoding="utf-8") as f:
        for r in canonical_operator_rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    summary_map: Dict[Tuple[str, str, str, str], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    failure_map: Dict[Tuple[str, str, str, str, str], int] = defaultdict(int)
    for r in canonical_operator_rows:
        phase = str(r.get("phase", ""))
        operator = str(r.get("destroy_operator", "")) if phase == "destroy" else str(r.get("repair_operator", ""))
        key = (str(r.get("scenario", "")), str(r.get("method", "")), phase, operator)
        summary_map[key]["attempts"] += float(r.get("attempts", 0) or 0)
        summary_map[key]["feasible_candidates"] += float(r.get("feasible_candidates", 0) or 0)
        summary_map[key]["infeasible_candidates"] += float(r.get("infeasible_candidates", 0) or 0)
        summary_map[key]["accepted"] += float(1 if bool(r.get("accepted", False)) else 0)
        summary_map[key]["improved"] += float(1 if bool(r.get("improved", False)) else 0)
        failure_counts = r.get("failure_reason_counts", {})
        if isinstance(failure_counts, dict):
            for reason, count in failure_counts.items():
                failure_map[(key[0], key[1], phase, operator, str(reason))] += int(float(count or 0))
        reason_values = r.get("reason_codes", [])
        if isinstance(reason_values, list):
            for reason in reason_values:
                if str(reason) and str(reason) not in {
                    "RANDOM_REMOVAL",
                    "WORST_COST_REMOVAL",
                    "RELATED_REMOVAL",
                    "SEQUENCE_SEGMENT_REMOVAL",
                }:
                    failure_map[(key[0], key[1], phase, operator, str(reason))] += 1
    with (run_dir / "canonical_operator_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "scenario",
            "method",
            "phase",
            "operator",
            "attempts",
            "feasible_candidates",
            "infeasible_candidates",
            "feasibility_rate",
            "accepted",
            "improved",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for key, vals in sorted(summary_map.items()):
            attempts = float(vals.get("attempts", 0.0))
            feasible = float(vals.get("feasible_candidates", 0.0))
            infeasible = float(vals.get("infeasible_candidates", 0.0))
            w.writerow(
                {
                    "scenario": key[0],
                    "method": key[1],
                    "phase": key[2],
                    "operator": key[3],
                    "attempts": attempts,
                    "feasible_candidates": feasible,
                    "infeasible_candidates": infeasible,
                    "feasibility_rate": float(feasible / max(feasible + infeasible, 1.0)),
                    "accepted": float(vals.get("accepted", 0.0)),
                    "improved": float(vals.get("improved", 0.0)),
                }
            )
    with (run_dir / "canonical_repair_failure_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["scenario", "method", "phase", "operator", "reason_code", "count"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for key, count in sorted(failure_map.items()):
            w.writerow(
                {
                    "scenario": key[0],
                    "method": key[1],
                    "phase": key[2],
                    "operator": key[3],
                    "reason_code": key[4],
                    "count": int(count),
                }
            )
    er_destroy_names = {
        "road_disruption_removal",
        "critical_task_reassignment_removal",
        "support_conflict_removal",
        "synchronization_risk_removal",
    }
    er_repair_names = {
        "critical_first_insertion",
        "risk_aware_insertion",
        "synchronized_insertion",
        "feasibility_restoration_insertion",
    }
    er_operator_rows = [
        r
        for r in canonical_operator_rows
        if (
            str(r.get("phase", "")) == "destroy"
            and str(r.get("destroy_operator", "")) in er_destroy_names
        )
        or (
            str(r.get("phase", "")) == "repair"
            and str(r.get("repair_operator", "")) in er_repair_names
        )
    ]
    with (run_dir / "er_operator_ledger.jsonl").open("w", encoding="utf-8") as f:
        for r in er_operator_rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    er_summary_map: Dict[Tuple[str, str, str, str], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    er_failure_map: Dict[Tuple[str, str, str, str, str], int] = defaultdict(int)
    support_coordination_rows: List[Dict[str, Any]] = []
    for r in er_operator_rows:
        phase = str(r.get("phase", ""))
        operator = str(r.get("destroy_operator", "")) if phase == "destroy" else str(r.get("repair_operator", ""))
        key = (str(r.get("scenario", "")), str(r.get("method", "")), phase, operator)
        er_summary_map[key]["attempts"] += float(r.get("attempts", 0) or 0)
        er_summary_map[key]["feasible_candidates"] += float(r.get("feasible_candidates", 0) or 0)
        er_summary_map[key]["infeasible_candidates"] += float(r.get("infeasible_candidates", 0) or 0)
        er_summary_map[key]["accepted"] += float(1 if bool(r.get("accepted", False)) else 0)
        er_summary_map[key]["improved"] += float(1 if bool(r.get("improved", False)) else 0)
        failure_counts = r.get("failure_reason_counts", {})
        if isinstance(failure_counts, dict):
            for reason, count in failure_counts.items():
                er_failure_map[(key[0], key[1], phase, operator, str(reason))] += int(float(count or 0))
        support_rows = r.get("support_coordination", [])
        if isinstance(support_rows, list):
            for sr in support_rows:
                if isinstance(sr, dict):
                    row_sr = dict(sr)
                    row_sr["scenario"] = key[0]
                    row_sr["method"] = key[1]
                    row_sr["operator"] = operator
                    row_sr["seed"] = r.get("seed", "")
                    row_sr["episode"] = r.get("episode", "")
                    row_sr["step"] = r.get("step", "")
                    support_coordination_rows.append(row_sr)
    with (run_dir / "er_operator_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "scenario",
            "method",
            "phase",
            "operator",
            "attempts",
            "feasible_candidates",
            "infeasible_candidates",
            "feasibility_rate",
            "accepted",
            "improved",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for key, vals in sorted(er_summary_map.items()):
            feasible = float(vals.get("feasible_candidates", 0.0))
            infeasible = float(vals.get("infeasible_candidates", 0.0))
            w.writerow(
                {
                    "scenario": key[0],
                    "method": key[1],
                    "phase": key[2],
                    "operator": key[3],
                    "attempts": float(vals.get("attempts", 0.0)),
                    "feasible_candidates": feasible,
                    "infeasible_candidates": infeasible,
                    "feasibility_rate": float(feasible / max(feasible + infeasible, 1.0)),
                    "accepted": float(vals.get("accepted", 0.0)),
                    "improved": float(vals.get("improved", 0.0)),
                }
            )
    with (run_dir / "er_repair_failure_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["scenario", "method", "phase", "operator", "reason_code", "count"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for key, count in sorted(er_failure_map.items()):
            w.writerow(
                {
                    "scenario": key[0],
                    "method": key[1],
                    "phase": key[2],
                    "operator": key[3],
                    "reason_code": key[4],
                    "count": int(count),
                }
            )
    with (run_dir / "support_coordination_ledger.jsonl").open("w", encoding="utf-8") as f:
        for r in support_coordination_rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")

    if v2_sortie_lifecycle_rows:
        out = run_dir / "tables" / "Table_v2_sortie_lifecycle.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(v2_sortie_lifecycle_rows).to_csv(out, index=False, encoding="utf-8-sig")
    if v2_support_lifecycle_rows:
        out = run_dir / "tables" / "Table_v2_support_command_lifecycle.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(v2_support_lifecycle_rows).to_csv(out, index=False, encoding="utf-8-sig")
    if v2_safety_lifecycle_rows:
        out = run_dir / "tables" / "Table_v2_safety_recovery_lifecycle.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(v2_safety_lifecycle_rows).to_csv(out, index=False, encoding="utf-8-sig")
    if v2_support_passenger_trace_rows:
        out = run_dir / "tables" / "Table_gate20_support_passenger_trace.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(v2_support_passenger_trace_rows).to_csv(out, index=False, encoding="utf-8-sig")
    if v2_support_quality_rows:
        out = run_dir / "tables" / "Table_gate112_support_opportunity_quality.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(v2_support_quality_rows).to_csv(out, index=False, encoding="utf-8-sig")

    config_used = run_dir / "config_used.yaml"
    config_used.write_text(
        yaml.safe_dump(
            {
                "matrix_config": cfg_matrix,
                "base_config": cfg_base,
                "mainline_v2_config": mainline_v2_yaml,
                "effective_config": cfg_effective_yaml,
                "resolved": {
                    "scales": sorted({str(s).upper() for s, _ in scale_scenario_pairs}),
                    "scenarios": sorted({str(sc).upper() for _, sc in scale_scenario_pairs}),
                    "scale_scenario_pairs": [f"{str(s).upper()}-{str(sc).upper()}" for s, sc in scale_scenario_pairs],
                    "methods": [str(x) for x in methods],
                    "seeds": [int(x) for x in seeds],
                    "episodes_per_seed": int(episodes_per_seed),
                    "monitor_snap_enabled": bool(monitor_snap),
                    "lightweight_metrics_only": bool(lightweight_metrics_only),
                    "lightweight_audit_flags": {k: bool(v) for k, v in lightweight_audit_flags.items()},
                    "ablation_flags_by_method": {
                        str(m): _method_ablation_flags(str(m))
                        for m in methods
                    },
                    "method_effective_flags_by_method": method_effective_flags_by_method,
                    "method_identity_by_method": method_identity_by_method,
                    "final_runtime_env_config": dict(cfg.__dict__),
                },
            },
            sort_keys=False,
            allow_unicode=False,
        ),
        encoding="utf-8",
    )

    run_meta = {
        "script": "run_experiment_matrix.py",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "code_commit_hash": _git_commit_hash(),
        "runtime_provenance": _runtime_provenance(
            [
                Path(args.config),
                Path(args.base_config),
                real_rc_cfg_path,
                PROJECT_ROOT / "configs" / "formal_l_map_bank_30_manifest.json",
                PROJECT_ROOT / "configs" / "paper_theoretical_freeze.json",
                PROJECT_ROOT / "tools" / "run_experiment_matrix.py",
                config_used,
            ]
        ),
        "args": vars(args),
        "methods": [str(x) for x in methods],
        "method_backend_map": {str(x): _method_backend_family(str(x)) for x in methods},
        "ablation_flags_by_method": {str(x): _method_ablation_flags(str(x)) for x in methods},
        "method_effective_flags_by_method": method_effective_flags_by_method,
        "method_identity_by_method": method_identity_by_method,
        "method_identity_note": "Each formal method is an isolated algorithm package bound to the same public environment interface; ER-HLNS capabilities are private to the ER-HLNS package and its named ablations.",
        "effective_config": cfg_effective_yaml,
        "monitor_snap_enabled": bool(monitor_snap),
        "lightweight_metrics_only": bool(lightweight_metrics_only),
        "lightweight_audit_flags": {k: bool(v) for k, v in lightweight_audit_flags.items()},
        "experiment_class": "ablation_only" if monitor_snap else "main",
        "outputs": {
            "episode_metrics": str(episode_csv),
            "seed_metrics": str(seed_csv),
            "task_outcome": str(run_dir / "task_outcome.csv"),
            "task_outcome_all_methods": str(run_dir / "tables" / "Table_task_outcome_all_methods.csv"),
            "config_used": str(config_used),
            "run_manifest": str(run_dir / "run_manifest.json"),
            "k2_sequence_ledger": str(run_dir / "k2_sequence_ledger.jsonl"),
            "k2_objective_comparison": str(run_dir / "k2_objective_comparison.csv"),
            "k2_feasibility_summary": str(run_dir / "k2_feasibility_summary.csv"),
            "k2_roll_forward_ledger": str(run_dir / "k2_roll_forward_ledger.jsonl"),
            "k2_runtime_sequence_ledger": str(run_dir / "k2_runtime_sequence_ledger.jsonl"),
            "k2_runtime_sequence_summary": str(run_dir / "k2_runtime_sequence_summary.csv"),
            "k2_sa_delta_ledger": str(run_dir / "k2_sa_delta_ledger.jsonl"),
            "k2_sa_delta_summary": str(run_dir / "k2_sa_delta_summary.csv"),
            "canonical_operator_ledger": str(run_dir / "canonical_operator_ledger.jsonl"),
            "canonical_operator_summary": str(run_dir / "canonical_operator_summary.csv"),
            "canonical_repair_failure_summary": str(run_dir / "canonical_repair_failure_summary.csv"),
            "er_operator_ledger": str(run_dir / "er_operator_ledger.jsonl"),
            "er_operator_summary": str(run_dir / "er_operator_summary.csv"),
            "er_repair_failure_summary": str(run_dir / "er_repair_failure_summary.csv"),
            "support_coordination_ledger": str(run_dir / "support_coordination_ledger.jsonl"),
            "support_execution_ledger": str(run_dir / "support_execution_ledger.jsonl"),
            "operator_weight_trajectory": str(run_dir / "operator_weight_trajectory.jsonl"),
            "event_trigger_ledger": str(run_dir / "event_trigger_ledger.jsonl"),
            "sa_delta_summary": str(run_dir / "sa_delta_summary.csv"),
            "mode_parity": str(run_dir / "mode_parity.csv"),
            "single_seed_results": str(run_dir / "single_seed_results.csv"),
            "runtime_matrix": str(run_dir / "runtime_matrix.csv"),
            "safety_matrix": str(run_dir / "safety_matrix.csv"),
        },
    }
    (run_dir / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    run_manifest = {
        **run_meta,
        "reproducibility_digest_fields": list(REPRO_DIGEST_FIELDS),
        "physical_freeze_export_fields": list(PHYSICAL_FREEZE_EXPORT_FIELDS),
        "episode_digest_records": [
            {
                "scale": str(r.get("scale", "")),
                "scenario": str(r.get("scenario", "")),
                "method": str(r.get("model", "")),
                "seed": int(r.get("train_seed", 0)),
                "episode": int(r.get("episode", 0)),
                **{k: str(r.get(k, "")) for k in REPRO_DIGEST_FIELDS},
                **{k: str(r.get(k, "")) for k in PHYSICAL_FREEZE_EXPORT_FIELDS},
            }
            for r in episode_rows
        ],
        "seed_digest_records": [
            {
                "scale": str(r.get("scale", "")),
                "scenario": str(r.get("scenario", "")),
                "method": str(r.get("model", "")),
                "seed": int(r.get("train_seed", 0)),
                **{k: str(r.get(k, "")) for k in REPRO_DIGEST_FIELDS},
                **{k: str(r.get(k, "")) for k in PHYSICAL_FREEZE_EXPORT_FIELDS},
            }
            for r in seed_rows
        ],
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if task_outcome_rows:
        out_task = run_dir / "task_outcome.csv"
        out_task_all = run_dir / "tables" / "Table_task_outcome_all_methods.csv"
        _write_rows_csv(out_task, task_outcome_rows, TASK_OUTCOME_FIELDS)
        _write_rows_csv(out_task_all, task_outcome_rows, TASK_OUTCOME_FIELDS)
        print(f"Saved: {out_task}")
        print(f"Saved: {out_task_all}")

    if not bool(lightweight_metrics_only):
        # Always emit ERC ablation summary artifacts when those methods are present.
        if any(str(m).startswith("erc_") and str(m) != "erc_rhc" for m in methods):
            _write_erc_ablation_outputs(run_dir=run_dir, seed_rows=seed_rows)
            _write_ablation_wiring_check(run_dir=run_dir, seed_rows=seed_rows)
        # Emit simplified-vs-rolling summary when both methods are present.
        if {"rolling_fixed", "erc_rhc"}.issubset({str(m).strip().lower() for m in methods}):
            _write_erc_rhc_simplified_vs_rolling(run_dir=run_dir, seed_rows=seed_rows)
        # Emit three-way comparison table for rolling vs old-erc vs new-erc runs.
        if {"rolling_fixed", "erc_rhc_old", "erc_rhc"}.issubset({str(m).strip().lower() for m in methods}):
            _write_compare_rolling_olderc_newerc(run_dir=run_dir, seed_rows=seed_rows)
        # Emit hard-event attribution diagnostics for erc_old/new runs.
        if {"erc_rhc_old", "erc_rhc"}.issubset({str(m).strip().lower() for m in methods}):
            _write_hard_event_attribution_outputs(
                run_dir=run_dir,
                seed_rows=seed_rows,
                hard_event_offender_rows=hard_event_offender_rows,
            )
        # Emit R-C hard no-op attribution outputs for new ERC runs.
        if "erc_rhc" in {str(m).strip().lower() for m in methods}:
            _write_rc_hard_noop_outputs(
                run_dir=run_dir,
                hard_event_reason_rows=hard_event_reason_rows,
                hard_event_offender_rows=hard_event_offender_rows,
            )
        if {"rolling_fixed", "erc_rhc_old", "erc_rhc"}.issubset({str(m).strip().lower() for m in methods}):
            _write_mc_routine_loss_outputs(
                run_dir=run_dir,
                task_outcome_rows=task_outcome_rows,
                routine_trace_rows=routine_trace_rows,
                truck_routine_summary_rows=truck_routine_summary_rows,
            )
        if "erc_rhc" in {str(m).strip().lower() for m in methods}:
            _write_rc_seed0_seed1_support_failure_outputs(
                run_dir=run_dir,
                seed_rows=seed_rows,
                task_outcome_rows=task_outcome_rows,
                support_trace_rows=support_trace_rows,
                uav_execution_rows=uav_execution_rows,
            )
        if tc_override_trace_rows:
            out_tc_override = run_dir / "tables" / "Table_TC_override_delivery_feasibility_trace.csv"
            out_tc_override.parent.mkdir(parents=True, exist_ok=True)
            fields = [
                "scenario",
                "seed",
                "method",
                "episode",
                "step",
                "truck_id",
                "routine_task_id",
                "route_dist_to_routine",
                "eta_to_routine",
                "uav_id",
                "tc_task_id",
                "launch_gain_m",
                "routine_delay_steps",
                "predicted_launchable",
                "predicted_full_sortie_feasible",
                "predicted_recovery_margin_m",
                "predicted_battery_margin_ratio",
                "predicted_lifeline_remaining",
                "recent_reject_hit",
                "override_allowed",
                "override_block_reason",
                "actual_launch_after",
                "actual_delivery_after",
                "actual_forced_recovery_after",
                "actual_reject_reason_after",
            ]
            with out_tc_override.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for row in tc_override_trace_rows:
                    w.writerow({k: row.get(k, "") for k in fields})
            print(f"Saved: {out_tc_override}")
        if len(switch_decision_ledger_rows) > 0:
            _write_switch_decision_audit_outputs(
                run_dir=run_dir,
                switch_rows=switch_decision_ledger_rows,
            )
        if agent_step_rows:
            out_agent_steps = run_dir / "agent_step_trace.csv"
            fields = sorted({str(key) for row in agent_step_rows for key in row})
            _write_rows_csv(out_agent_steps, agent_step_rows, fields)
            print(f"Saved: {out_agent_steps}")
        if task_proximity_decision_rows:
            out_task_proximity = run_dir / "task_proximity_decision_trace.csv"
            fields = sorted({str(key) for row in task_proximity_decision_rows for key in row})
            _write_rows_csv(out_task_proximity, task_proximity_decision_rows, fields)
            print(f"Saved: {out_task_proximity}")

    print(f"Saved: {episode_csv}")
    print(f"Saved: {seed_csv}")
    print(f"Saved: {config_used}")
    print(f"Saved: {run_dir / 'run_meta.json'}")


if __name__ == "__main__":
    main()
