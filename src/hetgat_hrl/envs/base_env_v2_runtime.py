from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Tuple

import numpy as np

from hetgat_hrl.core.mdp_spec import AgentKind
from hetgat_hrl.envs.physical_environment_v2 import (
    PhysicalEnvironmentV2,
    RecoveryPlanV2,
    RoadConditionV2,
    RoadEdgeV2,
    RouteSegmentV2,
    TruckStateV2,
    UAVOperationalState,
    UAVLifecycleTransition,
    UAVStateV2,
    WeatherStateV2,
)


def physical_v2_enabled(env: Any) -> bool:
    return str(getattr(getattr(env, "cfg", None), "physical_environment_version", "v1")).lower() == "v2"


def _edge_id(src: int, dst: int) -> str:
    a, b = sorted((int(src), int(dst)))
    return f"{a}-{b}"


def _nearest_node(env: Any, xy: Tuple[float, float] | None) -> int:
    if xy is None:
        return 0
    best_node = 0
    best_dist = float("inf")
    for nid, node in getattr(env.topology, "nodes", {}).items():
        d = float(np.hypot(float(node.x) - float(xy[0]), float(node.y) - float(xy[1])))
        if d < best_dist:
            best_node = int(nid)
            best_dist = d
    return int(best_node)


def _weather_from_v1(env: Any, node_id: int, step: int) -> WeatherStateV2:
    node = env.topology.nodes[int(node_id)]
    try:
        wx = env.hazards.weather_at((float(node.x), float(node.y)))
        wind = float(getattr(wx, "wind", 0.0))
        rain = float(getattr(wx, "rain", 0.0))
    except Exception:
        wind = 0.0
        rain = 0.0
    visibility = float(max(10.0 - rain / 10.0, 0.5))
    storm_level = float(np.clip(max(wind / 8.0, rain / 18.0), 0.0, 5.0))
    return WeatherStateV2(
        region_id="r0",
        step=int(step),
        wind_speed=wind,
        wind_direction=0.0,
        rain_intensity=rain,
        visibility=visibility,
        temperature=20.0,
        storm_level=storm_level,
        no_fly_status=bool(storm_level >= 5.0),
    )


def init_physical_v2_runtime(env: Any) -> None:
    env.physical_v2_enabled = bool(physical_v2_enabled(env))
    env.physical_v2 = None
    env.physical_v2_energy_eval_count = 0
    env.physical_v2_travel_eval_count = 0
    env.physical_v2_safety_intervention_count = 0
    env.physical_v2_blocked_entry_count = 0
    env.physical_v2_energy_fraction_total = 0.0
    env.physical_v2_minimum_energy_reserve_seen = float("inf")
    env.physical_v2_lifecycle_ledger = []
    env.physical_v2_energy_ledger = []
    env.physical_v2_launch_check_count = 0
    env.physical_v2_service_start_count = 0
    env.physical_v2_service_complete_count = 0
    env.physical_v2_recovery_event_count = 0
    env.physical_v2_forced_landing_count = 0
    env.physical_v2_uav_drop_runtime_count = 0
    env.physical_v2_service_reject_count = 0
    env.physical_v2_energy_deduction_count = 0
    env.physical_v2_lifecycle_transition_count = 0
    env.physical_v2_recovery_motion_step_count = 0
    env.physical_v2_recovery_bind_count = 0
    if not env.physical_v2_enabled:
        env.physical_v2_fairness_hash = ""
        env.physical_v2_road_digest = ""
        env.physical_v2_weather_digest = ""
        return

    edges: list[RoadEdgeV2] = []
    for src, nbs in sorted(getattr(env.topology, "adjacency", {}).items()):
        for dst in sorted(nbs):
            if int(src) >= int(dst):
                continue
            attr = env.topology.edge_attr(int(src), int(dst))
            length = float(max(getattr(attr, "length_m", 0.0), env.topology.edge_distance(int(src), int(dst))))
            base_time = float(length / max(float(getattr(env.cfg, "truck_speed_mps", 1.0)), 1e-6))
            edges.append(
                RoadEdgeV2(
                    edge_id=_edge_id(int(src), int(dst)),
                    from_node=int(src),
                    to_node=int(dst),
                    road_class=str(getattr(attr, "road_class", "collector")),
                    base_travel_time=base_time,
                    capacity=float(max(getattr(attr, "lanes", 1), 1)),
                    hazard_exposure=float(getattr(attr, "barrier_exposure", 0.0)),
                    vulnerability=float(getattr(attr, "base_vulnerability", 0.0)),
                    bridge_or_tunnel=bool(getattr(attr, "bridge_or_tunnel", False)),
                    repair_priority=1,
                    spatial_cluster="r0",
                )
            )
    max_step = int(max(getattr(env.cfg, "max_steps", 1), 1))
    weather = {("r0", step): _weather_from_v1(env, 0, step) for step in range(max_step + 1)}
    node_region = {int(nid): "r0" for nid in getattr(env.topology, "nodes", {})}
    env.physical_v2 = PhysicalEnvironmentV2(
        tuple(edges),
        weather,
        node_region,
        safety_protocol=str(getattr(env.cfg, "physical_environment_safety_protocol", "shielded_operation")),
    )
    env.physical_v2_fairness_hash = env.physical_v2.fairness_hash()
    env.physical_v2_road_digest = env.physical_v2.road_digest()
    env.physical_v2_weather_digest = env.physical_v2.weather_digest()


def v2_edge_accessible(env: Any, src: int, dst: int) -> bool:
    if not physical_v2_enabled(env) or getattr(env, "physical_v2", None) is None:
        return True
    cond = env.physical_v2.road_conditions.get(_edge_id(src, dst), RoadConditionV2())
    if not bool(cond.truck_accessible) or not math.isfinite(float(cond.travel_time_multiplier)):
        env.physical_v2_blocked_entry_count += 1
        return False
    return True


def v2_truck_speed_multiplier(env: Any, src: int, dst: int) -> float:
    if not physical_v2_enabled(env) or getattr(env, "physical_v2", None) is None:
        return 1.0
    edge_id = _edge_id(src, dst)
    step = int(getattr(getattr(env, "state", None), "step_index", 0))
    weather = env.physical_v2.weather_at(int(src), step)
    travel = float(env.physical_v2.travel_time(edge_id, weather))
    base = float(env.physical_v2.edge(edge_id).base_travel_time)
    env.physical_v2_travel_eval_count += 1
    if not math.isfinite(travel):
        return float("inf")
    return float(max(travel / max(base, 1e-9), 1e-6))


def v2_uav_energy_cost_fraction(env: Any, aid: str, dist_m: float, xy: Tuple[float, float]) -> float | None:
    if not physical_v2_enabled(env) or getattr(env, "physical_v2", None) is None:
        return None
    st = env.state.agents[str(aid)]
    if getattr(st, "kind", None) != AgentKind.UAV:
        return float("inf")
    node_id = _nearest_node(env, xy)
    step = int(getattr(getattr(env, "state", None), "step_index", 0))
    weather = env.physical_v2.weather_at(node_id, step)
    payload = float(max(getattr(st, "payload_kg_current", 0.0), 0.0))
    capacity = float(max(getattr(env.cfg, "uav_payload_capacity_kg", 1.0), 1e-6))
    payload_ratio = float(np.clip(payload / capacity, 0.0, 1.0))
    load_factor = float(1.0 + max(float(getattr(env.cfg, "uav_full_load_energy_penalty", 0.0)), 0.0) * payload_ratio)
    weather_factor = float(
        1.0
        + max(float(getattr(env.cfg, "uav_headwind_energy_coeff", 0.0)), 0.0) * max(float(weather.wind_speed), 0.0)
        + max(float(getattr(env.cfg, "uav_rain_energy_coeff", 0.0)), 0.0) * max(float(weather.rain_intensity), 0.0)
    )
    fraction = float(
        max(dist_m, 0.0)
        * max(float(getattr(env.cfg, "uav_flight_discharge_per_m", 0.0)), 0.0)
        * load_factor
        * weather_factor
    )
    env.physical_v2_energy_eval_count += 1
    env.physical_v2_energy_fraction_total += float(fraction)
    env.physical_v2_minimum_energy_reserve_seen = float(
        min(float(env.physical_v2_minimum_energy_reserve_seen), float(max(getattr(st, "battery", 0.0), 0.0) - fraction))
    )
    return float(fraction)


def record_v2_lifecycle(env: Any, event_type: str, *, aid: str = "", task_id: str = "", reason: str = "", **extra: Any) -> None:
    if not physical_v2_enabled(env):
        return
    ledger = getattr(env, "physical_v2_lifecycle_ledger", None)
    if ledger is None:
        env.physical_v2_lifecycle_ledger = []
        ledger = env.physical_v2_lifecycle_ledger
    ledger.append(
        {
            "step": int(getattr(getattr(env, "state", None), "step_index", 0)),
            "event_type": str(event_type),
            "agent_id": str(aid),
            "task_id": str(task_id),
            "reason": str(reason),
            **extra,
        }
    )


def record_v2_energy_deduction(
    env: Any,
    aid: str,
    *,
    energy_before: float,
    energy_after: float,
    distance_m: float,
    reason: str,
) -> None:
    if not physical_v2_enabled(env):
        return
    env.physical_v2_energy_deduction_count += 1
    row = {
        "step": int(getattr(getattr(env, "state", None), "step_index", 0)),
        "agent_id": str(aid),
        "energy_before": float(energy_before),
        "energy_after": float(energy_after),
        "energy_delta": float(energy_before) - float(energy_after),
        "distance_m": float(distance_m),
        "reason": str(reason),
    }
    ledger = getattr(env, "physical_v2_energy_ledger", None)
    if ledger is None:
        env.physical_v2_energy_ledger = []
        ledger = env.physical_v2_energy_ledger
    ledger.append(row)
    record_v2_lifecycle(env, "ENERGY_DEDUCTION", aid=str(aid), **row)


def record_v2_recovery_motion(
    env: Any,
    aid: str,
    *,
    target_truck_id: str | None,
    target_anchor: int | None,
    old_xy: Tuple[float, float],
    new_xy: Tuple[float, float],
    distance_m: float,
    reason: str,
    wind_speed: float = 0.0,
    wind_direction: float = 0.0,
    rain: float = 0.0,
    visibility: float = 10.0,
    energy_before: float | None = None,
    energy_after: float | None = None,
    reason_codes: tuple[str, ...] = (),
) -> None:
    if not physical_v2_enabled(env):
        return
    env.physical_v2_recovery_motion_step_count += 1
    record_v2_lifecycle(
        env,
        "RECOVERY_MOTION",
        aid=str(aid),
        reason=str(reason),
        target_truck_id="" if target_truck_id is None else str(target_truck_id),
        target_anchor=None if target_anchor is None else int(target_anchor),
        old_x=float(old_xy[0]),
        old_y=float(old_xy[1]),
        new_x=float(new_xy[0]),
        new_y=float(new_xy[1]),
        distance_m=float(distance_m),
        wind_speed=float(wind_speed),
        wind_direction=float(wind_direction),
        rain=float(rain),
        visibility=float(visibility),
        energy_before=float(0.0 if energy_before is None else energy_before),
        energy_used=float(0.0 if energy_before is None or energy_after is None else float(energy_before) - float(energy_after)),
        energy_after=float(0.0 if energy_after is None else energy_after),
        reason_codes="|".join(str(r) for r in reason_codes),
    )


def v2_authoritative_launch_check(env: Any, aid: str, task: Any | None) -> tuple[bool, str, bool] | None:
    if not physical_v2_enabled(env) or getattr(env, "physical_v2", None) is None:
        return None
    env.physical_v2_launch_check_count += 1
    st = env.state.agents.get(str(aid), None)
    if st is None or getattr(st, "kind", None) != AgentKind.UAV:
        return False, "not_uav", False
    if getattr(st, "follow_target", None) is None:
        return False, "not_docked", False
    if task is None:
        return False, "no_emergency_goal", True
    truck_id = str(getattr(st, "follow_target", ""))
    truck_st = env.state.agents.get(truck_id, None)
    if truck_st is None:
        return False, "support_binding_stale", True
    launch_anchor = int(getattr(truck_st, "node", getattr(st, "node", 0)) or 0)
    task_node = int(getattr(task, "demand_node", launch_anchor))
    task_xy = env._node_xy(task_node)
    uxy = getattr(st, "pos_xy", None) or env._node_xy(int(getattr(st, "node", launch_anchor) or launch_anchor))
    distance_km = float(np.hypot(float(task_xy[0]) - float(uxy[0]), float(task_xy[1]) - float(uxy[1]))) / 1000.0
    payload = float(max(getattr(st, "payload_kg_current", getattr(task, "demand_kg", 0.0)), 0.0))
    weather = env.physical_v2.weather_at(task_node, int(getattr(env.state, "step_index", 0)))
    uav = UAVStateV2(
        str(aid),
        UAVOperationalState.ON_TRUCK,
        int(getattr(st, "node", launch_anchor) or launch_anchor),
        float(max(getattr(st, "battery", 0.0), 0.0) * 100.0),
        100.0,
        payload_capacity=float(max(getattr(env.cfg, "uav_payload_capacity_kg", 5.0), 1e-6)),
    )
    truck = TruckStateV2(
        truck_id,
        launch_anchor,
        stopped=bool(getattr(truck_st, "transit", None) is None),
        dwell_steps=1,
    )
    proposal = env.physical_v2.check_launch(
        truck,
        uav,
        launch_anchor=launch_anchor,
        task_payload=payload,
        route_segments=(RouteSegmentV2(launch_anchor, task_node, distance_km, 0.0, service_time=1.0),),
        weather=weather,
        recovery_plan=RecoveryPlanV2(task_node, alternate_anchors=(launch_anchor, task_node), reserve_energy=float(env.physical_v2.safety_reserve_energy)),
        support_binding_conflict=False,
    )
    step = int(getattr(env.state, "step_index", 0))
    candidate_digest = hashlib.sha256(
        json.dumps(
            {
                "step": step,
                "agent_id": str(aid),
                "task_id": str(getattr(task, "task_id", "")),
                "launch_anchor": int(launch_anchor),
                "task_node": int(task_node),
                "reason_codes": tuple(proposal.reason_codes),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    decision = env.physical_v2.apply_safety_shield(
        proposal,
        step=step,
        agent_id=str(aid),
        candidate_digest=candidate_digest,
    )
    allowed = bool(decision.allowed and (proposal.feasible or str(env.physical_v2.safety_protocol) == "unshielded_stress"))
    reason = "v2_authorized_launch" if allowed else "|".join(decision.reason_codes or proposal.reason_codes) or "v2_launch_blocked"
    record_v2_lifecycle(
        env,
        "LAUNCH_CHECK",
        aid=str(aid),
        task_id=str(getattr(task, "task_id", "")),
        reason=reason,
        allowed=allowed,
        launch_anchor=int(launch_anchor),
        recovery_anchor=int(task_node),
        protocol=str(env.physical_v2.safety_protocol),
    )
    return allowed, reason, not allowed


def v2_authorize_service_start(env: Any, aid: str, task: Any) -> tuple[bool, str] | None:
    if not physical_v2_enabled(env) or getattr(env, "physical_v2", None) is None:
        return None
    st = env.state.agents.get(str(aid), None)
    if st is None:
        env.physical_v2_service_reject_count += 1
        return False, "agent_not_found"
    if getattr(task, "status", None) is None or str(getattr(getattr(task, "status", None), "value", getattr(task, "status", ""))).lower() not in {"pending", "claimed"}:
        env.physical_v2_service_reject_count += 1
        return False, "task_not_active"
    if bool(getattr(st, "crashed", False)):
        env.physical_v2_service_reject_count += 1
        return False, "agent_crashed"
    if getattr(st, "kind", None) == AgentKind.UAV:
        if not bool(env._uav_loaded(str(aid))):
            env.physical_v2_service_reject_count += 1
            return False, "uav_not_loaded"
        node_id = int(getattr(task, "demand_node", getattr(st, "node", 0)) or 0)
        weather = env.physical_v2.weather_at(node_id, int(getattr(env.state, "step_index", 0)))
        if bool(getattr(weather, "no_fly_status", False)):
            env.physical_v2_service_reject_count += 1
            return False, "no_fly_weather"
        if float(getattr(st, "battery", 0.0) or 0.0) <= 0.0:
            env.physical_v2_service_reject_count += 1
            return False, "uav_energy_depleted"
    # A BULK_RELAY task may have two explicitly contracted UAVs unloading in
    # parallel.  The second one sees CLAIMED status, which is intentional.
    if not bool(env.is_task_serviceable_by_agent(str(aid), task)):
        env.physical_v2_service_reject_count += 1
        return False, "task_not_serviceable"
    return True, "v2_service_authorized"


def record_v2_service_start(env: Any, aid: str, task: Any) -> None:
    if not physical_v2_enabled(env):
        return
    env.physical_v2_service_start_count += 1
    step = int(getattr(env.state, "step_index", 0))
    setattr(task, "actual_arrival_step", step)
    setattr(task, "service_start_step", step)
    st = env.state.agents.get(str(aid), None)
    record_v2_lifecycle(
        env,
        "SERVICE_START",
        aid=str(aid),
        task_id=str(getattr(task, "task_id", "")),
        actual_arrival_step=step,
        service_start_step=step,
        energy_before_service=float(getattr(st, "battery", 0.0) if st is not None else 0.0),
    )


def v2_authorize_service_completion(env: Any, aid: str, task: Any, *, transfer: float) -> tuple[bool, str] | None:
    if not physical_v2_enabled(env) or getattr(env, "physical_v2", None) is None:
        return None
    st = env.state.agents.get(str(aid), None)
    if st is None:
        env.physical_v2_service_reject_count += 1
        return False, "agent_not_found"
    if bool(getattr(st, "crashed", False)):
        env.physical_v2_service_reject_count += 1
        return False, "agent_crashed"
    if float(transfer) <= 0.0:
        env.physical_v2_service_reject_count += 1
        return False, "insufficient_payload"
    if getattr(task, "actual_arrival_step", None) is None or getattr(task, "service_start_step", None) is None:
        env.physical_v2_service_reject_count += 1
        return False, "missing_actual_service_start"
    if getattr(st, "kind", None) == AgentKind.UAV and float(getattr(st, "battery", 0.0) or 0.0) <= 0.0:
        env.physical_v2_service_reject_count += 1
        return False, "uav_energy_depleted"
    return True, "v2_service_complete_authorized"


def record_v2_service_complete(env: Any, aid: str, task: Any) -> None:
    if not physical_v2_enabled(env):
        return
    env.physical_v2_service_complete_count += 1
    step = int(getattr(env.state, "step_index", 0))
    setattr(task, "service_complete_step", step)
    st = env.state.agents.get(str(aid), None)
    record_v2_lifecycle(
        env,
        "SERVICE_COMPLETE",
        aid=str(aid),
        task_id=str(getattr(task, "task_id", "")),
        actual_arrival_step=int(getattr(task, "actual_arrival_step", -1)),
        service_start_step=int(getattr(task, "service_start_step", -1)),
        service_complete_step=step,
        energy_after_service=float(getattr(st, "battery", 0.0) if st is not None else 0.0),
    )


def v2_authoritative_recovery_transition(
    env: Any,
    aid: str,
    *,
    requested_event: str,
    reason: str,
) -> UAVLifecycleTransition | None:
    if not physical_v2_enabled(env) or getattr(env, "physical_v2", None) is None:
        return None
    st = env.state.agents.get(str(aid), None)
    if st is None:
        return UAVLifecycleTransition("", "UAV_DROP", False, None, 0.0, 0.0, ("agent_not_found",))
    node_id = int(getattr(st, "node", 0) or 0)
    weather = env.physical_v2.weather_at(node_id, int(getattr(env.state, "step_index", 0)))
    battery_energy = float(max(getattr(st, "battery", 0.0), 0.0) * 100.0)
    uav = UAVStateV2(str(aid), UAVOperationalState.AIRBORNE, node_id, battery_energy, 100.0)
    recovery_available = any(
        getattr(ts, "kind", None) == AgentKind.TRUCK and not bool(getattr(ts, "crashed", False))
        for ts in getattr(env.state, "agents", {}).values()
    )
    trucks = [
        ts
        for ts in getattr(env.state, "agents", {}).values()
        if getattr(ts, "kind", None) == AgentKind.TRUCK and not bool(getattr(ts, "crashed", False))
    ]
    alternate = None
    if trucks and str(requested_event) == "DIVERTING":
        truck_state = trucks[0]
        alternate = env.physical_v2.select_alternate_recovery(
            uav,
            TruckStateV2(
                str(getattr(truck_state, "agent_id", "truck")),
                int(getattr(truck_state, "node", node_id) or node_id),
                stopped=bool(getattr(truck_state, "transit", None) is None),
                dwell_steps=1,
            ),
            [int(getattr(truck_state, "node", node_id) or node_id), node_id],
            remaining_energy=battery_energy,
            weather=weather,
        )
    return env.physical_v2.decide_lifecycle_transition(
        uav,
        requested_event=str(requested_event),
        target_anchor=node_id,
        energy_after=battery_energy,
        weather=weather,
        recovery_available=bool(recovery_available),
        alternate_recovery=alternate,
        reason_codes=(str(reason),),
    )


def record_v2_recovery(env: Any, aid: str, event_type: str, reason: str = "") -> None:
    if not physical_v2_enabled(env):
        return
    transition = v2_authoritative_recovery_transition(env, str(aid), requested_event=str(event_type), reason=str(reason))
    if transition is not None:
        event_type = transition.new_state
        env.physical_v2_lifecycle_transition_count += 1
    env.physical_v2_recovery_event_count += 1
    if str(event_type) == "FORCED_LANDING":
        env.physical_v2_forced_landing_count += 1
    if str(event_type) == "UAV_DROP":
        env.physical_v2_uav_drop_runtime_count += 1
    if str(event_type) == "RECOVERED":
        env.physical_v2_recovery_bind_count += 1
    record_v2_lifecycle(
        env,
        str(event_type),
        aid=str(aid),
        reason=str(reason),
        old_state=str(transition.old_state if transition is not None else ""),
        new_state=str(transition.new_state if transition is not None else event_type),
        authorized=bool(transition.authorized if transition is not None else True),
        target_anchor=transition.target_anchor if transition is not None else None,
        energy_before=float(transition.energy_before if transition is not None else 0.0),
        energy_after=float(transition.energy_after if transition is not None else 0.0),
        reason_codes="|".join(transition.reason_codes if transition is not None else (str(reason),)),
    )


def v2_metrics(env: Any) -> dict[str, Any]:
    enabled = bool(physical_v2_enabled(env) and getattr(env, "physical_v2", None) is not None)
    physical = getattr(env, "physical_v2", None)
    physical_forced_landing = int(getattr(physical, "forced_landing_count", 0) if physical else 0)
    runtime_forced_landing = int(getattr(env, "physical_v2_forced_landing_count", 0))
    forced_landing = max(physical_forced_landing, runtime_forced_landing)
    physical_uav_drop = int(getattr(physical, "uav_drop_count", 0) if physical else 0)
    runtime_uav_drop = int(getattr(env, "physical_v2_uav_drop_runtime_count", 0))
    # The physical layer and runtime transition hook can observe one loss twice.
    # Preserve both sources while exporting a de-duplicated canonical count.
    uav_drop = max(physical_uav_drop, runtime_uav_drop)
    unsafe = int(getattr(physical, "unsafe_plan_proposal_count", 0) if physical else 0)
    unique_unsafe = int(getattr(physical, "unique_unsafe_proposal_count", 0) if physical else 0)
    duplicate_checks = int(getattr(physical, "duplicate_unsafe_check_count", 0) if physical else 0)
    minimum_reserve = float(
        0.0
        if not math.isfinite(float(getattr(env, "physical_v2_minimum_energy_reserve_seen", float("inf"))))
        else float(getattr(env, "physical_v2_minimum_energy_reserve_seen", 0.0))
    )
    metrics = {
        "physical_environment_version": "v2" if enabled else "v1",
        "physical_v2_enabled": bool(enabled),
        "physical_v2_fairness_hash": str(getattr(env, "physical_v2_fairness_hash", "")),
        "physical_v2_road_digest": str(getattr(env, "physical_v2_road_digest", "")),
        "physical_v2_weather_digest": str(getattr(env, "physical_v2_weather_digest", "")),
        "physical_v2_energy_eval_count": int(getattr(env, "physical_v2_energy_eval_count", 0)),
        "physical_v2_travel_eval_count": int(getattr(env, "physical_v2_travel_eval_count", 0)),
        "physical_v2_blocked_entry_count": int(getattr(env, "physical_v2_blocked_entry_count", 0)),
        "physical_v2_shield_intervention_count": int(getattr(physical, "shield_intervention_count", 0) if physical else 0),
        "physical_v2_unsafe_plan_proposal_count": int(getattr(physical, "unsafe_plan_proposal_count", 0) if physical else 0),
        "physical_v2_uav_drop_count": physical_uav_drop,
        "physical_v2_launch_check_count": int(getattr(env, "physical_v2_launch_check_count", 0)),
        "physical_v2_service_start_count": int(getattr(env, "physical_v2_service_start_count", 0)),
        "physical_v2_service_complete_count": int(getattr(env, "physical_v2_service_complete_count", 0)),
        "physical_v2_recovery_event_count": int(getattr(env, "physical_v2_recovery_event_count", 0)),
        "physical_v2_forced_landing_count": forced_landing,
        "physical_v2_forced_landing_physical_count": physical_forced_landing,
        "physical_v2_forced_landing_runtime_count": runtime_forced_landing,
        "physical_v2_uav_drop_runtime_count": runtime_uav_drop,
        "physical_v2_service_reject_count": int(getattr(env, "physical_v2_service_reject_count", 0)),
        "physical_v2_energy_deduction_count": int(getattr(env, "physical_v2_energy_deduction_count", 0)),
        "physical_v2_lifecycle_transition_count": int(getattr(env, "physical_v2_lifecycle_transition_count", 0)),
        "physical_v2_recovery_motion_step_count": int(getattr(env, "physical_v2_recovery_motion_step_count", 0)),
        "physical_v2_recovery_bind_count": int(getattr(env, "physical_v2_recovery_bind_count", 0)),
        "physical_v2_lifecycle_ledger_count": int(len(getattr(env, "physical_v2_lifecycle_ledger", []))),
        "physical_v2_energy_ledger_count": int(len(getattr(env, "physical_v2_energy_ledger", []))),
        "physical_v2_energy_fraction_total": float(getattr(env, "physical_v2_energy_fraction_total", 0.0)),
        "physical_v2_minimum_energy_reserve_seen": minimum_reserve,
        "physical_v2_road_event_ledger_count": int(len(getattr(physical, "road_event_ledger", [])) if physical else 0),
        "physical_v2_weather_ledger_count": int(len(getattr(physical, "weather_ledger", [])) if physical else 0),
        "physical_v2_unique_unsafe_proposal_count": unique_unsafe,
        "physical_v2_duplicate_unsafe_check_count": duplicate_checks,
        "physical_v2_interventions_per_unique_proposal": float(
            (float(getattr(physical, "shield_intervention_count", 0)) / max(float(getattr(physical, "unique_unsafe_proposal_count", 0)), 1.0))
            if physical
            else 0.0
        ),
        "physical_v2_interventions_per_mission": float(
            (float(getattr(physical, "shield_intervention_count", 0)) / max(float(getattr(env, "physical_v2_launch_check_count", 0)), 1.0))
            if physical
            else 0.0
        ),
        "physical_v2_shield_reason_counts": json.dumps(
            dict(sorted(getattr(physical, "shield_reason_counts", {}).items())) if physical else {},
            sort_keys=True,
        ),
    }
    metrics.update(
        {
            "UAV_DROP": int(uav_drop),
            "uav_drop_count": int(uav_drop),
            "FORCED_LANDING": int(forced_landing),
            "EMERGENCY_RETURN": int(getattr(env, "physical_v2_emergency_return_count", 0)),
            "MISSION_ABORT": int(getattr(env, "physical_v2_mission_abort_count", 0)),
            "UNSAFE_PROPOSAL": int(unsafe),
            "UNIQUE_SHIELD_INTERVENTION": int(unique_unsafe),
            "DUPLICATE_SHIELD_CHECK": int(duplicate_checks),
            "MINIMUM_ENERGY_RESERVE": float(minimum_reserve),
            "ENERGY_EXHAUSTION": int(getattr(env, "physical_v2_energy_exhaustion_count", 0)),
            "RECOVERY_FAILURE": int(getattr(env, "physical_v2_recovery_failure_count", 0)),
            "WEATHER_LOSS_OF_CONTROL": int(getattr(env, "physical_v2_weather_loss_of_control_count", 0)),
        }
    )
    return metrics


def digest_metrics_subset(row: dict[str, Any]) -> str:
    payload = json.dumps(row, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
