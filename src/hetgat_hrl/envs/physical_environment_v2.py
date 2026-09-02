from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Iterable, Mapping, Sequence
from pathlib import Path


class RoadState(str, Enum):
    OPEN = "OPEN"
    DEGRADED = "DEGRADED"
    BLOCKED = "BLOCKED"
    UNDER_REPAIR = "UNDER_REPAIR"
    REOPENED = "REOPENED"


class DamageLevel(str, Enum):
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    BLOCKED = "blocked"


class UAVOperationalState(str, Enum):
    ON_TRUCK = "ON_TRUCK"
    LAUNCH_PENDING = "LAUNCH_PENDING"
    AIRBORNE = "AIRBORNE"
    SERVING = "SERVING"
    RECOVERY_PENDING = "RECOVERY_PENDING"
    DIVERTING = "DIVERTING"
    EMERGENCY_RETURN = "EMERGENCY_RETURN"
    FORCED_LANDING = "FORCED_LANDING"
    RECOVERED = "RECOVERED"
    UAV_DROP = "UAV_DROP"


@dataclass(frozen=True)
class RoadEdgeV2:
    edge_id: str
    from_node: int
    to_node: int
    road_class: str
    base_travel_time: float
    capacity: float
    hazard_exposure: float
    vulnerability: float
    bridge_or_tunnel: bool
    repair_priority: int
    spatial_cluster: str


@dataclass(frozen=True)
class RoadConditionV2:
    state: RoadState = RoadState.OPEN
    damage_level: DamageLevel = DamageLevel.NONE
    travel_time_multiplier: float = 1.0
    truck_accessible: bool = True
    launch_recovery_allowed: bool = True
    repair_duration_steps: int = 0
    reopened_step: int | None = None


@dataclass(frozen=True)
class RoadEventRecordV2:
    step: int
    edge: str
    old_state: str
    new_state: str
    damage_level: str
    hazard_intensity: float
    travel_time_multiplier: float
    reason: str


@dataclass(frozen=True)
class WeatherStateV2:
    region_id: str
    step: int
    wind_speed: float
    wind_direction: float
    rain_intensity: float
    visibility: float
    temperature: float
    storm_level: float
    no_fly_status: bool = False

    def severity(self) -> float:
        wind = min(max(self.wind_speed / 20.0, 0.0), 1.0)
        rain = min(max(self.rain_intensity / 50.0, 0.0), 1.0)
        visibility = min(max((5.0 - self.visibility) / 5.0, 0.0), 1.0)
        storm = min(max(self.storm_level / 5.0, 0.0), 1.0)
        return float(max(wind, rain, visibility, storm, 1.0 if self.no_fly_status else 0.0))


@dataclass(frozen=True)
class WeatherLedgerRecordV2:
    step: int
    region_id: str
    wind_speed: float
    wind_direction: float
    rain_intensity: float
    visibility: float
    temperature: float
    storm_level: float
    no_fly_status: bool


@dataclass(frozen=True)
class UAVStateV2:
    uav_id: str
    operational_state: UAVOperationalState
    current_node: int
    battery_energy: float
    battery_capacity: float
    battery_degradation: float = 0.0
    payload_capacity: float = 5.0


@dataclass(frozen=True)
class TruckStateV2:
    truck_id: str
    current_node: int
    stopped: bool
    dwell_steps: int = 0


@dataclass(frozen=True)
class RouteSegmentV2:
    from_node: int
    to_node: int
    distance: float
    heading_degrees: float
    hover_time: float = 0.0
    service_time: float = 0.0


@dataclass(frozen=True)
class RecoveryPlanV2:
    recovery_anchor: int | None
    alternate_anchors: tuple[int, ...] = ()
    reserve_energy: float = 10.0


@dataclass(frozen=True)
class EnergyEstimate:
    takeoff_energy: float
    cruise_energy: float
    hover_energy: float
    service_energy: float
    recovery_energy: float
    reserve_energy: float
    total_required_energy: float
    remaining_energy_after_mission: float
    feasible: bool
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class LaunchRecoveryCheck:
    feasible: bool
    reason_codes: tuple[str, ...] = ()
    selected_anchor: int | None = None


@dataclass(frozen=True)
class RecoveryPathPlanV2:
    feasible: bool
    selected_anchor: int | None
    expected_path: tuple[int, ...]
    expected_energy: float
    expected_arrival_step: int | None
    transition_state: str
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class SafetyDecisionV2:
    allowed: bool
    protocol: str
    reason_codes: tuple[str, ...] = ()
    intervention: str | None = None


@dataclass(frozen=True)
class UAVLifecycleTransition:
    old_state: str
    new_state: str
    authorized: bool
    target_anchor: int | None
    energy_before: float
    energy_after: float
    reason_codes: tuple[str, ...]


SHIELD_REASON_ALIASES = {
    "INVALID_LAUNCH_ANCHOR": "ILLEGAL_LAUNCH_ANCHOR",
    "DWELL_TIME_TOO_SHORT": "DWELL_NOT_MET",
    "UAV_NOT_ON_TRUCK": "SUPPORT_CONFLICT",
    "SUPPORT_BINDING_CONFLICT": "SUPPORT_CONFLICT",
    "INSUFFICIENT_UAV_ENERGY_WITH_RESERVE": "INSUFFICIENT_MISSION_ENERGY",
    "PAYLOAD_EXCEEDS_CAPACITY": "PAYLOAD_INSUFFICIENT",
    "NO_FEASIBLE_ALTERNATE_RECOVERY": "NO_RECOVERY_PLAN",
    "STATE_STALE": "STALE_STATE",
}


def normalize_shield_reason(reason: str) -> str:
    return SHIELD_REASON_ALIASES.get(str(reason), str(reason))


@dataclass
class PhysicalEnvironmentV2:
    road_edges: tuple[RoadEdgeV2, ...]
    weather_by_region_step: Mapping[tuple[str, int], WeatherStateV2]
    node_region: Mapping[int, str]
    safety_protocol: str = "shielded_operation"
    minimum_dwell_steps: int = 1
    safety_reserve_energy: float = 10.0
    road_conditions: dict[str, RoadConditionV2] = field(default_factory=dict)
    road_event_ledger: list[RoadEventRecordV2] = field(default_factory=list)
    weather_ledger: list[WeatherLedgerRecordV2] = field(default_factory=list)
    unsafe_plan_proposal_count: int = 0
    shield_intervention_count: int = 0
    mission_abort_count: int = 0
    emergency_return_count: int = 0
    forced_landing_count: int = 0
    reserve_breach_proposal_count: int = 0
    uav_drop_count: int = 0
    minimum_energy_reserve_seen: float = math.inf
    unique_unsafe_proposal_count: int = 0
    duplicate_unsafe_check_count: int = 0
    shield_reason_counts: dict[str, int] = field(default_factory=dict)
    _unsafe_proposal_keys: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        for edge in self.road_edges:
            self.road_conditions.setdefault(edge.edge_id, RoadConditionV2())

    def _region_for_node(self, node: int) -> str:
        return str(self.node_region.get(int(node), "default"))

    def weather_at(self, node: int, step: int) -> WeatherStateV2:
        region = self._region_for_node(int(node))
        weather = self.weather_by_region_step.get((region, int(step)))
        if weather is None:
            weather = WeatherStateV2(region, int(step), 0.0, 0.0, 0.0, 10.0, 20.0, 0.0, False)
        self.weather_ledger.append(WeatherLedgerRecordV2(**asdict(weather)))
        return weather

    def damage_probability(self, edge: RoadEdgeV2, hazard_intensity: float, cluster_shock: float = 0.0) -> float:
        class_factor = {"highway": 0.75, "arterial": 1.0, "local": 1.15, "bridge": 1.35, "tunnel": 1.25}.get(
            edge.road_class, 1.0
        )
        structure_factor = 1.25 if edge.bridge_or_tunnel else 1.0
        z = (
            1.25 * float(hazard_intensity)
            + 0.9 * float(edge.hazard_exposure)
            + 1.1 * float(edge.vulnerability)
            + 0.55 * float(cluster_shock)
        ) * class_factor * structure_factor - 1.6
        return float(1.0 / (1.0 + math.exp(-z)))

    def apply_damage(
        self,
        edge_id: str,
        level: DamageLevel,
        *,
        step: int,
        hazard_intensity: float,
        reason: str,
    ) -> RoadConditionV2:
        old = self.road_conditions.get(edge_id, RoadConditionV2())
        if level == DamageLevel.MINOR:
            new = RoadConditionV2(RoadState.DEGRADED, level, 1.35, True, True, 2)
        elif level == DamageLevel.MODERATE:
            new = RoadConditionV2(RoadState.DEGRADED, level, 1.85, True, False, 4)
        elif level == DamageLevel.SEVERE:
            new = RoadConditionV2(RoadState.UNDER_REPAIR, level, 2.75, True, False, 8)
        elif level == DamageLevel.BLOCKED:
            new = RoadConditionV2(RoadState.BLOCKED, level, math.inf, False, False, 12)
        else:
            new = RoadConditionV2()
        self.road_conditions[edge_id] = new
        self.road_event_ledger.append(
            RoadEventRecordV2(
                step=int(step),
                edge=str(edge_id),
                old_state=str(old.state.value),
                new_state=str(new.state.value),
                damage_level=str(level.value),
                hazard_intensity=float(hazard_intensity),
                travel_time_multiplier=float(new.travel_time_multiplier),
                reason=str(reason),
            )
        )
        return new

    def reopen_edge(self, edge_id: str, *, step: int, reason: str = "repair_completed") -> None:
        old = self.road_conditions.get(edge_id, RoadConditionV2())
        new = RoadConditionV2(RoadState.REOPENED, DamageLevel.NONE, 1.0, True, True, 0, int(step))
        self.road_conditions[edge_id] = new
        self.road_event_ledger.append(
            RoadEventRecordV2(int(step), str(edge_id), old.state.value, new.state.value, "none", 0.0, 1.0, reason)
        )

    def travel_time(self, edge_id: str, weather: WeatherStateV2 | None = None) -> float:
        edge = self.edge(edge_id)
        condition = self.road_conditions.get(edge_id, RoadConditionV2())
        if not condition.truck_accessible or not math.isfinite(condition.travel_time_multiplier):
            return math.inf
        weather_factor = 1.0
        if weather is not None:
            weather_factor += min(float(weather.rain_intensity) / 80.0, 0.75)
            weather_factor += min(max((3.0 - float(weather.visibility)) / 10.0, 0.0), 0.35)
        return float(edge.base_travel_time * condition.travel_time_multiplier * weather_factor)

    def edge(self, edge_id: str) -> RoadEdgeV2:
        for edge in self.road_edges:
            if edge.edge_id == edge_id:
                return edge
        raise KeyError(edge_id)

    def estimate_uav_energy(
        self,
        uav_state: UAVStateV2,
        route_segments: Sequence[RouteSegmentV2],
        payload: float,
        weather_state: WeatherStateV2,
        recovery_plan: RecoveryPlanV2,
    ) -> EnergyEstimate:
        reasons: list[str] = []
        if float(payload) > float(uav_state.payload_capacity):
            reasons.append("PAYLOAD_EXCEEDS_CAPACITY")
        if weather_state.no_fly_status:
            reasons.append("NO_FLY_WEATHER")
        takeoff = 2.0 + 0.15 * float(payload)
        cruise = 0.0
        hover = 0.0
        service = 0.0
        wind_rad = math.radians(float(weather_state.wind_direction))
        for seg in route_segments:
            heading_rad = math.radians(float(seg.heading_degrees))
            headwind = float(weather_state.wind_speed) * math.cos(wind_rad - heading_rad)
            crosswind = abs(float(weather_state.wind_speed) * math.sin(wind_rad - heading_rad))
            wind_factor = max(0.55, 1.0 + 0.035 * headwind + 0.018 * crosswind)
            payload_factor = 1.0 + 0.06 * max(float(payload), 0.0)
            temp_penalty = 1.0 + max(abs(float(weather_state.temperature) - 20.0) - 10.0, 0.0) * 0.01
            degradation = 1.0 + max(float(uav_state.battery_degradation), 0.0)
            cruise += float(seg.distance) * 0.9 * wind_factor * payload_factor * temp_penalty * degradation
            hover += float(seg.hover_time) * 0.35 * (1.0 + 0.04 * crosswind)
            service += float(seg.service_time) * 0.25
        recovery = 1.5 if recovery_plan.recovery_anchor is not None else 5.0
        reserve = float(recovery_plan.reserve_energy if recovery_plan.reserve_energy > 0 else self.safety_reserve_energy)
        total = float(takeoff + cruise + hover + service + recovery + reserve)
        remaining = float(uav_state.battery_energy - total)
        if remaining < 0.0:
            reasons.append("INSUFFICIENT_UAV_ENERGY_WITH_RESERVE")
        self.minimum_energy_reserve_seen = min(self.minimum_energy_reserve_seen, remaining)
        return EnergyEstimate(takeoff, cruise, hover, service, recovery, reserve, total, remaining, not reasons, tuple(reasons))

    def check_launch(
        self,
        truck: TruckStateV2,
        uav: UAVStateV2,
        *,
        launch_anchor: int,
        task_payload: float,
        route_segments: Sequence[RouteSegmentV2],
        weather: WeatherStateV2,
        recovery_plan: RecoveryPlanV2,
        support_binding_conflict: bool = False,
    ) -> LaunchRecoveryCheck:
        reasons: list[str] = []
        if int(truck.current_node) != int(launch_anchor):
            reasons.append("INVALID_LAUNCH_ANCHOR")
        if not truck.stopped:
            reasons.append("TRUCK_NOT_STOPPED")
        if int(truck.dwell_steps) < int(self.minimum_dwell_steps):
            reasons.append("DWELL_TIME_TOO_SHORT")
        if uav.operational_state != UAVOperationalState.ON_TRUCK:
            reasons.append("UAV_NOT_ON_TRUCK")
        if weather.no_fly_status:
            reasons.append("NO_FLY_WEATHER")
        if support_binding_conflict:
            reasons.append("SUPPORT_BINDING_CONFLICT")
        if recovery_plan.recovery_anchor is None and not recovery_plan.alternate_anchors:
            reasons.append("NO_RECOVERY_PLAN")
        estimate = self.estimate_uav_energy(uav, route_segments, task_payload, weather, recovery_plan)
        reasons.extend(estimate.reason_codes)
        normalized = tuple(sorted({normalize_shield_reason(r) for r in reasons}))
        return LaunchRecoveryCheck(not normalized, normalized, launch_anchor if not normalized else None)

    def select_alternate_recovery(
        self,
        uav: UAVStateV2,
        truck: TruckStateV2,
        candidates: Iterable[int],
        *,
        remaining_energy: float,
        weather: WeatherStateV2,
    ) -> LaunchRecoveryCheck:
        reasons: list[str] = []
        if weather.no_fly_status:
            reasons.append("NO_FLY_WEATHER")
            return LaunchRecoveryCheck(False, tuple(reasons), None)
        best_anchor: int | None = None
        best_score = math.inf
        for anchor in candidates:
            truck_distance = abs(int(truck.current_node) - int(anchor))
            uav_distance = abs(int(uav.current_node) - int(anchor))
            required = float(uav_distance) * 0.9 + self.safety_reserve_energy
            if remaining_energy < required:
                continue
            score = float(truck_distance + uav_distance)
            if score < best_score:
                best_score = score
                best_anchor = int(anchor)
        if best_anchor is None:
            reasons.append("NO_FEASIBLE_ALTERNATE_RECOVERY")
        return LaunchRecoveryCheck(best_anchor is not None and not reasons, tuple(reasons), best_anchor)

    def plan_recovery_path(
        self,
        uav: UAVStateV2,
        truck: TruckStateV2 | None,
        *,
        target_recovery_anchor: int | None,
        candidate_alternate_anchors: Iterable[int],
        weather: WeatherStateV2,
        current_step: int,
        synchronization_window: int = 1,
    ) -> RecoveryPathPlanV2:
        reasons: list[str] = []
        if weather.no_fly_status:
            reasons.append("NO_FLY_WEATHER")
            return RecoveryPathPlanV2(
                False,
                None,
                (int(uav.current_node),),
                0.0,
                None,
                UAVOperationalState.EMERGENCY_RETURN.value,
                tuple(sorted(set(normalize_shield_reason(r) for r in reasons))),
            )
        anchors: list[int] = []
        if target_recovery_anchor is not None:
            anchors.append(int(target_recovery_anchor))
        anchors.extend(int(a) for a in candidate_alternate_anchors if int(a) not in anchors)
        if not anchors:
            return RecoveryPathPlanV2(
                False,
                None,
                (),
                0.0,
                None,
                UAVOperationalState.UAV_DROP.value,
                ("NO_RECOVERY_PLAN",),
            )

        selected: int | None = None
        best_energy = math.inf
        best_arrival: int | None = None
        for anchor in anchors:
            distance = float(abs(int(uav.current_node) - int(anchor)))
            energy = float(distance * 0.9 + self.safety_reserve_energy)
            arrival = int(current_step + max(1, math.ceil(distance)))
            if truck is not None:
                truck_eta = int(current_step + math.ceil(abs(int(truck.current_node) - int(anchor))))
                if abs(truck_eta - arrival) > int(max(synchronization_window, 0)):
                    continue
            if float(uav.battery_energy) >= energy and energy < best_energy:
                selected = int(anchor)
                best_energy = float(energy)
                best_arrival = int(arrival)

        if selected is None:
            if float(uav.battery_energy) <= 0.0:
                state = UAVOperationalState.UAV_DROP.value
                reasons.append("RESERVE_BREACH")
            elif weather.no_fly_status:
                state = UAVOperationalState.EMERGENCY_RETURN.value
            else:
                state = UAVOperationalState.FORCED_LANDING.value
                reasons.append("INSUFFICIENT_MISSION_ENERGY")
            return RecoveryPathPlanV2(
                False,
                None,
                tuple([int(uav.current_node)]),
                0.0 if not math.isfinite(best_energy) else float(best_energy),
                None,
                state,
                tuple(sorted(set(normalize_shield_reason(r) for r in reasons))),
            )

        state = UAVOperationalState.RECOVERY_PENDING.value
        if target_recovery_anchor is not None and int(selected) != int(target_recovery_anchor):
            state = UAVOperationalState.DIVERTING.value
        return RecoveryPathPlanV2(
            True,
            int(selected),
            (int(uav.current_node), int(selected)),
            float(best_energy),
            best_arrival,
            state,
            tuple(sorted(set(normalize_shield_reason(r) for r in reasons))),
        )

    def classify_unshielded_drop(
        self,
        uav: UAVStateV2,
        *,
        remaining_energy: float,
        weather: WeatherStateV2,
        recovery_available: bool,
    ) -> UAVOperationalState:
        if remaining_energy <= 0.0 or (not recovery_available and remaining_energy < self.safety_reserve_energy) or weather.storm_level >= 5:
            self.uav_drop_count += 1
            return UAVOperationalState.UAV_DROP
        if weather.no_fly_status:
            self.emergency_return_count += 1
            return UAVOperationalState.EMERGENCY_RETURN
        return uav.operational_state

    def _unsafe_proposal_key(
        self,
        proposal: LaunchRecoveryCheck,
        *,
        step: int | None,
        agent_id: str | None,
        candidate_digest: str | None,
    ) -> str:
        payload = {
            "step": int(step) if step is not None else None,
            "agent_id": str(agent_id or ""),
            "candidate_digest": str(candidate_digest or ""),
            "reason_codes": tuple(sorted(normalize_shield_reason(r) for r in proposal.reason_codes)),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    def _record_unsafe_proposal(
        self,
        proposal: LaunchRecoveryCheck,
        *,
        step: int | None,
        agent_id: str | None,
        candidate_digest: str | None,
        intervention: bool,
    ) -> bool:
        key = self._unsafe_proposal_key(proposal, step=step, agent_id=agent_id, candidate_digest=candidate_digest)
        if key in self._unsafe_proposal_keys:
            self.duplicate_unsafe_check_count += 1
            self.shield_reason_counts["DUPLICATE_CHECK"] = int(self.shield_reason_counts.get("DUPLICATE_CHECK", 0) + 1)
            return False
        self._unsafe_proposal_keys.add(key)
        self.unique_unsafe_proposal_count += 1
        self.unsafe_plan_proposal_count += 1
        if intervention:
            self.shield_intervention_count += 1
        normalized = tuple(sorted(normalize_shield_reason(r) for r in proposal.reason_codes))
        for reason in normalized:
            self.shield_reason_counts[reason] = int(self.shield_reason_counts.get(reason, 0) + 1)
        if "INSUFFICIENT_MISSION_ENERGY" in normalized or "RESERVE_BREACH" in normalized:
            self.reserve_breach_proposal_count += 1
        return True

    def apply_safety_shield(
        self,
        proposal: LaunchRecoveryCheck,
        *,
        protocol: str | None = None,
        step: int | None = None,
        agent_id: str | None = None,
        candidate_digest: str | None = None,
    ) -> SafetyDecisionV2:
        active = str(protocol or self.safety_protocol)
        reason_codes = tuple(sorted(normalize_shield_reason(r) for r in proposal.reason_codes))
        if active == "unshielded_stress":
            if not proposal.feasible:
                self._record_unsafe_proposal(
                    LaunchRecoveryCheck(False, reason_codes, proposal.selected_anchor),
                    step=step,
                    agent_id=agent_id,
                    candidate_digest=candidate_digest,
                    intervention=False,
                )
            return SafetyDecisionV2(True, active, reason_codes, None)
        if proposal.feasible:
            return SafetyDecisionV2(True, active, (), None)
        self._record_unsafe_proposal(
            LaunchRecoveryCheck(False, reason_codes, proposal.selected_anchor),
            step=step,
            agent_id=agent_id,
            candidate_digest=candidate_digest,
            intervention=True,
        )
        return SafetyDecisionV2(False, active, reason_codes, "BLOCK_OR_REPAIR")

    def decide_lifecycle_transition(
        self,
        uav: UAVStateV2,
        *,
        requested_event: str,
        target_anchor: int | None,
        energy_after: float | None,
        weather: WeatherStateV2,
        recovery_available: bool,
        alternate_recovery: LaunchRecoveryCheck | None = None,
        reason_codes: Sequence[str] = (),
    ) -> UAVLifecycleTransition:
        old = str(uav.operational_state.value if isinstance(uav.operational_state, UAVOperationalState) else uav.operational_state)
        requested = str(requested_event)
        before = float(uav.battery_energy)
        after = float(before if energy_after is None else energy_after)
        reasons = [normalize_shield_reason(r) for r in reason_codes]
        authorized = True
        anchor = target_anchor
        new_state = requested

        if requested in {"RECOVERY_PENDING", "RECOVERED", "DIVERTING"}:
            if alternate_recovery is not None and alternate_recovery.feasible and requested != "RECOVERED":
                new_state = UAVOperationalState.DIVERTING.value
                anchor = alternate_recovery.selected_anchor
                reasons.append("ALTERNATE_RECOVERY_SELECTED")
            elif requested == "RECOVERED":
                new_state = UAVOperationalState.RECOVERED.value
            elif not recovery_available:
                new_state = UAVOperationalState.FORCED_LANDING.value
                authorized = False
                reasons.append("NO_RECOVERY_PLAN")
            elif after <= 0.0 or (after < self.safety_reserve_energy and not recovery_available) or weather.storm_level >= 5:
                new_state = UAVOperationalState.UAV_DROP.value
                authorized = False
                reasons.append("RESERVE_BREACH")
            elif weather.no_fly_status:
                new_state = UAVOperationalState.EMERGENCY_RETURN.value
                reasons.append("NO_FLY_WEATHER")
            else:
                new_state = requested
        elif requested == "UAV_DROP":
            new_state = UAVOperationalState.UAV_DROP.value
            authorized = False
        elif requested == "FORCED_LANDING":
            new_state = UAVOperationalState.FORCED_LANDING.value
            authorized = False
        elif requested == "EMERGENCY_RETURN":
            new_state = UAVOperationalState.EMERGENCY_RETURN.value

        if new_state == UAVOperationalState.UAV_DROP.value:
            self.uav_drop_count += 1
        elif new_state == UAVOperationalState.EMERGENCY_RETURN.value:
            self.emergency_return_count += 1
        elif new_state == UAVOperationalState.FORCED_LANDING.value:
            self.forced_landing_count += 1
        return UAVLifecycleTransition(
            old_state=old,
            new_state=str(new_state),
            authorized=bool(authorized),
            target_anchor=anchor,
            energy_before=before,
            energy_after=after,
            reason_codes=tuple(sorted(set(reasons))),
        )

    def fairness_hash(self) -> str:
        payload = {
            "road_edges": [asdict(x) for x in self.road_edges],
            "weather": [asdict(v) for k, v in sorted(self.weather_by_region_step.items(), key=lambda kv: repr(kv[0]))],
            "node_region": dict(sorted((str(k), str(v)) for k, v in self.node_region.items())),
            "safety_protocol": self.safety_protocol,
            "minimum_dwell_steps": self.minimum_dwell_steps,
            "safety_reserve_energy": self.safety_reserve_energy,
        }
        data = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def road_digest(self) -> str:
        data = json.dumps([asdict(x) for x in self.road_edges], sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def weather_digest(self) -> str:
        data = json.dumps([asdict(v) for k, v in sorted(self.weather_by_region_step.items(), key=lambda kv: repr(kv[0]))], sort_keys=True)
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def export_road_event_ledger(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in self.road_event_ledger:
                f.write(json.dumps(asdict(row), sort_keys=True) + "\n")

    def export_weather_ledger(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in self.weather_ledger:
                f.write(json.dumps(asdict(row), sort_keys=True) + "\n")
