from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

from hetgat_hrl.core.mdp_spec import AgentKind, TruckAction, UAVAction


REASON_CODES = {
    "TARGET_NODE_INVALID",
    "TARGET_NODE_UNREACHABLE",
    "ROAD_BECAME_BLOCKED",
    "TASK_NOT_FOUND",
    "TASK_ALREADY_COMPLETED",
    "TASK_NOT_ACTIVE",
    "TASK_NOT_SERVICEABLE",
    "TASK_ASSIGNED_TO_OTHER_AGENT",
    "TASK_STATE_STALE",
    "DUPLICATE_EXCLUSIVE_TASK_ASSIGNMENT",
    "INSUFFICIENT_INVENTORY",
    "INSUFFICIENT_UAV_PAYLOAD",
    "INSUFFICIENT_UAV_ENERGY",
    "UAV_NOT_DOCKED",
    "UAV_ALREADY_AIRBORNE",
    "UAV_STATE_CONFLICT",
    "INVALID_LAUNCH_ANCHOR",
    "INVALID_RECOVERY_ANCHOR",
    "RENDEZVOUS_NOT_FEASIBLE",
    "SUPPORT_BINDING_STALE",
    "ACTION_SHAPE_INVALID",
    "ACTION_OUT_OF_RANGE",
    "AGENT_STATE_CONFLICT",
    "UNKNOWN_INVALID_REASON",
}


@dataclass(frozen=True)
class InvalidActionRecord:
    scenario: str
    method: str
    seed: int
    episode_index: int
    step: int
    agent_id: int
    agent_type: str
    agent_state: str
    action_type: str
    raw_action: object
    normalized_action: object
    current_node: int | None
    target_node: int | None
    task_id: int | None
    task_status: str | None
    planner_goal: object | None
    support_binding: object | None
    validation_layer: str
    reason_code: str
    reason_detail: str
    planner_state_digest: Dict[str, object]
    environment_state_digest: Dict[str, object]
    local_repair_attempted: bool
    local_repair_succeeded: bool
    fallback_action: object | None
    source_code_location: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ActionValidationResult:
    action: Optional[object]
    invalid: bool
    ignore: bool


@dataclass(frozen=True)
class DispatchValidationResult:
    valid: bool
    reason_code: str = ""
    reason_detail: str = ""
    normalized_action: object | None = None
    fallback_action: object | None = None
    source_code_location: str = ""


def normalize_truck_step_action(
    action: Any,
    *,
    in_transit: bool,
    replenish_timer: int,
) -> ActionValidationResult:
    if in_transit or replenish_timer > 0:
        return ActionValidationResult(action=None, invalid=False, ignore=True)
    if action is None:
        return ActionValidationResult(action=None, invalid=False, ignore=True)
    if not isinstance(action, TruckAction):
        return ActionValidationResult(action=None, invalid=True, ignore=False)
    if bool(action.stay) or action.target_node is None:
        return ActionValidationResult(action=action, invalid=False, ignore=True)
    return ActionValidationResult(action=action, invalid=False, ignore=False)


def normalize_uav_step_action(action: Any) -> ActionValidationResult:
    if action is None:
        return ActionValidationResult(action=UAVAction(), invalid=False, ignore=False)
    if not isinstance(action, UAVAction):
        return ActionValidationResult(action=None, invalid=True, ignore=False)
    return ActionValidationResult(action=action, invalid=False, ignore=False)


def safe_noop_for_agent_state(agent_state: Any) -> object:
    if getattr(agent_state, "kind", None) == AgentKind.TRUCK:
        return TruckAction(stay=True)
    return UAVAction(vx=0.0, vy=0.0)


def _action_public_dict(action: object | None) -> Dict[str, object]:
    if action is None:
        return {}
    out: Dict[str, object] = {}
    for name in ("target_node", "stay", "vx", "vy", "takeoff", "bind_truck_id"):
        if hasattr(action, name):
            out[name] = getattr(action, name)
    return out


def _agent_numeric_id(agent_id: str) -> int:
    digits = "".join(ch for ch in str(agent_id) if ch.isdigit())
    return int(digits) if digits else -1


def _task_context(env: Any, aid: str) -> Tuple[int | None, str | None]:
    goal = getattr(env, "_effective_goals", {}).get(str(aid), getattr(env, "_recommended_goals", {}).get(str(aid), None))
    tasks = getattr(env.state, "tasks", {})
    task = tasks.get(str(goal)) if goal is not None else None
    if task is None and goal is not None:
        task = next((t for t in tasks.values() if str(getattr(t, "task_id", "")) == str(goal)), None)
    if task is None:
        return None, None
    try:
        task_id = getattr(task, "task_id", None)
        if isinstance(task_id, int):
            return int(task_id), str(task.status.name)
        digits = "".join(ch for ch in str(task_id) if ch.isdigit())
        return (int(digits) if digits else None), str(task.status.name)
    except Exception:
        return None, None


def _state_digest(env: Any, aid: str) -> Dict[str, object]:
    st = env.state.agents.get(str(aid))
    if st is None:
        return {}
    blocked = getattr(getattr(env, "topology", None), "blocked_edges", set())
    return {
        "step": int(getattr(env.state, "step_index", -1)),
        "agent_kind": str(getattr(getattr(st, "kind", None), "value", getattr(st, "kind", ""))),
        "node": None if getattr(st, "node", None) is None else int(st.node),
        "follow_target": None if getattr(st, "follow_target", None) is None else str(st.follow_target),
        "battery": float(getattr(st, "battery", 0.0)),
        "crashed": bool(getattr(st, "crashed", False)),
        "blocked_edge_count": int(len(blocked)),
    }


def make_invalid_action_record(
    env: Any,
    aid: str,
    raw_action: object | None,
    normalized_action: object | None,
    *,
    validation_layer: str,
    reason_code: str,
    reason_detail: str,
    local_repair_attempted: bool = False,
    local_repair_succeeded: bool = False,
    fallback_action: object | None = None,
    source_code_location: str = "",
) -> InvalidActionRecord:
    if reason_code not in REASON_CODES:
        reason_code = "UNKNOWN_INVALID_REASON"
    st = env.state.agents.get(str(aid))
    task_id, task_status = _task_context(env, str(aid))
    goal = getattr(env, "_effective_goals", {}).get(str(aid), getattr(env, "_recommended_goals", {}).get(str(aid), None))
    target_node = getattr(normalized_action, "target_node", None)
    return InvalidActionRecord(
        scenario=str(getattr(getattr(env, "cfg", None), "map_complexity", "")) + "-" + str(getattr(getattr(env, "cfg", None), "scenario", "")),
        method=str(getattr(env, "current_method", "")),
        seed=int(getattr(getattr(env, "cfg", None), "seed", 0)),
        episode_index=int(getattr(env, "current_episode_index", 0)),
        step=int(getattr(env.state, "step_index", -1)),
        agent_id=_agent_numeric_id(str(aid)),
        agent_type=str(getattr(getattr(st, "kind", None), "value", getattr(st, "kind", ""))),
        agent_state=str(_state_digest(env, str(aid))),
        action_type=type(raw_action).__name__ if raw_action is not None else "None",
        raw_action=_action_public_dict(raw_action),
        normalized_action=_action_public_dict(normalized_action),
        current_node=None if st is None or getattr(st, "node", None) is None else int(st.node),
        target_node=None if target_node is None else int(target_node),
        task_id=task_id,
        task_status=task_status,
        planner_goal=goal,
        support_binding=getattr(env, "_planner_truck_assist_waypoint_by_truck", {}).get(str(aid), None),
        validation_layer=str(validation_layer),
        reason_code=str(reason_code),
        reason_detail=str(reason_detail),
        planner_state_digest={
            "effective_goal": goal,
            "recommended_goal": getattr(env, "_recommended_goals", {}).get(str(aid), None),
        },
        environment_state_digest=_state_digest(env, str(aid)),
        local_repair_attempted=bool(local_repair_attempted),
        local_repair_succeeded=bool(local_repair_succeeded),
        fallback_action=_action_public_dict(fallback_action),
        source_code_location=str(source_code_location),
    )


def validate_action_for_dispatch(env: Any, aid: str, action: object | None) -> DispatchValidationResult:
    st = env.state.agents.get(str(aid))
    if st is None:
        return DispatchValidationResult(
            valid=False,
            reason_code="AGENT_STATE_CONFLICT",
            reason_detail=f"missing agent {aid}",
            normalized_action=None,
            fallback_action=None,
            source_code_location="base_env_step_validation.validate_action_for_dispatch",
        )
    if st.kind == AgentKind.TRUCK:
        normalized = normalize_truck_step_action(
            action,
            in_transit=bool(getattr(st, "transit", None) is not None),
            replenish_timer=int(getattr(st, "truck_replenish_timer", 0)),
        )
        if normalized.ignore:
            return DispatchValidationResult(valid=True, normalized_action=normalized.action)
        if normalized.invalid or not isinstance(normalized.action, TruckAction):
            return DispatchValidationResult(
                valid=False,
                reason_code="ACTION_SHAPE_INVALID",
                reason_detail="truck action is not a TruckAction",
                normalized_action=normalized.action,
                fallback_action=TruckAction(stay=True),
                source_code_location="base_env_step_validation.validate_action_for_dispatch:truck_shape",
            )
        target = normalized.action.target_node
        if normalized.action.stay or target is None:
            return DispatchValidationResult(valid=True, normalized_action=normalized.action)
        if getattr(st, "node", None) is None:
            return DispatchValidationResult(
                valid=False,
                reason_code="AGENT_STATE_CONFLICT",
                reason_detail="truck has no current node",
                normalized_action=normalized.action,
                fallback_action=TruckAction(stay=True),
                source_code_location="base_env_step_validation.validate_action_for_dispatch:truck_node",
            )
        src = int(st.node)
        dst = int(target)
        if dst not in set(int(x) for x in getattr(env.topology, "adjacency", {}).get(src, set())):
            return DispatchValidationResult(
                valid=False,
                reason_code="TARGET_NODE_INVALID",
                reason_detail=f"target {dst} is not adjacent to node {src}",
                normalized_action=normalized.action,
                fallback_action=TruckAction(stay=True),
                source_code_location="base_env_step_validation.validate_action_for_dispatch:truck_adjacency",
            )
        if bool(getattr(env, "_decision_is_blocked", lambda *_: False)(src, dst)):
            return DispatchValidationResult(
                valid=False,
                reason_code="ROAD_BECAME_BLOCKED",
                reason_detail=f"decision edge {src}-{dst} is blocked",
                normalized_action=normalized.action,
                fallback_action=TruckAction(stay=True),
                source_code_location="base_env_step_validation.validate_action_for_dispatch:truck_decision_blocked",
            )
        if bool(getattr(env.topology, "is_blocked", lambda *_: False)(src, dst)):
            return DispatchValidationResult(
                valid=False,
                reason_code="ROAD_BECAME_BLOCKED",
                reason_detail=f"physical edge {src}-{dst} is blocked",
                normalized_action=normalized.action,
                fallback_action=TruckAction(stay=True),
                source_code_location="base_env_step_validation.validate_action_for_dispatch:truck_physical_blocked",
            )
        return DispatchValidationResult(valid=True, normalized_action=normalized.action)

    normalized = normalize_uav_step_action(action)
    if normalized.invalid or not isinstance(normalized.action, UAVAction):
        return DispatchValidationResult(
            valid=False,
            reason_code="ACTION_SHAPE_INVALID",
            reason_detail="uav action is not a UAVAction",
            normalized_action=normalized.action,
            fallback_action=UAVAction(vx=0.0, vy=0.0),
            source_code_location="base_env_step_validation.validate_action_for_dispatch:uav_shape",
        )
    act = normalized.action
    if bool(getattr(st, "crashed", False)):
        return DispatchValidationResult(valid=True, normalized_action=UAVAction(vx=0.0, vy=0.0))
    if act.bind_truck_id is not None:
        tid = str(act.bind_truck_id)
        truck = env.state.agents.get(tid)
        if truck is None or truck.kind != AgentKind.TRUCK:
            return DispatchValidationResult(
                valid=False,
                reason_code="INVALID_RECOVERY_ANCHOR",
                reason_detail=f"bind target {tid} is not a truck",
                normalized_action=act,
                fallback_action=UAVAction(vx=0.0, vy=0.0),
                source_code_location="base_env_step_validation.validate_action_for_dispatch:uav_bind_target",
            )
        if getattr(st, "pos_xy", None) is not None:
            uxy = st.pos_xy
        else:
            uxy = getattr(env, "_node_xy")(int(getattr(st, "node", 0) or 0))
        txy = truck.pos_xy if getattr(truck, "pos_xy", None) is not None else getattr(env, "_node_xy")(int(getattr(truck, "node", 0) or 0))
        dx = float(uxy[0]) - float(txy[0])
        dy = float(uxy[1]) - float(txy[1])
        dist = float((dx * dx + dy * dy) ** 0.5)
        bind_window = float(getattr(env, "_uav_bind_window_m")(truck))
        if dist > bind_window and not bool(getattr(env, "_uav_recovery_required", lambda *_: False)(str(aid))):
            return DispatchValidationResult(
                valid=False,
                reason_code="RENDEZVOUS_NOT_FEASIBLE",
                reason_detail=f"bind target {tid} is {dist:.3f}m away, window={bind_window:.3f}m",
                normalized_action=act,
                fallback_action=UAVAction(vx=0.0, vy=0.0),
                source_code_location="base_env_step_validation.validate_action_for_dispatch:uav_bind_window",
            )
    return DispatchValidationResult(valid=True, normalized_action=act)
