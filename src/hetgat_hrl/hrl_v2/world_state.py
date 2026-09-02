from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from hetgat_hrl.core.mdp_spec import AgentKind, TaskKind, TaskStatus


@dataclass(frozen=True)
class TaskState:
    task_id: str
    kind: TaskKind
    task_class: str
    demand_node: int
    status: TaskStatus
    demand_left: float = 0.0
    remaining_demand_kg: float = 0.0
    lifeline_current: float = 0.0
    deadline_step: int = 0
    service_started: bool = False

    @property
    def is_time_critical(self) -> bool:
        return self.kind == TaskKind.EMERGENCY or "time_critical" in str(self.task_class)


@dataclass(frozen=True)
class TruckState:
    agent_id: str
    node: Optional[int]
    pos_xy: Tuple[float, float]
    crashed: bool = False
    is_servicing: bool = False
    current_goal_id: Optional[str] = None


@dataclass(frozen=True)
class UAVState:
    agent_id: str
    node: Optional[int]
    pos_xy: Tuple[float, float]
    follow_target: Optional[str]
    battery: float = 1.0
    loaded: bool = True
    crashed: bool = False
    is_airborne: bool = False
    is_servicing: bool = False
    current_goal_id: Optional[str] = None


@dataclass
class EventQueue:
    events: List[dict] = field(default_factory=list)


@dataclass
class CommitmentState:
    agent_id: str
    task_id: Optional[str]
    kind: str
    started_step: int
    status: str = "active"
    reason: str = ""


@dataclass
class WorldState:
    step: int
    trucks: Dict[str, TruckState]
    uavs: Dict[str, UAVState]
    tasks: Dict[str, TaskState]
    events: EventQueue = field(default_factory=EventQueue)
    commitments: Dict[str, CommitmentState] = field(default_factory=dict)

    @classmethod
    def from_env(cls, env) -> "WorldState":
        step = int(getattr(env.state, "step_index", 0))
        trucks: Dict[str, TruckState] = {}
        uavs: Dict[str, UAVState] = {}
        goals = getattr(env, "_effective_goals", {}) or getattr(env, "_recommended_goals", {})
        for aid, agent in env.state.agents.items():
            pos = getattr(agent, "pos_xy", None)
            if pos is None and getattr(agent, "node", None) is not None and hasattr(env, "_node_xy"):
                pos = env._node_xy(int(agent.node))
            if pos is None:
                pos = (0.0, 0.0)
            if agent.kind == AgentKind.TRUCK:
                trucks[str(aid)] = TruckState(
                    agent_id=str(aid),
                    node=getattr(agent, "node", None),
                    pos_xy=(float(pos[0]), float(pos[1])),
                    crashed=bool(getattr(agent, "crashed", False)),
                    is_servicing=bool(getattr(agent, "service_task_id", None)),
                    current_goal_id=goals.get(str(aid)),
                )
            elif agent.kind == AgentKind.UAV:
                follow_target = getattr(agent, "follow_target", None)
                uavs[str(aid)] = UAVState(
                    agent_id=str(aid),
                    node=getattr(agent, "node", None),
                    pos_xy=(float(pos[0]), float(pos[1])),
                    follow_target=None if follow_target is None else str(follow_target),
                    battery=float(getattr(agent, "battery", 1.0)),
                    loaded=bool(
                        getattr(env, "_uav_loaded", lambda _aid: not bool(getattr(agent, "uav_needs_reload_flag", False)))(str(aid))
                    ),
                    crashed=bool(getattr(agent, "crashed", False)),
                    is_airborne=follow_target is None and not bool(getattr(agent, "crashed", False)),
                    is_servicing=bool(getattr(agent, "service_task_id", None)),
                    current_goal_id=goals.get(str(aid)),
                )
        tasks: Dict[str, TaskState] = {}
        for tid, task in env.state.tasks.items():
            tasks[str(tid)] = TaskState(
                task_id=str(tid),
                kind=task.kind,
                task_class=str(getattr(task, "task_class", "")),
                demand_node=int(task.demand_node),
                status=task.status,
                demand_left=float(getattr(task, "demand_left", 0.0)),
                remaining_demand_kg=float(getattr(task, "remaining_demand_kg", 0.0)),
                lifeline_current=float(getattr(task, "lifeline_current", 0.0)),
                deadline_step=int(getattr(task, "deadline_step", 0)),
                service_started=getattr(task, "first_service_step", None) is not None,
            )
        return cls(step=step, trucks=trucks, uavs=uavs, tasks=tasks)
