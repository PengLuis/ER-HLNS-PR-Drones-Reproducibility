from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from hetgat_hrl.core.mdp_spec import TaskKind


@dataclass(frozen=True)
class TruckRouteFeasibility:
    feasible: bool
    distance_m: float = float("inf")
    eta_steps: float = float("inf")
    reject_reason: str = ""


@dataclass(frozen=True)
class UAVSortieFeasibility:
    launch_feasible: bool
    service_feasible: bool
    recovery_feasible: bool
    full_sortie_feasible: bool
    energy_margin: float = 0.0
    recovery_margin: float = 0.0
    expected_completion_step: float = float("inf")
    expected_lifeline_remaining: float = 0.0
    reject_reason: str = ""


@dataclass(frozen=True)
class SupportChainFeasibility:
    feasible: bool
    expected_gain_m: float = 0.0
    reject_reason: str = ""


@dataclass(frozen=True)
class RoutineNearCompletionFeasibility:
    near_completion: bool
    route_dist_m: float = float("inf")
    eta_steps: float = float("inf")
    has_path: bool = False


class FeasibilityOracle:
    def __init__(self, env):
        self.env = env

    def evaluate_truck_route(self, truck_id: str, task_id: str) -> TruckRouteFeasibility:
        task = self.env.state.tasks.get(str(task_id))
        agent = self.env.state.agents.get(str(truck_id))
        if task is None or agent is None or getattr(agent, "node", None) is None:
            return TruckRouteFeasibility(False, reject_reason="missing_agent_or_task")
        try:
            dist = float(self.env._decision_shortest_path_distance(int(agent.node), int(task.demand_node)))
        except Exception:
            return TruckRouteFeasibility(False, reject_reason="path_error")
        if not (dist < float("inf")):
            return TruckRouteFeasibility(False, reject_reason="unreachable")
        step_m = max(float(getattr(self.env.cfg, "truck_speed_mps", 8.0)) * float(getattr(self.env.cfg, "dt_seconds", 60.0)), 1.0)
        return TruckRouteFeasibility(True, distance_m=dist, eta_steps=dist / step_m)

    def evaluate_routine_near_completion(
        self,
        truck_id: str,
        task_id: str,
        eta_threshold_steps: float = 5.0,
        dist_threshold_m: float = 1000.0,
    ) -> RoutineNearCompletionFeasibility:
        route = self.evaluate_truck_route(truck_id, task_id)
        near = bool(route.feasible and (route.distance_m <= dist_threshold_m or route.eta_steps <= eta_threshold_steps))
        return RoutineNearCompletionFeasibility(
            near_completion=near,
            route_dist_m=route.distance_m,
            eta_steps=route.eta_steps,
            has_path=route.feasible,
        )

    def evaluate_uav_sortie(self, uav_id: str, task_id: str, launch_anchor: Optional[int] = None) -> UAVSortieFeasibility:
        task = self.env.state.tasks.get(str(task_id))
        agent = self.env.state.agents.get(str(uav_id))
        if task is None or agent is None:
            return UAVSortieFeasibility(False, False, False, False, reject_reason="missing_agent_or_task")
        if task.kind != TaskKind.EMERGENCY:
            return UAVSortieFeasibility(False, False, False, False, reject_reason="not_time_critical")
        launch_ok = True
        reason = "ok"
        if hasattr(self.env, "_uav_launch_gate_check"):
            goals = getattr(self.env, "_effective_goals", {})
            missing = object()
            previous = goals.get(str(uav_id), missing)
            try:
                goals[str(uav_id)] = str(task_id)
                launch_ok, reason, _ = self.env._uav_launch_gate_check(str(uav_id), task=task, count_reject=False)
            except Exception as exc:
                launch_ok, reason = False, f"launch_gate_error:{exc.__class__.__name__}"
            finally:
                if previous is missing:
                    goals.pop(str(uav_id), None)
                else:
                    goals[str(uav_id)] = previous
        loaded_fn = getattr(self.env, "_uav_loaded_for_task", None)
        if callable(loaded_fn):
            loaded = bool(loaded_fn(str(uav_id), task))
        else:
            loaded_fn = getattr(self.env, "_uav_loaded", None)
            loaded = bool(loaded_fn(str(uav_id))) if callable(loaded_fn) else not bool(getattr(agent, "uav_needs_reload_flag", False))
        battery = float(getattr(agent, "battery", 1.0))
        service_feasible = bool(loaded and battery > 0.0)
        recovery_feasible = bool(launch_ok and "recovery" not in str(reason) and "no_recovery" not in str(reason))
        full = bool(launch_ok and service_feasible and recovery_feasible)
        lifeline = float(getattr(task, "lifeline_current", 0.0))
        return UAVSortieFeasibility(
            launch_feasible=bool(launch_ok),
            service_feasible=service_feasible,
            recovery_feasible=recovery_feasible,
            full_sortie_feasible=full,
            energy_margin=max(0.0, battery - 0.2),
            recovery_margin=1.0 if recovery_feasible else 0.0,
            expected_completion_step=float(getattr(self.env.state, "step_index", 0) + 1),
            expected_lifeline_remaining=lifeline,
            reject_reason="" if full else str(reason),
        )
